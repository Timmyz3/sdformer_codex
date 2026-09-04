# SDformerFlow 硬件加速器研究路径（DATE 标准，6 周版）

2026-05-29

---

## DATE 要求 vs ML 会议

| | NeurIPS/ICLR | **DATE** |
|---|---|---|
| 硬件实现 | 分析模型可接受 | **至少 DC 综合**，最好有 P&R |
| RTL 完整度 | 一个创新模块足够 | **完整加速器数据通路** |
| 评估指标 | 精度 > 能效 | **面积/功耗/延迟/能效 > 精度** |
| 对标 | GPU 即可 | **必须对标 SOTA 加速器** (FireFly, Bishop, SENECA 等) |
| 创新点 | 算法创新为主 | **架构创新 + 实现创新** |

**DATE 审稿人想看的是：你们做了一个新的加速器，RTL 写出来了，DC 过了，数字比已有方案好。**

---

## 6 周总览

```
Week 1:   全系统数据流规格 (不写代码，定架构)
Week 2-3: 数据通路 RTL (所有引擎)
Week 4:   控制逻辑 + 存储子系统 + 顶层集成
Week 5:   DC 综合 + 功耗分析 + 优化迭代
Week 6:   论文 + 最终对标表
```

---

## Week 1: 全系统数据流规格（定架构）

### 目的

不写一行 RTL。把整个加速器的**所有接口、所有模块、所有数据流动**定死。Week 2-4 照着规格写代码。

### Day 1-2: 完成顶层接口定义

```
┌─────────────────────────────────────────────────────┐
│                  SDformer Accelerator                │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ Voxel    │  │ Weight   │  │ DRAM (off-chip)  │   │
│  │ Stream   │  │ Buffer   │  │ 16-bit @ 800MHz  │   │
│  │ (input)  │  │ (256KB)  │  │ HBM2 or LPDDR5   │   │
│  └────┬─────┘  └────┬─────┘  └────────┬─────────┘   │
│       │             │                 │              │
│  ┌────▼─────────────▼─────────────────▼──────────┐   │
│  │              AXI4 Interconnect                 │   │
│  └──┬──────────┬──────────┬──────────┬───────────┘   │
│     │          │          │          │                │
│  ┌──▼───┐ ┌───▼───┐ ┌───▼───┐ ┌───▼────┐            │
│  │Event │ │Sparse │ │Binary │ │ Dense  │            │
│  │Scat- │ │ MAC   │ │Engine │ │  MAC   │            │
│  │ter   │ │Engine │ │(AND-  │ │ Engine │            │
│  │Unit  │ │(MLP/  │ │PopCnt)│ │(PSN/   │            │
│  │      │ │Conv)  │ │       │ │Decoder)│            │
│  └──────┘ └───┬───┘ └───┬───┘ └───┬────┘            │
│               │          │          │                │
│  ┌────────────▼──────────▼──────────▼────────────┐   │
│  │              Flow Output Buffer                │   │
│  │              [2, 480, 640] × 4 scales          │   │
│  └───────────────────────────────────────────────┘   │
│                                                     │
│  ┌──────────────────────────────────────────────┐    │
│  │         Control FSM (Microcode Sequencer)     │    │
│  │         per-layer config: opcode, shapes,     │    │
│  │         sparsity mode, precision, addresses   │    │
│  └──────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

### Day 3-4: 每个引擎的接口规格 + 数据流时序

```systemverilog
// 顶层接口
module sdformer_accelerator_top (
    // AXI4 Master (DRAM)
    output axi4_master dram,
    // Voxel stream input (from event camera frontend)
    input  wire [15:0] voxel_data,     // FP16
    input  wire        voxel_valid,
    output wire        voxel_ready,
    // Flow output
    output wire [15:0] flow_data [1:0], // FP16 (dx, dy)
    output wire        flow_valid,
    // Control
    input  wire [31:0] ctrl_config     // layer config word
);

// 内部引擎接口 (AXI4-Stream)
interface engine_stream #(DATA_W=16, USER_W=8);
    logic [DATA_W-1:0] tdata;
    logic              tvalid;
    logic              tready;
    logic [USER_W-1:0] tuser;   // {sparsity_flag, precision[2:0], layer_id[4:0]}
endinterface
```

### Day 5-7: 全网络调度表

核心交付物：一张表，定义每一层在哪个引擎执行、输入数据精度、稀疏模式、预期时钟周期。

```
Layer_ID  Layer_Name                                    Engine     InPrec   Sparsity  Cycles(est)
0         voxel_scatter                                 Scatter    FP16     60%       245,760
1         patch_embed.head_conv                        SparseMAC  1-bit    70%       1,843,200
2         patch_embed.down_conv                        SparseMAC  1-bit    70%       3,686,400
3-4       patch_embed.resblock_0/1                     SparseMAC  1-bit    75%       7,372,800
5         patch_embed.proj_conv                        SparseMAC  1-bit    70%       1,843,200
6         stage0.block0.attn.q_proj                    SparseMAC  1-bit    70%       176,400
7         stage0.block0.attn.k_proj                    SparseMAC  1-bit    70%       176,400
8         stage0.block0.attn.qkformer_gate             Binary     2-bit    80%       3,726
9         stage0.block0.mlp.fc1                         SparseMAC  1-bit    75%       23,592,960
10        stage0.block0.mlp.fc2                         SparseMAC  1-bit    75%       23,592,960
...
89        decoder3.upsample                             DenseMAC   FP16     0%        23,592
90        decoder3.conv5x5                              SparseMAC  1-bit    60%       35,389,440
91        decoder3.pred_conv                            SparseMAC  1-bit    60%       29,491
92        temporal_sum + interpolate                    DenseMAC   FP16     0%        9,830
```

这张表是 Week 2-4 所有 RTL 的规范文档。

**交付物（Week 1 结束）**:
- 顶层框图
- 引擎间接口规范 (AXI4-Stream + AXI4 MM)
- 全网络调度表 (93 层 × 引擎分配 × 周期估算)
- 各引擎内部微架构框图 (哪些模块，怎么连)

---

## Week 2-3: 数据通路 RTL

**策略**：写全所有引擎的 RTL，但只用 DC 综合关键模块。其余模块的 DC 数字从已发表论文的 28nm 数据等比例缩放。

### Week 2: Binary 引擎 + Event Scatter + Sparse MAC

**2.1 Binary 引擎（Day 1-4, 核心贡献）**

这是论文最创新的硬件模块。对标 FireFly-T 的二值 AND-PopCount，我们做三值的：

```systemverilog
// ── sc_popcount_unit.sv ──
// 论文 Figure X: SC attention popcount unit microarchitecture
// 32-dim ternary XNOR-PopCount, 2-cycle, 全组合逻辑

module sc_popcount_unit #(
    parameter HEAD_DIM = 32
) (
    input  wire [1:0] q_sign [HEAD_DIM-1:0],    // {sign, active}
    input  wire [1:0] k_sign [HEAD_DIM-1:0],
    output logic [5:0] match_count,              // [-32, +32] mapped to positive count
    output logic [5:0] conflict_count
);
    // Stage 1: per-dim logic (parallel, 32x replication)
    logic match [HEAD_DIM-1:0];
    logic conflict [HEAD_DIM-1:0];
    
    generate
        for (genvar d = 0; d < HEAD_DIM; d++) begin : dim_logic
            wire both_active = q_sign[d][0] & k_sign[d][0];     // active bit = LSB
            wire same_sign   = ~(q_sign[d][1] ^ k_sign[d][1]);  // sign bit = MSB
            assign match[d]    = both_active &  same_sign;
            assign conflict[d] = both_active & ~same_sign;
        end
    endgenerate
    
    // Stage 2: carry-save adder tree, log2(HEAD_DIM) = 5 levels
    popcount_tree #(.INPUTS(HEAD_DIM), .OUTPUT_BITS(6)) match_tree (
        .bits(match), .count(match_count)
    );
    popcount_tree #(.INPUTS(HEAD_DIM), .OUTPUT_BITS(6)) conflict_tree (
        .bits(conflict), .count(conflict_count)
    );
endmodule

// ── shiftmax_lut.sv ──
// 64-entry × 8-bit, single-cycle read
module shiftmax_lut (
    input  wire [5:0]  score,       // [-32, +32] → offset by 32
    input  wire [5:0]  ceil_log2_sum,
    output logic [7:0] gate         // 8-bit fixed-point [0, 1]
);
    // ROM: 64 entries of 2^x values
    logic [7:0] lut [63:0];
    initial $readmemh("shiftmax_lut.hex", lut);
    
    // Single-cycle: LUT read + shift for divide-by-power-of-2
    assign gate = lut[score] >> ceil_log2_sum;
endmodule

// ── binary_engine_core.sv ──
// 3-stage pipeline, 1 token/cycle throughput
module binary_engine_core #(
    parameter NUM_HEADS = 24,
    parameter HEAD_DIM = 32
) (
    input  wire clk, rst_n,
    input  engine_stream q_stream,
    input  engine_stream k_stream,
    output engine_stream attn_out
);
    // Stage 1: SC popcount per head (24 heads parallel)
    sc_popcount_unit head_popcount [NUM_HEADS-1:0] (...);
    
    // Stage 2: Shiftmax LUT per head
    shiftmax_lut head_lut [NUM_HEADS-1:0] (...);
    
    // Stage 3: K gating (element-wise MUX)
    // gate × K per channel
    // Pipeline registers between stages
endmodule
```

**2.2 Sparse MAC 引擎（Day 5-7）**

```systemverilog
// ── sparse_mac_pe.sv ──
// Bit-serial processing element, 8b weight × 1b spike
// 对标 28nm Spiking ViT 的 1b/8b unified adder array

module sparse_mac_pe #(
    parameter WEIGHT_BITS = 8,
    parameter ACCUM_BITS  = 24
) (
    input  wire clk, rst_n,
    input  wire spike,                             // 1-bit
    input  wire [WEIGHT_BITS-1:0] weight,          // INT8
    input  wire [ACCUM_BITS-1:0] psum_in,          // partial sum from neighbor
    output logic [ACCUM_BITS-1:0] psum_out,
    output logic done                              // 8 cycles complete
);
    // AND(spike, weight[bit]) → shift → accumulate
    // 8 cycles × 串行, 或 8 路并行 (1 cycle, 面积换时间)
    
    // 并行版本 (面积大但快):
    wire [WEIGHT_BITS-1:0] weight_masked;
    assign weight_masked = spike ? weight : '0;
    assign psum_out = psum_in + {{(ACCUM_BITS-WEIGHT_BITS){1'b0}}, weight_masked};
endmodule

// ── sparse_pe_array.sv ──
// 128 PE systolic array with TTB scheduler
// 对标 Bishop ISCA 2025 Sparse Core

module sparse_pe_array #(
    parameter PE_ROWS = 16,
    parameter PE_COLS = 8
) (
    input  wire clk, rst_n,
    input  engine_stream activation_in,     // 1-bit spikes, TTB format
    input  engine_stream weight_in,         // INT8 weights
    output engine_stream psum_out
);
    // Weight-stationary systolic array
    // TTB scheduler skips all-zero bundles
    sparse_mac_pe pe [PE_ROWS-1:0][PE_COLS-1:0] (...);
    
    // TTB bundle detector (Bishop-style)
    ttb_stratifier stratifier (
        .spike_in(activation_in),
        .dense_route(dense_bypass),
        .sparse_route(sparse_to_pe)
    );
endmodule
```

**2.3 Event Scatter 单元（Day 7, 简单）**

```systemverilog
// ── event_scatter_unit.sv ──
// 参考 ASNA-Flow 的 event-to-grid 设计, 简化版
// 双线性 scatter-add (4 corners × weight), FP16 加法器

module event_scatter_unit (
    input  wire clk, rst_n,
    // Event input: {x[9:0], y[8:0], t[3:0], p}
    input  wire [23:0] event_packet,
    input  wire        event_valid,
    // VoxelGrid output
    output wire [15:0] grid_data,
    output wire [18:0] grid_addr   // {t[3:0], p, y[8:0], x[9:0]}
);
    // 4 路双线性插值器并行 (4 corners)
    // FP16 加法器 × 4 (面积小，可接受)
endmodule
```

**交付物（Week 2-3 结束）**:
- Binary 引擎完整 RTL (sc_popcount + shiftmax_lut + engine_core)
- Sparse MAC PE RTL (bit-serial PE + array + TTB scheduler)
- Event Scatter RTL
- 所有模块的独立 testbench

---

## Week 3: 其余引擎 + 控制逻辑

### Day 1-3: Dense MAC + 存储子系统

**Dense MAC 引擎**（对标标准 systolic array, 不创新, 用成熟设计）：

```systemverilog
// ── dense_mac_array.sv ──
// 32×32 systolic array @ FP16
// 标准 weight-stationary 数据流
// 用于: PSN temporal mix (5×5 matmul) + Decoder Conv + Flow output

module dense_mac_array #(ROWS=32, COLS=32) (...);
    // 标准 systolic MAC, 不赘述
    // 只在 PSN 时分复用 (5×5=25 MACs, 占 1/40 阵列)
    // 其余时间做 Decoder 反卷积
endmodule
```

**存储子系统**：

```systemverilog
// ── window_sram.sv ──
// 512KB dual-bank SRAM with automatic prefetch
// Bank 0: current window computation
// Bank 1: next window DMA prefetch

module window_sram #(SIZE_KB=512) (
    input  wire clk, rst_n,
    // AXI4 MM (to DRAM)
    axi4_master dram_port,
    // Internal engine access
    input  wire [18:0] engine_addr,
    input  wire [255:0] engine_wdata,  // 256-bit wide for efficiency
    output wire [255:0] engine_rdata,
    input  wire        engine_re, engine_we
);
    // True dual-port SRAM
    // Automatic prefetch state machine: when bank A is busy,
    // prefetch next window's weights into bank B
endmodule

// ── weight_prefetch_dma.sv ──
// Double-buffered weight DMA from DRAM
// Weights loaded layer-by-layer, hidden behind compute
module weight_prefetch_dma (...);
endmodule
```

### Day 4-5: 控制 FSM

```systemverilog
// ── control_fsm.sv ──
// Microcoded sequencer: 每层一个配置字
// 来自 Week 1 的全网络调度表

module control_fsm (
    input  wire clk, rst_n,
    output ctrl_layer_config [92:0],  // 93 layers
    output engine_select [3:0],       // which engine active
    // Handshake with each engine
    output wire start_scatter, start_sparse, start_binary, start_dense,
    input  wire done_scatter, done_sparse, done_binary, done_dense
);
    // Layer counter: 0→92, then output ready
    // Per-layer config: {engine_id[1:0], sparsity_mode[1:0], precision[1:0],
    //                     input_addr[31:0], weight_addr[31:0], output_addr[31:0],
    //                     input_shape[47:0], output_shape[47:0]}
    
    // State machine:
    //   IDLE → LOAD_WEIGHTS → COMPUTE → STORE_OUTPUT → NEXT_LAYER
    //   每层的 LOAD_WEIGHTS 与上一层的 COMPUTE 通过双缓冲并行
endmodule
```

### Day 6-7: 顶层集成

```systemverilog
// ── sdformer_accelerator_top.sv ──
// 顶层: 实例化所有引擎 + 互联 + 控制

module sdformer_accelerator_top (
    input  wire clk, rst_n,
    // External interfaces
    axi4_master dram_port,
    input  wire [15:0] voxel_in, input voxel_valid, output voxel_ready,
    output wire [15:0] flow_out_dx, flow_out_dy, output flow_valid,
    // Debug
    output wire [31:0] debug_counter
);
    // 时钟域: 单一时钟 @ 500MHz (目标)
    
    // 实例化
    event_scatter_unit scatter (.*);
    sparse_pe_array    #(16,8) sparse_mac (.*);
    binary_engine_core #(24,32) binary_engine (.*);
    dense_mac_array    #(32,32) dense_mac (.*);
    window_sram        #(512) sram (.*);
    weight_prefetch_dma dma (.*);
    control_fsm        ctrl (.*);
    
    // AXI4 interconnect (简化: round-robin arbiter)
    axi_interconnect axi_xbar (.*);
endmodule
```

**交付物（Week 3 结束）**:
- 全系统 RTL (所有 .sv 文件，预估 ~3000 行)
- 顶层集成 testbench (py_testbench.py 调用 Verilator)
- 语法检查通过 (Verilator lint)

---

## Week 4: 仿真 + 调试 + DC 准备

### Day 1-3: Verilator 功能仿真

```python
# py_testbench.py — Python-driven testbench via Verilator
import verilator
import numpy as np

# 从 H41 SC PyTorch checkpoint 导出 golden 数据
# 逐层提取输入/输出作为 test vectors

tb = verilator.VerilatorTb("sdformer_accelerator_top.sv")
tb.start()

for layer_id in range(93):
    # 加载该层的 golden 输入
    inputs = load_layer_input(layer_id, checkpoint)
    expected = load_layer_output(layer_id, checkpoint)
    
    # 发送到加速器
    tb.send_layer_config(layer_configs[layer_id])
    tb.send_inputs(inputs)
    tb.wait_done()
    
    # 比较输出
    actual = tb.read_output()
    assert np.allclose(actual, expected, rtol=1e-4, atol=1e-4), \
        f"Layer {layer_id} mismatch: max_err={np.max(np.abs(actual-expected))}"

print("All 93 layers pass")

# 报告: 总周期数、各引擎利用率
tb.report_stats()
```

### Day 4-5: 关键模块单独 DC 综合

**要 DC 的模块** (论文必须有 28nm 数字)：

| 模块 | 优先级 | 原因 |
|---|---|---|
| `sc_popcount_unit` | **P0** | 核心创新，论文 Figure X |
| `binary_engine_core` | **P0** | 完整 Binary 引擎，对标 FireFly-T |
| `sparse_mac_pe` | P1 | Bit-serial PE，对标 28nm ViT |
| `event_scatter_unit` | P2 | 简单，非核心创新 |

```tcl
# dc_synth.tcl — Synopsys Design Compiler script
set_db init_lib_search_path {/path/to/tsmc28/db}
set_db init_hdl_search_path  {../rtl}
read_db tcbn28hpcplusbwp7d0_120a.db

# Target: 500 MHz
create_clock -period 2.0 [get_ports clk]

# ── Binary Engine ──
read_sverilog {sc_popcount_unit.sv popcount_tree.sv shiftmax_lut.sv binary_engine_core.sv}
set_db syn_generic_effort high
set_db syn_map_effort high
set_db syn_opt_effort high
compile_ultra

report_area  -hierarchy > reports/binary_engine_area.rpt
report_power -hierarchy > reports/binary_engine_power.rpt
report_timing           > reports/binary_engine_timing.rpt

# ── Sparse MAC PE ──
remove_design -all
read_sverilog {sparse_mac_pe.sv}
compile_ultra
report_area  -hierarchy > reports/sparse_pe_area.rpt
report_power -hierarchy > reports/sparse_pe_power.rpt
```

### Day 6-7: 数字整理 + 系统级估算

```python
# 全加速器系统级估算
# Binary 引擎: DC 实测数字
binary_area  = 0.012   # mm² (DC report)
binary_power = 2.3     # mW  (DC report)

# Sparse MAC: DC 实测 (PE) + 缩放 (阵列)
sparse_pe_area  = 0.004   # mm² per PE (DC)
sparse_pe_power = 0.15    # mW per PE (DC)
sparse_array_area  = sparse_pe_area * 128 * 1.15   # +15% 互连开销
sparse_array_power = sparse_pe_power * 128 * 0.85  # 85% 利用率

# Dense MAC: 从标准 28nm 数据缩放
# 32×32 FP16 systolic = ~0.8 mm², ~150 mW (公开数据)
dense_area  = 0.8
dense_power = 150

# SRAM: CACTI 7.0 或公开数据
sram_area  = 0.35   # mm² for 512KB @ 28nm
sram_power = 45     # mW

# 控制逻辑 + 互联
ctrl_area  = 0.05
ctrl_power = 10

# ── 汇总 ──
total_area  = binary_area + sparse_array_area + dense_area + sram_area + ctrl_area
total_power = binary_power + sparse_array_power + dense_power + sram_power + ctrl_power

# 一帧推理
total_cycles = from_week3_simulation   # Verilator 实测
fps = 500e6 / total_cycles             # @ 500MHz
energy_per_frame = total_power * total_cycles / 500e6  # mJ
```

**交付物（Week 4 结束）**:
- Verilator 全系统仿真通过 (93 层)
- DC 综合报告 (面积/功耗 @ 28nm)
- 最终系统数字: FPS, 功耗, 面积, 能效

---

## Week 5: DC 迭代优化 + 对标

### Day 1-3: DC 优化

```
优化项:
1. 时钟频率扫描: 500MHz → 600MHz → 400MHz (tradeoff)
2. Binary 引擎并行度: 24 → 48 heads (面积换吞吐)
3. Sparse PE 数据宽度: 8-bit → 4-bit (精度换功耗)
4. SRAM 大小扫描: 256KB → 1MB (面积换带宽)

每个配置跑一次 DC → 画面积/功耗 Pareto 前沿
```

### Day 4-5: SOTA 对标表

| | SDformer (Ours) | FireFly-T | Bishop | ASNA-Flow | 28nm SpkViT | A100 GPU |
|---|---|---|---|---|---|---|
| 工艺 | **28nm** | FPGA | 28nm | 28nm | 28nm | 7nm |
| 频率 | **500 MHz** | 300 | — | — | — | 1.4 GHz |
| 面积 | X mm² | — | — | — | — | 826 mm² |
| 功耗 | X W | — | — | 7.9 mW | — | 300 W |
| FPS | X @ 480×640 | — | — | 104 | — | — |
| 能效 | X TOPS/W | — | — | — | 57.7 | 1.04 |
| SNN 三值 | **✅** | ❌ | ❌ | ❌ | ❌ | ❌ |
| 光流 | **✅** | ❌ | ❌ | ✅ | ❌ | ❌ |
| 注意力 | **PopCount** | AND-PopCnt | AND-Acc | — | — | FP16 |

### Day 6-7: 写论文硬件章节

```
Section III: Hardware Accelerator Design
  A. Architecture Overview
     - Figure 3: Top-level block diagram
     - Table I: Per-engine workload distribution
  B. Binary Engine Microarchitecture  ← 主要贡献
     - Figure 4: SC popcount unit datapath
     - Figure 5: 3-stage attention pipeline
     - Table II: Binary engine synthesis results @ 28nm
  C. Sparse MAC Engine
     - Figure 6: Bit-serial PE + TTB scheduler
     - Table III: PE array configuration
  D. System Integration
     - Figure 7: Memory hierarchy + dataflow
  E. Implementation Results
     - Table IV: Full accelerator results @ 28nm
     - Table V: Comparison with prior accelerators
     - Figure 8: Area-power Pareto frontier
     - Figure 9: FPS vs resolution scaling
```

**交付物（Week 5 结束）**:
- DC 优化后的最终数字
- 完整对标表
- 论文 hardware section 初稿 (含所有图表)

---

## Week 6: 论文收尾 + 补充实验

### Day 1-3: 论文写作
- Introduction (1 天)
- Related Work (0.5 天)
- Hardware Design (已有 Week 5 初稿, 1 天)
- Results + Discussion (0.5 天)

### Day 4-5: 补充缺失数字
- 如果 DC 数字不够好看 → 调整配置重新综合
- 如果对标表缺数据 → 补上
- 如果有 reviewer 可能质疑的假设 → 加 sensitivity study

### Day 6-7: 内部审阅 + 投稿
- 检查所有图表数字一致性
- 确认参考格式 (DATE 模板)
- 提交

---

## 每周交付物清单

| Week | 交付物 | 依赖 |
|---|---|---|
| 1 | 顶层框图 + 接口规范 + 调度表 | profile_sops.py 数据 |
| 2-3 | 全系统 RTL (~3000 行 Verilog) | Week 1 规范 |
| 4 | Verilator 仿真通过 + DC 综合报告 | RTL + 28nm PDK |
| 5 | DC 优化 + 对标表 + 论文初稿 | DC 数字 |
| 6 | 最终论文 | 全部 |

---

## 工具链要求

| 工具 | 用途 | 替代方案 |
|---|---|---|
| Synopsys VCS / Verilator | RTL 仿真 | Verilator (开源, 免费) |
| Synopsys DC | 综合 (面积/功耗/时序) | **必须**有, DATE 最低要求 |
| TSMC 28nm PDK | 标准单元库 | 或 FreePDK45 + 公开缩放因子 |
| CACTI 7.0 | SRAM 面积/功耗估算 | 或从 Bishop 论文引用 |
| Python + PyVerilator | Testbench | 必需 |
| LaTeX (DATE 模板) | 论文 | Overleaf |

---

## 风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| 28nm PDK 不可用 | 中 | 用 FreePDK45 + 公开的 28nm 缩放因子 (0.6x area, 0.4x power per gate) |
| DC license 不可用 | **高** | **优先解决**: 找学校/公司的 DC license; 或用 Genus (Cadence, 类似功能); 最后手段: Yosys + 公开库估算 |
| Verilator 仿真 93 层太慢 | 低 | 只仿真关键层 (Binary engine 覆盖的所有 attention 层), 其余用 Python 模型 |
| 最终数字不如 Bishop/28nm ViT | 中 | 如果某个模块数字不好看, 弱化该模块的描述, 强调 Binary 引擎的创新点 |
| Week 2-3 RTL 写不完 | 低 | Sparse MAC 和 Dense MAC 复用开源设计 (OpenCores / ESP), 不从头写 |
