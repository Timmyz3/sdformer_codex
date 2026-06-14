# 模块接口划分与协议定义

## 1. 顶层层次结构

```
nts07_top
├── nts07_controller          # 全局 FSM + 稀疏调度
├── event_scatter_unit        # (外设 IP，接口定义见 §2)
├── dma_engine                # (外设 IP)
├── sparse_mac_cluster
│   ├── sparse_mac_pe[127:0]
│   └── psn_unit
├── atlif_encode_cluster      # 全线 SN 共用
│   └── atlif_unified_encode_unit[lanes-1:0]
├── h60_attention_engine      # 全线 encoder 注意力 (12 blocks)
│   ├── ternary_encode_unit   # ternary_en=1 for Q/K
│   ├── tx_sc_score_unit
│   ├── shiftmax_unit
│   └── k_gate_unit
├── legacy_attn_engine        # DEPRECATED — 不综合 (11bc+)
├── dense_mac_array           # Decoder
└── sram_subsystem
    ├── sram_if (feature ping-pong)
    ├── weight_buffer
    └── metadata_sram
```

---

## 2. 标准流接口（AXI-Stream 风格）

```systemverilog
interface nts07_stream #(
    parameter int LANES = 8,
    parameter int DATA_W = 8
);
    logic                  valid;
    logic                  ready;
    logic                  last;
    logic [LANES*DATA_W-1:0] data;
    logic [3:0]            user_stage;   // stage_id
    logic [1:0]            user_engine;  // 00=MAC 01=H60 10=LEGACY
    modport source (output valid, data, last, user_stage, user_engine, input ready);
    modport sink   (input valid, data, last, user_stage, user_engine, output ready);
endinterface
```

---

## 3. 稀疏元数据总线

每个 work item 一条 128-bit descriptor：

| 字段 | 位宽 | 说明 |
|------|------|------|
| `stage_id` | 2 | 0..3 |
| `block_id` | 3 | 0..5 |
| `timestep_id` | 4 | 0..9 |
| `window_id` | 10 | linearized window index |
| `head_id` | 5 | 0..23 max |
| `token_base` | 10 | window 内起始 token |
| `n_tokens` | 7 | 通常 98 |
| `head_dim` | 6 | 通常 32 |
| `timestep_enable` | 1 | 0=skip entire timestep group |
| `window_enable` | 1 | 0=skip window |
| `head_mask` | 24 | packed active heads |
| `token_mask` | 98 | packed (stored in metadata SRAM) |
| `engine_id` | 2 | 强制路由 |
| `neuron_mode` | 1 | 0=binary ATLIF，1=ternary ATLIF → `ternary_en` |

**语义**（与 `hw/docs/interfaces.md` 一致）：

- `timestep_enable=0` → 控制器不发射 DMA/MAC
- `window_enable=0` → 跳过整个 window（TTB skip）
- `head_mask[i]=0` → 跳过 head i 的 H60 计算

---

## 4. H60 引擎局部接口

```systemverilog
module h60_attention_engine #(
    parameter int HEAD_DIM = 32,
    parameter int MAX_TOKENS = 98
)(
    input  wire                     clk,
    input  wire                     rst_n,
    // Config (latched per window)
    input  wire [7:0]               mu_q8,          // μ≈0.05 → 13
    input  wire                     center_scores,
    input  wire                     preserve_mean,
    input  wire [5:0]               n_tokens,
    // Q/K ternary streams
    input  wire                     q_valid,
    output wire                     q_ready,
    input  wire [1:0]               q_ternary [0:HEAD_DIM-1],  // per cycle per ch
    input  wire                     k_valid,
    output wire                     k_ready,
    input  wire [1:0]               k_ternary [0:HEAD_DIM-1],
    input  wire [15:0]              k_orig      [0:HEAD_DIM-1], // for K×gate
    // Output
    output wire                     out_valid,
    input  wire                     out_ready,
    output wire [15:0]              attn_out    [0:HEAD_DIM-1],
    output wire                     out_last
);
```

### 子模块握手

```
ternary_encode → tx_sc_score (TX/SC 并行)
              → shiftmax_unit
              → k_gate (K_orig ⊙ gate)
```

---

## 5. 控制寄存器映射（APB）

| 偏移 | 名称 | R/W | 说明 |
|------|------|-----|------|
| 0x00 | `CTRL` | RW | bit0=start, bit1=soft_rst |
| 0x04 | `STATUS` | R | bit0=idle, bit1=busy, bit2=done |
| 0x08 | `FRAME_H` | RW | 图像高度 |
| 0x0C | `FRAME_W` | RW | 图像宽度 |
| 0x10 | `MU_Q8` | RW | H60 μ 定点（默认 13） |
| 0x14 | `ENGINE_MAP` | RW | **11bc+ 固定**：stage0..3 注意力=`ENG_H60`；或只读常量 |
| 0x18 | `THRESH_QK` | RW | 默认 qk 阈值 FP16 |
| 0x28 | `NEURON_MODE_LUT` | RW | per-layer ternary_en 影子表（调试） |
| 0x1C | `PERF_CYCLES` | R | 上帧总周期 |
| 0x20 | `PERF_H60_CYCLES` | R | H60 引擎周期 |
| 0x24 | `PERF_SKIP_WIN` | R | 跳过 window 计数 |

---

## 6. 存储器映射

| 区域 | 基址 | 大小 | 内容 |
|------|------|------|------|
| `WEIGHT_DRAM` | 0x8000_0000 | 64MB | 全模型 INT8 权重 |
| `FEAT_DRAM` | 0x8400_0000 | 16MB | 中间特征 ping-pong |
| `META_SRAM` | 片上 0x0000 | 4KB | mask + descriptor |
| `WINDOW_SRAM_A` | 片上 0x1000 | 256KB | Q/K tile |
| `WINDOW_SRAM_B` | 片上 0x41000 | 256KB | attn/proj tile |

---

## 7. 软件导出契约（golden 向量）

Python 导出路径（待接 `tools/export_hw_golden.py`）：

```json
{
  "layer": "sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.0.attn",
  "q_ternary": "int8[N,D]",
  "k_ternary": "int8[N,D]",
  "k_orig": "fp16[N,D]",
  "tx_scores": "fp32[N]",
  "sc_scores": "fp32[N]",
  "gate": "fp32[N]",
  "attn": "fp16[N,D]",
  "config": {"mu": 0.05, "center_scores": true, "alpha0": 0.02}
}
```

RTL 仿真对比容差：gate ±1 LSB(Q0.8)，attn ±2 LSB(FP16)。