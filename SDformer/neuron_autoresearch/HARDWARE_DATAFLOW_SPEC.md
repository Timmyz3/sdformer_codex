# SDformerFlow 全网络数据流硬件规格

2026-05-29 | H41 SC S012C epoch27 | 目标工艺 28nm

---

本文档跟踪从事件流输入到光流输出的**完整数据流动**。每一步定义：做什么计算、哪个硬件模块负责、张量形状、接口信号、状态变量。

---

## 零、全局约定

### 数据类型

| 软件类型 | 硬件位宽 | 编码 |
|---|---|---|
| FP32 weight/activation | **FP16** (1-5-10) | IEEE 754 half |
| INT8 weight (量化后) | **INT8** | 2's complement |
| Binary spike (PSN输出) | **1-bit** | 0=silent, 1=fire |
| Ternary spike (ATLIF输出) | **2-bit** | 00=silent, 01=+thre, 10=-thre |
| SC popcount score | **6-bit int** | [-32, +32] |
| Shiftmax gate | **8-bit fixed** | [0, 1], 8 fractional bits |

### 坐标系统

```
T: 时间步 [0, 4] (5 steps in PSN, 2 in window attention)
B: batch [0] (推理时固定为1)
C: 通道
H, W: 空间 (随 stage 变化)
N: token 数 = T × H_patch × W_patch
```

### 存储层次

| 层次 | 大小 | 位宽 | 延迟 | 用途 |
|---|---|---|---|---|
| DRAM (HBM2) | 8 GB | 256-bit | ~100 cyc | 权重 + 完整激活 |
| Weight Buffer (SRAM) | 256 KB | 256-bit | 2 cyc | 当前层权重 |
| Window SRAM (双bank) | 512 KB | 256-bit | 2 cyc | 窗口内Q/K/V + 中间激活 |
| RF (per PE) | 2 KB | 16-bit | 1 cyc | PSN 权重 + 累积 psum |

---

## 一、Stage 0: Event Scatter → VoxelGrid

### 做什么

事件相机输出流 `(x, y, t, p)` → 双线性 scatter 到 10-bin × 2-polarity 的体素网格。

### 硬件模块: `event_scatter_unit`

### 输入接口

```systemverilog
// 事件流 (从 DVS/事件相机前端来)
interface event_stream;
    logic [9:0]  x;          // 0..639
    logic [8:0]  y;          // 0..479  
    logic [23:0] t;          // 时间戳 (us)
    logic        p;          // 极性 0=OFF 1=ON
    logic        valid;
    logic        ready;      // 背压
endinterface
```

### 内部状态

```systemverilog
// VoxelGrid 双缓冲 (乒乓: 一个在积累, 一个在读出)
logic [15:0] voxel_grid [1:0][9:0][1:0][479:0][639:0];  
// [pingpong][bin][pol][y][x], FP16
// 大小: 2 × 10 × 2 × 480 × 640 × 2B = 24.6 MB → 必须放 DRAM

// 当前网格时间窗口
logic [23:0] window_start_t;     // 当前积累窗口起始时间
logic [23:0] window_duration;    // 窗口长度 (通常 50ms)

// 控制
logic        pingpong_idx;       // 0=正在积累, 1=可以读出
logic        grid_ready;         // 当前网格积累完成, 可读出
```

### 计算: 双线性 Scatter

```
对每个事件 (x, y, t, p):
  1. 确定时间 bin: t_bin = (t - window_start_t) / window_duration × 10
  2. 确定 4 个空间角: (x0,y0), (x1,y0), (x0,y1), (x1,y1)
  3. 计算双线性权重: w00, w10, w01, w11
  4. Scatter-add (原子加):
     grid[t_bin][p][y0][x0] += w00 × (p ? +1 : -1)
     grid[t_bin][p][y1][x0] += w10 × (p ? +1 : -1)
     grid[t_bin][p][y0][x1] += w01 × (p ? +1 : -1)
     grid[t_bin][p][y1][x1] += w11 × (p ? +1 : -1)
```

### 硬件实现

4 路并行的双线性插值器 + FP16 加法器。因为 scatter-add 是随机地址写入，需要通过小 cache 合并对同一地址的多次写入，然后批量写回 DRAM。

```
event_scatter_unit 延迟: ~100K cycles/frame (1M events, 4 events/cycle)
event_scatter_unit 面积: ~0.1 mm² (主要是 FP16 加法器 + 地址计算)
```

### 输出

```
VoxelGrid: [10, 2, 480, 640] × FP16
读出接口:
  output logic [15:0] grid_data;         // FP16
  output logic [18:0] grid_addr;         // {t_bin[3:0], pol, y[8:0], x[9:0]}
  output logic        grid_valid;
```

---

## 二、Stage 1: Patch Embedding（输入编码）

### 2.1 Temporal Folding（纯数据搬运）

**做什么**: 把 `[10 bins, 2 pol, 480, 640]` 重排成 `[T=5, C=4, 480, 640]`

```
bin 0, pol 0 → step 0, ch 0    bin 1, pol 0 → step 1, ch 0
bin 0, pol 1 → step 0, ch 1    bin 1, pol 1 → step 1, ch 1
...
bin 8, pol 0 → step 4, ch 0    bin 9, pol 0 → step 5≡0, ch 2
bin 8, pol 1 → step 4, ch 1    bin 9, pol 1 → step 5≡0, ch 3
```

**硬件**: 不需要专门模块。DMA 在从 DRAM 读 VoxelGrid 时按照目标步长重排地址。或者 Event Scatter 直接按 folded 格式写到 DRAM。

**状态变量**: 无。纯地址映射。

### 2.2 Head Convolution

**做什么**: `Conv2d(Cin=4, Cout=48, k=3, s=1, p=1)` → BN → PSN

```
输入:  [T=5, B=1, C=4,  H=480, W=640]
权重:  [48, 4, 3, 3]                    # 1728 个 FP16
BN:    [48] gamma, [48] beta             # 96 个 FP16
PSN:   [5, 5] weight, [5, 1] bias       # 30 个 FP16

计算:
  for each t in 0..4:
    h = Conv2d(x[t], weight)             # 5.3 GFLOPs, 密集计算
    h = BN(h, gamma, beta)
    spike[t] = PSN(h, psn_w, psn_b)      # 5×5 matmul → {0,1} binary

输出:  [T=5, B=1, C=48, H=480, W=640] × 1-bit binary
稀疏度: 91.3% (真实数据: PSN firing = 8.7%)
```

**硬件模块**: SparseMAC Engine

```systemverilog
// 层配置字
typedef struct packed {
    logic [15:0] cin, cout;           // 4, 48
    logic [3:0]  kernel;              // 3
    logic [1:0]  stride;              // 1
    logic        has_bn;              // 1
    logic        has_psn;             // 1
    logic [31:0] weight_addr;         // DRAM 地址
    logic [31:0] input_addr;
    logic [31:0] output_addr;
    logic [15:0] H, W;                // 480, 640
} layer_config_conv_t;
```

**接口**:
```
SparseMAC input:  [1-bit spike stream] + [INT8 weight stream]
SparseMAC output: [1-bit spike stream]
PSN unit input:   [FP16 membrane potential]
PSN unit output:  [1-bit spike]
```

### 2.3 Downsampling Convolution

```
Conv2d(Cin=48, Cout=96, k=3, s=2, p=1) → BN
输入:  [T=5, C=48, H=480, W=640] × 1-bit (SN 输出)
输出:  [T=5, C=96, H=240, W=320] × FP16 (BN 输出, 还没过 SN)
FLOPs: 31.9 GFLOPs theoretical → 2.8 GFLOPs effective (91.3% sparsity)
```

### 2.4 Residual Blocks ×2（Patch Embed 内）

```
每个 MS_ResBlock:
  SN1 → Conv2d(96,96,3×3) → BN → SN2 → Conv2d(96,96,3×3) → BN → ADD(shortcut)

Block 0:
  conv1 输入: SN_head 输出 (firing 50%*)       → 63.7→31.9 GFLOPs
  conv2 输入: SN1 输出 (firing 20%)            → 63.7→12.8 GFLOPs

Block 1:
  conv1 输入: SN2_block0 输出 (firing 7.7%)    → 63.7→4.9 GFLOPs
  conv2 输入: SN1_block1 输出 (firing 24.5%)   → 63.7→15.6 GFLOPs
```

*注: patch_embed.conv.sn 没有在 profile CSV 里，取默认 50%。

### 2.5 Projection Convolution

```
Conv2d(96, 96, k=3, s=2, p=1) → BN
输入:  [T=5, C=96, H=240, W=320] × 1-bit (firing 7.1%)
输出:  [T=5, C=96, H=120, W=160] × FP16
FLOPs: 15.9→1.1 GFLOPs

然后 permute: (T,B,C,H,W) → (B,C,T,H,W)
→ [B=1, C=96, T=5, H=120, W=160]
```

**Patch Embed 总结**:
```
总 FLOPs:  310 theoretical → 70 effective GFLOPs
总权重:    ~0.3 MB
中间激活峰值: 5 × 96 × 480 × 640 × 1bit = 14.7 MB (head SN 输出)
```

---

## 三、Stage 2: Swin Transformer Stages

这是循环结构。4 个 stage，每个 stage 有 depth 个 block。先看**一个 block** 的完整数据流。

### 3.1 单个 Swin Block 的完整数据流

以一个 Stage 0 Block 0 为例（dim=96, heads=3, H=120, W=160）。

```
输入: [B=1, H=120, W=160, T=5, C=96]
      ↓
  ┌──────────────────────────────────────┐
  │  Window Partition                     │  纯地址重排
  │  pad → [1, 126, 161, 5, 96]          │  H 补到 7 倍数
  │  window_partition(window=(2,7,7))     │  T 补到 2 倍数
  │  → [1242 windows, each (2,7,7,96)]   │
  └──────────────────┬───────────────────┘
                     │ 每个 window: [T'=2, tokens=49, C=96]
                     ↓
  ┌──────────────────────────────────────┐
  │  QKFormer Attention                   │  ← Binary Engine
  │                                       │
  │  Step A: proj_sn(x)                   │  SN: [2,49,96]→1-bit binary
  │          firing = 22.4%               │
  │                                       │
  │  Step B: linear_q → BN → SN_q          │  SparseMAC (1-bit input)
  │          [2,49,96]@[96,96]→[2,49,96] │  然后 SN: → 1-bit binary
  │          firing (sn_q) = 1.8%         │  极端稀疏!
  │                                       │
  │  Step C: linear_k → BN → SN_k          │  同上
  │          + pos_encoding               │  pos_enc shape: [2,1,49,96]
  │          firing (sn_k) = 4.4%         │
  │                                       │
  │  Step D: Q-sum gate                   │  ← Binary Engine (核心)
  │          q = reshape(q, [2,3,49,32])  │  reshape to multi-head
  │          k = reshape(k, [3,98,32])    │  T 和空间合并 = 2×49=98 tokens
  │          att_token = sum(q, dim=-1)   │  求和 over head_dim=32
  │          gate = sn2_q(att_token)      │  → 1-bit binary
  │                                       │
  │  Step E: K gating                     │  ← MUX, 无乘法
  │          attn = k * gate              │  element-wise multiply
  │          [3,98,32] = [3,98,32] * gate │  gate shape: [3,98,1]
  │                                       │
  │  Step F: attn_sn + proj               │  SparseMAC
  │          attn_sn(attn)  firing=0%!    │  ← 完全死掉! attn全零
  │          proj = Linear(96,96)@attn     │  处理全零输入, 浪费
  │          proj_bn                      │
  └──────────────────┬───────────────────┘
                     │
  ┌──────────────────▼───────────────────┐
  │  SEW Add (残差连接)                   │  element-wise FP16 add
  │  x = x + attn_out                     │  ← DenseMAC (FP16)
  └──────────────────┬───────────────────┘
                     │
  ┌──────────────────▼───────────────────┐
  │  MLP (FFN)                            │  ← SparseMAC
  │                                       │
  │  Step G: sn1(x)                       │  SN: → 1-bit binary
  │          firing = 11.6%               │
  │                                       │
  │  Step H: fc1 = Linear(96,384)@sn1_out │  1-bit × INT8 weight
  │          → BN1                        │  7.1→0.8 GFLOPs effective
  │                                       │
  │  Step I: sn2(bn1_out)                 │  SN: → 1-bit binary
  │          firing = 1.0%                │  极端稀疏!
  │                                       │
  │  Step J: fc2 = Linear(384,96)@sn2_out │  7.1→0.1 GFLOPs effective
  │          → BN2                        │  99% zero-skipping!
  └──────────────────┬───────────────────┘
                     │
  ┌──────────────────▼───────────────────┐
  │  SEW Add (残差连接)                   │  element-wise FP16 add
  │  x = x + mlp_out                      │
  └──────────────────┬───────────────────┘
                     │
  ┌──────────────────▼───────────────────┐
  │  Window Reverse + Unpad              │  纯地址重排
  │  → [1, 120, 160, 5, 96]             │
  └──────────────────────────────────────┘

输出: [B=1, H=120, W=160, T=5, C=96]
```

### 3.2 Window Attention 硬件实现细节

**窗口划分**:

```
H_pad = ceil(H/7)*7 = 126, W_pad = ceil(W/7)*7 = 161, T_pad = ceil(T/2)*2 = 6

nW = (T_pad/2) × (H_pad/7) × (W_pad/7) = 3 × 18 × 23 = 1242

每个 window:
  T_window = 2, H_window = 7, W_window = 7
  N_tokens = T_window × H_window × W_window = 98
  C = dim (96/192/384/768 随 stage 变)
```

**Binary Engine 在一个 window 上的操作序列**:

```
Cycle 0: 从 Window SRAM 读 Q_sign [N_tokens=98, head_dim=32, 2-bit]
         从 Window SRAM 读 K_sign [N_tokens=98, head_dim=32, 2-bit]
Cycle 1: 24 heads 并行 AND-PopCount
         → match_count [24, 6-bit], conflict_count [24, 6-bit]
Cycle 2: 24 heads 并行 Shiftmax LUT
         → gate [24, N_tokens, 8-bit]
Cycle 3: gate × K → 写回 Window SRAM
```

**关键: 每个 window 只需 3-4 周期。** 1242 个 window × 4 周期 = ~4968 周期完成 Stage 0 的全部 attention。

### 3.3 Q 的极端稀疏问题

真实数据: `sn_q` 的 firing 在 Stage 0 只有 1.8-3.2%, Stage 1 只有 1.0-3.0%。

```
这意味着:
  linear_q 计算了 2.2 GFLOPs 的理论值
  但 SN_q 输出后，98 个 token 中只有 1-3 个有非零 Q
  → K gating: 大部分 token 的 gate=0
  → attn 输出几乎全零 (attn_sn firing = 0%)
  → attn_proj 处理全零输入，完全浪费
```

**硬件优化机会**: 如果 `sn_q` 输出全零，跳过后续的 K gating 和 attn_proj，直接输出全零。

### 3.4 MLP 的 SN2 极端稀疏

真实数据: `sn2` firing = 0.5-9%, 中位数 ~2%。

```
fc2 = Linear(4*dim, dim) @ sn2_out
sn2 只有 2% 非零 → 98% 的 fc2 MAC 是乘零
→ 7.1 GFLOPs theoretical → 0.1 GFLOPs effective
→ 70x zero-skipping speedup
```

**这就是硬件价值最集中的地方。** SparseMAC Engine 的 zero-skipping 在这里收益最大。

### 3.5 各 Stage 参数汇总

```
Stage  dim   H    W    dep heads nW      attn FLOPs/w  MLP FLOPs/block
  0    96  120  160   2    3    1,242       7.3G         14.2G
  1   192   60   80   2    6      324       7.6G         14.2G
  2   384   30   40   6   12       90      15.4G*        14.2G
  3   768   15   20   2   24       27      12.3G*        14.2G
```

*Stage 2 有 depth=6，总 attn = 6×15.4G；Stage 3 因为 token 少，FLOPs 反而小。

### 3.6 Patch Merging（Stage 间降采样）

```
Stage 0→1: dim 96→192, H 120→60, W 160→80
  4-way split: [1,5,120,160,96] → [1,5,60,80,384]
  SN → Linear(384, 192) → BN
  → [1,192,5,60,80]  # B,C,T,H,W 格式

Stage 1→2: dim 192→384, H 60→30, W 80→40
Stage 2→3: dim 384→768, H 30→15, W 40→20
```

**硬件**: SparseMAC Engine（SN 后的 Linear 用 1-bit × INT8）。

---

## 四、Stage 3: Bottleneck + Decoder

### 4.1 Residual Bottleneck（2 × MS_ResBlock）

```
位置: encoder 最后 (Stage 3 输出后)
输入: [T=5, B=1, C=768, H=15, W=20] × 1-bit (来自 encoder 最后 SN)

Block 0: SN1→Conv2d(768,768,3×3)→BN→SN2→Conv2d(768,768,3×3)→BN→ADD
Block 1: 同 Block 0

每 conv: 15.9 GFLOPs theoretical
```

**硬件**: SparseMAC（Conv 输入是 1-bit SN 输出）或 DenseMAC（如果稀疏度低）。

### 4.2 Decoder（4 级上采样）

解码器把 encoder 特征逐级恢复到原分辨率。每级: skip-concat → SN → upsample → Conv5×5 → BN → pred。

```
Decoder 0: 输入 [T=5,C=768,H=15,W=20]
  concat(encoder_stage3, bottleneck_out) → [T=5,C=1536,H=15,W=20]
  SN → upsample(×2) → Conv2d(1536→384,k=5) → BN → [T=5,C=384,H=30,W=40]
  Pred: SN → Conv2d(384→2,k=1) → [T=5,C=2,H=30,W=40]

Decoder 1: 输入 [T=5,C=384,H=30,W=40]
  concat(encoder_stage2, dec0_out, pred0_out) → [T=5,C=770,H=30,W=40]
  ... → [T=5,C=192,H=60,W=80]

Decoder 2: → [T=5,C=96,H=120,W=160]
Decoder 3: → [T=5,C=96,H=240,W=320]
```

**最重的层: Decoder3 Conv5×5 = 357.6 GFLOPs**

```
输入: [T=5, C=194, H=240, W=320] × 1-bit (SN 输出, firing 28.9%)
权重: [96, 194, 5, 5] × INT8
输出: [T=5, C=96, H=240, W=320] × FP16
有效: 357.6 × (1-0.711) = 103.3 GFLOPs (3.5x zero-skipping)

存储压力: 输出 = 5×96×240×320×2B = 73.7 MB ← 必须 stripe 处理
```

**Decoder 存储优化（stripe 流式）**:

```python
# Decoder3 分成 8 个 stripe
stripe_height = 240 // 8  # 30 行/stripe

for stripe in range(8):
    y_start = stripe * 30
    y_end   = y_start + 30 + 4  # +4 for 5×5 conv padding
    
    # 只加载当前 stripe 的输入、权重、skip connection
    # 每 stripe: 5×194×34×320×1bit ≈ 1.3 MB (远小于 73.7 MB)
    # Stripe 间无依赖 → 可以完全流水
```

---

## 五、Stage 4: Output（时间求和 + 插值）

```
4 个预测头:
  pred[0]: [T=5, 2, 30, 40]
  pred[1]: [T=5, 2, 60, 80]
  pred[2]: [T=5, 2, 120, 160]
  pred[3]: [T=5, 2, 240, 320]

每个做:
  flow = sum(pred, dim=0)     # 时间求和 → [1, 2, H, W]
  flow = interpolate(flow, (480, 640))  # 双线性上采样

最终输出: 4 个 [480, 640, 2] 的光流图 (多尺度监督)
```

**硬件**: DenseMAC Engine 的插值器（4 路并行 bilinear interpolator）。

---

## 六、控制 FSM 状态转移

```systemverilog
module control_fsm (
    input  wire clk, rst_n,
    output layer_config_t current_config,
    // Engine 使能
    output logic scatter_en, sparse_en, binary_en, dense_en,
    input  wire  scatter_done, sparse_done, binary_done, dense_done
);
    // 93 层配置 ROM (由编译器从 PyTorch 模型生成)
    layer_config_t layer_rom [0:92];
    
    logic [6:0] layer_idx;  // 0..92
    
    typedef enum logic [2:0] {
        IDLE,
        LOAD_WEIGHTS,    // DMA: DRAM→Weight Buffer
        COMPUTE,         // 等待对应引擎完成
        STORE_OUTPUT,    // DMA: 输出→DRAM (如果是大张量)
        NEXT_LAYER
    } state_t;
    
    state_t state;
    
    always_ff @(posedge clk) begin
        case (state)
            IDLE: begin
                layer_idx <= 0;
                state <= LOAD_WEIGHTS;
            end
            
            LOAD_WEIGHTS: begin
                // 触发 DMA 加载当前层权重
                dma_load(layer_rom[layer_idx]);
                state <= COMPUTE;
            end
            
            COMPUTE: begin
                // 根据 layer_rom[layer_idx].engine 使能对应引擎
                // 等待 done 信号
                case (layer_rom[layer_idx].engine)
                    SCATTER: if (scatter_done) state <= NEXT_LAYER;
                    SPARSE:  if (sparse_done)  state <= NEXT_LAYER;
                    BINARY:  if (binary_done)  state <= NEXT_LAYER;
                    DENSE:   if (dense_done)   state <= NEXT_LAYER;
                endcase
            end
            
            NEXT_LAYER: begin
                if (layer_idx == 92) state <= IDLE;  // 完成
                else begin
                    layer_idx <= layer_idx + 1;
                    state <= LOAD_WEIGHTS;
                end
            end
        endcase
    end
endmodule
```

### 层配置字（Layer Config Word）

```systemverilog
typedef struct packed {
    logic [1:0]  engine;           // 00=Scatter, 01=SparseMAC, 10=Binary, 11=DenseMAC
    logic [1:0]  precision;        // 00=FP16, 01=INT8, 10=1bit_binary, 11=2bit_ternary
    logic [1:0]  sparsity_mode;    // 00=dense, 01=zero_skip, 10=TTB_bundle
    logic [31:0] input_dram_addr;  // 输入张量 DRAM 基地址
    logic [31:0] weight_dram_addr; // 权重 DRAM 基地址 (INT8 或 FP16)
    logic [31:0] output_dram_addr; // 输出 DRAM 基地址
    logic [15:0] input_h, input_w; // 输入空间尺寸
    logic [15:0] output_h, output_w;
    logic [9:0]  cin, cout;        // 通道数
    logic [3:0]  kernel;           // 卷积核大小 (0=Linear)
    logic [1:0]  stride;
    logic        has_bn;
    logic        has_sn;           // 后面是否跟 SN
    logic        has_psn;          // 后面是否跟 PSN
    logic        has_skip;         // 是否有残差连接
    logic [31:0] skip_dram_addr;   // 残差数据地址
    logic [31:0] psn_weight_addr;  // PSN 权重地址 (如果 has_psn)
    logic [15:0] window_t;         // 窗口 T 维度 (0=非窗口操作)
    logic [15:0] window_h, window_w;
    logic [15:0] num_heads;
    logic [15:0] head_dim;
} layer_config_t;  // 总共 ~200 bits = 25 bytes × 93 layers = 2.3 KB ROM
```

---

## 七、全网络执行时间线（一帧推理）

```
Layer  0:  Event Scatter      ████████░░░░░░░░  ~100K cycles
Layer  1:  PE head conv       ░░░███████████░░  ~200K cycles
Layer  2:  PE down conv       ░░░░░██░░░░░░░░░  ~30K cycles (高稀疏)
...
Layer 72:  Decoder3 conv5×5   ░░░░████████████  ~500K cycles (最重单层)
Layer 92:  Interpolate+Output ░░░░░░░░██░░░░░░  ~10K cycles
─────────────────────────────────────────────────
Total: ~5-10M cycles @ 500MHz → 100-200 FPS
```

**瓶颈**: Decoder3 = 357.6GFLOPs theoretical, ~500K cycles。如果要 30FPS@500MHz，Decoder3 压缩到 <1.6M cycles 即可满足。

---

## 八、接下来要做什么

这份规格定义了**每个模块要算什么**。Week 2 进入 RTL 时：

1. **Event Scatter Unit**: 双缓冲 + 4 路 scatter-add pipeline
2. **Window SRAM + Window Partition**: 地址生成器 + 双 bank SRAM
3. **SparseMAC PE Array**: bit-serial MAC + TTB scheduler + zero-skip controller
4. **Binary Engine**: AND-PopCount + Shiftmax LUT + K-gate MUX (3-stage pipeline)
5. **DenseMAC Array**: 标准 systolic array, 复用开源设计
6. **Control FSM**: 93-entry ROM sequencer
7. **Top-level**: AXI4 interconnect, DMA engine, clock/reset

每个模块的接口信号和状态变量已经在上面定义好了。
