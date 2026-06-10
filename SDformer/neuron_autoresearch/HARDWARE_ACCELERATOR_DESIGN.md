# SDformerFlow 硬件加速器设计方案

最后更新: 2026-05-29

---

## 一、设计目标与指标

| 指标 | 目标 | 对标 |
|---|---|---|
| 吞吐量 | 30 FPS @ 480×640 | 实时 event 光流 |
| 功耗 | < 5W (边缘) | ASNA-Flow: 7.9mW, Bishop: 6.11x 能效 vs PTB |
| 精度 | AEE < 1.6, AAE < 10 | 当前 H41 SC: AEE=1.62, AAE=9.45 |
| 工艺 | 28nm CMOS (TSMC) | 28nm Spiking ViT: 57.7 TOPS/W |
| 片上 SRAM | < 2 MB | Window 粒度缓存, 对标 SENECA |

---

## 二、当前架构数据流特征

基于 MS_SpikingformerFlowNet_en4 完整 forward pass 分析（975 GFLOPs 逐层拆解）。

### 2.1 FLOPs 分布

| Stage | GFLOPs | 占比 | 计算类型 |
|---|---|---|---|
| Patch Embedding (head+conv+res×2+proj) | 307 | 31.5% | 稀疏 Conv (输入 binary spike) |
| Swin Stage 0 (dim=96, depth=2) | 164 | 16.8% | MLP (binary spike → dense FC) |
| Swin Stage 1 (dim=192, depth=2) | 166 | 17.0% | 同上 |
| Swin Stage 2 (dim=384, depth=6) | 224 | 23.0% | 同上（6 blocks, 最重） |
| Swin Stage 3 (dim=768, depth=2) | 31 | 3.2% | 轻量, token 数少 |
| Residual Bottleneck (2 blocks) | 64 | 6.6% | Dense Conv |
| Decoders (4 levels) + Predictions | 89 | 9.1% | Dense Conv + 双线性上采样 |
| Temporal sum + Interpolation | ~0 | <0.1% | 纯搬运 |

**核心洞察：MLP + Conv 占了 ~90% FLOPs，但输入全是 binary spike（50-80% zero）。**

### 2.2 5 种不同的计算模式

SDformerFlow 不能一刀切——不同 stage 的计算特征完全不同：

```
Stage              计算类型         输入精度        稀疏度       最优硬件
─────────────────────────────────────────────────────────────────────────────
Voxel 编码         scatter-add     FP16            60% zero     Event Scatter 单元
Patch Embed Conv   密集3×3卷积      binary spike    50-80% zero  Sparse Conv 引擎
PSN 时间混合       小矩阵乘(5×5)    FP16            0%           Dense MAC 阵列
QKFormer Attention AND-PopCount     2-bit ternary   67% zero     Binary 引擎
FFN MLP            密集FC (dim×4)   binary spike    50-80% zero  Sparse MAC 引擎
Decoder Conv       密集反卷积        binary spike   50-80% zero  Sparse Conv 引擎
SC/TX Gate         PopCount+Shift   2-bit ternary   67% zero     Binary 引擎
Output Interp      双线性插值        FP16            0%           Dense MAC
```

### 2.3 注意力是 O(N) 不是 O(N²)

```
标准 ViT:   attn = softmax(Q@K^T/√d) @ V    ← N×N 矩阵, 98²×32 = 307K MAC/head
SDformer:   attn = K * sn2( Σ_channel(Q) )   ← element-wise, 98×32 = 3K MAC/head
                                              省 100 倍，且不需要矩阵乘法器
```

**QKFormer 本身就不需要矩阵乘法器。** 硬件上就是一个 reduce-sum + 比较器(SN) + element-wise MUX。

### 2.4 稀疏度全景

| 位置 | 稀疏类型 | 零值比例 | 硬件机会 |
|---|---|---|---|
| Voxel grid 输入 | 自然稀疏 | 30-60% zero | 压缩存储, 跳过零像素 |
| 每个 SN 之后 | Binary spike | **50-80% zero** | Bit-serial MAC 或 zero-skipping |
| ATLIF 三值输出 | Ternary {-T,0,+T} | **~67% zero** | 2-bit 编码, 比较器替代乘法 |
| Q-sum gate | Binary spike (sn2_q) | 50-80% zero | MUX 跳过整个 K channel |

---

## 三、异构多引擎架构总览

```
                    ┌──────────────────────────────────────┐
  事件流 ──────────►│  Event Scatter 单元                    │
                    │  - 增量 VoxelGrid 更新 (ASNA-Flow)    │
                    │  - 双线性 scatter-add, FP16            │
                    │  - 输出: [10, 2, 480, 640]            │
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │  Sparse Conv 引擎 (Patch Embed)       │
                    │  - 零跳过 MAC 阵列 (16×16 PE)          │
                    │  - 权重 FP16 / INT8                    │
                    │  - 输入 1-bit spike → 2-5x 加速       │
                    └──────────────┬───────────────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       ▼                       │
           │  ┌─────────────────────────────────────────┐  │
           │  │  Sparse MAC 引擎 (MLP + FFN)             │  │
           │  │  - 128 PE, bit-serial AND-Accumulate     │  │
           │  │  - 输入: 1-bit spike, 权重: INT8          │  │
           │  │  - Bishop TTB 时空 bundle 调度            │  │
           │  │  - 50-80% 输入稀疏 → 2-5x 有效吞吐       │  │
           │  └─────────────────────────────────────────┘  │
           │                                               │
           │  ┌─────────────────────────────────────────┐  │
           │  │  Binary 引擎 (Attention + SC/TX Gate)     │◄─┤
           │  │  - AND-PopCount 单元 (FireFly-T 同款)     │  │
           │  │  - 2-bit 三值输入，6-bit popcount 输出     │  │
           │  │  - Shiftmax: 2ˣ LUT 实现                  │  │
           │  │  - 每 token 3 周期完成                    │  │
           │  │  - 零乘法器、零浮点单元                    │  │
           │  └─────────────────────────────────────────┘  │
           │                                               │
           │  ┌─────────────────────────────────────────┐  │
           │  │  Dense MAC 阵列 (PSN + Decoder + Output)  │  │
           │  │  - 32×32 systolic array @ 500MHz          │  │
           │  │  - FP16 乘加                               │  │
           │  │  - 时分复用: PSN(5×5) + decoder + output   │  │
           │  └─────────────────────────────────────────┘  │
           │                                               │
           └───────────────────────────────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │  光流输出 [2, 480, 640]               │
                    └──────────────────────────────────────┘
```

### 为什么四引擎而不是单一架构

| 引擎 | 处理的数据 | 精度需求 | 为何专用 |
|---|---|---|---|
| **Binary** | Q/K sign, gate | 2-bit + 6-bit int | AND-PopCount 比 FP16 MAC 省 ~1000x 功耗 |
| **Sparse MAC** | MLP/Conv (90% FLOPs) | 1-bit in × INT8 weight | 零跳过收益最大, bit-serial 省 50x |
| **Dense MAC** | PSN, Decoder, Output | FP16 | 稀疏度低, 需要全精度 |
| **Event Scatter** | 事件 → Voxel | FP16 | scatter-add 不是 MAC, 需要专用 scatter 单元 |

如果全部用 Dense MAC 阵列做，功耗至少高 5-10x。

---

## 四、Binary 引擎详细设计（核心创新）

这是 SDformerFlow 最有硬件特色的模块，对标 FireFly-T 的 AND-PopCount 引擎。

### 4.1 QKFormer Attention 硬件化

QKFormer 公式：`attn = K * sn2( Σ_channels(Q) )`

```verilog
// 3 级流水线, 每 token 3 周期

// Stage 1: Q-sum (reduce over head_dim=32)
q_sum[5:0] = popcount(q_sign[31:0])
// 32 个 2-bit AND → 6-bit carry-save adder tree
// 硬件: ~32 LUT + 5级加法器 → 1 周期

// Stage 2: SN gate (threshold compare)
gate = (q_sum >= thresh) ? 1'b1 : 1'b0
// 硬件: 1 个比较器 → 1 周期

// Stage 3: K gating (element-wise MUX over 32 channels)
for ch in 0..31:
    attn[ch] = gate ? k[ch] : 2'b00
// 硬件: 32 个 2-bit MUX → 1 周期
```

**跟标准 softmax attention 的面积/功耗对比（per head）：**

| | 标准 Q@K^T Attention | SDformer QKFormer | 改善 |
|---|---|---|---|
| 计算量 | 307K MAC | 32 AND + popcount | — |
| 硬件单元 | FP16 MAC 阵列 | AND-PopCount | — |
| 延迟 | ~100 周期 | **3 周期** | **33x** |
| 面积 | ~5000 LUT | **~500 LUT** | **10x** |
| 功耗 @ 500MHz | ~5 mW | **< 0.1 mW** | **50x** |

### 4.2 SC Gate 硬件化

SC 公式：`score_i = Σ_d sign(q_i,d) × sign(k_i,d) / head_dim`

```verilog
// 2-bit ternary encoding: {sign, active}
// sign: 0=positive, 1=negative
// active: 0=silent, 1=firing

// 逐维度比较 (32路并行)
for d in 0..31:
    both_active = q_active[d] & k_active[d]          // 1 LUT
    same_sign   = ~(q_sign[d] ^ k_sign[d])            // 1 LUT (XNOR)
    match[d]    = both_active & same_sign             // 1 LUT
    conflict[d] = both_active & ~same_sign            // 1 LUT

// PopCount (carry-save adder tree)
score = popcount(match) - popcount(conflict)          // [-32, 32], 6-bit
// 总延迟: 2-3 cycles

// Shiftmax: 2^score 查表
gate = lut_2pow[score + 32] / lut_ceil_log2_sum
// 2 次 8-bit LUT 查表 + 1 次除法
// 延迟: 1 cycle
```

**关键：SC gate 全程无乘法，只有 AND / XOR / PopCount / LUT。** 这是 Bishop ISCA 2025 和 28nm Spiking ViT 论文中验证过的最高效注意力实现。

### 4.3 SC 为什么比 TX 更适合硬件

```
TX:  score = same×1 + silence×0.02 - opposite×0.25 - single×γ
     → 需要 FP16 乘法器 (0.02, 0.25 是浮点数)
     → 4 路比较 + 3 次浮点乘 + 累加
     
SC:  score = Σ sign(q)×sign(k) / d
     → 纯 AND/XOR/Popcount, 零乘法
     → 1 路比较 + 1 次 popcount + 1 次 LUT
```

**SC 的硬件面积比 TX 小 ~5x，功耗低 ~10x。** 这是论文硬件章节的核心卖点。

---

## 五、Sparse MAC 引擎（MLP + Conv, 占 90% FLOPs）

### 5.1 应用 Bishop 的 Token-Time Bundle (TTB)

```
TTB 调度策略 (Bishop ISCA 2025):

1. 打包: 将同一 window 内 98 tokens × 2 timesteps 打包成 TTB
2. 分类: TTB Stratifier 按稀疏度路由
   - 高稀疏 (>80% zero) → Sparse Core (bit-serial, zero-skipping)
   - 低稀疏 (<80% zero) → Dense Core (FP16 MAC)
3. 配置: 8 Dense Cores + 32 Sparse Cores (Bishop 原文比例)

SDformerFlow 适配:
  - MLP 输入: sn1 → binary spike, ~60-80% zero → 几乎全走 Sparse Core
  - FFN 输入: sn2 → binary spike, 同上
  - Conv 输入: SN → binary spike, 同上
  - → 85%+ 的计算走 Sparse Core
```

### 5.2 Bit-Serial AND-Accumulate PE

```
每个 Sparse PE:
  输入: 1-bit spike × 8-bit INT weight
  操作: for bit in 0..7:
          partial += (spike & weight[bit]) << bit
  8 周期完成 8b×1b 乘法 (vs FP16 MAC 的 1 周期)
  但功耗只有 FP16 MAC 的 ~1/50

  考虑 60% 输入稀疏 (zero-skipping):
  → 只有 40% 的 spike 触发计算
  → 有效吞吐 = 8/8 × 1/0.4 = 1 周期/spike (等效)
  → 实际加速 2-3x vs dense FP16
```

### 5.3 能效对比（28nm CMOS 实测数据, 来源: 28nm Spiking ViT 论文）

| 操作 | 功耗/OP | 相对 |
|---|---|---|
| FP16 MAC | ~3.7 pJ | 1x |
| INT8 MAC | ~0.4 pJ | **9.3x** |
| 1b × 8b AND-Acc (稀疏 60%) | ~0.03 pJ | **123x** |
| 1b 比较器 (阈值判断) | ~0.005 pJ | **740x** |

---

## 六、ATLIF 三值神经元硬件化

### 6.1 从软件到硬件

```python
# 软件 (PyTorch)
h = W @ x + b                                    # FP32 矩阵乘
ternary = (h >= thre) - (h <= -neg_thre)          # 三值比较
out = ternary * thre                              # {−thre, 0, +thre}

# 硬件 (28nm)
# Stage 1: MAC 阵列计算 h = Wx+b (共享 Dense MAC)
# Stage 2: 比较器阵列
pos = (h >= thre)    # 1 周期, 1 个比较器/通道
neg = (h <= -thre)   # 1 周期, 1 个比较器/通道
# Stage 3: 2-bit 编码输出
out = {neg, pos}     # 00=silent, 01=positive, 10=negative
```

**阈值比较器 vs FP16 乘法器：面积差 ~200x, 功耗差 ~740x。**

### 6.2 PSN 时间混合的轻量实现

PSN 只有 5×5=25 个权重（T=5），所有时间步共享。可以做成：

```
PSN 计算单元 (每个 SN 实例):
  - 25 个 FP16 寄存器 (存 W 矩阵)
  - 1 个 5×1 向量乘法器 (5 个 MAC)
  - 5 周期完成一次 PSN forward (5 次 5x1 向量乘)
  - 所有 SN 实例共享这 25 个寄存器和 5 个 MAC (时分复用)

  一个 window 内有 ~10 个 SN 实例
  → 10 × 5 周期 = 50 周期的 PSN 开销
  → vs MLP 的 ~1000 周期, 可忽略
```

---

## 七、片上存储架构

### 7.1 三级层次

```
Level 0: 寄存器堆 (RF) — 每 PE 本地
  - 延迟: 1 cycle, 功耗: ~12 fJ/bit
  - 用途: PSN 权重(25×FP16), Psum 累积, 阈值寄存器

Level 1: Window SRAM (片上, 512 KB)
  - 延迟: 2-3 cycles, 功耗: ~200 fJ/bit
  - 一个 window 全激活存储:
    * Q signs: 98 tokens × 32(head_dim) × 2-bit = 784 bytes/head
    * 24 heads 全并行: 784B × 24 = 18.8 KB
    * K signs: 同上, 18.8 KB
    * 中间激活: ~150 KB
    * 双缓冲 (next window prefetch): ×2
    * 总计: ~400 KB < 512 KB ✓

Level 2: DRAM + Weight Prefetch Buffer (片外)
  - 完整模型权重: ~55M params × 平均 1 byte (量化后) ≈ 55 MB
  - 片上 weight buffer: 128 KB (当前 layer)
  - 按 layer 顺序 prefetch, 隐藏 DRAM 延迟 (~50ns)
```

### 7.2 Decoder 存储优化（关键）

Decoder Stage 3 的中间激活最大：

```
(5, 1, 194, 240, 320) × 2B (FP16) = 149 MB
→ 片上放不下

解决方案: Stripe-based 流式处理

  240 行分成 8 个 stripe, 每个 30 行:
  (5, 1, 194, 30, 320) × 2B = 18.6 MB/stripe
  
  仍偏大 → 进一步: 逐 layer 流式
  decoder conv + pred + upsample → 不存完整中间激活
  参考 SENECA 的 event-driven depth-first 策略
```

### 7.3 Window Prefetch 流水线

```
        Load W0      Load W1      Load W2      Load W3
DRAM:   [====]       [====]       [====]       [====]
        Compute W0   Compute W1   Compute W2   Compute W3
PE:     [========]   [========]   [========]   [========]

双缓冲: 计算 Window N 时, DRAM 预取 Window N+1
DRAM 延迟 (~100 cycles) 完全隐藏在计算中 (>500 cycles/window)
```

---

## 八、光流专属优化

### 8.1 从 ASNA-Flow 学到的

| ASNA-Flow 技术 | SDformerFlow 适配 |
|---|---|
| 事件驱动异步处理 (无帧概念) | VoxelGrid 增量更新: 只计算新增/移除的事件贡献 |
| 空间局部性利用 | Window attention 天然利用 7×7 局部性 |
| 0.3 pJ/SOP @ 28nm, 7.9mW | 目标: SC gate < 0.01 pJ/SOP (AND-only) |
| 104 FPS @ 低功耗 | SDformer 更重 (975 vs ~10 GFLOPs), 目标 30FPS@5W |

### 8.2 从 SENECA 学到的

| SENECA 技术 | SDformerFlow 适配 |
|---|---|
| 3-level memory (RF/SRAM/HBM) | 直接采用 |
| Event-driven depth-first conv | Encoder 前几层可用 (输入 event-sparse) |
| Spike grouping (50% memory reduction) | TTB bundling 等效 |
| Int4/Int8/BF16 混合精度 | ATLIF 天然 2-bit + 权重 INT8 |

### 8.3 时间冗余利用

```
连续帧之间的事件流高度相关:
  帧 t 的 VoxelGrid:   30% 非零像素
  帧 t+1 的 VoxelGrid: 28% 非零 (其中 ~80% 像素与 t 相同)

增量 VoxelGrid 更新:
  - 维护上一帧的 grid 状态
  - 新帧只计算新增/移除事件的贡献
  - 跳过未变化区域
  - 预期: 1.5-2x 输入处理加速
```

---

## 九、精度-能效 Tradeoff

### 9.1 逐组件精度方案

| 组件 | 当前精度 | 目标精度 | 实现方式 | AEE 影响 |
|---|---|---|---|---|
| VoxelGrid 输入 | FP32 | **FP16** | 半精度 scatter-add | < +0.01 |
| Q/K 激活 (ATLIF) | FP32 | **2-bit ternary** | 原生 {-thre,0,+thre} | 0 (已是三值) |
| SC score | FP32 | **6-bit int** | PopCount 天然整数 | < +0.01 |
| K carrier | FP32 | **8-bit int** | Post-training 量化 | **+0.03-0.07** |
| Shiftmax gate | FP32 | **8-bit fixed** | LUT 输出 | < +0.01 |
| MLP/Conv 权重 | FP32 | **INT8** | QP-SNN 量化 + 2ep 微调 | < +0.02 |
| PSN 权重 (W,b) | FP32 | **FP16** | 5×5 小矩阵, 精度敏感 | < +0.02 |
| Decoder 权重 | FP32 | **INT8** | Post-training 量化 | < +0.02 |
| Flow 输出 | FP32 | **FP16** | 半精度 | < +0.01 |
| **累积** | | | | **+0.06-0.14** |

### 9.2 最大风险点

**K carrier 的 8-bit 量化。** K 既是 attention score 的输入 (sign 用于 popcount) 又是 value carrier (幅度用于最终输出)。当前 H41 把 K 也过 ATLIF 三值化，等于已经是最激进的 2-bit 量化。如果保留 K 的幅度信息（8-bit），反而可能提升精度。

建议：K carrier 用 8-bit INT，Q 和 K_xnor 继续用 2-bit 三值。**这比当前 H41 的三值 K 精度更高，不引入新风险。**

---

## 十、对标 SOTA 硬件加速器

| | Bishop | FireFly-T | 28nm SpkViT | **本设计** |
|---|---|---|---|---|
| 出处 | ISCA 2025 | 2025 | 2025 | — |
| 工艺 | 28nm | FPGA | 28nm | **28nm** |
| 网络类型 | SpikeBERT | Spiking ViT | Spiking ViT | **SDformerFlow (光流)** |
| 神经元 | LIF | LIF | LIF | **PSN + ATLIF 三值** |
| 注意力实现 | AND-Acc | AND-PopCount | EMA-free SA | **AND-PopCount + SC** |
| 稀疏策略 | TTB + ECP | Multi-lane decoder | Dual-path sparse | **TTB + 三值零跳过** |
| 能效 | 6.11x vs PTB | 1.39x | 57.7 TOPS/W | **预估 40-60 TOPS/W** |
| 支持光流 | ❌ | ❌ | ❌ | **✅** |
| 特点 | 异构核 | 双引擎 | 1b/8b统一阵列 | **三值+SC popcount** |

**本方案的三个独特卖点 (区别于所有现有工作)：**

1. **ATLIF 三值 {−thre, 0, +thre}** — 唯一硬件化三值 SNN 神经元的设计。所有现有加速器都只支持 {0,1} 二值
2. **SC PopCount 注意力** — 纯 AND/XOR/PopCount, 零乘法零浮点。FireFly-T 的 AND-PopCount 只做到部分, 本方案是全流水
3. **Event → Voxel → SNN 端到端光流** — ASNA-Flow 做光流但不是 transformer, Bishop/FireFly-T 是 transformer 但不做光流

---

## 十一、实验验证计划

### Phase 1: 精度验证（软件, 2 周）

```
1. 在 H41 SC 上做逐组件精度消融
2. Q/K 量化 2-bit (已经是) → baseline
3. K carrier INT8 量化 vs FP32 → 测 AEE 影响
4. MLP/Conv 权重 INT8 量化 → 测 AEE 影响
5. Shiftmax 8-bit LUT vs FP32 → 测 AEE 影响
6. 全量化端到端 profile valid816
```

### Phase 2: RTL 关键模块（4 周）

```
Week 1-2: Binary 引擎 (AND-PopCount + Shiftmax LUT)
Week 2-3: Sparse MAC 引擎 (TTB scheduler + bit-serial PE)
Week 3-4: ATLIF 三值比较器阵列 + PSN 小 systolic array
```

### Phase 3: 系统集成（3 周）

```
Week 5-6: 存储子系统 + DMA + Window prefetch
Week 6-7: 顶层集成 + C/RTL cosim
Week 7:   Synopsys DC/PTPX @ 28nm (面积/功耗/时序)
```

### Phase 4: FPGA 原型 (可选, 3 周)

```
Week 8-10: Xilinx VU13P 原型
验证: 实时 30FPS @ 5W 目标是否可达
```

---

## 十二、参考文献

| 论文 | 出处 | arXiv / 链接 | 借鉴要素 |
|---|---|---|---|
| **Bishop** | ISCA 2025 | arXiv:2505.12281 | TTB bundle, 异构 Dense/Sparse core, BSA 训练, ECP 剪枝；**核心 TTB 调度来源** |
| **FireFly-T** | IEEE TC 2026 | ~2505.12771 | Dual-engine (Sparse+Binary), AND-PopCount 注意力, LUT；**最直接 dual-engine 对标** |
| **Spiking Transformer HW in 3D Integration** | ICCAD 2024 | arXiv:2411.07397 | 首个 spiking transformer 3D 加速器；memory-on-logic stacking；空间/时间权重复用 |
| **Hardware Efficient Accelerator for Spiking Transformer (Reconfig Parallel Timestep)** | 2025 | arXiv:2503.19643 | 低功耗 spiking ViT 加速器；parallel timestep 计算；解决 non-spike computation |
| **ASNA-Flow** | IEEE TVLSI 2025 | — | Event-driven 异步光流, 0.3 pJ/SOP, 7.9mW, 104FPS；**Event Scatter + 光流应用直接对标** |
| **SENECA** (Tang/Yousefzadeh et al.) | 2023 (后续 NN 2025 对比) | PMC10326429 等 | 3-level memory (RF/SRAM/shared), event-driven depth-first, spike grouping；RISC-V 灵活 NPE |
| **Prosperity** | HPCA 2025 | arXiv:2503.03379 | Product sparsity, TCAM matching；spiking transformer 加速 |
| **Phi** | ISCA 2025 | arXiv:2505.10909 | Pattern-based hierarchical sparsity；两级稀疏用于高效 SNN 计算 |
| **SeaSNN** (FPGA spiking attention) | PeerJ Comput Sci 2025 | — | FPGA spiking channel attention (SECA)；轻量 attention 硬件 + 并行优化；原型验证参考 |
| **SpinalFlow** | ISCA 2020 | — | 经典 SNN 专用数据流 (compressed timestamped sorted spikes)；高度复用 membrane/input/weight |
| **FireFly / FireFly v2 / FireFly-S** | TVLSI 2023 / TCAD / TCAS-I 2024 | — | DSP 优化、spatiotemporal FPGA、dual-side sparsity；reconfigurable overlay |
| **QP-SNN** | ICLR 2025 | — | SVD 通道剪枝 + 4-bit 量化, 零额外训练（算法支撑激进量化） |

---

## 附录A: 全网络形状参考表

| 层 | 输入形状 | 输出形状 | 窗口数 | Token/窗口 | 主要操作 |
|---|---|---|---|---|---|
| PatchEmbed Head | (5,1,4,480,640) | (5,1,48,480,640) | — | — | Conv2d 3×3 |
| PatchEmbed Down | (5,1,48,480,640) | (5,1,96,240,320) | — | — | Conv2d 3×3 s=2 |
| PatchEmbed Res ×2 | (5,1,96,240,320) | (5,1,96,240,320) | — | — | MS_ResBlock ×2 |
| PatchEmbed Proj | (5,1,96,240,320) | (5,1,96,120,160) | — | — | Conv2d 3×3 s=2 |
| Swin Stage 0 | (1,5,120,160,96) | (1,5,120,160,96) | 1,242 | 98 | W/SW-MSA + MLP ×2 |
| Patch Merge 0→1 | (1,5,120,160,96) | (1,192,5,60,80) | — | — | 4-way split + Linear |
| Swin Stage 1 | (1,5,60,80,192) | (1,5,60,80,192) | 324 | 98 | W/SW-MSA + MLP ×2 |
| Patch Merge 1→2 | (1,5,60,80,192) | (1,384,5,30,40) | — | — | 4-way split + Linear |
| Swin Stage 2 | (1,5,30,40,384) | (1,5,30,40,384) | 90 | 98 | W/SW-MSA + MLP ×6 |
| Patch Merge 2→3 | (1,5,30,40,384) | (1,768,5,15,20) | — | — | 4-way split + Linear |
| Swin Stage 3 | (1,5,15,20,768) | (1,5,15,20,768) | 27 | 98 | W/SW-MSA + MLP ×2 |
| Bottleneck ×2 | (5,1,768,15,20) | (5,1,768,15,20) | — | — | MS_ResBlock ×2 |
| Decoder 0 | (5,1,1536,15,20) | (5,1,2,30,40) | — | — | SkipCat + Up + Conv5×5 |
| Decoder 1 | (5,1,770,30,40) | (5,1,2,60,80) | — | — | SkipCat + Up + Conv5×5 |
| Decoder 2 | (5,1,386,60,80) | (5,1,2,120,160) | — | — | SkipCat + Up + Conv5×5 |
| Decoder 3 | (5,1,194,120,160) | (5,1,2,240,320) | — | — | SkipCat + Up + Conv5×5 |
| Output | (5,1,2,H,W)×4 | (1,2,480,640)×4 | — | — | Temporal sum + interpolate |
