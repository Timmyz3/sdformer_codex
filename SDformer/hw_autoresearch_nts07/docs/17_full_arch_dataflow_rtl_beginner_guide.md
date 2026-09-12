# NTS-11bc 硬件架构 & 数据流小白完全指南

**读者**：熟悉PyTorch但不熟悉RTL/ASIC的硬件小白
**目标**：从网络数据流到RTL模块接口，一次性掰扯清楚，并提炼DATE 2027会议创新点
**学习口诀**：流 → 块 → 线 → 码（数据流 → 模块 → 接口连线 → RTL代码）

---

## 第一部分：先搞懂「这东西输入是什么、输出是什么、中间在干嘛」

### 0. 任务背景：我们在做什么？

我们做的是一个**芯片**，专门跑「事件相机光流估计」——给你一个事件相机拍的视频，芯片实时算出画面里每个像素往哪动（光流）。

| 名词 | 大白话解释 |
|------|-----------|
| **事件相机** | 不是普通相机拍帧，而是「像素亮度变了才输出一个事件」。输出是一串 `(x, y, 时间t, 极性p=亮/暗)`，像打机关枪一样。一帧约50ms，有~100万个事件。 |
| **光流** | 一个二维向量 `(u, v)` 表示每个像素在画面里往哪动、动多快。比如车往前开，画面里树往后流。最终输出 `[2, H, W]` 两个通道。 |
| **脉冲神经网络(SNN)** | 不像普通CNN用浮点数乘加，神经元只有「点火/不点火」（二值），像生物神经元。省电、快。 |
| **三值脉冲** | 不仅有点火/静默，还有「正脉冲/负脉冲/静默」三种状态，用2bit存，信息更多。 |

---

### 1. 整体大管道：一帧事件怎么变成光流？

按顺序记：

```
事件流(x,y,t,p) 约100万个
      │
      ▼
┌─────────────────┐
│  ① Event Scatter│  把事件按时间/位置/极性累加成体素网格
│  (事件变体素)    │
└─────────────────┘
      │
      ▼
体素网格 [10, 2, H, W]  FP16浮点数
      │  （10个时间bin，2个极性，H×W是画面分辨率）
      ▼
┌─────────────────┐
│  ② Patch Embed  │  卷积 + 二值脉冲神经元 → 切成1-bit脉冲特征
│  (补丁嵌入)      │
└─────────────────┘
      │
      ▼
特征图 [T=10, C=96, 240, 320]  1-bit脉冲
      │
      ▼
┌─────────────────────────────────────────────┐
│  ③ Encoder 编码器 4个Stage (S0→S1→S2→S3)     │
│  每个Stage：多个Swin Block + 下采样           │
│  每个Block：H60注意力 + MLP                  │  ← 芯片最核心的部分！
└─────────────────────────────────────────────┘
      │
      ▼
瓶颈特征 [T=10, C=768, 30, 40]  1-bit
      │
      ▼
┌─────────────────┐
│  ④ Bottleneck   │  2个ResBlock，全二值脉冲
│  (瓶颈)          │
└─────────────────┘
      │
      ▼
┌─────────────────┐
│  ⑤ Decoder      │  上采样 + 跳连 + 卷积，回到原分辨率
│  (解码器)        │  用Dense MAC（稠密乘加，不稀疏）
└─────────────────┘
      │
      ▼
光流输出 [2, H, W]  FP16浮点数
```

---

### 2. 四个Stage的几何参数

知道越往深分辨率越低、通道越多、头越多即可：

| Stage | Block数 | 通道数C | 注意力头数 | 特征图大小 | Window数量 | 谁负责注意力 |
|-------|---------|---------|-----------|-----------|-----------|------------|
| S0 | 2 | 96 | 3 | 240×320 | **800个**（最多！） | H60 |
| S1 | 2 | 192 | 6 | 120×160 | 200 | H60 |
| S2 | 6 | 384 | 12 | 60×80 | 50 | H60 |
| S3 | 2 | 768 | 24 | 30×40 | 13 | H60 |

**Window概念**：Swin Transformer不看整张图，把特征图切成 `7×7` 不重叠的小窗口，每个窗口 `2×7×7=98个token`（时间维T=2折叠进来），**注意力只在窗口内部算**，大大省算力。

---

### 3. 一个Swin Block里发生了什么？

四个Stage都一样，只是尺寸不同：

```
输入 x  [N=98 tokens, D=32通道/头]  1-bit/2-bit脉冲
  │
  ├─→ LayerNorm（归一化，浮点域）
  │
  ├─→ H60注意力（下一节细讲）
  │     │
  │     └─ 输出还是 [N, D]
  │
  ├─→ 残差连接：x = x + attn_out
  │
  ├─→ LayerNorm
  │
  ├─→ MLP（多层感知机）：二值脉冲，走Sparse MAC
  │     结构：sn1(二值) → Linear(C→4C) → sn2(二值) → Linear(4C→C)
  │
  └─→ 残差连接：x = x + mlp_out
        │
        ▼
      输出到下一个Block
```

---

### 4. 重中之重：H60注意力到底在算什么？

标准Transformer注意力是 `softmax(QK^T/√d)V`，要算矩阵乘、指数、除法，**非常费硬件**。我们的H60完全不用这些！

#### H60五步走：

```
输入：Q和K都是三值脉冲 {-1, 0, +1}，每个token 32维
      V是K_orig，是INT16激活值（不是脉冲）

步骤1：算两个分数 TX 和 SC（Q和K有多像）
  ┌─────────────────────────────────────────────────────────┐
  │ TX (α-XNOR相似度)：                                      │
  │   逐通道比：Q[d]和K[d]同正/同负 → 加1分                  │
  │            都是0 → 加α₀≈0.02小奖励                       │
  │            一正一负 → 减β=0.25惩罚                       │
  │            只有一个点火 → 减γ≈0.15小惩罚                  │
  │   → 32个通道统计完，就是TX分数                            │
  │                                                          │
  │ SC (有符号共识)：                                         │
  │   同号 → +1，异号 → -1，有0 → 0                           │
  │   → 32个通道加起来除以32，就是SC分数                       │
  └─────────────────────────────────────────────────────────┘
      │
      ▼
步骤2：融合分数 s = TX + μ·SC，μ≈0.05（从checkpoint冻结，不在线算）
      │
      ▼
步骤3（可选）：行中心化，s[i][j] -= mean(s[i][:])（减去平均分，更稳定）
      │
      ▼
步骤4：Shiftmax归一化（代替softmax！）
  ┌─────────────────────────────────────────────────────────┐
  │ 1. 找这一行最高分 max_s                                   │
  │ 2. 每个分数减max：shifted = s - max_s（全变≤0）            │
  │ 3. 算2^shifted：用小查找表，不用指数！                     │
  │    比如 s-max=0 → 2^0=1；s-max=-1 → 0.5；s-max=-8→1/256  │
  │ 4. 加起来总和 sum = Σ 2^shifted                           │
  │ 5. 归一化：gate[i] = 2^shifted[i] / (2^ceil(log2(sum)))   │
  │    → 分母是2的幂，只用移位！不用除法器！                   │
  │ 输出：gate是0~255之间的整数（Q0.8定点）                    │
  └─────────────────────────────────────────────────────────┘
      │
      ▼
步骤5：加权求和 V = Σ_j gate[j] · K_orig[j][d]
      （每个token的V值乘以门控系数，加起来就是注意力输出）
      │
      ▼
输出 attn_out [N=98, D=32]
```

**大白话总结H60**：Q和K都是三值（+1/-1/0），用「同号加分、异号减分」数出来TX和SC两个分数，合起来，用2的幂次归一化（不用exp不用除法），最后当权重加V。全程是整数加法、比较、移位，没有浮点、没有乘法阵列，**极度省电省面积**！

---

### 5. 谁是三值、谁是二值？（一张表记住）

| 位置 | 模式 | 多少bit | 去哪 |
|------|------|---------|------|
| Q（Query）投影后 | **三值ATLIF** | 2-bit {-1,0,+1} | → H60注意力 |
| K（Key）投影后 | **三值ATLIF** | 2-bit {-1,0,+1} | → H60注意力 |
| 3个downsample.sn | **三值ATLIF** | 2-bit {-1,0,+1} | → Sparse MAC（不进H60） |
| Patch、MLP、proj、Decoder、Bottleneck | **二值ATLIF** | 1-bit {0,+1} | → Sparse MAC |

**ATLIF神经元干了啥？** 就是带泄漏的积分点火：
- 膜电位 `u` 每步泄漏一点（乘15/16）
- 加进来输入 `u = u + input`
- 如果 `u ≥ 正阈值` → 发正脉冲，膜电位复位到0
- 如果开了三值且 `u ≤ 负阈值` → 发负脉冲，膜电位复位到0
- 否则不点火

---

## 第二部分：硬件架构——芯片里有哪些「工人」，谁干什么活？

### 1. 四大引擎（记住这四个工人）

```
┌─────────────────────────────────────────────────────────────┐
│                         芯片顶层                             │
│  ┌──────────┐  ┌──────────────┐  ┌─────────┐  ┌──────────┐ │
│  │ Event    │  │ Sparse MAC   │  │ H60     │  │ Dense    │ │
│  │ Scatter  │  │ (稀疏乘加)    │  │ Attn    │  │ MAC      │ │
│  │          │  │              │  │ Engine  │  │ (稠密)   │ │
│  │ 事件→体素│  │ 1-bit/三值    │  │ 注意力  │  │ Decoder  │ │
│  └──────────┘  │ ×INT8权重     │  │ 专用核  │  │ 光流头   │ │
│                │ 零跳过省电    │  │         │  │          │ │
│                └──────────────┘  └─────────┘  └──────────┘ │
│                             │                               │
│                       ┌─────┴─────┐                         │
│                       │ Controller│  控制器：指挥谁什么时候干活 │
│                       │ + SRAM    │  片上SRAM存权重和特征      │
│                       └───────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

| 引擎 | 干什么活 | 占算力比例 | 特点 |
|------|---------|-----------|------|
| **Event Scatter** | 最开头，把事件流散列加成体素 | 小 | 专用前处理 |
| **Sparse MAC** | 卷积、MLP、Patch嵌入、下采样、瓶颈 | **最大头** | 脉冲是0就不乘不加（零跳过），超级省电 |
| **H60 Attention Engine** | 12个block的所有注意力计算 | 约30%周期 | 专用硬件算TX+SC+Shiftmax，无浮点 |
| **Dense MAC** | Decoder上采样、光流预测头 | 小 | 普通定点乘加，因为这部分不是脉冲了 |

**已废弃**：旧版还有一个Legacy QKFormer引擎给S0/S1用，现在**全部12个block都用H60**，省掉一整套硬件面积！

---

### 2. 片上SRAM（工作台/仓库）

| SRAM | 大小 | 存什么 |
|------|------|--------|
| Window SRAM | 256KB | 当前window的Q/K/V脉冲和中间分数 |
| Weight SRAM | 128KB | INT8权重，按层加载 |
| Meta SRAM | 4KB | 层描述符、token_mask、阈值等控制信息 |

总共388KB片上SRAM，非常小！

---

### 3. 控制器怎么调度？（一帧生命周期）

```
上电复位
  ↓
IDLE（等新一帧事件）
  ↓
SCATTER：事件→体素
  ↓
PATCH_EMBED：Sparse MAC跑Patch卷积
  ↓
FOR timestep = 0..9（10个时间步）:
  FOR stage = 0..3:
    LOAD_STAGE_CONFIG(stage)  // 加载该stage的头数、window数等参数
    FOR each block in stage:
      FOR each window in stage:
        如果window全空（token_mask全0）→ 跳过！（省电核心！）
        加载Q/K/V到Window SRAM
        RUN_H60_ATTENTION  // 调用H60引擎
        RUN_MLP            // 调用Sparse MAC跑MLP
    IF stage < 3:
      RUN_DOWNSAMPLE  // 三值编码后Sparse MAC跑下采样
  ↓
BOTTLENECK：Sparse MAC跑2个ResBlock
  ↓
DECODE：Dense MAC跑Decoder和光流头
  ↓
OUTPUT_FLOW：输出光流到片外
  ↓
回IDLE等下一帧
```

---

## 第三部分：RTL模块详解——每个Verilog文件是干嘛的，接口怎么连，数据怎么流

所有RTL都在 `rtl/` 目录下。

### 总览：模块层次

```
nts07_top.v  （芯片顶层）
  ├─ nts07_controller.v  （控制器FSM）
  ├─ SRAM banks （片上存储）
  ├─ event_scatter_unit （事件散射）
  ├─ sparse_mac_cluster （稀疏乘加簇，在sparse_mac_pe.v里）
  │    └─ sparse_mac_lane × OUT_DIM
  │         └─ sparse_mac_pe × IN_DIM （最基础PE单元）
  ├─ h60_attention_engine.v （H60注意力顶层）
  │    ├─ atlif_encode_lane_array （Q/K三值编码）
  │    ├─ tx_sc_pair_score （TX+SC评分流水线）
  │    │    ├─ tx_sc_per_channel × 32
  │    │    └─ popcount_pipelined × 4
  │    ├─ score_fuse_unit （分数融合s=TX+μ·SC）
  │    ├─ shiftmax_unit.v （Shiftmax归一化）
  │    │    ├─ pipelined_max （找行最大值）
  │    │    ├─ pow2_lut × 98 （2^x查找表）
  │    │    ├─ pipelined_sum_tree （求和树）
  │    │    └─ clz_ceil_log2 （前导零计数算log2）
  │    └─ k_gate_unit × 32 （V值加权累加）
  └─ dense_mac_cluster （稠密乘加）
```

---

### 1. 包头文件：nts07_pkg.vh

不是模块，是全局常量定义，所有Verilog都include它。

| 常量 | 值 | 含义 |
|------|----|------|
| `NTS07_HEAD_DIM` | 32 | 每个注意力头的通道数（固定） |
| `NTS07_MAX_TOKENS` | 98 | 每个window最多token数（2×7×7） |
| `NTS07_MAX_HEADS` | 24 | 最大头数（S3用24头） |
| `NTS07_ACT_W` | 16 | 激活位宽 INT16/Q0.15 |
| `NTS07_WGT_W` | 8 | 权重位宽 INT8 |
| `NTS07_ACC_W` | 24 | 累加器位宽 |
| `NTS07_SCORE_W` | 8 | 分数位宽 Q4.3（范围-8~+7） |
| `NTS07_GATE_W` | 8 | 门控位宽 Q0.7（0~255/256） |
| `NTS07_THRESH_W` | 16 | 神经元阈值位宽 |
| `NTS07_MEMBRANE_W` | 18 | 膜电位位宽 |
| `NTS07_MU_Q8_DEFAULT` | 13 | μ≈0.05，13/256≈0.0508 |
| `NTS07_ALPHA0_Q8_DEFAULT` | 5 | α₀≈0.02，same-zero奖励 |
| `NTS07_BETA_Q8_DEFAULT` | 64 | β=0.25，opposite惩罚 |
| `NTS07_GAMMA_Q8_DEFAULT` | 38 | γ≈0.15，single-active惩罚 |
| `NTS07_LEAK_Q8` | 240 | 泄漏=240/256=15/16 |
| `TERN_SILENT/POS/NEG` | 2'b00/01/10 | 三值脉冲编码 |
| `ATLIF_MODE_BINARY/TERNARY` | 1'b0/1 | 神经元模式选择 |
| `ENG_SPARSE_MAC/H60/DENSE_MAC` | 2'd0/1/3 | 引擎ID（Legacy ENG_LEGACY_QK已废弃） |

---

### 2. ATLIF神经元编码：atlif_unified_encode_unit.v

一个神经元一个，32个并行组成lane array。

#### 顶层模块 `atlif_unified_encode_unit` 接口：

| 端口 | 方向 | 位宽 | 含义 |
|------|------|------|------|
| clk | input | 1 | 时钟 |
| rst_n | input | 1 | 复位，低有效 |
| en | input | 1 | 计算使能 |
| acc_clear | input | 1 | 清除膜电位（新时间步/新神经元开始） |
| ternary_en | input | 1 | 0=二值，1=三值 |
| input_acc | input | ACT_W=16 | MAC过来的输入激活（突触前电位） |
| pos_thresh | input | THRESH_W=16 | 正阈值（冻结，从LUT来） |
| neg_thresh | input | THRESH_W=16 | 负阈值（三值用） |
| spike_out | output | 2 | 三值输出：SILENT/POS/NEG |
| binary_out | output | 1 | 二值输出（正脉冲=1，给Sparse MAC直接用） |
| pos_fire | output | 1 | 正点火标志（调试用） |
| neg_fire | output | 1 | 负点火标志（调试用） |

#### 内部数据流：
```
u(膜电位寄存器)
  │
  ├─→ u_leak = u * 240 >>> 8  （泄漏，乘15/16）
  │
  ├─→ u_integrated = u_leak + 符号扩展(input_acc)
  │
  ├─→ pos_fire = (u_integrated ≥ pos_thresh)
  ├─→ neg_fire = ternary_en & (u_integrated ≤ neg_thresh)
  │
  └─→ 时序逻辑：
        复位/acc_clear → u=0, spike=SILENT, binary=0
        en且pos_fire → u=0, spike=POS, binary=1
        en且neg_fire → u=0, spike=NEG, binary=0
        否则 → u=u_integrated, spike=SILENT, binary=0
```

#### 还有三个封装/扩展模块：

| 模块 | 干什么 |
|------|--------|
| `ternary_encode_unit` | 薄封装，固定ternary_en=1，给Q/K路径用 |
| `binary_encode_unit` | 薄封装，固定ternary_en=0，给MLP等路径用 |
| `atlif_encode_lane_array` | LANES个并行（比如32个），一次处理一个神经元的所有通道 |

---

### 3. TX+SC评分单元：tx_sc_score_unit.v

计算一对(Q,K) token的TX和SC分数，完全流水线化。

#### 模块1：`tx_sc_per_channel`（单通道匹配分类）
纯组合逻辑，判断这一个通道两个三值是哪种情况：

| 端口 | 方向 | 含义 |
|------|------|------|
| q_ternary | input [1:0] | Q的这个通道三值 |
| k_ternary | input [1:0] | K的这个通道三值 |
| same_nonzero | output | 同正或同负（++/--） |
| same_zero | output | 都是0 |
| opposite | output | 一正一负（+-/-+） |
| single_active | output | 只有一个点火 |
| sc_sign | output | SC的符号位 |

#### 模块2：`popcount_pipelined`（流水线popcount树）
数32个通道里有多少个1，5级流水线（两两相加），延迟5周期。

#### 模块3：`tx_sc_pair_score`（一对token的完整评分）
**接口**：
| 端口 | 方向 | 含义 |
|------|------|------|
| clk/rst_n/en | input | 时钟/复位/使能 |
| q_ternary | input [1:0][0:31] | Q的32个通道三值 |
| k_ternary | input [1:0][0:31] | K的32个通道三值 |
| alpha0_q8 | input [7:0] | same-zero奖励系数 |
| beta_q8 | input [7:0] | opposite惩罚系数 |
| gamma_q8 | input [7:0] | single-active惩罚系数 |
| tx_score | output [SCORE_W-1:0] | TX分数（Q4.3） |
| sc_score | output [SCORE_W-1:0] | SC分数（Q4.3） |
| valid_out | output | 输出有效（延迟6周期） |

**内部数据流**：
```
32通道Q/K → 每通道tx_sc_per_channel → 4个32bit向量(v_same/v_zero/v_opp/v_single)
                                          │
                                          ▼
                                    4个popcount_pipelined并行数个数
                                    （5周期延迟）
                                          │
                                          ▼
                                    valid_pipe打6拍
                                          │
                                          ▼
                                    tx_full = cnt_same*256 + cnt_zero*α₀ - cnt_opp*β - cnt_single*γ
                                    tx_r = tx_full >>> 10 （归一化到Q4.3）
                                    sc_full = cnt_same - cnt_opp
                                    sc_r = sc_full >>> 2 （归一化到Q4.3）
```
总延迟6周期，**每周期能吃一对token**（流水线填满后）。

#### 模块4：`score_fuse_unit`（分数融合+中心化）
把TX和SC按μ融合，可选减均值：
```
score = tx + (μ·sc >>> 8)
如果center_en：score = score - row_mean
```
延迟1周期。

---

### 4. Shiftmax归一化：shiftmax_unit.v

把融合后的分数变成0~255的门控系数，**不用指数、不用除法**。

#### 子模块：

| 子模块 | 干什么 |
|--------|--------|
| `pow2_lut` | 硬线查找表，输入x∈[-15,0]，输出2^x的Q0.8定点值。0→255，-1→128，…，-8→1，≤-9→0 |
| `pipelined_max` | 7级流水线比较树，找一行98个分数的最大值 |
| `pipelined_sum_tree` | 7级流水线加法树，98个2^x加起来 |
| `clz_ceil_log2` | 组合逻辑，数前导零算ceil(log2(x))，确定移位位数 |

#### 顶层 `shiftmax_unit` 接口：
| 端口 | 方向 | 含义 |
|------|------|------|
| clk/rst_n | input | 时钟/复位 |
| start | input | 开始计算（一行分数就绪） |
| n_tokens | input [6:0] | 实际token数（≤98） |
| preserve_mean | input | 为1时乘以n_tokens保均值 |
| scores | input [SCORE_W-1:0][0:97] | 一行98个融合分数 |
| gates | output [GATE_W-1:0][0:97] | 输出98个门控系数0~255 |
| done | output | 计算完成 |

**内部数据流（总共约17周期流水线）**：
```
start=1
  │
  ├─ Stage1-7：pipelined_max找row_max（7周期）
  │     同时scores用移位寄存器延迟7拍
  │
  ├─ Stage8：shifted[i] = scores_d7[i] - row_max
  │         每个shifted进pow2_lut得到2^shifted
  │
  ├─ Stage9-15：pipelined_sum_tree算row_sum（7周期）
  │     同时pow2_val延迟7拍
  │
  ├─ Stage16：clz_ceil_log2算denom_pow=ceil(log2(row_sum))
  │
  └─ Stage17：每个gate[i] = (pow2_final[i] * scale) >>> denom_pow
              scale=preserve_mean?n_tokens:1
              结果裁剪到0~255
              done=1
```
**吞吐量**：每周期可以喂新一行（流水线级间有寄存器隔离开）。

---

### 5. Sparse MAC稀疏乘加：sparse_mac_pe.v

从最基础PE到簇，支持二值和三值脉冲，零跳过。

#### 模块1：`sparse_mac_pe`（单个处理单元）
1-bit spike × INT8 weight，spike=0就不加，省翻转功耗。支持负脉冲减权重。

| 端口 | 方向 | 含义 |
|------|------|------|
| en | input | 使能 |
| spike_in | input | 1-bit，1=点火 |
| neg_spike | input | 1=负脉冲（减权重） |
| weight | input [WGT_W-1:0] | INT8权重 |
| acc_clear | input | 清累加器 |
| acc_out | output [ACC_W-1:0] | 累加结果 |

逻辑：
- en且spike_in：若neg_spike则acc -= weight，否则acc += weight
- 否则acc保持

#### 模块2：`sparse_mac_lane`（一个输出通道的lane）
IN_DIM个PE并行，算一个输出通道（每个PE处理一个输入通道），然后组合加法树把PE结果加起来。

接口：
- 输入：fire、spike_vec[IN_DIM]、neg_mask[IN_DIM]、weight_vec[IN_DIM*8]、acc_clear
- 输出：acc_out（一个累加和）

#### 模块3：`sparse_mac_cluster`（OUT_DIM个lane组成簇）
算OUT_DIM个输出通道，带IDLE/ACC/DONE三状态FSM，支持流式输入。

| 端口 | 方向 | 含义 |
|------|------|------|
| start | input | 开始计算 |
| done | output | 完成 |
| is_ternary | input | 输入是三值（用neg_mask） |
| n_inputs | input [9:0] | 要累加的输入通道数 |
| in_valid/in_ready | 输入握手 | valid/ready流控 |
| spike_vec | input [IN_DIM-1:0] | 二值spike |
| neg_mask | input [IN_DIM-1:0] | 负spike掩码 |
| weight_block | input [OUT*IN*8-1:0] | OUT×IN的权重块 |
| out_valid | output | 输出有效 |
| acc_out | output [ACC_W-1:0][0:OUT_DIM-1] | OUT_DIM个结果 |

#### 模块4：`k_gate_unit`（H60里V值加权的门控累加器）
专用模块，算gate×K_orig累加：
- 输入：en、acc_clear、gate[7:0]、k_val[15:0][0:31]
- 输出：acc_out[15:0][0:31]
- 逻辑：每个通道 product = gate * k_val，acc += product >>> 8（Q0.8定点）

---

### 6. H60注意力顶层：h60_attention_engine.v

把上面所有模块串起来，实现一个window×一个head的完整H60注意力。8状态FSM。

#### 接口：
| 端口分组 | 端口 | 方向 | 含义 |
|---------|------|------|------|
| 控制 | clk/rst_n | input | 时钟/复位 |
| | start | input | 开始处理一个window-head |
| | done | output | 整个window-head处理完 |
| 配置 | mu_q8/alpha0_q8/beta_q8/gamma_q8 | input [7:0] | 融合系数（checkpoint冻结） |
| | center_scores | input | 是否行中心化 |
| | preserve_mean | input | Shiftmax是否保均值 |
| | n_tokens | input [6:0] | token数（≤98） |
| 加载端口 | load_en | input | 加载使能 |
| | load_qkv_sel | input [1:0] | 0=Q，1=K，2=V(K_orig) |
| | load_idx | input [6:0] | 写哪个token |
| | q_ternary | input [1:0][0:31] | Q一个token的32通道三值 |
| | k_ternary | input [1:0][0:31] | K一个token的32通道三值 |
| | k_orig | input [15:0][0:31] | V一个token的32通道INT16 |
| 输出 | out_valid | output | 一个输出token有效 |
| | out_idx | output [6:0] | 输出token编号i |
| | attn_out | output [15:0][0:31] | 输出token的32通道 |

#### 内部存储：
- `Q_mem[98][32]`：Q寄存器堆，2-bit三值
- `K_mem[98][32]`：K寄存器堆，2-bit三值
- `V_mem[98][32]`：V寄存器堆，INT16
- `score_buf[98]`：当前query行的分数缓存
- `acc[32]`：输出通道累加器
- `q_active[98]`：Q是否有脉冲（静默行跳过优化）

#### FSM状态机（8个状态）：
| 状态 | 干什么 |
|------|--------|
| **S_IDLE** | 等start，start来了cur_i=0，score_i=0，score_j=0 |
| **S_SCORE** | 喂j=0..n_tokens-1进评分流水线，每周期一个j。如果q_active[cur_i]=0（Q行全静默），直接全零分行跳Shiftmax，省电！ |
| **S_DRAIN** | 等流水线排空7周期，直到所有N个分数都写到score_buf |
| **S_CENTER** | 如果center_scores，算行均值，每个分数减均值。超范围token设成极负值。然后置shift_start=1 |
| **S_SHIFT** | 等shiftmax_unit完成（shift_done=1），清acc累加器，acc_j=0 |
| **S_ACCUM** | 逐j累加gate[j] * V[j][d]，acc_j从0到n_tokens-1。完了输出attn_out=acc>>>8，out_valid=1 |
| **S_NEXT** | 最后一个token去S_DONE，否则cur_i++，回S_SCORE算下一个query行 |
| **S_DONE** | done=1，等start撤销回IDLE |

#### 关键性能数字（一个window-head）：
- N=98 tokens，D=32
- 评分：98周期喂数据 + 7周期排空 = 105周期
- 中心化：1周期
- Shiftmax：17周期
- 累加：98周期
- 总共约 ~22k cycles/window-head @500MHz ≈ 44μs

---

### 7. 其他文件
| 文件 | 干什么 |
|------|--------|
| nts07_controller.v | 芯片主控制器，解析层描述符，调度四个引擎 |
| nts07_top.v | 芯片顶层，互连所有模块和SRAM |
| tb_compile.v | 编译测试平台，实例化所有核心模块查语法错误 |
| tb_compile.out | 编译日志输出 |

---

## 第四部分：DATE 2027会议创新点（可以写进论文的）

一共**六个核心创新点**，按重要性排序：

---

### 创新点1：H60 双分数注意力 ISA（指令集架构）—— 最核心

**问题**：标准Transformer softmax要O(N²)浮点乘加、指数、除法，脉冲硬件要么只做单一popcount相似度，精度不够；要么做全精度softmax，面积爆炸。

**我们的方法**：把推理冻结的H60算法固化成**四条硬件原语**，全整数、无浮点、无除法：
1. `DYAD_TX(q,k)`：α-XNOR四分类popcount，考虑同号/同零/异号/单点火四种情况
2. `DYAD_SC(q,k)`：有符号共识分数
3. `SHIFTMAX_GATE(scores, μ)`：融合+Shiftmax归一化，只用LUT+移位
4. `K_GATE(k, gate)`：V值加权累加

**证据**：
- 软件精度：NTS-07b valid825 AEE 1.485，比NB0基线1.585好
- 硬件：单window<600周期，无浮点MAC，面积小

**与别人不同**：
- FireFly-T只有单一popcount，没有signed SC融合，也没有α₀/β/γ四分类
- SDformerFlow原文只有软件算法，没定义ISA级硬件

---

### 创新点2：统一H60全线映射 —— 省面积的关键

**问题**：旧方案S0/S1用Legacy QKFormer，S2/S3用H60，硅片要**两套注意力引擎**+复杂控制器调度，面积大、叙事乱。

**我们的方法**：
- 软件把target_blocks扩展到全部12个encoder block（S0→S3所有block）
- 硬件**只实例化一个参数化H60引擎**，不同stage只换heads/windows/dim的参数表
- Legacy引擎完全不综合，直接删掉

**证据**：
- 面积：省去Legacy核约0.35 mm²
- 性能：即使S0 window多，靠TTB空窗跳过仍达 ~91 FPS / ~10 mJ
- 控制器简化：注意力相位engine_id恒为H60，没有分叉

**与别人不同**：从stage-aware双ISA变成单ISA全线覆盖，软硬件co-design更干净。

---

### 创新点3：统一ATLIF双模神经元算子 —— 面积小、叙事统一

**问题**：如果二值、三值、PSN三套神经元各做一套编码器，要三套比较器、三套状态机，面积重复，论文故事也讲不圆。

**我们的方法**：
- 一个`atlif_unified_encode_unit`，**一套膜电位泄漏+比较器**
- 只用1位`ternary_en`（来自层描述符neuron_mode）切换输出：
  - ternary_en=1 → 2-bit三值输出给H60
  - ternary_en=0 → 1-bit二值输出给Sparse MAC
- Q/K 2-bit / FFN 1-bit带宽节省不变

**证据**：
- 软件统一成`ATLIFTernaryPSN(output_mode="ternary"|"binary")`一个类
- Segment-1实验：统一编码+TTB→ 12.40 mJ（比12.97降4.4%），面积2.80 mm²

---

### 创新点4：Shiftmax原生归一化硬件 —— 无exp、无除法

**问题**：Softmax要算指数→LUT大（>2K LUT），还要除法器，面积延迟都大。Shiftmax软件上验证过，但没有针对有符号融合分数的专用ASIC实现。

**我们的方法**：
- 行最大值减法（所有分数≤0）
- 16项硬线`pow2_lut`算2^x（x=-15..0）
- 加法树求和
- 前导零计数（CLZ）算ceil(log2(sum))，分母是2的幂
- 只用**桶形移位**做归一化，完全没有除法器、没有指数单元

**证据**：面积估算<200 LUT，比softmax小一个数量级；位精度与BSA Shiftmax公式对齐（±1 LSB容差）。

---

### 创新点5：Profile引导的TTB空窗跳过 —— 省电核心

**问题**：均匀调度所有window，但脉冲 firing 高度稀疏不均匀，很多window全静默，白算浪费电。

**我们的方法**：
1. 离线从软件profile的`spike_profile.json`生成`token_mask`和`window_enable`位图
2. 在线控制器看到window_enable=0（整个window没脉冲）直接跳过，不送H60
3. Bishop式Token-Time Bundle（TTB）捆绑发射，减少调度开销

**证据**：Autoresearch实测：
- 关skip：能耗29.48 mJ（+13.6%）
- 开skip：基线25.94 mJ
- 和软件effective_flops降22%趋势一致

---

### 创新点6：Autoresearch驱动的硬件参数自动搜索 —— 方法学创新

**问题**：手工选PE数量、SRAM大小容易拍脑袋，没有可复现性，不一定最优。

**我们的方法**：
- 11轮自动网格搜索（`run_all_experiments.py`）
- 主指标：每帧能耗；次指标：SRAM大小
- Pareto最优选出终极配置

**结果**：
- 256路Sparse MAC：能耗12.97 mJ（比基线降50%）
- 388 KB片上SRAM（256KB window + 128KB weight + 4KB meta）
- 101 FPS @500MHz（解析模型），远超30 FPS实时要求

---

### 论文贡献一句话总结（中文）

> 本文提出面向SDformerFlow事件光流的脉冲Transformer异构加速器，具有：（i）推理冻结的H60双分数注意力ISA，融合三值α-XNOR与有符号共识及Shiftmax无除法归一化；（ii）统一H60单ISA全线12个encoder block映射，消除冗余Legacy通路；（iii）双模统一ATLIF神经元算子，单比较器树支持1-bit/2-bit混合脉冲编码；（iv）profile引导的TTB空窗跳过节能。28nm目标实现下帧能耗约13 mJ、片上SRAM仅388 KB，解析吞吐超100 FPS，满足边缘实时要求，精度在软件验证带内。

---

## 附录：学习路径与自检清单

### 学习顺序

| 顺序 | 任务 | 参考文档 |
|------|------|------|
| 1 | 把本文「事件→光流」的步骤能自己讲一遍 | 就是本文 |
| 2 | 算子→引擎映射：每个算子归哪个引擎 | 03_operator_hardware_mapping.md |
| 3 | 模块接口和描述符定义 | 05_module_interface_spec.md |
| 4 | 性能能耗模型跑一遍 | 08_perf_area_energy_model.md + `python scripts/nts07_perf_model.py` |
| 5 | 最后读RTL代码 | 就按本文「RTL模块详解」顺序读，先看端口再看内部逻辑 |

### 自检清单（能答出来就算数据流过了）

- [ ] 一帧事件经过哪几个大阶段（A→H）？
- [ ] 全部12个encoder block用哪种注意力？
- [ ] S0为啥window多最耗时？靠什么优化？
- [ ] downsample.sn走不走H60？
- [ ] H60五步是什么？为什么不用softmax？
- [ ] Shiftmax怎么用移位代替除法？
- [ ] 三值和二值ATLIF硬件上怎么统一？
- [ ] 四大引擎各干什么活？哪个被删了？
