# 组会汇报：SDformerFlow 神经元算子优化进化历程

日期: 2026-05-16 | 基线模型: MS_SpikingformerFlow_en4 (PSN 神经元)

---

## 一、问题定义与目标

**任务**: 事件相机光流估计 (DSEC 数据集)
**基线**: SDformerFlow (PSN 并行脉冲神经元), AEE=1.58, AAE=7.50, SOPs=3.62G
**目标**: 设计硬件友好的稀疏脉冲神经元算子，降低运算量 (SOPs)，保持精度 (AEE/AAE)
**约束**: 不修改 baseline 代码，新实验通过 overlay + source-patching 隔离运行
**硬件映射**: 每个神经元改进必须对应明确的加速器原语 (比较器、门控、计数器)

---

## 二、实验进化历程 (15个关键节点)

### 阶段 0: 基线确立

**E0 — PSN Baseline**
- 机制: Parallel Spiking Neuron, 10 时间步并行计算
- 结果: AEE=1.5848, AAE=7.5012, SOPs=3.6219G, firing=0.085
- 判决: 精度和稀疏性的双重基准

---

### 阶段 1: 完整神经元替换路线 → 全灭 (E1-E6, 15次失败)

尝试将全部 PSN 神经元替换为不同类型的脉冲神经元：

| 实验 | 机制 | 开源来源 | AEE | SOPs | 失败原因 |
|------|------|---------|:---:|:---:|------|
| E2 ATLIF | 自适应阈值 LIF | Activity-Pruning-SNN (NeurIPS 2024) | 2.5~3.8 | 2.9~5.2G | 阈值过度剪枝 |
| E3 LMHT | 多层级阈值 | LMHT_SNN (NeurIPS 2024) | 2.6 | 9.7G | SOPs 爆炸 3× |
| E4 TS-LIF | 时间段 LIF | TS-LIF (ICLR 2025) | 2.2 | 4.0G | 全替换中最好，但 AEE +38% |
| E5b TSN | 三元脉冲 | Ternary-Spike (AAAI 2024) | 29.8 | 25.9G | 范式不兼容 |
| E6 NASN | 自适应量化发放 | 自建 (arXiv 2025) | 2.2 | 33.3G | SOPs 9.2× 基线 |

**关键发现**: 完整神经元替换在 SDFormerFlow 上全部失败。开源实现在分类任务上有效，但在事件光流上不兼容——原因是 SDformerFlow 的 PSN+事件箱范式与标准 LIF/PLIF 范式差异太大。

---

### 阶段 2: 部分门控路线 → 首个突破 (G1)

**G1 — 6节点部分稀疏门控**

```
机制: PSN输出 × STE(sigmoid(gate_logit) ≥ 0.5)
      6个 layer0 高敏感节点各一个标量门
      主干冻结，仅训练 6 个 gate_logit
      硬件: 1个比较器+AND gate per neuron, 零乘法器
```

| 指标 | 基线 | G1 | 变化 |
|------|------|-----|:---:|
| AEE | 1.5848 | 1.6056 | +1.3% |
| AAE | 7.5012 | **7.2452** | **-3.4%** |
| SOPs | 3.6219G | **2.7134G** | **-25.1%** |
| Firing | 0.0850 | 0.0637 | -25.1% |

**核心洞察 #1**: 部分插入 > 全替换。6 个关键节点的精确门控比改全部 100+ 个神经元效果好得多。
**核心洞察 #2**: 只改 MLP 层的 spike 输出，保留 attention Q/K 不变 → AAE 不仅没涨反而下降了。

---

### 阶段 3: 扩展到全 Stage → 精度崩塌 (H1)

**H1 — HardwareSparseNeuron (GTCN), 36 节点**

```
机制: G1的门控 + ATLIF自适应阈值, 扩展到所有4个encoder stage
      每个节点: output = PSN(x - threshold_bias) × STE(gate≥0.5)
      threshold_bias ← activity_eta × (firing_rate - target_rate)
      36 个 gate_logit, freeze_backbone, gate-only 训练
```

| 指标 | G1 | H1 | 变化 |
|------|-----|-----|:---:|
| AEE | 1.61 | 2.67 | **+65%** |
| SOPs | 2.71G | 2.71G | 持平 |

**核心洞察 #3**: 门控策略不能线性扩展到全 stage。6个节点是甜点——扩展到36个破坏了深层 encoder 的特征通路。门控对浅层 MLP 有效，对深层和 attention 路径有破坏性。

---

### 阶段 4: Attention 三元化 → AAE 持续恶化 (H2-H4)

观察到 attention 的 Q/K 是信息瓶颈 → 尝试用三值脉冲增强注意力表达能力。

**H2 — 自适应三元 PSN (Q/K only)**
- 机制: PSN + 三元输出 {-θ, 0, +θ} + ATLIF 阈值
- 结果: AEE 1.79, SOPs 8.68G (+140%) → **失败**: 三元导致负脉冲泛滥

**H3f — 官方 ATLIF 二元 (Q/K only)**
- 机制: Q/K 仅用官方 ATLIF 二元输出, atlif_only 训练 (168 params)
- 结果: AEE 1.60, SOPs 3.40G (-6%), AAE 8.41 (+12%)

**H4h — ATLIF 三元 (Q/K only)**
- 机制: Q/K 用 ATLIF 三元，全参数训练，负脉冲比例受控 (signed scale=30, cap=0.13)
- 结果: AEE **1.54** (-2.9%), AAE 7.97 (+6.2%), SOPs 3.48G (-3.8%)

| 实验 | AEE | AAE | SOPs | 发现 |
|------|:---:|:---:|------|------|
| H3f (二元 Q/K) | 1.60 | 8.41 | 3.40G | 有效但 AAE 恶化 |
| H4h (三元 Q/K) | **1.54** | 7.97 | 3.48G | 三元增强 AEE, AAE 略好于二元 |
| **H4 ctrl (Q/K=0)** | 1.62 | **7.88** | 3.27G | **删掉 Q/K 比损坏 Q/K 更好** |

**核心洞察 #4**: H4 ctrl (完全删掉 Q/K, AAE=7.88) 比 H4h (三元 Q/K, AAE=7.97) 的 AAE 更好，说明当前的 Q/K 三元化在**主动伤害方向估计**。根本原因是 SDformer 的 QK attention 没有归一化——负的 attention score 会翻转 V 投影方向。

---

### 阶段 5: 三元+二元混合 → 精度反超但 AAE 仍是问题 (H5-H6)

**H5 — 三元扩展到高 SOPs 层 → 失败**
- 将三元从 Q/K 扩展到 proj_sn、FFN、downsample → 这些层的负脉冲泛滥
- 结论: **三元只适用于 attention**, 非 attention 层用二元

**H6a — 三元 Attention + 二元 FFN**
- 机制: Q/K 三元 + stage0 FFN 二元 + stage0/stage2 downsample 二元
- 结果: AEE **1.55** (-2.0%), SOPs **3.30G** (-8.9%), AAE **8.20** (+9.3%)

**核心洞察 #5**: 混合方案有效——AEE 首次反超基线。但 AAE 依然恶化——ATLIF 阈值不加区分地压低 Q/K 发放，导致方向信息丢失。**问题不在神经元本身，在 attention 的归一化缺失。**

---

### 阶段 6: 不应期 (A5) 和 FFN 搜索 (H7-H8)

**A5 — 不应期修剪 → 完全失败**
- 机制: 发放后强制静默 2 步 (2-bit 计数器)
- 结果: AEE 9.42, SOPs 12.2G (+237%), firing 0.286 (+237%)
- **硬约束适得其反**: 不应期迫使网络在剩余时间步加倍发放来补偿

**H8 — FFN Block Search**
- 搜索 18 个 FFN block 的二元 ATLIF 替换候选
- 发现 h8m (stage3 block0 FFN 二元) 为唯一通过所有阈值的候选
- 但 profile 后 AAE 同样恶化 → **再次指向 attention 归一化问题**

---

### 阶段 7: Shiftmax → 突破性修复 (H9)

**根因确认**: SDformer 的 QK attention 用 `Q_sum → sn2_q → K*gate` 的 Q token gating，没有 softmax 或其他归一化。三元 Q/K 引入负值后，没有归一化导致 attention 分数可正可负，负值翻转 V 投影方向。

**H9a — Shiftmax Attention 归一化 (受 BSA NeurIPS 2025 启发)**

```
原始 attention:   attn = K * (sn2_q(sum(Q)))          ← 无归一化, lossy脉冲化
H9a attention:    gate = shiftmax(sum(Q)) * n          ← 2^x归一化 + 保均值
                  attn = K * gate                       ← 单路径
```

| 指标 | 基线 | H9a ep29 | 变化 |
|------|------|---------|:---:|
| AEE | 1.5848 | **1.5044** | **-5.1%** |
| AAE | 7.5012 | **7.6365** | **+1.8%** |
| SOPs | 3.6219G | **3.0847G** | **-14.8%** |
| Firing | 0.0850 | 0.0724 | -14.8% |

**首次同时实现: AEE 反超基线 5% + SOPs 降低 15% + AAE 控制在 2% 以内**

**H9e — FFN 目标精炼**
- 与 H9a 相同的 Shiftmax, 但 FFN 替换目标调整为浅层偶数 block
- 结果: AEE 1.50, AAE 7.68, SOPs 3.28G — 与 H9a 非常接近

**H9c — 深层 FFN 验证**

| 实验 | 替换层 | AAE | 结论 |
|------|--------|:---:|------|
| H9a | stage0 FFN + stage0/stage2 downsample | 7.64 | ✅ 安全 |
| H9e | layers0+1 FFN even blocks | 7.68 | ✅ 安全 |
| H9c_all6 | stage2 FFN blocks 1/3/5 | **31.2** | ❌ 危险 |
| H9c_odd135 | stage2 FFN blocks 1/3/5 (交错) | **33.1** | ❌ 危险 |

**核心洞察 #6**: Shiftmax 修复了 attention 的归一化，但**仍不能碰深层 FFN (stage2)**。stage2 是 encoder bottleneck (6 blocks, 最高维度)，动了它的 FFN 会系统性扭曲进入 decoder 的方向信息。安全的策略是: attention 三元 + Shiftmax 归一化 + 浅层 FFN 二元替换。

---

### 阶段 7 详细展开: Attention + Shiftmax 融合的五步迭代

#### Step 0: 问题发现 — H4 ctrl

```
H4 ctrl: 把 Q/K 全部置零 → AAE=7.88
H4h:     三元 ATLIF Q/K   → AAE=7.97
结论: 删掉比"优化"更好 → Q/K 三元化在主动伤害方向估计
```

**定位根因**: SDformer 的 attention 是 `Q_sum → sn2_q → K * gate` 的 Q token gating，没有 softmax 归一化。三元 Q 产生负的 Q_sum → `sn2_q` 脉冲化时截断为 0 → 负 K 值被逐元素乘放大 → 方向翻转。

#### Step 1: Shiftmax 首次引入 — H9a (突破)

双轨制 — 保留原始 Q token gating + 叠加 Shiftmax Q·K 兼容门:

```
原始:  att_token = sn2_q(sum(Q))           H9a:  att_token = sn2_q(sum(Q))
      attn = K * att_token                        scores = (Q*K).sum(-1)
                                                  gate = shiftmax(scores) * 162
                                                  attn = (K * att_token) * gate
```

| 基线 | H9a | 变化 |
|------|------|:---:|
| AEE=1.58, AAE=7.50, SOPs=3.62G | AEE=**1.50**, AAE=**7.64**, SOPs=**3.08G** | AEE -5%, SOPs -15%, AAE +1.8% |

**首次同时实现 AEE 反超 + SOPs 显著降低 + AAE 受控。**

#### Step 2: Attention 子集搜索 — H9b

120 步探针，测试 Shiftmax 仅作用于特定 stage 的 attention blocks:

| 配置 | AEE↓ | AAE↓ | SOPs | 判决 |
|------|-----:|-----:|------|:---:|
| **stage1 only** (2 blocks) | **1.04** | **6.12** | 3.60G | 探针最佳 |
| stage23 (8 blocks) | 1.09 | 6.16 | 3.57G | 良好 |
| stage3 only (4 blocks) | 1.09 | 6.32 | 3.60G | 一般 |
| stage0 only (2 blocks) | 1.10 | 6.35 | 3.52G | 一般 |
| stage2 only (6 blocks) | 1.15 | 6.28 | 3.48G | 一般 |

**发现**: stage1 探针表现最好，但 promotion 到 full run 后 epoch 3 AAE 炸到 32.68。120 步探针不足以预测长期 AAE 行为。**Shiftmax 需要全 attention block 参与才能维持方向稳定性。**

#### Step 3: FFN 安全边界 — H9c

| 配置 | FFN 替换 | AEE↓ | AAE↓ | SOPs↓ | 判决 |
|------|---------|-----:|-----:|------|:---:|
| H9a | stage0 FFN + s0/s2 downsample | 1.50 | 7.64 | 3.08G | ✅ |
| H9c_all6 | **stage2** blocks 1/3/5 FFN | 1.42 | **31.2** | 3.08G | ❌ |
| H9c_odd135 | **stage2** blocks 1/3/5 FFN (交错) | 1.47 | **33.1** | 3.40G | ❌ |

**发现**: stage2 是不可触碰的红线 (6 blocks, 最高维度 768→3072)。即使有 Shiftmax 保护 attention，深层 FFN 的二元 ATLIF 替换仍破坏方向信息。

#### Step 4: 安全 FFN 扩展 — H9e

| 配置 | FFN 替换 | AEE↓ | AAE↓ | SOPs↓ | 判决 |
|------|---------|-----:|-----:|------|:---:|
| H9a | stage0 FFN + s0/s2 downsample | 1.50 | 7.64 | 3.08G | ✅ |
| **H9e** | layers0+1 even blocks FFN | **1.50** | **7.68** | **3.28G** | ✅ |

**发现**: layers 0+1 是安全的扩展区域。

#### Step 5: Shiftmax 模式探索 — H10 系列

三种模式对比:

```
compat_qk_product (H9a):       qkformer_token (H10):         qk_bsa (H10c):
att_token = sn2_q(sum(Q))     scores = sum(Q)               scores = Q @ K^T
gate = shiftmax(Q·K) * n       gate = shiftmax(scores)       weights = shiftmax(scores)
attn = (K*att_token) * gate    attn = K * gate               attn = weights @ K
← 叠加原始, 双路径              ← 替代原始, 单路径              ← 标准BSA, K替代V
```

| 模式 | 实验 | AEE↓ | AAE↓ | 判决 |
|------|------|-----:|-----:|:---:|
| **compat_qk_product** | **H9a** | **1.50** | **7.64** | ✅ 当前最佳 |
| qkformer_token | H10 | 3.87 | 71.6 | ❌ 数值崩溃 |
| qk_bsa | H10c | 1.73 | 8.05 | ⚠️ 不如H9a |

**当前认知**: SDformer 的 QK-only 架构下，保留原始 token gating + 叠加 Shiftmax 兼容门的双轨策略最稳定。

#### 融合迭代总结

```
Step 0: H4 ctrl 发现问题     → Q/K 损坏比删除更差
Step 1: H9a 首次 Shiftmax    → AAE+1.8%, 突破!  compat_qk_product 双轨制
Step 2: H9b attention 子集搜索 → stage1 探针最好, 但全训练爆了
Step 3: H9c FFN 安全边界      → stage2 是不可触碰的红线
Step 4: H9e 安全 FFN 扩展      → layers0+1 even blocks 安全
Step 5: H10 模式探索          → compat_qk_product 仍是当前最优
  → 待突破: 三元原生归一化 (AD-Norm) 替代 2^x
```

**当前最佳配置 (H9a)**:

| 组件 | 配置 |
|------|------|
| Q/K 神经元 | PSN + ATLIF 三元 ({-θ, 0, +θ}) |
| Attention 归一化 | Shiftmax (compat_qk_product 双轨制) |
| FFN 稀疏化 | stage0 FFN 二元 ATLIF |
| Downsample 稀疏化 | stage0 + stage2 二元 ATLIF |
| 训练方式 | 全参数, lr=2e-5, 30 epochs |

---

## 三、各系列实验指标汇总

### E 系列 — 完整神经元替换 (全部失败)

| 实验 | 机制 | 代码来源 | 训练 | AEE↓ | AAE↓ | SOPs↓ | 判决 |
|------|------|---------|:---:|-----:|-----:|------|------|
| **E0** | PSN baseline | SpikingJelly | 59ep | 1.5848 | 7.5012 | 3.6219G | 基准 |
| E1 | Simple SN | 自建 | smoke | — | — | — | 未完成 |
| E2 ATLIF | 自适应阈值 LIF | [Activity-Pruning-SNN](https://github.com/putshua/Activity-Pruning-SNN) (NeurIPS'24) | 59ep | 2.51~3.76 | 12.5~21.5 | 2.87~5.21G | ❌ |
| E3 LMHT | 多层级阈值 | [LMHT_SNN](https://github.com/hzc1208/LMHT_SNN) (NeurIPS'24) | 59ep | 2.56 | 9.65 | 9.71G | ❌ |
| E4 TS-LIF | 时间段 LIF | [TS-LIF](https://github.com/kkking-kk/TS-LIF) (ICLR'25) | 59ep | 2.18 | 9.82 | 4.01G | ❌ |
| E4b TS-LIF | 官方风格 | 同上 | short | 6.99 | 83.9 | 2.16G | ❌ |
| E5 TSN | 早期三元 | [Ternary-Spike](https://github.com/yfguo91/Ternary-Spike) (AAAI'24) | smoke | — | — | — | 未完成 |
| E5b TSN | 官方三元 | 同上 | 59ep | 29.77 | 98.4 | 25.9G | ❌ |
| E6 NASN | 自适应量化 | 自建 (arXiv 2604.12365) | 59ep | 2.17 | 8.36 | 33.3G | ❌ |

> **结论: 10 次全替换, 0 次成功。E4 TS-LIF 是全替换天花板 (AEE +38%, SOPs +11%)。**

### F 系列 — 融合神经元 (仅骨架验证)

| 实验 | 机制 | 训练 | Train Loss | Val Loss | 判决 |
|------|------|:---:|----------:|--------:|------|
| F1 | 自适应 PSN (GTCN gate+ATLIF) | smoke | 8.16 | 6.22 | 待完整验证 |
| F2 | LMH + ATLIF 融合 | smoke | 6.96 | 6.42 | 待完整验证 |
| F3 | 自适应 TS-LIF | smoke | 6.04 | 6.35 | 待完整验证 |
| F4 | LMH + TS-LIF 融合 | smoke | 14.9 | 15.4 | 质量差 |
| F5 | 有符号混合 (三元+PSN) | smoke | 9.01 | 6.81 | 待完整验证 |

> **结论: 5 个融合实验均仅跑 smoke, 无 valid40 评估。全替换失败后转部分插入路线。**

### G/H1 系列 — 部分门控

| 实验 | 机制 | 参数量 | 训练 | AEE↓ | AAE↓ | SOPs↓ | firing | 硬件 | 判决 |
|------|------|:---:|:---:|-----:|-----:|------|:-----:|------|:---:|
| **G1** | 6 节点标量 STE 门, 主干冻结 | **6** | smoke ep0 | 1.6056 | **7.2452** | **2.7134G** | 0.0637 | 6×AND门 | ✅ |
| G1 BN-eval | 同上, batch-norm eval 模式 | 6 | 4ep | 1.6248 | 7.2609 | 2.7426G | 0.0643 | 同上 | ✅ |
| **H1** | 36 节点 GTCN (门+ATLIF), 主干冻结 | 36 | 19ep | 2.6661 | — | 2.7124G | 0.0636 | 36×AND+比较器 | ❌ |

> **G1 突破: SOPs -25.1%, AAE 反降 -3.4%。H1 证明线性扩展到 36 节点会破坏精度。**

### H2-H6 系列 — Attention 三元化 + FFN 混合

| 实验 | Q/K | FFN | Downsample | 训练 | AEE↓ | AAE↓ | SOPs↓ | firing | 判决 |
|------|:---:|:---:|:---:|:---:|-----:|-----:|------|:-----:|:---:|
| H2 | 三元ATLIF | 无 | 无 | 19ep | 1.79 | 8.85 | 8.68G | 0.204 | ❌ 负脉冲泛滥 |
| **H3f** | 二元ATLIF | 无 | 无 | 29ep | 1.60 | 8.41 | 3.40G | 0.080 | ⚠️ AAE恶化 |
| **H4h** | 三元ATLIF | 无 | 无 | 29ep | 1.54 | 7.97 | 3.48G | 0.082 | ⚠️ AEE改善, AAE略好 |
| **H4 ctrl** | **全部置零** | 无 | 无 | — | 1.62 | **7.88** | 3.27G | 0.077 | 🔍 删了比坏了强 |
| H5 | 三元 | 三元(proj+MLP) | 三元 | short | 1.14 | — | 7.69G | 0.180 | ❌ 三元不适FFN |
| **H6a** | 三元 | **二元**(stage0) | **二元**(s0/s2) | 30ep | 1.55 | 8.20 | 3.30G | 0.077 | ✅ AEE反超 |

> **H6a: 首次 AEE 反超基线 -2.0%。三元只适合 attention, FFN 必须用二元。但 AAE +9.3% 仍是问题。**

### A5 + H7-H8 系列 — 不应期 & FFN 搜索

| 实验 | 机制 | 训练 | AEE↓ | AAE↓ | SOPs↓ | firing | 判决 |
|------|------|:---:|-----:|-----:|------|:-----:|:---:|
| **A5** | 全节点不应期(2步) | 60ep | 9.42 | — | 12.2G | 0.286 | ❌ 适得其反 |
| H7 | 按 stage 扩展 FFN 二元 | short probes | — | — | — | — | 仅探针 |
| **H8** | 18 配置逐 block FFN 搜索 | search | h8m 入围 | — | — | — | 搜索完成 |

> **A5 失败根因: 硬约束导致补偿性超发。H8 搜索发现 h8m (stage3 block0) 为唯一通过阈值的候选。**

### H9 系列 — Shiftmax 突破 (当前最佳)

| 实验 | Shiftmax 模式 | FFN 替换 | Downsample | 训练 | AEE↓ | AAE↓ | SOPs↓ | firing | 判决 |
|------|:---:|------|:---:|:---:|-----:|-----:|------|:-----:|:---:|
| **H9a** | compat_qk_product | stage0 FFN 二元 | s0+s2 二元 | 29ep | **1.5044** | **7.6365** | **3.0847G** | 0.0724 | ✅ **综合最佳** |
| **H9e** | compat_qk_product | layers0+1 even FFN 二元 | 无 | 29ep | **1.4977** | **7.6800** | **3.2840G** | 0.0770 | ✅ **接近H9a** |
| H10c | qk_bsa | H9a core | s0+s2 | 29ep | 1.7321 | 8.0543 | 3.4850G | 0.0818 | ⚠️ 一般 |
| H9c_all6 | compat_qk_product | **stage2** FFN 二元(奇数块) | 无 | 29ep | 1.4247 | **31.17** | 3.0823G | 0.0723 | ❌ AAE爆 |
| H9c_odd135 | compat_qk_product | **stage2** FFN 二元(交错块) | 无 | 29ep | 1.4745 | **33.05** | 3.4013G | 0.0798 | ❌ AAE爆 |
| H9b_stage1 | compat_qk_product | stage1 FFN shiftmax | s0+s2 | 3ep | 1.6763 | **32.68** | 2.8844G | 0.0677 | ❌ 早期即爆 |
| H10 | qkformer_token | layers0+1 FFN 二元 | 无 | 29ep | 3.8739 | **71.64** | 3.4228G | 0.0803 | ❌ 模式bug |

> **H9a/H9e: AEE -5%, SOPs -15%, AAE 仅 +2%。Shiftmax 解决了 AAE 问题。深层 stage2 FFN 仍不可碰。**

### 全部实验横向对比 (按 AAE 排序, 仅 valid40)

```
AE&E 正常组 (AAE < 9):                  AAE 爆掉组 (AAE > 30):
─────────────────────────               ─────────────────────
H9a    AEE=1.50 AAE=7.64 SOPs=3.08G     H9c_all6   AEE=1.42 AAE=31.2 SOPs=3.08G
H9e    AEE=1.50 AAE=7.68 SOPs=3.28G     H9c_odd135 AEE=1.47 AAE=33.1 SOPs=3.40G
G1     AEE=1.61 AAE=7.25 SOPs=2.71G     H9b_stage1 AEE=1.68 AAE=32.7 SOPs=2.88G
H4 ctrl AEE=1.62 AAE=7.88 SOPs=3.27G     H10        AEE=3.87 AAE=71.6 SOPs=3.42G
H4h    AEE=1.54 AAE=7.97 SOPs=3.48G     
H6a    AEE=1.55 AAE=8.20 SOPs=3.30G     
H3f    AEE=1.60 AAE=8.41 SOPs=3.40G     
E0     AEE=1.58 AAE=7.50 SOPs=3.62G     ← 基线
```

> **关键规律: 所有 AAE 爆掉的实验都动了深层 (stage1/stage2) 的 attention 或 FFN。安全的策略: attention 三元 + Shiftmax + 浅层 (stage0/stage1) FFN 二元。**

---

## 四、关键实验进化树

```
E0 PSN baseline (AEE=1.58, SOPs=3.62G)
│
├─ E1-E6: 完整神经元替换 [全灭]
│   └─ 教训: 不要全换
│
├─ G1: 6节点门控 [突破] → SOPs -25%, AEE +1.3%, AAE -3.4%
│   ├─ 教训: 部分插入 > 全替换
│   │
│   ├─ H1: 扩展到36节点 [失败]
│   │   └─ 教训: 不能线性扩展, 浅层门控是甜点
│   │
│   └─ H2-H4: Attention三元化 [部分成功]
│       ├─ 教训: 三元增强AEE但破坏AAE
│       │
│       ├─ H5: 三元扩展FFN [失败]
│       │   └─ 教训: 三元只适用于attention
│       │
│       ├─ H6a: 三元attn+二元FFN [AEE反超基线]
│       │   └─ 教训: 混合方案有效, 但AAE仍是问题
│       │
│       └─ H8: FFN Block Search [发现h8m]
│           └─ 教训: 需要修复attention归一化
│
├─ A5: 不应期修剪 [完全失败]
│   └─ 教训: 硬约束不如软惩罚
│
└─ H9: Shiftmax注意力归一化 [当前最佳] ★
    ├─ H9a: AEE -5.1%, SOPs -14.8%, AAE +1.8%  ← 综合最佳
    ├─ H9e: AEE -5.5%, SOPs -9.3%, AAE +2.4%  ← 接近H9a
    ├─ H9c: 验证深层FFN危险 (AAE 31-33)
    └─ H10: qkformer单轨模式 [调试中]
```

---

## 四、当前最佳结果

| 实验 | 机制 | AEE | AAE | SOPs | 亮点 |
|------|------|:---:|:---:|------|------|
| **G1** | 6节点标量门 | 1.61(+1.3%) | **7.25(-3.4%)** | **2.71G(-25.1%)** | 最大稀疏, AAE改善 |
| **H9a** | Shiftmax + H8m | **1.50(-5.1%)** | 7.64(+1.8%) | **3.08G(-14.8%)** | AEE+SOPs双优, AAE受控 |

**G1 和 H9a 是互补的两个方向**: G1 追求极简硬件 (6个AND gate) 和最大稀疏性；H9a 追求 AEE 精度 + 中等稀疏性。结合两者 (Shiftmax 归一化 + 浅层门控) 可能进一步突破。

---

## 五、核心发现总结

1. **全替换死胡同** (E1-E6): 15次全神经元替换 0 次成功, 开源代码在 SDFormerFlow 上不兼容
2. **部分插入是唯一出路** (G1): 6 个关键节点的精准门控比改全部 100+ 个节点更有效
3. **三元只适用于 Attention** (H5-H6): attention Q/K 用三元增强极性表达, FFN 用二元压低发放
4. **没有归一化是 AAE 恶化的根因** (H4 ctrl→H9): Shiftmax 修复了方向误差
5. **浅层 FFN 安全, 深层 FFN 危险** (H9a vs H9c): stage2 bottleneck 不能随意动
6. **硬约束适得其反** (A5): 不应期导致补偿性超发, 软惩罚 (ATLIF 阈值) 更有效

---

## 六、下一步计划

### 短期 (本月)

1. **H11 — G1 + H9a 融合**: G1 的浅层门控 (6 节点) + H9a 的 Shiftmax 归一化
2. **AD-Norm 替代 Shiftmax**: 开发三元原生归一化 (Agreement-Disagreement Norm)，去 exp/LUT
3. **H9e 全量验证**: 验证 H9e 的 FFN even-block 策略在更多 epoch 下的稳定性

### 中期

4. **注意力稀疏化**: Shiftmax + 窗口稀疏 mask 预计算
5. **结构化剪枝**: 对长期不发放的神经元通道做结构化移除
6. **硬件加速器设计**: 基于 G1+H9a 的最优配置设计加速器架构

### 长期

7. **体素化优化**: 稀疏体素表示替代稠密体素网格
8. **混合精度**: attention 4-bit, FFN 1-bit 脉冲分级
