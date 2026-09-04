# H36/H37 短测计划评价

日期: 2026-05-21

---

## 一、整体评价: 设计良好，4/5

这是整个项目里最干净的实验设计。Agent 回应了审计，做了三件正确的事：
1. 分离 "论文复现" 和 "自研适配" 两种 attention 模式
2. 统一神经元基线，只变 attention 一个变量
3. 收紧筛选门槛，加 360 步确认层

---

## 二、10 个 config 的设计矩阵

| # | 实验 | 注意力 | 范式来源 | V 分支 | norm | 定位 |
|---|------|------|---------|:---:|:---:|------|
| 1 | H36 signed_consensus | popcount + Shiftmax gate | **自研** | K 复用 | head_dim | 我们的方案 |
| 2 | H36 strict_bsa (old) | Q@K^T + Shiftmax (no V) | BSA 适配 | K 复用 | head_dim | 旧 baseline |
| 3 | H37 strict_bsa_qkv | Q@K^T/√d + Shiftmax (V) | **BSA 严格复现** | 独立 V | sqrt_head_dim | 论文对照 |
| 4 | H37 a2os2a_qkv | Q_bin @ K_relu + L1 (V) | **A2OS2A 严格复现** | 独立 V | L1 | 论文对照 |
| 5 | H37 binary_axnor (shiftmax) | {0,1} XNOR + Shiftmax | **alpha-XNOR 严格复现** | K 复用 | head_dim | 论文对照 |
| 6-8 | H37 binary_axnor (L1) | {0,1} XNOR + L1 | alpha-XNOR 复现 | K 复用 | L1 | 硬件友好 |
| 9-10 | H36 strict_bsa H37 neuronfast | 同上 + LR/阈值调参 | 各范式 | 同上 | 同上 | 超参探索 |

**设计优点**: 所有 config 共用同一个神经元骨架：
```
Q/K: symmetric_target_rate, max_thre=1.8, target=0.035, activity_eta=1.0
FFN: stage0+2 high-SOPs, official_atlif (no target, no symmetric)
No angular loss
```

这保证了 **注意力模式是唯一变量**，消融干净。

---

## 三、值得肯定的改进

### 1. 三级筛选逻辑正确

```
120-step → 初筛 (快速淘汰 crash/发散)
360-step → confirm (避免 120 步误导, H9b 教训)
valid40 gate → 全量准入
```

Agent 还加了 promotion 防误用：`120-step 即使有 valid40 也不会被 promote`。

### 2. 门槛收紧合理

| 指标 | 旧门槛 | 新门槛 | 说明 |
|------|:---:|:---:|------|
| SOPs | < 3.9G | < 3.35G | 更贴近 3G 故事 |
| AEE | < 1.70 | < 1.70 | 不变 |
| AAE | < 8.5 | < 8.5 | 不变 |

### 3. 三元健康指标加入筛选

Agent 加了三元健康检查：`max_zero_neg_modules`（检查负发放死亡）、`max_worst_pos_neg_ratio`（检查正负比失衡）。之前 H13n 发现 `neg_mean=0.023` 的好结果就是因为能跟踪三元健康。

### 4. 单测覆盖良好

25 个单测覆盖了：Shiftmax 行和边界、consensus 模式、strict_bsa 模式、target_rate 双向/单向、rapid_screen stage 语义。

---

## 四、仍然存在的问题

### 1. FFN 没有稀疏压力

```
FFN: official_atlif, target_rate=null, activity_eta=2.0
```

`activity_eta=2.0` 是对 Q/K 的强约束（对 FFN 是 `∑|out| * 2.0`），但 FFN 没有 per-neuron 的 target_rate。这意味着 FFN 的稀疏性完全靠 activity_eta 的软约束，没有 Q/K 那种 "target_rate → 阈值双向更新" 的精确控制。这是 H13n 和 H23b 中 `binary_activity=0.136` 偏高的根因。

**后果**: 即使 attention 模式正确，SOPs 可能仍然 > 3.3G，因为 FFN 没被充分稀疏化。

### 2. 没有 angular loss 对照组

所有 config 都 `lambda_ang=0`。虽然 agent 的 H13e 测试过 angular loss (guard120 λ=0.1)，但没系统比较 "有 angular loss vs 无 angular loss" 对 AAE 的影响。这是缺失的关键消融。

### 3. signed_consensus 用的 norm 不一致

H36 signed_consensus 用 `head_dim`（除以 d），H37 strict_bsa 用 `sqrt_head_dim`（除以 √d）。

`signed_consensus` 的分数是 `(Q_sign * K_sign).sum(-1)` = 整数范围 [-32, 32]。除以 head_dim=32 得 [-1, 1]。除以 √32≈5.7 得 [-5.6, 5.6]。

哪个更合适？**head_dim 归一化更适合 popcount**——因为最大值 32 除以 32=1，正好把 score 映射到 [-1,1]。`sqrt_head_dim` 对于 signed_consensus 会放大分数。

Agent 对 strict_bsa 用 `sqrt_head_dim` 是对的（BSA 标准 Q·K^T 的方差是 d，√d 是标准缩放），但对 signed_consensus 应该用 `head_dim`（因为是计数不是点积）。H36 的 `head_dim` 设置是正确的。

### 4. 保守版的 FFN 覆盖仍偏大

`stage02_highsop_official` = stage0 + stage2。Stage2 是 H9c 的 AAE 爆炸区。虽然用了 `official_atlif`（非 symmetric）和低 activity_eta，但 H13n/H23b 的数据表明即使用 `symmetric_target_rate` 保护 Q/K，stage2 FFN 替换仍会拉高 AAE。

**建议**: 加一个 `stage0_only` 的 FFN 对照组，验证"stage2 FFN 替换是 AAE 的主要贡献者"。

### 5. neuronfast 的含义模糊

`h37_*_neuronfast` vs `h37_*_consevative` 的区别是什么？从 config diff 看不出来（ATLIF 参数完全一样）。可能只是 LR 不同（1e-5 vs 2e-5），但没有在 config 里显式标注。

---

## 五、补充建议

| 优先级 | 建议 | 理由 |
|:---:|------|------|
| 1 | 加一个 `stage0_only_ffn` 对照组 | 验证 stage2 FFN 的 AAE 贡献 |
| 2 | 加 H37 + angular loss (λ=0.2) 变体 | 缺失的关键消融 |
| 3 | 给 FFN 加 target_rate (0.08~0.12) | 解决 FFN binary_activity 偏高 |
| 4 | H36 signed_consensus norm 标注清楚 | 区分 head_dim vs sqrt_head_dim 的选择理由 |
| 5 | neuronfast/conservative diff 文档化 | config 里加 note 说明变体差异 |

---

## 六、预期结果

| 注意力 | 我的预测 |
|------|------|
| **H36 signed_consensus** | AAE 最优（历史数据 7.37），SOPs 3.4-3.6G |
| H37 strict_bsa_qkv | BSA 严格复现，AAE 可能略高 (8.0+)，但论文对照价值大 |
| H37 a2os2a_qkv | 不确定性最大 — 之前 H18e 数据差，但这次是 paper-compliant |
| H37 binary_axnor | 纯二元，SOPs 可能最低（无三元开销），但 AEE 会退化 (1.6+) |
| neuronfast 变体 | LR 更低/阈值更激进 → SOPs 更低但 AAE 风险更高 |

**H36 signed_consensus conservative 最可能成为全量候选** — 它是唯一自研方案，论文故事最强，且历史 AAE=7.37 最优。
