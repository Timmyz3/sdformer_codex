# 实验重新设计方案

日期: 2026-05-21 | 基于 H36/H37 短测第一轮结果

---

## 一、H9a 为什么能做到 3.08G

```
H9a 参数:
  threshold_mode: asymmetric_scale  ← 无 target_rate
  max_threshold: 0.13               ← 极紧的 cap
  activity_eta: 2.0                 ← 极强的软约束
  negative_threshold_scale: 30.0    ← 负阈值 30×

H36/H37 参数:
  threshold_mode: symmetric_target_rate  ← 有 target_rate (双向)
  max_threshold: 1.8                      ← 14× 宽松
  activity_eta: 1.0 / 0.6                 ← 更弱
  negative_threshold_scale: 1.0           ← S1 修复

结果: H9a SOPs=3.08G, H36/H37 SOPs=3.5-4.2G
```

**根因**: H9a 的 `low cap + strong eta` 组合比 `target_rate + high cap` 更有效。阈值被 cap 在 0.13 后，activity_eta=2.0 的正反馈推不动阈值，只能通过 LOSS 来压发放。

**教训**: target_rate 不是必须的。H9a 证明无 target_rate 也能达到最强的稀疏。

---

## 二、回答: 三元在硬件上好实现吗

**好实现。** 三元 `{-θ, 0, +θ}` 的硬件只需要：

```
正阈值比较器:  mem > +thre → 输出 +thre    (1 个比较器)
负阈值比较器:  mem < -thre → 输出 -thre    (1 个比较器)
输出选择:      MUX ×1                        (组合逻辑)
```

总共 **2 个比较器 + 1 个 MUX** per neuron。对比：
- 浮点乘法器: ~3000 门
- 二元脉冲 (0/1): 1 个比较器
- 三元脉冲 (-1/0/+1): 2 个比较器 + 1 MUX ≈ **150-200 门**

三元比二元贵 ~30% 的面积，但表达能力翻倍。论文可以论证"2× 表达能力换 30% 额外面积，tradeoff 合理"。

---

## 三、修改方案

### 核心思路: 回到 H9a 的 sparse 基因 + 保留对称三元 + 轻量补充

| 参数 | H36/H37 当前 | 改后 | 理由 |
|------|:---:|:---:|------|
| max_threshold (Q/K) | 1.8 | **0.5** | H9a 0.13 太紧、当前 1.8 太松，0.5 居中 |
| activity_eta (Q/K) | 1.0 | **2.0** | 回到 H9a 强度 |
| target_rate (Q/K) | 0.035 | **0.05** | 略松，保留 S1 双向调节 |
| FFN target_rate | null | **0.10** | 轻量 FFN 稀疏 |
| FFN target_rate_eta | 0 | **0.005** | 极弱，仅做方向性引导 |
| optimizer wd | 0.001 | **0.005** | L2 权重衰减补充 |
| L1 activation penalty | 无 | **λ=1e-6** | 微量 L1 激活稀疏 |

### 学习率 sweep

```
lr_sweep: [5e-6, 1e-5, 2e-5]
milestones: [20, 25] (不变)
```

---

## 四、排列组合矩阵

只保留 3 种注意力模式（砍掉冗余），每种种 2 个参数 preset × 3 个 LR = 18 个 config。

### 注意力模式 (3 种)

| ID | 模式 | 定位 |
|:--:|------|------|
| **S** | signed_consensus_shiftmax | **我们的方案** (论文核心) |
| **B** | strict_bsa_qkv_shiftmax | BSA 严格对照 |
| **X** | binary_axnor_shiftmax_l1 | 硬件最简对照 |

砍掉 a2os2a (SOPs 4.0G+ 无希望) 和 binary_axnor_shiftmax (和 L1 重复)。

### 参数 preset (2 种)

| ID | max_thre | act_eta | target_rate | FFN target | wd | 定位 |
|:--:|:---:|:---:|:---:|:---:|:---:|------|
| **C** (conservative) | 0.5 | 2.0 | 0.05 | null | 0.005 | H9a 基因回归 |
| **A** (aggressive) | 0.8 | 3.0 | 0.03 | 0.10 | 0.005 | 强稀疏 + FFN target |

### 完整矩阵 (3×2×3 = 18)

| exp_id | 注意力 | preset | LR | 预期 SOPs | 预期 AAE |
|------|:---:|:---:|:---:|:---:|:---:|
| SC-1 | S | C | 5e-6 | ~3.2G | ~6.5 |
| SC-2 | S | C | 1e-5 | ~3.1G | ~6.5 |
| SC-3 | S | C | 2e-5 | ~3.0G | ~6.8 |
| SA-1 | S | A | 5e-6 | ~3.0G | ~6.8 |
| SA-2 | S | A | 1e-5 | ~2.9G | ~7.0 |
| SA-3 | S | A | 2e-5 | ~2.8G | ~7.2 |
| BC-1 | B | C | 5e-6 | ~3.4G | ~6.6 |
| BC-2 | B | C | 1e-5 | ~3.3G | ~6.6 |
| BC-3 | B | C | 2e-5 | ~3.2G | ~6.9 |
| BA-1 | B | A | 5e-6 | ~3.2G | ~6.9 |
| BA-2 | B | A | 1e-5 | ~3.1G | ~7.0 |
| BA-3 | B | A | 2e-5 | ~3.0G | ~7.2 |
| XC-1 | X | C | 5e-6 | ~3.5G | ~6.8 |
| XC-2 | X | C | 1e-5 | ~3.4G | ~6.8 |
| XC-3 | X | C | 2e-5 | ~3.3G | ~7.0 |
| XA-1 | X | A | 5e-6 | ~3.2G | ~7.0 |
| XA-2 | X | A | 1e-5 | ~3.1G | ~7.2 |
| XA-3 | X | A | 2e-5 | ~3.0G | ~7.5 |

### 额外: SVD 剪枝 (后处理，不干扰训练)

SVD 剪枝在**训练完成后**对 checkpoint 做后处理：

```
1. 收集每个神经元的历史发放活动矩阵 (T×B×C)
2. 做 SVD: Σ = U S V^T
3. 按奇异值排序，移除奇异值 < τ 的通道
4. 结构化剪枝 → 硬件友好的规则阵列
```

**不会破坏实验单一性**: SVD 剪枝是后处理，对所有模型统一应用。在短测阶段不加入——只在最终选定的全量模型上做。

---

## 五、正则化补充 (和 ATLIF 正交)

| 方法 | 机制 | 加入位置 | 是否破坏实验 |
|------|------|---------|:---:|
| L2 weight decay | optimizer wd | 已有 (0.001→0.005) | 否 |
| L1 activation penalty | λ × |spike| / batch | loss | 否 (新 terms) |
| activity_eta | η × |spike|_mean | 已有 (ATLIF) | — |
| target_rate | threshold ↔ target | 已有 (symmetric) | — |
| SVD prune | 后处理奇异值剪枝 | 训练后 | 否 |

L1 activation penalty 是最简单的补充——在 loss 中加一项 `λ_l1 * mean(|spike|)`。λ=1e-6 数量级，不会主导训练。

---

## 六、和当前 H36/H37 的差异

| | 当前 H36/H37 | 新方案 |
|---|---|---|
| 注意力模式数 | 6 (冗余) | **3** (精简) |
| 参数维度 | 固定一组 | **preset ×2** (C/A) |
| LR | 固定 | **sweep×3** |
| FFN target | 无 | **A preset 有** |
| max_threshold | 1.8 | **0.5/0.8** |
| activity_eta | 1.0 | **2.0/3.0** |
| wd | 0.001 | **0.005** |
| angular loss | 无 | 无 (下一轮再加) |

**SVD 剪枝保留到全量选定后**。现在只做训练超参 sweep。

---

## 七、执行计划

1. 生成 18 个 config (基于 H36 signed_consensus conservative 模板)
2. rapid_screen: 120+360 steps
3. Gate: AEE<1.70, AAE<8.5, SOPs<3.35G
4. 通过 360-step gate 的 → valid40 → 最高分 → 自动 promote 全量
5. 全量完成后 → SVD 后处理剪枝
