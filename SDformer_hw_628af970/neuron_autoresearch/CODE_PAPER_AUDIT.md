# 代码-论文范式符合性审计

日期: 2026-05-20 | 审计范围: bsa_attention.py, atlif_ternary_psn.py, installer.py, h9_losses.py

---

## 一、BSA 范式审计 (Bipolar Self-Attention, NeurIPS 2025)

### BSA 论文的核心公式

```
Q, K, V = TernarySpike(X)           # 三元脉冲 {-1, 0, +1}
scores = Q @ K^T / sqrt(d)           # 标准缩放点积
attn = Shiftmax(scores)             # 2^x / 2^ceil(log2(sum))
output = attn @ V                    # 加权V投影
```

### 当前代码 vs BSA 论文

#### 1. Shiftmax 函数 (第 60-70 行) — ✅ 符合

```python
shifted = scores - scores.amax()          # BSA 公式 (5)
numerator = torch.pow(2.0, shifted)       # BSA 公式 (6): 2^shifted
denom_power = ceil(log2(sum))             # BSA 公式 (7): next power of 2
denominator = pow(2.0, denom_power)       # BSA 公式 (8)
return numerator / denominator
```

**结论: 完全符合 BSA 论文的 Shiftmax 公式。** 每一个数学步骤都有对应代码行。

#### 2. strict_bsa_matrix_attention (第 175-213 行) — ⚠️ 部分符合

```
论文:  scores = Q @ K^T / sqrt(d), 使用 sign-only {-1,0,+1}
代码:  scores = Q_event @ K_event^T, 未除以 sqrt(d); 用 normalize_consensus_score 替代

论文:  attn = Shiftmax(scores) @ V
代码:  attn = Shiftmax(scores) @ K_orig     ← 没有独立 V, 用 K 替代
```

**差异点**:
- **没有 sqrt(d) 缩放**: BSA 原文明确要求 `scores/sqrt(d)`。代码用 `consensus_score_norm` 替代 (`head_dim`, `sqrt_head_dim`, `active`)，默认 `head_dim` 即除以 d 而非 sqrt(d)。差一个平方根。
- **K 作为 V**: SDformer 没有 V 投影，用 `k_orig` (阈值缩放的三元) 替代 V。这会使 V 带有 K 的极性信息，而 BSA 的 V 是独立于 Q/K 的。
- **声明不诚实**: 注释说 "matching the BSA ternary matrix product"，但没有提及 sqrt(d) 缺失和 V 替代。

**建议**: 注释应诚实标注 "近似 BSA" 而非 "匹配 BSA"，且需在论文中说明 sqrt(d) 被 head_dim 替代的原因。

#### 3. signed_consensus_shiftmax (第 518-532 行) — ❌ 不是 BSA

```
代码: scores = (Q_sign * K_sign).sum(per-channel)   ← token-wise popcount
      gate = Shiftmax(scores)
      attn = K * gate                                ← 逐元素门控
```

这不是 BSA 的 `Q@K^T` 矩阵注意力。**这是对 BSA Shiftmax 的 token-gating 适配，不应声称来自 BSA。** 论文应该把它定义为独立的 contribution: "Signed Consensus Gating"。

#### 4. qk_bsa/ternary_matrix (第 460-481 行) — ⚠️ 近似 BSA

```
代码: scores = Q_token @ K^T * score_scale   ← 使用原始 Q (带 θ), 非 sign-only
      gate = Shiftmax(scores)
      attn = gate @ K                        ← 使用原始 K 作为 V
```

**差异**: BSA 的 Q 是 sign-only {-1,0,+1}，这里的 Q 是原始值 (带 θ 幅值)。更接近标准 softmax attention 但用 Shiftmax 替代。

---

## 二、ATLIF 范式审计 (Activity Pruning SNN, NeurIPS 2024)

### ATLIF 论文的核心公式

```
thre[t] = thre[t-1] + sp * (firing_rate - target_rate) * lr
output = {thre if mem > thre else 0}     # 二元输出版本
```

### 当前代码 vs ATLIF 论文

#### 1. TernarySurrogate.forward (第 18-28 行) — ✅ 符合 + 三元扩展

```python
pos_active = (input >= thre).float()            # ATLIF 判定
neg_active = (input <= -neg_thre).float()        # 自研三元扩展
out = (pos_active - neg_active) * thre           # 三元输出
thre_update = sp * (pos_updates + neg_updates)   # ATLIF 阈值累加
```

**符合 ATLIF 的正向路径**: 输入 → 阈值判定 → 二元输出 → 阈值反馈。三元扩展是论文创新点，不是对 ATLIF 的违背。

#### 2. 阈值更新 (installer.py 第 324-360 行) — ✅ 符合 ATLIF

```python
target_rate = getattr(module, "target_rate", cfg.target_rate)
target_feedback = target_rate_eta * (module.r - target_rate)
module.thresh.data += (update_value + target_feedback) * lr * lr_scale
```

**对称 ternary 的 target_rate 机制完全符合 ATLIF**: 发放率 vs 目标率的差值驱动阈值双向更新 (可升可降)。

#### 3. symmetric_target_rate vs 原始 ATLIF — ⚠️ 扩展

```
原始 ATLIF:  target_rate 可选, 大多数实验不启用
当前代码:    symmetric_target_rate 模式下 target_rate 是核心机制
```

**差异**: 原始 ATLIF 主要在分类任务上用 `activity_eta` 做软约束。当前代码把 `target_rate` 提升为核心稀疏控制机制。这是扩展不是违背，但需要在论文中说明。

---

## 三、Alpha-XNOR 范式审计 (Spiking Transformer, CVPR 2025)

### 论文核心公式

```
# 二元版本的 alpha-XNOR
Q_bin = spike(Q).sign()     # {-1, +1}
K_bin = spike(K).sign()
score = Q_bin @ K_bin^T     # XNOR 等价
# alpha 参数区分 match/mismatch
```

### 当前代码 vs 论文

#### 1. _ternary_alpha_xnor_matrix_scores (第 266-292 行) — ⚠️ 三元扩展

```python
same_nonzero = Q_pos@K_pos + Q_neg@K_neg    # XNOR 核心: 同号=正
same_zero = Q_zero@K_zero                     # 三元扩展: 沉默同=小奖励
opposite = Q_pos@K_neg + Q_neg@K_pos          # 三元扩展: 异号=惩罚
score = same_nonzero + alpha0*same_zero - mismatch_penalty*opposite
```

**差异**:
- **论文是二元, 代码是三元**: CVPR 2025 的 alpha-XNOR 只处理 {0,1} 脉冲。代码把脉冲映射为 sign {-1,0,+1}，添加了 negative polarity 和 silence 的处理。
- **alpha0 和 mismatch_penalty 是原创扩展**: 论文只有 alpha 参数。代码加了 mismatch_penalty 专门惩罚三元中的异号冲突。
- **矩阵 vs token 版本**: 代码同时有矩阵版 (H18c) 和 token gate 版 (H18a)。论文只有矩阵版。

**结论**: 这是对 alpha-XNOR 的**重大三元扩展**，不应声称直接来自 CVPR 2025。正确说法: "We generalize the binary alpha-XNOR to signed ternary with polar agreement, mismatch penalty, and silence weighting."

---

## 四、A2OS2A 范式审计 (CVPR 2025)

### 论文核心公式

```
Q = BinarySpike(X)        # {0, 1}
K = ReLU(K_linear)         # ≥ 0
V = TernarySpike(X)        # {-1, 0, +1}
attn = L1_norm(Q @ K^T)
output = attn @ V
```

### 当前代码 vs 论文

#### 1. _a2os2a_matrix_scores (第 295-305 行) — ⚠️ 近似

```python
q_event = (Q > 0).float()             # {0, 1} — 符合 A2OS2A
k_nonnegative = K.clamp_min(0)        # ≥ 0 — 符合 A2OS2A
score = q_event @ k_nonnegative^T     # — 符合
```

但是: **V 仅用 sign(K) 而非 ternary(K)**。论文的 V 是 `TernarySpike(X)`，代码的 V 是 `_ternary_sign_ste(k_orig)` — 只有符号没有阈值幅值。

#### 2. h18e vs h18b — ⚠️ 两版都不完整

- **H18b (gate)**: 保留原始 QK gating，用 A2OS2A 分数作为辅助门。这不是论文的做法。
- **H18e (matrix)**: 用 A2OS2A 分数矩阵 + L1 norm。更接近论文，但 V 只有 sign(K)。

**结论**: 未经充分论证的近似。论文要求 Q binary + K ReLU + V ternary，代码没有完整实现三者。

---

## 五、Hamming Attention 审计 (SpikeVideoFormer, ICML 2025)

### 论文核心公式

```
# 二元 Hamming 版本
x = (2K - 1)^T V                # 第一步
x = (2Q - 1) x / (2 * dim)       # 第二步
```

### 当前代码 vs 论文

#### 1. _hamming_linear_attention (第 308-338 行) — ⚠️ 近似

```python
# 二元路径 (h21a)
q_h = (q > 0) * 2 - 1            # {0,1} → {-1, +1} — 符合
k_h = (k > 0) * 2 - 1            # — 符合
kv = k_h^T @ value                # 步骤 1
attn = q_h @ kv / (2*dim)         # 步骤 2

# 三元路径 (h21b)
q_h = _ternary_sign_ste(q)       # {-1, 0, +1} — 扩展
k_h = _ternary_sign_ste(k)
```

**二元路径 (h21a)** 完全符合 SpikeVideoFormer 公式。✅

**三元路径 (h21b)** 是自研扩展 — 保持沉默为 0 不参与，只有活跃脉冲贡献。应标注为原创扩展。

---

## 六、Loss 模块审计

### AngularFlowLossSupervised (h9_losses.py) — ✅ 正确

```python
flow_mag = sqrt(flow^2 + eps)
gt_mag = sqrt(gt_flow^2 + eps)
cosine = dot/(flow_mag * gt_mag)  # 余弦相似度
cosine = clamp(cosine, -1+eps, 1-eps)
loss = sum(acos(cosine) * mask) / num_valid_px  # 角度误差 (弧度)
```

**完全符合光学流标准角度误差定义。** 代码对 cos 做 clamp 防止 acos 输入越界，ε 防除零。实现质量好。

---

## 七、综合审计结论

| 模块 | 声称来源 | 符合度 | 主要问题 |
|------|---------|:---:|------|
| **Shiftmax** | BSA NeurIPS 2025 | ✅ 完全 | — |
| strict_bsa | BSA NeurIPS 2025 | ⚠️ 80% | 缺 sqrt(d), V 替代 |
| signed_consensus | 自研 | ✅ — | 应标注为独立贡献 |
| **ATLIF threshold** | ATLIF NeurIPS 2024 | ✅ 95% | target_rate 是扩展 |
| symmetric_target_rate | 自研 | ✅ — | S1 是独立贡献 |
| alpha-XNOR token | CVPR 2025 | ⚠️ 50% | 三元扩展过大 |
| alpha-XNOR matrix | CVPR 2025 | ⚠️ 50% | 同上 |
| A2OS2A gate/matrix | CVPR 2025 | ⚠️ 40% | Q/K/V 三路都不完整 |
| Hamming binary | ICML 2025 | ✅ 90% | — |
| Hamming ternary | 自研 | ✅ — | 应标注为自研扩展 |
| **Angular Loss** | 标准光流 | ✅ 完全 | — |

---

## 八、五个"幻觉"级问题

### 1. 声称 `strict_bsa` "matching the BSA ternary matrix product" — 不准确

缺少 `/sqrt(d)` 且 K 替代 V。应该诚实标注 "BSA-inspired, adapted for no-V architecture"。

### 2. 声称 `alpha_xnor` 来自 CVPR 2025 — 不准确

论文是二元 {0,1}，代码加了负极性、沉默权重、冲突惩罚。改动太大，应作为独立三元 alpha-XNOR 归因。

### 3. 声称 `a2os2a_direct` 实现 A2OS2A — 不准确

缺少独立 V 投影、缺少 Q 的脉冲化步骤。只是借鉴了 Q binary + K ReLU 的想法。

### 4. `_ternary_sign_ste` 的梯度问题 — 可能影响训练

```python
hard = x.sign()
return (hard - x).detach() + x
```

当 x = 0.001 时, hard = +1, STE 传回 `(1 - 0.001).detach() + 0.001 = 0.001` → 梯度为 `x` 的原始梯度。但当 x = 1.8 (θ 可达) 时, hard = 1, STE 传回 `(1 - 1.8).detach() + 1.8 = 1.8`。**STE 不缩放 x 的幅值**，但原始 ternary surrogate 的 backward 会通过 `tmp` 缩放。两者的梯度流不一致。

### 5. `h13_consensus_score_mean` 依赖 `locals()` — 脆弱

```python
self.h13_consensus_score_mean = float(
    locals().get("scores", torch.zeros((), device=x.device)).detach().mean().cpu()
)
```

`scores` 只在部分 if-elif 分支中定义。如果某个分支没有定义 `scores` 就直接使用它 (如 `signed_consensus_shiftmax` 用的是 `_signed_consensus_token_scores` 返回但未赋给 `scores` 吧... 实际上它赋给了 `scores`)，依赖 `locals()` 是反模式。

---

## 九、建议修复

| 优先级 | 问题 | 修复 |
|:---:|------|------|
| 0 | strict_bsa 声称不诚实 | 改注释为 "BSA-inspired, no-V adaptation" |
| 0 | alpha_xnor 声称不诚实 | 改注释为 "Ternary alpha-XNOR (generalized from CVPR 2025)" |
| 0 | a2os2a 声称不诚实 | 改注释为 "A2OS2A-inspired (incomplete V path)" |
| 1 | signed_consensus 无论文归属 | 标注为 "Signed Consensus Gating — proposed method" |
| 1 | _ternary_sign_ste 梯度 | 对照 ternary surrogate 的 backward 验证梯度一致性 |
| 2 | locals() 反模式 | 改为显式赋值 `self.h13_consensus_score_mean` 在各分支内 |
