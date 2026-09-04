# 三元负发放问题：根因分析与解决方案

日期: 2026-05-18 | 数据来源: I15 epoch 5

---

## 根因分析

### 数据

```
neg_mean = 0.00037 (0.037%)
pos_mean = 0.061   (6.1%)
ratio: 1 个负脉冲对应 165 个正脉冲
```

### 三层叠加的抑制机制

**第 1 层: negative_threshold_scale = 30**

```python
# atlif_ternary_psn.py line 20
neg_thre = thre × 30    # thre=0.12 → neg_thre=3.6
```

膜电位范围 [-1, +1]，触发负脉冲需要 mem < -3.6 → 概率 ≈ 0。

**第 2 层: 阈值增长的正反馈**

ATLIF 的阈值更新只关心总发放率 (pos_r + neg_r)，不区分正负:

```python
thre += (pos_r + neg_r - target_rate) × activity_eta × lr_scale
```

neg_r 几乎为零 → total_rate ≈ pos_r → 阈值更新只受正脉冲驱动 → 阈值增长后 neg_thre 也跟着 30× 增长 → neg_r 进一步被压缩 → 恶性循环。

**第 3 层: max_threshold=0.13 的 cap**

cap 到了但仍能压制负脉冲: thre=0.13 → neg_thre=3.9。即使 cap 住，neg_thre 已经大到任何训练都无法产生负脉冲的程度。

### 为什么正脉冲不受影响

正脉冲触发条件: mem > thre = 0.12。膜电位很容易超过 0.12 (尤其是 PSN 有 learnable weight+bias)。负脉冲触发条件: mem < -3.6。膜电位几乎不可能到 -3.6——PSN 的 bias 初始化在 [-1, 0] 附近，weight 也是正负混合，输出范围自然在 [-1, 1] 量级。

---

## 方案设计空间

### 方案 S1: 对称三元 (Symmetric Ternary)

**改动**: `neg_thre = thre` (不再乘 30)

```
改前: neg_thre = thre × 30 = 3.6  ← 不可能触发
改后: neg_thre = thre = 0.12       ← 和正阈值相同

输出: {+thre, 0, -thre}  ← 真正对称的三元
```

**直觉**: 正负脉冲应该平等——正负阈值为同一个可学习参数。总发放率 (pos+neg) 驱动阈值更新，natural 地让正负比例在训练中自行平衡。

**优势**: 改动最小 (1 行)，彻底解决负发放问题，训练稳定性高。

**风险**: 负脉冲数量增加 → 总发放率翻倍。需要增大 activity_eta 或调整 target_rate 来控制总稀疏性。

### 方案 S2: 分开的正负阈值 (Decoupled Dual Threshold)

**改动**: 两个独立的可学习参数

```python
self.pos_thre = nn.Parameter(0.1)   # 正阈值
self.neg_thre = nn.Parameter(0.1)   # 负阈值 (独立)

# 阈值更新分别计算:
self.pos_thre += (pos_r - target_pos) × eta_pos × lr_scale
self.neg_thre += (neg_r - target_neg) × eta_neg × lr_scale
```

**优势**: 正负稀疏性独立控制。可以设置不同的 target_rate。

**劣势**: 两个可学习参数 → 训练更复杂。论文需要额外解释"为什么正负需要不同阈值"。

### 方案 S3: 固定负阈值 (Fixed Neg Threshold)

**改动**: 负阈值是常数，不受 ATLIF 更新

```python
self.neg_thre = 0.15  # 固定值，不参与阈值更新
self.pos_thre = nn.Parameter(0.1)  # 正阈值正常更新
```

**优势**: 负脉冲稳定性最高（阈值不会漂移）。正稀疏性独立发育。

**劣势**: 失去对负发放的适应性控制。

### 方案 S4: 归一化阈值门 (Normalized Threshold Gate)

**改动**: 不用绝对阈值，改用相对比例

```python
pos_prob = sigmoid((mem - thre) / thre)          # 正发放概率
neg_prob = sigmoid((-mem - thre) / thre)          # 负发放概率

# 二值化:
pos_spike = (pos_prob > 0.5).float()
neg_spike = (neg_prob > 0.5).float()

out = (pos_spike - neg_spike) × thre
```

**优势**: sigmoid 提供平滑梯度。正负对等缩放。

**劣势**: sigmoid 引入了额外的非线性 (LUT 开销)。和当前 TernarySurrogate 的 STE 机制不兼容。

### 方案 S5: 正负绝对值约束 (Magnitude-Constrained Ternary)

**改动**: 保持当前 ternary surrogate，但输出恒为 `±thre`

当前代码中输出实际是 `spike × surrogate_gradient`，量级不确定。改为:

```python
pos_spike = (mem > thre).float()
neg_spike = (mem < -thre).float()
out = (pos_spike - neg_spike) * thre  # 绝对值恒为 thre
```

同时: `neg_thre = thre` (方案 S1 的条件)。正负用同一个 thre 判断，输出绝对值相等。

---

## 方案对比

| | S1 对称 | S2 独立 | S3 固定 | S4 门控 | S5 约束 |
|---|---|---|---|---|---|
| neg_thre | = thre | 独立参数 | = 0.15 常数 | sigmoid(thre) | = thre |
| 输出绝对值 | 相等 | 可能不等 | 相等 | 相等 | **严格相等** |
| 可学习参数 | 1 | 2 | 1 | 1 | 1 |
| 改动量 | 1行 | ~30行 | 1行 | ~20行 | ~10行 |
| 训练稳定性 | 高 | 中 | 高 | 中 | 高 |
| 论文新颖度 | 中 | 高 | 低 | 高 | 中 |
| 负活动恢复 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 对正活动影响 | 总发放率翻倍 | 可控 | 可控 | 可控 | 总发放率翻倍 |

---

## 推荐

**首选 S1 (对称三元) + S5 (幅值约束)**，作为一个组合方案:

```python
# 核心改动 (atlif_ternary_psn.py line 20, 原为 neg_thre = thre × 30)
neg_thre = thre  # 对称阈值

# 输出约束 (forward 中)
out = (pos_spike - neg_spike) * thre  # 绝对值严格等于 thre
```

**论文定位**: "Symmetric Ternary ATLIF: We replace the asymmetric threshold ratio (30×) with a symmetric design where both polarities share a single adaptive threshold. This recovers negative spike activity naturally and produces strictly magnitude-constrained {+θ, 0, -θ} ternary outputs."

**配合**: 和修正版 SOC、angular loss 组成 I17。改动量 <15 行，不修改已有文件。
