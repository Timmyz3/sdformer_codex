# 三元原生注意力设计空间：完整 Brainstorm

日期: 2026-05-16

---

## 零、SDformer 注意力机制的精确描述

### 0.1 架构约束

SDformer 使用的是 `Spiking_QK_WindowAttention3D`，**不是标准 Q·K^T·V 注意力**：

```
标准 Transformer:  attn = softmax(Q·K^T/√d) · V
SDformer:          attn = K · gate(Q_sum)
                   gate = sn2_q( sum(Q, dim=channel) )
```

关键特征：
- **没有 V 投影** → K 同时扮演 Key 和 Value 的角色  
- **没有 Q·K^T 点积** → 用 Q 的通道坍缩标量做 token gating
- **没有 softmax** → sn2_q 是二元脉冲化 (>=0 → 1, <0 → 0)
- **K 携带 pos_encoding** → 位置信息在 K 侧
- **每个 Q/K 神经元有独立的自适应阈值 θ** → θ ∈ [0.001, 0.13], 训练时动态变化
- **10 个时间步** → Q/K 在时间维度上独立处理

### 0.2 三元脉冲的实际形式

```
标准二元:  q ∈ {0, 1}            → Q_sum ∈ [0, d]       → sn2_q 正常
三元 ATLIF: q ∈ {-θ, 0, +θ}      → Q_sum ∈ [-Σθ, +Σθ]   → sn2_q 截断负值
```

sn2_q 的问题: Q_sum < 0 时输出 0 → 丢失所有负极性 token 的信息。

### 0.3 当前方案的问题

```
compat_qk_product (H9a):
  gate1 = sn2_q(Q_sum)              ← 原始 Q token gating, 负值丢失
  gate2 = shiftmax(Q·K, dim=ch) × n ← Q·K 兼容门, 需要 2^x
  gate = gate1 * gate2              ← 两套机制相乘

问题:
1. gate1 的 sn2_q 截断负 Q_sum → 信息丢失
2. gate2 的 2^x 对 θ-缩放实数不原生 → 硬件 expensive
3. gate1 × gate2 双轨制 → 难以解释, 论文不能作为独立贡献
```

---

## 一、设计空间：从五个视角出发

### 视角 1: 几何视角 — 脉冲作为 Hamming 空间中的点

三元脉冲 q,k ∈ {-θ, 0, +θ}^d。把每个 token 看作 d 维超立方体中的点。

**核心操作**: 衡量两个 token 的"方向一致性"

#### 方案 T1: θ-Weighted Token Voting (θ-WTV)

```
for each channel j:
  if Q_sign[j] == K_sign[j] and both ≠ 0:
    agree    += min(|Q[j]|, |K[j]|)
  elif Q_sign[j] != K_sign[j] and both ≠ 0:
    disagree += min(|Q[j]|, |K[j]|)

confidence = (agree - disagree) / (agree + disagree + silent + ε)
            ∈ [-1, +1]

gate = dual_path_L1_norm(confidence)
```

**为什么 min(θ)**：两个高 θ 的神经元同时发放且同号 → 更强的置信度。一个高 θ 一个低 θ → 置信度被低的一方限制 (短板效应)。两个低 θ → 弱信号。

**为什么分开 agree/disagree**：标准点积 Q·K 把"同向负×负=正"和"同向正×正=正"混在一起，丢失了符号信息。分开计数保留了"是正向一致还是负向一致"的区分。

#### 方案 T4: Angular Hamming Similarity (AHS)

```
把三元脉冲映射到单位球面:
q_norm = Q / max(|Q|, ε)     ∈ [-1, +1]  (注意 0 保持为 0)

cos_sim = (q_norm · k_norm) / (||q_norm|| · ||k_norm||)
        = Σ(q[j]·k[j]) / sqrt(Σq[j]² · Σk[j]²)

# 对三元脉冲, 这退化为:
num_nonzero_q = count(q_norm ≠ 0)
num_nonzero_k = count(k_norm ≠ 0)
ham_sim = Σ(q_norm[j] · k_norm[j]) / sqrt(num_nonzero_q · num_nonzero_k)

gate = softplus(ham_sim + pos_bias)
```

**物理直觉**: Hamming 相似度测量两个 token 在"哪些通道上同时活跃"以及"活跃时的符号是否一致"。除以活跃通道数的几何平均 → 归一化到 [-1, +1]。

#### 方案 T5: Sign-Only Consensus (SOC)

```
最极端的简化 — 完全忽略 θ, 只看符号:

agree_signs    = popcount(Q_sign == K_sign and both ≠ 0)
disagree_signs = popcount(Q_sign != K_sign and both ≠ 0)

consensus = (agree_signs - disagree_signs) / (agree_signs + disagree_signs + ε)
          ∈ [-1, +1]

gate = L1_norm(softplus(consensus))
```

**硬件**: 纯 popcount + 除法。θ 完全不参与 attention — θ 只用于 ATLIF 稀疏化。注意力和稀疏化完全解耦。

**论文角度**: "We disentangle spike confidence (θ) from attention computation — θ controls sparsity through ATLIF, while pure sign consensus drives attention. This two-factor design is both simpler in hardware and more interpretable."

---

### 视角 2: 信息论视角 — 注意力作为互信息估计

#### 方案 T6: Spike Coincidence Mutual Information (SCMI)

```
把 Q 和 K 视为两个二元随机变量的 d 次观测:

P(Q>0, K>0) = count(Q>0 & K>0) / d    ← 同正
P(Q<0, K<0) = count(Q<0 & K<0) / d    ← 同负
P(Q>0, K<0) = count(Q>0 & K<0) / d    ← 符号冲突
P(Q=0 or K=0) = count(any zero) / d    ← 沉默

# 点互信息 (PMI):
pmi_pos = log( P(Q>0,K>0) / (P(Q>0)·P(K>0)) )
pmi_neg = log( P(Q<0,K<0) / (P(Q<0)·P(K<0)) )
pmi_conflict = log( P(sign conflict) / (P(Q≠0)·P(K≠0)) )

mi_score = pmi_pos + pmi_neg - pmi_conflict
gate = softplus(mi_score + pos_bias)
```

**物理直觉**: PMI 测量 "Q 和 K 的符号一致性是否显著偏离随机独立"。正的 PMI → 两个 token 有显著的符号关联 → 它们的交互值得关注。负的 PMI → 符号随机 → 降低关注。

**硬件**: P 的估计需要除法, log 需要小 LUT (或近似 log2 via bit-length)。

#### 方案 T7: Entropy-Gated Attention (EGA)

```
# Q token 的"信息量" — 活跃通道越多, 熵越高
q_entropy = -Σ p_j log(p_j)
  where p_j = |Q[j]| / Σ|Q[j]|    ← θ-weighted 激活分布

# 均匀发放 (所有通道等概率) → 高熵 → 信息量大
# 集中发放 (少数通道主导) → 低熵 → 信息量小

# K token 的信息增益:
k_entropy_given_q = conditional entropy simplified:
  k_active_given_q = count(K≠0 | Q≠0) / count(Q≠0)
  
mutual_info ≈ q_entropy - (1 - k_active_given_q) × q_entropy
            = q_entropy × k_active_given_q

gate = softplus(mutual_info - baseline_entropy)  # 只有超预期的 MI 才被放大
```

**物理直觉**: 如果 Q 的激活模式很"信息丰富" (高熵) 且 K 倾向于在 Q 激活的通道上共同激活 → 高 MI → 强注意力。如果 Q 只有一两个通道激活 (低熵) → MI 自动降低。

---

### 视角 3: 统计视角 — 注意力作为异常检测

#### 方案 T8: Spike Outlier Detection Attention (SODA)

```
# 对每个 token pair, 计算 Q 和 K 的"异常一致性":

# 全局统计 (batch 级别):
E[Q·K] = mean over all token pairs of (Q_norm · K_norm)
Var[Q·K] = variance over all token pairs

# 对特定 token pair (i,j):
score_ij = (Q[i]·K[j] - E[Q·K]) / sqrt(Var[Q·K] + ε)

# 只有显著偏离均值的 pair 才获得高注意力:
gate_ij = softplus(score_ij)

# 然后坍缩到 token 级:
gate_i = Σ_j gate_ij / n
```

**物理直觉**: 大多数 token pair 的 Q·K 在某个均值附近。真正"相关"的 pair 会显著偏离这个均值 → 这些才是值得关注的。

**硬件**: 需要 batch 级统计 (均值/方差)。训练时 batch 内有足够样本；推理时可以 running mean/var。

#### 方案 T9: Rank-Based Attention (RBA)

```
# 不关心 Q·K 的绝对值, 只关心排位:

score_ij = Q[i] · K[j]                    # 原始兼容性分数
rank_ij = argsort(score_i)                # token i 对所有 j 的兼容性排名
                                          # 高排名 = K[j] 是 Q[i] 最兼容的 token

# Sigmoid 衰减的 rank 权重:
gate_ij = sigmoid( (n/2 - rank_ij) / temperature )

# Token 级 gate:
gate_i = Σ_j gate_ij / n
```

**物理直觉**: 不是所有 K token 都对 Q token 同等重要。只有 top-k 最兼容的值得关注。Rank 天然对 θ 的绝对大小不敏感 — 只关心相对顺序。

**硬件**: argsort 最贵 (O(n log n))。但 window size 只有 2×9×9=162 → 小规模排序可行。可用 top-k 选择器替代全排序。

---

### 视角 4: 生物视角 — STDP 启发

#### 方案 T10: Spike-Timing Coincidence Gate (STCG)

```
生物 STDP: 突触权重变化 ∝ Δt = t_post - t_pre
           pre-before-post → LTP (增强)
           post-before-pre → LTD (抑制)

在 SDformer 中, 10 个时间步可以看作 10 个"发放时刻":

# 每个时间步的 Q 和 K 激活:
Q_active[t] = (Q[t] ≠ 0)    ← bool
K_active[t] = (K[t] ≠ 0)    ← bool

# 时间上的重合度:
coincidence[t] = Q_active[t] & K_active[t]
early_coincidence = Σ_{t=1..3} coincidence[t]    # 早期重合
late_coincidence  = Σ_{t=7..10} coincidence[t]   # 晚期重合

# STDP-like gate (晚期重合更重要, 因为携带运动信息):
gate = Σ_t coincidence[t] × w[t]
  where w[t] = sigmoid((t - T/2) / τ)     # 单调递增的时域权重
```

**物理直觉**: 事件相机早期时间步主要是噪声, 晚期时间步携带相干运动信号。Q 和 K 在晚期时间步的重合比早期更有意义。

**硬件**: 时间维度的 popcount + 加权求和。权重 w[t] 是固定的 (查表)。

#### 方案 T11: Refractory-Coincidence Gate (RCG)

```
生物神经元发放后有不应期 → 连续发放之间有最小间隔。

对三元脉冲, "不应期重合"意味着:
两个 token 在相同时间步都是静默的(因为它们都刚发过) → 这不是负信号
两个 token 在相同时间步都是活跃的 → 这是正信号

refractory_state[t] = (Q[t-1]≠0 or Q[t-2]≠0)  # Q 在不应期中
valid_spike[t] = (Q[t]≠0) and not refractory_state[t]

coincidence[t] = valid_spike_Q[t] & valid_spike_K[t] & (sign(Q[t])==sign(K[t]))
gate = Σ_t coincidence[t] / Σ_t (valid_spike_Q[t] | valid_spike_K[t])
```

---

### 视角 5: 硬件原语视角 — 只用比较器和加法器能做什么

#### 方案 T12: Threshold-Max Pooling Attention (TMPA)

```
# 对每个 K token, 找出最"确信"的通道:
k_max_channel = argmax(|K[j]|)     ← 最大 θ 的通道
k_max_sign = sign(K[k_max_channel])
k_max_mag  = |K[k_max_channel]|

# 对每个 Q token, 同样:
q_max_channel = argmax(|Q[j]|)
q_max_sign = sign(Q[q_max_channel])
q_max_mag  = |Q[q_max_channel]|

# 如果 Q 和 K 的最大 θ 通道符号一致:
if q_max_sign == k_max_sign:
  confidence = min(q_max_mag, k_max_mag)
else:
  confidence = -min(q_max_mag, k_max_mag)

# 对所有通道也做一次加权:
global_agree = count(Q_sign == K_sign & both≠0)
global_disagree = count(Q_sign ≠ K_sign & both≠0)
global_conf = (global_agree - global_disagree) / d

# 组合:
gate = softplus(confidence * global_conf + pos_bias)
```

**硬件**: argmax = 比较器树 (log d 深度)。纯组合逻辑，零乘法器。

#### 方案 T13: Popcount-Ratio Attention (PRA)

```
最简单的方案 — 回到二元 popcount:

# 把三元退化到符号:
Q_sign_only = sign(Q) ∈ {-1, 0, +1}
K_sign_only = sign(K) ∈ {-1, 0, +1}

# 三种 popcount:
n_agree = popcount(Q_sign_only == K_sign_only & ≠ 0)
n_disagree = popcount(Q_sign_only == -K_sign_only & ≠ 0)
n_q_active = popcount(Q_sign_only ≠ 0)
n_k_active = popcount(K_sign_only ≠ 0)

# Jaccard-like 相似度:
jaccard = n_agree / (n_q_active + n_k_active - n_agree + ε)

# 调整重合度:
adjusted = (n_agree - n_disagree) / max(n_q_active, n_k_active, 1)

# 组合 (Jaccard 结构, adjusted 方向):
score = jaccard * sign(adjusted) * sqrt(|adjusted|)
gate = L1_norm(softplus(score))
```

**硬件**: 3 个 popcount + 2 除法。最便宜。

---

## 二、全部 13 个注意力方案的统一对比

| 方案 | 核心运算 | θ 使用 | 时域 | 硬件复杂度 | 论文差异度 | 直觉来源 |
|------|---------|:---:|:---:|:---:|:---:|------|
| T1 θ-WTV | min(θ)+popcount | 加权 | 否 | ⭐ 最低 | ⭐⭐⭐ | Hamming + 置信度 |
| T3 MCG | min(θ)+sum | 门控 | 否 | ⭐ 最低 | ⭐⭐⭐ | θ 作为可靠性 |
| T4 AHS | 归一化点积 | 隐式 | 否 | ⭐⭐ | ⭐⭐ | 余弦相似度 |
| **T5 SOC** | **纯popcount** | **无** | **否** | **⭐ 最低** | **⭐⭐⭐** | **符号共识** |
| T6 SCMI | PMI + log | 间接 | 否 | ⭐⭐ | ⭐⭐⭐ | 互信息 |
| T7 EGA | 熵 + MI | 加权 | 否 | ⭐⭐⭐ | ⭐⭐⭐ | 信息熵 |
| T8 SODA | 均值/方差+Zscore | 隐式 | 否 | ⭐⭐ | ⭐⭐ | 异常检测 |
| T9 RBA | argsort+sigmoid | 隐式 | 否 | ⭐⭐⭐ | ⭐⭐ | 排名 |
| **T10 STCG** | **时域popcount** | **无** | **是** | **⭐⭐** | **⭐⭐⭐** | **STDP生物** |
| T11 RCG | 不应期+popcount | 无 | 是 | ⭐⭐ | ⭐⭐ | 不应期生物 |
| T12 TMPA | argmax+比较 | 取max | 否 | ⭐⭐ | ⭐⭐ | 硬件原语 |
| **T13 PRA** | **3×popcount** | **无** | **否** | **⭐ 最低** | **⭐⭐** | **Jaccard** |

**加粗的五个 (T5, T10, T13 + T1, T3) 是最推荐的** — 它们代表五种完全不同的设计哲学。

---

## 三、AAE 防爆方案详细展开

### A1: 角度感知 Loss

**论文来源**: Cuadrado et al. "Optical flow estimation from event-based cameras and spiking neural networks" (Frontiers in Neuroscience, 2023) — 首次在 SNN 光流中使用角度 loss。

**公式**:

```
给定预测光流 (u_pred, v_pred) 和真值 (u_gt, v_gt):

cos_angle = (u_pred·u_gt + v_pred·v_gt + ε²) /
            (√(u_pred²+v_pred²+ε²) · √(u_gt²+v_gt²+ε²))

L_angular = arccos(clamp(cos_angle, -1+ε, 1-ε))
           ≈ √(1 - cos_angle²) / cos_angle   ← 小角度近似, 无 arccos

L_total = λ_mod × L1 + λ_ang × L_angular + λ_firing × firing_penalty
```

**为什么有效**: 当前的 L1 loss 只惩罚端点距离，不关心方向。两个光流向量的端点距离相同但方向不同 → L1 惩罚相同 → 模型没有动力纠正方向。L_angular 直接惩罚角度偏差 → 模型学会保护方向信息。

**为什么对 stage2 FFN 有效**: 当 stage2 FFN 输出分布被 ATLIF 改变时，decoder 收到扭曲的特征 → 产生"方向错误但幅值正确"的光流。L1 loss 不敏感 (幅值差不多)，但 L_angular 会强烈惩罚 → 迫使模型在训练中纠正方向。

**超参数建议**: λ_ang = 0.3~0.5 (角度 loss 的数值范围比 L1 小，需要稍大的权重)。ε = 1e-6。

### A3: 渐进稀疏调度

**论文来源**: ISKD "Iterative Structured Knowledge Distillation" (COLING 2025) — progressively replace transformer blocks。FuseGPT "From Pruning to Grafting" (arXiv 2024) — iterative knowledge redistribution。

**为什么直接加稀疏压力在深层会失败**:
- ATLIF 的阈值更新是局部的 (只看自己的发放率 vs target_rate)
- 深层 FFN 的阈值增长会改变整个 stage 的输出分布
- Decoder 在训练初期没有机会适应这个新分布
- 深层特征的变化被逐层放大 (bottleneck 效应)

**渐进调度**:

```
Stage0 FFN (浅层, 冗余多):
  epoch 0-30: activity_eta = 0.03 (全量, 始终)

Stage1 FFN (中层):
  epoch 0-5:   activity_eta = 0.0
  epoch 6-15:  activity_eta = 0.01
  epoch 16-30: activity_eta = 0.03

Stage2 FFN (深层 bottleneck, 最敏感):
  epoch 0-10:  activity_eta = 0.0        ← 纯三元动力学适应
  epoch 11-20: activity_eta = 0.005      ← 微弱的稀疏信号
  epoch 21-30: activity_eta = 0.02       ← 半量, 永不全量
```

**为什么有效**: decoder 在训练早期看到的是接近原始分布的特征 (eta=0 时期)，建立了正确的方向映射。稀疏压力逐渐引入时，decoder 已经学会了方向判断的基础模式，可以在保持方向的前提下适应特征分布的变化。

---

## 四、推荐实验路径

### 快速路线 (本周可跑)

```
I11: T13 (PRA) + A1 (角度Loss) + A3 (stage2 渐进)
  T13 在最简单的 popcount 层面验证"不需要 2^x"的假设
  A1 直接给方向信号
  A3 保护 stage2
```

### 创新路线 (适合论文)

```
I12: T5 (SOC) + T10 (STCG) + A1 (角度Loss) + A3 (渐进)
  T5 做符号共识: "θ 只管稀疏, 符号管注意力" 的两因素理论
  T10 做时域增强: "晚期重合比早期重, 事件数据的运动信息在晚期"
  A1 保护 AAE
```

### 消融实验设计 (论文必需)

```
Baseline:     H9a (compat_qk_product)
Ablation 1:   T5 (SOC only, 无角度Loss)
Ablation 2:   T13 (PRA only, 无角度Loss)
Ablation 3:   H9a + A1 (角度Loss only, 不改注意力)
Ablation 4:   T5 + A1 + A3 (完整方案)
Ablation 5:   T13 + A1 + A3
```

**可回答的问题**: "是注意力归一化重要还是角度 loss 重要？""符号共识是否足以替代 Shiftmax？""渐进调度贡献了多少？"
