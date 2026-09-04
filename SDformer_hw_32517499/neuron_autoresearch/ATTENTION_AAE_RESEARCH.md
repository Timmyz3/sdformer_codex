# Attention + AAE 研究：三元原生注意力和防 AAE 爆炸方案

日期: 2026-05-16 | 约束: 不修改已有文件，仅新增

---

## 一、问题重述

### 1.1 AAE 爆炸问题

| 实验 | FFN 替换层 | AAE | 现象 |
|------|----------|:---:|------|
| H9a | stage0 FFN + downsamples | 7.64 | ✅ 安全 |
| H9e | layers0+1 even blocks | 7.68 | ✅ 安全 |
| H9c | stage2 FFN blocks | **31.2** | ❌ 爆炸 |
| H10 | layers0+1 FFN (qkformer) | **71.6** | ❌ 完全崩溃 |

**根因定位**:
- stage2 是 encoder bottleneck (6 blocks, 12 heads, 768→3072 MLP)
- 二元 ATLIF 替换改变了 FFN 输出分布 → decoder 接收被扭曲的特征 → 方向误差累积
- 当前只用 flow L1 loss + firing penalty — **没有任何方向感知的正则项**

### 1.2 注意力设计问题

当前的 `compat_qk_product` 模式:
- 双轨制: 原始 Q token gating + Q·K^T 兼容门相乘
- Shiftmax 使用 `2^x` — 对三元 θ-scaled 输入不是原生的
- 需要 `sn2_q` 脉冲化 (丢掉一半负值信息)
- 论文角度: 不能被解释为独立创新, 只是 BSA 的 hack 式移植

---

## 二、AAE 防爆方案

### 方案 A1: 角度感知 Loss (Angular-Aware Loss)

**来源**: Cuadrado et al. (Frontiers in Neuroscience, 2023) 首次在 SNN 光流中使用角度 loss

**机制**:
```
L_total = L_supervised + L_sparse_activity + λ_ang × L_angular

L_angular = arccos( (u_pred·u_gt + v_pred·v_gt + ε²) /
                    (sqrt(u_pred²+v_pred²+ε²) × sqrt(u_gt²+v_gt²+ε²)) )
```

**改动**: 新增 `neuron_autoresearch/losses/angular_loss.py`，在 train.py 的 LOSS_PATCH 中注入。λ_ang 从 config 读。初始建议 λ_ang=0.1~0.5。

**预期**: 直接惩罚方向误差，AAE 不会在深层 FFN 替换时爆炸，因为 loss 有明确的方向信号。

**硬件**: 训练阶段 only (无推理开销)。推理时 AAE 改善来自更好的权重。

---

### 方案 A2: FFN 输出分布匹配 (Feature Distribution Matching)

**来源**: Prob-AMC (Mathematics 2025), Boosting Pruned Networks (ICASSP 2024)

**机制**:
```
# 只在 stage2 FFN 被替换时启用
L_fdm = KL( softmax(FFN_sparse_output / T) || softmax(FFN_baseline_output / T) )
```

在训练初期，用基线 PSN 模型计算 stage2 FFN 的原始输出分布，用 KL 散度约束稀疏版本不要偏离太远。随着训练推进，逐渐衰减 λ_fdm。

**改动**: 新增 `neuron_autoresearch/losses/fdm_loss.py`，需要跑一次基线 forward 收集 stage2 FFN 的 target distributions。

**预期**: stage2 FFN 的输出分布被约束在基线附近 → decoder 接收的特征保持一致性 → AAE 被保护。

**风险**: 需要额外的 teacher forward pass，增加显存。

---

### 方案 A3: 渐进稀疏调度 (Progressive Sparsity Schedule)

**来源**: ISKD (COLING 2025), FuseGPT (arXiv 2024)

**机制**: 对深层 FFN 的 activity_eta 做渐进式升温，而不是一开始就施加全量稀疏压力:

```
Stage2 FFN activity_eta 调度:
  epochs 0-5:   eta = 0.0        (纯适应 ATLIF 动力学)
  epochs 6-15:  eta = 25% max    (温和)
  epochs 16-25: eta = 60% max    (加速)
  epochs 26-30: eta = 100% max   (全量, 同时 λ_ang 激活)
```

**改动**: 仅修改 config — 在 `atlif_ternary_psn.target_groups` 的每个 group 中新增 `eta_schedule` 字段。需在 installer.py (新增文件) 中实现 schedule 解析。

**预期**: 深层 FFN 有足够时间在稀疏压力到来前适应 ATLIF 动力学分布，减少方向信息丢失。

---

### 方案 A4: 分层特征蒸馏 (Layer-wise Feature Distillation)

**来源**: ISKD, LAPTOP-Diff

**机制**: 训练时 teacher = PSN baseline (frozen)，student = H9a 配置。对 stage2 FFN 的输入和输出做 MSE 对齐:

```
L_distill = MSE(h_student, h_teacher) / dim
```

只在 epoch 0-10 启用，之后衰减到 0。

**预期**: 最强保护 — 直接迫使 student 在 stage2 的特征表示匹配 teacher。

**风险**: 额外显存 (需加载第二个模型)。

---

### 方案对比

| 方案 | 改动量 | 额外显存 | 论文创新性 | 预期效果 | 风险 |
|------|:---:|:---:|:---:|:---:|:---:|
| A1 角度 Loss | 小 (~50行) | 零 | 中 (首次在 SNN+Transformer 光流中用) | 中等 | 低 |
| A2 FDM | 中 (~150行) | 中 (teacher stats) | 高 (分布匹配用于 SNN) | 高 | 中 |
| A3 渐进调度 | 小 (~30行 config) | 零 | 低 (调度本身不算创新) | 中等 | 低 |
| A4 特征蒸馏 | 大 (~200行) | 高 (双模型) | 高 | 最高 | 中 |

**建议**: A1+A3 组合 — 改动量最小, 零额外显存, 论文可写 "we propose Angular-Aware Progressive Sparsity (AAPS) to protect direction-sensitive features during sparse training".

---

## 三、原生三元注意力方案

### 方案 T1: θ-Weighted Token Voting (θ-WTV)

**核心思路**: 完全替代 `compat_qk_product` 双轨制。基于三元脉冲的符号-量级结构直接计算 token confidence，零指数运算。

```
输入: Q, K ∈ {-θ, 0, +θ}^{T×B×H×N×d}
  θ 是 per-neuron 的自适应阈值

步骤:
1. 拆分符号和量级:
   Q_sign = sign(Q)  ∈ {-1, 0, +1}
   Q_mag  = |Q|      ∈ {0, θ}

2. 计算 token-wise 加权同意度:
   agree[i]    = Σ_j min(Q_mag[j], K_mag[j])    for Q_sign[j]==K_sign[j] 且都不为0
   disagree[i] = Σ_j min(Q_mag[j], K_mag[j])    for Q_sign[j]!=K_sign[j] 且都不为0
   silent[j]   = count(Q_sign[j]==0 or K_sign[j]==0)

3. Token confidence:
   confidence = (agree - disagree + ε) / (agree + disagree + silent + ε)
   confidence ∈ [-1, +1], 负值表示该 token 整体与参考方向相反

4. 正负路径分开归一化:
   pos_conf = ReLU(confidence)  / ΣReLU(confidence)  × n/2
   neg_conf = ReLU(-confidence) / ΣReLU(-confidence) × n/2
   gate = pos_conf - neg_conf    ← 符号感知的 gate

5. 输出:
   attn = K × gate
```

**论文差异化 (vs BSA/Shiftmax)**:
- Shiftmax 用 2^x 近似 softmax → θ-WTV 用 min(θ) 和 popcount
- Shiftmax 是通用 softmax 替代 → θ-WTV 是三元脉冲原生方案
- 零 LUT, 零 exp, 零 bit-shift → 纯比较器 + 加法树 + 除法器

**硬件映射**:
- XNOR 符号比较: 1 gate per channel pair
- min(θ_q, θ_k): 1 比较器 per channel pair
- 加法树: Σ agree, Σ disagree
- 除法: 1 per token (n tokens)

**改动**: 新增 `neuron_autoresearch/attention/theta_wtv.py` + 新 config section `bsa_attention.mode: theta_wtv`。在现有 H9 bsa_attention.py 的同目录下新增文件（不是修改）。

---

### 方案 T2: Signed Dual-Path Attention (SDPA)

**核心思路**: 把三元 Q 的正负极性分成两个独立路径，分别计算再合并。

```
pos_Q = ReLU(Q)  ∈ {0, +θ}    neg_Q = ReLU(-Q) ∈ {0, +θ}
pos_K = ReLU(K)               neg_K = ReLU(-K)

# 正极性路径 (标准 Q token gating):
pos_score = pos_Q.sum(dim=-1)
pos_gate = L1_norm(pos_score) × n/2

# 负极性路径 (符号翻转后走同样逻辑):
neg_score = neg_Q.sum(dim=-1)
neg_gate = L1_norm(neg_score) × n/2

# 合并:
gate = pos_gate - neg_gate
attn = K × gate
```

**论文角度**: BSA 的双极自注意力提出三元矩阵乘积 (TMP)，但需要 Q·K^T matrix multiply。SDPA 将它简化为 SDformer QK-attention 兼容的形式 — 正负分两路、L1 各自归一化、合并。

**硬件**:
- ReLU: 1 比较器
- L1 norm: 加法树 + 除法器
- 两路并行 → 2× 加法树 (可共享)

---

### 方案 T3: Min-Confidence Gating (MCG)

**核心思路**: 最简方案 — 用 min(θ_q, θ_k) 作为注意力置信度的唯一信息源，不依赖任何 softmax 类归一化。

```
# 每个 attention head 的 per-channel 置信度:
conf_c = min(|Q_c|, |K_c|)  / θ_max    ← 比较器, 零乘法器

# Token 级置信度 (通道维度求和):
token_conf = conf_c.sum(dim=-1) / d    ← ∈ [0, 1]

# 原始 Q token gating, 用置信度调制:
scores = Q.sum(dim=-1)                   ← 和 H10 qkformer 一样
scores = scores × token_conf             ← 低置信度 token 被 suppress
gate = L1_norm(scores) × n               ← 简单的 L1 归一化
attn = K × gate
```

**直觉**: 如果两个 token 的 θ 都很小 (廉价脉冲)，它们之间的 attention 是不可靠的 → suppress。如果 θ 都很大 (贵重脉冲) → amplify。

**论文角度**: 
- 和 Shiftmax 完全不同 — 不需要任何 softmax 类变换
- 直接利用 ATLIF 的 θ 作为 attention 可靠性度量
- "Self-calibrating attention without softmax: the adaptive threshold θ naturally encodes per-neuron confidence"

---

### 方案对比

| 方案 | 核心运算 | 乘法器 | 硬件复杂度 | 论文差异化 | 实现风险 |
|------|---------|:---:|:---:|:---:|:---:|
| T1 θ-WTV | popcount + min() + div | 零 | 最低 | 最高 | 中 |
| T2 SDPA | ReLU + L1 + div | 零 | 低 | 高 | 低 |
| T3 MCG | min() + sum + div | 零 | 最低 | 高 | 低 |
| 当前 compat | Q·K^T + 2^x + div×2 | 有 | 高 | 零 | — |

---

## 四、推荐实验组合

### 优先: A1 + T3 (最小改动, 最快验证)

```
I11: H9a 基底 + 角度 Loss + MCG 注意力 + 渐进 A3 调度
  - 角度 Loss 保护 AAE
  - MCG 替代 compat_qk_product (零 2^x, 零乘法器)
  - stage2 FFN 用渐进调度 (A3)
  - 目标: AAE < 7.5 (基线水平), SOPs < 3.0G
```

### 次选: A1 + T1 (最大创新, 适合论文)

```
I12: H9a 基底 + 角度 Loss + θ-WTV + 渐进 A3
  - θ-WTV 作为独立创新: "Ternary-Native Token Voting without Softmax"
  - 角度 Loss 保护 AAE
  - 目标: AEE < 1.50, AAE ≈ 7.5, SOPs < 3.0G
```

---

## 五、待新增文件清单

```
neuron_autoresearch/
├── losses/
│   ├── angular_loss.py          # A1: 角度 loss
│   └── fdm_loss.py              # A2: FFN 分布匹配 (可选)
├── attention/
│   ├── theta_wtv.py             # T1: θ-加权投票
│   ├── sdpa.py                  # T2: 双路径
│   └── mcg.py                   # T3: min-置信度门控
├── scheduling/
│   └── progressive_sparsity.py  # A3: 渐进稀疏调度解析
└── experiments/
    ├── i11_mcg_angular/configs/  # I11 配置
    └── i12_wtv_angular/configs/  # I12 配置
```

注: 所有文件都是新增，不修改任何已有文件。
