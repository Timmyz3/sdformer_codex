# 实验全景记录

最后更新: 2026-05-22 | 135+ short profiles, 4 轮实验

---

## 一、全部短测结果

### Phase 1: 14 attn × S0 FFN, max_thre=0.5, 80-step, valid5

注意: max_thre=0.5 太低，阈值被锁死，SOPs 普遍偏高。结果仅供参考相对排序。

| rank | attn | AEE | AAE | SOPs | firing | gate |
|---:|------|-----:|-----:|-----:|-----:|:---:|
| 1 | SC signed_consensus | 0.96 | 6.46 | 4.51G | 0.106 | ❌ |
| 2 | SN signed_shiftnorm | 0.95 | 6.54 | 4.51G | 0.106 | ❌ |
| 3 | SL signed_popcount_l1 | 1.05 | 6.83 | 4.36G | 0.102 | ❌ |
| 4 | HT hamming_ternary | 1.00 | 6.87 | 4.23G | 0.099 | ❌ |
| 5 | TX ternary_axnor | 0.95 | 6.93 | 4.45G | 0.104 | ❌ |
| 6 | CP compat_qk (H9a) | 1.03 | 7.02 | 4.28G | 0.100 | ❌ |
| 7 | BQ strict_bsa_qkv | 1.01 | 7.13 | 5.43G | 0.127 | ❌ |
| 8 | AD a2os2a_direct | 1.10 | 7.45 | 5.29G | 0.124 | ❌ |
| 9 | AQ a2os2a_qkv | 1.18 | 7.45 | 5.56G | 0.130 | ❌ |
| 10 | BS strict_bsa_adapt | 1.18 | 7.53 | 5.12G | 0.120 | ❌ |
| 11 | TL ternary_axnor_l1 | 1.22 | 7.77 | 5.34G | 0.125 | ❌ |
| 12 | HB hamming_binary | 1.30 | 8.51 | 4.80G | 0.113 | ❌ |

### Phase 2-S02: 6 attn × stage0+2 FFN, max_thre=2.0, 80-step, valid5

| rank | attn | AEE | AAE | SOPs | firing | gate |
|---:|------|-----:|-----:|-----:|-----:|:---:|
| 1 | **SN** signed_shiftnorm | 0.96 | **7.06** | **3.23G** | 0.076 | ✅ |
| 2 | **SC** signed_consensus | 1.00 | 7.08 | 3.29G | 0.077 | ✅ |
| 3 | SL signed_popcount_l1 | 1.01 | 7.31 | 3.35G | 0.079 | ❌ |
| 4 | TX ternary_axnor | 1.02 | 7.57 | 3.33G | 0.078 | ❌ pos_neg 炸 |
| 5 | HT hamming_ternary | 1.11 | 8.07 | 3.15G | 0.074 | ❌ |
| 6 | CP compat_qk | 1.08 | 8.11 | 3.09G | 0.072 | ❌ |

### Phase 2-S012: 6 attn × stage0+1+2 FFN, max_thre=2.0, 80-step, valid5

| rank | attn | AEE | AAE | SOPs | firing | gate |
|---:|------|-----:|-----:|-----:|-----:|:---:|
| 1 | **TX** ternary_axnor | 0.98 | **7.04** | **2.92G** | 0.069 | ✅ |
| 2 | **SN** signed_shiftnorm | 1.04 | 6.89 | 2.98G | 0.070 | ✅ |
| 3 | **HT** hamming_ternary | 1.11 | 7.30 | 2.96G | 0.069 | ✅ |
| 4 | SC signed_consensus | 1.01 | 6.51 | 3.11G | 0.073 | ❌ pos_neg 炸 |
| 5 | CP compat_qk | 1.15 | 8.18 | 2.82G | 0.066 | ❌ |
| 6 | SL signed_popcount_l1 | 1.09 | 7.49 | 2.99G | 0.070 | ❌ pos_neg 炸 |

### Phase 2-N: 6 attn × 无 FFN, max_thre=2.0, 80-step, valid5

| rank | attn | AEE | AAE | SOPs | firing | gate |
|---:|------|-----:|-----:|-----:|-----:|:---:|
| 1 | SC signed_consensus | 0.87 | 6.25 | 3.90G | 0.091 | ❌ |
| 2 | TX ternary_axnor | 0.88 | 6.45 | 4.00G | 0.094 | ❌ |
| 3 | SL signed_popcount_l1 | 0.89 | 6.99 | 3.89G | 0.091 | ❌ |
| 4 | SN signed_shiftnorm | 0.91 | 7.01 | 3.89G | 0.091 | ❌ |
| 5 | HT hamming_ternary | 0.94 | 7.20 | 3.83G | 0.090 | ❌ pos_neg 炸 |
| 6 | CP compat_qk | 1.03 | 7.59 | 3.78G | 0.089 | ❌ |

---

## 二、历史批次所有结果

### H37 main batch (max_thre=1.8, 120+360 steps, valid10)

| rank | attn | steps | AEE | AAE | SOPs | gate |
|---:|------|:---:|-----:|-----:|-----:|:---:|
| 1 | binary_axnor_l1 neuronfast | 360 | 1.07 | 6.35 | 3.54G | ❌ |
| 2 | strict_bsa_qkv conservative | 360 | 1.06 | 6.44 | 3.62G | ❌ |
| 3 | signed_consensus conservative | 360 | 1.02 | 6.17 | 3.72G | ❌ |
| 4 | strict_bsa_signv conservative | 360 | 1.09 | 6.46 | 3.68G | ❌ |
| 5 | binary_axnor_shiftmax cons | 360 | 1.07 | 6.51 | 3.82G | ❌ |
| 6 | a2os2a_qkv neuronfast | 360 | 1.04 | 6.41 | 3.91G | ❌ |
| 7 | a2os2a_qkv conservative | 360 | 1.07 | 6.39 | 4.08G | ❌ |

### H23 low LR + sparse combo (120+360 steps)

| rank | attn | steps | AEE | AAE | SOPs | gate |
|---:|------|:---:|-----:|-----:|-----:|:---:|
| 1 | h23e_h13v_lr1e5_target035 | 120 | **1.50** | **7.37** | 3.59G | valid40 |
| 2 | h23a_h18c_lr1e5_target040 | 120 | 1.52 | 7.47 | 3.63G | valid40 |
| 3 | h23b_h18c_lr1e5_target035 | 120 | 1.55 | 7.63 | 3.53G | valid40 |
| 4 | h23d_h13v_lr1e5_target040 | 120 | 1.56 | 7.81 | 3.64G | valid40 |

### H18 direct + H13fix (120 steps, valid10 unless noted)

| rank | attn | AEE | AAE | SOPs | gate |
|---:|------|-----:|-----:|-----:|:---:|
| 1 | h13v_lower_lr | 0.96 | 5.90 | 3.83G | — |
| 2 | h13w_sparse_stronger | 0.99 | 5.73 | 3.73G | — |
| 3 | h18c_alpha_xnor_direct | 1.09 | 6.73 | 3.81G | — |

### H22 H18c hyperparam sweep (120 steps, valid10)

| rank | attn | AEE | AAE | SOPs |
|---:|------|-----:|-----:|-----:|
| 1 | h22c_target035_eta08_act10 | 1.03 | 6.05 | 3.71G |
| 2 | h22j_sign_value | 1.07 | 6.87 | 3.81G |
| 3 | h22e_score0p5 | 1.09 | 7.00 | 3.80G |

### H21 Hamming (120 steps)

| rank | attn | AEE | AAE | SOPs |
|---:|------|-----:|-----:|-----:|
| 1 | h21b_hamming_ternary | 1.10 | 6.37 | 3.74G |
| 2 | h21c_hamming_binary_signv | 1.10 | 7.03 | 4.10G |

### H24 H9a scope + alpha-XNOR (120 steps)

| rank | attn | AEE | AAE | SOPs |
|---:|------|-----:|-----:|-----:|
| 1 | h24b_lr1e5 | 1.10 | 6.07 | 3.75G |
| 2 | h24a_base | 1.13 | 5.99 | 3.73G |

### H25 module combinations (120 steps)

| rank | attn | AEE | AAE | SOPs |
|---:|------|-----:|-----:|-----:|
| 1 | h25g_ffn_all_binary_no_ds | 1.52 | 7.76 | 3.72G |
| 2 | h25f_ffn_all_ternary | 1.62 | 7.48 | 3.63G |

### H26 attention revisit (120 steps)

| rank | attn | AEE | AAE | SOPs |
|---:|------|-----:|-----:|-----:|
| 1 | h26h_hamming_ternary_sparse035 | 1.60 | 7.70 | 3.51G |
| 2 | h26c_hamming_ternary_sparse040 | 1.61 | 8.33 | 3.65G |

### H27 strict BSA (120 steps)

| rank | attn | AEE | AAE | SOPs |
|---:|------|-----:|-----:|-----:|
| 1 | h27a_strict_bsa_signv_sqrt | 1.54 | 7.87 | 3.71G |
| 2 | h27b_strict_bsa_thetav_sqrt | 1.57 | 7.87 | 3.64G |

---

## 三、跨 FFN 对比结论

| FFN | 最佳 AAE | 最佳 SOPs | 最佳注意力 |
|:---:|-----:|-----:|------|
| S0 (stage0 only) | 6.46 | 4.23G | SC |
| **S02** (stage0+2) | **7.06** | **3.09G** | **SN** |
| **S012** (stage0+1+2) | **6.89** | **2.82G** | **SN** / TX |
| N (无 FFN) | 6.25 | 3.78G | SC |

**SN 是唯一跨 FFN 稳定的方案**。S012 SOPs 跌破 3G (2.82-2.98G)。

---

## 四、体素化和剪枝方案收录

### 体素化优化

| 方案 | 来源 | 核心机制 | 适配 SDformer |
|------|------|---------|:---:|
| **ECCFlow** (TemporalEventStereo) | ECCV 2024 | 时间差分立体征配，多帧事件流联合 | 中 — 需要立体 setup |
| **V2V** (Video-to-Voxel) | NeurIPS 2025 | 高效视频到体素仿真，自适应时间箱 | 高 — 可替代固定 10-bin |
| **EventFBP** (Functional BP) | ICLR 2026 | 事件处理的函数式反向传播 (2nd order) | 中 — 改体素梯度 |
| **EventDance** | CVPR 2024 | 事件表征学习，姿态估计 | 低 — 任务不匹配 |
| **OmniEvent** (Unified Repr) | AAAI 2026 | 统一事件表征框架，多模态融合 | 中 — 可作为预处理 |
| **RVT** (Recurrent VT) | CVPR 2023 | 循环视觉 Transformer 做事件 | 低 — 架构级改动 |
| **OpenESS** | CVPR 2024 | 开放事件立体视觉 benchmark | 参考 — benchmark |

### 剪枝方案

| 方案 | 来源 | 核心机制 | 适配方式 |
|------|------|---------|:---:|
| **QSD-Transformer** | ICLR 2025 | Spike Information Distortion 分析 + 量化剪枝 + Fine-Grained Distillation | 后处理 SVD 剪枝 |
| **QP-SNN** | ICLR 2025 | 奇异值阈值结构化剪枝 + 权重量化 | 全量后通道剪枝 |
| **SparseSpikFormer** | — | 基于脉冲活动的稀疏注意力剪枝 | 需搜索开源实现 |
| **EDCFlow** | CVPR 2025 | 时域稠密差分图 + 多尺度特征差 + GRU 迭代，DSEC SOTA | 中 — 体素差分可做预处理 |

## 五、当前最佳方案排名

### 注意力方案 (跨 FFN 平均)

| rank | attn | avg AAE | avg SOPs | 稳定性 |
|---:|------|-----:|-----:|:---:|
| 1 | **SN** signed_shiftnorm | 7.00 | 3.51G | 🏆 唯二双 PASS |
| 2 | SC signed_consensus | 6.58 | 3.55G | S012 崩 |
| 3 | TX ternary_axnor | 7.03 | 3.42G | S02 崩 |
| 4 | SL signed_popcount_l1 | 7.18 | 3.57G | S012 崩 |
| 5 | HT hamming_ternary | 7.25 | 3.45G | S02 AAE 崩 |
| 6 | CP compat_qk (H9a) | 7.82 | 3.23G | AAE 全线差 |

### FFN 方案 (跨注意力平均)

| rank | FFN | avg SOPs | avg AAE | 说明 |
|---:|:---:|-----:|-----:|------|
| 1 | **S012** | 2.97G | 7.18 | SOPs 最低 |
| 2 | S02 | 3.24G | 7.39 | 稳定 |
| 3 | S0 | 4.50G | 6.69 | max_thre=0.5 污染 |
| 4 | N | 3.88G | 6.88 | 无 FFN 替换 |

## 六、Phase 3 Preset Sweep (SN/SC/TX + C/A + S02/S012)

12 configs: 3 attn × 2 FFN × 2 preset, 80-step valid5 + 后续 valid40。

### 80-step valid5

| attn | FFN | preset | AAE | AEE | SOPs | gate |
|------|:---:|:---:|-----:|-----:|-----:|:---:|
| **TX** | **S012** | **A** | **6.95** | 1.01 | **2.93G** | ✅ |
| TX | S012 | C | 6.88 | 1.01 | 2.97G | ✅ |
| SC | S012 | C | 7.02 | 1.00 | 3.01G | ✅ |
| TX | S02 | A | 7.18 | 0.95 | 3.24G | ✅ |
| TX | S02 | C | 7.05 | 1.04 | 3.23G | ✅ |
| SN | S02 | C | 7.37 | 1.00 | 3.23G | ✅ |
| SC | S012 | A | 7.02 | 1.08 | 3.09G | ❌ pos_neg 炸 |

### 80-step valid40 (top 5)

| attn | FFN | preset | AEE | AAE | SOPs |
|------|:---:|:---:|-----:|-----:|-----:|
| **TX** | **S02** | **A** | **1.78** | **8.65** | 3.18G |
| SC | S012 | C | 1.82 | 8.75 | 2.90G |
| TX | S012 | A | 1.87 | 8.71 | 2.90G |
| TX | S012 | C | 1.86 | 8.82 | 2.89G |
| SN | S02 | C | 1.82 | 9.06 | 3.18G |

### 360-step confirm valid10 (H40 confirm batch)

| attn | FFN | AEE | AAE | SOPs | gate |
|------|:---:|-----:|-----:|-----:|:---:|
| SN | S02 | 1.10 | 6.26 | 3.26G | ✅ |
| SC | S02 | 1.09 | 6.19 | 3.31G | ✅ |
| SN | S012 | 1.18 | 7.09 | 3.06G | ✅ |
| SC | S012 | 1.19 | 6.84 | 3.02G | ✅ |

### 360-step confirm valid40

| attn | FFN | AEE | AAE | SOPs |
|------|:---:|-----:|-----:|-----:|
| SN | S02 | 1.78 | 8.39 | 3.07G |
| SC | S02 | 1.77 | 8.64 | 3.13G |
| SN | S012 | 1.87 | 8.84 | 2.89G |
| SC | S012 | 1.88 | 9.10 | 2.84G |

## 七、Angular Loss Sweep

| attn | λ_ang | AAE | SOPs | ternary |
|------|:---:|-----:|-----:|:---:|
| SN (基线) | 0 | 7.06 | 3.23G | 健康 |
| SN | 0.5 | 7.29 | 3.23G | 健康 |
| TX | 0.5 | 7.40 | 3.27G | 健康 |
| SN | 0.2 | 7.53 | 3.30G | ❌ 崩 |
| TX | 0.2 | 7.08 | 3.28G | ❌ 崩 |

结论: angular loss 未改善 AAE。λ=0.5 保护三元健康但不提精度。

## 八、短测完成度 & 全量候选

### 已完成的排列组合

| 维度 | 状态 |
|------|:---:|
| 14 注意力 mode | ✅ Phase 1 |
| 4 FFN 覆盖 (S0/S02/S012/N) | ✅ Phase 1+2 |
| 2 preset (C/A) | ✅ Phase 3 |
| Angular loss sweep | ✅ Phase 3 |
| 360-step confirm | ✅ |
| LR strategies (W/L) | ❌ 留全量 |
| Regularization sweep | ❌ 留全量 |

## 十、全量候选四实验详细方案

### 注意力机制

| | SN | TX | SC | BQ |
|---|---|---|---|---|
| 全称 | signed_consensus_shiftnorm | ternary_alpha_xnor_shiftmax | signed_consensus_shiftmax | strict_bsa_qkv_shiftmax |
| 核心操作 | Q/K 符号 popcount + ShiftNorm | 三值 α-XNOR (+1/+α/-β) + Shiftmax | Q/K 符号 popcount + Shiftmax | Q@K^T/√d + Shiftmax + 独立V |
| 归一化 | x / 2^ceil(log2(Σ)) (纯shift) | 2^x / 2^ceil(log2(Σ)) (LUT) | 2^x / 2^ceil(log2(Σ)) (LUT) | 2^x / 2^ceil(log2(Σ)) (LUT) |
| 硬件 | 零 LUT，最优 | LUT×1 | LUT×1 | LUT×1 + 矩阵乘 |
| 范式来源 | 自研 | CVPR 2025 α-XNOR 三元扩展 | 自研 | BSA NeurIPS 2025 严格复现 |
| 论文角色 | 自研硬件最简 | 自研三元拓展 | 自研对照 | 正统 BSA 基线 |

### 参数配置

| | SN S02 C | TX S02 A | SC S012 C | BQ S02 |
|---|---|---|---|---|---|
| Q/K max_threshold | 2.0 | 2.5 | 2.0 | 1.8 |
| Q/K target_rate | 0.05 | 0.03 | 0.05 | 0.05 |
| Q/K activity_eta | 2.0 | 3.0 | 2.0 | 1.0 |
| Q/K threshold_mode | symmetric_target_rate | symmetric_target_rate | symmetric_target_rate | symmetric_target_rate |
| FFN 范围 | stage0+2 | stage0+2 | stage0+1+2 | stage0+2 |
| FFN threshold_mode | official_atlif | official_atlif | official_atlif | official_atlif |
| FFN target_rate | 无 | **0.10** | 无 | 无 |
| FFN activity_eta | 2.0 | 2.0 | 2.0 | 2.0 |
| LR 策略 | differential | flat 1e-5 | flat 1e-5 | flat 1e-5 |
| Angular loss | 0 | 0 | 0 | 0 |
| 三元负阈值 | neg=thre (S1) | neg=thre (S1) | neg=thre (S1) | neg=thre (S1) |

### 短测指标

| | 80-step valid5 | | | 360-step valid10 | | | 80-step valid40 | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | AEE | AAE | SOPs | AEE | AAE | SOPs | AEE | AAE | SOPs |
| SN S02 C | 0.96 | 7.06 | 3.23G | 1.10 | 6.26 | 3.26G | 1.82 | 9.06 | 3.18G |
| TX S02 A | 0.95 | 7.18 | 3.24G | — | — | — | **1.78** | **8.65** | 3.18G |
| SC S012 C | 1.00 | 7.02 | 3.01G | 1.19 | 6.84 | 3.02G | 1.82 | 8.75 | **2.90G** |
| BQ S02 | — | — | — | 1.06 | 6.44 | 3.62G | 1.54 | 7.58 | 3.50G |

## 十二、三大注意力范式公式

### 共同前向

```
Q = SN_Q(Linear_Q(x))      ∈ {-θ_Q, 0, +θ_Q}^(T×B×H×N×d)     三元脉冲发放
K = SN_K(Linear_K(x) + PE) ∈ {-θ_K, 0, +θ_K}^(B×H×N×d)       带位置编码
T: 时间步=10, B: batch, H: 头数=3~24, N: token数, d: 头维度=32

Q 折叠 T→N: Q = reshape(Q, [B, H, T×N, d])
```

### SC (signed_consensus_shiftmax) — 符号共识 + Shiftmax

```
Q_sign = sign(Q)  ∈ {-1, 0, +1}
K_sign = sign(K)  ∈ {-1, 0, +1}

# 符号共识分数 (token 级 popcount)
S_token[j] = Σ_d (Q_sign[i,d] × K_sign[j,d])      ← 同号=+1, 异号=-1, 沉默/单边发放=0
S_token = S_token / head_dim × score_scale          ← 归一化到 [-1, +1]

# 中心化 + Shiftmax
S̃ = S - mean(S, dim=token)                         ← 去均值
gate = 2^S̃ / 2^ceil(log2(Σ 2^S̃))                     ← Shiftmax, 行和 ∈ (0.5, 1]
gate = gate × N                                      ← 保均值

# 输出 (K 复用为 V, 逐元素门控)
output = K × gate                     ← shape [B, H, N, d], token-wise mul
```

### SN (signed_consensus_shiftnorm) — 符号共识 + ShiftNorm

```
# 前三步同 SC
S_token[j] = Σ_d (Q_sign[i,d] × K_sign[j,d]) / head_dim

# 关键区别: ShiftNorm 替代 Shiftmax
S̃ = S - mean(S, dim=token)
gate = clamp(S̃ + bias, 0)                           ← ReLU 截断负值
gate = gate / 2^ceil(log2(Σ gate))                   ← 分母 2^n 可移位近似
gate = gate × N
output = K × gate
```

### TX (ternary_alpha_xnor_shiftmax) — 三元 α-XNOR 矩阵 + Shiftmax

```
Q_sign = sign(Q)  ∈ {-1, 0, +1}
K_sign = sign(K)  ∈ {-1, 0, +1}

# Token-token 三元 α-XNOR 矩阵 (区别于 SC 的 token 级)
S[i,j] = Σ_d [
    Q_sign⁺[i]·K_sign⁺[j] + Q_sign⁻[i]·K_sign⁻[j]    ← 同号激活: +1
  + α₀ × Q_sign⁰[i]·K_sign⁰[j]                        ← 同时沉默: +0.02
  - β  × (Q_sign⁺[i]·K_sign⁻[j] + Q_sign⁻[i]·K_sign⁺[j])   ← 异号: -0.25
  # 旧 TX/H45 未惩罚 0/非0，H46 会额外加 -γ × 单边发放
]
S = S / head_dim × score_scale                         ← 归一化

# 中心化 + Shiftmax
S̃ = S - mean(S, dim=-1)
gate = shiftmax(S̃)                                     ← 2^x / 2^ceil(log2(Σ2^x))
gate = gate × N                                        ← 保均值

# 输出 (K 复用为 V, 矩阵乘法)
output = gate @ K_threshold              ← shape [B, H, N, d], 矩阵乘
```

### TX SSA QKV (H42d/H44) — TX + 独立 V

```
# 前三步同 TX, 得 gate = shiftmax(S̃) × N

# 关键区别: 独立可训练 V 分支
V = Independent_Linear(x)               ← 独立的线性层, 从 K 权重初始化
V_threshold = ATLIF(V)

output = gate @ V_threshold             ← 矩阵乘, V 替代 K
```

### 三种范式对比

| | SC | SN | TX | TX SSA QKV |
|---|---|---|---|---|
| 评分对象 | token popcount | token popcount | **token-token 矩阵** | token-token 矩阵 |
| 核心操作 | Q_sign × K_sign | Q_sign × K_sign | Q_sign ⊗ K_sign (XNOR) | Q_sign ⊗ K_sign |
| 评分粒度 | O(Nd) | O(Nd) | O(N²d) | O(N²d) |
| α₀/β 加权 | ❌ | ❌ | ✅ (+0.02/-0.25) | ✅ |
| 归一化 | Shiftmax | **ShiftNorm** | Shiftmax | Shiftmax |
| 硬件 | LUT | **纯 shift** | LUT | LUT |
| V 分支 | K 复用 | K 复用 | K 复用 | **独立可训练 V** |

## 十三、H42 系列注意力公式详解

所有 H42 系列的共同基础：TX 三元 α-XNOR 评分矩阵

```
Q_event = STE_sign(Q)  ∈ {-1, 0, +1}^(B×H×N×d)
K_event = STE_sign(K)  ∈ {-1, 0, +1}^(B×H×N×d)

S_ij = Σ_d [ Q⁺ᵢK⁺ⱼ + Q⁻ᵢK⁻ⱼ           ← same non-zero (+1)
            + α₀ × Q⁰ᵢK⁰ⱼ                 ← same zero (+0.02)
            - β × (Q⁺ᵢK⁻ⱼ + Q⁻ᵢK⁺ⱼ) ]    ← opposite (-0.25)

S = S / head_dim × score_scale            ← 归一化
```

### 五种模式对比

| 模式 | V 分支 | 归一化 | 公式 |
|------|:---:|:---:|------|
| **H42b** (ssa_linear) | ❌ K 复用 | 无 (raw+bias) | `attn = (S + bias) @ K_threshold` |
| **H45** (ssa_kreuse_shiftmax) | ❌ K 复用 | Shiftmax | `attn = shiftmax(S) @ K_threshold` |
| **H42c** (ssa_qkv_linear) | ✅ 独立 V | 无 (raw+bias) | `attn = (S + bias) @ V_threshold` |
| **H42d** (ssa_qkv_shiftmax) | ✅ 独立 V | Shiftmax | `attn = shiftmax(S) @ V_threshold` |
| TX S02 C (H18c baseline) | ❌ K 复用 | Shiftmax | 同上，但用非 STE 评分 |

### 关键差异

**Linear (H42b/c)**: `gate = S + bias` — 分数可能无界，依赖 bias 和 score_scale 调参稳定。硬件最简（纯加法+矩阵乘）。

**Shiftmax (H42d/H45)**: `gate = shiftmax(S) = 2^S / 2^ceil(log2(Σ2^S))` — BSA 标准归一化。行和 ∈ (0.5, 1]。硬件需 2^x LUT。

**K 复用 vs 独立 V**: K 复用默认使用 `value_mode: threshold`，即保留 ATLIF 阈值幅度的 K，不新增 V 参数；独立 V (`_independent_value_tokens`) 是可训练的线性层，从 K 初始化后独立学习，H42d/H44 使用 `value_mode: threshold`。如果显式切到 `value_mode: sign`，公式才是 `@ sign(K/V)`。

**STE vs 非 STE**: H42 用 STE 版评分 (`_ternary_alpha_xnor_matrix_scores_ste`) 保留 Q/K 梯度。H18c 用非 STE 版 (`_ternary_alpha_xnor_matrix_scores`)，布尔运算阻断梯度。

### H46 单边发放惩罚

H46 修正一个三值相似度细节：旧 TX/SC/SN 都没有惩罚 `0/+1`、`0/-1`、`+1/0`、`-1/0`。这会让“一边沉默、一边发放”的 Q/K 通道贡献为 0，区分度偏弱。H46 增加：

```
single_active = (Q_active XOR K_active)
S = S - γ × single_active
```

当前备选统一设 `γ=0.15`，小于异号惩罚 `β=0.25`，表示单边发放比正负极性冲突轻一些，但不再完全无代价。默认 `single_active_penalty=0`，因此旧 H41/H42/H45 配置仍可复现。

`single_active_penalty_grad` 用来区分 hard 和 STE 两类实现：

- `hard`：前向分数使用严格布尔 XOR，实验最干净，但单边惩罚项本身不直接给 Q/K 梯度。
- `ste`：前向仍保持同样的 hard XOR 值，反向用 sigmoid active proxy 给正在孤立发放的一侧提供 surrogate 梯度；适合 SC/SN 后续认真调参。

| H46 配置 | 对应原方案 | 只新增的变化 | 配置文件 |
|---|---|---|---|
| H46-TX-g015 | H45 TX K-reuse relaxed | `single_active_penalty=0.15`，三值 alpha-XNOR 矩阵分数惩罚 0/非0 | `configs/generated/h46_tx_kreuse_singlepenalty_g015_full20.yml` |
| H47-TX-QKV-g015 | H44 TX QKV relaxed | `single_active_penalty=0.15`，独立 V + Shiftmax，检验 QKV 是否能弥补 K-reuse 表达力不足 | `configs/generated/h47_tx_qkv_singlepenalty_g015_shiftmax_full20.yml` |
| H46-SC-g015 | H41 SC S012 C ang02 | `single_active_penalty=0.15`，signed consensus + Shiftmax 惩罚 0/非0 | `configs/generated/h46_sc_s012c_singlepenalty_g015_full30.yml` |
| H46-SN-g015 | H41 SN S02 C continue | `single_active_penalty=0.15`，signed consensus + ShiftNorm 惩罚 0/非0 | `configs/generated/h46_sn_s02c_singlepenalty_g015_continue10.yml` |
| H47-SC/SN-ste | H46-SC/SN 的可导变体 | 额外设置 `single_active_penalty_grad: ste`，前向不变但惩罚项有 surrogate 梯度 | 待生成 |

### 状态

| 编号 | 模式 | V | 归一化 | 状态 |
|:---:|------|:---:|:---:|:---:|
| H42a | SN mild theta | — | — | ⏳ 没跑 |
| H42b | ssa_linear | ❌ | raw+bias | ⏳ 没跑 |
| H42c | ssa_qkv_linear | ✅ | raw+bias | ⏳ 没跑 |
| H42d | ssa_qkv_shiftmax | ✅ | Shiftmax | ✅ 配置已修复为 full30，尚未单独跑 |
| H44 | ssa_qkv_shiftmax | ✅ | Shiftmax | ⏹ 已按要求停止（独立 V 版本，不继续跑） |
| **H45** | ssa_kreuse_shiftmax | ❌ | Shiftmax | ✅ 已跑完并 profile；best valid40 为 epoch12：AEE 1.7150 / AAE 8.8009 / SOPs 3.1295G |
| H46-TX-g015 | ssa_kreuse_shiftmax + single active penalty | ❌ | Shiftmax | ▶️ 正在全量跑；截至 2026-05-25 00:59（约 UTC+8）已进入 epoch9，已保存 epoch0-8 checkpoints |
| H47-TX-QKV-g015 | ssa_qkv_shiftmax + single active penalty | ✅ | Shiftmax | ✅ 配置已写；尚未启动，目标是测试“独立 V + 单边惩罚”是否优于 H45/H46 |
| H46-SC-g015 | signed_consensus_shiftmax + single active penalty | ❌ | Shiftmax | ✅ 配置已写；对应 H41 SC S012 C ang02，γ=0.15 |
| H46-SN-g015 | signed_consensus_shiftnorm + single active penalty | ❌ | ShiftNorm | ✅ 配置已写；对应 H41 SN S02 C continue，γ=0.15 |

#### H47 启动前修复记录

独立 V 分支需要特别小心：H47 从 baseline checkpoint 续训，而 baseline 没有 `linear_v/bn_v/sn_v` 参数。如果在 checkpoint load 前直接 `copy.deepcopy(linear_k)`，V 会复制随机初始化的 K，而不是复制加载后的 baseline K。已修复：

- `bsa_attention.py` 增加 `sync_independent_value_branch_from_k()`。
- `entrypoints/train.py` 在 `load_state_dict()` 后检测 checkpoint 是否含 V 参数；如果不含 V，则把已经加载好的 K 同步到 V。
- 如果后续从 H47/overlay checkpoint 恢复，checkpoint 内含 V 参数，则不会再同步覆盖已训练的 V。
- H47 `launch.sh` 增加 `flock`，避免多个 watcher 同时触发重复启动。
- H47 watcher 等待 H46 `status.log` 出现 `profile_done` 后才启动，避免 H46 异常退出时误启动。

验证命令：

```
python -m py_compile bsa_attention.py entrypoints/train.py
python -m unittest neuron_experiments.H9_bipolar_self_attention.tests.test_bsa_attention
patched train source compile
```

验证结果：`test_bsa_attention` 共 10 个测试通过，其中包含“先创建 V、再加载 K 后同步 V”的回归测试。

#### H45 已完成结果记录

H45 是 `TX + K-reuse + Shiftmax` 的全量 20 epoch 对照，不含单边发放惩罚；训练从 baseline `checkpoint_epoch59.pth` 续训，配置为 `configs/generated/h45_tx_kreuse_relaxed_full20.yml`。该实验已完成训练与 valid40 profile，结果目录：

`neuron_experiments/H9_bipolar_self_attention/results/h45_tx_kreuse_relaxed_full20_20260524_130941`

valid40 checkpoint 排名如下：

| rank | checkpoint | AEE | AAE | SOPs(G) | firing | score |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `checkpoint_epoch12.pth` | **1.7150** | 8.8009 | 3.1295 | 0.07341 | **1.9507** |
| 2 | `checkpoint_epoch0.pth` | 1.7473 | 8.8525 | 3.1185 | 0.07315 | 1.9804 |
| 3 | `checkpoint_epoch9.pth` | 1.7728 | 9.0944 | 3.1698 | 0.07436 | 2.0300 |
| 4 | `checkpoint_epoch15.pth` | 1.8140 | 8.8244 | 3.1258 | 0.07332 | 2.0490 |
| 5 | `checkpoint_epoch6.pth` | 1.8275 | 9.4792 | 3.0306 | 0.07109 | 2.0645 |
| 6 | `checkpoint_epoch3.pth` | 1.8748 | 9.4866 | **2.9214** | **0.06853** | 2.1119 |
| 7 | `checkpoint_epoch19.pth` | 1.8656 | 9.2826 | 3.1294 | 0.07341 | 2.1133 |
| 8 | `checkpoint_epoch18.pth` | 1.9010 | 9.8748 | 3.0117 | 0.07065 | 2.1479 |

当前判断：H45 的最佳 AEE 在 epoch12，但 AAE 明显弱于 H9a/PSN baseline；SOPs 最低的 epoch3 精度又偏差。因此 H45 更适合作为“去掉独立 V 后表达力不足”的负对照，而不是主推方案。H46 正在验证同样 K-reuse 结构加入 `single_active_penalty=0.15` 是否改善单边发放匹配；H47 将在 H46 profile 完成后自动验证独立 QKV 版本。

#### 从 TX / SC / SN 三条路线后的实验演化路径

这一段只记录 Phase 3 之后的主线变化，目的是把“为什么从 TX/SC/SN 走到 H41/H42/H45/H46/H47”串起来。

| 阶段 | 实验/方案 | 改动逻辑 | 训练状态 | 最好/关键指标 | 结论 |
|---|---|---|---|---|---|
| 选择起点 | TX / SC / SN 三路线短测 | 从 `ternary_alpha_xnor_shiftmax`、`signed_consensus_shiftmax`、`signed_consensus_shiftnorm` 三个注意力里筛选；FFN 主要在 S02/S012 间搜索 | 已完成 80-step valid40 和 360-step confirm | 80-step valid40：TX S02 A 1.78/8.65/3.18G；SC S012 C 1.82/8.75/2.90G；SN S02 C 1.82/9.06/3.18G | 三条都能做稀疏，但短测只能排序，不能直接代表全量收敛 |
| H41-TX | TX S02 C slowbb full30 | TX 三元 alpha-XNOR + stage0/2 FFN 稀疏，保守慢阈值，追求 SOPs 破 3G 且精度不崩 | 已完成 full30 + valid40 profile | epoch27：AEE 1.732 / AAE 8.404 / SOPs 2.615G / firing 0.061；epoch24 SOPs 2.590G | 当前最适合讲“明显稀疏、精度可接受”的主线结果 |
| H41-SC | SC S012 C slowbb ang02 full30 | signed-consensus + Shiftmax，stage0/1/2 FFN 稀疏，加轻量 angular | 已完成 full30 + valid40 profile | epoch27：AEE 1.844 / AAE 8.553 / SOPs 2.749G；epoch12 AAE 8.385 | SOPs 好，但 AEE/AAE 不如 TX 主线；angular 没明显救回来 |
| H42-SC-raw | SC raw S012 C ang02 full30 | 去掉 Shiftmax 的减最大值版本，测试 raw Shiftmax 是否更适合三值 | 已完成 full30 + valid40 profile | epoch0：AEE 1.805 / AAE 8.453 / SOPs 2.875G；epoch12 AAE 8.433 | raw 版没有稳定优于 normal SC，暂不主推 |
| H41/H42-SN | SN S02 C dlr / continue | signed shiftnorm，stage0/2 FFN 稀疏；后续从中间 checkpoint 继续 | 部分 full/continue 已完成到短中程 profile | continue epoch6：AEE 1.743 / AAE 8.379 / SOPs 2.702G | 有单点不错，但训练稳定性和全量完整性弱于 TX；作为对照保留 |
| H45 | TX K-reuse Shiftmax full20 | 把 TX 做成真正 token-token attention：`shiftmax(S) @ K_threshold`，不再挂原 QKFormer carrier，不引入独立 V | 已完成 full20 + valid40 profile | epoch12：AEE 1.715 / AAE 8.801 / SOPs 3.129G；epoch3 SOPs 2.921G 但 AEE/AAE 1.875/9.487 | K-reuse 表达力不足，AAE 明显差；更像负对照 |
| H46-TX | H45 + single-active penalty | 修正 TX 评分里 `0/+1`、`0/-1` 单边发放不扣分的问题，设 `γ=0.15` | 本机正在全量跑；截至 2026-05-25 00:59（约 UTC+8）已到 epoch9，已保存 epoch0-8，未 profile | 暂无最终 AEE/AAE/SOPs；训练 loss 从 109.60 降到约 101.45；epoch8 发放统计约 ternary activity 0.041、pos/neg ratio 1.65 | 等 full/profile 判断；若不优于 H45，K-reuse TX 基本放弃 |
| H47-TX-QKV | H44/H42d 思路 + single-active penalty | 独立 V 分支：`shiftmax(S) @ V_threshold`，且修复了 V 从 loaded K 初始化的问题 | 本机已排队；等待 H46 `profile_done` 后自动启动 | 暂无 | 关键验证：如果 H47 明显优于 H45/H46，说明独立 V 是 TX direct attention 的必要条件 |
| 外部 H42B | H42B QKV P3 precision-first（外部服务器汇报） | `ternary_alpha_xnor_matrix_shiftmax`，Q/K/V 方向 TX 三值 alpha-XNOR matrix shiftmax + S02 FFN，`target_rate=0.08`，慢阈值增长，约 3G SOPs 保精度 | 外部服务器已完成 valid825 profile | epoch29：AEE 3.1461 / AAE 17.9545 / SOPs 3.2637G / firing 0.07656；epoch20：AEE 3.3607 / AAE 19.7203 / SOPs 3.2564G / firing 0.07639 | 精度崩，虽然 SOPs 接近目标但不值得继续押；应作为失败案例记录 |
| 外部 H46SC | H46-SC 从 epoch6 恢复 | H46 的 SC 分支：signed-consensus + single-active penalty；从 `checkpoint_epoch6.pth` 恢复跑剩余 23 epoch | 外部服务器运行中，PID 236458，保存格式 `resume_from_epoch6_local_epoch{}.pth` | 暂无最终 profile | 等恢复训练完成后看能否比 H41-SC/H42-SC-raw 更稳；若 AAE 仍 >8.4，则 SC 不作为主线 |

阶段性判断：

- **主线仍是 H41-TX S02 C**：它是目前唯一稳定做到 SOPs 2.6G 左右且 AEE < 1.75 的完整全量结果。
- **H45 暂时不主推**：K-reuse direct attention 的 AAE 太差，H46 只是验证单边惩罚能否修复它。
- **H47 是关键分叉点**：如果独立 V 能把 AAE 拉回 8.0 左右，同时 SOPs 不超过 3.2G，TX-QKV 才值得继续；否则回到 H41-TX 的 carrier/FFN 路线做精修。
- **H42B 外服结果已经判负**：精度崩到 AEE 3+ / AAE 18 左右，后续不再投入同类 P3 precision-first 配置。
- **SC/SN 更适合作对照或补充**：SC/SN 单点有不错指标，但全量稳定性和论文主线清晰度暂时弱于 TX。

## 十一、全量训练完整结果

### TX S02 C slowbb (三元 α-XNOR + stage0+2 + 保守)

| epoch | AEE | AAE | SOPs | firing |
|:---:|-----:|-----:|-----:|-----:|
| 0 | 1.758 | 8.454 | 2.945G | 0.069 |
| 6 | 1.945 | 9.557 | 2.708G | 0.064 |
| 12 | 1.873 | 9.377 | 2.706G | 0.063 |
| 18 | 1.748 | 8.492 | 2.676G | 0.063 |
| 24 | 1.754 | 8.545 | **2.590G** | 0.061 |
| **27** | **1.732** | **8.404** | 2.615G | 0.061 |
| 29 | 1.741 | 8.828 | 2.643G | 0.062 |

### SC S012 C slowbb ang02 (signed consensus + stage0+1+2 + angular 0.02)

| epoch | AEE | AAE | SOPs | firing |
|:---:|-----:|-----:|-----:|-----:|
| 0 | 1.808 | 8.355 | 2.845G | 0.067 |
| 3 | 1.856 | 8.773 | 2.877G | 0.067 |
| 6 | 1.918 | 8.728 | 2.798G | 0.066 |
| 9 | 1.896 | 8.690 | 2.918G | 0.068 |
| 12 | 1.889 | **8.385** | 2.878G | 0.068 |
| 15 | 2.057 | 9.386 | 2.838G | 0.067 |
| 18 | 1.962 | 9.008 | 2.843G | 0.067 |
| 21 | 1.943 | 8.699 | 2.910G | 0.068 |
| 24 | 1.918 | 8.694 | 2.760G | 0.065 |
| 27 | **1.844** | 8.553 | 2.749G | 0.064 |
| 29 | 1.945 | 8.945 | 2.780G | 0.065 |

### SC raw S012 C slowbb ang02 (不减 max 的 Shiftmax raw + stage0+1+2 + angular 0.02)

| epoch | AEE | AAE | SOPs | firing |
|:---:|-----:|-----:|-----:|-----:|
| **0** | **1.805** | 8.453 | 2.875G | 0.067 |
| 3 | 1.899 | 8.962 | 2.820G | 0.066 |
| 6 | 1.899 | 8.598 | 2.746G | 0.064 |
| 9 | 1.927 | 8.484 | 2.912G | 0.068 |
| 12 | 1.850 | **8.433** | 2.827G | 0.066 |
| 15 | 1.983 | 8.980 | 2.934G | 0.069 |
| 18 | 1.944 | 8.684 | 2.804G | 0.066 |
| 21 | 1.868 | 8.510 | 2.857G | 0.067 |
| 24 | 1.960 | 8.941 | **2.737G** | 0.064 |
| 27 | 1.888 | 8.824 | 2.826G | 0.066 |
| 29 | 1.921 | 8.718 | 2.806G | 0.066 |

### SN S02 C dlr / continue (signed shiftnorm + stage0+2 + 差分LR)

| 来源 | epoch | AEE | AAE | SOPs | firing |
|------|:---:|-----:|-----:|-----:|-----:|
| dlr 初始 | 0 | 1.789 | 8.728 | 3.122G | 0.073 |
| dlr 中途 | 9 | 1.947 | 9.977 | 2.881G | 0.068 |
| continue | 0 | 1.795 | 9.102 | 2.772G | 0.065 |
| continue | 3 | 1.860 | 8.997 | 2.742G | 0.064 |
| **continue** | **6** | **1.743** | **8.379** | 2.702G | 0.063 |
| continue | 9 | 1.815 | 9.209 | 2.800G | 0.066 |

### 全量汇总对比

| | 实验 | epoch | AEE | AAE | SOPs | firing |
|---|---|---|---|---|---|---|
| 🥇 | **TX S02 C slowbb** | **27** | **1.732** | **8.404** | 2.615G | 0.061 |
| 🥈 | TX S02 C slowbb | 24 | 1.754 | 8.545 | **2.590G** | 0.061 |
| 🥉 | SN S02 C continue | 6 | 1.743 | 8.379 | 2.702G | 0.063 |
| 4 | SC S012 C ang02 | 27 | 1.844 | 8.553 | 2.749G | 0.064 |
| 5 | SC raw S012 C ang02 | 0 | 1.805 | 8.453 | 2.875G | 0.067 |
| — | PSN baseline | — | 1.585 | 7.501 | 3.622G | 0.085 |
| — | H9a (历史最优) | — | 1.504 | 7.637 | 3.085G | 0.072 |

### 核心结论

✅ **TX S02 C 全量 SOPs 2.59-2.62G（-28% vs baseline），首次稳定破 3G**
⚠️ SC S012 C ang02 的 angular loss 未修复精度 (ep12 AAE=8.39 最优但仍差), SOPs 2.7-2.9G
⚠️ SC raw 不减 max 的 Shiftmax 没有改善 normal SC，AEE/AAE/SOPs 都未占优
❌ SN S02 C dlr 退化严重, continue 版本稍好但只到 ep9
📄 **论文主图: TX S02 C slowbb epoch27** — 唯一同时满足 SOPs<3G + AEE<1.75 的方案

### 第一档：精度保底 (AEE < 1.56, AAE < 7.70)

| 实验 | 注意力 | FFN | AEE | AAE | SOPs | 说明 |
|------|------|------|-----:|-----:|-----:|------|
| SC broad | SC signed_consensus | stage0+1+2+3 | **1.50** | **7.37** | 3.59G | 历史最优精度 |
| BQ S02 | BQ strict_bsa_qkv | stage0+2 | 1.54 | 7.58 | 3.50G | 正统 BSA |
| TX broad | TX ternary_axnor | broad FFN | 1.54 | 7.50 | 3.54G | 精度/SOPs 平衡 |

### 第二档：SOPs 潜力 (SOPs < 3.2G)

| 实验 | 注意力 | FFN | preset | AEE | AAE | SOPs | 亮点 |
|------|------|:---:|:---:|-----:|-----:|-----:|------|
| **TX S02 A** | TX | S02 | A | **1.78** | **8.65** | 3.18G | 综合最优 |
| SC S012 C | SC | S012 | C | 1.82 | 8.75 | **2.90G** | SOPs 最接近 3G |
| TX S012 A | TX | S012 | A | 1.87 | 8.71 | **2.90G** | SOPs 最低 |
| TX S012 C | TX | S012 | C | 1.86 | 8.82 | 2.89G | SOPs 极低对照 |
| SN S02 C | SN | S02 | C | 1.82 | 9.06 | 3.18G | shiftnorm 对照 |

### 全量三选一

| 优先级 | 实验 | 理由 |
|:---:|------|------|
| 1 | **TX S02 A** | 低 SOPs 线里 AEE+AAE 综合最优 |
| 2 | SC S012 C | SOPs 2.90G 最接近 3G 目标 |
| 3 | BQ S02 | 正统 BSA 精度最好，保底主线 |
| 2 | TX | S012 | A | 2.90G | 8.71 | SOPs 最低 |
| 3 | SC | S012 | C | 2.90G | 8.75 | AEE 最低 |
| 4 | SN | S02 | C | 3.18G | 9.06 | 跨 FFN 最稳定 |

## 九、所有 PASS 方案汇总（去重，按注意力分组）

共 23 个 unique PASS。以下为每种注意力的最优配置：

### SN (signed_consensus_shiftnorm) — 7 PASS

| FFN | preset | ang | AAE | AEE | SOPs | steps |
|:---:|:---:|:---:|-----:|-----:|-----:|:---:|
| S02 | C | 0 | **7.06** | 0.96 | 3.23G | 80 |
| S012 | C | 0 | 6.89 | 1.04 | 2.98G | 80 |
| S02 | A | 0 | 7.33 | 1.07 | 3.24G | 80 |
| S02 | C | 0.5 | 7.29 | 0.95 | 3.23G | 80 |
| S012 | C | 0 | 7.09 | 1.18 | 3.06G | **360** |

### SC (signed_consensus_shiftmax) — 3 PASS

| FFN | preset | ang | AAE | AEE | SOPs | steps |
|:---:|:---:|:---:|-----:|-----:|-----:|:---:|
| S02 | C | 0 | 7.08 | 1.00 | 3.29G | 80 |
| S012 | C | 0 | **6.85** | 0.98 | 2.98G | 80 |
| S02 | C | 0 | 6.19 | 1.09 | 3.31G | **360** |

### TX (ternary_alpha_xnor_shiftmax) — 6 PASS

| FFN | preset | ang | AAE | AEE | SOPs | steps |
|:---:|:---:|:---:|-----:|-----:|-----:|:---:|
| S012 | C | 0 | 6.88 | 1.01 | 2.97G | 80 |
| S012 | A | 0 | 6.95 | 1.01 | **2.93G** | 80 |
| S02 | C | 0 | 7.05 | 1.04 | 3.23G | 80 |
| S02 | A | 0 | **7.18** | **0.95** | 3.24G | 80 |
| S02 | A | 0.5 | 7.40 | 1.00 | 3.27G | 80 |

### HT (hamming_ternary_active) — 2 PASS

| FFN | preset | ang | AAE | AEE | SOPs | steps |
|:---:|:---:|:---:|-----:|-----:|-----:|:---:|
| S012 | C | 0 | 7.30 | 1.11 | 2.96G | 80 |
| S02 | C | 0.2 | 7.56 | 1.04 | 3.26G | 80 |

### 全量候选 TOP 3

| rank | 方案 | AAE | AEE | SOPs | 说明 |
|---:|------|-----:|-----:|-----:|------|
| 1 | **TX S02 A** | 8.65 | 1.78 | 3.18G | valid40 最优，A preset 稳定 |
| 2 | SC S012 C | 8.75 | 1.82 | 2.90G | SOPs 极低，360 步 AAEX=6.85 |
| 3 | TX S012 A | 8.71 | 1.87 | 2.90G | SOPs 最低，valid5 优秀 |

### 建议

短测全部完成。全量优先 **SC S012 C**（360 步最稳 + SOPs 2.90G + AEE 1.82 最优）或 **TX S02 A**（valid40 AAE 最优）。

### 执行计划

```
当前: Angular sweep 跑完 → 定全量方案 (SN+S02 或 SN+S012)
下一条线: V2V 自适应时间箱 + QSD 后处理剪枝 → 短测验证
优先级: V2V 体素化 > QSD 剪枝 > QP-SNN > EventFBP
```


## 七、H40 redesign autopilot 接管记录（20260522_020554）

- 自动生成补测配置：`generated/h40_p4_SNS02_ang05_dlr.yml`, `generated/h40_p4_SNS02_ang05_warm.yml`, `generated/h40_p4_SNS02_ang05_slowbb.yml`, `generated/h40_p4_SNS02_ang05_warm_slowbb.yml`, `generated/h40_p4_TXS02_ang05_dlr.yml`, `generated/h40_p4_TXS02_ang05_warm.yml`, `generated/h40_p4_TXS02_ang05_slowbb.yml`, `generated/h40_p4_TXS02_ang05_warm_slowbb.yml`, `generated/h40_p4_HTS02_ang02_dlr.yml`, `generated/h40_p4_HTS02_ang02_warm.yml`, `generated/h40_p4_HTS02_ang02_slowbb.yml`, `generated/h40_p4_HTS02_ang02_warm_slowbb.yml`, `generated/h40_p4_SCS02_ang05_dlr.yml`, `generated/h40_p4_SCS02_ang05_warm.yml`, `generated/h40_p4_SCS02_ang05_slowbb.yml`, `generated/h40_p4_SCS02_ang05_warm_slowbb.yml`, `generated/h40_p4_SCS012_ang05_dlr.yml`, `generated/h40_p4_SCS012_ang05_warm.yml`, `generated/h40_p4_SCS012_ang05_slowbb.yml`, `generated/h40_p4_SCS012_ang05_warm_slowbb.yml`, `generated/h40_p4_SNS012_ang05_dlr.yml`, `generated/h40_p4_SNS012_ang05_warm.yml`, `generated/h40_p4_SNS012_ang05_slowbb.yml`, `generated/h40_p4_SNS012_ang05_warm_slowbb.yml`, `generated/h40_p4_TXS012_ang05_dlr.yml`, `generated/h40_p4_TXS012_ang05_warm.yml`, `generated/h40_p4_TXS012_ang05_slowbb.yml`, `generated/h40_p4_TXS012_ang05_warm_slowbb.yml`, `generated/h40_p4_HTS012_ang05_dlr.yml`, `generated/h40_p4_HTS012_ang05_warm.yml`, `generated/h40_p4_HTS012_ang05_slowbb.yml`, `generated/h40_p4_HTS012_ang05_warm_slowbb.yml`, `generated/h40_p4_HTS02_ang05_dlr.yml`, `generated/h40_p4_HTS02_ang05_warm.yml`, `generated/h40_p4_HTS02_ang05_slowbb.yml`, `generated/h40_p4_HTS02_ang05_warm_slowbb.yml`, `generated/h40_p4_SLS02_ang05_dlr.yml`, `generated/h40_p4_SLS02_ang05_warm.yml`, `generated/h40_p4_SLS02_ang05_slowbb.yml`, `generated/h40_p4_SLS02_ang05_warm_slowbb.yml`
- 执行策略：160-step valid5 早筛 -> 360-step valid40 确认 -> valid40 pass 后串行 full。
- 并行策略：早筛使用 `parallel=2, bs4, workers4`。实测两个 bs4 并发显存约 42GB，稳定；confirm/full 仍使用 bs8 串行。


### H40 P4 早筛完成

- screen summaries：`neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_00_h40_p4_SNS02_ang05_dlr_20260522_020555/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_01_h40_p4_SNS02_ang05_warm_20260522_020555/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_02_h40_p4_SNS02_ang05_slowbb_20260522_020956/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_03_h40_p4_SNS02_ang05_warm_slowbb_20260522_020956/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_04_h40_p4_TXS02_ang05_dlr_20260522_021356/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_05_h40_p4_TXS02_ang05_warm_20260522_021356/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_06_h40_p4_TXS02_ang05_slowbb_20260522_021756/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_07_h40_p4_TXS02_ang05_warm_slowbb_20260522_021756/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_08_h40_p4_HTS02_ang02_dlr_20260522_022147/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_09_h40_p4_HTS02_ang02_warm_20260522_022147/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_10_h40_p4_HTS02_ang02_slowbb_20260522_022547/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_11_h40_p4_HTS02_ang02_warm_slowbb_20260522_022547/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_12_h40_p4_SCS02_ang05_dlr_20260522_022947/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_13_h40_p4_SCS02_ang05_warm_20260522_022947/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_14_h40_p4_SCS02_ang05_slowbb_20260522_023348/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_15_h40_p4_SCS02_ang05_warm_slowbb_20260522_023348/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_16_h40_p4_SCS012_ang05_dlr_20260522_023748/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_17_h40_p4_SCS012_ang05_warm_20260522_023748/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_18_h40_p4_SCS012_ang05_slowbb_20260522_024159/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_19_h40_p4_SCS012_ang05_warm_slowbb_20260522_024159/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_20_h40_p4_SNS012_ang05_dlr_20260522_024609/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_21_h40_p4_SNS012_ang05_warm_20260522_024609/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_22_h40_p4_SNS012_ang05_slowbb_20260522_025020/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_23_h40_p4_SNS012_ang05_warm_slowbb_20260522_025020/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_24_h40_p4_TXS012_ang05_dlr_20260522_025430/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_25_h40_p4_TXS012_ang05_warm_20260522_025430/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_26_h40_p4_TXS012_ang05_slowbb_20260522_025840/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_27_h40_p4_TXS012_ang05_warm_slowbb_20260522_025840/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_28_h40_p4_HTS012_ang05_dlr_20260522_030251/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_29_h40_p4_HTS012_ang05_warm_20260522_030251/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_30_h40_p4_HTS012_ang05_slowbb_20260522_030701/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_31_h40_p4_HTS012_ang05_warm_slowbb_20260522_030701/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_32_h40_p4_HTS02_ang05_dlr_20260522_031111/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_33_h40_p4_HTS02_ang05_warm_20260522_031111/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_34_h40_p4_HTS02_ang05_slowbb_20260522_031452/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_35_h40_p4_HTS02_ang05_warm_slowbb_20260522_031452/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_36_h40_p4_SLS02_ang05_dlr_20260522_031852/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_37_h40_p4_SLS02_ang05_warm_20260522_031852/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_38_h40_p4_SLS02_ang05_slowbb_20260522_032252/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_39_h40_p4_SLS02_ang05_warm_slowbb_20260522_032252/summary.csv`
- 进入 confirm 的配置：`generated/h40_p4_SNS02_ang05_dlr_steps160.yml`, `generated/h40_p4_SNS02_ang05_warm_steps160.yml`, `generated/h40_p4_SNS02_ang05_slowbb_steps160.yml`, `generated/h40_p4_TXS02_ang05_warm_steps160.yml`, `generated/h40_p4_HTS02_ang02_dlr_steps160.yml`, `generated/h40_p4_HTS02_ang02_slowbb_steps160.yml`, `generated/h40_p4_HTS02_ang02_warm_slowbb_steps160.yml`, `generated/h40_p4_SCS02_ang05_dlr_steps160.yml`, `generated/h40_p4_SCS02_ang05_warm_steps160.yml`, `generated/h40_p4_SCS02_ang05_slowbb_steps160.yml`, `generated/h40_p4_SCS012_ang05_dlr_steps160.yml`, `generated/h40_p4_SCS012_ang05_warm_steps160.yml`, `generated/h40_p4_SCS012_ang05_slowbb_steps160.yml`, `generated/h40_p4_SCS012_ang05_warm_slowbb_steps160.yml`, `generated/h40_p4_SNS012_ang05_dlr_steps160.yml`, `generated/h40_p4_SNS012_ang05_warm_steps160.yml`, `generated/h40_p4_SNS012_ang05_slowbb_steps160.yml`, `generated/h40_p4_SNS012_ang05_warm_slowbb_steps160.yml`, `generated/h40_p4_TXS012_ang05_warm_steps160.yml`, `generated/h40_p4_TXS012_ang05_slowbb_steps160.yml`, `generated/h40_p4_TXS012_ang05_warm_slowbb_steps160.yml`, `generated/h40_p4_HTS012_ang05_dlr_steps160.yml`, `generated/h40_p4_HTS012_ang05_warm_steps160.yml`, `generated/h40_p4_HTS012_ang05_slowbb_steps160.yml`, `generated/h40_p4_HTS012_ang05_warm_slowbb_steps160.yml`, `generated/h40_p4_HTS02_ang05_dlr_steps160.yml`, `generated/h40_p4_HTS02_ang05_warm_steps160.yml`, `generated/h40_p4_HTS02_ang05_slowbb_steps160.yml`, `generated/h40_p4_SLS02_ang05_warm_steps160.yml`, `generated/h40_p4_SLS02_ang05_slowbb_steps160.yml`


## 七、H40 redesign autopilot 续跑记录（20260522_112403）

- 复用早筛 summary：`neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_00_h40_p4_SNS02_ang05_dlr_20260522_020555/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_01_h40_p4_SNS02_ang05_warm_20260522_020555/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_02_h40_p4_SNS02_ang05_slowbb_20260522_020956/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_03_h40_p4_SNS02_ang05_warm_slowbb_20260522_020956/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_04_h40_p4_TXS02_ang05_dlr_20260522_021356/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_05_h40_p4_TXS02_ang05_warm_20260522_021356/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_06_h40_p4_TXS02_ang05_slowbb_20260522_021756/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_07_h40_p4_TXS02_ang05_warm_slowbb_20260522_021756/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_08_h40_p4_HTS02_ang02_dlr_20260522_022147/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_09_h40_p4_HTS02_ang02_warm_20260522_022147/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_10_h40_p4_HTS02_ang02_slowbb_20260522_022547/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_11_h40_p4_HTS02_ang02_warm_slowbb_20260522_022547/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_12_h40_p4_SCS02_ang05_dlr_20260522_022947/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_13_h40_p4_SCS02_ang05_warm_20260522_022947/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_14_h40_p4_SCS02_ang05_slowbb_20260522_023348/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_15_h40_p4_SCS02_ang05_warm_slowbb_20260522_023348/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_16_h40_p4_SCS012_ang05_dlr_20260522_023748/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_17_h40_p4_SCS012_ang05_warm_20260522_023748/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_18_h40_p4_SCS012_ang05_slowbb_20260522_024159/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_19_h40_p4_SCS012_ang05_warm_slowbb_20260522_024159/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_20_h40_p4_SNS012_ang05_dlr_20260522_024609/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_21_h40_p4_SNS012_ang05_warm_20260522_024609/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_22_h40_p4_SNS012_ang05_slowbb_20260522_025020/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_23_h40_p4_SNS012_ang05_warm_slowbb_20260522_025020/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_24_h40_p4_TXS012_ang05_dlr_20260522_025430/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_25_h40_p4_TXS012_ang05_warm_20260522_025430/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_26_h40_p4_TXS012_ang05_slowbb_20260522_025840/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_27_h40_p4_TXS012_ang05_warm_slowbb_20260522_025840/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_28_h40_p4_HTS012_ang05_dlr_20260522_030251/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_29_h40_p4_HTS012_ang05_warm_20260522_030251/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_30_h40_p4_HTS012_ang05_slowbb_20260522_030701/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_31_h40_p4_HTS012_ang05_warm_slowbb_20260522_030701/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_32_h40_p4_HTS02_ang05_dlr_20260522_031111/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_33_h40_p4_HTS02_ang05_warm_20260522_031111/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_34_h40_p4_HTS02_ang05_slowbb_20260522_031452/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_35_h40_p4_HTS02_ang05_warm_slowbb_20260522_031452/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_36_h40_p4_SLS02_ang05_dlr_20260522_031852/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_37_h40_p4_SLS02_ang05_warm_20260522_031852/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_38_h40_p4_SLS02_ang05_slowbb_20260522_032252/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_39_h40_p4_SLS02_ang05_warm_slowbb_20260522_032252/summary.csv`
- 从 confirm 阶段继续：360-step valid40 -> valid40 pass 后串行 full。


### H40 P4 早筛完成

- screen summaries：`neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_00_h40_p4_SNS02_ang05_dlr_20260522_020555/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_01_h40_p4_SNS02_ang05_warm_20260522_020555/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_02_h40_p4_SNS02_ang05_slowbb_20260522_020956/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_03_h40_p4_SNS02_ang05_warm_slowbb_20260522_020956/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_04_h40_p4_TXS02_ang05_dlr_20260522_021356/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_05_h40_p4_TXS02_ang05_warm_20260522_021356/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_06_h40_p4_TXS02_ang05_slowbb_20260522_021756/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_07_h40_p4_TXS02_ang05_warm_slowbb_20260522_021756/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_08_h40_p4_HTS02_ang02_dlr_20260522_022147/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_09_h40_p4_HTS02_ang02_warm_20260522_022147/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_10_h40_p4_HTS02_ang02_slowbb_20260522_022547/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_11_h40_p4_HTS02_ang02_warm_slowbb_20260522_022547/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_12_h40_p4_SCS02_ang05_dlr_20260522_022947/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_13_h40_p4_SCS02_ang05_warm_20260522_022947/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_14_h40_p4_SCS02_ang05_slowbb_20260522_023348/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_15_h40_p4_SCS02_ang05_warm_slowbb_20260522_023348/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_16_h40_p4_SCS012_ang05_dlr_20260522_023748/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_17_h40_p4_SCS012_ang05_warm_20260522_023748/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_18_h40_p4_SCS012_ang05_slowbb_20260522_024159/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_19_h40_p4_SCS012_ang05_warm_slowbb_20260522_024159/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_20_h40_p4_SNS012_ang05_dlr_20260522_024609/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_21_h40_p4_SNS012_ang05_warm_20260522_024609/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_22_h40_p4_SNS012_ang05_slowbb_20260522_025020/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_23_h40_p4_SNS012_ang05_warm_slowbb_20260522_025020/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_24_h40_p4_TXS012_ang05_dlr_20260522_025430/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_25_h40_p4_TXS012_ang05_warm_20260522_025430/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_26_h40_p4_TXS012_ang05_slowbb_20260522_025840/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_27_h40_p4_TXS012_ang05_warm_slowbb_20260522_025840/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_28_h40_p4_HTS012_ang05_dlr_20260522_030251/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_29_h40_p4_HTS012_ang05_warm_20260522_030251/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_30_h40_p4_HTS012_ang05_slowbb_20260522_030701/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_31_h40_p4_HTS012_ang05_warm_slowbb_20260522_030701/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_32_h40_p4_HTS02_ang05_dlr_20260522_031111/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_33_h40_p4_HTS02_ang05_warm_20260522_031111/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_34_h40_p4_HTS02_ang05_slowbb_20260522_031452/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_35_h40_p4_HTS02_ang05_warm_slowbb_20260522_031452/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_36_h40_p4_SLS02_ang05_dlr_20260522_031852/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_37_h40_p4_SLS02_ang05_warm_20260522_031852/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_38_h40_p4_SLS02_ang05_slowbb_20260522_032252/summary.csv`, `neuron_experiments/H9_bipolar_self_attention/results/h40_p4_ang05_screen160_bs4x2_39_h40_p4_SLS02_ang05_warm_slowbb_20260522_032252/summary.csv`
- 进入 confirm 的配置：`generated/h40_p4_SNS02_ang05_dlr.yml`, `generated/h40_p4_SNS02_ang05_warm.yml`, `generated/h40_p4_SNS02_ang05_slowbb.yml`, `generated/h40_p4_TXS02_ang05_warm.yml`, `generated/h40_p4_HTS02_ang02_dlr.yml`, `generated/h40_p4_HTS02_ang02_slowbb.yml`, `generated/h40_p4_HTS02_ang02_warm_slowbb.yml`, `generated/h40_p4_SCS02_ang05_dlr.yml`, `generated/h40_p4_SCS02_ang05_warm.yml`, `generated/h40_p4_SCS02_ang05_slowbb.yml`, `generated/h40_p4_SCS012_ang05_dlr.yml`, `generated/h40_p4_SCS012_ang05_warm.yml`, `generated/h40_p4_SCS012_ang05_slowbb.yml`, `generated/h40_p4_SCS012_ang05_warm_slowbb.yml`, `generated/h40_p4_SNS012_ang05_dlr.yml`, `generated/h40_p4_SNS012_ang05_warm.yml`, `generated/h40_p4_SNS012_ang05_slowbb.yml`, `generated/h40_p4_SNS012_ang05_warm_slowbb.yml`, `generated/h40_p4_TXS012_ang05_warm.yml`, `generated/h40_p4_TXS012_ang05_slowbb.yml`, `generated/h40_p4_TXS012_ang05_warm_slowbb.yml`, `generated/h40_p4_HTS012_ang05_dlr.yml`, `generated/h40_p4_HTS012_ang05_warm.yml`, `generated/h40_p4_HTS012_ang05_slowbb.yml`, `generated/h40_p4_HTS012_ang05_warm_slowbb.yml`, `generated/h40_p4_HTS02_ang05_dlr.yml`, `generated/h40_p4_HTS02_ang05_warm.yml`, `generated/h40_p4_HTS02_ang05_slowbb.yml`, `generated/h40_p4_SLS02_ang05_warm.yml`, `generated/h40_p4_SLS02_ang05_slowbb.yml`
