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

### STE 和评分函数版本

**STE (Straight-Through Estimator)** 是梯度传递技巧：

```python
# _ternary_sign_ste (bsa_attention.py:126-130)
def _ternary_sign_ste(x):
    hard = x.sign()                        # 前向: {-1, 0, +1}, 不可导
    return (hard - x).detach() + x          # 反向: ∂L/∂x 原样通过, hard部分detach
```

**关键**: 前向输出是硬三元 {-1,0,+1}，但梯度反向时把 sign() 当作恒等函数。这让三元注意力可以和普通可导层一样训练。

**两个评分版本的差异**:

```python
# 非STE (_ternary_alpha_xnor_matrix_scores): H18c / TX S02 C 使用
q_pos = q_event.gt(0).to(dtype)            # .gt() → boolean → 无梯度
q_neg = q_event.lt(0).to(dtype)            # 评分矩阵不可导

# STE (_ternary_alpha_xnor_matrix_scores_ste): H42b/c/d / H44 使用
q_pos = torch.relu(q_event)                 # relu → 可导
q_neg = torch.relu(-q_event)                # 对 Q/K 保留梯度
q_zero = torch.clamp(1 - q_event.abs(), 0, 1)  # STE 的 "零"近似
```

**为什么 H42 要用 STE 版**: H42 是 `gate = scores @ V` 的 direct attention——scores 是唯一的注意力对象，如果它不可导，梯度无法反向传播到 Q/K。H18c/TX S02 C 是 QKFormer gate 风格——评分仅做辅助门控，主梯度路径在原始 Q_sum gate 中。

**single_active_penalty (γ)**: 当 Q 发放但 K 沉默（或相反）时的惩罚项。默认 γ=0（不用）。稀疏脉冲下，大量 token 对出现单边发放——另一方沉默无法判断方向一致性。开启后给这种情况小惩罚。

## 十三、方案迭代演化史 (TX/SC/SN 筛选后)

### Phase 1-2: 14→6→3 注意力筛选

14 注意力 × S0 FFN (max_thre=0.5, SOPs 爆炸) → max_thre=2.0 修复 → 6 注意力 × S02/S012/N → **SN/SC/TX 三强**

### Phase 3: Preset Sweep

TX S02 C 首次稳定, SC S012 C 进入竞争。TX S012 C SOPs=2.97G 最低。

### 全量训练: TX 登顶

| 实验 | 注意力 | FFN | 最优 ep | AEE | AAE | SOPs |
|------|------|:---:|:---:|-----:|-----:|-----:|
| **TX S02 C** | TX | S02 | 27 | **1.73** | 8.40 | **2.62G** |
| SC S012 C ang02 | SC | S012 | 12 | 1.89 | 8.39 | 2.88G |
| SN S02 C cont | SN | S02 | 6 | 1.74 | 8.38 | 2.70G |
| SC raw S012 ang02 | SC raw | S012 | 27 | 1.89 | 8.82 | 2.83G |

**TX S02 C 碾压全场** — 唯一 SOPs<3G + AEE<1.75。SC S012 精度不够，SN 退化。

### H42/H44: 注意力升级 + 精度修复

| | TX S02 C | **H44 (进行中)** |
|---|---|---|
| 注意力 | TX 原始 (K 复用) | **TX SSA QKV (独立 V)** |
| target_rate | 0.05 | **0.06** |
| activity_eta | 2.0 | **1.5** |
| angular | 0 | **0.2** |
| threshold_eta | 0.001 | **0.0005** |

**H44 epoch 0 信号**: threshold_mean=1.05（远低 TX S02 C 同期 ~2.0），三元正负平衡 1.33，binary_activity=0.044（远低 0.136）。放松参数生效。

### 核心演化规律

1. **max_thre 是关键杠杆**: 0.5→SOPs炸, 1.8→AAE炸, 2.0+C→甜点
2. **S02 > S012**: 更广的 FFN 替换反而不如聚焦的 S02
3. **TX 三元 α-XNOR 最强**: token-token 矩阵 > token 级门控
4. **angular loss 不是万能**: 单独加不改善，配合放松参数才可能有效
5. **120/360步短测能排序不能预测绝对值**: H23b→360步 valid40 全部退化

### H42 未跑全量

| 编号 | V | 归一化 | 状态 |
|:---:|:---:|:---:|:---:|
| H42b | ❌ | raw+bias | ⏳ |
| H42b_qkv | ❌ | Shiftmax | ⏳ |
| H42c | ✅ | raw+bias | ⏳ |
| H42d | ✅ | Shiftmax | ⏳(被H44替代) |
| **H44** | ✅ | Shiftmax relaxed | 🔄 |

## 十五、下一步规划

### 当前瓶颈

| 指标 | PSN baseline | TX S02 C (最优) | 差距 |
|---|---|---|---|
| AEE | 1.585 | 1.732 | +9.3% |
| AAE | 7.501 | 8.404 | +12.0% |
| SOPs | 3.622G | 2.615G | -27.8% |

SOPs 已经超额完成任务。核心挑战：**在不显著增加 SOPs 的前提下把 AEE 从 1.73 降到 1.65 以内**。

### 三条路线

#### 路线 A: 注意力架构升级 (H44 验证中)

TX SSA QKV + 独立V + 放松参数。如果 H44 AEE<1.70→新注意力有效→在此基础上叠加路线B/C。

#### 路线 B: 训练策略优化

| 实验 | 内容 | 改动量 | 预期 |
|:---:|------|:---:|------|
| **H45** | TX S02 C + 渐进稀疏 schedule | 改 train.py | AEE↓, SOPs 微增 |
| **H46** | TX S02 C + teacher distillation | 新增 distill loss | AAE↓, SOPs 不变 |

**H45 渐进稀疏**: epoch 0-5 target=0.08 无约束→先学特征; epoch 6-12 target=0.06→温和压; epoch 13-20 target=0.05→目标稀疏。避免早熟稀疏化。

**H46 teacher distillation**: PSN baseline 作为 teacher，对中间特征加 MS E loss (stage2 FFN 输出)。保护深层特征不被 ATLIF 过度扭曲。只加特征蒸馏不加速率蒸馏（速率本来就是目标）。

#### 路线 C: 参数精调 (低风险)

| 实验 | 内容 | 改动量 | 预期 |
|:---:|------|:---:|------|
| **H47** | TX S02 C + angular 0.15 (更温和) | 改 config | AAE↓ |
| **H48** | TX S02 C + target 0.07 (更宽松) | 改 config | AEE↓, SOPs 微增 |

### 必做基线

| 实验 | 内容 | 理由 |
|:---:|------|------|
| **BQ S02** | strict_bsa_qkv 全量 30ep | 论文必须有的 BSA 对照 |

### 执行顺序

```
H44 完成 (今晚)
  ├─ AEE<1.70? → H45 + H46 on H44 base
  ├─ AEE≥1.70? → H45 + H46 on TX S02 C base
  └─ 同时: BQ S02 启动 (无论 H44 结果)
```

**不改动**: S012 FFN (全量证实不如 S02)、SC raw (无效)、angular loss 单独加 (无效)、并行短测 (multiprocessing 在 CUDA 下不稳定)。

## 十六、H42 系列注意力公式详解

所有 H42 系列的共同基础：TX 三元 α-XNOR 评分矩阵

```
Q_event = STE_sign(Q)  ∈ {-1, 0, +1}^(B×H×N×d)
K_event = STE_sign(K)  ∈ {-1, 0, +1}^(B×H×N×d)

S_ij = Σ_d [ Q⁺ᵢK⁺ⱼ + Q⁻ᵢK⁻ⱼ           ← same non-zero (+1)
            + α₀ × Q⁰ᵢK⁰ⱼ                 ← same zero (+0.02)
            - β × (Q⁺ᵢK⁻ⱼ + Q⁻ᵢK⁺ⱼ)      ← opposite (-0.25)
            - γ × [(Q⁺+Q⁻)ᵢK⁰ⱼ + Q⁰ᵢ(K⁺+K⁻)ⱼ] ]  ← single-active penalty (-γ)

S = S / head_dim × score_scale            ← 归一化

# STE (Straight-Through Estimator) 版本 vs 非STE版本
# _ternary_sign_ste: forward→sign(x)∈{-1,0,+1}, backward→∂L/∂x 原样传递
# STE版(_matrix_scores_ste): relu/clamp 保留梯度, H42b/c/d/H44用
# 非STE版(_matrix_scores): .gt()/.eq() 布尔运算阻断梯度, H18c/TX S02 C用
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
| H46-TX-g015 | ssa_kreuse_shiftmax + single active penalty | ❌ | Shiftmax | ✅ 已跑完并 profile；best valid40 为 epoch0：AEE 1.7309 / AAE 8.9586 / SOPs 3.1128G |
| H47-TX-QKV-g015 | ssa_qkv_shiftmax + single active penalty | ✅ | Shiftmax | ✅ 已跑完并 profile；best valid40 为 epoch18：AEE 1.7509 / AAE 8.7260 / SOPs 3.0217G |
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

当前判断：H45 的最佳 AEE 在 epoch12，但 AAE 明显弱于 H9a/PSN baseline；SOPs 最低的 epoch3 精度又偏差。因此 H45 更适合作为“去掉独立 V 后表达力不足”的负对照，而不是主推方案。H46/H47 后续 full-valid 与外部 valid825 结果均未超过 H41/H49 主线，direct TX-QKV、K-reuse single-penalty 和 SN STE 暂不作为主线。

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
| H46-TX | H45 + single-active penalty | 修正 TX 评分里 `0/+1`、`0/-1` 单边发放不扣分的问题，设 `γ=0.15` | 本机已完成 full20 + valid40 profile | best epoch0：AEE 1.7309 / AAE 8.9586 / SOPs 3.1128G；AAE 最低 epoch15：AEE 1.7852 / AAE 8.6964 / SOPs 3.0590G | 单边惩罚没有救回 K-reuse；比 H41-TX 主线差，K-reuse direct attention 基本放弃 |
| H47-TX-QKV | H44/H42d 思路 + single-active penalty | 独立 V 分支：`shiftmax(S) @ V_threshold`，且修复了 V 从 loaded K 初始化的问题 | 本机已完成 full20 + valid40 profile | best epoch18：AEE 1.7509 / AAE 8.7260 / SOPs 3.0217G；AAE 最低 epoch3：AEE 1.7851 / AAE 8.5693 / SOPs 3.0298G | 独立 V 比 K-reuse 略稳，但仍不如 H41-TX；direct TX-QKV 不适合作为当前主线 |
| 外部 H42B | H42B QKV P3 precision-first（外部服务器汇报） | `ternary_alpha_xnor_matrix_shiftmax`，Q/K/V 方向 TX 三值 alpha-XNOR matrix shiftmax + S02 FFN，`target_rate=0.08`，慢阈值增长，约 3G SOPs 保精度 | 外部服务器已完成 valid825 profile | epoch29：AEE 3.1461 / AAE 17.9545 / SOPs 3.2637G / firing 0.07656；epoch20：AEE 3.3607 / AAE 19.7203 / SOPs 3.2564G / firing 0.07639 | 精度崩，虽然 SOPs 接近目标但不值得继续押；应作为失败案例记录 |
| 外部 H46SC | H46-SC 从 epoch6 恢复 | H46 的 SC 分支：signed-consensus + single-active penalty；从 `checkpoint_epoch6.pth` 恢复跑剩余 23 epoch | 外部 A100 已完成 valid825 profile | local epoch22：AEE 1.9117 / AAE 11.4520 / SOPs 3.1778G；local epoch20：AEE 1.9361 / AAE 11.4979 / SOPs 3.1396G | 比外部 H42B/SN STE 好，但仍明显弱于本机 H41 SC/H49；不作为主线 |

阶段性判断：

- **主线仍是 H41-TX S02 C**：它是目前唯一稳定做到 SOPs 2.6G 左右且 AEE < 1.75 的完整全量结果。
- **H45/H46 暂时不主推**：K-reuse direct attention 的 AAE 太差，H46 的单边惩罚没有明显修复。
- **H47 已完成关键验证**：独立 V 能把 SOPs 控到约 3.02G，但 AAE 仍在 8.7 左右，且训练更慢、显存更高；因此 direct TX-QKV 也不作为当前主线。
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


## 十、H41 之后全量实验 full-valid816/valid825 统一推理（20260525）

本机当前 `valid_split_seq.csv` 只有 816 行，sha256 为 `571e6d06073df4a82b7abc10c186f061de0d93dd2beb0b7075053b9295c49f2e`，因此本节最早一批 full profile 实际是 **local full-valid816**。这批结果可以用于本机方案筛选和趋势判断，但不应作为最终论文主表的唯一口径。

最终论文/正式汇报建议统一使用 OF_EV_SNN/标准 DSEC split 对应的 825 行 `valid_split_seq.csv`，即另一台机器报告的 sha256：

```text
7f3dc2800653e12caca10379c51ee8e8988aaf6bb80c391224a454a5879325d0
```

正式 profile 协议应为：`split=valid`, `num_samples=999999`, `drop_last=False`，并确认 `sops_summary.json` 中 `samples=825`。每个结果同时统计 AEE/AAE、PE1/PE2/PE3(outlier)、总 SOPs 与全局发放率。当前本机 816 结果完整汇总文件：`neuron_experiments/H9_bipolar_self_attention/results/full_valid_profiles_20260525/FULL_VALID_RESULTS_ALL.md`。

注意：第一次队列里的 `h42_sc_raw_s012c_epoch0` 是无效项，因为本地不存在 `checkpoint_epoch0.pth`，日志没有 `Model restored`，结果为 0 firing/0 SOPs，已在最终表中排除。`h44_tx_qkv_relaxed_full20_20260524_125901` 没有保存 checkpoint，因此无法做 full-valid 推理。

### 本机 local full-valid816 结果（待 canonical valid825 重跑）

| rank | 实验 | checkpoint | AEE | AAE | PE1 | PE2 | PE3/outlier | SOPs | firing |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | baseline PSN | epoch59 | **1.3307** | **7.8132** | 0.4266 | 0.1526 | 0.0728 | 3.9633G | 0.09297 |
| 2 | H41 SC S012C | epoch27 | 1.6223 | 9.4549 | 0.5462 | 0.2342 | 0.1140 | 3.1278G | 0.07337 |
| 3 | H42 SC raw S012C | epoch27 | 1.6525 | 9.6163 | 0.5490 | 0.2405 | 0.1200 | 3.2123G | 0.07535 |
| 4 | H41 SC S012C | epoch0 | 1.6691 | 9.4516 | 0.5533 | 0.2446 | 0.1236 | 3.2133G | 0.07538 |
| 5 | H42 SC raw S012C | epoch29 | 1.6985 | 9.6130 | 0.5658 | 0.2567 | 0.1292 | 3.1825G | 0.07465 |
| 6 | H41 TX S02C | epoch27 | 1.7083 | 10.0881 | 0.5294 | 0.2230 | 0.1170 | **2.9482G** | **0.06916** |
| 7 | H46 TX K-reuse single-active | epoch0 | 1.8356 | 10.7802 | 0.5429 | 0.2381 | 0.1279 | 3.5217G | 0.08261 |
| 8 | H47 TX QKV single-active | epoch3 | 1.9178 | 10.6902 | 0.5306 | 0.2362 | 0.1298 | 3.4303G | 0.08047 |
| 9 | H41 SN S02C continue | epoch9 | 1.9190 | 10.8451 | 0.5386 | 0.2339 | 0.1247 | 3.1420G | 0.07370 |
| 10 | H46 TX K-reuse single-active | epoch15 | 1.9219 | 10.6675 | 0.5311 | 0.2388 | 0.1305 | 3.4565G | 0.08108 |
| 11 | H47 TX QKV single-active | epoch18 | 1.9221 | 10.7281 | 0.5292 | 0.2381 | 0.1302 | 3.4181G | 0.08018 |
| 12 | H45 TX K-reuse | epoch12 | 1.9346 | 10.9105 | 0.5274 | 0.2327 | 0.1285 | 3.5322G | 0.08286 |
| 13 | H47 TX QKV single-active | epoch19 | 1.9453 | 10.9052 | 0.5389 | 0.2466 | 0.1364 | 3.4426G | 0.08076 |
| 14 | H45 TX K-reuse | epoch19 | 1.9520 | 10.9585 | 0.5372 | 0.2438 | 0.1337 | 3.5337G | 0.08289 |
| 15 | H41 SN S02C continue | epoch6 | 1.9915 | 11.4317 | 0.5387 | 0.2329 | 0.1246 | 3.0068G | 0.07053 |
| 16 | H46 TX K-reuse single-active | epoch19 | 1.9943 | 11.0402 | 0.5405 | 0.2493 | 0.1395 | 3.6448G | 0.08550 |

### 外部 A100 机器 valid825 全量结果

以下结果来自另一台 A100 机器，只收“全量训练 + 已做 valid825”的 checkpoint，不混入短测。注意它的样本数是 valid825，本机主表是 valid816，且可能存在代码版本、配置、数据缓存和随机性差异，因此优先用于判断同一批方案在另一环境下是否稳定，不直接作为最终论文排序。

| 实验 | checkpoint | valid825 AEE | AAE | SOPs | firing | 备注 |
|---|---|---:|---:|---:|---:|---|
| H42B QKV P3 precision | epoch29 | 3.1461 | 17.9545 | 3.2637G | 0.07656 | 全量30跑完，精度很差 |
| H42B QKV P3 precision | epoch20 | 3.3607 | 19.7203 | 3.2564G | 0.07639 | 中途点，也差 |
| H46SC single-penalty g0.15 | local epoch22 | 1.9117 | 11.4520 | 3.1778G | 0.07454 | A100 已完成结果中最好，但明显弱于本机 H41/H49 |
| H46SC single-penalty g0.15 | local epoch20 | 1.9361 | 11.4979 | 3.1396G | 0.07365 | H46SC 中途点 |
| H47 SN STE g0.15 | epoch22 | 2.2373 | 12.5167 | 3.0548G | 0.07166 | 暂停在 epoch22，精度差 |
| H47 SN STE g0.15 | epoch20 | 2.1860 | 12.0454 | 3.0630G | 0.07185 | H47SN 最好点，但仍差 |
| H47 SN STE g0.15 | epoch12 | 2.7613 | 16.7426 | 2.9478G | 0.06915 | 早期点 |
| H48 SN STE g0.05 | epoch29 | 2.2394 | 12.2438 | 3.0395G | 0.07130 | 全量30跑完，SOPs好但精度差 |

外部 A100 结果的阶段判断：

- H42B QKV P3 precision 明确判负：SOPs 接近目标，但 AEE/AAE 崩到 `3.15/17.95`，不再继续押 direct QKV P3。
- H46SC 在 A100 上是这批里相对最好，但 `AEE=1.91, AAE=11.45`，仍明显弱于本机 H41 SC/H49，不足以作为主线。
- H47/H48 的 SN STE 路线 SOPs 可以到 3.0G 左右，但 AEE/AAE 退化太多；它说明“给 SN 单边惩罚加 STE 梯度”不能单独解决精度问题。
- 目前没有看到 **SC STE** 的完整全量 valid825 结果；已完成的是 H46SC single-penalty g0.15，以及 SN STE 的 H47/H48。

### 当前 local full-valid816 结论

- full-valid816 与 valid40 排名差异明显，后续方案筛选应至少看 full valid，不应只看 valid40。
- 由于本机 split 文件目前是 816 行，最终论文主表需要先同步 canonical 825 行 split，再对 baseline 和候选方案重跑 full-valid825。
- baseline PSN full-valid 为 `AEE=1.3307`, `AAE=7.8132`, `SOPs=3.9633G`, `firing=0.09297`，比之前 valid40 基准更强。
- 精度-稀疏折中最好的非 baseline 是 H41 SC S012C epoch27：`AEE=1.6223`, `AAE=9.4549`, `SOPs=3.1278G`，SOPs 相对 baseline 下降约 21.1%。
- SOPs 最低且仍可讲稀疏故事的是 H41 TX S02C epoch27：`AEE=1.7083`, `AAE=10.0881`, `SOPs=2.9482G`，SOPs 相对 baseline 下降约 25.6%，但 AAE 退化较大。
- H45/H46/H47 这条直接 TX attention 路线目前没有超过 H41 TX：K-reuse 和 QKV 版本 full-valid AEE 约 1.83-1.99，SOPs 约 3.42-3.64G，既没有更稀疏，也没有更准。
- H42 SC raw 不减 max 的 raw Shiftmax 没有带来收益：epoch27/29 的精度接近 H41 SC，但 SOPs 不低于 H41 SC。
- A100 外部 valid825 结果整体比本机 local-valid816 主线差，尤其 H42B/H47SN/H48SN；这更支持当前判断：direct QKV 和 SN STE 不宜作为主线，后续应围绕 H41/H49/H50 或另行重测 SC STE。

### 当前主线建议

1. 论文主线如果强调“最稳的精度-节能折中”，优先围绕 H41 SC S012C epoch27 讲，SOPs 从 3.9633G 到 3.1278G，精度损失相对可控。
2. 如果必须讲“3G SOPs 左右”的强稀疏故事，H41 TX S02C epoch27 是当前唯一靠谱候选，但需要继续压 AAE。
3. H45/H46/H47 的直接 TX attention 暂时不作为主线，除非后续重新设计 value/normalization/训练约束，否则现在的 full-valid 结果支持不足。

### TX 路线与 H47 方向判断

H41 TX 与 H47 的关键对比如下：

| 方案 | 注意力形式 | AEE | AAE | SOPs | 判断 |
|---|---|---:|---:|---:|---|
| H41 TX S02C epoch27 | 保留 QKFormer carrier，外接 TX 三值 alpha-XNOR Shiftmax gate | 1.7083 | 10.0881 | **2.9482G** | 当前唯一进入 3G SOPs 左右且没有崩的 TX 候选 |
| H47 TX QKV epoch3 | `Shiftmax(alpha-XNOR(Q,K)) @ V`，独立 V，single-active penalty | 1.9178 | 10.6902 | 3.4303G | 精度和 SOPs 均不如 H41 TX |
| H47 TX QKV epoch18 | 同上 | 1.9221 | 10.7281 | 3.4181G | 训练后没有改善 |
| H47 TX QKV epoch19 | 同上 | 1.9453 | 10.9052 | 3.4426G | 后期继续退化 |

结论：H47 的“直接 QKV 化三值注意力”理论上更优雅，但当前实现和训练范式没有带来收益。它引入了独立 V 分支，参数和训练路径更复杂，full-valid 下既没有降低 SOPs，也没有改善 AAE/AEE。因此 H47 目前只适合作为消融：说明直接把 SDFormerFlow/QKFormer 注意力替换成 QKV 三值矩阵注意力并不稳定，不能作为当前主线。

H41 TX 目前的论文叙事应改为：在不破坏 QKFormer 稳定 carrier 的前提下，引入三值 alpha-XNOR 兼容门控和 ATLIF 自适应稀疏，使 SOPs 从 3.9633G 降到 2.9482G，下降约 25.6%。它的问题是 AAE 从 7.8132 增至 10.0881，后续优化重点不是继续大幅换注意力，而是针对 H41 TX 压 AAE。

### H49：QKFormer-native 三值 Shiftmax selector 短测（2026-05-26）

H49 是为了回应“H45/H47 太像另起炉灶、H41 又像外挂 gate”的问题而开的中间路线。它不做 H45/H47 的 `N x N` token-token attention，也不再用原 QKFormer 的 `sn2(sum(Q))` 后再挂一个额外 gate，而是把 QKFormer 的 token selector 改成同 token 的三值 Q/K 一致性 selector：

```text
score_i = TX(q_i, k_i)
selector = Shiftmax(score over tokens)
out_i = k_i * selector_i
```

也就是说，H49 保留 K carrier 和线性复杂度，不引入独立 V，不做 `gate @ K/V` 的跨 token 混合。代码入口为 `bsa_attention.mode=ternary_alpha_xnor_qkselector_shiftmax`，生成脚本为 `entrypoints/make_h49_qkselector_configs.py`。

#### 160-step valid10 早筛

第一轮 11 个配置中有 6 个最初因为 H48 临时训练并行导致 OOM，已在 GPU 空闲后补跑，下面只列干净结果。valid10 只作为早筛，不作为论文指标。

| rank | 配置 | AEE | AAE | SOPs | firing | 备注 |
|---:|---|---:|---:|---:|---:|---|
| 1 | `h49_txsel_s02_tr07_score075` | 1.0552 | 6.2606 | 3.2889G | 0.07715 | score_scale=0.75，早筛最佳 |
| 2 | `h49_txsel_s02_tr07_nopreserve` | 1.1696 | 6.6089 | **3.1305G** | 0.07343 | 不保均值，SOPs 最低但精度回落 |
| 3 | `h49_txsel_s02_tr07_score125` | 1.0965 | **6.2294** | 3.2576G | 0.07642 | score_scale=1.25，AAE 早筛最低 |
| 4 | `h49_txsel_s02_tr07_softffn` | 1.1480 | 6.8619 | 3.2769G | 0.07687 | FFN 阈值增长放软，唯一早筛 zero_neg=0 |
| 5 | `h49_txsel_sn2only_tr07` | 1.0736 | 6.2723 | 3.5494G | 0.08326 | 精度可，但 SOPs 偏高 |
| 淘汰 | `h49_txsel_s02_tr07_warm` | 1.1464 | 6.6785 | 3.4980G | 0.08206 | worst pos/neg 比例异常，正负发放失衡 |

#### 360-step valid40 确认

将 160-step 前 4 个候选推进到 360-step valid40。排序显示 `softffn` 反而最稳，说明 FFN 阈值过快增长会损伤光流表达，放软 FFN 自适应阈值有收益。

| rank | 配置 | AEE | AAE | SOPs | firing | 判断 |
|---:|---|---:|---:|---:|---:|---|
| 1 | `h49_txsel_s02_tr07_softffn_steps360` | **1.8006** | **8.5361** | 2.9699G | 0.06967 | 当前 H49 全量候选 |
| 2 | `h49_txsel_s02_tr07_score075_steps360` | 1.8593 | 9.1438 | 2.9850G | 0.07002 | score_scale=0.75 保精度，但不如 softffn |
| 3 | `h49_txsel_s02_tr07_nopreserve_steps360` | 1.9002 | 9.3822 | **2.7302G** | 0.06404 | 最稀疏，但精度退化较明显 |
| 4 | `h49_txsel_s02_tr07_score125_steps360` | 1.9289 | 9.5937 | 2.9166G | 0.06842 | 放大 score 不利于精度 |

当前决定：启动 `h49_txsel_s02_tr07_softffn` 全量续训。它在 valid40 上比 H41 TX S02C 的典型 3G 结果 AAE 更低，同时 SOPs 仍约 3G；但最终是否能进入论文主线必须等 full-valid816 统一推理确认。

全量运行记录：

| run | 配置 | 入口 | checkpoint |
|---|---|---|---|
| `results/h49_txsel_s02_tr07_softffn_full30_20260525_181457` | `configs/generated/h49_txsel_s02_tr07_softffn.yml` | `entrypoints/train.py --prev_runid experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth` | `checkpoint_epoch{}.pth` 每轮保存 |

#### full-valid816 结果

已按统一 full-valid 协议补测 `checkpoint_epoch26` 和 `checkpoint_epoch29`。完整总表已更新到 `results/full_valid_profiles_20260525/FULL_VALID_RESULTS_ALL.md`。

| checkpoint | samples | AEE | AAE | PE1 | PE2 | PE3/outlier | SOPs | firing | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `epoch26` | 816 | **1.6725** | **9.9037** | 0.5194 | 0.2137 | 0.1083 | **3.1185G** | 0.07315 | H49 当前最佳 |
| `epoch29` | 816 | 1.6885 | 9.9646 | 0.5224 | 0.2147 | 0.1082 | 3.1324G | 0.07348 | 后期继续训练没有带来收益 |

与关键对照：

| 方案 | AEE | AAE | SOPs | firing | 结论 |
|---|---:|---:|---:|---:|---|
| baseline PSN | 1.3307 | 7.8132 | 3.9633G | 0.09297 | 精度上限 |
| H41 SC S012C epoch27 | 1.6223 | 9.4549 | 3.1278G | 0.07337 | 精度仍优于 H49，SOPs 接近 |
| H41 TX S02C epoch27 | 1.7083 | 10.0881 | **2.9482G** | 0.06916 | 更稀疏，但精度略低于 H49 |
| H49 TX selector softffn epoch26 | 1.6725 | 9.9037 | 3.1185G | 0.07315 | 比 H41 TX 稍准，但没有更稀疏 |

结论：H49 证明了 QKFormer-native 三值 selector 比 H45/H46/H47 的 direct TX attention 更稳，但没有超过 H41 SC 的精度，也没有超过 H41 TX 的稀疏度。它适合保留为“更优雅、精度略修复的 TX selector 消融”，暂时不替代 H41 SC/H41 TX 作为主线。

下一步只围绕 H41 TX 做三类修正：

1. 稀疏强度回退：把 SOPs 从 2.95G 放宽到 3.05-3.20G，换回 AAE/AEE，目标 `AEE<=1.70`, `AAE<9.2`, `SOPs<=3.2G`。
2. teacher 方向蒸馏：teacher 使用 baseline PSN epoch59，只在有效 flow 区域施加方向蒸馏，小运动/近零 GT 区域降低角度权重，避免之前粗 angular loss 与 QKFormer gate 冲突。
3. TX 层级约束：优先保持 H41 的替换范围和训练范式，只微调 target_rate、threshold_eta、activity_eta、lambda_sparse/方向蒸馏，不再同时改变 value 分支和注意力结构。

---

## 十七、外部顶会方案补充 (2026-05-26)

基于完整的 autoresearch 管道，针对 TX/SC/SN 三条线的精度问题和稀疏需求，从 2024-2025 顶会论文中提炼新方案。

### 方案来源

| 论文 | 会议 | 借鉴点 |
|------|------|--------|
| **SpiLiFormer** | ICCV 2025 | 双通道兴奋/抑制注意力，侧向抑制实现自然稀疏 |
| **A²OS²A** | CVPR 2025 | 异构精度 Q/K/V：Q=二值, K=非负连续, V=三值 |
| **Bishop** | DAC 2025 | Token-Time Bundle + 误差约束剪枝 + 异构硬件核心 |
| **QP-SNN** | ICLR 2025 | SVD 奇异值通道剪枝 + 4-bit 量化，零额外训练 |
| **Addition-Only SSA** | CVPR 2025 | 纯加法脉冲自注意力，零乘法零 softmax |

---

### 方案 G：兴奋/抑制双通道注意力 — 来自 SpiLiFormer (ICCV 2025)

**动机**：当前 TX/SC/SN 的 Q 是单一三值通道。SpiLiFormer 证明将 Q 分裂为兴奋 (Q_excite) 和抑制 (Q_inhibit) 两个通道，做**差分注意力**，能自然抑制噪声 token 并实现内生稀疏。

**设计**：
```
当前:  Q_ternary → scores → gate → 所有 token 参与注意力
改进:  Q → split → Q_excite (兴奋), Q_inhibit (抑制)
       A_excite = Q_excite · K  →  兴奋强度
       A_inhibit = Q_inhibit · K  →  抑制强度
       A_diff = SN(A_excite - A_inhibit)  →  差分门控 (SN = 脉冲)
       output = A_diff × K
```

**硬件**：兴奋和抑制共享同一 K，仅 Q 分裂。额外开销=1 个 split + 1 个 sign，无新增乘法器。抑制通道高发放 → 天然稀疏——抑制信号压过兴奋时 token 被跳过。

**对现有问题的解决**：AAE 退化的核心是 attention 被噪声 token 干扰。抑制通道主动识别并压制噪声区域，而不是让所有三元 token 平等参与投票。FFN 过稀疏压力也会减轻——attention 自己变精准后，FFN 不需要补偿。

**可行性**：`bsa_attention.py` 新增 `q_split_excite_inhibit` 模式，~80 行。Q 的 `linear_q` 权重原地 split，不需要额外参数。

---

### 方案 H：异构精度 Q/K/V — 来自 A²OS²A (CVPR 2025)

**动机**：A²OS²A 从信息论证明：Q/K/V 全二值化会损失 31/32 的信息。关键公式是 **Q=二值, K=非负连续, V=三值**——每个组件得到它需要的精度，而不是一刀切。

**当前问题对应**：H 系列把 Q 和 K 都设成三值 {-th, 0, +th}，K 的幅度信息被 `_ternary_sign_ste()` 丢弃。A²OS²A 的"K 保持非负连续"正好保护了 K 的幅度信息，又不增加 SOPs（K 的幅度只在 Q·K gate 里用一次）。

**设计**：
```
当前:  Q = ATLIFTernary(linear_q(x))    → {-th, 0, +th}
        K = ATLIFTernary(linear_k(x))    → {-th, 0, +th}
        gate = sign(Q) · sign(K) 或 α-XNOR(sign(Q), sign(K))

A²OS²A 改编:
        Q = ATLIFBinary(linear_q(x))      → {0, +th} (二值)
        K = ReLU(linear_k(x))             → [0, +∞)  (非负连续, 保留幅度!)
        V = ATLIFTernary(linear_v(x))     → {-th, 0, +th}
        scores = Q · K^T                  → 纯加法 (Q二值, 只做加法累加)
        gate = Shiftmax(scores)
        output = gate @ V
```

**关键变化**：
- K 不再走 ATLIF 阈值——保持 ReLU 连续值。硬件上 ReLU = 比较器 + MUX，比三值更简单。
- Q 从三值降为二值——更稀疏（只有正脉冲），硬件简洁。
- 需要加独立 V projection（同 H44）。

**硬件**：Q 二值 × K 连续 = 加法累加（Q 的二值性是选择器——Q=1 时累加 K，Q=0 时跳过）。Shiftmax 不变。V 三值 × gate = 带符号累加。

---

### 方案 I：Token-Time Bundle 稀疏训练 — 来自 Bishop (DAC 2025)

**动机**：Bishop 专为 spiking transformer 的硬件加速设计。BSA (Bundle Sparsity-Aware) 训练在训练时增强结构化稀疏，不是简单的 per-neuron threshold——它优化的是 **token-time 块的稀疏模式**，让硬件更容易利用。

**设计**：
```
当前:  per-neuron target_rate 反馈 → 每个神经元独立调节阈值
Bishop: 增加 Bundle Sparsity Loss
        Loss += λ_bundle × Σ (1 - sparsity(bundle))²  / num_bundles
        其中 bundle = 连续 T 个 timestep × K 个 token 的时空间块
        目标是: 让硬件上相邻的 spike 要么一起发要么一起不发
```

**为什么比当前好**：当前 target_rate 只管总发放率，不关心发放的时空分布。Bishop 的 bundle sparsity 优化的是硬件利用率——连续的零 = 可以跳过的连续时钟周期。

**对 SDformerFlow 的适用性**：bundle = 注意力窗口内的时空块。如果同一窗口内所有 token 都不发 spike，整个窗口的计算可以跳过——这个跳过粒度比 per-neuron gate 大得多。

**改动**：在 `h9_losses.py` 新增 bundle_sparsity_loss，~30 行。

---

### 方案 J：阈值驱动的通道剪枝 — 替代 SVD (基于 QP-SNN ICLR 2025 思路)

已在前面方案 E 中详述，此处不再重复。关键公式：

```
importance = (1 / threshold_mean) × ||weight||₂  
prune_mask = importance < quantile(p, all_importances)
fine_tune 2 epochs with pruned model
```

**补充硬件事**：剪枝后的矩阵维度减小 → PE 阵列列数减少 → 硬件面积和功耗直接下降。零额外训练开销（训练完直接剪，微调 2 epoch）。

---

### 方案 K：Key-as-Proxy V — 不加独立 V 的 A²OS²A 改编

**动机**：H47 加独立 V 失败了——破坏了 QKFormer 先验。A²OS²A 也需要独立 V。但如果不想加 V，可以保留 A²OS²A 的异构精度设计：

```
Q = ATLIFBinary(linear_q(x))          → {0, +th} (二值)
K = ReLU(linear_k(x))                 → [0, +∞)  (非负连续)
V = sign(K) 或 threshold(K)           ← 用 K 本身当 V, 不加独立 V
scores = Q · K^T                      → 加法累加
gate = Shiftmax(scores)
output = gate @ V
```

**与 H42/H44 对比**：H42d 用了独立 V（`_independent_value_tokens`），但加了太多改动。方案 K 只需改动 Q 和 K 的精度类型——Q 从三值→二值，K 从三值→ReLU——不需要独立 V。

---

## 十八、全部方案优先级总排序

| # | 方案 | 来源 | 解决什么 | 改动量 | 风险 | 优先级 |
|---|------|------|---------|--------|------|--------|
| **A** | 分层 target_rate | 自身经验 | FFN 过稀疏 | 只改 config | 低 | **P0** |
| **F** | Residual TX gate | Codex | AAE 退化 | ~15行 attention | 低 | **P0** |
| **C** | Progressive unfreezing | 自身经验 | 训练不稳定 | ~30行 train.py | 中 | **P0** |
| **G** | 兴奋/抑制双通道 (SpiLiFormer) | ICCV 2025 | 噪声抑制+自然稀疏 | ~80行 attention | 中 | **P1** |
| **K** | K-as-Proxy V (A²OS²A改编) | CVPR 2025 | 保护K的幅度信息 | ~40行 attention | 中 | **P1** |
| **E** | 阈值驱动通道剪枝 | QP-SNN ICLR 2025 | C3 贡献点 | ~100行 新文件 | 低 | **P1** |
| **H** | 全A²OS²A (Q二值+K连续+V三值) | CVPR 2025 | 异构精度 | ~120行 attention+installer | 高 | **P2** |
| **D** | 方向一致性 loss | 自身经验 | AAE 辅助 | ~30行 loss | 低 | **P2** |
| **I** | Token-Time Bundle (Bishop) | DAC 2025 | 硬件稀疏利用 | ~30行 loss+trainer | 中 | **P2** |
| **B** | SN+attention + TX+FFN 混合 | 自身经验 | 探索性 | ~50行 installer | 中 | **P3** |

### 建议分批推进

```
第一批 (P0 — 立刻):
  A + F + C → 修 FFN过稀疏 + AAE退化 + 训练稳定性

第二批 (P1 — 等第一批出结果):
  G + K + E → 兴奋/抑制注意力 + K保护幅度 + 论文C3贡献

第三批 (P2 — 消融/增强):
  H + D + I → 全A²OS²A对照 + 方向loss + Bundle稀疏

第四批 (P3 — 探索):
  B → 混合注意力
```

---

## 十九、valid40 → 全量 test set 问题

**当前状态**：所有 profile 使用 `--split valid --num-samples 40`，即 DSEC 验证集 40 帧。论文需要 report DSEC 完整 test set 结果（通常 ~10k 帧）。

**解决**：
1. 写 `tools/profile_full_test.py`：加载 checkpoint + config → 遍历 DSEC test set 全量帧 → 计算 AEE/AAE/SOPs
2. DSEC test set 路径：`data/Datasets/DSEC/saved_flow_data/test/`
3. 使用 DSEC 官方评测协议（不公开 GT，需要提交到 benchmark server）或使用公开的 test split
4. 短期：至少跑 DSEC `valid` split 的全量（不止 40 帧），得到统计显著的指标

**DSEC 评测注意**：
- DSEC benchmark 需要提交预测结果获取评测
- 训练用 `train/valid` split，最终 test 提交是盲测
- 当前 valid40 可以继续用于开发，但最终论文需要 (1) full valid 指标 (2) DSEC benchmark 提交结果

---

## 二十、H50/H51/H52：H49 后续分层稀疏与新注意力短测（2026-05-26）

### 本轮目标

H49 full-valid 结果说明 QKFormer-native TX selector 比 direct TX attention 稳，但仍未超过 H41 SC：

| 方案 | AEE | AAE | SOPs | firing |
|---|---:|---:|---:|---:|
| baseline PSN | 1.3307 | 7.8132 | 3.9633G | 0.09297 |
| H41 SC S012C epoch27 | 1.6223 | 9.4549 | 3.1278G | 0.07337 |
| H49 TX selector softffn epoch26 | 1.6725 | 9.9037 | 3.1185G | 0.07315 |

所以本轮不再大范围乱换结构，而是围绕 H49 做三条短测：

1. **H50：H49 + 分层 target_rate / 分层阈值增长**  
   目的：保留 H49 selector，按 stage 调整 Q/K ATLIF 的 `target_rate`、`threshold_eta`、`threshold_lr_scale` 和 `activity_eta`，避免所有 stage 一刀切导致 AAE 高。

2. **H51：双通道兴奋/抑制 selector**  
   目的：保留 H49 的线性 K carrier，不引入独立 V；但把三值 Q/K 的正负脉冲拆成兴奋和抑制证据：

   ```text
   excite  = Q+K+ + Q-K-
   inhibit = Q+K- + Q-K+ + one-sided activity
   score   = excite - beta * inhibit + alpha * silent_match
   gate    = Shiftmax(score over tokens)
   output  = K * gate
   ```

   这条线用于验证负脉冲是否能作为噪声抑制信号，而不是只参与 TX 一致性投票。

3. **H52：K-as-Proxy V 的 A²OS²A 改编**  
   目的：测试“不加独立 V”的 A²OS²A 路线。Q 走二值选择，K 保留非负幅度打分，V 直接复用 K：

   ```text
   scores = binary(Q) @ relu(K)^T
   gate   = Shiftmax(scores)
   output = gate @ K
   ```

   H52 属于风险较高的 direct attention 短测，只短测，不直接全量。它用于判断有没有必要继续走 A²OS²A/KASV 路线。

### 代码改动

| 文件 | 改动 |
|---|---|
| `overlay/models/STSwinNet_SNN/atlif_ternary_psn/installer.py` | 增加 `stage_threshold_eta`、`stage_threshold_lr_scale`、`stage_target_rate_eta`，让 H50 能真正按 stage 控制阈值增长和 target-rate 反馈 |
| `overlay/models/STSwinNet_SNN/bsa_attention.py` | 增加 `dual_channel_qkselector_shiftmax/h51` 和 `a2os2a_kasv_shiftmax/h52` 两个 attention mode |
| `entrypoints/make_h50_h51_h52_configs.py` | 生成 H50/H51/H52 短测配置 |
| `entrypoints/run_h50_h51_h52_autopilot.py` | 自动运行 H50/H51/H52 360-step 短测，随后从 H50 候选中选择一个启动 full30 |

### 短测候选

| 实验 | 注意力 | 分层稀疏策略 | 目的 |
|---|---|---|---|
| `h50a_h49_layered_precision` | H49 TX selector | rate 放松、阈值增长较慢 | 保精度优先，目标 SOPs 3.1-3.25G |
| `h50b_h49_layered_balanced` | H49 TX selector | stage0/2 稍强稀疏，后层放松 | 平衡精度和 3G SOPs |
| `h50c_h49_layered_sparse` | H49 TX selector | 更强分层稀疏 | 验证能否回到 3G 以下 |
| `h51a_dual_channel_balanced` | 双通道 selector | 沿用 H50b | 测负脉冲抑噪是否压 AAE |
| `h51b_dual_channel_precision` | 双通道 selector | 沿用 H50a | 双通道保精度版本 |
| `h52a_kasv_a2os2a_shiftmax` | KASV direct attention | 沿用 H50a | 测 A²OS²A/KASV 是否值得继续 |

### 执行策略

先跑全部候选的 360-step 短测，并做 valid10 + valid40 profile。短测结束后只从 H50 三个候选里选一个启动 full30，因为本轮全量主目标是验证 H49 分层稀疏；H51/H52 先只作为注意力方案筛选。

全量选择规则：

```text
pick_score = AEE + 0.035 * AAE + 0.25 * max(0, SOPs - 3.20)
```

若短测结果不可用，则默认启动 `h50a_h49_layered_precision`，避免 GPU 空闲。

---

## 二十一、H53：修正 ATLIF 范式与 stage3 替换范围（2026-05-26）

### 背景

H50c full-valid816 出现过稀疏：`AEE=1.8953`、`AAE=57.8867`、`SOPs=2.7896G`、`firing=0.06544`。复核后认为它不适合作为主线，主要问题是 Q/K 使用了额外的 target-rate 反馈，且 stage3 也被纳入替换，和“ATLIF 原论文阈值自适应范式 + stage3 尽量不动”的要求不一致。

因此 H53 重新收敛到一个更干净的 H49 变体：

- Q/K 仍是 **PSN + ATLIF + 三值阈值输出**，但 Q/K 不再使用 target-rate 控制。
- FFN 仍用 **official ATLIF binary**，即更接近 Activity-Pruning-SNN 的二值阈值增长范式。
- 单边发放惩罚 `0/±1`、`±1/0` 保留，并修复为 **hard 前向 + STE surrogate 梯度**。
- stage3 不做 Q/K 三值替换，也不做 H49 selector attention 替换。
- 本轮仍按当前约定使用 local valid816 split，后续正式论文口径再统一 valid825。

### 代码与配置

| 项 | 路径 / 设置 |
|---|---|
| 注意力代码 | `neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/bsa_attention.py` |
| 单测 | `neuron_experiments/H9_bipolar_self_attention/tests/test_bsa_attention.py` |
| 短测配置 | `configs/generated/h53b_h49_clean_no_stage3_s02.yml` |
| 全量配置 | `configs/generated/h53b_h49_clean_no_stage3_s02_full30.yml` |
| baseline checkpoint | `experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth` |
| train split | `valid816` 对应 backup train csv |
| valid split | `valid816` 对应 backup valid csv |

### H53b 替换范围

| 模块 | 替换范围 | 神经元 / 注意力 |
|---|---|---|
| Q/K | stage0 全部 block、stage1 全部 block、stage2 全部 6 个 block | PSN + ATLIF ternary，`target_rate=null` |
| attention | stage0/1/2 全部 block | `ternary_alpha_xnor_qkselector_shiftmax` |
| FFN | stage0 全部 block + stage2 的 block 0/2/4 | official ATLIF binary |
| stage3 | 不替换 | baseline PSN / baseline QK attention |

### 代码修复确认

`single_active_penalty` 修复点：H49 token selector 分支以前 forward 里扣了单边发放分，但 `single_active_penalty_grad=ste` 没进入 token-score 反向路径。现在 `_ternary_alpha_xnor_token_scores` 会在 STE 模式下用 active proxy 计算单边项，保持 hard forward 不变，同时给 Q/K 提供 surrogate 梯度。

已验证：

```text
python -m py_compile bsa_attention.py test_bsa_attention.py
python -m unittest neuron_experiments.H9_bipolar_self_attention.tests.test_bsa_attention
```

结果：11 个 attention 单测通过。

### H53b 720-step 短测结果

| 实验 | samples | AEE | AAE | SOPs | firing | ATLIF 模块 | target-rate 控制 | 备注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `h53b_h49_clean_no_stage3_s02_steps720` | 40 | 1.7438 | 8.5206 | 3.3115G | 0.07768 | 30 | 0 | stage3 已排除，正负三值发放未整体塌缩 |

短测日志关键值：

- `num_modules=30`：stage0/1/2 的 Q/K 20 个 + FFN 10 个，符合预期。
- `symmetric_bsa_tsn_modules=20`：只有 Q/K 三值模块。
- `official_atlif_modules=10`：只有 FFN 二值 ATLIF 模块。
- `target_rate_control_modules=0`：Q/K 没有再用 target-rate 反馈。
- `Shiftmax attention summary.num_modules=10`：attention 只替换 stage0/1/2 的 10 个 block。

### H53b full30 运行记录

| 项 | 内容 |
|---|---|
| 状态 | 已启动 |
| 目录 | `neuron_experiments/H9_bipolar_self_attention/results/h53b_h49_clean_no_stage3_s02_full30_20260526_203146` |
| PID | bash 2512993 / train 2512996 |
| 配置 | `neuron_experiments/H9_bipolar_self_attention/configs/generated/h53b_h49_clean_no_stage3_s02_full30.yml` |
| 日志 | `neuron_experiments/H9_bipolar_self_attention/results/h53b_h49_clean_no_stage3_s02_full30_20260526_203146/train.log` |

判断：H53b 不是追求最低 SOPs 的版本，而是“修正 ATLIF 范式 + 排除 stage3 + 保持 H49 selector 可解释性”的干净全量。若 full-valid816 不能超过 H49/H41-SC，则下一步优先讨论 SC 路线，而不是继续在 TX selector 上叠更多控制项。

---

## 二十二、H54：TX bipolar selector 两分数/三分数短测（2026-05-26）

### 动机

H49/H53 的 TX selector 为：

```text
score_i = TX(q_i, k_i)
gate_i  = Shiftmax(score_i) > 0
out_i   = gate_i * k_i
```

这个结构只能衰减或增强 K，不能反转 K 的方向。若 Q/K 异号代表强反向证据，正 gate 只能“不看它”，不能“方向纠正它”，这可能是 AAE 难以下降的原因之一。

### 新增注意力

| 实验 | mode | 公式 | 目的 |
|---|---|---|---|
| H54a | `bipolar_qkselector_shiftmax` | `gate = g_same - λ*g_opp` | 纯两分数 bipolar，验证反向 gate 是否能压 AAE |
| H54b | `tx_bipolar_qkselector_shiftmax` | `gate = g_tx + μ*(g_same - λ*g_opp)` | 保留 TX 稳定选择，再加 bipolar 方向修正 |

其中：

```text
same_score = count(q==k!=0) + α*count(q==0,k==0)
opp_score  = count(q==-k!=0) + γ*count(one-side-active)
tx_score   = same_score - β*opposite - γ*one_side

g_* = Shiftmax(score_*)
out = gate * K
```

H54 的关键点是：每一路 Shiftmax 仍然非负、硬件友好，但合成后的 effective gate 可以为负，从而允许 `K` 被反向证据翻转。

### 代码改动

| 文件 | 改动 |
|---|---|
| `overlay/models/STSwinNet_SNN/bsa_attention.py` | 新增 `_bipolar_token_score_components`、H54a/H54b 两个 mode、`bipolar_mu/bipolar_lambda/bipolar_gate_min/max` 配置 |
| `tests/test_bsa_attention.py` | 增加 H54 score split 和 tiny attention smoke |
| `entrypoints/make_h54_bipolar_configs.py` | 生成 H54 sweep 配置 |

验证：

```text
python -m py_compile bsa_attention.py test_bsa_attention.py
python -m unittest neuron_experiments.H9_bipolar_self_attention.tests.test_bsa_attention
```

结果：13 个单测通过。

### 短测排列组合

H53b full30 已在 `epoch2` 保存 checkpoint 后暂停，让 GPU 给 H54 短测。H53b 可从：

```text
neuron_experiments/H9_bipolar_self_attention/results/h53b_h49_clean_no_stage3_s02_full30_20260526_203146/checkpoint_epoch2.pth
```

恢复。

H54 sweep 使用 H53b 的干净替换范围：stage0/1/2 Q/K + attention，stage3 不动，FFN 使用 official ATLIF binary。每个候选 360 train steps + valid40 profile。

| 实验 | attention | λ | μ | 单边惩罚 γ | LR 策略 |
|---|---|---:|---:|---:|---|
| `h54a_two_l03_g10_base` | H54a two-score | 0.3 | 0 | 0.10 | base |
| `h54a_two_l05_g10_base` | H54a two-score | 0.5 | 0 | 0.10 | base |
| `h54a_two_l05_g20_base` | H54a two-score | 0.5 | 0 | 0.20 | base |
| `h54a_two_l05_g20_fast` | H54a two-score | 0.5 | 0 | 0.20 | fast neuron/threshold LR |
| `h54a_two_l08_g10_fast` | H54a two-score | 0.8 | 0 | 0.10 | fast neuron/threshold LR |
| `h54b_three_mu03_l08_g10_base` | H54b TX+bipolar | 0.8 | 0.3 | 0.10 | base |
| `h54b_three_mu05_l08_g10_base` | H54b TX+bipolar | 0.8 | 0.5 | 0.10 | base |
| `h54b_three_mu05_l10_g10_base` | H54b TX+bipolar | 1.0 | 0.5 | 0.10 | base |
| `h54b_three_mu05_l08_g20_fast` | H54b TX+bipolar | 0.8 | 0.5 | 0.20 | fast neuron/threshold LR |
| `h54b_three_mu07_l08_g10_fast` | H54b TX+bipolar | 0.8 | 0.7 | 0.10 | fast neuron/threshold LR |
| `h54b_three_mu02_l05_g10_warm` | H54b TX+bipolar | 0.5 | 0.2 | 0.10 | warmup |
| `h54b_three_mu03_l05_g15_warm` | H54b TX+bipolar | 0.5 | 0.3 | 0.15 | warmup |
| `h54b_three_mu03_l08_g15_slowbb_warm` | H54b TX+bipolar | 0.8 | 0.3 | 0.15 | slow-backbone warmup |
| `h54a_two_l03_g15_warm` | H54a two-score | 0.3 | 0 | 0.15 | warmup |

LR 策略：

| 策略 | backbone/norm LR | neuron LR | threshold LR |
|---|---:|---:|---:|
| base | `2e-7` | `1.2e-5` | `3e-6` |
| fast | `3e-7` | `2e-5` | `5e-6` |
| warmup | `2e-7` | `1.2e-5` | `3e-6`，前 100 step 从 0.2 倍线性 warmup |
| slow-backbone warmup | `1e-7` | `1.8e-5` | `4e-6`，前 120 step 从 0.15 倍线性 warmup |

运行目录：

```text
neuron_experiments/H9_bipolar_self_attention/results/h54_bipolar_sweep360_20260526_212301
```

### H54a 已完成短测结果

H54a 前 4 个组合已经给出趋势：纯两分数 signed gate 可以把 SOPs 压到 3.12-3.28G，但 AEE/AAE 没超过 H53b 的短测结果。强异号 `λ=0.8` 版本被跳过，避免继续把 GPU 花在已显弱势的方向上。

| 实验 | samples | AEE | AAE | SOPs | firing | 判断 |
|---|---:|---:|---:|---:|---:|---|
| `h54a_two_l05_g20_fast_steps360` | 40 | 1.8584 | 8.8976 | 3.1207G | 0.07320 | H54a 当前最好，SOPs 好但精度不如 H53b |
| `h54a_two_l05_g20_base_steps360` | 40 | 1.8512 | 8.9618 | 3.1841G | 0.07469 | 单边惩罚加大有帮助 |
| `h54a_two_l05_g10_base_steps360` | 40 | 1.8655 | 9.0605 | 3.1820G | 0.07464 | 中等 |
| `h54a_two_l03_g10_base_steps360` | 40 | 1.9291 | 9.4199 | 3.2787G | 0.07691 | 较弱 |

当前续跑目录：

```text
neuron_experiments/H9_bipolar_self_attention/results/h54b_bipolar_warm_sweep360_20260526_215331
```

### H54b/H54a 补充 sweep 结果

运行目录：

```text
neuron_experiments/H9_bipolar_self_attention/results/h54b_bipolar_warm_sweep360_20260526_215331
```

| rank | 实验 | 类型 | samples | AEE | AAE | SOPs | firing | 三值健康度 | 判断 |
|---:|---|---|---:|---:|---:|---:|---:|---|---|
| 1 | `h54b_three_mu05_l08_g20_fast_steps360` | H54b 三分数 | 40 | 1.7477 | 8.3664 | 3.3039G | 0.07750 | `worst_pos/neg=5.33` | 当前最佳，值得 720-step/valid816 |
| 2 | `h54a_two_l03_g15_warm_steps360` | H54a 两分数 | 40 | 1.7858 | 8.7273 | 3.2098G | 0.07529 | `worst_pos/neg=5.98` | 可作为纯 signed gate 对照 |
| 3 | `h54b_three_mu03_l08_g10_base_steps360` | H54b 三分数 | 40 | 1.7875 | 8.4672 | 3.3019G | 0.07745 | `worst_pos/neg=5.49` | AAE 好，值得 720-step/valid816 |
| 4 | `h54b_three_mu03_l05_g15_warm_steps360` | H54b 三分数 | 40 | 1.8657 | 9.0578 | 3.1712G | 0.07439 | 正常 | SOPs 低但精度弱 |
| 5 | `h54b_three_mu05_l10_g10_base_steps360` | H54b 三分数 | 40 | 1.8497 | 8.8944 | 3.2670G | 0.07664 | 正常 | 中等 |
| 6 | `h54b_three_mu05_l08_g10_base_steps360` | H54b 三分数 | 40 | 1.8928 | 9.1195 | 3.1649G | 0.07424 | 正常 | 中等偏弱 |
| 7 | `h54b_three_mu02_l05_g10_warm_steps360` | H54b 三分数 | 40 | 1.8742 | 9.2601 | 3.2384G | 0.07597 | 正常 | 修正太弱 |
| 8 | `h54b_three_mu07_l08_g10_fast_steps360` | H54b 三分数 | 40 | 1.9233 | 9.2985 | 3.2962G | 0.07732 | 正常 | 修正太强 |
| 9 | `h54b_three_mu03_l08_g15_slowbb_warm_steps360` | H54b 三分数 | 40 | 1.8094 | 8.7592 | 3.3290G | 0.07809 | `worst_pos/neg=66979.60` | 局部负发放塌缩，不作为主推 |

阶段结论：

- H54b 三分数比 H54a 两分数更符合当前目标：保留 TX 主选择，同时加入同号/异号方向修正，AAE 有改善空间。
- `μ=0.7` 过强，`μ=0.2` 过弱；当前最佳落在 `μ=0.5, λ=0.8, γ=0.20, fast LR`。
- warmup 本身没有带来稳定收益，`slowbb_warm` 还出现局部负发放塌缩。
- H54a warm 可以作为对照，但主线优先 H54b。

### H54 confirm 运行记录

为了验证 360-step 排序是否可靠，已启动 top3 的 720-step + valid40，并开启自动 valid816 promotion：

```text
neuron_experiments/H9_bipolar_self_attention/results/h54_top3_confirm720_valid816_*
```

候选：

| 实验 | 原因 |
|---|---|
| `h54b_three_mu05_l08_g20_fast` | 360-step 当前综合最优 |
| `h54a_two_l03_g15_warm` | H54a 纯 signed gate 最优对照 |
| `h54b_three_mu03_l08_g10_base` | AAE 稳定、三值健康，检验较轻修正是否更适合长训 |

promotion 阈值：`AEE <= 2.05`、`AAE <= 11.0`、`SOPs <= 3.6G`。通过者自动跑 local valid816，之后再决定是否进入 full30。

### H54 confirm 结果与 full30 决策

运行目录：

```text
neuron_experiments/H9_bipolar_self_attention/results/h54_top3_confirm720_valid816_20260526_230025
```

| 实验 | stage | samples | AEE | AAE | SOPs | firing | 三值健康度 | 判断 |
|---|---|---:|---:|---:|---:|---:|---|---|
| `h54b_three_mu05_l08_g20_fast_steps720_valid816` | confirm | 816 | 1.4624 | 8.9723 | 3.6025G | 0.08451 | `worst_pos/neg=4.94` | 精度最好，SOPs 比 baseline 低约 9.1%，作为 full30 主线 |
| `h54a_two_l03_g15_warm_steps720_valid816` | confirm | 816 | 1.4860 | 9.1416 | 3.5554G | 0.08340 | `worst_pos/neg=5.44` | 更稀疏但精度略差，保留为备选/对照 |
| `h54b_three_mu03_l08_g10_base_steps720` | screen | 40 | 1.8181 | 8.8346 | 3.3168G | 0.07780 | `worst_pos/neg=33489.80` | 局部负发放塌缩，不推进 |

与 baseline local valid816 对比：

| 实验 | AEE | AAE | SOPs | firing |
|---|---:|---:|---:|---:|
| baseline PSN epoch59 | 1.3307 | 7.8132 | 3.9633G | 0.09297 |
| H54b confirm720 | 1.4624 | 8.9723 | 3.6025G | 0.08451 |
| H54a confirm720 | 1.4860 | 9.1416 | 3.5554G | 0.08340 |

决策：优先跑 H54b full30。理由是 H54b 是更完整的“三分数 TX + 同号/异号方向修正”故事，精度优于 H54a，SOPs 虽略高于 3.6G 目标线但仍低于 baseline，且 full30 阈值继续更新后可能进一步降低 SOPs。

full30 运行记录：

| 项 | 内容 |
|---|---|
| 配置 | `neuron_experiments/H9_bipolar_self_attention/configs/generated/h54b_three_mu05_l08_g20_fast_full30.yml` |
| 运行目录 | `neuron_experiments/H9_bipolar_self_attention/results/h54b_three_mu05_l08_g20_fast_full30_20260526_234957` |
| PID | `2563936` |
| 入口 | `neuron_experiments/H9_bipolar_self_attention/entrypoints/train.py` |
| checkpoint | `checkpoint_epoch{}.pth`，每轮保存 |
| 数据口径 | 当前仍使用 local valid816 override；后续正式 profile 再统一 valid825 |

启动审查：

- `max_train_steps` 已从 full30 配置中移除，`loader.n_epochs=30`。
- Q/K 三值替换范围仍为 stage0/1/2，stage3 不动。
- FFN 为 stage0 全部 + stage2 block 0/2/4 的 official binary ATLIF。
- `target_rate_control_modules=0`，Q/K 不使用 target-rate，保持 H53/H54 的 ATLIF 范式修正。
- 分层学习率：backbone/norm `3e-7`，ATLIF neuron `2e-5`，threshold `5e-6`。
- attention mode：`tx_bipolar_qkselector_shiftmax`，`μ=0.5`，`λ=0.8`，`single_active_penalty=0.20`，`single_active_penalty_grad=ste`。
