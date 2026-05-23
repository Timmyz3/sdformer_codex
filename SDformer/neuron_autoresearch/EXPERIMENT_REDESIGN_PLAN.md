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

### 状态

| 实验 | 状态 | 结果 |
|------|:---:|------|
| SN S02 C dlr (H41) | 🔄 epoch 20/30 | ep9: AEE=1.95, SOPs=2.88G |
| SN S02 C cont (H41) | ✅ 完成 9ep | ep6: AEE=1.74, AAE=8.38, SOPs=2.70G |
| **TX S02 C slowbb** (H41) | ✅ **完成 30ep** | **ep27: AEE=1.73, AAE=8.40, SOPs=2.62G** |
| TX S02 A | ⏳ 待启动 | — |
| SC S012 C | ⏳ 待启动 | — |
| BQ S02 | ⏳ 待启动 | — |

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

### SC S012 C slowbb ang02 (signed consensus + stage0+1+2 + angular 0.2)

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
| — | PSN baseline | — | 1.585 | 7.501 | 3.622G | 0.085 |
| — | H9a (历史最优) | — | 1.504 | 7.637 | 3.085G | 0.072 |

### 核心结论

✅ **TX S02 C 全量 SOPs 2.59-2.62G（-28% vs baseline），首次稳定破 3G**
⚠️ SC S012 C ang02 的 angular loss 未修复精度 (ep12 AAE=8.39 最优但仍差), SOPs 2.7-2.9G
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
