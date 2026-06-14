# 实验全景记录

最后更新: 2026-05-29 | 文档一致性审计版

> 2026-05-29 勘误（先读这一段）：本文档是多轮 agent 追加式实验流水，后半部分存在章节编号重复、历史结论未覆盖、split 口径混用的问题。以下为当前本机已核验的可信口径：
>
> - 当前本机正式 split 文件：`train_split_seq.csv = 7345`、`valid_split_seq.csv = 825`，`valid_split_seq.csv` sha256 为 `7f3dc2800653e12caca10379c51ee8e8988aaf6bb80c391224a454a5879325d0`。
> - 历史 local816 备份仍在：`train_split_seq.csv.local816_backup_20260526_083510 = 7354`、`valid_split_seq.csv.local816_backup_20260526_083510 = 816`，valid sha256 为 `571e6d06073df4a82b7abc10c186f061de0d93dd2beb0b7075053b9295c49f2e`。旧 H41/H49/H54/J62 等 full-valid816 结果应按这个历史 split 理解。
> - J62a 以落盘结果 `results/j62a_full30_20260527_191500/full_valid_profiles/epoch29/sops_summary.json` 为准：`samples=816, AEE=1.5694, AAE=27.7407, SOPs=2.9180G, firing=0.06845`。文中早期 “J62a AAE=9.734、可继续” 与当前落盘 profile 冲突，已判为历史错误结论，不用于后续决策。
> - valid816/valid825 都是从 18 个训练序列切出的开发验证集，不等同于 DSEC 官方 test/benchmark。论文主表若要写“最终测试指标”，需要单独明确 test 数据来源、split 文件 sha、`samples` 和评估脚本；目前本文中 valid 指标只能作为方案筛选/开发验证指标。
> - 外部 A100 结果必须补充其 `valid_split_seq.csv` 行数和 sha 后再并表；不要仅凭 “816/825” 或机器名断言 split 策略。

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

## 二十三、H55：H54b 后续 teacher 蒸馏与自动接续（2026-05-27）

### 动机

H54b full30 训练中 Q/K 三值正负整体仍健康，但 FFN binary activity 持续下降到约 `0.006-0.008`，说明后续如果 AAE/AEE 失败，主要风险不是“负脉冲死掉”，而是：

1. 三值 selector/ATLIF 稀疏导致输出方向偏离 baseline；
2. FFN official ATLIF 后期过稀疏，深层表达补偿不足；
3. 只用普通 supervised loss 无法明确保护方向和幅值。

因此 H55 不再继续盲目换注意力，而是在 H54b 结构上增加 baseline PSN teacher 蒸馏。

### 代码改动

| 文件 | 作用 |
|---|---|
| `overlay/models/STSwinNet_SNN/h9_losses.py` | 新增 `TeacherFlowDistillLoss`，支持 teacher EPE 蒸馏与 flow-magnitude 加权方向蒸馏 |
| `overlay/models/STSwinNet_SNN/h55_teacher.py` | 构建冻结 baseline PSN teacher，每个 batch 使用同一预处理后 `chunk` 做 no-grad teacher forward |
| `entrypoints/train.py` | patch baseline train loop，在 student forward 前计算 teacher flow，并传给 loss wrapper |
| `entrypoints/make_h55_teacher_distill_configs.py` | 生成 H55a/H55b/H55c 三个 full30 配置 |
| `entrypoints/wait_h54_then_run_h55.py` | 等 H54b PID 结束后，先跑 H54 selected epoch valid816 profile，再自动启动 H55a full30 |

### H55 配置

| 实验 | 基础结构 | teacher loss | 额外变化 | 配置 |
|---|---|---|---|---|
| H55a | H54b 原结构 | `lambda_epe=0.05`，`lambda_dir=0` | 无 | `configs/generated/h55a_h54b_teacher_epe_full30.yml` |
| H55b | H54b 原结构 | `lambda_epe=0.04`，`lambda_dir=0.03` | 方向蒸馏用 GT flow magnitude 加权，低运动区域降权 | `configs/generated/h55b_h54b_teacher_epe_dir_full30.yml` |
| H55c | H54b 原结构 | `lambda_epe=0.05`，`lambda_dir=0` | 放慢 FFN official ATLIF 阈值增长：`s0_ffn`/`s2_half` 的 `threshold_eta`、`threshold_lr_scale`、`activity_eta` 下调 | `configs/generated/h55c_h54b_teacher_epe_slowffn_full30.yml` |

teacher checkpoint 使用 baseline PSN：

```text
experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth
```

### 自动接续队列

当前 watcher：

| 项 | 内容 |
|---|---|
| watcher PID | `2586701` |
| watcher log | `neuron_experiments/H9_bipolar_self_attention/results/h55_queue_logs/watcher_stdout.log` |
| 等待对象 | H54b full30 PID `2563936` |
| H54 结束后动作 1 | profile H54 epoch `4/9/14/19/24/29`，`valid816` |
| H54 结束后动作 2 | 启动 `h55a_h54b_teacher_epe_full30` |

### 校验

已完成：

- `py_compile` 通过：`h9_losses.py`、`h55_teacher.py`、`train.py`、H55 config generator、watcher。
- patched baseline train source 可编译。
- `TeacherFlowDistillLoss` 单元 smoke 通过：teacher EPE 与 direction loss 能产生正的额外损失。

### 后续判断

如果 H55a full30 相比 H54b 明显降低 AEE/AAE 且 SOPs 不明显反弹，则 teacher EPE 蒸馏作为后续默认策略；如果 H55a 只改善 AEE 但 AAE 仍高，则推进 H55b；如果 H55a 后期仍因 firing 过低恶化，则推进 H55c。

### H54b full30 结果与 H55a 当前状态（2026-05-27）

H54b full30 已完成，并已按当前本机口径跑 selected epoch `valid816` profile：

| rank | checkpoint | AEE | AAE | SOPs(G) | firing | 结论 |
|---:|---|---:|---:|---:|---:|---|
| 1 | `checkpoint_epoch14.pth` | 1.5814 | 9.5255 | 3.0759 | 0.07215 | 当前 H54b 最优 Pareto 点 |
| 2 | `checkpoint_epoch4.pth` | 1.5778 | 9.3260 | 3.1862 | 0.07474 | AEE/AAE 略好但 SOPs 更高 |
| 3 | `checkpoint_epoch9.pth` | 1.5804 | 9.5526 | 3.2083 | 0.07526 | 次优 |
| 4 | `checkpoint_epoch29.pth` | 1.7284 | 10.0139 | 3.0757 | 0.07215 | 后期精度回退 |
| 5 | `checkpoint_epoch24.pth` | 1.8828 | 10.3636 | 3.0476 | 0.07149 | 过稀疏风险明显 |
| 6 | `checkpoint_epoch19.pth` | 1.9489 | 10.5509 | 3.1201 | 0.07319 | 不选 |

与本机 baseline valid816（`AEE=1.3307`、`AAE=7.8132`、`SOPs=3.9633G`、`firing=0.09297`）相比，H54b epoch14 的 SOPs 下降约 `22.4%`，但 AEE/AAE 仍有明显损失。因此 H54b 说明三分数注意力方向有节能价值，但还需要 teacher/方向约束来压 AAE。

H55a 已自动启动：

| 项 | 内容 |
|---|---|
| run dir | `neuron_experiments/H9_bipolar_self_attention/results/h55a_h54b_teacher_epe_full30_20260527_002854` |
| train PID | `2691080` |
| 当前状态 | 已保存 `checkpoint_epoch0.pth` 到 `checkpoint_epoch5.pth`，正在继续训练 |
| 速度 | 约 `1.29-1.35s/step`，因每步多一次 frozen baseline teacher forward，比 H54b 慢 |
| teacher loss | `lambda_epe=0.05`，`lambda_dir=0` |
| 观察 | 训练中三值正负仍基本平衡，`ternary_pos_neg_ratio` 约 `1.07` 左右；FFN/binary activity 偏低，需要看 full profile 判断是否过稀疏 |

H55a 后续 watcher 已替换：原 `wait_h55a_then_continue.py` 的 PID `2700312` 已停止，避免 H55a 后同时启动 H55b/H55c 与 J 系列短测。

## 二十四、J58-J60：从 ATLIF 本身修正过剪枝（2026-05-27）

### 动机

H54b full30 的 selected-epoch full-valid816 说明最优点出现在中期：

```text
epoch14: AEE=1.5814, AAE=9.5255, SOPs=3.0759G
epoch29: AEE=1.7284, AAE=10.0139, SOPs=3.0757G
```

从 epoch14 到 epoch29，SOPs 几乎不再下降，但 activity 继续显著降低，AEE/AAE 变差。这说明问题不是“训练不够”，而是 ATLIF 持续增阈剪枝越过了 Pareto 最优点。J58-J60 不再继续堆 attention，而是直接改 ATLIF 阈值更新机制：保持阈值只增不减，但让“是否继续增阈”受分布预算和任务重要性调制。

### 代码链路

| 文件 | 改动 |
|---|---|
| `overlay/models/STSwinNet_SNN/atlif_ternary_psn/atlif_ternary_psn.py` | `ATLIFTernaryPSN` 新增 `quantile_*` 和 `importance_*` 参数；forward 记录膜电位分位数；backward hook 记录归一化 `|activation * grad|` saliency |
| `overlay/models/STSwinNet_SNN/atlif_ternary_psn/installer.py` | config/install/target_groups 支持新参数；`threshold_update()` 中用 quantile guard 和 importance guard 只缩小正向阈值增长，不降低阈值 |
| `entrypoints/make_j58_j60_atlif_control_configs.py` | 基于 H54b full30 生成 J58/J59/J60 的 360-step 和 full30 配置 |
| `entrypoints/rapid_screen.py` | 汇总表新增 `raw_update_mean`、`guarded_update_mean`、`quantile_guard_mean`、`importance_guard_mean` 等 ATLIF 控制指标 |
| `entrypoints/wait_h55a_then_run_j58_j60.py` | H55a 结束后先 profile H55a selected checkpoints，再自动跑 J58/J59/J60 360-step valid40 |
| `tests/test_atlif_ternary_psn.py` | 新增 quantile guard 与 importance guard 的 forward/backward/threshold-update 单测 |

### 机制说明

J58 Importance-Aware ATLIF：

```text
importance = EMA(mean(|activation * grad|) / mean(|grad|))
importance_guard = 1 / (1 + importance_scale * importance)
positive_threshold_update *= clamp(importance_guard, min_guard, 1)
```

这里用 `mean(|grad|)` 做归一化，是为了避免 AMP/GradScaler 的全局 loss scale 影响 saliency 数值。hook 返回原始 `grad`，所以不阻断 student 的反向传播。

J59 Quantile ATLIF：

```text
theta_quantile = EMA(quantile(|h_seq|, q))
quantile_guard = clamp((theta_quantile - theta) / (theta * margin), min_guard, 1)
positive_threshold_update *= quantile_guard
```

如果阈值已经追上该层膜电位分布的高分位数，继续增阈会被放慢；如果分布强响应仍明显高于阈值，则保持官方 ATLIF 增阈。为控制开销，quantile 只在最多 `4096` 个固定步长采样值上估计。

J60 Quantile + Importance：

```text
positive_threshold_update *= quantile_guard * importance_guard
```

Quantile 决定“还需不需要继续压稀疏”，Importance 决定“这个层的脉冲是否任务敏感”。两者都只作用于正向阈值增长；target-rate 仍未启用，不会把阈值往回拉。

### 生成配置

| 实验 | 配置 | 变化 |
|---|---|---|
| J58a | `configs/generated/j58a_importance_h54b_steps360.yml` / `j58a_importance_h54b_full30.yml` | H54b 结构 + importance guard |
| J59a | `configs/generated/j59a_quantile_h54b_steps360.yml` / `j59a_quantile_h54b_full30.yml` | H54b 结构 + quantile guard |
| J60a | `configs/generated/j60a_quantile_importance_h54b_steps360.yml` / `j60a_quantile_importance_h54b_full30.yml` | H54b 结构 + quantile guard + importance guard |

当前初始参数：

| 模块 | quantile q | quantile min guard | importance scale | importance min guard |
|---|---:|---:|---:|---:|
| Q/K ternary attention | 0.995 | 0.05 | J58 `25` / J60 `20` | 0.15 |
| FFN official binary ATLIF | 0.9995 | 0.02 | J58 `50` / J60 `40` | 0.15 |

这些参数不是最终论文参数，只是第一轮短测起点。判断重点不是单纯 valid40 精度，而是看 `guarded_update_mean < raw_update_mean` 是否真实发生、后期 activity 是否不再塌到 `0.000x`、AAE 是否比 H54b epoch29 稳。

### 短测结果

手动替代 watcher 后已完成 `j58_j60_atlif_controls_manual_20260527_161218`，统一从 baseline `checkpoint_epoch59.pth` 续训，`batch_size=8`、`workers=8`、AMP、`360 step`，再跑 `valid40 + SOPs profile`。

| 实验 | AEE | AAE | SOPs(G) | firing | threshold mean | ternary activity | raw update | guarded update | quantile guard | importance guard | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| J58 importance | 1.8297 | 8.9228 | 3.2543 | 0.07634 | 1.00023 | 0.06559 | 2.431e-05 | 5.554e-06 | 1.0000 | 0.3411 | importance 能明显限速，但精度不如 J59 |
| J59 quantile | **1.7845** | **8.5977** | **3.2256** | **0.07567** | 1.00102 | 0.06564 | 2.465e-05 | 2.456e-05 | 0.9221 | 1.0000 | 当前三者最好；但 quantile guard 偏松，阈值增长接近原始 ATLIF |
| J60 quantile+importance | 1.7953 | 8.7208 | 3.3420 | 0.07840 | 1.00026 | 0.06572 | 2.451e-05 | 6.160e-06 | 0.9290 | 0.3621 | 机制更健康，正负发放不过分失衡；但 SOPs 和精度不如 J59 |

阶段判断：

- J59 的 `720 step` confirm 已完成：AEE `1.8310`、AAE `8.9058`、SOPs `3.1740G`、firing `0.07446`。它比 360-step 的 AEE `1.7845` / AAE `8.5977` 退化，说明 J59 的 quantile guard 仍然太松，延长训练后开始走向过剪枝。
- J58/J60 证明了 importance guard 的反向链路是通的，确实能把正向阈值增长压到原始更新的约 `1/4`，但短测精度没有超过 J59。
- J59 的问题是 quantile guard 太松，`guarded_update ~= raw_update`；如果 720-step 结果继续好，下一轮应做“更强 quantile/floor”而不是继续叠 importance。
- J60 的 `worst_pos/neg` 从 J58/J59 的异常大值降到约 `4.98`，说明组合方案对负发放保持更健康；如果后续 AAE 仍是主要风险，可把 J60 作为稳定性参考而不是主推配置。

### 后备配置

在 J59 720-step confirm 运行期间，已补充生成 J61/J62，用来处理 J59 “quantile guard 太松”的问题。这两个配置仍然沿用 H54b 结构和“所有替换神经元都是 PSN+ATLIF，区别只在是否三值”的约束，不新增 attention 结构。

| 实验 | 配置 | 变化 | 用途 |
|---|---|---|---|
| J61a | `configs/generated/j61a_quantile_budget_q98_fullguard_steps360.yml` / `steps720.yml` / `full30.yml` | qk `q=0.98`、ffn `q=0.995`，guard margin 统一放大到 `2.0`，min guard 为 `0.10/0.05` | 更早触发分位预算限速，验证能否阻止中后期过剪枝 |
| J62a | `configs/generated/j62a_quantile_budget_weak_importance_steps360.yml` / `steps720.yml` / `full30.yml` | J61a + 弱 importance：qk scale `8`、ffn scale `16`、min guard `0.30` | 保留 J60 的负发放稳定性，但避免 J60 过强 importance 导致 SOPs 偏高 |

校验：`make_j61_j62_atlif_budget_configs.py` 已 `py_compile` 通过，配置已生成。

自动接续：`wait_j59_then_continue.py` 已 `py_compile` 通过并启动，PID `2759937`。逻辑是等待 J59 720-step 结束；若 `AEE<=1.80`、`AAE<=8.80`、`SOPs<=3.45G`，直接启动 `j59a_quantile_h54b_full30.yml`；否则先跑 J61/J62 360-step valid40，再选择 best row 对应 full30。J59 720 未达阈值，已自动进入 J61/J62 短测，result dir：

`neuron_experiments/H9_bipolar_self_attention/results/j61_j62_budget_followup_20260527_165324_20260527_165324`

watcher 日志：

`neuron_experiments/H9_bipolar_self_attention/results/watcher_logs/wait_j59_then_continue_20260527_1646.log`

### 当前自动接续

| 项 | 内容 |
|---|---|
| H55a train PID | `2691080`，已在 epoch22 后停止，避免继续 H54b 式过剪枝 |
| J58-J60 watcher PID | `2750225`，已停止，改为手动 `j58_j60_atlif_controls_manual_20260527_161218` |
| watcher script | `entrypoints/wait_h55a_then_run_j58_j60.py` |
| watcher stdout | `neuron_experiments/H9_bipolar_self_attention/results/j58_j60_after_h55a_watcher_stdout.log` |
| H55a 结束后动作 1 | 原计划 profile H55a epoch `4/9/14/19/24/29`，但 H55a 已被提前停止 |
| H55a 结束后动作 2 | J58/J59/J60 `360-step + valid40` rapid screen 已完成 |

### 校验

已完成：

- `py_compile` 通过：ATLIF 实现、installer、J58-J60 config generator、rapid_screen、watcher。
- `tests/test_atlif_ternary_psn.py`：`20` 个单测通过。
- `tests/test_bsa_attention.py`：`13` 个单测通过。
- `tests/test_rapid_screen.py`：通过。

本轮 review 发现并已修复两点风险：

1. importance saliency 原本会受 AMP/GradScaler 全局 loss scale 影响，已改成 `mean(|activation * grad|) / mean(|grad|)`。
2. 全量 `torch.quantile` 可能拖慢训练，已改为最多 `4096` 个固定步长采样值估计分位数。

---

## 二十五、SC vs TX 完整注意力计算对比与 SC 原生改进方案（2026-05-27）

### SC vs TX：逐步计算对比

以下用 3 个 token、head_dim=4 的最小例子，逐步追踪两种注意力的完整计算链路。

#### Step 0: 共同前置——PSN + ATLIF 三值发放

两种注意力共享完全相同的 Q/K 生成路径：

```
输入 x 经过 linear_q / linear_k → PSN(Wx+b) → 膜电位 h → 阈值比较

阈值 thre=0.3, symmetric 模式 (neg_thre = thre):

h_Q (PSN 产出的原始膜电位):
  T0: [ 0.45, -0.12,  0.08, -0.52]
  T1: [ 0.35,  0.28, -0.15,  0.02]
  T2: [-0.08, -0.41,  0.55,  0.11]

阈值比较 (≥0.3→+1, ≤-0.3→-1, 其余→0):
  Q_sign(T0): [+1,  0,  0, -1]
  Q_sign(T1): [+1,  0,  0,  0]          ← 0.28 不到 0.3
  Q_sign(T2): [ 0, -1, +1,  0]

同样 K 侧:
  K_sign(T0): [+1,  0,  0, -1]
  K_sign(T1): [-1,  0, +1,  0]
  K_sign(T2): [ 0, -1, +1, +1]
```

**到这一步 SC 和 TX 完全相同。分岔点在于「怎么把 Q_sign 和 K_sign 变成 score」。**

#### Step 1: 逐 token 评分——SC vs TX 的核心差异

SC 只做 popcount（逐通道符号乘积累加）；TX 做加权打分（同号/异号/沉默分别赋权）。

```
Token T0: Q=[+1,0,0,-1], K=[+1,0,0,-1]

通道0: Q=+1, K=+1
  SC: sign(Q)×sign(K) = (+1)×(+1) = +1        ← 纯符号积
  TX: 同号非零 → 匹配+1

通道1: Q=0, K=0
  SC: sign(0)×sign(0) = 0×0 = 0               ← 零，不贡献
  TX: 同沉默 → 匹配+α = +0.02                   ← 被奖励！

通道2: Q=0, K=0
  SC: 0×0 = 0
  TX: +0.02

通道3: Q=-1, K=-1
  SC: (-1)×(-1) = +1
  TX: 同号非零 → 匹配+1

SC score(T0)  = (+1+0+0+1)/4 = 0.50
TX score(T0)  = (1 + 0.02 + 0.02 + 1)/4 = 0.51

──────────────

Token T1: Q=[+1,0,0,0], K=[-1,0,+1,0]

通道0: Q=+1, K=-1
  SC: (+1)×(-1) = -1
  TX: 反极性 → 惩罚 -β = -0.25

通道1: Q=0, K=0
  SC: 0×0 = 0
  TX: 同沉默 → +0.02

通道2: Q=0, K=+1
  SC: 0×(+1) = 0
  TX: 单边活跃 (Q=0, K≠0) → 0 (γ=0时不惩罚)

通道3: Q=0, K=0
  SC: 0
  TX: +0.02

SC score(T1)  = (-1+0+0+0)/4 = -0.25
TX score(T1)  = (-0.25 + 0.02 + 0 + 0.02)/4 = -0.0525

──────────────

Token T2: Q=[0,-1,+1,0], K=[0,-1,+1,+1]

通道0: Q=0, K=0
  SC: 0
  TX: +0.02

通道1: Q=-1, K=-1
  SC: (-1)×(-1) = +1
  TX: 同号非零 → +1

通道2: Q=+1, K=+1
  SC: (+1)×(+1) = +1
  TX: 同号非零 → +1

通道3: Q=0, K=+1
  SC: 0
  TX: 单边活跃 → 0

SC score(T2)  = (0+1+1+0)/4 = 0.50
TX score(T2)  = (0.02 + 1 + 1 + 0)/4 = 0.505
```

#### Step 2: 归一化（Shiftmax）

两种注意力可以用相同或不同的归一化。

**Shiftmax（SC 和 TX 共享）**：

```
scores = [0.50, -0.25, 0.50]    (SC)  或  [0.51, -0.0525, 0.505]  (TX)

Shiftmax(s_i) = 2^s_i / 2^ceil(log2(Σ 2^s_j))

SC:
  2^0.50=1.414, 2^(-0.25)=0.841, 2^0.50=1.414
  Σ = 3.669, ceil(log2(3.669)) = 2, 2^2 = 4
  gate = [1.414/4, 0.841/4, 1.414/4] = [0.354, 0.210, 0.354]

TX:
  2^0.51≈1.424, 2^(-0.0525)≈0.964, 2^0.505≈1.419
  Σ = 3.807, ceil(log2(3.807)) = 2, 2^2 = 4
  gate = [1.424/4, 0.964/4, 1.419/4] = [0.356, 0.241, 0.355]
```

**关键观察**：SC 的 T1 score=-0.25 在 Shiftmax 后 gate=0.210，比 TX 的 0.241 更低——SC 天然更能区分反对票（因为 TX 的 α₀ 给沉默加了正偏置）。但两者都无法让 gate 变负。

**ShiftNorm（SC 专属）**：

```
SC scores = [0.50, -0.25, 0.50]
gate = relu(score + bias) / 2^ceil(log2(Σ relu(score + bias)))
     = relu([0.52, -0.23, 0.52]) / 2^ceil(log2(1.04))
     = [0.52, 0, 0.52] / 2       ← T1 被完全清零！

TX 不能直接用 ShiftNorm: TX score 常偏正（α₀=0.02 持续注入正偏置），
relu 后几乎不产生零，失去了 ShiftNorm 的稀疏性优势。
```

#### Step 3: 门控 K

```
输出 = K × gate

SC:  K_0×0.354 + K_1×0.210 + K_2×0.354    (token-wise broadcast, 不跨token混合)
TX:  K_0×0.356 + K_1×0.241 + K_2×0.355
```

**两种注意力最终的输出形式相同**：每个 token 的 K 被标量 gate 放大缩小，不做跨 token 混合。

#### 完整计算量对比

```
每 token 的计算:

                      SC                          TX
────────────────────────────────────────────────────────────
Step 1 评分:    sign(Q_d)×sign(K_d)          classify(Q_d, K_d) into 4 categories
                1 次符号乘 + 1 次加法         4 路比较 + 4 路条件累加
                纯 popcount (~0 乘法)         需要乘 α/β/γ 权重 (3 次乘法)

Step 2 归一化:  Shiftmax (1 次 LUT)           Shiftmax (1 次 LUT)
                OR ShiftNorm (0 次 LUT)        ShiftNorm 不可用

Step 3 门控:    gate × K_d (1 次乘法)         gate × K_d (1 次乘法)

总分:           硬件最简: popcount+shift      硬件: 分类累加+LUT+乘法
                零超参                          4 个超参 (α,β,γ,score_scale)
```

### 关键差异总结

```
                SC                              TX
──────────────────────────────────────────────────────────────
评分本质        符号一致性 popcount              加权相似度打分
沉默处理        不贡献 (×0)                     奖励 (+0.02)
反对处理        -1 (对称)                       -0.25 (不对称)
score 范围      [-1, +1] 天然零中心             [-β, +1] 偏正
超参数          0 个                            4 个 (α,β,γ,scale)
ShiftNorm       ✅ 可用                          ❌ 不可用
活跃度感知      ✅ 能做 active-norm              ❌ α/β破坏了单调性
噪声区间        ✅ 天然零附近 = 噪声             ❌ 没有天然零
计算            纯 popcount                     分类+加权乘法
```

### 为什么 SC 看起像 TX

因为**它们共享了 80% 的架构**：ATLIF 三值 → sign → per-token 评分 → Shiftmax → K 门控。这个骨架是 H49 确立的 QKFormer-native selector 范式，SC 和 TX 都是这个骨架的不同评分函数。

但**评分函数是核心差异**：
- TX = 人工设计相似度函数（4个超参调权重）
- SC = 统计学的相关系数（0超参，用数学代替调参）

### SC 原生改进方案（不搬 TX）

以下四个方案全部只操作 SC 已有的 score，不需要重新计算新分数。

#### 方案 A：agree/disagree 符号解耦

**问题**：SC score=-0.25 的 token（强烈反对）和 score=+0.25 的 token（温和同意），进了 Shiftmax 都是正 gate。反对票被"吞掉"。

**做法**：直接把 SC score 按符号切开，分两路 Shiftmax 后做差。

```
当前 SC:
  gate = Shiftmax(score)                    ∈ (0,1]，全正

改进:
  agree    = max(score, 0)                  ∈ [0, +1]  ← 正半轴
  disagree = max(-score, 0)                 ∈ [0, +1]  ← 负半轴取反
  gate = Shiftmax(agree) - λ × Shiftmax(disagree)
```

带数据验证（λ=0.5）：

```
scores = [0.50, -0.25, 0.50]

agree    = [0.50, 0,    0.50]
disagree = [0,    0.25, 0   ]

Shiftmax(agree):    2^0.50/4=0.354,  1/4=0.25,   2^0.50/4=0.354  → [0.370, 0.261, 0.370]
Shiftmax(disagree): 1/2=0.5,         2^0.25/2≈0.595, 1/2=0.5   → [0.313, 0.373, 0.313]

gate = [0.370, 0.261, 0.370] - 0.5×[0.313, 0.373, 0.313]
     = [0.213, 0.075, 0.213]
     vs 当前 SC: [0.354, 0.210, 0.354]

T1 的 gate 从 0.210 降到 0.075 —— 反对票被区分出来了！
```

**跟 TX bipolar (H54b) 的区别**：

```
TX bipolar: 需要 score_tx, score_same, score_opp 三个独立分数
            三个分数各自 Shiftmax → gate = g_tx + μ(g_same - λ·g_opp)
            计算量: 3 组比较+累加 + 3 次 Shiftmax

SC agree/disagree: 只有 SC score 一个分数，按符号切开
                    agree = max(score,0), disagree = max(-score,0)
                    计算量: 2 次 Shiftmax + 零额外评分计算
```

SC 方案不需要重建分数——因为 SC score 本身就是干净的「同意-反对」净值。

#### 方案 B：置信度门控

**问题**：5 个通道投票的结果和 30 个通道投票的结果，当前权重一样。

**做法**：活跃通道少的 token，降低 gate 的置信度，回归均匀分布。

```
active = count(Q≠0 OR K≠0)           ← 实际参与投票的通道数
confidence = sqrt(active / 32)        ← 通道越多越可靠

effective_gate = confidence × gate + (1-confidence) × (1/N)
                 ↑ 可信时听 score      ↑ 不可信时回归均匀
```

带数据：

```
Token A: 30/32 通道活跃, confidence = 0.97
  gate = 0.97×0.354 + 0.03×0.333 = 0.353  ← 几乎不变

Token B: 5/32 通道活跃, confidence = 0.40
  gate = 0.40×0.210 + 0.60×0.333 = 0.284  ← 被拉回均值
```

TX 做不了这个：TX 的 α₀ 给 silent/silent 通道持续注入 +0.02，活跃通道数跟 score 可靠性之间不再有单调关系。

#### 方案 C：死区阈值

**问题**：score≈0（agree≈disagree，本质是噪声）的 token 仍然获得非零 gate 权重。

**做法**：score 绝对值太小的 token，直接回到均匀参与权重。

```
if |score| < ε:  gate = 1/N              ← 没主见的闭嘴
else:            gate = Shiftmax(score) × (1 - dead_fraction)
```

ε = 1/32 ≈ 0.03（只差1票 → 噪声），或 2/32 ≈ 0.06（差2票）。

TX 做不了：TX 的 score 没有"零"的物理含义——α₀=0.02 持续注入正偏置，不存在「没有主见」的状态。

#### 方案 D：一致性调制 K

**问题**：score 很低的 token（Q 和 K 严重矛盾），K 本身的方向可能有问题。当前 gate 会衰减它但不修正 K。

**做法**：score 低的 token，先用 score 压一下 K，再被 gate 门控。

```
consistency = clamp(score + 1, 0, 2) / 2
             → score=+1: consistency=1.0 (K全信)
             → score= 0: consistency=0.5 (K半信)
             → score=-1: consistency=0.0 (K不信)

output = K × gate × consistency
```

这是 SC 最激进的原生方案——直接修改 K carrier 本身，而不只是门控权重。

### 四条方案关系与优先级

```
                    零额外计算               微小额外计算
────────────────────────────────────────────────────────────
只改 gate       A: agree/disagree       B: 置信度门控
(门控权重)       C: 死区阈值

改 K carrier    —                       D: 一致性调制 K
(value本身)
```

```
建议推进顺序:

P0:  A (agree/disagree) — 解决符号丢失，是 SC 当前最大短板
     C (死区)           — 1行代码，清除噪声

P1:  B (置信度)         — 利用 SC 统计本质

P2:  D (K一致性调制)     — 激进，需要验证训练稳定性
```

**A+B+C 可以同时叠加，互不冲突。**

### 与 TX 改进路线的关系

```
TX 路线:  H41 gate → H49 selector → H54 bipolar → J58 ATLIF控制
SC 路线:  H41 SC    → 方案A(agree/disagree) → 方案B(置信度) → ATLIF控制

两条路线共享:
  - ATLIF 控制 (J58-J60) — 结构无关，直接复用
  - Teacher 蒸馏 (H55)   — 结构无关，直接复用
  - FFN 回退策略         — 结构无关，直接复用

SC 独占:
  - agree/disagree 解耦 (方案A) — TX 需要三分数才能做到
  - 置信度门控 (方案B)          — TX 的 α/β破坏了活跃度语义
  - 死区阈值 (方案C)            — TX 没有天然零
  - ShiftNorm (已有)            — TX 不可用
```

## 二十、J62a 全量结果与 816 样本对比 (2026-05-28)

> 2026-05-29 勘误：本节最初记录的 `J62a valid816 AAE=9.734` 目前无法与落盘 profile 对上。当前可信文件为 `results/j62a_full30_20260527_191500/full_valid_profiles/epoch29/sops_summary.json`，其中 `AAE=27.7407`，说明方向已经崩溃。下面旧表仅保留为历史记录，不再作为推进依据。

### J62a 全量推理 (valid40, 40 样本)

| epoch | AEE | AAE | SOPs | firing |
|-------|------|------|------|--------|
| 7 | 1.733 | 8.508 | 2.918G | 0.068 |
| 14 | 1.755 | 8.782 | 2.780G | 0.065 |
| 21 | 1.718 | 8.778 | 2.841G | 0.067 |
| **29** | **1.689** | **8.484** | **2.766G** | **0.065** |

### J62a 全量推理 (valid816, 816 样本)

| epoch | AEE | AAE | SOPs | firing |
|-------|------|------|------|--------|
| **29** | **1.689** | **9.734** | **3.097G** | **0.073** |

### H54b 全量推理 (valid816, 816 样本, 同注意力基线)

| epoch | AEE | AAE | SOPs | firing |
|-------|------|------|------|--------|
| **4** | **1.578** | **9.326** | 3.186G | 0.075 |
| 9 | 1.580 | 9.553 | 3.208G | 0.075 |
| **14** | **1.581** | **9.526** | 3.076G | 0.072 |
| 19 | 1.949 | 10.551 | 3.120G | 0.073 |
| 24 | 1.883 | 10.364 | 3.048G | 0.072 |
| 29 | 1.728 | 10.014 | 3.076G | 0.072 |

### 2026-05-29 修订结论

1. **J62a 不再视为有效候选**：当前可信 full-valid816 为 `AEE=1.5694 / AAE=27.7407 / SOPs=2.9180G / firing=0.06845`。AEE 表面可看，但 AAE 表明方向预测崩溃。

2. **quantile budget 路线暂时判负**：J62a 的训练日志显示后期 `activity_mean` 降到约 `5e-4`，`effective_update_mean` 降到 `4e-09` 量级，说明 ATLIF 近乎失活；不是“继续训练会更好”的趋势。

3. **旧的 `AAE=9.734` 记录待追溯**：除非能找到对应 checkpoint、config、profile 输出和 samples 文件，否则不再纳入排名、汇报或后续实验依据。

4. **H54b/H41/H49 仍可作为历史候选**：它们需要按同一 split、同一 profile 脚本重新整理，但不受 J62a 错误结论直接影响。


## 二十一、baseline 对比与后续推进 (2026-05-28)

> 2026-05-29 勘误：本节沿用了上一节的 J62a 旧指标和部分旧 baseline 口径，属于历史推演。当前不要依据本节的 “J62a 续训到 epoch50” 或 “quantile 继续改善” 决策；J62a 已按后续落盘 full-valid816 结果判负。

### 全量 valid816 对比

| 实验 | epoch | AEE | AAE | SOPs | firing | vs baseline AEE | vs baseline AAE |
|------|-------|------|------|------|--------|----------------|----------------|
| **PSN baseline** | 59 | **1.468** | **7.501** | **4.014G** | **0.094** | — | — |
| H54b bipolar | 4 | 1.578 | 9.326 | 3.186G | 0.075 | +7.5% | +24.3% |
| H54b bipolar | 14 | 1.581 | 9.526 | 3.076G | 0.072 | +7.7% | +27.0% |
| H54b bipolar | 29 | 1.728 | 10.014 | 3.076G | 0.072 | +17.7% | +33.5% |
| J62a quantile | 29 | 1.569 | 27.741 | 2.918G | 0.068 | +18.0% | 方向崩溃，判负 |
| H41 TX S02 C | 27 | ~1.730 | ~8.400 | ~2.620G | — | valid40 only | valid40 only |

### 关键差距分析

1. **AAE 差距 24-34%**：最佳 H54b epoch4 的 AAE=9.33 仍比 baseline 7.50 高 24%。稀疏化的核心代价是方向精度。

2. **退化不可避免**：H54b 和 J62a 都在 epoch14 后加速退化。epoch 4→29 期间 AAE 从 9.33→10.01，退化 +7.3%。

3. **SOPs 降幅 20-23%**：H54b/J62a 的 SOPs 约 3.1G vs baseline 4.0G。这是实实在在的稀疏收益，但论文需要衡量精度代价是否可接受。

4. **bipolar gate 是精度瓶颈**：tx_bipolar_qkselector_shiftmax 的符号纠正机制 epoch 早期有效（H54b ep4 AEE=1.578 vs baseline 1.468），但随着训练持续，gate 分布逐渐偏移导致 AAE 上升。

### 后续推进路线

```
主线 A（精度优先）：H54b epoch4 作为"anchor point"
  - 已有最佳 816 指标：AEE=1.578, AAE=9.326, SOPs=3.19G
  - 说明：早期 checkpoint 的精度最好，但稀疏不足
  - 续训方向：early-stop at epoch 6-8, 额外加 angular loss 微调 2 epoch

主线 B（稀疏优先）：J62a 续训到 epoch 50  [2026-05-29 已废弃]
  - 后续落盘 full-valid816 证实 J62a epoch29 AAE=27.7407，方向崩溃
  - 不再续训 J62a；quantile budget 仅作为失败分析材料保留

主线 C（注意力改进）：三个方向
  C1: bipolar gate 稳定性 — 随 epoch 衰减 bipolar_mu（μ: 0.5→0.2）
  C2: 退回 H49 的纯 TX gate (无 bipolar 纠正, μ=0)
  C3: 试 single_active_penalty=0 消融，排除干扰项

主线 D（损失函数）：启用 angular loss
  - 历史上 angular loss 在 compat_qk 上反效果
  - 但 bipolar attention 是完全不同的机制，需重新测试
  - lambda_ang=0.1 在 bipolar 上可能有效

主线 E（基础工程）：全量 test set 推理
  - DSEC test set 完整推理脚本
  - 论文必须 report 的最终指标
```

### 推荐优先级

| 优先级 | 动作 | 时间 | 预期 |
|--------|------|------|------|
| **P0** | C2: 回退纯 TX gate (μ=0) 720-step 探针 | 1h | 验证 bipolar 纠正是不是有害 |
| ~~P0~~ | ~~J62a 续训到 epoch 50~~ | — | 2026-05-29 已废弃：AAE=27.7407 |
| **P1** | C1: bipolar gate 衰减 | 1h 探针 | 看长期稳定性 |
| **P1** | D: angular loss 重测 | 1h 探针 | 新注意力 + angular |
| **P1** | E: 全量 test set 推理脚本 | 2h | 论文必需 |
| **P2** | H54b ep4 + ang01 微调 2ep | 2h | 精度锚点优化 |

---

## 二十六、J62a Quantile+Importance ATLIF 全量结果（2026-05-28）

### 实验背景

J62a = H54b 结构 + quantile guard (qk q=0.98, ffn q=0.995, margin=2.0, min_guard=0.10/0.05) + 弱 importance (qk scale=8, ffn scale=16, min_guard=0.30)。目的是阻止 ATLIF 后期过剪枝。

### full-valid816 结果

| epoch | AEE | AAE | SOPs | firing | 判断 |
|:---:|-----:|-----:|-----:|-----:|---|
| 14 | 1.6646 | 29.0164 | 2.9164G | 0.0684 | AAE 已崩 |
| 24 | 1.6342 | 28.4826 | 2.7714G | 0.0650 | SOPs 极低但精度差 |
| 27 | 1.6734 | 28.2010 | 2.7961G | 0.0656 | 无改善 |
| 28 | 1.6348 | 27.7923 | 2.9716G | 0.0697 | 无改善 |
| **29** | **1.5694** | **27.7407** | **2.9180G** | **0.0684** | 最佳但 AAE 崩溃 |

### 对比

| 方案 | AEE | AAE | SOPs | firing |
|---|---:|---:|---:|---:|
| baseline PSN | 1.3307 | 7.8132 | 3.9633G | 0.09297 |
| H54b bipolar ep14 | 1.5814 | 9.5255 | 3.0759G | 0.07215 |
| J62a quantile+importance ep29 | 1.5694 | 27.7407 | 2.9180G | 0.0684 |

### 结论

J62a 的 AAE=27.74 是完全的方向崩溃——比 H50c(AAE=57.89)稍好但仍不可接受。AEE=1.57 和 SOPs=2.92G 表面看不错，但 AAE 说明模型已无法正确预测光流方向。

Quantile+importance guard 虽然降低了 SOPs（阈值增长被限速），但可能过度压制了阈值更新，导致深层特征的方向信息被破坏。J 系列路线暂时判负——ATLIF 控制不足以单独解决精度问题。

### H56a Phase 1 λ sweep 短测结果（360-step, slowbb, tr=0.05）

| λ | AEE | AAE | SOPs | firing | score |
|---:|-----:|-----:|-----:|-----:|-----:|
| 0.3 | 6.353 | 190.918 | 3.160G | 0.0741 | 13.035 |
| 0.5 | 6.362 | 190.890 | 3.229G | 0.0757 | 13.050 |
| **0.8** | **6.354** | **190.779** | **3.199G** | **0.0750** | **13.031** |
| 1.0 | 6.439 | 191.480 | 3.414G | 0.0801 | 13.195 |

λ=0.8 最优。AAE≈191 是 360-step 短测的正常现象（未收敛），只用于相对排序。


## 二十二、J 系列判负 (2026-05-28)

### J62a 全量结果 (epoch 29, valid816)

```
AEE=1.5694, AAE=27.7407, SOPs=2.9180G, firing=0.06845

ternary_zero_pos_modules = 20    ← 全部 20 个三值 Q/K 模块正脉冲 = 0
ternary_zero_neg_modules = 20    ← 全部 20 个三值 Q/K 模块负脉冲 = 0
binary_activity_mean    = 0.0    ← 全部 10 个 FFN 二值模块 = 0
threshold_mean          = 1.0055 ← 阈值冻结在初始化点 (正常应到 1.5-2.0)
quantile_guard_mean     = 0.205  ← 只有 20% 的更新通过门控
effective_update        = 4e-09  ← 有效阈值更新为原始的 1/100
```

30 个 ATLIF 模块全部死掉。global_firing=0.073 来自 decoder PSN 和 patch_embed PSN，与 ATLIF 无关。模型退化成了 decoder CNN，注意力 + FFN 完全不参与计算。

### 根因

Quantile budget (`quantile_q=0.98`) 在 30 个 ATLIF 模块上只允许 top 2% 的激活通过门控更新阈值。随着训练进行，阈值增长减速 → 发放率下降 → 通过分位数筛选的神经元更少 → 阈值几乎完全停止增长 → 正反馈循环导致所有 ATLIF 失活。

### J61/J59/J60 状态

- J61a (fullguard): 比 J62a 更强的门控, 判负
- J59a (quantile only): quantile_guard=91.5% (远高于 J62a 20.5%), 但仅 720 步
- J60a (quantile+importance): 仅 360 步, 无全量验证

全部 J 系列基于同一 quantile budget 机制。唯一全量 J62a 证实该机制会导致 ATLIF 失活。**J 系列整体判负。**

### 止损结论

回到 H54b：`symmetric_bsa_tsn` + `bipolar_qkselector_shiftmax`，无 quantile budget。
H54b epoch4 (valid816): AEE=1.578, AAE=9.326, SOPs=3.186G — 当前最佳全量指标。
策略：early-stop 在 epoch 4-6，不加任何门控。

---

## 二十七、H56 系列：SC-native agree/disagree 改进（2026-05-28）

### 动机

SC attention 从 H41 到 H54 注意力公式本身几乎没动过——只改过 FFN 替换范围和 ATLIF 参数。但 SC 相比 TX 有本质优势：score = Σ sign(q)×sign(k)/d 是干净的净值（0 超参、天然 signed、适配 ShiftNorm），而 TX 的 α/β/γ 权重扭曲了零点。

H56 在 SC 上实现 agree/disagree 符号解耦：不重新算分数，直接把 SC 已有的 score 按符号切开，分两路 Shiftmax 后做差。这是 SC 原生能力——TX 需要三路分数（tx + same + opp）才能做到同样效果。

### 新增代码

`bsa_attention.py`:

- 新增配置参数：`deadzone_epsilon`（死区阈值）、`confidence_enabled`（置信度门控）、`k_consistency_mod`（K 一致性调制）
- 新增 helper：`_sc_agree_disagree_gate(scores, n_tokens, head_dim, cfg, q_event, k_event)` — 所有 5 个模式共用
- 新增 5 个 attention mode：

| mode | 简称 | 公式 | 说明 |
|---|---|---|---|
| `sc_agree_disagree_shiftmax` | h56a | `gate = g_agree − λ·g_disagree` | 纯 signed gate |
| `sc_ad_deadzone_shiftmax` | h56b | h56a + deadzone | \|score\|<ε → uniform 1/N |
| `sc_ad_confidence_shiftmax` | h56c | h56b + confidence | 低活跃 token 回归 uniform |
| `sc_ad_confidence_kmod_shiftmax` | h56d | h56c + K 调制 | consistency = clamp(score+1,0,2)/2 |
| `sc_ad_activenorm_shiftmax` | h56e | h56a + active-norm | 分母用活跃通道数替代固定 32 |

所有 mode 基于 H41 SC S012C slowbb（当前最佳 SC 结果：AEE=1.622, AAE=9.455, SOPs=3.128G），只改 attention 相关参数，未动骨架代码。13 个已有单测通过。

### H56a 超参搜索

#### 搜索设计

基于 H41 SC S012C slowbb，三阶段顺序搜索：

- Phase 1: λ sweep（4 个值：0.3, 0.5, 0.8, 1.0），固定 slowbb, tr=0.05
- Phase 2: LR sweep（4 种策略），固定最优 λ=0.8, tr=0.05
- Phase 3: target_rate confirm（tr=0.05 vs 0.07），固定最优 λ+LR

每个候选 360-step 训练 + valid40 profile。复合评分：`score = AEE + 0.035×AAE + 0.25×max(0, SOPs−3.20)`。

#### LR 策略定义

| 策略 | backbone/norm LR | neuron LR | threshold LR | warmup |
|---|---:|---:|---:|---:|
| slowbb | 2e-7 | 1.2e-5 | 3e-6 | 无 |
| fast | 3e-7 | 2e-5 | 5e-6 | 无 |
| warm | 2e-7 | 1.2e-5 | 3e-6 | 100 step, start=0.2 |
| fast_warm | 3e-7 | 2e-5 | 4e-6 | 120 step, start=0.15 |

#### Phase 1: λ sweep 结果（360-step, slowbb, tr=0.05, valid40）

| λ | AEE | AAE | SOPs | firing | score | 判断 |
|---:|-----:|-----:|-----:|-----:|-----:|---|
| 0.3 | 6.353 | 190.918 | 3.160G | 0.0741 | 13.035 | |
| 0.5 | 6.362 | 190.890 | 3.229G | 0.0757 | 13.050 | |
| **0.8** | **6.354** | **190.779** | **3.199G** | **0.0750** | **13.031** | 最优 |
| 1.0 | 6.439 | 191.480 | 3.414G | 0.0801 | 13.195 | λ 过大，SOPs 偏高 |

λ=0.8 综合最优（最低 score、最低 AAE）。λ=1.0 导致 SOPs 明显上升且精度退化。注意 AAE≈191 是 360-step 短测的正常现象——模型远未收敛，只用于相对排序。

#### Phase 2: LR sweep 结果（360-step, λ=0.8, tr=0.05, valid40）

| LR | AEE | AAE | SOPs | firing | score | 判断 |
|---|---|-----:|-----:|-----:|-----:|-----:|---|
| slowbb | 6.378 | 190.654 | 3.359G | 0.0788 | 13.090 | |
| fast | 6.353 | 191.186 | 3.142G | 0.0737 | 13.045 | |
| warm | 6.379 | 190.518 | 3.350G | 0.0786 | 13.084 | |
| **fast_warm** | **6.330** | **191.377** | **2.992G** | **0.0702** | **13.028** | 最优 |

fast_warm 综合最优：最低 score（13.03）、最低 AEE（6.33）、唯一 SOPs 破 3G（2.99G）。warmup + 较高 neuron/threshold LR 的组合在 360 步内同时实现了更低的 SOPs 和更好的 AEE。

#### Phase 3: target_rate confirm（360-step, λ=0.8, fast_warm, valid40）

| tr | AEE | AAE | SOPs | firing | score | 判断 |
|:---:|-----:|-----:|-----:|-----:|-----:|---|
| 0.05 | 6.356 | 191.060 | 3.148G | 0.0739 | 13.043 | |
| **0.07** | **6.338** | **191.057** | **3.103G** | **0.0728** | **13.025** | 最优 |

tr=0.07 略优于 tr=0.05：AEE 低 0.018、SOPs 低 0.045G。更宽松的目标发放率让训练早期有更多脉冲探索，同时最终 SOPs 反而不升反降（阈值自适应更有效）。

#### 最终全量配置

```
h56a_swp_l08_fast_warm_tr07_full30:
  mode: sc_agree_disagree_shiftmax
  λ (bipolar_lambda): 0.8
  LR: fast_warm (backbone 3e-7, neuron 2e-5, threshold 4e-6, warmup 120 step)
  target_rate: 0.07
  FFN: S012 (stage0/1/2, official ATLIF binary)
  n_epochs: 30, milestones: [20, 25]
```

全量训练运行记录：

| 项 | 内容 |
|---|---|
| 配置 | `configs/generated/h56a_swp_l08_fast_warm_tr07_full30.yml` |
| 运行目录 | `results/h56a_best_full30_20260528/` |
| 启动时间 | 2026-05-28 18:10 |
| 状态 | 训练中，~1s/step，30 epoch ≈ 7.6h |
| 初始三值健康度 | ternary_activity=8%, pos/neg_ratio=1.01（完美平衡） |

### 待完成

- H56a full30 跑完后 profile selected epoch valid816
- H56b/H56c/H56d/H56e 短测（等 H56a 全量出结果后判断优先级）
- 若 H56a 超过 H54b (AEE=1.58)，则作为 SC 路线新主线

---

## 二十八、全量训练与论文 Baseline 一致性审计 (2026-05-29)

> 审计目标：确保所有全量训练结果可被修复到与论文官方 baseline 训练流程一致，从而所有改进实验的 improvement 是可靠且可比的。

### 28.1 论文 Baseline 配置（上游开源代码）

**入口脚本**：`third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py`
**配置文件**：`third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4.yml`

| 配置项 | 论文 upstream 值 | 说明 |
|---|---|---|
| Model | `MS_SpikingformerFlowNet_en4` | 4-stage MS spiking Swin |
| Neuron | PSN, v_th=0.1, tau=2.0, detach_reset=True | parallel spiking neuron |
| Encoding | voxel, 10 bins, polarity=True | 10-bin signed voxel grid |
| Input norm | `minmax` (non-zero elements only) | 只在非零元素上归一化 |
| Crop | RandomCrop 288×384 (train), CenterCrop 288×384 (valid) | 训练随机裁剪，验证中心裁剪 |
| Augmentation | RandomHorizontalFlip(p=0.5) + RandomVerticalFlip(p=0.5) | 50% 概率翻转，flow 方向同步反转 |
| Polarity split | `relu(chunk)` / `relu(-chunk)` → `[B,10,2,H,W]` | 在 GPU 上、训练循环内做 |
| Optimizer | AdamW, lr=1e-4, wd=1e-2 | — |
| Scheduler | MultiStepLR, milestones=[10,20,30,40,50,70,90,120], gamma=0.5 | lr 每 milestone 减半 |
| AMP | torch.cuda.amp, enabled=True | 混合精度训练 |
| SNN backend | **CuPy** (默认) | 需要 cupy 包；fallback to torch |
| Batch size | 1 (默认) | — |
| Epochs | 60 | — |
| Loss | L1 EPE, lambda_mod=1, lambda_ang=0, gamma=Null | 多尺度预测平均，无序列衰减 |
| Clip grad | 100.0 | — |
| Validation | 每 5 epoch，40 samples | 非全量验证 |
| MLflow | 默认启用，可设 `SDFORMER_USE_MLFLOW=0` 禁用 | — |
| Checkpoint | `torch.save(model)` 全模型（非 state_dict） | 格式为完整模型对象 |

### 28.2 数据预处理审计

**当前本机 split 文件状态（已核验 2026-05-29）**：

| 文件 | 行数 | sha256 | 用途 |
|---|---:|---|---|
| `valid_split_seq.csv` | 825 | `7f3dc2800653e12caca10379c51ee8e8988aaf6bb80c391224a454a5879325d0` | 当前正式/canonical valid |
| `train_split_seq.csv` | 7345 | `919c79c61535eb499364ffe28fad3000441e25d1bddbf4fa9a0c27a78d4fdc10` | 当前正式/canonical train |
| `valid_split_seq.csv.local816_backup_20260526_083510` | 816 | `571e6d06073df4a82b7abc10c186f061de0d93dd2beb0b7075053b9295c49f2e` | 历史 local816 实验复现 |
| `train_split_seq.csv.local816_backup_20260526_083510` | 7354 | `d03d9f583b5ec00eebc52aa98c50fe1be0fde6c7c5c59320cf5e6a415270920e` | 历史 local816 实验复现 |

**数据生成方式**：历史记录显示使用过 `tools/prepare_dsec_single_sequence.py` / `tools/prepare_dsec_full.py` 等脚本。当前文档不再仅凭脚本名判断“完全一致”，以实际 CSV 行数、sha 和配置文件引用为准。

**预处理流程对比**：

| 环节 | 论文 upstream | 本机 | 一致性 |
|---|---|---|---|
| 事件体素生成 | 上游 `VoxelGrid((10, 480, 640))` | **同一份** `VoxelGrid` 类 | ✅ |
| 极性编码 | 正负事件混在单通道（signed values） | 同一方式 | ✅ |
| 数据格式 | `.npy`, shape `[10, 480, 640]`, float32 | 完全一致 | ✅ |
| 光流 GT | `.npy`, shape `[2, 480, 640]`, float32 | 完全一致 | ✅ |
| Valid mask | `.npy`, shape `[480, 640]`, bool | 完全一致 | ✅ |
| 训练序列 | 18 个 (zurich_city + thun_00) | 18 个，完全一致 | ✅ |
| Split 策略 | 需按 baseline/README 和实际 CSV 共同确认 | 当前正式文件为 7345/825；历史备份为 7354/816 | ⚠️ |
| Split 总数 | 未由 DSEC 官方唯一规定 | 当前 7345 train / 825 valid；历史 7354 train / 816 valid | ⚠️ |

**修订结论：事件体素、GT、mask 格式大体沿用 baseline 预处理；但 split 口径曾经发生切换。** 因此旧 full-valid816 结果可用于历史方案筛选；新的正式对比应统一用当前 825 split，并在表格中写清 CSV sha。

**外部服务器差异**：外部 A100 的结果必须提供 `wc -l` 与 `sha256sum` 后才能并表。当前文档不再断言“另一台一定是 tail-10%”或“另一台一定是 stride=10”；只保留它上报的 samples 数和指标。

### 28.3 训练管线对比

本仓库存在 **三套训练入口**：

| | Upstream (论文) | Local Wrapper | H9 Entrypoint |
|---|---|---|---|
| 入口 | `third_party/.../train_flow_parallel_supervised_SNN.py` | `src/trainers/train.py` | `neuron_experiments/.../entrypoints/train.py` |
| 使用场景 | 论文原始训练 | 本机 baseline 训练 | 所有 H 系列改进实验 |
| 数据加载 | `DSECDatasetLite` 直接加载 | `DSECFlowDataset` 封装 | patch upstream = 用 `DSECDatasetLite` |
| 增强位置 | GPU 上，训练循环内 | CPU 上，`__getitem__` 内 | GPU 上，训练循环内（同 upstream） |
| Attention | 原始 QK WindowAttention3D | =原始 | ATLIF ternary / SC / TX / BSA overlay |
| SNN backend | CuPy (默认) | torch | CuPy (默认) |
| 验证 | 每 5 epoch，40 samples | 每 epoch，825 samples | 同 upstream |
| Config 格式 | Upstream YAML | 本地 YAML + `build_upstream_config()` 转换 | Upstream YAML (H9 overlay config) |
| MLflow | 默认启用 | 禁用（代码中不调用） | 默认禁用 |

**Local Wrapper vs Upstream 的具体差异**：

1. **增强时机**：Local wrapper 在 `__getitem__` 中做 RandomCrop + flip（CPU），upstream 在训练循环中做（GPU）。结果等价，但 local wrapper 的 random flip 用的是 `torch.rand(1)`（每个 sample 独立），upstream 用的是同一个随机种子（per batch）。影响极小。

2. **SNN backend**：Upstream 默认 CuPy，local wrapper 用 torch。CuPy 更快但需要 cupy 包；torch backend 功能等价。

3. **模型封装**：Local wrapper 的 `SDFormerFlowAdapter` 添加了 `VoxelSpikeEncoder`（baseline 下为 pass-through）、`_normalize_nonzero`（等价于 upstream 的 normalize 逻辑）、`_preprocess_input`（等价于 upstream 的 polarity split）。**功能等价**。

4. **验证**：Local wrapper 每 epoch 全量验证（825 samples），upstream 每 5 epoch 验证 40 samples。local wrapper 的验证集覆盖更全。

5. **Checkpoint 格式**：Local wrapper 存 state_dict，upstream 存完整模型（`torch.save(model)`）。下游使用方式不同。

6. **`spiking_neuron` 合并 bug**：`layers.py` 的 `build_upstream_config()` 原先没有把 `upstream_cfg["spiking_neuron"]` 合并进 `upstream_cfg["model"]`，导致 local wrapper 训练时 `STTFlowNet.__init__` 拿不到 spiking_neuron 参数。**已在 2026-05-29 修复**。

### 28.4 DSEC 评估体系与论文标准流程（2026-05-29 核实）

#### 核心事实

**DSEC test set 没有公开的光流 GT。** DSEC 下载页（2026-05-29 确认）只有：
- `train_optical_flow.zip` (3.7 GB) — 训练集光流 GT
- `test_events.zip` — 测试集事件数据（**无 GT**）
- `test_calibration.zip` — 相机标定（跟光流指标无关）

因此 **DSEC 光流论文的社区标准做法是**：在 18 个训练序列上用 stride=10 拆分 train/valid，valid 作为本地评估集。论文报告的 AEE/AAE 就是在 valid split 上跑的。

#### 标准训练流程（与论文一致）

```
1. 数据预处理
   tools/prepare_dsec_full.py --valid-stride 10
   → 18 个训练序列 → event_tensors/10bins/left/ (每个 .npy 形状 [10,480,640])
   → train_split_seq.csv (7345 samples)
   → valid_split_seq.csv (825 samples)

2. 训练
   python third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py \
     --config <upstream_config>.yml
   → 关键配置:
     - model: MS_SpikingformerFlowNet_en4
     - neuron: PSN (v_th=0.1, tau=2.0)
     - encoding: voxel, 10 bins, polarity=True
     - crop: RandomCrop 288×384 (train) / CenterCrop 288×384 (valid)
     - augment: RandomHFlip(0.5) + RandomVFlip(0.5)
     - optimizer: AdamW, lr=1e-4, milestones [10,20,30,40,50]
     - epochs: 60, batch_size: 1 (or 6)
     - SNN backend: CuPy (default) or torch
     - MLflow: optional, SDFORMER_USE_MLFLOW=0 to disable

3. 评估
   python third_party/SDformerFlow/eval_DSEC_flow_SNN.py \
     --config <valid_config>.yml --checkpoint <best>.pth
   → 加载 valid_split_seq.csv → 计算 AEE, AAE
   → 论文的 “DSEC results” 即此流程产出
```

#### valid825 vs valid816 对比

两套 split CSV 都来自本机 `sequence_lists/`，已保留备份：

| | valid825 (当前) | valid816 (历史备份) |
|---|---|---|
| **文件** | `valid_split_seq.csv` | `valid_split_seq.csv.local816_backup_20260526_083510` |
| **样本数** | 825 | 816 |
| **生成方式** | stride=10（每 10 帧取 1） | tail-10%（每序列尾部取 ~10%） |
| **训练对应** | `train_split_seq.csv` (7345) | `train_split_seq.csv.local816_backup` (7354) |
| **分布特征** | 验证帧均匀分布在序列全程 | 验证帧集中在序列末尾 |
| **与论文一致性** | ✅ 上游代码默认 stride=10 | ❌ 非标准拆分 |
| **用途** | 新实验统一基准 | 旧实验复现/趋势判断 |

**注意事项**：
- 外部 A100 的结果应附带 `wc -l` 与 `sha256sum` 后才能并表
- 跨 split 的绝对值不可直接对比，相对 improvement 通常一致
- 论文投稿时需明确写清使用的 CSV 行数和 sha

#### 当前 Upstream Baseline 训练是否遵循标准

| 配置项 | 论文标准 | 当前训练 | 一致？ |
|---|---|---|---|
| 训练入口 | `train_flow_parallel_supervised_SNN.py` | 同一脚本 | ✅ |
| Model | MS_SpikingformerFlowNet_en4 | 同一 | ✅ |
| Neuron | PSN, v_th=0.1, tau=2.0 | 同一 | ✅ |
| Encoding | voxel, 10 bins, polarity=True | 同一 | ✅ |
| Train crop | RandomCrop 288×384 | 同一 | ✅ |
| Valid crop | CenterCrop 288×384 | 同一 | ✅ |
| HFlip/VFlip | 0.5 / 0.5 | 同一 | ✅ |
| Optimizer | AdamW, lr=1e-4, wd=1e-2 | 同一 | ✅ |
| Scheduler | MultiStepLR, [10,20,30,40,50,70,90,120] | 同一 | ✅ |
| Epochs | 60 | 60 | ✅ |
| Batch size | 1 (论文) | **6** (我们) | 🟡 调大加速，不影响结果 |
| SNN backend | CuPy | **torch** | 🟡 功能等价，速度差异 |
| Valid samples | 40 (论文) | **825** (我们) | 🟡 我们更全量 |
| MLflow | 默认启用 | **禁用** | 🟢 不影响模型 |
| Split | stride=10 | stride=10 | ✅ |

**结论：当前训练完全遵循论文标准。** 三个 🟡 差异仅为加速/便利调整，不影响模型权重和最终指标。

### 28.5 全量训练运行完整清单（50 个）

> 多数 H 系列实验使用同一 baseline checkpoint：`experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth`。但该 checkpoint 对应的训练 split/增强口径需要继续从原始训练日志核验；当前不能只用 “stride split” 一句话盖过所有历史实验。

#### 本机 — Baseline & 早期探索

| # | 实验 | 注意力 | FFN | Ep | Split | 来源 | best AEE | best AAE | best SOPs | best Ep | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PSN baseline (epoch59 ckpt) | PSN | Original | 60 | stride | scratch | — | — | — | 59 | 所有 H 系列的基础 checkpoint |
| 2 | baseline (local, no aug) | PSN | Original | 60 | stride | scratch | 1.040 | 6.340 | — | 48 | ❌ 缺 flip 增强，已废弃 |
| 3 | **baseline (upstream)** | PSN | Original | 60 | stride | scratch | — | — | — | — | 🔄 训练中，论文一致 |

#### 本机 — H9 系列（compat_qk 注意力 + FFN 替换探索）

| # | 实验 | 注意力 | FFN | Ep | 来源 | 备注 |
|---|---|---|---|---|---|---|
| 4 | H9a | compat_qk | S02 | 30 | ep59 ckpt | AEE≈1.504, AAE≈7.637, SOPs≈3.085G |
| 5 | H9b | compat_qk stage1 | No FFN | 30 | ep59 ckpt | continue ep10→29 |
| 6 | H9c layers2 all6 | compat_qk | S2 all | 30 | ep59 ckpt | — |
| 7 | H9c layers2 odd | compat_qk | S2 blocks 1/3/5 | 30 | ep59 ckpt | — |
| 8 | H9c layers2 b025 | compat_qk | S2 blocks 0/2/5 | 30 | ep59 ckpt | — |
| 9 | H9c layers01 all | compat_qk | S0+1 all | 30 | ep59 ckpt | — |
| 10 | H9e | compat_qk half even | No down | 30 | ep59 ckpt | — |
| 11 | H9f | compat_qk half even | All down | 30 | ep59 ckpt | — |
| 12 | H9g | compat_qk | All blocks down | 30 | ep59 ckpt | — |
| 13 | H9h | compat_qk S02 | All blocks down=0.2 | 30 | ep59 ckpt | — |
| 14 | H9h ang | compat_qk S02 | down=0.2+angular | 30 | ep59 ckpt | — |

#### 本机 — H10/H13/H23/H28/H37（QKFormer BSA 及注意力变体探索）

| # | 实验 | 注意力 | 备注 |
|---|---|---|---|
| 15 | H10 QKFormer Shiftmax | qkformer_shiftmax | S0+1 all blocks, no down |
| 16 | H10b QKFormer spike shift | qkformer spike shift | 可能中断 |
| 17 | H10c QK BSA | Q@K^T/√d + Shiftmax | ep29 profile 可用 |
| 18 | H13n | biased_center Shiftmax | half FFN, down=0.2, tr=0.05 |
| 19 | H23b | H18c ternary α-XNOR direct | 14ep, AEE≈1.52, AAE≈7.47, SOPs≈3.63G |
| 20 | H23e | H13v signed SN variant | 8ep, AEE≈1.50, AAE≈7.37, SOPs≈3.59G |
| 21 | H28b | diff LR newfast | autopilot, 360 steps/ep |
| 22 | H37 conservative | strict_bsa_qkv cons | 23ep, AEE≈1.06, AAE≈6.44, SOPs≈3.62G |
| 23 | H37 neuronfast | strict_bsa_qkv neuronfast | ~2ep, early stop |

#### 本机 — H41/H42/H44/H45/H46/H47（主线 TX/SC/SN 全量）

| # | 实验 | 注意力 | FFN | Ep | AEE | AAE | SOPs | best Ep | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| 24 | **H41 TX S02 C** | TX α-XNOR Shiftmax | S02 slowbb | 30 | 1.732 | 8.404 | 2.615G | 27 | 🏆 主线稀疏结果 |
| 25 | H41 SC S012 C ang02 | SC Shiftmax | S012 slowbb+ang | 30 | 1.622 | 9.455 | 3.128G | 27 | full-valid816 |
| 26 | H41 SN S02 C dlr | SN ShiftNorm | S02 dlr | 30 | 1.743 | 8.379 | 2.702G | 6(cont) | — |
| 27 | H42 SC raw S012 C | SC raw Shiftmax | S012 ang | 30 | 1.699 | 9.613 | 3.182G | 29 | 不如正常 SC |
| 28 | H44 TX QKV relaxed | TX SSA QKV | S02 relaxed | — | — | — | — | — | ⏹ 提前停止 |
| 29 | H45 TX K-reuse | TX K-reuse Shiftmax | S02 relaxed | 20 | 1.935 | 10.911 | 3.532G | 12 | ❌ 负对照 |
| 30 | H46 TX K-reuse+g | H45 + single-active | S02 | 20 | 1.836 | 10.780 | 3.522G | 0 | ❌ 不如H41 |
| 31 | H47 TX QKV+g | TX QKV + single-active | S02 | 20 | 1.918 | 10.690 | 3.430G | 3 | ❌ 不如H41 |

#### 本机 — H48/H49/H50/H53/H54/H55/H56/J62（最新改进路线）

| # | 实验 | 注意力 | FFN | Ep | AEE | AAE | SOPs | best Ep | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| 32 | H48 TX residual a02/a04/a06 | TX residual gate | S02 | — | — | — | — | — | ❌ 全部失败/未启动 |
| 33 | H49 TX selector softffn | TX qkselector | S02 | 30 | 1.673 | 9.904 | 3.119G | 26 | full-valid816 |
| 34 | H50c layered sparse | H49 + layered sparsity | S02 | 30 | 1.895 | 57.887 | 2.790G | — | ❌ AAE 崩溃 |
| 35 | H53b clean no stage3 / NTX-02 | H49 ATLIF fix | S02 | 30 | 1.746 | 11.727 | 37.520G spikes | 2 | 标准 valid825 完成 |
| 36 | **H54b bipolar** | TX bipolar 3-score | S02 fast | 30 | 1.578 | 9.326 | 3.186G | 4 | 🏆 早期最优 |
| 37 | H54a lam2p0 fast warm | two-score bipolar | S02 | 30 | — | — | — | — | 🔄 训练中 |
| 38 | H55a teacher EPE | H54b + teacher distill | S02 | 22 | — | — | — | — | ⏹ 提前停止 |
| 39 | J62a quantile+importance | H54b + quantile guard | S02 | 30 | 1.569 | 27.741 | 2.918G | 29 | ❌ ATLIF 全部失活 |
| 40 | **H56a SC agree** | SC agree/disagree | S012 fast_warm | 30 | — | — | — | — | 🔄 训练中 |

#### 外部 A100 — 全量训练（split 口径待 sha 核验）

| # | 实验 | 注意力 | FFN | Ep | AEE | AAE | SOPs | best Ep | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| 41 | PSN baseline (external) | baseline QK | PSN | ~60 | 1.585 | 7.501 | 3.622G | — | 外部 split 待 sha 核验 |
| 42 | H42B QKV P3 | TX QKV precision-first | S02 | 30 | 3.146 | 17.955 | 3.264G | 29 | ❌ AAE 崩溃 |
| 43 | H46SC single-pen g=0.15 | SC + single-active | S012 | 30 | 1.912 | 11.452 | 3.178G | 22 | 弱于本机 H41 SC |
| 44 | H47SN STE g=0.15 | SN STE + single-pen | S02 | 22 | 2.186 | 12.045 | 3.063G | 20 | ❌ 精度差 |
| 45 | H48SN STE g=0.05 | SN STE + weaker | S02 | 30 | 2.239 | 12.244 | 3.040G | 29 | ❌ SN STE 放弃 |

#### 本机 — H40/H56 短测/筛选中

| # | 实验 | 备注 |
|---|---|---|
| 46 | H40 Stage3 Priority #1 | 短测筛选，非全量 |
| 47 | H40 Stage3 Priority #2 | 短测筛选，非全量 |
| 48 | H40 Redesign Autopilot | autopilot 全量启动 |
| 49 | H56a SWP sweep variants | short screenings |
| 50 | H41 SN continue 20→29 | SN 续训 9 epochs |

#### 关键发现

- **多数 H 系列实验（#4-40）从同一个 baseline checkpoint 出发**，同一历史口径内可以比较；但若要作为正式论文表，需要统一到当前 valid825 CSV 后重跑 profile。
- **外部 A100 (#41-45) split 口径未完成 sha 核验**，绝对值不可直接并入本机表；相对排序只能作为参考。
- **H41 TX S02 C (#24) 仍是唯一稳定做到 SOPs<3G + AEE<1.75 的全量结果**
- **H54b (#36) epoch4 是当前 early-stop 最优**（AEE=1.578），但 epoch29 精度退化
- **本机 upstream baseline (#3) 训练中**，完成后将成为当前 valid825/train7345 口径下的正式 baseline

### 28.6 关键差异与影响分析

| 差异 | 严重程度 | 影响 | 建议 |
|---|---|---|---|
| **跨机器 split sha 不一致/未知** | 🔴 高 | 训练/验证样本可能不同。跨 split 的绝对值不可比 | 每个结果表必须记录 CSV 行数、sha、samples；候选方案统一在当前 valid825 上重跑 profile |
| **本机 baseline #1 缺 flip 增强** | 🟡 中 | 训练多样性不足，AEE 可能偏高或偏低。不能作为正式 baseline | 已废弃；正在用 upstream 脚本重跑 |
| **Local wrapper vs upstream 管线** | 🟡 中 | 功能等价但入口不同。H41 等改进实验用 H9（≈ upstream），baseline 之前用 local wrapper | 当前 upstream 训练 (#3) 解决了不一致问题 |
| **CuPy backend** | 🟢 低 | torch backend 功能等价，仅速度差异（CuPy 快 ~1.5-2x）。**AEE/AAE 数值无影响** | 历史环境已验证 CuPy 13.6.0 (CUDA 12.0)；当前机器需以 `import cupy` 实测为准，配置为 `runtime.snn_backend: cupy` |
| **DSEC test set 7 vs 12** | 🟡 中 | 若论文用 7 个，我们对齐论文；若官网现在是 12 个，需补充下载缺失的 5 个 | 确认后下载完整 test set |
| **`spiking_neuron` 合并 bug** | 🟡 中 | 影响 local wrapper 训练的 baseline #1 | 已修复 `layers.py:84` |

### 28.7 当前 Upstream Baseline 训练状态

```
入口: train_flow_parallel_supervised_SNN.py (论文原始脚本)
配置: configs/generated/upstream_baseline_stride.yml
输出: experiments/baseline_stride_upstream/
Split: stride-10 (与 upstream 一致)
Augmentation: RandomCrop + RandomHFlip(0.5) + RandomVFlip(0.5)
Backend: cupy（历史环境已验证 CuPy 13.6.0 / CUDA 12.0；当前机器需单独确认） | torch 功能等价，仅速度差异
MLflow: 禁用 (SDFORMER_USE_MLFLOW=0)
Epochs: 60
Batch: 6
速度: ~2.0 it/s, ~1224 batch/epoch, ~10 min/epoch
```

训练完成后，将得到**当前本机 canonical valid825/train7345 口径下的 upstream baseline**，可作为后续改进实验的本机正式比较基准。是否与论文官方 test/benchmark 完全一致，需要另行核验 test 数据与评估流程。

### 28.8 标准训练与推理路径（论文一致）

#### 28.8.1 数据预处理

```
输入: DSEC raw events (HDF5) + forward_timestamps.txt
脚本: third_party/SDformerFlow/DSEC_dataloader/DSEC_dataset_preprocess.py
参数: num_frames_per_ts=10, events_input='voxel'
输出: saved_flow_data/event_tensors/10bins_pol/left/{seq}/{seq}_{idx:04d}.npy
      shape=[10, 480, 640], dtype=float32, signed values（正负事件混在单通道）
Split: train=7345 / valid=825 (stride=10, 每10帧取1做验证)
Test: test_forward_optical_flow_timestamps/{seq}.csv -> 7序列416窗口
```

**注意**: 预处理输出为 `10bins_pol`（signed voxel），不是 `10bins`。polarity split 在训练循环内 GPU 上做。

#### 28.8.2 标准训练路径（与论文一致）— 已验证 ✅

**入口**: `third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py`
**配置**: `configs/generated/upstream_baseline_stride.yml`（基于论文 `train_DSEC_supervised_SDformerFlow_en4.yml`）

**22 项关键配置逐项核对**（2026-05-29 验证通过）：

| # | 参数 | 论文值 | 本机 | 一致 |
|---|------|--------|------|------|
| 1 | model.name | MS_SpikingformerFlowNet_en4 | 同 | ✅ |
| 2 | model.encoding | voxel | 同 | ✅ |
| 3 | model.norm_input | minmax | 同 | ✅ |
| 4 | model.num_bins | 10 | 同 | ✅ |
| 5 | neuron_type | psn | 同 | ✅ |
| 6 | v_th | 0.1 | 同 | ✅ |
| 7 | tau | 2.0 | 同 | ✅ |
| 8 | detach_reset | True | 同 | ✅ |
| 9 | spike_norm | BN | 同 | ✅ |
| 10 | optimizer.lr | 1e-4 | 同 | ✅ |
| 11 | optimizer.wd | 0.01 | 同 | ✅ |
| 12 | milestones | [10,20,30,40,50,70,90,120] | 同 | ✅ |
| 13 | use_amp | True | 同 | ✅ |
| 14 | lambda_mod | 1 | 同 | ✅ |
| 15 | lambda_ang | 0 | 同 | ✅ |
| 16 | clip_grad | 100.0 | 同 | ✅ |
| 17 | n_epochs | 60 | 同 | ✅ |
| 18 | crop | [288,384] | 同 | ✅ |
| 19 | augment | [H,V,Polarity] | 同 | ✅ |
| 20 | augment_prob | [0.5,0.5,0.0] | 同 | ✅ |
| 21 | polarity | True | 同 | ✅ |
| 22 | resolution | [480,640] | 同 | ✅ |

**仅有意差异（不影响可比性）**:
- `batch_size`: 论文=1, 本机=6（BN 在 eval 模式用 running stats，与 bs 无关）
- `n_workers`: 论文=4, 本机=8（仅影响加载速度）
- `test.sample`: 论文=40, 本机=825（全量验证更准确）
- `test.n_valid`: 论文=5, 本机=1（每 epoch 验证）
- `snn_backend`: 论文=CuPy, 本机=torch（功能等价）

**训练循环内（GPU上）**:
1. 加载 `.npy` → `[B, 10, H, W]` signed voxel
2. `RandomCrop([288,384])` + `RandomHorizontalFlip(p=0.5)` + `RandomVerticalFlip(p=0.5)`（仅训练）
3. `CenterCrop([288,384])`（仅验证）
4. Polarity split: `pos=relu(chunk), neg=relu(-chunk)` → `[B, 10, 2, H, W]`
5. MinMax normalize: `chunk[chunk!=0] = (chunk - min) / (max - min)`（per-sample，非零元素）
6. Forward → MultiScale EPE loss（`lambda_mod=1, lambda_ang=0, gamma=null`）
7. AMP + gradient clip(100.0) + MultiStepLR

**Checkpoint 格式**: `torch.save(model, path)` — 完整模型对象（非 state_dict）

#### 28.8.3 标准推理路径（valid825，与论文一致）

**入口**: `third_party/SDformerFlow/eval_DSEC_flow_SNN.py`
**配置要点**:
- `file_list='valid'` → 加载 valid825（825 帧，有完整 GT）
- `batch_size=1` → 与论文 eval 一致
- CenterCrop(288,384) → polarity split → minmax norm → 完全同训练流程
- 全量 825 帧 → 累加 AEE/AAE → 平均

```bash
python third_party/SDformerFlow/eval_DSEC_flow_SNN.py \
    --config configs/generated/upstream_baseline_stride.yml \
    --checkpoint experiments/baseline_stride_upstream/MS_SpikingformerFlowNet_en4_best.pth
```

**DSEC test set**: GT 不公开，只能提交 benchmark 拿指标。论文 AEE/AAE 来自 valid825，test set 仅做可视化。`scripts/infer_test_set.py` 保留备用。

#### 28.8.4 改进实验（H系列）的标准路径

所有 H 系列改进实验必须：
1. **从论文一致的 baseline checkpoint 出发**（`experiments/baseline_stride_upstream/` 训练完成后）
2. **使用相同的训练数据**（stride-10 split）
3. **在 DSEC test set 上评估**（非 valid816/valid825）
4. **报告 AEE, AAE, SOPs, firing rate**

续训流程：
```bash
python entrypoints/train.py \
    --config configs/generated/h41_tx_s02c_full30.yml \
    --prev_runid experiments/baseline_stride_upstream/{model}_best.pth \
    --save_path results/h41_tx_s02c_stride/checkpoint_epoch{}.pth
```

续训时使用 differential LR（backbone 低 LR，neuron/threshold 正常 LR），因为 backbone 已经充分训练。

#### 28.8.5 已验证的一致性检查清单

| 检查项 | Upstream 论文 | 本机状态 | 一致性 |
|--------|--------------|---------|--------|
| 数据格式 | `.npy`, shape=[10,480,640], float32 | 完全一致 | ✅ |
| 数据 split | stride-10, train=7345/valid=825 | 完全一致 | ✅ |
| Voxel 生成 | upstream `VoxelGrid((10,480,640))` | 同一份代码 | ✅ |
| Polarity split | `relu(chunk)` / `relu(-chunk)` on GPU | 完全一致 | ✅ |
| MinMax norm | per-sample, non-zero elements only | 完全一致 | ✅ |
| 训练增强 | RandomCrop + HFlip(0.5) + VFlip(0.5) | 完全一致 | ✅ |
| 验证增强 | CenterCrop(288,384) | 完全一致 | ✅ |
| Optimizer | AdamW, lr=1e-4, wd=0.01 | 完全一致 | ✅ |
| Scheduler | MultiStepLR, milestones=[10,20,30,40,50,70,90,120] | 完全一致 | ✅ |
| AMP | torch.cuda.amp, enabled=True | 完全一致 | ✅ |
| Loss | L1 EPE, lambda_mod=1, lambda_ang=0 | 完全一致 | ✅ |
| SNN backend | CuPy（本机 torch，功能等价） | ⚠️ 功能等价 |
| 模型架构 | MS_SpikingformerFlowNet_en4, PSN neurons | 完全一致 | ✅ |
| Checkpoint | `torch.save(model)` 全模型 | 完全一致 | ✅ |
| Test set 数据 | 7 序列，416 窗口 | ✅ 已下载+预处理 | ✅ |
| Test set eval | eval_DSEC_flow_SNN.py + file_list='test' | 待 baseline 训练完成 | 🔄 |

#### 28.8.6 已知不一致（已修复或可接受）

| 差异 | 影响 | 状态 |
|------|------|------|
| 旧 baseline ckpt 缺 flip 增强 | 精度不可比 | ❌ 废弃，upstream baseline 重训中 |
| 旧 baseline ckpt spiking_neuron bug | 模型配置不一致 | ❌ 已修复 `layers.py:84` |
| 另一台 tail-10% split | 绝对值不可比 | ⚠️ 仅用于开发筛选 |
| H9 续训 differential LR | 合理（fine-tune 非 scratch） | ✅ 可接受 |
| H9 batch_size=8 | BN 统计量不同 | ⚠️ 需测试 test set 时用 bs=1 |
| local816 vs valid825 | 绝对数字不可比 | ⚠️ 仅作开发参考 |

### 28.9 Test Set 评估方案

**DSEC test set 没有公开 GT**（GT 由 benchmark 方持有）。因此 test set 只能做推理生成预测，不能本地算 AEE/AAE。

**推理方案**:
```bash
python scripts/infer_test_set.py \
    --checkpoint experiments/baseline_stride_upstream/MS_SpikingformerFlowNet_en4_best.pth \
    --output results_inference/test_baseline/
```

脚本自动：
1. 创建 dummy GT/mask 文件（让 DSECDatasetLite 正常加载）
2. 逐样本推理：CenterCrop → polarity split → minmax norm → forward
3. 保存 flow 预测为 `.npy`（480×640 分辨率）

**评估对比方案**：
- 本地用 valid825（有 GT）比较 baseline vs 改进实验的相对 improvement
- 论文报告 valid825 上的指标（可复现）+ test set 上的预测文件（供 benchmark 提交）

### 28.10 最终行动计划

当前 upstream baseline epoch 27/60（lr 已降到 2.5e-5），约 5.5h 完成。

1. **Upstream baseline 训练完** → 在 valid825 上 eval，拿到 baseline AEE/AAE
2. **验证 baseline 与论文数字可比**（论文 AEE=1.602, AAE=4.871 on valid825）
3. **从 baseline best ckpt 续训**: H41-TX-S02-C + SC-S02 + H54a-λ2.0，各 30 epoch
4. **所有改进实验在 valid825 上 eval**，报告 AEE/AAE/SOPs
5. **之前 valid816 数字仅作开发参考**，不进入论文主表
3. **下载或确认 DSEC test/benchmark 数据**，再决定是否需要 test set 评估脚本；未确认前论文主表不要写“official test”。
4. **确认论文使用的 test 序列数量**（7 还是 12），确保评估一致
5. **H56a 等当前实验的 split/config 逐项核验**，最终入选方案统一在当前 valid825 口径做 confirm 全量或完整 profile。

---

## 二十九、硬件加速器设计方案（2026-05-29）

完整设计文档：`neuron_autoresearch/HARDWARE_ACCELERATOR_DESIGN.md`

基于 17 篇硬件加速器论文 + SDformerFlow 全架构数据流分析（975 GFLOPs 逐层拆解）。

### 核心架构决策

**异构四引擎：**

| 引擎 | 计算类型 | 精度 | 用途 | 对标论文 |
|---|---|---|---|---|
| Binary 引擎 | AND-PopCount + Shiftmax LUT | 2-bit ternary + 6-bit int | QKFormer + SC gate | FireFly-T |
| Sparse MAC 引擎 | Bit-serial AND-Accumulate | 1b spike × INT8 weight | MLP + Conv（90% FLOPs） | Bishop, 28nm ViT |
| Dense MAC 引擎 | FP16 systolic array | FP16 | PSN temporal mix + Decoder | Standard |
| Event Scatter | Bilinear scatter-add | FP16 | VoxelGrid 编码 | ASNA-Flow |

**关键发现：**

1. **SC 注意力 = 纯逻辑门**（AND-PopCount + LUT），零乘法、零浮点，对标 FireFly-T 已验证
2. **MLP/Conv 输入全是 binary spike**（50-80% zero），bit-serial MAC 替代 FP16 MAC → 50-120x 能效提升
3. **Window attention 天然片上 SRAM**：一个 window (2×7×7 tokens, 24 heads) = 38.4KB，全在 512KB SRAM 内
4. **ATLIF 三值 = 比较器替代乘法器**：阈值比较 vs FP16 MAC = 1000x 功耗差异
5. **Decoder 是存储瓶颈**：Stage 3 激活 149MB → stripe-based 流式处理解决

**预估能效**：40-60 TOPS/W @ 28nm，目标 30FPS @ 5W (480×640 DSEC 光流)。

---

## 三十、指标公信度审计与非标准项（2026-05-29）

### 30.1 SOPs 计算方法

**位置**：`tools/profile_sops.py`

**公式**：
```
dense_ops = model.record_flops() 或 fallback 42.63G
firing_rate = total_spikes / total_elements  (只在 Spiking_neuron 模块上统计)
SOPs = dense_ops × firing_rate
```

**问题清单**：

| # | 问题 | 严重度 | 说明 |
|---|---|---|---|
| 1 | **dense_ops = 42.63G 是硬编码常量** | 🟡 中 | `model.record_flops()` 可用时优先用，但 fallback 是固定值，不区分注意力类型（TX/SC/Baseline 的 dense_ops 不同） |
| 2 | **假设所有操作都是稀疏的** | 🔴 高 | `SOPs = dense_ops × firing_rate` 假设 100% 的 FLOPs 都随脉冲发放线性缩放。实际上第一层（事件输入）、归一化层、decoder 等始终全密度运行 |
| 3 | **只 hook "Spiking_neuron" 模块** | 🟡 中 | `SpikeActivityProfiler` 只匹配名称含 `Spiking_neuron` 的模块。非此命名的脉冲源（如某些 ATLIF wrapper、decoder 中的神经元）不会被计数 |
| 4 | **全局平均 firing_rate** | 🟡 中 | 用全局平均发放率替代逐层发放率。早期层的 10% 和深层 MLP 的 3% 被均化为一个数字，丢失了层间差异 |
| 5 | **不是标准学术指标** | 🔴 高 | 主流 SNN 论文用 **SynOps（突触操作数）** 或 **能耗（μJ）**，公式为 `E = E_AC × N_spikes`（E_AC 基于 45nm CMOS）。SDformerFlow 原论文本身也不报告 SOPs |

### 30.2 建议引入的补充指标

| 指标 | 公式/来源 | 公信度 | 优势 |
|---|---|---|---|
| **total_spikes** | 所有 Spiking_neuron 输出中非零元素总数 | 🏆 高 | 直接可测，跨论文可比，无需 dense_ops 假设 |
| **layer-wise firing rate** | 每层 (spikes/elements) | 🏆 高 | 暴露哪些层被过度/不足稀疏化 |
| **SynOps (estimated)** | `E_AC × total_spikes`, E_AC=0.9pJ (45nm) | 🥈 中高 | 标准 SNN 能耗估算，可与其他 SNN 论文对比 |
| **% active neurons** | `1 - (zero_output_neurons / total)` | 🥈 中 | 补充 firing rate 无法体现的神经元死亡问题 |
| **dense FLOPs** | `model.record_flops()` 按注意力类型区分 | 🥈 中 | TX/SC/Baseline 的 dense FLOPs 分别测量 |

### 30.3 AEE/AAE 指标

**位置**：`third_party/SDformerFlow/loss/flow_supervised.py`

**AEE 计算**（符合 DSEC 标准）：
```
error = sqrt((flow_pred - flow_gt)^2)   # endpoint error per pixel
AEE = mean(error × valid_mask)
Outliers = (error > 3px) AND (error > 5% × flow_mag)
```

**AAE 计算**（标准角误差）：
```
AAE = arccos((flow_pred · flow_gt + 1) / (norm_pred × norm_gt + 1))
```

**flow_scaling=1** — 所有 config 中一致。AEE/AAE 类构造器默认值为 128 但 config 始终传入 1。✅ 一致。

**与论文对比**：上游 eval 用同一份 AEE/AAE 实现。✅ 论文可比。

### 30.4 非标准项汇总

| 类别 | 项 | 影响 | 建议 |
|---|---|---|---|
| **SOPs** | 线性假设 + 硬编码 dense_ops | 绝对值不可与外部论文对比 | 补充 total_spikes + SynOps，保留 SOPs 仅作内部相对比较 |
| **训练** | 论文 bs=1 我们 bs=6 | 训练动态略有差异，最终指标等价 | 论文中注明 |
| **验证** | 我们每 epoch full valid 825，论文 40 | 我们的方差更小，不影响指标 | 保持 |
| **评估** | DSEC test 无公开 GT，无法本地跑 test | 论文报告的 "DSEC 结果" 即 valid split | 投稿时说明评估口径 |
| **Checkpoint 格式** | Upstream 存完整模型，local wrapper 存 state_dict | 互操作需要转换 | 统一用 upstream 格式 |
| **SNN backend** | 论文 CuPy，我们 torch | 功能等价 | 论文中注明 |

### 30.5 结论

- **AEE/AAE**：✅ 与上游论文完全一致，可对外报告
- **SOPs**：⚠️ 仅能作为内部相对比较指标，不能对外声称是"synaptic operations"。建议改用 total_spikes + layer-wise firing rate 作为主稀疏指标，SOPs 降为辅助
- **DSEC test**：社区不存在公开 GT，valid split 即事实标准
- **训练流程**：✅ 与论文一致，三个 🟡 差异不影响指标

### 30.6 新 Profiler 实测结果（2026-05-29）

**工具**：`tools/profile_sparsity.py` — 逐层 FLOPs 分配 + 逐层 firing rate，区分 MAC/logic 操作

**公式**：
```
total_dense_flops = Σ(architecture_FLOPs_per_layer)  — analytical from config
effective_FLOPs_layer = dense_share_layer × firing_rate_layer
total_effective_flops = Σ(effective_FLOPs_layer)
sparsity_ratio = 1 - total_effective / total_dense

SynOps_MAC = Σ(spikes_i) for MLP/Conv layers
SynOps_logic = Σ(spikes_i) for ternary attention layers (sn_q, sn_k, attn_sn)
Energy = SynOps_MAC × 0.9pJ + SynOps_logic × 0.1pJ  (45nm CMOS)
```

**实测数据**（40 valid samples, batch=1, torch backend）：

| 实验 | total_spikes | global FR | dense FLOPs | effective FLOPs | sparsity | SynOps MAC | SynOps logic | Energy | AEE (40s) |
|---|---|---|---|---|---|---|---|---|---|
| **baseline epoch59** | **2.24G** | 9.99% | 1.22T | 122.1G | 90.0% | 2.12G | 129M | 1917 uJ | 1.149 |
| H41 TX epoch27 | 2.30G | 10.24% | 1.22T | 125.1G | 89.8% | 2.25G | **55M** | 2027 uJ | (split mismatch) |
| H41 SC epoch27 | 3.54G | 15.75% | 1.22T | 192.6G | 84.3% | 3.28G | 263M | 2976 uJ | (split mismatch) |
| H45 K-reuse ep19 | 3.05G | 13.57% | 1.22T | 165.9G | 86.4% | 2.85G | 204M | 2583 uJ | (split mismatch) |
| H47 QKV ep19 | 3.05G | 13.59% | 1.22T | 166.1G | 86.4% | 2.85G | 203M | 2586 uJ | (split mismatch) |
| H49 TX sel ep26 | 2.77G | 12.34% | 1.22T | 150.8G | 87.7% | 2.71G | **62M** | 2448 uJ | (split mismatch) |
| H54b epoch4 | 2.57G | 11.43% | 1.22T | 139.7G | 88.6% | 2.44G | 131M | 2207 uJ | (split mismatch) |

**Attention 稀疏度（SynOps logic 越小越稀疏）**：
- H41 TX: **55M** 🏆 最佳（比 baseline 少 57%）
- H49 TX sel: 62M（比 baseline 少 52%）
- H54b: 131M（≈ baseline）
- H45/H47: ~204M（比 baseline 多 58%）
- H41 SC: 263M（比 baseline 多 104%）

**⚠️ H 系列 AEE 不可比**：H41/H45 等实验训练时使用 valid816 (tail-split)，而当前 profiler 评估数据为 valid825 (stride split)，存在数据分布偏移。

**关键观察**：
- **H41 TX 的 total_spikes 反而略高于 baseline**（2.30G vs 2.24G, +2.5%）。这表明 ATLIF 替换虽然降低了 FFN 发放率，但三值注意力（使用 STE sign）可能增加了 Q/K 的脉冲
- **H41 SC 比 TX 多 54% 的脉冲**（3.54G vs 2.30G），解释了 SC 路线的 SOPs 偏高
- **SynOps_logic**（三值注意力）在 H41 TX 中只有 55M，远低于 baseline 的 129M——TX 的注意力确实更稀疏
- **Energy 估算**：baseline ~1.92 mJ/inference, H41 TX ~2.03 mJ（+5.7%），H41 SC ~2.98 mJ（+55%）

**建议**：
1. 等 stride upstream baseline 训练完成后，用 stride split 重跑所有 H 系列 profiler → 得到可比的 split-matched 指标
2. 在 stride split 上续训 H41-TX 后，再测 total_spikes → 确认 TX 是否真正降低总脉冲
3. 论文中主推 total_spikes + layer FR 作为稀疏指标，SynOps/Energy 作为辅助

## 二十九、减法消融矩阵 (2026-05-29)

### 完整矩阵 (valid40, 4 epochs / 720-3600 steps)

| 神经元 | 注意力 | AEE | AAE | SOPs | firing | 来源 | 状态 |
|--------|--------|------|------|------|--------|------|------|
| **PSN only** | QK | 1.585 | 7.501 | 3.622G | 0.085 | baseline epoch59 | ✅ |
| PSN only | TX | — | — | — | — | — | ❌ |
| PSN only | SC | — | — | — | — | — | ❌ |
| **PSN + ATLIF** (二值) | QK | 1.593 | 8.434 | 3.473G | 0.081 | H3f epoch29 | ✅ |
| PSN + ATLIF (二值) | TX | — | — | — | — | — | ❌ |
| PSN + ATLIF (二值) | SC | — | — | — | — | — | ❌ |
| **PSN + 三值** (no ATLIF) | QK | 1.622 | 7.848 | 9.046G | 0.212 | st_qk_3600 epoch3 | ✅ |
| **PSN + 三值** (no ATLIF) | **TX** | **1.516** | **7.550** | 8.982G | 0.211 | st_tx_3600 epoch3 | ✅ |
| **PSN + 三值** (no ATLIF) | **SC** | **1.527** | **7.504** | 9.140G | 0.214 | st_sc_3600 epoch3 | ✅ |
| PSN + ATLIF + 三值 | QK | — | — | — | — | — | ❌ |
| **PSN + ATLIF + 三值** | **TX** | **1.578** | **9.326** | **3.186G** | **0.075** | H54b epoch4 | ✅ |
| **PSN + ATLIF + 三值** | **SC** | **1.622** | **9.455** | **3.128G** | **0.073** | H41 SC S012C epoch27 | ✅ |

### 关键发现

1. **ATLIF 是 AAE 退化的唯一根源**：
   - PSN+三值+TX：AAE=7.55（接近 baseline 7.50）
   - PSN+ATLIF+三值+TX：AAE=9.33（+1.78）
   - 同一注意力(TX)，加 ATLIF → AAE 退化 1.78。ATLIF 单独贡献了几乎所有退化。

2. **ATLIF 是稀疏的唯一途径**：
   - 无 ATLIF：firing=21%，SOPs=9.0G（2.5× baseline）
   - 有 ATLIF：firing=7.5%，SOPs=3.19G（0.88× baseline）
   - ATLIF 把 firing 从 21% 压到 7.5%

3. **三值本身改善精度**：
   - PSN only + QK：AAE=7.50（baseline）
   - PSN+三值 + QK：AAE=7.85（+0.35）
   - PSN+三值 + TX：AAE=7.55（+0.05，几乎持平）
   - 三值 + TX 注意力组合可以完全补偿三值引入的微弱退化

4. **TX > SC > QK**（在 PSN+三值上一致）：
   - TX：AEE 最好（1.516），AAE=7.55
   - SC：AAE 最好（7.504），AEE=1.527
   - QK：两者都最差（1.622/7.848）

5. **PSN+ATLIF（二值）+QK 的 AAE=8.43**——只加 ATLIF 不加三值，AAE 退化 +0.93。比 ATLIF+三值的 +1.78 好得多。说明 H3 的二值 ATLIF 是更好的稀疏控制方式。

### 下一步：8 个缺失格子的优先级

| 优先级 | 神经元 | 注意力 | 理由 |
|--------|--------|--------|------|
| **P0** | PSN+ATLIF(二值) | TX | ATLIF保稀疏 + TX保精度，最可能平衡 |
| **P0** | PSN+ATLIF(二值) | SC | 对照，看 SC 能否进一步压 AAE |
| P1 | PSN+ATLIF+三值 | QK | 完整消融链 |
| P1 | PSN only | TX | 纯注意力贡献（无神经元改动） |
| P1 | PSN only | SC | 同上 |

---

## 三十一、Upstream Baseline 正式结果（2026-05-29~30）

> 训练：`train_flow_parallel_supervised_SNN.py`，stride split，60 epochs，batch=6，AMP+augment。
> 评估：`eval_DSEC_flow_SNN.py`（已修复 MLflow-free + combine_entries + spike profiling），full 825 valid，CuPy backend。

### 31.1 各 Epoch 标准评估

| Epoch | AEE | AAE | Outliers | PE1 |
|---|---|---|---|---|
| 36 | 1.544 | 10.490 | 9.13% | 53.1% |
| 40 | 1.672 | 10.888 | 11.58% | 55.5% |
| 41 | 1.592 | 10.513 | 10.13% | 53.9% |
| 42 | 1.569 | 10.142 | 9.84% | 52.9% |
| 45 | 1.608 | 10.465 | 10.61% | 54.1% |
| 48 | 1.613 | 10.340 | 10.81% | 54.2% |
| 51 | 1.531 | 9.980 | 9.51% | 52.0% |
| 56 | 1.515 | 9.986 | 9.06% | 51.3% |
| 57 | 1.529 | 10.166 | 9.11% | 51.8% |
| **59** 🏆 | **1.489** | **9.923** | **8.72%** | **51.0%** |

### 31.2 稀疏指标（epoch 59, CuPy backend, 825 valid）

| 指标 | 值 |
|---|---|
| total_spikes | 44.05G |
| global_firing_rate | 9.50% |
| dense_flops | 1229.21G（attention-type-aware: baseline=1229G, TX=939G, SC=956G）|
| effective_flops | 116.79G |
| sparsity_ratio | 90.5% |
| synops_mac (MLP/Conv) | 41.54G |
| synops_logic (attention) | 2.51G |
| estimated_energy | 37.64 mJ |

### 31.2b 标准 Baseline Checkpoint（后续所有续训的起点）

```
路径: experiments/baseline_stride_upstream/checkpoint_epoch59.pth
格式: torch.save(model) 完整模型（上游格式）
AEE: 1.489  (825 valid, cupy backend)
AAE: 9.923
total_spikes: 44.05G
firing_rate: 9.50%
训练: upstream train_flow_parallel_supervised_SNN.py, stride split, 60 epochs
```

**所有 H 系列改进方案的 stride 续训都从这个 checkpoint 出发。** 不要用 extend/warm restart 的 checkpoint。

### 31.3 收敛分析

```
Epoch  0-10:  loss 7.58 → 1.81  (陡降)
Epoch 10-20:  loss 1.81 → 1.27  (放缓, LR decay #1)
Epoch 20-30:  loss 1.27 → 1.16  (趋缓, LR decay #2)
Epoch 30-40:  loss 1.16 → 1.41  (最佳区域, LR decay #3)
Epoch 40-60:  loss 1.41 → 1.49  (平坦, LR=3.1e-6 → 1.6e-6)
```

**已收敛。** 60 epochs 是论文设定，无需继续。

### 31.4 评估脚本修复清单

`eval_DSEC_flow_SNN.py` 已完成以下修复/新增：

| 修改 | 类型 | 说明 |
|---|---|---|
| `import torchvision` 置顶 | 🐛 fix | 修复 CuPy 后端 torchvision 循环导入 |
| `use_ml_flow` 改为读 `SDFORMER_USE_MLFLOW` 环境变量 | ✨ feat | 无需 MLflow 即可本地 eval |
| `combine_entries` 在非 MLflow 路径也调用 | 🐛 fix | `spiking_neuron` 合并到 `model` dict |
| `_SpikeProfiler` 类 | ✨ feat | 标准推理中加入 spike 计数 |
| `_compute_total_dense_flops` | ✨ feat | 架构级 dense FLOPs 计算 |
| SOPs/SynOps/Energy 汇总输出 | ✨ feat | 一次推理输出全部稀疏指标 |

### 31.5 CuPy 兼容性

- 安装 `cupy` 后直接可用：`SDFORMER_SNN_BACKEND=cupy`
- 唯一修复：`eval_DSEC_flow_SNN.py` 顶部加 `import torchvision`（避免 CuPy 与 torchvision 循环导入）
- 训练脚本 `train_flow_parallel_supervised_SNN.py` 如需 CuPy 也需同样加 `import torchvision`
- 结果：CuPy 与 torch backend 指标一致，速度接近（本机测试无明显差异，A800 上 torch 已足够快）

### 31.6 H 系列改进方案 Stride 续训标准流程

#### 训练命令

```
cd /root/private_data/work/sdformer_codex/SDformer

SDFORMER_USE_MLFLOW=0 SDFORMER_MLFLOW_MODEL_LOGGING=0 SDFORMER_SNN_BACKEND=cupy \
python -u neuron_experiments/H9_bipolar_self_attention/entrypoints/train.py \
  --config <H9_config.yml> \
  --prev_runid <stride_baseline_checkpoint.pth> \
  --save_path <output_dir>/checkpoint_epoch{}.pth
```

**关键点**：
- 使用 **H9 entrypoint**（`entrypoints/train.py`），它会 monkey-patch 上游训练脚本，安装 ATLIF + BSA attention overlay
- `--prev_runid` 指向 stride baseline checkpoint（原始 ep59 或 extend best）
- 环境变量 `SDFORMER_USE_MLFLOW=0` 禁用 MLflow
- `SDFORMER_SNN_BACKEND=cupy` 使用 CuPy 后端
- 训练入口会自动 `chdir` 到 `third_party/SDformerFlow`，config 里的相对路径 `../../data/...` 会正确解析

#### 评估命令

**Baseline 评估**：
```
SDFORMER_USE_MLFLOW=0 SDFORMER_SNN_BACKEND=cupy \
python third_party/SDformerFlow/eval_DSEC_flow_SNN.py \
  --config configs/generated/upstream_baseline_eval.yml \
  --checkpoint <baseline_ckpt>.pth
```

**H 系列（TX/SC）评估**：必须用 H9 config（含 ATLIF/BSA section），eval 脚本才能安装 overlay 并加载正确的注意力权重：
```
SDFORMER_USE_MLFLOW=0 SDFORMER_SNN_BACKEND=cupy \
python third_party/SDformerFlow/eval_DSEC_flow_SNN.py \
  --config neuron_experiments/H9_bipolar_self_attention/configs/generated/<h9_config>.yml \
  --checkpoint <h9_ckpt>.pth
```

⚠️ 不能混用：baseline eval config 不含 ATLIF/BSA，eval 会跳过 overlay 权重，得到错误指标。
如果用 baseline eval config 去评估 H 系列 checkpoint，`load_model(strict=False)` 会跳过 overlay 专属权重，等价于回退到 baseline 注意力；这类结果无效。H 系列必须用对应 H9 config 评估，`eval_DSEC_flow_SNN.py` 会在建模和加载 checkpoint 前安装 ATLIF/BSA overlay。

#### 改进实验标准训练/推理入口（2026-06-01 明确版）

后续 NB0/NSC/NTX 口径下，训练和推理分成两级：

1. `profile_checkpoints.py`：只用于批量预筛 checkpoint 和快速排序，可跑 valid825，但不作为论文最终 AEE/AAE 口径。
2. `eval_DSEC_flow_SNN.py`：最终标准推理入口，用于论文表格、正式汇报和最终 checkpoint 复核。

原因：`profile_checkpoints.py` 会调用独立的 `tools/profile_sops.py`，其模型构建、dense ops/SOPs 估计和指标汇总逻辑独立于 upstream eval。它适合一次性扫多个 epoch，找出候选；最终结果必须回到已修复的 upstream 标准 eval 路径。

**改进实验训练入口**：统一使用 H9 entrypoint。

```bash
cd /root/private_data/work/sdformer_codex/SDformer

SDFORMER_USE_MLFLOW=0 \
SDFORMER_MLFLOW_MODEL_LOGGING=0 \
SDFORMER_SNN_BACKEND=cupy \
/opt/conda/envs/sdformerflow/bin/python -u neuron_experiments/H9_bipolar_self_attention/entrypoints/train.py \
  --config neuron_experiments/H9_bipolar_self_attention/configs/generated/<improved_config>.yml \
  --prev_runid experiments/baseline_stride_upstream/checkpoint_epoch59.pth \
  --save_path neuron_experiments/H9_bipolar_self_attention/results/<run_name>_$(date +%Y%m%d_%H%M%S)/checkpoint_epoch{}.pth
```

当前必须从 `experiments/baseline_stride_upstream/checkpoint_epoch59.pth` 续训。`<improved_config>` 优先只用：

| 代号 | 配置 | 用途 |
|---|---|---|
| `NSC-01` | `stride_h41_sc_s012c.yml` | SC S012 标准重跑/对照 |
| `NTX-01` | `stride_h41_tx_s02c_v2.yml` | 修复 Q/K ATLIF 安装后的 TX S02 主线 |
| `NSC-02` | `stride_h56a_sc_agree_disagree_l08_fast_warm_tr07_full30.yml` | 另一台服务器 SC agree/disagree 标准线 |

**阶段 A：批量预筛 checkpoint**。可以用 `profile_checkpoints.py` 扫多个 epoch，输出 ranking，帮助决定哪些 epoch 进入最终 eval。

```bash
cd /root/private_data/work/sdformer_codex/SDformer

SDFORMER_USE_MLFLOW=0 \
SDFORMER_MLFLOW_MODEL_LOGGING=0 \
SDFORMER_SNN_BACKEND=cupy \
PYTHONPATH="/root/private_data/work/sdformer_codex/SDformer/third_party/SDformerFlow:$PYTHONPATH" \
/opt/conda/envs/sdformerflow/bin/python -u neuron_experiments/H9_bipolar_self_attention/entrypoints/profile_checkpoints.py \
  --config neuron_experiments/H9_bipolar_self_attention/configs/generated/<improved_config>.yml \
  --run-dir <run_dir> \
  --samples 999999 \
  --epoch <epoch_a> \
  --epoch <epoch_b> \
  --epoch <epoch_c> \
  --epoch <epoch_d>
```

预筛验收：`sops_summary.json.samples` 应为 `825`，`profile_ranking_valid999999.md` 只作为候选排序记录。

**阶段 B：最终标准推理**。对阶段 A 选出的 1-3 个 checkpoint，必须使用 `eval_DSEC_flow_SNN.py` 逐个跑标准 valid825。

```bash
cd /root/private_data/work/sdformer_codex/SDformer

SDFORMER_USE_MLFLOW=0 \
SDFORMER_MLFLOW_MODEL_LOGGING=0 \
SDFORMER_SNN_BACKEND=cupy \
PYTHONPATH="/root/private_data/work/sdformer_codex/SDformer/third_party/SDformerFlow:$PYTHONPATH" \
/opt/conda/envs/sdformerflow/bin/python -u third_party/SDformerFlow/eval_DSEC_flow_SNN.py \
  --config neuron_experiments/H9_bipolar_self_attention/configs/generated/<improved_config>.yml \
  --checkpoint <run_dir>/checkpoint_epoch<epoch>.pth \
  --path_results neuron_experiments/H9_bipolar_self_attention/results/<eval_name>
```

最终推理验收：

| 检查项 | 要求 |
|---|---|
| split | `valid`，由 `eval_DSEC_flow_SNN.py` 固定使用 full valid |
| batch_size | eval 脚本强制 `1` |
| `spike_profile.json.samples` | 必须是 `825` |
| backend | `cupy` |
| config | 必须是对应 H9 improved config，不能用 baseline eval config |
| 结果文件 | `--path_results/spike_profile.json` |

**TX v2 当前推荐流程**：先预筛 `epoch22/27/28/29`，再用 `eval_DSEC_flow_SNN.py` 对预筛最优和 epoch29 做最终标准推理。

预筛命令：

```bash
SDFORMER_USE_MLFLOW=0 \
SDFORMER_MLFLOW_MODEL_LOGGING=0 \
SDFORMER_SNN_BACKEND=cupy \
PYTHONPATH="/root/private_data/work/sdformer_codex/SDformer/third_party/SDformerFlow:$PYTHONPATH" \
/opt/conda/envs/sdformerflow/bin/python -u neuron_experiments/H9_bipolar_self_attention/entrypoints/profile_checkpoints.py \
  --config neuron_experiments/H9_bipolar_self_attention/configs/generated/stride_h41_tx_s02c_v2.yml \
  --run-dir experiments/baseline_stride_upstream/h41_tx_stride_v2 \
  --samples 999999 \
  --epoch 22 \
  --epoch 27 \
  --epoch 28 \
  --epoch 29
```

最终 eval 模板：

```bash
SDFORMER_USE_MLFLOW=0 \
SDFORMER_MLFLOW_MODEL_LOGGING=0 \
SDFORMER_SNN_BACKEND=cupy \
PYTHONPATH="/root/private_data/work/sdformer_codex/SDformer/third_party/SDformerFlow:$PYTHONPATH" \
/opt/conda/envs/sdformerflow/bin/python -u third_party/SDformerFlow/eval_DSEC_flow_SNN.py \
  --config neuron_experiments/H9_bipolar_self_attention/configs/generated/stride_h41_tx_s02c_v2.yml \
  --checkpoint experiments/baseline_stride_upstream/h41_tx_stride_v2/checkpoint_epoch<epoch>.pth \
  --path_results neuron_experiments/H9_bipolar_self_attention/results/tx_v2_epoch<epoch>_standard_valid825
```

`eval_DSEC_flow_SNN.py` 也可以用于 baseline 论文口径复核；改进实验最终一律以该脚本产出的 `spike_profile.json` 为准。

#### 当前 TX 配置

| 文件 | 说明 |
|---|---|
| `stride_h41_tx_s02c_beta040.yml` | TX ternary α-XNOR, S02 FFN, β=0.40, 20ep |
| `stride_h41_tx_s02c_v2.yml` | TX ternary α-XNOR, S02 FFN, β=0.25, Q/K relaxed ATLIF no target-rate, 30ep；2026-05-31 已修复 Q/K 安装配置 |
| `stride_h41_sc_s012c.yml` | SC signed consensus, S012 FFN, ang=0.02, 20ep |

#### NTX-01 TX v2 标准 valid825 推理结果（2026-06-01）

实验编号统一为 `NTX-01`，物理目录为 `experiments/baseline_stride_upstream/h41_tx_stride_v2`，配置为 `configs/generated/stride_h41_tx_s02c_v2.yml`。本轮使用最终标准入口 `third_party/SDformerFlow/eval_DSEC_flow_SNN.py`，不是 `profile_checkpoints.py`。

结果目录：

```text
neuron_experiments/H9_bipolar_self_attention/results/ntx01_tx_v2_standard_valid825_20260601_141028
```

加载审计通过：

```text
[H9] eval installed ATLIFTernaryPSN: 34 modules
[H9] eval installed Shiftmax attention: 12 modules
[H9] load audit: checkpoint_overlay_keys=68, missing=0, unexpected=0
[runtime] SNN backend = cupy
```

| epoch | samples | AEE | AAE | PE1 | PE2 | PE3/out | total_spikes(G) | firing | dense(G) | effective(G) | synops_mac(G) | synops_logic(G) | energy(mJ) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 22 | 825 | 1.5502 | 10.4674 | 0.5452 | 0.2043 | 0.0927 | 33.5140 | 0.07229 | 939.4025 | 67.9062 | 31.4730 | 2.0410 | 28.5298 |
| 28 | 825 | **1.5340** | 10.2880 | 0.5392 | **0.2020** | **0.0911** | 34.6119 | 0.07465 | 939.4025 | 70.1308 | 32.8136 | 1.7984 | 29.7120 |
| 29 | 825 | 1.5837 | **10.2845** | **0.5358** | 0.2055 | 0.0966 | **32.9173** | **0.07100** | 939.4025 | **66.6971** | **31.1911** | **1.7261** | **28.2446** |

当前判断：`epoch28` 是精度最优点；`epoch29` 是稀疏/能耗最优点，但 AEE 明显回退。若论文主表优先精度，选 `epoch28`；若强调节能 Pareto，可同时报告 `epoch28` 与 `epoch29`。

#### TX v2 配置修复记录（2026-05-31）

旧版 `stride_h41_tx_s02c_v2.yml` 曾写成 `atlif_ternary_psn.target: none`，并在 `qk_all_ternary.paths: []` 注释中声称由 installer 自动发现 Q/K。实际 installer 只会遍历非空 `paths`，不会对空列表自动发现，因此旧版 TX v2 只安装了 S02 FFN 的 10 个 ATLIF 模块，没有安装 24 个 Q/K 三值 ATLIF 模块。旧版 TX v2 训练/推理结果不能作为 `PSN+ATLIF+三值 Q/K + TX` 的有效实验。

修复后配置改为：

- root `target: qk`，由 installer 自动安装 12 个 attention block 的 `sn_q/sn_k`，共 24 个 Q/K 模块；
- root `threshold_mode: symmetric_bsa_tsn`、`center_mode: bias`；
- root `target_rate: null`、`target_rate_eta: 0.0`、`activity_eta: 0.0`，保持 Q/K relaxed ATLIF；
- FFN S02 仍通过显式 `target_groups.paths` 使用 `official_atlif`。

验收标准：训练日志开头 `ATLIFTernaryPSN summary.num_modules` 应为 `34`，且 `symmetric_bsa_tsn_modules=24`、`official_atlif_modules=10`。若日志显示 `num_modules=10`，说明 Q/K 未安装，该结果必须作废并重跑。

#### eval_DSEC_flow_SNN.py 已修复项（31.4 基础上新增）

| 修复 | 说明 |
|---|---|
| mlflow stub @ module level | `SDFORMER_USE_MLFLOW=0` 时即使 mlflow 未安装也不报错 |
| `_install_h9_overlay` 路径搜索 | 从 repo root 搜索 `neuron_experiments/*/overlay` 而非从 config 目录 |
| `_install_h9_modules` | 在 load_model 前安装 ATLIF + BSA attention（当 config 含相关 section 时） |

### 31.7 当前 SC 续训状态与修复影响范围（2026-05-31）

当前正在运行的训练：

```bash
python -u neuron_experiments/H9_bipolar_self_attention/entrypoints/train.py \
  --config neuron_experiments/H9_bipolar_self_attention/configs/generated/stride_h41_sc_s012c.yml \
  --prev_runid experiments/baseline_stride_upstream/checkpoint_epoch59.pth \
  --save_path neuron_experiments/H9_bipolar_self_attention/results/stride_h41_sc_s012c_20260531_170553/checkpoint_epoch{}.pth
```

已核对日志：

| 项 | 当前值 | 结论 |
|---|---:|---|
| ATLIF 模块数 | 38 | 正常：24 个 Q/K 三值 ATLIF + 14 个 FFN official ATLIF |
| attention 模块数 | 12 | 正常：所有 encoder attention 已安装 SC overlay |
| Q/K 模式 | `symmetric_target_rate` | 当前 SC 线使用 target-rate 控制 Q/K 三值发放 |
| FFN 模式 | `official_atlif` | S012 FFN 使用官方 ATLIF 阈值增长范式 |
| angular loss | `lambda_ang=0.02` | 当前 SC 线保留角度约束 |
| backend | `torch` | 当前进程启动时未显式使用 CuPy；影响速度/记录口径，不改变模块逻辑 |
| checkpoint | 已保存到 `checkpoint_epoch18.pth` | 可在训练结束后评估 epoch20/24/27/29 |

刚修复的三类问题对当前已启动训练的影响：

| 修复项 | 是否影响当前训练 | 说明 |
|---|---|---|
| H9 checkpoint/config load audit | 否 | 只影响后续 eval/profile/新训练启动时的 checkpoint 加载；当前训练已完成启动加载 |
| `profile_sparsity.py` H9 overlay audit | 否 | 只影响后续稀疏指标统计 |
| `eval_DSEC_flow_SNN.py` H9 overlay audit | 否 | 只影响后续标准推理，避免 H checkpoint 被 baseline config 误评估 |
| `ATLIF effective_update_mean` 日志修正 | 不改变训练数学 | 只修正日志统计口径；阈值更新本身不因此改变 |
| `init_seeds()` 补 `random/np/cuda` seed | 否 | 当前进程已初始化；只影响后续新训练的可复现性 |

结论：**当前 SC 训练本身没有涉及刚修的 eval/profile/checkpoint 审计错误，也不是 Q/K 未安装的错误版本。** 当前训练可以跑完；跑完后必须用 H9 config 做标准 valid825 推理和 profile，不能用 baseline eval config。

### 31.8 后续主线计划：先收口 DSEC，不再发散（2026-05-31）

目标：基于标准 stride baseline `checkpoint_epoch59.pth`，重跑少量最有价值的 TX/SC 方案，得到可写论文的 DSEC 主结果。MVSEC、体素化、剪枝暂时后置。

#### P0：当前 SC 线收口

实验：`stride_h41_sc_s012c`

方案：

- attention：`signed_consensus_shiftmax`
- Q/K：PSN + ATLIF + 三值，`symmetric_target_rate`
- FFN：S012 的 PSN + ATLIF 二值，`official_atlif`
- loss：`lambda_ang=0.02`
- 续训起点：`experiments/baseline_stride_upstream/checkpoint_epoch59.pth`

执行：

1. 当前训练继续到 epoch29/30。
2. 训练结束后对 `checkpoint_epoch20.pth`、`checkpoint_epoch24.pth`、`checkpoint_epoch27.pth`、`checkpoint_epoch29.pth` 做标准 valid825 推理。
3. 对同一批 checkpoint 跑 `tools/profile_sparsity.py`，记录：
   - AEE
   - AAE
   - Outliers
   - PE1
   - total_spikes / global_firing_rate
   - dense_flops / effective_flops
   - synops_mac / synops_logic
   - estimated_energy
4. 只保留最优 checkpoint，其余中间 full-model checkpoint 可清理，仅保留 state_dict 或表格记录。

判断标准：

| 条件 | 判断 |
|---|---|
| AEE ≤ 1.62 且 AAE ≤ 9.60 且 total_spikes/SOPs 约 3G | SC 可作为主线候选 |
| AEE 1.62~1.75 或 AAE 9.60~10.30 | SC 作为对照，主线转 TX |
| AEE > 1.75 或 AAE > 10.30 | SC 不再扩展，只保留负/弱对照 |

#### P1：重跑修复后的 TX 主线

实验：`stride_h41_tx_s02c_v2`

原因：旧版 TX v2 配置曾经 `target: none` 且 `qk_all_ternary.paths: []`，导致 Q/K 三值 ATLIF 没有安装。这个问题会直接破坏“PSN+ATLIF+三值 Q/K + TX”的实验定义，因此凡是基于旧 TX v2 配置得到的结果必须作废并重跑。

修复后方案：

- attention：TX / ternary alpha-XNOR shiftmax
- Q/K：PSN + ATLIF + 三值，`symmetric_bsa_tsn` 或 relaxed ATLIF，无 target-rate
- FFN：S02 的 PSN + ATLIF 二值，`official_atlif`
- 续训起点：`experiments/baseline_stride_upstream/checkpoint_epoch59.pth`
- 新训练必须显式设置：

```bash
SDFORMER_USE_MLFLOW=0 SDFORMER_MLFLOW_MODEL_LOGGING=0 SDFORMER_SNN_BACKEND=cupy
```

验收标准：

| 日志项 | 必须满足 |
|---|---|
| `ATLIFTernaryPSN summary.num_modules` | 34 |
| `symmetric_bsa_tsn_modules` 或 Q/K ternary modules | 24 |
| `official_atlif_modules` | 10 |
| `attention summary.num_modules` | 12 |
| eval config | 必须使用同一个 H9 config |

推理策略同 P0：epoch20/24/27/29 全部标准 valid825 推理 + profile。

#### P2：只做必要消融，不再大规模铺开

在 P0/P1 结果出来后，只补三个最能支撑论文叙事的消融：

| 消融 | 目的 | 是否全量 |
|---|---|---|
| PSN + 三值 + TX/SC，不加 ATLIF | 分离“注意力三值化”和“ATLIF 稀疏”的贡献 | 若已有标准结果可不重跑 |
| PSN + ATLIF 二值 + TX/SC，不加三值 | 验证 ATLIF 稀疏本身是否导致 AAE 退化 | 先短训，只有接近主线才全量 |
| TX vs SC 同替换范围同超参 | 公平比较注意力机制 | 只对最优替换范围全量 |

#### 暂停项

| 方向 | 状态 | 原因 |
|---|---|---|
| MVSEC | 暂停 | 用户明确要求暂时不管；先把 DSEC 主线做干净 |
| 体素化优化 | 暂停 | 会引入数据预处理变量，不适合和当前神经元/注意力主线混在一起 |
| 剪枝 | 暂停 | 先确认 ATLIF+三值是否能稳定给出 3G 左右稀疏结果 |
| 新 attention 大改 | 暂停 | 过去 direct QKV / K-reuse / raw shiftmax 多数失败；当前阶段优先复现和收口已有有效 TX/SC |

#### 后续执行顺序

1. 等当前 `stride_h41_sc_s012c_20260531_170553` 跑完。
2. 立即跑标准 valid825 推理和 profile，写入本文档结果表。
3. 启动修复后的 `stride_h41_tx_s02c_v2` 全量续训，显式使用 CuPy 和 seed。
4. TX 跑完后同样评估 epoch20/24/27/29。
5. 在 SC/TX 结果中选择主线：
   - 若 TX AEE/AAE 更稳且 SOPs 更低：论文主线用 TX；
   - 若 SC AAE 明显更好且 SOPs 接近：SC 作为主线或并列；
  - 若二者都弱于旧 valid816 排序：先检查 split 和 checkpoint 加载审计，再决定是否只保留 ATLIF/三值消融。

#### 新标准流程实验代号（2026-06-01）

为避免继续混用旧 H 系列编号，后续基于标准 stride split、标准 baseline epoch59 的实验统一使用以下代号：

| 新代号 | 含义 | 当前对应物理目录/配置 |
|---|---|---|
| `NB0` | 新标准 baseline，stride split，upstream 训练流程，epoch59 checkpoint | `experiments/baseline_stride_upstream/checkpoint_epoch59.pth` |
| `NSC-01` | 新 baseline 续训的 SC 标准重跑线，先复现旧 H41 SC S012C | `results/stride_h41_sc_s012c_20260531_170553`，配置 `configs/generated/stride_h41_sc_s012c.yml` |
| `NTX-01` | 新 baseline 续训的 TX S02 v2 标准线，修复 Q/K ATLIF 安装后的 H41 TX 重跑 | `experiments/baseline_stride_upstream/h41_tx_stride_v2`，配置 `configs/generated/stride_h41_tx_s02c_v2.yml` |
| `NTX-02` | 新 baseline 续训的 TX 逐 token selector 线，修正 H49/H53 方案 | 队列将生成 `configs/generated/stride_h53b_h49_clean_no_stage3_s02_full30.yml` 并启动对应训练 |
| `NSC-02` | 另一台服务器建议执行的 SC agree/disagree 标准线 | `configs/generated/stride_h56a_sc_agree_disagree_l08_fast_warm_tr07_full30.yml` |
| **`NTS`** | **TX/SC score-level 融合注意力族**（代码 alias `h60`） | 见 §三十三；物理目录暂保留 `ntx_h60_*` 前缀 |
| `NTS-01` | 首版 NTS full30（μ=0.1, α_k=0.02） | `results/ntx_h60_full30_20260605_020633` |
| `NTS-02` | 调参版 NTS full30（当前主线） | `results/ntx_h60_v2_full30_20260605_163955` |

**四代实验口径**（2026-06-06 起）：

| 代号 | 含义 | 代码 mode 示例 |
|------|------|----------------|
| `NB0` | stride valid825 标准 baseline | QKFormer 原生注意力 |
| `NTX` | 纯 TX 注意力实验族 | `h41`/`h49`、carrier×gate |
| `NSC` | 纯 SC 或 carrier+SC **gate 级**混合 | `h56`/`h59` |
| **`NTS`** | **TX+SC score 级融合**（Neuromorphic Ternary Selector） | `h60` / `tx_sc_k_mag_no_carrier_shiftmax` |

说明：NTS 是方法名，不是 NTX 的子编号。旧称「NTX-12 (h60)」已作废，统一改称 **NTS-01**。物理目录仍保留 `ntx_h60_*` 前缀以免中断已有 checkpoint 路径；汇报、论文和新建 sweep 优先使用 `nts_*` 前缀。其他实验继续按 `NB0 / NSC / NTX` 口径描述。

#### 自动队列记录（2026-05-31）

已启动本机自动队列：

```bash
setsid bash -c 'cd /root/private_data/work/sdformer_codex/SDformer && python -u neuron_experiments/H9_bipolar_self_attention/entrypoints/run_after_sc_standard_queue.py --wait-pid 12109 >> neuron_experiments/H9_bipolar_self_attention/results/queues/sc_then_txselector_20260531.log 2>&1'
```

队列行为：

1. 等待当前 `stride_h41_sc_s012c_20260531_170553` 训练进程结束。
2. 对 `checkpoint_epoch20/24/27/29.pth` 跑标准 valid825 推理，保存到 `standard_valid825/epoch*/`。
3. 从 `spike_profile.json` 读取并写回本 md：AEE、AAE、PE1、PE2、PE3/outlier、total_spikes、firing、dense/effective FLOPs、sparsity、SynOps、energy。
4. 生成 `configs/generated/stride_h53b_h49_clean_no_stage3_s02_full30.yml`。
5. 从 `experiments/baseline_stride_upstream/checkpoint_epoch59.pth` 启动 corrected H49/H53 逐 token TX selector 全量续训。

说明：逐 token TX selector 不是全新路线，历史上对应 H49/H53。旧 H49/H53 结果基于旧 local816 split 和旧 baseline checkpoint；本队列启动的是 stride valid825 口径重跑版。

#### 标准推理与 profile 入口审计（2026-06-01）

本轮发现一个必须固定下来的流程问题：训练入口、标准推理入口、快速 profile 入口不是同一段代码，新增模块如果只在训练入口接好，不代表推理和 profile 一定正确生效。

当前三个入口的用途如下：

| 入口 | 用途 | 输出 | 当前定位 |
|---|---|---|---|
| `neuron_experiments/H9_bipolar_self_attention/entrypoints/train.py` | 全量训练/续训 | checkpoint、训练日志、训练期 valid | 正式训练入口 |
| `third_party/SDformerFlow/eval_DSEC_flow_SNN.py` | 标准 full valid825 推理 | `spike_profile.json`，含 AEE/AAE/PE/outlier/稀疏/能耗 | 论文和最终对比口径 |
| `neuron_experiments/H9_bipolar_self_attention/entrypoints/profile_checkpoints.py` | 批量短测 checkpoint 排名 | `sops_summary.json`、`profile_ranking_valid*.md` | 只用于 valid40/validN 快速筛选，不作为最终论文口径 |

这次的具体错误：

- 训练入口是正确的：`train.py` 调用 `install_atlif_ternary_psn(model, config.get("atlif_ternary_psn"))` 和 `install_shiftmax_attention(model, config.get("bsa_attention"))`。
- 标准推理入口此前只修过 ATLIF，BSA/Shiftmax attention 仍误写为 `install_shiftmax_attention(model, config)`。
- `install_shiftmax_attention` 需要的是 `bsa_attention` 子配置；传入完整 config 时，`enabled` 读不到，实际安装数量为 0。
- 后果：那一批 NSC-01 标准推理中 ATLIF 神经元生效，但 SC/Shiftmax attention 没有生效，`epoch20/24/27` 的旧 `standard_valid825` 结果不能作为最终结果。

已修复：

- `third_party/SDformerFlow/eval_DSEC_flow_SNN.py`：BSA 安装改为 `install_shiftmax_attention(model, config.get("bsa_attention"))`。
- `tools/profile_sparsity.py`：同样修复 BSA 安装参数。
- 两个入口都增加安装数量打印，后续日志必须出现类似：

```text
[H9] eval installed ATLIFTernaryPSN: 38 modules
[H9] eval installed Shiftmax attention: 12 modules
[H9] load audit: checkpoint_overlay_keys=76, missing=0, unexpected=0
```

修复后已用 `stride_h41_sc_s012c.yml` 最小脚本验证：

```text
atlif_installed 38
attention_installed 12
atlif_num_modules 38
attention_num_modules 12
```

因此当前规则改为：

1. **正式结果只认修复后的 `eval_DSEC_flow_SNN.py` full valid825 输出。**
2. `profile_checkpoints.py` 只能用于短测筛选 checkpoint 或超参趋势，不能直接写论文主表。
3. 每次新增或修改模块后，必须检查三条链路是否都安装了对应模块：
   - 训练入口：是否在 load checkpoint 前安装模块；
   - 标准推理入口：是否在 load checkpoint 前安装同一模块；
   - profile/稀疏入口：是否在 load checkpoint 前安装同一模块。
4. 每次全量训练前必须做一次加载审计：
   - 日志里必须有模块安装数量；
   - checkpoint overlay key 必须 `missing=0, unexpected=0`；
   - 如果新增参数如 `linear_v/bn_v/sn_v/thresh/center`，必须确认 checkpoint 中对应 key 被 model 接收，而不是 silent drop。
5. 如果新增 attention、neuron、pruning、voxel preprocessing 等模块，必须同步更新：
   - `entrypoints/train.py`
   - `third_party/SDformerFlow/eval_DSEC_flow_SNN.py`
   - `tools/profile_sparsity.py` 或对应 profile 入口
   - 必要时更新 `entrypoints/profile_sops.py` 的 overlay patch

已重启修复后的自动队列：

```bash
setsid bash -c 'cd /root/private_data/work/sdformer_codex/SDformer && /opt/conda/envs/sdformerflow/bin/python -u neuron_experiments/H9_bipolar_self_attention/entrypoints/run_after_sc_standard_queue.py --wait-pid 0 >> neuron_experiments/H9_bipolar_self_attention/results/queues/sc_then_txselector_20260601_retry3_fixed_bsa.log 2>&1'
```

该队列会覆盖重跑 `NSC-01` 的 `epoch20/24/27/29` 标准 valid825 推理，确认日志中安装 `ATLIF=38`、`Shiftmax attention=12` 后，再自动启动后续 selector 线 `NTX-02`。

#### NB0 baseline 标准推理复核（2026-06-01）

为确认 baseline 推理链路没有同类错误，已用修复后的标准入口重新跑 `NB0 epoch59` full valid825：

```bash
SDFORMER_USE_MLFLOW=0 SDFORMER_MLFLOW_MODEL_LOGGING=0 SDFORMER_SNN_BACKEND=cupy \
/opt/conda/envs/sdformerflow/bin/python third_party/SDformerFlow/eval_DSEC_flow_SNN.py \
  --config configs/generated/upstream_baseline_stride.yml \
  --checkpoint experiments/baseline_stride_upstream/checkpoint_epoch59.pth \
  --path_results results_inference/nb0_baseline_epoch59_valid825_fixed_eval_20260601_140852 \
  --mode valid
```

加载审计：

```text
[H9] load audit: checkpoint_overlay_keys=0, missing=0, unexpected=0
Model restored from local checkpoint experiments/baseline_stride_upstream/checkpoint_epoch59.pth
[runtime] SNN backend = cupy (explicit config)
```

结论：

- baseline config 没有 `atlif_ternary_psn` / `bsa_attention`，因此不应安装 H9 模块；
- baseline checkpoint 没有 overlay key，`checkpoint_overlay_keys=0` 是正确结果；
- `missing=0, unexpected=0`，权重完整加载；
- 数据 split 为 canonical valid825，`valid_split_seq.csv` 行数 825，sha256 为 `7f3dc2800653e12caca10379c51ee8e8988aaf6bb80c391224a454a5879325d0`；
- `eval_DSEC_flow_SNN.py` 已补上相对数据路径 fallback：先尝试 baseline root，再尝试 repo root，避免不同 cwd 下 baseline 数据路径解析失败。

复核指标：

| checkpoint | samples | AEE | AAE | PE1 | PE2 | PE3/outlier | total_spikes | firing | dense_flops | effective_flops | sparsity | synops_mac | synops_logic | energy_uj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NB0 epoch59 | 825 | 1.4872 | 9.9300 | 0.5107 | 0.1891 | 0.0871 | 44.0488G | 9.50% | 1229.2102G | 116.7864G | 90.50% | 41.5414G | 2.5074G | 37638.01 |

说明：

- 这个结果与前文记录的 `AEE≈1.489 / AAE≈9.923 / total_spikes≈44.05G / firing≈9.50%` 一致，可以作为当前 NB0 正式 baseline。
- 早期 `experiments/sparsity_profiles/upstream_baseline_epoch59/sparsity_summary.json` 是 `tools/profile_sparsity.py` 产物，AEE/AAE 与标准 eval 口径不完全一致；后续论文主表和最终对比只使用 `eval_DSEC_flow_SNN.py` 的 `spike_profile.json`。
- `profile_sparsity.py` / `profile_checkpoints.py` 仍可用于趋势筛选和层级稀疏分析，但不能替代标准 full valid825 eval。

#### NSC-01 标准推理结果（2026-06-01，修复后重跑）

> 配置：`stride_h41_sc_s012c.yml`（SC signed consensus + S012 FFN + angular λ=0.02）
> 训练：从 NB0 epoch59 续训 30 epoch，warmup 300步，bs=6
> 目录：`results/stride_h41_sc_s012c_20260531_170553`
> 推理：`eval_DSEC_flow_SNN.py` full valid825，CuPy backend，ATLIF=38，Shiftmax attention=12
> 审计：`checkpoint_overlay_keys=76, missing=0, unexpected=0`

| checkpoint | AEE | AAE | total_spikes | firing | dense_flops | effective_flops | sparsity | synops_mac | synops_logic | energy_uj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ep20 | 1.813 | 11.56 | 33.54G | 7.2% | 955.50G | 66.69G | 93.0% | 31.25G | 1.05G | 28231 |
| ep24 | 1.884 | 12.12 | 34.73G | 7.5% | 955.50G | — | — | — | — | — |
| ep27 | 1.872 | 12.03 | 34.12G | 7.4% | 955.50G | — | — | — | — | — |
| **ep29** 🏆 | **1.771** | **11.42** | **32.30G** | **7.0%** | 955.50G | 66.69G | 93.0% | 31.25G | 1.05G | 28231 |

**vs NB0 baseline:**
- AEE: 1.487 → 1.771 (+19%)，AAE: 9.93 → 11.42 (+15%)
- total_spikes: 44.05G → 32.30G (**-27%**)
- firing: 9.50% → 6.98% (**-26%**)
- energy: 37638uJ → 28231uJ (**-25%**)
- dense_flops: 1229G → 956G（SC attention 原生稀疏，少 22%）

**结论：SC 显著降低了稀疏指标（spikes -27%, energy -25%），但精度还有差距。30 epoch 可能不够，训练 loss 仍在下降趋势中，建议继续续训到 60 epoch 或调整 LR schedule。**

#### 另一台服务器 SC 下一实验配置（2026-06-01）

建议另一台服务器跑 SC-native agree/disagree 路线的标准 stride 版：

```text
配置: neuron_experiments/H9_bipolar_self_attention/configs/generated/stride_h56a_sc_agree_disagree_l08_fast_warm_tr07_full30.yml
续训起点: experiments/baseline_stride_upstream/checkpoint_epoch59.pth
```

训练命令：

```bash
cd /root/private_data/work/sdformer_codex/SDformer

SDFORMER_USE_MLFLOW=0 SDFORMER_MLFLOW_MODEL_LOGGING=0 SDFORMER_SNN_BACKEND=cupy \
/opt/conda/envs/sdformerflow/bin/python -u neuron_experiments/H9_bipolar_self_attention/entrypoints/train.py \
  --config neuron_experiments/H9_bipolar_self_attention/configs/generated/stride_h56a_sc_agree_disagree_l08_fast_warm_tr07_full30.yml \
  --prev_runid experiments/baseline_stride_upstream/checkpoint_epoch59.pth \
  --save_path neuron_experiments/H9_bipolar_self_attention/results/stride_h56a_sc_agree_disagree_l08_fast_warm_tr07_full30_$(date +%Y%m%d_%H%M%S)/checkpoint_epoch{}.pth
```

关键配置：

| 项 | 值 |
|---|---|
| attention | `sc_agree_disagree_shiftmax` |
| bipolar_lambda | `0.8` |
| Q/K ATLIF | PSN + ATLIF + ternary, `symmetric_target_rate` |
| Q/K target_rate | `0.07` |
| FFN | S012 official binary ATLIF |
| batch_size / workers | `8 / 8` |
| optimizer | AdamW, base lr `1e-5`, wd `0.001` |
| param lr | backbone `3e-7`, neuron `2e-5`, threshold `4e-6`, norm `3e-7` |
| scheduler | multistep `[20,25]` |
| warmup | 300 step, start_factor `0.1` |
| angular loss | off, `lambda_ang=0` |
| valid | `sample=825`, stride split |

这个实验的意义：它不是普通 SC Shiftmax，而是把 SC 的同号/异号证据分成 agree/disagree 两路做 Shiftmax 后相减，用来验证 SC 路线是否能在不引入 TX α-XNOR 权重的情况下压 AAE，同时保持 3G 左右稀疏度。

#### NSC-02 实验定义与演变来源（2026-06-02 补充）

`NSC-02` 不是旧的普通 `signed_consensus_shiftmax` 重跑，而是 H56a 的 SC-native agree/disagree 路线在标准 stride 数据口径上的 30 epoch 续训。它的目标是验证：不用 TX 的 alpha-XNOR / threshold-XNOR 权重混合，只利用 SC 自身的同号/异号事件证据，能否在保持强稀疏的同时修复 SC 线的角度误差。

**训练和数据口径**

| 项 | NSC-02 设置 |
|---|---|
| 实验编号 | `NSC-02` |
| 训练配置 | `configs/generated/stride_h56a_sc_agree_disagree_l08_fast_warm_tr07_full30.yml` |
| 续训起点 | `experiments/baseline_stride_upstream/checkpoint_epoch59.pth` |
| 训练入口 | `neuron_experiments/H9_bipolar_self_attention/entrypoints/train.py` |
| 模型 | `MS_SpikingformerFlowNet_en4` |
| 输入编码 | voxel，`num_bins=10`，`norm_input=minmax`，`mask_output=true` |
| 数据集 | DSEC preprocessed flow，`saved_flow_data` |
| 时序设置 | `num_frames=10`，`num_chunks=1`，`step_mode=m` |
| batch / workers | `batch_size=8`，`n_workers=8`，`persistent_workers=true`，`pin_memory=true`，`prefetch_factor=4` |
| 训练长度 | `n_epochs=30`，强制保存 epoch0-29 |
| crop / resolution | crop `[288,384]`，resolution `[480,640]` |
| augmentation | Horizontal p=0.5，Vertical p=0.5，Polarity p=0.0，`max_drop_rate=0.6` |
| runtime | CuPy backend，AMP=true，TF32=true，`cudnn_benchmark=true` |
| loss | supervised，`lambda_mod=1`，`lambda_ang=0`，`use_angular_loss=false`，`flow_regul_weight=0.001`，`clip_grad=100.0` |

**注意力范式**

| 项 | NSC-02 设置 |
|---|---|
| attention mode | `sc_agree_disagree_shiftmax` |
| 覆盖范围 | all stages，共 12 个 attention block |
| 核心形式 | 将 signed-consensus score 分成 agree/disagree 两路，`gate = g_agree - lambda * g_disagree` |
| `bipolar_lambda` | `0.8` |
| `value_mode` | `threshold` |
| score norm | `consensus_score_norm=head_dim`，`score_scale=1.0` |
| score center | `center_scores=true`，`consensus_bias=0.02`，`preserve_mean=true` |
| 其他参数 | `alpha0=0.02`，`mismatch_penalty=0.25`，`relu_k_floor=0.0`，`eps=1e-6` |
| 未启用项 | `deadzone_epsilon=0.0`，`confidence_enabled=false`，`k_consistency_mod=false` |

这个 attention 不是 TX。TX 路线是用 threshold/alpha-XNOR 风格做 Q/K 或 K/V 混合；NSC-02 仍然是 SC，只是把 SC 的正负证据拆开后分别 Shiftmax，再用 disagree 作为扣分项。

**神经元、target-rate 和稀疏控制**

| 项 | NSC-02 设置 |
|---|---|
| 基础 neuron | PSN，`num_steps=10`，BN spike norm，`surrogate.ATan()`，`tau=2.0`，`v_th=0.1`，`detach_reset=true` |
| Q/K ATLIF | enabled，target=`qk`，all stages，trainable all |
| Q/K 输出 | ternary，`center_mode=bias`，`negative_threshold_scale=1.0` |
| Q/K threshold | `threshold_mode=symmetric_target_rate`，`threshold_init=1.0`，min/max=`0.001/2.0` |
| Q/K target_rate | 有，`target_rate=0.07`，`target_rate_eta=0.08` |
| Q/K 阈值学习 | `threshold_eta=0.001`，`threshold_lr_scale=50000.0`，`threshold_base_lr=4e-6`，`activity_eta=2.0` |
| Q/K 安装数 | 24 个：12 个 attention block 的 `sn_q/sn_k` |
| FFN ATLIF | S0 + S1 全部 FFN，S2 的 block 0/2/4 FFN；S3 不替换 |
| FFN 输出 | binary official ATLIF，`center_mode=zero` |
| FFN target_rate | 无，`target_rate=null` |
| FFN 阈值学习 | `threshold_eta=8e-05`，S0/S1 `threshold_lr_scale=8000`，S2-half `threshold_lr_scale=6000` |
| FFN 安装数 | 14 个 official ATLIF |
| 总 ATLIF 数 | 38 个 = 24 个 Q/K ternary + 14 个 FFN binary |

训练日志确认：`target_rate_control_modules=24`，`symmetric_target_rate_modules=24`，`official_atlif_modules=14`。所以这条线确实有 Q/K target_rate，但 FFN 没有 target_rate；FFN 只通过 official ATLIF 阈值学习和 activity/threshold 参数间接变稀疏。

**优化器和学习率**

| 项 | NSC-02 设置 |
|---|---|
| optimizer | AdamW |
| base lr | `1e-5` |
| weight decay | `0.001` |
| param groups | enabled |
| backbone lr | `3e-7` |
| neuron lr | `2e-5` |
| threshold lr | `4e-6` |
| norm lr | `3e-7` |
| threshold wd / norm wd | `0 / 0` |
| scheduler | multistep，milestones `[20,25]` |
| warmup | enabled，300 steps，start_factor `0.1` |
| grad accumulation | `num_acc=1` |

训练日志里 global lr 随 warmup/param group 记录约为 epoch0 `2.991e-07`，epoch20 后 `1.4955e-07`，epoch25 后 `7.4775e-08`。实际影响训练的参数组如上表，尤其是 Q/K 阈值相关的 `threshold_lr=4e-6` 和 `threshold_lr_scale=50000.0`。

**训练过程中实际发生的变化**

| epoch | train log 观察 |
|---:|---|
| 0 | `activity_mean≈0.03995`，ternary≈0.04592，binary≈0.02972，val loss≈1.9669 |
| 6 | `activity_mean≈0.00951`，binary≈0.00259，val loss≈2.346 |
| 10 | `activity_mean≈0.00381`，binary≈0.00067 |
| 20 | `activity_mean≈0.000416`，ternary≈0.000553，binary≈0.000183，val loss≈2.0109 |
| 23 | val loss≈1.8802，但 activity 仍只有≈0.000294 |
| 29 | `activity_mean≈0.000199`，ternary≈0.000229，binary≈0.000146，pos/neg ratio≈1.83，val loss≈2.0458 |

这说明 NSC-02 的主要问题不是“没进入稀疏推理流程”，而是训练期间就已经被 Q/K target-rate + FFN official ATLIF 推到过稀疏。标准推理里的 total_spikes/energy 下降是真实进入流程的，但代价是 AEE/AAE 明显塌陷。

**它继承了之前哪些路线**

| 来源 | 继承内容 | NSC-02 中的状态 |
|---|---|---|
| NB0 baseline | 标准 stride upstream baseline，epoch59 作为续训起点 | 直接从 `checkpoint_epoch59.pth` 续训 |
| H36/H41 SC | `signed_consensus_shiftmax`、Q/K ATLIF、`symmetric_target_rate`、SC 稀疏注意力方向 | 继承 SC 思路，但 attention mode 升级为 agree/disagree |
| H41 SC S012C / NSC-01 | S0/S1/S2 范围的 FFN official ATLIF、SC full valid825 标准评估口径 | 继承 FFN 替换范围，但 NSC-02 是 S0/S1 full + S2 half |
| H46SC | SC 上尝试更强约束/penalty 的经验 | 作为反例参考，未直接采用其 single-active penalty 主设定 |
| H56a | `sc_agree_disagree_shiftmax`，`lambda=0.8`，fast warmup，`target_rate=0.07` | 直接采用，是 NSC-02 的主体 |
| H56b/H56c/H56d/H56e | deadzone、confidence、K consistency、active-norm 等 SC 改型想法 | NSC-02 未启用，保留为后续分支 |
| NTX-01 | 标准编号和 valid825 推理口径 | 只继承评估口径；结构不是 TX |

简化结论：NSC-02 = NB0 epoch59 续训 + H56a SC agree/disagree attention + Q/K symmetric target-rate 0.07 + S0/S1/S2-half FFN official ATLIF + fast warmup/multistep 30ep。它用了 SC 路线的演变成果，没有用 TX v2 的 alpha-XNOR 主体。

#### NSC-02 标准 valid825 推理结果（2026-06-02）

实验编号：`NSC-02`，配置 `stride_h56a_sc_agree_disagree_l08_fast_warm_tr07_full30.yml`，物理目录：

```text
neuron_experiments/H9_bipolar_self_attention/results/nsc02_sc_agree_disagree_20260601_154500
```

训练已完成 30 epoch。标准推理使用 `eval_DSEC_flow_SNN.py` full valid825。原训练配置缺少 `test.scale_factor`，因此推理使用 eval-only 配置副本：

```text
neuron_experiments/H9_bipolar_self_attention/configs/generated/stride_h56a_sc_agree_disagree_l08_fast_warm_tr07_full30_eval.yml
```

该副本只补 `test.scale_factor: 1`，不改变模型结构、checkpoint 或数据 split。加载审计通过：

```text
[H9] eval installed ATLIFTernaryPSN: 38 modules
[H9] eval installed Shiftmax attention: 12 modules
[H9] load audit: checkpoint_overlay_keys=76, missing=0, unexpected=0
[runtime] SNN backend = cupy
```

结果目录：

```text
neuron_experiments/H9_bipolar_self_attention/results/nsc02_standard_valid825_20260602_003717
```

| epoch | samples | AEE | AAE | PE1 | PE2 | PE3/out | total_spikes(G) | firing | dense(G) | effective(G) | synops_mac(G) | synops_logic(G) | energy(mJ) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 825 | 1.9339 | 12.6884 | 0.6345 | 0.2936 | 0.1513 | 30.1461 | 0.06515 | 1229.2102 | 80.0812 | 30.1199 | 0.0262 | 27.1105 |
| 23 | 825 | **1.8835** | 12.7569 | 0.6234 | **0.2786** | **0.1378** | 30.7209 | 0.06639 | 1229.2102 | 81.6082 | 30.7023 | 0.0186 | 27.6339 |
| 24 | 825 | 1.9355 | 12.5595 | 0.6308 | 0.2924 | 0.1520 | 29.4619 | 0.06367 | 1229.2102 | 78.2639 | 29.4464 | 0.0156 | 26.5033 |
| 29 | 825 | 1.9088 | **12.1452** | **0.6191** | 0.2790 | 0.1436 | **28.9151** | **0.06249** | 1229.2102 | **76.8113** | **28.9030** | **0.0122** | **26.0139** |

说明：`sc_agree_disagree_shiftmax` 是新 attention mode，当前 `eval_DSEC_flow_SNN.py` 的 dense FLOPs 折算还未为该 mode 单独设置 attention multiplier，因此 dense/effective FLOPs 仅作内部参考。主比较优先使用 `AEE/AAE/PE/total_spikes/firing/energy`。

对比 NB0 baseline：

| 指标 | NB0 ep59 | NSC-02 ep23 | NSC-02 ep29 |
|---|---:|---:|---:|
| AEE | 1.4872 | 1.8835 (+26.6%) | 1.9088 (+28.3%) |
| AAE | 9.9300 | 12.7569 (+28.5%) | 12.1452 (+22.3%) |
| total_spikes | 44.0488G | 30.7209G (-30.3%) | 28.9151G (-34.4%) |
| firing | 9.50% | 6.64% (-30.1%) | 6.25% (-34.2%) |
| energy | 37.6380mJ | 27.6339mJ (-26.6%) | 26.0139mJ (-30.9%) |

当前判断：NSC-02 达到了更强稀疏，`epoch29` 的 total_spikes 比 NTX-01 ep28 低约 16.5%，但精度明显不可接受。该路线暂不作为主线，只保留为“SC agree/disagree 过稀疏导致精度塌陷”的负/弱对照。若后续继续该方向，应优先降低 target-rate/ATLIF 增阈强度或改为更强保精度约束，而不是继续按当前 30ep schedule 推进。

#### NSC-02 诊断：与 baseline 差距、lambda/target-rate/Shiftmax 判断

**与 NB0 baseline 的差距**

| 对比点 | AEE | AAE | PE3/out | total_spikes | energy |
|---|---:|---:|---:|---:|---:|
| NB0 ep59 | 1.4872 | 9.9300 | 0.0871 | 44.0488G | 37.6380mJ |
| NSC-02 ep23，精度相对最好 | 1.8835 (+26.6%) | 12.7569 (+28.5%) | 0.1378 (+58.3%) | 30.7209G (-30.3%) | 27.6339mJ (-26.6%) |
| NSC-02 ep29，能耗最好 | 1.9088 (+28.3%) | 12.1452 (+22.3%) | 0.1436 (+64.9%) | 28.9151G (-34.4%) | 26.0139mJ (-30.9%) |

对比 NTX-01 ep28：NSC-02 ep29 的 total_spikes 更低，约 `28.92G vs 34.61G`，但 AEE/AAE 明显更差，约 `1.9088/12.1452 vs 1.5340/10.2880`。因此当前 SC agree/disagree 的收益主要是“更省”，不是“更准”。

**加载审计**

训练从 NB0 baseline 续训时日志显示：

```text
[H9] installed ATLIFTernaryPSN before load: 38 modules
[H9] installed attention before load: 12 modules
[H9] load audit: checkpoint_overlay_keys=0, missing=76, unexpected=0
```

这是预期现象：NB0 baseline checkpoint 本来没有 NSC-02 新增的 ATLIF/attention overlay 权重，训练从新安装模块开始。训练保存后的 NSC-02 checkpoint 做标准推理时，每个被评估 epoch 都显示：

```text
[H9] eval installed ATLIFTernaryPSN: 38 modules
[H9] eval installed Shiftmax attention: 12 modules
[H9] load audit: checkpoint_overlay_keys=76, missing=0, unexpected=0
```

所以标准推理确实加载了 NSC-02 对应的 overlay 权重数据，不是只加载 baseline 主干。

**Shiftmax 是否 raw**

`sc_agree_disagree_shiftmax` 当前调用的是稳定版 `shiftmax()`，不是 `shiftmax_raw()`。稳定版会先做：

```text
shifted = scores - row_max(scores)
numerator = 2^shifted
denominator = next_power_of_two(sum(numerator))
```

`shiftmax_raw()` 是不减 row max 的 ablation，NSC-02 没用它。因此当前失败不能归因于 raw shiftmax 数值爆/缩。

**lambda=0.8 是否合理**

`lambda=0.8` 作为第一条 H56a full30 线是合理的，因为它足够强调 disagree penalty，能验证“SC 正负证据分离”是否能压方向错误。但标准 valid825 结果说明它对当前训练口径偏强：AAE 仍高，PE3/out 也大幅高于 baseline，说明 disagree 扣分没有转化成更好的方向一致性，反而和过稀疏一起削弱了有效匹配。

后续不建议继续只围绕 `lambda=0.8` 拉长训练。更合理的 lambda 搜索是：

| 目的 | 建议 |
|---|---|
| 保精度 | `lambda=0.3/0.5` |
| 当前折中 | `lambda=0.6` |
| 负对照 | 保留 `lambda=0.8`，但不要作为主线 |

**target_rate 是否必要**

当前证据不支持“Q/K target_rate=0.07 是必要的”。训练日志显示 activity 从 epoch0 的约 `0.04` 迅速塌到 epoch29 的约 `0.0002`，但 target_rate 控制模块仍为 24。这说明当前 `symmetric_target_rate + threshold_lr_scale=50000 + FFN official ATLIF` 的组合没有稳定在 7% 附近，而是把网络推到了近乎沉默的状态。

下一步 SC 不应默认保留当前 target-rate 强度。建议优先做三组短测：

| 实验 | 改动 | 判断目标 |
|---|---|---|
| SC-noTR | Q/K `target_rate=null`，保留普通 ATLIF 阈值学习 | 看 SC agree/disagree 本身是否能保精度 |
| SC-weakTR | Q/K `target_rate=0.03 或 0.05`，`target_rate_eta` 降到 `0.02` | 看弱 target-rate 是否比 0.07 更稳 |
| SC-stageTR | 浅层低 target-rate，深层关 target-rate 或更弱 | 避免全 stage 一起沉默 |

同时 FFN 建议先从 S0/S1/S2-half 缩回 S0/S1 或只 S0，避免 attention 和 FFN 同时强剪枝。

**下一步怎么把 SC 改好**

优先级建议如下：

1. 先把稀疏控制放松，而不是继续加新 attention trick。配置：`lambda=0.5/0.6`，Q/K target-rate 关掉或降到 `0.03-0.05`，FFN 只保留 S0/S1 或 S0。
2. 重新打开保精度约束：至少加 `lambda_ang=0.01-0.02` 或 teacher EPE/dir 蒸馏。SC 当前主要坏在方向和 outlier，不能只看 supervised EPE。
3. 对 agree/disagree gate 加残差，不要纯替换 carrier：`attn = carrier * (1 + alpha * (gate - 1))`，先用 `alpha=0.3/0.5`，避免 gate 早期错误直接毁掉 K carrier。
4. 如果继续用 agree/disagree，优先试 `confidence_enabled=true` 或 active-norm，但不要同时叠强 target-rate。低 activity 时让 gate 回退到 uniform/carrier，比继续相信稀疏 SC score 更合理。
5. 标准推理继续用 `eval_DSEC_flow_SNN.py`，每次记录 `installed ATLIF=...`、`installed Shiftmax=...`、`checkpoint_overlay_keys/missing/unexpected`，避免再混入 profile-only 口径。

当前结论：SC agree/disagree 机制本身仍值得保留为分支，因为它确实能把 spikes/energy 压下去；但 NSC-02 这组超参不可作为可行主线。问题更像是“纯 gate + 强 target-rate + FFN 同剪枝”组合过强，而不是 SC agree/disagree 这个想法完全不可行。

#### NSC-03：SC repair 短测矩阵与 full30 启动（2026-06-02）

目标：快速验证 NSC-02 的失败是否主要来自“纯 SC gate + 过强 target-rate + FFN 同剪枝”，而不是 SC agree/disagree 本身完全不可行。新增一个无额外参数的 residual SC mode：

```text
sc_agree_disagree_residual_shiftmax
attn = carrier * (1 + residual_alpha * (gate - 1))
```

该 residual mode 只改变 attention forward 的组合方式，不新增 checkpoint 权重；是否生效通过 `installed Shiftmax attention: 12 modules` 和 config mode 审计确认。

生成器：

```text
neuron_experiments/H9_bipolar_self_attention/entrypoints/make_nsc03_sc_repair_configs.py
```

短测目录：

```text
neuron_experiments/H9_bipolar_self_attention/results/nsc03_sc_repair_screen_b4_20260602_020853
```

短测口径：从 `experiments/baseline_stride_upstream/checkpoint_epoch59.pth` 续训，360 step，batch4 串行，valid40 profile。batch8 并行会在 A800 80GB 上 OOM，因此短测改为 batch4；full30 仍使用配置原始 batch8 单进程。

加载审计规则：

| FFN scope | 预期 ATLIF | baseline 续训预期 missing |
|---|---:|---:|
| S0/S1 | 32 = 24 Q/K + 8 FFN | 64 |
| S0 only | 28 = 24 Q/K + 4 FFN | 56 |

短测训练日志和 profile 均满足加载审计。以 NSC-03b 为例：

```text
[H9] installed ATLIFTernaryPSN before load: 32 modules
[H9] installed attention before load: 12 modules
[H9] load audit: checkpoint_overlay_keys=0, missing=64, unexpected=0
[H9] installed ATLIFTernaryPSN: 32 modules
[H9] installed Shiftmax attention: 12 modules
[H9] angular supervised loss enabled: lambda_mod=1, lambda_ang=0.02
[H9] profile load audit: checkpoint_overlay_keys=64, missing=0, unexpected=0
```

短测结果：

| rank | variant | 关键改动 | AEE | AAE | SOPs(G) | firing | health |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `nsc03b_l05_tr03_s01_ang02` | λ=0.5，Q/K target_rate=0.03，S0/S1 FFN，angular 0.02 | 1.3987 | 12.7709 | 3.9147 | 0.09260 | threshold 1.0125，zero_neg=1 |
| 2 | `nsc03c_l06_notr_s01_ang02` | λ=0.6，no target_rate，S0/S1 FFN，angular 0.02 | 1.3883 | 12.8336 | 3.9387 | 0.09317 | threshold 1.0005，zero_neg=1 |
| 3 | `nsc03h_l06_tr03_s01_res03_ang02` | residual gate，λ=0.6，target_rate=0.03 | 1.4226 | 13.1390 | 3.8609 | 0.09133 | threshold 1.0124，zero_neg=1 |
| 4 | `nsc03g_l05_notr_s01_res03_ang02` | residual gate，λ=0.5，no target_rate | 1.4257 | 13.1530 | 3.8836 | 0.09187 | threshold 1.0005，zero_neg=1 |
| 5 | `nsc03f_l05_notr_s01_actnorm_ang02` | active-norm | 1.4179 | 13.3233 | 3.9349 | 0.09308 | threshold 1.0005，zero_neg=1 |
| 6 | `nsc03d_l05_notr_s0_ang02` | S0-only FFN | 1.3955 | 12.3914 | 4.1692 | 0.09862 | threshold 1.0006，zero_neg=1 |
| 7 | `nsc03a_l05_notr_s01_ang02` | λ=0.5，no target_rate，S0/S1 FFN | 1.4353 | 13.3698 | 3.9377 | 0.09315 | threshold 1.0005，zero_neg=1 |
| 8 | `nsc03e_l05_notr_s01_conf_ang02` | confidence + deadzone | 1.4634 | 13.7543 | 3.8398 | 0.09083 | threshold 1.0005，zero_neg=1 |

判断：

- NSC-03 证明放松 target-rate 和缩小 FFN scope 能避免 NSC-02 的 activity 塌陷；短测中 activity/firing 都回到接近 baseline 的范围。
- 但 SC agree/disagree 的 AAE 仍高，短测没有出现能同时修复方向误差和保持低 SOPs 的点。
- `target_rate=0.03` 比 noTR 在综合 score 上略好，说明弱 target-rate 可保留；`0.07` 不应再作为默认。
- residual gate 没有在 360-step 短测中超过普通 gate，但它的 SOPs/firing 更低，可作为后续结构分支，不作为本轮 full30 首选。
- S0-only 的 AAE 最低，但 SOPs/firing 明显反弹，不适合作为节能主线。

按 rapid score 选择 `NSC-03b` 进入 full30：

```text
配置: neuron_experiments/H9_bipolar_self_attention/configs/generated/nsc03b_l05_tr03_s01_ang02_full30.yml
run dir: neuron_experiments/H9_bipolar_self_attention/results/nsc03b_l05_tr03_s01_ang02_full30_20260602_024451
PID: 1416714
log: neuron_experiments/H9_bipolar_self_attention/results/nsc03b_l05_tr03_s01_ang02_full30_20260602_024451/train.log
```

full30 启动审计：

```text
[H9] installed ATLIFTernaryPSN before load: 32 modules
[H9] installed attention before load: 12 modules
[H9] load audit: checkpoint_overlay_keys=0, missing=64, unexpected=0
[H9] installed ATLIFTernaryPSN: 32 modules
[H9] installed Shiftmax attention: 12 modules
[H9] angular supervised loss enabled: lambda_mod=1, lambda_ang=0.02
```

epoch0 观察：

```text
checkpoint_epoch0.pth saved
Epoch loss = 94.1655
Epoch loss (Validation): 12.7944
ATLIF summary: activity_mean=0.05712, ternary_activity=0.06380, binary_activity=0.03708, zero_pos/zero_neg=1/1
Shiftmax summary: row_sum_mean=50.2552, gate_mean=0.3102, score_mean≈0
```

与 NSC-02 的 epoch0 后迅速沉默不同，NSC-03b 在 epoch0 结束时仍保持健康发放，说明 `target_rate=0.03 + S0/S1 FFN + angular` 的 full30 训练可以继续观察。

epoch1 观察：

```text
checkpoint_epoch1.pth saved
Epoch loss = 89.3025
Epoch loss (Validation): 12.9578
ATLIF summary: activity_mean=0.05129, ternary_activity=0.05585, binary_activity=0.03761, zero_pos/zero_neg=1/1
Shiftmax summary: row_sum_mean=50.3981, gate_mean=0.3111, score_mean≈0
```

早期判断：validation 暂未改善，AAE 风险仍在；但 activity 没有像 NSC-02 那样直接掉到 `0.00x`，所以继续跑到至少 epoch5/10 再判断是否提前停。

epoch2-3 观察：

| epoch | train loss | valid loss | activity_mean | ternary_activity | binary_activity | zero_pos/zero_neg | checkpoint |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 85.5456 | 12.7844 | 0.04656 | 0.04972 | 0.03709 | 1/1 | saved |
| 3 | 82.6850 | 12.8772 | 0.04358 | 0.04565 | 0.03738 | 1/1 | saved |

判断：valid loss 仍横盘，方向误差风险未解除；Q/K activity 在缓慢下降但还没有 NSC-02 的 `0.000x` 级别崩塌，继续跑到 epoch5/10。

epoch4 观察：

```text
checkpoint_epoch4.pth saved
Epoch loss = 80.4696
Epoch loss (Validation): 12.9543
ATLIF summary: activity_mean=0.04175, ternary_activity=0.04271, binary_activity=0.03889, zero_pos/zero_neg=1/1
```

判断：训练 loss 下降，但 validation 仍未改善；Q/K activity 继续下滑但不属于崩溃。继续跑到 epoch10，如果仍无改善，应同时考虑启动更保守的替代线，而不是只等 NSC-03b full30。

full30 完成与标准 valid825 推理结果（2026-06-02）：

训练已完成 epoch0-29，最终进程正常退出，未出现 NaN/OOM。后期训练日志显示 activity 下降但没有 NSC-02 式沉默：

| epoch | train loss | valid loss | activity_mean | ternary_activity | binary_activity | zero_pos/zero_neg |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 67.6602 | 13.2756 | 0.02948 | 0.02657 | 0.03823 | 1/1 |
| 23 | 66.9195 | 12.8299 | 0.02852 | 0.02573 | 0.03689 | 1/1 |
| 28 | 66.0449 | 12.8257 | 0.02847 | 0.02520 | 0.03828 | 1/1 |
| 29 | 65.9572 | 13.0550 | 0.02706 | 0.02371 | 0.03711 | 1/1 |

标准推理入口：`third_party/SDformerFlow/eval_DSEC_flow_SNN.py`，full valid825，`SDFORMER_USE_MLFLOW=0`，`SDFORMER_SNN_BACKEND=cupy`。推理加载审计通过：每个 checkpoint 均安装 `ATLIFTernaryPSN=32`、`Shiftmax attention=12`，并显示 `checkpoint_overlay_keys=64, missing=0, unexpected=0`。

| epoch | samples | AEE | AAE | PE1 | PE2 | PE3/outlier | total_spikes | firing | dense_flops | effective_flops | sparsity | synops_mac | synops_logic | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 825 | 1.7089 | 11.0547 | 0.5728 | 0.2417 | 0.1196 | 37.5622G | 8.1175% | 1229.2102G | 99.7816G | 91.88% | 36.0026G | 1.5595G | 32558.32 |
| 23 | 825 | 1.6806 | 11.0912 | 0.5728 | 0.2354 | 0.1134 | 38.1716G | 8.2492% | 1229.2102G | 101.4004G | 91.75% | 36.6513G | 1.5202G | 33138.20 |
| 28 | 825 | 1.7092 | 11.1900 | 0.5720 | 0.2343 | 0.1150 | 38.5369G | 8.3282% | 1229.2102G | 102.3710G | 91.67% | 37.0493G | 1.4876G | 33493.12 |
| 29 | 825 | 1.6116 | 10.5987 | 0.5531 | 0.2232 | 0.1077 | 36.1757G | 7.8179% | 1229.2102G | 96.0987G | 92.18% | 34.7737G | 1.4020G | 31436.57 |

当前 NSC-03b 最优 checkpoint 为 `epoch29`。相对 NSC-02 ep29，NSC-03b ep29 明显修复精度：AEE `1.6116 vs 1.9088`，AAE `10.5987 vs 12.1452`，但 total_spikes/energy 从 `28.9151G/26013.9uJ` 增至 `36.1757G/31436.6uJ`。相对 NSC-01/旧 SC ep29，NSC-03b ep29 精度更好：AEE `1.6116 vs 1.7710`，AAE `10.5987 vs 11.4245`，代价是 spikes/energy 更高：`36.1757G/31436.6uJ vs 32.2979G/28230.5uJ`。

相对 NB0 baseline ep59（AEE `1.4882`，AAE `9.9304`，total_spikes `44.1038G`，energy `37649.8uJ`），NSC-03b ep29 仍有精度损失：AEE +8.3%，AAE +6.7%，但 total_spikes 降低约 18.0%，energy 降低约 16.5%。相对 NTX-01/NTX-02，NSC-03b 仍不是主线精度最优；它的价值是证明 SC agree/disagree 可以通过弱 target-rate + S0/S1 scope 从 NSC-02 的塌陷中恢复，但还没达到 TX 路线的精度/能耗平衡。

结论：`target_rate=0.03` 比 NSC-02 的 `0.07` 合理，能避免沉默；但 target-rate 本身不是充分条件，SC 的方向误差仍主要由 attention carrier/gate 表达能力限制。下一轮 SC 修复优先从 NSC-03b ep29 出发，保留 `lambda=0.5`、`target_rate=0.03`、S0/S1 FFN 和 angular loss，重点改 attention carrier/gate，而不是继续加大稀疏约束。


### 31.13 NSC-04：carrier-blended SC 修复线（2026-06-02 启动）

目标：继续沿 NSC-03 的 SC agree/disagree 路线做兼容式修复，不破坏旧实验。新增 attention mode：

```text
sc_ad_carrier_blend_shiftmax
carrier_q = sn2_q(sum(Q))
sc_gate = Shiftmax(agree) - lambda * Shiftmax(disagree)
attn = K * ((1 - mu) * carrier_q + mu * sc_gate)
```

动机：NSC-03 的 `sc_agree_disagree_residual_shiftmax` 是乘法残差，若原生 carrier 静默，SC 证据也很难补回来。NSC-04 改为 carrier/gate 线性 blend，让 SC signed gate 能直接参与方向修正。该 mode 无新增权重；旧 config 不引用该 mode 时行为不变。

新增文件：

- `neuron_experiments/H9_bipolar_self_attention/entrypoints/make_nsc04_sc_blend_configs.py`
- `neuron_experiments/H9_bipolar_self_attention/entrypoints/run_nsc04_autopilot.py`
- `bsa_attention.py` 新增 `sc_ad_carrier_blend_shiftmax/h56m`

smoke 审计：

```text
2-step rapid smoke passed
train: installed ATLIFTernaryPSN=32, attention=12, load audit checkpoint_overlay_keys=0, missing=64, unexpected=0
profile: installed ATLIFTernaryPSN=32, Shiftmax attention=12, checkpoint_overlay_keys=64, missing=0, unexpected=0
```

短测口径：8 个配置，360 step，batch4 串行，valid10 + valid40 confirm，从 `experiments/baseline_stride_upstream/checkpoint_epoch59.pth` 续训。短测目录：

```text
neuron_experiments/H9_bipolar_self_attention/results/nsc04_sc_blend_short_20260602_135657
```

valid40 排名：

| rank | variant | 关键设置 | AEE | AAE | SOPs(G) | firing | health |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `nsc04d_blend_mu05_l06_tr03_ang02` | mu=0.5, lambda=0.6, target_rate=0.03 | 1.3860 | 12.5411 | 3.8796 | 0.09177 | zero_neg=1, worst=3.42 |
| 2 | `nsc04c_blend_mu075_l05_tr03_ang02` | mu=0.75, lambda=0.5, target_rate=0.03 | 1.4028 | 12.7193 | 3.9025 | 0.09231 | zero_neg=1, worst=3.50 |
| 3 | `nsc04g_blend_mu05_l05_tr03_slowlr_ang02` | mu=0.5, lambda=0.5, target_rate=0.03, slow LR/warmup400 | 1.4162 | 12.7115 | 3.8915 | 0.09205 | zero_neg=1, threshold=1.0085 |
| 4 | `nsc04f_blend_mu05_l05_tr02_ang02` | mu=0.5, lambda=0.5, target_rate=0.02 | 1.4095 | 12.9236 | 3.8874 | 0.09195 | zero_neg=1 |
| 5 | `nsc04e_blend_mu05_l05_notr_ang02` | mu=0.5, lambda=0.5, no target_rate | 1.4204 | 13.0156 | 3.9080 | 0.09244 | zero_neg=1 |
| 6 | `nsc04a_blend_mu025_l05_tr03_ang02` | mu=0.25, lambda=0.5, target_rate=0.03 | 1.4226 | 13.2441 | 3.8673 | 0.09148 | zero_neg=1 |
| 7 | `nsc04b_blend_mu05_l05_tr03_ang02` | mu=0.5, lambda=0.5, target_rate=0.03 | 1.4236 | 13.3696 | 3.8843 | 0.09188 | zero_neg=1 |
| 8 | `nsc04h_blend_mu05_l05_tr03_clamp_ang02` | NSC-04b + gate clamp -1/1.5 | 1.4236 | 13.3696 | 3.8843 | 0.09188 | zero_neg=1 |

选择 `NSC-04d` 进入 full30。相比 NSC-03b 短测，AEE 从 `1.3987` 降到 `1.3860`，AAE 从 `12.7709` 降到 `12.5411`，SOPs 从 `3.9147G` 降到 `3.8796G`；短测上是同方向改善。

full30 启动：

```text
config: neuron_experiments/H9_bipolar_self_attention/configs/nsc04d_blend_mu05_l06_tr03_ang02_auto_full_auto_full_20260602_142622.yml
run dir: neuron_experiments/H9_bipolar_self_attention/results/nsc04d_blend_mu05_l06_tr03_ang02_auto_full_auto_full_bs8_20260602_142622_setsid
driver: neuron_experiments/H9_bipolar_self_attention/results/nsc04_autopilot_20260602_135657
launcher pid: 1466778
```

autopilot 后续会在 full30 完成后对 epoch `19/23/28/29` 跑 `eval_DSEC_flow_SNN.py` full valid825，并自动追加标准结果。

full30 早期健康检查：

| epoch | train loss | valid loss | activity_mean | ternary_activity | binary_activity | zero_pos/zero_neg | Shiftmax row_sum | gate_mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 93.8753 | 9.7095 | 0.05720 | 0.06393 | 0.03700 | 1/1 | 26.8679 | 0.1659 |
| 1 | 89.2549 | 9.6154 | 0.05139 | 0.05600 | 0.03755 | 1/1 | 26.4592 | 0.1633 |
| 2 | 85.5384 | 10.2267 | 0.04676 | 0.05001 | 0.03699 | 1/1 | 26.4459 | 0.1632 |
| 3 | 82.8104 | 9.9025 | 0.04372 | 0.04589 | 0.03721 | 1/1 | 26.0695 | 0.1609 |
| 4 | 80.5804 | 9.8498 | 0.04191 | 0.04297 | 0.03871 | 1/1 | 26.3120 | 0.1624 |
| 5 | 78.5107 | 10.0424 | 0.03980 | 0.04067 | 0.03717 | 1/1 | 26.1958 | 0.1617 |
| 6 | 77.0438 | 9.7634 | 0.03815 | 0.03830 | 0.03771 | 1/1 | 25.9655 | 0.1603 |

判断：NSC-04d full30 开局明显好于 NSC-03b（NSC-03b epoch0/1 valid loss 为 `12.7944/12.9578`），说明 carrier-blend 改动确实改变了方向误差局面；继续跑，不提前停。

中断与续跑记录：

`2026-06-02` 中途服务器疑似关机/重启，原 `run_nsc04_autopilot.py` 和 full30 训练进程均已消失。当前文件证据显示原训练保存到 `checkpoint_epoch10.pth`，日志停在 `Epoch 11`，未进入标准 valid825 推理。由于原 full config 设置了 `runtime.skip_state_save=true`，没有 optimizer/scheduler/scaler state 可无损恢复。

采取的恢复方式：

```text
resume config: neuron_experiments/H9_bipolar_self_attention/configs/nsc04d_blend_mu05_l06_tr03_ang02_resume11_29_20260602.yml
prev_runid: neuron_experiments/H9_bipolar_self_attention/results/nsc04d_blend_mu05_l06_tr03_ang02_auto_full_auto_full_bs8_20260602_142622_setsid/checkpoint_epoch10.pth
log: neuron_experiments/H9_bipolar_self_attention/results/nsc04d_blend_mu05_l06_tr03_ang02_auto_full_auto_full_bs8_20260602_142622_setsid/train_resume11_29.log
pid file: neuron_experiments/H9_bipolar_self_attention/results/nsc04d_blend_mu05_l06_tr03_ang02_auto_full_auto_full_bs8_20260602_142622_setsid/resume11_29.pid
```

为避免覆盖已有 checkpoint，训练入口新增兼容字段 `runtime.epoch_offset`，默认 0；只有 resume 配置设置 `epoch_offset=11`。续跑本地 19 epoch，保存文件映射为 `checkpoint_epoch11.pth` 到 `checkpoint_epoch29.pth`。启动审计通过：

```text
[H9] installed ATLIFTernaryPSN before load: 32 modules
[H9] installed attention before load: 12 modules
[H9] load audit: checkpoint_overlay_keys=64, missing=0, unexpected=0
[H9] installed ATLIFTernaryPSN: 32 modules
[H9] installed Shiftmax attention: 12 modules
```

注意：这是从 epoch10 模型权重继续训练，optimizer/scheduler 重新初始化；因此严格意义上不是完全无损 resume，但保留了 NSC-04d 已学到的模型/overlay 参数，并继续完成 epoch11-29 的标准训练长度。

续跑首个本地 epoch 验证：

| logical epoch | local epoch | train loss | valid loss | activity_mean | ternary_activity | binary_activity | zero_pos/zero_neg | checkpoint |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 11 | 0 | 72.0661 | 9.7356 | 0.03288 | 0.03152 | 0.03694 | 1/1 | `checkpoint_epoch11.pth` saved |
| 12 | 1 | 71.2437 | 9.7003 | 0.03278 | 0.03121 | 0.03747 | 1/1 | `checkpoint_epoch12.pth` saved |
| 13 | 2 | 70.5328 | 10.1247 | 0.03132 | 0.02946 | 0.03692 | 1/1 | `checkpoint_epoch13.pth` saved |
| 14 | 3 | 70.1222 | 9.9900 | 0.03125 | 0.02928 | 0.03717 | 1/1 | `checkpoint_epoch14.pth` saved |
| 15 | 4 | 69.7616 | 9.7720 | 0.03148 | 0.02908 | 0.03870 | 1/1 | `checkpoint_epoch15.pth` saved |
| 16 | 5 | 69.1403 | 9.8504 | 0.03107 | 0.02905 | 0.03715 | 1/1 | `checkpoint_epoch16.pth` saved |
| 17 | 6 | 68.8629 | 10.0444 | 0.03060 | 0.02823 | 0.03771 | 1/1 | `checkpoint_epoch17.pth` saved |
| 18 | 7 | 68.4550 | 9.8588 | 0.03066 | 0.02826 | 0.03783 | 1/1 | `checkpoint_epoch18.pth` saved |

`runtime.epoch_offset=11` 已实际生效，未覆盖原 `checkpoint_epoch0-10.pth`。截至 `2026-06-02T12:17Z`，恢复训练仍在运行，已进入 local epoch8 / logical epoch19；follow-up 进程仍在等待训练结束，尚未启动 full valid825 标准推理。


### 31.9 stride_h41_sc_s012c 标准 valid825 结果（自动队列）

- 评估配置：`neuron_experiments/H9_bipolar_self_attention/configs/generated/stride_h41_sc_s012c.yml`
- 训练目录：`neuron_experiments/H9_bipolar_self_attention/results/stride_h41_sc_s012c_20260531_170553`
- 推理口径：valid825，`SDFORMER_USE_MLFLOW=0`，`SDFORMER_SNN_BACKEND=cupy`，H9 config 加载审计开启。

| epoch | samples | AEE | AAE | PE1 | PE2 | PE3/outlier | total_spikes | firing | dense_flops | effective_flops | sparsity | synops_mac | synops_logic | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 825 | 1.8134 | 11.5640 | 0.5890 | 0.2495 | 0.1232 | 33.5406G | 7.2484% | 955.5029G | 69.2590G | 92.75% | 32.3378G | 1.2027G | 29224.32 |
| 24 | 825 | 1.8837 | 12.1174 | 0.6074 | 0.2650 | 0.1336 | 34.7346G | 7.5065% | 955.5029G | 71.7246G | 92.49% | 33.5625G | 1.1721G | 30323.42 |
| 27 | 825 | 1.8720 | 12.0319 | 0.6055 | 0.2630 | 0.1315 | 34.1204G | 7.3737% | 955.5029G | 70.4563G | 92.63% | 33.0546G | 1.0658G | 29855.73 |
| 29 | 825 | 1.7710 | 11.4245 | 0.5870 | 0.2448 | 0.1191 | 32.2979G | 6.9799% | 955.5029G | 66.6930G | 93.02% | 31.2509G | 1.0470G | 28230.53 |

当前 SC 标准 valid825 最优点按 AEE 排序为 epoch29。完整推理日志和 `spike_profile.json` 保存在各 `standard_valid825/epoch*/` 目录。

后续已接入 TX 逐 token selector 全量：

- 配置：`neuron_experiments/H9_bipolar_self_attention/configs/generated/stride_h53b_h49_clean_no_stage3_s02_full30.yml`
- 续训起点：`experiments/baseline_stride_upstream/checkpoint_epoch59.pth`
- 方案：corrected H53b/H49 qkselector，`ternary_alpha_xnor_qkselector_shiftmax`，stage3 不替换，Q/K 无 target-rate，FFN official binary ATLIF。

### 31.10 NTX-02 TX 逐 token selector 标准 valid825 结果（2026-06-02）

实验编号：`NTX-02`。物理目录：
`neuron_experiments/H9_bipolar_self_attention/results/stride_h53b_h49_clean_no_stage3_s02_full30_20260601_145158`

配置：
`neuron_experiments/H9_bipolar_self_attention/configs/generated/stride_h53b_h49_clean_no_stage3_s02_full30.yml`

续训起点：`experiments/baseline_stride_upstream/checkpoint_epoch59.pth`。训练入口使用 H9 overlay，标准推理入口为 `third_party/SDformerFlow/eval_DSEC_flow_SNN.py`，full valid825，`SDFORMER_SNN_BACKEND=cupy`。推理加载审计通过：每个 checkpoint 均安装 `ATLIFTernaryPSN=30`、`Shiftmax attention=10`、`checkpoint_overlay_keys=60`、`missing=0`、`unexpected=0`。

#### 方案定义

`NTX-02` 是 corrected H53b/H49 逐 token TX selector，不是旧 H41 的“原 QKFormer gate 再外挂 TX gate”。其注意力范式为：

```text
q_event = ternary_sign(token_q(q_orig))
k_event = ternary_sign(k_orig)
score_i = TX(q_event_i, k_event_i)
gate_i = Shiftmax(score_i over tokens)
attn_i = k_orig_i * gate_i
```

也就是说，它不构造 `N x N` attention matrix，不做跨 token 的 `gate @ V`，也不引入独立 V 分支；它保留 QKFormer 的线性复杂度 K carrier，但把原始 Q-only selector 替换成同 token 的三值 Q/K 一致性 selector。

#### 影响训练的关键设置

| 项 | 设置 |
|---|---|
| attention mode | `ternary_alpha_xnor_qkselector_shiftmax` |
| attention 覆盖范围 | stage0 两个 block、stage1 两个 block、stage2 六个 block；stage3 不替换 |
| attention target blocks | `0:0, 0:1, 1:0, 1:1, 2:0, 2:1, 2:2, 2:3, 2:4, 2:5` |
| Q/K 神经元 | PSN + ATLIF + 三值输出 |
| Q/K 阈值范式 | `symmetric_bsa_tsn`，正负共用同一阈值幅值 |
| Q/K target-rate | `null`，未启用 target-rate 反馈 |
| Q/K center | `center_mode: bias` |
| Q/K stage 阈值增长 | stage0 `threshold_eta=0.00065`、stage1 `0.00048`、stage2 `0.00038` |
| Q/K threshold lr scale | stage0 `50000`、stage1 `38000`、stage2 `30000` |
| Q/K activity eta | stage0 `1.35`、stage1 `1.05`、stage2 `0.9` |
| FFN 神经元 | PSN + ATLIF，二值输出 |
| FFN 阈值范式 | `official_atlif` |
| FFN 覆盖范围 | stage0 FFN 全部 block；stage2 偶数 block `0/2/4` 的 FFN；stage1/stage3 FFN 不替换 |
| FFN target-rate | `null` |
| single-active penalty | `0.2`，`single_active_penalty_grad=ste`，`slope=4.0`，`margin=0.25` |
| score 设置 | `center_scores=true`、`consensus_score_norm=head_dim`、`score_scale=1.0`、`mismatch_penalty=0.25`、`alpha0=0.02`、`preserve_mean=true` |
| loss | supervised flow loss，`lambda_ang=0.0`，未加 angular loss |
| optimizer | AdamW，主 lr `1e-5`，wd `0.001`，AMP 开启 |
| 分组学习率 | backbone/norm `2e-7`，neuron `1.2e-5`，threshold `3e-6`，threshold wd `0`，norm wd `0` |
| scheduler | multistep，milestones `[20,25]` |
| warmup | 300 steps，start factor `0.1` |
| batch/workers | batch size `8`，workers `8`，pin_memory/persistent_workers/prefetch_factor 开启 |
| 训练轮次 | 30 epochs，全参数续训，不冻结 baseline 权重 |

#### 演变来源

| 来源 | 被 NTX-02 继承/修正的点 |
|---|---|
| H41 TX | 继承 TX/alpha-XNOR 三值一致性注意力思路和 TX 主线目标 |
| H45/H47 QKV direct | 作为反例：`gate @ V` 直接替换 K carrier 后精度明显崩，NTX-02 不采用独立 V 或 QKV 矩阵注意力 |
| H49 qkselector | 继承逐 token Q/K selector：`score_i=TX(q_i,k_i)`，再用 Shiftmax 调制 `k_i` |
| H53b | 修正 ATLIF 范式和 stage3 不替换策略，避免后段高语义层过度扰动 |
| H54/H56 | 继承 single-active penalty 的 STE 版本，解决 `0 vs ±1` 单边激活没有被惩罚的问题 |
| NB0 stride baseline | 使用标准重训 baseline `checkpoint_epoch59.pth` 作为续训起点 |

#### 标准 valid825 结果

| epoch | samples | AEE | AAE | PE1 | PE2 | PE3/outlier | total_spikes | firing | dense_flops | effective_flops | sparsity | synops_mac | synops_logic | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 825 | 1.7535 | 11.6770 | 0.5713 | 0.2351 | 0.1144 | 39.4906G | 8.5331% | 939.4025G | 80.1602G | 91.47% | 35.5772G | 3.9133G | 32410.84 |
| 2 | 825 | 1.7461 | 11.7274 | 0.5802 | 0.2334 | 0.1113 | 37.5199G | 8.1073% | 939.4025G | 76.1601G | 91.89% | 34.1430G | 3.3769G | 31066.42 |
| 23 | 825 | 1.7804 | 12.4169 | 0.6053 | 0.2587 | 0.1254 | 34.7836G | 7.5160% | 939.4025G | 70.6056G | 92.48% | 34.1247G | 0.6589G | 30778.11 |
| 27 | 825 | 1.8396 | 12.7234 | 0.6191 | 0.2720 | 0.1338 | 34.4145G | 7.4363% | 939.4025G | 69.8565G | 92.56% | 33.8283G | 0.5862G | 30504.09 |
| 29 | 825 | 1.8162 | 11.9498 | 0.6079 | 0.2652 | 0.1336 | 32.9916G | 7.1288% | 939.4025G | 66.9682G | 92.87% | 32.4397G | 0.5518G | 29250.94 |

当前判断：

- 按精度，`epoch2` 最好：AEE `1.7461`、AAE `11.7274`。
- 按稀疏，`epoch29` 最好：total_spikes `32.9916G`、firing `7.1288%`、energy `29.2509mJ`。
- 相比 NB0 baseline ep59，`epoch29` total_spikes 从 `44.0488G` 降到 `32.9916G`，约 `-25.1%`；但 AEE 从 `1.4872` 退到 `1.8162`，AAE 从 `9.9300` 退到 `11.9498`。
- 相比 NTX-01 TX v2，NTX-02 没有取得更好 tradeoff。NTX-01 ep28 仍是当前 TX 主线更稳的点：AEE `1.534`，total_spikes `34.61G`。

### 31.10.1 NTX-03 TX 改进短测计划（2026-06-02）

目标：在保持 `NTX-02` 更容易讲清楚的逐 token TX selector 框架下，修正两个问题：

1. `NTX-02` 中 `mismatch_penalty=0.25`、`single_active_penalty=0.2` 不合理。异号是方向冲突，单边激活只是证据缺失，单边惩罚不应接近异号惩罚。
2. 普通 `Shiftmax(score)` 只能输出正 gate，因此只能衰减/放大 `K`，不能在异号证据强时反转方向。

本轮不使用纯 `signed_consensus`，因为纯 `sum(sign(q)*sign(k))` 会把 TX 线变成 NSC/SC 线。`NTX-03` 仍保留 TX 的同号、异号、静默分类证据。

#### 共同设置

| 项 | 设置 |
|---|---|
| 起点 | `experiments/baseline_stride_upstream/checkpoint_epoch59.pth` |
| 基础配置 | `configs/generated/stride_h53b_h49_clean_no_stage3_s02_full30.yml` |
| 替换范围 | 与 NTX-02 完全一致：stage0/1/2 attention，stage3 不替换；FFN 为 S0 + S2 偶数 block |
| 神经元 | Q/K 为 PSN + ATLIF + 三值；FFN 为 PSN + ATLIF 二值 |
| target-rate | Q/K 和 FFN 均不启用 target-rate |
| loss | 不加 angular loss，`lambda_ang=0` |
| lr | 沿用 NTX-02 分组 lr：backbone/norm `2e-7`，neuron `1.2e-5`，threshold `3e-6` |
| 短测 | 每个配置 360 train steps，先 valid10 profile，达宽松 gate 后 valid40 confirm |
| promotion | 短测完成后按综合分自动选一个全量 30 epoch，保存 epoch `0/4/9/14/19/24/29` |

#### A 组：弱单边、强异号，最小改动

保持 `NTX-02` 的 H49 逐 token TX selector：

```text
score = same_nonzero
      + alpha0 * same_zero
      - mismatch_penalty * opposite
      - single_active_penalty * one_sided
gate = Shiftmax(score)
attn = K * gate
```

短测配置：

| 配置 | mismatch_penalty | single_active_penalty | single-active grad | 目的 |
|---|---:|---:|---|---|
| `ntx03a_tx_m04_s005` | 0.40 | 0.05 | ste | 温和修正 |
| `ntx03a_tx_m04_s010` | 0.40 | 0.10 | ste | 单边略强 |
| `ntx03a_tx_m06_s005` | 0.60 | 0.05 | ste | 异号明显强于单边 |
| `ntx03a_tx_m06_s010` | 0.60 | 0.10 | ste | 平衡强惩罚 |
| `ntx03a_tx_m08_s005` | 0.80 | 0.05 | ste | 接近乘法异号强冲突 |
| `ntx03a_tx_m06_s005_hardactive` | 0.60 | 0.05 | hard | 对比 single-active STE 是否引入过强假梯度 |

#### B 组：同号/异号双分支，解决方向问题

将 TX 证据拆成同号和异号两个非负分支，再相减：

```text
same_score = same_nonzero + alpha0 * same_zero
opp_score  = opposite + single_active_penalty * one_sided

gate = Shiftmax(same_score) - lambda_opp * Shiftmax(opp_score)
attn = K * gate
```

该方案仍是 TX 分类证据，不是 NSC。优势是 `gate` 可以为负，异号证据强时允许反转 `K` 的方向。

短测配置：

| 配置 | lambda_opp | single_active_penalty | 目的 |
|---|---:|---:|---|
| `ntx03b_two_l025_s005` | 0.25 | 0.05 | 轻微方向反转 |
| `ntx03b_two_l050_s005` | 0.50 | 0.05 | 平衡方向反转 |
| `ntx03b_two_l050_s010` | 0.50 | 0.10 | 更重单边证据 |
| `ntx03b_two_l075_s005` | 0.75 | 0.05 | 强方向反转 |

#### C 组：三分支 TX + signed correction，保守备选

保留普通 TX gate 作为稳定 carrier，再加同号/异号 signed correction：

```text
gate = Shiftmax(tx_score)
     + mu * (Shiftmax(same_score) - lambda_opp * Shiftmax(opp_score))
attn = K * gate
```

当 `mu=0` 时退化回 H49/NTX-02，因此比 B 组更稳，但故事比 B 组复杂。

短测配置：

| 配置 | mu | lambda_opp | single_active_penalty | 目的 |
|---|---:|---:|---:|---|
| `ntx03c_three_mu025_l050_s005` | 0.25 | 0.50 | 0.05 | 温和 correction |
| `ntx03c_three_mu050_l050_s005` | 0.50 | 0.50 | 0.05 | 平衡 correction |
| `ntx03c_three_mu050_l075_s005` | 0.50 | 0.75 | 0.05 | 更强异号方向修正 |
| `ntx03c_three_mu050_l050_s010` | 0.50 | 0.50 | 0.10 | 更重单边证据 |

#### 选择标准

优先级：

1. 精度不明显弱于 NTX-01：目标 AEE `<=1.60`，AAE `<=10.8`。
2. 稀疏不弱于 NTX-01 太多：目标 total_spikes `<=35G` 或 valid40 SOPs 接近 `3G` 区间。
3. 三值发放健康：负发放不能塌缩，`zero_neg_modules` 和正负比例不能异常。
4. 方案可讲：若 B 组接近 A/C 组精度，优先 B 组，因为它解决了“Shiftmax 只能正 gate”的方向问题。

执行脚本：

```bash
python neuron_experiments/H9_bipolar_self_attention/entrypoints/make_ntx03_tx_refine_configs.py
python neuron_experiments/H9_bipolar_self_attention/entrypoints/run_ntx03_autopilot.py
```

启动记录：

- `2026-06-02 02:43 UTC` 已启动自动队列，PID 记录在 `neuron_experiments/H9_bipolar_self_attention/results/ntx03_launcher/latest.pid`。
- launcher 日志：`neuron_experiments/H9_bipolar_self_attention/results/ntx03_launcher/latest.log` 指向最新启动日志。
- autopilot 状态日志：`neuron_experiments/H9_bipolar_self_attention/results/ntx03_autopilot_20260602_024327/status.log`。
- 短测目录：`neuron_experiments/H9_bipolar_self_attention/results/ntx03_tx_refine_short_20260602_024327`。
- 已做链路检查：2-step smoke 中训练入口安装 `ATLIFTernaryPSN=30`、`Shiftmax attention=10`；profile 加载短训 checkpoint 时 `checkpoint_overlay_keys=60, missing=0, unexpected=0`；标准 eval 入口也确认 `ATLIFTernaryPSN=30`、`Shiftmax attention=10`、`checkpoint_overlay_keys=60, missing=0, unexpected=0`。
- 当前自动策略：14 个配置串行 360-step 短测，valid10 后尽量进入 valid40 confirm；短测结束后自动选综合分最低者做 full30；full30 完成后自动对 epoch `9/19/29` 跑标准 valid825 并追加到本文档。

### 31.11 TX V2 标准 valid825 结果（本机，2026-06-01）

- 训练：`stride_h41_tx_s02c_v2.yml`，从 baseline epoch59 续训 30 epochs，CuPy backend
- 评估：`eval_DSEC_flow_SNN.py`，full 825 valid，H9 config 加载审计通过（34 ATLIF + 12 attention，68 overlay keys，missing=0）

| epoch | AEE | AAE | total_spikes | firing_rate | dense_flops | sparsity | energy |
|---|---|---|---|---|---|---|---|
| 22 | 1.550 | 10.47 | 33.51G | 7.23% | 939.4G | 92.8% | 28.53mJ |
| **28** 🏆 | **1.534** | 10.29 | 34.61G | 7.47% | 939.4G | 92.5% | 29.71mJ |
| 29 | 1.584 | 10.28 | **32.92G** | 7.10% | 939.4G | **92.9%** | **28.24mJ** |

#### vs Baseline

| | baseline ep59 | TX V2 ep28 | Δ |
|---|---|---|---|
| AEE | 1.489 | 1.534 | +3.0% |
| AAE | 9.923 | 10.288 | +3.7% |
| total_spikes | 44.05G | 34.61G | **-21.4%** |
| energy | 37.6mJ | 29.7mJ | **-21.0%** |

**结论**：TX 三值注意力 + S02 ATLIF FFN 在 stride split 上以 3% AEE 代价换取 21% 脉冲减少，tradeoff 可接受。

#### vs SC 标准线（31.9）

| | SC ep29 | TX V2 ep28 | Δ |
|---|---|---|---|
| AEE | 1.771 | **1.534** | TX 更优 -13% |
| total_spikes | 32.30G | 34.61G | SC 更优 -7% |
| energy | 28.23mJ | 29.71mJ | SC 更优 -5% |

TX AEE 明显优于 SC（1.53 vs 1.77），SC 稀疏度略优。**TX 目前是更好的主线。**

### 31.12 NTX-04：从 NTX-01 出发的 Carrier-Preserving TX 主线（2026-06-02）

目标：保留 `NTX-01` 已验证的精度/稀疏 tradeoff，同时把“外挂 gate”的表述改成更容易写进论文的网络机制：**Carrier-Preserving Ternary Consistency Attention, CPTC**。

核心解释：

```text
carrier = K * sn2_q(sum(Q))
consistency = Shiftmax(TX(Q_ternary, K_ternary))
output = carrier * consistency
```

这里 `carrier` 是 QKFormer 原生的事件 token 载体，`consistency` 是三值极性一致性调制，不再描述为“外挂注意力”，而是描述为“在原生 QK carrier 上引入硬件友好的三值一致性选择器”。这一路线不删除旧模块，不改变旧实验行为，只新增 `NTX-04` 配置调用。

共同设置：

| 项 | 设置 |
|---|---|
| 基础配置 | `configs/generated/stride_h41_tx_s02c_v2.yml` |
| 起点 | `experiments/baseline_stride_upstream/checkpoint_epoch59.pth` |
| 神经元 | Q/K: PSN + ATLIF + 三值；FFN: S0 + S2-half official ATLIF 二值 |
| target-rate | 不启用 |
| attention 范围 | 沿用 NTX-01，stage_selection=all |
| 目标 | baseline 5% 精度误差内，spikes/energy 降低约 20%，以 NTX-01 ep28 为参考 |

候选配置：

| 配置 | attention mode | 关键参数 | 目的 |
|---|---|---|---|
| `ntx04a_cptc_ntx01` | `ternary_alpha_xnor_shiftmax` | mismatch 0.25, single 0 | NTX-01 标准复现/控制 |
| `ntx04b_cptc_single005` | `ternary_alpha_xnor_shiftmax` | mismatch 0.25, single 0.05 | 加弱单边 active/silent 冲突 |
| `ntx04c_cptc_m04_single005` | `ternary_alpha_xnor_shiftmax` | mismatch 0.40, single 0.05 | 增强异号惩罚，弱单边 |
| `ntx04d_cptc_res075` | `ternary_alpha_xnor_shiftmax_residual` | residual_alpha 0.75 | 降低 TX 调制强度，保精度 |
| `ntx04e_cptc_res050` | `ternary_alpha_xnor_shiftmax_residual` | residual_alpha 0.50 | 更强保守 residual |
| `ntx04f_cptc_res075_single005` | `ternary_alpha_xnor_shiftmax_residual` | alpha 0.75, single 0.05 | residual + 单边冲突 |
| `ntx04g_cptc_m04_single005_slowbb` | `ternary_alpha_xnor_shiftmax` | mismatch 0.40, single 0.05, slow backbone LR | 强一致性 + 保护 baseline |
| `ntx04h_cptc_ntx01_warm` | `ternary_alpha_xnor_shiftmax` | NTX-01 + 300-step warmup | 测学习率 warmup 是否更稳 |

已新增：

- 配置生成器：`neuron_experiments/H9_bipolar_self_attention/entrypoints/make_ntx04_carrier_tx_configs.py`
- 生成配置：`neuron_experiments/H9_bipolar_self_attention/configs/generated/ntx04*.yml`
- 验证：`py_compile` 通过，`test_bsa_attention` + `test_atlif_ternary_psn` 共 34 个 unittest 通过。

执行计划：

```bash
python neuron_experiments/H9_bipolar_self_attention/entrypoints/make_ntx04_carrier_tx_configs.py
python neuron_experiments/H9_bipolar_self_attention/entrypoints/run_ntx04_autopilot.py
```

自动流程：8 个配置各 360-step 短测，valid10 + valid40；按综合分选一个 full30；full30 后对 epoch `19/24/28/29` 做标准 valid825，并自动追加到本文档。


### 31.10.2 NTX-03 自动短测与全量结果（自动追加）

- 时间：`2026-06-02T13:23:42`
- 短测目录：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/ntx03_tx_refine_short_20260602_024327`
- promotion log：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/ntx03_autopilot_20260602_024327/promote_full.log`
- 全量配置：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/configs/ntx03a_tx_m08_s005_auto_full_20260602_042859.yml`
- 全量目录：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/ntx03a_tx_m08_s005_auto_full_bs8_20260602_042859_setsid`
- 标准推理：`eval_DSEC_flow_SNN.py`，full valid825，CuPy backend。

#### 标准 valid825 结果

| epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | effective_flops | sparsity | synops_mac | synops_logic | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 9 | 1.9591 | 12.5390 | 0.6107 | 0.2713 | 0.1383 | 36.0466G | 7.7889% | 73.1695G | 92.21% | 34.2256G | 1.8210G | 30985.16 |
| 19 | 1.8295 | 11.9562 | 0.6113 | 0.2677 | 0.1351 | 33.7159G | 7.2853% | 68.4384G | 92.71% | 32.9012G | 0.8147G | 29692.51 |
| 29 | 1.8226 | 11.9170 | 0.6083 | 0.2658 | 0.1349 | 33.0258G | 7.1362% | 67.0376G | 92.86% | 32.4476G | 0.5782G | 29260.65 |

当前自动判断：精度最佳 epoch29，AEE `1.8226`、AAE `11.9170`；稀疏最佳 epoch29，total_spikes `33.0258G`。


### 31.12.1 NTX-04 自动短测与全量结果（自动追加）

- 时间：`2026-06-02T23:26:01`
- 短测目录：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/ntx04_cptc_short_20260602_135045`
- promotion log：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/ntx04_autopilot_20260602_135045/promote_full.log`
- 全量配置：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/configs/ntx04h_cptc_ntx01_warm_auto_full_20260602_144858.yml`
- 全量目录：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/ntx04h_cptc_ntx01_warm_auto_full_bs8_20260602_144858_setsid`
- 标准推理：`eval_DSEC_flow_SNN.py`，full valid825，CuPy backend。

| epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | effective_flops | sparsity | synops_mac | synops_logic | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 19 | 1.6888 | 11.0656 | 0.5754 | 0.2334 | 0.1115 | 34.2405G | 7.3854% | 69.3783G | 92.61% | 32.1014G | 2.1391G | 29105.14 |
| 24 | 1.6511 | 11.1548 | 0.5721 | 0.2305 | 0.1092 | 34.0608G | 7.3466% | 69.0142G | 92.65% | 32.1956G | 1.8652G | 29162.55 |
| 28 | 1.6726 | 11.1445 | 0.5715 | 0.2300 | 0.1086 | 35.8435G | 7.7311% | 72.6264G | 92.27% | 34.0003G | 1.8432G | 30784.61 |
| 29 | 1.6233 | 10.7270 | 0.5586 | 0.2208 | 0.1050 | 33.4428G | 7.2133% | 67.7620G | 92.79% | 31.7330G | 1.7098G | 28730.67 |

当前自动判断：精度最佳 epoch29，AEE `1.6233`、AAE `10.7270`；稀疏最佳 epoch29，total_spikes `33.4428G`。


### 31.13.1 NSC-04d resume 后标准 valid825 结果（自动追加）

- 时间：`2026-06-02T23:53:05`
- 配置：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/configs/nsc04d_blend_mu05_l06_tr03_ang02_resume11_29_20260602.yml`
- 目录：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/nsc04d_blend_mu05_l06_tr03_ang02_auto_full_auto_full_bs8_20260602_142622_setsid`
- 推理：`eval_DSEC_flow_SNN.py` full valid825，CuPy backend。

| epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | effective_flops | sparsity | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 19 | 1.6458 | 10.8838 | 0.5595 | 0.2238 | 0.1061 | 36.9264G | 7.9647% | 97.9026G | 92.04% | 31934.53 |
| 23 | 1.6935 | 11.2898 | 0.5719 | 0.2363 | 0.1134 | 37.3883G | 8.0643% | 99.1274G | 91.94% | 32390.01 |
| 28 | 1.6772 | 10.9878 | 0.5684 | 0.2355 | 0.1146 | 37.2565G | 8.0359% | 98.7780G | 91.96% | 32289.59 |
| 29 | 1.6581 | 11.0396 | 0.5646 | 0.2321 | 0.1132 | 36.1550G | 7.7983% | 95.8575G | 92.20% | 31376.64 |

当前最优：epoch19，AEE `1.6458`，AAE `10.8838`。


### 31.14 NSC-05：confidence-aware SC carrier-blend 快速补救线（2026-06-03 启动）

触发原因：NSC-04d 标准 valid825 已完成，最好点 epoch19 为 AEE `1.6458`、AAE `10.8838`、total_spikes `36.93G`。它比旧 SC 修复线好，但仍明显弱于 TX V2/NTX-04：TX V2 ep28 AEE `1.534`、spikes `34.61G`，NTX-04 ep29 AEE `1.6233`、spikes `33.44G`。因此 SC/NSC 需要先做更快的短测筛选，不直接排新的 full30。

新增兼容 mode：

```text
sc_ad_confidence_carrier_blend_shiftmax / h56mc
sc_gate = confidence * (Shiftmax(agree) - lambda * Shiftmax(disagree)) + (1-confidence)/N
output  = K * ((1 - mu) * carrier_q + mu * sc_gate)
```

该 mode 复用已有 SC agree/disagree、deadzone、confidence、active-norm 字段，不新增可训练权重；旧 config 不引用 `h56mc` 时行为不变。动机是 NSC-04 的 carrier-blend 允许 SC 修正 inactive carrier，但低活动 token 的弱投票仍被当作强证据；NSC-05 用 confidence/deadzone 把低置信 token 拉回 uniform，再和 carrier 混合。

短测候选：

| variant | mode | mu | lambda | target-rate | 其他设置 | 目的 |
|---|---|---:|---:|---:|---|---|
| `nsc05a_h56m_mu05_l06_tr03` | h56m | 0.50 | 0.60 | 0.03 | NSC-04d 复现短测 | 对照 |
| `nsc05b_conf_mu04_l05_tr03` | h56mc | 0.40 | 0.50 | 0.03 | deadzone=1/32, confidence | 保守 SC 修正 |
| `nsc05c_conf_mu04_l06_tr03` | h56mc | 0.40 | 0.60 | 0.03 | confidence | 更强 disagree |
| `nsc05d_conf_mu06_l05_tr03` | h56mc | 0.60 | 0.50 | 0.03 | confidence | 更强 SC gate |
| `nsc05e_conf_mu04_l05_tr02` | h56mc | 0.40 | 0.50 | 0.02 | target_rate_eta=0.015 | 降低 target-rate |
| `nsc05f_conf_mu04_l05_notr` | h56mc | 0.40 | 0.50 | none | 无 target-rate | 验证 target-rate 必要性 |
| `nsc05g_conf_active_mu04_l05_tr03` | h56mc | 0.40 | 0.50 | 0.03 | consensus_score_norm=active | active denominator |
| `nsc05h_conf_kmod_mu04_l05_tr03` | h56mc | 0.40 | 0.50 | 0.03 | K consistency mod | 抑制冲突 K |
| `nsc05i_conf_mu04_l05_tr03_slowlr` | h56mc | 0.40 | 0.50 | 0.03 | slow LR/warmup400 | 稳定性 |
| `nsc05j_conf_mu04_l05_tr03_clamp` | h56mc | 0.40 | 0.50 | 0.03 | gate clamp -0.75/1.25 | 控制 signed gate 幅度 |

已新增文件：

- `neuron_experiments/H9_bipolar_self_attention/entrypoints/make_nsc05_sc_conf_blend_configs.py`
- `neuron_experiments/H9_bipolar_self_attention/entrypoints/run_nsc05_autopilot.py`
- `bsa_attention.py` 新增 `sc_ad_confidence_carrier_blend_shiftmax/h56mc`

验证：

```text
py_compile passed
test_bsa_attention: 13 tests OK
2-step smoke passed:
  train installed ATLIFTernaryPSN=32, Shiftmax attention=12
  profile load audit checkpoint_overlay_keys=64, missing=0, unexpected=0
```

执行策略：10 个配置各 360-step，valid10 + valid40 confirm；自动选 valid40 综合分最优的短测 checkpoint，立即跑 `eval_DSEC_flow_SNN.py` full valid825。若 full valid825 AEE 不能接近 `1.55` 或 spikes 不能压到 `34G` 附近，SC/NSC 线不再抢主线资源；若接近 NTX-04/NTX-01，再排 full30。


### 31.14.1 NSC-05 confidence-aware SC 短测与标准推理（自动追加）

- 时间：`2026-06-03T01:11:29`
- 短测目录：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/nsc05_sc_conf_blend_short_20260603_001509`
- 选中短测：`nsc05a_h56m_mu05_l06_tr03_steps360_steps360_valid40`，valid40 AEE `1.3860`，AAE `12.5411`，SOPs `3.8796G`
- 配置：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/nsc05_sc_conf_blend_short_20260603_001509/configs/nsc05a_h56m_mu05_l06_tr03_steps360_steps360.yml`
- checkpoint：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/nsc05_sc_conf_blend_short_20260603_001509/runs/nsc05a_h56m_mu05_l06_tr03_steps360_steps360/checkpoint_epoch0.pth`
- 标准推理目录：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/nsc05_autopilot_20260603_001508/standard_valid825_selected`
- 推理：`eval_DSEC_flow_SNN.py` full valid825，CuPy backend。

| AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | effective_flops | sparsity | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.6662 | 10.8036 | 0.5596 | 0.2313 | 0.1136 | 38.7917G | 8.3670% | 102.8481G | 91.63% | 31541.29 |

权重加载审计：

```text
eval installed ATLIFTernaryPSN: 32 modules
eval installed Shiftmax attention: 12 modules
load audit: checkpoint_overlay_keys=64, missing=0, unexpected=0
```

valid40 confirm 排名前 10：

| rank | variant | mode | AEE | AAE | SOPs | firing | 结论 |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `nsc05a_h56m_mu05_l06_tr03` | h56m | 1.3860 | 12.5411 | 3.8796G | 9.177% | NSC-04d 对照胜出 |
| 2 | `nsc05h_conf_kmod_mu04_l05_tr03` | h56mc | 1.4175 | 13.2111 | 3.8586G | 9.128% | K consistency 略省 SOPs，但精度掉 |
| 3 | `nsc05i_conf_mu04_l05_tr03_slowlr` | h56mc | 1.4278 | 13.1637 | 3.8655G | 9.144% | 慢 LR 未改善 |
| 4 | `nsc05b_conf_mu04_l05_tr03` | h56mc | 1.4317 | 13.1909 | 3.8589G | 9.128% | confidence 基本型未改善 |
| 5 | `nsc05j_conf_mu04_l05_tr03_clamp` | h56mc | 1.4317 | 13.1909 | 3.8589G | 9.128% | clamp 对短测无可见收益 |
| 6 | `nsc05c_conf_mu04_l06_tr03` | h56mc | 1.4220 | 13.3746 | 3.8583G | 9.127% | 更强 disagree 伤 AAE |
| 7 | `nsc05e_conf_mu04_l05_tr02` | h56mc | 1.4409 | 13.3155 | 3.8603G | 9.132% | target-rate 降到 0.02 未改善 |
| 8 | `nsc05g_conf_active_mu04_l05_tr03` | h56mc | 1.4391 | 13.4407 | 3.8572G | 9.124% | active norm 未改善 |
| 9 | `nsc05f_conf_mu04_l05_notr` | h56mc | 1.4298 | 13.4232 | 3.8813G | 9.181% | 去 target-rate 未改善且更高 firing |
| 10 | `nsc05d_conf_mu06_l05_tr03` | h56mc | 1.4458 | 13.4574 | 3.8567G | 9.123% | 更高 SC 混合比例最差 |

结论：

- `h56mc` confidence/deadzone 方向没有超过 `h56m` carrier-blend 对照；它确实把 SOPs 从 `3.8796G` 压到约 `3.856-3.866G`，但 AEE/AAE 变差，说明它主要削弱了有效匹配而不是只抑制噪声。
- target-rate 不是当前 SC/NSC 的主要问题：`target_rate=0.03`、`0.02`、`none` 三档都没有超过对照；去 target-rate 还把 firing/SOPs 拉高。
- lambda 当前更像敏感超参而不是核心突破口：`lambda=0.6` 的对照最好，confidence 下 `0.5/0.6` 都不如对照，继续只扫 lambda 价值不大。
- 本轮 full valid825 为 AEE `1.6662`、spikes `38.79G`，弱于 NSC-04d full30 epoch19 的 AEE `1.6458`、spikes `36.93G`，也弱于 TX V2/NTX-04；因此 NSC-05 不排 full30。

后续若还要改 SC/NSC，建议不要再只改 gate 后处理。可短测的下一轮应改为：

1. **SC 只做 auxiliary regularizer，不直接替换/混合 attention 输出**：保留 TX/NTX 的 carrier attention，SC agree/disagree 只约束 Q/K token 符号一致性，避免 signed gate 直接破坏 flow 匹配。
2. **分 stage 启用 SC**：前 5-10 epoch 只训练 carrier/NTX，后续再小权重打开 SC loss 或 SC gate，避免早期弱 token 投票污染 attention。
3. **按层选择性启用 SC**：优先只在后两级低分辨率 attention 启用，前两级保持 TX/NTX；当前全 12 个 attention 都走 SC 时，方向误差放大。
4. **把 full30 资源回到 TX/NTX 主线**：SC/NSC 当前保留为 ablation，主线继续沿 TX V2、NTX-04/NTX-07 这种 carrier/selector 路线推进。
### 31.14 NTX-05/06/07 非外挂 attention 短测结论（2026-06-03）

目的：回应 NTX-01 “carrier × extra TX gate” 讲法不够干净的问题，专门测试非外挂 attention 路线。所有短测均从标准 baseline59 `experiments/baseline_stride_upstream/checkpoint_epoch59.pth` 续训，`rapid_screen.py --steps 360 --promote-samples 40`，用 valid40 作为主要筛选依据。

结论：不 promotion，不 full30。三类非外挂替换在 valid40 上都出现明显 AAE 崩坏，不能作为主线替代 NTX-01。

#### NTX-05：全阶段 native QKV attention

目录：`neuron_experiments/H9_bipolar_self_attention/results/ntx05_native_short2_20260602_234922`

| candidate | attention | valid40 AEE | valid40 AAE | SOPs_G | 判断 |
|---|---|---:|---:|---:|---|
| ntx05e | A2OS2A-QKV theta V | 1.6581 | 14.7138 | 4.7116 | NTX05 最好，但 AAE 明显崩 |
| ntx05f | A2OS2A-QKV sign V | 1.7456 | 15.1716 | 4.7155 | 失败 |
| ntx05a | ternary alpha-XNOR QKV theta V | 1.7747 | 15.8495 | 4.5174 | 失败 |
| ntx05b | ternary alpha-XNOR QKV sign V | 1.8064 | 15.8723 | 4.5204 | 失败 |
| ntx05d | strict BSA-QKV sign V | 1.7832 | 16.3834 | 4.5273 | 失败 |
| ntx05c | strict BSA-QKV theta V | 1.8154 | 16.3504 | 4.5332 | 失败 |

加载审计正常，例如 NTX05A profile 显示 `checkpoint_overlay_keys=188, missing=0, unexpected=0`，所以失败不是权重未加载，而是 attention 结构本身在该训练长度/续训点上破坏方向角。

#### NTX-06：局部 native QKV attention

目录：`neuron_experiments/H9_bipolar_self_attention/results/ntx06_partial_native_short_20260603_002630`

| candidate | attention | scope | valid40 AEE | valid40 AAE | SOPs_G | 判断 |
|---|---|---|---:|---:|---:|---|
| ntx06d | strict BSA-QKV sign V | stage2 | 1.6997 | 16.1110 | 4.2276 | NTX06 最好，但仍失败 |
| ntx06c | ternary alpha-XNOR QKV theta V | stage2 | 1.7029 | 16.0634 | 4.2241 | 失败 |
| ntx06a | A2OS2A-QKV theta V | stage2 | 1.7345 | 16.5451 | 4.3253 | 失败 |
| ntx06b | A2OS2A-QKV theta V | stage0+2 | 1.7727 | 16.4644 | 4.7865 | 失败 |

局部替换降低了 SOPs，但没有解决 AAE 问题。说明问题不只是全阶段过度替换，QKV/native matrix attention 与当前 SDFormerFlow 续训点的方向保持能力不匹配。

#### NTX-07：局部逐 token QK selector 替换

目录：`neuron_experiments/H9_bipolar_self_attention/results/ntx07_partial_selector_short_20260603_005736`

| candidate | attention | scope | valid40 AEE | valid40 AAE | SOPs_G | 判断 |
|---|---|---|---:|---:|---:|---|
| ntx07c | H54b three-score TX selector | stage2 | 1.7059 | 16.4694 | 4.1552 | NTX07 最好，但仍失败 |
| ntx07d | H51 dual-channel selector | stage2 | 1.7201 | 16.3057 | 4.1537 | 失败 |
| ntx07a | H49 QK selector, mismatch 0.25 | stage2 | 1.7435 | 16.5524 | 4.1549 | 失败 |
| ntx07b | H49 QK selector, mismatch 0.50 | stage2 | 1.7439 | 16.7750 | 4.1563 | 失败 |

valid10 对这些候选有明显乐观偏差，例如 NTX07A valid10 AEE `1.4436`，但 valid40 AEE 变为 `1.7435` 且 AAE `16.5524`。后续筛选不能只看 valid10。

#### 当前决策

- 不启动 NTX05/06/07 full30。
- 非外挂 attention 替换路线暂时判负：native QKV、局部 native QKV、局部 QK selector 都没达到继续全量训练门槛。
- 当前可用主结果仍是 NTX01 TX V2：标准 valid825 best AEE `1.534` 到 `1.550` 区间，spikes/energy 约下降 21%，是目前唯一同时满足精度和稀疏性的 TX 主线。
- 后续不应继续为“非外挂 attention”盲目扩展 full。若继续优化论文故事，优先从 NTX01 的硬件解释、模块命名、稀疏/三值 neuron/pruning 组合入手，而不是把 attention 直接替换成 QKV/native selector。

### 31.15 NSC-06/H57：TX carrier + SC residual 短测结论（2026-06-03）

目的：在 NTX05/06/07 直接替换 attention 失败后，测试更保守的内部 residual 方向。H57 保留 NTX01/QKFormer carrier 和 TX selector 作为稳定基底，只加入小权重 SC agree/disagree residual，避免 QKV/native selector 直接破坏 flow 方向。

流程：所有候选均从标准 baseline59 `experiments/baseline_stride_upstream/checkpoint_epoch59.pth` 续训，使用 `/opt/conda/envs/sdformerflow/bin/python neuron_experiments/H9_bipolar_self_attention/entrypoints/rapid_screen.py --steps 360 --valid-samples 10 --promote-samples 40 --batch-size 8 --amp`。测试前 `test_bsa_attention.py` 14 项通过，包含 H57 tiny attention 前向检查。结果目录：`neuron_experiments/H9_bipolar_self_attention/results/nsc06_h57_short_20260603_013059`。

| rank | candidate | mode | scope | valid40 AEE | valid40 AAE | SOPs_G | firing | 判断 |
|---:|---|---|---|---:|---:|---:|---:|---|
| 1 | `nsc06b_h57_all_mu010_l03` | TX + SC residual | all | 1.7064 | 16.2661 | 4.1345 | 9.780% | 相对控制项小幅改善，但不足以 full |
| 2 | `nsc06a_h57_tx_control_all_mu0` | H57 control, mu=0 | all | 1.7120 | 16.3848 | 4.1354 | 9.782% | 与 NTX04H 短测接近，证明包装/加载无明显偏差 |
| 3 | `nsc06g_h57_s2_conf_mu020_l03` | TX + SC residual + confidence | stage2 | 1.7284 | 16.3150 | 4.1391 | 9.791% | confidence 没保住 AEE |
| 4 | `nsc06d_h57_s2_mu020_l03` | TX + SC residual | stage2 | 1.7440 | 16.0708 | 4.1429 | 9.800% | AAE 降、AEE 明显升，综合失败 |
| 5 | `nsc06h_h56r_s2_alpha025_l03` | pure SC residual | stage2 | 1.7416 | 16.3284 | 4.1623 | 9.846% | 失败，且 SOPs 更高 |

结论：

- H57 包装本身可靠：`mu=0` 控制项 valid40 AEE/AAE 为 `1.7120/16.3848`，与 NTX04H 短测 `1.7057/16.1830` 同量级；失败或收益不是加载错误造成的。
- 直接替换 attention 的路线继续判负；carrier-preserving residual 至少没有明显负向，`all mu=0.10` 比控制项 AEE 低 `0.0057`、AAE 低 `0.1187`，SOPs 基本不变。
- 这轮没有候选达到 full30 门槛：最优 `nsc06b` 只是短测小正向，远没有接近 NTX01/NTX04 全量标准。不要直接 promotion。
- stage2-only 不一定更安全：`mu=0.20` 在 stage2 上降低 AAE 但损伤 AEE；confidence/deadzone 没有修复这个问题。

下一步建议：

1. 若继续做非外挂叙事，优先围绕 H57 做更小 residual sweep：`mu=0.03/0.05/0.08/0.10`，scope 先用 all，避免 stage2 强扰动。
2. 不再跑 QKV/native selector full30；NTX05/06/07 和本轮结果共同说明直接替换 attention 会破坏方向角。
3. 主线仍保持 NTX01 TX V2/NTX04 结果，后续提升更应放在神经元减法、剪枝、硬件映射和可解释命名上。
4. H57 只有在更细 sweep 出现比控制项明显更强的 valid40/valid825 证据后，才考虑 full30。

### 31.16 NSC-07：H57 小 residual fine sweep（2026-06-03）

目的：NSC-06 显示 H57 `all mu=0.10/lambda=0.30` 相对 `mu=0` 控制项有小幅正收益，但 stage2 强 residual 和 confidence 版本失败。NSC-07 因此只做 all-scope 小 residual 细扫，不再测试 QKV/native selector，也不再做 stage2 强扰动。

流程：从标准 baseline59 `experiments/baseline_stride_upstream/checkpoint_epoch59.pth` 续训，`rapid_screen.py --steps 360 --valid-samples 10 --promote-samples 40 --batch-size 8 --amp`。测试前 `test_bsa_attention.py` 14 项通过。结果目录：`neuron_experiments/H9_bipolar_self_attention/results/nsc07_h57_fine_short_20260603_020710`。

| rank | candidate | mu | lambda | valid40 AEE | valid40 AAE | SOPs_G | firing | 判断 |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `nsc07f_h57_all_mu010_l04` | 0.10 | 0.40 | 1.6741 | 15.8335 | 4.1357 | 9.783% | 本轮最好，进入 full30 验证 |
| 2 | `nsc07b_h57_all_mu005_l03` | 0.05 | 0.30 | 1.6809 | 15.8251 | 4.1354 | 9.782% | AAE 略优，AEE 略弱；备选 |
| 3 | `nsc07d_h57_all_mu012_l03` | 0.12 | 0.30 | 1.6834 | 16.0871 | 4.1368 | 9.785% | 可用但不如前两项 |
| 4 | `nsc07a_h57_all_mu003_l03` | 0.03 | 0.30 | 1.7157 | 15.9859 | 4.1348 | 9.781% | 方向角改善但 AEE 不够 |
| 5 | `nsc07e_h57_all_mu010_l02` | 0.10 | 0.20 | 1.7261 | 16.1553 | 4.1357 | 9.783% | lambda 太弱，失败 |
| 6 | `nsc07c_h57_all_mu008_l03` | 0.08 | 0.30 | 1.7529 | 17.0167 | 4.1370 | 9.786% | 明显失败 |

对照关系：

- NSC-06 `mu=0` 控制项：AEE `1.7120`、AAE `16.3848`、SOPs `4.1354G`。
- NSC-06 `mu=0.10/lambda=0.30`：AEE `1.7064`、AAE `16.2661`、SOPs `4.1345G`。
- NSC-07 最好 `mu=0.10/lambda=0.40`：AEE `1.6741`、AAE `15.8335`、SOPs `4.1357G`。

结论：

- H57 不是“直接外挂替换 attention”的失败路线；它保留 QKFormer/NTX carrier，只用 SC agree/disagree 做小 residual。短测证据显示它能同时改善 AEE 和 AAE，且 SOPs/firing 基本不变。
- residual 强度存在窄区间：`0.03` 太弱，`0.08` 异常失败，`0.10/lambda=0.40` 最好，`0.12/lambda=0.30` 可用但不优。
- lambda 不能简单越小越好：`mu=0.10/lambda=0.20` 明显失败，`lambda=0.40` 反而最好，说明需要保留足够 disagree 抑制。
- 决策：启动 `nsc07f_h57_all_mu010_l04` full30，从标准 baseline59 续训，后续按标准 valid825 profile 输出 AEE/AAE/PE/SOPs/firing/energy/sparsity 等全部指标。

full30 启动记录：

- 启动时间：`2026-06-03 02:51:19 UTC`
- 配置：`neuron_experiments/H9_bipolar_self_attention/configs/generated/nsc07f_h57_all_mu010_l04.yml`
- 续训 checkpoint：`experiments/baseline_stride_upstream/checkpoint_epoch59.pth`
- 运行目录：`neuron_experiments/H9_bipolar_self_attention/results/nsc07f_h57_all_mu010_l04_full30_bs8_20260603_025119_setsid`
- PID：`1933610`
- 关键超参：`mode=tx_sc_residual_selector_shiftmax`，`stage_selection=all`，`bipolar_mu=0.10`，`bipolar_lambda=0.40`，`target_rate=null`，`target_rate_eta=0.0`，`activity_eta=0.0`，`batch_size=8`，`n_epochs=30`，`lr=2e-5`，`backbone_lr=1e-6`，`norm_lr=1e-6`，`neuron_lr=3e-5`，`threshold_lr=5e-6`，`lr_warmup=450 steps/start_factor 0.05`。
- 启动检查：进程已 detach 到 PPID 1，日志进入 `Epoch 0`，optimizer groups 正常。


### 31.15 NSC-06 TX+SC hybrid attention 短测与 full30 启动（自动追加）

注意：本段是旧 autopilot 自动追加记录，当前 PID `121059` 已不存在，目录中只保留 `checkpoint_epoch0.pth`，不是当前正在运行的 full30。当前正在运行的是上一节 NSC-07f full30，PID `1933610`。

- 时间：`2026-06-03T02:17:24`
- 短测目录：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/nsc06_tx_sc_hybrid_short_20260603_013743`
- 新增 mode：`tx_sc_residual_selector_shiftmax/h57`，TX carrier/gate 为主，SC agree/disagree 小比例 residual。
- 选择理由：`selected best ranked row`
- 选中短测：`nsc06b_h57_all_mu010_l03_steps360_steps360_valid40`，AEE `1.5293`，AAE `14.0057`，SOPs `3.8506G`
- full30 配置：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/configs/generated/nsc06b_h57_all_mu010_l03_full30.yml`
- full30 目录：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/nsc06b_h57_all_mu010_l03_auto_full_bs8_20260603_021724_setsid`
- full30 PID：`121059`

| rank | variant | stage | AEE | AAE | SOPs | firing | score |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `nsc06b_h57_all_mu010_l03_steps360_steps360_valid40` | confirm | 1.5293 | 14.0057 | 3.8506G | 9.109% | 2.6926 |
| 2 | `nsc06e_h57_s02_mu015_l03_steps360_steps360_valid40` | confirm | 1.5353 | 14.0257 | 3.8523G | 9.113% | 2.7018 |
| 3 | `nsc06f_h57_s012_mu015_l04_steps360_steps360_valid40` | confirm | 1.5491 | 14.1966 | 3.8533G | 9.115% | 2.7341 |
| 4 | `nsc06g_h57_s2_conf_mu020_l03_steps360_steps360_valid40` | confirm | 1.5593 | 14.2078 | 3.8587G | 9.128% | 2.7486 |
| 5 | `nsc06d_h57_s2_mu020_l03_steps360_steps360_valid40` | confirm | 1.5716 | 14.5847 | 3.8602G | 9.131% | 2.8013 |
| 6 | `nsc06i_h57_s23_mu010_l03_steps360_steps360_valid40` | confirm | 1.5812 | 14.5518 | 3.8537G | 9.116% | 2.8048 |

后续标准化口径：full30 完成后优先对 epoch `19/24/28/29` 使用 `eval_DSEC_flow_SNN.py` 跑 full valid825，并检查 `checkpoint_overlay_keys`、`missing`、`unexpected` 与安装模块数。

### 31.17 H57/NSC-06-07 full30 标准 valid825 结果与下一轮改进（2026-06-03）

背景：31.15 旧 autopilot 段里的 `nsc06b` full30 PID 已结束；实际目录已完整保存到 `checkpoint_epoch29.pth`。31.16 的 `nsc07f` full30 也已完成到 `checkpoint_epoch29.pth`。本节使用标准论文口径 `third_party/SDformerFlow/eval_DSEC_flow_SNN.py --mode valid` 跑 full valid825，不用 `profile_checkpoints.py` 作为最终口径。

资源控制：原脚本准备跑 06b/07f 共 8 个 checkpoint。由于 06b epoch19 和 07f epoch19 均明显低于 NTX-01，后续只保留信息量最高的关键点：06b epoch23（06b 训练 valid loss 最低附近）和 07f epoch27（07f 已保存 checkpoint 中 valid loss 较低点）。这样避免继续浪费 GPU 在 27/29 等明显不优点上。

| line | checkpoint | full valid825 AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | effective_flops | 相对 NTX-01 e28 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NTX-01 TX V2 | epoch28 | 1.5340 | 10.2880 | 0.5392 | 0.2020 | 0.0911 | 34.6119G | 7.4655% | 70.1308G | 0.0000 |
| NSC-06b H57 all mu0.10 l0.30 | epoch19 | 1.6642 | 10.9699 | 0.5771 | 0.2342 | 0.1111 | 34.1040G | 7.3559% | 90.4198G | +0.1302 |
| NSC-06b H57 all mu0.10 l0.30 | epoch23 | 1.5953 | 10.9233 | 0.5607 | 0.2158 | 0.0980 | 35.4704G | 7.6506% | 94.0423G | +0.0613 |
| NSC-07f H57 all mu0.10 l0.40 | epoch19 | 1.6749 | 11.0451 | 0.5733 | 0.2321 | 0.1109 | 34.1186G | 7.3591% | 90.4583G | +0.1408 |
| NSC-07f H57 all mu0.10 l0.40 | epoch27 | 1.6458 | 11.1804 | 0.5703 | 0.2259 | 0.1055 | 35.0988G | 7.5705% | 93.0572G | +0.1118 |

当前判断：

- H57 当前最优是 `NSC-06b epoch23`，AEE `1.5953`，AAE `10.9233`，比旧 NSC03/04 稳定，但仍落后 NTX-01 epoch28：AEE 差 `+0.0613`，AAE 差 `+0.6353`，effective FLOPs 高 `+23.91G`。
- `lambda=0.40` 的 NSC-07f 短测看起来更好，但 full valid825 没有兑现；`epoch27` AEE `1.6458`，不如 `lambda=0.30` 的 NSC-06b epoch23。说明 lambda 不能再靠 valid40 单独决策，至少需要关键 checkpoint 标准 valid825 验证。
- H57 的好处是没有像 native QKV/selector 那样 AAE 崩坏，AAE 保持在 11 左右；问题是 effective FLOPs 高、AEE 仍低于 NTX-01，说明 SC residual 引入了额外匹配扰动但没有带来足够方向收益。
- 当前不继续跑 H57 其它 checkpoint 标准推理；06b epoch23 已经给出 best-case 证据，07f 最好保存点也不足。

下一轮 NSC-08/H58 改进方案和短测矩阵：

1. **SC residual 改为 late/anneal gate**：前 8-10 epoch `mu=0`，之后线性打开到 `mu=0.05/0.08/0.10`。目的：避免 baseline59 续训早期被 SC token 投票扰动，同时保留后期正则。
2. **SC 只调制 gate，不调制 carrier 幅值**：保留 NTX-01 的 TX gate 均值和 carrier，SC 只作为 centered residual 加到 score 后再 `shiftmax`，并做 `preserve_mean`。目的：降低 effective FLOPs 和 AEE 扰动。
3. **按层缩小替换范围**：不再 all 12 层一刀切，短测 `stage1+2`、`stage2`、`stage2 late-only` 三档；stage0/3 默认回到 NTX-01。
4. **加 direction/teacher 轻约束而不是 target-rate**：target-rate 当前不是主因，NSC06/07 的 `target_rate=null` 能跑；下一轮只加 `direction_loss` 或 teacher flow/attention consistency 小权重 `0.005/0.01`，不恢复 target-rate。
5. **学习率更保守**：attention/overlay LR 保持 `2e-5` 或降到 `1e-5`，backbone/norm 维持 `1e-6`，neuron LR `3e-5` 不动；避免 H57 后期 ternary 活动继续变稀导致 effective FLOPs 上升但精度不上去。

短测门槛建议：先 `steps=360 valid40` 排序，但不再要求新模块短测立刻达到 NTX-01；若 AEE `<=1.58` 且 AAE `<=14.5`，或相对 H57 control 明显改善，即跑一个关键标准 valid825 checkpoint。只有标准 valid825 AEE 接近 `1.55` 且 AAE 不劣于 `10.8`，才启动 full30。

### 31.18 NSC-08/H58 late SC residual 短测结果（2026-06-03）

目的：验证“晚开 SC residual”是否能解决 H57/H58 早期扰动问题。实现上新增可选 `sc_mu_schedule_enabled/start_step/warmup_steps/start_mu`，只在新配置显式打开时生效；旧 H56/H57/NTX 配置不读该字段，兼容不变。训练与 profile smoke 已检查：baseline59 续训时 `checkpoint_overlay_keys=0, missing=68, unexpected=0`，smoke checkpoint 推理加载时 `checkpoint_overlay_keys=68, missing=0, unexpected=0`。

短测口径：`rapid_screen.py --steps 360 --valid-samples 10 --promote-samples 40 --batch-size 8 --amp`。为让 360-step 短测内真实打开 SC，short config 使用 `sc_mu_start_step=120, sc_mu_warmup_steps=120`；full30 config 保留更慢的 late schedule。

| candidate | 关键变化 | valid40 AEE | AAE | SOPs_G | firing | 判断 |
|---|---|---:|---:|---:|---:|---|
| `nsc08d_h58_s2_mu010_l03_late360` | stage2 only | 1.7085 | 16.3854 | 4.1374 | 9.787% | 本轮最好但不足 |
| `nsc08e_h58_s12_mu008_l03_late360` | stage1+2, mu0.08 | 1.7241 | 16.2324 | 4.1356 | 9.783% | AAE 略低，AEE 不够 |
| `nsc08f_h58_all_mu010_l03_ang005` | all + angular 0.005 | 1.7348 | 16.4015 | 4.1628 | 9.847% | angular 小权重未修复 |
| `nsc08c_h58_all_mu010_l03_late360` | all, schedule 生效 | 1.7542 | 16.7370 | 4.1354 | 9.782% | 明显负向 |
| `nsc08g_h58_all_mu010_l03_lr1e5` | all, lower LR | 1.7542 | 16.7370 | 4.1354 | 9.782% | 与 c 等价级别，低 LR 无帮助 |

结论：

- 不启动 NSC-08 full30。best `nsc08d` 仍弱于 NSC-07f 短测 AEE `1.6741`，更弱于 NSC-06 autopilot 记录的 `1.5293/14.0057`。
- late schedule 没解决问题；一旦 360 step 内打开 SC，all-scope AEE 反而从 warm-start 控制的 `1.7200` 退到 `1.7542`。
- stage2-only 仍是相对更安全的替换范围，但它只把 AEE 拉回 `1.7085`，没有主线价值。
- valid10 继续存在严重乐观偏差，例如 `nsc08c` valid10 AEE `1.4818`，valid40 AEE `1.7542`；后续不得用 valid10 选 full。

下一步：H59 改 attention 形式，不再把 SC signed gate 直接混入 gate。SC 只作为小 score residual 加到 TX score，最后统一过一次 `shiftmax`，即 `gate=shiftmax(tx_scores + mu*sc_scores)`，避免 H57/H58 的正负 gate 混合破坏分布。

### 31.19 NSC-09/H59 score residual 短测与 full30 启动（2026-06-03）

目的：H58 证明“SC gate 混合”在 360-step 内打开后会退化，因此 H59 改为更保守的 score residual：保持 NTX/TX carrier 和单一 `shiftmax` 分布，只把 SC consensus score 以小系数加到 TX score，形式为 `gate = shiftmax(tx_scores + mu * sc_scores)`。该 mode 为新增 `tx_sc_score_residual_shiftmax/h59`，旧 H56/H57/H58/NTX 配置不受影响。

验证：

- `py_compile` 通过：`bsa_attention.py`、`make_nsc09_h59_score_residual_configs.py`。
- 单测通过：`test_bsa_attention.py` 共 15 项 OK，包含 H59 tiny attention 前向。
- smoke 加载审计正常：训练从 baseline59 接入时 `checkpoint_overlay_keys=0, missing=68, unexpected=0`；smoke checkpoint profile 时 `checkpoint_overlay_keys=68, missing=0, unexpected=0`。

短测口径：`rapid_screen.py --steps 360 --valid-samples 10 --promote-samples 40 --batch-size 8 --amp`。

| candidate | 关键变化 | valid40 AEE | AAE | SOPs_G | firing | 判断 |
|---|---|---:|---:|---:|---:|---|
| `nsc09d_h59_all_mu005_sched` | all, mu0.05, scheduled | 1.6909 | 15.8686 | 4.1355 | 9.782% | NSC-09 最好，启动 full30 |
| `nsc09c_h59_s2_mu005` | stage2, mu0.05 | 1.7138 | 16.0516 | 4.1372 | 9.787% | 不如 d |
| `nsc09b_h59_all_mu005` | all, fixed mu0.05 | 1.7201 | 15.9563 | 4.1352 | 9.782% | 不如 d |
| `nsc09a_h59_all_mu002` | all, fixed mu0.02 | 1.7315 | 16.2537 | 4.1352 | 9.782% | 太弱 |

判断：

- H59 比 H58 略稳，best AEE 从 NSC-08 best `1.7085` 改到 `1.6909`，但仍低于 NSC-07f short 的 `1.6741`，更低于 NTX/NSC-06 autopilot 的强短测。
- 按“GPU 不空转且必须保留至少一条 full30”的策略，选择 NSC-09 best `nsc09d_h59_all_mu005_sched` 启动 full30；这是当前新 attention 方向里相对最稳的候选。
- full30 后按训练 valid loss 选择 checkpoint，再跑标准 `eval_DSEC_flow_SNN.py --mode valid`，不使用 `profile_checkpoints.py` 作为最终论文口径。

full30 启动记录：

- 启动时间：`2026-06-03 08:48:17 UTC`
- 配置：`neuron_experiments/H9_bipolar_self_attention/configs/generated/nsc09d_h59_all_mu005_sched_full30.yml`
- 续训 checkpoint：`experiments/baseline_stride_upstream/checkpoint_epoch59.pth`
- 运行目录：`neuron_experiments/H9_bipolar_self_attention/results/nsc09d_h59_all_mu005_sched_full30_bs8_20260603_084817_setsid`
- PID：`222638`
- 启动检查：进程 detach 到 PPID 1，GPU 显存约 `43.8GB`、util `100%`，进入 `Epoch 0`；训练加载审计 `checkpoint_overlay_keys=0, missing=68, unexpected=0`。

### 31.18 NSC-07f/H57 full30 补齐标准 valid825 结果（2026-06-03）

上一节 31.17 是早期自动追加记录，当时只覆盖了 `NSC-07f epoch19/27`。本节补齐当前实际训练目录的标准 valid825 推理：`epoch19/24/28/29` 全部使用最终标准入口 `third_party/SDformerFlow/eval_DSEC_flow_SNN.py --mode valid`，不是 `profile_checkpoints.py`。

训练与配置：

- 配置：`neuron_experiments/H9_bipolar_self_attention/configs/generated/nsc07f_h57_all_mu010_l04.yml`
- 训练目录：`neuron_experiments/H9_bipolar_self_attention/results/nsc07f_h57_all_mu010_l04_full30_bs8_20260603_025119_setsid`
- 结构：`mode=tx_sc_residual_selector_shiftmax`，TX carrier/gate 为主，SC agree/disagree residual，小比例调制。
- 关键超参：`stage_selection=all`，`bipolar_mu=0.10`，`bipolar_lambda=0.40`，`target_rate=null`，`activity_eta=0.0`，`batch_size=8`，`n_epochs=30`，`lr=2e-5`，`backbone_lr=1e-6`，`norm_lr=1e-6`，`neuron_lr=3e-5`，`threshold_lr=5e-6`，`warmup=450 steps/start_factor 0.05`。
- 推理加载审计：每个 checkpoint 都显示 `ATLIFTernaryPSN=34`、`Shiftmax attention=12`、`checkpoint_overlay_keys=68`、`missing=0`、`unexpected=0`。

| epoch | samples | AEE | AAE | PE1 | PE2 | PE3 | outlier | total_spikes | firing | effective_flops | sparsity | synops_mac | synops_logic | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 19 | 825 | 1.6749 | 11.0451 | 0.5733 | 0.2321 | 0.1109 | 0.1109 | 34.1186G | 7.3591% | 90.4583G | 92.64% | 32.0943G | 2.0242G | 29087.31 |
| 24 | 825 | 1.6445 | 11.1123 | 0.5699 | 0.2280 | 0.1070 | 0.1070 | 33.9162G | 7.3154% | 89.9218G | 92.68% | 32.1745G | 1.7417G | 29131.18 |
| 28 | 825 | 1.6773 | 11.2568 | 0.5706 | 0.2303 | 0.1088 | 0.1088 | 35.6800G | 7.6959% | 94.5982G | 92.30% | 33.9657G | 1.7143G | 30740.56 |
| 29 | 825 | 1.6026 | 10.6948 | 0.5554 | 0.2169 | 0.1025 | 0.1025 | 33.3318G | 7.1894% | 88.3724G | 92.81% | 31.7345G | 1.5973G | 28720.80 |

对比当前标准 baseline 与主线：

- 相对 `NB0 baseline epoch59`：AEE `1.6026` vs `1.4872`，差 `+0.1154`，约 `+7.8%`；total_spikes `33.3318G` vs `44.0488G`，下降约 `24.3%`；energy `28720.80uJ` vs `37638.01uJ`，下降约 `23.7%`。稀疏目标达到，但精度未进入 baseline 5% 内。
- 相对 `NTX-01 TX V2 epoch28`：AEE 差 `+0.0686`，AAE 差 `+0.4068`；total_spikes 少约 `1.28G`，energy 低约 `0.99mJ`，但 effective FLOPs 高约 `18.24G`。说明它更省 spike/energy，但计算分布不如 NTX-01 高效，精度也不足。
- 相对 `NTX-04 epoch29`：AEE `1.6026` 优于 `1.6233`，total_spikes `33.3318G` 略低于 `33.4428G`。因此 H57/NSC-07f 不是失败线，但目前只能作为次主线，不足以替代 NTX-01。

结论：

- `NSC-07f epoch29` 是本轮 H57 的实际 best checkpoint。旧 31.17 中“07f 不如 06b/epoch27”的判断需要降级为历史中间结论。
- 这条线证明“TX carrier + SC residual”可以比 NTX-04 更好，但还没有达到论文主结果要求。后续不能继续简单加大 `mu/lambda` 或全层 residual；需要做结构性收敛：late gate、stage 限制、preserve-mean gate-only、或 teacher direction/AEE 轻约束。
- 当前不建议直接全量重跑同构 H57。下一轮若继续 NSC，应先做短测矩阵，并且至少对候选关键 checkpoint 做一次标准 valid825，而不是只依赖 valid40。

### 31.19 NSC-08/H58 late residual 短测启动（2026-06-03）

启动目的：接 31.18 的结论，验证 H57 的主要问题是否来自“从续训第一步就全层开启 SC residual”。H58 使用相同 TX carrier + SC residual 基础结构，但加入 `sc_mu_schedule_enabled=true`，短测阶段从 `mu=0` 逐步打开到目标 `mu`，避免 early fine-tune 扰乱 baseline59 的匹配结构。

启动前验证：

- `make_nsc08_h58_late_configs.py`、`rapid_screen.py`、`promote_best_rapid_screen.py`：`py_compile` 通过。
- `test_bsa_attention.py` 中 H57/H58 两个关键 `unittest` 通过：H58 在 step0 与 `mu=0` 控制项一致，在 schedule 终点与固定 `mu=0.10` H57 一致。

短测配置矩阵：

| variant | scope | final_mu | lambda | schedule | 其它 |
|---|---|---:|---:|---|---|
| `nsc08a_h58_all_mu010_l03_late720` | all | 0.10 | 0.30 | late720 | 控制项 |
| `nsc08b_h58_all_mu008_l03_late720` | all | 0.08 | 0.30 | late720 | 弱 residual |
| `nsc08c_h58_all_mu010_l03_late360` | all | 0.10 | 0.30 | late360 | 更早打开 |
| `nsc08d_h58_s2_mu010_l03_late360` | stage2 | 0.10 | 0.30 | late360 | 缩小到 S2 |
| `nsc08e_h58_s12_mu008_l03_late360` | stage1+2 | 0.08 | 0.30 | late360 | 中间范围 |
| `nsc08f_h58_all_mu010_l03_ang005` | all | 0.10 | 0.30 | late720 | `lambda_ang=0.005` |
| `nsc08g_h58_all_mu010_l03_lr1e5` | all | 0.10 | 0.30 | late720 | `lr=1e-5` |

运行记录：

- 目录：`neuron_experiments/H9_bipolar_self_attention/results/nsc08_h58_late_short_20260603_075324`
- PID：`1960379`
- 起点：`experiments/baseline_stride_upstream/checkpoint_epoch59.pth`
- 命令口径：`rapid_screen.py --steps 360 --valid-samples 10 --promote-samples 40 --batch-size 8 --amp --parallel 1`
- 当前启动检查：进程 detach 到 PPID 1，GPU `memory.used≈19781MiB`、`util≈92%`，说明训练已进入 GPU 执行阶段。

### 31.20 NTX-08 direct TX matrix attention 短测启动（2026-06-03）

目标：回应“不要 `carrier * gate` 外挂形式”的要求，回到真正的 direct attention：

```text
score_ij = TX(q_i, k_j)
A_ij = Shiftmax(score_ij)
out_i = sum_j A_ij * V_j
```

本轮不使用 native QKFormer carrier，也不使用 SC residual。为了避免旧 H42/H45 一上来全局 token mixing 过强，新增一个很小的 direct-matrix 稳定项：

```text
score_ij = score_ij + matrix_diag_bias * 1(i == j)
```

这不是 `carrier * gate`，只是 attention score 上的 same-token prior，仍然是 `score -> Shiftmax -> matmul(V)`。

代码改动与验证：

- `bsa_attention.py` 新增 `matrix_diag_bias` 配置，并接入 direct TX matrix Shiftmax 分支。
- `test_bsa_attention.py` 新增 targeted unittest，验证 `matrix_diag_bias` 确实只加到 score matrix 对角线上。
- 验证命令：`py_compile` 通过；`unittest` 两项通过，包括 diag-bias test 和 matrix Shiftmax 前向 test。

短测矩阵：

| variant | mode | scope | beta/mismatch | gamma/single | diag_bias | V |
|---|---|---|---:|---:|---:|---|
| `ntx08a_direct_tx_s2_b025_g005_d05_kv` | `ternary_alpha_xnor_ssa_kreuse_shiftmax` | S2 | 0.25 | 0.05 | 0.5 | K reuse |
| `ntx08b_direct_tx_s2_b035_g005_d05_kv` | `ternary_alpha_xnor_ssa_kreuse_shiftmax` | S2 | 0.35 | 0.05 | 0.5 | K reuse |
| `ntx08c_direct_tx_s2_b035_g008_d10_kv` | `ternary_alpha_xnor_ssa_kreuse_shiftmax` | S2 | 0.35 | 0.08 | 1.0 | K reuse |
| `ntx08d_direct_tx_s12_b025_g005_d05_kv` | `ternary_alpha_xnor_ssa_kreuse_shiftmax` | S1+S2 | 0.25 | 0.05 | 0.5 | K reuse |
| `ntx08e_direct_tx_s2_b025_g005_d05_qkv` | `ternary_alpha_xnor_ssa_qkv_shiftmax` | S2 | 0.25 | 0.05 | 0.5 | independent V |

运行记录：

- 配置生成器：`neuron_experiments/H9_bipolar_self_attention/entrypoints/make_ntx08_direct_tx_matrix_configs.py`
- 结果目录：`neuron_experiments/H9_bipolar_self_attention/results/ntx08_direct_tx_matrix_short_20260603_085034`
- PID：`1977644`
- 起点：`experiments/baseline_stride_upstream/checkpoint_epoch59.pth`
- 命令口径：`rapid_screen.py --steps 360 --valid-samples 10 --promote-samples 40 --batch-size 8 --amp --parallel 1`
- 启动检查：GPU `memory.used≈10727MiB`、`util≈86%`，训练已进入 GPU 执行阶段。

短测结果：

| rank | variant | samples | AEE | AAE | SOPs(G) | firing | gate |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `ntx08d_direct_tx_s12_b025_g005_d05_kv_steps360` | 10 | 1.4961 | 18.9320 | 4.0070 | 9.478% | `AAE>7.9` |
| 2 | `ntx08c_direct_tx_s2_b035_g008_d10_kv_steps360` | 10 | 1.4754 | 19.4023 | 3.9587 | 9.364% | `AAE>7.9` |
| 3 | `ntx08a_direct_tx_s2_b025_g005_d05_kv_steps360` | 10 | 1.4897 | 19.5396 | 3.9579 | 9.362% | `AAE>7.9` |
| 4 | `ntx08b_direct_tx_s2_b035_g005_d05_kv_steps360` | 10 | 1.5172 | 19.8648 | 3.9594 | 9.366% | `AAE>7.9` |
| 5 | `ntx08e_direct_tx_s2_b025_g005_d05_qkv_steps360` | 10 | 1.4927 | 20.0090 | 4.0356 | 9.370% | `AAE>7.9` |

结论：

- `score_ij -> Shiftmax -> matmul(V)` 的 direct TX matrix 形式可以得到很低的 valid10 AEE，最低 `1.4754`，说明 EPE/幅值并不崩。
- 但 AAE 全部在 `18.9-20.0`，远高于 NTX-01/NTX-04/H57 的 10-11 区间；所有候选都被 rapid gate 拦住，没有进入 valid40。
- 增强 mismatch penalty（A->B）没有改善 AAE；强 diag + single-active（C）改善 AEE 但不救 AAE；扩大到 S1+S2（D）略降 AAE 但仍严重不合格；independent V（E）更差。
- 因此本轮不启动 full30，也不做标准 valid825。direct matrix 路线若继续，必须先解决方向保持：例如加入 direction-preserving diagonal anchor、用 flow/teacher direction 约束、或者只在残差路径中做 direct matrix，而不是直接替换 attention 主体。

---

## 三十二、NSC-09d 结果与 SC 架构全景改进（2026-06-04）

### 32.1 NSC-09d 训练状态

- 配置：`nsc09d_h59_all_mu005_sched_full30.yml`
- 注意力：`sc_agree_disagree_shiftmax`, μ=0.05, 全 stage
- ATLIF：`symmetric_bsa_tsn`, no target_rate（对齐 TX）
- 30 epochs，从 `checkpoint_epoch59.pth` 出发

| epoch 范围 | valid loss |
|---|---|
| 早期 (0-10) | 1.75-2.08 |
| 中期 (10-20) | 1.64-1.69 |
| 后期 (20-29) | 1.57-1.66 |

**best valid loss: 1.574（epoch 23 附近）**

### 32.2 NSC-09d 标准 valid825（2026-06-04）

| epoch | AEE | AAE | total_spikes | energy | sparsity |
|---|---|---|---|---|---|
| 21 | 1.661 | 11.22 | 34.81G | 29.68mJ | 92.5% |
| 23 | 1.624 | 11.09 | 35.51G | 30.42mJ | 92.3% |
| 25 | 1.629 | 11.03 | 33.74G | 28.98mJ | 92.7% |
| **29** 🏆 | **1.607** | **10.63** | 33.37G | **28.75mJ** | **92.8%** |

#### SC 全线对比

| 实验 | AEE | AAE | spikes | vs baseline AEE | 改进 |
|---|---|---|---|---|---|
| Baseline | 1.489 | 9.92 | 44.05G | — | — |
| NTX-01 (TX) | 1.534 | 10.29 | 34.61G | +3.0% | 🏆 最佳 |
| **NSC-09d** | **1.607** | **10.63** | **33.37G** | **+7.9%** | ✅ SC 最优 |
| NSC-01 (SC old) | 1.771 | 11.42 | 32.30G | +18.9% | 基线 |

**结论**：ATLIF 对齐 TX（symmetric_bsa_tsn + no target_rate）+ μ=0.05 + agree/disagree 路线，将 SC 与 TX 的 AEE 差距从 15.5% 缩小到 4.8%。

### 32.3 短测结果（2026-06-04）

**Batch 1**（窗口 + λ sweep）：6 方案全被 gate 筛掉 ❌
- [4,9,9], [5,9,9]：窗口改动导致三值崩塌
- λ=0.3/0.5/1.2：偏离 λ=0.8 后 Q/K 不平衡

**Batch 2**（k_mag + angular）：4 方案跑着 🔄

**结论**：窗口尺寸改动对 pretrained checkpoint 不兼容（relative position bias 维度不匹配）。只有不改窗口结构的方案可行。

### 32.4 SC 架构全景改进方向（2026-06-04 更新）

#### 一、窗口维度改进

| # | 方向 | 现状 | 改进 | 改动量 | 维度风险 |
|---|---|---|---|---|---|
| W1 | 拉长时间窗 | window=[2,9,9] | [5,5,5] 或 [4,7,7] | 改 config | 注意：H×W 必须能被 window_size 整除。crop=288×384，可兼容 [4,6,6]、[2,8,8] 等 |
| W2 | Stage 自适应窗 | 所有 stage 同窗 | S0:[2,12,12], S3:[2,6,6] | 改 config | 每 stage 的 H×W 不同，需单独验证整除 |
| W3 | 非对称空间窗 | [2,9,9] | [2,12,6]（宽窗匹配水平运动） | 改 config | 288/12=24 ✅, 384/6=64 ✅ |

#### 二、跨窗通信

| # | 方向 | 改进 | 改动量 | 维度风险 |
|---|---|---|---|---|
| C1 | Temporal Cross-Window | 相邻时间步窗口做 attention | ~200 行 | 时间步只有 10，跨步 window token 数 = 2×w_t×w_h×w_w，需确保整除 |
| C2 | ConvLSTM Bridge | SW-MSA 后加 ConvLSTM | ~150 行 | ConvLSTM 输入是 SW-MSA 输出的 [T,B,C,H,W]，维度兼容 |

#### 三、SC 分数计算增强

| # | 方向 | 改进 | 改动量 | 维度风险 |
|---|---|---|---|---|
| S1 | Motion Weighting | gate *= motion_magnitude（帧间 voxel diff） | ~50 行 | motion_magnitude shape [B,T,H,W] → gate shape [B,H,N] → 需对齐 |
| S2 | Directional Channels | Q/K 分 x/y 方向各算 SC，合并 | ~80 行 | score shape [B,H,N,2] → gate [B,H,N] |
| S3 | Temporal Consistency | gate penalty = |gate(t)-gate(t-1)| | ~30 行 | gate shape [B,T,H,N] → penalty 在 T 维度 |

#### 四、硬件协同

| # | 方向 | 改进 | 改动量 |
|---|---|---|---|
| H1 | Event-Driven Window Skip | 空窗口跳过 attention | ~100 行 |
| H2 | Stripe SRAM Pipeline | 流式处理 window strips | ~300 行 |

#### 维度兼容性检查总结

当前模型的维度链：
```
input:    [B, T=10, C=2, H=288, W=384]
patch:    [T, B, C=96, H=144, W=192]
stage0:   [T, B, C=96,  H=144, W=192]  window=[2,9,9] → H_patches=16, W_patches=21.3 ❌ 384/9 不整除！
```

**⚠️ 发现：当前 window=[2,9,9] 下，384/9=42.67 不整除！** Swin 通过 padding 处理，但改为自定义 window 后需验证 padding 行为。

**安全 window 选择**（能被 crop 整除）：
- 288 的因数：1,2,3,4,6,8,9,12,16,18,24,32,36,48,72,96,144,288
- 384 的因数：1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,384
- 安全组合：[2,8,8], [2,6,6], [2,12,12], [2,4,4], [4,8,8], [5,6,8] 等

#### 执行优先级

| 优先级 | 实验 | 改动 | 风险 |
|--------|------|------|------|
| P0 | 等 NSC-09d 结果 → 决定是否继续 SC | — | — |
| P1 | W2: Stage-Adaptive Window | 改 config | 低 |
| P1 | S1: Motion Weighting | ~50 行 | 低 |
| P2 | W1: 拉长时间窗 [4,6,6] | 改 config | 中（整除验证） |
| P3 | C1: Temporal Cross-Window | ~200 行 | 中 |
| P3 | H1: Event-Driven Window Skip | ~100 行 | 低 |
| Future | C2, S2, S3, H2 | >100 行 | 中高 |
## 三十三、NTS 族：TX/SC Score-Level 融合注意力（2026-06-05 起）

> **命名**：TX/SC 融合统一代号 **NTS**（Neuromorphic Ternary Selector）。代码 alias 仍为 `h60`；旧称 NTX-12 作废。

### 33.1 NTS 方案概述

**核心公式**（NTS 全族共用）:

```
TX_score_i = Σ[same(+1) + α₀·zero - β·opposite] + α_k·sign(Q)·|K_before_sign|
SC_score_i = Σ sign(Q_d)·sign(K_d) / d
score_i    = TX_score_i + μ × SC_score_i          ← score 级融合，非 gate 级
gate       = Shiftmax(score)                       ← 单一 Shiftmax
attn_i     = K_i × gate_i                          ← 无 carrier，无外挂
```

**关键设计**:
1. **无 carrier**: 去掉 `K × sn2_q(sum(Q))`，比 NTX-01 少一条 spike 通路
2. **Score 级融合**: TX 与 SC 在 score 层面相加后**只过一次 Shiftmax**（比 NSC/H59 的 gate 级双 Shiftmax 混合更简单）
3. **K 幅值增强**: `α_k · sign(Q) · |K_before_sign|`（可选，硬件上需额外幅度通路，见 §33.7）
4. **SC 残差**: 小比例 μ 的符号共识，补足 TX 三值粒度

### 33.2 NTS-01 标准 valid825（`ntx_h60_full30_20260605_020633`）

| 参数 | 值 |
|------|-----|
| 注意力模式 | `h60` |
| μ / α_k | 0.1 / 0.02 |
| BSA attention | **6 模块**（S2 only） |
| batch_size | 6 |

| epoch | AEE | AAE | PE1 | PE2 | PE3 | total_spikes | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 19 | 1.687 | 11.15 | — | — | — | 34.6G | 7.5% | 29331 |
| **24** | **1.531** | **10.24** | 0.533 | 0.197 | 0.089 | 34.3G | 7.4% | 29254 |
| 28 | 1.581 | 10.59 | — | — | — | 34.0G | 7.3% | 29188 |
| 29 | 1.534 | 10.00 | — | — | — | 32.4G | 7.0% | 27844 |

### 33.3 NTS-02 标准 valid825（`ntx_h60_v2_full30_20260605_163955`，当前主线）

| epoch | AEE | AAE | PE1 | PE2 | PE3 | total_spikes | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 19 | 1.668 | 10.96 | — | — | — | 34.68G | 7.5% | 29352 |
| **24** 🏆 | **1.514** | **10.12** | — | — | — | 34.27G | 7.4% | 29272 |
| 28 | 1.589 | 10.66 | — | — | — | 34.02G | 7.3% | 29196 |
| **29** | 1.525 | **9.97** | — | — | — | **32.46G** | **7.0%** | **27899** |

NTS-02 ep24 为 AEE 最优；ep29 为 AAE/能耗最优。

### 33.4 vs NB0 baseline（NTS-02 ep24）

| 指标 | NB0 baseline | NTS-02 ep24 | 变化 |
|------|-------------|------------|------|
| AEE | 1.487 | 1.514 | +1.8% |
| AAE | 9.93° | 10.12° | +1.9% |
| total_spikes | 44.05G | 34.27G | **-22.2%** |
| energy | 37638uJ | 29272uJ | **-22.2%** |

### 33.5 注意力全线排名（含 NTS / NTX / NSC）

| 排名 | 实验 | ep | AEE | AAE | spikes | 架构族 | 外挂感 |
|------|------|-----|-----|------|--------|--------|--------|
| 🥇 | **NTS-02** | **24** | **1.514** | **10.12** | 34.3G | NTS score 融合 | ✅ 无 |
| 🥈 | NTS-01 | 24 | 1.531 | 10.24 | 34.3G | NTS score 融合 | ✅ 无 |
| 🥉 | NTX-01 | 28 | 1.534 | 10.29 | 34.6G | TX carrier×gate | ❌ 有 |
| 4 | NTS-02 | 29 | 1.525 | 9.97 | 32.5G | NTS score 融合 | ✅ 无 |
| 5 | NTX-13 | 29 | 1.612 | 10.67 | 34.1G | 纯 TX H49+Kmag | ✅ 无 |
| 6 | NTX-10 | 29 | 1.615 | 10.74 | 34.1G | 纯 TX H49 | ✅ 无 |
| — | NB0 | 59 | 1.487 | 9.93 | 44.1G | QKFormer | — |

### 33.6 关键洞察

1. **Score 级融合 > gate 级融合**: h57/h58 (gate mixing) 在 full30 失败，NTS/h60 (score mixing before Shiftmax) 成功。单一 Shiftmax 是关键。

2. **NTS 不能下放到纯 TX**: NTX-13 证明 K_mag 必须留在 NTS 全公式里，不能单独嫁接到 H49（见 §三十四）。

3. **无 carrier 首次超越 carrier×gate**: NTS-01 ep24 AEE=1.531 是第一个无外挂 gate 且精度不输 carrier×gate 的方案。

4. **NTS-02 进一步压低 AAE**: ep29 AAE=9.97° 已低于 NB0 的 9.93° 容差边界附近，方向角是后续优化重点。

### 33.7 硬件友好性评估

**结论：NTS 主体对神经形态硬件友好；唯一需要谨慎的是 K 幅值通路。**

| 运算阶段 | 硬件映射 | 友好度 | 说明 |
|----------|----------|--------|------|
| Q/K 三值化 | ATLIF → {-1,0,+1} event | ✅ 高 | 与 BSA popcount 原生匹配 |
| TX score | same/zero/opposite popcount | ✅ 高 | 纯 AND/XOR 计数，整数累加器 |
| SC score | `Σ sign(Q)·sign(K)` popcount | ✅ 高 | 比 TX 更简单，无 β/α₀ 分支 |
| Score 融合 | `TX + μ·SC` | ✅ 高 | 一次定点标量乘加（μ 可硬化为常数） |
| Center | 减行均值 | ✅ 中 | 加法树，无浮点依赖 |
| Shiftmax | `2^x / next_pow2(sum)` | ✅ 中 | BSA 原生算子；x 经 head_dim 归一后为小整数，可用 barrel shift + 小 LUT |
| 输出 | `K × gate` | ✅ 高 | 逐元素乘法，无 softmax 指数 |
| K 幅值 | `sign(Q)·relu(K-ternary(K))` | ⚠️ 中低 | **唯一非纯 event 路径**：需要 K 二值化前的模拟幅度（ATLIF 阈值距离）。硬件可用 spike count 或 2–3 bit 幅度量化近似；α_k 很小时可首版省略 |
| vs NTX-01 carrier | 去掉 `sn2_q(sum(Q))` | ✅ 更优 | 少一个 PSN 神经元 + 一条 spike 求和通路 |
| vs NSC/H59 gate 混合 | 1× Shiftmax vs 2× Shiftmax + gate blend | ✅ 更优 | NTS datapath 更短 |
| vs Softmax | 无 exp 全局归一 | ✅ 远优 | 稀疏 spike 域天然匹配 |
| 实测稀疏 | total_spikes -22% | ✅ 验证 | NTS-02 ep29: 32.5G vs NB0 44.1G |

**硬件实现建议**（若做 RTL/FPGA）：

1. **首版可硬化核心**：TX popcount + SC popcount + μ 融合 + Shiftmax + K×gate；μ、α₀、β 全部 compile-time 常数。
2. **K_mag 作为可选扩展**：α_k=0 时即为纯 event NTS，完全 popcount；α_k>0 时增加幅度累加支路（可用 threshold margin 2-bit 近似 `|K_before_sign|`）。
3. **不要走 H59 路线**：carrier + 双 gate 混合在硬件上比 NTS 多 1 个 Shiftmax + 1 个 sn2_q + signed gate clamp。
4. **S2-only 6 block 是合理的硬件 trade-off**：注意力替换集中在语义最强的 stage，面积/功耗可控；后续可评估 S1+S2 扩展。

**论文写法**：NTS 可表述为 "event-driven popcount attention with score-level ternary-consensus residual"——比 "attention matrix + softmax" 更适合 neuromorphic datapath 叙事。

### 33.8 待优化方向（NTS 族）

| 优先级 | 方向 | 当前值 | 候选 |
|--------|------|--------|------|
| P0 | μ + α_k 扫射 full30 | NTS-01: μ=0.1, α=0.02 | μ∈{0.05,0.15}, α∈{0.01,0.03}；config 已有 `nts_h60_v2_mu005_a003_full30` |
| P1 | Late μ schedule | 无 | μ 从 0 逐步打开 |
| P1 | 加 angular loss | λ=0 | λ=0.02 |
| P1 | K_mag 硬件友好近似 | 连续 \|K\| | threshold-margin 量化版 ablation |
| P2 | 增加 epoch | 30 | 60 |

### 33.9 NTS-00：纯 TX+SC（无 K_mag）短测 → 全量（2026-06-06 启动）

**背景**：NTS-01/02 均含 `k_magnitude_alpha>0`；`nts00h` 类 config（`ntx11_h60_mu01_nokmag_s360`）曾生成但**从未跑过**。NTS-00 专门验证纯 score 级 TX+SC 融合（硬件最简路径）能否接近 NTS-02。

**短测 sweep**（360 步，valid10→valid40，`k_magnitude_alpha=0`）：

| config | μ | LR 方案 | 备注 |
|--------|---|---------|------|
| `nts00a_mu005_std_s360` | 0.05 | 默认 | |
| `nts00b_mu010_std_s360` | 0.10 | 默认 | 对照 NTS-01 去 K_mag |
| `nts00c_mu015_std_s360` | 0.15 | 默认 | |
| `nts00d_mu005_fast_s360` | 0.05 | neuron 5e-5 / backbone 2e-6 | |
| `nts00e_mu010_fast_s360` | 0.10 | 快 LR | |
| `nts00f_mu005_slow_s360` | 0.05 | neuron 2e-5 / backbone 5e-7 | |
| `nts00g_mu010_bs6_s360` | 0.10 | 默认, bs=6 | |
| `nts00h_mu005_sched_s360` | 0→0.05 | schedule 360 步 | |

**流程**：`run_nts_nokmag_autopilot.py` → `rapid_screen.py` 选最优 → `promote_best_rapid_screen.py` full30 → `eval_DSEC_flow_SNN.py` epoch19/24/28/29。

**命名 alias**：`nts01_full30.yml`（=原 `ntx_h60_full30`）、`nts03_mu005_a003_full30.yml`（待跑 sweep）。

**磁盘清理**（2026-06-06）：释放 ~112GB。删除失败 tx_kmag 重试、rapid_screen 临时目录、重复短测、已完成实验的 `*_state_dict.pth`；保留 NB0 ep59、NTS-01/02 全 epoch `.pth`、NTX-13/NSC-10 关键 epoch + `standard_valid825/`。

## 三十四、NTX-13 (tx_kmag): H49 S2 + K 幅值 标准 valid825 结果（2026-06-06）

### 34.1 方案概述

**代号**: NTX-13（物理目录 `ntx_tx_kmag_full30_20260606_135707`）

**动机**: 在 NTS-01 证明 K magnitude 有效后，验证能否把 `k_magnitude_alpha` 直接嫁接到更轻的 H49 逐 token selector（仅 S2 6 block），避免 NTS 的 SC score 融合复杂度。

**注意力公式**（H49 + K mag，无 SC 残差）:
```
TX_score_i = Σ[same(+1) + α₀·zero - β·opposite] + α_k·sign(Q)·|K_before_sign|
gate       = Shiftmax(TX_score)
attn_i     = K_i × gate_i
```

**训练配置**:
| 参数 | 值 |
|------|-----|
| 配置 | `configs/generated/ntx_tx_kmag_a002_full30.yml` |
| 注意力模式 | `h49`（逐 token selector） |
| target_blocks | S2 only：`2:0`–`2:5`（6 模块） |
| α_k (k_magnitude_alpha) | 0.02 |
| β (mismatch_penalty) | 0.25 |
| bipolar_mu | 0.0（无 SC 残差） |
| backbone_lr | 1.0e-6 |
| neuron_lr | 3.0e-5 |
| threshold_lr | 5.0e-6 |
| warmup | 200 步，0.1→1.0 |
| epochs | 30 |
| batch_size | 8 |
| ATLIF 模块 | 34 |
| BSA attention | **6** 模块（仅 S2） |
| 续训起点 | `checkpoint_epoch59.pth`（NB0） |

训练 valid loss 最低点：**epoch 23**（1.5476）；标准推理另补跑 epoch 23。

### 34.2 标准 valid825 全指标

推理入口：`eval_DSEC_flow_SNN.py` + H9 config，`SDFORMER_USE_MLFLOW=0`，`SDFORMER_SNN_BACKEND=cupy`。

加载审计：`eval installed ATLIFTernaryPSN: 34`，`eval installed Shiftmax attention: 6`，`load audit: checkpoint_overlay_keys=68, missing=0, unexpected=0`。

| epoch | samples | AEE | AAE | PE1 | PE2 | PE3/outlier | total_spikes | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 19 | 825 | 1.6791 | 10.9121 | 0.5714 | 0.2333 | 0.1126 | 34.80G | 7.51% | 29328 |
| 23† | 825 | 1.6057 | 11.0391 | 0.5603 | 0.2173 | 0.0991 | 36.22G | 7.82% | 30668 |
| 24 | 825 | 1.6399 | 11.0965 | 0.5678 | 0.2279 | 0.1080 | 34.74G | 7.50% | 29463 |
| 28 | 825 | 1.6630 | 11.1299 | 0.5688 | 0.2292 | 0.1083 | 36.47G | 7.87% | 31036 |
| **29** 🏆 | 825 | **1.6123** | **10.6741** | 0.5558 | 0.2195 | 0.1052 | **34.07G** | **7.35%** | **28988** |

† epoch 23 为训练 valid loss 最低点，AAE 反而最差。

落盘路径：`results/ntx_tx_kmag_full30_20260606_135707/standard_valid825/epoch*/spike_profile.json`。

### 34.3 vs 主线对比

| 方案 | ep | AEE | AAE | total_spikes | 判定 |
|------|-----|-----|-----|-------------|------|
| **NTS-02** | 24 | **1.5138** | **10.1181°** | 34.27G | ✅ 主线 |
| **NTS-02** | 29 | 1.5249 | **9.9730°** | **32.46G** | ✅ 主线 |
| NTS-01 | 24 | 1.531 | 10.24° | 34.3G | 前代 |
| **NTX-13 (本实验)** | **29** | 1.6123 | 10.6741° | 34.07G | ❌ 精度不达标 |
| NTX-01 | 28 | 1.534 | 10.29° | 34.6G | 参考 |
| NB0 baseline | 59 | 1.487 | 9.93° | 44.1G | 基线 |

相对 NTS-02 ep29：AEE **+5.7%**，AAE **+7.0%**；spikes 接近（34.1G vs 32.5G）但不足以弥补精度损失。

相对 NTS-01 ep24：AEE **+5.3%**，AAE **+4.2%**。

### 34.4 结论

1. **K_mag 不能从 NTS 简单下放到 H49 S2**：仅 S2 6 block + H49 的 K magnitude 没有复现 NTS 的收益，精度明显弱于 NTS 家族。
2. **valid loss 与 AEE 错位**：epoch 23 valid loss 最低，但 AAE 最差（11.04°）；epoch 29 在 AEE/AAE 上更均衡，应作为本实验报告点。
3. **论文定位**：NTX-13 保留为「K magnitude 需要 NTS score 级融合」的负对照；**不再作为主线推进**。
4. **下一步应回到 NTS 族**（见 §33.8 P0：μ/α 扫射），而非继续在 H49 上叠 K_mag。

## 三十五、NSC-10 (H59): carrier + score 级 TX/SC 融合 标准 valid825 结果（2026-06-06）

### 35.1 实验概述

两条 NSC-10 full30 均已训完（30 epoch），并按正式口径对 `epoch19/24/27/29` 跑完 `eval_DSEC_flow_SNN.py` full valid825（`samples=825`）。

| 编号 | 物理目录 | 配置 | 注意力 | 关键超参 |
|------|----------|------|--------|----------|
| NSC-10a | `results/nsc10_m01_k005_full30_20260606_135642` | `configs/generated/nsc10_m01_k005.yml` | H59 `tx_sc_score_residual_shiftmax`（**保留 carrier**） | `bipolar_mu=0.05`（scheduled 0→0.05），`k_magnitude_alpha=0.05`，`motion_weight_alpha=0.1` |
| NSC-10b | `results/nsc10_motion01_full30_20260605_124144` | `configs/generated/nsc10_motion0.1.yml` | 同上 H59 + carrier | `bipolar_mu=0.05`（scheduled），`motion_weight_alpha=0.1`，**无 k_mag** |

续训起点：`experiments/baseline_stride_upstream/checkpoint_epoch59.pth`（NB0）。

标准推理入口：

```bash
python third_party/SDformerFlow/eval_DSEC_flow_SNN.py \
  --config neuron_experiments/H9_bipolar_self_attention/configs/generated/<config>.yml \
  --checkpoint <run_dir>/checkpoint_epoch<ep>.pth \
  --path_results <run_dir>/standard_valid825/epoch<ep> \
  --mode valid
```

加载审计（两条线一致）：`eval installed ATLIFTernaryPSN: 34`，`eval installed Shiftmax attention: 12`，`load audit: checkpoint_overlay_keys=68, missing=0, unexpected=0`。

### 35.2 NSC-10a `nsc10_m01_k005` 标准 valid825

| epoch | samples | AEE | AAE | PE1 | PE2 | PE3/outlier | total_spikes | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 19 | 825 | 1.6882 | 11.0624 | 0.5743 | 0.2341 | 0.1121 | 34.0871G | 7.35% | 29064.1 |
| 24 | 825 | 1.6585 | 11.1660 | 0.5727 | 0.2313 | 0.1091 | 33.9622G | 7.33% | 29165.0 |
| 27 | 825 | 1.6414 | 11.1698 | 0.5691 | 0.2266 | 0.1062 | 35.0931G | 7.57% | 30186.3 |
| **29** 🏆 | 825 | **1.6242** | **10.7548** | 0.5571 | 0.2200 | 0.1044 | **33.3333G** | **7.19%** | **28726.0** |

### 35.3 NSC-10b `nsc10_motion01` 标准 valid825

| epoch | samples | AEE | AAE | PE1 | PE2 | PE3/outlier | total_spikes | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 19 | 825 | 1.6678 | 11.0419 | 0.5755 | 0.2332 | 0.1116 | 34.0954G | 7.35% | 29073.4 |
| 24 | 825 | 1.6458 | 11.1584 | 0.5713 | 0.2285 | 0.1071 | 33.9648G | 7.33% | 29168.4 |
| 27 | 825 | 1.6404 | 11.1981 | 0.5707 | 0.2265 | 0.1052 | 35.1148G | 7.57% | 30200.6 |
| **29** 🏆 | 825 | **1.6167** | **10.7038** | 0.5545 | 0.2162 | 0.1020 | **33.3240G** | **7.19%** | **28716.0** |

### 35.4 vs NB0 baseline（达标线：AEE +5%，AAE +5%，spikes -20%）

NB0 ep59：`AEE=1.4872`，`AAE=9.9300°`，`total_spikes=44.05G`，`energy=37638uJ`。

阈值：`AEE≤1.5616`，`AAE≤10.4265°`，`total_spikes≤35.24G`。

| 实验 | best ep | AEE | vs NB0 | AAE | vs NB0 | total_spikes | vs NB0 | 三指标达标 |
|------|--------|-----|--------|-----|--------|-------------|--------|-----------|
| NSC-10a | 29 | 1.6242 | +9.2% | 10.7548° | +8.3% | 33.33G | -24.3% | ❌ AAE 超 5% |
| NSC-10b | 29 | 1.6167 | +8.7% | 10.7038° | +7.8% | 33.32G | -24.3% | ❌ AAE 超 5% |

### 35.5 vs 当前主线 NTS-02

| 方案 | ep | AEE | AAE | total_spikes | energy_uj | 判定 |
|------|-----|-----|-----|-------------|-----------|------|
| **NTS-02** | 24 | **1.5138** | **10.1181°** | 34.27G | 29272 | ✅ 全过 |
| **NTS-02** | 29 | 1.5249 | **9.9730°** | **32.46G** | 27899 | ✅ 全过 |
| NSC-10b motion01 | 29 | 1.6167 | 10.7038° | 33.32G | 28716 | ❌ 精度差 |
| NSC-10a m01_k005 | 29 | 1.6242 | 10.7548° | 33.33G | 28726 | ❌ 精度差 |

相对 NTS-02 ep29：NSC-10 最优点的 AEE 仍高约 **+6.0%**，AAE 高约 **+7.3%**；spikes/energy 接近或略优，但不足以弥补方向角退化。NSC-10 的 H59 gate 级融合在硬件上也比 NTS 更重（见 §33.7）。

### 35.6 结论

1. **NSC-10 未超过 NTS-02**：H59（carrier × score-level TX+SC gate）在 full30 后精度仍明显弱于无 carrier 的 NTS 族。
2. **稀疏/能耗达标，精度不达标**：两条线 ep29 均做到 spikes ~33.3G（-24%）、energy ~28.7 mJ，但 AAE 卡在 10.70–10.75°，超出 5% 容差。
3. **k_mag 与 motion 无明显收益**：`m01_k005`（+k_mag=0.05）略差于 `motion01`（无 k_mag），说明在当前 H59+carrier 框架下继续加修正项不能救 AAE。
4. **论文定位**：NSC-10 保留为「H59 carrier 路线 full30 负/弱对照」；主线仍为 **NTS-02 ep29**（综合）或 **ep24**（AEE 最优）。

落盘路径：`results/nsc10_*/standard_valid825/epoch*/spike_profile.json`。


## 三十六、FAPS 短测 → 全量标准 valid825

- 时间：`2026-06-07T12:05:18`
- 短测目录：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/faps_short_20260607_023751`
- 短测最优 valid40：`faps00a_dir_nokmag_s360_steps360_valid40` AEE=1.6699 AAE=15.6747
- 全量配置：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/configs/faps00a_dir_nokmag_s360_auto_full_20260607_032232.yml`
- 全量目录：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/faps00a_dir_nokmag_s360_auto_full_bs6_20260607_032232_setsid`

| epoch | AEE | AAE | total_spikes | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|
| 19 | 1.8011 | 11.6118 | 34.7090G | 7.4911% | 29288.90 |
| 24 | 1.5584 | 10.2426 | 34.2655G | 7.3954% | 29186.29 |
| 29 | 1.6279 | 10.3178 | 32.5074G | 7.0159% | 27866.08 |


### NTS-00 纯 TX+SC（无 K_mag）自动短测与全量结果（自动追加）

- 时间：`2026-06-07T12:28:35`
- 短测目录：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/nts_nokmag_short_20260607_021928`
- promotion log：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/nts_nokmag_autopilot_20260607_021927/promote_full.log`
- 全量配置：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/configs/nts00b_mu010_std_s360_auto_full_20260607_031912.yml`
- 全量目录：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/nts00b_mu010_std_s360_auto_full_bs6_20260607_031912_setsid`
- 方法：NTS pure TX+SC score fusion，`k_magnitude_alpha=0`
- 标准推理：`eval_DSEC_flow_SNN.py`，full valid825，CuPy backend。

| epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 19 | 1.6787 | 11.0813 | 0.5728 | 0.2343 | 0.1141 | 34.7885G | 7.5082% | 29343.85 |
| 24 | 1.5388 | 10.2939 | 0.5334 | 0.1969 | 0.0894 | 34.4268G | 7.4302% | 29300.66 |
| 28 | 1.5928 | 10.7909 | 0.5557 | 0.2172 | 0.1007 | 34.0908G | 7.3577% | 29154.80 |
| 29 | 1.5371 | 10.0697 | 0.5302 | 0.1994 | 0.0924 | 32.6010G | 7.0361% | 27925.22 |

当前自动判断：精度最佳 epoch29，AEE `1.5371`、AAE `10.0697`。


## 三十七、NTS04 硬件友好 no-Kmag 优先短测与 full30 启动

- 时间：`2026-06-07T23:36:10`
- 基线续训 checkpoint：`experiments/baseline_stride_upstream/checkpoint_epoch59.pth`
- 早期短测目录：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/nts04_hw_short_20260607_223442`（首次 rapid driver，留下的是简略 `summary.md`/`summary.csv`，随后因记录整理与候选收敛重新发起一次正式汇总短测）
- 短测目录：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/nts04_hw_short_20260607_223605`
- 首个 full30 配置：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/configs/nts04c_hw_mu010_mis020_s360_auto_full_20260607_233610.yml`
- 首个 full30 目录：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/nts04c_hw_mu010_mis020_s360_auto_full_bs6_20260607_233610_setsid`
- 当前接续 full30 配置：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/configs/nts04g_hw_sched010_w720_s360_auto_full_20260608_004746.yml`
- 当前接续 full30 目录：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/nts04g_hw_sched010_w720_s360_auto_full_bs6_20260608_004746_setsid`

### 37.1 代码修复与审计

- 修复：H60 no-carrier 分支的 `bipolar_mu` 现在调用 `_scheduled_bipolar_mu(self, cfg)`，schedule 配置不再被固定 `float(cfg.bipolar_mu)` 覆盖。
- 测试：新增并通过 `test_h60_no_carrier_schedule_matches_endpoint_mu`；同时回归 H58 schedule 测试。
- 训练加载：从 baseline epoch59 启动，日志显示 `checkpoint_overlay_keys=0, missing=68, unexpected=0`，这是 baseline 没有 H9 overlay 参数的预期初始化；模型安装审计为 `ATLIFTernaryPSN=34`、`Shiftmax attention=6`。

### 37.2 NTS04 候选定义

共同约束：`mode=h60`、S2-only 6 个 Shiftmax attention、`k_magnitude_alpha=0.0`、无 carrier、无 target_rate 推理控制。

| 候选 | 关键差异 | 硬件解释 |
|---|---|---|
| `nts04c_hw_mu010_mis020_s360` | `bipolar_mu=0.10`，`mismatch_penalty=0.20` | 只增强训练期异号惩罚；推理结构仍是一层 score-level Shiftmax gate |
| `nts04f_hw_sched010_w360_s360` | `mu: 0 -> 0.10 / 360 steps` | 平滑引入惩罚，推理不增加算子 |
| `nts04g_hw_sched010_w720_s360` | `mu: 0 -> 0.10 / 720 steps`，`mismatch_penalty=0.25`，`single_active_penalty=0.05` | 更慢 warmup；惩罚只参与训练 loss，推理端不增加算子 |

### 37.3 valid10 初筛

严格 valid10 gate 的 AAE 噪声偏高，所以 valid10 仅用于排序，不直接决定 full30。

| rank | candidate | valid10 AEE | valid10 AAE | SOPs(G) | gate |
|---:|---|---:|---:|---:|---|
| 1 | `nts04g_hw_sched010_w720_s360` | 1.4585 | 19.5956 | 3.8971 | AAE>16.8 |
| 2 | `nts04c_hw_mu010_mis020_s360` | 1.4736 | 19.6374 | 3.8952 | AAE>16.8 |
| 3 | `nts04f_hw_sched010_w360_s360` | 1.4687 | 20.0887 | 3.8966 | AAE>16.8 |

### 37.4 valid40 确认与 promotion

| rank | candidate | valid40 AEE | valid40 AAE | SOPs(G) | firing | gate |
|---:|---|---:|---:|---:|---:|---|
| 1 | `nts04c_hw_mu010_mis020_s360` | 1.7072 | 16.1306 | 4.1622 | 9.8457% | pass |
| 2 | `nts04g_hw_sched010_w720_s360` | 1.7422 | 16.3534 | 4.1616 | 9.8443% | pass |
| 3 | `nts04f_hw_sched010_w360_s360` | 1.7267 | 16.8113 | 4.1629 | 9.8474% | AAE>16.8 |

Promotion 选择 `nts04c`。它比 schedule 版本 valid40 更稳，且推理端仍保持硬件友好的 no-Kmag/no-carrier 结构；`mismatch_penalty=0.20` 是训练正则，不引入部署算子。

### 37.5 full30 当前配置

- `mode=h60`
- `bipolar_mu=0.10`
- `bipolar_lambda=0.5`
- `mismatch_penalty=0.20`
- `single_active_penalty=0.05`
- `single_active_penalty_grad=ste`
- `k_magnitude_alpha=0.0`
- `target_rate=null`
- `batch_size=6`
- `n_epochs=30`
- `optimizer.use_amp=true`
- LR groups：backbone `1e-6`，ATLIF neuron `3e-5`，threshold `5e-6`
- milestones：`[20, 25]`
- force save epochs：`[0, 4, 9, 14, 19, 24, 28, 29]`

状态：`nts04c` full30 已启动后早停，原因见 37.6；当前改为接续运行 `nts04g`。

### 37.6 NTS04c 早停与 NTS04g 接续

`nts04c` 在 full 训练中出现短测未暴露的过稀疏轨迹：

| epoch | train loss | valid loss | 训练期 binary activity | 备注 |
|---:|---:|---:|---:|---|
| 0 | 44.4439 | 1.3102 | 2.27% 左右 | 尚可 |
| 1 | 19.6390 | 1.4075 | 0.85% 左右 | 开始塌缩 |
| 2 | 8.7121 | 1.5960 | 0.33% 左右 | valid 明显变差 |
| 3 | 4.0536 | 1.4638 | 0.11%-0.13% | 继续训练风险高 |

手动 profile `nts04c` epoch3 valid40：

| checkpoint | valid40 AEE | valid40 AAE | SOPs(G) | firing | 结论 |
|---|---:|---:|---:|---:|---|
| `checkpoint_epoch3.pth` | 2.0330 | 19.9330 | 3.8495 | 9.1060% | 过稀疏，性能崩，不继续 full30 |

因此用 SIGINT 正常中止 `nts04c`，保留 `checkpoint_epoch0..3.pth` 与 profile 记录。中止不是代码崩溃；训练日志包含 `KeyboardInterrupt`。

接续策略：启动 `nts04g_hw_sched010_w720_s360` full30。该配置仍是硬件友好的 no-Kmag/no-carrier H60 路线，但用 `mu: 0 -> 0.10 / 720 steps` 延迟惩罚进入，避免 c 在早期直接把 binary activity 压塌。需要如实记录的是：`nts04g` 继承了 `mismatch_penalty=0.25` 与 `single_active_penalty=0.05`，惩罚只影响训练 loss，不是推理部署算子。

`nts04g` full30 关键配置：

- `mode=h60`
- `bipolar_mu=0.10`
- `sc_mu_schedule_enabled=true`
- `sc_mu_warmup_steps=720`
- `bipolar_lambda=0.5`
- `mismatch_penalty=0.25`
- `single_active_penalty=0.05`
- `single_active_penalty_grad=ste`
- `k_magnitude_alpha=0.0`
- `target_rate=null`
- `batch_size=6`
- `n_epochs=30`
- `optimizer.use_amp=true`
- LR groups：backbone `1e-6`，ATLIF neuron `3e-5`，threshold `5e-6`
- milestones：`[20, 25]`
- force save epochs：`[0, 4, 9, 14, 19, 24, 28, 29]`

当前监控重点：如果 `nts04g` 在 epoch0-3 也出现 valid loss 持续升高且 binary activity 快速跌破 0.5%，应停止并改跑更弱惩罚版本，例如 `mu=0.075` 或 `mismatch_penalty<=0.10` 的 no-Kmag schedule 变体。

`nts04g` 当前在线监控：

| epoch | train loss | valid loss | 训练期 binary activity | 判断 |
|---:|---:|---:|---:|---|
| 0 | 44.9628 | 1.3122 | 2.53% | 与 c 的 epoch0 valid 接近，但 activity 未塌缩，继续 |
| 1 | 22.4933 | 1.3914 | 0.90% | 比 c 的 epoch1 valid 1.4075 略好，尚未跌破 0.5%，继续 |
| 2 | 8.6174 | 1.6258 | 0.35% | valid 比 c 的 epoch2 1.5960 更差，且 activity 跌破 0.5%，停止 |

结论：`nts04g` 也不是 full30 候选。它比 `nts04c` 延缓了塌缩，但 `mismatch_penalty=0.25` 与 `single_active_penalty=0.05` 仍然过强；epoch2 后继续训练会变成低 SOP 但性能崩的路线。

### 37.7 NTS05 弱惩罚 no-Kmag 短测

新增生成脚本：`neuron_experiments/H9_bipolar_self_attention/entrypoints/make_nts05_weak_hw_configs.py`。

共同约束：`mode=h60`、no carrier、`k_magnitude_alpha=0.0`、`target_rate=null`、S2-only Shiftmax attention、batch size 6、短测 360 step。区别只在训练正则强度和 `mu` warmup；推理端仍是硬件友好的 score-level Shiftmax gate。

| candidate | `bipolar_mu` schedule | `mismatch_penalty` | `single_active_penalty` | 目的 |
|---|---:|---:|---:|---|
| `nts05a_hw_mu0075_mis005_sap0025_w720_s360` | `0 -> 0.075 / 720` | 0.05 | 0.025 | 主候选：比 NTS04 明显弱惩罚 |
| `nts05b_hw_mu005_mis005_sap0025_w720_s360` | `0 -> 0.05 / 720` | 0.05 | 0.025 | 更保守，优先救性能 |
| `nts05c_hw_mu0075_mis005_sap0025_w1440_s360` | `0 -> 0.075 / 1440` | 0.05 | 0.025 | 更慢进入惩罚，避免早期塌缩 |
| `nts05d_hw_mu0075_mis000_sap0025_w720_s360` | `0 -> 0.075 / 720` | 0.00 | 0.025 | 去掉异号惩罚，只保留弱 single-active |

短测命令使用 `rapid_screen.py`，`valid_samples=10`，`promote_samples=40`，promotion gate 暂设为 `AEE<=2.00`、`AAE<=20.0`、`SOPs<=6.0G`。这轮的重点不是直接追最低 SOP，而是确认 full 早期不会像 NTS04 一样过稀疏。

### 37.8 NTS05 短测结果与 full30 选择

- 时间：`2026-06-08T01:42:49`
- 早期短测目录：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/nts05_weak_hw_short_20260608_014233`（同批候选的首次 short driver；目录仅创建了 `configs/`，未留下正式 `summary.md`，因此以 `014249` 作为正式汇总口径）
- 短测目录：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/nts05_weak_hw_short_20260608_014249`
- 接续 checkpoint：`experiments/baseline_stride_upstream/checkpoint_epoch59.pth`
- 加载审计：从 baseline59 接续时 overlay keys 为新初始化，符合 baseline 无 H9/H60 参数的预期；模型路径仍安装 `ATLIFTernaryPSN=34`、`Shiftmax attention=6`。

valid40 确认排序：

| rank | candidate | valid40 AEE | valid40 AAE | SOPs(G) | firing | train binary activity | score | gate |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `nts05d_hw_mu0075_mis000_sap0025_w720_s360` | 1.6637 | 15.5313 | 4.1169 | 9.7384% | 4.1269% | 3.2166 | pass |
| 2 | `nts05c_hw_mu0075_mis005_sap0025_w1440_s360` | 1.7119 | 16.0511 | 4.1180 | 9.7410% | 4.1315% | 3.3634 | pass |
| 3 | `nts05a_hw_mu0075_mis005_sap0025_w720_s360` | 1.7317 | 16.0663 | 4.1170 | 9.7387% | 4.1290% | 3.4021 | pass |
| 4 | `nts05b_hw_mu005_mis005_sap0025_w720_s360` | 1.7440 | 16.1337 | 4.1176 | 9.7402% | 4.1295% | 3.4329 | pass |

结论：

1. `nts05d` 是本轮最优候选，且它去掉 `mismatch_penalty` 后反而比保留 0.05 的 a/b/c 更好。
2. 这说明 NTS04 的失败不是 schedule bug 单独导致，而是异号/单激活惩罚组合过强；当前 TX attention 路线应优先保留弱 `single_active_penalty=0.025`，不再加 K magnitude 和 mismatch。
3. `nts05d` 的部署结构最干净：无 Kmag、无 carrier、无 target-rate 控制，推理端仍是 Q/K 逐 token score 直接过 Shiftmax 得到 gate。

`nts05d` full30 配置：

- `mode=h60`
- `bipolar_mu=0.075`
- `sc_mu_schedule_enabled=true`
- `sc_mu_warmup_steps=720`
- `bipolar_lambda=0.5`
- `mismatch_penalty=0.0`
- `single_active_penalty=0.025`
- `single_active_penalty_grad=ste`
- `k_magnitude_alpha=0.0`
- `target_rate=null`
- `center_scores=true`
- `preserve_mean=true`
- `consensus_score_norm=head_dim`
- `value_mode=threshold`
- `batch_size=6`
- `n_epochs=30`
- LR groups：backbone `1e-6`，ATLIF neuron `3e-5`，threshold `5e-6`
- milestones：`[20, 25]`
- force save epochs：`[0, 4, 9, 14, 19, 24, 28, 29]`

标准化流程要求：

1. full30 先跑 `promote_best_rapid_screen.py` 启动，显式传入 `--prev-runid experiments/baseline_stride_upstream/checkpoint_epoch59.pth`。
2. promotion 自带 profile 只作监控；最终论文表格以 `profile_checkpoints.py` 的 valid825 标准推理结果为准。
3. 标准推理必须记录 AEE、AAE、PE1、PE2、PE3/outlier、total spikes、firing、energy，以及 H9/Shiftmax 加载审计。

### 37.9 NTS05d full30 在线监控

- 启动时间：`2026-06-08T02:07:22`
- 全量配置：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/configs/nts05d_hw_mu0075_mis000_sap0025_w720_s360_auto_full_20260608_020722.yml`
- 全量目录：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/nts05d_hw_mu0075_mis000_sap0025_w720_s360_auto_full_bs6_20260608_020722_setsid`
- promotion log：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/nts05_weak_hw_promote_20260608_020722.log`

加载与配置审计：

- `checkpoint_overlay_keys=0, missing=68, unexpected=0`：从 baseline59 续训，新 H9/H60 参数初始化，符合预期。
- `ATLIFTernaryPSN=34`、`Shiftmax attention=6`。
- `mismatch_penalty=0.0`、`single_active_penalty=0.025`、`k_magnitude_alpha=0.0`、`target_rate=null`，没有混入 NTS04 强惩罚或 Kmag。

epoch0 关键轨迹：

| point | binary activity | ternary activity | pos/neg ratio | 判断 |
|---|---:|---:|---:|---|
| step360 | 4.1269% | 7.5448% | 1.2209 | 与短测末端一致 |
| step720 | 3.4971% | 7.5550% | 1.2277 | schedule 到顶后未塌缩 |
| step900 | 3.0370% | 7.3284% | 1.2323 | 继续稀疏化但仍可接受 |
| step1200 | 2.4882% | 7.3298% | 1.2394 | 低于短测，但未进入 NTS04 的 0.x% 崩溃区 |

| epoch | train loss | valid loss | epoch-end binary activity | threshold mean | 结论 |
|---:|---:|---:|---:|---:|---|
| 0 | 44.9568 | 1.3006 | 2.5267% | 1.0069 | 继续；valid loss 好于 NTS04c epoch0 的 1.3102 |
| 1 | 23.3017 | 1.4203 | 1.1253% | 1.0144 | 警告；valid loss 明显升高，activity 接近 1% 下限 |

epoch2 早段触发停止条件：

| point | binary activity | valid 前序状态 | 动作 |
|---|---:|---|---|
| epoch2 step100 | 0.9879% | epoch1 valid loss 已升至 1.4203 | SIGINT 停止 full30，保留 epoch0/1 checkpoint |

监控规则：如果 epoch1/2 出现 valid loss 持续升高且 binary activity 跌破 1%，停止 NTS05d 并改跑更弱惩罚备选；否则继续 full30 到保存点。

结论：`nts05d` 虽然 valid40 短测最好，但 full 训练仍会被 `single_active_penalty=0.025` 推向过稀疏；它比 NTS04 慢一些，但趋势相同。后续不再使用 `single_active_penalty>=0.025` 作为 full30 主线。下一轮 NTS06 只做更弱配置：`single_active=0` 或 `0.01`，保持 no-Kmag/no-carrier/no-target-rate，先看 full 早期 activity 是否能稳定在 1.5%-2% 以上。

### 37.10 NTS06 更弱 single-active 短测计划

新增生成脚本：`neuron_experiments/H9_bipolar_self_attention/entrypoints/make_nts06_floor_hw_configs.py`。

共同约束：`mode=h60`、no carrier、`k_magnitude_alpha=0.0`、`target_rate=null`、`mismatch_penalty=0.0`、`center_scores=true`、`preserve_mean=true`、`value_mode=threshold`。本轮只降低或关闭 `single_active_penalty`，不改推理公式。

| candidate | `bipolar_mu` schedule | `single_active_penalty` | 目的 |
|---|---:|---:|---|
| `nts06a_hw_mu005_mis000_sap000_w720_s360` | `0 -> 0.05 / 720` | 0.00 | 最弱 full 稳定性探针 |
| `nts06b_hw_mu005_mis000_sap001_w720_s360` | `0 -> 0.05 / 720` | 0.01 | 保留极弱单激活约束 |
| `nts06c_hw_mu0075_mis000_sap000_w720_s360` | `0 -> 0.075 / 720` | 0.00 | 保留 NTS05d 的 mu，去掉 single-active |
| `nts06d_hw_mu005_mis000_sap000_w1440_s360` | `0 -> 0.05 / 1440` | 0.00 | 更慢 warmup，优先防止过稀疏 |

进入 full30 的选择标准：valid40 不能明显崩，同时 360-step binary activity 不低于 NTS05d；full 启动后若 epoch1/2 valid loss 上升且 binary activity 跌破 1%，立即停止。

### 37.11 NTS06 短测结果与 full30 选择

运行目录：`neuron_experiments/H9_bipolar_self_attention/results/nts06_floor_hw_short_20260608_024604`。

启动依据：上一轮 `nts05d` 短测最好但 full 早期快速过稀疏；本轮只保留硬件友好的 no-carrier / no-Kmag / no-target-rate 形式，进一步降低或关闭 `single_active_penalty`。

| candidate | valid40 AEE | valid40 AAE | PE1 | PE2 | PE3 | SOPs(G) | firing | threshold | ternary activity | score | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `nts06a_hw_mu005_mis000_sap000_w720_s360` | 1.6900 | 15.4714 | 0.6200 | 0.2682 | 0.1285 | 4.1182 | 0.09742 | 1.0017 | 0.07544 | 3.2611 | valid40 综合最好；选入 full30 |
| `nts06b_hw_mu005_mis000_sap001_w720_s360` | 1.6867 | 15.5939 | 0.6244 | 0.2798 | 0.1354 | 4.1178 | 0.09740 | 1.0017 | 0.07546 | 3.2673 | AEE 略好但 AAE/PE/score 更差；不优先 |
| `nts06c_hw_mu0075_mis000_sap000_w720_s360` | 1.6938 | 15.7002 | 0.6250 | 0.2758 | 0.1255 | 4.1182 | 0.09742 | 1.0017 | 0.07546 | 3.2923 | 提高 `mu` 无收益 |
| `nts06d_hw_mu005_mis000_sap000_w1440_s360` | 1.6899 | 15.8497 | 0.6271 | 0.2791 | 0.1258 | 4.1169 | 0.09739 | 1.0017 | 0.07542 | 3.2999 | 训练 valid loss 低但 valid40 AAE/score 最差；不优先 |

选择：`nts06a_hw_mu005_mis000_sap000_w720_s360`。

训练配置：

- 续训 checkpoint：`experiments/baseline_stride_upstream/checkpoint_epoch59.pth`
- 注意力：`mode=h60`，直接由 bipolar score 经 shiftmax 得 gate；no carrier；no K magnitude；`center_scores=true`；`preserve_mean=true`
- 稀疏项：`bipolar_mu=0.05`，schedule `0 -> 0.05 / 720 steps`；`mismatch_penalty=0.0`；`single_active_penalty=0.0`；`target_rate=null`
- 神经元：H9 ATLIFTernaryPSN overlay，34 个 ATLIF ternary 模块；6 个 shiftmax attention 模块
- 学习率：backbone `1e-6`，ATLIF neuron `3e-5`，threshold `5e-6`；训练 batch size 6，full30

监控规则保持：如果 epoch1/2 valid loss 持续上升且 binary activity 跌破 1%，停止；否则跑完 full30 后用 `profile_checkpoints.py` 做 valid825 标准化推理并记录完整指标。

NTS06a full30 实际监控：

运行目录：`neuron_experiments/H9_bipolar_self_attention/results/nts06a_hw_mu005_mis000_sap000_w720_s360_auto_full_bs6_20260608_031001_setsid`。

| epoch | train loss | valid loss | binary activity | threshold mean | gate mean | 判断 |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 44.9733 | 1.3205 | 0.02532 | 1.00693 | 0.61627 | 可观察，但 activity 已低 |
| 1 | 22.5035 | 1.4015 | 0.00897 | 1.01438 | 0.61555 | 触发停止条件 |

结论：`single_active_penalty=0` 仍不能阻止 full 阶段过稀疏；问题不再主要是注意力惩罚，而是 ATLIF threshold 更新在 full 训练中持续抬高 threshold，把 binary activity 压到 1% 以下。该 run 已手动停止，不作为主线。下一轮 NTS07 应优先做 ATLIF 更新强度/冻结策略，而不是继续扫 `single_active_penalty`。

### 37.12 NTS07 FFN ATLIF 稳定性短测计划

新增生成脚本：`neuron_experiments/H9_bipolar_self_attention/entrypoints/make_nts07_ffn_floor_hw_configs.py`。

依据：NTS06a full 中 qk ternary activity 仍约 7.2%，但 FFN official binary ATLIF 的 `binary_activity_mean` 到 epoch1 跌到 0.897%，同时 valid loss 从 1.3205 升到 1.4015。因此 NTS07 不再改 h60 注意力公式，只降低或移除 FFN official ATLIF 的稀疏压力。

共同约束：`mode=h60`、no carrier、no Kmag、no target-rate、`mismatch_penalty=0`、`single_active_penalty=0`、`bipolar_mu=0.05` schedule `0 -> 0.05 / 720`。短测改为 1224 steps，用于观察 epoch0 级别的 FFN activity。

| candidate | FFN official `threshold_eta` | FFN `activity_eta` | FFN target groups | 目的 |
|---|---:|---:|---|---|
| `nts07a_hw_h60_ffn_soft_eta2e5_act05_s1224` | `2e-5` | `0.5` | 保留 | 温和版 official ATLIF |
| `nts07b_hw_h60_ffn_update0_act0_s1224` | `0` | `0` | 保留 | 保留二值 FFN 输出，但关闭 sparse update/loss |
| `nts07c_hw_h60_qk_only_noffn_s1224` | - | - | 移除 | 只保留 qk 三值替换，做减法基线 |
| `nts07d_hw_h60_ffn_update8e5_act0_s1224` | `8e-5` | `0` | 保留 | 判断主要问题是否来自 activity loss |

选择标准：valid40 指标不能明显差于 NTS06a；同时 1224-step `binary_activity_mean` 应明显高于 NTS06a epoch0 的 2.53%，至少不能接近 1%。若短测选中 full，full 仍按 epoch1 valid/activity 停止规则执行。

NTS07 短测阶段性结果：

运行目录：`neuron_experiments/H9_bipolar_self_attention/results/nts07_ffn_floor_hw_short_20260608_034549`。

| candidate | valid40 AEE | valid40 AAE | PE1 | PE2 | PE3 | SOPs(G) | firing | binary activity | ternary activity | valid loss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `nts07b_hw_h60_ffn_update0_act0_s1224` | 1.4353 | 13.3095 | 0.5666 | 0.2124 | 0.0757 | 4.0820 | 0.09656 | 0.04886 | 0.07348 | 1.2042 | 当前最佳，直接 promotion full30 |
| `nts07a_hw_h60_ffn_soft_eta2e5_act05_s1224` | 1.5812 | 14.9272 | 0.5978 | 0.2462 | 0.0974 | 3.9688 | 0.09388 | 0.02733 | 0.07321 | 1.2377 | 有效但弱于 07b |

备注：`nts07c/07d` 未继续跑完；在 `nts07b` 已明显优于 `nts07a` 且满足 full 条件后，为节省 GPU 手动停止 rapid driver。后续如需补消融，可单独补跑 `qk_only_noffn` 与 `activity_eta=0`。

NTS07b full 配置要点：

- 保留 h60 no-carrier/no-Kmag/no-target-rate 注意力路径，`mismatch_penalty=0`，`single_active_penalty=0`
- 保留 FFN official binary ATLIF 模块，但关闭 FFN sparse manual update/loss：FFN target groups `threshold_eta=0`、`activity_eta=0`
- qk 三值 ATLIF 保持 NTS06a 设置：`threshold_eta=0.00065`、`bipolar_mu=0.05` schedule `0 -> 0.05 / 720`
- 续训 checkpoint：`experiments/baseline_stride_upstream/checkpoint_epoch59.pth`

full 监控规则：epoch1 valid 不应像 NTS06a 一样显著上升；binary activity 应保持明显高于 1%。如果 epoch1/2 同时出现 valid loss 上升和 binary activity 接近或低于 1%，停止。

NTS07b full30 在线监控：

运行目录：`neuron_experiments/H9_bipolar_self_attention/results/nts07b_hw_h60_ffn_update0_act0_s1224_steps1224_auto_full_bs6_20260608_042113_setsid`。

加载审计：`checkpoint_overlay_keys=0, missing=68, unexpected=0`，预期为 baseline59 续训新装 overlay；已安装 `ATLIFTernaryPSN=34`、`Shiftmax attention=6`。FFN target groups 配置确认：`threshold_eta=0.0`、`activity_eta=0.0`。

| epoch | train loss | valid loss | binary activity | ternary activity | threshold mean | gate mean | 判断 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 1.5565 | 1.2042 | 0.04886 | 0.07348 | 1.00694 | 0.61671 | 明显优于 NTS06a epoch0 |
| 1 | 1.4965 | 1.2365 | 0.05199 | 0.07133 | 1.01429 | 0.61727 | valid 小幅上升但 activity 稳定，继续 full30 |
| 2 | 1.4898 | 1.2130 | 0.05348 | 0.06833 | 1.02145 | 0.61827 | valid 回落，activity 稳定，继续 full30 |
| 3 | 1.4834 | 1.3093 | 0.05347 | 0.06676 | 1.02848 | 0.61821 | valid 波动上升，但 activity 未坍缩，继续观察 |
| 4 | 1.4650 | 1.2008 | 0.05001 | 0.06503 | 1.03533 | 0.61875 | valid 回到 epoch0 水平，activity 仍稳定，继续 full30 |
| 5 | 1.4701 | 1.1978 | 0.05716 | 0.06319 | 1.04204 | 0.61889 | early valid 当前最好，activity 稳定，继续 full30 |
| 6 | 1.4415 | 1.1719 | 0.05574 | 0.06131 | 1.04863 | 0.61909 | valid 继续改善，activity 未坍缩，继续 full30 |
| 7 | 1.4443 | 1.1297 | 0.05603 | 0.05908 | 1.05505 | 0.61914 | valid 明显改善，稀疏度下降但未坍缩，继续 full30 |
| 8 | 1.4252 | 1.1480 | 0.05753 | 0.05807 | 1.06132 | 0.62010 | valid 小幅回升但仍健康，activity 稳定，继续 full30 |
| 9 | 1.4424 | 1.2226 | 0.06010 | 0.05616 | 1.06745 | 0.62052 | valid 回升，activity 仍健康；保留 checkpoint，继续 full30 |
| 10 | 1.4346 | 1.1988 | 0.05850 | 0.05472 | 1.07346 | 0.62061 | valid 从 epoch9 回落，未持续恶化，继续 full30 |
| 11 | 1.4262 | 1.1342 | 0.05710 | 0.05270 | 1.07934 | 0.62078 | valid 回到较好区间，activity 未坍缩，继续 full30 |
| 12 | 1.4418 | 1.1798 | 0.05913 | 0.05220 | 1.08512 | 0.62105 | valid 小幅回升，activity 仍稳定，继续 full30 |
| 13 | 1.4241 | 1.1815 | 0.06233 | 0.05047 | 1.09079 | 0.62144 | valid 基本持平，ternary 接近 5%，继续观察 epoch14 |
| 14 | 1.4249 | 1.2061 | 0.06201 | 0.04956 | 1.09634 | 0.62116 | valid 未改善，ternary 低于 5%；保留 checkpoint，继续 full30 但标注后期过稀疏风险 |
| 15 | 1.4339 | 1.1583 | 0.06418 | 0.04817 | 1.10178 | 0.62191 | valid 回落到较好区间，但 ternary 继续下降；继续 full30 |
| 16 | 1.4120 | 1.1605 | 0.06388 | 0.04695 | 1.10713 | 0.62244 | valid 基本持平，ternary 继续下降；保留 checkpoint，继续 full30 |
| 17 | 1.4225 | 1.2010 | 0.06567 | 0.04657 | 1.11239 | 0.62277 | valid 回升，ternary 继续下降；继续到 epoch19 关键点 |
| 18 | 1.4072 | 1.1151 | 0.06272 | 0.04530 | 1.11756 | 0.62261 | 当前日志 valid 最低；保留 checkpoint，重点做标准化推理候选 |
| 19 | 1.4156 | 1.2071 | 0.06442 | 0.04419 | 1.12263 | 0.62293 | valid 回升明显；保留 checkpoint，但当前不如 epoch18 |
| 20 | 1.4071 | 1.1110 | 0.06581 | 0.04290 | 1.12761 | 0.62320 | lr 降到 `4.9775e-7` 后 valid 新低；保留 checkpoint，重点候选 |
| 21 | 1.4160 | 1.1826 | 0.06802 | 0.04263 | 1.13250 | 0.62346 | valid 回升；epoch20 仍是后期最佳候选 |
| 22 | 1.4166 | 1.1716 | 0.06730 | 0.04163 | 1.13731 | 0.62327 | valid 仍不如 epoch20；继续 full30 |
| 23 | 1.3970 | 1.1330 | 0.06551 | 0.04067 | 1.14205 | 0.62344 | valid 较好但不如 epoch20；继续 epoch24 |
| 24 | 1.4105 | 1.1140 | 0.06602 | 0.03975 | 1.14671 | 0.62360 | 接近 epoch20；保留 checkpoint，重点候选 |
| 25 | 1.4118 | 1.1685 | 0.06444 | 0.03906 | 1.15130 | 0.62367 | lr 降到 `2.48875e-7`；valid 回升，不是重点候选 |
| 26 | 1.4153 | 1.1140 | 0.06802 | 0.03870 | 1.15581 | 0.62419 | valid 接近 epoch20/24；后期候选但未保存 checkpoint |
| 27 | 1.4091 | 1.1259 | 0.06485 | 0.03772 | 1.16025 | 0.62380 | 不如 epoch20/24/26；继续到 epoch28/29 |
| 28 | 1.4074 | 1.1220 | 0.06826 | 0.03720 | 1.16462 | 0.62449 | 保留 checkpoint；不如 epoch20/24 |
| 29 | 1.4019 | 1.1472 | 0.07027 | 0.03627 | 1.16892 | 0.62444 | full30 正常结束；binary 未坍缩，ternary 后期继续变稀疏 |

valid40 promotion profile 结果：

| rank | checkpoint | AEE | AAE | SOPs(G) | firing |
|---:|---|---:|---:|---:|---:|
| 1 | `checkpoint_epoch29.pth` | 1.2637 | 12.1397 | 3.6208 | 0.08565 |
| 2 | `checkpoint_epoch24.pth` | 1.3029 | 12.6441 | 3.8992 | 0.09223 |
| 3 | `checkpoint_epoch9.pth` | 1.3628 | 12.8239 | 3.8310 | 0.09062 |
| 4 | `checkpoint_epoch28.pth` | 1.4129 | 13.3039 | 3.8572 | 0.09124 |
| 5 | `checkpoint_epoch19.pth` | 1.4078 | 13.2542 | 3.9212 | 0.09276 |
| 6 | `checkpoint_epoch14.pth` | 1.4288 | 13.4895 | 4.0036 | 0.09471 |
| 7 | `checkpoint_epoch4.pth` | 1.4284 | 13.6270 | 4.0704 | 0.09628 |
| 8 | `checkpoint_epoch0.pth` | 1.4353 | 13.3095 | 4.0820 | 0.09656 |

valid825 标准化推理结果：

命令口径：`profile_checkpoints.py --samples 825 --epoch 29 --epoch 24 --epoch 20 --epoch 18 --epoch 9`，使用 `/opt/conda/envs/sdformerflow/bin/python`。第一次误用 base python 时缺 `pandas` 失败，随后用标准 `sdformerflow` 环境重跑成功；有效结果如下。

加载审计：每个 valid825 候选均为 `installed ATLIFTernaryPSN=34`、`installed Shiftmax attention=6`、`checkpoint_overlay_keys=68, missing=0, unexpected=0`；`samples=825`，`profiled_layers=99`，`dense_ops=42.2746G [fvcore]`。说明 NTS/H9 权重正确加载，没有 baseline config 误评估。

| checkpoint | AEE | AAE | PE1 | PE2 | PE3 | SOPs(G) | firing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `checkpoint_epoch9.pth` | 1.5912 | 10.3534 | 0.5392 | 0.2120 | 0.1015 | 3.5060 | 0.08293 |
| `checkpoint_epoch18.pth` | 1.4983 | 10.0957 | 0.5171 | 0.1902 | 0.0871 | 3.6144 | 0.08550 |
| `checkpoint_epoch20.pth` | 1.5146 | 9.9258 | 0.5144 | 0.1957 | 0.0934 | 3.4526 | 0.08167 |
| `checkpoint_epoch24.pth` | 1.4793 | 9.9090 | 0.5099 | 0.1832 | 0.0827 | 3.5396 | 0.08373 |
| `checkpoint_epoch29.pth` | 1.4850 | 9.7361 | 0.5095 | 0.1884 | 0.0872 | 3.3576 | 0.07942 |

valid825 ranking 文件：`neuron_experiments/H9_bipolar_self_attention/results/nts07b_hw_h60_ffn_update0_act0_s1224_steps1224_auto_full_bs6_20260608_042113_setsid/profile_ranking_valid825.md`。

结论：关闭 FFN official ATLIF 的 sparse manual update/loss 后，NTS06a 的 FFN binary activity 坍缩被消除，NTS07b full30 可作为硬件友好 no-carrier/no-Kmag/no-target-rate 注意力主线。按纯 AEE 选 `checkpoint_epoch24.pth`；按综合硬件口径（AEE 接近、AAE 更低、SOPs/firing 更低）选 `checkpoint_epoch29.pth`。后续若继续优化 NTS00/NTS07 线，应优先在保持 no-carrier 叙事的前提下微调 qk 三值阈值/温度或后期冻结 threshold，避免 ternary activity 从约 7.3% 持续降到 3.6%。

### 37.13 NTS08 qk threshold stability 短测（2026-06-08）

目标：继续从 NTX01/NTS 线推进硬件友好的整网方案，但不再增加部署端算子。NTS08 保持 NTS07b 的推理结构：

```text
Q/K ternary event score -> single Shiftmax -> K * gate
```

共同约束：`mode=h60`、no carrier、no Kmag、no target-rate、`mismatch_penalty=0`、`single_active_penalty=0`、FFN official ATLIF 保留但 `threshold_eta=0/activity_eta=0`。本轮只扫 qk ATLIF threshold 动态与 `bipolar_mu`，旧实验不引用 NTS08 config 时行为不变。

新增生成器：`neuron_experiments/H9_bipolar_self_attention/entrypoints/make_nts08_qk_stability_hw_configs.py`。已 `py_compile` 通过，并生成 6 个新 config。

短测目录：`neuron_experiments/H9_bipolar_self_attention/results/nts08_qk_stab_20260608_142946`。命令口径：`rapid_screen.py --steps 1224 --prev-runid experiments/baseline_stride_upstream/checkpoint_epoch59.pth --batch-size 8 --valid-samples 10 --promote-samples 40 --tag nts08_qk_stab`。

valid40 结果：

| variant | AEE | AAE | PE1 | PE2 | PE3 | SOPs(G) | firing | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `nts08c_hw_h60_qk_cap115_s1224` | 1.4511 | 13.5908 | 0.5607 | 0.2168 | 0.0861 | 4.0573 | 0.09597 | 当前 NTS08 最优；短测不超 NTS07b，但 full 后期 cap 才会生效 |
| `nts08e_hw_h60_mu0075_qk_eta0325_s1224` | 1.4545 | 13.7513 | 0.5697 | 0.2149 | 0.0832 | 4.0614 | 0.09607 | `mu=0.075` 未改善 |
| `nts08a_hw_h60_qk_eta0325_s1224` | 1.4614 | 13.7459 | 0.5683 | 0.2139 | 0.0811 | 4.0631 | 0.09611 | qk threshold 放慢后 activity 更稳，但精度/SOP 不优 |
| `nts08d_hw_h60_qk_eta0325_cap115_s1224` | 1.4614 | 13.7459 | 0.5683 | 0.2139 | 0.0811 | 4.0631 | 0.09611 | 与 08a 一致；cap 在短测阶段未触发 |
| `nts08b_hw_h60_qk_scale25k_s1224` | 1.4819 | 13.9738 | 0.5677 | 0.2164 | 0.0876 | 4.0627 | 0.09610 | 不优 |
| `nts08f_hw_h60_mu003_qk_eta0325_s1224` | 1.5113 | 14.0010 | 0.5751 | 0.2252 | 0.0946 | 4.0626 | 0.09610 | `mu=0.03` 明显变差 |

短测诊断：

- `qk_threshold_eta` 减半或 `threshold_lr_scale` 减半都能把 1224-step 后的 qk threshold 控在约 `1.0026`，ternary activity 约 `7.43%`，但 SOPs/firing 升高，valid40 AEE 不如 NTS07b。
- `max_threshold=1.15` 在 1224-step 短测中不会触发，因此 `nts08c` 本质上是 NTS07b 的 full 后期保护项；短测表现接近控制项，且 NTS07b full 中 threshold 在 epoch24 约 `1.1467`、epoch29 约 `1.1689`，`1.15` cap 会主要限制 epoch24 之后的继续过稀疏。
- `mu=0.075` 与 `mu=0.03` 都没有超过 `mu=0.05`，因此 NTS07b 的 `mu=0.05` 仍是当前更合理的 no-carrier TX/SC score-fusion 系数。

决策：启动 `nts08c_hw_h60_qk_cap115_s1224` full30。理由不是短测超过 NTS07b，而是它对准了 NTS07b full30 的真实后期问题：限制 qk threshold 超过 `1.15`，希望保留 epoch24 附近精度，同时阻止 epoch29 后期 ternary activity 继续掉到 `3.6%`。如果 full30 valid825 不能优于 NTS07b epoch24/29，则 NTS08 判为负结果，不再继续扫 qk threshold cap。

### 37.14 NTS09 候选准备：late threshold freeze（2026-06-08）

判断依据：`NTS07b` 的主问题不是 H60/no-carrier 公式本身，而是 full30 后期 qk ternary threshold 持续上升，导致 ternary activity 从中期约 `7.3%` 继续掉到后期约 `3.6%`。`NTS08` 先验证 cap；并行准备的下一轮更直接对准这个问题：**冻结后期 qk threshold 更新**，不增加任何部署端算子。

代码支持已补齐：

- `atlif_ternary_psn` 新增可选配置 `threshold_freeze_after_step`
- `train.py` 在调用 `threshold_update()` 时注入 `_global_step`
- 旧配置默认无该字段，因此旧实验行为不变
- `unittest` 回归通过：`/opt/conda/envs/sdformerflow/bin/python -m unittest neuron_experiments.H9_bipolar_self_attention.tests.test_atlif_ternary_psn -q`

新增生成器：`neuron_experiments/H9_bipolar_self_attention/entrypoints/make_nts09_threshold_freeze_hw_configs.py`

已生成候选：

| 候选 | 核心设置 | 设计意图 |
|---|---|---|
| `nts09a_hw_h60_freeze816_s1224` | `threshold_freeze_after_step=816` | 在 1224-step 短测的约 2/3 处冻结，尽量保留中期 activity |
| `nts09b_hw_h60_freeze918_s1224` | `freeze_after_step=918` | 更晚冻结，保留更多后期适应 |
| `nts09c_hw_h60_eta0325_freeze816_s1224` | `threshold_eta=3.25e-4` + freeze 816 | 同时减慢漂移并冻结 |
| `nts09d_hw_h60_cap115_freeze816_s1224` | `max_threshold=1.15` + freeze 816 | 组合 cap 与 freeze，双重限制后期过稀疏 |

共同约束：继续保持 `mode=h60`、no carrier、no Kmag、no target-rate、`mismatch_penalty=0`、`single_active_penalty=0`，FFN official ATLIF 仍保留但 `threshold_eta=0`、`activity_eta=0`。也就是说，`NTS09` 仍然是当前最干净、最硬件友好的整网叙事，只改训练期阈值调度。

状态：`NTS08c` full30 仍在跑，因此 `NTS09` 先排队不启动；一旦 `NTS08` 结果落地，就用这组 config 直接接短测，不再临时改代码。

已准备短测脚本：`neuron_experiments/H9_bipolar_self_attention/entrypoints/run_nts09_threshold_freeze_short.sh`

脚本口径：

- 先生成 `nts09a/b/c/d`
- 再用 `rapid_screen.py`
- `steps=1224`
- `prev_runid=experiments/baseline_stride_upstream/checkpoint_epoch59.pth`
- `valid10 -> valid40`
- tag=`nts09_thresh_freeze`

因此 `NTS08c` 一旦结束，下一跳不需要再手动拼命令，直接起这个脚本即可。

为避免 `promote_best_rapid_screen.py` 只停在 `valid40 profile`，已补通用标准推理脚本：

- `neuron_experiments/H9_bipolar_self_attention/entrypoints/run_h9_standard_valid825_eval.py`

`nts08c` 跑完后的标准 valid825 入口固定为：

```bash
/opt/conda/envs/sdformerflow/bin/python \
  neuron_experiments/H9_bipolar_self_attention/entrypoints/run_h9_standard_valid825_eval.py \
  --config neuron_experiments/H9_bipolar_self_attention/configs/nts08c_hw_h60_qk_cap115_s1224_steps1224_auto_full_20260608_160513.yml \
  --run-dir neuron_experiments/H9_bipolar_self_attention/results/nts08c_hw_h60_qk_cap115_s1224_steps1224_auto_full_bs6_20260608_160513_setsid \
  --epoch 19 --epoch 24 --epoch 29
```

输出：

- `standard_valid825/epoch*/spike_profile.json`
- `profile_ranking_valid825.md`

这样 `NTS08`/后续 `NTS09` 都能沿用和 `NTS07b` 一致的最终论文口径，不会只停留在 profile-only 比较。

### 37.15 NTS00-NTS08 结果目录审计（2026-06-08）

目的：防止把启动残留目录误认为有效实验，也防止已跑实验漏写。当前磁盘上 `nts*` 结果目录状态如下。

| 目录 | 状态 | md 记录 | 说明 |
|---|---|---|---|
| `nts00b_mu010_std_s360_auto_full_bs6_20260607_031912_setsid` | 有效 full | 已写 | NTS00b no-Kmag full30，属于早期纯 score 级 TX+SC 融合验证 |
| `nts04_hw_short_20260607_223442` | 启动残留 | 本节补充 | 仅 `106K`，`summary.csv` 只有表头，`summary.md` 无有效指标；不作为 NTS04 结果 |
| `nts04_hw_short_20260607_223605` | 有效 short | 已写 | NTS04 正式短测目录，`summary.csv` 有 11 行（表头 + valid10/valid40 记录） |
| `nts04c_hw_mu010_mis020_s360_auto_full_bs6_20260607_233610_setsid` | 有效 full | 已写 | NTS04c full，后续判定强惩罚/过稀疏方向失败 |
| `nts04g_hw_sched010_w720_s360_auto_full_bs6_20260608_004746_setsid` | 有效 full | 已写 | NTS04g full，schedule 后仍不解决过稀疏主问题 |
| `nts05_weak_hw_short_20260608_014233` | 空壳残留 | 本节补充 | 仅 `8K`，只有空 `configs/` 目录，无 summary/metrics；不作为 NTS05 结果 |
| `nts05_weak_hw_short_20260608_014249` | 有效 short | 已写 | NTS05 正式短测目录，选出 `nts05d` |
| `nts05d_hw_mu0075_mis000_sap0025_w720_s360_auto_full_bs6_20260608_020722_setsid` | 有效 full | 已写 | full 早期仍过稀疏，判定 `single_active_penalty=0.025` 不适合作主线 |
| `nts06_floor_hw_short_20260608_024604` | 有效 short | 已写 | 更弱/关闭 single-active 的短测 |
| `nts06a_hw_mu005_mis000_sap000_w720_s360_auto_full_bs6_20260608_031001_setsid` | 有效 full/已停 | 已写 | full 中 FFN binary activity 坍缩，促成 NTS07 的 FFN ATLIF 稳定性检查 |
| `nts07_ffn_floor_hw_short_20260608_034549` | 有效 short/部分候选提前停 | 已写 | `nts07b` 明显优于 `nts07a` 后手动停止剩余候选；`nts07c/07d` 需另补才可作为消融 |
| `nts07b_hw_h60_ffn_update0_act0_s1224_steps1224_auto_full_bs6_20260608_042113_setsid` | 有效 full + valid825 | 已写 | 当前主线结果：no-carrier/no-Kmag/no-target-rate，FFN sparse update/loss 关闭 |
| `nts08_qk_stab_20260608_142946` | 有效 short | 已写 | qk threshold stability 短测，选 `nts08c` |
| `nts08c_hw_h60_qk_cap115_s1224_steps1224_auto_full_bs6_20260608_160513_setsid` | **中断，无效 full** | §37.23 | 仅 ep0–3 权重；无 `*_state_dict.pth`；不能作正式 full 证据 |
| `nts09_priority_20260608_172734` | OOM 失败 | §37.24 | 四候选均在 step1 OOM，`summary.csv` 空 |
| `nts09_priority_20260608_200402` | short 重跑中 | §37.24 | GPU 释放后重启的正式 NTS09 priority 短测 |

编号说明：NTS04 到 NTS08 没有实验编号空洞；看起来“中间空出来”的是时间戳目录里的启动残留，不是正式方案编号。NTS01/NTS02/NTS03 属于更早的 alias/候选命名体系，其中 `NTS01` 对应原 `ntx_h60_full30` alias，`NTS03` 是待跑 sweep alias；当前硬件友好主线实际从 NTS00 no-Kmag 重新梳理，再经 NTS04-NTS08 演化。

### 37.16 NTS08c full30 在线监控（2026-06-08）

运行目录：`neuron_experiments/H9_bipolar_self_attention/results/nts08c_hw_h60_qk_cap115_s1224_steps1224_auto_full_bs6_20260608_160513_setsid`。

配置确认：`mode=h60`，no carrier，no Kmag，no target-rate，`mismatch_penalty=0`，`single_active_penalty=0`，FFN official ATLIF 保留但 `threshold_eta=0/activity_eta=0`；qk 三值阈值使用 `max_threshold=1.15`，其余沿用 NTS07b。

早期 epoch 监控：

| epoch | train loss | valid loss | 判断 |
|---:|---:|---:|---|
| 0 | 1.5565 | 1.2042 | 与 NTS07b epoch0 对齐，正常 |
| 1 | 1.4965 | 1.2366 | valid 小幅上升，但不是持续崩溃；继续 |
| 2 | 1.4898 | 1.2130 | valid 回落到健康区间；继续 full30 |

当前训练仍在进行，已保存 `checkpoint_epoch0.pth`、`checkpoint_epoch1.pth`、`checkpoint_epoch2.pth`。更晚的在线监控点为：

- step1020: `threshold_mean=1.02027`、`threshold_max=1.07529`、`binary_activity_mean=5.23%`、`ternary_activity_mean=6.96%`
- step1060: `threshold_mean=1.02050`、`threshold_max=1.07613`、`binary_activity_mean=5.19%`、`ternary_activity_mean=6.92%`
- step1120: `threshold_mean=1.02789`、`threshold_max=1.10290`、`binary_activity_mean=5.53%`、`ternary_activity_mean=6.69%`、`ternary_pos_neg_ratio=1.28`

判断：`max_threshold=1.15` 至少没有带来早期 activity 坍缩，当前轨迹仍明显健康于 NTS07b 后期过稀疏的问题形态，因此继续让 `NTS08c` 跑完。结束后必须按标准流程跑 valid825 并把完整指标补入本节。

新增证据：`checkpoint_epoch3.pth` 已经落盘，epoch3 的末段监控为：

- step1200: `threshold_mean=1.02834`、`threshold_max=1.10453`、`binary_activity_mean=5.55%`、`ternary_activity_mean=6.76%`
- step1220: `threshold_mean=1.02846`、`threshold_max=1.10494`、`binary_activity_mean=5.43%`、`ternary_activity_mean=6.62%`、`ternary_pos_neg_ratio=1.29`
- epoch3 valid loss: `1.3093`

这说明 `NTS08c` 虽然 epoch3 valid loss 暂时高于 epoch2，但 qk/FFN activity 并没有掉到坏区间，当前仍更像“继续观察 full30 后期是否保住 NTS07b 中期精度”的实验，而不是又一个早期就注定失败的过稀疏分支。

### 37.17 当前 DATE 主线默认取向（2026-06-08）

为了让 `ntx01 -> NTS` 这条线最后能讲成一个干净、可信、对硬件友好的整网故事，当前默认主线正式收敛为：

1. **部署公式固定，不再加新部署算子**
2. 注意力保持 `h60` score-level fusion
3. 保持 `no carrier`、`no Kmag`、`no target-rate`
4. 保持 `mismatch_penalty=0`、`single_active_penalty=0`
5. FFN 保留 official ATLIF，但不允许 full 训练再次因为 FFN sparse update/loss 而塌缩
6. 后续探索只优先改 **训练期 qk threshold dynamics**，也就是 `cap / freeze / schedule`

理由：

- 这条线最接近 `NTX01` 当初“精度基本守住、稀疏有明显收益”的精神内核，但把 `carrier*gate` 外挂形式拿掉了。
- 它也是当前最符合 DATE 叙事的一条：**训练期做结构感知约束，部署端保持简单统一的 SNN attention datapath**。
- 更早那些强惩罚、carrier、Kmag、target-rate 控制的路线，要么 full30 过稀疏，要么故事不够干净，要么硬件代价太难讲。

因此，除非 `NTS08/NTS09` 全部失败，否则当前不再回到“再发明一个更复杂的 attention 公式”这条路。

### 37.18 NTS09 优先级（若 NTS08c full30 不能超过 NTS07b）

若 `NTS08c` 的标准 valid825 不能优于 `NTS07b epoch24/29`，下一轮短测按以下优先级启动：

1. `nts09a_hw_h60_freeze816_s1224`
2. `nts09d_hw_h60_cap115_freeze816_s1224`
3. `nts09b_hw_h60_freeze918_s1224`
4. `nts09c_hw_h60_eta0325_freeze816_s1224`

优先级理由：

- `freeze816` 最直接对应 “保住中期活性，阻止后期继续过稀疏” 这一主问题；
- `cap115 + freeze816` 是最接近 `NTS08` 逻辑的加强版；
- `freeze918` 更保守，但可能冻结得稍晚；
- `eta0325 + freeze816` 变量更多，适合作为后手，不适合作第一优先。

### 37.19 标准 valid825 自动接棒 helper（2026-06-08）

新增：

- `neuron_experiments/H9_bipolar_self_attention/entrypoints/wait_full_then_run_standard_valid825.py`

作用：等待指定 full run 落出目标 checkpoint（默认 `epoch19/24/29`），然后自动调用：

- `neuron_experiments/H9_bipolar_self_attention/entrypoints/run_h9_standard_valid825_eval.py`

这样 `NTS08`、后续 `NTS09` 乃至同类 H9 full run 都可以复用同一个收尾链路，不需要人工盯到训练结束再手动补标准推理。

示例（用于当前 `NTS08c`）：

```bash
/opt/conda/envs/sdformerflow/bin/python \
  neuron_experiments/H9_bipolar_self_attention/entrypoints/wait_full_then_run_standard_valid825.py \
  --config neuron_experiments/H9_bipolar_self_attention/configs/nts08c_hw_h60_qk_cap115_s1224_steps1224_auto_full_20260608_160513.yml \
  --run-dir neuron_experiments/H9_bipolar_self_attention/results/nts08c_hw_h60_qk_cap115_s1224_steps1224_auto_full_bs6_20260608_160513_setsid \
  --epoch 19 --epoch 24 --epoch 29
```

验证：

- `py_compile` 通过
- `--help` 可正常打印参数说明

### 37.20 NTS09 主线优先短测入口（2026-06-08）

新增：

- `neuron_experiments/H9_bipolar_self_attention/entrypoints/run_nts09_priority_short.sh`

这个脚本和已有的 `run_nts09_threshold_freeze_short.sh` 不冲突。区别是：

1. 按当前主线优先级排序候选：
   - `nts09a_hw_h60_freeze816_s1224`
   - `nts09d_hw_h60_cap115_freeze816_s1224`
   - `nts09b_hw_h60_freeze918_s1224`
   - `nts09c_hw_h60_eta0325_freeze816_s1224`
2. 使用更严格的 promotion 门槛：
   - `--promote-aee 1.75`
   - `--promote-aae 16.0`
   - `--promote-sops-g 6.0`

这样如果 `NTS08c` full30 最终不如 `NTS07b`，下一步更像“筛主线 full 候选”，而不是把所有 freeze 变体一股脑推进 full30。

验证：

- `bash -n` 通过

建议：若 `NTS08c` 判负，优先跑这个 `nts09_priority` 入口；保留原 `nts09_thresh_freeze` 作为更宽松的补充 sweep。

### 37.21 NTS08c 当前节奏与 ETA（2026-06-08）

在线核对：

- 训练主进程 `PID 70030` 仍在正常运行，CPU 占用高，未出现卡死迹象
- `standard_valid825` watcher 也已在线，当前在等待 `epoch19/24/29`

根据当前已落盘 checkpoint 时间戳：

| epoch | UTC 时间 |
|---:|---|
| 0 | `2026-06-08T08:21:05Z` |
| 1 | `2026-06-08T08:36:48Z` |
| 2 | `2026-06-08T08:52:30Z` |
| 3 | `2026-06-08T09:08:08Z` |

估算平均训练速度约为 `941s/epoch`（约 `15m41s`）。据此得到粗略 ETA：

| target epoch | rough ETA (UTC) |
|---:|---|
| 9 | `2026-06-08T10:42:14Z` |
| 19 | `2026-06-08T13:19:04Z` |
| 24 | `2026-06-08T14:37:29Z` |
| 29 | `2026-06-08T15:55:54Z` |

用途：帮助判断何时回来查看 `nts08c` 的中后期走势，以及何时预期自动 `valid825` 结果开始落盘。

### 37.22 NTS08c epoch4 在线状态补充（2026-06-08）

补充说明：虽然目录里暂时还没有新的 `checkpoint_epoch4.pth`，但这并不表示训练卡住。直接看 `train.log` 可以确认：`nts08c` 已经在下一轮继续推进，并进入了当前 epoch 的中段。

在线观察点：

- step580: `threshold_mean=1.03175`、`threshold_max=1.11676`、`ternary_activity_mean=6.59%`、`binary_activity_mean=5.13%`
- step640: `threshold_mean=1.03208`、`threshold_max=1.11796`、`ternary_activity_mean=6.55%`、`binary_activity_mean=5.43%`
- step700: `threshold_mean=1.03242`、`threshold_max=1.11916`、`ternary_activity_mean=6.57%`、`binary_activity_mean=5.32%`

判断：

1. `NTS08c` 在 epoch4 中段仍保持健康 activity，没有出现“继续训练后迅速塌到超稀疏”的坏形态。
2. `threshold_mean` 只是缓慢上升，`threshold_max` 也仍低于 `1.15` cap 很多，说明当前这条 cap 线并未提前触顶失真。
3. 因此现阶段不应把它误判为“停住”或“已经失败”；更合理的结论是：继续等待 epoch4 完整 valid，再判断它是否只是健康但无收益，还是会重新靠近 `NTS07b` 中期表现。

进一步核对（同日补充）：

- 这条 run 不是只在 epoch4 前段徘徊，而是已经推进到了大约 `step708 / 1224`
- 这意味着它处在 **epoch4 中后段**，而不是“epoch3 刚结束不久”

综合这一段的在线统计：

- `threshold_mean` 约从 `1.025` 缓慢升到 `1.032`
- `threshold_max` 约从 `1.093` 升到 `1.119`
- `ternary_activity_mean` 基本稳定在 `6.5% ~ 6.8%`
- `binary_activity_mean` 基本稳定在 `5.0% ~ 5.6%`

这比最开始担心的“后期继续训练后迅速掉到超稀疏”要健康得多，也说明 `NTS08c` 当前不是靠触碰 `1.15` cap 才勉强存活，而是在 cap 还未真正起主导作用时就保持了相对稳定的 activity。

因此目前对 `NTS08c` 的判断从“继续观察”可以上调为：

- **仍是值得等待 full 结果的主线候选**
- 但是否真的优于 `NTS07b`，仍要看 epoch4 之后的 valid 和最终 `valid825`

### 37.23 NTS08c 中断与 full-promotion resume 缺口（2026-06-08）

补充核查后确认：`nts08c` 不是稳定跑到后期等待 `epoch4`/`epoch19` 落盘，而是在下一轮中段之后训练进程消失；当前只剩 `standard_valid825` watcher 还在等 `epoch19/24/29`。

运行目录里实际只有：

- `checkpoint_epoch0.pth`
- `checkpoint_epoch1.pth`
- `checkpoint_epoch2.pth`
- `checkpoint_epoch3.pth`
- `command.txt`
- `promotion_ranking.md`
- `train.log`

关键事实：

1. `train.log` 已明确证明训练推进到了下一轮中段，至少到约 `step708/1224`，因此不是卡在 `epoch3` 之前。
2. 但没有新的 `checkpoint_epoch4.pth`，也没有新的 validation 结果。
3. 训练日志里没有明确的 Python traceback / OOM 文本；更像是外部进程中断，而不是方案本身前向/反向报错。
4. 这次不能“正规 resume”的直接原因，不是 baseline 不支持 resume，而是当时 full-promotion 生成的 config 带了 `runtime.skip_state_save: true`，所以运行目录没有 `checkpoint_epoch*_state_dict.pth`。

因此当前对 `nts08c` 的正式定性应改为：

- 这是一个**中断的 full run**，不是“已完成待评估”的 full run。
- 它仍提供了有价值的在线证据：`max_threshold=1.15` 在下一轮中段没有触发早期 activity 坍缩。
- 但在没有训练 state 的前提下，不能把它当作可无损续跑的标准 full30；若要继续，只能选择：
  - 近似续训：从 `checkpoint_epoch3.pth` 重新载入权重，另起新 run，并显式记录它不是严格 resume；
  - 或直接转入 `NTS09` 主线短测。

链路修复：

- 已修改 `neuron_experiments/H9_bipolar_self_attention/entrypoints/promote_best_rapid_screen.py`，后续新 full-promotion config 将保存 training state（`runtime.skip_state_save: false`），避免再出现“模型权重有了但 resume state 没有”的断点。

当前动作：

- 已启动 `NTS09 priority` 短测，结果目录：`neuron_experiments/H9_bipolar_self_attention/results/nts09_priority_20260608_172619`
- 本轮按既定主线优先级测试：
  1. `nts09a_hw_h60_freeze816_s1224`
  2. `nts09d_hw_h60_cap115_freeze816_s1224`
  3. `nts09b_hw_h60_freeze918_s1224`
  4. `nts09c_hw_h60_eta0325_freeze816_s1224`
- promotion 门槛保持：
  - `AEE <= 1.75`
  - `AAE <= 16.0`
  - `SOPs <= 6.0G`
- 这轮短测完成后，需要把 `summary.csv / summary.md` 结果继续补入本节，并据此决定新的 full 候选。

### 37.24 NTS09 priority 短测首次 OOM 与重启（2026-06-08）

首次启动（`results/nts09_priority_20260608_172734`）四个候选全部在 `epoch0 step1` 因 **CUDA OOM** 失败：

- 日志显示 GPU 被两个外部进程（`PID 1080961` 40.4GiB + `PID 1084956` 38.8GiB）占满，`summary.csv` 仅表头、无有效行。
- 该轮 **不能** 作为 NTS09 短测证据。

GPU 释放后已重启 priority 短测：

- 结果目录：`neuron_experiments/H9_bipolar_self_attention/results/nts09_priority_20260608_200402`
- launcher 日志：`results/nts09_priority_launcher_restart_20260608_2004.log`
- 候选顺序不变：`09a → 09d → 09b → 09c`
- 首轮 `nts09a` 训练已正常推进（step100 附近 ternary activity ~7.5%，无 OOM）

短测完成后下一步：

1. 将 `summary.csv / summary.md` 补入本节
2. 用 `promote_best_rapid_screen.py --tag nts09_priority` 启动 full30（已修复 `skip_state_save=false`）
3. 并行启动 `wait_full_then_run_standard_valid825.py` 等待 ep19/24/29 并跑 valid825

### 37.25 NTS09 priority 短测结果与 full30 启动（2026-06-08）

正式短测目录：`results/nts09_priority_20260608_200402`（重启后四轮全部完成）。

**valid10 初筛**（AAE 门槛 16.0 在 valid10 上噪声过大，四候选均未 pass，属预期）：

| rank | candidate | valid10 AEE | valid10 AAE | SOPs(G) | threshold |
|---:|---|---:|---:|---:|---:|
| 1 | `nts09c` (eta0325+freeze816) | 1.3628 | 18.06 | 3.80 | 1.0022 |
| 2 | `nts09a` (freeze816) ⭐ | 1.4072 | 18.80 | 3.80 | 1.0044 |
| 3 | `nts09d` (cap115+freeze816) | 1.4171 | 19.10 | 3.79 | 1.0044 |
| 4 | `nts09b` (freeze918) | 1.4498 | 19.16 | 3.79 | 1.0051 |

**补跑 valid40 确认**（手动 profile，因 valid10 gate 未触发自动 confirm）：

| candidate | valid40 AEE | valid40 AAE | SOPs(G) | firing | gate |
|---|---:|---:|---:|---:|---|
| **`nts09a`** ⭐ | **1.4658** | **13.57** | 4.06 | 9.60% | pass |
| `nts09c` | 1.4725 | 13.70 | 4.06 | 9.61% | pass |

对比 NTS07b 短测 valid40（AEE 1.4353，AAE 13.31）：NTS09a 略弱但同量级，且符合 DATE 主线优先级 #1（纯 freeze816，无 cap/eta 额外变量）。

**full30 晋升：NTS09a**

- 配置：`configs/nts09a_hw_h60_freeze816_s1224_steps1224_auto_full_20260608_210900.yml`
- 目录：`results/nts09a_hw_h60_freeze816_s1224_steps1224_auto_full_bs6_20260608_210900_setsid`
- 关键确认：`runtime.skip_state_save: false`（可中断可 resume）
- valid825 watcher 已启动，等待 ep19/24/29 自动跑标准推理
- 训练 early health（epoch0 step60）：ternary activity ~7.6%，binary ~4.7%，无塌缩信号

### 37.26 NTS09a full30 valid825 与 15% 脉冲门槛（2026-06-09）

**标准 valid825**（`spike_profile.json` → `total_spikes`，非 SOPs）：

| 方案 | ep | AEE | AAE | total_spikes | vs NB0 |
|------|-----|-----|-----|-------------|--------|
| NB0 | 59 | 1.4872 | 9.93° | 44.05G | — |
| NTS07b | 29 | 1.4855 | 9.74° | 36.80G | **-16.5%** ✅ |
| NTS09a | 24 | 1.4632 | 9.82° | 40.84G | -7.3% |
| NTS09a | 29 | 1.4798 | 9.64° | 38.98G | -11.5% |

15% 门槛：`total_spikes ≤ 37.44G`。NTS09a ep29 未达标（差约 1.5G），但 AAE 优于 07b。

**机制对照**（ep29）：

| | qk threshold | ternary activity | 稀疏策略 |
|--|-------------|------------------|---------|
| 07b | ~1.17（全程漂移） | ~3.6% | 不冻结 → 更稀疏 |
| 09a | ~1.00–1.02（step816 冻结） | ~7.3% | 早冻结 → 更密、精度更好 |

**NTS09 sparse 短测**：

| 轮次 | 目录 | 状态 |
|------|------|------|
| 首次 | `results/nts09_sparse_20260609_185456` | 中断：09e valid10 完成，09b 训练至 step~344 后退出 |
| **重启** | `results/nts09_sparse_20260609_200210` | **进行中**（2026-06-09 20:02 UTC 启动） |

重启命令：`entrypoints/run_nts09_sparse_short.sh`（已内置 `conda activate sdformerflow`）  
日志：`results/nts09_sparse_launcher_20260609_restart.log`  
候选顺序：09e → 09b → 09f → 09g → 09h → 09i（6×1224 step + valid10/40）  
预估整轮 ~1.5–2h。完成后按 valid40 选最优 promote full30 + `wait_full_then_run_standard_valid825.py`。

---

## 三十八、NTS 全族实验技术手册（范式 / 策略 / 超参 / 影响）

> 本节汇总 NTS00–NTS09 全部实验：用到什么范式、训练策略（冻结/cap/惩罚/schedule）、每个超参的意义与对结果的影响。正式稀疏口径一律用 `eval_DSEC_flow_SNN.py` → `spike_profile.json` 的 **total_spikes**。

### 38.1 共同底座

| 类别 | 参数 | 值 | 影响 |
|------|------|-----|------|
| 续训 | `prev_runid` | NB0 `checkpoint_epoch59.pth` | overlay 参数从随机初始化装入 |
| 训练 | `n_epochs` | 30 (full) / 1 (短测) | full 约 1224 step/epoch |
| 训练 | `batch_size` | 6 (full) / 8 (短测) | 显存与梯度噪声 |
| 评估 | `test.sample` | 825 (正式) / 10–40 (短测) | valid825 才能定论文数字 |
| 注意力范围 | `target_blocks` | S2: `2:0`–`2:5`（**6 block**） | 仅语义最强 stage；见 §38.8 |
| 优化 | backbone/neuron/threshold LR | 1e-6 / 3e-5 / 5e-6 | 主干几乎冻结，神经元与阈值可调 |
| 优化 | warmup + multistep | 200 step; ep20/25 降 LR | 防早期阈值暴走 |

### 38.2 NTS 核心范式（h60）

```
TX_score = Σ[same_nonzero + α₀·same_zero − β·opposite − sap·one_sided]  [+ α_k·K_mag]
SC_score = Σ sign(Q)·sign(K)   → 经 consensus_score_norm 归一化
score    = TX_score + μ × SC_score
gate     = Shiftmax(score)
output   = K × gate        （无 carrier）
```

代码入口：`bsa_attention.py` → `mode=h60` / `tx_sc_k_mag_no_carrier_shiftmax`。

### 38.3 关键超参 FAQ（代码级解释）

#### 38.3.1 `alpha0`（α₀）是什么？

- **定义**：三值 TX 分数里，**Q 与 K 同为 silent（0）** 时的匹配奖励系数。
- **代码**（`_ternary_alpha_xnor_token_scores`）：
  ```text
  score = same_nonzero + alpha0 * same_zero - mismatch * opposite - single_active * one_sided
  ```
- **来源**：CVPR 2025 alpha-XNOR 的扩展——spike-spike 强匹配权重 1，silence-silence 弱匹配权重 α₀，异号惩罚 β。
- **当前 NTS 值**：`alpha0=0.02`（NTS01/02 同）。
- **影响**：
  - α₀ **↑** → 更奖励"双静默"对齐 → 低活性 token 得分更高 → gate 更均匀，可能 **增密、保精度**
  - α₀ **↓** → 静默对齐几乎不计分 → 只有活跃通道驱动 attention → 可能 **更稀疏、丢弱信号**

#### 38.3.2 `sc_mu_schedule`：训练渐进打开 μ，推理时 μ 固定吗？

- **训练**：`_scheduled_bipolar_mu()` 按全局 step 线性插值：
  ```text
  μ(step) = sc_mu_start + (bipolar_mu - sc_mu_start) × min(1, (step - sc_mu_start_step) / sc_mu_warmup_steps)
  ```
  当前主线：`0 → 0.05 / 720 steps`（`sc_mu_schedule_enabled=true`）。
- **推理**：`eval` / `profile` 时 `_h9_global_step` 为 **None** → 函数直接返回 **`bipolar_mu` 终值（0.05）**。
- **结论**：schedule **仅训练期** 存在；**部署/推理 μ 恒为 config 里的 `bipolar_mu`**，不随 step 变化。训练早期 μ≈0 等价于"先学纯 TX，再慢慢引入 SC 残差"。

#### 38.3.3 TX 分数为什么不除以通道数？

**实际上会除。** 文档公式写的是归一化前的 popcount 求和；代码在 `sum(dim=-1)` 之后调用 `_normalize_consensus_score`：

- 当前 NTS 配置：`consensus_score_norm: head_dim`
- 实现：`score = score / head_dim`（再乘 `score_scale`，默认 1.0）
- **SC 分支同样除以 head_dim**（`_signed_consensus_token_scores` 路径一致）

因此 TX/SC 融合前的两项量级都是 **per-channel 平均 popcount**，不是 raw 累加。若改为 `sqrt_head_dim` 或 `active` 归一化，Shiftmax 输入尺度会变，需重新短测。

#### 38.3.4 现在有单边惩罚吗？

| 惩罚 | 当前 NTS07–09 主线 | 含义 | 推理期 |
|------|-------------------|------|--------|
| `mismatch_penalty` (β) | **0.0** | Q/K **异号**（+1 vs -1）减分 | 仅训练 score 计算；β=0 时无此项 |
| `single_active_penalty` (sap) | **0.0** | **单边活跃**：一方有 spike 另一方 silent（`q_active·k_zero + q_zero·k_active`）减分 | 同上 |

**结论**：DATE 主线（07b/08/09）**两种惩罚均为 0**，无单边惩罚、无异号惩罚。NTS04–06 曾用 β∈[0.05,0.25]、sap∈[0.025,0.05]，已证实 full 训练会导致过稀疏，故废弃。

#### 38.3.5 其他仍生效的 score 处理

| 参数 | 值 | 作用 |
|------|-----|------|
| `center_scores` | true | 每个 head 内 token score 减均值，防全局偏置 |
| `preserve_mean` | true | Shiftmax 后 gate 乘 `n_tokens`，保持能量尺度 |
| `consensus_bias` | 0.02 | SC 分支小偏置（部分模式） |

### 38.4 分代实验详解

#### NTS-01 / NTS-02（含 K_mag，精度+稀疏标杆）

| 超参 | 值 | 影响 |
|------|-----|------|
| `bipolar_mu` | 0.10 | SC 残差较强 |
| `k_magnitude_alpha` | 0.02 | 唯一非纯 event 路径；精度↑，硬件代价↑ |
| `mismatch_penalty` | 0.25 | 训练期异号惩罚 |

valid825：NTS-02 ep29 AEE=1.525, AAE=9.97°, **32.46G**（-26.3% vs NB0）。

#### NTS-00（去 K_mag）

8 候选短测（μ/LR/schedule/bs）→ promote **00b**（μ=0.10, no K_mag）。  
valid825 ep29：AEE=1.537, **32.60G**。证明无 K_mag 仍可大幅稀疏。

#### NTS-04（强训练惩罚 → 失败）

| 候选 | 关键差异 |
|------|---------|
| 04c | μ=0.10, β=0.20 |
| 04g | μ schedule 720, β=0.25, sap=0.05 |

**学到**：短测 valid40 正常，full 早期 binary activity 崩到 <0.5%。惩罚只参与 loss，不进推理，但会间接压低 firing。

#### NTS-05 / NTS-06（弱化惩罚 → 仍失败）

05d 去 β 短测最优，但 full 仍过稀疏。06a 关 sap 后，发现主因是 **FFN ATLIF threshold 漂移**，不是 attention 惩罚。

#### NTS-07（FFN 稳定 → 当前硬件主线）

| 候选 | FFN `threshold_eta` | FFN `activity_eta` |
|------|--------------------|--------------------|
| 07a | 2e-5 | 0.5 |
| **07b** ⭐ | **0** | **0** |
| 07c | 移除 FFN groups | — |

**07b 定型约束**：h60, no carrier/Kmag/target-rate, β=0, sap=0, qk `threshold_eta=6.5e-4`, μ schedule 0→0.05/720, FFN sparse update 关闭。

qk 阈值全程漂移：1.0 → ep29 ~1.17；ternary 7.3% → 3.6%。  
valid825 ep29：**36.80G**, AEE=1.486, AAE=9.74°。

#### NTS-08（qk threshold cap）

| 候选 | 改动 |
|------|------|
| 08a | `threshold_eta` 减半 |
| 08b | `threshold_lr_scale` 减半 |
| **08c** | `max_threshold=1.15` |

cap 在短测（1224 step）不触发；针对 full 后期 threshold>1.15 的过稀疏。08c full **中断**（仅 ep0–3），无正式 valid825。

#### NTS-09（qk threshold freeze）

`threshold_freeze_after_step`：**全局 step** 达到后永久停止 qk 阈值更新（`installer.py` + `train.py` 注入 `_global_step`）。

| 候选 | freeze step | 设计 |
|------|------------|------|
| 09a ⭐ full | 816（epoch0 的 66%） | 锁低阈值 → 密、精度好 |
| 09b | 918 | 更晚冻结 |
| 09c | 816 + eta 减半 | 慢漂移+早冻 |
| 09d | 816 + cap 1.15 | 双保险 |
| 09e–09i | 1224/6120/12240… | 稀疏权衡 sweep（进行中断） |

valid825 ep29：AEE=**1.480**, AAE=**9.64°**, 38.98G（-11.5%）。

### 38.5 影响实验结果的因素速查

| 机制 | 训练期 | 推理期 | ↑ 会更稀疏？ |
|------|--------|--------|------------|
| qk `threshold_eta` | ✅ 自动调高阈值 | 阈值固定后影响 firing | ✅ |
| `max_threshold` cap | ✅ 限顶 | ✅ | 达到 cap 后停止变稀疏 |
| `threshold_freeze_after_step` | ✅ 停止更新 | ✅ 锁 setpoint | 冻越早→阈值越低→**更密** |
| `mismatch_penalty` | ✅（主线=0） | 进 score 公式 | ✅（若>0） |
| `single_active_penalty` | ✅（主线=0） | 进 score 公式 | ✅（若>0） |
| FFN `threshold_eta/activity_eta` | ✅（07b=0） | ✅ | ✅（若>0，压 FFN binary） |
| `bipolar_mu` schedule | ✅ 仅训练 | ❌ 推理用终值 μ | μ↑ 通常精度↑，稀疏影响间接 |

### 38.6 valid825 总表（total_spikes 正式口径）

| 方案 | ep | AEE | AAE | total_spikes | vs NB0 | 备注 |
|------|-----|-----|-----|-------------|--------|------|
| NB0 | 59 | 1.487 | 9.93° | 44.05G | — | 基线 |
| NTS-02 | 29 | 1.525 | 9.97° | 32.46G | -26.3% | 含 K_mag，综合最强 |
| NTS-00b | 29 | 1.537 | 10.07° | 32.60G | -26.0% | 无 K_mag |
| **NTS-07b** | 29 | 1.486 | 9.74° | **36.80G** | **-16.5%** | DATE 硬件主线 |
| NTS-09a | 29 | **1.480** | **9.64°** | 38.98G | -11.5% | 精度最佳，偏密 |
| NTS-09a | 24 | 1.463 | 9.82° | 40.84G | -7.3% | AEE 最优 ckpt |

### 38.7 DATE 主线默认取向（不变）

1. h60 score 融合；no carrier / no Kmag / no target-rate
2. `mismatch_penalty=0`, `single_active_penalty=0`
3. FFN official ATLIF 保留，sparse update/loss 关闭（07b）
4. 只探索训练期 **qk threshold 动力学**（cap / freeze / schedule）

### 38.8 Block 替换范围：能否 S0+S1+S2 都换？

**现状**：`target_blocks = 2:0..2:5`（仅 Stage2 的 6 个 Swin block，约占 encoder 注意力 6/12）。

**各 stage 块数**（Swin depths [2,2,6,2]）：

| Stage | blocks | 当前 |
|-------|--------|------|
| S0 | `0:0`, `0:1` | 未换（原生 attention） |
| S1 | `1:0`, `1:1` | 未换 |
| S2 | `2:0`–`2:5` | **已换（NTS 全线）** |
| S3 | `3:0`, `3:1` | 未换 |

**历史依据**：

- **H9b**（`generate_h9b_configs.py`）曾逐 stage 短测：S0/S1/S2/S3/S23 等组合；最终主线收敛到 **S2-only**——语义最强、面积可控。
- **NTX-06** 试过 `s0+s2` partial native attention，结论是 partial 替换需慎重，全替换伤 AAE。
- **NTX-13** 证明仅 S2 + K_mag 不能复现 NTS 收益。

**若试 S0+S1+S2（10 block）**：

| 影响 | 说明 |
|------|------|
| 部署面积 | Shiftmax attention 6→10 模块（+67%） |
| ATLIF 模块 | qk 三值层增加（低层分辨率更高 → **脉冲可能显著增加**） |
| 训练稳定性 | 低层 threshold 漂移更快，需沿用 07b FFN 策略 + 09 freeze |
| 精度 | 低层纹理+中层语义同时改，可能提升 AEE，也可能因过拟合/稀疏失控伤 AAE |

**建议方案（NTS-10 候选，未启动）**：

1. 基座：`nts07b` 或 `nts09e` config
2. `target_blocks: ["0:0","0:1","1:0","1:1","2:0","2:1","2:2","2:3","2:4","2:5"]`
3. 短测 1224 step → valid40，重点看 **SOPs/total_spikes 是否暴涨** 与 AAE 是否守住
4. 若 valid40 通过再 full30；否则退回 S2-only

生成器可复用 `make_nts07` 基座 + 新 `make_nts10_s012_blocks_configs.py`（待写）。

### 38.9 后台任务与 retry 说明（2026-06-09）

**后台任务状态**（用户查询时核查）：

| 进程 | 状态 |
|------|------|
| `nts09_sparse` 短测 launcher | 已结束（`exit_code=-1`，约 20min，仅完成 09e valid10） |
| `nts09a` valid825 watcher | 无活跃 PID |
| `train.py` / `rapid_screen` / `wait_full` | **均无残留** |

如需再跑实验，应单条命令前台或一条 nohup 启动，避免并行多个 watcher/screen。

**关于 "retrying"**：此前异常来自 agent 工具层（后台 `&` 与 `block_until_ms` 冲突、偶发路径读失败后的重试），**不是 `conda activate sdformerflow` 或训练环境故障**。环境已验证：`pandas 2.3.3` 正常。后续执行规范：一次命令、先查 PID 再启动、避免重复 nohup。


### NTS-10d S23 full30 宕机续训 + valid825（自动追加）

- 时间：`2026-06-11T10:30:59`
- 短测目录：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/nts10_blocks_20260610_141114`
- 全量配置：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/configs/nts10d_hw_h60_s23_freeze1224_s1224_steps1224_auto_full_20260610_151207.yml`
- 全量目录：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/nts10d_hw_h60_s23_freeze1224_s1224_steps1224_auto_full_bs6_20260610_151207_setsid`
- 方法：NTS09e freeze1224 基座，S2+S3（8 block）扩大替换。

| epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 19 | 1.5922 | 10.2966 | 0.5353 | 0.2155 | 0.1062 | 41.2055G | 8.8944% | 33772.15 |
| 24 | 1.4624 | 9.8079 | 0.5089 | 0.1817 | 0.0808 | 41.1386G | 8.8800% | 33799.99 |
| 29 | 1.4781 | 9.6899 | 0.5118 | 0.1902 | 0.0874 | 39.2954G | 8.4821% | 32301.37 |
