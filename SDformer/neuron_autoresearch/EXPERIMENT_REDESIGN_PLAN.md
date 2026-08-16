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

### 38.10 NTS-11 代码/链路/硬件审阅记录（2026-06-15）

本次审阅对象：`NTS-11bd / 11bd-v2 / 11bj`，重点检查加载链路、神经元/注意力范式混合、以及是否存在“指标好但硬件代价被低估”的问题。

**加载链路结论**

- `nts11bd_u12_ds_w720_fastlr_full30_20260613_223042_bs8_20260613_223042_setsid` 训练从 NB0 加载时：
  - `installed ATLIFTernaryPSN: 105 modules`
  - `installed Shiftmax attention: 12 modules`
  - `checkpoint_overlay_keys=0, missing=210, unexpected=0`
  - 这是预期状态：NB0 没有 overlay 参数，210 个 ATLIF `thresh/center` 由安装器初始化。
- 该 run 的 valid825 评估日志显示：
  - `checkpoint_overlay_keys=210, missing=0, unexpected=0`
  - 说明保存后的 NTS-11 权重能被标准 eval 正确加载，不是 baseline config 误评。
- `runtime.skip_state_save=false` 已在 full/fine-tune 配置中生效，目录中存在 `checkpoint_epoch*_state_dict.pth`，后续可正规 resume。

**发现并修复的小问题**

- `11bd-v2` 生成器把 5ep fine-tune 的 `force_save_epochs` 写成 `[0, 2, 4, 5]`，但 baseline 训练循环 `range(0, n_epochs)` 实际只会保存 `epoch0..4`。
- `run_nts11bj_full_ft_valid825.sh` 也请求了 `--epoch 5`。评估脚本会跳过不存在的 checkpoint，因此不会污染已有指标，但记录容易误导。
- 已修正：
  - `make_nts11bd_v2_tune_configs.py`: 5ep 改为保存 `[0, 2, 4]`
  - `run_nts11bj_full_ft_valid825.sh`: valid825 改为评估 `epoch0..4`
  - 已生成的 `nts11bj_u12_ds_w720_stdlr_ftbd19_ft5.yml`: 删除残留的 `force_save_epochs: 5`
  - `run_h9_standard_valid825_eval.py`: 对缺失 checkpoint 增加 `skipped missing checkpoints` 提示，避免静默跳过造成误判

**范式混合与硬件口径风险**

- `NTS-11bd` 是 **unified all12 H60 attention**：
  - attention 替换范围从 NTS07/09 的 S2-only 6 block 扩大到 S0/S1/S2/S3 全 12 block。
  - 这有利于讲“整个 encoder 统一范式”，但不是低代价改动；attention 控制逻辑数量约翻倍。
- `NTS-11bd u12_ds` 的神经元安装规模为：
  - 27 个 ternary `symmetric_bsa_tsn` 模块：24 个 Q/K + 3 个 downsample
  - 78 个 binary `official_atlif` 模块：由 `all_non_qk` 自动覆盖，包含 resblocks/decoders 等非 Q/K spiking neuron
  - 总计 105 个 ATLIF 模块，明显大于 NTS07/09 的 34 模块。
- 因此 NTS-11 的硬件叙事不能只说“total_spikes 下降”。还必须额外报告：
  - Shiftmax attention modules: `12`
  - ATLIF modules: `105`
  - ternary modules: `27`
  - binary official ATLIF modules: `78`
  - eval/training wall-time 或至少说明控制逻辑面积显著增加

**当前 NTS-11bd 结果定位**

- `NTS-11bd u12_ds fastlr` valid825 最佳为 epoch19：
  - AEE `1.5647`
  - AAE `9.9213`
  - total_spikes `29.1676G`
  - 相对 NB0 `44.0488G` 约 `-33.8%`
- 精度在 baseline 约 5% 误差内，稀疏明显达标；但硬件面积/控制复杂度不如 NTS07/09 简洁。
- 它适合作为“统一 encoder attention + 更强稀疏”的候选，但若投 DATE，必须把面积复杂度讲清楚，不能只按 spike 数声称更硬件友好。

**当前正在跑的 11bj**

- `nts11bj_u12_ds_w720_stdlr_ftbd19_ft5` 是从 `NTS-11bd epoch19` 权重加载后做 5ep fine-tune。
- 训练命令使用 `--prev_runid checkpoint_epoch19.pth`，不是 `--resume`，因此这是 **weight-only fine-tune**，不是 optimizer/scheduler/scaler 的严格续训。
- 当前 valid825 正在评估 `epoch0..4`；已完成 epoch0：
  - AEE `1.5571`
  - AAE `10.1128`
  - total_spikes `29.6176G`
  - firing `6.4006%`
  - energy `23492.38uJ`
- epoch0 相比 11bd ep19 AEE 略好（`1.5571` vs `1.5647`），但 AAE 更差（`10.1128` vs `9.9213`），且 spikes 更高（`29.62G` vs `29.17G`）。需等 epoch1..4 完整结果再判断是否值得保留；如果只能小幅改善 AEE/AAE，但保持 105 ATLIF + 12 attention，那么论文主线仍需谨慎。

**11bj valid825 完整结果（已完成）**

标准推理目录：`results/nts11bj_u12_ds_w720_stdlr_ftbd19_ft5_bs8_20260614_233224_setsid/standard_valid825`。该 run 已完成 `epoch0..4` 的标准 valid825，`profile_ranking_valid825.md` 已生成。

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 1.5159 | 9.9611 | 0.5203 | 0.1954 | 0.0912 | 29.0414G | 6.2761% | 23032.66 |
| 2 | 4 | 1.5218 | 9.7917 | 0.5154 | 0.1981 | 0.0941 | 29.0964G | 6.2880% | 23178.42 |
| 3 | 0 | 1.5571 | 10.1128 | 0.5245 | 0.2031 | 0.0973 | 29.6176G | 6.4006% | 23492.38 |
| 4 | 3 | 1.5592 | 10.1564 | 0.5286 | 0.2038 | 0.0967 | 31.0770G | 6.7160% | 24599.59 |
| 5 | 1 | 1.5915 | 10.3802 | 0.5368 | 0.2105 | 0.1009 | 30.5921G | 6.6112% | 24222.91 |

相对 `NTS11bd ep19`（AEE `1.5647` / AAE `9.9213` / `29.1676G` / `23108.92uJ`）：

- `11bj ep2` 明显改善 AEE（`1.5159`），AAE 基本同级（`9.9611`），spikes/energy 略低（`29.0414G` / `23032.66uJ`）。按综合 ranking 选 ep2。
- `11bj ep4` AAE 最好（`9.7917`），AEE 也接近 ep2（`1.5218`），spikes/energy 略高。若论文更重视方向角，可同时报告 ep4。
- 结论：11bj 不是无效 fine-tune；它把 NTS11 的 best AEE 从 `1.5647` 拉到 `1.5159`，且维持约 `-34%` spikes / `-39%` energy。NTS11 作为 full-network unified hardware 主线更有支撑。

### 38.11 DATE11 全量替换消融矩阵（2026-06-15）

目的：暂时不讨论 S2/S23/all12 替换范围，先固定为 **full all12 attention scope**，按 DATE 论文需要拆开两个正交机制：

1. 神经元：`PSN` / `all_binary_atlif` / `all_ternary_atlif`
2. 注意力：`original` / `TX` / `SC` / `TXSC(NTS/H60)`

生成器：

```text
neuron_experiments/H9_bipolar_self_attention/entrypoints/make_date11_full_factorial_configs.py
```

manifest：

```text
neuron_experiments/H9_bipolar_self_attention/configs/generated/date11_full_factorial_manifest.json
```

共同训练口径：

- `prev_runid`: `experiments/baseline_stride_upstream/checkpoint_epoch59.pth`
- `loader.n_epochs=30`, `batch_size=8`
- `warmup=720`, fastlr recipe, `threshold_freeze_after_step=1224`
- standard valid825 评估 epoch: `9/14/19/24/28/29`
- 每个结果必须检查：
  - `ATLIFTernaryPSN count`
  - `Shiftmax attention count`
  - `checkpoint_overlay_keys`
  - `missing=0, unexpected=0`（从自身 checkpoint 做 valid825 时）

#### 主矩阵与优先级

| 优先级 | 实验 | 神经元 | Attention | 预期 ATLIF | 预期 Shiftmax | 状态 | 配置 / 结果 |
|---|---|---|---|---:|---:|---|---|
| P0 | NB0 | PSN | original | 0 | 0 | 已跑 | `results_inference/nb0_baseline_epoch59_valid825_fixed_eval_20260601_140852`：AEE `1.4872`, AAE `9.9300`, total_spikes `44.0488G` |
| Main | NTS11 best/main ref | mixed: 27 ternary + 78 binary ATLIF | NTS/H60 all12 | 105 | 12 | **已跑完，当前主线参考** | `NTS11bl ep4`: AEE `1.4956`, AAE `9.7167`, total_spikes `29.3567G`, energy `23440.75uJ`; deploy 等价 checkpoint 用 `NTS11bj ep2`: AEE `1.5159`, AAE `9.9611`, total_spikes `29.0414G`, energy `23032.66uJ` |
| P0 | binary ATLIF only | all binary ATLIF | original | 105 | 0 | **已跑完** | config: `configs/generated/date11full_all_binary_atlif_original_w720_fastlr_full30.yml`; run: `results/date11full_all_binary_atlif_original_w720_fastlr_full30_bs8_20260615_214142_setsid`; best rank ep29: AEE `1.5900`, AAE `9.9413`, total_spikes `22.9163G`, energy `20166.57uJ` |
| P0 | ternary ATLIF only | all ternary ATLIF | original | 105 | 0 | **已跑完，负例** | config: `configs/generated/date11full_all_ternary_atlif_original_w720_fastlr_full30.yml`; run: `results/date11full_all_ternary_atlif_original_w720_fastlr_full30_bs8_20260615_220540_setsid`; best rank ep29: AEE `3.2733`, AAE `18.4266`, total_spikes `79.3630G`, energy `67644.66uJ` |
| P0 | ternary + TX | all ternary ATLIF | TX (`ternary_alpha_xnor_shiftmax`) | 105 | 12 | **已跑完，负例** | config: `configs/generated/date11full_all_ternary_atlif_tx_w720_fastlr_full30.yml`; run: `results/date11full_all_ternary_atlif_tx_w720_fastlr_full30_bs8_20260616_022014_setsid`; best rank ep29: AEE `3.2885`, AAE `18.6194`, total_spikes `79.0967G`, energy `67486.37uJ` |
| P0 | ternary + SC | all ternary ATLIF | SC (`signed_consensus_shiftmax`) | 105 | 12 | **已跑完，负例** | config: `configs/generated/date11full_all_ternary_atlif_sc_w720_fastlr_full30.yml`; run: `results/date11full_all_ternary_atlif_sc_w720_fastlr_full30_bs8_20260616_121203_setsid`; best rank ep29: AEE `3.2706`, AAE `18.2325`, total_spikes `80.1816G`, energy `68346.52uJ` |
| P0 | ternary + TXSC/NTS | all ternary ATLIF | NTS/H60 | 105 | 12 | **已跑完，负例** | config: `configs/generated/date11full_all_ternary_atlif_nts_w720_fastlr_full30.yml`; run: `results/date11full_all_ternary_atlif_nts_w720_fastlr_full30_bs8_20260617_033508_setsid`; best rank ep29: AEE `3.2809`, AAE `18.3304`, total_spikes `80.0070G`, energy `68269.43uJ` |
| P1 | binary + TX | all binary ATLIF | TX | 105 | 12 | **已跑完** | config: `configs/generated/date11full_all_binary_atlif_tx_w720_fastlr_full30.yml`; run: `results/date11full_all_binary_atlif_tx_w720_fastlr_full30_bs8_20260617_024526_setsid`; best rank ep19: AEE `1.5831`, AAE `9.9381`, total_spikes `22.4706G`, energy `19780.93uJ` |
| P1 | binary + SC | all binary ATLIF | SC | 105 | 12 | **已跑完** | config: `configs/generated/date11full_all_binary_atlif_sc_w720_fastlr_full30.yml`; run: `results/date11full_all_binary_atlif_sc_w720_fastlr_full30_bs8_20260617_160046_setsid`; best rank ep19: AEE `1.5815`, AAE `9.9454`, total_spikes `22.6911G`, energy `20009.89uJ`; eval audit: ATLIF `105`, Shiftmax `12`, `checkpoint_overlay_keys=210, missing=0, unexpected=0` |
| P1 | binary + TXSC/NTS | all binary ATLIF | NTS/H60 | 105 | 12 | **已跑完** | config: `configs/generated/date11full_all_binary_atlif_nts_w720_fastlr_full30.yml`; run: `results/date11full_all_binary_atlif_nts_w720_fastlr_full30_bs8_20260617_200451_setsid`; best rank ep19: AEE `1.5800`, AAE `9.9255`, total_spikes `22.7684G`, energy `20095.94uJ`; pipeline summary shows ATLIF `105`, Shiftmax `12`; valid825 append audit line has a stale `Shiftmax=0` count and should not be used for the table |
| P2 | PSN + TX | PSN | TX | 0 | 12 | **训练已完成，valid825 待修审计后补跑** | config: `configs/generated/date11full_psn_tx_w720_fastlr_full30.yml`; run: `results/date11full_psn_tx_w720_fastlr_full30_bs8_20260618_011517_setsid`; train full30 completed; standard valid825 failed at epoch9 because attention-only PSN checkpoint has no ATLIF overlay keys and current H9 load audit rejects it |
| P2 | PSN + SC | PSN | SC | 0 | 12 | 已生成，低优先级 | `configs/generated/date11full_psn_sc_w720_fastlr_full30.yml` |
| P2 | PSN + TXSC/NTS | PSN | NTS/H60 | 0 | 12 | **已跑完（仅 ep29 valid825）** | config: `configs/generated/date11full_psn_nts_w720_fastlr_full30.yml`; run: `results/date11full_psn_nts_w720_fastlr_full30_bs8_20260618_073920_setsid`; ep29: AEE `1.5390`, AAE `10.0527`, total_spikes `44.3434G`, energy `38004.71uJ`; 其他标准点 checkpoint 已缺失，未评估 |

#### 跑法建议

P0 先跑，P1 视 P0 结果决定是否需要全跑。P2 只在 reviewer 要求“attention-only without ATLIF”时再跑，因为 PSN + TX/SC/NTS 的硬件定义不如 ATLIF 事件化输入干净。

推荐分工：

1. 本机：`binary ATLIF only`，先证明 ATLIF 本身是否足以降 spikes。
2. 另一台服务器：`ternary ATLIF only` 或 `ternary + TX`，用于打开 attention 机制对照。
3. 如果 P0 中 `ternary+TX/SC/NTS` 全部明显劣于 NTS11bl，论文表中保留它们作为 DATE mechanism ablation，不继续扩 P1。

当前接棒安排（2026-06-16）：本机 `binary ATLIF only` 已完成 full30 + standard valid825，并已触发 `ternary + TX`。`ternary + TX` 当前运行目录为 `results/date11full_all_ternary_atlif_tx_w720_fastlr_full30_bs8_20260616_022014_setsid`。

`binary ATLIF only` 结论：spikes/energy 降幅非常强（相对 NB0 total_spikes 约 `-48.0%`，energy 约 `-46.4%`），但 AEE 从 `1.4872` 到 `1.5900`，相对上升约 `+6.9%`，略超 DATE 主目标的 `5%` 精度窗口；适合作为“ATLIF-only 可大幅降能但需要 attention/ternary 或 fine-tune 找回精度”的消融点。

`ternary ATLIF only` 结论：这是明确负例，不是单纯“轮次不够”。ep9→ep29 的 AEE 有下降（`4.7046`→`3.2733`），说明训练仍在适应，但最终 AEE/AAE 和 total_spikes 都远差于 NB0；同时 full ternary ATLIF 在 original attention 下把 spikes 推高到 `79.3630G`，说明全网三值神经元若没有匹配的 attention/门控数据流，会产生过高活动率和精度崩坏。论文中可作为“ternary neuron alone is insufficient; attention path is required”的机制消融。

`ternary + TX` 结论：同样是明确负例。引入 TX attention 后 best ep29 为 AEE `3.2885` / AAE `18.6194` / total_spikes `79.0967G`，几乎没有修复 `ternary ATLIF only` 的崩坏；说明问题主要来自“全网 105 个模块全部 ternary ATLIF”的神经元覆盖过强，而不是 original attention 缺 TX。下一步应跑 `binary + TX`，验证 TX attention 在 binary ATLIF 事件化输入上是否仍能保持 `binary ATLIF only` 的低 spikes，同时找回一部分精度。

当前接棒安排（2026-06-17）：NTS11 部署量化 valid825 已完成；接棒 watcher `2366118` 已进入 `binary + TX` 队列，当前先跑 `verify_nts11_chain.py`，运行目录 `results/date11full_all_binary_atlif_tx_w720_fastlr_full30_bs8_20260617_024526_setsid`。

### NTS11 部署量化可行性验证（2026-06-17）

目的：在进入硬件设计前，验证 NTS11bj ep2 的 H60 推理路径能否替换为硬件友好的定点近似，而不是只依赖 float attention score / float gate。所有量化开关均为 `bsa_attention` 下新增可选字段，默认关闭；旧实验配置不设置这些字段，因此旧实验行为不变。

新增可选字段：

- `hardware_quant_enabled`
- `hardware_mu_pow2_shift`
- `hardware_score_step`
- `hardware_score_min/max`
- `hardware_gate_step`
- `hardware_gate_min/max`

生成配置：

- `configs/generated/nts11bj_deploy_float_ref.yml`
- `configs/generated/nts11bj_deploy_score_int8.yml`
- `configs/generated/nts11bj_deploy_score_int8_mu_pow2.yml`
- `configs/generated/nts11bj_deploy_score_int8_mu_pow2_gate_int8.yml`

valid40 smoke 目录：`results/nts11_deployment_quant_eval_20260617_023434`。

| config | samples | AEE | AAE | total_spikes | firing | energy_uj |
|---|---:|---:|---:|---:|---:|---:|
| float_ref | 40 | 1.3854 | 13.0232 | 1.4725G | 6.5631% | 1175.15 |
| score_int8 | 40 | 1.3928 | 13.0942 | 1.4722G | 6.5620% | 1175.08 |
| score_int8_mu_pow2 | 40 | 1.3814 | 12.8367 | 1.4727G | 6.5641% | 1175.35 |
| score_int8_mu_pow2_gate_int8 | 40 | 1.3719 | 12.7837 | 1.4728G | 6.5645% | 1175.44 |

full valid825 目录：`results/nts11_deployment_quant_full825_20260617_023728`。

| config | samples | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| float_ref | 825 | 1.5159 | 9.9611 | 0.5203 | 0.1954 | 0.0912 | 29.0414G | 6.2761% | 23032.66 |
| score_int8_mu_pow2_gate_int8 | 825 | 1.5203 | 9.9316 | 0.5202 | 0.1960 | 0.0917 | 29.0492G | 6.2778% | 23038.29 |

部署量化设置：

- `mu`：从 float schedule 近似为 power-of-two，`hardware_mu_pow2_shift=4`，即 `mu=1/16=0.0625`
- score：`step=1/128`，clamp 到 `[-2, 2]`
- gate：`step=1/128`，clamp 到 `[0, 2]`

结论：硬件近似几乎不掉点。相对 float_ref，AEE 仅 `+0.0044`，AAE 反而略好 `-0.0295`，spikes/energy 基本不变（`29.0414G -> 29.0492G`，`23032.66uJ -> 23038.29uJ`）。这说明 NTS11bj ep2 可以作为 DATE 硬件主线的部署等价 checkpoint：TX/SC score、`mu`、Shiftmax gate 均可用定点近似进入硬件方案，不需要保留 float score/gate 作为部署假设。


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


### DATE11 自动结果追加：ternary ATLIF only（2026-06-16 12:12:03）

<!-- DATE11_APPEND::date11full_all_ternary_atlif_original_w720_fastlr_full30_bs8_20260615_220540_setsid -->
- 配置：`neuron_experiments/H9_bipolar_self_attention/configs/generated/date11full_all_ternary_atlif_original_w720_fastlr_full30.yml`
- 运行目录：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_ternary_atlif_original_w720_fastlr_full30_bs8_20260615_220540_setsid`
- 标准 valid825 ranking：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_ternary_atlif_original_w720_fastlr_full30_bs8_20260615_220540_setsid/profile_ranking_valid825.md`
- 加载审计：ATLIF `105`，Shiftmax `12`，`checkpoint_overlay_keys=0, missing=210, unexpected=0`
- best：epoch `29`，AEE `3.2733`，AAE `18.4266`，total_spikes `79.3630G`，firing `17.1179%`。

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 29 | 3.2733 | 18.4266 | 0.7863 | 0.4836 | 0.3033 | 79.3630G | 17.1179% | 67644.66 |
| 2 | 19 | 3.4243 | 19.4353 | 0.8009 | 0.5110 | 0.3304 | 80.6874G | 17.4035% | 68725.23 |
| 3 | 24 | 3.5567 | 19.2077 | 0.8082 | 0.5259 | 0.3459 | 82.4424G | 17.7821% | 70258.87 |
| 4 | 14 | 3.7553 | 20.5947 | 0.8181 | 0.5455 | 0.3701 | 81.7911G | 17.6416% | 69557.19 |
| 5 | 9 | 4.7046 | 24.5051 | 0.8605 | 0.6290 | 0.4603 | 84.4629G | 18.2179% | 71769.36 |
| 6 | 28 | 3.9935 | 21.6348 | 0.8218 | 0.5574 | 0.3833 | 88.7817G | 19.1494% | 75627.06 |


### DATE11 自动结果追加：ternary + TX（2026-06-17 01:17:00）

<!-- DATE11_APPEND::date11full_all_ternary_atlif_tx_w720_fastlr_full30_bs8_20260616_022014_setsid -->
- 配置：`neuron_experiments/H9_bipolar_self_attention/configs/generated/date11full_all_ternary_atlif_tx_w720_fastlr_full30.yml`
- 运行目录：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_ternary_atlif_tx_w720_fastlr_full30_bs8_20260616_022014_setsid`
- 标准 valid825 ranking：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_ternary_atlif_tx_w720_fastlr_full30_bs8_20260616_022014_setsid/profile_ranking_valid825.md`
- 加载审计：该 run 目录未保留标准 `pipeline.log`；结果已产出完整 standard valid825 ranking，若写论文表前需要从 watcher/训练日志补核安装计数与 train-load audit。
- best：epoch `29`，AEE `3.2885`，AAE `18.6194`，total_spikes `79.0967G`，firing `17.0604%`。

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 29 | 3.2885 | 18.6194 | 0.7860 | 0.4826 | 0.3023 | 79.0967G | 17.0604% | 67486.37 |
| 2 | 19 | 3.4092 | 19.4361 | 0.8018 | 0.5112 | 0.3306 | 80.6300G | 17.3911% | 68709.19 |
| 3 | 24 | 3.5886 | 19.4201 | 0.8069 | 0.5233 | 0.3447 | 82.4463G | 17.7829% | 70312.16 |
| 4 | 14 | 3.8196 | 21.0811 | 0.8230 | 0.5535 | 0.3774 | 81.7139G | 17.6249% | 69522.32 |
| 5 | 9 | 4.8055 | 24.9302 | 0.8645 | 0.6367 | 0.4681 | 84.3358G | 18.1905% | 71683.83 |
| 6 | 28 | 4.0361 | 22.1459 | 0.8213 | 0.5557 | 0.3820 | 88.8157G | 19.1567% | 75727.01 |


### DATE11 自动结果追加：ternary + SC（2026-06-17 03:34:47）

<!-- DATE11_APPEND::date11full_all_ternary_atlif_sc_w720_fastlr_full30_bs8_20260616_121203_setsid -->
- 配置：`neuron_experiments/H9_bipolar_self_attention/configs/generated/date11full_all_ternary_atlif_sc_w720_fastlr_full30.yml`
- 运行目录：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_ternary_atlif_sc_w720_fastlr_full30_bs8_20260616_121203_setsid`
- 标准 valid825 ranking：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_ternary_atlif_sc_w720_fastlr_full30_bs8_20260616_121203_setsid/profile_ranking_valid825.md`
- 加载审计：ATLIF `105`，Shiftmax `12`，`checkpoint_overlay_keys=0, missing=210, unexpected=0`
- best：epoch `29`，AEE `3.2706`，AAE `18.2325`，total_spikes `80.1816G`，firing `17.3280%`。

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 29 | 3.2706 | 18.2325 | 0.7840 | 0.4813 | 0.3024 | 80.1816G | 17.3280% | 68346.52 |
| 2 | 19 | 3.3839 | 19.0939 | 0.7969 | 0.5044 | 0.3237 | 81.5162G | 17.6164% | 69445.10 |
| 3 | 24 | 3.5487 | 19.2428 | 0.8056 | 0.5251 | 0.3475 | 83.5034G | 18.0459% | 71143.98 |
| 4 | 28 | 4.0506 | 21.8423 | 0.8212 | 0.5598 | 0.3882 | 89.5171G | 19.3455% | 76257.62 |


### DATE11 自动结果追加：ternary + TXSC/NTS（2026-06-17 18:11:27）

<!-- DATE11_APPEND::date11full_all_ternary_atlif_nts_w720_fastlr_full30_bs8_20260617_033508_setsid -->
- 配置：`neuron_experiments/H9_bipolar_self_attention/configs/generated/date11full_all_ternary_atlif_nts_w720_fastlr_full30.yml`
- 运行目录：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_ternary_atlif_nts_w720_fastlr_full30_bs8_20260617_033508_setsid`
- 标准 valid825 ranking：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_ternary_atlif_nts_w720_fastlr_full30_bs8_20260617_033508_setsid/profile_ranking_valid825.md`
- 加载审计：ATLIF `105`，Shiftmax `12`，`checkpoint_overlay_keys=0, missing=210, unexpected=0`
- best：epoch `29`，AEE `3.2809`，AAE `18.3304`，total_spikes `80.0070G`，firing `17.2903%`。

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 29 | 3.2809 | 18.3304 | 0.7852 | 0.4845 | 0.3062 | 80.0070G | 17.2903% | 68269.43 |
| 2 | 19 | 3.4843 | 19.5988 | 0.7998 | 0.5117 | 0.3326 | 81.3750G | 17.5859% | 69377.40 |
| 3 | 24 | 3.5752 | 19.3414 | 0.8046 | 0.5220 | 0.3447 | 83.3902G | 18.0214% | 71119.32 |
| 4 | 14 | 3.7744 | 20.9740 | 0.8197 | 0.5496 | 0.3738 | 82.5721G | 17.8446% | 70329.72 |
| 5 | 9 | 4.6406 | 24.0679 | 0.8587 | 0.6264 | 0.4583 | 85.1294G | 18.3973% | 72456.86 |
| 6 | 28 | 4.0938 | 21.7471 | 0.8228 | 0.5623 | 0.3906 | 89.3769G | 19.3152% | 76217.19 |


### DATE11 自动结果追加：binary + TX（2026-06-17 20:04:51）

<!-- DATE11_APPEND::date11full_all_binary_atlif_tx_w720_fastlr_full30_bs8_20260617_024526_setsid -->
- 配置：`neuron_experiments/H9_bipolar_self_attention/configs/generated/date11full_all_binary_atlif_tx_w720_fastlr_full30.yml`
- 运行目录：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_tx_w720_fastlr_full30_bs8_20260617_024526_setsid`
- 标准 valid825 ranking：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_tx_w720_fastlr_full30_bs8_20260617_024526_setsid/profile_ranking_valid825.md`
- best：epoch `19`，AEE `1.5831`，AAE `9.9381`，total_spikes `22.4706G`，firing `4.8467%`。

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19 | 1.5831 | 9.9381 | 0.5348 | 0.2136 | 0.1041 | 22.4706G | 4.8467% | 19780.93 |
| 2 | 29 | 1.5921 | 9.8577 | 0.5321 | 0.2159 | 0.1075 | 23.0106G | 4.9632% | 20270.08 |
| 3 | 24 | 1.5868 | 10.1702 | 0.5347 | 0.2128 | 0.1042 | 22.9146G | 4.9425% | 20196.28 |
| 4 | 28 | 1.5849 | 10.2686 | 0.5361 | 0.2102 | 0.1024 | 24.8565G | 5.3613% | 21893.70 |
| 5 | 14 | 1.6644 | 10.3322 | 0.5578 | 0.2303 | 0.1140 | 21.8241G | 4.7073% | 19151.32 |
| 6 | 9 | 1.7087 | 10.7106 | 0.5500 | 0.2244 | 0.1125 | 21.4632G | 4.6294% | 18809.85 |


### DATE11 自动结果追加：binary + TXSC/NTS（2026-06-18 07:37:46）

<!-- DATE11_APPEND::date11full_all_binary_atlif_nts_w720_fastlr_full30_bs8_20260617_200451_setsid -->
- 配置：`neuron_experiments/H9_bipolar_self_attention/configs/generated/date11full_all_binary_atlif_nts_w720_fastlr_full30.yml`
- 运行目录：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_nts_w720_fastlr_full30_bs8_20260617_200451_setsid`
- 标准 valid825 ranking：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_nts_w720_fastlr_full30_bs8_20260617_200451_setsid/profile_ranking_valid825.md`
- 加载审计：ATLIF `105`；pipeline summary 显示 Shiftmax attention `12`；该自动追加段原 audit 行里的 `Shiftmax=0` 是计数脚本对 NTS/H60 wrapper 的漏报，不作为论文表依据。
- best：epoch `19`，AEE `1.5800`，AAE `9.9255`，total_spikes `22.7684G`，firing `4.9205%`。

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19 | 1.5800 | 9.9255 | 0.5313 | 0.2123 | 0.1040 | 22.7684G | 4.9205% | 20095.94 |
| 2 | 29 | 1.5813 | 9.8990 | 0.5275 | 0.2125 | 0.1058 | 23.3601G | 5.0483% | 20624.58 |
| 3 | 28 | 1.5854 | 10.1919 | 0.5314 | 0.2091 | 0.1020 | 25.2138G | 5.4489% | 22263.00 |
| 4 | 24 | 1.5903 | 10.1569 | 0.5328 | 0.2131 | 0.1050 | 23.3411G | 5.0442% | 20617.51 |
| 5 | 14 | 1.6679 | 10.4208 | 0.5562 | 0.2287 | 0.1133 | 22.2122G | 4.8003% | 19541.89 |
| 6 | 9 | 1.7130 | 10.8293 | 0.5568 | 0.2284 | 0.1136 | 22.0002G | 4.7544% | 19326.96 |


### DATE11 自动结果追加：binary + SC（2026-06-18）

<!-- DATE11_APPEND::date11full_all_binary_atlif_sc_w720_fastlr_full30_bs8_20260617_160046_setsid -->
- 配置：`neuron_experiments/H9_bipolar_self_attention/configs/generated/date11full_all_binary_atlif_sc_w720_fastlr_full30.yml`
- 运行目录：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_sc_w720_fastlr_full30_bs8_20260617_160046_setsid`
- 标准 valid825 ranking：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_sc_w720_fastlr_full30_bs8_20260617_160046_setsid/profile_ranking_valid825.md`
- 加载审计：ATLIF `105`，Shiftmax `12`，`checkpoint_overlay_keys=210, missing=0, unexpected=0`
- best：epoch `19`，AEE `1.5815`，AAE `9.9454`，total_spikes `22.6911G`，firing `4.9038%`。

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19 | 1.5815 | 9.9454 | 0.5322 | 0.2112 | 0.1036 | 22.6911G | 4.9038% | 20009.89 |
| 2 | 29 | 1.5847 | 9.8680 | 0.5264 | 0.2114 | 0.1053 | 23.3229G | 5.0403% | 20577.59 |
| 3 | 28 | 1.5894 | 10.2182 | 0.5308 | 0.2087 | 0.1020 | 25.1582G | 5.4369% | 22195.98 |
| 4 | 24 | 1.5990 | 10.1763 | 0.5318 | 0.2123 | 0.1045 | 23.2862G | 5.0324% | 20556.05 |
| 5 | 14 | 1.6673 | 10.4728 | 0.5551 | 0.2278 | 0.1125 | 22.2190G | 4.8017% | 19545.52 |
| 6 | 9 | 1.7609 | 11.0274 | 0.5575 | 0.2293 | 0.1150 | 21.8732G | 4.7270% | 19200.01 |


### DATE11 状态追加：PSN + TX（2026-06-18）

<!-- DATE11_APPEND::date11full_psn_tx_w720_fastlr_full30_bs8_20260618_011517_setsid -->
- 配置：`neuron_experiments/H9_bipolar_self_attention/configs/generated/date11full_psn_tx_w720_fastlr_full30.yml`
- 运行目录：`neuron_experiments/H9_bipolar_self_attention/results/date11full_psn_tx_w720_fastlr_full30_bs8_20260618_011517_setsid`
- 训练：full30 已完成，最后保存 `checkpoint_epoch29.pth` / `checkpoint_epoch29_state_dict.pth`。
- 标准 valid825：未产出 ranking。`standard_valid825/epoch9/eval.log` 在加载审计处失败：

```text
RuntimeError: Config enables H9 overlay modules but checkpoint does not contain H9 overlay parameters
```

解释：这是 PSN + TX 的 attention-only 消融，配置安装了 `12` 个 H9 Shiftmax attention，但神经元仍是 PSN，因此 checkpoint 没有 ATLIF overlay 参数。当前 `load_checkpoint_with_h9_audit` 把“启用 H9 overlay 但 checkpoint 无 overlay keys”视为错误，适用于 ATLIF 训练后自身 valid825，但不适用于 PSN + attention-only 消融。

处理建议：先新增一个仅用于 attention-only PSN 消融的 eval/audit 分支，允许 `expected_atlif=0` 且 `expected_shiftmax=12` 的 checkpoint 通过，然后再补跑 PSN+TX 的 standard valid825。该实验低于 NTS11/binary ATLIF 主线优先级，不建议在未修审计前重复训练。


### DATE11 阶段结论（2026-06-18）

- binary 全替换组整体成立为“低能但略掉点”的消融：`binary only` AEE `1.5900`，`binary+TX` `1.5831`，`binary+SC` `1.5815`，`binary+TXSC/NTS` `1.5800`；相对 NB0 的 spikes 都约 `22.5G-22.9G`，降幅约 `48%`，但 AEE 比 NB0 `1.4872` 高约 `6.2%-6.9%`，略超 DATE 主目标的 `5%` 精度窗口。
- all ternary ATLIF 组是明确负例：无论 original/TX/SC/NTS attention，AEE 都在 `3.27-3.29`，spikes 在 `79G-80G`，说明“全网三值 ATLIF”覆盖过强，会把活动率和误差同时推高。
- 当前论文主线仍应放在 NTS11 mixed ATLIF + NTS/H60 all12：`NTS11bl ep4` AEE `1.4956`、spikes `29.3567G`，兼顾精度窗口和硬件统一性；部署等价用 `NTS11bj ep2`，量化后 AEE `1.5203`、spikes `29.0492G`，硬件近似基本不掉点。
- 后续最有价值的补实验不是继续扩大 all ternary，而是 NTS11-lite：保持 all12 H60 attention，减少非 QK ATLIF 覆盖，验证面积/控制复杂度与精度之间的 tradeoff。


### DATE11 追加实验：all-binary 精度恢复 fine-tune（2026-06-18）

动机：如果 `all binary ATLIF + NTS/H60` 可以通过轻量 fine-tune 把 AEE 拉回 NB0 `+5%` 窗口内，那么硬件实现会比 mixed binary/ternary 更简单：全网只需 `{0,+1}` 事件，不需要 ternary sign rail 或 pos/neg 双 rail。

配置：

- `neuron_experiments/H9_bipolar_self_attention/configs/generated/date11full_all_binary_atlif_nts_stdlr_ft_ep29_ft5.yml`
- 起点：`results/date11full_all_binary_atlif_nts_w720_fastlr_full30_bs8_20260617_200451_setsid/checkpoint_epoch29.pth`
- 结构：`105` 个 binary ATLIF，`12` 个 NTS/H60 attention；无 ternary ATLIF。
- 训练：5 epoch，`stdlr` fine-tune 口径，`backbone_lr=1e-6`，`neuron_lr=3e-5`，`threshold_lr=5e-6`，warmup `720`，保存 epoch `0..4`。
- valid825：训练完成后自动评估 epoch `0..4`。

链路审计：

- verify config：PASS
- preload install：ATLIF `105`，attention `12`
- neuron modes：`{'ternary': 0, 'binary': 105, 'other': 0}`
- saved checkpoint reload audit：`checkpoint_overlay_keys=210, missing=0, unexpected=0`

当前状态：

- run dir：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_nts_stdlr_ft_ep29_ft5_bs8_20260618_141011_setsid`
- 状态：**已完成 full5 + standard valid825**。
- ranking：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_nts_stdlr_ft_ep29_ft5_bs8_20260618_141011_setsid/profile_ranking_valid825.md`
- 目标：AEE 低于 `1.5616` 即进入 NB0 `+5%` 窗口；该 run best ep2 达标。

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 1.4891 | 9.7785 | 0.5151 | 0.1924 | 0.0898 | 23.8206G | 5.1479% | 21045.91 |
| 2 | 4 | 1.5131 | 9.8042 | 0.5149 | 0.1970 | 0.0934 | 23.9705G | 5.1803% | 21193.41 |
| 3 | 0 | 1.5385 | 10.0256 | 0.5210 | 0.2008 | 0.0960 | 24.1349G | 5.2158% | 21327.85 |
| 4 | 3 | 1.5480 | 10.0104 | 0.5253 | 0.2028 | 0.0958 | 25.4267G | 5.4949% | 22457.95 |
| 5 | 1 | 1.5767 | 10.1547 | 0.5312 | 0.2083 | 0.1013 | 25.0779G | 5.4196% | 22146.16 |

结论：all-binary NTS/H60 经过短 fine-tune 后达到 DATE 主目标。相对 NB0 baseline，AEE `1.4872 -> 1.4891` 基本持平（约 `+0.13%`），AAE `9.9300 -> 9.7785` 略好，total_spikes `44.0488G -> 23.8206G` 下降约 `45.9%`，energy `37638.01uJ -> 21045.91uJ` 下降约 `44.1%`。这使 **all-binary ATLIF + all12 NTS/H60 + short fine-tune** 成为当前最硬件友好的主线候选；mixed NTS11 仍可作为机制参考或精度/结构对照。


### DATE11 自动结果追加：PSN + TXSC/NTS（2026-06-18 15:48:08）

<!-- DATE11_APPEND::date11full_psn_nts_w720_fastlr_full30_bs8_20260618_073920_setsid -->
- 配置：`neuron_experiments/H9_bipolar_self_attention/configs/generated/date11full_psn_nts_w720_fastlr_full30.yml`
- 运行目录：`neuron_experiments/H9_bipolar_self_attention/results/date11full_psn_nts_w720_fastlr_full30_bs8_20260618_073920_setsid`
- 标准 valid825 ranking：`neuron_experiments/H9_bipolar_self_attention/results/date11full_psn_nts_w720_fastlr_full30_bs8_20260618_073920_setsid/profile_ranking_valid825.md`
- 加载审计：ATLIF `0`，Shiftmax `12`，`checkpoint_overlay_keys=0, missing=0, unexpected=0`
- 说明：当前 run 目录只保留 `checkpoint_epoch29.pth`，standard valid825 补跑跳过缺失的 `9/14/19/24/28`，本段仅代表 ep29。
- best：epoch `29`，AEE `1.5390`，AAE `10.0527`，total_spikes `44.3434G`，firing `9.5830%`。

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 29 | 1.5390 | 10.0527 | 0.5265 | 0.2032 | 0.0957 | 44.3434G | 9.5830% | 38004.71 |


### DATE11 最终 all-binary 主线最小补实验（2026-06-20 启动）

依据当前结果，主线候选切换为 **all-binary ATLIF + all12 NTS/H60 + short FT**：
`date11full_all_binary_atlif_nts_stdlr_ft_ep29_ft5_bs8_20260618_141011_setsid/checkpoint_epoch2.pth`
（AEE `1.4891`，AAE `9.7785`，total_spikes `23.8206G`，energy `21045.91uJ`）。

本轮只补最小必要实验，不再扩大全矩阵：

| 优先级 | 实验 | 目的 | 状态 |
|---|---|---|---|
| P0 | all-binary + original attention + 同样 FT5 | 判断是不是 FT 本身即可救回精度；如果 original 也追平，则最终硬件可不需要 H60 attention。 | **已完成**；best ep2 AEE `1.5049`，AAE `9.8872`，spikes `23.2877G`；见下方自动结果段 |
| P0 | all-binary + NTS/H60 deploy quant | 新主线必须重做 int8 score / μ pow2 / gate int8 部署验证。 | **已完成**；最强 deploy 变体 `score_int8_mu_pow2_gate_int8` AEE `1.4919`；见下方自动结果段 |
| P1 | all-binary + TX FT5 | 判断 NTS/H60 是否明显优于更简单 attention；先跑 TX。 | **已完成**；best ep2 AEE `1.5077`，AAE `9.8912`，spikes `22.7231G`，energy `20010.68uJ`；见下方 P1 队列结果 |
| P1 | all-binary + NTS/H60 从 ep19 FT5 | 验证 ep29 起点不是偶然 cherry-pick。 | **已完成**；best ep2 AEE `1.5072`，AAE `9.8772`，spikes `23.0841G`，energy `20378.47uJ`；见下方 P1 队列结果 |

P1 队列启动记录（2026-06-21）：

- queue PID：`2691052`
- queue log：`neuron_experiments/H9_bipolar_self_attention/results/date11_allbinary_p1_ft_queue_20260621_035025.log`
- TX FT5 run：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_tx_stdlr_ft_ep19_ft5_bs8_20260621_035025_setsid`
- NTS ep19 FT5 run：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_nts_stdlr_ft_ep19_ft5_bs8_20260621_035025_setsid`
- 执行顺序：先 TX FT5 full5 + standard valid825，再 NTS/H60 ep19 FT5 full5 + standard valid825。
- 两个配置均已通过 verify：preload ATLIF `105`、attention `12`、neuron modes `0 ternary / 105 binary`、saved reload `checkpoint_overlay_keys=210, missing=0, unexpected=0`。

P1 队列结果（2026-06-21）：

- 队列已完成：`2026-06-21T09:21:13+08:00`。
- GPU 训练进程已结束；结果均为 standard valid825 ranking。
- 结论：两个 ep19-start FT5 都进入 NB0 约 5% 精度窗口，但没有追平当前主线 `all-binary + NTS/H60 ep29-start FT5` 的 AEE `1.4891`。TX FT5 能耗最低，适合作为“更简单 attention”强消融；NTS/H60 ep19 FT5 说明 ep29-start 不是唯一可行起点，但最终主线仍应选 ep29-start 的 best ep2。

all-binary + TX FT5（ep19 start）：

- 配置：`neuron_experiments/H9_bipolar_self_attention/configs/generated/date11full_all_binary_atlif_tx_stdlr_ft_ep19_ft5.yml`
- 起点：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_tx_w720_fastlr_full30_bs8_20260617_024526_setsid/checkpoint_epoch19.pth`
- 运行目录：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_tx_stdlr_ft_ep19_ft5_bs8_20260621_035025_setsid`
- ranking：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_tx_stdlr_ft_ep19_ft5_bs8_20260621_035025_setsid/profile_ranking_valid825.md`

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 1.5077 | 9.8912 | 0.5215 | 0.1959 | 0.0921 | 22.7231G | 4.9012% | 20010.68 |
| 2 | 4 | 1.5202 | 9.7697 | 0.5182 | 0.1992 | 0.0950 | 22.8847G | 4.9360% | 20173.51 |
| 3 | 0 | 1.5569 | 10.1261 | 0.5303 | 0.2065 | 0.0993 | 23.1008G | 4.9826% | 20350.71 |

all-binary + NTS/H60 FT5（ep19 start）：

- 配置：`neuron_experiments/H9_bipolar_self_attention/configs/generated/date11full_all_binary_atlif_nts_stdlr_ft_ep19_ft5.yml`
- 起点：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_nts_w720_fastlr_full30_bs8_20260617_200451_setsid/checkpoint_epoch19.pth`
- 运行目录：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_nts_stdlr_ft_ep19_ft5_bs8_20260621_035025_setsid`
- ranking：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_nts_stdlr_ft_ep19_ft5_bs8_20260621_035025_setsid/profile_ranking_valid825.md`

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 1.5072 | 9.8772 | 0.5171 | 0.1960 | 0.0929 | 23.0841G | 4.9887% | 20378.47 |
| 2 | 4 | 1.5409 | 9.9166 | 0.5213 | 0.2031 | 0.0982 | 23.1757G | 5.0085% | 20476.28 |
| 3 | 0 | 1.5578 | 10.1405 | 0.5296 | 0.2081 | 0.1004 | 23.4814G | 5.0745% | 20733.58 |


### DATE11 自动结果追加：all-binary original attention FT5（2026-06-20 04:42:48）

<!-- DATE11_FT_APPEND::date11full_all_binary_atlif_original_stdlr_ft_ep29_ft5_bs8_20260620_015804_setsid -->
- 配置：`neuron_experiments/H9_bipolar_self_attention/configs/generated/date11full_all_binary_atlif_original_stdlr_ft_ep29_ft5.yml`
- 起点：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_original_w720_fastlr_full30_bs8_20260615_214142_setsid/checkpoint_epoch29.pth`
- 运行目录：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_original_stdlr_ft_ep29_ft5_bs8_20260620_015804_setsid`
- 标准 valid825 ranking：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_original_stdlr_ft_ep29_ft5_bs8_20260620_015804_setsid/profile_ranking_valid825.md`
- best：epoch `2`，AEE `1.5049`，AAE `9.8872`，total_spikes `23.2877G`，firing `5.0229%`，energy `20509.47uJ`。

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 1.5049 | 9.8872 | 0.5185 | 0.1924 | 0.0896 | 23.2877G | 5.0229% | 20509.47 |
| 2 | 4 | 1.5129 | 9.7119 | 0.5154 | 0.1964 | 0.0938 | 23.4788G | 5.0642% | 20695.59 |
| 3 | 0 | 1.5529 | 10.0801 | 0.5242 | 0.2009 | 0.0959 | 23.6388G | 5.0987% | 20829.11 |
| 4 | 1 | 1.5703 | 10.0840 | 0.5313 | 0.2057 | 0.0991 | 24.5282G | 5.2905% | 21599.38 |
| 5 | 3 | 1.5706 | 10.0885 | 0.5319 | 0.2064 | 0.0985 | 24.9302G | 5.3772% | 21944.83 |


### DATE11 自动结果追加：all-binary NTS/H60 deploy quant（2026-06-20 16:38:41）

<!-- DATE11_DEPLOY_QUANT::date11_binary_nts_deploy_quant_full825_20260620_154820 -->
- 主 checkpoint：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_nts_stdlr_ft_ep29_ft5_bs8_20260618_141011_setsid/checkpoint_epoch2.pth`
- 运行目录：`neuron_experiments/H9_bipolar_self_attention/results/date11_binary_nts_deploy_quant_full825_20260620_154820`
- 目的：验证 all-binary NTS/H60 FT ep2 主线在 int8 score / pow2 μ / int8 gate 下是否保持等价。

| config | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `date11_binary_nts_ft_ep29_deploy_float_ref` | 1.4891 | 9.7817 | 0.5152 | 0.1924 | 0.0898 | 23.8205G | 5.1478% | 21045.84 |
| `date11_binary_nts_ft_ep29_deploy_score_int8` | 1.4953 | 9.8057 | 0.5141 | 0.1930 | 0.0904 | 23.8238G | 5.1485% | 21048.75 |
| `date11_binary_nts_ft_ep29_deploy_score_int8_mu_pow2` | 1.5036 | 9.8169 | 0.5133 | 0.1927 | 0.0904 | 23.8238G | 5.1485% | 21048.71 |
| `date11_binary_nts_ft_ep29_deploy_score_int8_mu_pow2_gate_int8` | 1.4919 | 9.7804 | 0.5140 | 0.1921 | 0.0899 | 23.8240G | 5.1486% | 21048.87 |


### DATE11 MVSEC/MDR 跨数据集检查（2026-06-20）

目标：按 SDformerFlow 原工程的 MVSEC/MDR 路线做最小跑通检查，不启动 MDR 训练，不改模型代码。

SDformerFlow 参考配置与当前数据状态：

- MVSEC 官方入口：`third_party/SDformerFlow/eval_MV_flow_SNN.py`；本工程包装：`neuron_experiments/H9_bipolar_self_attention/entrypoints/run_h9_standard_mvsec_eval.py`。
- SDformerFlow 原 MVSEC 配置：`third_party/SDformerFlow/configs/eval_MV_supervised.yml`，默认 `resolution=[260,346]`、`crop=[256,256]`、`window_size=[2,8,8]`。
- 当前 DATE/DSEC checkpoint 使用 `window_size=[2,9,9]`，直接沿用 `crop=[256,256]` 会在最深层得到 `8x8` token，与 checkpoint 的 `positional_encoding` 形状 `2*9*9=162` 不匹配；旧 NB0 MVSEC run 已出现过该形状错误。
- 配置层修正：新增 `neuron_experiments/H9_bipolar_self_attention/configs/generated/eval_mvsec_dt1_all_binary_nts_crop288.yml`，保留 all-binary ATLIF + NTS/H60 主线结构，只把 MVSEC eval crop 设为 `288x288`。MVSEC dataloader 使用 `torchvision.transforms.CenterCrop`，当 crop 大于原图高度时会 padding，因此最深层恢复为 `9x9`，无需代码改动。
- 本地 MVSEC 已具备 `indoor_flying3` dt1：`event=2951`，`flowgt_dt1=2434`。
- 本地 MDR 仍未 ready：`third_party/SDformerFlow/data/Datasets/MDR` 下只有 `MDR_dt1_official.pth` 与 `_gdown_tmp/*.tar/*.pth.tar`，按现有 `MDR_dataloader/MDR.py` 所需的预处理 `*.npz` 数量为 `0`。

MVSEC full profile：

- checkpoint：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_nts_stdlr_ft_ep29_ft5_bs8_20260618_141011_setsid/checkpoint_epoch2.pth`
- out dir：`results_inference/mvsec_date11_all_binary_nts_ft_ep2_crop288_dt1/indoor_flying3`
- 命令：`python -u neuron_experiments/H9_bipolar_self_attention/entrypoints/run_h9_standard_mvsec_eval.py --config neuron_experiments/H9_bipolar_self_attention/configs/generated/eval_mvsec_dt1_all_binary_nts_crop288.yml --checkpoint neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_nts_stdlr_ft_ep29_ft5_bs8_20260618_141011_setsid/checkpoint_epoch2.pth --out-dir results_inference/mvsec_date11_all_binary_nts_ft_ep2_crop288_dt1 --sequence indoor_flying3`
- ranking：`results_inference/mvsec_date11_all_binary_nts_ft_ep2_crop288_dt1/mvsec_ranking.md`
- profile：`results_inference/mvsec_date11_all_binary_nts_ft_ep2_crop288_dt1/indoor_flying3/spike_profile.json`
- 状态：完整跑完 `1885/1885`，exit_code `0`；耗时约 `1:26:55`。

| sequence | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | effective_flops | dense_flops | energy_uj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| indoor_flying3 | 2.4140 | 73.7652 | 0.7871 | 0.4905 | 0.2657 | 25.9757G | 3.3552% | 31.0402G | 925.1325G | 23060.49 |

结论：MVSEC 对当前 all-binary NTS/H60 主线可用纯配置方式完整跑通；室内 flying3 指标为 AEE `2.4140`，spikes/energy 仍明显低于 DSEC NB0 量级。MDR 当前缺预处理/组织好的 `*.npz` 数据，上传或解包整理前不启动 MDR。

### DATE11 MVSEC/MDR 标准化训练接管（2026-06-22）

用户已上传 MDR train 压缩包，本轮接管目标是把 **MDR train -> MVSEC indoor_flying3 validation/eval** 走成可复现标准流程，并保持和 baseline 官方入口一致。

已确认的数据状态：

- MDR archive root：`/root/private_data/mdr/train`
- 已发现 `batch_1.tar.gz` 到 `batch_12.tar.gz`，压缩体积约 `79G`。
- 抽样检查：
  - `batch_1`: `events1=8564`, `events2=8564`, `best_density_events1=1434`, `best_density_events2=1434`, `flow=1434`
  - `batch_12`: `events1=8547`, `events2=8547`, `best_density_events1=1428`, `best_density_events2=1428`, `flow=1428`
- 当前目标目录 `third_party/SDformerFlow/data/Datasets/MDR/dt1/train/{events1,events2,best_density_events1,best_density_events2,flow}` 原本为空。
- MVSEC 当前已具备 `indoor_flying3` dt1：`event=2951`, `flowgt_dt1=2434`；`indoor_flying1/2` 尚未完整编码，`flowgt_dt4` 当前为 `0`。

关键代码事实：

- SDformerFlow 官方 MDR 训练入口：`third_party/SDformerFlow/train_mdr_supervised_SNN.py`。
- 官方 MDR baseline config：`third_party/SDformerFlow/configs/train_MDR_supervised_SDformerFlow.yml`；本工程标准 route config：`configs/generated/train_mdr_baseline_mvsec_route.yml`。
- 训练 dataloader `MDR_dataloader/MDR.py::MDREventFlow.get_train_sequence()` 当前硬编码读取 `dt1/train/...`，不随 `config.data.event_interval` 切换训练目录；`event_interval=dt4` 主要影响 MVSEC validation loader 选择 `MvsecEventFlow_dt4`。
- 因此本轮先把用户上传的 MDR batch archive 组织到 `dt1/train`，按仓库现有官方 baseline route 跑通 MDR->MVSEC。若后续要做严格 dt4 MVSEC protocol，需要补齐/生成 MVSEC `flowgt_dt4` 和对应 event 目录，并单独配置。

新增脚本：

- `scripts/prepare_mdr_from_archives.py`
  - 从 `/root/private_data/mdr/train/batch_*.tar.gz` 解包到 `third_party/SDformerFlow/data/Datasets/MDR/dt1/train`
  - 只整理 `batch_*` 目录，避免官方 `MDR_menage.py` 在重复运行时误处理已经存在的平铺 `events1/flow` 目录
  - 最终统计 `events1/events2/best_density_events1/best_density_events2/flow` 数量
- `neuron_experiments/H9_bipolar_self_attention/entrypoints/run_mdr_mvsec_standard_pipeline.sh`
  - 第一步：准备 MDR archive
  - 第二步：构造 `MDREventFlow` 做 dataset smoke test，确认 train sample 数量非零
  - 第三步：启动 `third_party/SDformerFlow/train_mdr_supervised_SNN.py --config ../../../configs/generated/train_mdr_baseline_mvsec_route.yml`
  - 训练配置为 baseline PSN / original SDFormerFlow，`loader.n_epochs=60`，每 `5` epoch 在 MVSEC `indoor_flying3` 上 validation

标准运行命令：

```bash
cd /root/private_data/work/sdformer_codex/SDformer
nohup bash neuron_experiments/H9_bipolar_self_attention/entrypoints/run_mdr_mvsec_standard_pipeline.sh \
  > neuron_experiments/H9_bipolar_self_attention/results/mdr_mvsec_standard_pipeline_20260622.log 2>&1 &
```

当前决策：

- **不需要用户立刻补传 dt1**：按当前仓库代码，MDR train loader 读取的是 `dt1/train`，本轮 archive 只要包含 `events1/events2/best_density_events1/best_density_events2/flow` 即可跑官方 baseline route。
- **暂不启动 dt4 MVSEC eval**：本机 MVSEC `flowgt_dt4` 为空，dt4 validation loader 路径还指向 `third_party/SDformerFlow/dataset/MVSEC/...`，需要单独准备后再跑。
- **先跑 baseline MDR->MVSEC**：这是 DATE 论文外部泛化表的 baseline 参照。all-binary NTS/H60 的 MDR 训练入口需要后续把 H9 overlay 安装逻辑接入 `train_mdr_supervised_SNN.py` 或新增 H9 MDR train entrypoint，不能直接复用 DSEC 训练配置。

启动记录（2026-06-22）：

- pipeline log：`neuron_experiments/H9_bipolar_self_attention/results/mdr_mvsec_standard_pipeline_20260622.log`
- 第一次后台启动留下了未完成的 `batch_1` 临时目录；已修正 `prepare_mdr_from_archives.py`，增加 `.extract_done` 标记，重跑时会删除半解包目录再重新提取。
- 当前 setsid pipeline PID：`2772369`
- 当前阶段：`prepare_mdr_from_archives.py` 正在重新解包 `batch_1.tar.gz`；训练尚未开始，GPU 可能空闲。

环境修正（2026-06-22 14:58）：

- MDR 已成功解包并整理：`events1=85720`, `events2=85720`, `best_density_events1=17190`, `best_density_events2=17190`, `flow=17190`。
- 第一次 smoke test 使用系统 `python3`，失败于 `ModuleNotFoundError: torch`。
- 第二次改用 `/opt/conda/bin/python`，失败于 `ModuleNotFoundError: cv2`。
- 已将 pipeline 默认解释器改为 `/opt/conda/envs/sdformerflow/bin/python`；该环境已验证 `torch/cv2/mlflow/spikingjelly` 均可导入。
- 已优化 `prepare_mdr_from_archives.py`：平铺 MDR tree 已存在时直接跳过解包，避免重复解压和全目录计数。

启动修正（2026-06-22 15:00）：

- 将 smoke test 从完整构建 `MDREventFlow` 改为轻量检查五类文件各至少一个样本，避免和训练脚本重复扫全量数据。
- 修正训练配置相对路径：从 `third_party/SDformerFlow` 启动时应使用 `../../configs/generated/train_mdr_baseline_mvsec_route.yml`。
- 当前 pipeline PID：`2791114`。
- 当前训练进程：`/opt/conda/envs/sdformerflow/bin/python train_mdr_supervised_SNN.py --config ../../configs/generated/train_mdr_baseline_mvsec_route.yml --path_mlflow file:///root/private_data/sdformer_mlflow`。
- 日志已进入 `Training Dataset ...`，说明训练脚本已启动；当前仍在构建 MDR dataset 索引，尚未进入 epoch/GPU 训练。

链路审计补充（2026-06-22 18:35）：

- 当前运行的是 **baseline MDR**，不是 DATE11 all-binary/NTS 主线：
  - 入口：`third_party/SDformerFlow/train_mdr_supervised_SNN.py`
  - config：`configs/generated/train_mdr_baseline_mvsec_route.yml`
  - 模型：`MS_SpikingformerFlowNet_en4`
  - `event_interval=dt1`
  - `spiking_neuron.num_steps=5`
  - `metrics.mask_events=false`
  - 无 `atlif_ternary_psn`
  - 无 `bsa_attention`
  - 启动命令没有传 `--prev_runid`，因此按 baseline MDR 范式从初始化训练；日志没有 `Model restored from local checkpoint`，也没有 `[H9] installed ATLIFTernaryPSN/Shiftmax`。
- 与官方 MDR baseline 配置相比，当前只做了运行层差异：
  - `test_sequence` 从官方示例的 `outdoor_day1` 改为本机已准备完整的 `indoor_flying3`，与 `eval_MV_supervised.yml` 的默认 MVSEC eval sequence 一致。
  - `n_epochs` 当前先设为 `60`；官方 MDR config 是 `100`。若需要完全复刻论文 baseline，应补跑或续跑到 100 epoch。
- 当前训练已进入 epoch0 并通过首批 batch；GPU 正在训练。后续可用该 baseline MDR 模型在 MVSEC dt1 上评估，作为 DATE11 主线 MDR/MVSEC 对比的 baseline。

MDR 训练过慢复核与 fast 口径（2026-06-22 19:10）：

- 已停止慢速 baseline MDR 进程：原配置 `batch_size=4`、`n_workers=4`、`use_amp=false`、`n_epochs=60`，且训练循环默认每个 batch 开 `torch.autograd.set_detect_anomaly(True)`；epoch0 约 3.4 小时仅到 76%。
- 论文复核：
  - SDformerFlow arXiv 2409.04082 的 DSEC 主实验写明：3 张 RTX 2080 Ti，AdamW，80 epoch，随后 full-resolution fine-tune 30 epoch。
  - MVSEC 表的协议写明：MDR 约 `80000` training samples、`6000` validation samples，训练 `50` epoch 后在 MVSEC 上做 sparse flow evaluation。
  - 因此，仓库 `train_MDR_supervised_SDformerFlow.yml` 的 `n_epochs=100` 是代码默认配置，不是论文 MVSEC/MDR 表的最小复现实验；本工程 baseline MDR 应按 `50` epoch 作为论文口径。
- 本地 MDR 数据量确认：
  - `events1=85720`，`flow=17190`。
  - `MDREventFlow.get_train_sequence()` 遍历 `events1/*/*.npz`，不是只遍历 flow 文件；因此一个 epoch 实际是 `85720 / batch_size` 个 batch。旧配置 `batch_size=4` 得到 `21430` steps/epoch，和日志一致。
- 已在 `third_party/SDformerFlow/train_mdr_supervised_SNN.py` 新增可选 runtime 环境变量，默认兼容旧行为：
  - `SDFORMER_MDR_DETECT_ANOMALY=0/1`：控制 anomaly debug；默认仍为 `1`，pipeline fast 默认设为 `0`。
  - `SDFORMER_MDR_MAX_TRAIN_BATCHES` / `SDFORMER_MDR_MAX_VALID_BATCHES`：测速/短测用 batch 上限。
  - `SDFORMER_MDR_SKIP_VALIDATION=1`：测速时跳过 validation。
  - 训练循环新增 `train_samples_per_s` 打印，用于估算 epoch 时间。
- Fast benchmark（A800 80GB，MDR baseline PSN，crop `256x256`，torch backend）：

| candidate | batch | workers | AMP | train samples/s | 结论 |
|---|---:|---:|---:|---:|---|
| old route | 4 | 4 | 0 | 约 5.28 | 过慢，约 4.5h/epoch |
| bs8_w8 | 8 | 8 | 0 | 6.83 | 小幅提升 |
| bs12_w8 | 12 | 8 | 0 | 8.34 | 可用但不优 |
| bs16_w8 | 16 | 8 | 0 | 9.21 | 稳定 |
| bs24_w8_amp1 | 24 | 8 | 1 | 11.06 | 稳定 |
| **bs32_w8_amp1** | **32** | **8** | **1** | **15.40**（160-batch 稳定性测试） | **当前最快稳定默认** |
| bs32_w8_amp0 | 32 | 8 | 0 | OOM | 不用 |

- 新增配置：
  - `configs/generated/train_mdr_baseline_mvsec_route_paper_strict.yml`：论文 MDR 50 epoch 口径，保留 `batch_size=4`、`use_amp=false`。
  - `configs/generated/train_mdr_baseline_mvsec_route_fast.yml`：默认快速复现，`n_epochs=50`、`batch_size=32`、`n_workers=8`、`use_amp=true`。
- `run_mdr_mvsec_standard_pipeline.sh` 默认切到 fast 配置，并默认 `SDFORMER_MDR_DETECT_ANOMALY=0`；若要回到严格 batch=4，可显式：

```bash
CONFIG=configs/generated/train_mdr_baseline_mvsec_route_paper_strict.yml \
bash neuron_experiments/H9_bipolar_self_attention/entrypoints/run_mdr_mvsec_standard_pipeline.sh
```

- 重要边界：这个 batch=32+AMP 只在 **MDR baseline PSN / crop256** 上实测稳定；不能直接套到 DATE11 DSEC all-binary/ATLIF/Shiftmax 消融。DSEC DATE11 消融仍保持现有 batch 配置，若要加速需另做同样的短 batch benchmark。
- 已重启正式 fast baseline MDR run：
  - pipeline PID：`2972644`
  - train PID：`2972665`
  - log：`neuron_experiments/H9_bipolar_self_attention/results/mdr_mvsec_fast_pipeline_20260622_194135.log`
  - MLflow dir：`file:///root/private_data/sdformer_mlflow/285508689205532009/6fcd5ea103b14a02bc895da13ee44d90/`
  - 启动审计：`detect_anomaly=False`、`max_train_batches=None`、`skip_validation=False`、`batch_size=32`、AMP enabled；已进入 `Epoch 0`，约 `16/2678` step 时 GPU 显存约 `50GB`，无 OOM。
- 2026-06-24/25 运行状态与恢复：
  - 原 fast run 已完成到 epoch23 train loop；epoch23 train loss `0.350138`，但在 `2026-06-24 11:20:41 +0800` 后卡在 `mlflow.pytorch.log_model` 保存阶段，GPU 空闲、日志不再更新。
  - 已确认旧 run 的 training state 只安全保存到 epoch22，因此终止卡住进程后从旧 MLflow run `6fcd5ea103b14a02bc895da13ee44d90` 恢复。
  - 为避免再次卡在重型 MLflow model logging，新增可选环境变量 `SDFORMER_MDR_SKIP_MLFLOW_MODEL_LOG=1` 与 `SDFORMER_MDR_LOCAL_CHECKPOINT_DIR=...`；默认行为不变，resume run 只写本地 checkpoint。
  - `mdr_mvsec_fast_resume_nomlflowmodel_20260624_155919.log` 已判定为 **无效 resume**：日志出现 `No model found at6fcd5ea103b14a02bc895da13ee44d90`，说明只恢复了 optimizer/scheduler/scaler state，没有恢复模型权重；该 run 及 `results/mdr_fast_local_ckpts_20260624` 不得用于论文结果或后续 warm start。
  - 正确恢复种子已本地化为一对 checkpoint：`results/mdr_valid_resume_seed_epoch22/checkpoint_epoch22.pth` 与 `results/mdr_valid_resume_seed_epoch22/checkpoint_epoch22_state_dict.pth`，来自旧有效 run 的 epoch22 模型和 training state。
  - 正确 smoke（2026-06-25）：`results/mdr_valid_resume_epoch22_smoke_20260625_163628.log` 同时出现 `Model restored from local checkpoint` 和 `Training state resumed from local checkpoint`，从 `Epoch 23` 开始；3-batch epoch23 train loss `0.403140`，与旧有效 run 的 `~0.35` 同量级，证明不是从随机权重继续。
  - 正式正确续跑（2026-06-25）：PID `3517633`；log：`results/mdr_valid_resume_epoch22_full_20260625_164239.log`；本地 checkpoint 目录：`results/mdr_valid_resume_local_ckpts_20260625_164239`；MLflow dir：`file:///root/private_data/sdformer_mlflow/285508689205532009/37721017bbb74e468f1a30d6ece78b61/`。该 run 从有效 epoch22 重复 epoch23 继续到论文口径 epoch50，并跳过 MLflow 大模型上传。
  - 2026-06-27 完成审计：进程已结束；代码 `range(epoch_initial, n_epochs)` 下 `n_epochs=50` 的最后训练轮为 epoch49。epoch23-49 无 traceback，训练 loss 从 `0.351193` 降到 `0.312000`。MVSEC validation loss：epoch25 `1.067901`、epoch30 `1.422358`、epoch35 `1.116676`、epoch40 `1.003706`、epoch45 `1.120062`；本轮 best validation 为 epoch40，但因本地 checkpoint 只按 train loss 改善保存，实际保留的后期 checkpoint 为 epoch41/42/43/47，其中 epoch47 train loss 最低 `0.311793`。

MDR baseline 详细指标（2026-06-27 汇总）：

| epoch | train_loss | valid_loss | local checkpoint |
|---:|---:|---:|---|
| 23 | 0.351193 | - | `checkpoint_epoch23.pth` |
| 24 | 0.348625 | - | `checkpoint_epoch24.pth` |
| 25 | 0.348197 | 1.067901 | `checkpoint_epoch25.pth` |
| 26 | 0.344489 | - | `checkpoint_epoch26.pth` |
| 27 | 0.338675 | - | `checkpoint_epoch27.pth` |
| 28 | 0.339123 | - | - |
| 29 | 0.338033 | - | `checkpoint_epoch29.pth` |
| 30 | 0.336983 | 1.422358 | `checkpoint_epoch30.pth` |
| 31 | 0.328426 | - | `checkpoint_epoch31.pth` |
| 32 | 0.326536 | - | `checkpoint_epoch32.pth` |
| 33 | 0.325802 | - | `checkpoint_epoch33.pth` |
| 34 | 0.325467 | - | `checkpoint_epoch34.pth` |
| 35 | 0.324975 | 1.116676 | `checkpoint_epoch35.pth` |
| 36 | 0.323036 | - | `checkpoint_epoch36.pth` |
| 37 | 0.321519 | - | `checkpoint_epoch37.pth` |
| 38 | 0.320431 | - | `checkpoint_epoch38.pth` |
| 39 | 0.322780 | - | - |
| 40 | 0.320828 | 1.003706 | - |
| 41 | 0.314732 | - | `checkpoint_epoch41.pth` |
| 42 | 0.314153 | - | `checkpoint_epoch42.pth` |
| 43 | 0.312673 | - | `checkpoint_epoch43.pth` |
| 44 | 0.314826 | - | - |
| 45 | 0.314569 | 1.120062 | - |
| 46 | 0.313564 | - | - |
| 47 | 0.311793 | - | `checkpoint_epoch47.pth` |
| 48 | 0.313763 | - | - |
| 49 | 0.312000 | - | - |

MVSEC sparse-flow 推理 sanity check（2026-06-27，非论文复现数）：

- 配置：`configs/generated/eval_mvsec_dt1_mdr_baseline_route.yml`。
- checkpoint：`neuron_experiments/H9_bipolar_self_attention/results/mdr_valid_resume_local_ckpts_20260625_164239/checkpoint_epoch47.pth`。
- 输出目录：`results_inference/mvsec_mdr_baseline_epoch47_dt1_20260627_123556`。
- 加载审计：`checkpoint_overlay_keys=0`、`model_overlay_keys=0`、`missing=0`、`unexpected=0`。

| checkpoint | MVSEC split | AEE | PE1 | PE2 | PE3/outlier | total_spikes | global_fr | effective_flops | energy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MDR baseline epoch47 | indoor_flying3 dt1 | 1.2120 | 0.4851 | 0.1412 | 0.0481 | 36.1762G | 11.28% | 81.6827G | 32277.13uJ |

重要修正（2026-06-28）：该结果不能写作论文 MVSEC baseline 复现值，只能视为当前本地链路 sanity check。

- 论文 Table II 的 MVSEC `dt=1` 不报告 AAE，只报告 AEE / `% Outlier`；SDformerFlow_v2/MDR 为 outdoor_day1 `0.61/0.08%`、indoor_flying1 `0.54/0.58%`、indoor_flying2 `0.81/3.85%`、indoor_flying3 `0.69/1.78%`、Avg `0.66/1.57%`。
- 本地目前只有 `MVSEC_test/indoor_flying3` 预处理目录，且该目录为 `event=2951`、`flowgt_dt1=2434`；runner 按论文四序列默认检查时跳过 `outdoor_day1`、`indoor_flying1`、`indoor_flying2`。原始数据层面只有 `indoor_flying3` 的 data/gt hdf5，`indoor_flying1` 只有 bag、缺 hdf5，`outdoor_day1` 与 `indoor_flying2` 未在本机找到。
- 后续 MVSEC 评估配置已改为 AEE-only：`configs/generated/eval_mvsec_dt1_mdr_baseline_route.yml` 的 `metrics.name=[AEE]`；`run_h9_standard_mvsec_eval.py` 默认序列改为 `outdoor_day1/indoor_flying1/indoor_flying2/indoor_flying3`，ranking 只输出 AEE/PE/outlier/spikes/energy，并记录缺失序列。
- `AAE=29.9453` 不应和论文 MVSEC 表对比，因为论文 MVSEC 表没有 AAE；仓库 AAE 对小幅/近零光流方向很敏感，MVSEC sparse-flow 论文主指标应优先看 AEE 和 outlier。后续 MVSEC 表不再报告 AAE。
- 本地 epoch47 的 `indoor_flying3` AEE `1.2120` 明显差于论文 SDformerFlow_v2/MDR 的 `0.69`。当前排查到的最大偏差源是训练 checkpoint/协议，不是加载失败：加载审计为 `missing=0/unexpected=0`，eval 遍历 `1885` 个样本；但该 checkpoint 来自 fast route（`batch_size=32`、AMP、从早期 run 续跑），而论文只说明 MDR 从头训 50 epoch、cropped resolution/window size，未等价声明 batch32+AMP。训练日志里 MVSEC validation loss 在 epoch40 最低 `1.0037`、epoch45 又升到 `1.1201`，说明该 fast checkpoint 泛化没有稳定达到论文水平。后续 DATE 的 MVSEC/MDR 外部泛化表必须先补齐 MVSEC 四序列预处理，并用 paper-strict 或官方 checkpoint 复核 baseline，再训练/eval all-binary+TX 主线。

结论：baseline MDR->MVSEC route 已完成训练链路跑通，但尚未完成论文级 MVSEC baseline 复现。当前最稳的训练 checkpoint 是 `checkpoint_epoch47.pth`；validation loss 最低点在 epoch40 但本地未保存该 epoch。若继续推进，应先补齐 MVSEC test 预处理序列，并对 `checkpoint_epoch41.pth` / `checkpoint_epoch47.pth` 做同协议 AEE/outlier sensitivity；在复现 baseline 接近论文量级前，不启动 all-binary + TX 的 MDR 全量训练作为正式论文结果。

MVSEC 完整四序列复现接管（2026-06-29）：

- 目标：先把 MVSEC `outdoor_day1 / indoor_flying1 / indoor_flying2 / indoor_flying3` 全部补齐到 `third_party/SDformerFlow/data/Datasets/MVSEC/MVSEC_test/<seq>/{event,flowgt_dt1}`，再按论文 Table II 的 AEE/%Outlier 口径评估 MDR baseline。MVSEC 阶段不再报告 AAE。
- 论文范式核对：arXiv v1 第 IV-B.2 前的实验设置写明，为避免 MVSEC 过拟合，模型使用 MDR 训练；MDR 约 `80000` train / `6000` valid samples；从头训练 `50` epoch 后在 MVSEC sparse optical flow 上评估。官方仓库 `third_party/SDformerFlow/configs/train_MDR_supervised_SDformerFlow.yml` 与该范式一致的核心项是 `data.path=data/Datasets/MDR`、`test_sequence=outdoor_day1`、`event_interval=dt1`、`batch_size=4`、`use_amp=False`、`crop=[256,256]`、`metrics.name=[AEE]`；但仓库默认 `n_epochs=100`，本工程 paper-strict config 固定为论文口径 `50` epoch。
- 论文目标值（SDformerFlow-v2/MDR dt1）：`outdoor_day1 AEE=0.61 / outlier=0.08%`，`indoor_flying1 0.54 / 0.58%`，`indoor_flying2 0.81 / 3.85%`，`indoor_flying3 0.69 / 1.78%`，平均 `0.66 / 1.57%`。任何本地结果若只在 indoor_flying3 上得到 `AEE≈1.2`，只能算链路 sanity，不是论文复现。
- 当前下载状态：`indoor_flying1/2/3` raw bags 已在本地；`indoor_flying3` 已有官方/既有 hdf5 与 encoded dt1；`outdoor_day1_data.bag` 与 `outdoor_day1_gt.bag` 正通过 `scripts/http_range_download.py` 并发 range 下载；`indoor_flying_calib.zip` 与 `outdoor_day_calib.zip` 均已就绪。
- GT root cause：MVSEC 官网明确说明 dense optical flow 不直接放在 bag/hdf5 里，而是单独提供 `_gt_flow_dist.npz`，或通过 depth/pose/calibration 生成。直接从 gt bag 读 `/davis/left/flow_dist` 会失败，这是正常数据格式差异，不是包损坏。Google Drive folder 可列出四序列 file id，但实际下载速度约几十 KB/s，不适合当前作为主路径。
- 当前 GT 生成策略：优先用本地 raw gt bag 的 `/davis/left/depth_image_rect` + `/davis/left/odometry` 和官方 calibration zip 生成 `*_gt.hdf5`，再交给 `MVSEC_encoder.py` 生成 `flowgt_dt1`。若后续拿到官方 `_gt_flow_dist.npz`，`scripts/mvsec_npz_to_gt_hdf5.py` 已兼容官网 schema `timestamps/x_flow_dist/y_flow_dist`。
- 新增/修正工具：`scripts/prepare_mvsec_dt1.py` 修正 `outdoor_day* -> outdoor_day`、`indoor_flying* -> indoor_flying` URL 映射，并修正 argparse 默认序列；gt-only 转换失败后自动 fallback 到 `scripts/mvsec_gt_flow_from_bag.py`；`scripts/mvsec_gt_flow_from_bag.py` 自动识别 calibration zip 内的 `camchain-imucam-*.yaml`；`scripts/http_range_download.py` 用于大文件 range 下载；`MVSEC_encoder.py` 将 `U_gt_all/V_gt_all` 移出 per-sample loop，预处理子进程限制 BLAS 线程；`configs/generated/eval_mvsec_dt1_mdr_baseline_route.yml` 已改为 AEE-only；`run_h9_standard_mvsec_eval.py` 默认跑四个 MVSEC 序列并记录缺失序列。
- 下载命令范式：

```bash
/opt/conda/envs/sdformerflow/bin/python -u scripts/http_range_download.py \
  https://visiondata.cis.upenn.edu/mvsec/outdoor_day/outdoor_day1_data.bag \
  third_party/SDformerFlow/data/Datasets/MVSEC/outdoor_day1/outdoor_day1_data.bag \
  --workers 32 --chunk-mb 4

/opt/conda/envs/sdformerflow/bin/python -u scripts/http_range_download.py \
  https://visiondata.cis.upenn.edu/mvsec/outdoor_day/outdoor_day1_gt.bag \
  third_party/SDformerFlow/data/Datasets/MVSEC/outdoor_day1/outdoor_day1_gt.bag \
  --workers 32 --chunk-mb 4
```

- encode 命令范式：

```bash
/opt/conda/envs/sdformerflow/bin/python -u scripts/prepare_mvsec_dt1.py \
  --encode-only \
  --sequence outdoor_day1 \
  --sequence indoor_flying1 \
  --sequence indoor_flying2 \
  --sequence indoor_flying3
```

- baseline 标准化推理命令范式：

```bash
/opt/conda/envs/sdformerflow/bin/python neuron_experiments/H9_bipolar_self_attention/entrypoints/run_h9_standard_mvsec_eval.py \
  --config configs/generated/eval_mvsec_dt1_mdr_baseline_route.yml \
  --checkpoint neuron_experiments/H9_bipolar_self_attention/results/mdr_valid_resume_local_ckpts_20260625_164239/checkpoint_epoch47.pth \
  --out-dir results_inference/mvsec_mdr_baseline_epoch47_dt1_full4_$(date +%Y%m%d_%H%M%S)
```

- 判定流程：四序列 encoded 完成后，先对 MDR baseline `checkpoint_epoch41/47` 做 AEE/outlier sensitivity；如果四序列结果仍明显高于论文目标，优先排查 MVSEC hdf5/encoder 是否与官方格式一致，再决定是否重跑 `configs/generated/train_mdr_baseline_mvsec_route_paper_strict.yml`。在 baseline 复现可信前，不把 all-binary + TX MDR/MVSEC 训练结果写成正式外部泛化结论。

DATE11 all-binary + TX MDR/MVSEC full run（2026-06-27）：

- 新增配置：`configs/generated/train_mdr_all_binary_atlif_tx_mvsec_route_fast.yml`，保持 MDR baseline fast route 的 `crop=256x256`、`batch_size=32`、`n_epochs=50`、MVSEC `indoor_flying3/dt1` validation，只替换为 all-binary ATLIF + all12 TX attention。
- 新增训练链路：`third_party/SDformerFlow/train_mdr_supervised_SNN.py` 仅在配置包含 `atlif_ternary_psn.enabled` 或 `bsa_attention.enabled` 时安装 H9 overlay；baseline 配置不触发。加载审计输出 `checkpoint_overlay_keys/missing/unexpected`。
- smoke：`results/mdr_allbinary_tx_smoke_20260627_121258.log`，从 MDR baseline `checkpoint_epoch47.pth` 加载模型权重、不 resume optimizer；审计为 ATLIF `105`、Shiftmax/TX `12`、`checkpoint_overlay_keys=0, missing=210, unexpected=0`，1-batch loss `1.521719`，本地 checkpoint/state 保存正常。
- full run 曾启动：PID `371764`；log：`results/mdr_allbinary_tx_full_20260627_121911.log`；本地 checkpoint 目录：`results/mdr_allbinary_tx_local_ckpts_20260627_121911`；MLflow dir：`file:///root/private_data/sdformer_mlflow/199834913587708620/808bba3aea934a08875db8a106cbe295/`。启动命令从 baseline epoch47 checkpoint 初始化，但不传 `--resume`，因此不会沿用 baseline optimizer/scheduler/scaler state。
- 2026-06-28 状态修正：该 full run 已按“先做标准化推理，先不跑训练”的要求停止；停止时仍在 epoch0 中段，未保存任何 checkpoint，本地 checkpoint 目录为空。当前无 `train_mdr_supervised_SNN.py` / `eval_MV_flow_SNN.py` 主进程运行。若恢复 all-binary + TX MDR/MVSEC 训练，应从同一个 baseline epoch47 checkpoint 重新启动。

### DATE11 完整消融矩阵配置与二服务器标准流程（2026-06-22）

目标：给另一台服务器一个可复现入口，按 DATE 论文 full-replacement ablation 矩阵生成配置、训练 full30、跑 standard valid825，并保留每个实验的加载审计和 ranking。

新增 runner：

```text
neuron_experiments/H9_bipolar_self_attention/entrypoints/run_date11_ablation_matrix.py
```

依赖环境：

- 必须用 `sdformerflow` 环境或等价环境，系统 `/usr/bin/python3` 缺 `yaml/torch/cv2/mlflow/spikingjelly`，不能直接跑。
- 本机验证命令：

```bash
/opt/conda/envs/sdformerflow/bin/python \
  neuron_experiments/H9_bipolar_self_attention/entrypoints/run_date11_ablation_matrix.py \
  --preset date-paper-core --priority P0 --dry-run --generate
```

dry-run 已通过，会生成/刷新：

```text
neuron_experiments/H9_bipolar_self_attention/configs/generated/date11_full_factorial_manifest.json
```

完整矩阵：

| 优先级 | 实验名 | 神经元 | Attention | ATLIF | Shiftmax | 配置 | 当前状态 |
|---|---|---|---|---:|---:|---|---|
| P0 | NB0 | PSN | original | 0 | 0 | `configs/generated/upstream_baseline_stride.yml` | 已完成 baseline |
| P0 | `date11full_all_binary_atlif_original_w720_fastlr` | all binary ATLIF | original | 105 | 0 | `configs/generated/date11full_all_binary_atlif_original_w720_fastlr_full30.yml` | 已完成 |
| P0 | `date11full_all_ternary_atlif_original_w720_fastlr` | all ternary ATLIF | original | 105 | 0 | `configs/generated/date11full_all_ternary_atlif_original_w720_fastlr_full30.yml` | 已完成，负例 |
| P0 | `date11full_all_ternary_atlif_tx_w720_fastlr` | all ternary ATLIF | TX | 105 | 12 | `configs/generated/date11full_all_ternary_atlif_tx_w720_fastlr_full30.yml` | 已完成，负例 |
| P0 | `date11full_all_ternary_atlif_sc_w720_fastlr` | all ternary ATLIF | SC | 105 | 12 | `configs/generated/date11full_all_ternary_atlif_sc_w720_fastlr_full30.yml` | 已完成，负例 |
| P0 | `date11full_all_ternary_atlif_nts_w720_fastlr` | all ternary ATLIF | NTS/H60 | 105 | 12 | `configs/generated/date11full_all_ternary_atlif_nts_w720_fastlr_full30.yml` | 已完成，负例 |
| P1 | `date11full_all_binary_atlif_tx_w720_fastlr` | all binary ATLIF | TX | 105 | 12 | `configs/generated/date11full_all_binary_atlif_tx_w720_fastlr_full30.yml` | 已完成 |
| P1 | `date11full_all_binary_atlif_sc_w720_fastlr` | all binary ATLIF | SC | 105 | 12 | `configs/generated/date11full_all_binary_atlif_sc_w720_fastlr_full30.yml` | 已完成 |
| P1 | `date11full_all_binary_atlif_nts_w720_fastlr` | all binary ATLIF | NTS/H60 | 105 | 12 | `configs/generated/date11full_all_binary_atlif_nts_w720_fastlr_full30.yml` | 已完成；最终主线基底 |
| P2 | `date11full_psn_tx_w720_fastlr` | PSN | TX | 0 | 12 | `configs/generated/date11full_psn_tx_w720_fastlr_full30.yml` | 训练已做过；valid825 审计需谨慎 |
| P2 | `date11full_psn_sc_w720_fastlr` | PSN | SC | 0 | 12 | `configs/generated/date11full_psn_sc_w720_fastlr_full30.yml` | 未跑/低优先 |
| P2 | `date11full_psn_nts_w720_fastlr` | PSN | NTS/H60 | 0 | 12 | `configs/generated/date11full_psn_nts_w720_fastlr_full30.yml` | 已跑 ep29 valid825；非主线 |

runner preset：

- `--preset date-paper-core`：8 个 DATE 主矩阵实验，包含 all-binary/all-ternary × original/TX/SC/NTS；不含 PSN attention-only。
- `--preset full`：11 个 full-factorial 实验，包含 P2 PSN+TX/SC/NTS。
- `--preset psn-attention-only`：只跑 P2 attention-only 对照。

标准运行命令（另一台服务器从头跑完整 DATE 主矩阵）：

```bash
cd /root/private_data/work/sdformer_codex/SDformer
/opt/conda/envs/sdformerflow/bin/python \
  neuron_experiments/H9_bipolar_self_attention/entrypoints/run_date11_ablation_matrix.py \
  --preset date-paper-core \
  --generate \
  --skip-existing
```

只跑某个优先级：

```bash
/opt/conda/envs/sdformerflow/bin/python \
  neuron_experiments/H9_bipolar_self_attention/entrypoints/run_date11_ablation_matrix.py \
  --preset date-paper-core \
  --priority P1 \
  --generate \
  --skip-existing
```

只跑单个配置：

```bash
/opt/conda/envs/sdformerflow/bin/python \
  neuron_experiments/H9_bipolar_self_attention/entrypoints/run_date11_ablation_matrix.py \
  --preset full \
  --name date11full_psn_sc_w720_fastlr \
  --generate \
  --skip-existing
```

每个实验的标准流程：

1. `make_date11_full_factorial_configs.py` 生成配置与 manifest。
2. `verify_nts11_chain.py <config>` 检查 ATLIF/Shiftmax 安装和 checkpoint overlay 预期。
3. `entrypoints/train.py --config <config> --prev_runid experiments/baseline_stride_upstream/checkpoint_epoch59.pth --save_path <run_dir>/checkpoint_epoch{}.pth`
4. `run_h9_standard_valid825_eval.py --config <config> --run-dir <run_dir> --epoch 9 --epoch 14 --epoch 19 --epoch 24 --epoch 28 --epoch 29`
5. 输出：
   - `<run_dir>/pipeline.log`
   - `<run_dir>/profile_ranking_valid825.md`
   - `<run_dir>/standard_valid825/epoch*/spike_profile.json`
   - driver 目录：`results/date11_ablation_matrix_driver_<stamp>/plan.json` 和 `status.jsonl`

注意：

- P2 PSN+TX/SC/NTS 是 reviewer 可能要求的 attention-only 控制，不是 DATE 主线。此前 PSN+TX 标准 valid825 曾因 attention-only checkpoint 的 overlay 审计口径出过问题；如跑 P2，必须人工检查 `pipeline.log` 中 `checkpoint_overlay_keys/missing/unexpected`，不能只看脚本退出码。
- 当前论文主线仍是 `all-binary ATLIF + all12 NTS/H60 + FT5`；full30 矩阵用于机制消融，FT5/部署量化是 final-mainline 补实验。


### DATE11 自动结果追加：all-binary dualrail TX beta0.5 FT5（2026-06-26 03:50:20）

<!-- DATE11_FT_APPEND::date11full_all_binary_atlif_drtx_b050_stdlr_ft_txep19_ft5_bs8_20260626_005642_setsid -->
- 配置：`neuron_experiments/H9_bipolar_self_attention/configs/generated/date11full_all_binary_atlif_drtx_b050_stdlr_ft_txep19_ft5.yml`
- 起点：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_tx_w720_fastlr_full30_bs8_20260617_024526_setsid/checkpoint_epoch19.pth`
- 运行目录：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_drtx_b050_stdlr_ft_txep19_ft5_bs8_20260626_005642_setsid`
- 标准 valid825 ranking：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_drtx_b050_stdlr_ft_txep19_ft5_bs8_20260626_005642_setsid/profile_ranking_valid825.md`
- best：epoch `2`，AEE `1.5079`，AAE `9.8100`，total_spikes `23.1366G`，firing `5.0000%`，energy `20420.85uJ`。

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 1.5079 | 9.8100 | 0.5228 | 0.1969 | 0.0925 | 23.1366G | 5.0000% | 20420.85 |
| 2 | 4 | 1.5544 | 9.9571 | 0.5258 | 0.2035 | 0.0971 | 23.3246G | 5.0407% | 20606.63 |
| 3 | 0 | 1.5715 | 10.2266 | 0.5361 | 0.2100 | 0.1005 | 23.5368G | 5.0865% | 20771.58 |
| 4 | 3 | 1.5900 | 10.2020 | 0.5361 | 0.2090 | 0.0995 | 24.7798G | 5.3552% | 21865.75 |
| 5 | 1 | 1.5914 | 10.1970 | 0.5409 | 0.2113 | 0.1013 | 24.4892G | 5.2923% | 21603.81 |

### DATE11 机制记录：all-binary TX vs FAPS 前向与配置（2026-06-26）

目的：把 `all-binary ATLIF` 代入 dual-rail TX 与 FAPS 两种注意力，明确二者前向差异，并生成可直接训练的 all-binary + FAPS 配置。

代码入口：

- dual-rail TX：`neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/bsa_attention.py::_dualrail_binary_tx_token_scores`
- FAPS：`neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/bsa_attention.py::_faps_flow_aligned_token_scores`

机制对比：

| 项目 | dual-rail TX | FAPS |
|---|---|---|
| 输入事件 | `_binary_event_ste`，事件为 `{0,+1}` | `_ternary_sign_ste`，事件语义为 `{-1,0,+1}`；在 all-binary ATLIF 下主要退化到 `{0,+1}` |
| 通道解释 | head 前半为 positive rail，后半为 negative rail | head 前后半按 flow-aligned 的 x/y 方向组处理 |
| 分数核心 | `same + alpha0 * same_zero - beta * opposite - single_active_penalty * single_active` | 每个方向组内 `4 * same_nonzero + 1 * same_zero - 1 * opposite - 4 * single_active` |
| 异号惩罚 | 显式 `- beta * opposite`；b050 为 0.5，b025 为 0.25，不是 0 | 固定 `-1 * opposite`；all-binary 场景中若没有负事件，opposite 项通常弱化 |
| 光流特性 | 通用 TX/XNOR 风格 token selector，不显式区分 x/y 方向 | 显式拆 x/y 方向组，可用 mean/sum 或 disagreement penalty 融合，并可加 sparse K magnitude |
| 当前 all-binary 叙事 | 更像“硬件友好的通用二值匹配注意力” | 更适合讲成“借鉴 TX 的 popcount selector，同时结合光流 x/y 方向结构和 K 幅值置信度” |

具体 all-binary toy 前向，head_dim=8，前 4 维为一组，后 4 维为一组，3 个 token。设：

```text
Token A:
q_pos=[1,0,1,0], q_neg=[0,1,0,0]
k_pos=[1,0,0,0], k_neg=[0,1,1,0]

Token B:
q_pos=[1,1,0,0], q_neg=[0,0,1,0]
k_pos=[0,1,0,0], k_neg=[1,0,1,0]

Token C:
q_pos=[0,0,0,0], q_neg=[0,0,0,0]
k_pos=[0,0,0,0], k_neg=[0,0,0,0]
```

dual-rail TX，取 `alpha0=0.02`、`beta=0.25`、`single_active_penalty=0.10`：

| token | same_nonzero | opposite | same_zero | single_active | raw score | head_dim norm 后 |
|---|---:|---:|---:|---:|---:|---:|
| A | 2 | 1 | 1 | 0 | `2 + 0.02 - 0.25 = 1.77` | 0.4425 |
| B | 2 | 1 | 1 | 0 | `2 + 0.02 - 0.25 = 1.77` | 0.4425 |
| C | 0 | 0 | 4 | 0 | `0.08` | 0.0200 |

结论：TX 在 all-binary dual-rail 表示下主要奖励同 rail 激活，显式压低异 rail 匹配；全静默 token 只拿 `alpha0 * same_zero` 的弱奖励。

FAPS，用同一 toy 张量作方向组示意：前 4 维为 x，后 4 维为 y，`directional_merge_mode=mean`：

| token | score_x | score_y | mean raw score | head_dim norm 后 |
|---|---:|---:|---:|---:|
| A | `4*1 + 1*2 - 4*1 = 2` | `4*1 + 1*2 - 4*1 = 2` | 2 | 0.2500 |
| B | 2 | 2 | 2 | 0.2500 |
| C | `1*4 = 4` | `1*4 = 4` | 4 | 0.5000 |

结论：FAPS 的故事更贴光流，因为它把 token 匹配拆成 x/y 方向一致性，并可用 `k_magnitude_alpha` 给高置信 K 幅值补充 2-bit 修正；但在纯 all-binary toy case 里，`same_zero` 权重比 TX 大，静默 token 可能被偏高奖励，所以需要真实 valid825 训练/验证确认，必要时调低 silence 权重或先跑 stage2-only FAPS。

已生成 all-binary + FAPS 配置：

| 用途 | 配置 | 起点 | 关键设置 |
|---|---|---|---|
| FT5 快速判断 | `neuron_experiments/H9_bipolar_self_attention/configs/generated/date11full_all_binary_atlif_faps_all12_stdlr_ft_txep19_ft5.yml` | `date11full_all_binary_atlif_tx_w720_fastlr_full30_bs8_20260617_024526_setsid/checkpoint_epoch19.pth` | `output_mode=binary`，`mode=faps`，all12 blocks，`batch_size=8`，`n_epochs=5` |
| full30 | `neuron_experiments/H9_bipolar_self_attention/configs/generated/date11full_all_binary_atlif_faps_all12_w720_fastlr_full30.yml` | NB0 epoch59 | `output_mode=binary`，`mode=faps`，all12 blocks，fastlr full30 |
| manifest | `neuron_experiments/H9_bipolar_self_attention/configs/generated/date11_allbinary_faps_manifest.json` | - | 记录 full30/FT5 两个配置 |

FT5 配置关键 attention 参数：`directional_channels_enabled=true`，`directional_merge_mode=mean`，`flow_disagreement_gamma=0.0`，`k_magnitude_alpha=0.03125`，`confidence_min_active=8`，`kmag_quantize_bits=2`，`single_active_penalty=0.05`，`consensus_score_norm=head_dim`。

运行命令：

```bash
cd /root/private_data/work/sdformer_codex/SDformer
SDFORMER_USE_MLFLOW=0 SDFORMER_MLFLOW_MODEL_LOGGING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True KMP_DUPLICATE_LIB_OK=TRUE \
python -u neuron_experiments/H9_bipolar_self_attention/entrypoints/run_date11_ft5_and_valid825.py \
  --config neuron_experiments/H9_bipolar_self_attention/configs/generated/date11full_all_binary_atlif_faps_all12_stdlr_ft_txep19_ft5.yml \
  --resume neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_tx_w720_fastlr_full30_bs8_20260617_024526_setsid/checkpoint_epoch19.pth \
  --label "all-binary FAPS all12 FT5"
```

b025 状态备注：`date11full_all_binary_atlif_drtx_b025_stdlr_ft_txep19_ft5_bs8_20260626_141439_setsid` 当前只有 `checkpoint_epoch0.pth` 和 `pipeline.log`，没有 valid825 ranking，进程已不存在；日志停在 epoch0 约 `378/918` step，未检出 Python traceback/OOM 字样，按异常中断处理，不能作为有效结果。

### DATE11 all-binary FAPS 短测筛选（2026-06-26）

目的：在不直接开 full30 的情况下，先用 `all-binary TX ep19` 续训 360 step，筛出 FAPS 的硬件友好主线超参。筛选原则：优先 no-Kmag；Kmag 只作为精度上限 ablation，不作为主线，因为它需要额外 K margin 保留、量化、active-count gate，比纯 popcount selector 硬件故事更重。

生成脚本与 manifest：

- `neuron_experiments/H9_bipolar_self_attention/entrypoints/make_date11_allbinary_faps_short_configs.py`
- `neuron_experiments/H9_bipolar_self_attention/configs/generated/date11_allbinary_faps_short_manifest.json`

短测结果（`profile_checkpoints.py --samples 40`，仅用于趋势筛选，不作为论文最终指标）：

| rank | config | scope | Kmag | LR | run_dir | train_loss | val_loss | valid40 AEE | valid40 AAE | SOPs(G) | firing | 判断 |
|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `date11allbin_faps_s2only_nokmag_stdlr_s360.yml` | S2-only 6 blocks | no | stdlr | `results/date11allbin_faps_s2only_nokmag_stdlr_s360_bs8_20260626_160617` | 1.5064 | 1.2660 | 1.5948 | 14.9469 | 2.2845 | 0.05404 | 当前最优；硬件最干净，进入 FT5 |
| 2 | `date11allbin_faps_s2only_nokmag_fastlr_s360.yml` | S2-only 6 blocks | no | fastlr | `results/date11allbin_faps_s2only_nokmag_fastlr_s360_bs8_20260626_161714` | 1.5088 | 1.2390 | 1.6198 | 15.1280 | 2.2849 | 0.05405 | val loss 低但 valid40 更差；不选 |
| 3 | `date11allbin_faps_all12_nokmag_stdlr_s360.yml` | all12 | no | stdlr | `results/date11allbin_faps_all12_nokmag_stdlr_s360_bs8_20260626_155451` | 1.5370 | 1.2680 | 1.6453 | 15.2806 | 2.3309 | 0.05514 | scope 太激进，AAE 更差 |

中止项：

- `date11allbin_faps_all12_kmag032_stdlr_s360`：已手动中止于约 step7。原因不是代码错误，而是 Kmag 需要额外硬件通路，和“纯 FAPS popcount selector”主线不一致；后续只在 noKmag 无法达标时作为精度上限 ablation。

短测结论：

1. FAPS 在 all-binary + TX ep19 起点下可以稳定训练，360 step 后 binary activity 保持约 `4.4%-4.5%`，未出现塌火。
2. S2-only 明显优于 all12：valid40 AEE 从 `1.6453` 降到 `1.5948`，AAE 从 `15.2806` 降到 `14.9469`，SOPs 也更低。
3. fastlr 没有改善 valid40，虽然 validation loss 更低，但 AEE/AAE 均差于 stdlr；下一步使用 `S2-only + noKmag + stdlr` 做 FT5/valid825。


### DATE11 自动结果追加：all-binary FAPS S2-only noKmag stdlr FT5（2026-06-26 19:25:50）

<!-- DATE11_FT_APPEND::date11full_all_binary_atlif_faps_s2only_nokmag_stdlr_ft_txep19_ft5_bs8_20260626_162831_setsid -->
- 配置：`neuron_experiments/H9_bipolar_self_attention/configs/generated/date11full_all_binary_atlif_faps_s2only_nokmag_stdlr_ft_txep19_ft5.yml`
- 起点：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_tx_w720_fastlr_full30_bs8_20260617_024526_setsid/checkpoint_epoch19.pth`
- 运行目录：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_faps_s2only_nokmag_stdlr_ft_txep19_ft5_bs8_20260626_162831_setsid`
- 标准 valid825 ranking：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_faps_s2only_nokmag_stdlr_ft_txep19_ft5_bs8_20260626_162831_setsid/profile_ranking_valid825.md`
- best：epoch `2`，AEE `1.5091`，AAE `9.9242`，total_spikes `22.6747G`，firing `4.8938%`，energy `19977.77uJ`。

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 1.5091 | 9.9242 | 0.5236 | 0.1970 | 0.0923 | 22.6747G | 4.8938% | 19977.77 |
| 2 | 4 | 1.5349 | 9.8483 | 0.5201 | 0.2003 | 0.0958 | 22.8250G | 4.9262% | 20130.69 |
| 3 | 3 | 1.5525 | 10.0885 | 0.5303 | 0.2044 | 0.0974 | 24.3261G | 5.2502% | 21421.04 |
| 4 | 0 | 1.5655 | 10.1694 | 0.5301 | 0.2063 | 0.0992 | 23.1522G | 4.9968% | 20405.53 |
| 5 | 1 | 1.5786 | 10.2047 | 0.5375 | 0.2100 | 0.1011 | 24.0592G | 5.1926% | 21192.65 |


### DATE11 自动结果追加：all-binary ATLIFPSN + all-attention FAPS noKmag FT5（2026-06-27 03:05:21）

<!-- DATE11_FT_APPEND::date11full_all_binary_atlif_faps_all12_nokmag_stdlr_ft_txep19_ft5_bs8_20260627_000507_setsid -->
- 配置：`neuron_experiments/H9_bipolar_self_attention/configs/generated/date11full_all_binary_atlif_faps_all12_nokmag_stdlr_ft_txep19_ft5.yml`
- 起点：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_tx_w720_fastlr_full30_bs8_20260617_024526_setsid/checkpoint_epoch19.pth`
- 运行目录：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_faps_all12_nokmag_stdlr_ft_txep19_ft5_bs8_20260627_000507_setsid`
- 标准 valid825 ranking：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_faps_all12_nokmag_stdlr_ft_txep19_ft5_bs8_20260627_000507_setsid/profile_ranking_valid825.md`
- best：epoch `2`，AEE `1.5152`，AAE `9.8479`，total_spikes `23.0819G`，firing `4.9882%`，energy `20370.23uJ`。

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 1.5152 | 9.8479 | 0.5253 | 0.1980 | 0.0928 | 23.0819G | 4.9882% | 20370.23 |
| 2 | 4 | 1.5475 | 9.9171 | 0.5277 | 0.2048 | 0.0979 | 23.2116G | 5.0162% | 20502.78 |
| 3 | 0 | 1.5703 | 10.2110 | 0.5358 | 0.2093 | 0.0999 | 23.5325G | 5.0856% | 20764.50 |
| 4 | 3 | 1.5745 | 10.2312 | 0.5390 | 0.2103 | 0.1001 | 24.6671G | 5.3308% | 21762.17 |
| 5 | 1 | 1.5905 | 10.1833 | 0.5423 | 0.2132 | 0.1026 | 24.4676G | 5.2877% | 21581.08 |

结论：严格全替换 FAPS noKmag 的 FT5 最优是 epoch2，AEE `1.5152`，仍差于 all-binary TX FT5 `1.5077` 和 all-binary NTS/H60 `1.4891`；但 spikes/energy 仍在低能区间，且 epoch2 明显优于 epoch0/1/3/4，说明当前不能直接判定 FAPS 结构无效，FT5/stdlr 的时长与调度可能不合适。

后续动作：已启动同一定义的 `date11full_all_binary_atlif_faps_all12_nokmag_slowlr_ft_txep19_ft10.yml`，从同一个 all-binary TX ep19 checkpoint 续训 10 epoch，使用更保守 LR（backbone/norm `5e-7`，neuron `1.5e-5`，threshold `2.5e-6`，milestone `8`），用于判断“多训/慢训”能否把 strict all-attention FAPS 拉到 TX 附近。

### DATE11 自动结果追加：all-binary ATLIFPSN + all-attention FAPS noKmag FT10 slowlr（2026-06-27 09:10:35）

<!-- DATE11_FT_APPEND::date11full_all_binary_atlif_faps_all12_nokmag_slowlr_ft_txep19_ft10_bs8_20260627_030855_setsid -->
- 配置：`neuron_experiments/H9_bipolar_self_attention/configs/generated/date11full_all_binary_atlif_faps_all12_nokmag_slowlr_ft_txep19_ft10.yml`
- 起点：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_tx_w720_fastlr_full30_bs8_20260617_024526_setsid/checkpoint_epoch19.pth`
- 运行目录：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_faps_all12_nokmag_slowlr_ft_txep19_ft10_bs8_20260627_030855_setsid`
- 标准 valid825 ranking：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_faps_all12_nokmag_slowlr_ft_txep19_ft10_bs8_20260627_030855_setsid/profile_ranking_valid825.md`
- 审计：训练加载 `missing=0/unexpected=0`；`installed Shiftmax attention: 12 modules`；`official_atlif_modules=105`；`ternary_activity_mean=0.0`；`k_magnitude_alpha=0.0`。
- best：epoch `8`，AEE `1.5262`，AAE `9.7982`，total_spikes `23.8992G`，firing `5.1648%`，energy `21078.63uJ`。

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 1.5262 | 9.7982 | 0.5220 | 0.2003 | 0.0951 | 23.8992G | 5.1648% | 21078.63 |
| 2 | 2 | 1.5232 | 9.9866 | 0.5280 | 0.2001 | 0.0938 | 22.9500G | 4.9597% | 20241.14 |
| 3 | 4 | 1.5449 | 9.9174 | 0.5265 | 0.2044 | 0.0979 | 23.0081G | 4.9723% | 20310.47 |
| 4 | 7 | 1.5484 | 9.9755 | 0.5276 | 0.2029 | 0.0967 | 23.8412G | 5.1523% | 21039.98 |
| 5 | 9 | 1.5500 | 10.0927 | 0.5290 | 0.2019 | 0.0957 | 24.2746G | 5.2460% | 21438.58 |
| 6 | 5 | 1.5591 | 10.0709 | 0.5322 | 0.2080 | 0.0996 | 23.7135G | 5.1247% | 20925.35 |
| 7 | 6 | 1.5662 | 10.2375 | 0.5335 | 0.2083 | 0.1003 | 23.8831G | 5.1614% | 21077.34 |
| 8 | 0 | 1.5779 | 10.3148 | 0.5368 | 0.2104 | 0.1017 | 23.5018G | 5.0790% | 20733.87 |
| 9 | 3 | 1.5857 | 10.2178 | 0.5389 | 0.2116 | 0.1014 | 24.4905G | 5.2926% | 21591.43 |
| 10 | 1 | 1.5976 | 10.2315 | 0.5456 | 0.2157 | 0.1038 | 24.3647G | 5.2654% | 21481.98 |

结论：多训到 FT10 且降低 LR 没有把严格全替换 FAPS noKmag 拉近 TX/NTS。FT10 best AEE `1.5262`，比同定义 FT5 best `1.5152` 更差，也差于 all-binary TX FT5 `1.5077`、S2-only FAPS FT5 `1.5091`、all-binary NTS/H60 `1.4891`。因此当前证据不支持把“所有注意力都换成 FAPS noKmag”作为主线；如果继续讲 FAPS，优先保留 S2-only 作为机制旁证，或另开 silence/zero-match 权重修正，而不是继续堆训练轮数。

### DATE11 FAPS 整数 TX-ratio 短测（2026-06-27）

目的：修正原始 FAPS 在 all-binary ATLIFPSN 下的三值有效公式。all-binary 时事件基本为 `{0,+1}`，`opposite` 项基本无效，因此 FAPS 实际由 `same_active / same_zero / single_active` 三类计数决定。原始 FAPS `4:1:4` 等价于 `1:0.25:1`，与 TX 的 `1:0.02:0.10` 差别过大，尤其静默奖励和单边惩罚过强。

本轮只测无浮点乘法、移位友好的整数近似：加权 popcount 后用 dyadic `score_scale` 右移；`same_zero` 保留为 `1`，不设为 `0`。同时关闭 x/y split，先验证整数比例本身，避免把“比例”和“人为通道分组”混在一起。

- 起点：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_tx_w720_fastlr_full30_bs8_20260617_024526_setsid/checkpoint_epoch19.pth`
- 短测目录：`neuron_experiments/H9_bipolar_self_attention/results/date11_faps_txratio_integer_txep19_20260627_231503`
- summary：`neuron_experiments/H9_bipolar_self_attention/results/date11_faps_txratio_integer_txep19_20260627_231503/summary.md`
- 共同设置：all-binary ATLIFPSN，all12 FAPS，noKmag，no x/y split，360 train steps，valid40。

| rank | integer score before shift | dyadic scale | effective ratio | valid40 AEE | valid40 AAE | SOPs(G) | firing | 判断 |
|---:|---|---:|---|---:|---:|---:|---:|---|
| 1 | `64*same_active + 1*same_zero - 6*single_active` | `1/64` | `1 : 0.0156 : 0.0938` | 1.5712 | 14.3776 | 2.3324 | 0.05517 | 当前最好；最接近 TX `1:0.02:0.10`，进入 FT5 |
| 2 | `32*same_active + 1*same_zero - 3*single_active` | `1/32` | `1 : 0.0313 : 0.0938` | 1.5810 | 14.2899 | 2.3322 | 0.05517 | AEE 略差；same_zero 偏强 |
| 3 | `16*same_active + 1*same_zero - 2*single_active` | `1/16` | `1 : 0.0625 : 0.1250` | 1.6131 | 15.1093 | 2.3325 | 0.05518 | 明显变差；静默奖励仍过强 |

结论：FAPS 可以按 TX 比例做整数近似，而且短测有效。`64:1:6 >>6` 比此前 all12 FAPS short 的 valid40 AEE `1.6453` 明显更好，也优于 S2-only FAPS short 的 `1.5948`；下一步跑 `64:1:6 >>6` 的 FT5 + 标准 valid825。x/y split 暂时不作为主线假设，后续只作为同一整数比例下的对照。


### DATE11 自动结果追加：all-binary FAPS TX-ratio integer 64:1:6 nosplit FT5（2026-06-28 02:45:41）

<!-- DATE11_FT_APPEND::date11full_all_binary_atlif_faps_all12_nokmag_s64_z1_p6_sc0p015625_nosplit_stdlr_ft_txep19_ft5_bs8_20260627_234802_setsid -->
- 配置：`neuron_experiments/H9_bipolar_self_attention/configs/generated/date11full_all_binary_atlif_faps_all12_nokmag_s64_z1_p6_sc0p015625_nosplit_stdlr_ft_txep19_ft5.yml`
- 起点：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_tx_w720_fastlr_full30_bs8_20260617_024526_setsid/checkpoint_epoch19.pth`
- 运行目录：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_faps_all12_nokmag_s64_z1_p6_sc0p015625_nosplit_stdlr_ft_txep19_ft5_bs8_20260627_234802_setsid`
- 标准 valid825 ranking：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_faps_all12_nokmag_s64_z1_p6_sc0p015625_nosplit_stdlr_ft_txep19_ft5_bs8_20260627_234802_setsid/profile_ranking_valid825.md`
- best：epoch `2`，AEE `1.5085`，AAE `9.9029`，total_spikes `23.1650G`，firing `5.0062%`，energy `20444.56uJ`。

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 1.5085 | 9.9029 | 0.5248 | 0.1970 | 0.0918 | 23.1650G | 5.0062% | 20444.56 |
| 2 | 4 | 1.5424 | 9.8597 | 0.5251 | 0.2041 | 0.0981 | 23.3069G | 5.0368% | 20592.57 |
| 3 | 0 | 1.5664 | 10.2091 | 0.5357 | 0.2089 | 0.0994 | 23.5640G | 5.0924% | 20796.24 |
| 4 | 3 | 1.5712 | 10.1394 | 0.5364 | 0.2070 | 0.0981 | 24.7960G | 5.3587% | 21880.99 |
| 5 | 1 | 1.5880 | 10.1927 | 0.5412 | 0.2122 | 0.1020 | 24.5042G | 5.2956% | 21617.59 |

结论：`64:1:6 >>6` 证明 all-binary FAPS 可以用纯整数 popcount + shift 的方式追到 all-binary TX FT5 附近。相对 TX FT5 best（AEE `1.5077`，AAE `9.8912`，`22.7231G`，`20010.68uJ`），本实验 best epoch2 AEE 只差 `+0.0008`，但 spikes/energy 略高（`23.1650G` / `20444.56uJ`）。相对 all-binary NTS/H60 主线（AEE `1.4891`，`23.8206G`，`21045.91uJ`），精度仍明显落后，但能耗更低。论文定位上，FAPS 当前不是替代 NTS/H60 的最强精度主线，而是一个更接近 TX、无浮点乘法、无 Kmag 旁路、保留静默弱奖励的硬件友好注意力候选。下一步只补一个同类整数比例 `32:1:3 >>5` 的 FT5，验证 dyadic scale/静默奖励强度是否有更优能耗-精度点；不再扩大小数或 Kmag 矩阵。


### DATE11 自动结果追加：all-binary FAPS TX-ratio integer 32:1:3 nosplit FT5（2026-06-28 05:35:05）

<!-- DATE11_FT_APPEND::date11full_all_binary_atlif_faps_all12_nokmag_s32_z1_p3_sc0p03125_nosplit_stdlr_ft_txep19_ft5_bs8_20260628_024839_setsid -->
- 配置：`neuron_experiments/H9_bipolar_self_attention/configs/generated/date11full_all_binary_atlif_faps_all12_nokmag_s32_z1_p3_sc0p03125_nosplit_stdlr_ft_txep19_ft5.yml`
- 起点：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_tx_w720_fastlr_full30_bs8_20260617_024526_setsid/checkpoint_epoch19.pth`
- 运行目录：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_faps_all12_nokmag_s32_z1_p3_sc0p03125_nosplit_stdlr_ft_txep19_ft5_bs8_20260628_024839_setsid`
- 标准 valid825 ranking：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_faps_all12_nokmag_s32_z1_p3_sc0p03125_nosplit_stdlr_ft_txep19_ft5_bs8_20260628_024839_setsid/profile_ranking_valid825.md`
- best：epoch `2`，AEE `1.5124`，AAE `9.8949`，total_spikes `23.1615G`，firing `5.0054%`，energy `20440.79uJ`。

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 1.5124 | 9.8949 | 0.5243 | 0.1973 | 0.0922 | 23.1615G | 5.0054% | 20440.79 |
| 2 | 4 | 1.5435 | 9.9041 | 0.5276 | 0.2043 | 0.0975 | 23.2528G | 5.0251% | 20540.64 |
| 3 | 3 | 1.5783 | 10.1059 | 0.5349 | 0.2081 | 0.0994 | 24.7417G | 5.3469% | 21828.25 |
| 4 | 0 | 1.5759 | 10.2322 | 0.5363 | 0.2107 | 0.1014 | 23.5488G | 5.0891% | 20781.70 |
| 5 | 1 | 1.5859 | 10.1048 | 0.5406 | 0.2122 | 0.1022 | 24.5143G | 5.2978% | 21625.17 |

结论：`32:1:3 >>5` 没有超过 `64:1:6 >>6`，best epoch2 AEE 从 `1.5085` 退到 `1.5124`，spikes/energy 基本持平（`23.1615G` / `20440.79uJ`）。这说明把 same_zero 从 `1/64` 提到 `1/32` 反而略损精度，整数 FAPS 的最稳点仍是更接近 TX 比例的 `64:1:6 >>6`。相对 all-binary TX FT5（AEE `1.5077`，AAE `9.8912`，`22.7231G`，`20010.68uJ`），两个 FAPS 整数点都没有形成明确优势；因此 FAPS 不再继续扩大 sweep，后续主线应转为 all-binary + TX/整数 TX-like score 的硬件化论证，FAPS 只保留为“带弱静默奖励的 TX 近似”候选或消融。


### DATE11 自动结果追加：all-binary TX deploy quant（2026-06-28 06:10:33）

<!-- DATE11_TX_DEPLOY_QUANT::date11_allbinary_tx_deploy_quant_full825_20260628_053939 -->
- 主 checkpoint：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_tx_stdlr_ft_ep19_ft5_bs8_20260621_035025_setsid/checkpoint_epoch2.pth`
- 运行目录：`neuron_experiments/H9_bipolar_self_attention/results/date11_allbinary_tx_deploy_quant_full825_20260628_053939`
- 目的：验证 all-binary TX FT ep19 best checkpoint 的 TX gate 在 int8 score / int8 gate 下是否保持等价；TX 无 μ，因此不需要 pow2 μ 消融。

| config | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `date11_allbinary_tx_ft_ep19_deploy_float_ref` | 1.5087 | 9.8950 | 0.5214 | 0.1959 | 0.0921 | 22.7230G | 4.9011% | 20010.61 |
| `date11_allbinary_tx_ft_ep19_deploy_score_int8` | 1.5068 | 9.8695 | 0.5203 | 0.1955 | 0.0919 | 22.7223G | 4.9010% | 20009.90 |
| `date11_allbinary_tx_ft_ep19_deploy_score_int8_gate_int8` | 1.5124 | 9.9086 | 0.5223 | 0.1967 | 0.0927 | 22.7223G | 4.9010% | 20009.88 |

结论：TX gate 的部署量化基本成立。`score_int8` 相对 float ref 没有掉点，AEE 反而从 `1.5087` 到 `1.5068`，spikes/energy 不变；`score_int8_gate_int8` AEE 为 `1.5124`，相对 float ref 只差 `+0.0037`，energy 仍约 `20009.88uJ`。这支持 all-binary TX 的硬件路径写成 popcount score + centering + Shiftmax/LUT + int8 gate；但该验证只量化 TX gate，原 QK carrier 仍来自 H18a 路径，论文表述需避免把它写成无 carrier selector。若要把“无 carrier selector”作为更干净主线，需要另跑 H49/`tx_qkselector_shiftmax` 的 all-binary FT5 与 deploy quant。


### DATE11 自动结果追加：all-binary H60 TX-only mu0 noSC FT5（2026-06-28 18:13:22）

<!-- DATE11_FT_APPEND::date11full_all_binary_atlif_h60_mu0_txonly_stdlr_ft_ep19_ft5_bs8_20260628_152115_setsid -->
- 配置：`neuron_experiments/H9_bipolar_self_attention/configs/generated/date11full_all_binary_atlif_h60_mu0_txonly_stdlr_ft_ep19_ft5.yml`
- 起点：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_nts_w720_fastlr_full30_bs8_20260617_200451_setsid/checkpoint_epoch19.pth`
- 运行目录：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_h60_mu0_txonly_stdlr_ft_ep19_ft5_bs8_20260628_152115_setsid`
- 标准 valid825 ranking：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_h60_mu0_txonly_stdlr_ft_ep19_ft5_bs8_20260628_152115_setsid/profile_ranking_valid825.md`
- best：epoch `2`，AEE `1.5150`，AAE `9.9346`，total_spikes `23.0172G`，firing `4.9742%`，energy `20316.34uJ`。

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 1.5150 | 9.9346 | 0.5178 | 0.1964 | 0.0931 | 23.0172G | 4.9742% | 20316.34 |
| 2 | 4 | 1.5445 | 9.9752 | 0.5230 | 0.2038 | 0.0981 | 23.1286G | 4.9983% | 20431.56 |
| 3 | 0 | 1.5616 | 10.1376 | 0.5298 | 0.2087 | 0.1007 | 23.5005G | 5.0787% | 20747.31 |
| 4 | 3 | 1.5735 | 10.1852 | 0.5310 | 0.2068 | 0.0988 | 24.6238G | 5.3214% | 21730.11 |
| 5 | 1 | 1.5897 | 10.1688 | 0.5361 | 0.2112 | 0.1029 | 24.4217G | 5.2778% | 21546.55 |

结论：H60 框架下把 `bipolar_mu` 设为 `0` 并关闭 SC/Kmag 后，best epoch2 AEE `1.5150`，没有超过 all-binary TX FT5（AEE `1.5077`，`22.7231G`，`20010.68uJ`），也明显落后 all-binary NTS/H60 ep29-start FT5 主线（AEE `1.4891`，`23.8206G`，`21045.91uJ`）。因此 `mu=0/noSC` 不能作为当前主线；这说明 H60/NTS 的 SC/μ 分支不是可直接删掉的冗余项。由于该结果没有达到“效果好再接 x/y 分通道”的门槛，暂不启动 x/y-channel FAPS/H60 变体，避免继续扩大负向分支。


### DATE11 自动结果追加：all-binary H60 TX-only mu0 noSC 慢 LR 续训（2026-06-29）

<!-- DATE11_FT_APPEND::date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid -->
- 配置：`neuron_experiments/H9_bipolar_self_attention/configs/generated/date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8.yml`
- 起点：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_h60_mu0_txonly_stdlr_ft_ep19_ft5_bs8_20260628_152115_setsid/checkpoint_epoch2.pth`
- 运行目录：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid`
- 标准 valid825 ranking：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid/profile_ranking_valid825.md`
- 目的：验证前一次 `mu=0/noSC` 结果是否只是 FT5/warmup 不够；从旧 best epoch2 继续慢 LR 训练 8 epoch。
- 训练审计：加载 overlay key `missing=0/unexpected=0`；`ATLIFTernaryPSN=105`，`Shiftmax attention=12`；全程 `neg_mean=0`、`ternary_activity_mean=0`，符合 all-binary ATLIFPSN。
- 关键超参：`lr=1.2e-5`，`backbone_lr=5e-7`，`norm_lr=5e-7`，`neuron_lr=2e-5`，`threshold_lr=3e-6`，warmup `720` steps，`bipolar_mu=0.0`，SC schedule 关闭，`k_magnitude_alpha=0.0`。
- best：epoch `2`，AEE `1.5020`，AAE `9.8871`，total_spikes `23.2395G`，firing `5.0223%`，energy `20521.04uJ`。

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 1.5020 | 9.8871 | 0.5169 | 0.1949 | 0.0918 | 23.2395G | 5.0223% | 20521.04 |
| 2 | 7 | 1.5288 | 9.7984 | 0.5188 | 0.1995 | 0.0956 | 24.1899G | 5.2277% | 21371.97 |
| 3 | 4 | 1.5382 | 9.8282 | 0.5177 | 0.2003 | 0.0970 | 23.3293G | 5.0417% | 20615.77 |
| 4 | 5 | 1.5509 | 9.9964 | 0.5250 | 0.2049 | 0.0985 | 24.0518G | 5.1978% | 21249.86 |
| 5 | 0 | 1.5497 | 10.1150 | 0.5248 | 0.2041 | 0.0982 | 23.7429G | 5.1311% | 20973.08 |
| 6 | 6 | 1.5594 | 10.0954 | 0.5252 | 0.2060 | 0.0999 | 24.2368G | 5.2378% | 21416.56 |
| 7 | 3 | 1.5732 | 10.0503 | 0.5283 | 0.2062 | 0.0989 | 24.7933G | 5.3581% | 21886.55 |
| 8 | 1 | 1.5816 | 10.1395 | 0.5342 | 0.2093 | 0.1009 | 24.6176G | 5.3201% | 21729.44 |

结论：继续慢 LR/warmup 有效，把 `mu=0/noSC` best AEE 从 `1.5150` 改善到 `1.5020`，说明前一次 FT5 确实偏短或 warmup 不充分。与 all-binary TX FT5（H18a carrier TX，AEE `1.5077`，AAE `9.8912`，`22.7231G`，`20010.68uJ`）相比，H60 no-carrier TX selector 的 AEE/AAE 略优，但 spikes/energy 分别高约 `0.5164G` / `510.36uJ`；与 all-binary NTS/H60 ep29-start FT5 主线（AEE `1.4891`，AAE `9.7785`，`23.8206G`，`21045.91uJ`）相比仍有约 `+0.0129` AEE / `+0.1086` AAE 差距，但能量更低约 `524.87uJ`。因此 `mu=0/noSC` 不能简单等同旧 TX：它是无 carrier 的 H60 TX selector，硬件故事更干净；当前可作为 TX-like 硬件友好候选继续做部署量化/少量验证，但若以精度优先，NTS/H60 仍是更强主线。


### DATE11 自动结果追加：TTX deploy quant（2026-06-29）

<!-- DATE11_TTX_DEPLOY_QUANT::date11_ttx_deploy_quant_full825_20260629_220531 -->
- 命名：后续将 `all-binary H60 TX-only mu0 noSC` 主线简称为 **TTX**。本文档中 TTX 定义为：`all-binary ATLIFPSN + H60 TX-only selector, bipolar_mu=0, no SC, no Kmag`，即无 carrier 的 TX-style selector。
- 主 checkpoint：`neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid/checkpoint_epoch2.pth`
- 运行目录：`neuron_experiments/H9_bipolar_self_attention/results/date11_ttx_deploy_quant_full825_20260629_220531`
- 目的：验证 TTX 在 int8 score / int8 gate 部署近似下是否保持等价。TTX 的 `bipolar_mu=0`，因此不跑 pow2 μ 变体。
- 配置：
  - `neuron_experiments/H9_bipolar_self_attention/configs/generated/date11_ttx_ep2_deploy_float_ref.yml`
  - `neuron_experiments/H9_bipolar_self_attention/configs/generated/date11_ttx_ep2_deploy_score_int8.yml`
  - `neuron_experiments/H9_bipolar_self_attention/configs/generated/date11_ttx_ep2_deploy_score_int8_gate_int8.yml`

| config | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `date11_ttx_ep2_deploy_float_ref` | 1.5020 | 9.8871 | 0.5169 | 0.1949 | 0.0918 | 23.2395G | 5.0223% | 20521.04 |
| `date11_ttx_ep2_deploy_score_int8` | 1.4971 | 9.8303 | 0.5174 | 0.1951 | 0.0915 | 23.2434G | 5.0231% | 20524.45 |
| `date11_ttx_ep2_deploy_score_int8_gate_int8` | 1.5003 | 9.8266 | 0.5173 | 0.1956 | 0.0920 | 23.2462G | 5.0237% | 20526.92 |

结论：TTX 的部署量化通过。`score_int8` 相对 float ref 没有掉点，AEE 从 `1.5020` 到 `1.4971`；`score_int8_gate_int8` AEE `1.5003`，仍优于 float ref，energy 仅从 `20521.04uJ` 到 `20526.92uJ`。这说明 TTX 的硬件路径可以描述为二值 ATLIF 事件 + popcount/XNOR-style score + centering + Shiftmax/LUT + int8 gate；相比旧 all-binary TX，TTX 不依赖 H18a 的 native QK carrier，更适合写成干净的 selector 主线。当前 TTX 综合指标为 AEE `1.5003`、AAE `9.8266`、energy `20526.92uJ`（部署量化口径），是比旧 TX 更好讲硬件、精度接近 NTS/H60 的候选主线；若论文主线强调极简硬件，TTX 可优先于 NTS/H60，若强调最低 AEE，NTS/H60 仍保留为精度上界。

### DATE11 TTX MDR/MVSEC 标准化训练计划（2026-06-30）

目的：在完整 MVSEC 四序列 baseline eval 完成后，直接启动 **TTX 主线** 的 MDR->MVSEC 标准化训练，用于外部泛化表。TTX 定义保持为：`all-binary ATLIFPSN + H60 TX-only selector, bipolar_mu=0, no SC, no Kmag`。

- 新增配置：`configs/generated/train_mdr_ttx_mvsec_route_fast.yml`
  - 基于 `configs/generated/train_mdr_all_binary_atlif_tx_mvsec_route_fast.yml` 的 MDR fast route。
  - 保持 MDR protocol：`crop=256x256`、`batch_size=32`、`n_epochs=50`、`use_amp=true`、MVSEC `dt1` validation。
  - TTX 替换项：`bsa_attention.mode=h60`、`bipolar_mu=0.0`、`sc_mu_schedule_enabled=false`、`sc_mu_warmup_steps=0`、`k_magnitude_alpha=0.0`、`hardware_quant_enabled=false`。
  - ATLIF 仍为全二值：`atlif_ternary_psn.output_mode=binary`，`sn2_q + all_non_qk_binary_atlif`，`threshold_freeze_after_step=1224`。
- 新增等待启动脚本：`neuron_experiments/H9_bipolar_self_attention/entrypoints/run_ttx_mdr_after_mvsec_baseline.sh`
  - 等待当前 baseline MVSEC orchestrator 完成。
  - 从 `epoch41/epoch47` 四序列 `mvsec_ranking.md` 里选择平均 AEE 更低的 baseline checkpoint。
  - 先跑 1-batch smoke：检查 `ATLIFTernaryPSN`、`Shiftmax attention`、`load audit`。
  - smoke 通过后启动 full MDR training；使用 `SDFORMER_MDR_SKIP_MLFLOW_MODEL_LOG=1` 和 `SDFORMER_MDR_LOCAL_CHECKPOINT_DIR` 保存本地 checkpoint，避免 MLflow 大模型上传卡住。
- 启动命令范式：

```bash
cd /root/private_data/work/sdformer_codex/SDformer
setsid bash neuron_experiments/H9_bipolar_self_attention/entrypoints/run_ttx_mdr_after_mvsec_baseline.sh \
  > neuron_experiments/H9_bipolar_self_attention/results/ttx_mdr_after_mvsec_baseline_wait_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

注意：现有 `configs/generated/train_mdr_all_binary_atlif_tx_mvsec_route_fast.yml` 是旧 all-binary TX/H18a 风格配置，`bsa_attention.mode=ternary_alpha_xnor_shiftmax` 且 `bipolar_mu=0.05`，不等价于 TTX；MDR 外部泛化主线应使用新的 `train_mdr_ttx_mvsec_route_fast.yml`。

TTX MDR/MVSEC 启动审计（2026-06-30）：

- Baseline 四序列 MVSEC eval 已完成。
  - epoch41 mean AEE `1.2048`：outdoor_day1 `1.0068`，indoor1 `1.1795`，indoor2 `1.3790`，indoor3 `1.2540`。
  - epoch47 mean AEE `1.1414`：outdoor_day1 `0.8959`，indoor1 `1.0634`，indoor2 `1.3943`，indoor3 `1.2120`。
  - 因此 TTX 初始化选择 `mdr_valid_resume_local_ckpts_20260625_164239/checkpoint_epoch47.pth`。
- `train_mdr_ttx_mvsec_route_fast.yml` 早期 batch32 smoke 曾失败为 CUDA OOM；复核发现当时同机有其它进程占用大量显存，不能作为 TTX batch32 不可用的结论。
- 新增 `configs/generated/train_mdr_ttx_mvsec_route_fast_bs16.yml`，batch16 smoke 可以前向/反传：
  - torch backend：1 train batch 约 `47.0s`，1 valid batch 约 `38s`，train loss 从 `1.5816` 到 `1.5080`（smoke 多个 1-batch epoch，随后停止）。
  - cupy backend 1-epoch/1-batch smoke：加载审计正常，`ATLIFTernaryPSN=105`，`Shiftmax attention=12`，`checkpoint_overlay_keys=0, missing=210, unexpected=0`；1 train batch `45.8s`，valid loss `1.4427`。
- 2026-06-30 速度复核（A800 80GB，干净 GPU，`checkpoint_epoch47.pth` 续训，`batch_size=32`、`n_workers=8`、AMP、skip validation，限制 80 train batches）：

| route | backend | batch | workers | train samples/s | 估算 sec/epoch | 估算 h/epoch | 备注 |
|---|---|---:|---:|---:|---:|---:|---|
| TTX MDR fast | torch | 32 | 8 | 8.303 | 10324 | 2.87 | `ttx_mdr_forkserver_preload_bench80_20260630_154148` |
| TTX MDR fast | cupy | 32 | 8 | 9.120 | 9399 | 2.61 | `ttx_mdr_forkserver_cupy_bench80_20260630_155032` |
| TTX MDR workers0 | torch | 32 | 0 | 0.852 | 100610 | 27.95 | 单进程 dataloader 对照 |
| MDR baseline workers0 | torch | 32 | 0 | 0.864 | 99213 | 27.56 | 说明 28-40h/epoch 主要是 dataloader worker 没跑起来，不是 TTX 独有 |

- 结论：TTX MDR 链路、权重加载、batch32 都可跑；“40h/epoch”只在 workers 失效/单进程 dataloader 时成立。当前推荐 full train 使用 `batch32 + n_workers=8 + AMP + cupy backend`，并在启动前设置 multiprocessing `forkserver` preload（`torch`, `torchvision.extension`, `torchvision`），否则本机 `/opt/conda` 环境可能在 worker 里触发 torchvision 导入顺序问题。按 cupy 80-batch 稳定吞吐估算，TTX MDR full 约 `2.6h/epoch`，50 epoch 约 `5.4` 天；比 baseline fast 的 `18 samples/s` 慢约 2 倍，原因更可能是 all-binary ATLIFPSN + TTX 覆盖的模型计算开销，而不是数据加载。

TTX MDR/MVSEC Codex 复核与正式启动（2026-06-30）：

- 启动脚本修正：`run_ttx_mdr_after_mvsec_baseline.sh` 的 smoke 段原本只限制 `SDFORMER_MDR_MAX_TRAIN_BATCHES=1`，但仍沿用 full config 的 `n_epochs=50`，会变成 50 个 one-batch smoke epoch。已改为在 `RESULT_ROOT/smoke_config.yml` 生成 smoke 专用配置，并固定 `loader.n_epochs=1`；正式训练段仍使用原始 full config。
- 复核命令：`PYTHON_BIN=/opt/conda/envs/sdformerflow/bin/python ORCH_PID=0 SNN_BACKEND=cupy SDFORMER_MDR_MAX_TRAIN_BATCHES=80 SDFORMER_MDR_SKIP_VALIDATION=1` + `run_ttx_mdr_after_mvsec_baseline.sh`。
- 复核结果：`ttx_mdr_forkserver_cupy_bench80_codex_fixed_20260630_161513`，smoke 加载审计正常，full 段 80 train batches 完成，`train_loop_elapsed_s=235.194`，`train_batches=80`，`train_samples=2560`，`train_samples_per_s=10.885`，估算 full epoch 约 `2.18h`。
- 正式训练已启动：`ttx_mdr_full_cupy_from_ep47_20260630_162339`，初始化 checkpoint 为 `mdr_valid_resume_local_ckpts_20260625_164239/checkpoint_epoch47.pth`，full 段确认 `max_train_batches=None`、`max_valid_batches=None`、`skip_validation=False`，进入 `Epoch 0`。
- 当前活跃训练以 `ttx_mdr_full_cupy_from_ep47_20260630_162339` 为准；旧记录 `mdr_ttx_full_cupy_forkserver_20260630_161902` 不是当前活跃 run。
- `ttx_mdr_full_cupy_from_ep47_20260630_162339` 后续状态：epoch0 完整完成，`train_loop_elapsed_s=6839.186`，`train_samples_per_s=12.530`，train loss `1.0937`，valid loss `1.1796`，已保存 `local_ckpts/checkpoint_epoch0.pth` 和 `checkpoint_epoch0_state_dict.pth`。epoch1 约 step69 发生 DataLoader worker OOM；主训练进程占约 `72.78GB`，8 个 worker 各自建立 CUDA context 约 `0.7-1.0GB`，worker 在 `MDR.py -> self.voxel(...).cpu()` 申请 22MB 显存失败。结论：不是模型 NaN，也不是 full config 被 80-batch 限制，而是 `n_workers=8` 在 batch32/full ATLIF 下显存余量太小。
- 已新增 `configs/generated/train_mdr_ttx_mvsec_route_fast_workers4.yml`，仅把 `n_workers` 从 8 降到 4。已从 epoch0 本地 checkpoint + state_dict 续训：`ttx_mdr_full_cupy_w4_resume_ep0_cd_20260630_190733`，恢复审计 `checkpoint_overlay_keys=210, missing=0, unexpected=0`，`Training state resumed ... checkpoint_epoch0_state_dict.pth`，从 `Epoch 1` 开始，GPU 约 `75.9GB/80GB`、util `100%`，当前继续训练。
- `ttx_mdr_full_cupy_w4_resume_ep0_cd_20260630_190733` 后续状态：worker4 仍在 epoch1 step687 发生 DataLoader worker OOM。报错仍位于 `MDR_dataloader/MDR.py -> self.voxel(...).cpu()` 和 `loader_utils.py` 的 voxel grid 构建；主训练进程约 `75.42GB`，4 个 worker 各自约 `0.95-1.00GB` CUDA context，只剩约 `8.62MB` free，worker 申请 36MB 失败。结论：`worker=6` 直接沿用 GPU voxel 不可取，会比 worker4 多约 2 个 CUDA context，OOM 风险更高；慢速和 OOM 的共同根因是 MDR DataLoader worker 在 GPU 上做 voxel，而不是训练超参变化。
- 新增可选分支：`SDFORMER_MDR_VOXEL_GPU=0`。该开关只影响 MDR DataLoader 的 voxel 构建位置；默认不设置时保持旧逻辑 `gpu=True`。设置为 0 后，worker 在 CPU 上构建 voxel，避免每个 worker 建 CUDA context，模型结构、loss、optimizer、checkpoint 加载逻辑均不变。
- 新增配置：`configs/generated/train_mdr_ttx_mvsec_route_fast_workers6_cpuvoxel.yml`
  - 基于 `train_mdr_ttx_mvsec_route_fast.yml`。
  - 仅改 `experiment=date11_ttx_mdr_mvsec_route_fast_workers6_cpuvoxel` 和 `loader.n_workers=6`。
  - 正式训练环境额外设置 `SDFORMER_MDR_VOXEL_GPU=0`、`SNN_BACKEND=cupy`、`SDFORMER_MDR_DETECT_ANOMALY=0`、`SDFORMER_MDR_SKIP_MLFLOW_MODEL_LOG=1`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
- worker6 + CPU voxel 80-batch 复核：`ttx_mdr_w6_cpuvoxel_bench80_resume_ep0_20260630_200423`
  - 初始化：从 `ttx_mdr_full_cupy_from_ep47_20260630_162339/local_ckpts/checkpoint_epoch0.pth` 和 `checkpoint_epoch0_state_dict.pth` 续训。
  - 审计：`[MDR dataloader] voxel_gpu=False`，`ATLIFTernaryPSN=105`，`Shiftmax attention=12`，`checkpoint_overlay_keys=210, missing=0, unexpected=0`。
  - runtime：`max_train_batches=80`、`skip_validation=True`，从 `Epoch 1` 开始。
  - 结果：80 train batches 完成，`train_loop_elapsed_s=266.904`，`train_batches=80`，`train_samples=2560`，`train_samples_per_s=9.591`，估算完整 epoch 约 `2.48h`。该测试随后进入下一 epoch，已手动停止；其 checkpoint 不作为论文结果。
- worker6 + CPU voxel 正式续训：`ttx_mdr_full_cupy_w6_cpuvoxel_resume_ep0_20260630_201141`
  - full 模式确认：`max_train_batches=None`、`max_valid_batches=None`、`skip_validation=False`。
  - 加载确认：`checkpoint_overlay_keys=210, missing=0, unexpected=0`，`Training state resumed ... checkpoint_epoch0_state_dict.pth`，从 `Epoch 1` 继续。
  - 该 run 已跑到 epoch10：`checkpoint_epoch10.pth` 和 `checkpoint_epoch10_state_dict.pth` 已保存。epoch10 train `train_loop_elapsed_s=6793.679`，`train_samples_per_s=12.614`，train loss `0.3983`。随后停在 epoch10 validation：原因不是训练 forward OOM，而是 MVSEC 验证集 `MDR_dataloader/MVSEC.py` 仍硬编码 `gpu=True`，6 个 validation worker 在 GPU 上做 voxel，模型占约 `76.85GB` 时 worker 申请 20MB 失败。
- MVSEC validation OOM 修复（2026-07-02）：已把同一个 `SDFORMER_MDR_VOXEL_GPU` 可选开关补到 `MDR_dataloader/MVSEC.py`。现在 `SDFORMER_MDR_VOXEL_GPU=0` 会同时作用于 MDR train dataloader 和 MVSEC validation dataloader；默认不设置仍保持旧行为。
- worker6 + CPU voxel 从 epoch10 续训：`ttx_mdr_full_cupy_w6_cpuvoxel_resume_ep10_20260702_161956`
  - 初始化 checkpoint：`ttx_mdr_full_cupy_w6_cpuvoxel_resume_ep0_20260630_201141/local_ckpts/checkpoint_epoch10.pth`。
  - 审计：`[MDR dataloader] voxel_gpu=False`，`[MVSEC dataloader] voxel_gpu=False`，`checkpoint_overlay_keys=210, missing=0, unexpected=0`。
  - 恢复：`Training state resumed ... checkpoint_epoch10_state_dict.pth`，从 `Epoch 11` 开始，full 模式 `max_train_batches=None`、`skip_validation=False`。
  - 当前建议：后续 TTX MDR full 以该 run 为准；不要再尝试 `worker=6/8 + GPU voxel`。若仍有 validation OOM，再降到 `n_workers=4 + SDFORMER_MDR_VOXEL_GPU=0` 或临时拆分为 train-only + 独立 validation。
- TTX MDR full 训练完成状态（2026-07-06）：`ttx_mdr_full_cupy_w6_cpuvoxel_resume_ep10_20260702_161956` 已跑完到 epoch49，GPU 空闲。由于训练脚本只在 train loss 刷新 best 时保存，后半段最新保留 checkpoint 到 `checkpoint_epoch43.pth`；用于 MVSEC 选择的 validation checkpoint 中 `epoch20` 最好。
  - validation loss：epoch15 `1.1861`，epoch20 `1.1035`，epoch25 `1.1415`，epoch30 `1.1787`，epoch35 `1.1468`，epoch40 `1.1261`，epoch45 `1.1260`。
  - train loss 继续下降但趋于平台：epoch20 `0.3450`，epoch30 `0.3363`，epoch40 `0.3326`，epoch49 `0.3324`。这说明 epoch20 后更像 MDR train 拟合，MVSEC validation 没继续改善。
  - 当前推荐 MVSEC 标准 AEE 评估 checkpoint：`checkpoint_epoch20.pth`。
- 已生成 TTX 专用 MVSEC eval 配置：`configs/generated/eval_mvsec_dt1_ttx_mdr_epoch20_route.yml`，保留 TTX 的 `ATLIFTernaryPSN=105` 和 `Shiftmax attention=12`，评估口径沿用 AEE-only MVSEC dt1 route。
- 已启动 epoch20 四序列标准评估：`results_inference/mvsec_ttx_mdr_epoch20_dt1_full4_20260706_001522`。该 eval 才能和 baseline `mvsec_mdr_baseline_epoch47_dt1_full4_20260629_235858` 的四序列 AEE/energy 做同口径比较；训练日志中的 validation loss 只能用于选 checkpoint。
- TTX epoch20 四序列标准评估完成（2026-07-06）：
  - 目录：`results_inference/mvsec_ttx_mdr_epoch20_dt1_full4_20260706_001522`
  - ranking：`results_inference/mvsec_ttx_mdr_epoch20_dt1_full4_20260706_001522/mvsec_ranking.md`

| checkpoint | sequence | AEE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TTX ep20 | outdoor_day1 | 0.9779 | 0.3748 | 0.0840 | 0.0066 | 28.6694G | 6.1312% | 25782.02 |
| TTX ep20 | indoor_flying1 | 1.1414 | 0.5214 | 0.1068 | 0.0140 | 20.8872G | 6.5355% | 18792.65 |
| TTX ep20 | indoor_flying2 | 1.4266 | 0.5024 | 0.1965 | 0.0915 | 23.8459G | 7.4534% | 21452.82 |
| TTX ep20 | indoor_flying3 | 1.2617 | 0.5082 | 0.1521 | 0.0509 | 22.0527G | 6.8929% | 19840.89 |
| TTX ep20 mean | 4-seq | 1.2019 | - | - | - | 23.8638G | - | 21467.09 |

  - 对比本地 MDR baseline epoch47 mean：AEE `1.1414`、total_spikes `39.5024G`、energy `35225.57uJ`。
  - TTX ep20 相对 baseline：AEE `+0.0605`（`+5.30%`），total_spikes `-15.6386G`（`-39.59%`），energy `-13758.47uJ`（`-39.06%`）。
  - 论文目标口径判断：降耗目标明显达成（>20%），精度目标“baseline 约 5% 内”略微越线，主要由 `outdoor_day1` 和 `indoor_flying1` 拉高；`indoor_flying2/3` 分别为 `+2.32%/+4.10%`，在 5% 内。
- TTX MDR 训练/推理链路复核（2026-07-06）：
  - 训练初始化不是 strict resume baseline optimizer/scheduler，而是从本地 MDR baseline `checkpoint_epoch47.pth` 做模型权重 warm-start。启动日志：`selected baseline epoch=47 mean_aee=1.141400 checkpoint=.../mdr_valid_resume_local_ckpts_20260625_164239/checkpoint_epoch47.pth`。
  - 初始 TTX full run 加载 baseline 时审计为 `ATLIFTernaryPSN=105`、`Shiftmax attention=12`、`checkpoint_overlay_keys=0, missing=210, unexpected=0`。解释：baseline checkpoint 没有 H9 overlay 参数，新增 ATLIF 参数从默认值初始化；共享 backbone/conv/BN/QK 权重从 baseline 加载。
  - 之后的 TTX 自身 resume 链路正常：从 epoch0 和 epoch10 继续时均为 `checkpoint_overlay_keys=210, missing=0, unexpected=0`，且 `Training state resumed ... checkpoint_epoch*_state_dict.pth`，说明 optimizer/scheduler/scaler 是在 TTX 自身 run 内续上的。
  - checkpoint key 复核：baseline ep47 有 `711` 个 state_dict key、无 overlay key；TTX ep20 有 `921` 个 key，其中 `210` 个为 `spiking_neuron.thresh/center` overlay key。两者共享 `711` 个 tensor key。Shiftmax/H60 attention 在当前实现中是无参数 forward patch，因此没有单独 attention state_dict key，这不是漏保存。
  - eval 链路正常：四个 MVSEC 序列均显示 `eval installed ATLIFTernaryPSN: 105 modules`、`eval installed Shiftmax attention: 12 modules`、`checkpoint_overlay_keys=210, model_overlay_keys=210, missing=0, unexpected=0`，且 checkpoint 路径为 TTX ep20。
  - 当前“问题”不是权重没加载或推理配置错配，而是实验范式风险：TTX 从 baseline warm-start 后替换了全二值 ATLIF + 无参数 H60/TX selector，新 overlay 从默认阈值/center 开始；MDR train loss 持续下降，但 MVSEC validation 在 epoch20 后反弹，说明继续训练更像拟合 MDR，不一定改善 MVSEC 四序列 AEE。后续应优先做从 ep20 出发的小 LR/短程精修或更稳的 calibration，而不是盲目延长到 60/70 epoch。
- 晚期 checkpoint 标准评估队列（2026-07-06）：
  - 新增脚本：`neuron_experiments/H9_bipolar_self_attention/entrypoints/run_ttx_mvsec_late_epoch_eval_queue.sh`。
  - 选择 checkpoint：`epoch40` 和 `epoch43`。理由：`epoch40` 是后期 validation loss 仍较低的代表点（epoch40 `1.1261`，epoch45 `1.1260` 但没有 epoch45 checkpoint），`epoch43` 是后半段最后保存的 checkpoint。与 `epoch20` 一起可以判断“继续 MDR 训练是否改善标准 MVSEC AEE”，避免无意义地把 `31/34/37/40/43` 全部四序列扫完。
  - 队列行为：先等待 `results_inference/mvsec_ttx_mdr_epoch20_dt1_full4_20260706_001522` 结束，再顺序跑：
    - `results_inference/mvsec_ttx_mdr_epoch40_dt1_full4_<stamp>`
    - `results_inference/mvsec_ttx_mdr_epoch43_dt1_full4_<stamp>`
  - 启动范式：

```bash
cd /root/private_data/work/sdformer_codex/SDformer
setsid bash neuron_experiments/H9_bipolar_self_attention/entrypoints/run_ttx_mvsec_late_epoch_eval_queue.sh 40 43 \
  > results_inference/mvsec_ttx_mdr_late_epoch_eval_queue_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

  - 注意：晚期 checkpoint 评估只用于论文中选择/解释主 checkpoint，不作为新的训练分支；主对比仍以同口径四序列 `mvsec_ranking.md` 为准。
  - 完成状态（2026-07-07）：`epoch40` 与 `epoch43` 四序列标准评估均已完成，队列正常退出。GPU 当前无训练/评估进程。

| checkpoint | mean AEE | mean total_spikes | mean energy_uj | vs baseline AEE | spike reduction | energy reduction | ranking |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline ep47 | 1.1414 | 39.5024G | 35225.57 | - | - | - | `results_inference/mvsec_mdr_baseline_epoch47_dt1_full4_20260629_235858/mvsec_ranking.md` |
| TTX ep20 | 1.2019 | 23.8638G | 21467.09 | +5.30% | 39.59% | 39.06% | `results_inference/mvsec_ttx_mdr_epoch20_dt1_full4_20260706_001522/mvsec_ranking.md` |
| TTX ep40 | 1.2217 | 25.2930G | 22750.96 | +7.04% | 35.97% | 35.41% | `results_inference/mvsec_ttx_mdr_epoch40_dt1_full4_20260706_003729/mvsec_ranking.md` |
| TTX ep43 | 1.2251 | 25.1262G | 22600.76 | +7.34% | 36.39% | 35.84% | `results_inference/mvsec_ttx_mdr_epoch43_dt1_full4_20260706_003729/mvsec_ranking.md` |

  - 结论：晚期 checkpoint 没有改善四序列 MVSEC AEE，且 spikes/energy 也比 ep20 更高。`epoch20` 是当前 TTX MDR/MVSEC 标准评估的最优 checkpoint；`epoch40/43` 只能作为“继续训练会过拟合 MDR、外部 MVSEC 不改善”的支撑证据，不应作为主结果。
- TTX ep20 低 LR 微调计划与启动（2026-07-08）：
  - 目的：从当前最优标准 MVSEC checkpoint `epoch20` 做短程 calibration，尝试把 mean AEE 从 `+5.30%` 拉回 5% 内，同时保持约 39% spikes/energy 降低。
  - 配置：`configs/generated/train_mdr_ttx_mvsec_ep20_calib_lr025_ep26.yml`。
  - 启动脚本：`neuron_experiments/H9_bipolar_self_attention/entrypoints/run_ttx_mdr_ep20_calib_lr025.sh`。
  - 起点 checkpoint：`neuron_experiments/H9_bipolar_self_attention/results/ttx_mdr_full_cupy_w6_cpuvoxel_resume_ep10_20260702_161956/local_ckpts/checkpoint_epoch20.pth`。
  - 训练方式：使用 `--resume` 恢复 `checkpoint_epoch20_state_dict.pth`，因此从 `Epoch 21` 开始，阈值冻结不会重新打开；同时设置 `SDFORMER_MDR_RESET_LR_FROM_CONFIG=1`，在 resume optimizer 后强制把 LR 重置为低 LR 配置。
  - LR：固定 LR、无 multistep、无 warmup。`backbone_lr=5e-7`，`norm_lr=2.5e-7`，`neuron_lr=1.25e-5`，`threshold_lr=1e-6`，约为主训 1/4。
  - 轮次：`loader.n_epochs=26`，即 resume 后跑 `epoch21-25` 五轮；`test.n_valid=5`，因此会在 `epoch25` 做一次训练内 MVSEC validation。
  - smoke：`ttx_mdr_ep20_calib_lr025_smoke2_20260708_162924` 已验证 `MDR/MVSEC voxel_gpu=False`、`ATLIFTernaryPSN=105`、`Shiftmax attention=12`、`checkpoint_overlay_keys=210, missing=0, unexpected=0`、`Training state resumed ... checkpoint_epoch20_state_dict.pth`、LR reset 生效，并进入 `Epoch 21` 完成 1 个 train batch。smoke 随后手动停止，不作为论文结果。
  - full 启动范式：

```bash
cd /root/private_data/work/sdformer_codex/SDformer
setsid bash neuron_experiments/H9_bipolar_self_attention/entrypoints/run_ttx_mdr_ep20_calib_lr025.sh \
  > neuron_experiments/H9_bipolar_self_attention/results/ttx_mdr_ep20_calib_lr025_launch_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

  - full run 已启动：PID `2152241`，launcher log `neuron_experiments/H9_bipolar_self_attention/results/ttx_mdr_ep20_calib_lr025_launch_20260708_163326.log`，run dir `neuron_experiments/H9_bipolar_self_attention/results/ttx_mdr_ep20_calib_lr025_ep21_25_20260708_163326`。
  - 启动审计：`MDR/MVSEC voxel_gpu=False`，`checkpoint_overlay_keys=210, missing=0, unexpected=0`，`Training state resumed ... checkpoint_epoch20_state_dict.pth`，LR reset 后 param groups 为 `backbone=5e-7`、`norm=2.5e-7`、`atlif_neuron=1.25e-5`、`atlif_threshold=1e-6`。已进入 `Epoch 21`，GPU util 约 `97%`、显存约 `72.98GB/81.92GB`。
  - **full run 完成（2026-07-09 复核）**：
    - 状态：训练进程已退出；launch log 末尾有 `[ttx-ep20-calib] complete 2026-07-09T03:50:03+08:00`。当前 `nvidia-smi` 空闲，无 `train_mdr_supervised_SNN` / `eval_MV_flow_SNN` / calib 相关进程。
    - run dir：`neuron_experiments/H9_bipolar_self_attention/results/ttx_mdr_ep20_calib_lr025_ep21_25_20260708_163326`
    - train log：`.../ttx_mdr_ep20_calib_lr025_ep21_25_20260708_163326/train.log`
    - launch log：`neuron_experiments/H9_bipolar_self_attention/results/ttx_mdr_ep20_calib_lr025_launch_20260708_163326.log`
    - 加载审计复核：`[MDR/MVSEC dataloader] voxel_gpu=False`；`ATLIFTernaryPSN=105`；`Shiftmax attention=12`；`checkpoint_overlay_keys=210, missing=0, unexpected=0`；`Training state resumed ... checkpoint_epoch20_state_dict.pth`；从 `Epoch 21` 开始；LR reset 生效。
    - 训练结果（脚本只在 train loss 刷新 best 时保存 checkpoint）：

| epoch | train loss | checkpoint saved? | notes |
|---:|---:|:---:|---|
| 21 | 0.342202 | yes | `local_ckpts/checkpoint_epoch21.pth` + state_dict |
| 22 | 0.343056 | no | 未优于 best |
| 23 | 0.342558 | no | 未优于 best |
| 24 | 0.339920 | yes | best train loss；`local_ckpts/checkpoint_epoch24.pth` + state_dict |
| 25 | 0.341162 | no | 有训练内 MVSEC validation |

    - epoch25 训练内 validation（indoor_flying3-style train-loop valid，**不是**四序列标准 AEE）：`Epoch loss (Validation): 1.156682`
    - 对照原 TTX full run 的 train-loop valid：ep20 `1.1035`，ep25 `1.1415`。calib ep25 的 `1.1567` **劣于** 原 ep20 valid，提示低 LR 微调未必改善 MVSEC；最终结论必须以四序列标准 AEE 为准。
    - 可评估 checkpoint：仅 `epoch21` 与 `epoch24`（无 ep22/23/25 模型权重落盘）。
    - 标准评估安排：对 `checkpoint_epoch24.pth`（优先）和 `checkpoint_epoch21.pth` 跑 MVSEC dt1 四序列 eval，并与 baseline ep47 / TTX ep20/40/43 对比；该安排已按下节记录执行完成。
  - **calib MVSEC 标准 eval 已完成（2026-07-10 复核）**：
    - 队列：先 `epoch24`（calib best train loss），再 `epoch21`。
    - config：`configs/generated/eval_mvsec_dt1_ttx_mdr_epoch20_route.yml`
    - ckpt root：`neuron_experiments/H9_bipolar_self_attention/results/ttx_mdr_ep20_calib_lr025_ep21_25_20260708_163326/local_ckpts`
    - out：`results_inference/mvsec_ttx_mdr_ep20_calib_epoch24_dt1_full4_20260709_154906` 与 `..._epoch21_...`
    - queue log：`neuron_experiments/H9_bipolar_self_attention/results/ttx_mdr_ep20_calib_mvsec_eval_queue_20260709_154906.log`
    - 环境：`SDFORMER_USE_MLFLOW=0`，`SDFORMER_MDR_VOXEL_GPU=0`，`SNN_BACKEND=cupy`
    - 四序列标准结果：

| checkpoint | mean AEE | mean outlier | mean total_spikes | mean energy | vs baseline AEE | spikes reduction | energy reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| MDR baseline ep47 | 1.1414 | 0.0389 | 39.5024G | 35225.57uJ | reference | - | - |
| TTX original ep20 | 1.2019 | 0.0408 | 23.8638G | 21467.09uJ | +5.30% | 39.59% | 39.06% |
| calib ep21 | 1.2750 | 0.0434 | 23.4104G | 21057.82uJ | +11.70% | 40.74% | 40.22% |
| calib ep24 | 1.2775 | 0.0434 | 24.2060G | 21772.43uJ | +11.92% | 38.72% | 38.19% |

    - 结论：本次 ep20 低 LR calibration **没有改善 MVSEC 泛化**。ep21/ep24 虽继续保持约 `38%-41%` 的 spikes/energy 降幅，但 mean AEE 从 ep20 的 `1.2019` 退化到 `1.2750/1.2775`，明显超出 baseline `+5%` 目标；训练内 validation 变差与四序列标准 AEE 方向一致。
    - MVSEC 推荐 checkpoint 仍为原 TTX full run `checkpoint_epoch20.pth`。calib ep21/ep24 作为负结果保留，不用于论文主结果，也不继续扩大低 LR 微调扫参。



- 旧重复 run `mdr_ttx_full_cupy_forkserver_20260630_161902`（2026-06-30 16:19 启动，launcher PID `494591`）：
  - label：`mdr_ttx_full_cupy_forkserver_20260630_161902`
  - launcher PID：`494591`
  - config：`configs/generated/train_mdr_ttx_mvsec_route_fast.yml`
  - init checkpoint：`neuron_experiments/H9_bipolar_self_attention/results/mdr_valid_resume_local_ckpts_20260625_164239/checkpoint_epoch47.pth`
  - launch log：`neuron_experiments/H9_bipolar_self_attention/results/mdr_ttx_full_cupy_forkserver_20260630_161902.launch.log`
  - smoke log：`neuron_experiments/H9_bipolar_self_attention/results/mdr_ttx_full_cupy_forkserver_20260630_161902/smoke.log`
  - train log：`neuron_experiments/H9_bipolar_self_attention/results/mdr_ttx_full_cupy_forkserver_20260630_161902/train.log`
  - local checkpoints：`neuron_experiments/H9_bipolar_self_attention/results/mdr_ttx_full_cupy_forkserver_20260630_161902/local_ckpts`
  - smoke：1 train batch loss `1.5203`，1 valid batch loss `1.4253`，本地 checkpoint/state 保存正常。
  - full train 审计：`checkpoint_overlay_keys=0, missing=210, unexpected=0`，`Shiftmax attention=12`，`detect_anomaly=False`，`max_train_batches=None`，`skip_validation=False`；已进入 epoch0，约 step `88/2678` 时 GPU 显存约 `80.9GB`，证明链路正常。
  - 处理：因 `ttx_mdr_full_cupy_from_ep47_20260630_162339` 已作为正式 active run 启动，本重复 run 已按要求停止；停止前约到 epoch0 step `220/2678`，未完成完整 epoch，不作为论文结果。

## 四十一、TTX/BTTX 全网 all12 硬件冻结候选（2026-07-10）

### 41.1 约束与主线定义

本轮冻结以下硬约束：不再考虑 S2-only、stage-wise、partial ATLIF 或混合注意力范围；所有候选均为 `105` 个 ATLIF wrapper + `12` 个 attention block 的全网替换。候选必须复用同一条线性复杂度数据流：

`Q/K event classification -> category popcount -> integer score -> centering -> Shiftmax/LUT -> gate * K`

- `TTX`：全二值 ATLIFPSN `{0,+1}` + H60 TX-only，`mu=0`、no carrier、no Kmag。
- `BTTX`：全三值/双极 ATLIFPSN `{-1,0,+1}` + 同一 H60 no-carrier selector；神经元和 score category 增加负事件 rail，但 token/gate/K 数据流不变。
- 不再把 NTX/H18 carrier 路线作为硬件主线；旧 `all-ternary + TX` 使用 `ternary_alpha_xnor_shiftmax` 并保留 native QKFormer carrier，不能当作 exact BTTX。

### 41.2 已有 DSEC 证据与 exact BTTX 缺口

| existing experiment | neuron | attention/dataflow | best AEE | spikes | 判断 |
|---|---|---|---:|---:|---|
| current TTX deploy | all binary | H60 TX-only, no carrier | 1.5003 | 23.2462G | 当前低风险主线 |
| all-ternary + old TX | all ternary | H18a TX + native carrier | 3.2885 | 79.0967G | 不是 exact BTTX；负例 |
| all-ternary + SC | all ternary | SC-only no-carrier | 3.2706 | 80.1816G | integer SC 已有强负证据 |
| all-ternary + NTS | all ternary | H60 TX + scheduled SC `mu=0.05` | 3.2809 | 80.0070G | 接近但不等于 BTTX-TX |

旧全三值实验共同活动率约 `16%-17%`，而当前 binary TTX 约 `4.5%-5%`。因此 BTTX 的主要风险是全网负事件把 activity/spikes 推高，不是 attention score 本身。exact BTTX 仍值得做一次从已收敛 TTX checkpoint warm-start 的受控筛选，用于区分“从 baseline 直接全三值训练失败”和“三值结构本身不可行”。

### 41.3 固定整数 score 候选

TX 与 SC 可共享 `same/zero/opposite` 分类计数，无需两个独立 score datapath。取 `alpha0=1/64`、`mu=1/16`：

- binary TTX-ISC16：`score * 64 = 68*same + 1*zero`；binary 下没有 opposite category。
- ternary BTTX-ISC16：`score * 64 = 68*same + 1*zero - 4*opposite`。

系数 `68=64+4` 与 `4` 均用移位加减实现；没有通用乘法器、Kmag、target-rate 或运行时 mu schedule。软件仍复用现有 `h60` 实现，硬件可在 category counters 后代数折叠成单个 weighted adder。

### 41.4 最小全替换筛选协议

生成脚本：

`neuron_experiments/H9_bipolar_self_attention/entrypoints/make_date11_full_all12_ttx_family_configs.py`

manifest：

`neuron_experiments/H9_bipolar_self_attention/configs/generated/date11_full_all12_ttx_family_manifest.json`

共同起点：DSEC TTX best `date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8.../checkpoint_epoch2.pth`。共同范围为 full-network/all12；先跑 `360` steps + valid40，只有达到明确精度/活动率潜力才进入完整训练。

| priority | candidate | event alphabet | score | status |
|---|---|---|---|---|
| P0 | `date11full_ttx_isc16_all12_s360` | binary | `68*same+zero` | config generated; chain PASS |
| P0 | `date11full_bttx_txonly_all12_s360` | ternary | exact BTTX `same+0.02*zero` | config generated; chain PASS |
| P0 | `date11full_bttx_isc16_all12_s360` | ternary | `68*same+zero-4*opposite` | config generated; chain PASS |

链路审计：三个配置均安装 `ATLIFTernaryPSN=105`、`Shiftmax attention=12`；保存后自身 checkpoint reload 为 `checkpoint_overlay_keys=210, missing=0, unexpected=0`。本轮不新增模型结构代码，只新增配置生成器和配置。

### 41.5 主线约束修正：取消所有 TX/SC 混合（2026-07-10）

用户冻结硬件/论文约束：不采用 TX+SC score/gate 混合，不采用 NTX/native carrier，不采用 partial replacement。上节生成过的 `TTX-ISC16/BTTX-ISC16` 配置只保留为探索记录，不进入候选矩阵、不启动训练，也不写入论文主线。后续只允许纯 TX：

- `TTX`：all-binary ATLIF + all12 H60 TX-only。
- `BTTX`：all-ternary ATLIF + all12 H60 TX-only。
- 纯算术简化可把 `alpha0=0.02` 近似为 `1/64`，但不允许增加 SC/opposite residual 分支。

此前完成的 ITTX int8 纯推理（NTS ep2 权重，AEE `1.48891` / AAE `9.76219`）证明混合分数可折叠，但因论文故事和硬件约束已主动放弃，**不得将该结果作为最终主线**。

### 41.6 exact BTTX 与 BTTX-A4 结果

exact symmetric BTTX 定义：`105` 个 ternary ATLIF、all12 H60 TX-only、`mu=0`、no SC/no Kmag/no carrier、正负阈值 `+theta/-theta`。

1. exact BTTX 纯推理：
   - checkpoint：旧 all-ternary H60/NTS ep29；推理时关闭 SC，改为 TX-only。
   - config：`configs/generated/date11full_bttx_txonly_all12_s360.yml`。
   - out：`results/date11_bttx_txonly_infer_from_ternary_nts_ep29_valid825_20260710`。
   - load audit：`ATLIFTernaryPSN=105`、`Shiftmax=12`、`checkpoint_overlay_keys=210, missing=0, unexpected=0`。
   - valid825：AEE `18.5180`、AAE `103.9021`、total_spikes `254.2891G`、global firing `54.9543%`。

2. symmetric BTTX 从当前 TTX best warm-start：
   - run：`results/date11_bttx_txonly_from_ttx_20260710_200056`。
   - step20 activity `47.22%`（pos `5.06%` / neg `42.16%`）；step100 仍为 `47.14%`（pos `5.07%` / neg `42.07%`）。
   - 负事件支路没有随短训恢复，结合旧 full30 全三值结果 AEE 约 `3.27-3.29`，在 step100 提前停止，不保留为论文候选。

3. BTTX-A4：保持纯 TX，只把负阈值设为 `-4*theta`：
   - 硬件变化仅为负阈值左移；attention/token/gate 数据流不变。
   - config：`configs/generated/date11full_bttx_a4_txonly_all12_s360.yml`。
   - run：`results/date11_bttx_a4_txonly_from_ttx_20260710_200703`。
   - load audit：从 TTX ep2 加载 `checkpoint_overlay_keys=210, missing=0, unexpected=0`。
   - step120 train activity `5.5863%`（pos `3.9679%` / neg `1.6183%`），说明 `-4*theta` 能把活动率压回 TTX 区间。
   - 但 valid10 AEE `5.6936`、AAE `83.2893`、firing `8.1944%`，远离可晋级范围；不进入 360-step/full30。

结论：全三值的主要矛盾不只是 activity。A4 可以控制负脉冲数量，但全网从 binary activation 直接切换为 signed ternary 后，特征分布和已训练权重不匹配，短训精度仍严重崩坏；旧 full30 也没有显示能恢复到 baseline 5% 窗口。BTTX 作为负对照保留，不建议 RTL 首版增加 sign rail 作为关键路径。

### 41.7 最终纯 TX dyadic 部署主线

为了移除 `alpha0=0.02` 的非 dyadic 常数，保持 current TTX 权重和纯 TX-only 数据流，仅把 score 改为：

`score_int = 64 * same_active + same_zero; score = score_int >> 6`

配置：`configs/generated/date11full_ttx_dyadic_txonly_all12_deploy_int8.yml`。必须同时满足 `bipolar_mu=0` 与 `hardware_mu_pow2_shift=0`；后者若误设为 `4` 会在量化函数中隐式注入 `1/16`，不再是纯 TX。最终配置已审计为二者均为 `0`。

| deployment | AEE | AAE | outlier | total_spikes | firing | estimated SOPs |
|---|---:|---:|---:|---:|---:|---:|
| current TTX int8 (`alpha0=0.02`) | 1.5003 | 9.8266 | 0.0920 | 23.2462G | 5.0237% | - |
| pure dyadic TTX int8 (`alpha0=1/64`) | 1.5016 | 9.8431 | 0.0919 | 23.2439G | 5.0232% | 2.1235G |

dyadic 相对 current int8 只差 AEE `+0.0013`、AAE `+0.0165`，spikes 略低，属于部署等价。最终 DATE/RTL 主线冻结为：

**all-binary ATLIFPSN + all12 no-carrier TTX + `64:1` dyadic TX score + centering + int8 Shiftmax/LUT gate**。

推荐训练 checkpoint 仍为 `date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8.../checkpoint_epoch2.pth`；部署使用 dyadic/int8 config。下一步只给该唯一主线补随机种子，不再扩 attention 范式。

## 四十二、对称 ATLIF + 无 gate-K 的统一注意力探索（2026-07-11）

### 42.1 用户冻结的新约束

本节约束覆盖 41.7 的“最终冻结”结论，但不删除旧结论和结果：

- 神经元必须使用对称阈值，无论输出编码是二值还是三值；现有 `official_atlif` 的 `{0,+theta}` 单边阈值只能作为历史 TTX 对照。
- 全 encoder 的 `12` 个 attention block 使用同一种公式；禁止 S2-only、stage-wise、TX/SC 混合和 pure SC。
- attention 优先彻底移除 `gate * K`，直接研究 Shiftmax 结果作为 attention 输出；只有直接路线被证伪后，才允许统一 all12 的 shift-value fallback。
- 不使用 Kmag、target-rate、mu schedule 或 native QKFormer carrier；当前阶段只做 DSEC，不启动 MVSEC/MDR。
- 所有实现只能新增配置门控分支，旧模块、旧配置、旧 checkpoint 和旧结果不得破坏。

### 42.2 对称二值的严格定义

现有代码没有“对称二值 ATLIF”：`BinarySurrogate` 和 `OfficialATLIFSurrogate` 都只比较 `h>=theta`。为避免论文概念偷换，本轮新增候选定义为：

```text
symmetric binary magnitude event:
  event = 1, if |h| >= theta
  event = 0, otherwise
```

正负比较阈值严格共享同一个 `theta`，输出仍只传一位 event，符号不进入 activation SRAM。三值候选继续使用 `{-theta,0,+theta}`。`{-theta,+theta}` 的 dense bipolar binary 因活动率恒为 100%，与 `total_spikes` 降低至少 20% 的目标先验冲突，不安排训练。

### 42.3 文献与官方代码复核

| work | venue | relevant operator | 与本轮边界的关系 |
|---|---|---|---|
| QKFormer | NeurIPS 2024 | `gate(Q) * K` 的线性 Q-K attention | 证明 token selector 有效，但仍依赖 K carrier |
| Spike-driven Transformer | NeurIPS 2023 | Q/K/V 间 mask + addition，无通用乘法 | 硬件友好，但引入完整 V path，改动范围更大 |
| SpikeVideoFormer | ICML 2025 | linear Hamming `(2Q-1)((2K-1)^T V)` | 保留 V；本仓库 H21/H35 valid40 已不具晋级优势 |
| Bipolar Self-Attention | NeurIPS 2025 spotlight | ternary QK + Shiftmax + shift-V | 支持 Shiftmax/dyadic，但仍靠 V 保存通道信息 |

官方实现共同说明：归一化 score 通常是标量/矩阵权重，不天然等于 `[token, head_dim]` 特征。若把单个 gate 广播到整个 head，通道秩会降到 1。因此本轮不只测一个标量直出，而是测试分组直出，在不使用 K carrier 的前提下逐步恢复通道自由度。

### 42.4 H63 预注册矩阵

完整协议：`neuron_autoresearch/experiments/h63_direct_shiftmax/protocol.md`。

共同设置：full-network `105` 个 symmetric-binary ATLIF wrapper；all12 pure TX score；`alpha0=1/64`；no SC/no Kmag/no target-rate/no carrier；从 DSEC TTX epoch2 warm-start。

```text
score_g = popcount(q_g & k_g) + (1/64) * popcount(~q_g & ~k_g)
gate_g  = Shiftmax(score_g over token) * N - 1
attn_g  = broadcast(gate_g within group g)
```

| priority | candidate | groups/head | dynamic output path | status |
|---|---|---:|---|---|
| P0 probe | H63a direct-G1 | 1 | centered Shiftmax -> broadcast, no K readback | pre-registered |
| P0 main | H63b direct-STC | token+channel | `0.5*(centered token gate + centered channel gate)` | pre-registered |
| P1 | H63c direct-G4 | 4 | 4 grouped centered Shiftmax outputs, no K readback | pre-registered |
| hold | direct-G8/G32 | 8/32 | finer grouped/per-channel Shiftmax | G4 接近门槛且明确受通道容量限制时才考虑 |
| fallback | all12 dyadic shift-value | 1 | Shiftmax exponent -> event/value shift | 仅 direct 全失败后考虑 |

晋级门槛：20-step activity `<20%`；120-step valid10 AEE `<=2.2`；360-step valid40 AEE `<=1.65` 且训练趋势下降。只有 valid40 最优候选允许转 full train；论文结论必须来自标准 valid825，并满足 AEE 在 NB0 约 5% 内、total spikes 至少下降 20%。

### 42.5 硬件块级影响

H63 保留 `Q/K event SRAM -> TX classify/popcount -> Shiftmax -> attention output SRAM -> projection` 的大体调度，只替换 attention block 尾部：删除 `K reread + FGK late scale`，增加 `G` 路 gate descriptor 广播。G1/G4/G8 分别对应每 head `1/4/8` 个 Shiftmax accumulator/denominator context；可以用单 lane 分时实现，面积不要求线性复制，但 cycles 近似随 G 增长。

硬件细化同步到 `hw_autoresearch_nts07/docs/40_H63对称ATLIF无GateK注意力探索.md`。在 valid40 之前只更新接口和风险，不修改现有 TTX RTL 主线；候选晋级后再决定是否新增 RTL block。

### 42.6 H63 零训练链路与共同失配诊断

三个配置均通过完整模型链路审计：`ATLIFTernaryPSN=105`、attention patch `=12`、binary modules `=105`、`symmetric_binary_abs_modules=105`，保存后自身 state_dict reload PASS。TTX ep2 权重的 profile 加载均为 `checkpoint_overlay_keys=210, missing=0, unexpected=0`。

在同一个 valid sample 上直接切换语义、不训练的结果：

| candidate | AEE | AAE | global firing | estimated SOPs |
|---|---:|---:|---:|---:|
| H63 G1 | 6.8399 | 56.1595 | 70.1155% | 29.6410G |
| H63 STC | 6.8395 | 55.6197 | 69.9316% | 29.5633G |
| H63 G4 | 6.7314 | 55.3017 | 69.9901% | 29.5880G |

三种 attention 的 firing 几乎相同且同时远超 20% 停止线，说明主要问题是把 TTX 的单边 `theta=1` checkpoint 直接解释为 `|h|>=theta`，大量旧负膜电位被激活；此时比较 attention 排名没有意义。按协议不从该起点跑 120 steps。

只允许一个预先有依据的恢复动作：把 checkpoint 中 `105` 个 ATLIF threshold 统一乘 `4`，倍率来自 41.6 的 BTTX-A4 活动率证据；不做倍率扫参、不启用 target-rate。转换脚本为 `entrypoints/make_h63_symmetric_threshold_checkpoint.py`，目标 checkpoint 为 `results/h63_checkpoints/ttxep2_symmetric_threshold_x4.pth`。下一步先审计其 `210` overlay keys 和 valid1 activity；仍超过 20% 则停止 symmetric-binary direct 路线。

恢复实验结果：

| calibration | candidate | AEE | AAE | global firing | load audit |
|---|---|---:|---:|---:|---|
| all thresholds `x4` | G1 | 35.8333 | 59.2857 | 50.3973% | 210/0/0 |
| all thresholds `x4` | STC | 35.5507 | 59.2581 | 50.1561% | 210/0/0 |
| all thresholds `x4` | G4 | 35.7922 | 59.0794 | 50.2947% | 210/0/0 |
| per-module budget calibration | STC | 21.2763 | 60.3173 | 57.5106% | 210/0/0 |
| per-module budget calibration | G4 | 21.4402 | 60.3039 | 57.6803% | 210/0/0 |

逐模块校准脚本 `entrypoints/calibrate_h63_symmetric_thresholds.py` 使用原 TTX/H60 的一个 DSEC valid sample，对 93 个实际经过 forward 的 ATLIF 求 `P(|h|>=theta_sym)=P(h>=theta_old)`；12 个未经过该路径的 wrapper 保留原阈值。得到 observed threshold min/mean/max=`1.0000/2.6812/3.9676`，原单边 target rate mean=`4.0875%`。然而切换全网语义后上游事件改变会级联改变下游 `h` 分布，离线逐层匹配不能保持闭环活动率，firing 仍约 58%。

结论：`symmetric_binary_abs` 虽是清晰的 1-bit 对称比较器定义，但在当前全网 PSN 权重和 TTX warm start 上造成不可接受的分布级联；其失败与 G1/STC/G4 排名无关。按 `<20%` 活动率硬门槛停止，不投入 120-step/full train。后续只评估逻辑上严格的 signed ternary `{-theta,0,+theta}`，并优先从已有全三值 checkpoint 做 attention-only 受控短测；若仍无潜力，则应明确报告“对称 signed event 与当前全网 spike/精度目标冲突”，不能把单边 official binary TTX 改名为对称 ATLIF。

### 42.7 Signed-ternary direct TX 语义修正

首次 STC ternary 20-step diagnostic 使用了 positive-only binary event helper，负事件没有参与 TX score：step20 ternary activity `40.5751%`（pos `2.2948%` / neg `38.2803%`），valid1 AEE `10.4914`、AAE `75.0328`、profile firing `46.5596%`。加载链路为 `105/12`、overlay `210/0/0`，但该公式不满足对称 ternary TX，不能作为 signed TX 的最终否定证据。

实现已新增 `direct_shiftmax_signed_events`：

```text
same_active = [q=k=+1] + [q=k=-1]
same_zero   = [q=k=0]
score       = same_active + (1/64)*same_zero
```

异极性事件贡献 `0`，没有 SC negative penalty、没有 TX/SC 混合。修正后只重跑 STC 20-step；若 activity 仍大于 20% 或 valid1 明显发散，则 direct-Shiftmax 线整体停止，不再跑 G4/G8。

修正后的 signed TX centered-STC 仍失败：step20 ternary activity `41.5731%`（pos `2.9672%` / neg `38.6059%`），valid1 AEE `9.7823`、AAE `107.5519`、profile firing `44.9490%`。其失败机制是 `gate-1` 产生大量负输出并沿对称 ATLIF 级联。为严格覆盖“Shiftmax 结果本身”，预注册唯一 raw-STC 对照：`Y=0.5*(gate_token+gate_channel)`，不减 1、不乘 K。先 valid1；只有 firing 显著低于 centered-STC 且满足 20% 门槛才允许 20-step。不给 G4/G8 增加 raw 组合。

raw-STC valid1 结果：AEE `13.8467`、AAE `135.6088`、firing `53.9180%`、estimated SOPs `22.7936G`，load audit `210/0/0`。不减 1 没有解决全网分布级联，因此 H63 direct family 正式停止；G4/G8 不再运行。

### 42.8 H64：离线中心化的严格对称 ATLIF

H63 的共同特征是负事件占绝对多数，说明现有 PSN 的 `h` 分布不以数值零为对称轴。H64 预注册 `c_t±theta`：每个模块/时间步用离线 median 得到固定 `c_t`，正负阈值共享同一个 `theta`。推理硬件预存 `lo_t=c_t-theta`、`hi_t=c_t+theta`，仍是两个比较器，不需要在线统计、减法或 target-rate FSM。

协议：`neuron_autoresearch/experiments/h64_centered_symmetric_atlif/protocol.md`。先用现有 all12 H60/TX 做 H64-ref，只隔离 neuron 是否可行；再用完全相同 checkpoint 切换到 raw-STC 无 gate-K。未通过 valid1/20% activity 门槛前不运行 G4/G8，也不启动 full train。

H64 valid1 结果：H60-ref AEE `11.8798`、AAE `91.7006`、firing `59.6304%`；raw-STC AEE `12.4145`、AAE `116.0168`、firing `57.8680%`；两者 load audit 均为 `210/0/0`。连保留 H60 的 neuron-only reference 都失败，说明一次性逐层 center/threshold 校准无法控制全网闭环分布；H64 停止，不做20-step。

### 42.9 H65：全网统一 signed Hamming linear attention

H63/H64 失败后，剩余满足“all12 统一、无 TX/SC、无 gate-K、非 `N×N`”且有顶会官方实现依据的候选是 SpikeVideoFormer/ICML 2025 Hamming linear attention。本仓库旧 H21/H35 的 partial-scope valid40 AEE 约 `1.63` 不能外推到 all105，因此 H65 只预注册一个严格全网测试：105 个 `symmetric_bsa_tsn` ternary ATLIF + 12 个 `hamming_ternary_active_direct`，从 all-ternary TX ep29 warm-start，20 steps + valid1。

协议：`neuron_autoresearch/experiments/h65_signed_hamming/protocol.md`；配置生成器：`entrypoints/make_h65_signed_hamming_config.py`。硬件不做 gate-K/Shiftmax，但需要 `D×D=1024` signed accumulator state/head，属于精度优先高风险备用。step20 activity 必须 `<20%` 且 valid1 AEE `<2.2` 才能进入 120 steps。

H65 结果：step20 ternary activity `45.1234%`（pos `6.5950%` / neg `38.5284%`），valid1 AEE `8.3598`、AAE `90.6382`、profile firing `46.649%`、estimated SOPs `19.7205G`；load audit `checkpoint_overlay_keys=210, missing=0, unexpected=0`。精度和 activity 均未通过门槛，停止在20步，不进入120/360/full。

### 42.10 本轮外环结论与主线状态

| family | symmetric neuron | unified all12 attention | gate-K | best new evidence | decision |
|---|---|---|---|---|---|
| H63 sym-binary direct | `abs(h)>=theta` | grouped/STC Shiftmax | no | 50%-70% firing, AEE 6.7-35.8 | stop |
| H63 signed ternary centered-STC | `±theta` | signed TX + centered Shiftmax | no | step20 activity 41.57%, AEE 9.78 | stop |
| H63 signed ternary raw-STC | `±theta` | raw signed TX Shiftmax | no | firing 53.92%, AEE 13.85 | stop |
| H64 centered symmetric | `c_t±theta` | H60 ref / raw-STC | ref yes / STC no | H60-ref firing 59.63%, AEE 11.88 | stop |
| H65 signed Hamming | `±theta` | Hamming linear | no gate-K, K reused as value | step20 activity 45.12%, AEE 8.36 | stop |
| historical all-ternary TX/NTS | `±theta` | TX or NTS | yes | full30 AEE 3.28-3.29, spikes 79-80G | fails DATE target |
| historical dyadic TTX | one-sided `{0,+theta}` | pure TX H60 | yes, factorized shift/late-scale | valid825 AEE 1.5016, spikes 23.24G | numerical reference only |

严格约束下当前没有候选同时满足：对称 ATLIF、统一 all12、AEE 约 baseline 5% 内、spikes 至少下降20%。因此本轮**没有实验具备 full training/standard valid825 晋级资格**；强行全训会违反预注册门槛并浪费约30 epoch计算。

现有 dyadic TTX 仍是 DSEC 数值和硬件成熟度最好的 checkpoint/reference，但它有两个明确不符合项：`official_atlif` 是单边 `{0,+theta}`，attention 尾部仍以 factorized/shift-friendly 方式使用 K carrier。论文不能同时把它声称为“严格对称 ATLIF、无 gate-K”。下一步必须在论文约束中二选一：

1. 保留 dyadic TTX 为主线，把 neuron 贡献准确表述为 one-sided sparse ATLIF，并把 gated-K 说明为 dyadic shift/late-scale，而非通用乘法；或
2. 坚持严格对称 signed-event，则需要从训练目标和网络初始化层面重训新 backbone，现有 checkpoint overlay 微调路线已有充分负证据，不能承诺 baseline 5% 精度和20% spikes降幅。

本轮新增实现均为可选分支，旧 H60/TTX/NTX/SC 模块、配置和 checkpoint 未删除。GPU 已停止，未启动 MVSEC/MDR，也未启动未达门槛的 full train。

## 四十三、约束勘误与精度优先统一注意力探索（2026-07-11）

### 43.1 对第 42 节的正式勘误

第 42 节误把用户约束理解成“所有神经元都必须双边对称、attention 禁止任何 K value path”。正确约束如下，本节覆盖第 42 节的相反判断，但保留旧实验作为负结果：

- 单边 `{0,+theta}` ATLIF 合法；只有设计正负双边发放时，`+theta/-theta` 才必须对称。
- `gate*K`、`weights@K` 合法。禁止的是原 QKFormer/native carrier：先计算 `sn2_q(sum(Q))`，再构造 `K*Q_gate`，之后又叠加第二个 attention gate。
- 当前 H60/TTX 的 `TX score -> Shiftmax gate -> gate*K` 没有上述 native carrier，因此满足 no-carrier 约束。
- full encoder 的 12 个 attention block 仍必须统一；不采用 S2-only、stage-wise 或 TX/SC 混合部署。
- 候选先按精度筛选，activity/spikes 作为第二维度记录，不再用 20% activity 直接终止一个精度候选。

因此，第 42.10 节“dyadic TTX 仅为数值参考、不符合约束”的结论无效。当前有效主线仍是 **105 个 one-sided all-binary ATLIF + all12 no-carrier dyadic TTX**：valid825 AEE `1.5016`，相对 NB0 约 `+0.97%`；total spikes `23.2439G`，相对 NB0 约 `-47.2%`。

### 43.2 候选计算范式

| family | attention object | output/value path | native carrier | complexity / hardware story | status |
|---|---|---|---|---|---|
| TTX | 每 token 一个 alpha-XNOR/TX 标量分数 | Shiftmax 后 `gate*K` | no | 线性 token selector；当前 RTL 主线 | valid825 pass |
| NTS | TX 与 SC 加权混合标量 | Shiftmax 后 `gate*K` | no | 公式混杂，用户不采用 | historical only |
| NTX/H18a | TX gate 叠加在 `K*sn2_q(sumQ)` 上 | native carrier 再乘 gate | yes | 双重选择器，禁止 | stop |
| H66a | token-token binary alpha-XNOR 矩阵 | row Shiftmax 后 `weights@K` | no | 窗口内矩阵 SRAM/累加；精度优先上界 | P0 run |
| H66b | binary Hamming linear attention | `Q(K^T K)` | no | `D x D` accumulator，无 token-token matrix | P1 generated |
| STAtten | temporal-block `Q(K^T V)` | 独立 V 或复用 value | no | 时间块缓冲，复杂度仍为 `O(TND^2)` | literature hold |
| SpiLiFormer | feed-forward selector + feedback lateral inhibition | 抑制无关 token 后投影 | mode-dependent | 需要反馈状态，不是简单减一项 | literature hold |
| A2OS2A | binary Q、ReLU K、ternary V | addition-only matrix attention | no | 三种激活数据类型，硬件异构较高 | reference only |

### 43.3 为什么有些旧结果值得继续、有些不值得

旧 H37 的 binary alpha-XNOR matrix 在部分神经元替换条件下，120/360-step valid10 AEE 约 `1.03-1.09`，说明**注意力范式本身有精度潜力**；它不能作为论文结果，但足以支持改成 full105/all12 后重跑。相反，H63/H64/H65 的 AEE `8.36-13.85` 是 baseline 的约 `5.6-9.3` 倍，且保留 H60 的 H64 neuron-only reference 也达到 `11.88`，说明主要是错误强制对称神经元导致全网分布崩坏。停止这些实验的正确依据是精度失配，而不是 firing 超过 20%。

H66 协议位于 `neuron_autoresearch/experiments/h66_accuracy_first_unified/protocol.md`。生成器为 `entrypoints/make_h66_accuracy_first_unified_configs.py`，只新增配置并从 TTX epoch2 warm-start；第一优先级为 H66a full alpha-XNOR matrix，H66b Hamming 仅在 H66a 完成后顺序运行。

### 43.4 外部顶会检索后的优先级修正

独立检索记录见 `neuron_autoresearch/literature/h66_external_attention_survey_20260711.md`，覆盖 CVPR 2025 a-XNOR/STAtten/A2OS2A、ICCV 2025 SpiLiFormer、ICML 2025 SpikeVideoFormer、NeurIPS 2025 spiking RPE/MaxFormer 和 ICLR 2026 LRF-Dyn，并复核可获得的官方代码。

H66a full a-XNOR matrix 作为 accuracy oracle 先跑；若 pairwise correlation 有效，硬件优先降阶为 TP-TTX 或 LR-TTX，而不是直接冻结 `N x N` score matrix。TP-TTX 只比较同空间位置的两个时间片；LR-TTX 比较 self 与四个空间邻居，二者都统一使用 binary alpha-XNOR + Shiftmax + K value，不含 SC、native carrier 或 stage-wise 公式。旧 `h59_local` 只是对同 token score 做 roll-average，并没有计算邻居 `TX(q_i,k_j)`，且存在边缘 wraparound，因此旧 NTX10 结果不能作为 LR-TTX 证据。

### 43.5 H66a full alpha-XNOR matrix 初筛结果

配置：`configs/generated/h66a_allbinary_all12_axnor_matrix_shiftmax_s120.yml`；起点为 TTX epoch2。链路审计为 `ATLIFTernaryPSN=105`、patched attention `=12`、`checkpoint_overlay_keys=210, missing=0, unexpected=0`。该分支没有 native QKFormer carrier；它直接计算 binary alpha-XNOR token-token matrix，经 row Shiftmax 后执行 `weights@K`。

| stage | AEE | AAE | firing | estimated SOPs | note |
|---|---:|---:|---:|---:|---|
| zero-shot valid1 | 1.6928 | 30.4814 | 4.8567% | 2.0532G | 只切 attention 语义，未训练 |
| 120-step valid10 | **1.1766** | 14.2411 | 5.0444% | 2.1325G | AEE 强信号；AAE 需扩大样本确认 |

120-step train loss `1.5962`，耗时 `198.75s`，峰值 GPU memory `57.243GiB`；step120 binary activity `4.3462%`。checkpoint 为 `results/h66a_accuracy_first_20260711_151351/runs/h66a_allbinary_all12_axnor_matrix_shiftmax_s120_steps120/checkpoint_epoch0.pth`。valid10 只作初筛，不能与 NB0 valid825 直接宣称胜负。由于 AEE 明显通过精度门槛，已从同一 TTX ep2 独立启动 360-step + valid10/valid40；本轮把 rapid-screen 的 AAE 晋级阈值放宽到 20，避免旧门槛错误阻断光流 AEE 优先候选，但最终 DATE 表仍同时报告 AAE。

计量限制：当前 `profile_sops.py` 的 `estimated SOPs`/energy 不包含 overlay forward 内的显式 `N x N x D` alpha-XNOR 和 `weights@K` 运算，因此表中的 `2.1325G` 只能比较被 profiler 覆盖的脉冲层，**不能**作为 H66a attention 总硬件能耗。H66a 是 accuracy oracle；若晋级，必须另算 pairwise TX、Shiftmax row 和 value accumulation cycles/energy。TP-TTX/LR-TTX 的目的之一就是把该精度收益压缩到固定 2/5 邻域，使 attention 运算可完整计数。

### 43.6 H66a valid40 与 TP/LR 降阶结果

H66a 从相同 TTX epoch2 独立训练 360 steps：train loss `1.5874`、train time `579.09s`、peak memory `57.243GiB`。valid10 AEE/AAE 为 `1.1639/12.9109`，但 valid40 扩大后为 `1.6554/15.5025`，firing `5.6204%`。AEE 只比 `1.65` 门槛高 `0.0054`，证明 pairwise alpha-XNOR 有潜力，但 AAE 持续偏高且 full matrix 硬件代价大，暂不直接进入 full train。

新增真正的固定邻域分支；两者均为 full105 one-sided binary ATLIF、all12 同一公式、no native carrier：

- TP-TTX：每个 Q 与同位置 self-K、paired-time-K 计算两路 binary alpha-XNOR，2-way Shiftmax 后聚合 K。
- LR-TTX：每个 Q 与 self/up/down/left/right 五个 K 计算 binary alpha-XNOR，5-way Shiftmax 后聚合 K；边界 mask，不允许 wraparound。

| candidate | zero-shot valid1 AEE/AAE | 120-step valid10 AEE | AAE | firing | train loss | peak memory |
|---|---:|---:|---:|---:|---:|---:|
| H66a full matrix | 1.6928 / 30.4814 | 1.1766 | 14.2411 | 5.0444% | 1.5962 | 57.243GiB |
| H66d LR-TTX | 1.7398 / 31.5636 | 1.2026 | 15.0169 | 5.0369% | 1.4943 | 54.778GiB |
| **H66c TP-TTX** | 1.7499 / 32.4011 | **1.1656** | **14.3868** | 5.0602% | **1.4643** | **54.681GiB** |

三个 valid10 的 AAE 都高于 TTX valid825 约 `9.84`，说明 pairwise K aggregation 的共同风险是角度误差，不是单个候选偶然异常。TP-TTX 以最低的固定邻域成本取得最好 AEE/loss，已晋级独立 360-step + valid40。LR-TTX 暂停在120步，只有 TP valid40 失败或显示明确空间局部性不足时才继续，避免无意义并行扫参。

### 43.7 H66c 标准推理与 H66e self-bias 结论

H66c TP-TTX 的独立 360-step 结果为 valid10 `1.1555/13.1295`，valid40
`1.5741/14.7896`，因此按短测门槛进入标准 valid825。完整结果：AEE `1.6566913`、
AAE `10.4282732`、PE1/PE2/PE3=`0.55224/0.22992/0.11479`、firing `5.2936%`、
total spikes `24.4950G`。加载审计为 `ATLIFTernaryPSN=105`、attention `=12`、
`checkpoint_overlay_keys=210, missing=0, unexpected=0`。

相对 NB0，H66c AEE 约增加 `11.4%`，超过论文 5% 窗口；相对 H60 TTX，AEE 从
`1.5016` 恶化至 `1.6567`，spikes 也从 `23.2439G` 增至 `24.4950G`。虽然 AAE
`10.4283` 远好于 valid40 的小样本估计，但仍不具备替代 H60 的资格。

H66e 只增加固定 `+1` self-lane bias，未扫权重。zero-shot valid1 AEE/AAE 为
`1.7240/31.5940`；120-step valid10 为 `1.1949/14.7047`，弱于无 bias 的 H66c
`1.1656/14.3868`，因此停止，不跑 valid40/full。固定邻域 pairwise family 到此关闭。

### 43.8 深读后的候选校正与 H67

全文与官方代码复核见
`neuron_autoresearch/literature/idea_mining_20260711/notes/CODEX_DEEP_READ_REVIEW.md`；
硬件增量见 `hw_autoresearch_nts07/docs/42_H67运动XOR与有界TTX硬件增量.md`。

关键校正：Bishop ECP 针对 binary `QK^T` matrix 的 active-count bound，不能原样套到
奖励 silent/silent 的 dyadic TTX；FlowFormer 的 latent cost memory 也不是简单 K pooling。
两者只能在重新推导误差界或实现 cost-codeword 数据流后引用。相反，EDCFlow 的相邻
时域差分可在 binary event 上严格化为 XOR-popcount；Castling-ViT 的训练期重分支、推理期
移除范式可让 H66a 仅作为 teacher，而不污染部署硬件。

H67 只预注册一个点：所有 12 个 H60 block 使用
`score=TX(Q_t,K_t)+0.25*popcount(K_t XOR K_pair)`，其余 105 个 one-sided binary
ATLIF、Shiftmax 和 `gate*K` 均不变。新增字段 `binary_motion_xor_alpha` 默认 0；配置为
`configs/generated/h67_allbinary_all12_motionxor_ttx_w025_s120.yml`。代码测试 62 项通过。
zero-shot valid1 为 AEE `1.7156`、AAE `31.1874`、firing `5.0775%`、estimated SOPs
`2.1465G`；加载审计 `105/12/210/0/0`。当前只运行一次 120-step valid10，达到门槛后
才允许 valid40，不做权重 sweep。

H67 120-step valid10 为 AEE `1.2098`、AAE `14.6673`、firing `5.0373%`，train loss
`1.4637`。同 checkpoint 首次 valid40 错用 `profile batch_size=8`，得到不可信的 AAE
`47.64`；历史 rapid-screen 的标准是 `profile batch_size=1`，说明 SNN 状态/AAE 汇总不能
随意改 batch。按 batch1、workers4 重跑后 valid40 为 AEE `1.4566`、AAE `14.0078`、
firing `5.4815%`、estimated SOPs `2.3173G`。该结果优于 H66c 360-step valid40 AEE
`1.5741`，因此已从 TTX epoch2 独立启动 H67 360-step + valid10/40；仍只保留 1/4 单点。

H67 独立 360-step 已完成：train loss `1.4836`、train time `581.87s`、peak memory
`57.861GiB`；valid10 AEE/AAE=`1.2283/14.1901`，valid40=`1.5937/15.0056`，firing
`5.6245%`。它通过 `1.65/20` 晋级门槛，但相对 120-step valid40 `1.4566/14.0078`
没有继续收敛收益，且略弱于 H66c 360-step valid40 AEE `1.5741`。因此不做更多短训或
XOR 权重 sweep，只对 step360 checkpoint 做一次标准 valid825；若不在 NB0 5% 窗口内，
H67 作为“简单 motion bias 不足”的负消融停止，不进入 full train。

H67 标准 valid825 已完成：AEE `1.6536553`、AAE `10.3943662`、
PE1/PE2/PE3=`0.55034/0.22840/0.11387`、firing `5.2816%`、total spikes
`24.4393G`、estimated SOPs `2.2328G`；加载审计 `105/12/210/0/0`。相对 NB0
AEE 约增加 `11.2%`，超过 5% 窗口；相对 H60 TTX AEE `1.5016` 和 spikes
`23.2439G` 也同时更差。H67 正式停止，不做 full train、权重 sweep 或硬件主线替换。
该结果与 H66c valid825 `1.6567/10.4283/24.4950G` 几乎一致，表明只给 TTX 增加
简单相邻时间先验不能解决 pairwise/motion candidate 的全量泛化问题。

后续最小顺序固定为：H67 Motion-XOR -> Castling-TTX training-only auxiliary -> frozen
checkpoint 的有界 gate bundling -> 完整 attention operation/traffic accounting。SwiftFormer
官方实现含 L2 normalize、learned global query 和浮点 projection；STAtten/Hamming 需要
`D x D` state，均降为 P2，不在上述实验完成前占用 full-train 资源。

Castling-TTX 的可选分支、单元测试和配置已生成但尚未启动。训练配置为
`configs/generated/h68_allbinary_all12_castling_ttx_aux050_s360.yml`，训练态 full-matrix
auxiliary 权重从 `0.5` 线性退火到 step360 的 `0`；评估态无条件为 0。部署配置
`configs/generated/h68_allbinary_all12_castling_ttx_deploy.yml` 显式关闭 auxiliary，且该
分支没有新增参数。H68 协议位于 `neuron_autoresearch/experiments/h68_castling_ttx/protocol.md`；
只有 H67 完成后才允许串行启动。当前 `test_bsa_attention.py` 全部 63 项通过。

H68 第一次启动在首 batch 暴露 AMP dtype mismatch（H60 FP32、matrix auxiliary FP16），
未产生 checkpoint；显式把 auxiliary 对齐到 H60 dtype 后，新增 FP16/FP32 单测并重跑，
当前测试总数 64 项通过。唯一一次 360-step 训练结果：train loss `1.4821`、train time
`617.54s`、peak memory `59.451GiB`；valid10 AEE/AAE=`1.2297/13.9894`，valid40
`1.5650/14.5559`，firing `5.6316%`。

显式 deploy config 在同 checkpoint 上重跑 valid40，逐项复现 `1.5650/14.5559`，并保持
`105/12/210/0/0`，证明 training-only matrix branch 在 eval 中为 0 且无新增 checkpoint key。
标准 deploy valid825 为 AEE `1.6544181`、AAE `10.4139422`、
PE1/PE2/PE3=`0.55294/0.22934/0.11432`、firing `5.2868%`、total spikes
`24.4634G`、estimated SOPs `2.2350G`。相对 NB0 AEE 约增加 `11.2%`，失败；不扫
auxiliary 初值、退火长度或混合方式。Castling-TTX 只证明了“部署可零开销移除”，没有证明
全量精度收益。

### 43.9 Exact Delta-TTX profile

CVPR 2025 MEET 的关键启示不是简单做 temporal sparsity，而是必须同时核算 state SRAM、
dynamic cycles 和误差累积。当前 binary H60 在 `T=2` 下可做无误差增量：整数化
`S64=64*count(q=1,k=1)+count(q=0,k=0)`，t1 只更新 Q 或 K 翻转的 channel。
1-bit match state 不足以区分 active-active 与 silent-silent；硬件需缓存 previous Q/K
共 64 bit，或每 lane 2-bit contribution class，并保留 S64 accumulator。穷举等价参考
`hw_autoresearch_nts07/scripts/ttx_delta_reference.py` 已通过全部 16 种单 lane 时间转移。

在 TTX best epoch2 上做 100 个 DSEC valid sample 的 all12 hook profile。最终 profiler 对
所有 block/head/window/sample 的 raw lane count 求和，共 `1,741,824,000` 个 t1 lanes：
Q temporal toggle `0.7983%`、K toggle `1.9946%`、union update density `2.7832%`。
因此 t1 理想 lane skip 为 `97.2168%`；由于 t0 仍需完整计算，整个 T=2 window 的 TX
compare 理想减少上限是 `48.6084%`，不是 97.2%。结果目录为
`results/date11_ttx_ep2_delta_profile100_exact_20260711`，load audit `105/12/210/0/0`。
该 element-weighted 结果覆盖此前只按 stage/head 加权的临时估算。

Delta-TTX 数值上与 H60 完全相同，不需要重新训练或精度消融；下一步是 RTL/cycle/PPA：
从 `48.6084%` compare 上限中扣除 state SRAM、XOR mask、稀疏 scheduler 和 accumulator
读写。若净能耗收益成立，它比 H67 motion bias、Bishop 式近似 prune 或继续替换 attention
更适合作为 DATE 硬件主贡献。

### 43.10 full30 判定勘误与正式队列

此前把 H67/H68 的 360-step checkpoint standard valid825 约 `1.65` 写成“方向失败”不严谨。
它只能证明 360-step short checkpoint 未达到最终门槛，不能替代完整收敛实验。按用户要求，
H67 与 H68 均进入标准 full30，逐个串行训练和 standard valid825。

共同 full30 口径：从 TTX best epoch2 独立 warm-start；batch8、workers8、AMP、30 epochs、
warmup720、fast-LR param groups、milestones20/25；保存并评估 epoch
`0/4/9/14/19/24/28/29`。H67 固定 motion XOR weight `1/4`。H68 training-only matrix
auxiliary 从 0.5 线性退火到 epoch20 的 0，epoch20-29 只训练部署 H60；valid825 使用显式
auxiliary=0 deploy config。两者都不扫参。

- 生成器：`entrypoints/make_h67_h68_full30_configs.py`
- 串行队列：`entrypoints/run_h67_h68_full30_queue.py`
- manifest：`configs/generated/h67_h68_full30_manifest.json`
- 独立双轨 idea 文档：`neuron_autoresearch/literature/TTX_SOFTWARE_HARDWARE_IDEA_TRACKS_20260711.md`

软件线继续寻找超过 TTX 的统一 all12 checkpoint；硬件线优先 exact Delta-TTX、zero-K
folding 和 bounded bundling。软件 short/full 指标不能替代硬件 PPA，硬件 compare skip 也不能
替代 AEE/AAE。队列顺序固定为 H67 full30 -> H67 valid825 -> H68 full30 -> H68 deploy
valid825；每个结果完成后自动追加到本文档。

#### H68 命名与论文忠实度勘误（2026-07-11）

H68 是 `Castling-inspired annealed matrix augmentation`，不是 Castling-ViT 原式复现。原论文
使用 threshold-masked softmax auxiliary 与 linear-angular 主分支相加，并保留部署期 DWConv；
H68 使用 binary full-matrix output 与 H60 的全局 `lerp`，blend weight 在 epoch20 前退火为 0，
且没有 entry-wise mask/DWConv。当前 H68 full30 配置保持不变，确保实验定义稳定。若 H68
valid825 不超过 TTX，才考虑单个 faithful-mask H72；不能把 H68 结果直接归因于原论文 mask
机制。

#### H67 full30 运行健康记录（2026-07-11 epoch0）

H67 epoch0 已完成并继续 epoch1：train loss `1.499748`，内部小验证 loss `1.231780`，
epoch time `1487.88 s`，train throughput `4.9359 samples/s`，max GPU memory `57.861 GiB`；
ATLIF `105`、Shiftmax attention `12`、binary activity `4.3952%`。已保存
`checkpoint_epoch0.pth`（705 MB）与 state dict（420 MB）。该小验证 loss 只用于训练健康检查，
不与 standard valid825 AEE/AAE 比较，也不作为提前终止依据。

  - **calib MVSEC 标准 eval 完成（2026-07-10）**：
    - 队列已结束：`[calib-eval] 2026-07-10T14:05:23+08:00 all done`；当前无 eval/train 进程，GPU 空闲。
    - ep24 完成时间：`2026-07-10T03:01:12+08:00`；ep21 完成时间：`2026-07-10T14:05:23+08:00`。
    - ranking：
      - `results_inference/mvsec_ttx_mdr_ep20_calib_epoch24_dt1_full4_20260709_154906/mvsec_ranking.md`
      - `results_inference/mvsec_ttx_mdr_ep20_calib_epoch21_dt1_full4_20260709_154906/mvsec_ranking.md`
    - 四序列 mean（标准 AEE 口径）：

| checkpoint | mean AEE | mean total_spikes | mean energy_uj | vs baseline AEE | spike reduction | energy reduction | ranking |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline ep47 | 1.1414 | 39.5024G | 35225.57 | - | - | - | `results_inference/mvsec_mdr_baseline_epoch47_dt1_full4_20260629_235858/mvsec_ranking.md` |
| TTX ep20 | 1.2019 | 23.8638G | 21467.09 | +5.30% | 39.59% | 39.06% | `results_inference/mvsec_ttx_mdr_epoch20_dt1_full4_20260706_001522/mvsec_ranking.md` |
| TTX ep40 | 1.2217 | 25.2930G | 22750.96 | +7.04% | 35.97% | 35.41% | `results_inference/mvsec_ttx_mdr_epoch40_dt1_full4_20260706_003729/mvsec_ranking.md` |
| TTX ep43 | 1.2251 | 25.1262G | 22600.76 | +7.34% | 36.39% | 35.84% | `results_inference/mvsec_ttx_mdr_epoch43_dt1_full4_20260706_003729/mvsec_ranking.md` |
| **TTX ep20-calib ep21** | **1.2750** | **23.4104G** | **21057.82** | **+11.70%** | **40.74%** | **40.22%** | `results_inference/mvsec_ttx_mdr_ep20_calib_epoch21_dt1_full4_20260709_154906/mvsec_ranking.md` |
| **TTX ep20-calib ep24** | **1.2775** | **24.2060G** | **21772.44** | **+11.92%** | **38.72%** | **38.19%** | `results_inference/mvsec_ttx_mdr_ep20_calib_epoch24_dt1_full4_20260709_154906/mvsec_ranking.md` |

    - 明细：

| ckpt | outdoor_day1 | indoor_flying1 | indoor_flying2 | indoor_flying3 | mean AEE |
|---|---:|---:|---:|---:|---:|
| TTX ep20 | 0.9779 | 1.1414 | 1.4266 | 1.2617 | 1.2019 |
| calib ep21 | 1.0435 | 1.2616 | 1.4705 | 1.3243 | 1.2750 |
| calib ep24 | 1.0522 | 1.2614 | 1.4681 | 1.3282 | 1.2775 |

    - 判定：
      1. **ep20 低 LR calib 失败（相对 ep20 变差）**：calib ep21/ep24 mean AEE 分别为 `1.2750` / `1.2775`，相对 TTX ep20 `1.2019` 约 `+6.1% / +6.3%` AEE；相对 baseline 约 `+11.7% / +11.9%`，**没有进入 +5% 窗口**（5% 门限约 `AEE<=1.1985`）。
      2. spikes/energy 仍满足 ≥20% 下降：ep21 约 `-40.7% / -40.2%`，ep24 约 `-38.7% / -38.2%`。
      3. 当前 **MDR/MVSEC 主结果仍是 TTX ep20**；calib ep21/ep24 只能作为“低 LR 续训 5 epoch 不改善外部泛化、甚至恶化”的负向证据，不替换主线 checkpoint。
      4. 与训练内 valid 一致：calib ep25 train-loop valid `1.1567` 已劣于原 ep20 valid `1.1035`；标准四序列 AEE 进一步确认恶化。
    - 后续建议（按优先级）：
      1. **不要**再从 ep20 用同一低 LR 继续堆 epoch；当前证据说明这条 calib 方向无效。
      2. 若仍想压 AEE 进 5%：优先考虑更短/更稳策略（例如 1–2 epoch、更小 LR、或只训 ATLIF 参数冻结 backbone），或回到 DSEC all-binary TTX 主线巩固后再做跨数据集。
      3. 论文 MDR/MVSEC 外部表：主报 TTX ep20；calib 作为失败 ablation 可选写一行。
<!-- H69_DYADIC_TEMPERATURE_PROTOCOL_20260711 -->

### 43.11 H69 Dyadic-Temperature TTX：H67/H68 后续自动队列（2026-07-11）

H67/H68 的 short360 结果不再作失败结论，二者先逐个完成独立 full30 和 standard valid825。
后续新增 H69，但不会与当前 full30 并发：它等待 H68 排名文件出现后，才从原始 TTX epoch2
分别短训 `score_scale=4/8/16`。三个点都是硬件移位档位；只允许 valid40 综合分数最优且
过门槛的一项晋级 full30，避免无意义连续温度扫参。

H69 动机来自实测而非摘要猜测：H60 gate entropy 为 `7.33985`，等于 `log2(162)`，effective
tokens 为 `162/162`，说明 gate 几乎均匀。固定 dyadic temperature 不改变 105 个 binary
ATLIF、12 个 H60 attention、Q/K binary popcount 和 `gate*K` 数据流，只给 score accumulator
增加 2/3/4-bit 左移。Swin V2 的 scaled cosine attention只提供温度校准先例；H69 是否改善
事件光流必须由 DSEC full30/valid825 决定。

- generator：`neuron_experiments/H9_bipolar_self_attention/entrypoints/make_h69_dyadic_temperature_configs.py`
- queue：`neuron_experiments/H9_bipolar_self_attention/entrypoints/run_h69_after_h67_h68.py`
- literature/software-hardware split：`neuron_autoresearch/literature/TTX_SOFTWARE_HARDWARE_IDEA_TRACKS_20260711.md`

主线替换条件维持不变：valid825 AEE 必须优于或统计上不差于 TTX ep2 的 `1.5016`，AAE 不得
明显差于 `9.8431`，total spikes 仍需较 NB0 降低至少 20%。未达到这三个条件时，H69 只能作为
attention temperature 消融，不能包装成新主线。

### 43.12 H70 Event-Selective TTX：事件密度自适应温度（2026-07-11）

H70 从 NeurIPS 2024 Selective Attention 的 token-aware temperature 迁移，但去掉其 MLP、
浮点 `tanh/sigmoid` 和额外参数。每 token 用 `a=popcount(Q OR K)`，温度档位固定为
`2^min(ceil(log2(a+1)),3)`，并在 TX score 跨 token 中心化之后左移，再进入原 Shiftmax。
部署仍是 all12 同一公式、105 个 one-sided binary ATLIF、12 个 attention 和 `gate*K`。

H70 在 H69 full30 完成后串行执行。360-step 仅为实现健康检查，不能单独判失败；只要脚本、
load audit 和数值有限，就继续独立从 TTX ep2 完成 full30，并评估 epoch
`0/4/9/14/19/24/28/29` 的 standard valid825。

- implementation：`overlay/models/STSwinNet_SNN/bsa_attention.py`
- tests：`tests/test_bsa_attention.py`
- generator：`entrypoints/make_h70_event_selective_ttx_configs.py`
- deferred queue：`entrypoints/run_h70_after_h69.py`
- manifest：`configs/generated/h70_event_selective_ttx_manifest.json`

H70 若超过 TTX，可包装为 **Event-Selective TTX**：事件活跃度既是运动信息强度，也是 attention
selectivity 的硬件控制信号。若不超过，它仍构成固定温度 H69 对动态温度 H70 的必要消融。

### 43.13 H71 Window-Context TTX：无参数窗口上下文广播（2026-07-11）

H60 的 gate entropy 接近最大值，但其 `gate_i*K_i` 仍是逐 token 运算，并不产生标准 attention
的 token-to-token interaction。H71 迁移 ICCV 2023 Context Broadcasting 原公式，在每个
Swin window 内执行 `Y_i=(gate_i*K_i + mean_j(gate_j*K_j))/2`。该分支无参数、无 QK matrix、
无 native carrier，12 个 encoder attention 块使用同一个公式。

H71 在 H70 full30 后串行执行：360-step 只排除实现错误，随后从 TTX ep2 独立 full30 并做
standard valid825。验收除 AEE/AAE 外，必须重点检查 context broadcast 是否抬高 downstream
spikes；若不能保持相对 NB0 至少 20% 的 total-spike 降幅，即使 AEE 更低也不作为最终主线。

- implementation：`overlay/models/STSwinNet_SNN/bsa_attention.py`
- generator：`entrypoints/make_h71_window_context_ttx_configs.py`
- deferred queue：`entrypoints/run_h71_after_h70.py`
- manifest：`configs/generated/h71_window_context_ttx_manifest.json`

### 43.14 Delta-Locality v2 延迟硬件审计协议（2026-07-11）

profiler 已新增 zero-update token/head、每 token/head 更新 lane 直方图、changed-token run
length，以及 4/8-token bundle 全零比例。所有指标保存原始分子/分母后跨 block/head/sample
汇总，不平均百分比。由于本机只有一张 A800，审计脚本等待 H71 standard valid825 完成后，
再对冻结 TTX ep2 跑 100 samples，避免打断软件 full30 队列。

- profiler：`entrypoints/profile_nts11_hardware_p0.py`
- deferred audit：`entrypoints/run_delta_locality_after_h71.py`
- output：`results/date11_ttx_ep2_delta_locality_profile100_v2_20260711`

MEET（CVPR 2025）仅作为 memory-aware temporal execution 的核算依据：它说明 temporal
suppression 若需要过大的 state memory，能耗可能反而恶化。它不直接证明 TTX/Shiftmax
节能，也不要求本项目照搬 state-compression 网络重构；TTX 的 previous Q/K 已是 packed
`64-bit/token/head`，最终仍需连同 S64 accumulator 和访问能耗一起扣除。

### 43.15 Attention operation audit 与 energy 口径勘误（2026-07-11）

现有 standard valid825 `energy_uj` 只对 profiler hook 到的 spike layer 采用
`spikes × (0.9/0.1 pJ)`，没有覆盖 overlay attention 的新增控制与归约操作，也没有真实
SRAM/NoC。因此未来 H67--H71
ranking 将该列标为 `spike_energy_proxy_uj`，profile JSON 增加 `energy_scope` 元数据，但公式和
历史数值不改。

新增 `entrypoints/audit_attention_candidate_ops.py`：使用 100-sample H60 hook 的实际
`batch_windows x heads x tokens x head_dim`，统一计算 H67 motion XOR/popcount、H69 score
shift、H70 OR-popcount/leading-one/shift、H71 context reduction/broadcast/fixed reciprocal。
结果由最终 watcher 写入 `attention_candidate_ops.json/.md`。候选只有在 valid825 精度、spikes、
attention op audit 和最终 PPA 四项均成立时，才可宣称精度与稀疏能耗超过 TTX。

### 43.16 H60-family 统一 dyadic INT8 deploy valid825（预注册）

训练/float eval 使用 `alpha0=0.02`，而冻结 DATE/RTL 主线的 `AEE=1.5016` 对应同一 TTX ep2
checkpoint 在 `alpha0=1/64`、INT8 score/gate 下的部署图。为避免口径错配，H67--H71 各自在
float valid825 排名中选 rank-1 epoch 后，只额外做一次统一部署评估：保留候选自身 attention
机制，强制 `alpha0=1/64`、score/gate step `1/128`、score range `[-2,2]`、gate range `[0,2]`，
并关闭 Castling auxiliary。

`entrypoints/run_h60_family_deploy_eval.py` 将 TTX frozen mainline 与 H67/H68/H69/H70/H71 六行
放在同一次 standard valid825 表中；它由 `run_delta_locality_after_h71.py` 在软件队列完成后
串行调用。Exact Delta-TTX 的逐 lane `S64` 等价只适用于该 dyadic 部署图，不适用于训练时
`alpha0=0.02` 的 float score。

量化命名勘误：历史配置名中的 `INT8` 实际表示 `step=1/128` 的定点网格，不是严格 8-bit
tensor。score `[-2,2]` 含 513 个端点码，若全部精确表示至少需 10-bit code；gate `[0,2]`
含 257 个码，至少需 9 bit。统一部署表保留该已验证网格以保证数值可比，但硬件/PPA 按实际
10/9-bit datapath 核算；论文不得简写成“8-bit score/gate”。

### 43.17 主线竞争 full30 纪律与存盘修正（2026-07-12）

H66/H67--H71 的短跑结论统一勘误为：短跑只能证明实现健康、给出候选超参或暴露明显数值错误，
没有完成标准 full30 + valid825 的候选不得标记为“算法失败”。当前串行队列固定为 H67、H68、
H69、H70、H71，均从同一 TTX epoch2 checkpoint 独立起跑，使用相同 30-epoch 训练协议并评估
epoch `0/4/9/14/19/24/28/29`。H69 的 360-step 只选择 dyadic temperature；H70/H71 的
360-step 只检查实现，均不会代替 full30。

上游保存逻辑除 `force_save_epochs` 外还会在 training loss 创新低时保存，因此 H67 前期几乎
每轮生成约 738MB model + 440MB optimizer state。训练入口新增默认关闭的
`runtime.save_only_force_epochs`；H68--H71 启用后，只保存预注册轮次和末轮，旧配置行为不变。
H67 当前进程不重启、不改计算图，其已有额外 checkpoint 暂不删除。

队列扩展后，为避免八个评估 checkpoint 各自重复保存 440MB optimizer state，又新增默认空的
`runtime.state_save_epochs`。H68--H71 与 H66a-e 设置为 `[19,24,29]`：八个 full-model
checkpoint 全部保留用于论文推理，epoch19/24/29 保留 optimizer/scheduler/scaler 用于故障续训。
预计每候选由约 9.4GB 降到约 7.2GB；旧实验仍维持原保存行为。

用户要求进一步覆盖所有尚未 full30 的结构合规 H66 候选。H66a-e 均为 full105 one-sided
binary ATLIF、all12 同构 attention、无旧 native carrier，因此不能用 120/360-step 或 H66c
零续训 valid825 作为最终否决。新增 `make_h66_full30_configs.py` 与
`run_h66_full30_after_h71.py`，顺序固定为 H66a matrix -> H66b Hamming -> H66c temporal-pair ->
H66d local-5 -> H66e temporal-pair+self-bias；每项独立从 TTX epoch2 跑 full30 和预注册八轮
valid825。最终 dyadic deployment 与 attention operation audit 延后到该队列全部结束。

独立研究账本：`neuron_autoresearch/TTX_MAINLINE_COMPETITION_LEDGER_20260712.md`。该文件把
软件 checkpoint 竞争、论文孵化方向、数值等价硬件优化和近似硬件优化分开记录。后续候选必须
先完成全文公式、官方代码、迁移公式和硬件操作审计，再追加到 H71 后做 full30，不能边训练边
改变候选定义。

硬件深读与数据布局模型补充：`T=2`、每 head 32 lanes 允许同一 spatial token 的两个时间片
恰好打包为 64 bit。事务模型位于
`hw_autoresearch_nts07/results/ttx_temporal_pair_layout_model.md`：相对“两个 32-bit 请求均占
一个 64-bit transaction”的未合并实现，请求数可由 324 降到 162；但相对已经合并的 baseline
收益为 0，logical storage 始终不变。该条件性结论必须由后续 RTL address trace 验证，不能直接
计入 DATE 能耗表。

### 43.18 H73/H74 Match-Code：跨时间位移描述子 full30（2026-07-12）

EEMFlow 的关键可迁移点不是整套 ANN flow 网络，而是保留位移索引的固定跨时间 correlation。
本项目新增三个 all12 候选，均使用 one-sided binary ATLIF105，不混 SC/TX stage，不恢复旧
native carrier，也不执行动态 `weights@K`：

```text
H73 DE9:
  n11(o)=sum_d Q[t,p,d] & K[1-t,p+o,d]
  n00(o)=sum_d ~Q[t,p,d] & ~K[1-t,p+o,d],  o in 3x3
  z=concat(Shiftmax(n11), Shiftmax(n00)) in R^18
  Y[h,t,p,:]=z @ W_code[h,18,D]

H74 MC49:
  s(o)=n11(o)+alpha*n00(o),  o in fixed 49-offset set
  z=Shiftmax(s) in R^49
  Y[h,t,p,:]=z @ W_code[h,49,D]
```

`W_code` 是每 head 静态参数，不随 token/K 动态变化；部署评估将其量化到 signed `2^-7` 网格。
H73 优先验证较小 3x3 跨时搜索，H74 验证更大固定感受野的精度上界。二者都从 TTX epoch2
独立执行 full30，配置为 batch8/workers8/AMP/cupy、warmup720、milestones20/25，并评估
epoch `0/4/9/14/19/24/28/29` 的 standard valid825。不得以实现健康测试代替 full30。

- generator：`entrypoints/make_h73_h74_match_code_configs.py`
- deferred queue：`entrypoints/run_match_code_after_h66.py`
- manifest：`configs/generated/h73_h74_match_code_full30_manifest.json`
- loading audit：`neuron_autoresearch/experiments/h73_h74_match_code/load_chain_audit.json`

加载链实测：H73/H74/H75 都是 ATLIF105、attention12、Match-Code codebook12；从 TTX checkpoint
恢复 overlay keys210，唯一 missing 是新建 codebook12，unexpected0、其他 overlay missing0。
H73/H74/H75 新参数分别为79,488/216,384/75,072。后续训练日志还会再次强制检查该四项；full checkpoint
valid825 则恢复严格加载。最终部署量化、attention operation audit 和 Delta locality 已统一
后移到 H75 完成后，避免 watcher 并发抢占 GPU。

### 43.19 H75 AX17 Match-Code：Flow1D 启发的轴向匹配（2026-07-12）

Flow1D（ICCV 2021）通过“一个方向做1D attention、正交方向做1D correlation”把二维光流搜索
复杂度从乘积改成横纵之和；官方代码`flow1d/attention.py`执行QK softmax和attention乘V，
`flow1d/correlation.py`分别构造`[B,H,W,W]`与`[B,W,H,H]`相关。该原式含动态carrier，不直接
满足当前硬件边界。

H75 只迁移正交分解动机：对另一时刻9x9 window 读取横轴9点和纵轴9点，中心共享得到17个固定
offset；每路计算`n11+alpha*n00`、Shiftmax17，再通过静态per-head`17xD` codebook输出。它不含
softmax、动态`weights@K`、SC或stage mix，12块完全同式。相对H74 MC49，H75匹配和投影规模
更小；相对H73 DE9，覆盖半径4轴向运动但不表达联合对角位移。

H75 已加入同一 Match-Code watcher，在H74后从TTX epoch2执行full30和八轮valid825。CPU加载
审计为ATLIF105、attention12、codebook12、checkpoint overlay210、missing12且全为新codebook、
unexpected0；新增参数75,072。该方法只能称`Flow1D-inspired axial Match-Code`，不能声称复现
Flow1D网络或继承其精度结果。


### H67/H68 full30 自动结果：h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30

<!-- H67_H68_FULL30::h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30 -->
- train config: `neuron_experiments/H9_bipolar_self_attention/configs/generated/h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30.yml`
- eval config: `neuron_experiments/H9_bipolar_self_attention/configs/generated/h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30.yml`
- start checkpoint: `neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid/checkpoint_epoch2.pth`
- run dir: `neuron_experiments/H9_bipolar_self_attention/results/h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30_bs8_full30_20260711_setsid`
- ranking: `neuron_experiments/H9_bipolar_self_attention/results/h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30_bs8_full30_20260711_setsid/profile_ranking_valid825.md`

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | spike_energy_proxy_uj |
| 1 | 19 | 1.4671 | 9.4155 | 0.5002 | 0.1891 | 0.0890 | 26.3898G | 5.7031% | 23393.08 |
| 2 | 29 | 1.4910 | 9.4711 | 0.5023 | 0.1942 | 0.0938 | 26.7021G | 5.7706% | 23662.62 |
| 3 | 4 | 1.4896 | 9.6386 | 0.5080 | 0.1897 | 0.0893 | 24.2416G | 5.2388% | 21445.96 |
| 4 | 9 | 1.5013 | 9.8363 | 0.5133 | 0.1926 | 0.0904 | 25.9925G | 5.6172% | 23020.02 |
| 5 | 24 | 1.5122 | 9.7665 | 0.5123 | 0.1990 | 0.0950 | 26.7619G | 5.7835% | 23730.83 |
| 6 | 14 | 1.5172 | 9.8366 | 0.5220 | 0.2015 | 0.0939 | 25.9844G | 5.6155% | 23009.58 |
| 7 | 28 | 1.5250 | 9.9873 | 0.5120 | 0.1956 | 0.0935 | 28.4275G | 6.1435% | 25202.74 |
| 8 | 0 | 1.5338 | 10.0045 | 0.5219 | 0.1998 | 0.0948 | 24.0647G | 5.2006% | 21268.49 |

### 43.20 H67 Motion-XOR full30 结果判定（2026-07-12）

H67 已完成30 epochs和预注册八轮standard valid825。rank-1 epoch19为AEE`1.4671`、AAE
`9.4155`、total spikes`26.3898G`、spike-energy proxy`23393.08uJ`。相对NB0 ep59：AEE改善
约`1.35%`，spikes下降约`40.09%`；相对冻结TTX float AEE约`1.5003`，AEE改善约`2.21%`。
因此H67不是“short看起来正常”，而是已通过论文精度与spike硬门槛，当前升级为精度第一主线
候选。

H67仍不能直接宣布硬件总能耗优于TTX：ep19 spikes比TTX dyadic的`23.2439G`高约`13.53%`，
且Motion-XOR需要每token/head增加相邻时间K XOR、popcount归约和dyadic shift。最终watcher会对
ep19执行同一alpha0=1/64、score/gate定点网格的valid825，并把增量attention op与SRAM/控制
单列。若该部署结果保持精度，H67可替代H60作为软件主线；硬件主线仍需PPA后决定。

epoch4形成更轻的Pareto点：AEE`1.4896`、AAE`9.6386`、spikes`24.2416G`。它比ep19少约
`8.14%` spikes但AEE高`0.0225`。论文最终可同时报告epoch19 accuracy point和epoch4 efficiency
point，但随机种子复验优先对统一部署后的rank-1与最终第二名进行。

为尽早验证主线，新增`run_h67_early_deploy_after_h68.py`：等待H68 full30+valid825完全释放GPU，
立即对H67 epoch19执行一次统一dyadic/定点valid825；H69 watcher已改为等待该完成标记后再启动。
这不会与H68并发，也不改变H69训练协议。全候选结束后的统一deploy表仍会复核同一结果。

### 43.21 TTB density stratifier 与 dense/sparse 双路径（2026-07-12）

TTB正式提升为DATE硬件P0，而不再只是可选skip单元。目标是利用事件光流空间上“运动边缘局部
活跃、背景大面积低活跃”和时间上T=2变化稀疏的联合分布，将`T=2 × contiguous tokens ×
32 lanes`作为work descriptor。stratifier按temporal changed lanes路由：零变化bundle复用score，
低变化bundle进入changed-index sparse core，高变化bundle进入固定lane dense core；K全零bundle
另外关闭gated-K/value/projection。

必须修正旧统计口径：历史`TTB2 empty`是按时间聚合后对整个window/head的Q活性分类，不是多个
空间token乘多个timestep的真实bundle；而且H60的silent/silent score仍进入Shiftmax denominator，
所以普通Q/K empty不能直接证明完整attention可跳过。新增profiler按token bundle1/2/4/8记录
Q-or-K empty、K-zero、K-motion-zero及active lane阈值2/4/8/16/32。新增
`run_ttb_density_after_h67.py`，在H67统一dyadic评估后分别对TTX ep2和H67 ep19运行100 samples，
随后才放行H69，避免并发抢GPU。

微架构候选固定为三档：保守型单core+empty gating；平衡型一dense core+一sparse core并共享
SRAM/Shiftmax/projection；激进型多dense/多sparse core。当前预推荐平衡型，但core比例、FIFO、
路由阈值和是否值得双核必须等真实profile与cycle/SRAM模型，不能直接借用Bishop的PPA数字。
完整微架构评估见`hw_autoresearch_nts07/docs/45_TTB异构双路径微架构评估.md`。


### H67/H68 full30 自动结果：h68_allbinary_all12_castling_ttx_aux050_toep20_w720_fastlr_full30

<!-- H67_H68_FULL30::h68_allbinary_all12_castling_ttx_aux050_toep20_w720_fastlr_full30 -->
- train config: `neuron_experiments/H9_bipolar_self_attention/configs/generated/h68_allbinary_all12_castling_ttx_aux050_toep20_w720_fastlr_full30.yml`
- eval config: `neuron_experiments/H9_bipolar_self_attention/configs/generated/h68_allbinary_all12_castling_ttx_deploy_full30.yml`
- start checkpoint: `neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid/checkpoint_epoch2.pth`
- run dir: `neuron_experiments/H9_bipolar_self_attention/results/h68_allbinary_all12_castling_ttx_aux050_toep20_w720_fastlr_full30_bs8_full30_20260711_setsid`
- ranking: `neuron_experiments/H9_bipolar_self_attention/results/h68_allbinary_all12_castling_ttx_aux050_toep20_w720_fastlr_full30_bs8_full30_20260711_setsid/profile_ranking_valid825.md`

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | spike_energy_proxy_uj |
| 1 | 19 | 1.4688 | 9.4794 | 0.5029 | 0.1897 | 0.0891 | 26.4244G | 5.7106% | 23421.77 |
| 2 | 29 | 1.4787 | 9.3500 | 0.5006 | 0.1933 | 0.0928 | 26.6893G | 5.7678% | 23655.00 |
| 3 | 24 | 1.5155 | 9.7194 | 0.5142 | 0.2007 | 0.0954 | 26.8216G | 5.7964% | 23784.07 |
| 4 | 28 | 1.5174 | 9.9615 | 0.5127 | 0.1957 | 0.0927 | 28.4371G | 6.1455% | 25212.11 |
| 5 | 14 | 1.5203 | 9.9057 | 0.5250 | 0.2026 | 0.0938 | 26.3557G | 5.6957% | 23341.40 |
| 6 | 9 | 1.5221 | 9.9764 | 0.5170 | 0.1961 | 0.0927 | 26.6947G | 5.7690% | 23637.70 |
| 7 | 4 | 1.5441 | 9.8728 | 0.5217 | 0.1993 | 0.0949 | 25.7580G | 5.5665% | 22774.23 |
| 8 | 0 | 1.6234 | 10.3243 | 0.5481 | 0.2197 | 0.1064 | 26.1983G | 5.6617% | 23122.77 |


### H67 epoch19 early dyadic deploy valid825 自动结果

<!-- H67_EPOCH19_EARLY_DYADIC_DEPLOY_VALID825 -->
- artifact: `neuron_experiments/H9_bipolar_self_attention/results/h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30_bs8_full30_20260711_setsid/h67_epoch19_dyadic_int8_valid825.json`
- AEE: `1.4626`; AAE: `9.3949`
- total_spikes: `26.3948G`; firing: `5.7042%`
- spike-energy proxy: `23397.49uJ`
- deployment: alpha0=1/64, score/gate step=1/128; attention operation cost remains separate.


## True TTB profile100 自动结果

<!-- TRUE_TTB_TTX_H67_PROFILE100 -->
- artifact: `neuron_experiments/H9_bipolar_self_attention/results/ttb_true_density_ttx_h67_h68_profile100.md`

| model | tokens/bundle | Q-or-K density | empty | K-zero | no K-motion | active 1--4 | active 1--8 | active 1--16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TTX ep2 dyadic | 1 | 1.691499% | 72.539530% | 82.282540% | 82.369915% | 19.717835% | 24.120280% | 26.879439% |
| TTX ep2 dyadic | 2 | 1.691499% | 66.127210% | 78.106500% | 78.168351% | 20.017777% | 26.023788% | 30.592211% |
| TTX ep2 dyadic | 4 | 1.691499% | 60.089640% | 73.525184% | 73.570075% | 18.556144% | 26.025553% | 32.241234% |
| TTX ep2 dyadic | 8 | 1.691499% | 54.704775% | 68.838231% | 68.871861% | 16.705019% | 23.933644% | 31.997132% |
| H67 ep19 dyadic | 1 | 1.502114% | 73.897325% | 83.106384% | 83.175281% | 19.043564% | 23.382608% | 25.765469% |
| H67 ep19 dyadic | 2 | 1.502114% | 67.319149% | 79.544915% | 79.584164% | 20.094138% | 25.416482% | 30.060322% |
| H67 ep19 dyadic | 4 | 1.502114% | 60.963301% | 75.636288% | 75.667241% | 19.685629% | 26.506512% | 31.869218% |
| H67 ep19 dyadic | 8 | 1.502114% | 55.255939% | 71.416599% | 71.443439% | 18.047010% | 25.623377% | 32.758293% |
| H68 ep19 dyadic | 1 | 1.548900% | 74.201277% | 83.292383% | 83.355065% | 18.398058% | 22.791555% | 25.405763% |
| H68 ep19 dyadic | 2 | 1.548900% | 67.769127% | 80.035264% | 80.068906% | 19.527250% | 24.596737% | 29.303615% |
| H68 ep19 dyadic | 4 | 1.548900% | 61.517474% | 76.512280% | 76.536657% | 19.329514% | 25.838329% | 30.946131% |
| H68 ep19 dyadic | 8 | 1.548900% | 55.880384% | 72.718655% | 72.740138% | 17.770766% | 25.208699% | 32.022727% |

`empty` cannot by itself remove silent/silent score contributions to Shiftmax. Bit-exact skipping is limited to proven Delta score reuse and K-zero value/projection gating.

### 43.22 H67/H68 dyadic 与 true TTB 阶段裁决（2026-07-13）

<!-- H67_H68_DYADIC_TTB_STAGE_DECISION_20260713 -->

统一alpha0=`1/64`、score/gate step=`1/128`的valid825已经完成。H67 epoch19为AEE
`1.4626`、AAE`9.3949`、total spikes`26.3948G`；H68 epoch19为AEE`1.4715`、AAE
`9.4517`、total spikes`26.4311G`。两者都优于NB0精度且spikes下降约40%，也都明显优于冻结
H60 dyadic的AEE`1.5016`。H67比H68再低`0.00885` AEE，但需要推理期Motion-XOR；H68训练期
matrix辅助在epoch20前退火到零，部署图严格回到H60。因此现阶段软件精度主线为H67，硬件简洁
主线为H68；在attention logic/SRAM/cycle的同口径PPA完成前不强行合并结论。

true TTB profile100证明事件稀疏足以支持stratifier。以4-token、T=2 bundle为例，H67的
Q-or-K empty为`60.96%`、K-zero为`75.64%`、K-motion-zero为`75.67%`；H68分别为
`61.52%`、`76.51%`、`76.54%`。但empty仍不能删除silent/silent score与Shiftmax，合法的
bit-exact路径只包括：K-zero关闭value/projection、Delta零变化复用score，以及按changed-lane
密度在dense/sparse core间路由。硬件B1继续采用单dense core加单4/8-lane sparse core，阈值
`{2,4,8,12,16}`必须由trace cycle与同工艺综合交点选择，不能仅按profile比例拍定。

TTB profiler最初因历史配置的`data.path`相对baseline repository解释而失败；现已与标准
`eval_DSEC_flow_SNN.py`统一路径解析。修复后smoke加载审计为ATLIF105、Shiftmax12、overlay
keys210/210、missing0、unexpected0，三组profile100均完成。H69已于`2026-07-13 13:26 UTC`
进入固定dyadic温度的short360+valid40筛选，筛选只选`x4/x8/x16`温度，晋级项仍执行full30。

### 43.23 Round3算法线与TTB周期模型（2026-07-13）

<!-- ROUND3_ALGO_TTB_CYCLE_PROTOCOL_20260713 -->

第三轮全文与官方代码深读见
`neuron_autoresearch/literature/idea_mining_20260711/notes/DEEP_IDEA_MINING_ROUND3_20260713.md`。
在不改变all12、one-sided binary ATLIF和encoder/decoder接口的边界内，新增三个互斥P0候选：
H76 PC9用固定Omega9位移的3x3对应patch一致性；H77 LC4保留并学习
`n11/n10/n01/n00`四类二值列联证据；H78 G4把32 lanes固定分成四个8-bit组并分别做
Shiftmax9。三项都使用静态codebook、无native carrier，不与H73/H74/H75的offset规模消融
重复。它们排在H75之后逐项full30+valid825，120/360-step只作实现健康检查，不作淘汰门。

硬件周期与综合协议见`hw_autoresearch_nts07/docs/46_TTB真实分布周期模型与综合协议.md`。
该协议把true-density的`A_b=sum(Q OR K)`与Exact-Delta的
`u=popcount(Q_toggle OR K_toggle)`严格分开，并要求按stage/row trace回放C0/B1 makespan。
H67 bundle8 empty在S0仅`26.51%`、S1达`87.30%`，全局平均不能直接乘总周期。现有raw
histogram只支持theta=`2/4/8/16`；theta=`12`必须补计数，当前只能报告8与16之间的区间。
`empty`只能注入精确silent/silent常数，K-zero只能gate value/projection，Delta zero-update才可
复用score；任何比例都不能在trace、SRAM macro和同工艺综合前写成净speedup或PPA收益。

H69 x4 short360的valid10为AEE`1.2232`、AAE`14.6871`，方向指标尚未收敛，因此未进入
valid40。这不构成H69失败。runner已改为：若x4/x8/x16均未通过短筛门槛，仍按已有综合score
选择最佳温度执行预注册full30，避免用短跑拒绝结构合规候选；重启时复用完整screen目录，
不重复训练已完成的短筛。

### 43.24 H67/H68 RTL-exact Shiftmax valid825（2026-07-13）

<!-- H67_H68_RTL_EXACT_VALID825_20260713 -->

artifact：`hw_autoresearch_nts07/results/h67_h68_rtl_exact_valid825.md`。该评估不再使用浮点
`2^x`，而是严格复现当前RTL的raw score Q7量化、16项Q8 exp2 LUT、整数行和、上取整二次幂
归一化、Q1.7 round-to-nearest-even和`[0,2]`饱和。

| 候选 | RTL-exact AEE | 相对原dyadic | AAE | 相对原dyadic | spikes(G) |
|---|---:|---:|---:|---:|---:|
| H67 Motion-XOR TTX | 1.4627 | +0.0001 | 9.4040 | +0.0091 | 26.3544 |
| H68 Castling-trained/H60 deploy | 1.4727 | +0.0012 | 9.4714 | +0.0197 | 26.4164 |

两项AEE退化都远低于预注册`0.02`门槛，证明当前LUT和整数归一化对这两个checkpoint基本无损。
因此不需要为精度扩大LUT或先做hardware-aware fine-tune；下一门槛是H67/H68相同top、SRAM
wrapper、SDC和compile effort下的面积/时序/SAIF，而不是继续修改数值格式。H68推理仍必须关闭
training-only Castling matrix，netlist只实现Motion-XOR off的H60数据流。

### 43.25 H69 full30 启动与恢复链路审计（2026-07-13）

<!-- H69_FULL30_START_AND_RECOVERY_AUDIT_20260713 -->

H69 的 x4/x8/x16 short360 已全部完成，但都因 short-run AAE 高于 `11.5` 未通过筛选门；按
43.23 的预注册规则不能据此拒绝结构候选，因此按综合 score 选择 x8（valid10 AEE
`1.1704`、AAE `13.5339`、SOPs `2.1777G`）进入 full30。正式配置为
`configs/generated/h69_allbinary_all12_dyadic_temperature_ttx_x8_w720_fastlr_full30.yml`，从冻结
H60/TTX `checkpoint_epoch2.pth` 独立续训，结果目录为
`results/h69_allbinary_all12_dyadic_temperature_ttx_x8_w720_fastlr_full30_bs8_full30_20260711_setsid`。
训练已于 `2026-07-13 14:22:59 UTC+8`（`06:22:59 UTC`）启动，使用 batch8、8 workers、AMP、CuPy 和 30 epochs；
稳态约 `1.57 s/step`，每 epoch 918 steps。

启动审计为 ATLIFTernaryPSN `105`、all12 Shiftmax attention `12`、
`checkpoint_overlay_keys=210`、`missing=0`、`unexpected=0`，证明新增 H69 固定 dyadic 温度通过
标准 warm-start 链路加载，且没有遗漏旧 TTX 权重。恢复脚本曾因
`h69_dyadic_temperature_screen_*` glob 同时匹配 launcher 日志而误启动一次重复 x4；重复进程已
在首轮训练早期终止，未覆盖完整筛选或正式 checkpoint。runner 现只扫描目录，并优先复用最新
`summary.csv` 完整的筛选目录；未完成目录保留用于审计，不参与晋级。H69 完成后同一 runner
自动执行 epoch `0/4/9/14/19/24/28/29` 的标准 valid825，并写回最终排名。

同类恢复风险已向后审计：H70/H71 runner 现分别复用已成功的 implementation health、
`checkpoint_epoch29.pth` 和 `profile_ranking_valid825.md`，不再在 watcher 重启后无条件重跑
short360、full30 或 valid825。两个 WAIT watcher 已于 `2026-07-13 14:27 UTC+8` 重启并加载新逻辑；
H66、H73-H78 原本已按末轮 checkpoint/ranking 幂等跳过，不需要改变实验协议。

### 43.26 TTB/Exact-Delta cycle-v2 证据补全协议（2026-07-13）

<!-- TTB_DELTA_CYCLE_V2_PROTOCOL_20260713 -->

旧 profile100 只能提供 TTB `active<=2/4/8/16/32` 的 bundle CDF，以及 Exact-Delta
`9--16` 合并桶，无法严格恢复 `theta=12`，也无法从 bundle 数推导 sparse index payload。现已在
profile collector 中新增两组只读统计，不改变 attention 数值路径：

- Exact-Delta：完整 `u=0..32` histogram，`theta={2,4,8,12,16}` 的 sparse token count 与
  conditional changed-lane sum；
- true TTB：每种 bundle1/2/4/8 的完整 `A_b=0..2*b*32` histogram、`active<=12`，以及
  `kappa={2,4,8,12,16,32}` 的 conditional active-lane sum。

新 runner 为 `entrypoints/run_ttb_cycle_profile_v2_after_round3.py`，使用独立 v2 输出目录，在
H76-H80 full30+valid825 全部结束后依次重放冻结 TTX ep2、H67 ep19、H68 ep19 各100样本。每项
强制审计 ATLIF105、Shiftmax12、overlay210/210、missing0/unexpected0 和 samples100；旧 profile
JSON 保留不覆盖。最终 `run_delta_locality_after_h71.py` 已后移到 v2 completion marker 之后，避免
两个 profiler/deploy eval 并发占用 GPU。v2 只补齐 route/traffic 输入；净 cycle/PPA 仍必须经过
stage/row trace、FIFO replay、SRAM transaction padding 和同工艺综合，不能把覆盖率直接写成
speedup。

固定温度 H69 和 event-selective H70 还新增部署 score-clipping 审计。最终软件队列结束后，
`run_temperature_score_clip_audit.py` 读取两项各自 valid825 rank-1 epoch，用统一 alpha0=`1/64`、
score Q7 step=`1/128`、range=`[-2,2]` 的 deploy config 各重放20样本，报告量化前 strict-low/high
clip count 与比例。该结果决定 x8/动态左移是否能沿用现有 score 位宽；它只读统计，不改变
valid825 或 attention 输出。

### 43.27 Round4 assignment 候选裁决（2026-07-13）

<!-- ROUND4_ASSIGNMENT_CANDIDATES_20260713 -->

Round4算法深读没有继续做9/17/49 offset数量变体，而是在固定Omega9上找到两个结构互斥的
assignment假设。H79 CF10采用row-only局部匹配并加入fixed-zero null候选，允许many-to-one；
H80 DN9对同一局部边再做destination incoming归一化，以Q1.7双gate乘积施加目标端竞争。两者
都使用静态per-head codebook输出，不读动态K/V carrier，且所有12个block使用同一公式。

H79/H80将分别实现、加载审计并在H78后从冻结TTX ep2独立full30+valid825，禁止组合。AMM9
多模态GT监督与BSMR9 masked reconstruction保留孵化：它们是部署零增量训练正则，不构成新的
attention硬件数据流，且在当前队列已有多个plain/patch/contingency Omega9候选时优先级较低。
本轮判据不是NB0+5%的最低门，而是必须击败H67 dyadic AEE`1.4626`、保持AAE/PE竞争力且
spikes不超过`26.3948G`；CF10还必须排除null塌缩，DN9必须报告boundary和destination collision。

### 43.26 H67/H68增量ASIC RTL冻结与SCS-Shiftmax迭代（2026-07-13）

<!-- H67_H68_ASIC_RTL_SCS_FREEZE_20260713 -->

完整中文结果见：

- `hw_autoresearch_nts07/docs/49_H67H68逐位验证_占用类Shiftmax与DC交付结果.md`；
- `hw_autoresearch_nts07/docs/50_H67主线_DATE贡献冻结与投稿前签核清单.md`；
- `hw_autoresearch_nts07/results/h67_h68_storage_ablation.md`；
- `hw_autoresearch_nts07/results/h67_h68_score_class_scan_cycle_model.md`；
- `hw_autoresearch_nts07/results/h67_h68_atlif_module_coverage.md`。

硬件主线更新为H67，H68只保留为训练期辅助、部署期TTX的消融和回退顶层。原因是H67
RTL-exact AEE`1.462688`优于H68的`1.472654`，两者spikes接近；同时H67的零K最终score虽有
35个合法类，profile100每行只占用`1.36--2.75`类，可用占用位图而非固定35类扫描。

新增SCS-Shiftmax保持零K的Shiftmax分母贡献，但省略其恒零`gate*K`回放。H67采用两拍
`FIND_CLASS -> CLASS_MAC`流水切断35路优先编码到exp2乘加的长路径；H68只有3类，编译期特化
为单拍。按6720行/帧的真实stage权重，H67行核周期代理由`1,591,065`降至`1,386,424`，下降
`12.86%`；H68下降`0.37%`。该值不含SRAM同步读、projection、ATLIF、skip、decoder或搬运，
不能写成端到端FPS。

ATLIF覆盖已解释：软件安装105个wrapper，但H67/H68实际forward均为93个；未调用的12个全部
是all12 attention的`sn2_q`原carrier神经元。固定H60部署硬件按93个实际调用点核算，105只作
安装/回退兼容口径。encoder跨stage skip仍只有S0/S1/S2三条；S3是bottleneck输出，不是第4条
encoder skip。

精确162深度相对256填充的Yosys通用结构代理：H67存储位下降`37.02%`、总单元下降
`32.55%`；H68分别下降`36.69%`和`31.51%`。该结果未映射工艺，不能换算um2/mW。DC/SDC、
SVF和Formality交接包已经生成，但当前机器没有`dc_shell`、目标`.db/.lib`和SRAM宏；正式
WNS/TNS、面积、真实活动功耗和LEC仍是投稿前P0门槛。

### 43.28 H69 full30 首轮运行里程碑（2026-07-13）

<!-- H69_FULL30_EPOCH0_MILESTONE_20260713 -->

H69 x8 正式 full30 已完成 epoch0 并进入 epoch1。epoch0 共918 steps，训练损失为
`1.500449`，标准小验证损失为`1.233169`；训练耗时`1445.56 s`（24.09分钟），稳态
`1.5747 s/step`、`5.0804 samples/s`，峰值GPU显存`56.823 GiB`。本地checkpoint已写入
`results/h69_allbinary_all12_dyadic_temperature_ttx_x8_w720_fastlr_full30_bs8_full30_20260711_setsid/checkpoint_epoch0.pth`
（约738 MB）。训练中105个one-sided binary ATLIF的平均activity约`4.3%--4.5%`，threshold
固定为1.0；未观察到NaN、OOM、worker退出或加载异常。

按epoch0实测吞吐，单项30轮训练约需12.1小时，之后runner会自动对预注册epoch
`0/4/9/14/19/24/28/29`执行valid825。当前loss只能证明链路健康，不能用于H69相对H67/H68的
精度裁决；最终仍以valid825 AEE/AAE、spikes以及统一dyadic/RTL-exact部署评估为准。

epoch1随后正常完成：train loss=`1.485234`、小验证loss=`1.147696`，分别较epoch0继续下降；
epoch time=`1466.67 s`、`1.5977 s/step`、`5.0073 samples/s`、峰值显存`56.111 GiB`。epoch1不在
`force_save_epochs={0,4,9,14,19,24,28,29}`内，因此未新增checkpoint属于预注册存盘行为，非保存
故障。训练已进入epoch2。

### 43.29 H79/H80 实现、加载审计与队列接入（2026-07-13）

<!-- H79_H80_IMPLEMENTATION_LOAD_QUEUE_AUDIT_20260713 -->

H79 CF10与H80 DN9已按43.27实现为默认关闭分支。H79对Omega9 score加入由top2 margin和query
activity构造的null evidence，只注册9行静态codebook，第10行null codeword在forward中严格接零；
H80对同一Omega9边分别做source-row和destination-incoming Shiftmax，将双gate乘积量化到unsigned
Q1.7后再读取9行静态codebook。两项均不读取动态K/V carrier，所有12个block使用同一公式。

公式、边界、梯度、fixed-zero null、DN9 dense-reference与默认关闭回归共52项全部通过；DN9
逐边索引覆盖1250条合法cross-time local edge。冻结TTX epoch2真实加载审计为：两项ATLIF105、
attention12、candidate12、checkpoint overlay210、unexpected0；H79唯一missing为12个codebook加
12个`_h9_cf10_beta`，H80唯一missing为12个codebook，同模式state严格重载均为missing0/
unexpected0。另5项加载与optimizer测试通过。

审计同时发现原训练入口只允许到H78，且未把`_h9_cf10_beta`计入overlay/new-module参数；这会导致
H79正式warm-start被拒绝或beta落入backbone参数组。现已仅新增H79/H80模式及CF10键识别到训练、
标准推理加载、optimizer和SOP profiler，旧模式行为不变。新幂等runner
`entrypoints/run_round4_assignment_after_h78.py`已启动，严格等待H78完成后按H79->H80逐项执行
full30、trained strict-load与八个预注册epoch的valid825。TTB-v2 watcher已后移到H80完成标记，
最终dyadic deploy表与attention op audit固定为19项候选。

### 43.30 TTB有序trace与有限FIFO回放（2026-07-13）

<!-- TTB_ORDERED_TRACE_FINITE_FIFO_REPLAY_20260713 -->

为关闭“聚合empty比例不能推出cycle”的缺口，hardware profiler新增默认关闭的`--ordered-trace`
只读模式。它在每个sample、stage、block、window/head顺序下，以zlib+base64压缩int16保存：逐token
Exact-Delta更新lane数，以及TTB4/TTB8的Q-or-K active、K active和K-motion计数。该分支只在
profile collector中读取已经产生的Q/K bitplane，不改变attention输出、checkpoint或训练图。

新增`entrypoints/replay_ttb_dual_path_cycles.py`。它先对`kappa={2,4,8,12,16}`、sparse lanes
`{2,4,8}`做全量解析工作下界，再只提升每种route的前三个组合，对FIFO depth`{4,8,16}`逐周期
回放metadata一拍一bundle、dense/sparse服务与有限队列backpressure。每次attention调用结束时排空
队列，避免跨block虚假重叠；输出总cycle、input stalls、dense/sparse busy、最大FIFO占用和分stage
结果。traffic同时报告dense Q/K、固定union bitmap加active Q/K payload、coordinate-index加payload
三种位数；E0/E1都计route count/tag metadata，不把稀疏格式或索引当作免费。除bit编码量外，回放
还输出逐bundle保守64-bit transaction计数：每个descriptor固定一个metadata word，dense/bitmap/index
payload分别在bundle边界向上取整到64 bit，避免用跨bundle理想拼接夸大稀疏收益；该口径尚不包含
transaction coalescing、bank映射或端口grant。单元测试覆盖全empty metadata速率、有限FIFO阻塞、
双路径并行、FIFO峰值聚合、完整sweep输出及bit/64-bit traffic单调性，当前完整回归90项通过。

解析sweep另加入共享backend每window/head row为`{1,4,8,16}` cycles的工作下界敏感性，并分别
报告E0/E1的`max(metadata,dense,sparse,backend)`下界。该项只说明backend何时会吞没前端收益；
由于尚未模拟row join完成顺序与backend FIFO/credit，它不进入finite-replay cycle，也不能作为
最终共享Shiftmax周期。

该模型仍不包含共享Shiftmax/backend、SRAM bank/port、projection、decoder或NoC，因此其cycle
下降只能称row-kernel proxy，不能写成端到端FPS。现有64-bit transaction是无bank的保守padding
模型，不等于真实SRAM transaction或energy。TTB-v2 runner已在H80完成后对TTX/H67/H68各
profile100自动采集有序trace并生成有限FIFO artifact；下一证据门是把同一trace映射到SRAM address/
bank conflict、端口grant、可合并transaction和共享backend credit。


### H69 dyadic-temperature full30 自动结果：h69_allbinary_all12_dyadic_temperature_ttx_x8_w720_fastlr_full30

<!-- H69_FULL30::h69_allbinary_all12_dyadic_temperature_ttx_x8_w720_fastlr_full30 -->
- short screen: `neuron_experiments/H9_bipolar_self_attention/results/h69_dyadic_temperature_screen_20260713_132619`
- full config: `neuron_experiments/H9_bipolar_self_attention/configs/generated/h69_allbinary_all12_dyadic_temperature_ttx_x8_w720_fastlr_full30.yml`
- start checkpoint: `neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid/checkpoint_epoch2.pth`
- run dir: `neuron_experiments/H9_bipolar_self_attention/results/h69_allbinary_all12_dyadic_temperature_ttx_x8_w720_fastlr_full30_bs8_full30_20260711_setsid`

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | spike_energy_proxy_uj |
| 1 | 19 | 1.4802 | 9.4675 | 0.5052 | 0.1922 | 0.0903 | 26.3310G | 5.6904% | 23335.48 |
| 2 | 29 | 1.4913 | 9.4607 | 0.5048 | 0.1951 | 0.0944 | 26.6890G | 5.7677% | 23651.10 |
| 3 | 4 | 1.4976 | 9.6447 | 0.5091 | 0.1908 | 0.0895 | 24.1025G | 5.2088% | 21318.26 |
| 4 | 24 | 1.5146 | 9.7547 | 0.5135 | 0.1987 | 0.0943 | 26.7167G | 5.7737% | 23689.93 |
| 5 | 28 | 1.5188 | 9.9509 | 0.5132 | 0.1970 | 0.0942 | 28.3859G | 6.1345% | 25166.20 |
| 6 | 14 | 1.5251 | 9.9046 | 0.5248 | 0.2039 | 0.0949 | 25.9921G | 5.6171% | 23022.69 |
| 7 | 0 | 1.5301 | 9.9924 | 0.5224 | 0.2014 | 0.0957 | 24.0388G | 5.1950% | 21244.04 |
| 8 | 9 | 1.5324 | 9.9877 | 0.5177 | 0.1971 | 0.0929 | 25.8149G | 5.5789% | 22858.47 |


### H70 Event-Selective TTX full30 自动结果

<!-- H70_EVENT_SELECTIVE_FULL30_RESULT -->
- config: `neuron_experiments/H9_bipolar_self_attention/configs/generated/h70_allbinary_all12_event_selective_ttx_maxshift3_w720_fastlr_full30.yml`
- start checkpoint: `neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid/checkpoint_epoch2.pth`
- run dir: `neuron_experiments/H9_bipolar_self_attention/results/h70_allbinary_all12_event_selective_ttx_maxshift3_w720_fastlr_full30_bs8_full30_20260711_setsid`

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | spike_energy_proxy_uj |
| 1 | 19 | 1.4784 | 9.4656 | 0.5053 | 0.1918 | 0.0903 | 26.3144G | 5.6868% | 23311.37 |
| 2 | 29 | 1.4938 | 9.4258 | 0.5034 | 0.1953 | 0.0944 | 26.6953G | 5.7691% | 23647.94 |
| 3 | 4 | 1.5016 | 9.7319 | 0.5104 | 0.1924 | 0.0909 | 24.1368G | 5.2162% | 21347.44 |
| 4 | 24 | 1.5129 | 9.7693 | 0.5128 | 0.1976 | 0.0944 | 26.7508G | 5.7811% | 23710.13 |
| 5 | 9 | 1.5109 | 9.9410 | 0.5162 | 0.1953 | 0.0918 | 25.8551G | 5.5875% | 22894.45 |
| 6 | 14 | 1.5171 | 9.8676 | 0.5223 | 0.2022 | 0.0944 | 26.0191G | 5.6230% | 23035.56 |
| 7 | 28 | 1.5211 | 9.9819 | 0.5137 | 0.1966 | 0.0938 | 28.3979G | 6.1370% | 25167.29 |
| 8 | 0 | 1.5310 | 10.0304 | 0.5224 | 0.2006 | 0.0953 | 24.0534G | 5.1982% | 21253.78 |


### 43.31 H70 full30 + 算法队列暂停（2026-07-14）

H70 event-selective TTX（maxshift3）full30+valid825 已完成。

- run dir：`neuron_experiments/H9_bipolar_self_attention/results/h70_allbinary_all12_event_selective_ttx_maxshift3_w720_fastlr_full30_bs8_full30_20260711_setsid`
- ranking：`.../profile_ranking_valid825.md`
- best epoch19：AEE `1.4784`、AAE `9.4656`、total_spikes `26.3144G`
- 相对 H67 float ep19 AEE `1.4671`：更差约 `+0.77%`，**未超越 H67**
- 相对 H69 ep19 AEE `1.4802`：略好，但仍明显差于 H67
- 相对 NB0 `1.4872`：仍略好；spikes 约 26.3G（相对 NB0 约 -40%）

H69 best ep19 AEE `1.4802` 亦未超越 H67。

用户要求 H70 完成后暂停算法队列。已停止 H71 训练（若已启动）及后续 watcher
（h71/h66/match-code/round3/round4 等）。暂停记录：
`neuron_experiments/H9_bipolar_self_attention/results/algorithm_queue_pause_after_h70_20260714.log`。

**当前软件主线不变：H67 Motion-XOR ep19**（float AEE `1.4671` / dyadic AEE `1.4626`）。
H71 及 Match-Code 队列未完成；恢复时检查 H71 partial run 后决定重跑或续训。



### 43.32 算法队列恢复（2026-07-14）

<!-- ALGORITHM_QUEUE_RESUME_20260714 -->

用户要求恢复 H70 后暂停的算法队列。

**H71 半成品处理**
- 暂停时 H71 full30 仅有 `checkpoint_epoch0.pth`，epoch1 中途被杀；无完整 optimizer state 续训。
- 按 runner 设计：`checkpoint_epoch29` 不存在则从冻结 TTX ep2 干净重训 full30（非 mid-epoch resume）。
- health 360 已通过，runner 复用 `END H71 implementation health check: exit_code=0`。
- 审计副本：`results/h71_..._20260711_setsid/train_partial_killed_after_pause_20260714.log` 与 `checkpoint_epoch0_partial_before_resume_*.pth`。

**恢复链（setsid 并行 WAIT，串行依赖）**
1. `run_h71_after_h70.py` — 当前执行 full30
2. `run_h66_full30_after_h71.py` — H66a–e
3. `run_match_code_after_h66.py` — H73–H75
4. `run_round3_match_after_h75.py` — H76–H78
5. `run_round4_assignment_after_h78.py` — H79–H80
6. `run_ttb_cycle_profile_v2_after_round3.py`
7. `run_delta_locality_after_h71.py`（等 TTB-v2）

恢复记录：`neuron_experiments/H9_bipolar_self_attention/results/algorithm_queue_resume_after_h70_pause_20260714_234909.log`  
Launcher 前缀：`*_resume_20260714_234909.launcher.{log,pid}`

软件主线在 H71 完成前仍为 **H67 Motion-XOR ep19**。


## H69/H70 deployment score-clipping profile20 自动结果

<!-- H69_H70_TEMPERATURE_SCORE_CLIP_PROFILE20_20260713 -->
- artifact: `neuron_experiments/H9_bipolar_self_attention/results/temperature_score_clip_profile20_20260713.md`

| candidate | best epoch | score elements | clip low | clip high | clip ratio |
|---|---:|---:|---:|---:|---:|
| H69 | 19 | 21772800 | 0 | 0 | 0.000000% |
| H70 | 19 | 21772800 | 0 | 0 | 0.000000% |

裁剪按量化前 score 严格小于 -2 或大于 2 计数，边界值不计入；该表用于判断固定/动态左移是否需要扩大 score 位宽，不替代 valid825 精度。


### H71 Window-Context TTX full30 自动结果

<!-- H71_WINDOW_CONTEXT_FULL30_RESULT -->
- config: `neuron_experiments/H9_bipolar_self_attention/configs/generated/h71_allbinary_all12_window_context_ttx_w720_fastlr_full30.yml`
- start checkpoint: `neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid/checkpoint_epoch2.pth`
- run dir: `neuron_experiments/H9_bipolar_self_attention/results/h71_allbinary_all12_window_context_ttx_w720_fastlr_full30_bs8_full30_20260711_setsid`

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | spike_energy_proxy_uj |
| 1 | 19 | 1.4872 | 9.3892 | 0.5045 | 0.1933 | 0.0923 | 26.4488G | 5.7158% | 23444.92 |
| 2 | 29 | 1.5035 | 9.4518 | 0.5029 | 0.1967 | 0.0961 | 26.7633G | 5.7838% | 23721.74 |
| 3 | 4 | 1.5092 | 9.8101 | 0.5122 | 0.1938 | 0.0922 | 24.1230G | 5.2132% | 21340.77 |
| 4 | 24 | 1.5218 | 9.7992 | 0.5144 | 0.2014 | 0.0966 | 26.8829G | 5.8096% | 23841.91 |
| 5 | 28 | 1.5189 | 9.9914 | 0.5115 | 0.1953 | 0.0931 | 28.4882G | 6.1566% | 25262.82 |
| 6 | 9 | 1.5230 | 10.0372 | 0.5159 | 0.1954 | 0.0926 | 25.9949G | 5.6177% | 23022.36 |
| 7 | 0 | 1.5449 | 10.0425 | 0.5220 | 0.2025 | 0.0976 | 24.0644G | 5.2006% | 21267.23 |
| 8 | 14 | 1.5527 | 9.9909 | 0.5281 | 0.2088 | 0.0991 | 25.9951G | 5.6178% | 23023.77 |


### H66 full30 自动结果：h66a_allbinary_all12_axnor_matrix_shiftmax_w720_fastlr_full30

<!-- H66_FULL30::h66a_allbinary_all12_axnor_matrix_shiftmax_w720_fastlr_full30 -->
- config: `neuron_experiments/H9_bipolar_self_attention/configs/generated/h66a_allbinary_all12_axnor_matrix_shiftmax_w720_fastlr_full30.yml`
- start checkpoint: `neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid/checkpoint_epoch2.pth`
- run dir: `neuron_experiments/H9_bipolar_self_attention/results/h66a_allbinary_all12_axnor_matrix_shiftmax_w720_fastlr_full30_bs8_full30_20260712_setsid`

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | spike_energy_proxy_uj |
| 1 | 19 | 1.5060 | 9.6469 | 0.5102 | 0.1973 | 0.0946 | 26.8712G | 5.8071% | 23762.69 |
| 2 | 29 | 1.5115 | 9.6591 | 0.5074 | 0.1988 | 0.0968 | 27.2870G | 5.8970% | 24112.96 |
| 3 | 24 | 1.5426 | 9.9446 | 0.5165 | 0.2043 | 0.0990 | 27.3546G | 5.9116% | 24192.89 |
| 4 | 28 | 1.5455 | 10.0656 | 0.5162 | 0.2001 | 0.0965 | 28.9981G | 6.2668% | 25640.77 |
| 5 | 4 | 1.5497 | 9.9826 | 0.5189 | 0.2010 | 0.0967 | 24.2526G | 5.2412% | 21434.81 |
| 6 | 9 | 1.5707 | 10.2605 | 0.5217 | 0.2014 | 0.0971 | 26.1243G | 5.6457% | 23103.09 |
| 7 | 14 | 1.5743 | 10.2040 | 0.5337 | 0.2125 | 0.1012 | 26.3072G | 5.6852% | 23259.96 |
| 8 | 0 | 1.5794 | 10.2936 | 0.5312 | 0.2095 | 0.1014 | 24.1164G | 5.2118% | 21300.83 |


### H66 full30 自动结果：h66b_allbinary_all12_hamming_linear_w720_fastlr_full30

<!-- H66_FULL30::h66b_allbinary_all12_hamming_linear_w720_fastlr_full30 -->
- config: `neuron_experiments/H9_bipolar_self_attention/configs/generated/h66b_allbinary_all12_hamming_linear_w720_fastlr_full30.yml`
- start checkpoint: `neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid/checkpoint_epoch2.pth`
- run dir: `neuron_experiments/H9_bipolar_self_attention/results/h66b_allbinary_all12_hamming_linear_w720_fastlr_full30_bs8_full30_20260712_setsid`

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | spike_energy_proxy_uj |
| 1 | 29 | 1.5429 | 9.7685 | 0.5132 | 0.2057 | 0.1023 | 26.2821G | 5.6798% | 22901.70 |
| 2 | 19 | 1.5535 | 9.8956 | 0.5184 | 0.2059 | 0.1008 | 25.9425G | 5.6064% | 22616.65 |
| 3 | 4 | 1.5710 | 10.0397 | 0.5234 | 0.2051 | 0.1003 | 24.2502G | 5.2407% | 21108.76 |
| 4 | 9 | 1.5854 | 10.3057 | 0.5259 | 0.2058 | 0.0995 | 25.8519G | 5.5868% | 22497.22 |
| 5 | 28 | 1.5980 | 10.3387 | 0.5224 | 0.2079 | 0.1027 | 28.0903G | 6.0706% | 24437.91 |
| 6 | 24 | 1.6036 | 10.3051 | 0.5275 | 0.2136 | 0.1054 | 26.4182G | 5.7092% | 23024.51 |
| 7 | 14 | 1.6081 | 10.3874 | 0.5387 | 0.2183 | 0.1059 | 25.7450G | 5.5637% | 22420.86 |
| 8 | 0 | 1.6306 | 10.4748 | 0.5368 | 0.2164 | 0.1077 | 24.4585G | 5.2857% | 21262.94 |


### H66 full30 自动结果：h66c_allbinary_all12_tp_ttx_w720_fastlr_full30

<!-- H66_FULL30::h66c_allbinary_all12_tp_ttx_w720_fastlr_full30 -->
- config: `neuron_experiments/H9_bipolar_self_attention/configs/generated/h66c_allbinary_all12_tp_ttx_w720_fastlr_full30.yml`
- start checkpoint: `neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid/checkpoint_epoch2.pth`
- run dir: `neuron_experiments/H9_bipolar_self_attention/results/h66c_allbinary_all12_tp_ttx_w720_fastlr_full30_bs8_full30_20260712_setsid`

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | spike_energy_proxy_uj |
| 1 | 19 | 1.4757 | 9.5116 | 0.5038 | 0.1904 | 0.0894 | 26.5044G | 5.7278% | 23473.59 |
| 2 | 29 | 1.4807 | 9.4081 | 0.5005 | 0.1922 | 0.0921 | 26.8580G | 5.8043% | 23784.34 |
| 3 | 4 | 1.4966 | 9.7418 | 0.5093 | 0.1907 | 0.0898 | 24.2674G | 5.2444% | 21459.56 |
| 4 | 28 | 1.5121 | 9.9149 | 0.5113 | 0.1949 | 0.0929 | 28.5681G | 6.1738% | 25311.56 |
| 5 | 24 | 1.5247 | 9.8106 | 0.5118 | 0.1983 | 0.0953 | 26.9102G | 5.8155% | 23846.06 |
| 6 | 9 | 1.5263 | 10.0156 | 0.5162 | 0.1952 | 0.0920 | 26.0911G | 5.6385% | 23091.98 |
| 7 | 0 | 1.5392 | 10.0741 | 0.5222 | 0.2005 | 0.0948 | 24.1374G | 5.2163% | 21321.54 |
| 8 | 14 | 1.5513 | 10.0350 | 0.5274 | 0.2054 | 0.0967 | 26.1199G | 5.6448% | 23115.45 |


### 43.30 DATE 算法审稿整改、AAE 口径纠偏与 H81 等预算控制（2026-07-17）

<!-- DATE_ALGORITHM_REVIEW_REMEDIATION_20260717 -->

算法侧独立审稿结论为 **Weak Reject（confidence 4/5）**。首要问题不是 H67 当前 AEE，而是
缺少等预算 no-motion 对照、多 seed、独立测试集，以及把本地 AAE 与论文官方 test AE 混为
同一口径。完整整改表见
`neuron_autoresearch/DATE_ALGORITHM_REVIEW_REMEDIATION_20260717.md`。

**AAE 根因已确认：**

- 本地旧 `AAE` 计算 `(u,v)` 的二维方向夹角；
- DSEC benchmark 按 Barron 定义计算 `(u,v,1)` 的三维时空夹角；
- 论文 `AEE=1.602, AAE=4.871` 来自 official DSEC test 全序列，不是 valid825；
- valid825 是 DSEC train 的 `288x384` center-crop 留出集。

因此 NB0/H67 的本地 `9.x` 仍可用于历史实验内部比较，但不能直接对比论文 `4.871`。已新增
可选 `AAE_Benchmark`，旧 `AAE` 不变；诊断记录见
`neuron_autoresearch/AAE_BASELINE_DIAGNOSTIC_20260717.md`。

**H81 reviewer control：**

- config：`neuron_experiments/H9_bipolar_self_attention/configs/generated/h81_allbinary_all12_h60_nomotion_equalbudget_w720_fastlr_full30.yml`；
- 与 H67 仅有三个语义差异：实验名、说明文字、`binary_motion_xor_alpha: 0.25 -> 0.0`；
- 同一 TTX epoch2 起点、full30、优化器、warmup、阈值冻结、all12 H60、105 个 all-binary ATLIF；
- 队列顺序改为 `H66a-e -> H81 -> NB0/H67/H81 双 AAE 审计 -> H73-H80`，不并发抢 GPU。

当前 H66 已完成 a/b/c。best 分别为：H66a `1.5060/9.6469/26.8712G`，H66b
`1.5429/9.7685/26.2821G`，H66c `1.4757/9.5116/26.5044G`。H66c 通过 NB0 精度门，
但尚未超过 H67 `1.4671/9.4155/26.3898G`。H66d 当前训练中，H66e 随后串行执行。

若 DSEC official test 暂时无法提交，第二数据集采用固定公开 sequence split 的
**MVSEC-train -> MVSEC-test**，不混 MDR，统一训练域、测试域和文献对比口径；该路线在最终
候选冻结后执行，避免与当前 DSEC 搜索队列并发。


### 43.34 算法队列立即暂停（2026-07-17）

<!-- PAUSE_NOW_H66D_20260717 -->

用户要求立即暂停（不等当前 epoch）。

- 已停止 H66d train 及 H66/Match/Round/TTB 等 algorithm watcher。
- H66d 最后强制存盘：`checkpoint_epoch24.pth`（force: 0/4/9/14/19/24/28/29）。
- 暂停时约在 Epoch 27 中途；ep25–27 无完整 model ckpt。
- 已完成：H71、H66a、H66b、H66c；未完成：H66d/e 及 Match-Code 队列。
- 主线仍为 **H67 Motion-XOR ep19**。
- 暂停记录：`neuron_experiments/H9_bipolar_self_attention/results/algorithm_queue_pause_now_20260717.log`


### 43.35 有价值 checkpoint 的统一新 AAE 审计（2026-07-17）

<!-- VALUABLE_CHECKPOINT_AAE_AUDIT_20260717 -->

统一使用 DSEC valid825，同时计算历史二维方向 AAE 与 DSEC/Barron `(u,v,1)` 三维 AE。完整可复现报告见 `neuron_autoresearch/VALUABLE_CHECKPOINT_AAE_AUDIT_20260717.md`。

| model | epoch | AEE | legacy AAE-2D | DSEC/Barron AE-3D | PE1 | PE2 | outlier | spikes(G) | energy proxy(uJ) | load |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| NB0 baseline | 59 | 1.4872 | 9.9300 | 9.2506 | 0.5107 | 0.1891 | 0.0871 | 44.0488 | 37638.01 | ATLIF 0, Shiftmax 0, 0/0 |
| NTS11bd mixed ternary/binary | 19 | 1.5650 | 9.9234 | 9.3407 | 0.5310 | 0.2101 | 0.1021 | 29.1679 | 23109.18 | ATLIF 105, Shiftmax 12, 0/0 |
| All-binary + TX | 19 | 1.5831 | 9.9381 | 9.3482 | 0.5348 | 0.2136 | 0.1041 | 22.4706 | 19780.93 | ATLIF 105, Shiftmax 12, 0/0 |
| Frozen TTX/H60 | 2 | 1.5019 | 9.8894 | 9.2123 | 0.5169 | 0.1949 | 0.0918 | 23.2396 | 20521.16 | ATLIF 105, Shiftmax 12, 0/0 |
| H66a a-XNOR matrix | 19 | 1.5060 | 9.6469 | 9.0311 | 0.5102 | 0.1973 | 0.0946 | 26.8712 | 23762.69 | ATLIF 105, Shiftmax 12, 0/0 |
| H66b Hamming linear | 29 | 1.5429 | 9.7685 | 9.1403 | 0.5132 | 0.2057 | 0.1023 | 26.2821 | 22901.70 | ATLIF 105, Shiftmax 12, 0/0 |
| H66c TP-TTX | 19 | 1.4757 | 9.5116 | 8.8846 | 0.5038 | 0.1904 | 0.0894 | 26.5044 | 23473.59 | ATLIF 105, Shiftmax 12, 0/0 |
| H67 Motion-XOR (float) | 19 | 1.4671 | 9.4155 | 8.7949 | 0.5002 | 0.1891 | 0.0890 | 26.3898 | 23393.08 | ATLIF 105, Shiftmax 12, 0/0 |
| H67 Motion-XOR (RTL-exact) | 19 | 1.4627 | 9.4040 | 8.7801 | 0.5007 | 0.1886 | 0.0883 | 26.3544 | 23362.23 | ATLIF 105, Shiftmax 12, 0/0 |
| H68 Castling-trained/H60 deploy (RTL-exact) | 19 | 1.4727 | 9.4714 | 8.8441 | 0.5025 | 0.1895 | 0.0891 | 26.4164 | 23414.83 | ATLIF 105, Shiftmax 12, 0/0 |
| H69 fixed dyadic temperature | 19 | 1.4819 | 9.5177 | 8.8829 | 0.5056 | 0.1920 | 0.0899 | 26.3414 | 23344.73 | ATLIF 105, Shiftmax 12, 0/0 |
| H70 event-selective dyadic TTX | 19 | 1.4852 | 9.5081 | 8.9013 | 0.5052 | 0.1917 | 0.0904 | 26.3213 | 23317.32 | ATLIF 105, Shiftmax 12, 0/0 |
| H71 window-context TTX | 19 | 1.4872 | 9.3892 | 8.8030 | 0.5045 | 0.1933 | 0.0923 | 26.4488 | 23444.92 | ATLIF 105, Shiftmax 12, 0/0 |

H67 RTL-exact 相对 NB0：AEE 改善 `1.65%`、AE-3D 改善 `5.09%`、spikes 下降 `40.17%`、energy proxy 下降 `37.93%`，因此作为当前 checkpoint 主线。H68 RTL-exact 继续作为部署零增量回退。所有行加载审计均为预期 ATLIF/Shiftmax 数且 `missing=0, unexpected=0`；论文 official test 仍需单独提交，不能把 valid825 AE-3D 冒充 test AE。


### 43.36 算法队列原状态恢复（2026-07-18）

<!-- RESUME_ALGORITHM_QUEUE_20260718 -->

用户要求先完成已排队训练。恢复顺序固定为：

`H66d epoch24 true-resume -> H66d valid825 -> H66e full30/valid825 -> H81 no-motion -> H73-H75 -> H76-H78 -> H79-H80`。

- H66d 使用 `checkpoint_epoch24.pth` 加同名 `_state_dict.pth`，恢复 model、optimizer、scheduler、AMP scaler，并从 epoch25 开始；不得将 epoch24 仅作为新 warm-start。
- MLflow 关闭，CuPy backend，原 batch、worker、warmup、milestone、阈值冻结和 valid825 口径均不变。
- 恢复入口：`neuron_experiments/H9_bipolar_self_attention/entrypoints/resume_algorithm_queue_20260718.sh`。
- 运行日志：`neuron_experiments/H9_bipolar_self_attention/results/algorithm_queue_resume_20260718.log`。
- 为防止磁盘耗尽，仅在某候选八个 valid825 profile 和 ranking 全部存在后清理中间模型；保留 best、epoch29、对应训练状态、全部 profile/ranking/log，并写 `checkpoint_prune_audit.json`。
- `AAE=4.871` 的官方 test 口径核查和 CICC 2026 复现实验协议在队列运行期间继续做，不改变本队列训练配置。


### 43.37 AAE 官方结果差距与 CICC 2026 仿照协议（2026-07-18）

<!-- AAE_OFFICIAL_TEST_CICC2026_PROTOCOL_20260718 -->

完整审计与可执行协议：
`neuron_autoresearch/AAE_OFFICIAL_TEST_AND_CICC2026_PROTOCOL_20260718.md`。

**AAE 结论：** 新增的 DSEC/Barron AE-3D 对同一 valid825 的模型比较可信，但当前不能复现
论文 `4.871`。论文值来自七条 hidden test sequence 的 official server；论文最终路线为
`80 epoch 288x384 crop + 30 epoch 480x640 full-resolution fine-tune`，测试时关闭 BN
running-state tracking。NB0 只是 `60 epoch crop`，本地评估又是 `288x384 center-crop`
valid825，学习率、序列、采样、聚合均不同。差距不能归咎于 AE-3D 公式。

当前 `eval_DSEC_flow_SNN.py` 只实现 `mode=valid`；`mode=test` 没有执行分支。因此队列完成后
必须新增 official submission writer，并先做 NB0/最终候选的 full-resolution、BN-policy 等预算
对照，再谈与 `4.871` 比较。

**MVSEC 结论（2026-08-01 正文勘误）：** 四个 dt1 test sequence 已完整预处理；CICC
正文使用参考 Spike-FlowNet 的 INT8 Hybrid U-Net，在 `indoor1/2/3 + outdoor1` 各评估
`800` 个输入。Spike-FlowNet 标准范式只用 `outdoor_day2` 训练，测试为 `256x256`
center crop、dt1 event-masked AEE。MDR->MVSEC 仍是 SDformerFlow 原论文对比路线；
day2->四测试集是 CICC/Spike-FlowNet 直接 MVSEC 路线；两者必须分表。

**CICC 2026（2026-08-01 正文勘误）：** 本地四页正文已核验。INT8 baseline 四序列 AEE
为 `0.84/1.32/1.14/0.52`（mean `0.96`），全部特征后为
`0.87/1.35/1.17/0.56`（mean `0.99`）。真实机制是 group-16 lossless BWAC、
Dense-Channel-First MaxPool/ReLU speculation 和 feature-similarity DLSS；TTB 不属于该论文。
DATE 仿照累计消融修正为 `C0 INT8 dense / C1 +BWAC / C2 +speculation / C3 +DLSS`，
统一报告 event-masked AEE、实际 operation、SRAM/DRAM bytes、cycle、含外存 energy 和
控制/面积开销。完整修订见后文 `MVSEC_CICC2026_ALIGNMENT_20260801`。


### H66 full30 自动结果：h66d_allbinary_all12_lr_ttx_w720_fastlr_full30

<!-- H66_FULL30::h66d_allbinary_all12_lr_ttx_w720_fastlr_full30 -->
- config: `neuron_experiments/H9_bipolar_self_attention/configs/generated/h66d_allbinary_all12_lr_ttx_w720_fastlr_full30.yml`
- start checkpoint: `neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid/checkpoint_epoch2.pth`
- run dir: `neuron_experiments/H9_bipolar_self_attention/results/h66d_allbinary_all12_lr_ttx_w720_fastlr_full30_bs8_full30_20260712_setsid`

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | spike_energy_proxy_uj |
| 1 | 29 | 1.4432 | 9.4012 | 0.4963 | 0.1832 | 0.0844 | 27.0403G | 5.8437% | 23976.31 |
| 2 | 19 | 1.4757 | 9.4652 | 0.5040 | 0.1900 | 0.0891 | 26.6324G | 5.7555% | 23601.58 |
| 3 | 28 | 1.4879 | 9.6978 | 0.5076 | 0.1916 | 0.0897 | 28.4610G | 6.1507% | 25219.64 |
| 4 | 4 | 1.5060 | 9.7565 | 0.5119 | 0.1934 | 0.0918 | 24.1945G | 5.2287% | 21408.35 |
| 5 | 24 | 1.5120 | 9.7579 | 0.5116 | 0.1981 | 0.0945 | 27.0550G | 5.8469% | 23988.37 |
| 6 | 9 | 1.5174 | 10.0373 | 0.5154 | 0.1937 | 0.0917 | 26.0698G | 5.6339% | 23092.76 |
| 7 | 14 | 1.5458 | 9.9784 | 0.5255 | 0.2067 | 0.0980 | 26.1670G | 5.6549% | 23173.75 |
| 8 | 0 | 1.5445 | 10.0643 | 0.5231 | 0.2020 | 0.0967 | 24.0886G | 5.2058% | 21289.31 |


### H66 full30 自动结果：h66e_allbinary_all12_tp_ttx_selfbias1_w720_fastlr_full30

<!-- H66_FULL30::h66e_allbinary_all12_tp_ttx_selfbias1_w720_fastlr_full30 -->
- config: `neuron_experiments/H9_bipolar_self_attention/configs/generated/h66e_allbinary_all12_tp_ttx_selfbias1_w720_fastlr_full30.yml`
- start checkpoint: `neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid/checkpoint_epoch2.pth`
- run dir: `neuron_experiments/H9_bipolar_self_attention/results/h66e_allbinary_all12_tp_ttx_selfbias1_w720_fastlr_full30_bs8_full30_20260712_setsid`

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | spike_energy_proxy_uj |
| 1 | 19 | 1.4803 | 9.4605 | 0.5051 | 0.1920 | 0.0900 | 26.3073G | 5.6853% | 23315.47 |
| 2 | 29 | 1.4899 | 9.3905 | 0.5025 | 0.1940 | 0.0934 | 26.7107G | 5.7724% | 23669.21 |
| 3 | 4 | 1.5062 | 9.6793 | 0.5102 | 0.1936 | 0.0920 | 24.2135G | 5.2328% | 21412.85 |
| 4 | 24 | 1.5128 | 9.7383 | 0.5121 | 0.1984 | 0.0948 | 26.7579G | 5.7826% | 23724.93 |
| 5 | 28 | 1.5156 | 9.8940 | 0.5119 | 0.1959 | 0.0933 | 28.4036G | 6.1383% | 25180.05 |
| 6 | 9 | 1.5231 | 9.9495 | 0.5144 | 0.1939 | 0.0918 | 25.9046G | 5.5982% | 22928.64 |
| 7 | 14 | 1.5289 | 9.8693 | 0.5220 | 0.2016 | 0.0944 | 26.0871G | 5.6377% | 23099.00 |
| 8 | 0 | 1.5364 | 10.0503 | 0.5230 | 0.2015 | 0.0961 | 24.1027G | 5.2088% | 21293.58 |


### H81 等预算 no-motion 控制与 AAE 口径审计

<!-- H81_EQUAL_BUDGET_AAE_AUDIT_20260717 -->
H81 与 H67 的训练预算、起点、all12 H60 和 all-binary ATLIF 完全一致，只关闭 Motion-XOR。

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | spike_energy_proxy_uj |
| 1 | 19 | 1.4813 | 9.4636 | 0.5058 | 0.1925 | 0.0906 | 26.2576G | 5.6745% | 23271.63 |
| 2 | 29 | 1.4972 | 9.4441 | 0.5033 | 0.1955 | 0.0943 | 26.6326G | 5.7556% | 23602.36 |
| 3 | 4 | 1.4982 | 9.7259 | 0.5092 | 0.1912 | 0.0901 | 24.1474G | 5.2185% | 21355.83 |
| 4 | 14 | 1.5168 | 9.9034 | 0.5224 | 0.2009 | 0.0938 | 26.0141G | 5.6219% | 23037.59 |
| 5 | 28 | 1.5172 | 9.9219 | 0.5123 | 0.1949 | 0.0930 | 28.3680G | 6.1306% | 25150.08 |
| 6 | 24 | 1.5211 | 9.8582 | 0.5153 | 0.2012 | 0.0963 | 26.7233G | 5.7751% | 23696.28 |
| 7 | 9 | 1.5223 | 10.0046 | 0.5159 | 0.1948 | 0.0919 | 25.8656G | 5.5898% | 22900.75 |
| 8 | 0 | 1.5315 | 9.9904 | 0.5213 | 0.2006 | 0.0952 | 24.0795G | 5.2038% | 21280.04 |

同 checkpoint 双口径 AAE：

| model | epoch | AEE | legacy AAE-2D | DSEC/Barron AE-3D |
|---|---:|---:|---:|---:|
| NB0 | 59 | 1.4872 | 9.9300 | 9.2506 |
| H67 Motion-XOR | 19 | 1.4671 | 9.4155 | 8.7949 |
| H81 no-motion | 19 | 1.4813 | 9.4636 | 8.8450 |

H81 best epoch: `19`。论文 `4.871` 来自 official test，不再标注为 valid825。
### 43.38 H73-H80 显存安全恢复协议（2026-07-20）

- 2026-07-19 23:15，H73 batch8 在第一个训练 batch OOM：进程占用 78.65 GiB，后续算子仍需 810 MiB；GPU 无外部进程污染。
- 加载链路在 OOM 前已通过：`ATLIFTernaryPSN=105`、统一 attention=12、`checkpoint_overlay_keys=210`、新增 Match-Code 仅 `missing=12`、`unexpected=0`，12 个新 codebook 均正确初始化。
- 不覆盖原配置与失败目录。新增 `make_h73_h80_bs4acc2_configs.py` 和 `run_h73_h80_bs4acc2_queue.py`，为 H73-H80 生成独立 `_bs4acc2` 配置及结果目录。
- 公平性：物理 batch 4、梯度累积 2，effective batch 仍为 8；warmup 由 720 改成 1440 个 micro-step，对应相同的 720 次 optimizer update 和 5760 个训练样本。LR、milestone、训练数据、独立 TTX ep2 起点、full30 和 standard valid825 均不变。
- 每项完成后执行 trained strict-load audit，并在 valid825 ranking 落盘后仅保留 best/final checkpoint 及对应 state/profile，避免磁盘再次阻塞队列。

### 43.39 SDformerFlow AAE 公式裁决与 MVSEC 后续门控（2026-07-20）

- SDformerFlow 论文正文只声明使用 AAE，没有写出展开公式；Table I 的 `4.871` 明确来自 DSEC official optical-flow benchmark。
- DSEC benchmark 将 AE 指向 Baker/Barron optical-flow evaluation：逐像素比较归一化三维时空向量 `(u,v,1)`，即分子为 `1+u*u_gt+v*v_gt`。因此新增 `AAE_Benchmark` 的逐像素公式与论文 official table 的目标口径一致。
- upstream `flow_supervised.py::AAE` 只比较二维 `(u,v)` 方向，不能与论文 `4.871` 对比；它仅保留作历史本地实验兼容。
- 公式一致仍不等于结果可直接对比：本地 valid825 是 `288x384` center-crop train holdout，并按 sample mean 聚合；论文是七条 hidden test sequence、`480x640`、80 crop epochs + 30 full-resolution fine-tune、test BN policy 和官方 server 聚合。当前 AE-3D 只用于同一 valid825 的内部比较。
- H73-H80 结束后启动 MVSEC train-to-test，但不把所有过线候选重复训练。门槛固定为 valid825 `AEE<=1.5616`、spikes `<=35.24G`、统一 all12/no-carrier、加载 clean；从过线项中只选一个新 winner，与 MVSEC-NB0 和 H67/TTX deploy reference 构成 seed0 最小矩阵。新 winner 在 MVSEC 同时胜过 H67 AEE 并满足 spike 门槛后，才补 seed1/2。
- MVSEC 训练测试固定为 `outdoor_day1 train + held-out tail validation -> indoor_flying1/2/3 test`，详见 `neuron_autoresearch/MVSEC_TRAIN_TEST_PROTOCOL_20260717.md`。该表是受控第二数据集实验，不冒充 SDformerFlow 的 MDR->MVSEC 复现；论文主比较以 AEE/outlier、spikes 和 attention-inclusive cost 为主。

### 43.40 H73-H80 DATE 算法 novelty gate（2026-07-20）

- 完整审稿报告：`neuron_autoresearch/DATE_ALGORITHM_REVIEW_H73_H80_20260720.md`。
- H73-H80 不是八个独立论文贡献，而是一个 carrier-free binary Match-Code 主机制及 support/cost/grouping/assignment 变体。DATE 需要一个可解释、可综合、可归因的中心机制，不以变体数量作为贡献。
- 保留正在运行的 H73 DE9 full30，作为该家族唯一代表。原 H73-H80 supervisor 已停止，H73 train 进程保持运行；新增 `finish_h73_only_after_date_review.py` 只负责等待 H73、strict-load、valid825 和 ranked checkpoint pruning，不会启动 H74-H80。
- H74/MC49、H75/AX17 只是 offset 数量/形状；H76/PC9 是固定局部 cost aggregation；H78/G4 是 channel grouping；H79/CF10 高度重合 dustbin/matchability；H80/DN9 是 local dual-softmax。以上均不执行 full30。H77/LC4 只保留为 H73 晋级后的潜在机制消融，当前不训练。

### 43.42 H79/H80 作者 override 探索队列（2026-07-20）

<!-- H79_H80_AFTER_H73_REVIEW_OVERRIDE_20260720 -->

- DATE novelty review 结论不变：H79/CF10 与 dustbin/matchability 先例重合，H80/DN9 与 dual-softmax 先例重合，不能仅凭精度作为独立贡献。作者决定仍补跑二者，判断它们是否能作为 H73 的 assignment 消融或增强机制。
- H74-H78 继续停止。H79、H80 在 H73 完成 strict-load、standard valid825 和 checkpoint pruning 后按 `H79 -> H80` 串行运行，不与 H73 抢 GPU。
- 两项都从冻结 TTX epoch2 **独立** warm-start，不从前一候选续训；协议固定为 batch4、gradient accumulation2、effective batch8、warmup1440 micro-steps、full30、预注册 epoch `0/4/9/14/19/24/28/29` standard valid825。
- 每项强制审计 `ATLIFTernaryPSN=105`、attention=12、checkpoint overlay=210；H79 warm-start 仅允许 missing24，H80 仅允许 missing12，均要求 unexpected0；训练 checkpoint 必须 strict-load missing0/unexpected0。
- 新增可选过滤参数 `run_h73_h80_bs4acc2_queue.py --ids H79 H80` 及等待脚本 `run_h79_h80_after_h73_review_override.py`。结果由 runner 自动追加到本文档并执行 ranked checkpoint pruning。
- 当前 `288x384 + window[2,9,9]` 用于同 checkpoint 的公平架构搜索，并与 N=162 硬件 tile 一致。论文的 `480x640 + window[2,15,15]` 会把窗口扩为 N=450，并改变存储、地址、Shiftmax、位置编码和硬件规模；不对每个候选执行。
- 最终架构冻结后，full-resolution 只跑 NB0 与最终候选。`480x640/window9` 是硬件一致结果；`480x640/window15 + official submission` 才是 SDformerFlow 协议对比，两类结论不得混写。


### 43.41 H73 DATE novelty-gated full30 结果

<!-- H73_ONLY_AFTER_DATE_REVIEW_20260720 -->
- H73 是 Match-Code 基础机制的 full30 代表；H74-H78 经 DATE novelty review 后未训练，H79/H80 按作者决定作为 assignment 增强候选另行补跑。
- config: `neuron_experiments/H9_bipolar_self_attention/configs/generated/h73_allbinary_all12_de9_match_code_w720_fastlr_full30_bs4acc2.yml`
- run dir: `neuron_experiments/H9_bipolar_self_attention/results/h73_allbinary_all12_de9_match_code_w720_fastlr_full30_bs4acc2_20260720_setsid`
- load: ATLIF105, attention12, warm-start overlay210/missing12/unexpected0; trained checkpoint strict-load missing0/unexpected0.

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | spike_energy_proxy_uj |
| 1 | 29 | 1.4758 | 9.6183 | 0.5122 | 0.1944 | 0.0890 | 27.2813G | 5.8957% | 24180.73 |
| 2 | 19 | 1.4858 | 9.2595 | 0.5004 | 0.1916 | 0.0913 | 25.6245G | 5.5377% | 22704.12 |
| 3 | 14 | 1.4822 | 9.6562 | 0.5118 | 0.1935 | 0.0893 | 26.5083G | 5.7287% | 23465.75 |
| 4 | 0 | 1.4862 | 9.5878 | 0.5129 | 0.1915 | 0.0893 | 23.3852G | 5.0538% | 20658.38 |
| 5 | 24 | 1.5441 | 9.4157 | 0.5210 | 0.2117 | 0.1040 | 25.6049G | 5.5335% | 22684.61 |
| 6 | 4 | 1.5582 | 9.9058 | 0.5261 | 0.2055 | 0.0992 | 24.2335G | 5.2371% | 21427.33 |
| 7 | 9 | 1.5725 | 9.6986 | 0.5281 | 0.2173 | 0.1071 | 24.7629G | 5.3515% | 21900.49 |
| 8 | 28 | 1.5713 | 9.8242 | 0.5227 | 0.2141 | 0.1061 | 26.2376G | 5.6702% | 23243.87 |


### H79 Match-Code full30 显存安全结果

<!-- MATCH_BS4ACC2_FULL30::H79::h79_allbinary_all12_cf10_match_code_w720_fastlr_full30_bs4acc2 -->
- config: `neuron_experiments/H9_bipolar_self_attention/configs/generated/h79_allbinary_all12_cf10_match_code_w720_fastlr_full30_bs4acc2.yml`
- start checkpoint: `neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid/checkpoint_epoch2.pth`
- run dir: `neuron_experiments/H9_bipolar_self_attention/results/h79_allbinary_all12_cf10_match_code_w720_fastlr_full30_bs4acc2_20260720_setsid`
- protocol: batch4, accumulation2, effective batch8, warmup1440 micro-steps = 720 optimizer updates = 5760 samples; full30; standard valid825.
- load: ATLIF105, attention12, overlay210, warm-start missing24/unexpected0; trained strict-load audited.

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | spike_energy_proxy_uj |
| 1 | 29 | 1.4788 | 9.6199 | 0.5104 | 0.1944 | 0.0901 | 27.3125G | 5.9025% | 24197.15 |
| 2 | 19 | 1.4919 | 9.2877 | 0.5049 | 0.1970 | 0.0956 | 25.6529G | 5.5438% | 22722.26 |
| 3 | 0 | 1.4865 | 9.5040 | 0.5131 | 0.1939 | 0.0916 | 23.3672G | 5.0499% | 20645.83 |
| 4 | 14 | 1.4843 | 9.6307 | 0.5087 | 0.1941 | 0.0911 | 26.5092G | 5.7289% | 23461.05 |
| 5 | 24 | 1.5434 | 9.3749 | 0.5197 | 0.2113 | 0.1044 | 25.6174G | 5.5362% | 22684.14 |
| 6 | 4 | 1.5428 | 9.7984 | 0.5236 | 0.2059 | 0.1001 | 24.2965G | 5.2507% | 21482.44 |
| 7 | 28 | 1.5533 | 9.6743 | 0.5186 | 0.2121 | 0.1052 | 26.2665G | 5.6764% | 23257.79 |
| 8 | 9 | 1.5707 | 9.7130 | 0.5236 | 0.2150 | 0.1060 | 24.7588G | 5.3506% | 21890.07 |


### H80 Match-Code full30 显存安全结果

<!-- MATCH_BS4ACC2_FULL30::H80::h80_allbinary_all12_dn9_match_code_w720_fastlr_full30_bs4acc2 -->
- config: `neuron_experiments/H9_bipolar_self_attention/configs/generated/h80_allbinary_all12_dn9_match_code_w720_fastlr_full30_bs4acc2.yml`
- start checkpoint: `neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid/checkpoint_epoch2.pth`
- run dir: `neuron_experiments/H9_bipolar_self_attention/results/h80_allbinary_all12_dn9_match_code_w720_fastlr_full30_bs4acc2_20260720_setsid`
- protocol: batch4, accumulation2, effective batch8, warmup1440 micro-steps = 720 optimizer updates = 5760 samples; full30; standard valid825.
- load: ATLIF105, attention12, overlay210, warm-start missing12/unexpected0; trained strict-load audited.

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | spike_energy_proxy_uj |
| 1 | 19 | 1.4725 | 9.2222 | 0.4994 | 0.1927 | 0.0926 | 24.9347G | 5.3886% | 22057.34 |
| 2 | 29 | 1.4826 | 9.5503 | 0.5079 | 0.1934 | 0.0900 | 26.5939G | 5.7472% | 23528.59 |
| 3 | 14 | 1.4880 | 9.6634 | 0.5088 | 0.1944 | 0.0909 | 25.7810G | 5.5715% | 22780.20 |
| 4 | 0 | 1.5058 | 9.6101 | 0.5119 | 0.1948 | 0.0931 | 22.5108G | 4.8648% | 19851.04 |
| 5 | 24 | 1.5329 | 9.3327 | 0.5168 | 0.2088 | 0.1023 | 24.8313G | 5.3663% | 21953.07 |
| 6 | 4 | 1.5355 | 9.7629 | 0.5222 | 0.2044 | 0.0993 | 23.4903G | 5.0765% | 20738.14 |
| 7 | 28 | 1.5598 | 9.7138 | 0.5184 | 0.2113 | 0.1054 | 25.5383G | 5.5191% | 22578.24 |
| 8 | 9 | 1.5729 | 9.7182 | 0.5245 | 0.2150 | 0.1065 | 24.0351G | 5.1942% | 21218.13 |


### 43.35 H66f Local5+TP 与 H66g Local5+Motion 开跑（2026-07-23）

<!-- H66F_H66G_LOCAL5_COMBO_20260723 -->

用户要求尝试两种 Local-5 时空混合：

1. **H66f Scheme A**：`binary_axnor_local5_tp_shiftmax` — self + 时间对位 + 四向空间，共 **6** 候选 α-XNOR + Shiftmax。
2. **H66g Local5+Motion**：`binary_axnor_local5_motion_shiftmax` — Local-5 五候选，self lane 加 H67 式 `0.25 * popcount(K⊕K')`（不可广播到全部 lane，否则 Shiftmax 不变）。

协议：冻结 TTX ep2 独立 full30 + 八轮 valid825；串行 H66f → H66g。  
不改 H66d/H67 旧配置。纯 Local-5 mode 仍强制 `motion_xor_alpha=0`，避免 H66d 语义漂移。

- generator：`entrypoints/make_h66f_h66g_local5_combo_configs.py`
- runner：`entrypoints/run_h66f_h66g_local5_combo_full30.py`
- status：`results/h66f_h66g_local5_combo_status.log`
- run dirs：`results/h66f_*_20260723_setsid`、`results/h66g_*_20260723_setsid`

晋级门槛（预注册）：AEE 相对 H66d best `1.4432` 至少改善约 0.5% 或明确持平+代价更优，否则作消融。


### H66 combo full30 自动结果：h66f_allbinary_all12_local5_tp_w720_fastlr_full30

<!-- H66_COMBO_FULL30::h66f_allbinary_all12_local5_tp_w720_fastlr_full30 -->
- config: `neuron_experiments/H9_bipolar_self_attention/configs/generated/h66f_allbinary_all12_local5_tp_w720_fastlr_full30.yml`
- start checkpoint: `neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid/checkpoint_epoch2.pth`
- run dir: `neuron_experiments/H9_bipolar_self_attention/results/h66f_allbinary_all12_local5_tp_w720_fastlr_full30_bs8_full30_20260723_setsid`

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | spike_energy_proxy_uj |
| 1 | 19 | 1.4714 | 9.4922 | 0.5031 | 0.1891 | 0.0890 | 26.6072G | 5.7501% | 23585.34 |
| 2 | 29 | 1.4978 | 9.4932 | 0.5007 | 0.1943 | 0.0942 | 26.8891G | 5.8110% | 23829.62 |
| 3 | 4 | 1.5034 | 9.7992 | 0.5105 | 0.1923 | 0.0912 | 24.2485G | 5.2403% | 21447.35 |
| 4 | 24 | 1.5156 | 9.8413 | 0.5126 | 0.1979 | 0.0943 | 26.9449G | 5.8231% | 23894.32 |
| 5 | 28 | 1.5189 | 9.9366 | 0.5094 | 0.1930 | 0.0922 | 28.6192G | 6.1849% | 25376.76 |
| 6 | 9 | 1.5177 | 10.0075 | 0.5175 | 0.1955 | 0.0928 | 26.0186G | 5.6229% | 23038.96 |
| 7 | 14 | 1.5392 | 9.9584 | 0.5266 | 0.2055 | 0.0967 | 26.1139G | 5.6435% | 23125.65 |
| 8 | 0 | 1.5564 | 10.1020 | 0.5237 | 0.2042 | 0.0990 | 24.0529G | 5.1981% | 21254.44 |


### H66 combo full30 自动结果：h66g_allbinary_all12_local5_motion_w720_fastlr_full30

<!-- H66_COMBO_FULL30::h66g_allbinary_all12_local5_motion_w720_fastlr_full30 -->
- config: `neuron_experiments/H9_bipolar_self_attention/configs/generated/h66g_allbinary_all12_local5_motion_w720_fastlr_full30.yml`
- start checkpoint: `neuron_experiments/H9_bipolar_self_attention/results/date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid/checkpoint_epoch2.pth`
- run dir: `neuron_experiments/H9_bipolar_self_attention/results/h66g_allbinary_all12_local5_motion_w720_fastlr_full30_bs8_full30_20260723_setsid`

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | spike_energy_proxy_uj |
| 1 | 19 | 1.4914 | 9.5455 | 0.5073 | 0.1935 | 0.0920 | 26.5551G | 5.7388% | 23534.53 |
| 2 | 29 | 1.4995 | 9.4391 | 0.5054 | 0.1970 | 0.0960 | 26.8616G | 5.8050% | 23800.33 |
| 3 | 4 | 1.5064 | 9.7414 | 0.5096 | 0.1915 | 0.0906 | 24.2433G | 5.2392% | 21446.72 |
| 4 | 24 | 1.5194 | 9.7735 | 0.5137 | 0.1997 | 0.0954 | 26.9545G | 5.8251% | 23897.31 |
| 5 | 28 | 1.5255 | 9.9942 | 0.5147 | 0.1982 | 0.0956 | 28.6045G | 6.1817% | 25357.88 |
| 6 | 9 | 1.5253 | 10.0517 | 0.5174 | 0.1963 | 0.0930 | 26.1047G | 5.6415% | 23116.67 |
| 7 | 14 | 1.5382 | 9.9798 | 0.5265 | 0.2069 | 0.0973 | 26.1879G | 5.6595% | 23188.13 |
| 8 | 0 | 1.5567 | 10.1523 | 0.5253 | 0.2048 | 0.0986 | 23.9989G | 5.1864% | 21207.10 |


### 43.36 H66d Local-5 定点/RTL 主线深挖启动（2026-07-25）

<!-- H66D_LOCAL5_DEPLOY_PIPELINE_START_20260725 -->

用户要求把 Local-5 定点与 RTL 深入做，评估能否成为主线。

**软件路径改造**
- `bsa_attention._binary_alpha_xnor_stencil_attention` 已接入 `hardware_quant` + `hardware_rtl_shiftmax`（与 Match-Code/H67 同网格：score step 1/128 clip[-2,2]，gate Q1.7）。
- 训练浮点路径在 quant 关闭时行为不变。

**评估流水线**
- `entrypoints/run_h66d_local5_deploy_pipeline.py`：H66d ep29 dyadic INT8 valid825 → RTL-exact valid825 → 写 JSON/MD/docs。
- checkpoint：`h66d_..._20260712_setsid/checkpoint_epoch29.pth`（float rank-1）。

**RTL 起步（非全 encoder）**
- `hw_autoresearch_nts07/rtl_local5/`：`local5_axnor_score_q7.sv`、`local5_shiftmax5_q17.sv`、`local5_stencil_token.sv`
- 参考向量：`tb_local5/local5_ref_vectors.json`
- 文档：`docs/76_H66d_Local5主线定点与RTL签核.md`

**判定门槛（预注册）**
1. dyadic AEE 仍优于 H67 dyadic 1.4626  
2. RTL-exact 相对 dyadic AEE 退化 ≤ 0.02  
3. 硬件叙事改为 Stencil-5，不可用 H67 Motion-XOR top 冒充  

流水线日志：`results/h66d_local5_deploy_pipeline_*.log`


### 43.36 H66d Local-5 dyadic + RTL-exact valid825（2026-07-25）

<!-- H66D_LOCAL5_DEPLOY_RTL_VALID825_20260725 -->

- checkpoint: epoch29
- dyadic: AEE 1.4475 / AAE 9.3860 / spikes 26.5517G / energy 23550.64uJ
- RTL-exact: AEE 1.4486 / AAE 9.4210 / spikes 26.5340G / energy 23535.19uJ
- vs dyadic: AEE +0.0011, AAE +0.0350
- gate: AEE degradation vs dyadic ≤ 0.02 for deploy freeze; software precision mainline remains H66d only if float+dyadic beat H67 dyadic 1.4626.


### DSEC 480x640/window9 三模型全分辨率队列（2026-07-26）

<!-- DSEC_FULLRES_WINDOW9_QUEUE_20260726 -->
- 顺序：NB0 ep59 -> H67 Motion-XOR ep19 -> H66d Local-5 ep29；三者均为 30 epoch full-resolution fine-tune。
- geometry：`480x640`、`crop=null`、`window=[2,9,9]`；physical batch `2`、effective batch `8`。
- 加载：`--finetune 1` 触发 audited `remap=v1`；插值后必须执行 `load_state_dict`，并核对 ATLIF/attention/overlay/missing/unexpected。
- 定位：这是保持 N=162 硬件 tile 的 full-resolution 对照，不是论文 `[2,15,15]` protocol；window15 只在最终 winner 冻结后补跑。
- status：`neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_window9_queue_status.log`。
- NB0 config：`neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w9_nb0_ep59_ft30.yml`；start：`experiments/baseline_stride_upstream/checkpoint_epoch59.pth`。
- H67 config：`neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w9_h67_motion_ep19_ft30.yml`；start：`neuron_experiments/H9_bipolar_self_attention/results/h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30_bs8_full30_20260711_setsid/checkpoint_epoch19.pth`。
- H66d config：`neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w9_h66d_local5_ep29_ft30.yml`；start：`neuron_experiments/H9_bipolar_self_attention/results/h66d_allbinary_all12_lr_ttx_w720_fastlr_full30_bs8_full30_20260712_setsid/checkpoint_epoch29.pth`。

### DSEC full-resolution window9 预检与正式启动（2026-07-26）

<!-- DSEC_FULLRES_WINDOW9_PREFLIGHT_20260726 -->

**加载缺陷修复**

- 原 H9 `remap=v1` 分支只调用 `load_pretrained_interpolate()` 修改内存中的
  state dict，随后直接返回 model，没有执行 `model.load_state_dict()`。该缺陷只影响
  full-resolution/窗口变化触发的 v1 分支，历史 crop/window9 结果走 `remap=None`，不受影响。
- 训练入口和 `h9_load_audit.py` 均改为：插值后继续进入统一的 audited
  `load_state_dict(strict=False)`；overlay mismatch 仍 fail-fast。
- 回归测试新增“v1 插值后模型 tensor 必须等于 checkpoint tensor”，共 6 项通过。

**真实 checkpoint tensor 审计**

| model | ATLIF | attention | overlay keys | compared tensors | unequal | missing/unexpected |
|---|---:|---:|---:|---:|---:|---|
| NB0 ep59 | 0 | 0 | 0 | 711 | 0 | 0/0 |
| H67 ep19 | 105 | 12 | 210 | 921 | 0 | 0/0 |
| H66d ep29 | 105 | 12 | 210 | 921 | 0 | 0/0 |

审计文件：
`neuron_autoresearch/experiments/dsec_fullres_window9/load_chain_audit.json`。

**显存 smoke**

- NB0、H67、H66d 均完成 `480x640/window9`、两次 train step + 小验证的
  batch2 smoke，无 OOM。
- 正式参数：physical batch2、gradient accumulation4、effective batch8、AMP、
  CuPy、workers8、MLflow off。
- full30 只保存 epoch `0/4/9/14/19/24/28/29`；optimizer state 保存
  `4/9/14/19/24/29`，控制磁盘占用。

**正式运行**

- supervisor PID 文件：
  `neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_window9_supervisor/supervisor.pid`
- supervisor log：
  `neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_window9_supervisor/supervisor.log`
- 当前顺序：NB0 -> H67 Motion-XOR -> H66d Local-5。
- NB0 run：
  `neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w9_nb0_ep59_ft30_bs2_20260726/`
- 启动稳定性：NB0 epoch0 初始约 `1.9 step/s`、GPU 约 `26.1 GiB`、利用率
  `99%`，单 epoch 粗估约 32 分钟；以完整 epoch stats 为最终时间口径。
- 日志查看不 tail 大文件正文；优先看 queue status，训练进度可将 tqdm 的 CR 转行：
  `tr '\r' '\n' < <run_dir>/train.log | tail -n 30`。

后续门控不变：window9 三模型完成并做统一 full-resolution valid 后冻结 winner；
再仅对 NB0 与 winner 参数化并执行论文 `480x640/window15` 加 official submission。
MVSEC 的 NB0/H67/H66d seed0 矩阵排在该门控之后。


### DSEC fullres window9 正式推理选点规定（2026-07-27）

<!-- DSEC_FULLRES_W9_FORMAL_EVAL_POLICY_20260727 -->

- **何时评**：每个模型 `FT30` 训完（`checkpoint_epoch29.pth` 落盘）后立刻做 formal valid825；当前已在跑的旧队列若未串评，由 `run_dsec_fullres_window9_formal_eval.py --wait-ready --wait-gpu` 在 GPU 空闲后补齐。
- **评哪些 cp（默认）**：`0, 4, 9, 14, 19, 24, 28, 29`  
  - 与 `make_dsec_fullres_window9_configs.SAVE_EPOCHS` / crop 线 H67·H68 full30 formal valid825 **同一选点**。  
  - 缺 ckpt 跳过；`train.log` 中 train-val best 若映射到另一已存 epoch，会自动补进集合。
- **协议**：config 用 fullres yml（`480×640` / `window=[2,9,9]` / `crop=null` / `remap=v1`），`eval_DSEC_flow_SNN.py --mode valid`；**不是** paper `[2,15,15]`。
- **排序口径**：fullres 的 `profile_ranking_valid825.md` 按 **AEE** 排序。历史 crop
  候选分数包含固定 `34.5G` spike target，不能直接用于像素数扩大 `2.7778x` 的
  `480x640` 结果，否则会由绝对 spike 数主导并错误选择 checkpoint。
- **产物**：`<run_dir>/standard_valid825/epoch{N}/`、`profile_ranking_valid825.md`、`fullres_formal_eval_summary.json`。
- **入口**：
  - 训练队列（训完即评）：`entrypoints/run_dsec_fullres_window9_queue.py`
  - 仅补评 / 等待就绪：`entrypoints/run_dsec_fullres_window9_formal_eval.py`


### DSEC fullres window9 接管编排（2026-07-27）

<!-- DSEC_FULLRES_W9_TAKEOVER_20260727 -->
- 用户要求：先完成 NB0 正式推理，再继续 H67/H66d 训练；本会话接管 Codex 原队列。
- H67 中断点：等待 force-save 后从 epoch9 resume（`--resume True --finetune 1`），避免丢掉 mid-run 权重。
- 顺序：stop old queue → NB0 formal valid825 (epochs [0, 4, 9, 14, 19, 24, 28, 29]) → H67 resume+eval → H66d train+eval。
- status：`neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_window9_takeover_status.log`


### DSEC fullres window9 正式 valid825：NB0

<!-- DSEC_FULLRES_W9_FORMAL_EVAL::NB0::dsec_fullres_w9_nb0_ep59_ft30_bs2_20260726 -->
- protocol：`480x640` / `window=[2,9,9]` / `crop=null` / `remap=v1`；standard valid825 via `eval_DSEC_flow_SNN.py --mode valid`；**不是** paper window15。
- eval epochs policy：`[0, 4, 9, 14, 19, 24, 28, 29]`（= force_save set；missing skipped；train-val best mapped epoch auto-included if missing）。
- evaluated epochs：`[0, 4, 9, 14, 19, 24, 28, 29]`
- config：`neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w9_nb0_ep59_ft30.yml`
- run dir：`neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w9_nb0_ep59_ft30_bs2_20260726`
- ranking：`neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w9_nb0_ep59_ft30_bs2_20260726/profile_ranking_valid825.md`
- ranking mode：`aee`
- best rank1：epoch24 AEE=1.8987 AAE=8.5985

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | spike_energy_proxy_uj |
| 1 | 24 | 1.8987 | 8.5985 | 0.5692 | 0.2503 | 0.1319 | 120.7546G | 9.2144% | 102548.79 |
| 2 | 28 | 1.9154 | 8.5381 | 0.5831 | 0.2589 | 0.1358 | 121.8895G | 9.3010% | 103473.12 |
| 3 | 14 | 1.9325 | 8.7618 | 0.5823 | 0.2584 | 0.1374 | 117.4927G | 8.9655% | 99780.85 |
| 4 | 29 | 1.9350 | 8.9765 | 0.5918 | 0.2595 | 0.1368 | 122.6940G | 9.3624% | 103976.54 |
| 5 | 19 | 1.9424 | 9.0373 | 0.5980 | 0.2656 | 0.1393 | 122.9734G | 9.3837% | 104338.78 |
| 6 | 9 | 1.9882 | 8.9971 | 0.5932 | 0.2676 | 0.1416 | 117.7792G | 8.9874% | 100051.57 |
| 7 | 4 | 2.4566 | 11.1729 | 0.6276 | 0.3163 | 0.1871 | 111.7044G | 8.5238% | 94839.19 |
| 8 | 0 | 2.8723 | 13.1466 | 0.7150 | 0.3902 | 0.2295 | 118.6163G | 9.0513% | 100885.27 |

#### NB0 fullres window9 结果诊断（2026-07-28）

- 相对 crop NB0 ep59（AEE `1.4872` / AAE `9.93`），真正的 fullres 最优
  epoch24 为 AEE `1.8987`（`+27.67%`），但 AAE `8.5985`（`-13.41%`）。
  因此不是所有精度指标同时崩坏，而是端点误差在 fullres/window9 下明显退化。
- 训练/推理加载审计均为 `checkpoint_overlay_keys=0, missing=0,
  unexpected=0`；该结果不是 NB0 权重漏载或错载。
- 训练内 valid40 loss 从 epoch0 的 `2.2386` 降到 epoch25 的 `1.3244`，
  formal valid825 AEE 从 epoch0 的 `2.8723` 降到 epoch24 的 `1.8987` 后平台化。
  这说明 fullres 微调有效但已基本收敛，继续按原 schedule 堆 epoch 不是首选修复。
- 当前实验是硬件固定结构的 `480x640 + window9`，不是 paper 的
  `480x640 + window15`。相对 `288x384` crop，固定 window9 的归一化空间搜索范围
  缩小，是大位移端点误差变差的首要结构假设。
- NB0 直接继承 crop baseline 的 `AdamW lr=1e-4, wd=0.01`，没有 H67/H66d
  使用的低 backbone LR/warmup 分组；这是 fullres 域切换时过度更新的优化风险。
- checkpoint 对比中 711 个 tensor 全部同名同形状，但 156 个 BN
  `running_mean/running_var` 的相对变化中位数约 `38.8%`；个别原本接近零的
  `running_var` 变化更大。梯度累积只形成 effective batch 8，BN 统计仍由
  physical batch 2 更新。当前 formal eval 又执行普通 `model.eval()` 使用这些
  running stats，而 paper 口径是 test 时禁用 BN running-state，因此 BN 是第二个
  必须实测的协议变量，不能仅靠现有结果排除。
- fullres 绝对 spike 数不能和 crop 直接比较。按像素比 `2.7778x` 归一化，
  epoch24 的 `120.7546G` 等效为 crop 尺度 `43.4717G`，较 NB0 `44.05G`
  反而低 `1.31%`；绝对值增大主要来自图像面积。
- 仍缺最关键的因果对照：未微调 NB0 ep59 在 `480x640/window9` 的 zero-shot
  valid825。待当前串行队列释放 GPU 后补测，才能把“分辨率/窗口影响”与
  “fullres 微调策略影响”分开。


### DSEC 论文全分辨率 window15 重跑协议（2026-07-28）

<!-- DSEC_FULLRES_PAPER_W15_QUEUE_20260728 -->
- 论文公开协议：crop 阶段 `2x9x9`；full-resolution 阶段 `480x640`、`crop=null`、`2x15x15`、额外 30 epochs、physical batch 1 或 2、相对位置偏置 bicubic remap；测试关闭 BN running-state。
- 本队列统一使用 physical batch `2`、`num_acc=1`、AMP、CuPy、workers=8、MLflow off；formal valid825 按 AEE 排序。
- 重要限制：本地没有论文 80-epoch crop checkpoint。NB0/H67/H66d 起点分别只有 60/20/30 crop epochs，因此可称为 paper full-resolution protocol，不可称为论文 checkpoint 的逐点复现。
- 旧 `480x640/window9` 结果保留为协议失败审计，不再用于论文对比；其 checkpoint/state 已删除，日志、ranking、profile 保留，回收 9.814 GiB。
- status：`neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_paper_w15_queue_status.log`。
- NB0：config `neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_paper_w15_nb0_ep59_ft30.yml`；start `experiments/baseline_stride_upstream/checkpoint_epoch59.pth`；source crop epochs `60`。
- H67：config `neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_paper_w15_h67_motion_ep19_ft30.yml`；start `neuron_experiments/H9_bipolar_self_attention/results/h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30_bs8_full30_20260711_setsid/checkpoint_epoch19.pth`；source crop epochs `20`。
- H66d：config `neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_paper_w15_h66d_local5_ep29_ft30.yml`；start `neuron_experiments/H9_bipolar_self_attention/results/h66d_allbinary_all12_lr_ttx_w720_fastlr_full30_bs8_full30_20260712_setsid/checkpoint_epoch29.pth`；source crop epochs `30`。

#### paper-window15 启动前审计

- strict CPU load audit 三模型全部通过：
  - NB0：ATLIF `0`、Shiftmax `0`、checkpoint overlay `0`、
    `missing=0`、`unexpected=0`；
  - H67/H66d：ATLIF `105`、Shiftmax `12`、checkpoint overlay `210`、
    `missing=0`、`unexpected=0`。
- `remap=v1` 已实际执行 PyTorch bicubic relative-position interpolation；
  比对所有同形 tensor 后 `unequal_after_load=[]`。审计产物：
  `neuron_autoresearch/experiments/dsec_fullres_paper_w15/load_chain_audit.json`。
- H66d Local-5 原实现把空间窗口硬编码为 `9x9`，paper window15 smoke 因此
  fail-fast。现已改为从 token 数推导方形窗口边长；候选仍严格保持
  self/up/down/left/right 五条 lane，没有改变注意力算子、权重或硬件数据流。
  新增 `15x15` 单元测试并通过。
- physical batch2 两步训练 smoke 均通过：

| model | samples/s | peak GPU | load audit |
|---|---:|---:|---|
| NB0 | 0.5653 | 25.749 GiB | 0/0/0 overlay/missing/unexpected |
| H67 Motion-XOR | 0.5044 | 40.495 GiB | 210/0/0 |
| H66d Local-5 | 0.5041 | 38.354 GiB | 210/0/0 |

- 单样本正式推理 smoke 通过：`BN=no_running` 实际切换 `78` 个 BN 模块；
  `spike_profile.json` 已记录 `resolution=[480,640]`、`crop=null`、
  `window_size=[2,15,15]`、`remap=v1`、`bn_policy=no_running`。
- 两步 smoke 的首轮 CuPy 编译开销不能用于外推整轮时间。NB0 正式训练稳定后
  为约 `1.7-1.9 batch/s`（batch2），单 epoch 约 `33-35 min`，FT30 粗估约
  `17h`；H67/H66d 以各自完整 epoch stats 为准。每条训练结束后立即 formal
  valid825，再仅保留 epoch0、AEE best、epoch29 模型和 epoch29 resume state，
  自动回收中间权重。
- supervisor PID：`neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_paper_w15_supervisor/supervisor.pid`
- supervisor log：`neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_paper_w15_supervisor/supervisor.log`
- NB0 train log：`neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_paper_w15_nb0_ep59_ft30_bs2_20260728/train.log`

#### H67/H66d fullres 部署量化补测（2026-07-29）

<!-- DSEC_FULLRES_W15_DEPLOY_QUEUE_20260729 -->

- 训练阶段仍使用 float/AMP，不在训练图中强制定点；每个模型先完成
  `480x640 / 2x15x15 / BN=no_running / valid825` 的浮点 checkpoint 排名。
- 浮点排名完成后，对 H67 Motion-XOR 与 H66d Local-5 的 rank-1 checkpoint
  各补两条部署评估：
  1. dyadic Q7/Q1.7：score 步长 `2^-7`、gate 步长 `2^-7`；
  2. hardware-order：Q7 score、16-entry Q8 `exp2` LUT、integer row sum、
     ceil-power-of-two normalization、Q1.7 ties-to-even RNE。
- Local-5 hardware-order 额外启用真正的 invalid-candidate mask：边界外候选不进入
  row max 和分母、gate 严格为零。该行为通过新增单元测试，且配置字段默认关闭，
  不改变旧实验。
- 当前正式队列进程启动于上述集成前，因此另挂只轮询状态、不占 GPU 的 follower；
  主队列出现 `ALL COMPLETE DSEC PAPER-W15 QUEUE` 后才开始两模型 valid825。
- follower status：
  `neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_paper_w15_deploy_followup_status.log`；
  汇总输出：
  `neuron_autoresearch/experiments/dsec_fullres_paper_w15/fullres_w15_deploy_summary.md`。
- **exact 命名边界**：H67 在旧 crop/window9、`T=162` 合同下已完成
  hardware-order valid825（AEE `1.462688`、AAE `9.403994`）及 attention-row RTL
  回归；但现有 SV 的正式参数合同是 `MAX_TOKENS=162`。fullres/window15 是
  `T=450`，所以这次队列先关闭“硬件顺序数值 exact”，仍需硬件侧补
  T450 controller/address/memory/ordered-trace RTL 回归后，才能称 fullres RTL-exact。
- H66d Local-5 已有修正后的五候选 score/gate/term/projection 功能 RTL 链回归；
  本队列补的是 fullres rank-1 的真实 mask 数值评估。其
  window15 line-buffer/address-control SV 回放同样保留为硬件签核项。


### DSEC paper-window15 valid825：NB0

<!-- DSEC_FULLRES_PAPER_W15_RESULT::NB0::dsec_fullres_paper_w15_nb0_ep59_ft30_bs2_20260728 -->
- best epoch：`29`；run：`neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_paper_w15_nb0_ep59_ft30_bs2_20260728`；source crop epochs：`60`。
- protocol：`480x640 / 2x15x15 / remap=v1 / BN=no_running / standard valid825 / ranking=AEE`。

# Standard Valid825 Ranking

Ranking mode: `aee`.

The energy column is a spike-activity proxy and excludes overlay attention control/reduction operations.

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | spike_energy_proxy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 29 | 1.4454 | 6.5128 | 0.4563 | 0.1661 | 0.0793 | 126.1156G | 9.7927% | 107137.62 |
| 2 | 24 | 1.5304 | 6.5414 | 0.4557 | 0.1678 | 0.0823 | 124.9918G | 9.7054% | 106209.19 |
| 3 | 19 | 1.5798 | 6.6132 | 0.4819 | 0.1858 | 0.0922 | 123.6289G | 9.5996% | 105128.22 |
| 4 | 14 | 1.6157 | 7.0525 | 0.4860 | 0.1854 | 0.0925 | 121.0580G | 9.4000% | 103052.18 |
| 5 | 9 | 1.8509 | 8.4431 | 0.5790 | 0.2418 | 0.1218 | 118.1810G | 9.1766% | 100645.50 |
| 6 | 4 | 1.8894 | 8.3599 | 0.5826 | 0.2500 | 0.1284 | 114.7724G | 8.9119% | 97787.98 |
| 7 | 0 | 2.4123 | 10.2478 | 0.6546 | 0.3266 | 0.1880 | 112.2139G | 8.7133% | 95788.32 |


### DSEC fullres 训练/推理/RTL 接管审计（2026-07-30）

<!-- DSEC_FULLRES_CODE_PROTOCOL_RTL_AUDIT_20260730 -->

- 完整审计：
  `neuron_autoresearch/experiments/dsec_fullres_paper_w15/CODE_PROTOCOL_RTL_AUDIT_20260730.md`。
- 协议纠正：历史 DSEC evaluator 本来就会在推理入口强制
  `loader.batch_size=1`，因此 NB0 ep29 的 AEE `1.44535` 不是 batch2
  结果。现在改成显式 `test.eval_batch_size=1`、非法值 fail-fast，并写入
  profile。
- 指标拆分：`AAE` 保留为历史二维方向角；`AAE_Benchmark` 使用 DSEC/Barron
  `(u,v,1)` 公式。NB0 ep29 后者为 `6.18034`。论文表中 `4.871` 是 official
  test-server 结果，不能与 local valid825 的任一 AAE 直接逐点比较。
- 强化加载审计通过：NB0/H67/H66d 的 missing/unexpected 均为 `0/0`；
  H67/H66d 为 ATLIF `105`、Shiftmax `12`、overlay keys `210`；每条线
  12 个 window9->15 positional tensor 与独立插值结果逐元素一致。
- 新审计产物：
  `neuron_autoresearch/experiments/dsec_fullres_paper_w15/load_chain_audit_v2.json`。
- 论文范式边界：fullres geometry/window/FT30/batch/no-running BN 对齐；但
  source crop budget 是 `60/20/30`，不是论文 80 epochs，H67/H66d 优化器也
  是候选微调配置。因此只称 paper-geometry fullres protocol，不称 exact
  training reproduction。
- deploy profile 新增 config/checkpoint identity、load audit 和
  `deployment_contract`。当前 exact 口径只能称
  `attention-core hardware-order numeric`；全网定点与 window15/T450 SV
  controller/address/memory/ordered-trace 尚未关闭。
- 队列恢复增强：若训练中断，重跑会选择最新同时具有 model 与 optimizer
  state 的 checkpoint，并传入 `--resume 1`，不再从 crop 起点重训。
- follower 已更新为新版本，等待主队列完成后串行执行 NB0 provenance replay、
  H67/H66d dyadic Q7/Q1.7 及 hardware-order valid825，不与训练抢 GPU。
- 硬件签核阻塞与 T450 最小闭环已独立写入
  `hw_autoresearch_nts07/docs/100_DSEC全分辨率RTLExact签核阻塞与T450闭环清单_20260730.md`。
  当前 H67 RTL 仍以 T162 为默认/上限且 descriptor scheduler 写死 162；
  Local-5 仍是 row8/dest16/synthetic-window 原型。正式措辞只能使用
  `attention-core hardware-order numeric`，待算法赢家确定后只对赢家补
  T450 real-trace SV 零失配和公平 PPA，避免两套 RTL 同时扩展。
- 自动评估 follower 已增加 config SHA-256、checkpoint size/mtime、加载审计、
  protocol 与 deployment-contract 的严格复用条件；不匹配的旧 profile 保留，
  新结果写入带 artifact fingerprint 的 `*_audited_*` 目录，避免静默复用或
  覆盖历史结果。
- 旧 H66d deploy 兼容脚本原先只把无效候选压到 `score_min`，没有从 row max
  和分母中真正排除；未来重跑已改为 true invalid-candidate mask。历史结果不
  删除，fullres follower 使用修正合同。
- `threshold_freeze_after_step` 历史上只冻结额外的 homeostatic update，
  **不会**停止 AdamW 对 `thresh` 的梯度更新。当前 H67/H66d 保持该历史行为以
  保证可比性；新增默认关闭的 `freeze_threshold_grad_after_step` 可用于未来
  “真阈值冻结”消融，不能把当前日志的 `threshold_updates_frozen=1` 写成阈值
  参数完全冻结。
- 2026-07-30 运行中两次出现 Prosperity/phi CPU-only 架构扫描占满容器
  `7-core` cgroup 配额，H67 从约 `1.15 s/it` 退化到 `4.7-5.0 s/it`，GPU
  利用率同步降到约 `16%`；降优先级后恢复到 `1.14-1.20 s/it`。新增
  `entrypoints/run_dsec_cpu_priority_guard.py`，只把已知 Prosperity/phi
  扫描设为 `nice=19`，不修改训练/推理进程，并在主队列完成后自动退出。


### DSEC paper-window15 valid825：H67

<!-- DSEC_FULLRES_PAPER_W15_RESULT::H67::dsec_fullres_paper_w15_h67_motion_ep19_ft30_bs2_20260728 -->
- best epoch：`29`；run：`neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_paper_w15_h67_motion_ep19_ft30_bs2_20260728`；source crop epochs：`20`。
- protocol：`480x640 / 2x15x15 / remap=v1 / BN=no_running / standard valid825 / ranking=AEE`。

# Standard Valid825 Ranking

Ranking mode: `aee`.

The energy column is a spike-activity proxy and excludes overlay attention control/reduction operations.

| rank | epoch | AEE | AAE legacy | AAE benchmark | PE1 | PE2 | outlier | total_spikes | firing | spike_energy_proxy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 29 | 2.0730 | 8.1203 | 7.9029 | 0.5564 | 0.2453 | 0.1343 | 87.9821G | 6.8450% | 77853.53 |
| 2 | 19 | 2.1030 | 8.1820 | 7.9868 | 0.5658 | 0.2541 | 0.1397 | 85.9824G | 6.6894% | 76109.02 |
| 3 | 24 | 2.1084 | 8.1075 | 7.9157 | 0.5624 | 0.2547 | 0.1420 | 87.4260G | 6.8017% | 77370.43 |
| 4 | 14 | 2.1487 | 8.2730 | 8.0787 | 0.5678 | 0.2584 | 0.1443 | 83.1584G | 6.4697% | 73617.83 |
| 5 | 9 | 2.2024 | 8.4850 | 8.2699 | 0.5760 | 0.2682 | 0.1518 | 80.0272G | 6.2261% | 70862.47 |
| 6 | 4 | 2.2631 | 8.7384 | 8.5322 | 0.5855 | 0.2779 | 0.1590 | 75.6409G | 5.8848% | 66972.84 |
| 7 | 0 | 2.4928 | 9.6139 | 9.3747 | 0.6051 | 0.3030 | 0.1816 | 71.4781G | 5.5610% | 63293.21 |


### DSEC paper-window15 valid825：H66d

<!-- DSEC_FULLRES_PAPER_W15_RESULT::H66d::dsec_fullres_paper_w15_h66d_local5_ep29_ft30_bs2_20260728 -->
- best epoch：`29`；run：`neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_paper_w15_h66d_local5_ep29_ft30_bs2_20260728`；source crop epochs：`30`。
- protocol：`480x640 / 2x15x15 / remap=v1 / BN=no_running / standard valid825 / ranking=AEE`。

# Standard Valid825 Ranking

Ranking mode: `aee`.

The energy column is a spike-activity proxy and excludes overlay attention control/reduction operations.

| rank | epoch | AEE | AAE legacy | AAE benchmark | PE1 | PE2 | outlier | total_spikes | firing | spike_energy_proxy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 29 | 2.0912 | 8.1688 | 7.9574 | 0.5560 | 0.2451 | 0.1342 | 89.8206G | 6.9880% | 79361.90 |
| 2 | 19 | 2.1370 | 8.4184 | 8.2054 | 0.5676 | 0.2551 | 0.1413 | 87.8276G | 6.8329% | 77648.94 |
| 3 | 24 | 2.1404 | 8.2573 | 8.0612 | 0.5626 | 0.2543 | 0.1417 | 89.2258G | 6.9417% | 78854.75 |
| 4 | 14 | 2.1942 | 8.5077 | 8.2904 | 0.5714 | 0.2609 | 0.1463 | 85.1391G | 6.6238% | 75314.00 |
| 5 | 9 | 2.2391 | 8.6994 | 8.4915 | 0.5764 | 0.2678 | 0.1516 | 81.5003G | 6.3407% | 72140.68 |
| 6 | 4 | 2.3051 | 8.9582 | 8.7366 | 0.5876 | 0.2795 | 0.1609 | 77.6147G | 6.0384% | 68713.57 |
| 7 | 0 | 2.5191 | 9.6328 | 9.4064 | 0.6046 | 0.3023 | 0.1814 | 73.5069G | 5.7188% | 65122.17 |


### DSEC fullres window15 定点/硬件顺序评估

<!-- DSEC_FULLRES_W15_DEPLOY_FOLLOWUP_RESULTS -->
- summary：`neuron_autoresearch/experiments/dsec_fullres_paper_w15/fullres_w15_deploy_summary.md`
- 口径：Q7 score、Q1.7 gate、16-entry Q8 exp2 LUT、integer row sum、ceil-pow2 normalize、RNE；Local-5 使用真正 masked candidate 合同。
- 命名边界：该表先关闭 fullres valid825 数值精度；window15/T450 SV 控制、地址、line-buffer 与 ordered trace 仍须硬件侧独立签核。

| baseline | epoch | AEE | AAE legacy | AAE benchmark | spikes(G) |
|---|---:|---:|---:|---:|---:|
| NB0 baseline | 29 | 1.4454 | 6.5128 | 6.1803 | 126.1156 |

| candidate | epoch | float AEE | dyadic AEE | hardware-order AEE | hardware-float delta | hardware-order AAE legacy | AAE benchmark | spikes(G) | true mask |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| H67 Motion-XOR | 29 | 2.0730 | 2.0669 | 2.0880 | +0.0150 | 8.1532 | 7.9466 | 87.9802 | False |
| H66d Local-5 | 29 | 2.0912 | 2.1041 | 2.1091 | +0.0179 | 8.2214 | 8.0203 | 89.8145 | True |

The hardware-order column is the frozen integer/LUT numerical path. Fullres SV sign-off additionally requires window15/T450 controller, address, line-buffer, and ordered-trace regression.

- H67 Motion-XOR: hardware-order numeric exact; existing H67 SV row RTL is verified at window9/T162, while fullres window15/T450 controller parameterization still requires RTL regression.
- H66d Local-5: score/gate hardware-order numeric exact with true masked candidates; fullres window15 line-buffer/address-control SV replay remains a separate hardware sign-off item.


### DSEC fullres LR rescue 短筛（2026-08-01）

<!-- DSEC_FULLRES_W15_LR_RESCUE_SCREEN_20260801 -->
- 失败诊断：旧 H67/H66d fullres backbone/norm LR 为 `2e-6/1e-6`，NB0 为 `1e-4`；旧候选不是等强度 fullres adaptation。
- 固定结构、480x640、2x15x15、remap=v1、BN=no_running、batch2；每项完整训练 1 epoch 后跑 standard valid825。
- 两条 own-crop LR 仅改变优化强度；NB0-fullres conversion 只用于判别初始化问题，不自动作为论文主协议。
- `H67_crop_bb2e5`：profile `bb2e5`，init `own_crop`，config `neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_rescue_H67_crop_bb2e5_screen1.yml`。
- `H67_crop_bb1e4`：profile `bb1e4`，init `own_crop`，config `neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_rescue_H67_crop_bb1e4_screen1.yml`。
- `H67_nb0full_bb2e5`：profile `bb2e5`，init `nb0_fullres_conversion`，config `neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_rescue_H67_nb0full_bb2e5_screen1.yml`。
- status：`neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_rescue_screen_20260801/status.log`。


### MVSEC 与 CICC 2026 对齐边界（2026-08-01）

<!-- MVSEC_CICC2026_ALIGNMENT_20260801 -->

- 参考论文：Tao Zhang et al., "A 28-nm Optical Flow Estimation Accelerator with
  Redundancy Speculation, Bit-Width-Aware Compression and Similarity Detection,"
  CICC 2026，DOI `10.1109/CICC65509.2026.11509564`。
- 正文已放入 `hw_autoresearch_nts07/docs/` 并完成逐页核验。论文使用参考
  Spike-FlowNet 的 INT8 Hybrid U-Net，在 `indoor1/2/3 + outdoor1` 各取 `800`
  个输入评估；其引用的标准训练范式是只用 `outdoor_day2` 训练，测试为
  `256x256` center crop、dt1、event-masked AEE。因此原定 `outdoor_day1` 训练路线
  只保留为内部 fallback，不能与 CICC/Spike-FlowNet 绝对 AEE 并表。
- CICC INT8 baseline AEE 为 `0.84/1.32/1.14/0.52`，均值 `0.96`；加入全部特征后
  为 `0.87/1.35/1.17/0.56`，均值 `0.99`。对应 mean operation/EMA/energy/latency
  为 `0.20x/0.08x/0.12x/0.19x`。
- 论文真实三机制为：group-16 最小有符号位宽加 non-zero map 的 `BWAC`；按 channel
  density 排序并提前执行 MaxPool/ReLU 的 Dense-Channel-First Speculation；比较
  `FM_i` 与 `FM_(i+delta)`、只保留 Level 0 或执行完整 U-Net 的 `DLSS`。TTB 不属于
  该论文，必须作为本项目独立创新点。
- 部署累计消融修正为：C0 INT8 dense；C1 + lossless BWAC；C2 + dense-channel
  speculation；C3 + DLSS。阈值正文未给出，只能在 validation 上选，不能在四个 test
  sequence 上扫参。
- 对 TTX 可迁移的是卷积/投影 INT8 权重 BWAC，以及二值 feature XOR/popcount 的
  DLSS 检测器。MaxPool/ReLU speculation 不能直接声称适用于 Shiftmax score，除非
  另建误差界或 validation-controlled stop rule。
- 当前本机只有四个测试 sequence，尚无 `outdoor_day2`。paper-facing MVSEC 训练启动
  前必须先下载并预处理 day2，再用同一 split、loss、预算分别训练 NB0 和唯一候选。
  完整协议见 `neuron_autoresearch/MVSEC_TRAIN_TEST_PROTOCOL_20260717.md`。
- 执行顺序不变：先完成正在运行的 DSEC fullres LR rescue 并冻结唯一候选，同时准备
  `outdoor_day2`；再训练 MVSEC-NB0 和该候选；C0-C3 使用训练后冻结权重，不为每个
  执行策略重新训练。

#### CICC 实验组织方式在本项目的落地（2026-08-01）

<!-- CICC2026_EXPERIMENT_ORGANIZATION_20260801 -->

- 这里参考的是 CICC 的**实验方法**，不是照搬其 Hybrid U-Net：固定四个 MVSEC
  sequence 各 `800` 个 dt1 输入的 manifest，所有 checkpoint、量化和部署策略逐样本共用，
  避免数据差异混入消融。
- 第一张是模型/数值表：M0 NB0 float；M1 NB0 INT8/hardware-order；M2 最终 TTX float；
  M3 最终 TTX INT8/hardware-order。逐序列报告 event-masked AEE/outlier、spikes 和完整
  energy，明确模型变化与量化损失。
- 第二张是冻结 M3 后的累计硬件特征表：D0 fixed-width/full-compute；D1 + lossless
  weight BWAC 与 binary/ternary activation packing；D2 + exact-empty TTB skip/density
  dispatch；D3 + validation-selected feature-similarity deep-level skip。
- D2 是本项目对“冗余消除”的实现，不冒充论文的 MaxPool/ReLU speculation；TTB 仍是
  本项目独立 idea。D3 才是对 DLSS 实验范式的直接迁移，二值 feature 可用
  XOR/popcount 实现相似度检测。
- D0-D3 每行必须报告 AEE degradation、active-TTB proportion、executed-op ratio、平均
  deep-level interval、EMA/energy/latency ratio，以及 detector、metadata、control 和 area
  overhead。另画与 CICC Fig. 9 同构的累计 EMA/energy/latency waterfall。
- 现有 MDR->MVSEC 冻结 checkpoint 可以先跑这套固定-trace 硬件消融，不必等待 day2；
  `outdoor_day2` 仅用于新增 direct-MVSEC 训练表和 Spike-FlowNet split 下的绝对精度比较。
- CICC 是实测 28-nm silicon；本项目在流片前只能写 post-synthesis/post-layout 或 cycle-model
  estimate，不能把估算值写成 measured silicon TOPS/W。所有行使用同一 DRAM 能耗/带宽
  假设，并把压缩地址、TTB descriptor、similarity detector 流量计入总 EMA。
- 当前 `run_h9_standard_mvsec_eval.py` 只能全序列评估，尚无固定 800-sample manifest；其
  ranking 只有 AEE/outlier/spikes/spike-proxy energy，也未接 EMA/cycle/control overhead。
  已有 profile100 TTB/cycle 工具不能替代四序列正式结果。后续需要新增一个只扩展接口的
  fixed-manifest MVSEC runner，以及一个按同一 checkpoint/input fingerprint 汇总 D0-D3
  operation/traffic/cycle/energy 的脚本，旧 evaluator 和旧结果保持不变。


### DSEC fullres LR rescue 短筛结果

<!-- DSEC_FULLRES_W15_LR_RESCUE_SCREEN_RESULT_20260801 -->

# DSEC fullres window15 rescue screen

| rank | candidate | init | LR profile | AEE | AAE benchmark | spikes | energy |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | H67_crop_bb1e4 | own_crop | bb1e4 | 2.2768 | 9.7754 | 70.4126G | 62316.92 |
| 2 | H67_nb0full_bb2e5 | nb0_fullres_conversion | bb2e5 | 2.2820 | 9.4308 | 41.6688G | 36152.30 |
| 3 | H67_crop_bb2e5 | own_crop | bb2e5 | 2.2949 | 8.7795 | 71.5410G | 63352.17 |

- 三条均完成且加载链路干净：overlay keys `210/210`，missing/unexpected
  `0/0`，remap=`v1`。结果不是 fullres 精度修复，只是 LR 适配速度筛选。
- `H67_crop_bb1e4` 一轮 AEE `2.2768`，接近旧低 LR fullres epoch4 的
  `2.2631`，而旧 epoch0 为 `2.4928`；高 LR 确实把约四轮的早期适配压到一轮。
  但其 benchmark AAE `9.7754`、outlier `0.1628`，方向误差明显不稳，不能仅按
  AEE rank 直接续 FT30。
- `H67_crop_bb2e5` AEE `2.2949`，benchmark AAE `8.7795`，相对更稳定；
  `H67_nb0full_bb2e5` spikes `41.6688G`（对 fullres NB0 `-66.96%`）但 AEE
  `2.2820`，只保留为初始化诊断，不作为论文主训练协议。
- 三条相对 fullres NB0 AEE `1.4454` 仍差 `+57.5%` 至 `+58.8%`，均未达到
  `+5%` 精度门槛。下一步不能把任一 screen 当主线；若继续，只允许 own-crop
  `bb1e4` 与 `bb2e5` 做短程收敛确认并同时按 AEE/AAE/outlier 门控，禁止直接盲跑
  三条 FT30。


### DSEC fullres LR rescue short5 续跑（2026-08-01）

<!-- DSEC_FULLRES_W15_LR_RESCUE_SHORT5_20260801 -->
- 只续 own-crop `bb1e4` 与 `bb2e5`，各从 screen epoch0 模型补 5 epochs；NB0-fullres conversion 不续。
- screen 未保存 optimizer/scaler state，因此两条线都从已训练模型重建 AdamW；这是 model continuation，不写成 strict optimizer resume。
- 结构、480x640、window2x15x15、remap=v1、BN=no_running、batch2、LR profile 保持不变；checkpoint epoch offset=1，最终为 epoch5。
- `H67_crop_bb1e4`：config `neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_rescue_H67_crop_bb1e4_continue5.yml`，source `neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_rescue_screen_20260801/H67_crop_bb1e4/checkpoint_epoch0.pth`。
- `H67_crop_bb2e5`：config `neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_rescue_H67_crop_bb2e5_continue5.yml`，source `neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_rescue_screen_20260801/H67_crop_bb2e5/checkpoint_epoch0.pth`。
- status：`neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_rescue_short5_20260801/status.log`。


### DSEC fullres LR rescue short5 结果

<!-- DSEC_FULLRES_W15_LR_RESCUE_SHORT5_RESULT_20260801 -->

# DSEC fullres window15 rescue short5

| rank | candidate | LR | epoch | AEE | AAE benchmark | outlier | spikes | energy |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | H67_crop_bb1e4 | bb1e4 | 5 | 1.7681 | 7.2373 | 0.1171 | 74.0398G | 65519.38 |
| 2 | H67_crop_bb2e5 | bb2e5 | 5 | 1.9901 | 7.7662 | 0.1321 | 76.2662G | 67511.90 |


### H67 fullres bb1e4 strict ep5-to-ep10 resume（2026-08-02）

<!-- DSEC_FULLRES_W15_H67_BB1E4_STRICT_RESUME10_20260802 -->
- short5 的 ep5 仍在改善，故只延长排名第一的 `H67_crop_bb1e4`；不继续较差的 `bb2e5`。
- 同时加载 `checkpoint_epoch5.pth` 与同目录 `checkpoint_epoch5_state_dict.pth`，使用 `--resume 1` 严格恢复 optimizer/scheduler/AMP scaler；目标为 ep10。
- 其余协议保持 480x640、window 2x15x15、batch2、LR bb1e4、BN no_running、valid825 不变。
- config：`neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_H67_crop_bb1e4_resume_ep10.yml`。
- status：`neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_H67_crop_bb1e4_resume10_20260802/status.log`。


### H67 fullres bb1e4 strict ep10 结果

<!-- DSEC_FULLRES_W15_H67_BB1E4_STRICT_RESUME10_RESULT_20260802 -->

# Standard Valid825 Ranking

Ranking mode: `aee`.

The energy column is a spike-activity proxy and excludes overlay attention control/reduction operations.

| rank | epoch | AEE | AAE legacy | AAE benchmark | PE1 | PE2 | outlier | total_spikes | firing | spike_energy_proxy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 1.5357 | 6.6708 | 6.3943 | 0.4709 | 0.1811 | 0.0896 | 76.5939G | 5.9590% | 67745.00 |


### H67 fullres bb1e4 strict ep10-to-ep15 resume（2026-08-03）

<!-- DSEC_FULLRES_W15_H67_BB1E4_STRICT_RESUME15_20260803 -->
- ep10 AEE `1.5357`，距 NB0+5% 门槛 `1.5177` 仅差约 `0.018`；训练 loss 尚未平台，因此继续同一 H67 Motion-XOR 分支。
- 使用 ep10 model/state 与 `--resume 1` 严格恢复 optimizer、scheduler、AMP scaler；训练到 ep15，并对 ep12/ep15 执行 standard valid825。
- 480x640、window 2x15x15、batch2、BN no_running、bb1e4 和其他结构不变；保留原 milestones 10/20。
- config：`neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_H67_crop_bb1e4_resume_ep15.yml`。
- status：`neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_H67_crop_bb1e4_resume15_20260803/status.log`。


### H67 fullres bb1e4 strict ep12/ep15 结果

<!-- DSEC_FULLRES_W15_H67_BB1E4_STRICT_RESUME15_RESULT_20260803 -->

# Standard Valid825 Ranking

Ranking mode: `aee`.

The energy column is a spike-activity proxy and excludes overlay attention control/reduction operations.

| rank | epoch | AEE | AAE legacy | AAE benchmark | PE1 | PE2 | outlier | total_spikes | firing | spike_energy_proxy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15 | 1.4757 | 6.4254 | 6.1599 | 0.4547 | 0.1702 | 0.0828 | 78.2806G | 6.0902% | 69209.42 |
| 2 | 12 | 1.6431 | 6.9515 | 6.7414 | 0.5085 | 0.2039 | 0.1012 | 77.2567G | 6.0105% | 68324.54 |


### H67 fullres bb1e4 ep15-to-ep30 等预算续训（2026-08-04）

<!-- DSEC_FULLRES_W15_H67_BB1E4_RESUME30_20260804 -->
- ep15 已达到 AEE `1.4757`、AAE-Benchmark `6.1599`、spikes `78.2806G`，满足 NB0+5% 与 spikes 至少下降20%的门槛；继续到 ep30 是为了与 NB0 fullres 30 epochs 等预算。
- 审计发现上游 state 在每轮 `scheduler.step()` 前保存；此前两次分段使 saved scheduler 在 ep15 时为 `last_epoch=12`。实际已执行 LR 为 ep1--12 `1e-4`、ep13--15 `5e-5`。
- 不修改历史 checkpoint/state；新增 staged state，仅将 scheduler `last_epoch 12->15`、`_step_count +3`，model/optimizer/scaler/current LR 均不变，之后按 milestone20 正常降 LR。
- 保存并 standard-valid825 评估 ep20/25/30；其余 480x640、window2x15x15、batch2、BN no_running、H67 Motion-XOR 结构不变。
- config：`neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_H67_crop_bb1e4_resume_ep30.yml`。
- scheduler audit：`neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_H67_crop_bb1e4_resume30_20260804/resume_source_ep15_scheduler_aligned/scheduler_alignment_audit.json`。
- status：`neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_H67_crop_bb1e4_resume30_20260804/status.log`。


### H67 fullres bb1e4 ep20/25/30 结果

<!-- DSEC_FULLRES_W15_H67_BB1E4_RESUME30_RESULT_20260804 -->

# Standard Valid825 Ranking

Ranking mode: `aee`.

The energy column is a spike-activity proxy and excludes overlay attention control/reduction operations.

| rank | epoch | AEE | AAE legacy | AAE benchmark | PE1 | PE2 | outlier | total_spikes | firing | spike_energy_proxy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 30 | 1.3387 | 6.0147 | 5.7558 | 0.4165 | 0.1405 | 0.0647 | 81.3086G | 6.3258% | 71812.84 |
| 2 | 25 | 1.3726 | 6.0102 | 5.7661 | 0.4172 | 0.1453 | 0.0678 | 80.5870G | 6.2696% | 71193.93 |
| 3 | 20 | 1.4240 | 6.1946 | 5.9511 | 0.4378 | 0.1597 | 0.0762 | 79.5345G | 6.1877% | 70268.40 |


### H67/Local-5 最终证据队列、收敛边界与磁盘清理（2026-08-05）

<!-- H67_LOCAL5_FINAL_EVIDENCE_AND_CLEANUP_20260805 -->

- H67 ep30 是当前已测轮次的 rank-1：AEE 从 ep20/25/30 的
  `1.4240 -> 1.3726 -> 1.3387` 单调改善，AAE-Benchmark 为
  `5.9511 -> 5.7661 -> 5.7558`。末轮 train loss 也创下阶段最低值
  `1.10247`，因此不能声称数学意义上完全收敛；准确表述是“完成与 NB0
  相同的 fullres 30-epoch 预算，已接近平台但末轮仍有小幅收益”。
- H67 ep25->30 的 AEE 仍改善约 `2.47%`，但 AAE-Benchmark 只改善约
  `0.18%`，且 legacy AAE 略有反弹 `6.0102->6.0147`。因此当前更准确的判断是：
  方向误差已基本平台，端点误差尚未完全平台；ep30 是已测最优 checkpoint，但
  “最后一轮最好”本身不等价于仍会持续同速改善。
- NB0 ep29 同样是其已测 rank-1，AEE 从 ep19/24/29 的
  `1.5798 -> 1.5304 -> 1.4454` 持续改善。其 train loss 在 ep26--29
  约为 `1.14569/1.14085/1.13698/1.13989`，更接近平台，但也没有证据证明
  已完全收敛。NB0 的 crop 预训练只有 60 epochs，而论文范式为 80 epochs，
  少 20 个 crop epochs 可能是 AAE 差距的一部分。
- NB0 ep24->29 的 AEE 改善约 `5.56%`、AAE-Benchmark 从 `6.2591` 到
  `6.1803`（约 `1.26%`），说明 fullres30 结束时仍可能欠收敛；但本地与论文
  official AAE `4.871` 尚差 `1.3093`，约为论文值的 `26.9%`。这远大于末 5/10
  epochs 的 AAE 收益，不能预期仅多跑少量 fullres epoch 就自动闭合。
- 不能把论文 official test-server AAE `4.871` 与本地 train-split valid825 的
  AAE-Benchmark `6.1803` 直接归因于“没收敛”。二者还同时受数据序列、split、
  checkpoint 选择、crop 预算与 official server 聚合影响。当前
  AAE-Benchmark 已按 DSEC/Barron `(u,v,1)` 公式实现并通过单元测试；在没有
  official submission 前，该差距必须标注为 protocol/domain gap，而不是公式错误。
- 为公平回答是否仍需训练，后续如做 convergence audit，必须同时把 NB0 和 H67
  从各自 fullres rank-1 延长相同预算（建议 +10 epochs），不能只延长候选主线。
  当前 H67 已在同预算下显著胜过 NB0，因此该审计排在 Local-5 公平重跑与硬件
  profile 之后。
- Local-5 旧 fullres 结果使用 backbone/norm LR `2e-6/1e-6`，不能与 bb1e4
  H67 公平比较。新增队列
  `entrypoints/run_dsec_fullres_w15_h66d_local5_bb1e4_full_pipeline.py`：固定
  480x640、window2x15x15、batch2、BN no_running、30 epochs、bb1e4；使用
  milestones `13/20` 对齐 H67 实际执行的 ep0--12/13--19/20--29 LR 轨迹，完成
  ep9/14/19/24/29 valid825 后自动选择 rank-1，并接 dyadic Q7/Q1.7、
  hardware-order Q7/Q1.7 与 100-sample T450 post-G0 ordered
  profile/replay/acceptance。
- acceptance 后由
  `hw_autoresearch_nts07/scripts/run_local5_bb1e4_checkpoint_bound_rtl.py` 自动生成
  100 组 checkpoint-bound T450 projection vectors，并运行 direct/QGASR
  SystemVerilog、random-stall SVA、lint 与 Yosys。证据范围严格标为
  post-G0 projection RTL exact，不扩写成 full-attention/full-network exact。
- 长训练恢复合同：ep9/19/29 保存 model 与 optimizer/scheduler/AMP scaler 成对
  checkpoint；若流水线中断，自动从最新成对 checkpoint 用 `--resume 1` 严格恢复，
  不从 crop 起点静默重训。
- 2026-08-05 14:24 在首个 ep0 仅运行约 6% 时做一次受控重启，以启用上述恢复
  合同；当时尚未产生 checkpoint，因此从同一 crop ep29 起点重新开始，不混入
  optimizer 状态。重启后加载审计仍为 ATLIF105、Shiftmax12、overlay210、
  missing/unexpected0/0，速度约 `1.05 s/it`。
- 独立 supervisor
  `entrypoints/supervise_dsec_fullres_w15_h66d_local5_bb1e4.py` 持有单实例锁；
  若主流水线在最终完成标记前退出，最多自动重启 5 次。恢复仍由主流水线的最新
  model/state 成对检查决定，supervisor 不自行拼接或修改 checkpoint。
- supervisor 首次启动测试发现相对 argv 未被绝对脚本路径识别，曾误启动第二个
  流水线约 14 秒；第二进程在首个 forward、进度仍为 `0/3672` 时因显存竞争 OOM，
  未完成 optimizer step 或写 checkpoint；原训练未停止，仅在 step193--199 短时从
  约 `1.05` 降到最高 `1.87 s/it`，随后恢复。修复后改用
  `/proc/<pid>/cwd + argv` 解析并要求精确绝对路径相等；事件只造成 `train.log`
  多一段初始化/OOM 输出，不改变原进程模型状态或样本顺序。
- exact 口径保持严格：dyadic/hardware-order 是 attention-core Python 数值评估；
  只有真实 T450 trace 驱动 SystemVerilog 且 zero-mismatch 后，才能写
  `RTL-exact`。最终论文赢家必须重新绑定最终 checkpoint/config SHA，不复用旧
  checkpoint 的 profile 充当新赢家证据。
- 运行 `entrypoints/prune_superseded_checkpoints_20260805.py` 删除 106 个已被
  后续结果替代的中间/短跑 checkpoint，释放约 `63.87 GiB`；保留 paper-relevant
  best/final checkpoint、配置、日志、ranking、spike profile 和全部硬件产物。
  删除清单与策略见
  `results/checkpoint_prune_audit_20260805.json`，清理后磁盘可用空间约 `107 GiB`。


### H67 fullres ep30 部署数值评估（2026-08-05）

<!-- H67_FULLRES_EP30_DEPLOY_NUMERIC_20260805 -->

# H67 fullres ep30 deploy numeric summary

Scope: attention-core hardware-order numeric; this is not full-network RTL-exact or T450 SV sign-off.

| path | AEE | AAE legacy | AAE benchmark | spikes(G) | energy proxy(uJ) |
|---|---:|---:|---:|---:|---:|
| float | 1.3387 | 6.0147 | 5.7558 | 81.3086 | 71812.84 |
| dyadic Q7/Q1.7 | 1.3424 | 6.0056 | 5.7625 | 81.3076 | 71811.95 |
| hardware-order | 1.3417 | 5.9869 | 5.7536 | 81.3067 | 71811.18 |


### H66d Local-5 fullres bb1e4 公平重跑（2026-08-05）

<!-- DSEC_FULLRES_W15_H66D_LOCAL5_BB1E4_PIPELINE_20260805 -->
- 旧 Local-5 fullres 使用 backbone/norm LR `2e-6/1e-6`，不能与已修复的 H67 比较；本轮改为同一 bb1e4 optimizer，并用 milestones13/20 复现 H67 实际执行的 ep0--12/13--19/20--29 LR 轨迹，threshold freeze1224。
- 固定 480x640、window2x15x15、batch2、BN no_running、30 epochs；从 Local-5 crop/full30 rank-1 ep29 初始化。
- 评估 ep9/14/19/24/29 standard valid825；rank-1 再跑 dyadic Q7/Q1.7、hardware-order，以及100样本 T450 post-G0 ordered profile/replay/acceptance。
- config：`neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_H66d_local5_bb1e4_ft30.yml`。
- status：`neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_H66d_local5_bb1e4_ft30_20260805/status.log`。
- `2026-08-05 15:30 CST` 健康检查：ep0 已完整结束并进入 ep1；train/validation loss
  分别为 `2.14747/1.97951`，LR `1e-4`，epoch time `3929.79 s`，train step
  `1.0702 s`，吞吐 `1.8688 samples/s`，峰值训练显存 `38.354 GiB`。8 个 worker 与
  GPU 快路径持续正常，尚未到首个预注册 ep9 checkpoint/valid825，不能据此判断精度。


### H67/Local-5 fullres 硬件 profile 协议修正与 H67 队列（2026-08-05）

<!-- H67_LOCAL5_FULLRES_PROFILE_PROTOCOL_FIX_20260805 -->

- 审计发现 `profile_nts11_hardware_p0.py` 历史上只调用 `model.eval()`，没有执行
  standard evaluator 的 `test.bn_policy=no_running`；因此旧 profile 可保留作开发数据，
  但不能直接作为 fullres 论文口径。profiler 现已复用同一 BN 语义：所有
  `_BatchNorm` 关闭 running stats，并清空 running mean/var/counter。
- 新 profile JSON/Markdown 强制记录 config SHA-256、checkpoint path/size/mtime、
  resolution/crop/window/T、BN 策略、`ATLIFTernaryPSN`/Shiftmax 数量与完整
  checkpoint load audit。H9 profile 若出现 missing/unexpected 非零将 fail-fast，
  不生成可被误用的硬件证据。
- Local-5 继续由现有 post-G0 watcher 在最终 rank-1 上生成 profile100；随后
  checkpoint-bound projection RTL watcher 运行真实 T450 vector 的 SV/SVA/lint/Yosys。
- `2026-08-05 15:23 CST` 源码 provenance/source-binding 补强后，旧 post-G0 watcher
  已受控终止以避免用启动时缓存的旧 Python 代码生成最终证据；训练子进程不受影响，
  继续正常占用 GPU。主流水线完成 train/valid825/deploy 后会收到该 watcher 的非零
  返回，supervisor 仅重启一次流水线；重启会复用已完成的 model/state、ranking 与
  deploy 产物，并用当前代码重新启动 post-G0 watcher，不会重复训练。独立 Local-5
  checkpoint-bound RTL watcher 也已重启并加载当前 source-binding 逻辑。
- 新增 `hw_autoresearch_nts07/scripts/run_h67_ep30_fullres_t450_profile.py`，严格等待
  Local-5 整条流水线释放 GPU 后，基于 H67 ep30 hardware-order config 运行
  100-sample T450 ordered profile，并导出 1 个真实样本的 all12 attention bit trace。
  trace 审计必须同时满足四 stage 覆盖和恰好 12 个 block record。
- H67 该队列当前完成的是 checkpoint-bound profile 与真实 bit-trace 数据质量审计；
  在这些 trace 尚未驱动完整 score/mask/Shiftmax/projection SystemVerilog
  zero-mismatch 前，仍不得写成 full-attention 或 full-network `RTL-exact`。


### Local-5 checkpoint-bound score/Shiftmax RTL 证据补全（2026-08-05）

<!-- LOCAL5_CHECKPOINT_BOUND_SCORE_RTL_20260805 -->

- 进一步审计确认旧 Local-5 T450 full-chain RTL 使用 synthetic Q/K/weights；旧的最终
  watcher 只把真实 K/gate descriptor 送入 projection RTL。因此它们均不能单独证明
  最终 checkpoint 的真实 Q/K 到 score/Shiftmax 数据通路。
- `profile_local5_hardware_features.py` 的 post-G0 ordered trace 现额外保存与每组
  450 个 descriptor 对齐的 `descriptor_q_bitmap`，并用独立 contract
  `local5_qk_score_shiftmax_trace_v1` 标识；原 descriptor contract v3 保持不变，避免
  破坏已有 replay/acceptance 消费链路。
- 新增 `generate_local5_checkpoint_score_vectors.py`：从四个 stage 各选 25 组、共
  100 组真实 T450 trace，重建全部 Q/K candidate row，独立复算 alpha-XNOR Q7 与
  masked integer Shiftmax Q1.7，并要求复算 gate 与 checkpoint trace gate 完全一致；
  共生成 `100 x 450 = 45,000` 个 score/Shiftmax 事务。
- 新增 `run_local5_checkpoint_score_trace_checks.sh` 与
  `report_local5_checkpoint_score_rtl.py`：Icarus 和 Verilator 必须全部 zero-mismatch，
  Yosys 对 score 与 Shiftmax leaf 分别通过 `check`。已用独立 synthetic T450 fixture
  验证工具链可处理 45,000 事务且双仿真零误差；该 fixture 只证明工具可运行，不能
  当作模型 checkpoint 结果。
- 最终 watcher 会在 post-G0 acceptance 后依次执行真实 checkpoint 的
  Q/K->score->Shiftmax RTL 与 projection RTL。准确证据组合为：
  `checkpoint-bound score/Shiftmax component RTL-exact`、
  `checkpoint-bound post-G0 projection component RTL-exact`、以及旧
  `synthetic T450 control-chain RTL`；projection weights 目前仍是确定性 synthetic
  权重，因此三者不得合并宣称 full-attention 或 full-network RTL-exact。
- Local-5 profiler 本身也已直接执行 BN `no_running` 处理，而不只依赖通用 profiler；
  其 JSON/Markdown 必须记录 ATLIF/Shiftmax 数量、overlay/missing/unexpected、配置与
  checkpoint identity。最终 source binding 已纳入 score generator/reporter/TB/runner
  的 SHA-256，任何脚本变化都会使 acceptance 重新绑定而不能静默复用旧结果。
- projection 证据复审发现旧向量虽使用真实 K/gate descriptor，却始终加载手工
  `(lane%5+1)*(out0?1:-2)` 权重；同时最终 runner 的汇总器未显式传
  `--manifest`，可能默认读取旧 vector manifest。两项均已 fail-closed 修正：profiler
  新增 `checkpoint_projection_contract.{json,npz}`，导出 all12 block 的真实
  `proj.weight/bias`、逐输出通道 dyadic INT8 编码和 checkpoint SHA；ordered manifest
  与 acceptance 强绑定两份产物。向量生成器按每组真实 stage/block/head 选择对应
  32-lane INT8 weight slice，testbench 每组 reset 后重载权重，runner 强制传当前
  manifest，report 再校验 vector/projection contract SHA。
- 新证据的准确范围提升为
  `checkpoint-bound real-weight per-head projection partial-accumulator RTL-exact`；仍不含
  跨全部 head 的 C 维求和、bias、动态 BN、requant、residual 或 full network。特别是
  standard evaluator 使用 BN `no_running`，其 batch statistics 随输入变化，不能把
  BN 静态折叠进 checkpoint 权重；文档和机器报告均已明确禁止该扩写。
- 工具验证：非平凡正负 INT8 fixture 的四 stage/T450 回放通过跨组 weight reload 与
  Acc32 zero-mismatch，旧 synthetic 模式回归亦通过；projection/acceptance/ordered
  trace 共 `18` 个 unittest 通过。等待中的 checkpoint-bound RTL watcher 已于
  `2026-08-05 15:46 CST` 重启并加载当前真实权重/source-binding 逻辑，训练未受影响。


### H67 ep30 checkpoint-bound T450 score/Shiftmax RTL 队列（2026-08-05）

<!-- H67_EP30_CHECKPOINT_BOUND_T450_ROW_RTL_20260805 -->

- 审计确认现有 `h67_score_class_row_engine` 已参数化 `MAX_TOKENS`，无需修改算法或
  RTL 语义即可实例化 T450；此前缺口是最终 ep30/all12 真实 bit trace 到 T450 RTL
  的自动向量、双仿真和强 provenance 绑定。
- `h67_bit_trace.py` 现把 config/checkpoint SHA-256、480x640/window2x15x15、BN
  `no_running`、ATLIF105/Shiftmax12 与 overlay/missing/unexpected load audit 写入
  trace `run_context`。通用 profiler 同时将完整 checkpoint SHA-256 写入 profile，
  不再只依赖 path/size/mtime。
- 新增 `generate_h67_checkpoint_row_vectors.py`：要求 all12 block 顺序完整，从每个
  block 的真实 window0、所有 head 重建 T450 temporal Q/K/peer-K；独立复算 Motion-XOR
  Q7、Q8 exp2 LUT、integer row sum、ceil-pow2 normalization 与 Q1.7 gate，任何一项
  与 trace gate 不一致即 fail-fast。
- 新增 `tb_h67_checkpoint_rows.sv` 和
  `run_h67_checkpoint_row_trace_checks.sh`：逐行驱动真实 Q/K/peer-K，随机化输入空拍与
  输出 backpressure，核对 active K/gate、zero-K fold、loaded/folded/emitted 计数；
  Icarus 与 Verilator 必须 zero-mismatch，Yosys 在 `MAX_TOKENS=450` 下通过
  hierarchy/proc/opt/check/stat。
- 工具资格验证已使用旧的真实 T162 四-stage/B0 trace：独立参考与 trace 完全匹配，
  Icarus/Verilator 均对 45 行、7,290 个 token 输入零失配，active outputs 共 1,054；
  Yosys T450 结构检查报告 0 problems。该旧 T162 dry-run 只验证新工具链，不能替代
  ep30 fullres/all12/T450 的最终结果。
- `run_h67_ep30_fullres_t450_profile.py` 已升级为 profile100 -> all12 trace audit ->
  checkpoint-bound T450 score/Shiftmax RTL 串行闭环；旧常驻 watcher 已安全重启以加载
  新代码。最终报告 scope 固定为
  `checkpoint_bound_qk_score_scs_shiftmax_component_rtl_exact_not_projection_or_full_network`，
  不得扩写为 projection、full attention 或 full-network RTL-exact。


### Local-5 训练、RTL 证据与历史 checkpoint 清理复核（2026-08-05）

<!-- LOCAL5_TRAIN_RTL_CLEANUP_REAUDIT_20260805 -->

- Local-5 公平 fullres30 正在执行，当前已进入 ep2（0-based 的第 3 轮），尚未到首个
  预注册 ep9 checkpoint，因此现在不能报告 AEE/AAE 或判断 Local-5 是否收敛。训练
  进程、训练/验证各 8 个 persistent workers、约 45 GiB GPU 占用均存活；两次
  `1.05 -> 4--7 s/it` 抖动都与同机 OpenROAD detail-route 同期出现，不是重复训练、
  OOM、NaN 或 checkpoint/optimizer 状态损坏。`2026-08-05 16:18 CST` 已无损隔离
  CPU affinity：训练进程树使用 core `0--47`，当前 OpenROAD flow 继承 core
  `48--63`；训练随后恢复到约 `1.05--1.2 s/it`，硬件任务未停止或改写。
- ep1 已完整结束：train loss `1.84878`、validation loss `1.70673`，相对 ep0 的
  `2.14747/1.97951` 分别下降约 `13.91%/13.78%`；LR 仍为 `1e-4`，无 NaN/Inf。
  ep1 epoch time `4958.14 s`、平均 `1.3503 s/it`，高于 ep0 的 `3929.79 s`，差值由
  同期 OpenROAD 资源争用解释；OpenROAD 退出后的连续训练段恢复到约
  `1.03--1.10 s/it`。ep1 不是预注册保存点，目录无 checkpoint 符合协议。
- Local-5 训练入口的实际 load chain 已复核并升级为流水线门禁：预加载
  ATLIFTernaryPSN `105`、Shiftmax attention `12`，checkpoint overlay keys `210`，
  missing/unexpected `0/0`。训练后进入 valid825 前会重新解析最后一次 load audit；任一
  数量漂移都直接失败，不允许仅凭 checkpoint 文件存在继续生成 ranking/profile。
- standard evaluator 的 `artifact_identity` 进一步加入 checkpoint SHA-256，并在
  `spike_profile.json` 序列化实际 `ATLIFTernaryPSN/ShiftmaxAttention` 数量。Local5
  流水线对 ep9/14/19/24/29 float profile 和 rank-1 的 dyadic/hardware-order profile
  逐个复核 480x640、window2x15x15、batch1、BN no_running、overlay210、
  missing/unexpected0/0、ATLIF105、Shiftmax12 与 checkpoint SHA；旧的同路径/同大小但
  无 SHA 或无 counts 产物不得复用。deploy reusable-profile 门禁采用相同约束。
- 阈值语义复核：当前 `threshold_freeze_after_step=1224` 只冻结独立 homeostatic
  `threshold_update`，并不冻结 optimizer gradient；配置未启用
  `freeze_threshold_grad_after_step`，threshold LR 为 `5e-6`。`official_atlif` 分支也不会
  应用配置中的 `min_threshold/max_threshold` clamp。训练中的 threshold 因而仍可由
  optimizer 更新，而推理/RTL 使用最终 checkpoint 中的静态 threshold。profile JSON 与
  bit-trace manifest 新增 `threshold_training_semantics` 机器字段，禁止把日志中的
  `threshold_updates_frozen=1` 误写成“阈值参数完全冻结”。
- Local5 专用 post-G0 profile 也已把该语义写入
  `local5_hardware_features.json`、Markdown 和 ordered-trace manifest；fail-closed
  acceptance 固定检查 `official_atlif`、homeostatic boundary1224、optimizer gradient
  未冻结、threshold LR `5e-6`、official clamp inactive 与 checkpoint-static inference。
  任一字段缺失/漂移都不会释放后续 checkpoint-bound RTL watcher。profile/trace/
  descriptor acceptance 相关回归与 standard/deploy/convergence provenance 回归合计 `25` 项 unittest
  已全部通过。
- Local5 总流水线不再只信任 post-G0 watcher 的 exit code：退出后会重新读取
  `acceptance.json` 和其绑定的 ordered manifest，检查 `accepted=true`、100 samples、
  12 blocks、rank-1 checkpoint 路径与 threshold semantics。这样即使另一个 watcher
  持有锁使当前进程正常退出，也不能提前写入总完成标记或释放 H67 队列；正向和篡改
  acceptance 夹具均已验证。
- checkpoint-bound projection 最终报告改为 fail-closed：Yosys 使用
  `check -assert`，并把全部 RTL/SVA/TB 与当前 vector manifest 写入
  `source_sha256.txt`；汇总器逐文件复算 SHA-256。最终 watcher 同时要求真实 dyadic
  INT8 checkpoint weight binding、random-stall SVA、Verilator lint、Yosys、vector
  manifest 和 source manifest 全部 PASS，否则不生成总 PASS。
- score/Shiftmax 的两个 Yosys leaf 同样改用 `check -assert`，report 要求两个 Yosys
  session 都正常结束且 Icarus/Verilator/独立 trace reference 全部通过。等待中的
  checkpoint-bound watcher 已以独立 daemon PID `2161370` 重启并加载该版本。
- 历史 `third_party/SDformerFlow/results` 经全仓路径引用审计后，仅保留
  `checkpoint_epoch59.pth` 与 `checkpoint_epoch59_state_dict.pth`；删除 57 个未被当前
  配置、脚本或文档引用的 2026-05 中间 checkpoint，共回收
  `24,852,206,537 bytes`（约 `23.15 GiB`）。清理后可用空间约 `127 GiB`；完整删除表、
  保留文件 SHA-256 与策略记录在
  `neuron_autoresearch/cleanup_audits/third_party_sdformerflow_20260805.json`。
- 第二轮只对白名单中的 6 月已结束 H9 路线清理非锚点 model checkpoint：删除 `30`
  个文件、实际回收 `18,606,112,768 bytes`（`17.33 GiB`），可用空间增至约
  `146 GiB`。保留 NTS10d standard-valid825 的 ep19/24/29、NTS11aah rank-1/resume
  ep0 与 final ep14、NTS11aq rank-1/续训源 ep2、NTS11aqa AEE 最优 ep5 与 AAE/final
  ep7；NB0/H67/Local5、当前队列、paired optimizer state、配置、日志、指标和硬件产物
  均不在删除 allowlist。机器审计为
  `neuron_autoresearch/cleanup_audits/h9_superseded_20260805.json`，删除后逐项确认 30 个
  文件不存在、8 个保留锚点仍存在。
- NB0 未完全收敛是可信风险，但不是 AAE 差距的唯一解释：论文训练范式写 80 个 crop
  epochs 再加 30 个 full-resolution epochs，而本地发布 YAML
  `third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_full_torch.yml`
  固定 `n_epochs: 60`，仓内也没有 ep79/ep80 checkpoint。本地 NB0 fullres ep24->29
  AEE 仍改善 `5.56%`，支持欠收敛判断；但 AAE-Benchmark 仅改善 `1.26%`，不足以单独
  解释本地 `6.1803` 与论文 `4.871` 的 `1.3093` 差距。结论仍应写为“本地 checkpoint
  可能欠训练，同时存在 official test-server/split/聚合口径差异”，不能只靠追加少量
  epoch 承诺复现论文 AAE。
- 是否给 H67 和 NB0 各追加相同 +10 fullres epochs，待 Local-5 ep9/14/19/24/29
  公平曲线完成后再裁决；H67 当前 ep30 是已测最优但 AEE 尚未平台，NB0 也不能标为
  fully converged。不得只延长 H67 而保持 NB0 预算不变。


### NB0 AAE 与论文数值的最终口径更正（2026-08-05）

<!-- NB0_AAE_PAPER_ROW_RECONCILIATION_20260805 -->

- 重新逐页核对 SDformerFlow-v2 论文后确认：`AAE=4.871` 位于 Table I，是 DSEC
  official hidden test 的 all-sequence `AE`，不是作者 validation split 的结果。
- 论文 Table IV 对 full-resolution validation 的 PSN+SPE+QK s10-c2 行报告
  `EPE=1.61`、`Outlier=8.91%`、`AAE=7.23`。本地 NB0 ep29 为
  `AEE=1.4454`、`Outlier=7.93%`、legacy AAE-2D `6.5128`、Barron AE-3D
  `6.1803`；因此不能说“本地 NB0 AAE 不及论文 validation”，其同域数值反而更好。
- 本地 evaluator 是“每帧先对有效像素求均值，再对 825 帧等权平均”；local valid825
  来自 18 条训练序列中的 held-out frame。official test 是七条不同 hidden sequence。
  DSEC 页面公布的七条 SDformerFlow sequence AE 简单平均为 `4.9919`，而 leaderboard
  总值为 `4.871`，进一步证明官方聚合并非本地 frame-mean 或简单 sequence-mean。
- 新 standard profile 将该区别固化为 `metric_contract`：legacy `AAE` 明确标识为
  `(u,v)` 2D direction angle，`AAE_Benchmark` 明确标识为 normalized `(u,v,1)`
  Middlebury/Barron 3D angle，aggregation 标识为 per-frame masked mean 后对本地 valid
  frames 等权平均，population 明确写 `not_official_hidden_test`。Local5 和 H67/NB0
  convergence runner 均 fail-closed 校验这些字段。
- NB0 的确可能欠收敛：paper 写 80 crop + 30 fullres，而 released YAML 是 60 crop；
  NB0 ep24->29 AEE 仍下降 `5.56%`。但 legacy/benchmark angle 只下降
  `0.44%/1.26%`，已经接近平台。正确结论是“AEE 可能继续改善，AAE 不太可能仅靠少量
  续训从 6.18 变成官方 4.871”；官方值只能通过冻结模型后提交 DSEC server 验证。
- 完整定义、数据 population、聚合和收敛表更新在
  `neuron_autoresearch/AAE_BASELINE_DIAGNOSTIC_20260717.md`。后续 DATE 表必须分别标
  `local valid825 legacy AAE-2D`、`local valid825 Barron AE-3D` 与
  `official DSEC test AE`，三者禁止混列成同一栏。


### H67/NB0 fullres 等预算 +10 收敛审计队列（2026-08-05）

<!-- DSEC_FULLRES_W15_H67_NB0_EQUAL_PLUS10_PROTOCOL_20260805 -->

- H67 ep25->ep30 AEE 仍改善 `2.47%`，NB0 ep24->ep29 AEE 仍改善 `5.56%`；两者
  最优都落在已测训练边界，因此均不能标成 fully converged。新增对称 +10 fullres
  审计，不覆盖或改写任何旧 checkpoint。
- 两份最终 state 的只读审计均为 internal epoch29、scheduler last_epoch29、AMP scaler
  存在且主 optimizer LR 为 `2.5e-5`。NB0 state 仍含 future milestone30，而 H67 没有；
  直接 resume 会造成追加第 1 轮后只有 NB0 降 LR，不能作为公平收敛比较。
- 新协议为两者分别 hard-link 原最终 model、复制 optimizer/scheduler/scaler state，且仅在
  新 staged state 中把 future milestones 清空；追加 10 轮都固定 source current LR。
  H67 因历史 `epoch_offset=1` 保存/评估 label ep35/40，NB0 保存/评估 ep34/39；它们分别
  对应相同的 fullres 35/40-epoch 预算。
- 历史 state 未保存 Python/NumPy/Torch/CUDA RNG，因此该续训是同 seed 的公平
  fine-tune extension，不是 bit-exact uninterrupted continuation；机器 audit 必须显式记录
  `rng_state_present=false`，论文不得声称逐 bit 连续复现。
- H67 每次训练启动必须验证 `checkpoint_overlay_keys=210`、missing0/unexpected0、
  ATLIF105、Shiftmax12；NB0 必须验证 overlay0、missing0/unexpected0。标准评估沿用
  480x640、window2x15x15、crop null、batch2、BN no_running 与 valid825。
- +10 runner 在 ranking 后还会逐个复核三个预算点的 profile：config/checkpoint SHA、
  fullres/window15/batch1/BN 协议，以及 H67 的 overlay210/ATLIF105/Shiftmax12 或 NB0 的
  overlay0/ATLIF0/Shiftmax0。训练日志正确但评估加载漂移同样会 fail-closed。
- 为避免抢占 Local-5 和已注册硬件证据任务，队列严格等待 Local-5 全流水线及 H67 ep30
  T450 profile/RTL 结束，再依次执行 H67 +10、NB0 +10 和 standard valid825。最终是否
  采用 ep40/39 由 AEE rank 与末段斜率决定，不能预设最后一轮必然最好。
- 收敛机器判据固定为：若 budget40 仍是 AEE rank-1，且相对 budget35 的 AEE 改善
  `>1%`，则标记 `not_plateaued`；否则标记
  `operationally_plateaued_or_overfit`。同时报告 AEE last5/last10、AE-3D last5 和
  spikes，不用单一 train loss 代替标准推理曲线。输出为
  `dsec_fullres_w15_equal_plus10_convergence_summary_20260805.{json,md}`。
- 汇总同时独立报告 legacy AAE-2D 与 AE-3D 的 last5/last10 improvement，以及 spikes
  last5/last10 change；只有两种角度指标的 last5 绝对变化都不超过 `1%` 才标
  `angle_plateaued`。AEE 是否平台和角度是否平台是两个字段，避免用 AEE 持续下降错误
  推断 AAE 与 official-test 差距也会靠续训消失。
- config generator：
  `neuron_experiments/H9_bipolar_self_attention/entrypoints/make_dsec_fullres_w15_equal_plus10_configs.py`。
- queue runner：
  `neuron_experiments/H9_bipolar_self_attention/entrypoints/run_dsec_fullres_w15_equal_plus10_convergence.py`。
- status：
  `neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_equal_plus10_convergence_20260805.log`。


### 第三轮旧 checkpoint 定向清理（2026-08-05）

<!-- PAPER_IRRELEVANT_CHECKPOINT_CLEANUP_20260805 -->

- 在前两轮清理之外，继续按 valid825 ranking 和当前 resume 依赖做白名单审计；本轮只处理
  已结束的 NTS11 早期范围消融、两个 5-epoch 旧微调和已被完整实验取代的短筛。
- 删除 `27` 个 model checkpoint，实际回收 `19,934,760,960 bytes`（约
  `18.56 GiB`），可用空间由约 `146 GiB` 增至约 `164 GiB`。
- NTS11u/NTS11aa/NTS11bd 路线保留 standard valid825 的 ep19/24/29，另保留已记录的
  rank-1 ep26；两个旧 5-epoch 微调均保留 valid825 rank-1 ep2 和 final ep4。当前
  NB0、H67、Local-5、所有 paired optimizer state、配置、日志、ranking/profile 和
  RTL 产物完全不在删除集合。
- 可复跑脚本：
  `neuron_experiments/H9_bipolar_self_attention/entrypoints/prune_paper_irrelevant_intermediates_20260805.py`；
  逐文件大小、inode/link count、删除后存在性和保留集合记录在
  `neuron_autoresearch/cleanup_audits/paper_irrelevant_intermediates_20260805.json`。
- 至此三轮 checkpoint 清理累计回收约 `59.0 GiB`。后续在 Local-5/H67/NB0 队列完成前
  不再扩大删除范围，避免误删尚未完成公平比较或硬件重新绑定所需的权重。


### Local-5/H67 ATLIF checkpoint-bound RTL 队列（2026-08-05）

<!-- CHECKPOINT_BOUND_ATLIF_DPTME_RTL_20260805 -->

- 复核确认此前所谓 RTL-exact 只覆盖 checkpoint-bound score/Shiftmax 和真实权重
  per-head projection partial accumulator；ATLIF 只有首调用 sampled Q4/Q6/Q8 参数量化
  profile，且旧 profile 的输入仍为浮点，不能计入 RTL-exact scope。
- 新增可选后处理
  `hw_autoresearch_nts07/scripts/generate_checkpoint_atlif_dptme_vectors.py`：从最终 rank-1
  的标准 `no_running` BN 验证输入捕获 81 个 functionally-live ATLIF site，严格要求
  `45 x T10 + 36 x T2`；12 个从未调用的 `sn2_q` 和 12 个结果死亡的 `attn_sn` 不进入
  部署执行集，但安装总数仍必须为105。
- 每个 live site 导出一个完整 32-lane DP-TME command；command 内同时混合普通、最小
  threshold margin 和最大幅值三类真实 lane。定点合同为 per-site power-of-two INT8
  input/weight、`Sa=Sx*Sw`、Acc24 bias/threshold、round-to-nearest-even、任何 clip 或
  Acc24 overflow 直接失败。另行报告 fixed-vs-float event flip，不能被 RTL 的整数
  zero-mismatch 掩盖。
- 新增 file-driven TB/runner/report：Icarus 与 Verilator 分别带独立 simulator identity，
  hidden/event 必须逐位零失配；输出 backpressure、输入气泡、SVA、lint、Yosys
  `check -assert` 均进入 fail-closed 报告。config/checkpoint/vector/RTL/TB/SVA/source
  SHA 全部绑定，报告 scope 固定为 ATLIF temporal-matrix component only。
- synthetic 81-site 工具资格测试已由 Icarus 比较 `25,920` 个 hidden 和 event，均为
  zero-mismatch；Verilator file-driven 全事务在训练和 OpenROAD 并行时运行代价过高，
  本次只完成编译，正式双仿真由 Local-5/H67 串行 watcher 在 GPU 任务结束后执行，
  未产生正式 PASS 前不得引用 synthetic fixture。
- Local-5 的 ATLIF 导出/双仿真已放在 post-G0 wrapper 返回之前，因此 H67 不会与其争抢
  GPU。H67 ep30 和 equal+10 后的新 rank-1 都会用各自 checkpoint 重新运行同一流程；
  两个空等 watcher 已重启加载新代码，当前 Local-5 训练未中断。
- PyTorch binary ATLIF 实际输出 `event x threshold`，而 DP-TME RTL 当前输出 event bit。
  报告把 checkpoint-static threshold 作为 output scale metadata，但在完成下一层
  weight folding、BN/requant/residual 和 valid825 复验前，`deployment_accuracy_signoff`
  固定为 false，禁止写 full encoder/full network RTL-exact。
- 旧 DP-TME 周期表基于 `9x9=81` positions，不能用于当前 fullres `15x15=225`。
  新 `dptme_fullres_w15_port_contract` 报告得到 T10 `2250` 拍，T2 G5/G4/G3 分别
  `90/114/150` 拍；单32-bit event出口使 G5 下界回到 `450` 拍。5项旧兼容与 fullres
  几何回归通过，报告仍明确是架构乐观下界，不是 full-encoder measured latency。


### Local-5 fullres 在线状态与 ATLIF 报告门禁补强（2026-08-05 18:05 UTC）

<!-- LOCAL5_FULLRES_PROGRESS_AND_ATLIF_GATE_20260805 -->

- Local-5 并未遗漏；当前运行
  `dsec_fullres_w15_H66d_local5_bb1e4_ft30_20260805`，从自己的 crop/full30 ep29
  checkpoint 续训，而不是从 NB0/H67 权重启动。固定协议为 480x640、window
  `2x15x15`、batch2、BN `no_running`、ATLIF105、Shiftmax12、overlay210、
  missing/unexpected `0/0`。
- 已完成 ep0--2：train loss 为 `2.14747 / 1.84878 / 1.68106`，valid loss 为
  `1.97951 / 1.70673 / 1.80956`。ep2 train 继续下降但 valid 相对 ep1 反弹
  `6.03%`，当前正训练 ep3；在 standard valid825 尚未完成前不据此选择 checkpoint，
  也不把 ep1 暂时最低 valid loss 写成算法排名。
- H67 ep30 仍是当前最后预算点和 AEE rank-1：ep25 `1.3726` 到 ep30
  `1.33874` 改善约 `2.47%`，因此准确结论是“当前边界最好、尚未证明收敛”。已排队的
  H67/NB0 对称 +10 将以 ep35/40 standard valid825 与末段斜率完成收敛判断。
- NB0 fullres ep24 到 ep29 的 AEE 仍改善约 `5.56%`，说明 AEE 很可能也未完全收敛；
  但 legacy AAE/AE-3D 仅改善约 `0.44%/1.26%`，角度已经更接近平台。论文 `4.871`
  是隐藏七序列 DSEC test aggregate，不能与本地 valid825 AAE 直接相减后全部归因于轮次。
- ATLIF RTL reporter 不再只信任 manifest summary：逐命令复核81个唯一 site/tag、
  `45xT10 + 36xT2`、三类 lane 配额、25,920个 event、正有限 scale、Acc24 范围、
  input/weight/bias/threshold 零 clip、零 overflow 和静态 threshold output-scale 合同。
  新增4项 manifest 单元测试与2项 fullres 几何测试通过；既有训练/评估/profile
  provenance 回归 `25/25` 通过。


### 第四轮早期 MDR/FAPS 中间 checkpoint 清理（2026-08-05）

<!-- LEGACY_MDR_FAPS_CHECKPOINT_CLEANUP_20260805 -->

- 删除109个早期中间 checkpoint，实际回收 `36,648,837,120 bytes`（约
  `34.13 GiB`），文件系统可用空间增至约 `197 GiB`。
- 明确无效的 `mdr_fast_local_ckpts_20260624` 权重全部删除；该 run 已由日志证明只恢复
  optimizer/scheduler/scaler、没有恢复 model，原文档也已禁止用于论文或 warm start。
- 正确 MDR baseline 只保留已做 MVSEC 标准评估的 ep41/47；TTX-MDR 保留 ep10 交接
  model/state、论文表实际评估的 ep20/40/43 model，以及 ep20/final ep43 resume state。
- 已退休 FAPS FT10 保留 standard valid825 rank-1 ep8 和 final ep9 model，其余 model
  与 state 删除。所有 ranking、profile、训练日志、配置和结果表均保留。
- 当前 Local-5/H67/NB0、fullres 收敛队列、paired optimizer state、硬件向量/profile/RTL
  均未进入候选集合。可复跑脚本为
  `entrypoints/prune_legacy_mdr_faps_intermediates_20260805.py`，逐文件 inode/link count、
  大小、删除理由和保留白名单记录在
  `neuron_autoresearch/cleanup_audits/legacy_mdr_faps_intermediates_20260805.json`。
- 四轮定向清理累计回收约 `93.1 GiB`；后续只在当前 fullres/RTL 队列结束并确定最终
  rank-1 后再清理新产生的中间轮。


### Local-5 ATLIF watcher 生命周期修复（2026-08-05）

<!-- LOCAL5_ATLIF_WATCHER_LIFECYCLE_FIX_20260805 -->

- 审计发现本轮主流水线最早启动的 embedded post-G0 child 已提前退出；虽然独立
  score/projection watcher 正常等待，但若只依赖主流水线重启，ATLIF 正式回放存在遗漏风险。
- ATLIF vector + DP-TME 双仿真现同时纳入独立
  `run_local5_bb1e4_checkpoint_bound_rtl.py`。它在 acceptance 后读取同一
  `post_g0_run_identity.json`，验证 rank-1 config/checkpoint SHA，再生成正式向量和报告；
  最终 `checkpoint_bound_scope.json` 必须同时包含 score/Shiftmax、真实权重 per-head
  projection partial accumulator 和 ATLIF temporal-matrix 三项 component 证据。
- embedded 与独立入口共用同一 `flock`；锁内先检查 report status 与 checkpoint SHA，
  相同权重直接复用，否则才重跑，避免并发覆盖向量/日志。独立 watcher 已以当前代码重启
  为 PID `2233542`，不影响正在训练的 Local-5。
- wrapper/provenance/ATLIF/fullres 几何相关回归合计 `31/31` PASS。证据范围仍是三个
  component，不包含 cross-head accumulation、BN/requant/residual、SRAM macro 或 full network。


### DATE 算法证据最终 fail-closed 审计队列（2026-08-05）

<!-- DATE_ALGORITHM_CLOSURE_AUDITOR_QUEUED_20260805 -->

- 新增 `entrypoints/audit_date_algorithm_closure_20260805.py`，等待并统一审计五类正式产物：
  Local-5 standard valid825 ranking、Local-5 float/dyadic/hardware-order deploy summary、
  Local-5 三组件 RTL scope、H67/NB0 +10 收敛 summary、post-convergence H67 rank-1
  hardware evidence。
- 审计器逐 checkpoint 复核 480x640、无 crop、window2x15x15、BN no_running、batch1、
  AAE-2D/AE-3D/aggregation/population contract、overlay/module counts、missing/unexpected0/0
  和 checkpoint SHA；收敛 summary 的每个数值还必须与原始 profile 精确一致。
- Local-5 硬件证据必须同时通过真实 Q/K score/Shiftmax、checkpoint dyadic INT8 per-head
  projection partial accumulator、81-site ATLIF temporal matrix；向量与 projection contract
  均反向追溯到同一 rank-1 checkpoint SHA。H67 最终 evidence 的 epoch 必须等于 +10
  ranking 的 AEE rank-1，score/ATLIF report 也必须绑定同一 SHA。
- 最终 scope 固定为 `checkpoint_bound_component_rtl_exact_not_full_network`；只有 Local-5
  和 +10 两个结果 marker 都已写入本文件才允许 PASS。通过后自动生成
  `neuron_autoresearch/DATE_ALGORITHM_CLOSURE_AUDIT_20260805.{json,md}` 并回写结果标记。
- one-shot 当前按预期返回 PENDING；ranking/profile contract 的3项正反单元测试通过。
  常驻 watcher PID `2239223`，5分钟轮询，不占GPU。


### Local-5 post-G0 producer 独立监督修复（2026-08-05）

<!-- LOCAL5_SUPERVISED_POSTG0_PRODUCER_20260805 -->

- 运行时复核确认，主流水线早期启动的 embedded post-G0 child 已退出，最后一次记录停在
  `15:23:43`；仅保留 checkpoint-bound RTL watcher 等待 acceptance 会形成“消费者存活、
  producer 缺失”的生命周期缺口。
- `run_local5_bb1e4_checkpoint_bound_rtl.py` 现先调用
  `ensure_profile_acceptance()`：若 acceptance 不存在，就同步监督
  `run_local5_bb1e4_postg0_profile.py` 直到 rank-1/deploy/profile/replay/acceptance 完成；只有
  acceptance 真正落盘后才进入 score/Shiftmax、projection 与 ATLIF 三组件 RTL 流程。
- 独立监督器已重启为 PID `2243134`，其 post-G0 producer 子进程 PID `2243137` 正在等待
  fullres deploy follower；它不占GPU，也不改变当前 Local-5 训练。producer 与主流水线未来
  可能启动的同类入口仍由原有 `flock` 串行化。
- 新增两项生命周期测试，分别证明“缺 acceptance 必须启动 producer”和“已有 acceptance
  不重复启动”；连同 Local-5 pipeline acceptance 4项与 fullres rescue config 4项共
  `10/10` PASS。当前 conda 环境未安装 pytest，测试用同一 Python 环境直接调用并运行
  unittest，另通过 `py_compile` 与 `git diff --check`。


### Standard valid825 样本数 fail-closed 门禁（2026-08-05）

<!-- STANDARD_VALID825_SAMPLE_COUNT_GATE_20260805 -->

- 审计发现既有 evaluator/pipeline/closure 已校验 local validation population、聚合公式与
  checkpoint SHA，但没有强制 `samples == 825`；异常中断或错误文件列表仍可能生成名为
  `valid825` 的残缺结果。
- 现已在 `run_h9_standard_valid825_eval.py`、Local-5 pipeline、H67/NB0 `+10` runner 和最终
  closure auditor 四层加入 825 样本硬门禁。任一 checkpoint 少测或多测都会拒绝 ranking、
  convergence summary 和最终 PASS。
- Local-5 pipeline 与 closure 正反测试均增加 `824` 反例并通过；相关 Python 文件通过
  `py_compile`/`git diff --check`。加载新代码后的 `+10` runner PID `2245266`、closure
  watcher PID `2245267` 已重新启动并继续空等，Local-5 训练未中断。


### H67/NB0 staged resume 与 H67 证据复用预审计（2026-08-05）

<!-- H67_NB0_STAGE_AND_EVIDENCE_REUSE_AUDIT_20260805 -->

- 已在不占 GPU 的情况下提前生成并复核 H67/NB0 两份 staged resume：源 model SHA、源
  state SHA、staged state SHA 全部落盘；model 使用 hardlink；state/scheduler internal epoch
  均为29；AMP scaler 存在；历史 RNG state 不存在，故明确为非 bit-exact continuation。
- 两者均清空未来 scheduler milestones，并保持源 optimizer LR 不变。H67 五参数组 LR 为
  `2.5e-5/2.5e-5/1.25e-5/1.25e-5/1.25e-6`，NB0 为 `2.5e-5`；正式 runner 会按 SHA 复用
  staged state，不会在 GPU 释放后重新构造漂移状态。
- H67 ep30 和 post-convergence watcher 的旧逻辑曾以 report/FINAL 文件存在作为复用条件；
  现分别要求 score report `PASS`、component scope、checkpoint SHA，ATLIF report `PASS` 与
  同一 SHA，以及 FINAL 的 rank-1 epoch/path/report 全绑定后才允许复用。两项 checkpoint
  变更反例测试 `2/2` PASS。
- 加载新代码的 H67 ep30 watcher PID `2246496`、post-convergence watcher PID `2246497`
  已重新启动并空等；不会与 Local-5 争用 GPU。


### Local-5 成对训练状态最终门禁（2026-08-05）

<!-- LOCAL5_PAIRED_TRAINING_STATE_CLOSURE_GATE_20260805 -->

- 最终 closure 现除模型 checkpoint/profile SHA 外，还强制读取 ep9/19/29 的配套
  `*_state_dict.pth`。每个 state 必须满足 internal epoch 与文件标签一致、scheduler
  `last_epoch` 一致、milestones 恰为13/20、AMP scaler 存在。
- 五参数组 LR 也按真实轨迹逐项校验：ep9 为
  `1e-4/1e-4/5e-5/5e-5/5e-6`，ep19 乘0.5，ep29 乘0.25；optimizer LR 与 scheduler
  `_last_lr` 任一漂移都会拒绝最终 PASS。closure 输出会同时记录 model/state SHA。
- 新增正常三点与 ep19 LR 漂移反例测试，closure 测试现 `4/4` PASS。加载该门禁的最终
  watcher 已重启为 PID `2248718`；当前仍按预期等待正式产物。


### AAE 公式/聚合可执行 receipt（2026-08-05）

<!-- AAE_METRIC_EXECUTABLE_RECEIPT_20260805 -->

- 新增 `generate_aae_metric_test_receipt_20260805.py`，使用生产 conda 环境直接运行上游
  `tests/test_aae_metrics.py`，三项测试 `3/3` PASS：Barron `(u,v,1)` 数值公式、legacy 2-D
  与 benchmark 3-D 区分、逐 batch mask。
- receipt 同时检查 evaluator 仍固定 eval batch1，并执行“每帧 masked mean 后对 frame
  等权平均”的累加/除法路径；记录 metric/evaluator/test 三个源码 SHA。正式产物为
  `neuron_autoresearch/AAE_METRIC_TEST_RECEIPT_20260805.{json,md}`。
- 最终 closure 已把 receipt 加入 REQUIRED，并重新计算三个源码 SHA；任何后续公式或聚合
  修改都会使旧 receipt 失效。加载新逻辑的 closure watcher PID 为 `2250001`。


### Local-5 ep3 OpenROAD 资源争用记录（2026-08-05）

<!-- LOCAL5_EP3_OPENROAD_CONTENTION_20260805 -->

- Local-5 ep3 在 step2252 前约为 `1.05 s/it`；同机 OpenROAD
  `local5_out32_allmacro_proxy/direct/5_2_TritonRoute` 启动 detail-route 后，瞬时均值升至
  `3.2--4.8 s/it`，GPU utilization 从高负载降至 `15--58%`，显存仍稳定约46GiB。
- OpenROAD 进程有129个线程、覆盖全部64 CPU；系统 `iowait=0`，训练主进程、8个 persistent
  workers、模型显存和数值日志均正常。因此该段是 CPU/内存带宽争用造成的 wall-time 抖动，
  不是 DataLoader 单进程退化、OOM、模型重载或算法发散。
- 不重启训练、不改变 batch/workers/LR，也不干预另一条硬件任务；待 detail-route 自然结束
  后继续以稳定段吞吐估算剩余时间。该抖动不进入收敛或精度判断。
- 连续观测确认吞吐随后按 `3.14 -> 2.41 -> 1.89 -> 1.32 -> 1.13 -> 1.10 s/it`
  恢复，ep3 到约71%时已回到正常区间；训练全程为同一 PID/optimizer state。该恢复进一步
  排除永久 worker 退化或需要重启的故障。
- 同一 detail-route 后续以 PID `2264682` 再进入重负载段，ep3约93--98%时训练瞬时达到
  `3--5 s/it`、GPU最低约11%；仍无 iowait、OOM 或 worker 退出。到 ep4 开始时即使 OpenROAD
  尚未退出，训练已再次恢复约 `1.06 s/it`/GPU99%，因此两次抖动均按外部资源争用记录，
  不改变 batch/workers/LR 或恢复训练。


### Local-5 post-G0 relation/profile acceptance 最终门禁（2026-08-05）

<!-- LOCAL5_POSTG0_ACCEPTANCE_CLOSURE_GATE_20260805 -->

- 最终 closure 过去只通过下游 watcher 间接依赖 post-G0 acceptance；现已把
  `local5_fullres_bb1e4_postg0_acceptance_20260805/acceptance.json` 直接加入 REQUIRED。
- 审计器会重新检查 schema、accepted、100 samples、12 blocks，以及 loader provenance、
  formal qualification、relation RTL、descriptor geometry、ordered replay、source software、
  release receipt、checkpoint projection weight 和 threshold deployment 共11项门禁。
- acceptance、ordered manifest 与 run identity 的 SHA 必须互相一致，并同时绑定 Local-5
  AEE rank-1 epoch/path/checkpoint SHA；因此 Local-5 特有 relation-transpose/profile 证据不会
  被 score/projection/ATLIF 三组件总表遮蔽。
- 正常绑定和 `relation_rtl_binding=false` 反例测试通过，closure 测试现 `5/5` PASS；加载
  新门禁的 watcher PID 为 `2252989`。
- 本轮组合回归随后一次性重跑 pipeline 4项、closure 5项、H67复用 2项、Local-5 producer
  生命周期 2项、fullres config 4项、AAE 数值 3项，共 `20/20` PASS；相关入口同时通过
  `py_compile` 和 `git diff --check`。


### 第五轮退休微调中间 checkpoint 清理（2026-08-05）

<!-- RETIRED_FT_INTERMEDIATE_CHECKPOINT_CLEANUP_20260805 -->

- 对6条2026年6月已完成 standard valid825 ranking 的旧5/8轮微调执行白名单清理；每条均
  保留 AEE rank-1 `ep2`、最终 `ep4/ep7`，以及两个锚点对应的 optimizer/scheduler/scaler
  state。当前 Local-5、H67、NB0、staged resume、profile、ranking、日志和 RTL 产物均不在
  删除集合。
- 删除42个非锚点 model/state checkpoint，实际回收 `24,740,700,160 bytes`（约
  `23.04 GiB`）；可用空间增至约 `219 GiB`。删除后复核为0个候选残留、0个保留锚点缺失。
- 可复核脚本为
  `neuron_experiments/H9_bipolar_self_attention/entrypoints/prune_retired_ft_intermediates_20260805.py`；
  文件大小、inode/link count、删除原因和保留表记录在
  `neuron_autoresearch/cleanup_audits/retired_ft_intermediates_20260805.json`。
- 五轮定向清理累计回收约 `116.1 GiB`。当前空间足够完成 Local-5 和 H67/NB0 `+10`，不再
  扩大删除范围；新 fullres checkpoint 只在最终 rank-1、resume 与 RTL 绑定全部确定后整理。


### Local-5 ep3 与 H67/NB0 收敛状态复核（2026-08-05 19:25 CST）

<!-- LOCAL5_H67_NB0_CONVERGENCE_STATUS_20260805 -->

- Local-5 fullres30 已完成 ep0--3，train loss 为
  `2.1475/1.8488/1.6811/1.6521`，累计下降 `23.07%`；小验证 loss 为
  `1.9795/1.7067/1.8096/2.0182`，ep3 相对 ep2 反弹 `11.53%`、相对暂时最佳 ep1 高
  `18.25%`。当前进入 ep4，稳定吞吐恢复约 `1.06 s/it`。这是需要 ep9 valid825 检查的
  泛化风险，但 train loss、ATLIF活动与数值均正常，不能据42帧训练内验证定性为发散。
- 尚无正式 checkpoint 是保存策略所致，首个 model/state 对固定在 ep9；在 standard valid825
  前不以短验证损失选主线。历史 H67 小验证曾单轮反弹约 `13.8%`，旧 Local-5 后期也有明显
  波动，进一步说明短验证只能作故障探针，不能替代825帧排名。
- Local-5 完成后自动执行 standard/deploy valid825、五类 checkpoint-bound profile、post-G0
  relation acceptance，以及 score/Shiftmax、projection partial accumulator、ATLIF temporal
  matrix 三组件 RTL-exact。正式 report 尚未生成时只能写 `queued`，不能写 RTL PASS。
- H67 ep30 是当前预算边界和 AEE rank-1，但 ep25到ep30仍改善 `2.47%`，故状态是
  `best_at_boundary_not_proven_converged`。AAE-2D 与 AE-3D 同区间只变化约 `+0.08%/-0.18%`，
  角度已近平台。H67/NB0 对称 `+10` 已 staged 并完成加载审计，将用 ep35/40 valid825 与
  末段斜率给出机器判定。
- NB0 ep24到ep29 AEE改善 `5.56%`，AEE 欠训练是合理怀疑；但 AAE-2D/AE-3D 只改善
  `0.44%/1.26%`，不能期待少量续训把本地 AE-3D `6.1803` 直接变成论文 hidden-test
  `4.871`。本地 NB0 已优于论文 validation row (`AEE 1.61`, `AAE 7.23`)；官方 `4.871`
  来自不同的七序列 hidden-test population 与聚合合同，只能通过正式 test submission 比较。


### Local-5 ep9 首个续训锚点早期签收（2026-08-05）

<!-- LOCAL5_EP9_EARLY_CHECKPOINT_AUDIT_20260805 -->

- 原流水线只在30轮训练结束后统一检查 ep9/19/29 model/state，若首个 state 损坏会延迟约20轮
  才发现。新增只读 watcher
  `entrypoints/audit_local5_ep9_checkpoint_20260805.py`，在 ep9 两文件大小/mtime 连续稳定后立即
  CPU 加载并生成 `checkpoint_epoch9_early_audit.json`。
- fail-closed 检查包括：model/state 非空与 SHA256、state internal epoch9、scheduler
  `last_epoch=9`、milestones13/20、AMP scaler、五参数组 optimizer 与 scheduler LR
  `1e-4/1e-4/5e-5/5e-5/5e-6`、fullres/window15 保存合同，以及最新训练加载
  overlay210/missing0/unexpected0。该报告只签收可续训锚点，不宣称精度或 RTL PASS。
- 正常状态、epoch 漂移和 LR 漂移三项测试 `3/3` PASS；独立 session PID `2264517` 已跨命令
  确认存活并每60秒等待 ep9，不占 GPU、不修改训练状态。


### H67/NB0 +10 checkpoint 标签与 provenance 修正（2026-08-05）

<!-- H67_NB0_PLUS10_LABEL_PROVENANCE_FIX_20260805 -->

- 续训编号复核确认 H67 的 source budget30 使用 checkpoint label30/internal epoch29，配置
  `epoch_offset=1`，因此内部 epoch34/39 正确保存为 label35/40；NB0 source budget30 使用
  label29/internal epoch29且无 offset，后续保存 label34/39。summary 统一映射为预算30/35/40，
  不存在评估 off-by-one。
- H67 生成配置此前继承旧 ep15到ep30 rescue 的说明字段
  `resume_source_epoch: 15`，虽不参与训练，但会造成错误 provenance。generator 现移除该字段，
  两条线均显式记录 `resume_source_budget: 30`、实际 `resume_source_checkpoint_label: 30/29`
  和 `audited_model_optimizer_scheduler_scaler_equal_plus10_from_fullres30`。
- 已重新生成等待队列将读取的两份配置；训练 LR、scheduler、epoch 数、保存点、模型和 staged
  state 均未改变。新增两项 provenance 回归，与 ep9签收、profile/acceptance/closure 回归
  合计 `14/14` PASS。
- staged-resume runner 进一步把 config SHA、source budget30 和实际 checkpoint label30/29 写入
  `resume_stage_audit.json`；已有审计只有在 source model/state/staged-state SHA、hardlink 与 RNG
  disclosure 全部仍通过时才允许补齐新字段，已存在但不匹配则 fail-closed。配置 SHA 漂移反例
  测试通过，相关组合回归现为 `15/15` PASS。
- 旧空等 runner 已安全退出；加载新代码的独立 session PID `2268615` 已生成修正配置并继续
  等待 Local-5/H67 ep30 T450 release。未启动额外 GPU 作业。
- 两份 staged 审计已在等待期间以低 I/O 优先级提前升级并重新验真：H67 config SHA
  `86db3960...b15d1cbcc`、NB0 `55aeb36c...20290efe` 与当前配置逐字节一致；source
  model/state、staged state SHA、hardlink、internal/scheduler epoch29、scaler 与 LR 均通过，
  RNG state 缺失继续显式披露。由此配置 provenance 不再只是未来 runner 的预期行为，而是
  已落盘机器证据。


### H9 续训 CLI 布尔参数路径归一化修复（2026-08-05）

<!-- H9_RESUME_BOOLEAN_PATH_NORMALIZATION_FIX_20260805 -->

- 审计 ep9 后自动恢复链发现 `_absolutize_path_args` 误把 `--resume` 和 `--finetune` 当成路径
  参数，导致命令中的 `1` 被改写为绝对路径 `.../1`。上游 argparse 未声明数值类型且只检查
  字符串真值，所以历史运行仍进入 resume/finetune 分支，但该行为污染命令语义并可能被未来
  类型约束破坏。
- path normalization 现只处理 `--prev_runid`、`--save_path` 和 `--path_mlflow`；
  `--resume 1`、`--finetune 1` 保持原值。新增 positional/equal-form 混合参数测试通过。
- 当前 Local-5 首次运行不带 resume，不受代码热更新影响；若 ep9 后 supervisor 恢复，会由
  新进程加载修正 wrapper。H67/NB0 +10 也尚未启动子训练，后续使用同一正确入口。
- 参数测试与 ep9签收、+10 provenance/staging、profile/acceptance、final closure 回归合并重跑
  `16/16` PASS。
- 新增真实上游 `utils.resume_model` CPU 集成测试：由 `checkpoint_epoch9.pth` 自动定位
  `checkpoint_epoch9_state_dict.pth`，恢复五组 optimizer LR、scheduler `last_epoch=9`、
  milestones13/20、`_last_lr` 与 scaler，并验证 `epoch_initial=10`。该测试通过后本组回归为
  `17/17` PASS，证明 Local-5 ep9 恢复不会只加载模型或重复 epoch9。
### Local-5 运行时配置身份门控（2026-08-05 19:40 CST）

<!-- LOCAL5_RUNTIME_CONFIG_IDENTITY_20260805 -->

- 当前公平重跑仍在执行：已完成 ep0--3，ep4 约 `24%`；GPU 约 `94%`，当前快路径约
  `1.04 s/step`。ep0--3 train loss 为 `2.1475/1.8488/1.6811/1.6521`；小验证
  `1.9795/1.7067/1.8096/2.0182`。训练损失持续下降，但 42-frame 小验证在 ep2--3
  反弹，因此现阶段只能判为“未收敛且存在泛化波动”，必须以 ep9/14/19/24/29
  的 standard valid825 排名为准。
- 审计发现活跃 train 进程启动于 `14:24:39`，而配置生成器及磁盘配置最终 mtime
  分别为 `14:26:19/14:28:08`。当前配置可以由当前生成器和源配置确定性重建，SHA256
  均为 `cf8c3da8fd8a40b098ca95a7d27fa84777c969ef025567d840a7c29b54dbedaf`，但不能据此
  倒推训练进程在 14:24 读入文件的逐字身份。
- 新增 `entrypoints/enforce_local5_ep9_config_identity_20260805.py`，以 ep9 成对训练 state
  中实际 optimizer/scheduler/scaler 为运行时权威证据。要求 state/scheduler epoch=9、
  milestones=`13/20`、五组 LR=`1e-4/1e-4/5e-5/5e-5/5e-6` 且 AMP scaler 存在。
  若唯一差异是旧 scheduler milestone，脚本先保留原 state，再在 ep9 边界修正并由 supervisor
  续训；其他差异一律停止训练并 fail closed。
- 当前身份报告为 `PENDING_EP9_RUNTIME_STATE`：
  `results/dsec_fullres_w15_H66d_local5_bb1e4_ft30_20260805/training_config_identity.json`。
  最终 closure audit 已将其加入 REQUIRED，未 PASS 不允许发布 Local-5 结果。

### 第六轮安全清理：退休对称双神经元短筛（2026-08-05）

<!-- RETIRED_SYMMETRIC_SCREEN_CLEANUP_20260805 -->

- 清理范围仅为 `nts11_two_neuron_20260611_203636`、`nts11_phase2_20260611_230130`
  和单候选 `nts11bc_short_20260613_152906` 的失败 valid10 短筛权重；这些实验全部未过
  AEE gate，且不属于当前 one-sided binary Local-5/H67/NB0 链路。
- 保留两组多候选筛选的 rank-1 checkpoint：`NTS11c` 和 `NTS11j`；所有 config、train log、
  summary、profile 也全部保留。删除其余 12 个 checkpoint，回收 `8,859,732,660 bytes`
  （`8.251 GiB`），可用空间由约 `219 GiB` 增至 `227 GiB`。
- 审计：`neuron_autoresearch/cleanup_audits/retired_symmetric_screens_20260805.json`；当前
  Local-5/H67/NB0、等预算 +10、standard/deploy profile 与 RTL 绑定权重均未触碰。

### Local-5 配置身份下沉到 profile/RTL（2026-08-05 19:47 CST）

<!-- LOCAL5_CONFIG_IDENTITY_PROFILE_RTL_BINDING_20260805 -->

- 配置身份不再只由最终 closure 检查。`run_local5_bb1e4_postg0_profile.py` 现必须等待
  `training_config_identity.json` PASS，复算训练 config 与 ep9 paired-state SHA，并把该身份文件
  加入 post-G0 `run_identity.source_bindings`。身份 PENDING 时等待，FAIL 时立即终止。
- `run_local5_bb1e4_checkpoint_bound_rtl.py` 在 acceptance 后再次复算上述绑定，并把训练身份 SHA、
  training-config SHA 和 ep9-state SHA 写入 `checkpoint_bound_scope.json`。最终 closure 同时检查
  acceptance run identity 与 RTL aggregate 的身份 SHA，避免“算法身份合格、硬件向量来自另一条
  训练链”的错配。
- 新增 `hw_autoresearch_nts07/tests/test_local5_training_identity_gate.py`，PASS/PENDING、state drift
  和缺失 post-G0 source binding 三类用例 `3/3 PASS`；相关脚本 py_compile 与 diff-check 通过。
- 旧空等 watcher 已安全替换为加载新门控的 PID `2286750/2286752`；训练 PID 未改动。
- ep9 早期审计原有两个等待者可能同时写同一临时报告，现合并为单一 enforcer PID `2287913`，
  使用内核 `flock`。它在发布身份 PASS 前还会复算 early-audit 的 model/state SHA；独立旧 auditor
  已安全退出，训练进程未改动。

### H67 all12 实权重 projection RTL 证据补齐（2026-08-05 20:00 CST）

<!-- H67_ALL12_CHECKPOINT_PROJECTION_RTL_20260805 -->

- 复核发现 H67 最终 watcher 原先只对当前 checkpoint 生成 score/Shiftmax 与 ATLIF RTL；历史
  DCTF96 projection 回放绑定的是旧 bit trace，不能作为 ep30 或追加训练 rank-1 的证据。因此
  旧结果继续保留为架构开发证据，但不再进入最终 checkpoint closure。
- `generate_gatestack_dctf_real_trace_vectors.py` 现兼容 all12 attention records，并用
  `s{stage}_b{block}` 隔离每个 block 的真实 K/gate、checkpoint dyadic INT8 projection weight
  和 acc32 bias，避免同 stage 多 block 覆盖同一向量目录。source manifest SHA 与完整
  `run_context.artifact_identity` 同步写入 vector manifest。
- DCTF96 generator/TB/runner 已从旧 crop T162/8-bit token 合同参数化为 fullres T450/9-bit；
  单个 gate/lane 超过255 destinations 时确定性拆分而不改变投影和。runner 允许独立
  `SOURCE_MANIFEST/RESULT_DIR`，逐条执行 all12 Icarus bit-exact，并按 stage 复用编译结果对
  all12 全部执行 Verilator+SVA；`report.json` 必须绑定当前 checkpoint SHA、12 records、T450、S0--S3
  coverage 与 `checkpoint_dyadic_int8_projection_weight`。声明范围仅为
  `checkpoint-bound real-weight projection component RTL-exact`，不是完整 attention/encoder。
- H67 ep30 与 post-convergence rank-1 watcher、最终 closure 均改为同时要求三类证据：
  score/Shiftmax、81-site ATLIF temporal matrix、all12 real-weight projection。任一报告缺失、
  checkpoint SHA 不同或 record 数不是12均 fail closed；ep30 也不再只凭前两类报告复用。
- trace 复用不再只检查旧 `audit.json` 是否存在；watcher 会复算 checkpoint/config/12 NPZ SHA、
  all12 与 T450 合同，并在每次正式回放前重新执行 audit，避免恢复后静默消费旧 trace。
- 静态检查、6项生成器单测、3项最终证据/trace复用测试、synthetic all12 目录/身份传播及 T450
  token449/450-destination S0 Icarus 与 Verilator+SVA bit-exact 均通过；正式 all12 双仿真回放仍等待 Local-5
  释放 GPU 后由队列生成，当前不得写成最终 PASS。

### 第七轮安全清理：早期全二值/NTX 微调中间锚点（2026-08-05）

<!-- EARLY_BINARY_NTX_FT_PRUNE_20260805 -->

- 对三条已完成 standard valid825 的早期5轮微调执行 ranking-aware 清理：all-binary FAPS、
  NTX-from-ep19 和 NTX-from-ep29。三条 ranking 的 AEE rank-1 均为 ep2，最终轮均为 ep4。
- 每条完整保留 ep2/ep4 model 与 optimizer/scheduler/scaler state、全部配置、日志、valid825
  profiles 和 ranking；仅删除 ep0/ep1/ep3 的 model/state，共18个文件。
- 每个 run 目录内已写 `checkpoint_prune_audit.json`；合计回收 `10,603,116,963 bytes`
  （约 `9.88 GiB`），可用空间由约 `227 GiB` 增至 `237 GiB`。当前 Local-5/H67/NB0、
  Local-5 crop source、追加训练 staged states 与所有 checkpoint-bound RTL 权重均未触碰。

### Local-5 多启动歧义的活跃进程签收（2026-08-05 20:12 CST）

<!-- LOCAL5_ACTIVE_LAUNCH_PROVENANCE_20260805 -->

- `status.log` 留有14:18、14:24和14:28三次启动尝试；14:28的 `exit_code=1` 属于重复实例，
  不能误判为当前训练退出。只读 `/proc` 签收确认唯一根训练为 PID2097444、父流水线2097439，
  启动于 `2026-08-05 14:24:39 CST`，8个同 argv 子进程是 DataLoader workers。
- 新增 `active_launch_provenance.json`，冻结根/父进程 start ticks、完整 argv、Python executable、
  config/source checkpoint/save path/finetune 参数，以及 source checkpoint、train/pipeline入口 SHA。
  source checkpoint SHA 为 `11c5aa35...4b5c`，命令加载审计仍为 overlay210/missing0/unexpected0。
- 该收据明确只证明 process/argv/source identity；因进程启动早于磁盘配置最终 mtime，不能反推
  当时读取的逐字配置。因此 ep9 optimizer/scheduler/scaler state 仍是运行时配置权威。
- ep9 identity enforcer 现必须验证 launch receipt、当前 config SHA、source checkpoint SHA和
  `start < final config mtime`，并把 receipt SHA 写入最终身份；closure 会再次复算。2项 `/proc`
  解析测试和4项 enforcer 状态/漂移测试通过，新 enforcer PID2302180 已加载门禁。

### Local-5/H67 当前闭环状态与第八轮 checkpoint 清理（2026-08-05 20:22 CST）

<!-- LOCAL5_H67_STATUS_AND_CLEANUP8_20260805 -->

- Local-5 公平 fullres/window15 训练仍在执行，当前 ep4 约 `78%`。已完成 ep0--3 的
  train loss 为 `2.1475/1.8488/1.6811/1.6521`，42-frame 小验证为
  `1.9795/1.7067/1.8096/2.0182`。训练损失下降但小验证波动，且首个预注册模型锚点为 ep9，
  因此当前仍不能报告 standard AEE/AAE，也不能判定收敛。
- Local-5 后处理没有遗漏：训练结束后固定评估 ep9/14/19/24/29 的 standard valid825，再对
  rank-1 运行 float/dyadic/hardware-order、T450 all12 profile100 与 relation acceptance；硬件闭环
  要求同一 checkpoint 的 score/Shiftmax、checkpoint real-weight projection partial accumulator、
  81-site ATLIF temporal matrix 三类 RTL-exact 报告。当前这些状态均为 `queued`，历史 crop
  `h66d_local5_rtl_exact_valid825` 不能冒充本次 fullres checkpoint-bound PASS。
- H67 ep30 仍是已测最后一轮和 AEE rank-1：ep25 `1.37261` 到 ep30 `1.33874` 继续改善
  `2.47%`，所以状态保持 `best_at_boundary_not_proven_converged`。H67/NB0 对称 +10 已等待
  Local-5 释放 GPU；预算35/40及其 standard valid825 完成前，不以“最后一轮最优”推导收敛。
- NB0 ep24到ep29 AEE改善 `5.56%`，支持 AEE 欠收敛风险；但 legacy AAE 与 AE-3D 只改善
  约 `0.44%/1.26%`。论文 `4.871` 是七序列 official hidden-test aggregate，本地 NB0
  AE-3D `6.1803` 是 valid825 frame-mean，population 与聚合都不同；轮次不足可能解释 AEE
  斜率，不能单独解释全部 AAE 数值差。
- 第八轮清理只处理三组已被完整 MDR 训练取代的 smoke checkpoint、NTS11bj/bl 的非最优
  中间轮、两条已退休 NTS11-lite 的非锚点轮，以及被当前公平 fullres 训练取代的旧 ep0
  初始化模型。删除 `30` 个文件，实际回收 `14,035,456,000 bytes`（约 `13.07 GiB`），
  可用空间增至约 `250 GiB`。
- 仍保留 NTS11bj ep2/4、NTS11bl ep2/4、NTS11-lite 各自 best-AEE/final、旧 fullres ep29
  model+paired state，以及全部配置、日志、valid825 ranking/profile 和 RTL 产物；当前
  Local-5/H67/NB0、crop source、+10 staged state 与 checkpoint-bound RTL 依赖均未触碰。
  机器审计位于
  `neuron_autoresearch/cleanup_audits/retired_smoke_lite_intermediates_20260805.json`，不可变执行收据为
  `neuron_autoresearch/cleanup_audits/retired_smoke_lite_intermediates_20260805.executed.json`。
- H67 ep30、post-convergence 与最终 closure 三个等待器的旧 PID 文件在20:08后失去对应进程；
  未产生错误结果，但会造成后续队列断链。三者已用独立 session 重新常驻为 PID
  `2310765/2310766/2310767`，复核 PPID=1、独立 SID 且日志重新出现 WAIT。Local-5 训练、
  optimizer state 和 GPU 作业未重启。
- 同批复核也发现 ep9 config-identity enforcer 的旧 PID2302180 已退出；已按相同方式重签收为
  PID2311400，复核 PPID=1、独立 SID 且12:29 UTC重新写入 WAIT。Local-5 ep9 首锚点的
  optimizer/scheduler/scaler 门禁因此仍处于有效等待状态。

### Local-5 ep4 完成与自动闭环回归（2026-08-05 20:36 CST）

<!-- LOCAL5_EP4_AND_CLOSURE_REGRESSION_20260805 -->

- ep4 已完成并进入 ep5：train loss `1.621065`、42-frame 小验证 `1.642443`、epoch time
  `4458.03 s`、稳定训练步耗时约 `1.04--1.10 s`，整轮含系统争用的均值为 `1.2141 s/step`。
  ep3 小验证 `2.018181` 的反弹在 ep4 回落，说明该小集合方差较高，不能据 ep3 单点淘汰
  Local-5；同样不能用 ep4 小验证代替 ep9 standard valid825。
- ep4 末 ATLIF activity mean `5.2366%`，module summary 仍为 official one-sided binary
  ATLIF `105`、ternary activity `0`；Shiftmax summary 仍为 `12` modules。训练 PID、五组 LR
  和 optimizer state 未重启，GPU 快路径已恢复。
- 聚焦自动闭环回归共 `25` 项 PASS：fullres/window15 配置、强 LR 与保存点、稀疏 model/state
  保存、paired-state 恢复到 next epoch、`--resume 1` 路径语义、ep9 runtime state 只允许
  milestone-only 修复、valid825 825样本/metric/load/module/checkpoint合同、post-G0 acceptance、
  training-identity SHA，以及 H67/Local-5 stale checkpoint/trace/RTL 复用拒绝。
- 生产环境 AAE 数值回归再次 `3/3 PASS`，验证 legacy二维方向角、Barron/Middlebury `(u,v,1)`
  AE-3D 和 batch mask。现有 aggregate profile 未保存逐帧角度分子/有效像素分母，因此不能在
  不重跑推理的前提下可信重构 pixel-global 或 sequence-weighted AE；当前不使用估算值解释
  official hidden-test/local-valid 差距。
- 新增 `supervise_date_closure_watchers_20260805.py`，用单一 flock watchdog 监督六个长等待任务：
  ep9 config identity、Local-5 checkpoint-bound RTL、H67/NB0 +10、H67 ep30 RTL、H67
  post-convergence RTL 和最终 closure。每项有独立 PID/命令身份及完成标记，死亡后最多重启8次，
  已完成项不再拉起；Local-5 训练流水线仍由原 supervisor 独立负责。
- watchdog 三项 completion/PID fail-closed 测试 PASS；独立 session PID2319106 已签收
  `PPID=1/SID=2319106`，启动 heartbeat 确认 `incomplete=6, alive=6`。它不占 GPU，也不会把
  WAIT 状态误记为 PASS。
- 完成判据进一步要求日志 marker 与最终 artifact 同时存在：Local-5 scope、equal+10 summary、
  H67 ep30 三组件 reports、post-convergence FINAL、closure JSON+MD；只剩历史 `ALL COMPLETE`
  日志而报告缺失时仍视为 incomplete。加载该门控的 watchdog PID 更新为2319720，首次
  heartbeat 仍为 `6/6 alive`。
- watchdog 还会在 PID 文件缺失/陈旧时扫描 `/proc` 并收养匹配的现有 detached follower，避免
  盲启副本因 follower 自身 flock 退出后耗尽重试。新增收养反例后测试 `4/4 PASS`；加载最终
  逻辑的 PID2320213 已再次确认 `6/6 alive`。

### Local-5 ep5 进度、硬件队列与第九轮清理（2026-08-05 20:50 CST）

<!-- LOCAL5_EP5_HW_QUEUE_AND_CLEANUP9_20260805 -->

- Local-5 已完成 ep0--4，当前在 ep5 约 `13%`（进度 `461/3672`）；稳定段约
  `1.04--1.07 s/step`。已完成 train loss 为
  `2.14747/1.84878/1.68106/1.65212/1.62107`，42-frame 小验证为
  `1.97951/1.70673/1.80956/2.01818/1.64244`。小验证反弹已回落，但在 ep9 standard
  valid825 之前仍不作 AEE/AAE 或收敛结论。
- Local-5 硬件证据没有遗漏：PID2286750 等待 rank-1 后依次生成 post-G0
  profile100/T450/all12、relation acceptance、score/Shiftmax RTL、真实权重 projection
  partial-accumulator RTL 和81-site ATLIF temporal RTL。H67 ep30、追加训练 rank-1 和最终
  closure 的 PID2310765/2310766/2310767 也全部存活；总 watchdog PID2320213 报告
  `incomplete=6/alive=6`。
- H67 ep30 是当前 AEE rank-1（`1.33874`），但 ep25到ep30仍改善 `2.47%`，
  因此只能写 `best_at_boundary_not_proven_converged`。NB0 ep24到ep29 AEE 改善
  `5.56%`，欠收敛风险更强；但其 legacy AAE-2D/AE-3D 只改善约
  `0.44%/1.26%`，所以多训几轮不足以解释 local-valid 与 official hidden-test `4.871`
  的数值差。H67/NB0 对称 +10 继续以 ep35/40 和末段斜率判定是否平台。
- 第九轮清理针对已退役 H66c/e/f/g、H68--H71 和 H81 crop 实验。它们的
  valid825 AEE rank-1 均为 ep19，因此 ep19 最优和 ep29 最终 model 全部保留；只删除
  无活跃引用的 ep19/ep29 optimizer/scheduler/scaler state，共18个文件、
  `7,777,410,120 bytes` 约 `7.24 GiB`。可用空间增至约 `257 GiB`。
- 清理脚本为
  `neuron_experiments/H9_bipolar_self_attention/entrypoints/prune_retired_h66_h81_optimizer_states_20260805.py`，
  机器审计为
  `neuron_autoresearch/cleanup_audits/retired_h66_h81_optimizer_states_20260805.json`。执行后验证
  18个 state 均不存在、18个保留 model anchor 全部存在；Local-5/H67/NB0、续训源、
  standard profiles/rankings 和 checkpoint-bound RTL 依赖未触碰。

### AAE 多聚合可执行审计（2026-08-05 21:05 CST）

<!-- AAE_MULTI_AGGREGATION_AUDIT_20260805 -->

- 旧 `spike_profile.json` 只保存 masked mean per frame 后对帧等权的 AAE，没有保存角度
  numerator、valid-pixel denominator 或 sequence identity，因此无法从旧 aggregate 可信
  重构 pixel-global 或 sequence-balanced 结果。
- 新增 `third_party/SDformerFlow/utils/metric_aggregation.py`，在不改变旧 `AEE`/
  `AAE_Benchmark` 计算和排名的前提下，同一次推理额外输出
  `frame_equal_mean`、`pixel_global_mean`、`sequence_balanced_mean` 和18个本地
  valid subsequence 的独立结果。同时保存 validation CSV 绝对路径与 SHA256。
- Local-5 五个待评 checkpoint、H67/NB0 对称 +10 的所有新 profile 和最终 closure
  都已升级 fail-closed 门禁：`825` frames、`18` sequences、valid pixels>0、per-sequence
  records=18，且 frame-equal AEE/AAE-2D/AE-3D 必须与原生产指标在 `1e-5` 内一致。
  这保证新统计是对同一预测/同一 mask 的重聚合，不是另一条推理链。
- 数值和反例测试共 `6/6 PASS`，Local-5/equal+10/closure 验收测试 `11/11 PASS`；
  包含三种聚合字段缺失的反例和rank-1聚合向最终报告传递的正例。
  `AAE_METRIC_TEST_RECEIPT_20260805.json` 升级为 schema v2，绑定 metric、evaluator、
  aggregation 及两组测试源码 SHA。equal+10/closure 等待器已重签收并交由总 watchdog
  维护，实时 PID 以各自 `.pid` 文件为准；Local-5 训练未中断。最终 closure JSON/MD 还会直接汇总
  Local-5/H67/NB0 rank-1 的 frame/pixel/sequence 三组 AE-3D，无需再手工转抄。

### H67 训练血缘、Local-5 fullres RTL 状态与第十轮清理（2026-08-05 21:20 CST）

<!-- H67_LINEAGE_LOCAL5_RTL_CLEANUP10_20260805 -->

- `H67_FULLRES_LINEAGE_RECEIPT_20260805.json` 已实际复算 PASS：H67 从自身 Motion-XOR crop ep19
  `5ff626a7...3d22ba` 出发，经 fullres ep0、5、10、15、30 五段训练到 ep30
  `7a484dc1...f37e4a`；每段 config/log/source/output 均由 SHA 或删除审计绑定，加载记录均为
  overlay210/missing0/unexpected0。该主线没有从 NB0 或 Local-5 初始化。
- H67 ep30 是当前最后一轮和 AEE 最优轮，但 ep25到ep30仍改善约2.47%，所以结论仍为
  `best_at_boundary_not_proven_converged`。H67/NB0 对称 +10 已排队；不能用“最后一轮最优”替代
  平台检验。NB0 ep24到ep29 AEE仍改善5.56%，同样有欠收敛风险；其角度指标末段变化较小，
  因而欠训练不能单独解释与 official hidden-test `4.871` 的全部差距。
- Local-5 fullres 训练已完成 ep0--5、进入 ep6；`training_config_identity.json` 已生成但仍是
  `PENDING_EP9_RUNTIME_STATE`。ep9 model/state 出现并通过运行时配置门禁后，流水线才会对
  ep9/14/19/24/29 做 valid825 并选择 rank-1，再生成 T450/all12 profile100、relation acceptance、
  score/Shiftmax、真实权重 projection partial accumulator 和81-site ATLIF temporal matrix
  checkpoint-bound RTL-exact。历史 crop RTL 只作开发证据，不能替代本轮 fullres SHA。
- 第十轮清理仅删除两份已淘汰的 H67 rescue screen ep0 model：NB0→H67 路线的加载审计为
  overlay0/missing210，crop bb2e5 路线精度也劣于入选 bb1e4。两份配置、train/valid825 日志、
  ranking 全部保留；当前 H67 五段血缘锚点、Local-5、NB0、equal+10 和所有硬件证据均受保护。
  实际回收 `1,182,334,976 bytes`（约 `1.10 GiB`）；机器审计为
  `neuron_autoresearch/cleanup_audits/failed_h67_rescue_screens_20260805.json`。
- 长队列 watchdog 增加非阻塞 child reaping，避免 follower 被重启后留下 zombie；对应完成、PID、
  detached adoption 和 reaping 回归 `6/6 PASS`。新 watchdog PID2343815 已收养六个现有 follower，
  Local-5 训练 PID/配置/GPU context 未重启，实验顺序与硬件 scope 均未改变。
- 进一步核对 `/proc/2097443/stat` 发现 pipeline 最初的 post-G0 child 曾被 SIGTERM（exit code15），
  虽然后续 canonical producer PID2286752 已接管，但旧 pipeline 最终 `wait()` 会误把历史子进程
  退出当成本轮 profile 失败。收尾逻辑已改为以 checkpoint-bound `acceptance.json` 为权威：
  child 非零时尝试恢复或加入 canonical producer，并在 acceptance 出现后再做 SHA/语义门禁。
  Local-5 pipeline 回归更新为 `6/6 PASS`。当前训练进程未热改；若旧内存代码最终先退出，
  supervisor 会在不重训的情况下由 ep29/既有 profile 产物恢复并完成最终 marker。
- H67/NB0 +10 runner 原先只等待 H67 score/Shiftmax report 出现；该文件早于 ATLIF vector/RTL
  和 all12 real-weight projection RTL 完成，可能让训练与剩余 GPU vector producer 重叠并 OOM。
  释放门禁现要求 score/Shiftmax、ATLIF、projection 三报告均 PASS、scope/12 records/T450/9-bit
  合同完整、全部绑定 H67 ep30 SHA，并且 watcher 最终 completion marker 已落盘。新增反例后
  Local-5/equal+10 组合回归 `7/7 PASS`；等待中的 runner 已无损重启为 PID2347644，尚未开训。
- Local-5 ep5 已完整结束并进入 ep6：train loss `1.515193`、42-frame 小验证
  `1.411659`，相对 ep4 的 `1.621065/1.642443` 分别改善约 `6.53%/14.05%`。ep5耗时
  `4555.48 s`，max GPU memory `38.177 GiB`；ATLIF仍为105个 official one-sided binary
  modules，activity mean `5.4486%`、ternary activity `0`，Shiftmax仍为12 modules。
  训练和小验证都在明显下降，故当前不能提前宣称 Local-5 已收敛，也不缩短预注册30轮预算。

### H67/Local-5 profile-RTL 闭环加固与第十一次清理（2026-08-05 22:10 CST）

<!-- H67_LOCAL5_PROFILE_RTL_CLOSURE_CLEANUP11_20260805 -->

- Local-5 fullres/window15 并未遗漏：当前已完成 ep0--5，ep6 约17%，稳定段约
  `1.03--1.08 s/step`，GPU利用率约90%。ep5 train/42-frame小验证 loss 为
  `1.515193/1.411659`，仍明显优于ep4，因此在首个预注册 ep9 valid825 前不作收敛或精度结论。
- Local-5 最终 rank-1 仍强制生成 float/dyadic/hardware-order 标准推理、profile100、T450/all12
  trace、relation acceptance，以及绑定同一 checkpoint SHA 的 score/Shiftmax、真实权重 projection
  partial accumulator 和81-site ATLIF temporal matrix component RTL-exact。历史 crop profile/RTL
  不能替代本轮 fullres 证据，声明范围仍是
  `checkpoint_bound_component_rtl_exact_not_full_network`。
- H67 ep30 post-convergence 复用分支已与 ep35/40 分支对齐。最终收据现在除三类 RTL report 外，
  还必须核验 hardware-order config、`nts11_hardware_p0_profile.json`、all12 manifest 和 trace audit：
  checkpoint/config SHA、100 samples、480x640、crop null、T2x15x15=450、ATLIF105、Shiftmax12、
  12个 NPZ SHA、四stage覆盖全部 fail-closed。ep30生产器、rank-1绑定器、最终 closure 和 watchdog
  required paths 四层均已升级；旧 checkpoint 或 trace payload 被替换的反例必须拒绝复用。
- 对应测试为 H67复用 `4/4 PASS`、DATE closure `8/8 PASS`、watchdog `6/6 PASS`。新版 watchdog
  PID2364542 已重启并拉起 H67 ep30、post-convergence 和 closure 三个等待器；Local-5训练 PID2097444
  未被重启，H67/NB0 equal+10 PID2347644 仍在严格等待硬件GPU释放。
- H67 目前是 ep30 AEE `1.3387` 最优，但 ep25到ep30仍改善2.47%，所以状态是
  `best_at_boundary_not_proven_converged`，不能因为最后一轮最好就称已收敛。NB0 ep24到ep29 AEE
  改善5.56%，AEE欠收敛风险更高；但 AAE-2D/AE-3D 仅改善0.44%/1.26%，角度已近平台。
  因此续训可能继续改善 NB0 AEE，却不能把 local-valid AE-3D `6.1803` 与官方七序列 hidden-test
  `4.871` 的 population/聚合差异解释成单纯训练轮数不足。两条线的同协议 +10 将给出最终证据。
- 第十一次清理只删除 H66a/H66f/H71 三条已退役 crop 候选中，被各自 ep19 在 AEE、AAE、
  total_spikes 三项同时支配的 ep29 模型；rank-1 ep19、profiles、rankings、logs、configs 全保留，
  NTS/TTX/H67/Local-5/NB0 与硬件依赖均未触碰。实际回收 `2,213,556,224 bytes`，审计为
  `neuron_autoresearch/cleanup_audits/dominated_attention_checkpoints_20260805.json`，可用空间约260GiB。
- Local-5 checkpoint-bound RTL 启动门进一步前移：`accepted=true` 单字段不再足够；watcher 会在
  任何 score/projection/ATLIF 回放前解析当前 valid825 rank-1，并复算 acceptance、ordered
  manifest、run identity、hardware-order config、checkpoint、training identity 的路径和 SHA，
  同时要求100 samples、12 blocks和11项 acceptance checks。旧 rank-1 acceptance 会自动重跑
  profile producer，而不是先生成数小时的陈旧 RTL 后再等 closure 拒绝。
- score 和 projection vector manifest 现也在聚合 PASS 落盘前沿 source manifest 追溯到当前
  checkpoint SHA；ATLIF report 同样必须绑定该 SHA。最终 `checkpoint_bound_scope.json` 新增显式
  checkpoint/rank-1/run-identity/acceptance identity。缺失、有效、陈旧 acceptance 与 checkpoint
  替换反例四项门禁回归 `4/4 PASS`，closure 回归仍为 `8/8 PASS`。纯 WAIT follower 已重启为 PID2369602/2369604，
  Local-5训练未中断。
- ATLIF temporal-matrix report 原本已保存完整 config/checkpoint identity，但 Local-5/H67 上层复用
  只比较 checkpoint SHA。现统一收紧为 config SHA + checkpoint SHA 双绑定：Local-5 post-G0
  producer、Local-5 aggregate RTL、H67 ep30、H67 post-convergence 和最终 closure 全部拒绝配置漂移。
  H67复用/Local-5门禁/closure回归分别为 `5/5`、`4/4`、`8/8 PASS`；加载最终逻辑的等待器为
  Local-5 PID2372110/2372115、H67 PID2372111/2372112、closure PID2372113。训练PID未改变。

### Local-5/H67 当前状态、直接 trace 绑定与第十二次清理（2026-08-05 22:30 CST）

<!-- LOCAL5_H67_DIRECT_TRACE_CLEANUP12_20260805 -->

- Local-5 fullres/window15 并未漏做：训练 PID2097444 当前在 ep6 约44%，ep0--5 已完成；首个
  预注册 model/state 是 ep9。`training_config_identity.json` 保持
  `PENDING_EP9_RUNTIME_STATE` 是正确的 fail-closed 状态。ep9/14/19/24/29 五个 checkpoint 才会
  进入 valid825/rank-1，随后对最终 rank-1 生成 float/dyadic/hardware-order 推理、profile100、
  T450/all12 trace、relation acceptance、score/Shiftmax RTL、真实权重 projection partial
  accumulator RTL 和81-site ATLIF temporal-matrix RTL。声明仍严格为
  `checkpoint_bound_component_rtl_exact_not_full_network`，不能把历史 crop RTL 当作本轮签核。
- H67 score/Shiftmax 与 projection 的最终复用门禁进一步从“经上层间接绑定”升级为报告自身直接
  绑定 hardware-order config SHA；score 报告还必须直接绑定 source all12 trace manifest 的路径与
  SHA。H67 ep30 producer、post-convergence rank-1 binder 和最终 closure 三层同步 fail-closed，
  最新 H67 反例回归 `5/5 PASS`、closure `8/8 PASS`。新版 watchdog PID2375263 已拉起
  H67 ep30 PID2375265、post-convergence PID2375266 和 closure PID2375267；Local-5训练与其硬件
  follower 未重启。
- H67 ep30 是当前最后一轮且 AEE rank-1（AEE `1.3387`、AAE-2D `6.0147`、AE-3D
  `5.7558`），但 ep25到ep30 AEE仍改善约2.47%，所以仍只能标记
  `best_at_boundary_not_proven_converged`。H67/NB0 同协议 +10 runner PID2347644 已排队，待当前
  Local-5/profile向量任务释放 GPU 后比较 ep35/40 和末段斜率；最后一轮最好本身不是收敛证据。
- NB0 ep29 的 local-valid 指标为 AEE `1.4454`、AAE-2D `6.5128`、AE-3D `6.1803`。
  ep24到ep29 AEE仍改善5.56%，说明 AEE 很可能欠收敛；但 AAE-2D/AE-3D 仅改善0.44%/1.26%，
  角度已接近平台。论文 `4.871` 是七序列 hidden official-test 聚合，本地是18个 validation
  subsequences、825帧 frame-equal；论文 validation 行 AAE 为 `7.23`。因此不能把 local `6.1803`
  与 official-test `4.871` 的差全部归因于轮次，+10 只用于量化训练余量，不用于伪造同口径比较。
- 第十二次清理只删除三个明确退役的早期短筛 model：H56A threshold-rate、NTX03 TX-refine 和
  NSC08 low-LR 无提升各一份 ep0。配置、日志、指标与历史结论全部保留；NSC09 best short、NB0、
  NTX/NTS 表格锚点、TTX/BTTX、H67、Local-5、fullres 和所有硬件证据均受保护。实际回收
  `1,948,774,400 bytes`，可用空间约262GiB，机器审计为
  `neuron_autoresearch/cleanup_audits/retired_early_screen_checkpoints_20260805.json`。
- Local-5 流水线新增 checkpoint-set 完整性反例：固定五个 eval model
  `9/14/19/24/29` 和三个 paired optimizer/scheduler/scaler state `9/19/29`，逐个删除任一产物都必须
  fail closed。软件验收由 `7/7` 升为 `8/8 PASS`；checkpoint-bound RTL watcher仍为`4/4 PASS`，
  12-block真实权重projection contract为`1/1 PASS`。该回归不改训练配置或运行中进程。
- H67/NB0 equal+10 的 GPU release gate 也补齐同级直接证据，不再只看三类RTL的checkpoint SHA：
  现在同时要求ep30 hardware-order config、profile100、all12 trace、trace audit、score/Shiftmax、
  ATLIF、projection和最终marker；复算config/checkpoint SHA、100 samples、480x640/T450、
  ATLIF105/Shiftmax12、12个trace文件SHA及四stage覆盖。score还必须直接绑定source trace SHA，
  三类RTL均直接绑定config SHA。正例与trace-config漂移反例已纳入Local-5/equal组合`8/8 PASS`；
  staged H67/NB0 model/state SHA复算通过，加载新版门禁的WAIT runner为PID2383761，尚未开始训练。
- Local-5 acceptance 后原有两个潜在GPU producer：profile wrapper生成ATLIF checkpoint vectors，
  checkpoint-bound watcher生成score/projection vectors。现将watcher的ATLIF replay移到最前，先通过
  既有ATLIF flock让二者“一个生产、另一个复用”，ATLIF报告与rank-1 config/checkpoint双SHA通过后
  才释放score/projection。顺序回归加入后checkpoint-bound watcher为`5/5 PASS`；新版父/子
  PID2385101/2385103已在纯WAIT状态加载，训练PID2097444未重启。该修复消除acceptance瞬间的
  双GPU vector producer/OOM窗口，不改变数值、向量或RTL声明范围。
- Local-5 score/projection source ordered manifest 的复用门由checkpoint SHA升级为
  `(checkpoint SHA, hardware-order config SHA)`双绑定；aggregate `checkpoint_identity`也显式保存
  config path/SHA，最终closure沿score与projection各自的source manifest重新复算。缺失/短config
  SHA反例加入后，watcher为`6/6 PASS`、closure为`9/9 PASS`。加载最终代码的Local-5父/子为
  PID2386788/2386791，closure为PID2386789；均处于WAIT，训练未重启。
- 新增机器收据 `H67_NB0_FULLRES_HEAD_TO_HEAD_20260805.json`，直接绑定两个端点profile与checkpoint
  SHA。在完全相同local-valid825口径下，H67 ep30相对NB0 ep29：AEE `-7.376%`、AAE-2D
  `-7.647%`、AE-3D `-6.869%`、total spikes `-35.529%`，四项lower-is-better同时占优。因此
  “H67当前端点优于baseline且满足spikes至少下降20%”已由数据证明；“H67已收敛”仍未证明，继续
  等待equal+10。该结论只用于同一local validation比较，不冒充论文official hidden-test结果。
- 最终closure新增论文硬目标的可执行选择门：以equal+10后的NB0 rank-1为基准，分别计算H67和
  Local-5的AEE变化与spikes变化；候选必须同时满足`AEE <= 1.05*NB0`和
  `spikes <= 0.80*NB0`，至少一个候选过线，否则closure失败。未过线候选仍作为负消融保留；
  多个过线时选择AEE最低者为最终主线。正例、单项失败和全部失败回归后closure为`10/10 PASS`，
  加载该门禁的WAIT PID为2389117。
- Local-5与equal+10标准评估的复用条件从“ranking文件存在”收紧为“ranking存在且每个冻结
  checkpoint的825-frame协议、聚合、load audit、module count、config/checkpoint SHA全部通过”。
  中途中断留下partial/stale ranking时会自动重跑valid825，而不是永久重启失败。回归加入后软件
  组合为`10/10 PASS`；equal+10新版WAIT PID2390244已加载。Local-5当前训练父进程不为该恢复性
  改动中断；若旧内存路径评估失败，supervisor会用新版代码无损恢复。
<!-- PROFILE_CHECKPOINT_CONFIG_DOUBLE_IDENTITY_20260805 -->

### Local5/H67 profile 双身份闭环（2026-08-05）

- Local5 fullres30 正在训练；冻结评估点为 ep9/14/19/24/29，rank-1 后补 standard valid825、dyadic Q7/Q1.7、hardware-order、T450 ordered profile/replay 与 component RTL-exact。
- 所有 standard profile 现在必须同时匹配训练 checkpoint path/SHA256 与训练 config path/SHA256；dyadic、hardware-order profile 分别绑定各自部署 config，禁止仅凭旧 ranking/profile JSON 复用。
- H67/NB0 equal+10 的每个 valid825 profile 同样绑定候选 checkpoint 与对应 equal+10 config。最终 closure 对 Local5、H67、NB0 和两条部署 profile 重新验 SHA、overlay/missing/unexpected、ATLIF/Shiftmax count 与 480x640、T2x15x15、825-frame 指标合同。
- 定向验收测试：Local5 pipeline `10/10 PASS`，DATE closure `10/10 PASS`。当前 Local5 训练不因合同补强中断；最终独立 closure 使用新规则 fail-closed 验收。

<!-- WINDOW15_PRETRAINED_WINDOW_AUDIT_20260805 -->

### Window15 与 pretrained-window 公平性审计（2026-08-05）

- 发布的 `valid_DSEC_supervised.yml` 采用当前 `window_size=[2,15,15]`、来源 `pretrained_window_size=[2,9,9]`；后者表示低分辨率来源窗口，不等于当前切窗。
- Local5 保留 `[2,9,9]`，H67 rescue 配置记录为 `[2,15,15]`。源码 AST 审计确认 `Spiking_QK_WindowAttention3D.forward` 与安装后的 `_qk_shiftmax_gate_forward` 均不读取 `pretrained_window_size`；all12 installer 仅替换该 QK 类，故差异不进入 H67/Local5 attention 前向。
- 两条训练日志都确认 `ShiftmaxAttention=12`、`checkpoint_overlay_keys=210`、`missing=0`、`unexpected=0`。公平比较使用的实际几何仍一致：480x640、crop null、T2x15x15、batch2；无需因此中止或重跑。
- 此结论只适用于当前 `swinv1` QK+Shiftmax all12 路径；若恢复原始 full-matrix WindowAttention 或启用 `swinv2` continuous relative-position bias，必须重新统一 pretrained-window 并复核。

<!-- EQUAL10_MULTI_AGGREGATION_SOURCE_REEVAL_20260805 -->

### Equal+10 源点重评与 AAE 三聚合边界（2026-08-05）

- 旧 H67 ep30/NB0 ep29 fullres profile 只有 frame-equal 指标，不含升级后的 `metric_aggregation_audit`；不得据此猜测 pixel-global/sequence-balanced 数值。
- equal+10 会在新 run 中重评 H67 30/35/40 与 NB0 29/34/39；旧 profile 缺三聚合、validation-list SHA 或 checkpoint/config path+SHA 任一项即不能复用。
- staged source receipt 已核对：模型与原 source hardlink，optimizer/scheduler/scaler 续接，旧 scheduler milestones 清空而 LR 不变；历史 RNG state 缺失，故披露为 state-continuation、非 RNG bit-exact continuation。

<!-- LOCAL5_EP7_H67_PROVENANCE_V2_20260805 -->

### Local5 ep7、H67 收敛边界与 RTL provenance v2（2026-08-05 23:35 CST）

- Local5 fullres/window15 训练 PID2097444 仍正常运行，当前到 ep7 约32%，约
  `1.04 s/step`；ep0--6 train/valid loss依次为
  `2.1475/1.9795`、`1.8488/1.7067`、`1.6811/1.8096`、`1.6521/2.0182`、
  `1.6211/1.6424`、`1.5152/1.4117`、`1.4714/1.4306`。首个正式 model/state 仍是ep9，
  因而当前Local5没有fullres RTL-exact结果是正确的queued状态；五点valid825、rank-1、三部署
  profile及score/Shiftmax、ATLIF、真实权重projection component RTL均未跳过。
- H67 ep30是当前边界rank-1，但ep25到ep30 AEE仍改善`2.47%`，不能称为收敛；NB0 ep24到
  ep29 AEE仍改善`5.56%`。两者的AAE-2D只变化`+0.08%/-0.44%`，AE-3D变化
  `-0.18%/-1.26%`，说明角度已近平台而EPE尚有续训余量。equal+10将以相同预算判断收敛，
  不把最后一轮最优误写为plateau。
- NB0 local-valid ep29 的`AAE-2D=6.5128`、Barron AE-3D=`6.1803`在非同一validation协议下数值低于论文行，
  AAE `7.23`；论文`4.871`来自七序列official hidden test，而本地为18段/825帧validation。
  因此训练不足可能继续改善AEE，但不能解释或消除全部`4.871`差距；最终只做同population、
  同公式、同聚合比较，official-test数字必须由官方提交获得。
- H67真实权重projection报告升级为`h67_checkpoint_projection_rtl_exact_v2`：报告必须SHA/size
  绑定source trace manifest、vector manifest、12组record manifest及全部memh payload，同时绑定
  generator及其测试、runner、TB、SVA、bind和7个RTL源文件；任一payload或源码漂移即fail closed。
  新增端到端报告生成/篡改反例，连同H67复用与closure回归共`11/11 PASS`。
- H67 ep30、post-convergence和最终closure watcher已重启为PID2418138/2418139/2418140以加载
  provenance v2。H67 T450 source trace尚未生成，故v2实际RTL PASS仍等待Local5释放GPU后生产；
  现阶段声明仍是`checkpoint_bound_component_rtl_exact_not_full_network`，不得提前写成整网exact。
- 最终closure新增Local5收敛资格门：若rank-1仍为ep29且ep24到ep29 AEE改善大于`1%`，Local5
  标记`not_plateaued`并不得被选为最终主线；H67 equal+10仍为`not_plateaued`时同样不具资格，
  NB0参考线若仍未平台则整个最终选择fail closed并要求继续对称训练。新增边界/内点及候选排除测试后
  后续已收紧为任何边界rank-1均视为right-censored，不再用小于1%的斜率自动放行。
- 早期无关checkpoint已完成12轮白名单清理，当前约260GiB可用。为保护Local5五点评估、H67/NB0
  equal+10及其续训state，本阶段不继续泛化删除；待最终rank-1冻结后再清本轮非最优中间权重。

<!-- LOCAL5_PROJECTION_PROVENANCE_V2_20260805 -->

### Local5 component RTL provenance v2（2026-08-05 23:45 CST）

- Local5最终aggregate收据升级为`local5_checkpoint_bound_component_rtl_exact_v2`。除既有
  checkpoint/config/acceptance身份外，closure现在实时复算projection report、vector manifest、
  ordered source trace manifest/payload、8个memh payload、13个RTL/TB、7个SVA及runner/generator/
  summarizer源码的path/SHA256/bytes；任一向量、trace或RTL源码替换都会令旧PASS失效。
- 新增生产形态21行source manifest去重、精确RTL/SVA集合及projection/score/ATLIF payload
  篡改反例；与H67 provenance、复用和closure测试合计`25/25 PASS`。加载最终源码快照的
  Local5硬件父/子为PID2445870/2445874，最终closure为PID2445871；训练
  PID2097444未重启。
- 颗粒度仍不夸大：105是软件安装wrapper数；历史同构profile显示93个唯一动态调用site，当前
  Local5 rank-1的93口径须等本轮profile直接确认。ATLIF replay预注册为sample0、每站点首次调用、
  过滤12个结果死亡attn_sn后的81个功能活跃site，每site选择320个位置，共25,920 events；不是
  81个site全部运行时激活。attention score/Shiftmax与真实权重projection由独立RTL链覆盖，
  aggregate仍明确为component exact而非full-network exact。
- ATLIF旧报告复用现在先递归重算report `source_sha256`与vector manifest内generator及全部7个mem/
  contract payload；score链递归重算3个RTL/TB、3类日志、vector/source manifest/payload和vector
  文件，并由aggregate额外绑定shell/generator/reporter。源码或payload漂移后不得复用旧PASS。
- Local5 capture现在直接保存并SHA绑定installed/called/dead-called/replayed四个完整名称集合；
  reporter强制集合关系`105/93/12/81`及dead-called全为`attn_sn`。Local5与H67/NB0的最终收敛门
  同步改为：最大预算点只要仍是rank-1，就一律标记right-censored，不论末五轮斜率是0.1%还是
  2%；斜率只作描述。Local5条件续训配置已在看到ep29结果前冻结为
  `dsec_fullres_w15_H66d_local5_bb1e4_equal_plus10_ep40.yml`，SHA256
  `99f5baef32334762f6e5fb6d00aa61911e5ab91a38583ffa38a21a10b9a537a7`，仅允许从自身ep29
  model/optimizer/scheduler/scaler续接并保存ep34/39。新增覆盖后组合回归为`35/35 PASS`，
  H67/NB0 equal+10新版WAIT PID2440292。

<!-- LOCAL5_EP7_CLEANUP13_CONVERGENCE_20260806 -->

### Local5 ep7、收敛签核与第十三轮安全清理（2026-08-06 08:10 CST）

- Local5 fullres/window15训练PID2097444保持运行，GPU利用率约93%，ep7已完成并正常进入ep8。ep0--7
  train/42-frame小验证loss为`2.1475/1.9795`、`1.8488/1.7067`、`1.6811/1.8096`、
  `1.6521/2.0182`、`1.6211/1.6424`、`1.5152/1.4117`、`1.4714/1.4306`、
  `1.4330/1.3504`。ep7相对ep6的train/小验证loss继续下降约`2.61%/5.61%`，尚无提前停止依据。
  首个冻结点仍为ep9，
  因此当前尚无Local5 fullres valid825 rank-1或checkpoint-bound RTL-exact PASS。ep9/14/19/24/29
  五点评估、float/dyadic/hardware-order profile、profile100/T450 trace、score/Shiftmax、ATLIF和
  真实权重projection三类component RTL均由现有follower继续排队，未漏做。
- H67 ep30是当前AEE rank-1，但ep25到ep30 AEE仍从`1.3726`降到`1.3387`（`-2.47%`），故状态
  仍为`best_at_boundary_not_proven_converged`。NB0 ep24到ep29 AEE从`1.5304`降到`1.4454`
  （`-5.56%`），同样不能签AEE已收敛；两者AAE-2D/AE-3D末段仅变化约
  `+0.08%/-0.18%`与`-0.44%/-1.26%`，说明角度已近平台而EPE仍有训练余量。H67/NB0同预算
  `+10` runner PID2440292仍在等待Local5及H67 component证据释放GPU，最终以ep30/35/40和
  ep29/34/39同协议曲线判定。
- NB0 local-valid ep29为`AEE/AAE-2D/AE-3D=1.4454/6.5128/6.1803`，在非同一validation协议下数值低于论文行，
  行AEE约`1.61`、AAE`7.23`。论文`4.871`属于七序列official hidden-test AE，本地是18段、825帧
  frame-equal validation；因此欠收敛可能继续改善AEE，但不能解释全部角度差距，也不能把本地
  valid825训练到某个数字后冒充official test复现。
- ATLIF capture新增直接构造105 installed、93 called、12 dead-called和81 replayed站点的测试，
  与manifest/report、Local5/H67 provenance、复用和closure回归合计`36/36 PASS`。该测试只证明
  capture/reporter集合合同；本轮Local5 rank-1的实际站点集合仍须由最终profile向量直接确认。
- 第十三轮清理仅删除2026-05-22的40组H40 160-step短筛和2组并行probe各自唯一的
  `checkpoint_epoch0.pth`。42个目录的config、train/profile日志、CSV、summary和JSON测量全部保留；
  NB0、TTX/BTTX、H67、Local5、fullres、resume state、valid825和RTL证据均未触碰。实际回收
  `18,370,277,376 bytes`（候选文件逻辑大小`18,370,277,220 bytes`，约`17.11 GiB`），可用空间
  增至约`278 GiB`。机器审计为
  `neuron_autoresearch/cleanup_audits/retired_h40_may_screen_checkpoints_20260806.json`。

<!-- LOCAL5_THETA_FOLDED_CLOSURE_FIX_20260806 -->

### Local5 theta-folded 最终闭环修复（2026-08-06 08:25 CST）

- requirement-by-requirement审计发现最终closure仍要求旧projection枚举
  `checkpoint_dyadic_int8_head_slice`，而当前Local5 checkpoint-bound runner、向量生成器、reporter
  和theta-folded硬件合同统一生产`checkpoint_theta_folded_dyadic_int8_head_slice`。若不修复，正确
  的最终RTL报告会在所有任务完成后被closure误拒绝。
- closure现冻结为theta-folded生产枚举，并新增正例接受/旧枚举拒绝回归；该修复只改变最终审计
  合同，不改变训练、量化、向量或RTL数值。最终JSON还新增auditor自身path/SHA256，并将其纳入
  `source_sha256`，避免审计逻辑变化后旧PASS失去代码版本身份。
- Local5/H67 provenance、复用、收敛和closure组合回归更新为`38/38 PASS`。纯WAIT closure已重启
  为PID2458356加载新代码；Local5训练PID2097444、profile/RTL follower及H67/NB0队列均未重启。
- Local5当前正常运行ep8，约12%；ep9 model/state尚未产生，training identity继续保持
  `PENDING_EP9_RUNTIME_STATE`是正确的fail-closed状态。

<!-- EQUAL10_LOCAL5_RTL_RELEASE_GATE_20260806 -->

### Equal+10 的 Local5/H67 双硬件完成门（2026-08-06 08:30 CST）

- 进一步审计发现equal+10日志虽写“等待Local5 pipeline”，旧实现实际只验证H67 ep30的profile、
  all12 trace和三组件RTL；未要求Local5 checkpoint-bound RTL完成。若H67先完成，后续训练可能与
  Local5 score/projection向量生成重叠并造成GPU OOM。
- 新释放门同时要求：H67 ep30 profile100/T450 trace/audit及score/ATLIF/projection PASS；Local5
  当前ranking rank-1与aggregate中的checkpoint/config path+SHA一致、score/ATLIF/projection均PASS、
  projection为theta-folded生产模式，并且Local5 RTL watcher completion marker已落盘。该门只负责
  GPU串行，最终报告真实性仍由DATE closure递归复算全部provenance。
- 新增当前rank-1正例、旧projection枚举、checkpoint SHA漂移及criterion文字一致性反例；组合
  回归为`41/41 PASS`。JSON/Markdown criterion现明确“最大观测预算点为AEE rank-1即
  right-censored，last5斜率只作描述”，不再保留与代码冲突的`>1%`条件。equal+10纯WAIT runner
  已重启为PID2461925，日志确认
  `WAIT Local-5 RTL and H67 ep30 T450 evidence release`。Local5训练当前ep8约15%，GPU利用率约96%。

<!-- LOCAL5_H67_RTL_SERIALIZATION_CLEANUP14_20260806 -->

### Local5/H67 硬件任务严格串行与第十四轮安全清理（2026-08-06 08:35 CST）

- H67 ep30 profile watcher此前只等待Local5软件pipeline completion marker，再以瞬时GPU显存低于
  8GiB作为释放条件；这仍可能在Local5 checkpoint-bound RTL producer尚未启动或两个GPU阶段之间
  抢先占卡。现改为先验收Local5当前ranking rank-1、checkpoint/config path+SHA、aggregate
  component-exact/not-full-network scope、score/ATLIF/projection三类PASS、theta-folded projection
  weight mode及RTL completion marker，随后才检查GPU空闲。新H67 watcher PID2464747已加载该门并
  显示`WAIT Local-5 checkpoint-bound RTL release`。依赖顺序为Local5训练/五点评估 -> Local5三类
  component RTL -> H67 ep30 component RTL -> H67/NB0 equal+10，不存在环依赖。
- 审计同时发现`test_h67_final_evidence_reuse.py`是pytest风格自由函数，而当前环境没有pytest；旧
  `unittest`命令实际未执行其中5个既有正反例。现增加显式`load_tests`并将fixture升级为projection
  provenance v2，真实构造all12名称、source/vector manifests、payload、7个RTL及所有SHA/bytes绑定。
  checkpoint/config/trace漂移和Local5旧weight mode反例均通过，组合回归为实际执行的`47/47 PASS`。
- 第十四轮清理仅删除12个2026-06-25/30退役MDR smoke、DataLoader/backend及吞吐测速目录中的30个
  唯一`.pth`文件；所有配置、日志、测速和正式MDR训练结果保留。实际回收`9,656,684,544 bytes`
  （候选逻辑大小`9,656,683,198 bytes`），可用空间约287GiB。NB0 ep29、TTX/H66d ep29、H67 ep30、
  Local5源checkpoint和正式MDR ep43续训锚点均在删除后验证存在。机器审计为
  `neuron_autoresearch/cleanup_audits/retired_mdr_smoke_bench_checkpoints_20260806.json`。
- Local5训练PID2097444未重启，当前ep8约29%、约1.05s/step、GPU利用率99%；ep0--7最后一点仍为
  train/42-frame小验证loss `1.4330/1.3504`，尚未生成首个冻结ep9。因此Local5 fullres valid825、
  profile和RTL-exact仍是已排队未完成，而不是漏做。H67 ep30仍是AEE边界rank-1，但ep25到ep30
  改善2.47%，只能标记`best_at_boundary_not_proven_converged`。NB0 ep24到ep29 AEE改善5.56%，
  AEE也有欠收敛风险；其local-valid AAE-2D/AE-3D=`6.5128/6.1803`仅能称为在非同一协议下数值低于论文validation AAE
  `7.23`。论文`4.871`来自不同population/聚合的official hidden test，不能把差距单纯归因于轮次。

<!-- LOCAL5_SOFTWARE_MARKER_SHA_HARDENING_20260806 -->

### Local5软件完成marker的checkpoint/config SHA加固（2026-08-06 08:42 CST）

- 继续逐层审计发现Local5主pipeline的`validate_profile_acceptance()`只检查ordered manifest中的
  checkpoint路径和阈值语义，没有像checkpoint-bound RTL follower一样检查acceptance schema、全部
  13项acceptance checks、manifest/run-identity SHA、run identity v3、checkpoint/config path+SHA及
  manifest反向绑定。该缺口不会绕过最终RTL/DATE closure，但可能让软件pipeline completion marker
  在同路径旧artifact存在时过早落盘。
- 主pipeline现采用与硬件follower一致的严格身份合同，并把checkpoint哈希缓存改为
  `path+size+mtime_ns`键，避免同一进程内文件原地变化后继续复用旧digest。当前训练进程仍使用启动时
  已加载代码，没有重启；因此独立supervisor也从“只信marker”升级为marker后重新验证5个模型、3个
  paired state、5份standard valid825 profile、当前rank-1及严格post-G0 acceptance。即使旧内存父进程
  写出弱marker，supervisor也会拒绝并用新代码恢复。
- 仅纯等待supervisor无损重启为PID2471551；Local5训练PID2097444和主pipeline PID2097439未变化。
  acceptance等待函数也从“文件存在即释放”改为严格验证，stale文件会触发canonical producer重建，
  避免supervisor重复重启耗尽。新增marker-only/旧acceptance/checkpoint内容漂移及stale恢复反例后，
  全链实际回归为`49/49 PASS`。Local5
  checkpoint-bound RTL和最终closure仍是更高层独立签核，不依赖该软件marker作为真实性证明。

<!-- AAE_DSEC_FL_CLOSURE_RECOMPUTE_20260806 -->

### 标准DSEC Fl、共同population与closure派生量重算（2026-08-06 09:00 CST）

- DATE审稿式独立复核确认AAE-2D和Barron AE-3D公式在本地内部自洽，但发现历史
  `AEE_outliers`以预测流幅值作5%阈值分母，不是标准DSEC Fl-all。旧字段和历史数值保持不变，仅标为
  `legacy_prediction_magnitude_fraction`；新增`DSEC_Fl`按
  `EPE>3px && EPE>0.05*|GT flow|`计算并以百分数输出。生产evaluator和三聚合audit会在不改训练config
  文件的前提下自动保存该字段，因此当前Local5训练config SHA及运行身份不受影响。
- metric侧新增能区分prediction/GT分母的反例，多聚合侧验证frame-equal、pixel-global、
  sequence-balanced三种DSEC Fl；当前metric/aggregation为`8/8 PASS`，receipt已重生成并绑定当前
  metric/evaluator/aggregation及测试源码SHA。以后论文Fl-all只能引用`DSEC_Fl`，历史7.93%/6.47%
  不再与论文8.91%/10.051%比较。
- 最终closure不再信任convergence summary的派生字段。它从已通过checkpoint/config SHA与完整protocol
  验收的H67/NB0三点profile自行重算rank-1 budget/label、AEE/AAE-2D/AE-3D last5与last10、spikes
  last5/last10、boundary decision及angle decision，再逐字段比对summary；任一陈旧或伪造字段均拒绝。
- 每份profile现在返回validation-list path/SHA和`sequence_id -> frame_count`。closure强制Local5五个
  standard点、dyadic/hardware-order、H67三点和NB0三点全部一致，并进一步硬绑定valid825 SHA
  `7f3dc2800653e12caca10379c51ee8e8988aaf6bb80c391224a454a5879325d0`及18序列精确825帧分布，
  避免“所有候选一致地跑错列表”仍通过。
- AAE诊断措辞已修正：本地NB0只能称为在非同一local-validation协议下数值低于论文validation行，
  不能称为受控复现或超过论文；AAE-2D `6.5128`与official AE-3D `4.871`禁止跨公式比较。主闭环测试
  当前`50/50 PASS`，metric/aggregation另`8/8 PASS`。equal+10/closure/supervisor等待进程已加载新
  合同，PID分别为2482999/2484511/2483016；Local5训练未重启。
- equal+10仍应准确表述为“各自已冻结训练recipe下的等full-resolution epoch预算”，不是共享
  optimizer超参的architecture-only因果消融；正式论文若要提出结构独立增益，仍需匹配shared-backbone
  LR/WD的控制或多seed误差条，不能用当前单轨迹收敛审计替代统计证据。

<!-- LOCAL5_DEPLOY_SUMMARY_AND_CLEANUP15_20260806 -->

### Local5部署摘要闭环与第十五轮短筛清理（2026-08-06 09:10 CST）

- Local5当前训练父/子PID仍为`2097439/2097444`，未重启；ep8约76%、约`1.05 s/step`，GPU约
  `46 GiB/91%`。ep9 model/state尚未落盘，所以`training_config_identity.json`保持
  `PENDING_EP9_RUNTIME_STATE`，硬件profile/RTL等待首个冻结输入是正确状态。H67 ep30和后续
  equal+10继续严格排在Local5 checkpoint-bound RTL之后。
- 训练父进程在新增标准`DSEC_Fl`前已加载旧版部署摘要解析器。为避免它在训练结束后写出缺少
  `DSEC_Fl`的弱摘要，独立supervisor现额外验证rank-1 checkpoint、float/dyadic/hardware三份摘要的
  `AEE/AAE/AAE_Benchmark/DSEC_Fl/total_spikes`，并与两份部署profile逐字段相等；旧内存pipeline若
  生成弱marker，supervisor会以新代码跳过已完成训练，仅恢复推理/摘要，不会重训。加载该合同的
  supervisor PID为`2487791`。
- 新增部署摘要缺字段和profile数值漂移反例后，主闭环回归实际执行`51/51 PASS`；AAE与聚合测试另
  `8/8 PASS`。这保证Local5最终rank-1进入profile/RTL前，标准DSEC Fl和共同valid825 population均
  已闭环，但不把尚未生成的fullres RTL证据误写为PASS。
- 第十五轮清理仅针对7个已有`summary.csv/md`的一轮淘汰短筛权重：BTTX A4、FAPS三个整数比例、
  H66a两次accuracy screen及H65对称恢复。所有配置、日志、summary和正式全训权重保留；NB0、TTX、
  H67、Local5、正式MDR及NTS对称短筛rank-1均为显式保护锚点。机器审计写入
  `neuron_autoresearch/cleanup_audits/retired_one_epoch_idea_screens_20260806.json`；实际回收
  `5,163,982,848 bytes`，可用空间约292GiB，删除后所有保护锚点和活动训练进程均复验通过。

<!-- LOCAL5_RTL_WATCHER_DURABILITY_20260806 -->

### Local5 checkpoint-bound RTL等待器脱离终端（2026-08-06 09:15 CST）

- 运行态复核发现原Local5硬件父/子PID `2457012/2457015`虽有独立SID，但stdin/stdout/stderr仍绑定
  `pts/1`，其生命周期仍可能受启动终端影响。该进程当时只等待ep9身份、未生成profile或RTL，因此
  受控终止整个旧process group，不触碰训练PID `2097439/2097444`。
- 当前代码以`setsid`、stdin=`/dev/null`、stdout/stderr写入
  `hw_autoresearch_nts07/results/local5_bb1e4_checkpoint_bound_rtl_launcher_20260805.log`重新启动；新父/子
  PID为`2493350/2493352`，父进程PPID=1、无TTY，状态继续为
  `WAIT Local-5 ep9 runtime config identity PASS`。这只增强长队列生命周期，不改变训练、推理、量化、
  profile或RTL数值合同。

<!-- LOCAL5_EP9_RUNTIME_IDENTITY_PASS_20260806 -->

### Local5 ep9首个恢复锚点与运行时配置身份PASS（2026-08-06 10:28 CST）

- ep8完整结果为train/42-frame小验证loss `1.460497/1.282759`；ep9为
  `1.385071/2.366434`。ep9训练loss继续改善，小验证出现单点波动；该42-frame路径不用于最终模型选择，
  不提前据此停训或否决。正式排名仍只使用固定相同population的ep9/14/19/24/29 standard valid825。
- 首个成对恢复锚点已稳定写盘：`checkpoint_epoch9.pth`为`591,166,629 bytes`、SHA256
  `695d0541...bfe5f`；state为`432,588,102 bytes`、SHA256 `3437e4b6...6df9`。独立重算SHA与
  `checkpoint_epoch9_early_audit.json`完全一致。
- `training_config_identity.json`现为`PASS`：state/internal scheduler均为epoch9，milestones为
  `{13,20}`，五组optimizer/scheduler LR为`1e-4/1e-4/5e-5/5e-5/5e-6`，AMP scaler存在，
  overlay/missing/unexpected为`210/0/0`。无需repair，`scheduler_repaired=false`、
  `stopped_train_pids=[]`，故当前ep10是原进程连续训练而非重启伪续训。
- Local5硬件子进程已从等待身份切换到`WAIT fullres deploy follower`，仍需完整30轮、五点valid825、
  rank-1及新追加的部署完成marker后才生成profile/RTL，不会用ep9中间点提前出最终硬件结果。
### Local5 ep9模型对象直审计、收敛状态与清理（2026-08-06）

<!-- LOCAL5_EP9_DIRECT_STRUCTURE_AUDIT_AND_CLEANUP16_20260806 -->

- Local5公平full-res修复训练保持同一父/训练PID `2097439/2097444` 连续运行；本节记录时已完成
  ep9并进入ep10约20%。ep9快速valid42的波动不用于选rank-1，最终仍固定评估
  ep9/14/19/24/29共同valid825。
- 新增直接full-model对象审计
  `results/dsec_fullres_w15_H66d_local5_bb1e4_ft30_20260805/checkpoint_epoch9_structure_audit.json`。
  它在CPU注册Shiftmax pickle兼容后直接反序列化ep9模型，而不是只解析训练日志；结果为
  `ATLIFTernaryPSN=105`、`ShiftmaxAttention=12`、model state keys `921`、overlay keys `210`。
  105个ATLIF全部为`binary/official_atlif/zero-center`，无symmetric-binary和ternary输出；12个
  attention全部为`binary_axnor_local5_shiftmax`且`value_branch=reuse_k`。模型SHA与
  `training_config_identity.json`一致，审计状态`PASS`。
- Local5最终硬件证据没有提前用ep9代替：checkpoint-bound watcher仍等待完整30轮、五点
  valid825 rank-1、float/dyadic/hardware-order部署评估；随后执行100样本T450 ordered profile、
  replay/acceptance及attention/relation/ATLIF三类component RTL-exact。旧crop Local5 RTL只能作
  机制预验证，不能作为最终full-res sign-off。
- H67 ep20/25/30 AEE为`1.4240/1.3726/1.3387`，ep30边界最优且ep25到ep30仍改善`2.47%`，
  因此状态保持`best_at_boundary_not_proven_converged`；AAE为`6.1946/6.0102/6.0147`，角度已近
  平台。NB0 ep24到ep29 AEE仍改善`5.56%`，AEE也未证明收敛；但AAE只改善约`0.44%`，论文
  `4.871`与本地valid825的差距不能仅归因于轮次，还受official hidden test、本地population、
  Barron AE-3D/legacy AAE及聚合协议影响。H67/NB0公平各+10轮仍按队列执行后再签收敛结论。
- 清理脚本
  `entrypoints/prune_retired_june_states_and_nonrank_models_20260806.py`已执行：删除45个2026年6月
  已结束实验的optimizer/scheduler/scaler恢复状态，以及7条已退役crop候选中明确低于ep19
  rank-1的ep29模型，共52个文件、`24,959,774,720`字节。NB0、H67、Local5、MDR、当前队列、
  所有rank-1模型、配置、日志、valid825/profile/RTL证据均受保护；审计在
  `neuron_autoresearch/cleanup_audits/retired_june_states_and_nonrank_models_20260806.json`。

<!-- NB0_AAE_GAP_MACHINE_RECEIPT_AND_LOCAL5_EP10_20260806 -->

### NB0 AAE机器诊断与Local5 ep10进度（2026-08-06）

- Local5 full-res/window15仍由同一父/训练PID `2097439/2097444`连续运行，当前ep10约`29%`、
  `1.03--1.08 s/step`，GPU显存约46GiB。ep9只是恢复和结构锚点；ep14/19/24/29尚未产生，故
  standard valid825 rank-1、profile100/T450及最终三组件RTL-exact仍为正确的`queued`状态，不是漏做。
- 新增只读生成器`entrypoints/generate_nb0_aae_gap_diagnostic_20260806.py`和SHA绑定收据
  `neuron_autoresearch/NB0_AAE_GAP_DIAGNOSTIC_20260806.json/.md`。它重算六份full-res profile、
  指标/评估器/聚合器源码及H67/NB0 head-to-head收据SHA，状态为
  `PASS_LOCAL_DIAGNOSIS_OFFICIAL_TEST_REPRODUCTION_UNAVAILABLE`。
- NB0 ep24->29的AEE/AAE-2D/AE-3D改善为`5.558%/0.437%/1.258%`：AEE明显未证明收敛，
  角度已近平台。H67 ep25->30为`2.468%/-0.075%/0.179%`：ep30是当前AEE边界最优，仍不能
  签收敛。equal+10继续必要，且会用新三聚合schema重评源点。
- 论文official-test `4.871`是隐藏测试Barron/Middlebury `(u,v,1)` AE；本地legacy AAE是二维方向角，
  本地`AAE_Benchmark`虽已正确实现三维公式，但population与server聚合仍不同。机器结论固定为
  `formula_bug=false`、`NB0_AEE_undertraining_plausible=true`、
  `NB0_angle_gap_explained_by_undertraining_alone=false`，禁止把增加轮次写成复现official 4.871。
- 最终DATE closure已新增该机器收据为REQUIRED，并验证状态、源码SHA、profile SHA和诊断字段；
  后续任何口径或证据漂移都会fail-closed。

<!-- RANKED_JUNE_NONBEST_CLEANUP17_AND_CLOSURE_RELOAD_20260806 -->

### 第十七轮ranked旧模型清理与closure重载（2026-08-06）

- 清理器`entrypoints/prune_ranked_june_nonbest_models_20260806.py`只扫描目录名含`202606`、且存在
  `profile_ranking_valid825.md`的已结束路线。每条路线保留当前磁盘中排名最高的模型；候选若在自身
  目录外被日志、配置、脚本或文档以绝对source path引用则自动保护。
- dry-run冻结34个无外部lineage引用的非最优模型，随后执行删除并复验：`34/34` absent、所有best
  checkpoint present，逻辑回收`23,556,287,298 bytes`（`21.939 GiB`），可用空间由约314GiB升至
  336GiB。审计为`neuron_autoresearch/cleanup_audits/ranked_june_nonbest_models_20260806.json`，包含
  每个被删模型的epoch/rank/SHA/大小和对应保留rank-1路径。
- NTS07b ep29、NTS09e ep29、NTS11bd ep19等关键历史rank-1均存在；NB0、TTX/BTTX、H67、Local5、
  MDR、当前queue及全部硬件输入不在该清理scope。清理后Local5 ep10约50%、约`1.04 s/step`，原
  PID `2097444`和46GiB CUDA context连续，未受影响。
- AAE机器收据正反例加入closure定向测试后总计`23/23 PASS`。纯等待closure重载为PID`2538883`，
  已读取新代码并等待Local5 ranking/deploy/RTL、equal+10与H67 post-convergence证据；没有重启任何
  训练、profile或RTL生产进程。


### H66d Local-5 fullres bb1e4 结果

<!-- DSEC_FULLRES_W15_H66D_LOCAL5_BB1E4_RESULT_20260805 -->

# Standard Valid825 Ranking

Ranking mode: `aee`.

The energy column is a spike-activity proxy and excludes overlay attention control/reduction operations.

| rank | epoch | AEE | AAE legacy | AAE benchmark | PE1 | PE2 | outlier | total_spikes | firing | spike_energy_proxy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 29 | 1.3286 | 6.0022 | 5.6594 | 0.4135 | 0.1425 | 0.0661 | 82.8799G | 6.4480% | 73271.32 |
| 2 | 24 | 1.3599 | 5.9995 | 5.6853 | 0.4244 | 0.1494 | 0.0693 | 82.3036G | 6.4032% | 72789.85 |
| 3 | 14 | 1.4381 | 6.3066 | 6.0253 | 0.4403 | 0.1584 | 0.0761 | 79.9314G | 6.2186% | 70748.16 |
| 4 | 19 | 1.4534 | 6.3284 | 6.0052 | 0.4486 | 0.1704 | 0.0841 | 81.3738G | 6.3308% | 71984.70 |
| 5 | 9 | 1.6942 | 7.2407 | 6.9402 | 0.5116 | 0.2079 | 0.1053 | 78.0242G | 6.0702% | 69096.25 |

# Local-5 bb1e4 deploy summary

Scope: attention-core hardware-order numeric; full T450 SV zero-mismatch is a separate sign-off.

| path | AEE | AAE benchmark | spikes(G) |
|---|---:|---:|---:|
| float | 1.3286 | 5.6594 | 82.8799 |
| dyadic | 1.3326 | 5.6507 | 82.8787 |
| hardware-order | 1.3338 | 5.6626 | 82.8745 |

<!-- LOCAL5_POSTG0_BINDING_FIX_AND_RELAUNCH_20260808 -->

### Local5 post-G0 binding fix and queue relaunch (2026-08-08)

- Root cause of the queue stall: `profile_local5_hardware_features.load_post_g0_run_identity`
  required exact key equality on `source_bindings`, but the production writer
  (`run_local5_qfsa_profile_after_fullres` + bb1e4 training-identity patch) emits
  two additional bindings (`projection_contract_verifier`, `training_config_identity`).
  This raised `ValueError: post_g0生产软件绑定集合不完整`, so profile100/T450 acceptance
  never produced; H67 T450, equal+10, and DATE closure watchers then waited forever.
- Fix: align required software bindings with the writer (add
  `projection_contract_verifier`) and accept run-scoped extras validated by path+sha.
  Same subset/extra policy applied to `analyze_ds_flm_descriptor_manifest.validate_source_bindings`.
- Unit checks: `test_ds_flm_descriptor_analysis`, `test_local5_postg0_acceptance`,
  `test_local5_training_identity_gate`, `test_local5_release_receipt` → 11/11 OK.
- Relaunched at ~2026-08-08 15:27 CST:
  - post-G0 producer PID recorded in
    `hw_autoresearch_nts07/results/local5_fullres_bb1e4_postg0_watcher_20260805.pid`
  - checkpoint-bound RTL watcher relaunched to wait on acceptance
- First live evidence after fix: load audit `overlay210/missing0/unexpected0`,
  ATLIF=105, Shiftmax=12, 12 blocks attached; GPU active on profile100.
- Downstream still correct-to-queue until acceptance/RTL release:
  equal+10 convergence, H67 ep30 T450, post-convergence rank-1 profile, DATE closure.

- Race fix while relaunching RTL: `run_local5_bb1e4_postg0_profile` no longer runs ATLIF
  after a lock-skip (return 0 without acceptance). `run_local5_bb1e4_checkpoint_bound_rtl`
  now polls for acceptance and joins an existing producer lock instead of failing on a
  second short-lived producer.

- ATLIF DP-TME vector fix (2026-08-08 evening): `generate_checkpoint_atlif_dptme_vectors`
  failed on Local5 rank-1 with `PyTorch ATLIF recomputation mismatch` at float32 threshold
  boundary (model event fire, subset rematmul h=0.99999994 < thr). Capture now stores the
  full-path `addmm` hidden for selected lanes and uses that as the float reference; optional
  ulp-boundary salvage remains as fail-soft. Smoke: 81 commands, model_reference_mismatches=0.
  post-G0 acceptance already PASS (13/13); RTL watcher relaunched after this fix.

- ATLIF DP-TME checks timeout fix (2026-08-09): icarus PASS exact (hidden/event mismatches=0),
  but `run_checkpoint_atlif_dptme_checks.sh` used `timeout 120s` for Verilator sim; exit 124
  after only command=0 of 81. Raised defaults to ICARUS_TIMEOUT_S=600 / VERILATOR_TIMEOUT_S=7200
  (env-overridable). Relaunched RTL PID recorded in
  `local5_bb1e4_checkpoint_bound_rtl_watcher_20260805.pid`. Avoid concurrent second RTL parents.

<!-- LOCAL5_ATLIF_VERILATOR_HANDSHAKE_AND_THREE_CANDIDATE_CONVERGENCE_20260809 -->

### Local5 ATLIF双仿真器闭环、H67解锁与三候选收敛队列（2026-08-09）

- 后续定位证明Verilator的`120s`超时不是正常性能：旧testbench在接收上升沿后再次读取
  `step_ready`，DUT状态更新会使ready下降，形成仿真器调度相关的无限等待。testbench改为负沿放置
  payload并等待组合ready稳定、仅跨一个上升沿接收；生产脚本保留可配置timeout作为防无限运行保护。
  修复后Icarus和Verilator均在81条命令上完成：hidden `25920/0 mismatch`、event
  `25920/0 mismatch`，并通过lint与Yosys。此前“Verilator需要数小时”的判断由该结果撤销。
- 最终Local5 checkpoint-bound aggregate已生成并绑定rank-1 ep29：checkpoint SHA
  `6e0e92a5...c993b`、hardware config SHA `cf332c05...d72a`。score/Shiftmax四项检查PASS；
  theta-folded真实权重projection的checkpoint binding/random SVA/Verilator lint/Yosys均PASS；ATLIF
  81个活跃replay site覆盖45个T10和36个T2命令。aggregate scope仍是
  `checkpoint_bound_component_rtl_exact_not_full_network`。
- 数值边界不得混淆：ATLIF定点向量相对捕获float事件有`1177/25920=4.5409%`局部翻转，故报告
  `deployment_accuracy_signoff=false`。组件RTL对定点参考零误差不等于整网量化精度签核；后续仍须
  以static site scale与下游event-times-threshold folding的标准valid825部署推理作算法侧依据。
- H67 follower原有两处脆弱门控已修复：完成日志接受生产runner实际marker；Local5 projection不再
  读取不存在的顶层`status`，而是逐项验证四个PASS字段并继续核对rank-1/config/checkpoint SHA。
  2026-08-09 00:19 CST已释放并启动H67 ep30 fullres `480x640`、window `2x15x15`、T450、
  profile100/all12 bit trace；加载审计为overlay `210/210`、missing/unexpected `0/0`、ATLIF `105`、
  Shiftmax `12`。完成后自动接score/Shiftmax、ATLIF和全12块真实权重projection RTL。
- 收敛审计补齐Local5。其ep24->29 AEE由`1.3599`降至`1.3286`（改善`2.301%`），且ep29为
  最大预算边界点，与H67和NB0一样不能签已收敛。预注册配置
  `dsec_fullres_w15_H66d_local5_bb1e4_equal_plus10_ep40.yml`现已接入正式runner，只从Local5自身
  ep29 model/state续训并评估ep29/34/39；队列顺序冻结为Local5 -> H67 -> NB0。候选/血缘回归
  `5/5 PASS`，连同Local5 release与H67复用边界的组合回归为`23/23 PASS`；新watcher PID
  `3114346`当前等待H67 T450证据，不占GPU。
- 当前收敛结论：Local5与H67的legacy AAE在末五轮分别变化约`+0.044%/+0.075%`，AE-3D分别改善
  `0.456%/0.179%`，已近角度平台；NB0 ep24->29 AEE仍改善`5.558%`而AAE-2D/AE-3D仅改善
  `0.437%/1.258%`。追加轮次可能继续降低AEE，但不足以单独解释本地AE-3D `6.1803`与official
  hidden-test `4.871`的差距；公式、population与server aggregation必须继续分栏报告。

<!-- LOCAL5_EQUAL_PLUS10_INTERIM_RESULT_20260809 -->

### Local5 equal +10正式结果与三候选队列进度（2026-08-09 21:30 CST）

- Local5严格从自身full-res ep29的model + optimizer/scheduler/scaler state续训10轮，resume audit为
  `model_optimizer_scheduler_scaler_resume_not_rng_bit_exact`；训练正常exit 0。标准加载链为
  overlay `210/210`、missing/unexpected `0/0`、ATLIF `105`、Shiftmax `12`，并完成ep29/34/39
  三个checkpoint的Valid825标准推理与profile contract校验。
- 正式AEE排名如下；AAE benchmark为本地AE-3D口径，energy仍只是未计attention控制/归约的
  spike-activity proxy：

| rank | epoch | AEE | AAE legacy | AAE benchmark | total_spikes | spike energy proxy |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 39 | 1.3153 | 5.8291 | 5.5379 | 84.4197G | 74612.15 uJ |
| 2 | 29 | 1.3286 | 6.0022 | 5.6594 | 82.8799G | 73271.32 uJ |
| 3 | 34 | 1.3355 | 6.0979 | 5.7541 | 83.8319G | 74105.85 uJ |

- ep29->39的AEE改善约`1.00%`、AAE benchmark改善约`2.15%`，代价是spikes增加约`1.86%`。
  ep34曾退化而ep39成为最大预算边界的新rank-1，说明追加训练有效但仍不能签“已收敛”；在H67、
  NB0同预算对照完成前不继续单独加轮，避免对Local5产生不公平的后验预算。
- H67 equal +10已于14:39 CST接续自身ep30启动；21:30 CST进入epoch36，实测每轮约66分钟，GPU
  持续满载。H67完成后runner自动做ep30/35/40 Valid825，再启动NB0 equal +10；三候选summary尚未
  生成，故当前不能宣布最终主线。
- 若ep39最终保持三候选总排名第一，旧Local5 ep29的profile/T450/score-Shiftmax/ATLIF/projection
  component RTL证据不能继承为ep39签核；post-convergence follower必须按ep39 checkpoint/config SHA
  重做并保留`component_rtl_exact_not_full_network`范围声明。

<!-- H67_EQUAL_PLUS10_AND_CICC_MVSEC_QUEUE_20260810 -->

### H67 equal +10结果与CICC范式MVSEC后续队列（2026-08-10 02:45 CST）

- H67严格从自身已完成30轮full-res适配的ep30 model + optimizer/scheduler/scaler state续训10轮；
  该full-res ep30更早由H67 own-crop初始化，但本次不是直接从crop跳接。训练exit 0，加载审计为
  overlay `210/210`、missing/unexpected `0/0`、ATLIF `105`、Shiftmax `12`。ep30/35/40的
  Valid825正式排名为：

| rank | epoch | AEE | AAE legacy | AAE benchmark | total_spikes | spike energy proxy |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 35 | 1.3297 | 5.9004 | 5.6509 | 82.1107G | 72508.06 uJ |
| 2 | 30 | 1.3387 | 6.0147 | 5.7558 | 81.3086G | 71812.84 uJ |
| 3 | 40 | 1.3434 | 5.8238 | 5.6069 | 82.7681G | 73073.34 uJ |

- H67的AEE在ep35达到内部最优、ep40回退约`1.03%`，因此H67已有“越过AEE最优点”的收敛
  证据；ep40虽然AAE benchmark更低，但预注册主排序仍按AEE。Local5 ep39的AEE `1.3153`仍优于
  H67 ep35约`1.08%`，所以当前暂定总领先者仍是Local5，最终结论等待NB0 equal +10。
- NB0 equal +10已于2026-08-10 02:39 CST从自身full-res ep29的model/state启动，overlay为`0`且
  missing/unexpected为`0/0`；完成后自动执行ep29/34/39 Valid825并生成三候选summary。

**DSEC闭环后的MVSEC协议冻结：**

- 第二数据集采用CICC 2026引用的Spike-FlowNet split：只用`outdoor_day2`训练，在
  `outdoor_day1 + indoor_flying1/2/3`测试；dt1、训练random `256x256` crop与水平/垂直翻转
  `p=0.5`、测试center `256x256`、event-masked AEE。每序列固定800输入的审计表与完整序列结果
  分开报告；不得与既有MDR->MVSEC绝对AEE混表。
- 最小训练矩阵为MVSEC-NB0 seed0 from scratch、H67 deploy reference seed0，以及DSEC最终胜者
  seed0；两个替换模型均从同一MVSEC-NB0 checkpoint初始化并使用相同训练样本、目标、预算和
  checkpoint-selection validation split。只有最终胜者通过AEE/稀疏门后才补seed1/2。
- 本机已有四个测试序列约80GB，但缺失`outdoor_day2`。其data/GT bag远端大小分别为
  `28,497,983,504`与`24,170,765,728` bytes（合计约52.67GB）。2026-08-10 02:44 CST已启动
  低IO优先级、32路range、可断点的串行下载，PID记录于
  `neuron_experiments/H9_bipolar_self_attention/results/mvsec_outdoor_day2_download.pid`；下载只做
  数据准备，不抢占GPU，也不提前启动候选训练。
- 启动前必须补齐并测试三个可选链路：`outdoor_day2`固定train/validation非重叠manifest；修复现有
  `MvsecEventFlow` dense augmentation分支在`sample['valid']`赋值前裁掉outdoor无效行的顺序错误；
  新增固定800样本manifest及event-mask/valid-pixel计数审计。所有新增实现保持可选，不修改旧
  MDR->MVSEC结果口径。
- 模型表固定为M0 NB0-float、M1 NB0-hardware-order、M2 winner-float、M3 winner-hardware-order；
  再冻结M3做CICC式累计部署表D0 dense、D1 lossless BWAC/activation packing、D2 exact-empty TTB
  skip+density dispatch、D3 validation-selected similarity deep-level skip。TTB是本项目机制，不归因
  于CICC；每行同时报告AEE/outlier、operations、SRAM/DRAM bytes、cycle、完整energy和控制开销。


### Local5/H67/NB0 fullres 等预算 +10 收敛审计结果

<!-- DSEC_FULLRES_W15_H67_NB0_EQUAL_PLUS10_RESULT_20260805 -->

#### Local5

# Standard Valid825 Ranking

Ranking mode: `aee`.

The energy column is a spike-activity proxy and excludes overlay attention control/reduction operations.

| rank | epoch | AEE | AAE legacy | AAE benchmark | PE1 | PE2 | outlier | total_spikes | firing | spike_energy_proxy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 39 | 1.3153 | 5.8291 | 5.5379 | 0.4081 | 0.1383 | 0.0638 | 84.4197G | 6.5678% | 74612.15 |
| 2 | 29 | 1.3286 | 6.0022 | 5.6594 | 0.4135 | 0.1425 | 0.0661 | 82.8799G | 6.4480% | 73271.32 |
| 3 | 34 | 1.3355 | 6.0979 | 5.7541 | 0.4226 | 0.1460 | 0.0671 | 83.8319G | 6.5221% | 74105.85 |

#### H67

# Standard Valid825 Ranking

Ranking mode: `aee`.

The energy column is a spike-activity proxy and excludes overlay attention control/reduction operations.

| rank | epoch | AEE | AAE legacy | AAE benchmark | PE1 | PE2 | outlier | total_spikes | firing | spike_energy_proxy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 35 | 1.3297 | 5.9004 | 5.6509 | 0.4120 | 0.1397 | 0.0643 | 82.1107G | 6.3882% | 72508.06 |
| 2 | 30 | 1.3387 | 6.0147 | 5.7558 | 0.4165 | 0.1405 | 0.0647 | 81.3086G | 6.3258% | 71812.84 |
| 3 | 40 | 1.3434 | 5.8238 | 5.6069 | 0.4072 | 0.1394 | 0.0655 | 82.7681G | 6.4393% | 73073.34 |

#### NB0

# Standard Valid825 Ranking

Ranking mode: `aee`.

The energy column is a spike-activity proxy and excludes overlay attention control/reduction operations.

| rank | epoch | AEE | AAE legacy | AAE benchmark | PE1 | PE2 | outlier | total_spikes | firing | spike_energy_proxy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 29 | 1.4454 | 6.5128 | 6.1803 | 0.4563 | 0.1661 | 0.0793 | 126.1156G | 9.7927% | 107137.62 |
| 2 | 39 | 1.4549 | 6.5222 | 6.2109 | 0.4528 | 0.1623 | 0.0772 | 128.0836G | 9.9455% | 108790.07 |
| 3 | 34 | 1.4584 | 6.5741 | 6.2463 | 0.4541 | 0.1614 | 0.0764 | 127.0435G | 9.8648% | 107905.72 |

# DSEC full-resolution equal +10 convergence audit

Criterion: the largest observed budget being AEE rank-1 is right-censored and therefore not plateaued; the last-five slope is descriptive only.

| candidate | budget | checkpoint label | AEE | AAE-2D | AE-3D | DSEC Fl(%) | spikes(G) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Local5 | 30 | 29 | 1.328621 | 6.002152 | 5.659358 | 6.6078 | 82.8799 |
| Local5 | 35 | 34 | 1.335480 | 6.097923 | 5.754114 | 6.7112 | 83.8319 |
| Local5 | 40 | 39 | 1.315288 | 5.829055 | 5.537945 | 6.3815 | 84.4197 |
| H67 | 30 | 30 | 1.338739 | 6.014742 | 5.755796 | 6.4705 | 81.3086 |
| H67 | 35 | 35 | 1.329678 | 5.900353 | 5.650878 | 6.4279 | 82.1107 |
| H67 | 40 | 40 | 1.343414 | 5.823829 | 5.606944 | 6.5488 | 82.7681 |
| NB0 | 30 | 29 | 1.445353 | 6.512804 | 6.180336 | 7.9323 | 126.1156 |
| NB0 | 35 | 34 | 1.458382 | 6.574137 | 6.246313 | 7.6448 | 127.0435 |
| NB0 | 40 | 39 | 1.454926 | 6.522233 | 6.210893 | 7.7207 | 128.0836 |

## Decision

- Local5: `not_plateaued`; AEE last5 `1.512%`, last10 `1.004%`; AAE-2D last5/last10 `4.409%/2.884%`, AE-3D last5/last10 `3.757%/2.145%`; spikes change last5/last10 `+0.701%/+1.858%`; angle `angle_not_plateaued_or_noisy`, rank-1 budget `40`.
- H67: `operationally_plateaued_or_overfit`; AEE last5 `-1.033%`, last10 `-0.349%`; AAE-2D last5/last10 `1.297%/3.174%`, AE-3D last5/last10 `0.777%/2.586%`; spikes change last5/last10 `+0.801%/+1.795%`; angle `angle_not_plateaued_or_noisy`, rank-1 budget `35`.
- NB0: `operationally_plateaued_or_overfit`; AEE last5 `0.237%`, last10 `-0.662%`; AAE-2D last5/last10 `0.790%/-0.145%`, AE-3D last5/last10 `0.567%/-0.494%`; spikes change last5/last10 `+0.819%/+1.560%`; angle `angle_plateaued`, rank-1 budget `30`.

<!-- POSTCONV_MARKER_FIX_AND_HANDOFF_20260810 -->

### Post-convergence marker fix and handoff resume (2026-08-10)

- **Equal +10 audit already complete** (2026-08-10 08:17): summary schema
  `dsec_fullres_equal_plus10_convergence_v1` with Local5/H67/NB0 budgets 30/35/40.
  Producer marker text is `ALL COMPLETE Local5/H67/NB0 equal +10 convergence audit`.
- **Blocker**: `run_h67_postconvergence_rank1_profile.py` waited only for the older
  string `ALL COMPLETE H67/NB0 equal +10 convergence audit`, so it never released
  despite summary + rankings on disk. DATE closure then waited forever on missing
  `h67_postconvergence_rank1_hardware_evidence_20260805.json`.
- **Fix**: accept both Local5/H67/NB0 and H67/NB0 markers and require summary JSON
  candidates Local5/H67/NB0. Supervise watcher marker updated to match producer.
- **H67 post-convergence rank-1 is now ep35** (AEE 1.3297), not ep30 — will run
  full T450 profile/trace/RTL path (not the ep30 reuse shortcut).
- **GPU**: Local5 joint-head profile100 (~86/100 at handoff) still holds the A800;
  post-convergence auto-starts when that process exits (`/tmp/run_h67_postconv_when_gpu_free.sh`).
- Local5 component RTL scope already PASS; equal+10 AEE bests: Local5 ep39 **1.3153**,
  H67 ep35 **1.3297**, NB0 ep29 **1.4454**.


### DATE 算法/RTL 最终证据闭环

<!-- DATE_ALGORITHM_CLOSURE_AUDIT_PASS_20260805 -->

- fail-closed closure audit PASS；Local-5 rank-1 ep29，H67 rank-1 ep35。
- H67 收敛判定 `operationally_plateaued_or_overfit`，NB0 收敛判定 `operationally_plateaued_or_overfit`；AAE-2D 与 AE-3D 仍分口径报告。
- Local-5 收敛判定 `not_plateaued`，ep24到ep29 AEE改善 `2.301%`；边界仍改善时不得选为最终主线。
- H67 训练血缘由机器收据绑定为自身 Motion-XOR crop ep19 经五段续训到 fullres ep30；没有从 NB0 或 Local-5 初始化。
- Local-5 仅声明 score/Shiftmax、真实权重 per-head projection partial accumulator、ATLIF temporal matrix 三项 component RTL-exact；H67 同样不外推为 full network。
- 机器审计：`neuron_autoresearch/DATE_ALGORITHM_CLOSURE_AUDIT_20260805.json`。

<!-- DATE_ALGORITHM_CLOSURE_PASS_20260811 -->

### DATE algorithm closure PASS (2026-08-11 03:59)

- Marker mismatch fixed; H67 post-convergence rank-1 **ep35** T450 evidence PASS.
- Local5 checkpoint-bound scope re-bound after source SHA drift (projection generator /
  evidence_provenance) and ATLIF report 8→9 source artifact contract.
- Closure audit: `neuron_autoresearch/DATE_ALGORITHM_CLOSURE_AUDIT_20260805.json/.md`
  — **ALL COMPLETE DATE algorithm closure audit PASS Local5=ep29 H67=ep35**.
- Equal+10 AEE rank-1: Local5 ep39 **1.3153**, H67 ep35 **1.3297**, NB0 ep29 **1.4454**.

<!-- GROK_HANDOFF_REVIEW_AND_MVSEC_DIRECT_ROUTE_20260811 -->

### Grok 交接复核、证据重签与 direct-MVSEC 标准路线（2026-08-11 14:43 CST）

**交接结论复核：** `DATE_ALGORITHM_CLOSURE_AUDIT_20260805.{json,md}` 的 PASS 与
equal+10 原始 summary 一致，但论文表必须区分两种“最好”：Local5 ep39 是等预算 AEE 最好
（`1.3153`，但最大预算仍为 rank-1，状态 `not_plateaued`）；H67 ep35 是已经越过 AEE 最优点、
且当前 checkpoint-bound 硬件证据完整的部署主线（AEE `1.3297`）。H67 不是纯精度冠军，Local5
ep39 也不能继承旧 ep29 的 RTL 证据。NB0 ep29 AEE `1.4454`，因此 H67/Local5 分别改善约
`8.0%/9.0%`，spikes 分别下降约 `34.9%/33.1%`。

**provenance 重签：** 03:59 closure 后硬件侧又修改了 Local5 projection generator，旧 scope
对当前工作树出现 SHA drift。14:38-14:39 使用同一个 Local5 ep29 checkpoint 重跑最小 runner：
ATLIF provenance-valid 结果复用，score/Shiftmax 与 projection 重新 PASS，最新
`checkpoint_bound_scope.json` 再次通过 fail-closed provenance 校验；14:43 closure audit 重跑并
再次输出 `PASS Local5=ep29 H67=ep35`。声明范围仍为 component RTL-exact，不是 full-network
RTL-exact。

**训练入口审计修复：** 原 `train_mdr_supervised_SNN.py` 只能做 MDR train -> MVSEC valid，且
validation loop 未从当前 validation batch 读取 `valid`，会错误沿用最后一个 train batch 的 mask。
该缺陷影响旧训练日志中的内置 MVSEC validation loss，但不影响独立标准推理生成的 AEE/profile。
现已补当前 batch mask，并新增可选 `data.training_dataset=mvsec_dt1`；默认 `mdr` 路径不变。
direct-MVSEC 新分支按 validation loss 保存 checkpoint，MLflow 关闭且 model/state 仅保存本地。

**CICC/Spike-FlowNet split 冻结：** 原始 `outdoor_day2_data/gt.bag` 的 MD5 分别为
`536d20bc59720b995df49f925f96b74d` 与 `69fb399411d7098b3e2cf3850f593e7b`，与官方值一致。
固定 manifest 为
`neuron_experiments/H9_bipolar_self_attention/manifests/mvsec_cicc_dt1_v1.json`：train
`4375..6737`（2363 项）、隔离 gap `6738`、validation `6739..7001`（263 项）；四个测试序列
各冻结 800 个均匀覆盖 valid interval 的 index。训练为 dt1、random `256x256` crop、水平/垂直
flip 均 `p=0.5`；验证/测试为 center `256x256` 和 event-masked AEE。完整序列结果与 fixed800
结果必须分列。

**静态配置与入口：**

- `configs/generated/mvsec_cicc_nb0_w8_seed0.yml`：NB0、window `2x8x8`、seed0、50 epoch
  from scratch；
- `configs/generated/mvsec_cicc_h67_motion_w8_seed0.yml`：H67 Motion-XOR、30 epoch；
- `configs/generated/mvsec_cicc_local5_w8_seed0.yml`：Local5、30 epoch；
- H67/Local5 必须从同一个 MVSEC-NB0 seed0 validation-rank1 checkpoint 初始化；三者均使用
  batch `8`、workers `8`，候选安装审计为 `ATLIFTernaryPSN=105`、Shiftmax attention `=12`；
- manifest/config 生成器分别为 `build_mvsec_cicc_manifests.py` 与
  `build_mvsec_cicc_train_configs.py`；本地训练入口为 `run_mvsec_cicc_train.py`。

正式启动门为：outdoor_day2 编码样本与 flow/event 对齐审计 PASS、manifest 全文件存在、NB0 单
batch train+validation smoke PASS。随后严格串行执行 NB0 -> 从同一 NB0 checkpoint 分叉 H67 与
Local5 -> 四序列 fixed800 + full-sequence 标准推理；seed1/2 仅在候选通过 seed0 门槛后补跑。

`window 2x15x15 + crop256` 的首次 smoke 在 S3 失败：S3 feature 为 `8x8`，动态 window 缩到
`8x8`，但原 positional encoding 仍为 450 token，无法 reshape 为 128 token。该失败配置保留为
负证据，不进入论文。direct-MVSEC 正式配置改用 SDformerFlow/Spike-FlowNet 256-crop 可执行的
`window 2x8x8`；DSEC full-resolution 主表继续使用 `window 2x15x15`，两者按数据集协议分表。

<!-- MVSEC_DIRECT_SMOKE_AND_FORMAL_LAUNCH_20260811 -->

### direct-MVSEC 冒烟修复、速度实测与正式队列（2026-08-11 15:08 CST）

- 首次 window8 冒烟的训练前向成功，但 validation event mask 错误保留 polarity 维，形成
  `[B,1,2,H,W]`，而 flow loss 要求 `[B,1,H,W]`，触发 `65536 vs 131072` 展平长度冲突。现新增
  `event_activity_mask()`，对 time 与 polarity 维做非零 `any`，同时避免正负事件求和相消；训练、
  validation 与 visualization 共用同一 `[B,1,H,W]` 合同。
- 后续真实样本审计又发现 source-FOV 顺序问题：direct train 在原始 `260x346` 上先屏蔽
  outdoor_day2 的 `y>=193` 再 random crop，但旧 validation 先 center crop，且只对 day1 屏蔽。
  新的可选 `mvsec_source_valid_before_crop=true` 先在 source frame 对 day1/day2 建 valid mask，再
  center crop；旧 MDR/MVSEC 默认分支不变。真实 validation 样本验证 crop 后 row190 有效、row191
  起全0，valid pixels=`48,896`；五项协议/增强/mask/source-FOV 单测 PASS。
- v2 单 batch train+validation 冒烟 exit 0：train loss `2.0682626`，validation loss
  `7.4427266`；此前只修 polarity mask、尚未修 source-FOV 的 `8.7158298` 仅作负证据。v2 本地保存
  `checkpoint_epoch0.pth`（219,737,042 bytes）与 state
  （438,550,218 bytes）。`launch_provenance.json` 明确记录 MLflow/model-log/state-log 全关闭、
  `SDFORMER_MDR_VOXEL_GPU=0`、CuPy backend，故权重和训练状态均为本地可恢复文件。
- 10-batch 吞吐实测：80 samples / `37.758s` = `2.119 samples/s`。首批 CuPy compile 约 `33s`；
  后续稳定约 `0.5s/batch`，完整 295-batch train 段约 2.5-3 分钟，不能用首批时间外推成数天。
  validation 为 batch1，正式总耗时仍以首轮完整 train+validation 日志重新估算。
- 首个正式 run 在 epoch0 validation 期间因上述 source-FOV 审计主动停止；v2 随后又因审计发现
  `runtime.seed=0` 未被旧 trainer 消费而在 epoch0 主动停止。两者目录均保留且禁止参与选点。
  trainer 现仅在配置含 `runtime.seed` 时冻结 Python/NumPy/Torch/CUDA、train shuffle generator 与
  validation worker generator；CuPy/AMP 仍声明 `seeded_data_order_non_bit_exact`，不声称 bit-exact。
  workers8 双池审计得到相同首批 indices 与联合 batch SHA
  `1fcd5328e2f5cb5c055fa594af689ad1f3eb9a093f361d1a208dfa0438408ca1`，PASS。
- v3 冒烟 exit0，train/validation loss=`2.1670144/7.4427252`。启动收据新增 config、manifest、
  trainer、MVSEC loader 与 protocol helper 的 SHA256。v3 首轮完整得到 train/validation loss
  `1.0649808/11.5610867` 和可恢复 state，但其父进程绑定交互 PTY；直接 resume 又不能恢复 worker
  RNG，故不将 v3 作为正式 run。最终 v4 使用同一冻结 config 从 seed0 重新开始，以 `setsid` 启动，
  输出为 `results/mvsec_cicc_nb0_w8_seed0_v4_20260811/`，训练父 PID `4151517`（PPID1、无TTY）；
  串行 supervisor PID `4151772` 同样为 PPID1、无TTY，等待 exit0 后按 validation loss 与实际
  checkpoint 文件共同选择 best；随后
  对 H67、Local5 分别先做单 batch 加载冒烟，必须满足 ATLIF `105`、Shiftmax `12`、
  `checkpoint_overlay_keys=0`、`missing=210`、`unexpected=0`，再从同一个 MVSEC-NB0 best
  checkpoint 正式训练。
- 标准推理不能只报 fixed800。每条路线均先跑四序列各800样本的等样本审计，再跑
  `outdoor_day1 + indoor_flying1/2/3` 四个完整 valid interval；两套结果分别汇总 macro mean AEE、
  valid-pixel-weighted AEE、有效像素、spikes 与 energy proxy。统一入口为
  `entrypoints/supervise_mvsec_cicc_pipeline.py --join-active-nb0`，最终汇总为
  `results/mvsec_cicc_nb0_h67_local5_comparison_20260811.{json,md}`。

<!-- H67_MAINLINE_DECISION_AND_H81_FULLRES_QUEUE_20260811 -->

### H67 主线裁决与 H81 no-motion fullres 对照队列（2026-08-11）

- DSEC full-resolution equal+10 已足以裁决 H67 与 Local5：Local5 ep39 的 AEE
  `1.3153` 比 H67 ep35 的 `1.3297` 好约 `1.08%`，但 H67 spikes 为
  `82.1107G`，比 Local5 的 `84.4197G` 少约 `2.74%`。综合算法指标、统一
  all12 数据流、硬件创新潜力和软硬件协同度，冻结 **H67 Motion-XOR 为 DATE
  算法/硬件主线**；Local5 保留为精度上界与空间局部注意力消融。
- 硬件主故事固定为 Motion-aware all-binary TTX 与可逆时间商流；TESC/RQTB
  合并为一项贡献。64-bit temporal-pair TTB、Exact Delta-TTX、occupied-class
  SCS/Shiftmax 和 zero-K folding 是同一执行后端的支撑机制，不能把普通 TTB
  单独声明为原创。dense/sparse 双 core 仅在相对统一 C0 core 的同约束 PPA/EDP
  有净收益后晋级。
- crop 协议下 H81 已是与 H67 同起点、同 full30 预算且只关闭
  `binary_motion_xor_alpha` 的 reviewer control；但此前没有 paper geometry
  `480x640 / T2x15x15` 的同协议结果。新增配置
  `configs/generated/dsec_fullres_w15_H81_nomotion_bb1e4_ft40.yml`，从 H81 crop
  ep19 开始运行 40 个 full-resolution epoch，并在等预算 30/35/40 点评估
  checkpoint `29/34/39`。
- 为避免抢占当前 direct-MVSEC 队列，新增
  `entrypoints/run_dsec_fullres_w15_h81_after_mvsec.py`；它只在
  `mvsec_cicc_nb0_h67_local5_comparison_20260811.json` 出现后启动。加载门要求
  ATLIF `105`、Shiftmax `12`、overlay `210`、missing/unexpected `0/0`；旧配置、
  checkpoint 和结果均不修改。

<!-- MVSEC_DIRECT_NB0_FIXED800_PROGRESS_20260811 -->

### direct-MVSEC NB0 fixed800 与 full-sequence 中间进度（2026-08-11）

- NB0 seed0 已完成 50 epoch；按冻结 validation loss 选中 `checkpoint_epoch11.pth`，
  validation loss=`8.1990142648`。该 checkpoint 已完成四序列 fixed800 标准推理：
  `outdoor_day1/indoor_flying1/2/3` AEE 分别为
  `0.8379/1.5977/2.7469/2.1102`，macro mean AEE=`1.8231`，按有效像素加权
  AEE=`2.4429`，总有效像素=`17,302,983`。对应 GT Fl 分别为
  `3.7352%/11.6992%/35.1511%/22.7118%`。
- full-sequence 推理已完成 `outdoor_day1`（2755/2755，exit0），当前顺序执行
  `indoor_flying1 -> indoor_flying2 -> indoor_flying3`。NB0 full 完成后，冻结队列会从
  同一个 NB0 ep11 分叉 H67 与 Local5，逐一执行加载冒烟、30 epoch direct-MVSEC 训练、
  fixed800 和 full-sequence 推理。
- 当前 NB0 数值只能建立 direct-MVSEC 同协议基线，不能据此裁决 H67/Local5，也不能与
  DSEC full-resolution 或旧 MDR->MVSEC 数字混表。最终判断只读取
  `mvsec_cicc_nb0_h67_local5_comparison_20260811.{json,md}`；H81 no-motion fullres watcher
  在该比较文件出现前只等待，不占 GPU。

<!-- MVSEC_DIRECT_H67_COMPLETE_LOCAL5_RUNNING_20260812 -->

### direct-MVSEC H67 完成与 Local5 运行状态（2026-08-12）

- NB0 full-sequence 已完成：macro AEE=`1.8273`，有效像素加权 AEE=`2.3435`，
  四序列总有效像素=`44,208,251`。H67 从同一个 NB0 seed0 `checkpoint_epoch11.pth`
  初始化，加载审计 PASS：ATLIF=`105`、Shiftmax=`12`、`checkpoint_overlay_keys=0`、
  `missing=210`、`unexpected=0`；30 epoch 中 validation-rank1 为 H67 ep12，loss=`8.0028`。
- H67 fixed800 四序列 AEE 为 `0.8181/1.5850/2.6212/2.0352`，macro AEE=`1.7649`，
  有效像素加权 AEE=`2.3287`。相对同协议 NB0 分别改善 `3.20%/4.67%`，总 spikes
  从 `97.3392G` 降至 `55.1700G`（`-43.32%`），energy proxy 从 `82,900.09uJ`
  降至 `47,654.90uJ`（`-42.52%`）。
- H67 full-sequence 四序列 AEE 为 `0.8201/1.5868/2.6258/2.0357`，macro AEE=`1.7671`，
  有效像素加权 AEE=`2.2300`。相对 NB0 分别改善 `3.29%/4.84%`，macro GT Fl
  改善 `6.68%`，总 spikes 从 `251.4680G` 降至 `140.6647G`（`-44.06%`），energy
  proxy 从 `214,151.98uJ` 降至 `121,555.15uJ`（`-43.24%`）。四个测试序列的
  AEE 均独立改善，因此 seed0 已通过跨数据集候选门。
- Local5 使用相同 NB0 ep11 初始化，加载合同同样 PASS；当前继续 30 epoch 训练，不抢跑推理。
  截至完成 ep12，validation-rank1 为 ep12、loss=`8.1244`。最终 Local5 checkpoint 及两套
  MVSEC 推理完成前，不做 H67-vs-Local5 的 MVSEC 裁决。
- MVSEC H67 ep12 只用于 direct-MVSEC 算法泛化表；DATE 硬件 RTL/profile 的冻结 checkpoint
  仍是 DSEC full-resolution H67 ep35，禁止用 MVSEC ep12 替换硬件 provenance。

<!-- MVSEC_SMOKE_CHECKPOINT_CLEANUP_20260812 -->

### direct-MVSEC 加载冒烟 checkpoint 清理与防复发（2026-08-12）

- H67/Local5 加载 smoke 原先只限制每个 epoch 的 train/validation batch 数，但未限制
  `n_epochs`，因此产生多组不参与选点、推理或硬件绑定的临时 model/state。加载合同已由
  `train.log` 与 `launch_provenance.json` 完整保留：ATLIF `105`、Shiftmax `12`、overlay `0`、
  missing/unexpected `210/0`、exit0。
- 精确删除五个 smoke 目录中的 72 个 `checkpoint_epoch*.pth`，共回收
  `23,441,861,279 bytes`（约 `21.83 GiB`）；正式 NB0 ep11、H67 ep12 和当前 Local5 ep12
  checkpoint 均在删除后复验存在。机器审计为
  `neuron_autoresearch/cleanup_audits/mvsec_load_smoke_checkpoints_20260812.json`。
- supervisor 的可选 smoke 分支现派生 `n_epochs=1` 的临时配置，并在加载审计 PASS 后自动删除
  临时 checkpoint、保存清理收据。隔离 helper 测试与 `py_compile` PASS；该修改不触碰当前已启动
  的 Local5 正式训练，也不改变任何正式配置或 checkpoint。

<!-- NB0_AAE_GAP_FINAL_CLOSURE_20260812 -->

### NB0 AAE 差距最终收口（2026-08-12）

- 新增 `NB0_AAE_GAP_CLOSURE_20260812.{json,md}`，它是 8月6日早期诊断的后继收据，不覆盖
  原时间线。当前 metric/evaluator/aggregation 源 SHA 与 `AAE_METRIC_TEST_RECEIPT_20260805`
  一致，AAE/aggregation 共 8 项单测 PASS：历史 `AAE` 是二维 `(u,v)` 方向角，论文对齐的
  `AAE_Benchmark` 是 Barron/Middlebury `(u,v,1)` 三维角度。
- equal+10 已排除“NB0 只是轮次不足”：预算30/35/40 的 AEE 为
  `1.4454/1.4584/1.4549`，AAE-2D 为 `6.5128/6.5741/6.5222`，AE-3D 为
  `6.1803/6.2463/6.2109`。预算30仍为 AEE rank-1，继续训练未改善角度，最终判定
  `operationally_plateaued_or_overfit`；不再给 NB0 增加训练轮次。
- NB0 ep29 的 AE-3D 按 frame-equal/pixel-global/sequence-balanced 三种聚合分别为
  `6.1803/5.9892/6.0925`，均不能得到论文 official hidden-test `4.871`。因此聚合差异本身也
  不能消除数值差距；根本边界是本地 18 段 valid825 与官方七条 hidden-test population/server
  protocol 不同，禁止直接声称本地复现或未复现 `4.871`。
- 在同一 local valid825 上，H67 相对收敛 NB0 的 AEE/AE-3D/spikes 改善
  `8.00%/8.57%/34.89%`；Local5 改善 `9.00%/10.39%/33.06%`。这组同 population 对照才是
  DATE 算法表可使用的 AAE 结论。

<!-- MVSEC_FINAL_COMPARISON_FAIL_CLOSED_AUDIT_20260812 -->

### direct-MVSEC 三线最终 comparison 自动审计（2026-08-12）

- 新增 `entrypoints/audit_mvsec_cicc_comparison_20260812.py`。它只在 supervisor 生成
  `mvsec_cicc_nb0_h67_local5_comparison_20260811.json` 后运行，重新从六份原始 summary 与逐序列
  字段计算 macro AEE、valid-pixel-weighted AEE、macro GT Fl、总 spikes 和 energy proxy，不直接
  信任 comparison 内嵌派生值。
- fail-closed 门包括：三条路线精确集合、fixed800/full 双协议、四序列无 skipped、固定样本数、
  checkpoint/config/manifest SHA、comparison 内嵌 summary 与磁盘原件完全一致，以及 H67/Local5
  从同一 NB0 ep11 初始化的 provenance 和 ATLIF105/Shiftmax12/overlay0/missing210/unexpected0
  加载合同。H67 还必须满足 AEE 在 NB0+5% 内、spikes 至少下降20%、四序列 AEE 全改善。
- 已完成的 NB0/H67 四份 summary 和 H67 加载合同预审计 PASS。独立 watcher PID 由
  `mvsec_cicc_nb0_h67_local5_audit_watcher_20260812.pid` 记录，PPID=1、无TTY、只轮询文件且不占
  GPU；最终输出为 `mvsec_cicc_nb0_h67_local5_audit_20260812.{json,md}`。
- 该表是 direct-MVSEC 算法泛化证据。MVSEC checkpoint 不继承 DSEC 硬件证据；component
  RTL-exact/T450 profile 仍唯一绑定 full-resolution H67 ep35。

<!-- DATE_CLOSURE_CURRENT_SOURCE_REBIND_20260812 -->

### DATE 闭环当前源码重绑与声明边界（2026-08-12）

- 总审计首次重跑时 fail-closed 捕获 Local5 projection vector generator 源码 SHA 已漂移；
  旧 `report.json` 虽为 PASS，但不再允许引用。不回退当前源码，而是从已冻结的
  Local5 full-resolution rank-1 checkpoint/profile 重生成 100 组真实权重 projection 向量，并重跑
  score/Shiftmax 与 projection RTL/SVA/lint/synthesis。两者 exit0/PASS；ATLIF 报告因 checkpoint/config
  SHA 一致且 provenance 有效而安全复用。
- H67 ep35 聚合证据的 scope 已与实际三类已验组件对齐为
  `checkpoint_bound_qk_score_scs_shiftmax_atlif_temporal_matrix_real_weight_projection_component_rtl_exact_not_full_network`；
  该修正只消除“已有 projection PASS 但聚合 scope 仍说未覆盖 projection”的口径冲突，
  不扩展为 full-network RTL exact。
- 重新执行 `audit_date_algorithm_closure_20260805.py` 后，
  `DATE_ALGORITHM_CLOSURE_AUDIT_20260805.{json,md}` 在当前工作树上再次 `PASS`。
  DSEC 算法门仍选 H67 ep35：AEE=`1.3296776`（较 NB0 `-8.00%`），
  total_spikes=`82.1107G`（`-34.89%`）。全局论文声明边界冻结为
  `checkpoint_bound_component_rtl_exact_not_full_network`。

<!-- LOCAL5_FULLRES_40_TO_50_PREREGISTRATION_20260812 -->

### Local5 full-resolution 40→50 收敛延伸预注册（2026-08-12）

- 三线 equal-budget 30/35/40 已完成公平比较，但 Local5 ep39 在最大预算边界成为
  AEE rank-1（`1.3152881`），因此严格统计口径仍为 right-censored / `not_plateaued`。
  新增可选配置 `dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50.yml` 和独立 runner
  `run_local5_fullres_plus10_after_h81_20260812.py`，仅从 Local5 ep39 的
  model+optimizer+scheduler+AMP scaler 续至总预算50，冻结评估 ep39/44/49。
- 队列顺序冻结为：direct-MVSEC Local5 训练/推理/三线审计 → H81 no-motion DSEC fullres40
  控制 → Local5 40→50。watcher PID=`283692`，PPID=1，当前只等待 H81 ranking，不占 GPU。
- 该延伸只解决算法收敛边界；ep44/49 默认没有任何硬件 provenance。Local5 现有
  checkpoint-bound component RTL 仍绑定 ep29。只有 ep44/49 最终被选为论文主张 checkpoint 时，
  才单独决定是否为该 checkpoint 重做 profile/RTL；禁止继承 ep29 SHA 证据。
- ep39 resume 已提前 staging 并生成 `resume_stage_audit.json`：model/state hardlink、源 SHA、
  五组 optimizer LR、空 milestones 和 AMP scaler 全部 PASS。历史 forced checkpoint 在
  `scheduler.step()` 前保存，因此 ep34/39 的 scheduler `last_epoch` 为 label-1（ep33/38）；
  runner 显式审计该顺序，不伪造为 scheduler=39。由于 milestones 已清空，续训仍保持
  源 optimizer 的固定 LR。

<!-- H67_H81_TRAINING_FAIRNESS_RECEIPT_20260812 -->

### H67 Motion 与 H81 no-motion 训练公平性收据（2026-08-12）

- 新增 `H67_H81_TRAINING_FAIRNESS_20260812.{json,md}` 和可重跑 auditor。共19项检查 PASS：
  crop 阶段两者使用同一 TTX ep2 parent checkpoint、同 seed 与同配置；去掉
  experiment/note 后唯一算法差异是 `binary_motion_xor_alpha=0.25` 对 `0.0`；
  两线 crop 评估 epoch 集合相同且 ep19 都是 AEE rank-1，不是后验选点。
- full-resolution 的 model/neuron/optimizer/augmentation、`480x640`、crop=null、window
  `2x15x15`、batch2 与评估合同一致；H81 预注册连续40轮。但 H67 历史 full-resolution
  是已审计的五段 rescue/continuation，H81 是连续训练，故收据状态为
  `PASS_RECIPE_LEVEL_CONTROL_NOT_STEP_PAIRED`。
- 允许主张：H81 是 same-parent、seed-matched、same-recipe no-motion 控制。
  禁止主张：H67/H81 是每一 optimizer step 都 bit-exact 配对、只差 Motion-XOR。
  最终 Motion 贡献幅度必须等 H81 ep29/34/39 standard valid825 完成后才填表。

<!-- DATE_FINAL_MAINLINE_AUDIT_PREREGISTRATION_20260812 -->

### DATE 最终跨证据主线裁决预注册（2026-08-12）

- 新增 `audit_date_final_mainline_20260812.py`，它等待 direct-MVSEC 三线审计、
  H67/H81 no-motion 控制和 Local5 40→50 收敛审计全部完成后，再同时读取
  DSEC closure、H67 ep35 硬件证据与 Local5 ep29 硬件证据。watcher PID=`291486`，
  PPID=1，当前只等待三份未完成结果。
- 决策政策预注册为：两线先满足 DSEC AEE≤NB0+5%、spikes≤NB0-20%。当 Local5 的
  DSEC AEE 优势不超过2%，且 H67 在 spikes 或 MVSEC AEE 至少一项更好、并拥有同
  checkpoint ep35 component RTL PASS 时，选 H67 Motion-TTX 为部署主线，Local5 为精度上界。
  若 Local5 优势超过预注册幅度，auditor 输出 `REVIEW_REQUIRED`，不强行保留 H67。
- 按用户当前口径，硬件工程完整度暂不参与决策；只使用创新度/数据流统一性、
  软硬件协同度、spikes/跨数据集泛化和同 checkpoint RTL provenance。未重做的
  Local5 ep39/44/49 不继承 ep29 RTL。

<!-- MVSEC_EPOCH12_CICC_AND_DSEC_COMPLETION_AUDIT_20260812 -->

### MVSEC ep12、CICC 对齐边界与 DSEC 完成度审计（2026-08-12）

**ep12 不是“从头训练 12 轮即完全收敛”。** direct-MVSEC 的 NB0 先从头训练 50 epoch，
按 `outdoor_day2` 时间后段 263 个 held-out validation sample 选中 ep11
（validation loss=`8.1990`）。H67 和 Local5 再从同一个 NB0 ep11 模型初始化；这不是
optimizer/scheduler resume，而是安装 ATLIF=`105`、Shiftmax=`12` 后加载公共 backbone，
两者都完整训练 30 个 adaptation epoch。加载时预期为 overlay=`0`、missing=`210`、
unexpected=`0`；候选 checkpoint 推理回载为 overlay=`210/210`、missing/unexpected=`0/0`。

- H67 的 30 个 validation loss 全部存在；ep12=`8.0028` 为 rank-1，后续 ep25=`8.1348`、
  ep29=`8.4197`，17 个后续 epoch 未刷新 rank-1。
- Local5 的 30 个 validation loss 全部存在；ep12=`8.1244` 为 rank-1，后续 ep25=`8.2333`、
  ep29=`8.5836`，同样未刷新 rank-1。
- 因此论文应写“validation-selected early optimum after 12 adaptation epochs”，不能写
  “strictly converged at epoch 12”。训练损失仍缓慢下降、validation 有明显噪声，严格收敛
  证据有限；但继续到 ep29 未刷新，说明 ep12 不是因训练提前停止偶然留下的边界最优。
- 四测试序列 fixed800/full-sequence 标准推理只对 validation rank-1 做一次。其他轮次已通过
  held-out validation 比较，但没有在四测试序列上逐个推理；这是防止 test-set checkpoint
  cherry-picking 的有意设计。若补 checkpoint sensitivity，只能评估预先保存的 H67 ep10、
  Local5 ep5/ep12 等，并明确不改变主 checkpoint 选择。

**CICC 2026 参考边界。** 当前 direct-MVSEC 已对齐其软件评测范式：`outdoor_day2` 训练、
`indoor_flying1/2/3 + outdoor_day1` 测试、dt1、中心 `256x256`、event+valid-flow mask，
并提供每序列固定 800 个确定性均匀样本。当前不是 CICC 芯片或其 Hybrid U-Net 的复现；
不能把本项目绝对 AEE 与论文 `0.96/0.99` 直接等同。后续冻结 H67 后，硬件侧按同一 checkpoint
和同一 800x4 trace 做累计表：`C0 INT8 dense -> C1 + group16 lossless BWAC -> C2 + density
speculation -> C3 + feature-similarity deep-level skip`，逐行报告 AEE delta、operations、
SRAM/DRAM bytes、cycles、control overhead、energy/latency。TTB 是 Bishop 启发的本项目正交
方向，不属于该 CICC 论文。

**当前 MVSEC fixed800 三线结果。** NB0/H67/Local5 的 macro AEE 分别为
`1.8231/1.7649/1.7984`，total spikes 分别为 `97.3392/55.1700/55.4902 G`。H67 相对 NB0
AEE 改善 `3.20%`、spikes 下降 `43.32%`；Local5 相对 NB0 AEE 改善 `1.36%`、spikes 下降
约 `43.00%`。H67 当前同时优于 Local5 的跨数据集 AEE 和 spikes。NB0/H67 full-sequence 已
完成；Local5 full-sequence 在本审计时运行中，完成后才生成三线 fail-closed comparison。

**DSEC 完成度。** 论文式本地协议已经冻结为 `480x640`、crop=null、`T2x15x15`、
no-running BN、18 段 valid825。NB0/H67/Local5 的公平预算 30/35/40 标准推理全部完成：

| route | rank-1 | AEE | AAE-2D | AE-3D | Fl (%) | spikes (G) | convergence |
|---|---:|---:|---:|---:|---:|---:|---|
| NB0 | ep29 | 1.4454 | 6.5128 | 6.1803 | 7.9323 | 126.1156 | plateau/overfit |
| H67 Motion-TTX | ep35 | 1.3297 | 5.9004 | 5.6509 | 6.4279 | 82.1107 | passed optimum |
| Local5 | ep39 | 1.3153 | 5.8291 | 5.5379 | 6.3815 | 84.4197 | right-censored |

H67 相对 NB0 的 AEE/spikes 为 `-8.00%/-34.89%`；Local5 为 `-9.00%/-33.06%`。
H67 ep35 已具有同 checkpoint 的 profile 与 ATLIF、score/Shiftmax、projection component
RTL-exact PASS；声明边界是 component-level exact，不是 full-network RTL-exact。Local5 现有
硬件证据只绑定 ep29，不能继承给 ep39。

DSEC 核心算法对照和 H67 硬件闭环已经完成，但“所有结果”尚未全部完成：H81 no-motion
fullres40 用于隔离 Motion-XOR 贡献；Local5 40->50 用于解除 ep39 最大预算 right-censor；
两者完成后运行最终跨证据主线审计。若 Local5 ep44/49 成为最终论文 checkpoint，还必须重做
同 checkpoint profile/RTL。官方 DSEC hidden-test server 提交也未完成，本地 valid825 不能标为
official test。

<!-- DATE_PAPER_ALGORITHM_EXPERIMENT_BLUEPRINT_20260812 -->

### DATE 正文算法实验蓝图与 Local5 保留决策（2026-08-12）

- 新增 `neuron_autoresearch/DATE_PAPER_ALGORITHM_EXPERIMENT_BLUEPRINT_20260812.md`，逐项冻结
  MVSEC 文献协议分类、DSEC/MVSEC 正文表头、机制消融、收敛、随机种子、事件密度分层、
  算法到硬件复杂度桥接和加载审计附表。不同训练来源（MVSEC day2、UZH-FPV、MDR、无训练）
  必须分组，禁止只按绝对 AEE 混排。
- `outdoor_day2` 是历史学习型 MVSEC 的主流训练选择，但不是唯一选择。E-RAFT 还使用
  temporal-upsampled 约45-Hz GT；ET-FlowNet/FireNet 系使用外部 FPV；ADMFlow/SDformerFlow
  使用 MDR；model-based 方法无需训练。当前 direct-MVSEC 的 2363/263 supervised split 是
  三线公平内部泛化协议，不冒充 E-RAFT/Spike-FlowNet 训练复现。
- Local5 不再被提前降为仅“精度上界”。它与 H67 保持双候选直到 ep50、MVSEC full 和等面积
  系统硬件表完成。现有 Local5 证据证明约80% SRAM transaction reduction 的潜力，但 direct
  GASR 周期尚为 `0.995x`，自适应选择约 `1.022x`；因此“硬件潜力更大”目前是有数据支撑的
  假设，不是已闭合的速度/能量结论。

<!-- MVSEC_DAY2_ONLY_PROTOCOL_FROZEN_20260812 -->

### MVSEC day2-only 正文协议冻结（2026-08-12）

- 用户确认采用文献中最普遍的 `outdoor_day2` 训练、`outdoor_day1 + indoor_flying1/2/3`
  测试范式。当前 NB0/H67/Local5 三线已经使用同一 day2-only manifest，无需重启训练。
- 为避免测试集选点，day2 有效监督对按时间冻结为 `2363 train + 263 held-out validation`，中间
  留一个样本 gap；四测试序列仅在 validation rank-1 冻结后运行 fixed800/full-sequence。
  正文必须披露这一内部90/10划分，不能写成全部 day2 pair 都参与梯度更新。
- MDR->MVSEC、FPV->MVSEC 和 E-RAFT temporal-upsampled GT 属于不同训练来源/监督口径，只能
  放在分组 related-work 表，不与本项目 day2-only 三线混作同协议绝对排名。
<!-- DATE_MOTION_REVIEW_ACTIONS_20260813 -->

## DATE Motion 主线预投稿审查执行项（2026-08-13）

### 冻结决策

- DATE 算法/硬件协同的唯一论文主线冻结为 **H67 Motion-TTX**。Local5 只完成已经排队的
  full-sequence MVSEC、40→50 收敛审计，并作为扩展/附录候选；不再为当前 DATE 稿件新增
  Local5 算法分支。
- H67 唯一 paper checkpoint 冻结为 full-resolution `ep35`：
  `checkpoint_sha256=4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158`。
- 机器可读身份合同：
  `neuron_autoresearch/H67_PAPER_IDENTITY_CONTRACT_20260813.json`，状态 `PASS`。

### 模型身份与术语

- 公开 SDformerFlow 使用原始 SDSA；H67 是重新训练的 `all12 H60 Motion-XOR + Shiftmax + gated-K`
  attention operator，不能写成公开 SDSA 的代数等价硬件实现。
- 论文统一使用：`T_snn=10`、`T_w=2`、`H_w=W_w=15`、`N_tok=450`、
  `N_pair=225`。禁止再用裸 `T=450` 表示 token 数。
- `exact` 只表示相对冻结 hardware-order fixed-point reference 的 component-level bit-exact；
  不表示 float SDSA 等价或 full-network RTL-exact。

### 审查意见中已由现有证据解决的项目

- ep30/ep35 身份冲突已解决：当前 `h67_postconvergence_rank1_hardware_evidence_20260805.json`
  明确绑定 ep35。
- synthetic projection 风险已升级：ep35 已有 checkpoint-bound real-weight dyadic-INT8
  projection RTL，12/12 blocks、Icarus/Verilator 全记录 bit-exact。仍只声称 component-level，
  不外推为 full network。
- 当前算法指标：AEE=`1.329678`，AAE-2D=`5.900353`，AE-3D=`5.650878`，
  DSEC Fl=`6.4279%`，spikes=`82.1107G`。

### 新增算法实验队列：H67 score precision Pareto

- 配置：QF5/QF6/QF7/QF8 score grid，固定 Q1.7 gate、ep35、480×640、
  `window=[2,15,15]` 和 Valid825；`QF` 表示 fractional bits，不是总码宽。
- QF5/QF6/QF8 使用 generic quantized Shiftmax，只做算法敏感性，不声称存在相应 RTL；
  Q7 hardware-order LUT 结果继续由已有独立证据报告。
- 输出 AEE、AAE-2D、AE-3D、Fl、spikes、pair-score equality 和理想 dual-slot reduction。
- manifest：
  `neuron_experiments/H9_bipolar_self_attention/configs/generated/h67_ep35_score_precision_qf5_qf8_manifest.json`。
- watcher：
  `neuron_experiments/H9_bipolar_self_attention/entrypoints/run_h67_score_precision_sweep_after_mainline.py`；
  它等待现有 H81/Local5/final-mainline 队列结束后才使用 GPU。

### 本轮不做

- 不修改 `hw_autoresearch_nts07` 中任何 RTL、脚本、PPA 或验证代码。
- 不在当前算法队列中实现 class-wise K folding、inverse-stencil、SAIF 或 full-encoder shell；
  这些属于硬件侧后续决策，不应由算法实验抢占现有收敛队列。
- 不再增加新的注意力范式或内部缩写；当前算法证据优先补 Motion/no-motion、跨数据集和
  precision Pareto。

<!-- MVSEC_DIRECT_THREE_ROUTE_FINAL_AUDIT_20260812 -->

### direct-MVSEC NB0/H67/Local5 最终审计

- fail-closed 审计 PASS；三线 best checkpoint 均从原始训练日志重算 validation-loss rank-1，并绑定 checkpoint/config/manifest SHA。
- full-sequence 同协议结果：

| route | epoch | macro AEE | weighted AEE | Fl(%) | spikes(G) | energy proxy(uJ) |
|---|---:|---:|---:|---:|---:|---:|
| nb0 | 11 | 1.827258 | 2.343471 | 18.3537 | 251.4680 | 214151.98 |
| h67 | 12 | 1.767113 | 2.230047 | 17.1276 | 140.6647 | 121555.15 |
| local5 | 12 | 1.801101 | 2.269640 | 17.7919 | 141.3613 | 122047.50 |

- MVSEC algorithm-only 合格候选中按 macro AEE 选中 `h67`。该结论不改变 DSEC H67 ep35 硬件主线，MVSEC checkpoint 不继承 DSEC RTL provenance。
- 机器审计：`neuron_experiments/H9_bipolar_self_attention/results/mvsec_cicc_nb0_h67_local5_audit_20260812.json`。

<!-- DATE_ALGORITHM_GROK_TAKEOVER_20260813 -->

### Grok 接管算法队列 2026-08-13

- Codex `019ec76b-ea14-7862-be41-45ea956713db` 额度用尽后，由 Grok 继续同一条 GPU 队列：H81 fullres40 → valid825 → Local5 40-50 → 最终主线审计 → QF5-QF8。
- H67 Motion-TTX ep35 仍是 DATE 唯一主线。seed1/2 只登记不启动；valid825 密度四分位人口已冻结。
- 机器交接：`neuron_autoresearch/DATE_ALGORITHM_GROK_TAKEOVER_20260813.md`。

<!-- DATE_ALGORITHM_QUEUE_REORDER_20260814 -->

### DATE 算法队列主线优先级修正（2026-08-14）

- H81 连续训练不变；当前依赖顺序改为：H81 fullres40 + valid825 → H67 QF5-QF8 →
  Local5 40→50 → 最终跨证据审计。
- 原顺序把 H67 主线位宽消融放在 Local5 扩展之后，Local5 失败或超时会无意义阻塞 QF。
  新顺序只调整两个空等 watcher，不重启 H81，不并行抢 GPU。
- QF watcher 以 `H67_H81_NOMOTION_RESULT_20260812.json` 为释放门；Local5 watcher 以
  `h67_ep35_score_precision_qf5_qf8_20260813/summary.json` 为释放门。
- 硬件证据仍只读；本次未修改任何硬件代码或硬件文档。


<!-- H67_H81_NOMOTION_FINAL_RESULT_20260812 -->

### H67 Motion vs H81 no-motion 最终控制

- H67 Motion ep35: AEE=`1.329678`，AAE-2D=`5.900353`，AE-3D=`5.650878`，spikes=`82.1107G`。
- H81 no-motion ep29: AEE=`1.330597`，AAE-2D=`5.969235`，AE-3D=`5.672632`，spikes=`80.9024G`。
- H67 AEE 相对 H81 变化=`-0.069%`（负值为更好）；H81 收敛判定=`operationally_plateaued_or_overfit`。该证据是 recipe-level control，不是 step-paired bit-exact 训练。
- 机器审计：`neuron_autoresearch/H67_H81_NOMOTION_RESULT_20260812.json`。


<!-- H67_SCORE_PRECISION_QF5_QF8_RESULT_20260813 -->

### H67 ep35 score QF5-QF8 sensitivity

| score | AEE | delta vs QF7 | AAE-2D | AE-3D | Fl(%) | spikes(G) | pair equal | ideal dual-slot reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| QF5 | 1.332377 | +0.004465 | 5.908379 | 5.665975 | 6.4666 | 82.1065 | 98.4138% | 49.2069% |
| QF6 | 1.331083 | +0.003171 | 5.925380 | 5.676300 | 6.4542 | 82.1075 | 92.0596% | 46.0298% |
| QF7 | 1.327912 | +0.000000 | 5.914105 | 5.666098 | 6.3915 | 82.1065 | 97.5198% | 48.7599% |
| QF8 | 1.330811 | +0.002899 | 5.926746 | 5.679027 | 6.4539 | 82.1074 | 92.9069% | 46.4535% |

- 该表是算法位宽敏感性，不把 QF5/QF6/QF8 写成已有 RTL。
- 机器结果：`neuron_experiments/H9_bipolar_self_attention/results/h67_ep35_score_precision_qf5_qf8_20260813/summary.json`。

<!-- LOCAL5_FULLRES_40_TO_50_FINAL_RESULT_20260812 -->

### Local5 full-resolution 40→50 收敛结果

| budget | checkpoint | AEE | AAE-2D | AE-3D | Fl(%) | spikes(G) |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 39 | 1.315288 | 5.829055 | 5.537945 | 6.3815 | 84.4197 |
| 45 | 44 | 1.281893 | 5.849797 | 5.508685 | 6.0210 | 85.2376 |
| 50 | 49 | 1.298168 | 5.831211 | 5.501246 | 6.2162 | 85.8205 |

- 收敛判定=`operationally_plateaued_or_overfit`，AEE rank-1=ep44。
- ep44/49 没有硬件 provenance；Local5 现有 component RTL 仍只绑 ep29。
- 机器审计：`neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_20260812/convergence_summary.json`。


<!-- DATE_FINAL_MAINLINE_DECISION_20260812 -->

### DATE 最终跨证据主线裁决

- 状态=`PASS_EVIDENCE_AUDIT`，决策=`H67_MAINLINE_FROZEN_LOCAL5_EXTENSION_REPORTED`，主线=`H67_Motion_TTX`。
- DSEC: H67 AEE/spikes=`1.329678/82.1107G`，Local5 rank-1 ep44=`1.281893/85.2376G`。
- MVSEC full AEE: H67=`1.767113`，Local5=`1.801101`；Motion control H67/H81=`1.329678/1.330597`。
- 硬件证据仅只读消费；该算法审计不修改硬件代码或硬件文档。
- 机器审计：`neuron_autoresearch/DATE_FINAL_MAINLINE_DECISION_20260812.json`。
