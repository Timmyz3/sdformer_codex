# Rapid Screen 交叉分析

42 个 valid40 profile, 7 个实验族, 10 种注意力模式, 5 月 19-20 日完成。

---

## 一、AAE 排名 (方向误差最重要)

| rank | 实验 | 注意力 | FFN | AEE | AAE | SOPs | Δ vs H9a |
|---:|---|---|:---:|---:|---:|---:|:---:|
| **1** | **H23e** | signed_consensus (H13v) | H13v (st0+1+2) | **1.50** | **7.37** | 3.59G | AAE -3.5% |
| 2 | H24b | alpha_xnor | H9a safe | 1.56 | 7.44 | 3.57G | AAE -2.5% |
| 3 | H23a | alpha_xnor (H18c) | H18c | 1.52 | 7.47 | 3.63G | AAE -2.2% |
| 4 | H25f | signed_consensus | FFN 全三值 | 1.62 | 7.48 | 3.63G | 精度差 |
| 5 | H13w | compat_qk_product | H9a safe+strong | 1.54 | 7.56 | 3.58G | SOPs 高 |
| **H9a** | compat_qk_product | H9a safe | **1.50** | **7.64** | **3.08G** | 基准 |
| 6 | H22j | alpha_xnor (H18c) sign_v | H18c | 1.57 | 7.58 | 3.59G | |
| 7 | H23b | alpha_xnor (H18c) | H18c | 1.55 | 7.63 | 3.53G | ← **当前全量** |

### 发现 #1: H23e 是唯一在 AAE 上显著超过 H9a 的

AAE 7.37 vs H9a 7.64 = -3.5%。但 SOPs 3.59G vs H9a 3.08G = +16.5%。这是精度-稀疏 tradeoff: H9a 用 compat_qk_product 双轨制既保精度又降 SOPs，H23e 的 signed_consensus + 更强稀疏反馈牺牲了 SOPs 换 AAE。

### 发现 #2: 最佳 AAE 实验集中在 H23/H24 族

H23/H24 的共同特征是低 LR (1e-5) + 更强的 target_rate/activity_eta。低 LR 让 ATLIF 阈值更新更平滑，避免阈值 overshoot 杀 Q/K。这是关键超参发现。

### 发现 #3: FFN 全三值 (H25f) AAE 可控但 AEE 差

H25f 用全三值 FFN (sn1 sn2 都是 ternary)，AAE 7.48 尚可，但 AEE 1.62 显著恶化。全三值 FFN 增加负脉冲噪声，破坏 decoder 特征。

---

## 二、SOPs 排名

| rank | 实验 | SOPs | AEE | AAE | 注意力 |
|---:|---:|---:|---:|---:|------|
| H9a | **3.08G** | 1.50 | 7.64 | compat_qk_product |
| 1 | H18a | 3.43G | 1.68 | 7.93 | alpha_xnor shiftmax |
| 2 | H22d | 3.50G | 1.63 | 8.02 | alpha_xnor (target030) |
| 3 | H26h | 3.51G | 1.60 | 7.70 | hamming ternary |
| 4 | H23b | 3.53G | 1.55 | 7.63 | alpha_xnor (target035) |
| 5 | H22c | 3.51G | 1.59 | 7.75 | alpha_xnor (target035) |
| 6 | H24a | 3.55G | 1.66 | 7.75 | alpha_xnor base |
| 7 | H22b | 3.57G | 1.58 | 7.66 | alpha_xnor (target040) |

**所有实验的 SOPs 都 ≥ 3.43G**，没有接近 H9a 的 3.08G。compat_qk_product 双轨制有系统性的稀疏优势。

### 发现 #4: 双轨制 > 单轨制的 SOPs 优势

H9a 的双轨制 (原始 QK gating × Shiftmax gate) 在 SOPs 上有 ~0.4G 的系统优势。所有替换原始 QK gating 的单轨制 (direct attention) 都损失了这 0.4G。

推测原因: 原始 QK gating 中的 `sn2_q` 脉冲化提供了一个额外的稀疏化步骤，direct attention 替代了它，去掉了这个稀疏瓶颈。

---

## 三、注意力模式 vs 性能

| 注意力模式 | 实验数 | AEE 均值 | AAE 均值 | SOPs 均值 | 最佳 variant |
|------|:---:|:---:|:---:|:---:|------|
| compat_qk_product | 2 | 1.54 | 7.59 | 3.58G | H13w |
| signed_consensus | 2 | 1.53 | 7.43 | 3.62G | H23e (AAE 最佳) |
| alpha_xnor matrix | 15+ | 1.59 | 7.76 | 3.60G | H24b |
| strict BSA | 6 | 1.61 | 8.08 | 3.68G | H27a |
| hamming ternary | 3 | 1.63 | 8.15 | 3.58G | H26h |
| hamming binary | 1 | 1.63 | 8.27 | 3.65G | H26d |

### 发现 #5: signed_consensus 在 AAE 上有优势

signed_consensus 的符号共识 + head_dim normalization 保留了方向信息最好。alpha_xnor 的 AAE 偏高 (~7.76)，可能因为 XNOR 矩阵更"硬"（符号突变无平滑）。

### 发现 #6: strict BSA 全矩阵没有比 token-wise 更好

strict_bsa 用 Q@K^T 全矩阵乘法，AEE/AAE 反而中等（均值 1.61/8.08），且 SOPs 更高。对 SDformer 的 no-V 架构，token-wise 门控比全矩阵更有效。

---

## 四、FFN 覆盖度 vs AAE

| FFN 覆盖 | 实验 | AAE |
|------|------|:---:|
| stage0 only | H24a (H9a safe) | 7.75 |
| stage0+downsamples | H9a | 7.64 |
| stage0+1+2+3 | H13n, H23e | 7.37~31.2 |
| 全三值 FFN | H25f | 7.48 |

### 发现 #7: FFN 覆盖度和 AAE 不直接单调

H23e (stage0+1+2) 的 AAE=7.37 是全局最佳。但 H9c 的 stage2 FFN 替换让 AAE 爆炸到 31。关键不是覆盖度，而是 **activity_eta 和 threshold_lr_scale 的设置**——H23e 用 low LR+gentle sparsity 保护了 direction。

---

## 五、关键参数相关性

| 参数 | 与 AAE 的关系 | 说明 |
|------|:---:|------|
| LR | 1e-5 < 2e-5 | 低 LR = 更低 AAE |
| target_rate | 0.035~0.045 最优 | 太低(0.02)太激进, 太高(>0.05)没效果 |
| activity_eta (Q/K) | 弱负相关 | 太高 (1.2+) 推阈值过快 |
| FFN activity_eta | 需 > 0 但 < 0.03 | null target 的 FFN 可用 activity_eta 微控 |
| score_scale | 1.0~1.5 最优 | <0.75 或 >2.0 均有害 |

---

## 六、未覆盖的维度

1. **angular loss 全量**: H24e/f 有 α=0.2/0.5 的 valid40, 但没有系统性 sweep。H9a+I15 的 angular loss 结果未纳入对比。
2. **max_threshold sweep**: 所有 H23/H24 用 max_threshold=2.5。H9a 的 0.13 和 H13n 的 1.8 是不同数量级。中间值 (0.5, 1.0) 未探索。
3. **多 seed**: 全部单次推理，统计显著性未知。
4. **多数据集**: DSEC only。

---

## 七、当前 H23b 全量的预期

H23b guard120 valid40: AEE=1.55, AAE=7.63, SOPs=3.53G

如果 30 epoch 全量后:
- SOPs 能降到 ~3.3G → 和 H9a 的差距缩小到 <7% → 可接受
- AAE 保持 <7.8 → angular loss 不需要
- AAE >8.0 → 需要 angular loss 变体
