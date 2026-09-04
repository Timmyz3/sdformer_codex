# H13 系列深度审查

日期: 2026-05-19 | 当前运行: H13n

---

## 一、三元设计（BSA / ATLIF / TSN 范式）

### 已做到的

| 组件 | H9a (旧) | H13n (当前) | 来源 |
|------|---------|-----------|------|
| 三元输出 | `{-θ,0,+θ}` | `{-θ,0,+θ}` | BSA / TSN |
| 正负阈值比 | `neg = thre × 30` | `neg = thre × 1` (对称) | 自研改进 |
| 阈值模式 | asymmetric_scale | **symmetric_target_rate** | 自研 |
| PSN 底座 | 保留 weight/bias | 保留 weight/bias | PSN (NeurIPS 2023) |
| ATLIF 阈值更新 | activity × total_r | activity × (total_r - target_rate) | ATLIF (NeurIPS 2024) |

### 评估

**对称三元 (neg=thre) 已经工作**: 数据证明 `ternary_pos_mean ≈ ternary_neg_mean ≈ 0.023`，正负完全对称。这是 H13 系列最核心的改进。

**但 `symmetric_target_rate` 对 BSA 范式的引用不完整**: BSA 原文有两个关键组件——三元矩阵乘积(TMP) 和 Shiftmax。H13 的 `signed_consensus_shiftmax` 用 `Q_sign * K_sign` 取代了 TMP 的点积，这是一个架构适配（因为 SDformer 没有 V），但论文需要在 Related Work 中明确说明这一点与 BSA 的差异。

**ATLIF 阈值更新的变化未充分论证**: 从原始的 `activity_eta × (total_r - target_rate)` 改为 `symmetric_target_rate` 模式，阈值开始可以**下降**（因为 target_rate=0.05，而 total_r 可能低于或高于 0.05）。H13n 日志显示 `raw_update_mean` 为负值（-6e-05），验证了阈值正在被双向调节。这是 H9a 没有的能力——H9a 的阈值只能上升。

**TSN 范式覆盖不足**: TSN (Ternary Spike Neurons) 的原始设计包括可学习的正负阈值权重和独立的 threshold update。H13 的对称模式让负阈值等于正阈值，这在简化上有优势，但丢失了 TSN 的"正负通道可能需求不同阈值"的灵活性。论文需要讨论这个 trade-off。

### 缺失

- **没有可学习的正负阈值比**。BSA 和 TSN 的原始设计都允许正负通道有不同的阈值。H13 强制 ratio=1 是一种简化假设。
- **没有 Ternary Quantization 参照**。QP-SNN (ICLR 2025) 的三元量化方案基于奇异值阈值，ATLIF 的 threshold 学习可能有借鉴。

---

## 二、注意力改进

### H13 注意力模式矩阵

| 模式 | 核心操作 | 归一化 | 范式来源 |
|------|---------|:---:|------|
| **compat_qk_product** (H9a) | `Q_sum gate × Shiftmax(Q·K)` | 双轨制 | BSA 适配 |
| **signed_consensus_shiftmax** (H13b/H13n) | `(Q_sign * K_sign).sum(-1) → shiftmax` | 单轨 | BSA + 自研 |
| **signed_consensus_shiftnorm** (H13c) | `(Q_sign * K_sign).sum(-1) → shiftnorm` | 单轨+L1 | 自研 |
| **strict_bsa_shiftmax** (H14?) | `Q_ternary @ K_ternary^T → shiftmax @ K` | 标准矩阵 | BSA 严格 |
| **biascenter_shiftmax** (H13f/H13n) | signed_consensus + center_scores → shiftmax | 单轨+中心化 | 自研 |

### 评估

**`signed_consensus_shiftmax` 是最重要的创新**: 它将 Q·K 点积退化为符号乘积求和——等效于 Hamming-distance popcount——然后用 Shiftmax 归一化。这和 BSA 的 TMP 不同（TMP 是全矩阵乘法），但更高效（不需要 O(n²d) 矩阵乘，只需要 O(nd) 符号矩阵 + O(n²) Shiftmax）。

**`biascenter` 的作用未经验证**: H13f vs H13b 的区别仅在于中心化分数 `scores = scores - scores.mean()`。这在 H9a 兼容模式中存在，但 H13n 将其保留在 signed_consensus 中。需要消融实验确认其贡献。

**`strict_bsa_shiftmax` 应该被测试**: 这是对 BSA 最忠实的复刻——三元 Q·K^T 矩阵 + Shiftmax + K 作为 V。它比 signed_consensus 更接近 BSA 原文，且可以作为 H13 的对照实验来验证"签名共识 vs 全矩阵乘积"是否等价。

### 缺失

- **没有 Shiftnorm 的完整测试** (H13o 已配置但未运行)。Shiftnorm = shiftmax 的 2^x 换成 L1 norm。这对硬件友好度论证很重要。
- **没有 POPCOUNT-only (无 Shiftmax) 的控制实验**。我的 SOC/PRA 虽然失败了，但概念上验证了"popcount + divider"是可能的。H13 可以加一个 `popcount_l1_norm` 模式。
- **没有时域注意力**。所有 H13 注意力模式都只做空间 attention，忽略了 10 时间步的时序信息。

---

## 三、负阈值问题

### 当前状态: 已基本解决

| 指标 | H9a | H13n | 改善 |
|------|:---:|:---:|:---:|
| negative_scale | 30 | **1** | ✅ |
| ternary_neg_mean | ~0.0004 | **0.023** | ✅ 57× |
| ternary_pos_mean | ~0.056 | 0.023 | — |
| 正负比例 | 140:1 | **1:1** | ✅ |

H13n 的 `ternary_pos_mean=0.023, ternary_neg_mean=0.022` 证明负阈值问题已经解决。

### 残留问题

**target_rate=0.05 对三元是否合适？** 三元神经元的总活动率 (pos+neg) ≈ 0.046，正好接近 target_rate=0.05。但这是总发放率——如果后续实验需要降低 SOPs，需要降低 target_rate（如 H13p 的 target=0.2 或 H13q 的 target=0.35）。target_rate sweep 覆盖了这个维度。

**阈值仍在上升**: `threshold_mean=0.407` 且继续上升（虽然速度变慢）。`max_threshold=1.8` 提供了充足空间。但阈值上升到什么程度会损害精度？需要一个 `threshold_mean` vs AAE 的关联分析。

---

## 四、AAE 防爆

### 当前状态: 未直接解决

H13n 配置中 `lambda_ang=0, use_angular_loss=false`。AAE 问题完全依赖对称三元 + 正确的注意力来间接防护。

**H13e 测试了 angular loss (λ=0.1)**: 但只是 guard120 烟雾测试。没有全量 angular loss 实验。

### 风险

**H13n 把 FFN 放到了 stage2**: `stage2_half_even_ffn_binary` — H9c 已经证明 stage2 FFN 替换是 AAE 爆炸区。H13n 有三个保护层：
1. 对称三元（保持极性信息）
2. signed_consensus 注意力（保持方向信号）
3. activity_eta 极低 (0.006 for stage2)

但**没有直接的 AAE 保护机制（如 angular loss）**。如果 H13n 的 AAE 仍然 > 8.0，需要加 angular loss。

### 缺失

- **H13n 的全量 angular loss 版本** (可以叫 H13r): λ_ang=0.2~0.3
- **AAE 与 threshold 的关联分析**: 在什么阈值下 AAE 开始爆炸？
- **Per-stage AAE 敏感性**: stage2 FFN 的阈值与 AAE 的关系

---

## 五、硬件友好度

### 当前状态

| 组件 | 硬件运算 | 复杂度 |
|------|---------|:---:|
| signed_consensus | `Q_sign * K_sign` → XNOR + popcount | O(nd) |
| Shiftmax | `2^x` via LUT | O(n²) |
| ATLIF 阈值更新 | 乘法 + 加 | O(1) per neuron |
| 三元输出 | 比较器 ×2 | O(1) per neuron |

### 问题

**Shiftmax 的 2^x 仍然存在**: H13 没有去掉 Shiftmax 的 LUT 成本。biascenter_shiftmax 的中心化增加了额外的均值计算。

**gate * n 乘法**: 所有模式的 gate 归一化后都乘以 n_tokens——这是一个额外乘法。

**signed_consensus 的 XNOR popcount 是可综合验证的**: 这个操作的硬件映射非常清晰（XNOR gate bank + adder tree）。论文中应该包含一个简单的面积/延迟估算。

### 缺失

- **没有硬件综合数据**: 没有 FPGA/ASIC 的面积或功耗估算
- **shiftnorm (H13o) 未在全量验证**: shiftnorm = L1 norm 替代 2^x → 零 LUT，纯加法+除法
- **没有混合精度方案**: attention 和 FFN 可以用不同的 bit-width

---

## 六、可借鉴的外部工作

| 工作 | 可复用的点 | H13 当前 |
|------|-----------|:---:|
| **QP-SNN (ICLR 2025)** | 奇异值阈值选择法；权重+激活联合量化 | 未覆盖 |
| **Spike-driven Transformer V2 (ICLR 2024)** | 分层混合精度 (attention 4-bit, FFN 1-bit) | 未覆盖 |
| **TTFSFormer (ICML 2025)** | 时域注意力 (time-to-first-spike gating) | 未覆盖 |
| **Dual-Sparse LoAS (MICRO 2024)** | 权重+激活双稀疏的数据流设计 | 部分覆盖 |
| **QSD-Transformer (ICLR 2025)** | Spike Information Distortion (SID) 分析 | 未覆盖 |
| **SEMM (NeurIPS 2024)** | Spike-driven MoE routing | 未覆盖 |
| **IM-SNN (IEEE 2024)** | 整数-only SNN, 三元膜电位 | 部分覆盖 |

### 最值得借鉴的三个

1. **QSD-Transformer 的 SID 分析**: "尖峰信息失真"现象——量化后的尖峰分布与原始分布不一致。H13 的三元脉冲也存在类似问题：三元化后的 Q/K 分布与原始 PSN 分布不同。SID 的量化方法是 "Information-Enhanced LIF" + "Fine-Grained Distillation"。H13 可以把这个蒸馏 loss 加到训练中。

2. **TTFSFormer 的时域注意力**: H13 的注意力完全忽略了 10 个时间步的时序信息。TTFSFormer 的 "时间到首脉冲" 门控可以利用事件数据的时序特性——早期脉冲（噪声）和晚期脉冲（信号）应有不同的注意力权重。这个可以直接作为 `consensus_score_norm` 的一个新选项。

3. **LoAS 的双稀疏数据流**: H13 已经实现了激活稀疏（三元发放）和权重稀疏（ATLIF 阈值）。但对硬件数据流的设计仅停留在概念层面。LoAS 的"全时域并行数据流"可以直接引用作为 H13 的硬件映射方案。

---

## 七、补充建议

### 短期 (H13 可加)

1. **H13r: H13n + angular loss (λ=0.2)**: 最直接的 AAE 保护实验
2. **H13s: H13n + shiftnorm (替代 shiftmax)**: 验证 L1 norm 是否等价，去掉 2^x
3. **H13t: signed_consensus + popcount_only (无 Shiftmax, 纯 L1)**: 最硬件友好的验证
4. **TTFS-aware attention norm**: `consensus_score_norm: time_weighted` —— 早期时间步低权，晚期高权

### 中期

5. **QSD-style SID 分析**: 量化 H13 三元脉冲与原始 PSN 脉冲的分布差异
6. **硬件综合 (FPGA)**: 对 signed_consensus + shiftnorm 做基本的资源估算
7. **混合精度**: attention 用 ternary+shiftmax, FFN 用 binary+L1

### 长期

8. **多数据集验证**: MVSEC, HQF
9. **多 seed 统计**: 3-5 seeds for key experiments
10. **MoE routing**: SEMM-style spike-driven expert selection

---

## 八、总体评分

| 维度 | 评分 | 说明 |
|------|:---:|------|
| 三元设计 | 4/5 | 对称 S1 解决核心问题，阈值双向学习是进展 |
| 注意力改进 | 4/5 | signed_consensus 有创新，缺 strict_bsa 对照 |
| 负阈值 | **5/5** | **已解决**，数据证实 |
| AAE 防爆 | 2/5 | **最大风险**，H13n 无 angular loss 且碰了 stage2 |
| 硬件友好 | 3/5 | signed_consensus 硬件友好，但 shiftmax 的 LUT 未去除 |
| 实验系统度 | 4/5 | 消融设计好，但缺 angular loss 全量和多 seed |
