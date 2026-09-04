# H18-H27 短测实验全景与代码审查

日期: 2026-05-20

---

## 一、实验族谱：10 种注意力模式

SDformer 原始 attention 是 QKFormer 风格：`K × sn2_q(sum(Q))`——Q 通道坍缩为标量门控 K，无点积，无 softmax。

所有新模式的共同约束：**没有 V 投影**（SDformer 架构没有 V），K 同时扮演 Key 和 Value。

```
compat_qk_product (H9a)
│  双轨制: 原始QK gating + Shiftmax(Q·K) 相乘
│
├── H13b  signed_consensus_shiftmax
│   popcount(Q_sign × K_sign) + Shiftmax
│
├── H13c  signed_consensus_shiftnorm  
│   popcount + power-of-two norm (去 2^x)
│
├── H13t  signed_consensus_popcount_l1
│   popcount + L1 norm (纯 popcount, 零指数)
│
├── H18c  alpha_xnor_matrix_shiftmax  
│   XNOR 相似度矩阵 + Shiftmax
│
├── H18d  alpha_xnor_matrix_l1
│   XNOR + L1 norm (去 2^x)
│
├── H18e  a2os2a_direct
│   Q 二元, K ReLU, L1 norm
│
├── H21a  hamming_binary_direct
│   SpikeVideoFormer Hamming (二元)
│
├── H21b  hamming_ternary_active_direct
│   Hamming (三元活跃, 沉默不参与)
│
└── strict_bsa_shiftmax
    Q@K^T 全矩阵 + Shiftmax (最接近 BSA 原文)
```

---

## 二、每个短测批次的原理

### H18: 第一个 direct attention 探针

**问题**: 能不能用真正的 token-token attention 矩阵替代 QKFormer 的 token gating？

**原理**:
- H18a/c (alpha_xnor): Q/K 转 {-1,0,+1} → `Q_event * K_event^T` → XNOR 式相似度 → Shiftmax → gate @ K。本质是 XNOR popcount 泛化到矩阵形式。
- H18d (alpha_xnor L1): 同上但用 L1 norm 替代 Shiftmax
- H18e (A2OS2A): Q 二元、K ReLU、V 三值，来自 A2OS2A (CVPR 2025) 的简化版

**结论**: 120-step 探针显示 direct attention 可行 (AEE 1.05~1.13)，但 SOPs 高 (3.81~4.33G)。H18c 综合最佳。

### H21: Hamming attention 探针

**问题**: SpikeVideoFormer (ICML 2025) 的 Hamming linear attention 能否适配三元脉冲？

**原理**:
- H21a (binary): 脉冲映射 {-1,+1}，Hamming 相似度 = `0.5*(1 + dot/max)`。沉默 = -1 贡献强烈负信号 → 稀疏下不公平。
- H21b (ternary_active): 沉默 = 0 不参与，只有活跃符号贡献。对稀疏脉冲更公平。
- H21c (binary_signv): 同上但 V 用二值符号。

**结论**: H21b valid40 AEE=1.68, AAE=8.42, SOPs=3.59G。精度不如 H13v/H9a。降级。

### H22: H18c 超参 sweep

**问题**: H18c (alpha_xnor_shiftmax) 的核心超参——target_rate, activity_eta, score_scale——对稀疏度/精度的影响？

**原理**: 固定 H18c 注意力 + H18c 的 FFN 模块设计，扫 target_rate (0.03~0.045), activity_eta (0.4~1.2), score_scale (0.5~1.5), alpha 正则 (0~0.01)。

**结论**: valid10 最优 H22c (AEE=1.03, AAE=6.05, SOPs=3.71G)。valid40 最优 H22j sign_value (AEE=1.57, AAE=7.58, SOPs=3.59G)。没追上 H9a 的 3.08G SOPs。

### H23: 低 LR + 强稀疏反馈组合

**问题**: 更低的 LR (1e-5) 结合更强的稀疏控制能否压 SOPs？

**原理**: 
- H23a-c: H18c 基线 + LR 1e-5 + target 0.035~0.040
- H23d-e: H13v 基线 (signed_consensus) + LR 1e-5 + target 0.035~0.040
- 更强 target_rate 让 ATLIF 阈值更积极压发放

**结论**: H23e valid40 AEE=1.50, AAE=7.37, SOPs=3.59G。AEE 追平 H9a，AAE 略好，但 SOPs 比 H9a 高 16%。**H23b 被 promote 到全量**。

### H24: H9a 安全 FFN + alpha-XNOR

**问题**: 回到 H9a 的安全 FFN 集合 (stage0+s3b0+downsamples)，换上 alpha_xnor 注意力，扫 LR/稀疏/角度 loss？

**原理**: 去掉 stage1/stage2 的 FFN 替换 (H23 的教训)，只保留 H9a 验证过的安全 FFN。测试 H24a-c (base/lr1e5/sparse040)，H24d-g (sparse035/ang002/ang005/flowreg0003)。

**结论**: H24b valid40 AEE=1.56, AAE=7.44, SOPs=3.57G。SOPs 仍高。

### H25: 模块排列组合

**问题**: Q/K 三元固定，FFN 升维/降维、二值/三值、downsample 的各种排列？

**原理**: 系统性测试 FFN 的替换粒度和输出模式。不扫描太多超参，只看"哪个模块组合的 valid10 SOPs 最低且精度合理"。

### H26: 降级注意力回收测试

**问题**: 之前被降级的注意力 (alpha_xnor_l1, a2os2a, hamming) 是否只是当时超参不合适？

**原理**: 回收 H18d/e, H21a/b, 在 H9a 安全 FFN + stronger sparsity 下重新测试。H26a-i 涵盖 L1 变体、值模式、FFN 细分替换、flow reg 调整。

### H27: strict BSA 标准测试

**问题**: 最接近 BSA 原文的 strict_bsa_matrix 在所有 stage 上的表现？

**原理**: Q@K^T 全矩阵 + Shiftmax + K 作为 V。value_mode 扫 sign/threshold，stage 扫 all/stage0/stage1/stage2/stage3。

---

## 三、代码审查

### 3.1 bsa_attention.py — 整体结构 (3/5)

**优点**:
- 单一入口 `_qk_shiftmax_gate_forward` 通过 mode 字符串路由到 15+ 种注意力实现
- 统一接口：所有模式输入 Q/K 原始张量，输出 attn 张量
- 归一化函数族 (shiftmax, shiftnorm, l1norm) 各自独立，易于测试
- monkey-patch 方式干净，不影响 baseline

**问题**:

**P1: 路由函数过长 (第 401-640 行, ~240行)**
单一 `_qk_shiftmax_gate_forward` 包含 15 种模式的 if-elif 链，阅读困难。建议抽离为 `_MODE_DISPATCH` dict，将每个模式的实现独立为函数。

**P2: gate * n_tokens 乘法重复出现 (~10处)**
```python
if cfg.preserve_mean:
    gate = gate * float(n_tokens)
```
这是所有模式的统一后处理，但每处都要手动写。应该提取到统一出口。

**P3: `_ternary_sign_ste` 的 STE 设计有风险 (第 107-108行)**
```python
hard = x.sign()           # {-1, 0, +1}
return (hard - x).detach() + x
```
当 x=0 时 sign()=0, STE 传回原始梯度。但当 x 接近 0 时，STE 的梯度近似 `1 * x_grad`(如果 x>0) 或 `-1 * x_grad`(如果 x<0) 或 `x_grad`(如果 x≈0)。x 的绝对值可能很大（θ 可达 1.8），STE 的 scale 问题被忽略。

**P4: 未使用的变量**
```python
row_sum = gate.sum(dim=-1)  # 多处赋值但仅在部分路径传递
```
在 compat_qk_product 等旧模式中 `row_sum` 用于日志，但在新 direct 模式中 gate 直接用于 `torch.matmul(gate, value)`。row_sum 的语义随模式变化（有时是 row sum of gate，有时未定义），易用错。

**P5: shiftnorm 和 l1norm 的功能重叠 (第 73-101行)**
shiftnorm 分母 ceil 到 2^n（除法可近似为移位），l1norm 精确除法。但在硬件论证中，l1norm 的除法器成本被低估。应该在一篇论文中只选一个归一化方案，避免读者困惑。

### 3.2 atlif_ternary_psn.py — 神经元实现 (4/5)

**优点**:
- symmetric_target_rate 模式正确解决了负阈值问题
- 三元输出严格为 {-thresh, 0, +thresh}，S5 幅值约束
- 阈值双向更新（可升可降），target_rate 机制工作正常

**问题**:

**P1: binary_activity 无 target_rate (中等)**
```python
# 在 target_groups 中
- name: stage0_ffn_binary
  target_rate: null     # ← FFN 二元没有目标率
```
FFN 二元模块的 activity 无法被 target_rate 控制，只能靠 activity_eta 做 soft 约束。这是 H23b full 训练中 binary_activity=0.136 的根因。应该给 FFN 二元也加上 target_rate。

**P2: threshold_lr_scale 默认值依赖 (低)**
```python
self.threshold_lr_scale = 50000.0 if threshold_lr_scale is None else float(threshold_lr_scale)
```
默认 50000 在 config 中有显式覆盖，但万一 config 遗漏，50000×lr 会极大。应该改为必传参数。

### 3.3 rapid_screen.py — 管线 (4/5)

**优点**:
- 自动生成 guard120 config，训练 120 步，profile valid10 + 可选 valid40
- 跳过已有结果的 config（幂等），可安全重跑
- 生成 summary.md 排名表

**问题**:

**P1: 120-step 可靠性 (严重)**
H9b 证明了 120 步 valid10 结果和 30 epoch valid40 之间可能完全不一致（AAE 6.12 → 32.7）。当前管线把 120-step 结果作为主要筛选依据。补救：promotion 前加 epoch-3 验证。

**P2: profile 超时处理 (中等)**
```python
subprocess.run(profile_cmd, timeout=600)  # 10 min timeout
```
大型 valid40 profile 可能超过 10 分钟（尤其 SOPs 统计需要 layer-by-layer profiling）。应设为 20 分钟。

**P3: promote 阈值格式问题 (低)**
```python
parser.add_argument("--promote-aee", type=float, default=1.70)
```
H9a AEE=1.50，允许退化到 1.70 = +13%。应该收窄。

---

## 四、总结评分

| 维度 | 评分 | 关键点 |
|------|:---:|------|
| 注意力设计覆盖 | 4/5 | 10 种模式，从 token gate → direct matrix → Hamming |
| 超参 sweep 覆盖 | 4/5 | target_rate/LR/activity_eta 已系统扫描 |
| 代码结构 | 3/5 | 单函数过长、重复逻辑多 |
| 管线可靠性 | 3/5 | 自动 promotion 工作，但 120-step 可靠性存疑 |
| 硬件诚实度 | 3/5 | l1norm/shiftnorm 诚实，但除法成本未量化 |
