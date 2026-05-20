# H13 Signed-Consensus Attention 深度分析

Date: 2026-05-19

## 一、H13 做了什么（四层改进叠加）

### 第 1 层：Bias-Centered 三元标定（H13f 起，核心创新）

```
原来 (H4-H9):  h = PSN_weight @ x + PSN_bias(-1)
              out = {-th, 0, +th}
              → bias≈-1 被误判为负信号 → neg_scale=30 暴力压制

H13:          h = PSN_weight @ x + PSN_bias(-1) - PSN_bias(-1)
              = PSN_weight @ x + 0
              out = {-th, 0, +th}
              → 正负完全对称, neg_scale=1
```

**代码位置**: `atlif_ternary_psn.py:131-137`
```python
center = self.bias.detach().clone() if center_mode == "bias" else zeros
h_seq = h_seq - center  # 减去 PSN bias, 归零中心
```

**效果**: neg_mean 全程稳定在 ~1%，30 epoch 不衰减。不再需要 H4 的 neg_scale=30 hack。

---

### 第 2 层：对称目标发放率阈值（H13j 起）

```
threshold_mode = symmetric_target_rate
正阈值 = 负阈值 (neg_scale=1，完全对称)
target_rate = 0.05 → 总发放率向 5% 靠拢
max_threshold = 1.8 → 允许阈值大幅增长以削峰
activity_eta = 0.6 → 强活动惩罚
```

对比 H4-H9 的 `asymmetric_scale`：正负阈值独立缩放，无法统一控制总发放。H13 改为对称+目标发放率反馈。

---

### 第 3 层：Signed Consensus 注意力（H13b 起）

```
原来 (H9 compat_qk_product):  Q_token × K × Shiftmax(Q×K 浮点积) → 浮点输出

H13 signed_consensus_shiftmax: 
  1. Q_sign = STE(sign(Q))   ∈ {-1, 0, +1}
  2. K_sign = STE(sign(K))   ∈ {-1, 0, +1}
  3. consensus = (Q_sign · K_sign).sum(head_dim) / head_dim   ∈ [-1, +1]
  4. gate = Shiftmax(consensus × score_scale)
  5. output = K × gate
```

**硬件优势**:
- sign(Q)·sign(K): XOR(符号位) + AND(有效位) → 2 个门, 0 MAC
- sum over head_dim: popcount → N 个加法器, 0 MAC
- Shiftmax: 桶形移位器 + 前导1检测, 0 MAC
- 全程整数运算, 零浮点乘法器

**代码位置**: `bsa_attention.py:102-130` (`_signed_consensus_token_scores`), `bsa_attention.py:324-338` (forward path)

---

### 第 4 层：范围扩展（H13m/n）

| 变体 | Q/K 三元 | FFN 二元 | Downsample | 总模块 |
|------|---------|---------|-----------|--------|
| H13f/j | 24 | ~8 | 2 | ~34 |
| **H13m** | 24 | **27** (全 stage0-3) | **3** (全 stage0/1/2) | **51** |
| **H13n** | 24 | **14** (部分 even block) | **2** (stage0/2) | **~40** |

---

## 二、关键实验结果

### Guard120 (valid40，120 步)

| 变体 | AEE | AAE | SOPs | 范围 |
|------|------|------|------|------|
| **H13f** | **1.502** | **7.234** | 3.746G | bias-center Q/K + H9a FFN |
| **H13j** | 1.527 | 7.222 | 3.659G | + target_rate=0.05 |
| H13m | 1.573 | 7.564 | 3.628G | + all FFN/downsample |
| **H13n** | **1.500** | **7.365** | 3.651G | + partial FFN + st0/2 down |
| H13p | 1.541 | 7.797 | 3.590G | H13n + target_rate=0.02 |
| H13q | 1.579 | 7.524 | 3.649G | H13n + target_rate=0.035 |

### Full 训练

| 变体 | epoch | AEE | AAE | SOPs | Firing |
|------|-------|------|------|------|--------|
| **H13m** | 29 | **2.196** | **10.213** | 3.458G | 0.081 |

### H13m 训练过程中指标变化

| Epoch | thr_mean | act_mean | pos_mean | neg_mean | valid_loss |
|-------|----------|----------|----------|----------|------------|
| 0 | 0.537 | 0.092 | 0.082 | 0.010 | 1.144 |
| 5 | 0.446 | 0.093 | 0.083 | 0.011 | 1.091 |
| 10 | 0.393 | 0.092 | 0.081 | 0.011 | 1.283 |
| 15 | 0.353 | 0.091 | 0.082 | 0.010 | 1.319 |
| 20 | 0.319 | 0.091 | 0.080 | 0.011 | 1.282 |
| 25 | 0.307 | 0.087 | 0.077 | 0.010 | 1.293 |
| 29 | 0.303 | 0.089 | 0.079 | 0.010 | 1.186 |

**neg_mean 全程稳定在 1%** — 负发放问题已解决。

### 历史最佳对比

| 实验 | AEE | AAE | SOPs | 备注 |
|------|------|------|------|------|
| PSN baseline | 1.585 | 7.50 | 3.62G | 基准 |
| G1 gate-only | 1.606 | 7.25 | 2.71G | 最佳稀疏 |
| H9a compat_qk_product | 1.504 | 7.64 | 3.08G | 此前最佳 AEE |
| H9e half-blocks even | 1.498 | 7.68 | 3.28G | |
| **H13f guard120** | **1.502** | **7.234** | 3.75G | **AAE 史上最低** |
| **H13n guard120** | **1.500** | **7.365** | 3.65G | **精度最强** |
| H13m epoch29 | 2.196 | 10.213 | 3.458G | 全量退化 |

---

## 三、不足与缺口

### 1. 全量训练仍然退化

H13m 从 guard120 的 AEE=1.57/AAE=7.56 退化到 epoch29 的 AEE=2.20/AAE=10.21。

**根因**:
- `lambda_ang=0`（H13a/b 用过但又丢弃了）
- `max_threshold=1.8` 让部分 Q/K 阈值涨到 1.63，几乎完全关闭
- 全参数训练 + EPE-only loss = 权重漂移无方向约束

### 2. 没有隔离 Signed Consensus 本身的贡献

无纯 SOC（无 Shiftmax）消融。不清楚收益来自"符号共识评分"还是"Shiftmax 归一化"。

### 3. 没有正规 BSA 矩阵注意力

H13 的 signed consensus 是 token 级标量投票，不是 BSA 的 `Q@K^T` 矩阵乘。令牌间关系未被建模。H14 计划做但还没开始。

### 4. 负脉冲存着但参与注意力太少

neg_mean=1% vs pos_mean=7.8%。在 head_dim=32 的求和中，1% 的负脉冲贡献极微弱。

### 5. 硬件友好但缺数值验证

声称硬件友好（纯 popcount + 整数），但没有量化 SOPs 分解——没有和 compat_qk_product 的浮点 Shiftmax 做能源对比。

### 6. 可借鉴但缺失

| 来源 | 可借鉴 | H13 缺失 |
|------|--------|---------|
| BSA paper | V projection + Q@K^T 矩阵注意力 | H13 仍用 K 兼 V |
| TSN paper | 二元化 K（二值输出）减少噪声 | K 仍是三值连续量值 |
| ATLIF paper | `regularize_spike` 激活惩罚 | 用 activity_eta 替代但未分离正负 |
| I17 负反馈 | 负发放率独立反馈 | `negative_target_rate` 已实现但**未使用** |
| T5 SOC | 纯 popcount 比率，无 Shiftmax | 未做消融 |
| 渐进稀疏 | target_rate warm-up | target_rate 固定 |

### 7. 最关键的缺失：角度损失

`lambda_ang=0` 贯穿 H13m/n/p/q。**这是 H13m 全量退化的直接原因**——全参数训练 + EPE-only loss = 无方向约束。

---

## 四、建议补充的实验

| # | 补充 | 改动 | 预期 |
|---|------|------|------|
| 1 | **H13n + lambda_ang=0.2** | config 一行 | 约束全参数漂移，AAE 不退化 |
| 2 | **纯 SOC 消融**（无 Shiftmax） | bsa_attention 加 mode | 隔离贡献 |
| 3 | **渐进 target_rate**：0.15→0.03 | installer + config | 早期保持精度，后期稀疏化 |
| 4 | **负脉冲加权**：neg_weight=3 | consensus 公式 | 1% 负脉冲产生 3× 影响力 |
| 5 | **Stage 差异化阈值上限** | config groups | 保护深层不被过度剪枝 |
| 6 | **启用 negative_target_rate** | config 开关 | 独立控制负发放率 |
| 7 | **角度蒸馏**：baseline PSN 作 teacher | loss 加项 | 方向一致性约束 |

---

## 五、H 系列进化路线图

```
H3  二值 ATLIF (Q/K only, atlif_only 训练)
├── AAE 可控 (~8.4) 但 SOPs 不降
│
H4  三元 ATLIF (Q/K only, atlif_only 训练)  
├── 三元输出 + 不对称负阈值 neg_scale=30
├── AAE ~8.0, SOPs ~3.48G
│
H6  三元 ATLIF + 二值 FFN/downsample (全参数训练)
├── 全参数训练放开 backbone
├── AAE 爆炸到 21.6  ← 首次出现 AAE 危机
│
H8  FFN 块搜索 (全参数训练)
├── 短探针 AAE 低至 5.92 (valid10)
├── 全量 AAE 爆炸到 22.8  ← 确认 AAE 是系统性问题
│
H9  引入 Shiftmax 注意力 (compat_qk_product)
├── H9a: AEE=1.50, AAE=7.64, SOPs=3.08G  ← 首次 AAE 可控 + AEE 改善
├── H9e: AEE=1.50, AAE=7.68, SOPs=3.28G
├── H9c: AAE=31  ← stage2 FFN 全换=灾难
│
I16 尝试降低 neg_scale (1/2/4/8)
├── 短探针可行，全量仍然压制
│
I17 负发放率反馈控制器
├── neg_mean 维持 0.34-0.43% (30 epoch) ← 首次在全量训练稳定负发放
├── 但达不到 0.5% 目标
│
H13  偏置中心 + 签名共识注意力  ← 当前
├── 解决了 neg_scale=1 对称阈值下的负发放稳定性 (neg_mean=1%)
├── guard120 AAE=7.23 历史最低
├── H13m 全量退化 (AAE=10.21) ← λ_ang=0
└── H13n 全量进行中
│
H14  正规 BSA 矩阵注意力 (计划中)
├── Q@K^T → Shiftmax → V
```

---

## 六、当前运行状态 (2026-05-19)

- **H13n full**: 正在跑 (PID 400779, 开始于 14:27, epoch 3+)
  - Config: `h13n_biascenter_shiftmax_target05_halfffn_down02_full.yml`
  - 范围: 24 Q/K 三元 + 14 FFN 二元 + 2 downsample
  
- **H13m full**: 已完成 (epoch 29 checkpoint 已保存)
  - 最佳 epoch: guard120 (epoch 0)
  - 全量退化，不建议使用 epoch29
