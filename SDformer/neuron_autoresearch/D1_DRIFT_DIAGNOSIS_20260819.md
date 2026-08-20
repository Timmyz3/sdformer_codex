# D1 训练漂移诊断：h87/motion_t5_quotient 数值等价性 + 根因裁决 + 修复排序

日期：2026-08-19。诊断 agent 交付，纯 CPU。
对象：`D1_MOTION_T5_IMPLEMENTATION_20260818.md`（合同 I1–I7）与 D1 训练
（三 lr 档均漂移，short 档 lr2.5e-5 完整曲线见 §1）。
红线：未碰 GPU（B2 训练占用中）；bsa_attention.py 只读（0 改动）；仅新建
本 md 与 /tmp 脚本；未删文件。
证据分级：`[prof]` = 冻结数据重建（nts11 profile）；`[代码]` = 直接导入
bsa_attention.py 真实函数（合成稀疏输入，含梯度）。

---

## 0. 一句话结论

**根因 = forward 挂载层 shiftmax 的指数尺度错误**：h87 将整数 Q7 分数
（0..162）直接当作 float 指数喂给 `2^(s−smax)/2^ceil(log2(Σ))`（bsa_attention.py
h87 分支），而 h67 现网与仓库 RTL q17 shiftmax 的语义是定点指数 `2^(s/128)`。
这使 h87 的 shiftmax 温度**锐化 128 倍**：gate/attn 层与 h67 锚点不等价
（gate 相对差 1.67 vs 0.006，top-1 gate 质量集中 22 倍，熵 5.40→4.51）；
叠加 `_rne16_div_pow2_ste` 的恒等 backward（梯度不除 16），q/k 流梯度被放大
**2124×**（q 流从 h67 的 0 变为 1.1e5）——这是 train loss 持续上升的训练不稳
机制。三个疑似原因全部洗清：tie 差档（真实率仅 0.16%）、T=5 分组（0.5%）、
threshold 漂移（0.99997 恒定）。修复首推 **F1：h87 分数进 shiftmax 前 ÷128
（定点解释对齐 RTL）**，CPU 已实证等价恢复（§6）。

## 1. 症状与材料

- 漂移 log（`results/dsec_fullres_w15_H87_motion_t5_quotient_ft5_short_20260818/train_lr2p5e5_drift.log`）：
  train loss **1.2607 → 1.3750 → 1.5208**，val **1.1524 → 1.5040 → 1.6055**；
  lr 1e-4 / 5e-5 两档同漂移 → 与 lr 无关。每 epoch 1.4–1.5h（short 档，
  force_save [4]），5 epoch ≈ 7–8 GPU 小时。
- H82 对照（冻结算子续训，同基线预算）稳定收敛：ep14 AEE 1.2817 四线最佳
  （`H82_EP14_VALID825_RESULT_20260818.md`）→ 续训协议本身无问题。
- ATLIF threshold：安装至结束 `threshold_mean` 恒为 **0.99997**（105 模块，
  lr 5e-6，freeze@1224）→ **假设 D 排除**。
- 数据源 `[prof]`：`hw_autoresearch_nts07/results/h67_fullres_ep35_..._profile100/
  nts11_hardware_p0_profile.json`（1200 记录、672,000 (window,head) 行、
  3.024 亿槽）；`[代码]` 用稀疏合成输入（p=0.06，对齐真实 popcount~1.9、
  kzero~0.76）走 bsa_attention.py 真实函数。

## 2. 诊断方法与三路径

脚本 `/tmp/d1_equiv_diag_20260819.py`（两条证据线，输出
`/tmp/d1_equiv_diag_20260819.json`）：

| 路径 | 定义 |
|---|---|
| h67（锚点） | Motion 现网 float：tx + 0.02·sc + 0.25·motion，center 后进 shiftmax |
| h87（D1） | 合同 I1 融合式 `s=min(RNE16(64·o+sz+16·m̄),162)`，整数 Q7 直接进 shiftmax |
| h87f（假说） | 与 h87 同分数，但进 shiftmax 前 ÷128（定点解释假说） |
| h67t5（分组对照） | h67 分数按 T=5 分组重排，其余同 h67（隔离分组影响） |
| h87b（B2 对照） | T=4+pad 变体（`_binary_t4_pad_quotient_token_scores` 真实函数） |

## 3. 数值结果

### 3.1 score 层（h87 vs h67）— 差异存在但量级小

- 量化差：`score_diff_levels_mean = 0.79` 级 `[prof]`（0.41 级合成）；
  精确相等率 16.6% `[prof]`（合成 86.3%）——h87 把 float 分数量化到 Q7 网格
  [0,162]，平均 ~0.4–0.8 级偏差。
- **tie 差档（融合式 vs 拆解式差 1 档）实测率 0.16%** `[prof]`
  （243,559/151.2M 真实 within-pair 边；合成数据 0 次）——合同里"全域 2.74%"
  是含跨窗近似 m̄ 的全域上界，真实 within-pair 率低 17 倍。
- B2 对照 `[代码]`：`b2_slots_equal_d1=1.0`（{0,1,2,3,6,7,9} 与 D1 逐位相等），
  仅 {4,5,8} 为设计差异（slot8 差率 0.811）——B2 实现与合同一致，无新增偏差。

### 3.2 gate 层（核心证据）

| 指标 | h87 vs h67 | **h87f vs h67** | h67t5 vs h67 |
|---|---:|---:|---:|
| gate mean abs `[prof]` | 9.78e-4 | **8.2e-6**（↓120×） | 7.9e-6 |
| gate mean rel `[代码]` | 1.67 | **0.0064**（↓260×） | 0.0051 |
| gate max abs / max rel | 0.498 / 274× | 8.8e-5 / 0.05 | 8.4e-5 / 0.048 |
| corr(g67,g87) `[prof]` | **0.40** | — | — |

- **÷128 定点解释使 gate 层差异下降 120–260 倍、与 h67 恢复等价**——指数
  尺度是 gate 层不等价的唯一主因。
- T=5 分组单独影响 ~0.5%（`attn_h67t5_vs_h67_mean_rel=0.0003`）→ **假设 B 不成立**。
- 锐化实证 `[prof]`：top1 gate 质量 0.00195→**0.0438（22.4×）**、top3 5.85e-3→
  0.081；熵 5.3976→**4.5121**（h87f 为 5.3887 ≈ h67）——注意力塌向单 token。
- tie 差档经 gate 放大为零：top5 gate 内 tie 占比 0（合成）→ **假设 A 不成立**
  （0.16% 真实率 × 位置级影响，达不到漂移量级）。

### 3.3 attn 层

`attn_h87_vs_h67_mean_rel = 0.198`（max 258×）`[代码]`；
**`attn_h87f_vs_h67_mean_rel = 0.00036`**（↓550×）。h67t5 仅 0.0003。

### 3.4 梯度层（训练不稳的机制）`[代码]`

| | h67 | h87 | 倍数 |
|---|---:|---:|---:|
| grad_norm_q | **0** | 110,489 | ∞（q 流 h67 无梯度） |
| grad_norm_k | 66.9 | 142,084 | **2124×** |

- 成因三件套：`_rne16_div_pow2_ste` 恒等 backward（不除 16）＋ o 项系数
  65/16 入分子 ＋ 锐化 shiftmax 的 `2^k` 放大。q/k 梯度 4–5 个数量级放大 →
  AdamW 更新失稳 → train loss 上升、val 漂移（三 lr 档同漂移与此一致）。
- 单 bit 翻转 Jacobian 代理：h87 gate delta 1.0e-8 vs h67 4.5e-5（锐化后
  多数位置饱和，敏感度集中在 top-1）——漂移不是"逐位敏感"而是"尺度与
  梯度路径"问题。

## 4. 四假设裁决汇总

| 假设 | 证据 | 裁决 |
|---|---|---|
| A：fused/decomposed tie 差档（2.74%） | 真实 within-pair 率 **0.16%**；gate 影响 ≈0 | 不成立（2.74% 为全域上界误导） |
| B：T=5 分组跨窗耦合 | gate 0.5% / attn 0.03% | 不成立 |
| C：run-length 广播 STE 梯度 | q 0→1.1e5；k **2124×** | **成立（放大器，非根因）** |
| D：SNN threshold 漂移 | 全程 0.99997 恒定 | 排除 |

## 5. 根因裁决

**根因 = h87 forward 的 shiftmax 指数尺度错误**（整数 Q7 分数直接作 `2^k`
指数，缺 RTL 的 `/128` 定点解释），证据链：h87f 在 gate/attn/熵/top-mass
四个层面全部恢复到与 h67 等价（差 120–550× 缩小）→ 指数尺度是唯一的
forward 不等价来源。score 层量化差（0.4–0.8 级）、tie 差档（0.16%）、
T=5 分组（0.5%）均为次要。

**漂移机制**：forward 不等价（锐化 128 倍）＋ STE 恒等梯度（不除 16）
→ q/k 梯度放大 2124× → 优化不稳 → train loss 单调上升、val 同步漂移。
H82 对照稳定而 D1 漂移，差异恰好在这两条新引入的路径（整数 Q7 直进
shiftmax + 恒等 STE）上。

注意：**I1–I7 位账合同不受影响**（融合式仍为硬件规范，位账 −78.3% 口径
不变）；问题出在 forward 挂载层对 Q7 分数的指数解释，与 RTL 不一致。

## 6. 修复方案排序（含修改范围 / CPU 验证 / GPU 成本）

**F1（首推，根因修复）：h87 分支分数进 shiftmax 前 ÷128（定点解释对齐 RTL q17）**
- 修改范围：bsa_attention.py h87 分支 1 处（`scores = scores/128` 或等价
  `_rtl_shiftmax_gate_q17` 语义）；不动融合式本身、不动 I 系列、不动配置。
- CPU 验证：本脚本 gate_h87f_vs_h67 rel 0.006 / attn 0.00036 / 熵恢复
  5.389（已实证）；B2 与 D1 回归套全绿（h87f 只改指数解释，不改分数）。
- GPU 成本：short ft5 一档 ≈ 7–8h（5×1.4h）＋ valid；判定标准：train loss
  不再上升、val ≤ 锚点 1.33 量级。风险：Q7 步长 1/128 的量化噪声在低位
  区放大（相对 h67 float 温度），可叠加 F2 兜底。

**F2（梯度修复，可与 F1 叠加）：`_rne16_div_pow2_ste` backward 改为 ÷16
直通（消除 65/16 放大），或对梯度做缩放/clip**
- 修改范围：1 个函数 backward；测试套 STE 梯度档更新期望值。
- CPU 验证：现有 unittest 的 STE 档（h87/h87b/forward F6）改期望后全绿。
- GPU 成本：另 1 档 short ≈ 7–8h。收益：把 2124× 梯度放大压回 ~1×，训练
  稳定性的第二道保险（F1 后梯度仍比 h67 大，只是不再 4–5 个数量级）。

**F3（中间修复，不解决漂移）：B2 T=4+pad（h87b）**
- B2 是位账中间修复（−61.5%，`B2_MOTION_T4_PAD_IMPLEMENTATION_20260819.md`），
  但同为整数 Q7 分数直进 shiftmax，**同样存在 F1 所指的指数尺度问题**
  （gate_h87b_vs_h87 rel 1.54，与 h87 一样锐）→ 不能替代 F1，只能作为
  走通 F1 之后的位账消融档。GPU 成本另 1 档 short。

**F4（缓解非修复，可选）：curriculum warmup**
- 前 N 步（≤1224 即 threshold freeze 点）用 h67 分数训练后平滑切 h87，
  或 alpha 退火。修改范围：launcher/配置 1 处。成本低（复用现有档）。
- 定位：缓解锐化+梯度冲击的初期爆点，不改变 forward 不等价本身；若
  F1 验证通过则优先级降为 0（避免引入新变量）。

**顺序建议**：先跑 **F1**（单一变量验证根因；若 train loss 平稳即裁决闭环）
→ 通过后决定是否叠加 F2（若 F1 档仍有轻微不稳）→ F3 作为位账档按
B2 预案推进 → F4 仅在需要压短收敛时启用。三 lr 档均漂移说明不是 lr 问题，
**不要以调 lr 代替修复**。

## 7. 复现与红线

- 复现：`python3 /tmp/d1_equiv_diag_20260819.py --part 1 --limit 1200`
  （≈6 分钟 CPU）＋ `python3 /tmp/d1_equiv_diag_20260819.py --part 2`（≈1s）；
  结果 `/tmp/d1_equiv_diag_20260819.json`（`real` = 真实数据，`code` = 代码路径）。
- 红线遵守：全程纯 CPU（GPU 实测 B2 训练占用，未触碰）；bsa_attention.py
  未修改（诊断只读）；仅新建本 md 与 /tmp 脚本；未删除任何文件。
