# Motion/H67 quotient-file 侧车：exact 设计与证据锚定计划

日期：2026-08-18。本文件是 ROUND2 攻击（`CLAUDE_INNOVATION_ATTACK_ROUND2_MOTION_20260818.md`）裁决的幸存候选（
`CONDITIONAL_PROFILE_GATE_SUPPORT_ONLY_NO_RTL`，创新 3.0–3.2 → +SAIF/PPA 3.5–3.7）的推进设计。
红线遵守：CPU-only、不写 RTL、不改任何现有文件（`docs/359`、selector、生产 RTL、194436Z 不动）、
不碰 GPU。证据分档沿用项目惯例：`[rtl]` / `[prof]` / `[模型]`；模型数字不冒充周期。
本文件只新增一个 md，作为"写 sidecar RTL 之前的 exact 设计冻结与证据清单"。

前置锚点（全部来自冻结 ep35 全量 672,000 行 `[prof]`，ROUND2 文档 §1/§3，不重测）：

- 真窗口 C（occupied score class）：p50=3 / **p95=16** / p99=19 / max=46 / mean=5.116；
- descriptor/token：**p95=0.578**（≈260 条）/ mean=0.511（≈230 条）；per-row desc max=435
  （docs/406 同数据集口径 p95=265，见 §5 前置 2 的对账）；
- equal-pair 率 E/P：mean 219.5/225=0.9756（ep30），逐 stage 0.9952/0.9971/0.9590/0.8795；
- 三腿门：C p95≤192 ✓（16）、descriptor/token≤0.60 ✓（0.578，保守 0.589）、
  state 相对 fused pair-gather 降 ≥20% ✓（−23.4%/−26.1%，513 口径）;
- 周期腿：450+C+desc = 696 vs 450+C+PAIRS = 691 → +0.7% 持平，**不主张周期**；
- 现网贡献身份：TESC（数学合同）+ RQTB（物理流接口）一条贡献，公平锚
  `112589/94891 = 1.1865×` `[rtl]`（docs/359/362/263）。

---

## 1. 对象合同（quotient 流字段级定义）

### 1.1 侧车对象的五个存储单元

| 单元 | 字段 | 位宽（Motion 档位口径） | 位宽（H82 兼容保守口径） | 说明 |
|---|---|---|---|---|
| occupancy | 每窗口 occupied class bitmap | **163 bit**（MAX_SCORE=162 → 163 bins） | 513 bit（H82 bin 数） | 取代 C7 的 class hist + 450 带分名单；无 multiplicity 字段 |
| compact gate file（quotient-gate 变体） | 每 occupied class 一条 `gate_q17` | **9 bit** × C | 9 bit × C | 现网 `h67_certified_gate_quant_q17` 同宽；每类写一次 |
| quotient descriptor 流 | `class_id` | **8 bit**（0..162） | 9 bit（513 类） | 取代"450 带分 token 名单" |
| quotient descriptor 流 | `k_mask` | **2 bit**（K0/K1 active，与现网 `active_k_mask` 同义） | 2 bit | 保持 destination identity |
| quotient descriptor 流 | `pair_last` | **1 bit**（pair 内最后一条 descriptor） | 1 bit | 使 expand 端可恢复 temporal membership（common=2 / split=1） |
| denominator certificate（denom-only 变体） | `row_max` + `denom_shift` | **9 + 5 = 14 bit** | 同左 | 取代整份 class gate 物化；expand 按 class_id 重算 exp2 |

即每条 descriptor record = `class_id[8] + k_mask[2] + pair_last[1]` = **11 bit**（513 口径 12 bit）。

### 1.2 总数上界（合法域 vs workload 锚定，两类分开）

- **容量上界（合法域，写 RTL 的 SRAM 尺寸依据，不做 profile 拟合）**：
  `D ≤ 2P = 450`（与现网 `MAX_DESCRIPTORS = 2*PAIRS` 参数一致）、`C ≤ 163`（Motion bins）。
  容量必须按合法域定，不能按 profile max 静态声称 exact（教训来自 docs/389 §4：四码表被全量
  max=63 证伪）。
- **workload 锚定（能量腿与门限依据）**：冻结 ep35 全量 672,000 行实测 `C p95=16`、
  `D p95=260`（保守取 265，见 §5 前置 2）、`D mean=230`。门限语义：p95 是"workload 代表性"门，
  不是容量门。

### 1.3 存储对象 bit 账（公式）

记 `O`=occupancy 宽（163 或 513），`W`=class_id 宽（8 或 9），`C`=occupied class 数，
`D`=descriptor 数，`P=225`，`T=450`，`W_g=9`，`W_k=2`，`W_l=1`，`W_rm=9`，`W_ds=5`。

```
fused pair-gather 强基线（docs/445 定义：同一 one-vote 数值语义、固定 pair record、
   读 K 时在线 gather compact class gate、不写 token-gate）：
   B_pair  = O + C·W_g + P·(2·W + 1)

quotient-gate 变体：
   S_qg    = O + C·W_g + D·(W + W_k + W_l)

denom-only 变体：
   S_do    = O + (W_rm + W_ds) + D·(W + W_k + W_l)
   expand 按 class_id 重算 exp2，不存 gate（expand_exp2 = D 次/窗）

对照（弱基线，仅上下文）：现网 C7 物化 = O + C·W_g + T·W + T·W_g（token-gate 物化，
   docs/445 已明确其不是强基线）。
```

### 1.4 Motion 实测分布算出的最终数字

**Motion 档位口径（O=163, W=8, descriptor=11 bit）**：

| 点 | 参数 (C, D) | pair-gather 强基线 | quotient-gate | denom-only | qg 降幅 | do 降幅 |
|---|---:|---:|---:|---:|---:|---:|
| p95 联合点（ROUND2 口径） | (16, 260) | **4132** | **3167** | **3037** | **−23.4%** | **−26.5%** |
| p95 保守点（406 口径 D=265） | (16, 265) | 4132 | 3222 | 3092 | −22.0% | −25.2% |
| mean 点 | (5, 230) | 4132 | 2837 | 2707 | −31.3% | −34.5% |
| p99 点 | (19, 323) | 4159 | 3887 | 3730 | −6.5% | −10.3% |
| 合法域最坏行 | (163, 450) | 5455 | 6580 | 4962 | **+20.6%（反超）** | −9.0% |

**H82 兼容保守口径（O=513, W=9, descriptor=12 bit，与 ROUND2 §3 完全一致）**：
p95 联合点 baseline **4932**、quotient-gate **3777**（−23.4%）、denom-only **3647**（−26.1%）；
mean 点 −30.7% / −33.4%。

**诚实边界（写进论文的）**：≥20% 门在 p95 与 mean 点通过；p99 处降至 −6.5%/−10.3%
（descriptor 项主导）；合法域最坏行 quotient-gate 反超基线——门的口径是"p95 工作量代表性"，
不是"每行都赢"，能量主张同样按分布而非最坏行。

descriptor SRAM 宏粒度：450×11 = 4,950 bit → fakeram45_256x32（8,192 bit）单宏可容纳，
与现网 K store 同族宏（见 §4.2）。

---

## 2. 执行合同

### 2.1 K 恢复顺序（pair 序，bit-exact 约束）

与现网 `h67_temporal_quotient_shiftmax_gate_top` 输出边界逐项一致（`[rtl]` 接口，只读引用）：

1. pair 以 scan 序 `pair_id = 0..224` 进入（双时间平面交错）；quotient 发射保持 pair 序
   （现网 `h67_temporal_score_quotient` 保序 + FIFO 保序）。
2. 每条 descriptor 在 expand 时按 `k_mask` 选 K0/K1 双 bank（`h67_sync_dual_bank_k_store`）；
   K bank 内容与读事务与现网完全相同（docs/263：Fixed/RQTB 的 K read bits 相同）。
3. 输出 token 序：每 pair 先 time0（`token_id = 2·pair_id`）后 time1（`2·pair_id+1`）；
   `pair_last` 终止 pair；`window_seal/window_done` 握手、`out_last`、`threshold_q8`
   （`cfg_preserve_mean` 0/1 两档）全部原样保留。
4. 侧车不改变任何 K 数据、不合并 K、不做 class-wise K folding（389/433 已否决的类别）。

### 2.2 one-vote normalization 在 Motion Q7 档位下的硬件语义

H82 一类一票合同（docs/445、`h82_multiplicity_free_quotient_model.py`）迁移到 Motion Q7 档位：

```
Z = Σ_{c ∈ occupied} exp2(c − row_max)，禁止乘 multiplicity；
row_max = max over occupied classes（multiplicity 不影响 max → 与现网 row_max 逐位一致）；
exp2 用现网同款整数 LUT（1/16 分数步，score 域 0..162）；
denom_shift[5] = 分母累加器的 Q 点，使 quant_q17 复刻现网 `certified_gate_quant_q17`
    的取整位置（round 位置必须逐 bit 复刻，见 §3）；
gate_q17 每 class 算一次，common descriptor 的两个 token 共用（同现网按类算 gate）。
```

Mode B（one-vote）的 `Z` 与现网 C7 的 `Z_C7 = Σ exp2(c−rm)·mult(c)` 数值不同
（equal-pair 率 97.5% ⇒ 几乎每窗不同）——这是**新数值合同**，不是 RQTB 换名（见 §3/§5）。

### 2.3 与 TESC 的接口（侧车不能吃掉 TESC 的身份）

- TESC = 数学合同（无损 post-quantization temporal score quotient、归一化域保留
  multiplicity、gated-K 边界迟展开）；RQTB = 该合同的物理流接口（common/split slot →
  weighted SCS → Shiftmax → active-K bank → 原序展开）。docs/263：**二者只能合并为一条贡献**。
- 侧车的插入位置：`h67_temporal_quotient_scs_frontend` 之后的 **directory 物化层**——
  把 "class hist + 450 带分名单 + exp×multiplicity 直方图更新" 换成
  "occupancy bitmap + quotient descriptor 流 + denominator certificate"。
  侧车**不改** quotient 发射规则（`D_min = 2P − E` 逐 pair 达到，docs/406 `[rtl]` 账本），
  **不改**归一化域语义（Mode A）或明确声明数值合同变更（Mode B），**不改** gated-K 输出边界。
- 措辞边界（对 docs/359 封存列）：`112589/94891=1.1865×`、slot `−45.09%`、equal `28001`
  全列不动；侧车**不新增贡献名**、不新增缩写；论文中只能作为"同一 quotients 流的目录
  物理实现变体"，在 **energy 腿**下呈现（周期腿不报新数字，+0.7% 持平）。
- 计数器口径：复用现网 perf 总线 8 项（`perf_pairs/quotient_descriptors/original_tokens/
  active_entries/equal_pairs/class_transactions/exp_transactions/emitted_tokens`），
  保证 SAIF 对比与现网 VCD 同口径。

---

## 3. exact 边界（与现网 C7 exp×multiplicity 逐项对照）

两个模式，措辞必须分开：

**Mode A（C7-exact 存储侧车，默认路径）——bit-exact 等价现网**：

| 路径 | 现网 C7 | 侧车 Mode A | 等价性 |
|---|---|---|---|
| 发射的 descriptor 集合 | `{score, temporal_mask, active_mask}`，D=2P−E | `{class_id, k_mask, pair_last}`，D=2P−E，同 pair 序 | **bit-exact**（pair_last 编码 temporal membership：pair 单条 ⇒ common=2，双条 ⇒ split=1；成员 popcount 恒 2P） |
| row_max | SCS 类 max | certificate `row_max[9]` | **bit-exact**（max 不受 multiplicity 影响） |
| 分母 | Σ exp(c−rm)·mult(c)，mult=popcount(mask) | 展开期每 descriptor 加 exp 一次/两次（common 加两次 = exp+exp），证书 `denom_shift` 复刻现网 Q 点 | **bit-exact**（整数 LUT 加法交换律成立；前提：逐项不加中间舍入，round 位置复刻现网 RTL，由 miter 证明） |
| gate_q17 | 每 class 一次 quant_q17 | 同 | **bit-exact** |
| K 恢复与 token 序 | 双 bank + active mask + 原序 | 同 | **bit-exact** |
| threshold 截断 | cfg_threshold_q8 | 同 | **bit-exact** |
| 新增对象 | — | occupancy bitmap + descriptor 流 + certificate；移除 hist/450 名单/token-gate 物化 | **仅存储对象改变**，执行语义保持 |

**Mode B（one-vote 数值侧车，H82 一类一票迁移）——新增数值合同，不等价现网**：

| 路径 | 现网 C7 | 侧车 Mode B | 差异 |
|---|---|---|---|
| descriptor 流 / row_max / K 序 / token 序 | 同 A | 同 A | bit-exact（可逆性、destination identity、pair 序全保） |
| 分母 | Σ exp·mult（equal pair 计 2） | Σ exp，禁乘 multiplicity | **数值不同**（E/P≈0.975 ⇒ 几乎每窗不同） |
| gate_q17 | 现网值 | 新值 | **不同**；且 gate 平移可能翻转 threshold 截断 → 发射 token 集合可改变（K 输出边界随之改变） |
| 准入前提 | — | 需要算法侧 one-vote 语义的 accuracy 证据（H82 硬件-order AEE 门为模板） | **跨线依赖**，Motion 本线不自行裁决 |

**结论一句话**：Mode A 是"同算子、新存储对象、可证明 bit-exact"的诚实侧车（energy 腿载体）；
Mode B 是"同 quotient 流、新归一化合同"——它才是与 H82/H86 算法线接线的对象，Motion 侧车
只在不依赖现网 gate 边界时才可用 B。两个模式都不构成 4.0（433：无新算法算子合同）。

---

## 4. 证据锚定计划（分步、每步证据等级与门槛）

### 4.1 第一步：CPU golden 参考与三方 miter 前置 `[模型]`（纯 CPU，写 RTL 前）

- 从冻结 ep35 profile 的 ordered count 独立重算 Q7 score（方法同 ROUND2 `/tmp` 脚本），
  重建 descriptor 流、row_max、Mode A/B 两种分母、gate_q17、token/K 序列；
- 与现网 138 行公平包 RTL 账本（docs/406 逐行 descriptor 账本 + `[rtl]` 联合 Acc32 结果）
  逐行对照：**descriptor 集合 0 mismatch、Mode A 的 gated-K 输出 0 mismatch**（Mode B 只对照
  结构项，数值项记录差异分布）——这是 sidecar RTL 的 golden；
- 同时产出 one-vote vs C7 的差异证据包：gate_q17 差值分布、threshold 翻转导致 token 发射
  集合改变的行数占比（预注册门槛：报告即可，不做门）。

### 4.2 第二步：SAIF 能量锚定 `[待验证]`（新思机器，写 RTL 后）

- 宏库：**fakeram45 族**（现网 `fakeram45_256x32_bb.sv` 同族宏，descriptor SRAM 450×11
  → 256×32 单宏；directory/gate file 与 K store 同宏族），逻辑用 Nangate45 开放库
  （docs/264 同流程，时钟 5 ns）；
- corner：**TT 25°C 标称 + SS/FF 敏感**，5 ns；
- 对照三方：Fixed2S / RQTB2S（现网）/ sidecar，同一 LFSR 公平 replay + 同一确定性反压
  （docs/264 §3.1 公平边界 9 项）；VCD→SAIF 在 **score+directory+SCS+expand 组件边界**；
- **energy ≥15% 的证法**：先给组件动态能量（SAIF-annotated，活动 bit 来自同一 VCD 回放）
  sidecar vs RQTB2S 的对比，≥15% 才算过门；同时必须给出总 row 能量占比（诚实披露
  组件/总行边界，不把组件腿写成整行/encoder 腿，docs/262 口径）；
- 翻盘预案：若组件能量 <15%，主张降级为 storage bit 账 + workload 统计（3.0–3.2 档），
  不后验改门。

### 4.3 第三步：同端口宏面积 / Fmax 门 `[待验证]`（新思机器）

- 对照 RQTB2S（docs/264：278,348 um²、5ns post-route WNS +0.0686 ns，均按该报告口径）；
- 门：逻辑+宏面积 ≤ +10%，Fmax ≥ −5%（docs/445/433 合同）；
- 记账要求：builder metadata、descriptor SRAM、denominator certificate、occupancy bitmap
  全部计入，与 K store/FIFO 同一端口模型（不许只报目录差分）。

### 4.4 第四步：miter 计划（sidecar RTL 与现网三方，写 RTL 后 `[rtl]`）

- **flat**：138 行 LFSR 公平包（与 1.1865× 同一包），逐行 gated-K + Acc32 checksum
  0 mismatch；preserve_mean 0/1 各一遍；
- **banked**：双 bank K + descriptor SRAM bank 冲突注入 + **随机反压**（docs/445 要求），
  0 mismatch；
- **compiled**：12-block 全 record（S0..S3 串联）回放，输出通道联合 Acc32 0 mismatch；
- directed corner：max-descriptor 行（D=435）、空行、全等行、全 split 行、threshold 扫
  （cfg_threshold_q8 边界）、seal 竞争；
- 样本规格：与 138 行公平包同 LFSR seed 集，另加 3 个随机 seed 包（记录级）。

---

## 5. 审稿人三杀 + 应对

**杀 1：「为什么不是 RQTB 换名？」**
应对锚定两条硬证据：(a) 存储对象合同层面改变——bit 账可审计：
`B_pair=4132 → S_qg=3167（−23.4%）/ S_do=3037（−26.5%）`（163 口径），且与现网 C7 物化
（含 450 名单 + token-gate）差异更大；换名不换对象，侧车换对象不换算子（433 的
"新算子 + 同时改存储/执行对象"中存储腿占住，算子腿诚实放弃 → 天花板 3.5–3.7 非 4.0）；
(b) Mode B 的 one-vote 与现网 pair-quotient 的**数值差异证据**：equal-pair 率 97.5% 下
两分母几乎每窗不等，§4.1 的 gate 差值分布与 threshold 翻转统计直接量化"不是同一对象换名字"。
论文措辞：sidecar = "同一 quotients 流的目录物化变体"，不新增贡献名。

**杀 2：「周期持平为什么还能有 energy 主张？」**
周期腿 +0.7% 持平反而排除周期撒谎风险（无周期表可写）。energy 腿的载体是**活动位宽差**：
每窗目录写活动 = occupancy 置位 + D 条 11-bit descriptor（≈16+2860 bit）vs 现网 450 带分
名单 + token-gate 物化写（450×9 + 450×9）或 pair-gather 的 225 条 17-bit 固定记录写
（3825 bit）；SCS 直方图更新 bit 与 exp 事务（现网已有 −22.04% 的 VCD 观测，docs/263 §3.3）
一并计入。主张口径硬约束：**组件动态能量 ≥15%（同端口、同宏、同反压、SAIF 实测）**，
组件边界与总行占比如实披露；SAIF 不过门就退到 bit 账档。周期的 +0.7% 在论文中作为
"持平（non-negative）"如实陈述，不与 energy 合写成 EDP 表。

**杀 3：「0.578 的门余量会不会被多样本翻盘？」**
分四层：(a) 口径对账先行——ROUND2 的 0.578（≈260）与 docs/406 的 p95=265（0.589）差
1.9%，设计采用**保守 0.589**，余量收窄到 1.8%，但仍在门下（§5 前置 2 先对账后定稿）；
(b) 翻盘条件量化：desc/token p95 破 0.60 需 E/P<0.80；冻结 ep35 逐 stage E/P
0.995/0.997/0.959/0.879，最深的 S3 也只到 0.879（→ 0.560），100 样本全量 p95 已含深尾
（desc max=435 的行已存在，p95 对 1200 record/sample 粒度稳健）；(c) **预注册降级规则**：
任何未来样本 p95>0.60 → 能量主张自动降为仅 storage bit 账（3.0–3.2），不后验改门；
(d) 容量是合法域（450/163），spill 无损，门只约束"workload 代表性"，不约束正确性。

---

## 6. 写 sidecar RTL 之前还缺的前置证据（3–5 项）

| # | 前置证据 | 等级 | 执行域 |
|---|---|---|---|
| 1 | **CPU golden 参考 + 现网账本对齐**：从冻结 ep35 profile 独立重建 descriptor 流/row_max/两种分母/gate_q17/token 序，与 138 行公平包 RTL 账本逐行 0 mismatch（Mode A）；附带 one-vote vs C7 差异统计包（§4.1） | `[模型]`（golden 的数值链挂靠现网 `[rtl]` 账本） | **纯 CPU** |
| 2 | **descriptor p95 口径对账 + 逐 stage 分布**：0.578 与 265（0.589）归因（ratio-of-p95 vs p95-of-ratio、per-window vs per-head-row）；输出 C/desc/E–P 联合分布与 S3 密行上界，定稿 4.1 的保守 D 值 | `[prof]` | **纯 CPU**（冻结 profile 已在盘上，不重测全量，只重放统计） |
| 3 | **容量与 spill 语义冻结**：合法域 450/163 下的最坏行 bit 账与周期边界（p99 −6.5%/−10.3%、最坏行反超表 §1.4）、fakeram45 宏粒度映射（4,950 bit → 256×32）、同端口记账表 | `[模型]` | **纯 CPU** |
| 4 | **端口契约冻结（RTL 前的唯一设计冻结）**：以 `h67_temporal_quotient_shiftmax_gate_top` 接口为边界（pair_in → out 8 项 + perf 8 计数器）、default-off 插入点、preserve_mean/threshold/seal 语义、denom_shift 与 round 位置逐 bit 复刻声明 | `[模型]`（文档冻结） | **纯 CPU** |
| 5 | **SAIF/PPA 同宏基线**：Fixed2S/RQTB2S/sidecar 三方同宏 VCD→SAIF 能量基线（fakeram45 + Nangate45，TT/SS/FF，5ns）与面积/Fmax 门执行（§4.2/§4.3） | `[待验证]` | **新思机器**（docs/433：DC/STA/SAIF/PTPX 在另一台有目标库服务器） |

GPU profile 任务：**本线为零**——Motion 冻结 ep35 profile 已落盘，全部统计可 CPU 重放；
若 Mode B 需要 one-vote 语义的 accuracy 证据（H82 硬件-order AEE），那是 H82 算法线（H82
rank-1 GPU）的交付物，Motion 侧车只引用不自行裁决。

---

## 7. 主张边界总表

| 主张 | 状态 |
|---|---|
| 三腿 profile 过门（C p95=16 / 0.578–0.589 / −23.4%~−26.5%） | `[prof]`+`[模型]`，已过（ROUND2） |
| Mode A 与现网 bit-exact（descriptor 流/row_max/分母/gate/K 序） | `[待验证]`（§4.1 golden + §4.4 miter） |
| 组件动态能量 ≥15% 同端口 | `[待验证]`（新思 SAIF，§4.2） |
| 逻辑+宏面积 ≤+10%、Fmax ≥−5% | `[待验证]`（新思，§4.3） |
| 创新分：过门+sidecar exact 3.0–3.2；+SAIF/PPA 同宏 3.5–3.7 | 与 ROUND2/445 一致 |
| 4.0 | **NO**（无新算法算子合同；解锁需算法侧新合同，另一条线） |

本文不修改 `docs/359`、selector、生产 RTL、194436Z；未抢 GPU；无 RTL 文件产出。
