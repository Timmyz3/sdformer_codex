# Motion/H67 quotient-file 侧车 Mode B（one-vote normalization）硬件对象设计

日期：2026-08-19。证据等级：`[模型]`（数值链挂靠 golden `[模型]+[prof]` 与现网 `[rtl]` 只读端口）+ `[prof]`（冻结 ep35 重放）。
红线：纯 CPU、不写 RTL、不碰 GPU（Mode B valid825 GPU 实验只排队不执行）、不改任何现有文件；只新增本 md。
母文档：`docs/CLAUDE_MOTION_SIDECAR_DESIGN_20260818.md`（设计文档）、`docs/CLAUDE_MOTION_SIDECAR_CAPACITY_PORTS_20260818.md`（证据 ③）、`docs/CLAUDE_MOTION_SIDECAR_PORT_CONTRACT_20260818.md`（证据 ④）、`dc_handoff/CLAUDE_SIDECAR_SAIF_PPA_RUNBOOK_20260818.md`（runbook §11：Mode B GPU 评估规格）。
golden：`results/motion_sidecar_golden_evidence_20260818/`（golden.py / report.json / report.md）。

## 0. 冻结结论一句话

Mode B 是 Mode A 同一目录数据通路的**参数化数值变体**：五个对象中三个零 diff 复用（occupancy bitmap、descriptor 流、row_max），一个语义变（denominator certificate：累加器从 `exp×mult` 改为 `exp×1`，denom_shift 变为新 Q 点），**一个对象消除**（per-class count 文件 163×9=1,467 bit 全部去掉，class 流 `class_multiplicity` 常量 1）。Mode B 相对 Mode A **存储 −22.2%、无任何位宽增长**（gate 饱和 256 是现网既有 clamp 哨兵值，非新格式）；能量侧净符号不定（省 count RMW 与乘法，但 t=128 下发射活动增加），**Mode B 不做独立能量主张**。全部设计元素绑定 GPU 精度门：**AEE ≤ 1.3430 才生效，FAIL 则整体归档 HOLD（不进 RTL）**。

---

## 1. Mode A 对象构造回顾（diff 的基准，证据 ③ 冻结）

| # | 对象 | 容量/位宽 | 宏 | 角色 |
|---|---|---|---|---|
| 1 | occupancy bitmap | 163 bit flop RF | flop 寄存器文件 | 取代 C7 class hist + 450 带分名单；build 期按位置位 |
| 2 | quotient descriptor 流 | 450×11 = 4,950 bit | fakeram45_256x32 单宏（225 字 × 2 条/字） | `{class_id[7:0], k_mask[1:0], pair_last[0]}`；D=2P−E，pair 序 |
| 3 | denominator certificate | 14 bit（row_max[8:0] + denom_shift[5:0]）| flop | Mode A：denom_shift 复刻现网 C7 Q 点（bit-exact 义务）|
| 4 | **per-class count 文件** | **163×9 = 1,467 bit** | flop/小 RAM | 证据 ③ 对账项 #1：保 top 的 `row_sum += exp × class_multiplicity` 与 C7 逐位一致；build 期每 descriptor 1 次 RMW（Mode A：+= popcount(temporal_mask)）|
| 5 | 目录状态机 | — | — | build/class/expand 相位，class 流/active 流发射 |

K store、slot FIFO、exp LUT、gate quant、emit 后端、8 项 perf = 现网原封（证据 ④）。Mode A 的 golden 链：672,000 行 11 项 0-mismatch（`Z_A == Z_C7` 逐行成立）。

---

## 2. 对象级 diff 清单（Mode B vs Mode A）

| 对象 | Mode A | Mode B | diff 定性 |
|---|---|---|---|
| occupancy bitmap | build 期置位 | **复用，零 diff** | occupancy 与 multiplicity 无关（occupied 集合两模式逐位相同）|
| quotient descriptor 流 | D=2P−E，pair 序 | **复用，零 diff** | D、k_mask、pair_last、token 序全由发射规则决定，与归一化无关 |
| row_max | certificate row_max[8:0]，与 C7 逐位一致 | **复用，零 diff** | max 不受 multiplicity 影响（golden：逐位一致）|
| denominator certificate | 累加器 `Σ_class exp(c−rm)·count[c]`（需乘 count）；denom_shift = ceil_log2(Z_C7)，**逐 bit 复刻现网 Q 点**（bit-exact 义务） | 累加器 `Σ_class exp(c−rm)×1`（**乘法消除**）；denom_shift = ceil_log2(Z_B)，**新数值合同，无 C7 bit-exact 义务**；miter 目标改为"与 golden ds_B 0-mismatch" | **语义变**；端口 `denominator_shift[5:0]`/`row_max[8:0]`/`certificate_valid/pass` 原样（位宽不变）；内部累加器 17→16 bit（Z_B ≤ 41,728 < 2^16，见 §4.2）|
| **count 文件** | 163×9 = 1,467 bit；build 期每 descriptor 1 次 RMW；class 相位按 `count[c]` 发射 multiplicity | **消除（elaboration 级不实例化）**；class 流 `class_multiplicity ≡ 1`（常量，9-bit 总线恒 1） | **存储 −1,467 bit；RMW 活动（D 次/窗）+ C 次读全消**；class 流接口本身不变（证据 ④ 契约：class_score/class_multiplicity/class_last 信号与位宽原样）|
| class 流（目录→top）| `{class_score, class_multiplicity=count[c], class_last}` | `{class_score, 1, class_last}` | 仅 multiplicity 数据源变（Mode A=count[c]，Mode B=常量 1）；top 的 row_sum 硬件原封 |
| gate quant（top 共享）| `min(256, RNE(exp×57,600 >> ds))` | 同一公式，ds 更小 → 饱和更频繁（见 §4.1/§4.2） | **top 零 RTL 改动**；行为差异即数值合同本身 |
| K 恢复 / 发射边界 | 与 C7 逐位一致（138 行账本锚）| 发射集改变（t=128 全 450 token 翻，见 §4.2）| **Mode B 的验收对象 = 自 golden，不比对 C7 账本**（§6/§7）|
| perf 8 项 | 单源，与现网同口径 | 单源；`perf_emitted_tokens` 数值不同（语义不变）| 零 RTL 改动 |

**diff 总括**：Mode B 不新增任何对象、不扩大任何位宽；全部改动收敛为三个点——(a) certificate 累加器输入语义（乘 count → 乘常量 1），(b) count 文件消除 + class_multiplicity 常量 1，(c) denom_shift 值域变化（workload 下 ds_B ∈ [8,14]）。目录主状态机（build/class/expand 相位、bitmap 升序扫描、descriptor 打包、fail-closed 容量检查）两模式完全共用。

**参数化**：`SIDECAR_DIRECTORY`（0/1）× `MODE_B`（0/1），合法组合：(0,*) = 现网 C7；(1,0) = Mode A；(1,1) = Mode B。全部 elaboration 级静态选择（同证据 ④ default-off 惯例），无运行时 mux。

---

## 3. one-vote normalization 的硬件语义

### 3.1 denom_shift 全负（p50=−7，range [−9,−3]，zero=0.00%）的物理含义

- 数值事实（golden `[模型]`）：Z_C7 = Σ exp·mult（加权和，equal 对计 2），Z_B = Σ exp（每 occupied class 一票）；672,000 行 **100% 窗口 Z_B ≠ Z_C7**，ΔZ=Z_C7−Z_B 的 rel mean 98.85% ⇒ **Z_B ≈ 0.011×Z_C7**（典型行约 1/100）。
- 硬件含义：`denom_shift = ceil_log2(Z)` 是分母累加器的 Q 点（gate = exp×57,600 >> ds 的取整位置）。Z_B 比 Z_C7 小两个数量级 ⇒ **ceil_log2 结果系统性右移 ~7 bit**（Δds 全负、p50=−7、100% 非零）。右移 7 ⇒ gate 数值整体放大 ~2^7=128×：这就是"分母右移，gate 变大"的物理机制。
- 位宽影响：**denom_shift 端口不变**。合法域 Z_B ∈ [256, 41,728]（C≤163、exp≤256）⇒ ds_B ∈ [8,16]；现网/C7 侧 ds_C7 ∈ [8,17]。两者都装进 `[5:0]` 6-bit 端口（证据 ③ 对账项 #2 口径），零变化。workload 下 C≤46 ⇒ Z_B ≤ 11,776 ⇒ **ds_B ∈ [8,14]**（[prof] 观测，非容量）。

### 3.2 gate 饱和到 256 与 t=128 全翻的硬件处理

- 机制链（golden 数值链 `[模型]`，已核对 golden.py）：top 类 exp=256 ⇒ 乘积 14,745,600；ds_B ≤ 14 时 `14,745,600>>ds_B ≥ 900` ⇒ **撞 `min(256, ·)` clamp，gate_B = 256**。workload 下 ds_B ≤ 14（§3.1）且最小 exp=102 的乘积 5,875,200 > 256×2^14=4,194,304 ⇒ **本 workload 中全部 occupied 类、全部窗口的 gate_B = 常量 256**（[prof] 观测；合法域 ds_B=15/16 时低类可不饱和，硬件按通用数据通路实现，容量不降级）。
- t=64 零翻的对照解释：全等行（73.5%）top 类 gate_C7 = RNE(14,745,600>>17) = RNE(112.5) = **112**（≥64 发射，<128 抑制）；观测上全部 occupied 类 gate_C7 ≥ 64 ⇒ t=64 零翻转、t=128 全 450 token 翻转（302.4M/302.4M）。
- **硬件处理 = 什么都不加**：(a) 溢出——无新溢出点：乘积 14,745,600 < 2^24（与 C7 同宽），右移 ds ∈ [8,17]（同硬件），累加器 16-bit 够（Z_B ≤ 41,728 < 2^16），无进位扩展需求；(b) clamp——`min(256)` 是现网 `certified_gate_quant_q17` 既有饱和语义（C7 深尾行同样会饱和），Mode B 复用，不新增 clamp/饱和逻辑；(c) 发射比较（gate ≥ threshold）不变——t=128 全翻是**数值合同声明的直接后果**，不是硬件异常；(d) 唯一真实的硬件后果是**输出边界**：gated-K 发射集合与现网不同，Mode B 的 miter 与 138 行账本验收不能复用（§6）。
- **Q1.7 兼容**：out_gate_q17 = 9-bit；256 = 0b1_0000_0000 在 9-bit 无符号口径内可表示，且是现网 clamp 哨兵值（现网 C7 在稀疏行已输出 256，深尾行 gate_C7 最低可到 76，max|Δ|=180 即 256−76 的直接证据）。Mode B **不改变表示格式、不改变数据通路宽度**，只改变 clamp 触发频率（深尾稀有 → 全 workload 100% 类 100% 窗）。

### 3.3 certificate 语义变化（Mode A → Mode B）

- Mode A：certificate 的 denom_shift 必须与现网 C7 Q 点**逐 bit 复刻**（bit-exact 义务，miter 对 C7 账本）。
- Mode B：certificate 计算**新 Q 点** `ceil_log2(Z_B)`（同一 ceil_log2 硬件、同一 exp LUT、同一 RNE 取整位置——runbook §11.2 改动面唯一性：只改分母）。累加器输入从 `exp×count[c]`（乘法）变为 `exp×1`（常量，乘法折叠）；端口模式（row_load_start/load_accept/load_pair_id → certificate_valid/certificate_pass/denominator_shift[5:0]/row_max[8:0]）与证据 ③ §3.3 原样。

---

## 4. bit 账（Mode B 相对 Mode A，正负都算）

### 4.1 存储位账（163 口径，denom-only 变体）

| 对象 | Mode A | Mode B | Δ |
|---|---:|---:|---:|
| occupancy bitmap | 163 | 163 | 0 |
| descriptor 流 | 4,950 | 4,950 | 0 |
| certificate | 14 | 14 | 0 |
| count 文件 | **1,467** | **0** | **−1,467** |
| **合计** | **6,594** | **5,127** | **−1,467（−22.2%）** |

上下文列：vs 现网 C7 物化 7,957（证据 ③ §2.4 口径）= **−35.6%**；vs fused pair-gather 强基线 4,132（p95 保守点）= +24.1%（Mode A 的 qg'/do' 口径已 +10.3%~+13.5% 且 ≥20% 腿 FAIL，Mode B 的存储腿同样不成立——**能量腿口径维持 SAIF ≥15%（Mode A），不因 Mode B 重开**）。H82 兼容口径：count 消除 513×9=4,617 bit 同逻辑成立。

### 4.2 能量/活动账（每窗口径，workload 锚定 [prof]）

**负项（省）**：
- count 文件 RMW 全消：build 期 D 次写/窗（D mean≈230）+ class 相位 C 次读/窗（C mean≈5.1，p95=16）；
- certificate 乘法消除：exp×mult 折叠为 exp×1（侧车目录边界内）；
- gate 流切换骤降：workload 下 gate_B 恒 256 ⇒ 9-bit gate 总线、阈值比较输入切换 ≈0（对比 Mode A 逐类变化）。

**正项（费）**：
- 发射活动增加：t=128 下 Mode B 恒发 450 token（vs C7/Mode A 全等行近零发射）⇒ K store 双 bank 读、out_valid/out_k_bits/emit 后端活动增加（发射后端为共享块"现网不动"，但活动随发射集改变）；
- top 侧共享乘法器仍在（top 原封，default-off 要求）：乘数常量 1 只降切换、不消除硬件。

**净结论**：目录边界（sidecar 自有对象）内净减；含发射后端边界后符号不定（threshold 依赖）。**Mode B 不做独立能量主张**——能量腿主主张 = Mode A（SAIF ≥15%）；Mode B 在论文中只作为"数值合同变体 + 精度锚"，runbook §11.4 措辞原样。SAIF 实验（runbook Phase 0–5）只跑 Mode A，Mode B 无 SAIF 门。

### 4.3 逻辑/时序增量

- 累计逻辑净减：count 文件读写逻辑、certificate 乘法、17→16 bit 累加器；无任何新增宽位或比较器。Fmax/面积门（≤+10% / ≥−5%）按 Mode A 执行；Mode B 不另设 PPA 门。

---

## 5. 准入条件绑定（GPU 精度门 AEE ≤ 1.3430）

判定：`AEE_ModeB ≤ 1.3297 × 1.01 = 1.3430`（runbook §11.2，valid825 全网络推理，队列 D1 short → D3 short → Mode B，<0.5 h，不执行只排队）。证据分档：GPU 数值 `[模型]+[prof]`。

### 5.1 PASS（AEE ≤ 1.3430）→ 设计元素生效清单

| # | 生效元素 | 动作 |
|---|---|---|
| 1 | `MODE_B=1` 参数点 | 与 `SIDECAR_DIRECTORY` 组合进入 RTL 前置（证据 ③④ 冻结合同 + 本文档 §2/§3 为输入）|
| 2 | count 文件消除 | 实例化时（1,1）组合不产生 count 文件；class_multiplicity=1 常量 |
| 3 | certificate one-vote 语义 | 累加器 exp×1、denom_shift 新 Q 点、miter 目标 = golden ds_B/gate_B 0-mismatch |
| 4 | gate 恒 256 / t=128 全翻入档 | 作为 workload 事实 `[prof]` 与 GPU 数字并列呈现（上下文，不合成新指标）|
| 5 | 论文一句精度锚 | 侧车主主张不变 = Mode A energy 腿；Mode B 只补"one-vote 合同精度锚"一句（差异包 + GPU 数字双证据）|
| 6 | Mode B 自 golden miter 范围 | flat/banked/compiled + directed corner 全部以 golden.py 的 Mode B 数值链为参照（**不比对 C7 138 行账本**：发射集与 gate 已不同）|

### 5.2 FAIL（AEE > 1.3430）→ 整体归档 HOLD

- Mode B **不进 RTL**：不写 miter、不建实例、不跑 SAIF；设计文档 §2/§3 全文降级为"已归档的数值合同变体设计"，无后续工作项。
- 论文措辞（runbook §11.4 已冻结）：只报 Mode A（C7-exact）+ 差异统计包（证据 ① Mode B 节已归档），"one-vote 数值差异客观存在且已量化，精度证据未过门，不主张"。
- **Mode A 主线零影响**：default-off bit-exact 是构造性质（证据 ④ §4），不依赖 Mode B 任何元素。

### 5.3 绑定纪律

- 本文档是 PASS 后**立即写 RTL 的冻结前置**（消除 GPU 空窗）；FAIL 时不产生任何 RTL 侧工作。
- 不后验改门：AEE 门 1.3430 预注册；gate 恒 256 的观测不构成"再修一版量化"的理由——数值合同是算法线（H82/H86）裁决物，Motion 侧车只引用不自行裁决（设计文档 §2.3/§6）。

---

## 6. 审稿人风险预答：「one-vote 是改阈值就能复现的 C7 变体」

三条硬证据（全部来自 golden `[模型]`+`[prof]`，672,000 行）：

1. **分母是函数改变，不是参数微调**：改阈值只动发射比较常数，Z 与 denom_shift 逐位不变（ΔZ≡0、Δds≡0 于 100% 行）；观测到的是 **Δds 全负（p50=−7，range [−9,−3]）且 100% 窗口非零、Z_B ≈ 0.011×Z_C7**——这是"分母从加权和换成无多重性总和"的函数级指纹，阈值扰动不可能产生任何一行 Δds≠0。
2. **"96.9% 条目级 Δ=0"的正确解读（防反向引用）**：96.86% 是 **672,000×163 全矩阵口径**，其中未占用 bin 占 96.9%（Δ 平凡为 0）；**占用类条目实质上 100% 改变**（100% 行 gate 集变，top 类 112→256）。审稿人若引"96.9% 没变"为"微调"证据，此分解直接消解；若引"100% 窗口分母变"为"大改"，则算法线 AEE 门兜底。
3. **t=64 零翻 + t=128 全翻的极端对**：阈值假说预测翻转集中在 [t, t+δ) 窄带；实测 t=64 翻转 0 token、t=128 翻转 302.4M/302.4M token——只有"gate 值整体左移 2^Δds 并撞 min(256) clamp"（§3.2 机制链）能同时满足两档。这一对极端是"分母 Q 点改变"的充分判别式。

**定位一句话**：one-vote 是 H82 算法线 multiplicity-free 合同（docs/445、`h82_multiplicity_free_quotient_model.py`）在 Motion Q7 档位的目录物化；与 C7 的关系在论文中如实陈述为"同一 quotient 流、不同归一化合同"，数值差异包 + GPU 精度锚（AEE ≤ 1.3430 门）双证据；不新增贡献名、不构成 4.0（无新算法算子合同，设计文档 §2.3 措辞边界原样）。

---

## 7. 剩余前置与证据分档

| # | 项 | 状态 | 分档 |
|---|---|---|---|
| 1 | Mode B 差异统计包（Δds/ΔZ/Δgate/翻转） | **已完成**（golden 证据 ① 续节）| `[模型]`+`[prof]` |
| 2 | Mode B 精度评估（valid825，AEE 门）| **排队中**（D1→D3→ModeB，<0.5 h）| GPU `[模型]` |
| 3 | PASS 后的 RTL + 自 golden miter | 未开始（绑定 §5.1）| `[rtl]`（写 RTL 后）|
| 4 | Mode A SAIF/PPA ⑤ | 新思机器（与 Mode B 无关）| `[待验证]` |

本文档不修改任何现有文件；无 RTL 产出；不触碰 GPU 队列纪律（CLAUDE_ALGORITHM_CONTRACT_QUEUE_20260818.md）。
