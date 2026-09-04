# TCAS-II 2027 C1/C2/TSBG 物理证据与最小收口审计 r1

日期：2026-09-04（Asia/Shanghai）  
模式：只读证据审计；没有运行 VCS/simv、DC、PT、PTPX、Formality、GPU、license query 或远端任务；没有修改任何既有源码、RTL、结果、合同、论文或 `docs/359`。

## 裁决

**现有证据足以支撑一篇边界清楚的 TCAS-II execution-island brief，但还不是“matched physical/energy closed”的电路稿。** C1 已有九个真实 SRAM leaf 的面积、PT setup/hold、mapped-to-mapped Formality 和 PTPX 组件锚点；C2/TSBG 已有可信的 VCS 周期分布和 logic-only DC 面积/3-ns setup，却仍缺 matched hold、mapped power 和存储能量。

对 TCAS-II 最合适的收口不是增加第四个稀疏机制，而是把 TSBG 从“权重请求减少”补成“相同 288-KiB 权重容量下，计算、逻辑和 bank-activation/energy 同时计价”的电路-存储协同结果。现有 C1/C2/TSBG 机制不需要互相替换。

## 一、逐 claim 的可引用状态

| 对象 | 已可引用 | 必须同句限定 | 当前未准入 |
|---|---|---|---|
| C1 周期 | `648,741,051 / 382,848,700 = 1.694510x`，十个 `zurich_city_09_a` 样本、51.84M source-row | same-ledger cycle model；不是 RTL、全网或系统周期 | 多序列 RTL 周期、系统 FPS |
| C1 物理 | `166,514.312 um2`；9 个 `128x128b` 1RW SRAM；PT setup/hold `+27.871/+1.827 ps`；16,549 mapped-to-mapped compare points | prelayout、ideal clock、ZeroWireload；这是九宏 execution island | 214,912-B 全存储的集成 timing/PPA |
| C1 功耗 | 253-cycle/759-ns directed window：`29.0763016 mW`、`22.0689129144 nJ`；parent scratch `36.1%` | stdcell TT 0.9 V 25 C + SRAM SSG 0.9 V 125 C mixed-corner；无 SPEF；不是 energy/frame | 单一 PVT signoff、全存储/系统能量 |
| C2 等带宽 | K8/K1x8 为 `1913/1945` VCS cycles；面积 `131,086.241/585,479.154 um2`；`1.016728x` cycle、`4.541078x` throughput/logic-area、logic `-77.6104%` | 五个 directed workload；logic-only、pre-macro；周期与面积效率必须同句 | hold、power、macro-inclusive PPA、系统加速 |
| TSBG 周期/流量 | 1,920 个固定 ep34 workload、40 sample、4 sequence：`12,522,876 -> 5,124,365` cycles，`2.443791x`，time `-59.07997%`；scalar requests `8,774,304 -> 3,136,608`，`-64.25234%` | post-load component VCS；自然非零 descriptor 为 `+1`；8 个 G>48 FC2 层和全 token 不在该人口；7 个非空 case 略慢 | full-FC/full-network、真实权重 wall time、系统 speedup |
| TSBG logic | ordinary/TSBG `249,710.452/249,739.810 um2`，`+0.0117568%`；两轴 setup met | 相同 M2018/M803 RTL source 的 schedule-mode ablation；standard-cell state、0 macro | hold `-16.4 ps`、power、external weight SRAM |

以上数字的独立审阅目录内外 seals 均已在本次审计重新验证。`docs/359` SHA 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 二、SRAM/容量证据的真实边界

### C1：九宏物理锚点不等于完整 240-KiB 账本

冻结 C1 storage ledger 是 `214,912 B`：parent `18,432 B`、metadata/reserve `24,448 B`、psum `122,880 B`、weight `49,152 B`。映射为同一种 `2,048-B` SRAM 时，需要 `9+12+60+24=105` 个宏，容量 `215,040 B`，保守面积为 **`0.988049 mm2 [macro-area model]`**。

当前 `166,514.312 um2` mapped top 只实际绑定九个 parent SRAM leaf。105 宏值没有集成进 netlist，也没有完整 adapter/interconnect、timing 或 PTPX。论文不能把“9 个已实现宏”与“约 215 KiB 完整容量”写成同一个物理对象。TCAS-II 主表应拆成：

1. `C1 execution island, 9 generated SRAM leaves [mapped]`；
2. `complete 214,912-B ledger, 105 macros, 0.988049 mm2 [macro-area model]`。

若保持 execution-island 叙事，这一拆分不要求新 EDA；若声称完整 C1 PPA，则必须重新集成 105 宏并重跑 VCS/DC/PT/Formality/PTPX。

### C2/TSBG：288-KiB 权重 SRAM 是当前最大的未计价共同项

C2 权重容量为 `8 x 2304 x 128b = 288 KiB`，每 bank 1R。现有 foundry QRT exact-capacity 组织为每 bank `2048x128 + 256x128 SP`，总计 16 宏、**`558,507.032 um2 [foundry-QRT model]`**；最慢宏 local `tcyc/tacc=0.800/0.701 ns`。它对 K8/K1x8 和 ordinary/TSBG 都是共同部署容量，不能从 candidate 面积中删除，也不能把少激活 bank 写成少部署 SRAM。

当前 TSBG 的最强可执行存储事实是 scalar-bank request 从 `8,774,304` 降到 `3,136,608`；单个预注册 mapped-energy workload 的冻结请求数是 ordinary `14,304`、TSBG `4,608` 个 128-bit bank reads。将此访问人口与同 PVT 的 SRAM access/leakage 模型相乘，是 TCAS-II 最低成本、最有价值的新增物理轴，但必须保持 `[foundry-QRT model]`，不能冒充生成宏或 PTPX。

## 三、hold 和 SAIF/PTPX 的失败根因

### C2 hold：不是几条可手工加 buffer 的控制路径

M1877 的 fast-min 诊断为 K8 hold WNS `-23.259 ps`、30,442 个 violations。完整路径分类显示：

- 30,267/30,442（`99.425%`）是 exact same-register Q-to-D feedback；
- 28,602（`93.956%`）集中在 Acc24 context、weight slots 和两个 bitmap 状态族；
- 50-ps hold uncertainty 不能写成 60 ps；当前是 ideal/unpropagated clock、ZeroWireload、无 SPEF 的 prelayout 诊断。

M1960 的一次 `set_fix_hold`/incremental 尝试把 K8 面积从 `130,822.775` 推到 `141,886.71 um2`，增长 `8.457%`，超过 +5% 门，而且没有可引用的 post-fix timing/netlist。继续做 DC buffer-only sweep 没有价值。唯一可信的硬闭环是两轴相同约束的 placement/CTS/route/RC extraction 后 hold repair；若不做 P&R，则 TCAS-II 只能写“3-ns setup met, prelayout hold open”，不能写 timing-closed 或 333-MHz signoff。

### C2/TSBG mapped power：当前是零有效 SAIF、零 PTPX

不要把已消费失败坐标重跑或拼接：

1. M1845：K8 case0 在 SAIF window 打开后，public fault/endpoint-fault 聚合向量含 X/Z；`SAIF=0, PTPX=0`。
2. M2058：ordinary 在 preload 完成后、第一次 execute 前，bridge/commit/control 聚合 X/Z；`SAIF=0, PTPX=0`。
3. M2061：settled-phase 版本定位到 `ordinary.cycle_count` X；根因包含同步 reset 同 active-slot 释放的 TB race，以及 reset 被合入无 reset-pin DFF 的四态 X-pessimism；`SAIF=0, PTPX=0`。
4. M2063：`+vcs+initreg+0` 后能进入 ordinary power window并跑到 base done，但十项 completion ledger 至少一项漂移；日志不打印每项 actual/expected，无法定位；仍没有完成 SAIF，也没有 PTPX，TSBG 轴未运行。

这说明下一步不能再直接重发同类 mapped-gate power runner。TCAS-II 时间窗内存在两条可信的活动路径：

1. **优先路径：RTL DUT-only SAIF -> 新鲜 DC `saif_map` transformation tracking -> PT mapped annotation/PTPX。** 它绕开当前 mapped VCS 的四态收敛问题，但必须用最终 ordinary/TSBG 两轴网表分别生成精确 name-map，且不能把它写成 mapped-gate VCS activity。M431 已证明 Synopsys 路径可导出 7,035 个 essential map entries；M437r2 将 essential 与 4,100 个 default sequential entries 做 union 后，PT 仍只有 `30.86%` annotated nets（低于 95% 门，53.59% switching coverage），因此这只是可复用的 source pattern 和负向 gate，不是成功功耗证据。
2. **较慢后备路径：direct mapped-gate VCS SAIF。** 若新的 `saif_map` 在最终 TSBG 两轴上仍过不了覆盖门，或者论文必须声称门级动态活动，则先做一个 diagnostic-only identity：ordinary/TSBG、RTL/mapped 并排，逐项打印十个 completion ledger 的 actual/expected，并记录 first-divergent cycle；保留所有 fault/X 检查且不产生 paper power。

无论选择哪条路径，功耗都必须绑定同一最终 area/timing/Formality identity；不能用旧负 hold 网表测功耗，再对另一份 hold-repaired 网表报 timing。

## 四、最小可执行 P0（按依赖顺序）

### P0-1：先固定 TCAS-II 物理对象和表格口径（零 EDA）

- C1 明确拆成九宏 mapped island 与 105 宏 full-ledger area model。
- C2/TSBG 明确拆成 logic-only mapped/DC 与 288-KiB common SRAM QRT model。
- 任何面积效率都用 `logic` 或 `logic + identical common SRAM` 明确标注；不能把共同 SRAM 只加到 baseline/candidate 一侧。

### P0-2：选择并冻结活动传递路径（优先 `saif_map`）

- 为 ordinary/TSBG 固定完全相同的 workload、预载、power window 和统计分母；RTL DUT-only SAIF 必须满足 TX=0，且 T0/T1/TX 与 duration 守恒。
- 在最终两轴综合 identity 中启用 Synopsys `saif_map` transformation tracking，分别导出 exact name-map；记录 essential/combinational、default/sequential、intersection 和 union entries。
- PT 端必须报告 annotated/unannotated nets、annotation percentage、nonzero-toggle coverage、inconsistent annotation 和关键计算/控制锥活动；低于冻结覆盖门或关键锥为零即 fail closed，不得 `report_power` 后挑选数字。
- M431/M437r2 只可作 runner/source pattern：M437r2 的 `30.86%` annotation 已明确失败，不能沿用旧 map 或把 53.59% switching coverage冒充 annotation pass。
- 若最终 TSBG 两轴仍无法过 `saif_map` 门，再回到 direct mapped-gate diagnostic：复用 M2056/M2063 的 fixed slot42、383-cycle preload、ordinary `20,292` 和 TSBG `7,569` execute denominator；逐项打印 cycles、rows、issues、products、miss/hit/evict、bundle beats、bank requests/responses，输出 first-divergent signal/cycle。该 diagnostic 通过前禁止新的门级 SAIF/PTPX production attempt。

### P0-3：生成一个最终 matched physical identity，再在其上测功耗

- 最优证据：ordinary 与 TSBG 使用相同 floorplan/placement/CTS/route/RC、PVT、IO、macro/common-capacity 和 hold-repair策略；两轴 setup/hold WNS 均 `>=0`、DRC=0、Formality PASS。
- 若时间不允许 P&R，至少对相同最终 netlist 做 matched setup/DC + 明示 hold-open；此时主表不得写 timing-closed/Fmax，只能把 hold 当限制项。
- power 必须在这个最终身份上重做；不能先用旧负 hold 网表测功耗，再对另一份 hold-repaired 网表报 timing。

### P0-4：matched SAIF/PTPX + external SRAM energy 分列

- 若采用 direct mapped-gate 路径，mapped functional 必须先完整 PASS；若采用推荐的 `saif_map` 路径，RTL functional/VCS 和 mapped Formality 必须分别 PASS。两个 axis 都产生 DUT-only SAIF；duration=`cycles*3 ns`、TX=0，并报告 annotation 与 nonzero-toggle coverage。`saif_map` 路径必须标作 **mapped-netlist power driven by transformation-mapped RTL activity**，不得标作 mapped-gate VCS activity 或门级动态等价。
- PTPX 按同一 fixed workload 报 internal/switching/leakage/total 和 `E=P*time`；普通与 TSBG 两轴必须同一 PVT/约束。
- 288-KiB SRAM area/leakage对两轴相同；dynamic 按真实 128-bit bank requests 计价。若只有 QRT，单独标 `[model]`，不得与 logic PTPX 相加成“measured total”而隐藏证据等级。
- TCAS-II 主句建议同时给：post-load cycles、scalar bank reads、logic energy、SRAM dynamic-energy model、logic/macro area breakdown。

## 五、可直接复用与禁止复用的入口

### 可直接复用的 sealed evidence

- C1 cycle：`reviews/m1597_m1590_ep34_c1_same_ledger_cycle_model_result_hammer_r1_20260901/`。
- C1 PT/Formality：`dc_handoff/runs/m1740_c1_readonly_formality_pt_salvage_r1_20260901/`。
- C1 SAIF/PTPX：`results/m1782_c1_expected_macro_leaf_blackbox_energy_r1_20260902/` + M1789 review。
- C2 same-campaign area/cycle：M872 mapped identity + `reviews/m903_m872_m803_c2_r16_three_axis_dc_result_hammer_r1_20260829/`。
- TSBG matched logic：M2029 mapped netlists/SDCs + M2030 review。
- TSBG 1,920-workload population：M2057 result/review。
- SRAM：`reviews/tsmc28_sram_macro_audit_r1_20260827/`；C1 full-ledger model：M1591/M1596。

### 仅可复用 source pattern，必须新 identity/review/release

- `tb_m2018/tb_m2056_m2018_tsbg_matched_mapped_energy.sv` 和 M2063 版本：可复用 workload、bridge、two-stop/UCLI 结构；必须修成逐项 divergence diagnostic。
- `run_ptpx_m2063_m2018_tsbg_matched_mapped_energy.tcl`：可复用 annotation/duration/分量解析框架；只有新的 mapped functional PASS 后才能进入 production runner。
- M431/M437r2：`run_dc_m431_m414_saif_tracked_selected_slice.tcl`、`*.ptpx_saif_map.tcl` 和 `run_m437r2_m431_union_saif_annotation_recovery_exact_sha.sh` 可复用 transformation-tracking、mapping-class 审计及 PT annotation 框架。M431 仅是 mapping diagnostic；M437r2 annotation=`30.86%`、NO-GO power，二者都不是可引用的正功耗结果，也不能直接复用旧 map。
- M2088 continuation DC runner：已过 M2087 source hammer，但只测 G96/G192 continuation 的 logic-only area/setup；需要一个干净、独立准入的 960-workload VCS 前置。它不是 power、hold 或 macro closure，不能替代上述 P0。

### 永久禁止原身份重跑/引用为成功

M1845、M1858、M1877、M1960、M2058、M2061、M2063 都是已消费 failure/quarantine 坐标。它们只可作为失败诊断，不可自动重试、不可拼出成功表、不可引用半轴功耗或 hold closure。

## 六、对 TCAS-II 的工作取舍

- **不需要 FPGA 才能投稿。** 当前商业 28-nm flow 的 VCS/DC/PT/Formality/PTPX 路径比临时 FPGA 映射更一致。
- **不新增 S2/有损、decoder matcher 或 attention 稀疏。** 它们不会修复当前最显眼的电路证据缺口。
- **FC2 continuation 是 P1 evaluation-depth，不是 physical P0。** 它若成功，可把 TSBG 从 4/12 FC2 扩到 12/12；但不应阻挡先定位 mapped power 和固定存储口径。
- **最适合的“新增点”不是新算法，而是 storage-aware TSBG evaluation。** 现有机制已经在被省资源之前抑制 weight bank reads；把同容量 SRAM 的 bank-activation 能量和 logic PTPX补齐，会提高 TCAS-II circuits relevance，又不会稀释 novelty。

## 最终建议

按 `P0-1 -> P0-2 -> P0-3 -> P0-4` 收口；功耗优先尝试新的两轴 RTL-SAIF + `saif_map`，覆盖失败才投入 direct mapped-gate divergence。若能获得同身份的 hold-clean + mapped-netlist PTPX，并把 288-KiB common SRAM分列计价，C2/TSBG 就从“RTL周期 + logic area”提升为完整的电路-存储协同结果；这是比再加新稀疏 idea 更可靠的 TCAS-II 增量。若 P&R/hold 来不及，保留 execution-island 口径仍可投稿，但必须公开 hold/power/macro open，预期只能是边缘到普通 Accept，而不是强证据稿。
