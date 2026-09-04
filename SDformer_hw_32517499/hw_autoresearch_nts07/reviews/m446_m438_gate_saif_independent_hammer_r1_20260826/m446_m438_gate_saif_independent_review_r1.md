# M446：M438 mapped-gate SAIF 独立打铁

结论：**94/100，P0=0、P1=2、P2=3。GO 进入一个新的、独立封印的 prelayout PTPX 里程碑；M438 本身仍 NO-GO 功耗/能量/系统性能/论文 PPA。**

本评审没有从 M438 两份 receipt JSON 派生数字。所有数字来自冻结 `memh`、原始 VCS compile/sim/assert 日志、门级 SAIF、PT annotation 报告，以及 wrapper/TB/网表/UCLI/PT Tcl/runner 的直接审计。

## 独立复算结果

- 完整性：M438 `RUN_MANIFEST.sha256` 的 27 项全部通过，外封通过；全部输入 exact-SHA 通过；`docs/359` 仍为 `dedde7ce...`。
- 网表：把 M431 源网表首个 top module 名机械改为 `m405_q32_elastic_selected_slice_mapped_gate` 后，与 M438 仿真网表逐字节相同；没有第二处改动。
- 负载：三份冻结 `memh` 独立复算为 64 phases、192,000 rows、61,285 pass1、11,923 early、93,037 zero、25,755 pop1、63,067 PWP rows、504,536 low、416,630 high、87,906 narrow、416,630 wide、921,166 contributions、48,435,456 reconstructed lanes。
- 功能：TB 对真实 `u_gate` 的每个 accepted matcher result 和 contribution 逐项比对；wide block 的每个 12-bit lane 都重构比对。metadata/matcher/codec/reconstruction/bitmap/X/protocol/SVA failure 全为 0。
- SAIF：作用域只有 `tb.../dut/u_gate`；无 wrapper scoreboard signal。22,800 entries 中 21,827 个 `TC>0`、973 个 `TC=0`；`TX>0` 为 0；总 TX duration 为 0；所有条目的 `T0+T1+TX` 与 duration 一致；`protocol_error` 的 TX/TC 都为 0。
- PT annotation：22,800/22,800 nets 由 activity file **exact mapping**（100%）；20,803/20,803 leaf cells fully annotated（100%）；有至少一次 toggle 的 net 为 21,827/22,800 = **95.732456%**；inconsistent object 为 0。因此“exact annotation ≥95%”和“nonzero-toggle coverage ≥95%”两道门都严格通过。
- 功耗边界：PT Tcl 只 `read_saif` 和 `report_switching_activity`。没有 `update_power`、没有 `report_power`、没有 power report。VCS UCLI 的 `power` 命令只负责 gate-level activity monitoring/SAIF 导出，不是 PT 功耗计算。

## Wrapper 结论

外置的 10 个 32-bit debug counter 恰好替代被综合删除的 320 个纯 debug DFF，只从真实门级 handshake 和四个被观测内部网驱动，且不在 `u_gate` SAIF scope。它们只承担最终 population ledger；真正的功能 oracle 是 TB 对真实门级输出的逐事务/逐 lane 比对。因此 wrapper 没有把 RTL 功能“旁路成参考模型”，也没有污染门级 SAIF。

四个层次化 observer 会增加零延时仿真中的逻辑观察扇出，但不进入 PT 读取的 mapped design，也不进入 scoped SAIF。由于是层次名依赖，任何重综合都必须重新做兼容性审计。

## 风险与准入边界

P1：这是 zero-delay/no-SDF、prelayout、0 macro、无 extracted interconnect 的活动；且只覆盖 64-phase/192k-row stratified subset，不是 51.84M rows 或 full network。下一轮只能称“prelayout standard-cell PTPX”。

P2：合法 workload 没有触发 global-fault cover；973 个 net 在该 subset 中零 toggle；wrapper 的四个内部 observer 依赖 frozen hierarchical name。三项均不阻止 PTPX，但必须保留口径。

最终准入：可以启动新的 exact-SHA PTPX；必须同时报 100% exact annotation 与 95.73% nonzero-toggle coverage，并在下一轮再做独立打铁。严禁把 M438 自身称为已有功耗、能量、系统倍速、Conv 倍速或 paper-PPA 证据。
