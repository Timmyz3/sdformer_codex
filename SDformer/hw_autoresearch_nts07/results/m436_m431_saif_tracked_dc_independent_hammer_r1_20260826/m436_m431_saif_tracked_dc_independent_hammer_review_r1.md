# M436：M431 SAIF-tracked DC 独立打铁

## 结论

评分 **82/100**。M431 的 DC 复现和 Synopsys essential map 导出本身通过；它只能准入到“继续做完整映射或 gate-SAIF 诊断”。当前 mapped RTL-SAIF 直接注释只有 `3734/22800 = 16.38%`，`report_switching_activity -coverage` 为 `4962/22800 = 21.76%`，远低于冻结的 95% 门槛，因此 **功耗、能量和 paper PPA 均 NO-GO**。

## 独立重算

- M416、M425R4、M429 和 M431 的双封全部通过；M431 exact-SHA runner 当前 SHA 与封存值一致。
- 封存 DC log 中 `saif_map -start` 位于 `analyze` 和 `elaborate` 之前。
- M431 与 M416 完全相同：面积 `24,548.705582 um^2`、cell `20,803`、FF `4,100`、logic levels `52`、setup WNS `+0.7636 ns`、hold WNS `+0.0250 ns`。
- 五类 constraint report 均为 no violation；macro/blackbox 为 0，未发现 unresolved reference、inferred latch、timing loop、DC Error 或 Fatal。
- essential map 有 7,035 条命令，7,035 个唯一 RTL 名和 7,035 个唯一 gate 名；语法全部合法，PrimeTime 接受全部命令，且 7,035 个 gate 对象在 linked netlist 中全部存在。
- “gate 对象存在”不等于“RTL SAIF 名可解析”。PrimeTime 仍报告 `u_matcher`、`u_matcher/u_balanced`、`u_adapter` 为 `None Found`；只有 `799/4100 = 19.49%` sequential outputs 来自 activity file。
- M431 的 `saif_map -report` 同时写着 `No objects found to report`。因此不能拿 7,035 条 map 数量当覆盖证据。

## 缺陷分级

- P0：0。
- P1：2。
  - essential map 目标合法且存在，但无法覆盖大部分带层级的 RTL SAIF 状态。
  - 95% 门槛当前混用了“名字注释完整度”和“至少一次 toggle 的 coverage”；稀疏负载下两者不是同一个量，必须分列。
- P2：3。
  - `saif_map -report` 的空报告与非空 essential map 并存，应披露。
  - 当前仍是 0 macro、ideal clock、ZeroWireload 的 pre-macro screen。
  - DC log 有四次 PWR-24：库中存在未做 internal-power characterization 的 cell；将来 PTPX 必须统计受影响实例。

## 修复路线

优先尝试从同一 DC/SVF 流程输出 **完整、层级可解析** 的 RTL-to-gate map（不仅以 `-essential` 条目数作为成功标准），随后只跑 read-SAIF/coverage 诊断。这个方案成本最低，也能直接复用 M425R4 工作负载。

若完整映射仍无法覆盖 flatten/ungroup 后的内部状态，则转向 **zero-delay gate simulation 生成 gate-level SAIF**。它绕过 RTL-to-gate 名字变换，鲁棒性更高，但需要重新验证 reset、X、协议、时窗和仿真吞吐。

无论选哪条路，在独立注释门槛通过前都不得运行或引用 PTPX power/energy。完整统计见 `m436_independent_audit_result_r1.json`；`pt_diagnostic_failed_power_disabled` 是被显式拒绝的前置条件失败尝试，不是封存成功结果。
