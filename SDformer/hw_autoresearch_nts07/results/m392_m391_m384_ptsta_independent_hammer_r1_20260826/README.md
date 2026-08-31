# M392：M391/M384 prelayout PT-STA 独立打铁

结论：**93/100，P0=0、P1=0、P2=6**。接受 M391 r1b 作为 prelayout PT-STA 证据；不接受首轮 r1；不放行 postroute/physical timing、reset recovery/removal signoff 或 activity-backed PTPX。

独立重算得到 setup `+0.6302 ns`、hold `+0.0179 ns`。setup/hold 报告各保留 100 条路径且全部 MET，global timing 无 setup/hold violation，constraint violation 为 0。r1b 的 exact inputs、runner、18 个输出文件、manifest 和二层 seal 全部验证。

覆盖分母没有被隐藏：setup/hold 各 `1482 = 1170 met + 312 constant-disabled`；out setup/hold 各 `793 = 731 met + 62 no-path constant outputs`。全报告 7,144 个 untested check 精确分成 `constant_disabled=1872`、`no_paths=124`、`no_startpoint_clock=1716`、`no_clock=3432`。

其中 recovery/removal 各 858 个检查全部未测，合计 1,716，恰好对应唯一无 clock-relative input delay 且被 false-path 的异步 `reset_n`。所以本文只能说数据 setup/hold 在该 prelayout 点 MET，不能说 reset timing 已签核。

两类工具警告均记 P2。`SDC-2` 来自 DC 输出 SDC 2.1、PrimeTime reader 2.2；3 ns clock、0.1 ns uncertainty、I/O delay 和 reset false-path 都已加载并产生有效覆盖，因此不构成 P0/P1。`PTE-003` 与 1,872 个 `constant_disabled` 检查一致，`check_timing` 的 loop 检查没有报问题；但本轮没有封存 `report_disable_timing`，所以不能把它升级到 signoff 口径。

边界保持：无 SPEF、ideal clock、ZeroWireload、0 macro、descriptor/PWP SRAM 非物理实例。PT-STA 也没有提供 exact-workload SAIF；没有 activity annotation coverage 前，PTPX 继续 NO-GO。`physical_timing=false`、`energy=false`、`system_speedup=false`、`paper_ppa_ready=false`、`date_headline=false`。

首轮 r1 有 `RUN_FAILED_OR_INCOMPLETE`、exit code 42，且没有 RUN_COMPLETE、runner hash、output manifest 或 seal，明确不可引用。
