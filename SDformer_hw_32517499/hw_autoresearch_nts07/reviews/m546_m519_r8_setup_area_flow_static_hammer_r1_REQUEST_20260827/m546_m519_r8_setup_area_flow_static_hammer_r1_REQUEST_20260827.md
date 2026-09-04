# M546 / M519 R8 setup-area flow fresh independent static-hammer request

只读审阅全新的 R8 identity。禁止运行或探测性调用 `dc_shell`、`common_shell_exec`、VCS、PT、PTPX、Formality 或任何其他 Synopsys executable；禁止 CPU 大任务、远端任务和创建 launch admission；禁止修改 R5-R7、`docs/524` 或 `docs/359`。允许只读查看 wrapper 文本、文件元数据与 SHA。

必须逐条打铁审计：

1. `dc_shell` 入口、resolved `snps_shell` wrapper、实际 `common_shell_exec` 和两角 DB 是否都在 contract/future-admission closed identity 中，runner 是否在第一次 preflight 前对当前字节逐项 `expect`；contract `exact_files` 17 项是否全部 launch-time `expect`。
2. fork 后 capture 是否严格依靠 parent/starttime/UID/actual-exe 和完整 NUL-safe argv `common_shell_exec -shell dc_shell -r <install-root> -f <exact-R8-Tcl>`；capture 失败是否立即只 TERM 精确 birth tuple、限时后只 KILL 同一 tuple、wait 并 quarantine，绝不允许无 monitor 的 DC 继续运行。
3. future admission 是否将五个 R5 basis 的 path/outer-seal-file SHA 与 contract 闭合集合逐一交叉；runner 是否既验证每个 basis 的内外 seal，又比较实际 outer-seal 文件 SHA。R6 failed review 与 R7 disqualified review是否同样由 path/status/outer-seal SHA 硬绑定，且 R7 reviewer 禁止用于 launch admission。
4. 每个 external collision/mismatch 是否写出可独立重建的 timestamp/label/kind/PID/PPID/UID/starttime/state/comm-hex/exe-hex/完整 NUL-preserving cmdline-hex；PID identity change 是否仍保留变化前完整 tuple；campaign descendant 排除是否继续要求祖先链 `(PID,starttime)` 二次校验。
5. exact birth 的 zombie 是否只被识别为 completed/absent，而任何 PID/starttime/UID/parent/exe/cmdline 不一致仍锁存失败且永不误杀复用 PID。
6. runtime loop/final shared gate、final ack、preflight、attempt/quarantine 双封等 R7 已通过内容是否没有回归；R8 Tcl 是否仍恰好一次 `compile_ultra`、零 incremental、零 pre-CTS hold-only，三轴条件一致。

只有未执行任何 EDA 的 fresh reviewer 得到 `P0=0 && P1=0`，主 agent 才可另建一次性双封 launch admission。本 request 自身不授权任何运行，不证明面积、时序、功耗、能量、吞吐/面积、完整 FC2 或系统加速。
