# M538 / M519 R6 setup-area flow 独立静态打铁审阅 r1

日期：2026-08-27  
裁决：`FAIL_STATIC_HAMMER__RETURN_TO_AUTHOR__NO_LAUNCH_ADMISSION`

## 总评

**84/100，P0=0、P1=3、P2=1。** R6 已经正确移除 R5 的重复 incremental/min-path hold 流程，并把 K1/K8/K1x8 放到同一 setup/area Tcl、filelist、SDC、库和 3 ns 约束下；三样本 preflight、轴间恢复、runtime 连续低水位计数和嵌套双封的主体结构也已出现。但是 source-only runner 仍有三个会破坏 fail-closed 启动语义的问题，因此本审阅**不建议、也不授权创建 launch admission**。

本 reviewer 只做只读静态审计；未运行 DC、VCS、PT、PTPX、Formality、CPU 大任务或远端任务，未修改冻结输入、R5 quarantine、`docs/524` 或 `docs/359`。

## P1：必须修复后重新 fresh hammer

### P1-1｜实际综合 RTL/库身份没有被 contract 在启动时闭合

contract 的 `exact_files` 在第 140–157 行冻结了 runner、Tcl、filelist、SDC、12 个 RTL 和 `docs/359`。当前实体确实 **17/17** 与这些 SHA 匹配；但 runner 第 92–117 行只按未来 admission 提供的值检查 contract、R5 seals、Tcl、filelist、SDC、两角库、DC 和 `docs/359`，没有遍历 `contract.exact_files`，也没有把 admission 的 Tcl/filelist/SDC/库/DC/R5 seal 字段与 contract 内已冻结字段逐项交叉相等。

因此，filelist 内容本身不变但其引用的 RTL 后续漂移时，runner 仍可启动；同理，一份内部自洽但与 recovery contract 不一致的 future admission 可以重新定义库或输入 SHA。当前“17/17 匹配”只是审阅时快照，不能替代 launch-time fail-closed 检查。

修复门：第一条 preflight、attempt 消费和 DC 之前，逐项验证 `contract.exact_files`；并要求 admission 的所有 identity 值分别等于 contract 中的冻结值，而非只要求 admission 与当前文件互相一致。

### P1-2｜`runtime_final` 被记录但没有执行资源判门

runner 第 361–413 行只在 `while kill -0 child` 循环内更新 `commit_bad_count` 并判断 commit/MemAvailable/SwapFree/cgroup/OOM/外部 EDA。child 消失后，第 415–418 行会记录一次 `runtime_final`，但没有再次更新连续计数或执行任何即时门。

这会漏掉两类合同失败：前两次 commit `<32 GiB` 后 child 在第三个样本前结束，final 恰为第三次低水位；或者 final 首次出现 `MemAvailable <128 GiB`、`SwapFree <32 GiB`、cgroup/OOM 非零或新外部 EDA。日志会留下坏值，但 monitor 仍返回 0，canonical 仍可能 PASS。

修复门：把每个 snapshot 的计数和判门抽成同一函数，循环样本和 final 样本都必须调用；final 命中时即使 child 已退出，也必须让 monitor 非零并 quarantine。

### P1-3｜campaign child 只用裸 PID/祖先链识别，PID 重用可误排碰撞或误 TERM

第 125–137 行的 descendant 判定只沿 `/proc/<pid>/stat` 的 PPID 回溯到数字 root；第 361 行的 liveness 和第 409 行的 TERM 也只使用裸 PID。runner 没有在 launch 时冻结 child 的 `/proc/<pid>/stat` starttime，也没有在排除外部 EDA、判活或发 TERM 前复核 PID+starttime。

若 child 在 parent/monitor 的竞态窗口退出且 PID 被重用，新进程可被当作 campaign root：其后代可能被错误排除出 external-EDA collision，资源 latch 也可能 TERM 非本 campaign 进程。这违反“只排除/终止本 campaign child”的 fail-closed 合同。

修复门：launch 后立即记录 root PID、starttime、UID 和可执行身份；所有 descendant、liveness、collision exclusion 和 kill 都必须验证同一 root birth identity。身份不一致应锁存失败而不是继续或发信号。

## P2：证据完整性

### P2-1｜合同要求记录 command，实际只记录 `comm`

contract 第 102–108 行要求每个 runtime 样本记录 descendant 的 PID、PPID、**command**、VmPeak/VmSize/VmRSS/VmSwap；runner 第 359–377 行的 TSV 只写 `/proc/<pid>/status` 的 `Name` 到 `comm` 列，没有完整 `args`/cmdline。高水位表同样只保留 `comm`。这不会单独改变本次启动裁决，但不足以独立重建进程身份与调用参数。

修复建议：在不暴露凭据的前提下增加规范化 `/proc/<pid>/cmdline` 或 `ps args` 字段，并把它纳入高水位/根级双封；若参数可能含秘密，应记录经过明确规则脱敏的命令和原始可执行 inode/hash。

## 已通过的静态项

- 冻结 Tcl SHA256：`b5c56877e8fdb920cfaf916e7f93783277557f3a00010a4eb259a89f1f463ba1`；独立命令计数为一次 `compile_ultra`，没有 `set_fix_hold`、incremental compile 或 `only_hold_time`。
- 冻结 runner SHA256：`7a7cebe33c9e078bd341cd93009b3a313edf194da3bf04607c93186d8ae643d7`；`bash -n` 通过。
- 冻结 contract SHA256：`205203bd9f3c3d8bac3187d66b94fae6d2bb7af99d460ca98cf427d46c24e576`；JSON 可解析，当前 `exact_files` 17/17 匹配。
- author handoff、R5 static/VCS/VCS-review/final-failure/quarantine 的内外 seals 均独立通过；五个 R5 outer-seal file SHA 与 contract 完全一致。
- preflight 使用三次样本、相邻两次 `sleep 10`；`-lt` 失败条件使 64/128/32 GiB 边界本身可通过，H0 取三次 commit headroom 最小值。
- runtime commit 使用严格 `<32 GiB`，循环内连续计数在恢复样本清零；MemAvailable、SwapFree、cgroup/OOM、碰撞在循环内为单样本锁存。
- K8、K1x8 前均重新做三样本 preflight，K1x8 后存在 final recovery；attempt 消费后的失败进入 work-root quarantine 双封，嵌套 preflight seal 会被根 seal 覆盖。
- future launch admission、R6 canonical 和 R6 attempt sentinel 当前均不存在；R5 仍为 `FAILED_OR_INCOMPLETE_DO_NOT_CITE`。
- `docs/359` SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 裁决与 claim boundary

当前只能声称 R6 source 已被独立审阅并发现阻塞问题。`dc=false`、`ppa=false`、`hold_closed=false`、`power=false`、`energy=false`、`throughput_per_area=false`、`complete_fc2=false`、`system_speedup=false`、`headline=false`。

只有 author 产生新 identity 的 bounded repair，且新的 fresh independent static hammer 达到 `P0=0 && P1=0`，主 agent 才可另建一次性 launch admission。此 r1 结果永久不授权运行。
