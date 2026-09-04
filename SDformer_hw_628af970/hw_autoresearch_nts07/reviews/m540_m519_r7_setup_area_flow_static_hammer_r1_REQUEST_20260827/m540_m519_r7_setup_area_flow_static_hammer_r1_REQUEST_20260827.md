# M540 / M519 R7 setup-area flow fresh independent static-hammer request

只读审阅以下新 R7 identity；禁止运行 DC/VCS/PT/PTPX/Formality、CPU 大任务或远端任务，禁止创建 launch admission，禁止修改 R5/R6、`docs/524` 或 `docs/359`。

必须逐条审计：

1. contract `exact_files` closed set 的 launch-time 全量 path/SHA 验证，以及 future admission closed identity key set 与 contract path/SHA 的逐项交叉一致；
2. loop 与 `runtime_final` 是否确实共用同一资源判门、final 是否更新连续 commit 计数、是否在 exact child exit 后同步采样并写由父 runner 检查的 ack，monitor rc/liveness 是否 fail-closed；
3. campaign `(PID,starttime,UID,exe)`、每级祖先 `(PID,starttime)` 二次校验是否覆盖 liveness、collision exclusion、TERM 和 cleanup，PID reuse 是否只锁存不误排/误杀；
4. preflight PID-tree、每个 runtime sample 和 highwater 是否完整、NUL-safe 地记录 comm/executable/cmdline/starttime 并进入嵌套/根双封；
5. R7 Tcl 仍仅一次 `compile_ultra`、零 incremental、零 pre-CTS hold-only，三轴 setup/area 条件一致。

只有 fresh reviewer 得到 `P0=0 && P1=0`，主 agent 才能另建一次性双封 launch admission。本 request 不授权运行。
