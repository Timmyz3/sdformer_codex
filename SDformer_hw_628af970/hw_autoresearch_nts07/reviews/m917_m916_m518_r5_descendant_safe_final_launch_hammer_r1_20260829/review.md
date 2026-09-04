# M917/M916：M518 r5 descendant-safe Fixed DC 最终静态打铁

本评审只允许运行 `independent_static_fault_hammer.py`；它不得调用 DC、VCS、Formality、PT、PTPX 或许可证工具。只有 100/100 且 P0/P1/P2=0/0/0 时，M917 admission 中的唯一 Fixed corrective attempt 才可由 root 在新的 live go/no-go 后消费。

重点故障注入覆盖：真实 `/proc` 进程树的 direct child、grandchild、external sibling、错误 root starttime、root 死亡后的 orphan；独立 `setsid` process-group 排空；移除 descendant 调用或 safe HOME 的源码突变；incoming HOME 与错误 runner SHA 的 pre-EDA 负例。原 r4 quarantine 保持 DO_NOT_CITE，原 r4 attempt 保持 consumed；新身份在 hammer 期间必须完全为空。

唯一命令必须使用 `env -i`、固定 `/usr/bin/bash`、固定 runner SHA 与 admission SHA。该命令只授权 logic-only、ideal-clock、ZeroWireload、3 ns、单次 compile_ultra 的 Fixed setup/area 点；不授权 rank3、hold closure、STA、功耗、能量、PPA、系统倍速或 headline。
