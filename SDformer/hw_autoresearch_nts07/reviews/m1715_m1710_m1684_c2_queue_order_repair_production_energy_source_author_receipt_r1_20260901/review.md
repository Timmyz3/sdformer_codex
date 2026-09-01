# M1715 作者源收据

状态：`PASS_M1715_AUTHOR_SOURCE_ONLY__READY_FOR_M1716_DIFFERENT_AUTHOR_REVIEW__NO_EDA`

M1710 的双封失败为 `SOURCE_CHAIN`、`attempt_consumed=false`、VCS/simv/SAIF/PTPX=`0/0/0/0`。M1715 不重试或修改 M1710，而使用全新命名空间。

唯一行为变化是队列顺序：M1715 先以阻塞 `flock(LOCK_EX)` 等待共享 Synopsys 队列，再执行锁后同 UID/祖先感知 collision gate，随后重绑定六个直接执行源的 exact SHA、active-force、initreg 与 lexists；attempt 前重复 collision 与 runtime rebind。每次 VCS/PTPX 调用前的 collision rescan 保留。

两套 Python 运行时均通过 12/12；7/7 队列顺序、rebind、lexists、failure-binder 变异均被拒绝。源自检输出逐字节一致。M1684 五个 workload、K8/K1×8 两轴和 `2/10/10/10` 预算未改。

本收据没有启动 license query、VCS、仿真、SAIF、PTPX 或任何 EDA，也没有创建 attempt、result 或 release。下一步必须由不同作者执行 M1716 source hammer；只有随后双封的 M1717 才能授权一次 M1715 attempt。
