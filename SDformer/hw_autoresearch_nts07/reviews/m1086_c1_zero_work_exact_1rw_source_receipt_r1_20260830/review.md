# M1086：C1 zero-work exact-1RW additive source receipt

M1086 只新增 source/test，不修改冻结的 M1056、M1072、M1074。修复对象是 M1085 在 canonical task 207 发现的 `work_cycles=0`：该任务现在不创建 psum event/grant、不改变 `last_write`，所有计数为 0，nominal/effective end 均等于已经支付的 work start；正工作任务直接调用冻结 M1056，未另写调度语义。

生产入口保持 fail-closed：work-domain preflight 与 full iterator 均为零参数，只从 canonical reader 内部重导 row/work/provenance；生产 work 只允许 exact int 的 0 或 >=15，bool、负数、1--14 均拒绝。task207→task208 的 bounded regression 与小 oracle 可供独立 hammer 使用，但不构成 full replay。

root 已用 Python 3.10 独立运行当前精确 source 的 bounded unit suite，结果 12/12 PASS；该观察没有替代带封印、带攻击面的 M1087 hammer。本 receipt 不执行 exhaustive preflight 或 full replay，不消费 attempt，也不授权 EDA/GPU/remote。下一步只允许不同作者的 M1087 bounded source hammer；其通过前不得建立或启动一次性 full-replay runner。M1074 保持 DO NOT RETRY，任何 cycle/speedup/RTL/PPA 结论均为 false。
