# M827：M814/M533 R18 VCS 运行态外部服务阻塞独立审计

## 裁决

审计通过，且只关闭“为什么没有进展”这一问题：R18 高置信卡在 Synopsys VCS 运行库的外部服务/telemetry 路径，尚未进入可观察的 Verilog 时间；不是 RTL 仿真执行慢。精确子类（更新、usage telemetry、合规核验或许可辅助）在不 attach 的约束下不能进一步断言。

本审计没有发送信号、没有 attach、没有关闭 fd、没有执行 VCS/simv/许可证命令、没有改 result/release，也不产生任何功能、周期、PPA、能量或论文主张。

## 直接证据

- R17 从资源监控首样本到仿真终止约 36.5 秒，compile.log 到 5,527,500 ps 终止约 8.7 秒。
- R18 的 simv 在 01:20:11 启动；到 01:34:41 已约 870 秒。`sim.log` 从 01:20:12 起固定为 480 B，只含 ASLR 提示和 VCS runtime banner，没有任何仿真时间或 TB token。
- 主线程 231606 在 futex 等待 231704；231704 每秒 poll 两个 fd。fd10 是 `10.17.22.76:60454 -> 198.182.50.26:80`，TCP 为 ESTABLISHED，快照时发出 1804 B、已确认 1805 B，但应用数据接收为 0，且出现 4 次重传。
- 连续 8 秒采样中两个线程 CPU jiffy 均不变，sim.log 也不增长。
- 本地 FlexNet 的两个 simv socket 均为 ESTABLISHED；resource heartbeat 每秒增长，session/user failcnt 和 OOM 计数全为 0，主机可用内存约 397 GiB。因此没有资源饥饿或本地 daemon 下线证据。
- VCS 自己的 banner 明示产品会联系 Synopsys server；本进程映射的 `libvcsnew.so` 与 `libreader_common.so` 含 `/tmp/.snps_telemetry` 标记。这支持“Synopsys runtime 外部服务路径”分类，但不足以把精确功能说死。

## Fail-closed 处置

保守等待到 simv 启动满 30 分钟，即 2026-08-29 01:50:11+08:00。若届时仍同时满足：sim.log 480 B、无 HDL token、CPU 不推进、外部 socket 无应用数据接收、heartbeat 正常且无 OOM，则继续等待不再增加验证价值。

此后 root 可以对完整 R18 进程组走 runner 已有的可捕获 TERM 失败路径，禁止先用 SIGKILL。终止后必须验证 `RUN_FAILED_OR_INCOMPLETE.json`、`FAILED_DO_NOT_CITE`、`SHA256SUMS` 和外层 seal；若 graceful seal 自身失败，则作为未封存但已消费的 infrastructure failure 隔离，R18 身份不得重跑、恢复或引用。

## Successor 最小修复

不改 RTL/TB/SVA，只创建 additive runner identity：

1. 使用当前 sim.log 明确记录的 `-no_save`，避开 ASLR 触发的 simv re-exec；该参数不应被描述成关闭 telemetry。
2. 用已安装的 `/usr/bin/timeout` 给 simv 加 300 秒 wall-clock gate，并保留 30 秒 TERM→KILL 兜底：`/usr/bin/timeout --signal=TERM --kill-after=30s 300s ./simv -no_save`。
3. timeout 且没有 HDL 进度时，明确封为 `infrastructure_timeout_before_verilog_time`；所有既有功能、coverage、P2、六攻击、资源和双封门保持不变。
4. 不猜测、不加入任何未由本机文档或工具输出证实的 telemetry-disable 环境变量。

## Claim 边界

功能 VCS、RTL、TB、SVA、coverage、攻击、timing、cycle、speedup、area、power、PPA、energy、full-network、paper-citable 和 headline 全部为 false。
