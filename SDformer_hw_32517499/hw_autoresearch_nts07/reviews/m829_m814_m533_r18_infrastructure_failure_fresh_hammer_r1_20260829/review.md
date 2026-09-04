# M829：M814/M533 R18 VCS 基础设施失败终态独立锤击

## 裁决

**PASS，100/100，P0/P1/P2 = 0/0/0。** R18 已永久消费并以双封的 `FAILED_DO_NOT_CITE` 基础设施失败收口。它不能重跑、恢复、重标或引用，也不能把失败归因于 RTL、TB、SVA、foundry UNIT_DELAY 模型或功能行为。

R18 的编译、展开和链接完成；`simv` 随后在可观察 Verilog 时间之前停滞。30 分钟固定门满足后，runner 的可捕获 TERM 路径产生 `runner_exit_rc=143`、`child_rc=143`、`phase=signal_term`、`failure_message=caught_term` 和 `monitor_status=cleanup_wait_rc_143`。这是一条已封存的运行态外部服务/基础设施失败，不是功能 VCS 结果。

## 双封与全清单重验

- R18 result 的 `SHA256SUMS` 含 117 个 regular-file 成员；117/117 全部复算通过。成员 manifest SHA 为 `21526247e8803cc3d1379694f70535255731e05f63d72072ffae8f3b3692394c`，外层 seal 文件 SHA 为 `69b17a01deb01906102b2730a3c7900fcbe716de0c85f0c2d1f364ed1503fb68`。
- M827 stall audit 的 3 个成员 3/3 复算通过。成员 manifest SHA 为 `b3bd30f4bf98d91273f841f8950a87fffddd08ee2737493f248fc3c6959623be`，外层 seal 文件 SHA 为 `821c3a6acfeeae62445d539f10cc0c1ed92dee8350ead10d1fc6e4f520445118`。
- result 的 `ARTIFACT_INVENTORY.json` 记录 `terminal_kind=failure`，131 个条目，并确认所有 symlink 均为内部 regular target 且内容已绑定。
- 终态回执 SHA 为 `52517f22d654a4e13e9e39b522cfbd921f1473883d8da1667886405f0b62e3f8`；`FAILED_DO_NOT_CITE` marker SHA 为 `fee567ebca2864d0d9c8c6ff42b731f6edad40882126bd761f0ee052e7cf6e36`。

## 运行前沿与 30 分钟门

- `compile.log` 为 3290 B，VCS 报告 4 个 module 完成，compile/elab/link 分别为 1.187/0.597/0.190 s；`simv` 已生成。该事实只说明编译阶段完成，不能升级为功能 PASS。
- `sim.log` 从 2026-08-29 01:20:12 起到终态始终为 480 B，SHA `4f9a83e53657ad7f20e9f97843665ff6f2f8c4b0e3923fe98cf9906f1a7cefdf`。内容只有 ASLR save/restore 提示与 VCS runtime banner；没有 Verilog 时间、13-cover、P2、held-final、六攻击、PASS 或功能失败 token。
- M827 只读现场审计确认主线程等待每秒 poll 外部 HTTP fd 的 helper；8 秒内两个线程 CPU jiffy 与 `sim.log` 均不推进。本地 FlexNet socket 与资源 heartbeat 正常，无法证明精确 proprietary 子服务，因此只允许“Synopsys runtime external-service/infrastructure stall before Verilog time”这一保守分类。
- simv 起点为 01:20:11；resource monitor 连续记录到 01:50:39，共 1806 行，所有 failcnt/under_oom/oom_kill 字段均为 0；终态 receipt 于 01:50:40 封存。即从 simv 起点到终态约 1829 s，超过固定 1800 s 门。
- `monitor_status=cleanup_wait_rc_143`；原 runner/simv/helper PID 均已消失，未发现该 R18 identity 的残留进程。资源监控没有 OOM、cgroup failcnt 或本地 license-daemon 下线证据。

## 失败归属与 claim 边界

由于 HDL scheduler 没有产生可观察时间，不能从本次结果推断任何 RTL 快慢、正确/错误、coverage、攻击覆盖、cycle、speedup、timing、area、power、PPA、energy、full-network 或论文结论。R18 的唯一合法表述是：**永久消费、双封、不可引用的基础设施失败**。

`RUN_FAILED_OR_INCOMPLETE.json`、`FAILED_DO_NOT_CITE`、成员 manifest 与外层 seal 均已闭合。禁止删除或修改 result，禁止恢复 `simv`，禁止在同一 R18 identity 下重跑，禁止把 compile 完成或 480 B banner 包装成 VCS 功能证据。

## 唯一授权的后继动作

只授权一个 additive successor **source author**，不授权 VCS/simv/license/EDA、release、attempt 或 result：

1. 冻结 exact top RTL r2、SVA r2、TB r8、macro adapter、macro binding plan 与 foundry UNIT_DELAY Verilog 模型；SHA 依次保持 `726039...`、`b9f66f...`、`cd0cf9...`、`8fd008...`、`db4075...`、`8343ac...`。
2. 完整冻结 13 个 normal minima：`dead_plus_read`、`deadline_read_write`、`same_address_forward`、`pending_plus_forward`、`full_no_credit`、`liveness_sequences`、`parent_modes`、`stalled_raw_recovery`、`stalled_raw_forward_recovery`、`stalled_raw_response_recovery`、`pingpong_overlap`、`endpoint_rows`、`all_slices`。
3. 冻结 P2 的 `consecutive_distinct_reads>=1` 与 `response_identity_checks>=2`、held-final test，以及六项攻击：dirty-reserved、stale-epoch、overflow、wrong-parent/dead-live、read-before-write、parent-only-nonzero atomic。
4. runner 的执行语义只允许两项 additive delta：将运行命令改为 `./simv -no_save`，并由 `/usr/bin/timeout --signal=TERM --kill-after=30s 300s` 包住它。timeout 且没有 HDL 进度时必须双封为 `infrastructure_timeout_before_verilog_time`。
5. 不猜测、不加入任何 telemetry 环境变量；`-no_save` 只依据 VCS 自身提示用于避免 ASLR re-exec，不能宣称关闭 telemetry 或外部通信。
6. 后继仍须走 source hammer → candidate/release → final hammer → 唯一一次 attempt 的完整新身份链；本评审本身不授权 launch。

`docs/359` 未修改，SHA 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
