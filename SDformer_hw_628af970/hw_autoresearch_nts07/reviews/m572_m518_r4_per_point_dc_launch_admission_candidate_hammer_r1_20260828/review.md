# M572｜M518 r4 Fixed / rank3 两点 DC admission candidate 独立打铁评审

## 结论

**PASS，100/100，P0/P1/P2 = 0/0/0。** 两份 `launch_now=false` candidate 均通过严格 JSON、双封、身份、公平性、资源门、碰撞门和缺席性检查。它们仍然不授权 DC；现在只允许分别编写 Fixed 与 rank3 的 `launch_now=true` 单点 release，并且每个 release 都必须再经一轮 fresh independent final-release hammer 才能推荐一次精确启动。

本评审没有调用 EDA、DC、VCS、runner、remote 或大 CPU 任务，没有创建 result、attempt、true release 或 paired-comparison admission，也没有修改 `docs/359`。

## 两点身份与隔离

| 点 | top | candidate SHA256 | canonical result | attempt | 现状 |
|---|---|---|---|---|---|
| Fixed | `m518_matched_fixed_t10_atlif` | `e83e2a47319a5fca165fb918adfb64659d1d968022aa946c52e8788bd5aa82a4` | `m518_r4_fixed_setup_area_logic_only_dc_3p000ns_r1_20260828` | `.m518_r4_fixed_setup_area_attempt_consumed` | 均不存在 |
| rank3 | `m273_integrated_rank3_atlif` | `7c6fb69062707f542e310b9bcf2ab227ec0ee9397ada3d891e8dd8aea82f2958` | `m518_r4_rank3_setup_area_logic_only_dc_3p000ns_r1_20260828` | `.m518_r4_rank3_setup_area_attempt_consumed` | 均不存在 |

两点共享同一 runner、Tcl、filelist、SDC、两份 RTL corpus、slow/fast DB、3 ns、一次 `compile_ultra`、0 incremental、0 hold-fix；只允许 `point/top/result/attempt` 按候选身份分别选择。两点 result 与 attempt 路径互异，候选没有混用身份。

## 机械核验

- 两个 candidate payload、各自 member sidecar 和 outer seal 均逐字节通过。
- request、author handoff、M568 PASS100 静态评审、M555 r2 failure review、r2 quarantine 与 r2 attempt 均通过递归 member/outer seal。
- runner / Tcl / contract / filelist / SDC / Fixed RTL / rank3 RTL 的 live SHA 与 candidate 一致。
- `dc_shell`/`snps_shell` wrapper、实际 `common_shell_exec`、slow DB、fast DB 的 live SHA 与 candidate 一致；`dc_shell` realpath 为冻结 wrapper。
- 独立解析两份 source port：各 50 个有序 `(direction,width,name)` tuple，完全相同；按冻结参数展开后各 1175 个 bit-level port。两个计数保留为不同命名空间。
- runner 中预检门为 64 GiB commit headroom（三次、间隔 10 秒），runtime soft/hard 门分别为 48/40 GiB，另有 128 GiB MemAvailable、32 GiB SwapFree、cgroup、同 UID EDA collision、PID/starttime/uid/parent/exe/cmdline 身份门；`runtime_final` 更新连续计数、执行第三样本决策并要求 ACK/monitor rc。
- 两个 candidate 均是 `launch_now=false`、`max_attempts=0`、`run_dc=false`，其余 run flag 也全为 false。
- Fixed/rank3 true release、两个 result、两个 attempt 与 paired-comparison admission 均不存在。paired comparison 必须等待两个点各自双封结果以及各自 P0=0/P1=0 receipt hammer，点 runner 不得生成。
- `docs/359` SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 准入边界

本结论只把两点从“候选待审”推进到“可分别编写 true point release”。它不等于 launch authority，也不产生 timing、area、STA、hold、power、energy、throughput/area 或 paper-PPA 结果。r2 中间 Fixed QoR 继续保持 `FAIL_MATCHED_DC__SEALED_QUARANTINE__DO_NOT_CITE`，不得追认。
