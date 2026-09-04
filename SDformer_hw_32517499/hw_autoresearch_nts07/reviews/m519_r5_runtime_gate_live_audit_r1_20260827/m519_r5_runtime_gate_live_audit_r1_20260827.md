# M519 R5 DC 运行期资源门只读独立审计 r1

## 裁决

**96/100，P0=0、P1=1、P2=1。** 现有 R5 runner 的资源门存在可证明的 fail-closed 假杀风险：启动前和运行期复用了同一个 `CommitLimit-Committed_AS >= 64 GiB` 绝对门，因而 DC 自身启动后的合法 commit 也会压低该全局 headroom 并被锁存为失败。本审计只读、未向进程发送信号、未改运行目录、未启动任何 EDA/仿真/CPU DSE。

截至 `2026-08-27T20:11:11+08:00` 的 live prefix 已经触发该问题：92 个 runtime 样本中 69 个低于 64 GiB，故当前 `runtime_resource_latch` 的最终语义已不可恢复为 0；即使 K1 的 DC 子进程最终正常返回，R5 runner 也必须 fail-closed。此结论只针对资源门和预期 quarantine 语义，**不是最终 run receipt hammer**；R5 结束后仍必须对最终 quarantine、child/monitor rc、信号来源、封印和文件完整性做另一轮独立失败收据审阅。

## 冻结观察点与身份

- runner：`dc_handoff/scripts/run_dc_m519_r5_channel_local_fault_three_axis_exact_sha.sh`，SHA256 `ec20959d83c7e3e7f027e9cf34792b73361871644348e8aabbaaca5904473519`。
- launch admission：`contracts/m519_r5_channel_local_fault_dc_launch_admission_r1_20260827.json`，SHA256 `2b564af969bdec98c64f25471d0086dca7a99c1f9f3b1c539d17955db2261a7c`。
- recovery contract：`contracts/m519_r5_channel_local_fault_recovery_contract_r1_20260827.json`，SHA256 `779180ed7ca889a92c83273476f6d70a970ed5f8a713e235fd18c4600919160a`。
- live work identity：`dc_handoff/runs/.m519_r5_channel_local_fault_dc_work.4165439`；审计时 K1 仍在运行。
- live log：上述目录的 `k1/resource_runtime.log`。观察冻结为前 92 行，prefix SHA256 `0b17ea99f97657e90828d1ea4674a2f0d2ff5ff4da7cfde28a01853ac14bf440`。后续追加不改变此前缀身份。
- docs/359 SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`；本审计未修改它。

## 为什么是假杀风险，而不是安全门正常工作

Runner 的 `m519_resource_snapshot` 无论 label 是 `preflight_*` 还是 `runtime`，都要求：

1. commit headroom `>= 67,108,864 KiB`；
2. `MemAvailable >= 134,217,728 KiB`；
3. `SwapFree >= 33,554,432 KiB`；
4. cgroup `failcnt=0`、`under_oom=0`、`oom_kill=0`。

启动前三次样本实际在同一秒内完成，数值完全相同：headroom `72,314,884 KiB`（68.964848 GiB）。它离 64 GiB 运行门只剩 `5,206,020 KiB`（4.964848 GiB）。全局 `Committed_AS` 会计包含当前启动的 DC 及其工作进程的可提交虚拟内存，所以“工具自身合法 commit 不超过 4.96 GiB”被无意中变成了运行准入条件。审计时仍存活的 DC 主进程 `VmPeak=4,359,996 KiB`，单这一进程的虚拟高水位已经接近全部启动余量；运行中短时优化子进程还可能额外增长。

冻结的 92 个 runtime 样本显示：

| 指标 | 观察值 |
|---|---:|
| commit headroom 最大值 | 72,314,884 KiB（68.964848 GiB） |
| commit headroom 最小值 | 59,595,656 KiB（56.834846 GiB） |
| 相对 preflight 最大下降 | 12,719,228 KiB（12.130001 GiB） |
| 低于 64 GiB 的样本 | 69 / 92 |
| 首次低于 64 GiB | 2026-08-27T19:56:29+08:00，66,207,320 KiB |
| `MemAvailable` 最小值 | 408,921,680 KiB（389.978104 GiB） |
| `SwapFree` 最小值 | 57,265,404 KiB（54.612545 GiB） |
| cgroup fail/under-oom/oom-kill 异常 | 0 / 92 |

该日志是全局会计，不能把 12.13 GiB 的全部下降逐页归因给 DC；因此本审计不声称“全部下降都由 DC 导致”。但假杀风险仍是可证明的：runner 不做归因，任何由 DC 自身带来的 headroom 下降都与外部压力同样锁存；而当前门已在 `MemAvailable`、swap 和 cgroup 三组独立危险信号均有巨大余量时连续失败。故“64 GiB prelaunch reserve”可以作为是否允许启动的条件，却不能原样作为运行期不允许工具消费的 floor。

## 最小 R6 恢复建议

R6 应仅修资源监控合同和 runner，不改 RTL、Tcl、filelist、SDC、库、VCS 证据或三轴顺序；仍需全新静态审阅、launch admission 和一次性 attempt 身份。

### 1. Prelaunch 保持保守，但让三样本真正独立

- 保留 `commit headroom >= 64 GiB`、`MemAvailable >= 128 GiB`、`SwapFree >= 32 GiB`、cgroup 三项全 0。
- 三个 prelaunch 样本之间间隔 10 秒，而不是当前同一秒内连续读取；记录 `H0=min(三次 headroom)` 作为本次 campaign baseline。
- 保留全局 DC/FM/PT 无碰撞、当前 uid VCS/vlogan/simv 无碰撞；在创建 attempt 前重新做最后一次相同检查。
- 不建议仅把 prelaunch 提到 72 GiB 再继续复用 64 GiB runtime floor：这只是在启动前预留固定 8 GiB，不能覆盖工具/设计规模变化，也没有解决“自身消耗被当作外部故障”的口径错误。

### 2. Runtime 使用危险红线，不再重复启动 reserve

建议最小绝对门为：

- `commit headroom >= 48 GiB`；
- `MemAvailable >= 128 GiB`；
- `SwapFree >= 32 GiB`；
- cgroup `failcnt` 不增加，`under_oom=0`、`oom_kill` 不增加。

48 GiB 不是性能或 PPA 参数，只是宿主安全红线；它比本次已观察最小值 56.835 GiB 低约 8.835 GiB，同时仍保留远大于当前 DC 进程工作集的全局 commit 缓冲。任何一次绝对红线或 cgroup 异常仍应锁存、停止后续 K8/K1x8、封存 quarantine。

### 3. 增加 delta/high-water 证据，避免放松外部压力检测

- 每 10 秒记录 campaign 进程树 PID、`VmPeak/VmSize/VmRSS/VmSwap` 高水位，以及全局 headroom 相对 `H0` 的下降。
- 将 `H0-H(t)` 与 campaign 自身进程树的虚拟/驻留高水位并列写入日志；先作为诊断，不把不精确的 `VmSize` 抵扣直接做 PASS 条件。
- 若要设置 delta 门，应采用“双门”：绝对 48 GiB 红线始终有效；另在 campaign 进程树没有相应增长而全局 headroom 额外下降超过 8 GiB 时锁存 external-pressure。不能只用 `H0-H(t)`，否则仍会惩罚自身合法高水位；也不能简单把全部 `VmSize` 加回，避免共享映射/未计费映射导致过度抵扣。

### 4. Runtime collision 只拒绝新外部进程

- 每个 runtime 样本检查新的 DC/FM/PT 与当前 uid 的 VCS/vlogan/simv，但必须排除 runner 已记录的 campaign 子孙 PID。
- 检到外部碰撞时锁存，不向外部进程发送信号；当前点自然结束或按新合同的安全退出流程处理，并 quarantine。

## 严格边界

- R5 的 64 GiB runtime 失败已经发生，不能事后改阈值把当前 attempt 变成 PASS。
- 当前 K1 任何 raw area/timing/netlist 都不得引用；K8/K1x8 若未运行也不得补写或推断。
- 本审计不授权 R6，不授权重跑，不授权修改现有 work/quarantine。
- 只有 R5 最终退出后的独立失败收据 hammer，才能确认 quarantine 双封、最终 rc、signal 和实际执行到的 architecture points。

