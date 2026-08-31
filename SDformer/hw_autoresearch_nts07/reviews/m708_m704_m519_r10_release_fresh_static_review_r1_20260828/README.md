# M708｜M704 M519 R10/r3 DC release fresh static review

## Verdict

**GO_ONE_M519_R10_DC_ONLY_ATTEMPT__FINAL_LIVE_RECHECK_REQUIRED**，`99/100`，
`P0=0, P1=0, P2=1`。

M704 additive R10/r3 已关闭 M701 的两处 `set -u` 同声明依赖，并在不运行 EDA
的条件下通过作者 selftest、fresh failure injection、exact-SHA/双封与 20 项安全源码/
JSON/identity mutation 检查。R10 canonical、work、preflight staging 和 attempt sentinel
均不存在，因此一次性 DC attempt 尚未消耗。

这个 verdict 只授权**一次 R10、DC-only、三轴 setup/area、logic-only** attempt；它不
授权 VCS、Formality、PT、PTPX、remote，也不准入任何 PPA、性能或论文 headline。

## Exact one-shot command

以下命令只有在 root **紧邻执行前**完成 fresh live recheck 且全部 gate 通过时才有效：

```bash
env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 M519_R10_EXPECTED_DC_RUNNER_SHA256=7dc7d79c27b85820c621ac142e104cc155afd949ffa8b2ec46dd7279a314d27f M519_R10_EXPECTED_DC_LAUNCH_ADMISSION_SHA256=f4bccc501dea216396d2755ef6b1f627209efe18346701cd5d448367cf4a3424 /home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/dc_handoff/scripts/run_dc_m519_r10_setup_area_three_axis_exact_sha_r3.sh
```

不得增加、删除或改写环境变量，不得重跑。任何 preflight rejection 不消耗 attempt；
一旦 runner 建立 sealed attempt sentinel，即使后续失败也算唯一 attempt 已消耗。

## Mandatory final live recheck

本次是静态 review，当前 live state **没有**被当成 launch evidence。执行命令前必须重新
确认：

1. R10 canonical、attempt、work 和 preflight staging 仍为空；
2. 同 UID 的 DC/FM/PT/VCS/simv collision 为零；
3. 三次、间隔 10 秒的 CommitLimit−Committed_AS 均不少于 64 GiB；
4. MemAvailable 均不少于 128 GiB，SwapFree 均不少于 32 GiB；
5. cgroup failcnt、under_oom、oom_kill 均为零；
6. 不向任何 foreign process 发信号，也不得因为 foreign process 属于不同 UID 就豁免
   全局资源门；
7. runner SHA、admission SHA 与上方命令仍精确一致。

runner 会在 attempt 消耗前再次执行同样的三样本 preflight，并在三轴之间及最终恢复点
重复检查；static GO 不能绕开 machine gate。

## Fresh checks performed

- M704 handoff 内部 `SHA256SUMS` 4/4 通过，manifest SHA 为
  `ca484c753f4615050e357a131f52fa455f71faf0e44c52c7894a852c8eee2931`，
  outer seal 精确绑定。
- runner / contract / admission SHA：
  `7dc7d79c...14d27f` / `2ba563ed...501a4e` / `f4bccc50...a3424`。
- contract 与 admission file-sidecar/outer-seal 均通过。
- M694 review/manifest/outer：`8026ceb1...ac0b` / `cbd67895...44b6` /
  `c6903561...918`；状态 exact match。
- M701 review/manifest/outer：`8bf29fa5...33a4` / `70333157...e87f` /
  `ff8a1161...d0f4`；状态 exact match。
- R9 runner/contract/admission、R8 admission、M580 review 与 docs359 SHA 全部保持作者
  contract 中的冻结值。
- `static_no_eda_test.sh` fresh 10/10 PASS；成功 selftest 与注入退出码 86 failure
  receipt 均未启动 EDA。
- `static_boundary_mutation_review.py` 20/20 PASS；所有 mutation 只在内存中发生。

## Set-u and execution-boundary closure

- `payload`、`sidecar`、`outer` 已分离声明后赋值；历史 compound local 模式不存在。
- `id`、`mode`、`point` 已分离声明后赋值；同类第二缺陷不存在。
- byte-order 静态检查为：
  `bash -n (6306) < selftest (6335) < admission (7857) < preflight (36879) <`
  `attempt (41606) < dc launch (58863)`。
- 注入 pre-attempt failure 返回 86，生成 fresh noncanonical `FAILURE.txt`、
  `SHA256SUMS`、`SHA256SUMS.seal.sha256`，并明确
  `attempt_consumed=false`。

## Preserved P2

`P2=1` 是共享主机 live-state 风险，不是静态源码失败。M694 的 foreign/same-UID/
resource 风险没有被删去、降级或转换成默认许可：64/128/32 GiB 门、同 UID collision、
runtime exact birth tuple、final synchronous ack 与不信号 foreign process 的边界全部保留。

## Claim boundary

- DC 尚未启动或完成。
- area/setup timing/hold closure/power/energy/throughput-per-area 均未准入。
- `paper_ppa_ready=false`、`complete_fc2=false`、`system_speedup=false`、
  `headline=false`。
- 任何结果必须另做 fresh receipt-blind result hammer，才可决定是否进入 Table-A。
- `docs/359` 未修改，SHA 仍为
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

