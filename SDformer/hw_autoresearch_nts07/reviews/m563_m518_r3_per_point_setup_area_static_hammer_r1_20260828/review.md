# M563｜M518 r3 per-point setup/area DC source static hammer

日期：2026-08-28  
模式：fresh independent、source-only、read-only  
结论：**FAIL_STATIC__NO_POINT_ADMISSION__NO_LAUNCH**  
评分：**92/100；P0/P1/P2 = 0/2/0**

本审阅没有调用 DC、VCS、runner、远端任务或大 CPU 任务；没有创建 point
admission、result、attempt 或 paired comparison；没有修改 `docs/359`。

## 裁决

r3 已正确修复 M555 的主体问题：Fixed/rank3 的 canonical result、attempt、
failure quarantine 和未来 admission 完全分离；paired comparison 只是后置
schema；50 个 source declaration tuple 与 1175 个 DC bit-level ports 分口径；
Tcl 仅一次 `compile_ultra`，没有 incremental/hold fix/hold report；结构化
`check_design`/`check_timing` 和 area macro/black-box=0 取代了宽泛 dc.log
grep；actual `common_shell_exec` 的 PID/starttime/UID/parent/exe/NUL-safe
cmdline 也被绑定。

但是当前 runner 仍有两个 P1 fail-closed 缺口，因此 **Fixed 与 rank3 都不得签
launch admission**。

## P1 findings

### P1-1｜runtime-final 未并入 `<48 GiB` 连续三样本门

普通 runtime 样本在 runner 444--455 行更新 `soft_bad` 并在三次连续低于
48 GiB 时失败。`runtime_final` 在 482--497 行重新采样后，只检查 `<40 GiB`、
MemAvailable、SwapFree、cgroup 和 collision；它既不更新 `soft_bad`，也不检查
`soft_bad >= 3`。

因此序列“倒数第二、最后一个普通 runtime 样本低于 48 GiB，final 仍低于
48 GiB 但不低于 40 GiB”会在 final ACK 中错误放行。contract 冻结的是
`runtime_soft_consecutive_samples=3` 且 `runtime_final_sample_is_mandatory_and_gated=true`；
当前实现没有对 final 完整应用该策略。

最小修复：final snapshot 后用同一阈值更新/清零 `soft_bad`，再依次检查 hard
40 GiB immediate、`soft_bad >= 3`、Mem/Swap/cgroup/collision，并把 final 的
soft gate 决策写入 ACK。

### P1-2｜M555/r2 前序冻结 seal 只被记录，没有在 launch 前验真

contract 冻结了 M555 review、r2 quarantine、r2 attempt 三个 outer-seal 文件
SHA。runner 116--148 行只验 contract/exact_files/tool/DB；到 attempt 已消费后，
才在 361--371 行用 `sha256sum` 把三个**当前 live** seal 写入
`input_sha256.txt`。runner 没有：

- 将三个 live outer-seal 文件 SHA 与 contract 的冻结值比较；
- 在各目录执行 member manifest 和 outer seal 的递归 `sha256sum -c`。

因此前序包若在本次 static review 后被替换，runner 仍可消费 attempt 并启动
DC，只会把替换后的身份记账。这不满足 request 的 SHA/双封门，也削弱了 r2
`DO_NOT_CITE` 先例链。

最小修复：在 preflight 与 attempt 消费前，按 contract 三个 expected SHA
逐一 `expect` outer-seal 文件，并对 M555 review、r2 quarantine、r2 attempt
分别递归验证 member manifest 和 outer seal；任一失败必须在 point attempt
消费前退出。

## 已通过的静态门

- runner `bash -n` PASS；contract/request/handoff JSON PASS。
- runner/Tcl/contract/contract outer-seal/docs359 live SHA 全部命中冻结值。
- contract 的 7 个 exact files、DC wrapper/actual executable、slow/fast DB 全部
  命中冻结值；`dc_shell` realpath 为冻结的 `snps_shell`。
- request、author handoff、M555 review、r2 quarantine、r2 attempt 的 member
  manifest 与 outer seal 当前均递归通过。
- 两份 RTL 的有序 source tuple 均为 50，且 direction/width/name 完全相同；
  runner/Tcl 将其与 1175 个 DC bit ports 明确分离。
- Tcl 只有一个命令级 `compile_ultra`；incremental、hold-fix、hold-only 和
  min-delay report 命令均为 0。未来结果只可称 setup/area-only，不可称 hold
  closed、full STA、power、energy 或 paper PPA。
- runner 没有对完整 `dc.log` 作宽泛 black-box grep；postcheck 使用
  `check_design_ok=1`、`check_timing_ok=1`、1175 bit ports 与 area report 的
  macro/black-box exact zero。
- preflight 为 64/128/32 GiB、3 样本、间隔 10 秒；普通 runtime 的 48 GiB
  连续三次、40 GiB immediate、Mem/Swap/cgroup/collision immediate 均存在。
- Fixed/rank3 的 future admission、canonical result、attempt sentinel 和 paired
  admission 当前全部不存在；r3 没有运行证据或可引用 QoR。
- M555/r2 quarantine/attempt 保持双封，r2 Fixed 中间 QoR 仍为 DO_NOT_CITE。

## 最小后续

仅修 P1-1/P1-2 并生成新 runner/contract/request identity；重新做 fresh
source-only static hammer。P0/P1 清零后，root 才能为 Fixed、rank3 分别生成
独立双封 one-shot admission。任何一点失败不得消费另一点；paired comparison
仍须等两点各自 clean receipt review 后再单独准入。

`docs/359_DATE终局冻结_20260813.md` SHA256：
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

