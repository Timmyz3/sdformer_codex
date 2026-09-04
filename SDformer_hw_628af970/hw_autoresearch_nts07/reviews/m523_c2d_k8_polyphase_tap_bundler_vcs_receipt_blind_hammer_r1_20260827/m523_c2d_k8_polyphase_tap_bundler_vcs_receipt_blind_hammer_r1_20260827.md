# M523 ConvTranspose K8 descriptor VCS receipt-blind 独立打铁 r1

日期：2026-08-27  
证据：`results/m523_c2d_k8_polyphase_tap_bundler_vcs_r2_20260827`  
结论：`PASS_DESCRIPTOR_ONLY_DIRECTED_FUNCTIONAL_VCS__NO_DIRECT_C2_PERFORMANCE_ENERGY_PPA_OR_HEADLINE_ADMISSION`  
评分：**100/100**；P0/P1/P2：**0/0/0**

## 独立裁定

M523 r2 canonical 可准入为冻结 directed campaign 下的 **descriptor-only 功能证据**。VCS 身份为
`V-2023.12-SP1 Full64`，identity/compile/simulation RC 全为 0；`sim.log` 中恰好一条 exact
PASS。冻结 workload 完成 6 events、43 taps、8 bundles，其中 full-8 为 4、one-tap tail 为 1，
四相计数为 `6/10/10/17`。随机 backpressure 下记录 7 stall、2 same-edge replacement；tag、time、
stream-last、FIFO-full 与 sticky fault-drain 边界均真实触发。

十个合同要求的 named cover 全部非零，旧 r1 缺失的 `cp_fault_drain_complete` 本次为 1 match。
assert report 未见 assertion failure，disable log 没有具体被禁用的 assertion instance。receipt 的
schema、status、identity、测量值与 claim boundary 和独立从 sealed inputs/logs/assert/topology
恢复的结论完全一致。

这项准入不把八条 transport lane 解释为 M218/C2 weight bank。它不证明 flattened
`(source_channel,kernel_index)` weight identity、bank-conflict deferral 或 stored-weight identity，
也不准入 component bundle reduction、decoder speedup、performance、energy、power、area、timing、
STA、DC、Formality、PPA、system speedup、paper-ready PPA 或 DATE headline。

## 封存与拓扑

- canonical member/outer seal 均通过；`SHA256SUMS` 含 105 个成员；
- 当前 105 个 regular file、`TOPOLOGY.json` 和 manifest 三者的集合精确相等；
- 恰有 2 条 VCS-generated symlink；raw target 与 inventory 一致，均解析到 canonical 内的
  regular target，resolved-target SHA 一致，无 absolute/outbound/dangling link；
- receipt、contract、topology、symlink inventory 与 wrong-runner receipt 均可解析，且没有
  `NaN`/`Infinity`；
- `input_sha256.txt` 对 runner、VCS binary、RTL、SVA、TB、filelist、contract、docs/359、旧失败
  review 和 authorizing static review 的当前文件核验全部通过；两个上游 review 的 inner/outer
  seal 均通过。

canonical manifest SHA 为
`1a83609697f6e3a0da0de7775428b36d39a3f35275ef6552e8f2560e28dc29ee`，outer-seal file SHA 为
`b9f8ef8ce1dfcdb69c87c37c4e55dd81e7e22dd72a6c86bb05506e391662de51`，receipt SHA 为
`aeb99262e85962ba45d77d83c80ba47c13eb66588f9241b6b9a39271f8dfaf7b`。

## one-shot 与 negative control

wrong-runner 子调用以全零 runner SHA 触发第一道 gate，exit=10，stdout/stderr 为空，记录
`vcs_invocations=0`、`attempt_consumed=false`、`canonical_created=false`；其完成时间早于正向
attempt 发布。nested member/outer seal 均通过。

正向 attempt marker 只有 `ATTEMPT_CONSUMED.txt`、`identity.sha256`、`SHA256SUMS`、
`SHA256SUMS.seal.sha256` 四个 regular file。member/outer seal、全部 identity current-file check
通过，状态为 `CONSUMED_BEFORE_EXACT_VCS_ID_AND_COMPILE`；mtime 顺序也为 wrong-runner → attempt →
VCS identity → compile → simulation → canonical completion。

## 功能账本与 cover

| 项 | 数值 |
|---|---:|
| events / taps / bundles | 6 / 43 / 8 |
| full8 / tails1 | 4 / 1 |
| stalls / replacements | 7 / 2 |
| boundaries / cross-event bundles | 6 / 2 |
| tag / time / stream flush | 1 / 1 / 2 |
| stream-last isolation | 1 |
| FIFO maximum occupancy | 18 |
| phases 00/01/10/11 | 6/10/10/17 |
| protocol attacks | 1 |

Named cover matches 为：full8=4、one-tail=1、cross-event=2、stream-last=2、partial=4、stall=2、
same-edge=2、FIFO-full=2、protocol-fault=1、fault-drain=1。没有零命中 cover，也没有 assertion
failure signature。

## 决策

准许将 M523 写为 H67 decoder ConvTranspose 网络完整性的 descriptor support；不能把它写成
第四个加速创新或 C2 已集成。只有 M511/M513 exact decoder trace 表明值得继续时，才应另起
reviewed adapter，补齐 weight identity、8-bank mapping、conflict deferral、stored-weight proof 与
同资源 cycle/memory 证据。

本审阅未运行任何 EDA 或开源 RTL 工具，未修改输入或 `docs/359`。`docs/359` SHA 仍为
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
