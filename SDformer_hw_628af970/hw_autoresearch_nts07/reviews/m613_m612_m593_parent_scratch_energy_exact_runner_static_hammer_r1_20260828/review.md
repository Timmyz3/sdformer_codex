# M613｜M612/M593 parent-scratch energy exact-runner fresh static hammer

日期：2026-08-28  
模式：`FRESH_INDEPENDENT_READ_ONLY_RUNNER_STATIC_HAMMER__NO_FORMAL_EXECUTION_EDA_GPU_REMOTE`  
裁决：`PASS_RUNNER_STATIC__TRUE_LAUNCH_ADMISSION_AUTHORING_ONLY__NO_EXECUTION`  
评分：**100/100**；`P0/P1/P2 = 0/0/0`

## 1. 裁决

M612 r4 关闭了 M607 的 P1/P2，并保持 M606 已关闭的 exact-result、terminal rehash 与 no-replace 合同。
本次只在临时命名空间执行 fault injection 和手工 synthetic tree verifier；没有运行 formal analyzer，没有创建
canonical result/attempt/consumed/authorization，没有 EDA/GPU/remote，也未修改 docs/359。

本 PASS **只允许后续作者另行起草 M614 true-launch admission**。本评审自身不授权直接执行；M614 仍须绑定本
review 的 SHA、manifest/outer seal、M612 runner/adapter SHA、唯一 canonical 坐标及 component-only claim。

## 2. M607 P1｜arbitrary-entry quarantine

独立执行 30 组矩阵：`RESULT / ATTEMPT / CONSUMED / runner staging / adapter internal staging` 五类坐标，分别
注入 live symlink、dangling symlink、FIFO、Unix socket、普通文件、含 regular/subdir/symlink/FIFO/socket 的
嵌套目录。结果为 **30/30 PASS**：

- 每次只生成一个 `m612_energy.failed_or_incomplete.*` qfinal；
- qfinal exact member set 只有 `filesystem_evidence.json`、`failure_receipt.json` 与两级 seal；全部是 plain file；
- 所有 canonical、`.m612_energy.failed_raw.*`、`.m612_energy.failed_quarantine.staging.*` 均 absent；
- live-symlink 外部 target bytes 保持不变；symlink target 以原始 bytes base64 记录；
- FIFO/socket 不被读取，嵌套特殊成员只被 `lstat` 序列化并 no-follow 删除。

另做一组同时混合六项（四个 canonical + 两个 adapter staging），结果 present evidence=6、qfinal=1、raw/qstage=0、
canonical=0、target unchanged。M607-P1-01 关闭。

## 3. M607 P2｜lexical path before resolve

runner `plain_chain` 与 adapter `plain_chain_before_resolve` 对中间/末端 live/dangling symlink 共 8/8 拒绝；
authorization endpoint/intermediate 四组、static caller identity endpoint/intermediate 四组也全部在内容使用前拒绝。
实现先 `abspath` 固定绝对 lexical 坐标，再逐 component `lexists+lstat`，最后才比较 realpath。M607-P2-01 关闭。

人工放置 `.m612_energy.failed_raw.manual` 与 `.m612_energy.failed_quarantine.staging.manual`，两次 preflight 均
fail-close；不存在带病重启窗口。

## 4. M604/M606 回归

手工构造完整、双封的 synthetic final result（不调用 analyzer）先通过 baseline exact verifier；随后逐项重封攻击：

- 缺失顶层字段、额外顶层字段、错误 frozen source identity；
- 错误物理能量方程；
- `RUN_COMPLETE=FAIL_NOT_COMPLETE`；
- CSV/JSON 行漂移；
- terminal five-member map 缺项；
- terminal adapter identity 漂移。

八项均被拒绝。M612 绑定 byte-identical M606 core SHA
`3896c348b809b3094396bc64f63ffc7802866b3a5034e222c8addba8b21640fa`；publish 后和 consume 后仍顺序重验
authorization、static identity、final result，且 consumed attempt 使用 exact-member double-seal verifier。

adapter publish、result publish、attempt consume 三处 `RENAME_NOREPLACE` 碰撞全部拒绝，source=`SOURCE`、
target=`TARGET` bytes 均不变。

## 5. 身份与 absence

- shell runner：`b6082e1492b8d4885addb0343970917b79073e8b9cad1414ffac01ecff55f98f`
- Python runner：`82cf5a6d7d33a78246b9c88fa5a4db50be4821b4a30c8ffb198f114a59b76727`
- adapter：`65f6f006c62a5e7732eefc62106af14b76eb708567da995a3b45ad9a9d78daba`
- source candidate：`e455a16aded4e077313e563885e5924da8cd866e61103086922b44efd6e8fe23`
- preflight：`PASS_M612_M593_SOURCE_PREFLIGHT_ONLY__NO_RESULT_ATTEMPT_OR_LAUNCH`
- 无授权 execute：exit 70，result/attempt/consumed/M614 auth 均 absent。
- docs359 SHA：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 6. Claim boundary

本评审只准入 runner 的 fail-closed 静态执行合同。它不产生、也不准入 component energy 数字；
`38.2283079189%` / `1.2622562287 mJ` 仍须一次被授权的 bounded run 和独立 result hammer 后才可升级，且即便升级
也只是九个 parent-scratch macro、十个 frozen sampled inference 的 component model，不是 camera-frame/full-network/
silicon/system-energy/headline 数据。
