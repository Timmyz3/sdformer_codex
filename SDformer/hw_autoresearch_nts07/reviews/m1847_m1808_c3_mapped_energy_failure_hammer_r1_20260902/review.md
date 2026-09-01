# M1847｜M1808 C3 mapped-energy 唯一生产失败独立审阅

结论：**审计 PASS（99/100），M1808 生产准入 FAIL_CLOSED；P0=0、P1=1、P2=0。不得重跑 M1808，不得产生 C3 功耗/能量数字。**

## 故障定性

这不是 token-only 或 checker 文本匹配问题。唯一的 mapped VCS compile 已成功；唯一的 mapped simulation 在 31.7 ns 到达第三个 post-reset quiescent edge 后，TB 第 287 行因十一项 public debug counter 的聚合 `$isunknown` 仍为真而 `$fatal`。因此：

- 三拍 settling 修复没有闭合，至少一个 debug counter 在边界仍含 X/Z；
- 没有进入 configuration、warmup、数值 scoreboard workload 或 SAIF window；
- `PASS` token 为 0 是前述 `$fatal` 的后果，不是根因；
- 由于功能流量从未开始，本次不证明 ATLIF 算术/scoreboard 功能错误；
- M1808 的聚合 fatal 没打印逐 counter 值，所以不能声称第三拍仍未知的具体 counter 集合。M1806 的首拍六项定位只能作为前序上下文，不能替代本次边界证据。

## 执行与封存

- attempt latch、ordinary failure、preflight-governance quarantine 均双封校验通过；M1841 release 与 M1842 release audit 身份也一致。
- production 数量严格为 `VCS compile=1 / simv=1 / SAIF=0 / PTPX=0`，`automatic_retry=false`。
- canonical result 不存在；raw build 仅保留在 `private_build.unsealed_do_not_cite`。
- 最初的 source-chain preflight failure 是 `attempt_consumed=false`、四项工具计数全 0，和本次已消费生产失败清楚分离。
- 本审阅未运行 EDA、GPU、远端、license query、commit 或 push，也未修改/删除/移动任何前序证据。

## 论文边界与下一步

M1808 不产生 mapped functional、SAIF、power 或 energy 准入，也不能用于 headline。既有 M1456 C3 area/timing 证据不因本次 activity campaign 失败而失效。

唯一合法后续是：保持 M1808 已消费状态不动，在新 namespace 先做逐 counter 的只读边界诊断或修复 mapped reset/initialization 根因，再走新的不同作者 source review 与 exact release。禁止把 checker 的 `runtime PASS count` 异常当作只需改 token 的修复。

