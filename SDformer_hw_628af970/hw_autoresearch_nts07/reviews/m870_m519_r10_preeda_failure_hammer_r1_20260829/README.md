# M870｜M519 R10 stale pre-EDA failure hammer

裁决：`PASS_FAILURE_AUDIT__STALE_R10_RETRY_REJECTED__R16_REMAINS_ONLY_CURRENT_SUCCESSOR`。

## 直接失败原因

目标回执 `m519_r10_pre_attempt_shell_failure.1676368.receipt` 双封存通过，内容明确为
`exit_code=3`、`attempt_consumed=false`。冻结 R10 runner 第 246 行把 `\` 写进单引号
包围的 jq 程序。该字节会原样传给 jq；对冻结 M576 `review.json` 精确重放稳定返回 3，
错误为 `unexpected INVALID_CHARACTER`。8 个 `jq -e` 块的同类静态扫描只命中第 246 行。

失败点早于第 707 行首次 axis preflight、第 823 行 attempt 发布和第 1171 行首次
`dc_shell`。现场不存在 R10 canonical、attempt、work、preflight staging 或 quarantine，
也没有同 UID 的 `dc_shell/common_shell_exec/fm_shell/pt_shell` 进程和本次 R10 工具输出。
所以绝不能声称 DC started、area、timing、PPA 或性能结果。

## P0：这是已禁止的陈旧 R10 重试

磁盘上已有更早的 `m519_r10_pre_attempt_shell_failure.693765.receipt`，且 M740 已双封裁定
`run_r10_again=false`。本目标是不同 PID 的第二份 R10 pre-attempt 回执，发生在 M740 之后；
因此内部 attempt sentinel 虽仍为空，也不能恢复已经撤销的一次性 release authority。
R10 在治理口径是 `CONSUMED_NO_RETRY`。本次重试没有启动 EDA，但属于 P0 身份/执行纪律问题。

## 不能回退到 R11

R11--R15 时间线已经存在：R11、R12、R15 的 attempt 与 failure quarantine 均双封；R13、
R14 在 source static hammer 阶段失败，没有 launch。当前最高已消费身份是 R15；M800 已把
R15 三轴定为 noncitable，并仅授权一个 additive R16 **source-only** 修复，尚未授权 VCS/DC。

因此后续绝不能新建或复用 R11，也不能重跑 R10/R15。唯一当前 successor 是 additive R16：
按 M800 隔离 K8 request fault 与 legal response completion；使用全新的 runner、RTL、contract、
admission、attempt 和 canonical。R16 runner 必须保留完整 pre-attempt no-EDA 回归：真正执行所有
jq/admission predicate，并检查单引号 jq 内无字面续行反斜杠；仅跑 `bash -n` 不足以覆盖本故障。

## Claim boundary

本 review 只接纳失败原因和身份状态。R10 没有面积、时序、功耗、PPA 或性能证据；R11/R12/R15
quarantine 也不能拼表。`docs/359` SHA 仍为
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
