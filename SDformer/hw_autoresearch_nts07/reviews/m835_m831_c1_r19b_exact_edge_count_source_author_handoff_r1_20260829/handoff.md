# M835：C1 R19b exact-edge-count source author handoff

M835 是 M831/R19 的 additive source-only 计数修复。原 runner SHA `2db504...` 保持不变；RTL、TB、SVA、宏适配、binding plan、foundry `UNIT_DELAY`、timeout/no_save、13 normal、P2、held-final、六攻击、资源与终态门均未修改。

修复内容只有一项：`require_regular_sha` 的冻结总数从错误的 94 更正为 **95 个唯一逻辑调用**，其物理分解为 **94 个单行 + 1 个跨行**。唯一跨行项是 runner 第 1125–1126 行的 `docs/359` 校验。新增 continuation-aware Python 3.6 parser 会先剔除 heredoc，再合并反斜杠续行；synthetic self-test 与 exact runner 均通过。

作者重新运行了 TB static、34/266/21 closure、三负变异、fake timeout 四路与 pre-mkdir rc86 dry-run，全部通过且 live VCS/license/compile/simv/result 副作用为 0。

本 handoff 不是 fresh hammer，不是 release，也不授权 launch。下一步只能由不同 reviewer 按 M836 request 做 source fresh hammer；只有 PASS100、P0/P1/P2=0/0/0 才可继续后续 admission 设计。
