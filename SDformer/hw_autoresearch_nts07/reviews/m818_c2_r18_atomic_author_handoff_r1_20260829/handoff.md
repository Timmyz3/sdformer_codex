# M818/C2 R18 attempt-publication accounting source handoff

M818 是 M814 唯一 P1 的 additive 修复，不改 M803 RTL/SVA/TB/filelists、五档 exact 周期门或 M813 旧文件。当前只是 source-only package；不授权 VCS、simv、license 查询、formal attempt/result 或任何 EDA。

## 修复

- attempt publication 被拆成三个显式阶段：平坦三件套 generic/exact 预校验、`renameat2(RENAME_NOREPLACE)`、shell 成功锁存后独立 canonical post-verify。
- failure receipt 不再只信 shell 内存标志。它同时审计 canonical attempt、source stage 与 exact identity：shell 已锁存或 canonical exact identity 均判 consumed；在 publication/postcheck 阶段 canonical 已出现且 stage 已移走时，即使 canonical 被注入损坏导致 post-verify 失败，也保守判 consumed。
- 发布前失败时 canonical 不存在而 stage 仍在，receipt 精确判 `attempt_consumed=false`；no-replace collision 中 attacker 目标不精确且 stage 仍在，同样判 false。
- failure receipt 新增 `attempt_publication` 结构，封存 authority、phase、shell latch、canonical/stage 存在性及 identity 校验结果。

## 动态验证

- Python 3.6.8 下 10/10 unittest 通过。新增 actual CLI receipt 注入同时覆盖发布前失败和 rename 后 post-verify 失败：前者双封 receipt 为 false，后者即使 canonical `attempt.json` 被损坏仍为 true。
- strict duplicate/nonfinite JSON、flat double seal、result no-replace、attempt collision、PRE/POST non-paper failure 与 primary collision fallback 全部保留。
- source closure、wrong-SHA 和 positive dry-run 通过；dry-run 在 live VCS/license boundary 前返回 86，所有正式身份与工具副作用为零。

下一步仅允许 receipt-blind M819 source hammer。PASS100 后仍需另建 exact true release 与 final hammer；不得由本 handoff 直接运行 VCS。
