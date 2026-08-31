# M822/C2 R19 exact-collision accounting source handoff

M822 是 M819 唯一 P1 的 additive 修复。它不改 M803 RTL/SVA/TB/filelists、五档 exact 周期门或 M818 旧文件，当前仍是 source-only package，不授权 VCS、simv、license 查询、formal attempt/result 或任何 EDA。

## 修复边界

- shell 已锁存 rename-success 时始终记 `attempt_consumed=true`。
- shell 尚未锁存时，canonical exact identity 只有在 source stage 已移走且 phase 位于 attempt publication/postcheck 边界时才能证明本次消费。
- pre-existing exact canonical 的 no-replace collision 保持 destination 与 source stage 双封不变，并记 `attempt_consumed=false`。
- rename 后 canonical exact 或被注入损坏、且 stage 已移走时，分别用 exact identity 或 conservative moved evidence 记 `attempt_consumed=true`。

## 动态验证

Python 3.6.8 下 12/12 unittest 通过。一个 CLI 测试生成并校验四份双封 non-paper receipt，结果严格为 false/false/true/true：发布前无 canonical、exact 预存在 collision、rename 后 exact、rename 后损坏。碰撞两侧均 no-clobber。

strict duplicate/nonfinite JSON、扁平 attempt、result no-replace、failure quarantine collision fallback、函数闭包与 future launch-chain binding 未弱化。wrong-SHA 和 positive source dry-run 通过，后者在 live VCS/license 边界返回 86，所有正式身份与工具副作用为零。

下一步只允许 receipt-blind M823 source hammer。即使 PASS100，仍需另建 true release 和 final launch hammer；本 handoff 不授权直接运行 VCS。
