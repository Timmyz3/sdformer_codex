# M814 / M813 C2 R17 fresh source hammer

结论：**FAIL_SOURCE_GATE，96/100；P0/P1/P2 = 0/1/0。** M813 已经关闭 M811 的 result 覆盖、扁平 attempt、strict JSON 和 failure-quarantine 主问题，冻结的 M803 RTL/SVA/TB 与五档 exact 周期门也没有变化；但是正式 attempt 的 `renameat2` 成功与 shell 的 `attempt_consumed=1` 之间仍存在一个可失败的发布后校验窗口，因此不能授权 true release 或 VCS。

## 通过项

- 请求、作者交接、M811 评审、contract、candidate、runner 的内外双封全部独立重算通过；contract 的 35 项 live SHA 与三份 filelist 全部闭合。`docs/359` 仍为 `dedde7ce...`。
- M803 adapter/K8 top/matched shell/SVA/attack TB/full TB 分别仍为 `cd264021...` / `2588f890...` / `3328e52d...` / `6d7803e5...` / `b89948e7...` / `6d1c1612...`。五档 exact 门仍为 `51/53, 131/133, 486/499, 1231/1246, 14/14`，numeric/tuple/weight/stall/full8/out-of-order 门未减弱。
- Python 3.6.8 下重跑 `bash -n`、函数闭包、undefined-function 负例、35 SHA closure、6/6 atomic unittest 和 wrong-SHA/positive source dry-run 全过。dry-run 在 live VCS/license 边界返回 86，VCS/license/simv/formal attempt/result/failure quarantine 副作用均为 0。
- 独立追加攻击通过：重复 `status`、嵌套 `authorization.launch_now`、嵌套 identity SHA、NaN/Infinity 全拒；result 文件碰撞、attempt symlink 碰撞均 no-replace 且不污染；污染 stage 在 rename 前拒绝；clean attempt 发布后仍是平坦双封三件套；PRE/POST failure 与 primary collision fallback 均生成双封 non-paper receipt；合成 future chain 的 exact binding 和 caller final-outer pin 通过。

## 阻断项 M814-P1-01

Runner 第 228--230 行调用 guard 的 `publish-no-replace`。Guard 第 220 行先完成 `renameat2(RENAME_NOREPLACE)`，随后第 221--226 行仍执行 source 消失、destination 双封与 identity 校验；只有整个子进程成功返回后，runner 第 231 行才设置 `attempt_consumed=1`。failure receipt 第 125--132 行完全依赖这个 shell 标志。

独立临时目录注入让发布后的第二次 verifier 失败，结果可重复为：source stage 已消失、canonical attempt 已存在且仍是 exact 平坦双封三件套，但 guard 命令非零返回，shell 标志仍为 0。因此 EXIT trap 虽会发布双封 failure receipt，却会把实际已经消费的 attempt 错记成 `attempt_consumed=false`。进程组信号落在同一窗口也有相同结果。这违反“任何 post-consumption failure 都有准确封存证据”的修复目标，和 M809 曾出现的 accounting gap 同类。

## 裁决与最小修复

本评审不授权 true release、final hammer、VCS/simv/license 查询或任何 EDA。允许一个新 identity 只修 attempt publication/accounting，不改 M803 RTL/SVA/TB/filelists 或 exact 周期门。最小修复应让 failure cleanup 根据 canonical attempt 的实际占用/身份记录 post-rename 状态，而不是只依赖 guard 子进程成功返回后的内存标志；并新增“rename 已成功、postcheck 注入失败”的 runner 级负例，要求 failure receipt 明确记录 canonical attempt 已消费/已占用。修复后需要新的 receipt-blind source hammer。
