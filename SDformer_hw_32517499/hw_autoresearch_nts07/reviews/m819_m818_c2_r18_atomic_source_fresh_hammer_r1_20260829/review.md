# M819 / M818 C2 R18 fresh source hammer

结论：**FAIL_SOURCE_GATE，96/100；P0/P1/P2 = 0/1/0。** M818 已经修复 M814 指出的 rename 后、shell latch 前 accounting gap；普通发布前失败、普通 collision、rename 后 exact/damaged canonical 都能生成语义正确的双封 receipt。但 receipt-blind 攻击发现一个对称边界：若 no-replace 的碰撞目标恰好已是同一份 exact canonical identity，当前 source stage 仍原封不动，guard 却把“看见 exact canonical”误当成本次 move 已完成并记 `attempt_consumed=true`。因此不能授权 true release 或 VCS。

## 通过项

- 请求、作者交接、M814、contract、candidate、runner 的双封均独立重算；contract 的 35 个 source SHA 全部 live replay 通过，三份 filelist 无重复、缺失或 symlink。`docs/359` 保持 `dedde7ce...`。
- M803 adapter/K8 top/matched shell/SVA/attack TB/full TB SHA 未变；五档 exact gate 仍为 K8 `51,131,486,1231,14` 与 K1x8 `53,133,499,1246,14`。numeric/tuple/weight/stall/full8/out-of-order 门未削弱。
- Python 3.6.8 下 `bash -n`、函数闭包、undefined-function 负例、10/10 unittest、wrong-SHA 与 positive source dry-run 全过。dry-run 在 live VCS/license boundary 返回 86，formal identity、quarantine 和工具副作用均为 0。
- strict JSON 对顶层重复 `status`、嵌套 `authorization.launch_now`、嵌套 identity SHA、NaN、Infinity、负 Infinity 全部拒绝。
- stage verify → rename-only → shell latch → canonical postverify 排序明确；扁平 attempt 三件套、result `renameat2(RENAME_NOREPLACE)`、失败 quarantine primary collision fallback、future release/final-hammer/caller outer-seal binding 均保留。

## 阻断项 M819-P1-01

独立临时目录中先建立一个与 expected identity 完全一致的 canonical attempt，再用另一份同 identity 的双封 source stage 执行 `publish_attempt_noreplace`。`renameat2(RENAME_NOREPLACE)` 正确拒绝碰撞，canonical 未改、stage 仍存在且双封不变，shell latch 为 false，说明本次 move 没有发生。

然而 `attempt_publication_state()` 在 canonical exact 校验通过后直接以 `CANONICAL_EXACT_IDENTITY` 判 `attempt_consumed=true`，没有同时要求 source stage 已移走。CLI 生成的 failure quarantine 也把错误值永久双封。它违反请求中的明确合同：“canonical 未由本次 move 建立且 stage 仍在”以及“no-replace collision”必须记 false。

## 裁决与最小修复

本评审不授权 true release、final hammer、VCS/simv/license 查询或任何 EDA。允许用新 identity 做 additive 修复：仅当 shell rename-success latch 为 true，或 canonical exact 且 stage 已移走，才把 exact identity 当作本次消费证据；publication/postcheck 阶段 canonical 存在且 stage 已移走时仍保守记 true，以保持 damaged-canonical 修复。新增 exact-identity collision CLI 负例，要求双封 receipt 为 `attempt_consumed=false`。冻结的 M803 RTL/SVA/TB/filelists 与周期门不得改。
