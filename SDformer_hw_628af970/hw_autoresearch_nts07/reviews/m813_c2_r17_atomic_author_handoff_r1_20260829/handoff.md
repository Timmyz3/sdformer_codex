# M813/C2 R17 atomic source-only author handoff

M813 是对 M811 三个 runner 边界问题的全新身份修复，不改 M803 RTL/SVA/TB/filelist，不改五档 exact 周期门，也不改 `docs/359`。当前仅为 source-only package，不授权 VCS、simv、license 查询或任何 EDA。

## 修复内容

- canonical result 使用 Linux `renameat2(RENAME_NOREPLACE)` 从 sibling stage 原子发布；发布前验 seal，发布后再验 seal 和 canonical-root 四件套。post-precheck 目标碰撞必须非零失败，不嵌套 stage，不改目标。
- attempt 先在 sibling stage 生成平坦三件套：`attempt.json + SHA256SUMS + outer seal`；通用 verifier 通过后才 no-replace 发布，发布后用 exact identity 再验。
- intentional source dry-run 之后、launch-chain gate 之前就安装 failure trap。任意 PRE/POST-attempt 失败都生成双封 `FAILED_OR_INCOMPLETE_DO_NOT_CITE_PERFORMANCE` 回执；primary quarantine 碰撞时保留旧目标，用原子 fallback 发布新回执。
- contract/candidate/release/review 全部权限 JSON 都通过 strict parser，重复 key 和 NaN/Infinity 均拒绝。

## 作者源级验证

- Python 3.6.8：atomic adversarial unittest `6/6 PASS`。包含 result 碰撞、attempt 污染/碰撞、三层 duplicate JSON key、PRE/POST-stage failure 和 failure destination collision。
- `bash -n` PASS；函数闭包 PASS；删除 `publish_failure_receipt` 的负例被捕获。
- wrong runner SHA 在 trace 前返回 3；positive source dry-run 在 live VCS/license/formal identity 边界前返回 86。所有 VCS/license/simv/attempt/result/failure-quarantine 计数均为 0。
- M803 五档 hard gates 仍为 `51/53, 131/133, 486/499, 1231/1246, 14/14`；numeric/tuple/weight/stall/full8/out-of-order 门未弱化。

下一步只是 fresh independent M814 source hammer。PASS 也只可授权作者新建一份 true release 与 final-hammer request，不能直接运行 VCS。
