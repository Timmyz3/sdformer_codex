# M1111Dr2 decoder production runner publish-gate repair author receipt

结论：**M1112D 的 P0/P1 在 additive r2 source 中关闭；只允许另一作者 final hammer，不允许 launch 或 production。**

r2 冻结并保留 r1 的单次 attempt、atomic seal、`renameat2(RENAME_NOREPLACE)` 和 post-attempt quarantine 协议，但把发布边界改成独立的严格 validator。发布前必须同时满足：顶层只有 result JSON、120 行 schedule JSONL 和完成 token 三个文件；seal bundle 只有 manifest/outer；JSON 无 duplicate key、NaN 或 extra key；call ordinal、transaction interval、cycle interval、traffic、digest 和 resource 逐项守恒；D1 使用精确 `1065353139` 且不 fold；final-checkpoint rebind 为 true；M700 字段不存在；任何 ratio/speedup/performance/citable/headline/system-admission 字段只能缺席或为精确 `false/null`。

作者阶段只在 `/tmp` 构造了一个 120 行、720 synthetic transaction 的 schema 候选。合法候选通过，13 个 claim/file/JSON/numeric 变异全部拒绝。M1112D P1 的 same-byte manifest symlink、outer symlink，以及额外 flat file 也全部拒绝。

没有调用 runner `main`、`execute_production` 或 `publish_result`，没有打开 canonical payload，没有创建 attempt/result/work/quarantine。下一步必须由不同作者绑定本 runner、contract 和本 sealed receipt，执行 final runner hammer。

`docs/359_DATE终局冻结_20260813.md` 未修改，SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
