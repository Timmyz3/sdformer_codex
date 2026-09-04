# M837 / C2 R22 identity-compatibility source handoff

R21 release authoring 前的 synthetic chain 揭示：M833 runner 虽已修复 Unicode，但仍调用冻结 M826 `validate-launch-chain`，其 source/release/final status 及 source-review 三键 target 都写死为 M826 身份。双封 M834 R21 PASS100 因此在读取 release 前必然失败。

R22 只增加身份兼容层：原 M826 guard 和 M833 runner 均未修改；新 wrapper 复用全部 strict JSON、seal、renameat2 no-replace、attempt/receipt 逻辑，只精确绑定 M834 R21 的 status 与四键 target、M832 的 spent-release 结论，以及未来 R22 source/release/final status。旧 M826/M833 错 status、三键 target、缺键和额外键均被负测拒绝。

Python 3.6/3.12 均通过原 atomic 12/12、final-auth 8/8、Unicode 5/5、R22 identity 11/11、source closure、synthetic launch-chain，以及 actual runner outer-C wrong-SHA rc3 / positive rc86 零副作用。M803、五档周期、四 receipt、15 键授权和 12+1 局部 C.UTF-8 边界未改。

当前仍是 source-only：未运行 VCS/simv、未查询 license、未创建 attempt/result/quarantine，也未制作 release。下一步只能由 fresh reviewer 按 M838 request 打铁；PASS100 后才允许另一作者制作一次 R22 true release。
