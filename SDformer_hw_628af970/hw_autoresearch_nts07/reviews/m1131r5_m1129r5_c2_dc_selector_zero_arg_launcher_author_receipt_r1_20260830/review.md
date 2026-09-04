# M1131r5 M1129r5 zero-argument launcher 作者回执

结论：**GO 仅授权不同作者 M1132r5 final launch hammer；STOP 真实 launcher、attempt、VCS 和 DC。**

Launcher 是 additive 新文件，绑定 M1129r5 engine/contract/author receipt、M1121 和 M1130r5 的 exact identities。它只接受零参数且要求精确 `env -i` root environment；子进程环境从常量构建，不转发 caller environment。

执行前必须通过 fresh attempt/result/work/failure/lock、EDA process collision、MemAvailable 和 CommitLimit-Committed_AS 门。最多调用一次 pinned engine child，原样返回 child return code，无 loop/自动 retry，并清理 private HOME。

Launch receipt 绑定 launcher SHA 及所有已存在 authority，但不含未来 M1132r5 outer。Engine 在最终执行时自一致发现 M1132r5，因此链无 hash 环。

作者测试仅用 mock resource/collision 与单 child `subprocess.run`，484 checks、9 attacks；没有调用真实 launcher/engine，没有创建 r5 namespace。
