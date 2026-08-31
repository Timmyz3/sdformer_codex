# M511 decoder S10 capture 独立静态打铁 r2

结论：`STATIC_NO_GO__POSTPUBLISH_FINALLY_CAN_LEAVE_CANONICAL_PASS`，89/100，P0=1、P1=4。没有运行生产 capture、加载 checkpoint/model 或接触 CUDA。

r1 的主要修复有效：contract start/end SHA 已闭合；seal 现在要求 actual member 与 sealed member 完全一致；sequence CSV 以及每个 S10 样本的 event/mask/flow 文件在 capture 前记录 path/bytes/SHA，并在发布前重哈希；发布后 try 内异常会把 canonical 原子移入 quarantine。样本源路径与 `DSECDatasetLite.__getitem__` 一致。

拓扑和容量复算也通过：MS 路径是 `sn -> deconv`；四层名称、顺序、Cin/Cout、K3/S2/P1/OP1、weight shape 与 M510 一致。S10 是 40 records、696,240,000 bit、87,030,000 B，所有 call 整字节对齐。合同没有越界到 cycle、speedup、RTL、energy、PPA 或 DATE headline。

仍有一个阻断项：

- `M511-R2-P0-01`：canonical 发布成功后才进入 `finally: handle.remove()`。Python 的 finally 异常不会回到同一 try 的 sibling except；因此 `remove()` 一旦抛错，进程失败但 canonical PASS 仍保留。应在 seal/rename 之前、仍位于受保护的 try 内移除 hooks 并清空 `handles`，发布之后不再做任何可能抛错的清理。

P1：quarantine 仅用 PID，历史目录与 PID 复用可在 rename 前再次抛错；修复版仍复用 r1 文件名/v1 schema，必须用 SHA pair 与独立 r1/r2 review 严格区分；decision_policy 引用 M512 kill/PGPR/TDR 但未 pin M512 seal；运行时最好断言完整 ConvTranspose 集合恰为四个目标。

当前 producer/contract 身份是 `73e26e73...` / `69f948f0...`。本 review 不准许 GPU launch。修 P0 后需重新锁 SHA 并做一轮短静态复审；通过后，capture 仍只授权独立 payload verifier、exact envelope repair、PGPR/TDR 离线 fast-kill，不授权 RTL，且绝不复活 M512 已杀的 phase-balanced EPD scheduler。
