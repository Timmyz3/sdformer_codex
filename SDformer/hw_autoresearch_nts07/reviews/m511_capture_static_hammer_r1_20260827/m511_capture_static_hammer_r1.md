# M511 decoder S10 capture 独立静态打铁 r1

结论：`NO_GO__SUPERSEDED_AFTER_P0_REPAIR`，78/100。该轮只审源码与合同，未运行生产 capture、未加载 checkpoint/model、未接触 CUDA。

正向复算全部成立：H67 的 MS decoder 是 `sn -> ConvTranspose2d`；四个目标按 decoder 0..3 顺序出现，参数均为 K3/S2/P1/output-padding1/group1/bias-null；Cin/Cout、输入输出 shape 与 M510 一致；ConvTranspose 权重布局为 `[Cin,Cout,3,3]`。S10 共 40 个 call、696,240,000 bit、87,030,000 B（82.9983 MiB），所有 call 都整字节对齐。合同也正确禁止 cycles/speedup/RTL/energy/PPA/headline。

阻断项：

- `M511-R1-P0-01`：canonical 原子发布后若最终 seal 复核或打印失败，旧实现只在父目录写旁路 failure，canonical 内的 PASS manifest 与 `RUN_COMPLETE` 仍存在，消费者可能误收。
- `M511-R1-P0-02`：contract 没有 start/end 身份比较；运行使用旧内存对象，却可能在 manifest 中记录捕获期间被替换后的合同 SHA。

P1：seal verifier 不拒绝未列成员；valid 样本只绑定文件名、未绑定 sequence CSV 和 event/mask/flow 字节；运行时未显式断言全部 ConvTranspose 集合恰为四个目标。

本 r1 记录只保留为发现与修复历史。原 producer/contract 身份分别为 `201a4013...` / `b3eb127d...`，已经被修复版取代，不得用于 launch GO。
