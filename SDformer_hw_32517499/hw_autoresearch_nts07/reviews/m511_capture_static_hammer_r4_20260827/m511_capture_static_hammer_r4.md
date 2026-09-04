# M511 decoder S10 capture 独立静态打铁 r4

最终结论：`STATIC_GO__FINAL_ONE_SHOT_REMOTE_S10_CAPTURE_ONLY`，99/100，P0=0、P1=1。本轮严格静态，没有运行生产 capture、checkpoint/model、CUDA、VCS、DC 或 DSE。

审定身份：producer `e16a454d...`，contract `e556743d...`；21 个 inputs 全部 SHA 匹配，`docs/359` 仍为 `dedde7ce...`。

r4 新增的完整集合门正确：producer 枚举 runtime model 中所有 `isinstance(torch.nn.ConvTranspose2d)` 的命名模块，并要求名称和顺序严格等于 contract 的 decoder 0..3 四项。该 gate 位于 hook 注册和 capture 之前；额外、缺失或重排的 ConvTranspose 都会在 canonical 不存在时 fail closed。r3 的 P1-02 已关闭。

r3 全部关键结论回归不变：MS `sn -> deconv` 拓扑、K3/S2/P1/output-padding1、四层通道/shape/weight identity、checkpoint exact load、no-running BN、first10 样本与 hook order、精确二值 little-bit pack 均闭合。S10 仍为 40 records、696,240,000 bit、87,030,000 B。原始 sequence/event/mask/flow、producer/contract/21 inputs 均在发布前复哈希；hooks 在 manifest 之前清空；seal 要求 actual/listed 集合完全一致；quarantine 在 publish 前生成，postpublication exception 第一项恢复操作就是 canonical 原子 rename。

唯一 P1 是有意保留的 r1 文件名/v1 schema。接受条件是未来 exact runner 必须绑定 producer `e16a454d...`、contract `e556743d...` 和本 r4 outer seal，并明确标记 r1/r2/r3 review superseded；不复制合同，避免产生第二份身份源。

授权边界不变：只准在当前 GPU 训练结束后做一次远端 S10 capture；结果先过独立 payload verifier，之后只准 exact envelope repair 与 A0/A1、PGPR/TDR 离线 fast-kill。仍不授权 RTL 或性能声明，M512 已杀的 phase-balanced EPD scheduler 不得恢复。
