# M511 decoder S10 capture 独立静态打铁 r3

结论：`STATIC_GO__ONE_SHOT_REMOTE_S10_CAPTURE_ONLY`，97/100，P0=0、P1=2。本轮没有运行生产 capture、checkpoint/model、CUDA、VCS、DC 或 DSE。

身份闭合：producer `7a4f6f36...`，contract `2b0d7bc9...`，21 个 contract inputs 全部逐文件 SHA 匹配；M512 的 verdict/manifest/outer-seal 也已纳入 pin，EPD phase-balanced scheduler 的 kill 不会被 capture 偷偷复活。`docs/359` 仍为 `dedde7ce...`。

拓扑与容量复算：MS 路径确为 `sn -> ConvTranspose2d`；decoder 0..3 的名称、顺序、Cin/Cout、K3/S2/P1/output-padding1/dilation1/group1/bias-null、weight shape 与 M510 全部一致。S10 是 40 records、696,240,000 bit、87,030,000 B（82.9983 MiB），所有 call 整字节对齐；输入只接受精确 `{0,1}`，不容忍阈值化或非零 coercion。

成功事务顺序已经安全：最终 CUDA fence 后，四个 hooks 在 manifest/seal/publish 之前用 `while-pop-remove` 清空；任一 remove 失败都会在 canonical 不存在时进入失败路径。随后 producer、contract、21 个 pinned inputs、sequence CSV 和每个 S10 样本的 event/mask/flow 都复哈希；seal 要求 actual member 集合与列出的集合完全相等；同父目录原子发布后再次验证 canonical。

post-publication 异常路径也闭合：唯一 quarantine target 在 publish 前生成并确认不存在；发布后的 except 第一条恢复操作就是 `os.replace(output, quarantine)`。在正常 POSIX 原子 rename 语义下，没有发现“caught nonzero exit 但 canonical PASS 仍存在”的路径；底层文件系统 rename 自身失败属于外部不可软件消除故障，不作为静态 P0。

两个非阻断 P1：修复版仍沿用 r1 文件名/v1 schema，因此 exact runner 必须锁定本 r3 的 producer/contract/review SHA tuple，并把 r1/r2 review 标成 superseded；独立 payload verifier 最好再断言运行时完整 ConvTranspose2d 名称集合恰为这四项。

授权边界很窄：只有在现有 GPU 训练结束、canonical output 不存在、exact-SHA runner 绑定本 r3 outer seal 时，允许一次远端 S10 capture。结果必须先过独立 payload verifier，之后才可做 exact envelope repair 以及 A0/A1、PGPR/TDR 离线 cycle fast-kill。仍不授权 RTL、cycle/speedup、energy/PPA 或 DATE headline；M512 已杀的 EPD scheduler 永久不因本 capture 恢复。
