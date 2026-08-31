# M176 indexed-nonzero96 exact-payload DSE 独立打铁评审 r1

结论：**86/100，`PASS_EXACT_ANALYTIC_FRONTEND_DSE_NATIVE_INDEXED_PRODUCER_REQUIRED`，P0/P1/P2 = 3/6/3。**

M176 的冻结 payload 计数和 analytic frontend recurrence 全部可独立复现，且
K1/K4 公平使用同一 indexed transport、每 token 显式计入一个 EOT、没有预测未来
非零 beat、没有跨 beat 拼 group。它是可信的机会边界，但其零 beat 跳过只有在
**ATLIF-native indexed tap 或预建稀疏索引**下才形成延迟优势；如果先读 raw bitmap
再做后处理压缩，优势会消失。

评审期间 M176 README/contract 已从含糊的 absolute base-row 修正为 5-bit absolute
beat index，并规定硬件以 `index×12` 恢复 base-row、按 stage 的 4/8/16/32 beat
extent 检查范围。当前结果目录的 manifest 文件 SHA 是
`57f8e77b70a58f10cf7045011aa2446e2faa4cd61fd73bcbb950f49a55706927`，已完整
`sha256sum -c` 通过；它取代任务开始时的 `21d704ff...`。

## 独立 exact-payload 复算

新脚本没有导入 M172/M173/M176 analyzer，而是用
`numpy.unpackbits(bitorder="little")` 直接解 120 个 FC2 payload，再按
`channel_index mod 8` 重建 bank population。全部 **437,760,000 bytes** 的逐文件
SHA、大小和 popcount 均重查。

| 项 | 独立结果 |
|---|---:|
| samples / FC2 records | 10 / 120 |
| tokens | 5,580,000 |
| input elements / events | 3,502,080,000 / 143,894,510 |
| raw96 / nonzero96 / zero96 beats | 36,480,000 / 18,869,376 / 17,610,624 |
| explicit EOT / indexed descriptors | 5,580,000 / 24,449,376 |
| recurrence oracle cases / mismatch | 17,416 / 0 |
| published aggregate + four-stage integer mismatch | 0 |

17,416 个 recurrence oracle 同时覆盖 variable-length descriptor 和有序 release-time；
vector recurrence 与逐 cycle scalar FSM 逐项比较，0 mismatch。

## 可复现的 M176 frontend 数字

| 候选 | K1 latency sum | K4 latency sum | K1/K4 |
|---|---:|---:|---:|
| raw96 | 437,234,151 | 157,504,597 | 2.776009x |
| indexed96，native/preindexed source | 424,060,394 | 144,146,504 | **2.941871x** |
| raw128 | 432,951,702 | 146,423,753 | 2.956841x |

因此 production 中的以下整数和比例正确：

- raw96/indexed descriptor = `36,480,000 / 24,449,376 = 1.492062620x`；
- raw96/indexed96 K4 = `157,504,597 / 144,146,504 = 1.092670253x`；
- raw128/indexed96 K4 = `146,423,753 / 144,146,504 = 1.015798156x`；
- indexed96 四 stage K1/K4 = `2.215418343 / 2.947335343 / 3.112231098 /
  3.159226338x`。

这些只能命名为冻结 payload 上、always-ready、one-descriptor-per-cycle source 的
analytic frontend latency sums。`1.092670253x` 和 `1.015798156x` 都隐含 native 或
preindexed producer；不是物理、完整 FC2 或系统倍率。

## 核心打铁发现：posthoc scanner 会把优势翻转

本评审增加了一个 production 没有的 release-aware 下界：producer 每拍检查一个
raw96 beat；非零 descriptor 在发现当拍即可使用；固定 extent 扫完后才产生 EOT；不收
任何 compactor、FIFO、SRAM 或 wire 额外代价。这个模型对 posthoc scanner 已经很乐观。

| 项 | native/preindexed M176 | optimistic posthoc scan |
|---|---:|---:|
| indexed96 K1 | 424,060,394 | 439,475,145 |
| indexed96 K4 | 144,146,504 | **159,902,252** |
| K1/K4 | 2.941871x | 2.748399x |

posthoc K4 相对 raw96 变为
`157,504,597 / 159,902,252 = 0.985005496x`，即 **慢 1.5223%**；相对 raw128
则慢 **9.2051%**。它还比 M176 理想 indexed K4 多 10.9304% latency。原因尤其清楚：
1,863,944 个 all-zero token 在 analytic indexed stream 中可以立即送 EOT，但 posthoc
producer 必须先检查完整 4/8/16/32 beat extent 才知道它们为零。

所以硬件路线必须是：在 ATLIF 产生/写入 activation 时顺手形成 beat index，或把稀疏
索引作为已存在的数据结构；不能把“读取 raw bitmap 后再压缩”包装成 M176 加速来源。
即使 producer scan 可以靠 replay 和缓冲部分隐藏，raw memory read 与 compactor energy
仍未计入。

## index、EOT 与 grouping 公平性

从 raw beat position 推导的真实非零 index 全部严格递增、范围合法；真实 token 中有
815,084 个 leading gap、735,443 个 internal gap、798,344 个 trailing gap，最大 internal
gap 为 30 beats。注意这些 index 是从 raw payload **推导**的，当前尚没有 materialized
indexed producer trace，因此只能证明目标序列合法，不能证明 producer 实现正确。

修正后的 EOT 语义可实现：前一 token 未 done-accept 时 EOT 非法；同拍 done-accept+EOT
代表合法的新 all-zero token。K1/K4 均使用相同 compact stream，每个非零 96-bit beat
独立形成 modulo-8 bank-unique groups，事件不跨 beat 合并，group/event 数严格守恒。

## descriptor 数量不是物理带宽

统一覆盖最多 32 beats 至少需要 5-bit index，再加一个 EOT flag。若只作说明性估算，
`96+5+1=102` bit descriptor 会把 raw96 bitmap bits 从 3,502,080,000 降到
2,493,836,352，只有 `1.404294x`，不是 descriptor-count 的 `1.492063x`；若物理 SRAM
按 128 bit 对齐，bit reduction 只剩 10.6382%。

这两项都不是已准入数字，因为 M176 尚未规定 descriptor word、sideband、tag、SRAM
布局或 macro。README 中“bitmap payload width 比 raw128 低 25%”只能理解为 96 对 128
的 payload 字段，不能扩展成总 wire/storage/energy 降低 25%。

## P0/P1/P2

P0：

1. 落地 ATLIF-native tap 或 prebuilt sparse-index memory，并用 120 payload materialize
   descriptor stream；禁止用 posthoc raw scan 的免费跳零口径。
2. 把 producer、finite descriptor FIFO 和修正后的 5-bit beat-index frontend 组成一个
   exact-payload VCS/DC shell，覆盖 producer stall、index extent、EOT/backpressure、
   same-cycle re-arm 和 descriptor/event conservation。
3. 再接四 bank weight response、M169 arithmetic、accumulator context 和 commit，之后才可
   讨论 complete FC2 或 physical speedup。

P1：封一份 terminology overlay，因为 immutable analyzer/result 仍写 absolute base-row，
而最新 README/contract 已改成 beat index；规定并综合 102/对齐后的 descriptor word；做
raw96/raw128/indexed96 matched physical A/B；PAFT 后重放；扩 sequence/tail distribution；
计入 producer/memory/compactor energy。

P2：把 release-aware producer sensitivity 收进 production DSE；分别报告 descriptor
count、payload bits、sideband bits、aligned storage bits 与 cycles；保存至少一份实际
indexed trace，直接验证 index/EOT 顺序。

安全口径是：

> M176 证明冻结 H67 FC2 payload 上存在 native/preindexed sparse-beat frontend 机会，
> indexed96 K1/K4 analytic latency-sum ratio 为 2.941871x；它没有证明 posthoc scanner、
> 完整 FC2、物理或系统加速。

机器可读裁决见 `m176_independent_hammer_review_r1.json`，独立全量复算见
`independent_recompute_result.json`。本评审未修改 M176 analyzer/result 或 `docs/359`。
