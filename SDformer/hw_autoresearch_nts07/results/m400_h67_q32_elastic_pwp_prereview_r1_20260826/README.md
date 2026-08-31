# M400-pre：H67 q32/O4 exact elastic-width PWP 预审

结论：**GO 全 phase/center/tile 正式重放；当前探索值只通过合理性审查，不准入
正式速度。** 评分 88/100，P0/P1/P2 = 0/3/5。

硬件合同可实现且容量安全。每个 96-lane signed12 PWP block 固定存为 96 B
low8 加 48 B high4、补零到 64 B sidecar，共 160 B。只有全部 96 lane 均在
`[-128,127]` 时才用一个 SHARED96 issue beat；否则 low/high 两 beat，并逐 lane
拼回 signed12。每 center/O4 固定 stride 是 640 B，地址和 run 长度均 32 B
对齐。

q32 的 64 B patterns 与 `32 centers × 8 output blocks = 256 bit` narrow bitmap
共打包为 96 B config：一个 cmd32、三个 data beat。slot0 布局为 config@0、
weight@96、PWP@6240，最坏容量为
`96+6144+32*640=26,720 B`，余 6,048 B；slot1 weight@32768、
PWP@38912。两槽都合法。

我用 exact-SHA M41 INT8 权重和 M338 前 32 个 nested center 独立重建了全部
442,368 个静态 center/output-block PWP：

- 数值范围 `[-1089,1059]`，maxabs=1089，signed12 足够；
- 静态 narrow block 为 112,167，占 25.356%；
- 所有声明 narrow block 的 low8 sign-extension round-trip 为 0 mismatch。

因此探索的 maxabs 和方向合理。M397 冻结的 PWP rows 是 16,971,357，乘八恰为
135,770,856 block descriptors；24,586,812/135,770,856 也确为 18.1090499%。
M397 同时冻结 used-center occurrences=549,754。但 runtime-weighted narrow count
仍不能从这些 aggregate 推出，必须按每个 phase 的 selected center 和八个 block
重放，所以 24,586,812 暂不准入。

周期上不能直接从 M397 总数减去 narrow descriptors。固定 640 B stride 比原
576 B 每 center/tile 多 64 B；DMA 只能按每 phase used center 计一次，绝不能按
descriptor 频次相乘。正式 q32 非空 phase 必须计算：

`config + matcher + seal + dma0 + max(replay0,dma1) + replay1 + tail2`

其中两 tile replay 分别用自己的 narrow block descriptor 数。缩短 replay0 后，
原先隐藏的 tile1 DMA 可能暴露。

1.15 门对应 candidate cycles 不超过 645,346,422；M397 q32 尚需减少
23,665,914 cycles。仅按未准入探索值，raw issue task 最多减少 24,586,812；但
必付一拍 config/phase 共 17,280 cycles，以及 tile0 每 used center 两拍的暴露
增量 1,099,508 cycles。即使假定 tile1 增量全隐藏，elastic-only 的乐观总减少
上界也只有 23,470,024，仍差 195,890。这个结论只是条件规划界，不是正式
candidate cycle 或 speedup。

因此正式执行必须同时报四行：

- M397 anchor：旧 fixed12/576 B/64 B config，精确复现 669,012,336；
- elastic-only：新 640 B/96 B config，原 q32 matcher；
- early-only：旧存储/DMA，加 M399 q32 prefix1 distance-zero early stop；
- combined：elastic 加 early-hit。

若 tile1 增量全隐藏且 codec 无 bubble，combined 至少需要 195,890 个
`F16_eligible` early-hit cycles；实际需求还要加新暴露的 tile1 DMA 和任何实名
控制开销。

当前 M384 硬编码 576 B stride 与旧地址，不能承接 M400。M79/M82 是 256-bit
variable-width stream，M133 是 512-bit stream，都只能作设计先例。准入前必须
做新的 768-bit SHARED96 one/two-beat codec 与 controller VCS/SVA、3 ns DC、
Formality 和 PrimeTime；物理 SRAM、SAIF/PTPX、能耗和 paper PPA 仍在边界外。

`docs/359` 与既有证据均未修改，本预审没有产生新 speedup 或 headline。
