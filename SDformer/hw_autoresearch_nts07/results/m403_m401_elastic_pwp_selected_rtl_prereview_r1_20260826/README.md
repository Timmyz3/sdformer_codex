# M403-pre：M401 selected-RTL 最小投稿边界预审

结论：**M402 已独立接受 M401 combined，可进入 selected RTL；两项新模块足够，
但两个裸模块不够。** 最小边界是：

1. 96-lane exact elastic PWP issue adapter；
2. q32 serial16 distance-zero prefix controller；
3. 非贡献性的薄 config/integration shell；
4. 把 M384 的旧 576 B 布局改成 q32/640 B，并重跑其全部 VCS/Synopsys。

本预审 89/100，P0/P1/P2=`0/4/6`，不产生新速度。

## 上游候选与脆弱门槛

M402 双封已复验，独立重建了 M401 的 17,280 phase 和全部账本：combined 为
641,790,704 cycles，相对同一 742,148,386 baseline 是 1.1563713550x。1.15 的
整数上限为 645,346,422，因此只余 3,555,718 cycles。折到 54,611,136 个 replay
descriptor，仅容许平均 0.0651098 个额外 cycle；0.25 的敏感性已经降到
655,443,488 cycles/1.1322843x。

这仍是独立接受的 cycle-simulator 结果，不是 RTL 测得速度。

## 模块 A：exact elastic PWP issue

每个 block 固定存 low8 96 B 和 high4 payload 48 B；high 物理侧车另有 16 B
零 padding，共 64 B，block 合计 160 B。单一最大 96 B 的输入口承接 low96 或
high64，不新增第二个 PWP 口；high 逻辑口多出的 bytes64..95 由 adapter 零化，
不算存储或 DMA。

- narrow：96 lane 全在 `[-128,127]`，每 lane sign-extend int8，一项 contribution；
- wide low：low8 按 unsigned zero-extend；
- wide high：每 4-bit nibble sign-extend 后左移 8；
- 两项的 signed13 和必须逐 lane 等于原 signed12，之间不能 saturation/rounding。

wide 必须原子化：先缓存 low，再接受并核验同 tag/tile/center/block 的 high 和
16 B 零 padding；只有核验通过后才允许 low/high contribution 对外 accept。否则
sticky error 且两项都不 commit，禁止先累加 low 后才发现 high 错误。

VCS 应重放全部 442,368 个静态 block/42,467,328 lane，并验证 narrow 一项、wide
两项、done 位置、极值、nibble 顺序、宽度切换、随机 backpressure 和协议攻击。
bitmap 语义真值来自离线 exhaustive codec；仅看 low8 的硬件无法发现“格式合法但
语义错误”的 narrow bit，必须明确这是受保护 config 的信任边界。

## 模块 B：q32 serial16 zero-stop

一个 96 B config command 分三拍 32 B：前两拍装 32 个 16-bit center，第三拍装
`32×8=256 bit` narrow bitmap 并 commit。768 bit config 只能有一个 owner，向
matcher 和 adapter 导出；若复制必须计面积并做一致性检查。

推荐逐行方案：每行先跑 centers0..15。只有 `popcount>=2 && pass0 best_distance>0`
才在下一 task cycle 跑 centers16..31；完整 pass0 的 distance0 才能停，tie 取最低
global center ID。单行 register scratch 至少实名保存 original16、row_id12、pop5、
last1、pass0 best id4/distance5，共 43 bit，另计 valid/state/tag/output skid。
它不做 source reread，也不写读 descriptor scratch，因此输出保持原序，任务数与
M401 的 `3000 + unresolved + 2` 同构。

这里有一个硬门：pass0 必须在正常 registered task boundary 给出决定，使 pass1
或下一行 pass0 在下一拍发射。M321 的 16-way tree 只能借拓扑；它 tie 按 center
值、语义是 tau，而且是两级结果 pipeline，不能继承其 VCS/DC admission。若新设计
在 3 ns 必须更深 pipeline，就要实名增加 multi-context/ROB，或把 bubble 加回全
phase recurrence，不能偷用 641,790,704。

另一种复用两片 3000x48 descriptor bank 的 batch scratch 在位宽上可行：
`final_addr12 + row_id12 + original16 + pass0_id4 + (positive_distance-1)4=48`。
但它增加 unresolved W/R、final random patch W 和能量，破坏“phase ping-pong”口径；
单口 scratch 在 pass0 写入时也不能凭空预取。除非证明 1R1W/双口或实名加入 L8，
否则不能采用。本轮冻结逐行 register 方案。

## M384 必改

当前 exact-SHA M384 RTL/SVA/TB 硬编码：tile0 PWP base=6208、stride/run=576，
旧 bounds=24640/57344。selected q32 必须改为：

- slot0 config@0、weight@96、PWP@6240、end<=26720；
- slot1 weight@32768、PWP@38912、end<=59392；
- center stride 和 run multiplier 都为 640。

32 centers、48-bit descriptor、D8/L8、II1、两次 replay 和 no cross-phase overlap
保持不变。所有 legal start/length、两 tile、bitmap 边界、backpressure 和既有攻击
都要重跑；旧 M384 VCS/DC/FM/PT 收据绑定旧 SHA，不能沿用。

## VCS 与 Synopsys 边界

不需要跑完整 641M VCS。M401/M402 已负责全 phase cycle replay；VCS 负责模块的
exact arithmetic、任务/贡献守恒、registered dependency、ready-high 无内部 bubble、
随机 stall 和 fail-closed。VCS 不能证明外部 SRAM 永远 ready、tile1 DMA 物理隐藏、
平均 blocking<0.0651，也不能把 cycle-sim 叫 RTL speedup。

顺序冻结为：VCS → 3.000 ns DC（A/B/M384/integrated 分层报告）→ exact-SHA
Formality → integrated PrimeTime。DC/PT setup 和 hold 都必须 `WNS>=0`；候选与
baseline 用同一 3 ns。1.15637 的 cycle margin 只容许 0.551% 频率损失。

预宏 logic-only integrated guardrail 冻结为 30,000 um²，必须包含 A 的原子 buffer、
B 的 768-bit config/scratch/skid、修订 M384 和 shell。它由旧 M384、1.5×旧 M133、
2×旧 M321 加约 8.09% 集成余量得到，只是 selected-slice 工程门，不是芯片面积或
paper PPA。

即使全部通过，也只准入 standalone selected RTL 与 logic-only area/timing；物理
SRAM、SAIF/PTPX、system speedup、paper PPA 和 DATE headline 仍为 false。

`docs/359` 未修改。
