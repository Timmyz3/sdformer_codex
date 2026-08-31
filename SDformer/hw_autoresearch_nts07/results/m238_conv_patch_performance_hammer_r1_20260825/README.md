# M238 Conv/Patch 性能独立打铁评审

**评分 82/100，P0=2、P1=7、P2=4。** M238 的热点重算和路线选择 GO；standalone Conv 硬件倍速、full-network 倍速与 paper PPA 仍 NO-GO。

最重要的路线修正是：**Patch 更贵，但 bottleneck Conv 更值得冲；Conv 不再走 M147+M149，而转成 M152 bank-aware source-K4 + M154/M155 融合。** M152 比 M147 只多 3,196 cycles（0.00426%），却绕开 30,958.704 um2 的 M149 conflict combiner及其未修的 stalled-result fault P1。

## 真实 H67 Conv 热点

冻结 620,302,905-cycle compute envelope 中：

| 范围 | cycles | share |
|---|---:|---:|
| 全部 Conv2d | 279,322,733 | 45.0301% |
| patch embed 全部 8 层 | 199,420,620 | 32.1489% |
| patch 六层 exact-binary Conv3x3 | 172,321,077 | 27.7801% |
| patch 两层 nonbinary head/shortcut | 27,099,543 | 4.3688% |
| bottleneck 四层 Conv3x3 | 79,630,957 | 12.8374% |
| prediction 四层 Conv | 271,156 | 0.0437% |

最贵的单层是 `patch_embed.residual_encoding.resblocks.1.conv1.0`，43,561,877 cycles，占 envelope 7.02%。因此“最贵层在哪”答案是 patch；但“哪条硬件性能线最强”答案不是 patch。

## Patch：热点大，但强基线下没有强创新倍率

60 条真实 bitpack 的公平重算保持 M222/M223 结论：

| 点 | 8×128b 合法 | product lanes | serial cycles | vs strong K1 |
|---|---:|---:|---:|---:|
| K1/D96 row-striped | 是 | 96 | 1,883,708,587 | 1.000× |
| K4/D32 | 是 | 128 | 1,704,834,936 | **1.104921×** |
| K8/D16 M218-like | 是 | 128 | 1,990,634,940 | **0.946285×** |
| K8/D32 | 否，需 16 banks | 256 | 1,050,037,470 | 1.793944× |
| K8/D48 | 否，需 24 banks | 384 | 736,504,980 | 2.557632× |

这说明 source K-group 的 group-count 看起来很漂亮，但 destination slice 和强 K1 weight striping 会吃掉收益。合法最优点还多用了 33.3% add lanes 和 1024-bit issue，1.1049× 必须等面积 DC 后才知道是否还有正收益。

其他 patch trick 也不够：

- 全跳过 7,257,197 个 empty-output commit，理想上限只有 1.003868×，还没有 sparse downstream token 合同。
- 假设 line-buffer scan 完全免费，上限只有 1.038091×；连同 empty commit 都免费也只有 1.042260×。
- 六层 parent residual 的 zero/local source-bit ratio 为 1.309144×，但 dual-parent 对 local 只再减少 0.60%；这是 source work，不是 Conv cycle。
- nonbinary `head.conv.0` 的 motion delta 比 local 多 34.37%，明确更差；`proj.conv_res` 只少 0.326%，而且仍是元素统计，不是 weighted cycles。

所以 patch 应冻结为公平性负结果/消融，不占性能主线 Synopsys 资源。

## Conv：最强 standalone 目标及正确的 baseline 层次

| 比较 | parent cycles | target cycles | ratio | 口径 |
|---|---:|---:|---:|---|
| M152 vs block-K4，双方 PWP1024 | 126,581,635 | 75,032,786 | **1.687018×** | 首选同 source-port 增量口径 |
| M152 vs M143 B4/PWP512 | 135,461,009 | 75,032,786 | **1.805358×** | 同 lineage，但包含 512→1024 port widening |
| M152 vs fixed8 dense service model | 1,114,863,448 | 75,032,786 | **14.858351×** | 传统 dense 次级口径，必须先做 matched simulator baseline |

这种三层表述最接近 Prosperity/Phi 的做法：可以最终报告对 dense baseline 的高倍率，同时必须给对强 sparse/same-port baseline 的增量倍率。当前三个数字都仍是 cycle-model opportunity，不是已实现硬件结果。

M147 mosaic packing本身在相同 PWP512 下是 1.107908×；PWP1024 在 block-K4 上是 1.070147×。两者组合出现 1.805×，有明显 overlap/queue 非线性。PWP1024 虽减少 49.37% beats，却多传 1.263% raw bits并需要双倍峰值带宽，不能写成能耗节省。

关键新判断是 M152 pivot：

- M147：47,037,211 descriptors、75,029,590 cycles，但 75.95% descriptor 有 repeated destination，离开 M149 combine 后反而只有 0.98768×。
- M152：47,040,777 descriptors、75,032,786 cycles；只损失 3,566 descriptors / 3,196 cycles，却用 destination modulo-4 + low/high half 保证一次最多四个独立 bank update。
- 因此 M149 不值得为 0.00426% 再付面积、3,072-bit 输入和协议修复成本。M152/M154 直接接 accumulator 是更干净的论文机制。

## 真正的硬件瓶颈

| 模块 | logic-only 诊断 | 结论 |
|---|---|---|
| M149 combiner | 30,958.704 um2，setup +1.0787 ns | 被选择路径绕过；r1 仍非 integration-ready |
| M154 supplier | 13,282.668 um2，setup/hold +1.6514/+0.0002 ns | 98,304-bit weight SRAM 被排除；3,072-bit result FF 是融合目标 |
| M155 accumulator debug | 76,994.064 um2，14,519 seq、50 levels、setup/hold +0.0164/+0.0001 ns | **当前具体瓶颈**；unsealed、0 macro，不能当 PPA |

M155 的大成本里有 7,296-bit same-address forwarding payload、3,072-bit lazy-valid bitmap和 384-lane overflow reduction。M157 的 source-major row interleave 在 46,971,957 个 heldout context-internal adjacent pair 上观测到 0 hazard；M157 的 22.747×则只是 destination-vector read-work reduction，而且 M152 已经收费 cached phase load，不能再当新 cycle 倍数。

M158 signed tuple/Acc19 数据证明可作为 specialization 依据，但最新独立评审仍有 P0：若 stall/reset/stale replay 造成 accepted transaction 重复，系数可到 2，Acc19 界会失效。因此第一版 fused RTL 必须保留 overflow guard；只有 exact-once coefficient miter 通过后才能做删除树的 matched ablation。

## M238 选择的下一 VCS/DC 里程碑

单独模块即可，不造全网 scheduler：

`M152 descriptor/order → M154 macro-output token → row-interleaved no-forward Acc19 accumulator`

VCS 必须：

1. 绑定 sequence/operator/window/partition/source/destination-half/row 全身份，使用真实 signed INT8 checkpoint vector 和 bounded ordered trace。
2. 对 M155 reference 做逐 accepted-transaction arithmetic miter，证明四 bank update、tail、negate 和 row order。
3. 覆盖 stall/reset/stale/replay/cache-alias，证明 accepted exact-once；任何 younger fault 不得吞掉已接受 macro response 或 accumulator write。
4. 先保留 lazy-valid 和 overflow guard，证明 row-interleave 下 no-forward II=1；exact-once 通过后才允许另开“删除 overflow tree”消融。

DC 必须做 matched pair：现有 M155 forwarding baseline 对 fused no-forward candidate，同一 3 ns/库/约束；分别报告：

- 删除 M154 3,072-bit result register 的收益；
- 删除 M155 7,296-bit forwarding payload 的收益；
- overflow guard 保留/删除的增量；
- area、setup/hold、fanout，以及 4×32×768 INT8 weight SRAM 和 4×768×96×19 accumulator SRAM 的明确 macro cut。

周期仿真器用 126,581,635-cycle block-K4/PWP1024 作主 parent，给 source-key/phase load、SRAM latency、backpressure、RMW/write和 commit 全收费；75,032,786 只有在 RTL 小样逐事务校准后才能晋级。

## Amdahl 只作方向判断

- Patch 合法 K4/D32 映射到冻结 envelope：1.02709× sensitivity。
- Bottleneck 1.687 same-port target：1.05516× sensitivity。
- 两者一起：1.08537× sensitivity。
- 1.805 bundled Conv 比率映射 envelope：1.06075× sensitivity。

这些来自不同 population 的 ledger/trace 组合，不是 full-network cycle。它们说明论文应该明确报 **Conv3x3 accelerator module**，而不是试图把 1.7–1.8× 写成全网 2×。

## P0

1. **选择的 Conv 链未连接。** M152/M154/M155 尚无一个真实 trace-driven fused module；M155 仍是 unsealed debug，M158 exact-once P0 未闭。
2. **同资源 macro/cycle/energy 未闭。** 还没有选 SRAM macro、load-to-use、commit、matched baseline area/energy；因此 1.687/1.805/14.858 都不能晋级。

## P1

1. Patch 强 K1 下只有 1.1049×，M218-like 还更慢，应停止主线 RTL。
2. M149 有 stalled-result fault，并且已被 M152 路线结构性绕开。
3. M154 排除了 98,304-bit SRAM/load/output-hold；wide result FF 尚未实际删除。
4. M155 50-level、+0.0164 ns 是明确 timing bottleneck。
5. 22.747× cache ratio 是 read work，不是新增 cycle speedup。
6. row-interleave zero hazard 仅对 frozen heldout order 成立。
7. profile100 envelope、patch s10 与 bottleneck heldout20 不是同一运行 population。

## P2

1. M36 是 intrusive statistics-only census。
2. Patch empty-token skip 缺 typed sparse consumer，且上限仅 1.00387×。
3. PWP1024 是 bandwidth/cycle trade，不是 energy reduction。
4. prediction Conv 只占 0.0437%，不再值得做性能 RTL。

## Allowed / forbidden

允许：报告 Conv 45.03% 热点拆分；Patch 合法最优 1.104921× 负结果；M152 比 M147 只慢 0.00426%；把 1.687018×写成 **same-PWP1024 cycle-model target**；把 1.805358×写成 bundled genealogy；把 22.747×写成 read-work reduction。

禁止：achieved standalone/full-network speedup、throughput、energy、paper PPA；把 1.805×说成 primary same-resource；把 14.858×说成已测硬件；声称 M149 integration-ready；声称 M154/M155 wide registers、forwarding 或 overflow tree 已删除；把 patch wide-bank 的 1.79–4.45×冒充 8-bank结果。

本评审只新增本目录，未改 M235、production 或 `docs/359`；`docs/359` SHA 仍为 `dedde7ce...`。
