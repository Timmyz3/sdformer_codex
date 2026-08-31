# M172 independent hammer review r1

## Verdict

**85/100 — CONDITIONAL PASS.** M172 可以称为：

> 冻结 H67 ep35、120 个 FC2 payload 上，M171 always-ready 前端的
> **逐 token 隔离延迟解析边界**。

现有整数账本和 `446,528,624 / 179,057,955 = 2.493765909x` 独立复算
通过；但不应把这两个和数称为连续 token 流的总 wall cycles。M171 在
`token_done` 被消费之后还需要一个 re-arm edge 才能接收下一 token。120 个
payload 一共对应 5,580,000 token，连续串行执行要补 5,579,999 个 token
边界周期，变成：

| 口径 | K1 | K4 | K1/K4 |
|---|---:|---:|---:|
| 隔离 token 延迟之和（M172 原数） | 446,528,624 | 179,057,955 | 2.493765909x |
| M171 连续串行 token 流 | 452,108,623 | 184,637,954 | 2.448622362x |

因此 2.493766x 仍可保留，但名称必须是 isolated-token analytic latency-sum
ratio；若写 frontend serialized wall-cycle ratio，应使用 2.448622x。

## 独立复算证据

- 独立检查 120/120 payload、437,760,000 bytes；SHA、大小、popcount、几何、
  `tail_used_bits` 均 0 mismatch。
- 小端 bit packing 逐 byte/bit 重构 2,048 项，0 mismatch。
- 重新得到 3,502,080,000 input elements、143,894,510 events、54,720,000
  scan beats、1,863,944 zero tokens。
- 不导入 production analyzer；另写解析调度器和逐 RTL edge 状态机。
  6,216 exhaustive、16,000 randomized、720 real-payload probe，共 22,936
  case，0 recurrence mismatch。
- aggregate 与四个 stage 的全部整数项 0 mismatch。stage wall ratios 独立得到
  1.761925091x、2.425725077x、2.707184454x、2.853770787x。
- K4 per-beat replay `144,999,276` 相对 M168 cross-beat ideal
  `106,536,803` 为 **1.361025222x**。分 stage fragmentation 为
  1.415861534x、1.345958623x、1.351275169x、1.363638832x。

## Claim boundary

可以称 standalone frontend analytic boundary，但必须同时保留 frozen
payload、always-ready、one-beat prefetch、64-bit scan、per-beat grouping、
isolated-token latency-sum 等限定。

不能称 RTL speedup：真实 120 payload 没有在 RTL 中逐周期 replay；M171 VCS
只证明了 5 个 directed token 的协议/守恒语义，Python 才产生本次大规模周期数。

不能称 physical speedup：M171 DC 是 fail-closed，103 logic levels 超过预锁定
80；`5,082.713990 um2` 和接近零的 setup margin 不能替代一个通过 admission 的
物理结果，而且仍是 ideal-clock、ZeroWireload、0 macro。

不能称 complete FC2/FFN speedup：模型没有 weight SRAM request/response、bank
macro conflict、M169 arithmetic handshake、2304-bit accumulator context、BN2、
residual 或 FC2 commit。

不能称 system speedup：没有 FC1、attention、conv、全局 memory traffic、调度、
FPS/energy，也只有一个 checkpoint 的十个样本。

## P0 / P1 / P2

### P0 — 2

1. 修正“wall”口径：生产结果/合同应把 2.493765909x 改称 isolated-token
   latency-sum ratio，或加入跨 token 状态机后报告连续流 2.448622362x。未修前
   不得把 2.493766x 写成 frontend serialized wall speedup。
2. 在升级为 FC2 性能结论前，必须把有限带宽 weight SRAM response、M169 K4
   arithmetic 和 accumulator context 接入同一 ready/valid 周期模型；当前的
   always-ready group consumer 只是前端上界。

### P1 — 4

1. M171 64-bit 平坦 selector 的 DC 已因 103 logic levels fail-closed；要做分级
   priority/局部压缩或直接取消二次 bitmap scan，并重新通过预锁定 DC。
2. 对 8 个 bank 的 cross-beat reservoir 做容量/occupancy/overflow DSE。理想收益
   上限已经由 1.361025222x fragmentation 给出，但无限队列不能直接实现。
3. 把 ATLIF 96-lane producer 与小型 8-bank reservoir 融合，直接产生 FC2 source
   descriptors；这是优先于单纯把 scanner 从 64-bit 加宽的下一硬件 trick。
4. 增加 per-sample 分布与更多 DSEC sequence/checkpoint，避免十样本总和掩盖尾部
   bank 冲突和 frontend stall。

### P2 — 3

1. M172 README 的 fragmentation 写成 `1.361033x`，应与整数账本一致为
   `1.361025x`。
2. 在结果 JSON 同时记录 isolated-latency 与 serialized-stream 两套 numerator /
   denominator，避免后续表格误拿口径。
3. 给真实 payload 抽样的 RTL/VCS trace replay 增加 cycle-by-cycle 对账；它不是
   120 payload 全跑 RTL 的替代品，但能把 Python edge convention 锁到 VCS 时间轴。

## Next architecture recommendation

最优下一步不是继续扩大平坦 scan。应做 **ATLIF-native 96-lane event tap +
8-bank finite reservoir + M169 K4 consumer**：producer fusion 直接省掉 54,720,000
个 64-bit SRAM scan beats；reservoir 再跨 producer beat 配对，目标恢复 M168 的
106,536,803 K4 replay 下界。单独 cross-beat reservoir 只能回收当前 replay 的
1.361025x fragmentation；单独 wide-scan 会加重已经 103-level 的 selector
关键路径。新的 milestone 必须同时给 reservoir occupancy、overflow/bypass、
SRAM banking 和 VCS/DC 证据。

`docs/359` SHA256 复核仍为
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

