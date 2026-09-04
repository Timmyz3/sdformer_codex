# M119 PWP weight tail-bypass mapper 独立打铁评审 r1

日期：2026-08-24  
评分：**88/100**  
严重度：**P0=1，P1=6，P2=5**

结论：M119 在 standalone behavioral 1-cycle SRAM 模型内真实补上了 M117 缺失的 256-bit 数据口、三拍 768-bit 重组、load2→首 event tail-bypass 和 signed19 mapper；但它未与 M117/M118 集成，也没有 sequence-aware 去重，因此 M117 P0 只能判为 **部分关闭（standalone kernel closed）**，M118 exact-once P0 **未关闭**。

本评审只写本目录，没有修改 production RTL、contract、既有 sealed evidence 或 docs/359。

## 独立商业 VCS 结果

Synopsys VCS V-2023.12-SP1 exact-SHA run 位于 `vcs_run_r1/`。

| 指标 | 独立结果 |
|---|---:|
| groups | 129 |
| reverse / seeded-permuted keys | 64 / 64 |
| repeated-key groups | 1 |
| 256-bit loads/reads/responses | 387 / 387 / 387 |
| events / updates | 513 / 513 |
| signed lane checks | 49,248 |
| tail-bypassed first events | 129 |
| INT8 -128 / +127 checks | 513 / 513 |
| `-(-128)=128` checks | 257 |
| event input backpressure | 5 cycles |
| old retire + new accept | 1 |
| malformed attack classes | 7 |
| older accepted update fault drains | 1 |

三个数字已独立复算：

```text
129 groups × 3 beats = 387 reads
128 groups × 4 events + 1 group × 1 event = 513 events
513 updates × 96 lanes = 49,248 signed lane checks
```

## M117 P0：standalone 关闭，整体部分关闭

独立 memory model 是固定一周期响应的同步 256-bit 端口。每组的时序为：

```text
load0: issue beat0
load1: consume response0 + issue beat1
load2: consume response1 + issue beat2
event0: consume response2 through tail bypass
```

129 个首 event 全部在 response2 同周期接受，没有额外 tail cycle。数据不是 identity-only：三拍真实携带 96 个 signed INT8，组装成 768-bit，再映射成 96×signed19。

数值边界全部通过：lane0 固定为 -128，lane1 固定为 +127；negate 时 `-(-128)` 精确得到 signed19 的 +128，`-(+127)` 得到 -127。

因此 M117 review 指出的 identity-only/tail-bypass P0，在 **M119 standalone kernel** 上关闭。但下列内容仍开放：

- M117 scheduler 没有实例化 M119；
- 没有 shared payload arbiter；
- SRAM 仍是 behavioral fixed-latency，不是 foundry macro；
- weight request 没有 M117 context；
- 没有 exact heldout descriptor/payload replay。

所以整体判定是 **PARTIALLY_CLOSED_STANDALONE_KERNEL_ONLY**，不能写成集成关闭。

## M118 P0：exact-once 仍未关闭

beat retry、beat skip、错误 key、错误 type 都会 sticky fail-closed。一个更年轻的 malformed token 出现时，已经接受的旧 update 会保持完整的 block/row/1824-bit delta，并在 `update_ready` 恢复后正常 drain；这一点通过。

但 M119 没有 sequence ID 或 event 去重状态。独立反例是：

1. event 被接受并占据 output skid；
2. `update_ready=0`，相同 event 仍保持 valid；
3. 旧 update 退休时，完全相同的 event 再次被接受；
4. 最终产生两个数值相同的 update，没有 `protocol_error`。

严格 ready/valid 合同下，生产者在握手后继续重试相同 token 属于上游违规；这个反例并不说明 M119 在其 standalone 合同内错误。但它明确证明 M119 自身不能提供 M118 P0 要求的 retry/replay exact-once 证明。要关闭 P0，必须把 M117 sequence identity 一路带到 M119/M118，并对 planned/accepted/accumulated transaction 做一一守恒。

## 尚未完成的物理与性能闭环

- `weight_rd_data` 没有 response-valid、可变 latency、ECC 或 error；
- shared arbiter/bank conflict 可能打破固定三拍连续响应；
- 1824-bit update 尚未接 M118，也没有物理布线/STA/PTPX；
- M119 request 只有 7-bit key，没有 context-dependent weight bank identity；
- 不是 heldout full-trace replay，也不是 matched physical baseline。

因此 M109 的 **2.535462× 仍为 precompacted same-clock 软件投影**。M119 只是证明在指定 standalone timing contract 下，可以避免那个确定的一-cycle tail bubble；没有把投影升级为 scheduled RTL、physical、system 或 headline speedup。

## GO / NO-GO

| 项目 | 判定 |
|---|---|
| production exact-SHA sealed VCS | **GO** |
| independent commercial VCS | **GO** |
| standalone 3×256b→768b | **GO directed behavioral** |
| load2→first-event tail-bypass | **GO directed behavioral** |
| INT8→signed19 与 negate 边界 | **GO directed** |
| event/update backpressure | **GO directed** |
| older update fault drain | **GO directed** |
| M117 standalone payload/tail kernel | **CLOSED** |
| M117 integrated P0 | **PARTIALLY CLOSED** |
| M118 integrated exact-once P0 | **NO-GO** |
| foundry SRAM/shared arbiter | **NO-GO** |
| M117→M119→M118 heldout replay | **NO-GO** |
| M109 2.535462× projection | **GO，qualifier mandatory** |
| scheduled/physical/system/headline 2.535462× | **NO-GO** |
