# M169/M170 独立打铁评审

结论：**83/100，通过 standalone arithmetic-island boundary；完整 FC2、物理加速、能效和系统 headline 均未通过。**

我独立审阅了 M169 K4、M170 matched K1 的 RTL/SVA/TB、sealed VCS 和同条件 DC 报告，并用两个新随机种子重跑现有 VCS 可执行文件。两次 fresh run 都通过：M169 seed `16917031` 为 361/361 issue/result、89 次连续 II=1、358 次 same-cycle replace、99 个 stall；M170 seed `16917037` 为 361/361、89 次连续 II=1、358 次 replace、79 个 stall。协议攻击和 signed24 overflow 攻击均保持已接受结果并 fail-close。

## 独立核算

| 指标 | K4 / M169 | K1 / M170 | 独立判断 |
|---|---:|---:|---|
| 96-lane signed24 update | 4 个 distinct-bank INT8 weight/issue | 1 个 INT8 weight/issue | 意图差异正确 |
| DC cell area | 18,522.881882 um2 | 11,940.011991 um2 | ratio = **1.5513285829161612** |
| Sequential cells | 2,343 | 2,341 | K1 常量 source-count 优化掉 2 bit，属真实 K4 开销 |
| Setup slack @ 3 ns | +0.8670 ns | +1.6146 ns | 两边均 MET |
| Hold slack | +0.0224 ns | +0.0221 ns | 两边均 MET |
| Logic levels | 38 | 9 | K4 reduction/overflow path 明显更深 |
| Macro / mapped multiplier | 0 / 0 | 0 / 0 | 仅 logic-only |
| 顶层 ports | 8,152 | 5,548 | K4 多 2,604，ZeroWireload 没有计线代价 |

M168 冻结 payload 给出的 output-block cycles 是 K1 `412,900,394`、K4 `106,536,803`：

```text
bank-service boundary = 412900394 / 106536803
                      = 3.8756597004323474x

logic-area ratio      = 18522.881882 / 11940.011991
                      = 1.5513285829161612x

area-normalized bound = 3.8756597004323474 / 1.5513285829161612
                      = 2.498284208202332x
```

因此收据中的 `1.551328583` 和 `2.498284208` 数值正确。可以称为：

> frozen H67 payload 下、matched 3 ns logic-only DC 的 **arithmetic-island logic-area-normalized bank-service boundary**。

不能称为 physical speedup、完整 FC2/FFN/system speedup，也不能称为能效或 paper-ready PPA。

## 公平性审查

逻辑岛 A/B 是合理的：两边同为 96 lanes、signed24 外部 accumulator、相同 tag/last/count/mask 结果状态、同一 one-entry elastic 接口、相同 fail-close，并使用相同 DC 版本、库、corner、3 ns 约束、ideal clock、ZeroWireload 和 flatten flow。K4 多出来的 4-row inputs、unique-bank legality 和 reduction tree 是应计入的创新成本。

但它不是物理公平比较。K4 每拍需要 `4 × 96 × 8 = 3072 bit` weight response，K1 只需 768 bit；SRAM macro、bank queue、response tag 和 routing 都没进 DC。尤其 M169 比 M170 多 2,604 个顶层端口，而 ZeroWireload 不惩罚这些线。故 2.498x 只能是算术岛逻辑面积归一化上界。

## RTL 语义审查

M169 的 nonempty/prefix/unique-bank legality 正确，slot0 在所有 legal issue 中必有效；两级 signed9/signed10 reduction 后再做 signed25 accumulator add，`extended_sum[24] != extended_sum[23]` 是 signed24 越界检查。M170 的 8-to-25 sign extension 和同一 overflow 规则也正确。两边 accepted overflow result 都可在 sticky fault 后继续 drain；one-entry result 支持消费旧结果并同拍接受新 issue。

VCS scoreboard 覆盖随机正负权重、随机 accumulator、backpressure、II=1、same-cycle replace、overflow 和非法请求。Fresh run 证明结果不依赖 sealed seed=1。不过它仍是 module-directed test，不是 120 个真实 payload 的 RTL schedule replay。

M168 链接可信：result SHA 与 contracts/preflight 一致，evidence manifest 对 analyzer、原 manifest、1.3 GB archive 和 docs/359 全部校验通过；writer 使用 little-bit-first pack，M168 的 `bank=input_channel mod 8` 解码吻合。公式 `max(max_bank_count, ceil(events/K))` 对“每 bank 每拍最多一项、全局最多 K 项”的独立队列模型是精确最小周期，但 compactor/queue 尚未实现。

## P0（4）

1. 实现有限深度 event compactor + 8 个 persistent bank queues，并在 frozen payload 上证明无丢失、无越界、周期数可达 M168 bound。
2. 给出四个 96-byte weight row/cycle 的 SRAM banking、响应 tag、macro area/timing/energy；当前 3072 bit/cycle 是免费输入假设。
3. 集成 accumulator context owner/feedback/writeback，按真实 FC2 顺序重放 120 个 payload 并对软件 golden；当前 2304-bit context 全由外部每拍提供。
4. PAFT/valid825 必须证明 released checkpoint 的 sn2 threshold 仍全为 1，或重建 folded-weight numeric bridge。

## P1（5）

1. 用相同 SRAM/context wrapper 做 macro-aware matched K4/K1；目前多出的 2,604 ports 没有 routed cost。
2. 对 exact-payload replay 产 SAIF，跑 macro-inclusive PTPX；activity mask 只是 row enable，不是 bit toggle。
3. 补 M169/M170 RTL-to-netlist Formality。
4. 加入 FC2 requant、BN2 fold 和 residual commit 后再报告完整 FC2 cycles。
5. 做一个单一 composition receipt，把 payload、queue、SRAM、context、K4/K1 和 cycle counter 锁到同一证据链。

## P2（3）

1. 两个 DC log 都有 truncation part-select 的 VER-318 signed-to-unsigned warning；应显式 cast 或留审计 waiver。
2. 增加 `-128/127`、全部 legal bank combination、全部 duplicate pair 的定向覆盖。
3. M170 旧 `r1` 已正确标成 `FAILED_OR_INCOMPLETE_DO_NOT_CITE`；下游只能选 sealed `r1b`。

下一里程碑建议做最小可执行 FC2 shell，而不是扩大裸算术 lane：8-bank finite queues + 4-bank weight response + 一个有界 context slot + end-of-block commit。这样才能判断 3.875660x 在 memory/context 成本之后还能保留多少。

机器可读裁决见 `m169_m170_independent_hammer_review.json`，独立数值与 fresh VCS 记录见 `fresh_recompute.json`。
