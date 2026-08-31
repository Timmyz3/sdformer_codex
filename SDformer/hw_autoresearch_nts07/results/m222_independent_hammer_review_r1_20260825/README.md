# M222 独立打铁评审

结论分成两层，不能混写：

- **M222 exact 筛选本身 GO**。独立从 60 个 M51 原始 bitpack 重新解码，3×3/pad1/stride1-or-2、`source_key=9*channel+tap`、K-bank group、product conservation、scan/service/commit 及表中 ratio 全部复算一致。这个里程碑是一份可信的负结果，评分 **88/100**；就 exact screen 单项为 95/100、P0=0。
- **M222 写出的 next gate NO-GO，P0=2**。不能做“96 INT8 MAC K1 对 add-only KxD”的 matched sparse DC，也不能把需要 16 个等价 128-bit read 的 K8D32 当成现有 8×128-bit 设计推进。

## 独立复算

| 项 | 独立结果 |
|---|---:|
| records | 60 |
| decoded active input bits | 325,287,254 |
| valid 3×3 source contributions | 1,774,268,587 |
| source×destination updates | 170,329,784,352 |
| ideal input-vector scan | 69,120,000 cycles |
| ideal 96-channel commit | 40,320,000 cycles |

复算器没有 import/call M222 analyzer；它对 K=1/2/4/8 分别直接建立 bank occupancy。所有 M222 integer 字段精确相等，ratio 误差不超过 `1e-12`。

## 公平性裁决

| 点 | 8×128b 合法 | serial ratio | lanes | 裁决 |
|---|---:|---:|---:|---|
| K1×D96 | 是，6 个 read | 1.000× | 96 | 强 service 基线成立，但需补 equal-capacity rotating stripe/routing |
| K4×D32 | 是，8 个 read | 1.104921× | 128 | 最好合法点，仍远低于 1.5× gate |
| K8×D16 | 是，8 个 read | 0.946285× | 128 | 比 K1 慢 5.37% |
| K8×D32 | 否，16 个 read | 1.793944× | 256 | 资源扩展点，不是现有 8-bank 加速 |

K8D16 的 group 数确实减少 `5.658962×`，但 D16 必须扫 6 个 destination slice，所以 service 是 1,881,194,940 cycles，反而高于 K1 的 1,774,268,587。K8D32 把 slice 降到 3 才出现 1.794× serial，但它相对 K1 同时用了 `2.667×` peak weight bits 和 `2.667×` add lanes；serial speedup/lane 只有 `0.672729×`。

最关键的 P0 是：六层输入都是 exact binary，活动 source 的乘法在 K1 中同样退化为 signed INT8 weight-add。若用 96 MAC 给 K1 定价、只给 KxD 用 adders，就把双方共同拥有的二值化收益只算给 candidate。正确 sparse matched 参考必须是 **96-add K1 vs add-only KxD**；96-MAC 只能另列 conventional dense baseline。

## 未闭合的 P1

仍缺 K1 rotating stripe 的等容量 bank 布局和 selector、accumulator 位宽/端口/K-way tree、96-channel commit 带宽、bounded line-buffer/stall schedule、真实 SRAM/DC/energy，以及 current-batch dynamic BN。M162 已表明 `no_running` AEE 为 1.309925，而可静态 fold 的 `running` AEE 为 1.469151；因此不能默认把 patch BN 折掉。

十个 clip 的结论很稳定：K4D32 为 `1.0999–1.1085×`，K8D16 为 `0.9388–0.9520×`，K8D32 为 `1.7823–1.8052×`。但它们全部来自 `zurich_city_09_a`，profile100 字段仍只是 ledger sensitivity。

最终建议：保留 M222 作为“强 K1 杀掉虚假 patch K8 headline”的负证据；停止 96-MAC-vs-add-only DC，也不为 K8D32 扩 bank 制造倍数。若只为防御审稿，可低优先级做一个 96-add K1 vs K4D32-128add 的小型 matched DC；性能主线按原合同转 FC1 Acc19。

机器可读评审见 `m222_independent_hammer_review_r1.json`，逐 raw receipt 见 `m222_independent_raw_recompute_receipt.json`。
