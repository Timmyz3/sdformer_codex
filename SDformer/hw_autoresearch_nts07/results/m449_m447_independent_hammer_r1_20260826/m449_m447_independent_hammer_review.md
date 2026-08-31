# M449 对 M447 的独立打铁结论

评分 **82/100**，P0=0、P1=3、P2=2。M447 的全量算术 DSE 是准确的，但 `2.187x/2.413x` 只能保留为单 96 B/cycle 基线下的机会值，不能进入硬件性能主表。

独立程序没有导入 M40/M401/M423/M430/M447 analyzer，从冻结 M40 packed plane 和 M430 catalog 重建了 51,840,000 行、17,280 phase。M430 K1 separate 517,041,352 cycle、strong zero K1 742,148,386 cycle，以及 M447 六点、逐 phase 字段、early matcher、distance histogram 全部 0 mismatch。

## 最关键的公平对照

| 点 | cycle | 对单口 zero K1 | 对 equal-K zero | 对 fused 总字节宽度下的 zero |
|---|---:|---:|---:|---:|
| K1 fused | 430,154,216 | 1.72531x | 1.72531x | 1.01161x vs zero K2 |
| K2 fused | 339,335,872 | 2.18706x | 1.28236x | 0.98884x vs zero K3 |
| K4 fused | 307,609,552 | 2.41263x | 0.94192x | 0.85686x vs zero K5 |

独立 equal-K zero K1/K2/K3/K4/K5 分别为 742,148,386 / 435,149,895 / 335,550,364 / 289,743,472 / 263,579,630 cycle。K4 连 equal-K zero 都打不过；K2 按 352 B/cycle fused 输入总宽度折算后也略输 zero K3。因此 >2x 主要来自增加 correction 端口宽度，不能称为资源归一化优势。

## 语义与位宽

行级公式正确：PWP separate 为 `1+ceil(d/K)`，PWP fused 为 `max(1,ceil(d/K))`，fallback 为 `ceil(popcount/K)`。只要未来 RTL 对每个 delta chunk 执行 `new_psum=old_psum+delta`，就不会重演 M426 丢失 old_psum 的 P0。

冻结权重全量重算得到 PWP 范围 `[-1089,1059]`，保守 K4 fused 范围 `[-1601,1571]`。signed13 安全，但 signed12 已足够；signed19 downstream accumulator bound 218,338 仍成立。

## 去留

- 保留 M430 K1 separate/M433，M447 不替换现有 admitted point。
- K4 RTL 直接 NO-GO。
- K2 只允许作为 matched-port falsification probe，不能为了保住 2.187x 而上 RTL。
- 若确认 M430 已有 PWP 与 correction 端口可同拍读取，K1 fused 可作为最小增量 probe；它相对 M430 是 1.20199x，而不是 >2x headline。
- 先做同 SRAM/互连假设下的 banked zero K2/K3/K4/K5 定价，再决定是否值得投入 composer RTL。

`docs/359` 在审计前后均为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
