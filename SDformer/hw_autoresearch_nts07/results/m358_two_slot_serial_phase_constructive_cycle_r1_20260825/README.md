# M358：双 tile-slot、单 DMA、phase-serial 构造性周期

M358 在 M351 修正 pattern DMA 重复收费后，移除了全部跨 phase
preprocess overlap。每个 phase 先依次完成一次 pattern DMA、matcher 和
packer；首个 output tile DMA 完全暴露，后续 tile 只允许通过两个
32-KiB slot 与当前 tile compute 乒乓重叠。32 B/cycle DMA 只有一个，
容量不足必须 stall。

固定存储仍是 65,536-byte tile cache + 36,000-byte 双 descriptor bank，
合计 101,536 bytes。10 个冻结 S10 样本、四个 bottleneck Conv 的结果：

| q/O | Port | Matcher | 相对同口径 bit-sparse |
|---|---|---|---:|
| 16/8 | SHARED96 | SERIAL16 | 1.064001x |
| 32/4 | SHARED96 | SERIAL16 | 1.077553x |
| 64/2 | SHARED96 | SERIAL16 | 1.065782x |
| 128/1 | SHARED96 | SERIAL16 | 0.997319x |
| 128/1 | SHARED96 | SYSTOLIC128 | 1.223476x |
| 64/2 | WIDE144 | SERIAL16 | 1.352908x |
| 128/1 | WIDE144 | SYSTOLIC128 | 1.639256x |

这说明 q128 的 2.04394x exact vector-work 不能自动兑现成硬件周期：在同
96-bit 端口和 SERIAL16 matcher 下，8-pass 匹配开销把收益完全吃掉。
WIDE144 和 128-stage systolic 的高点分别增加端口宽度和 matcher PE，
均未面积归一。

当前结论：GO 双槽/单 DMA/无跨 phase overlap 的构造性账本；NO-GO 把
任何一行称为 executable RTL、等面积或系统倍速。下一步应以 q32/q64
为实用候选，对 q/PE/端口做 Synopsys DC 的 throughput/mm2 Pareto，而
不是继续只放大 q128 work reduction。
