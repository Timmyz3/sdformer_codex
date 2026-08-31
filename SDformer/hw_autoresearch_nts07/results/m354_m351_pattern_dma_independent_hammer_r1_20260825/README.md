# M354：M351 pattern-DMA correction 独立打铁评审

结论：**91/100，P0/P1/P2 = 0/0/3。有限口径 GO M351 correction、固定容量与 analytical recurrence arithmetic；NO-GO executable performance、面积公平性能、系统倍速和 DATE headline。**

M351 与 M339 均由冻结合同完整重跑并和封存 JSON 字节一致；M339/M344/M347/M351 的两层 SHA 封存也由 runner 逐项验证。M351 wrapper 对 M344 只有两个 attribute assignment：`candidate_tile_load_cycles` 是唯一数值 monkeypatch，`strict_json` 只兼容 overlay contract schema。修正函数保留 weight、selective-PWP 和 32 B/cycle rounding，只从每个 output-tile DMA 删除 pattern；原 `candidate_tile_bytes` 未改，所以 pattern 仍保留在容量证明中。

10×4×432 = 17,280 个 phase 中，q16/q32/q64/q128 pattern 分别仍每 phase 搬一次，即 32/64/128/256 bytes，合计 DMA service 为 17,280/34,560/69,120/138,240 cycles。output-tile DMA 不再重复搬 pattern；weight 与实际使用的 PWP 没有漏收。16 个 q/O/port/matcher 组合的非 DMA 字段均与 M344 相同，修正周期差全部落在仅删除 pattern 可产生的严格边界内，speedup 重新除法逐项一致。

独立容量重算为：q16/O8 30,752 B/context、q32/O4 24,640 B、q64/O2 21,632 B、q128/O1 20,224 B；双 context 均小于 65,536 B。固定配置应报告 65,536 B tile cache 加 36,000 B descriptor SRAM，合计 **101,536 B**，不能简写成整个模块 64 KiB。

最可信实现种子 q128/O1 + SHARED96 + SERIAL16 的串行 analytical recurrence 为 389,278,750 cycles、1.396902x；乐观 overlap 为 380,479,705 cycles、1.429207x。这两项仍不是 executable bound，因为 pattern 物理归属、两个 cache slot 的释放时刻、单 DMA 仲裁、descriptor 端口、bank conflict、有限 queue、RTL cycle match、面积/Fmax 和能量均未实现。

三个 P2 是可审计性与后续实现缺口：pattern 只传一次但物理 residence 未明确；作者结果没有 per-phase DMA/component ledger；M339 work 在 M351 中以 SHA+boolean 继承而未展开。M354 已重放并公开所有 q work rows，但真正晋级仍需 finite-context simulator 的 accepted-transaction ledger。
