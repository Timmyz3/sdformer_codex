# M351：M344 pattern DMA 修正覆盖层

M351 仅修正 M347 指出的重复收费：每个 phase 的 q-entry pattern table
在 matcher 前加载一次，后续 output tile DMA 只搬 weight 与实际使用的
PWP。封存的 M344、M347 和 docs/359 均未修改。

固定容量结论不变：双 32-KiB tile cache 为 65,536 bytes，双
`3000x48-bit` descriptor SRAM 为 36,000 bytes，当前明确计价的物理存储
合计 101,536 bytes。q128/O1 + SHARED96 + SERIAL16 的修正后串行首 tile
递推为 389,278,750 cycles、相对同口径 bit-sparse 为 1.396902x；该数字
仍只是 analytical recurrence。

宽口径 q128/O1 + WIDE144 + SYSTOLIC128 为 1.971777x，乐观跨 phase
递推为 2.038818x。二者均未准入：128 个 matcher PE、144-bit PWP 端口、
有限 cache 状态、单 DMA 仲裁、descriptor 端口、bank conflict、RTL
cycle match、面积/Fmax、能量与系统倍速尚未收口。

当前结论：GO pattern-DMA 修正与 64-KiB tile 容量证明；NO-GO 把任一
递推称为 executable cycle bound、面积公平性能或 DATE headline。下一步
必须使用两个有限 tile slot、两个 descriptor bank 和一个 32 B/cycle DMA
的可执行模块模拟器。
