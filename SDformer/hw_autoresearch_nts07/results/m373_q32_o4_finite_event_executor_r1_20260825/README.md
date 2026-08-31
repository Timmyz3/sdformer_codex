# M373：q32/O4 全行 descriptor 有限事件执行

M373 把 M359 首选的 k16/q32/O4、SHARED96、SERIAL16 点落实为带地址与
时间戳的模块级事件 trace：单 32 B/cycle DMA、两个 32 KiB slot、两个
3000x48-bit descriptor bank、8-entry bundle FIFO、全行 matcher-to-SRAM
write、两次 descriptor replay，以及更强的 next-phase-overlap bit-sparse
baseline。

- baseline 精确复现 M358：`543,784,143 cycles`；
- candidate：`563,718,054 cycles`；
- 相对 baseline：`0.964639x`，低于冻结 1.05x gate，故本版本 NO-GO；
- 120,970 个 candidate event、69,130 个 baseline event；单 DMA overlap 0；
- slot0 最大 24,640 B，全部 payload 32-byte 对齐；
- 两次 tile replay 共付 `60,736,222` 个 zero-work descriptor dispatch cycle。

负结果定位出一个具体、无损且可实现的后继：matcher 仍扫描 3000 行，但只
把非零 fallback/PWP descriptor 连续写入 active stream；row ID 已在 48-bit
descriptor 中，不需要零行占 SRAM/replay 拍。仅从 M373 账本删除零回放得到
`1.081121x` 是后继上界/预检，不是已准入周期；必须另立合同重新生成地址、
计数、有限队列和时间戳。

M373 是 timestamped module-cycle evidence，不是 RTL match、Synopsys area、
energy、system speedup 或 DATE headline。
