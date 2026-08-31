# M378：M373 q32/O4 finite-event executor 独立打铁

结论：**92/100，P0/P1/P2 = 0/0/5。全行 descriptor 版本性能 NO-GO；active-only compact 只能进入重新生成与验证，不能预先准入 1.0811×。**

M378 从 exact-SHA runner fresh replay M373。fresh result、candidate CSV、baseline CSV 与原封存逐字节一致；原 M373 双层 seal 也在正确的上级目录逐文件通过。独立扫描 120,970 个 candidate event 与 69,130 个 baseline event，按各 sample 最终时间戳重算：baseline 543,784,143 cycles，candidate 563,718,054 cycles，速度比 0.964638509×，candidate 慢 19,933,911 cycles（3.666%）。因此全行版本 NO-GO 稳固。

有限资源核账通过的部分：17,280 phase；candidate/baseline DMA event 为 51,840/17,280，单 DMA overlap 0。slot0/slot1 分别为 `[0,32768)`/`[32768,65536)`；最大 tile payload 24,576 B，pattern+tile0 最大 24,640 B。所有 DMA 地址和长度为 32 B 对齐。两个 3,000×48-bit descriptor bank 分别为 `[65536,83536)` 和 `[83536,101536)`，容量与非重叠成立。

M378 重新解包 40 个 raw record、核算 51,840,000 行：zero 30,368,111，active 21,471,889。两次 tile replay 的 zero dispatch 正好是 `2×30,368,111=60,736,222`；active exact work 为 422,285,576，二者和为 M373 descriptor compute 的 483,021,798。所有 partition 都满足聚合 `active work >= 4×active rows`；从 exact rule 看，非零 bundle 的 O4 服务至少 4 拍，零 bundle 1 拍，而 reader 至多每拍产一个，所以首包之后不会 underflow。

但 M373 没有逐拍更新 FIFO occupancy，只把深度 8 当上界；也没显式收费 descriptor read/FIFO 首包启动。若保守地给 34,560 次 tile replay 各加 1 拍，candidate 为 563,752,614、速度比 0.964579373×，仍 NO-GO。这个修正只让负结论更强。

从 M373 直接减掉 60,736,222 个零分派拍得到 502,981,832、1.081120845×；再加每 tile 一拍得到 503,016,392、1.081046566×。两者都只是 active-only 后继的算术预检。紧凑写入会改变 descriptor 数量、地址递增、空 partition、首包和 FIFO 时间戳，必须 fresh 生成事件并独立核对，之后才可能进入 VCS controller。当前无 RTL cycle match、Synopsys area、energy、system speedup、paper PPA 或 DATE headline 准入。
