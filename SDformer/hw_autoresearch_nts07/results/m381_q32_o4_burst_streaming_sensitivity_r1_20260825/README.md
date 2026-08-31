# M381：q32/O4 PWP burst 与 streaming SRAM 压力测试

M381 在写 M377 controller RTL 前，补上两类会吃掉 `1.081047x` 的实际开销：

1. 不再把离散 PWP center 聚成一个虚构的大 burst，而是从冻结 S10 原始
   trace 重建每个 phase 的 q32 `used_center_bitmap`，按 center-ID 最大连续段
   计 DMA 命令；
2. 显式收费 active-count/bitmap seal、DMA 命令建立时间与 descriptor SRAM
   固定响应延迟，并把 II=1 streaming 与逐 descriptor 阻塞分开。

精确重建结果：

- 17,280 个 phase，51,840,000 source rows；
- 30,368,111 个 exact-zero，21,471,889 个 active descriptor；
- 12,709,384 个 PWP descriptor，8,762,505 个 fallback；
- 6,762,595 个 popcount-1 fallback 全部保留；
- 每 phase 平均使用 31.396/32 个 center，但只形成 1.472 个连续 run；最大
  used center/run 分别为 32/10；
- M377 candidate 与 543,784,143-cycle bit-sparse baseline 均精确复现。

有限开销结果：

- `cmd=32 cycles, SRAM L=8, II=1` 的预冻结 robust 点为
  505,195,832 cycles，即四个 bottleneck Conv 模块级 **1.076383x**；
- `cmd=8, SRAM L=8, II=1` 为 1.079460x；
- 相同点每 replay descriptor 串行阻塞 0.25 cycle 仍为 1.056935x；
- 串行阻塞 0.345 cycle 降到 1.048620x，1 cycle 降到 0.994668x。

因此仅准许实现 phase-exclusive、in-order、固定 `L<=8`、steady-state
`II=1`、D8 credit 的 minimal streaming active-descriptor controller；任何
blocking/shared-port 实现均不属于该 GO 点。上述数字不是全网 speedup、energy、
PPA 或 DATE headline，仍需 M382 独立打铁、VCS cycle miter 和 Synopsys 结果。

