# M359：M358 双槽 phase-serial 构造周期独立打铁

结论：**86/100，P0=0、P1=3、P2=4。GO 构造性模块 DSE；NO-GO executable cycle、等面积或论文 headline。**

M358 双重 seal 已通过。另从冻结的 40 个 packed payload 和 40 个 value
payload 重新校验 SHA，独立解包并重建 51,840,000 个 row-partition 观察、
17,280 个 phase 和全部 q16/q32/q64/q128 工作。独立审计器不 import
M358/M344/M339/M43；16 个周期行与 M358 均为 0 mismatch。fresh M358
producer replay 也与封存结果逐字节一致。

## 已通过的构造性边界

- 每个 phase 只收费一次 `ceil(q*2/32)` pattern DMA，随后严格执行
  matcher、packer、首 tile DMA；
- 首 tile DMA 全暴露，后续只有当前 tile compute 与下一 tile 的同一 DMA
  server 取 max，最后 tile、2-cycle tail 和公共 commit 都收费；
- 独立时间戳检查为 0 DMA overlap、最多两个 live slot、0 cross-phase
  overlap；
- q/O=16/8、32/4、64/2、128/1 的 slot0 pattern+tile 最大占用分别为
  30,752、24,640、21,632、20,224 bytes，均小于 32 KiB。一个合法容量
  witness 是把唯一 pattern table 固定保留在 slot0 的独立地址区；
- 64 KiB 只指两个 tile slot。加上两份 3000x48-bit descriptor bank，固定
  cache+descriptor 为 101,536 bytes。

## Admission 分层

所有 16 行可以保留为“显式资源下、可复现的构造性 analytical DSE”，但
没有一行可以称为 executable/RTL/equal-area/system cycle。

同资源 seed 只看 SHARED96+SERIAL16：

| q/O | M358 | 全 3000 行 descriptor packer 敏感性 | 裁定 |
|---|---:|---:|---|
| 16/8 | 1.064001x | 1.053491x | GO exploratory |
| 32/4 | 1.077553x | 1.067225x | **GO preferred** |
| 64/2 | 1.065782x | 1.055889x | GO exploratory |
| 128/1 | 0.997319x | 0.988776x | NO-GO accelerator |

WIDE144 相对 SHARED96 增加 1.5x PWP payload width；SYSTOLIC_Q 在 q32、
q64、q128 分别使用 SERIAL16 的 2x、4x、8x distance PE。故峰值
1.639256x 只能作为显式资源扩张 Pareto endpoint，不能作为公平 headline。

## 三个 P1

1. M358 为每个 raw row 预留 48-bit descriptor，却只按 assignment_rows
   收 packer；fallback 写、跨 output-tile 读、端口、bank arbitration 与
   backpressure 未执行化。最简单的全 3000 行写敏感性已使最佳公平点从
   1.077553x 降到 1.067225x。
2. 两槽与单 DMA 的抽象时间序列合法，但仍是 closed-form recurrence；没有
   finite queue、地址/valid generation、SRAM 端口与 RTL cycle miter。
3. 峰值行依赖额外端口或额外 matcher PE，尚无 DC throughput/mm² 归一。

四个 P2 是：pattern 物理地址未显式；动态 PWP gather/burst 粒度未定义
（q128 per-vector 32B 对齐使公平点变为 0.995639x）；M358 未保留逐 phase
事件账本；公共 output accumulation/commit 仅以双方相同的末尾标量收费。

下一步 GO q32/O4 SHARED96+SERIAL16：补 descriptor 读写端口、pattern/tile
地址、单 DMA queue、PWP gather burst 的 finite-state 模块调度器，再做
zero-mismatch cycle；同时用 Synopsys DC 对 PE16/32/64/128 和 96/144-byte
端口做 throughput/mm² Pareto。此前 NO-GO 任何 DATE headline。
