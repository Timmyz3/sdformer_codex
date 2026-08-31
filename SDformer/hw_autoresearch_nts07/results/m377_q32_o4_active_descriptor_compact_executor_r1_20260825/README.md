# M377：q32/O4 active-descriptor compact finite executor

M377 修复 M373 测到的全行 descriptor replay 瓶颈，但不跳过 matcher：每个
partition 仍通过 SERIAL16 扫描全部 3000 行；`original16==0` 的行不写
descriptor，所有非零 fallback/PWP descriptor 携带 `row12` 连续写入，再
在两个 O4 tile 中按 active count 回放。

冻结 S10 结果：

- source rows：51,840,000；
- exact elided zero rows：30,368,111；
- active descriptors：21,471,889（41.4195%）；
- 最大单 partition active rows：2400，即 14,400 B descriptor；
- empty partition：0；
- candidate：503,016,392 cycles；
- bit-sparse baseline：543,784,143 cycles；
- module speedup：**1.081047x**；
- 相对 M373 少 60,701,662 cycles，并额外付 34,560 个 tile startup cycles；
- candidate/baseline events：120,970/69,130；单 DMA overlap 0。

该点超过预冻结 1.05x VCS 准入门，下一步仅是 active-descriptor scheduler
controller 的 RTL cycle miter；完整 q32 matcher、SRAM macro、DC/PTPX 和
系统 Amdahl 尚未准入。1.081047x 只属于冻结四个 bottleneck Conv 的模块
执行，不是全网、energy 或 DATE headline。
