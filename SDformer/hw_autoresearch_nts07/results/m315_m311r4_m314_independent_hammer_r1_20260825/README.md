# M315：M311r4 VCS 与 M314 DC 独立打铁评审

结论：91/100，P0=0、P1=1、P2=2。M311r4 的 matcher 功能与模块边界 II=1 可以 admission；M314 的 **logic-only 标准单元面积**可以限定口径 admission。3 ns 物理时序、中心 SRAM、完整 Conv、accuracy、功耗、系统加速比和 headline 全部 NO-GO。

## 复核结果

- 重新调用 exact-SHA Synopsys VCS，而不是复用旧日志：generated/accepted/retired 为 2200/2200/2200，numeric/order mismatch 为 0。
- 动态命中：exact 3、positive-distance 323、distance/tau reject 1872、population<2 guard 2、stall cycle 786、定向 tie 1。
- 六个 SVA cover 全部非零：stall-transition 579、exact 3、positive 323、tau0 941、guard 2、distance-reject 1872。stall 的 786 是“停顿周期数”，579 是“stall 后恢复 ready 的转移数”，两者不是同一指标。
- 独立 Python oracle 共 181,078 例、0 mismatch：包含 65,536 个输入模式在 tau0/tau1 下的全遍历、50,000 个随机 16-center bundle，以及 guard/tau0/tau1/tie 定向例。tie 选择最低 packed unsigned center `0x0007`；tau0 只允许 exact；tau1 覆盖 distance-one positive。
- 修改 RTL 副本后实际运行 r4 runner，预检以 exit 10 拒绝且未启动 VCS。当前 runner 不转发 CLI 参数，没有重复/late-option 覆盖面；普通 `watchdog=3525` 不再被当作 timeout，字面 `watchdog timeout` 仍会拒绝。

## r1/r2/r3/r4 失败链

- r1：在终局回执前报 coverage/termination fatal，但六个 cover 已非零；与披露的 active-region 终局采样竞争一致。
- r2：明确出现 generated=2093、written=2201、read=2200，再 fatal，直接暴露重复 payload driver 问题。
- r3：2200/2200/2200、0 mismatch 且仿真打印 PASS，随后目录仍被标成失败；与旧 runner 把正常 watchdog 计数误判为错误一致。
- r4：生产 RTL/SVA 与之前 hash 相同，TB 与 r3 hash 相同；当前只缩窄 runner regex，独立重放通过。因此“r4 相对 r3 仅 runner 修复”成立。

历史失败目录没有封存旧 runner 源文件，r1/r2 的旧 TB 也只留下 contract SHA，所以失败链是**日志与合同一致**，但不能做完整历史 byte diff。这列为 P2，不伪装成已重放旧版本。

## M314 逻辑综合

冻结 SDC 是 TSMC28、3.000 ns，clock uncertainty 0.100 ns、input/output delay 0.200 ns、max transition 0.500 ns、max fanout 32。独立复核得到：

- cell area 1965.977999 um²；leaf/combinational/sequential cells 为 2481/2433/48；macro=0。
- 135 级逻辑，critical path 2.68 ns；setup slack 0.0005 ns，hold slack 0.0024 ns。
- 五类 constraint report 均无 violation。DC 官方结论是 max fanout 不超过 32；DDC 独立结构查询得到 data-net connected-minus-driver 近似最大值 29。

最关键的 P1 是 0.5 ps setup 裕量：在 ideal clock、ZeroWireload、零宏、零 net area 下几乎没有容错，不能把 3 ns 当作物理可实现频率。应先流水化或重构 16-way Hamming/min tree，再接真实中心 SRAM 与 propagated-clock/extracted-parasitic STA。

## 可引用边界

可引用：M311r4 matcher 功能、连续 ready 下模块边界 II=1，以及 M314 限定为 logic-only 的面积/单元数/零宏与综合报告无 violation。

不可引用：3 ns 物理时序、333 MHz post-route、中心 SRAM 免费、完整 Conv、accuracy、SAIF 功耗/能效、系统加速比、论文 PPA、DATE headline 或 best-paper 结论。

复算入口：`python3 results/m315_m311r4_m314_independent_hammer_r1_20260825/audit_m315_independent.py`。
