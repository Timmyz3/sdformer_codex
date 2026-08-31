# M323：M321 VCS 与 M322 DC 独立打铁评审

结论：**93/100，P0=0、P1=1、P2=2**。M321 matcher 功能、两级 elastic 协议、模块边界 II=1 和无停顿 latency=2 可以 admission；M322 的 setup 架构门与 logic-only 面积可限定口径 admission。hold 技术上 MET，但只有 **0.1 ps**，列 P1；物理时序、SRAM、完整 Conv、accuracy、功耗、系统加速比、paper PPA 和 headline 全部 NO-GO。

## M321 独立 VCS 复核

- 重新调用 exact-SHA Synopsys VCS，而不是复用旧日志：generated/accepted/retired 为 **3000/3000/3000**，numeric/order mismatch 为 **0**。
- 最大连续 accept/retire 为 **259/257**。动态命中为 exact 1、positive-distance 433、distance/tau reject 2564、population<2 guard 2、stall cycle 909、定向 tie 1。
- 七个生产 SVA cover 全部非零：stage0-full 2561、stall-transition 698、exact 1、positive 433、tau0 1284、guard 2、distance-reject 2564。
- 额外的 exact-production-RTL 定向 VCS 得到 accepted/retired=8/8、0 mismatch；覆盖 stage0-only、both-full、连续两拍 full stall、两级 payload 稳定、5 次 simultaneous push/pop、stalled output 后空 stage0 吸收、tau0 exact-only、tau1 positive、guard 和跨 group 等距时 lower-center tie。
- 独立语义 oracle 共 **181,078** 例、0 mismatch；独立两级 FIFO 模型另跑 continuous-ready 4096/4096 与随机背压 50,000/50,000，未见丢失、重复或重排。

## Runner 与封存

- 修改 RTL **副本**后实际调用生产 runner，SHA 预检以 exit 10 拒绝，VCS 未启动；生产 RTL 未修改。
- 当前失败 regex 只匹配字面 `watchdog timeout` 等错误形式，正常 `watchdog=4492` 不会误伤；runner 不转发 `"$@"`，没有 late/duplicate option 覆盖面。
- fresh VCS、独立 directed VCS、M322 DC 的 nested manifests 均重新验签通过；最终目录另有外层双层 seal。

## M322 logic-only DC 与 M314 对比

M322 在相同 3.000 ns、TSMC28 pre-macro 口径下为：cell area **1942.541962 um²**、leaf/combinational/sequential cells **2496/2340/156**、macro=0、36 级逻辑、critical path 1.58 ns、setup slack **+1.1047 ns**、hold slack **+0.0001 ns**；五类 constraint 均 clean。

相对 M314 的 1965.977999 um²、2481 cells、48 FF、135 levels、setup +0.0005 ns：

- 面积 -23.436037 um²（-1.192%），leaf +15、comb -93、FF +108。
- 逻辑级数 -99（-73.33%），critical path -1.10 ns，setup slack +1.1042 ns。
- setup 架构门要求 margin≥0.5 ns 且 levels≤60；M322 分别有 **0.6047 ns** 和 **24 levels** 额外余量，所以该门是真 PASS，不是刚好擦线。

最大风险是 hold：M322 插入 DEL025 235 个、DEL075 76 个，共 **311 个 delay cell / 203.741995 um²**，最差 hold 仍只剩 **0.1 ps**。因此“technical hold MET”可以报告，但在 ideal clock、ZeroWireload、零 macro、无寄生下不能外推物理时序。1.192% 面积下降也对 hold-fix 流程敏感，只能保留 logic-only 口径。

## 缺口与可引用边界

P1：用真实 SRAM 边界、propagated clock 和 extracted parasitics 做物理 STA，并同时收口 setup/hold。P2：把 stage0 payload 稳定、occupancy conservation、simultaneous retire/shift/accept 和 empty-stage0 absorption 四类内部不变量补入生产 SVA；物理流完成前不要把小幅面积下降升级成物理优势。

可引用：M321 matcher 语义、两级 elastic 边界的 II=1/latency=2/背压顺序稳定；M322 pre-macro setup 架构门；M322 限定为 logic-only 的面积、单元数和零宏事实。

不可引用：可靠物理 hold/3 ns post-route、333 MHz 物理工作频率、中心 SRAM 免费、完整 Conv、valid825、SAIF 能效、系统加速比、论文 PPA、DATE headline 或 best-paper 结论。

复算入口：`python3 results/m323_m321_m322_independent_hammer_r1_20260825/audit_m323_independent.py`。
