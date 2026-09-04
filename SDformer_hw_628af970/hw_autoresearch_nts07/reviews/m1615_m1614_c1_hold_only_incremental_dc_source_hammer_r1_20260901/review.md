# M1615｜M1614 C1 hold-only DC source 不同作者审阅

日期：2026-09-01

状态：`PASS_M1615_M1614_C1_HOLD_SOURCE_HAMMER__AUTHORIZE_ONE_FUTURE_DC_ATTEMPT`

评分：98/100；P0=0，P1=0，P2=1。

## 裁决

M1614 source 包通过不同作者 compile-free 审阅。允许作者编写并封存 M1616 release；只有 M1616 精确绑定本 review、M1614 runner 与 source contract，且调用者同时 pin runner/release SHA 后，才允许一次未来 DC attempt。本审阅没有运行或直接授权当前 DC，也没有创建 release。

Tcl 保持唯一 `set_fix_hold [get_clocks core_clk]` 与唯一 `compile -incremental_mapping -only_hold_time`，不存在第二次 compile、generic incremental、`compile_ultra` 或六类 timing concealment。输入模型仍为 3.000 ns、setup uncertainty 0.200 ns、hold uncertainty 0.050 ns、ideal clock、ZeroWireload 和零例外。

九个 `TS1N28HPCPHVTB128X128M4S` 宏在优化前后都有硬门，优化前执行 `set_dont_touch`；standard-cell 与 SRAM slow/max、fast/min 均有独立 `set_min_library`，工具、四个输入和所有库均 exact-SHA pin。

成功状态必须同时满足：setup 与 hold 均 `WNS>=0/TNS=0/violations=0`、面积不超过 154608.7116945 um²、宏数前后均为 9、DRC violating nets 为 0，并保持输出 SDC 的时钟/uncertainty/零例外。失败或超过面积门只发布 sealed negative，`retry=false`。

## Mutation hammer

Python 3.6 与 3.12 均复算冻结 audit：60/60 攻击被拒。攻击覆盖：双 compile、generic/ultra、删 `set_fix_hold`、Tcl/SDC 六类例外、3 ns 与两项 uncertainty、propagated clock、九宏/dont-touch/min-library/ZeroWireload、面积 +5%、缺报告、忽略 setup/hold/WNS/TNS/violations/DRC、attempt 移到 tool 后、插入其他 EDA、retry、release gate、DDC/SDC/netlist/SVF/tool/library SHA、same-UID/ancestry/common-shell collision，以及 contract 越权。

原作者 12 项静态测试也在 Python 3.6 和 3.12 下各 12/12 PASS；runner `bash -n` PASS。M993、original quarantine、M1006、M1612、author handoff 和 contract 的内外封印全部复核通过。

## P2

最终 publication 使用早期 freshness check、独占 lock 和已消费 attempt 后执行 `mv`，但没有 post-move canonical exact-topology assertion。它在同 UID 单次运行边界内不阻塞本次 release authoring；未来 result hammer 必须拒绝 nested、额外或异常 canonical topology，才能准入结果。

## 边界

本审阅不证明 hold closure、setup/hold timing、面积结果、Formality、PrimeTime、power、energy、speedup 或 paper-ready PPA。即使未来 DC positive，也仍需不同作者 result hammer、M993 gate-to-gate Formality、direct RTL Formality 与独立 PT slow/max + fast/min。
