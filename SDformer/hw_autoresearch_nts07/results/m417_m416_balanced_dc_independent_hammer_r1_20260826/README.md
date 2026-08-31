# M417：M416 平衡树 DC 独立打铁评审

结论：**94/100，P0/P1/P2 = 0/0/3；接受 M416 的逻辑级 DC 里程碑，GO Formality，GO PrimeTime STA diagnostic。**

## 独立复算结果

M416 在与 M412 完全相同的 TSMC28、3.0 ns SDC、slow/fast library、ZeroWireload、ideal clock 与 25 ps mapping hold guard 下得到：

| 指标 | M412 串行选择 | M416 精确平衡树 | M416−M412 |
|---|---:|---:|---:|
| cell area | 24,885.377609 um2 | 24,548.705582 um2 | -336.672027 um2（-1.3529%） |
| leaf cells | 21,582 | 20,803 | -779（-3.6095%） |
| sequential cells | 4,100 | 4,100 | 0 |
| logic levels | 111 | 52 | -59（-53.1532%） |
| setup worst slack | +0.0008 ns | +0.7636 ns | +0.7628 ns |
| hold worst slack | +0.0250 ns | +0.0250 ns | 0 |
| macro / black box | 0 | 0 | 0 |

五类约束（max/min delay、max capacitance、max transition、max fanout）全部无违例；DC log 未见 Error/Fatal、unresolved reference、black box、inferred latch 或 timing loop。映射后的 Verilog、SDC、DDC、SVF 均存在且进入 M416 seal。

## 功能与账本

M414 原始 Synopsys VCS seal 和 M415 独立 VCS seal 均复核通过。directed、integration、51,840,000-row full-runtime 三组 compile/sim 返回码均为 0，full-runtime mismatch 为 0。冻结 M401 matcher-cycle 账本仍为 **67,912,100**，增加 pipeline stage 为 0，task-ledger delta 为 0；testbench 的 67,981,225 raw cycles 仍不得当 speed。

## DEL 与边界

M416 有 4,815 个 DEL-class cell、面积 8,098.272110 um2，占逻辑级 cell area 的 32.9886%。相对 M412，DEL 数量 +67、DEL 面积 -44.351999 um2；没有 hold-guard A/B，不能做因果归因或把 DEL 从面积中直接扣除。

三个 P2 分别是：M414 的三处 VER-318 signed/unsigned warning 尚待 Formality 封口；DEL 面积占比仍高且缺 A/B；当前仍是 ideal-clock、ZeroWireload、0-macro 的 pre-macro screen。

## GO / NO-GO

- GO：M414 RTL ↔ M416 mapped netlist Formality；旧 M405 串行 RTL ↔ M414 平衡 RTL 的顺序等价可一并形式封口。
- GO：对 M416 exact mapped netlist + mapped SDC 做 PrimeTime STA diagnostic。
- NO-GO：PTPX/power（无真实 SAIF）、物理频率、SRAM-inclusive area、系统倍速、paper PPA、DATE headline。

`docs/359_DATE终局冻结_20260813.md` 未修改，SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
