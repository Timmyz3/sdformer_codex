# M421：M420 双 Formality 独立打铁

结论：M420 的两个 Synopsys Formality 作业都真实通过，且 M421 独立重跑得到相同的 5,368 个 passing compare points（1,268 个 port、4,100 个 DFF），因此准许 M416 mapped netlist 进入 PrimeTime STA 诊断。M420 不足以支撑“all internal state equivalence”；论文和后续收据必须改称“all-input-trace observable sequential equivalence”。

## 独立复现

- serial RTL → balanced RTL：`verify_return=1`，5,368 passing，0 failing、0 aborted、0 unverified。
- balanced RTL → M416 netlist：`verify_return=1`，5,368 passing，0 failing、0 aborted、0 unverified。
- M416、M417、M420 的嵌套 manifest 和外层 seal 全部复核通过；输入 SHA mismatch 为 0。
- `docs/359_DATE终局冻结_20260813.md` 仍为 `dedde7ce...`，未修改。

## 320 个 unread DFF 的真实语义

它们不是“不可达状态”，而是当前 selected-slice 顶层中的不可观察 debug counter：

- adapter：`low_accepts_q`、`high_accepts_q`、`narrow_blocks_q`、`wide_blocks_q`、`contributions_q`，共 5×32 bit；
- matcher：`source_rows_q`、`pass0_tasks_q`、`pass1_tasks_q`、`early_stops_q`、`results_q`，共 5×32 bit。

子模块把这些计数器连到 debug output，selected-slice shell 又只接到无消费者的本地 `matcher_debug_*` / `adapter_debug_*` wires；它们不驱动顶层输出、控制、协议检查或数据路径。

- RTL→RTL 中，320 对寄存器按名字匹配，但 Formality 把它们列为 `Not Compared / Unread`。
- RTL→netlist 中，Formality 明列 `320 reference, 0 implementation` unmatched unread DFF。
- M416 DC 恰好用 `OPT-1207` 删除同名 10×32=320 bit，数量和名字逐项一致。

因此，mapped netlist 删除它们在 observable cone 上是合法的；但 M420 没有证明这些内部计数器值相等。默认 `report_unmatched_points` 会隐藏 unread 点，不能据此把 receipt 写成笼统的 `unmatched_points=0`。

## VER-318 不是“优化寄存器”

M416 的三个 `VER-318` 是 signed/unsigned elaboration conversion warning：两处 loop-index part select 和一处 `ROWS_PER_PHASE` 比较。实际的寄存器移除使用 `OPT-1207`，对象是上述 320 bit debug counters。balanced RTL→netlist Formality 证明覆盖了三处 conversion 对全部可观察 compare point 的语义影响，但 warning 分类仍须保留，不能写成“三个被优化寄存器”。

## 准许口径

可以写：

> 在未添加输入约束的 Formality 设置下，serial→balanced RTL 和 balanced RTL→M416 mapped netlist 均通过所有 1,268 个 port 与 4,100 个 observable-DFF compare point；320 bit debug counters 在本顶层不可观察，RTL→RTL 未比较，RTL→netlist 被综合删除。

不可以写：

- all internal state equivalence；
- every legal and unreachable state；
- 不加限定的“0 unmatched points”；
- Formality 已证明 PT、物理时序、SRAM、功耗、能效、系统倍速或 paper-ready PPA。

评分：88/100，P0/P1/P2 = 0/2/2。结论为 conditional PASS：技术 proof 接受，GO PrimeTime STA diagnostic；M420 receipt 的 all-state 和笼统 unmatched wording 必须由 M421 边界覆盖。
