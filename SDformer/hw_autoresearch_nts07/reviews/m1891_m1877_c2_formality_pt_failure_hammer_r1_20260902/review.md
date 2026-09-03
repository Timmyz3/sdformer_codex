# M1891｜M1877 C2 Formality/PT 唯一失败独立审阅

结论：**审计 PASS（98/100），M1877 production admission 维持 `FAIL_CLOSED`；P0=0、P1=2、P2=1。** K8 Formality 是真实的等价诊断；K8 PT 报告集完整，但有 30,442 条 hold 违例，不能称时序闭合；K1×8 没有运行。整个 quarantine 仍为 `DO_NOT_CITE`，不得把其中 K8 半轴升格。

## 身份、终态与不重试

- attempt latch、PID 3055288 failure quarantine 的 manifest 和外层 seal 均独立校验通过；runner、M1879 release、输入身份与 `docs/359` SHA 精确一致。
- 唯一 attempt 在第一项 EDA 前已消费；canonical result=0、failure quarantine=1、遗留 work/launch-lock=0，failure terminal 明确为 `retry=false`。
- M1877 顺序是 K8 Formality → K8 PT → K1×8 Formality → K1×8 PT。K8 PT 的日志触发 runner 的任何 `Error/Fatal` 即失败策略，所以 K1×8 从未启动。
- 本审阅只读，没有运行或查询 EDA/license、没有重试、没有改 RTL/runner/前序证据或 `docs/359`。

## K8 Formality：真实成功，但仅是 quarantine 内诊断

K8 使用有效的 `ARCH_MODE=0` reference/implementation pair，`formality.rc=0`，报告明确写 `Verification SUCCEEDED`：

- passing compare points = 33,656；
- failing / aborted / unverified / unmatched = 0；
- passing/failing 两行的 BBPin 均为 0；
- DESIGN LIBRARY 的非零 `u/e/*` 实例为 0；
- 两侧非零 `e SNPS_BUSHOLD / 2 of 2` 只在 TECH LIBRARY，且路径集合精确、对称；非零 `m` 是 `.db` technology macro。

所以 K8 不是逻辑不等价，也不是黑盒偷过证明。可是它属于整体失败隔离，且 K1×8 未证明，因此只允许写在内部故障审计，不能作为论文中“C2 双轴 Formality 已闭合”的证据。

## K8 PT：报告完整不等于时序通过

`pt.rc=0`，internal-complete marker、setup/hold、coverage、constraint、library、clock、wire-load 等报告均存在且已封。唯一 `Error/Fatal` 是日志首行：

`Error: Library Compiler executable path is not set. (PT-063)`

该错误没有阻止 PrimeTime 读取 netlist/SDC、更新 timing 或生成完整报告，故可判为**工具启动/环境配置错误**，而不是时序数值本身的来源。但 runner 的规则明确禁止任何 `Error/Fatal`；它将整个 attempt 隔离是正确的 fail-closed 行为。合法 successor 应修正 Library Compiler 路径或启动环境，不能把 parser 改成忽略 PT-063。

更重要的是，报告本身揭示了独立的真实时序失败：

- slow-max setup WNS = **+0.001767 ns**，setup violating paths = 0；
- fast-min hold WNS = **−0.023259 ns**，hold violating paths = **30,442**；
- 32,429 个寄存器 setup checks 全 met；hold checks 只有 1,987 met，30,442 violated，二者均 0 untested；
- 1,228 个 output setup/hold checks 中各有 140 个 `no_paths` untested，因此 All Checks 仍有 280 untested，报告里的百分比显示为 0% 只是舍入，不能写成“零 untested”。

`check_timing succeeded` 只说明检查命令完成及基本约束可解析，不表示 setup/hold closure。`pt.rc=0` 也只表示 Tcl 正常退出，不会覆盖负 slack。

## 第一性原理后续

### A｜只修 parser/environment 不够

修 Library Compiler 路径能消除 PT-063，并让新 runner 的错误卫生通过；它不能改变 −0.023259 ns hold WNS、30,442 条 hold 违例，也不能补出未运行的 K1×8。弱化 Error/Fatal parser 更不可接受。即便对同一 K8 netlist重跑，最多得到“环境干净但 hold 失败”的半轴诊断。

### B｜必须生成 hold-repaired 双轴新网表并重跑完整链

应在新的 additive DC successor 中对 K8 与 matched K1×8 都做真实 hold repair，例如在同一冻结 clock/uncertainty 下 `set_fix_hold [get_clocks core_clk]` 后 `compile_ultra -incremental`；若采用 P&R/PT ECO，则也必须冻结流程、合法单元和等资源口径。随后：

1. 重新导出两轴 mapped netlist、SDC 和各自 SVF；
2. 对两轴分别重跑 RTL→新 mapped netlist Formality，要求 passing>0，failing/aborted/unverified/unmatched=0，BBPin=0，无非零 DESIGN `u/e/*`；
3. 对两轴分别重跑 slow-max setup + fast-min hold PT，保持 OCV、0.05 ns uncertainty、无 timing exception/false-path 美化；
4. setup/hold WNS 均 ≥0，违反路径数均 0；140+140 个 `no_paths` output checks 必须通过删除真实死端口、补正确约束或逐位结构证明来解释，不能用百分比舍入隐藏；
5. hold 插入后的面积、setup 回退和功耗需重新量化，两轴同表报告。

## 论文边界

M1877 只能支持三条内部诊断事实：K8 raw Formality 成功、K8 setup 为 +0.001767 ns、K8 hold 为 −0.023259 ns/30,442 violations。由于目录终态是 `FAILED_OR_INCOMPLETE_DO_NOT_CITE`，这些都不得成为论文新准入行。当前合法论文口径仍只能引用既有 M1811 DC 数字并标注 PT/Formality 未完成，直到新的两轴 repaired campaign 通过独立结果审阅。
