# M1635｜M1634/M1609 C2 三轴 logic-only DC 源码独立 hammer

日期：2026-09-01

状态：`PASS_M1635_M1634_M1609_C2_THREE_AXIS_DC_SOURCE_HAMMER__AUTHORIZE_ONE_FUTURE_ATTEMPT`

评分：99/100；P0=0，P1=0，P2=1。

## 结论

M1634 通过不同作者的只读源码门。12 个 RTL 文件中只有一个 `m214_fc2_raw4_to_descriptor4_terminal_hint_compactor` 定义，即 M1609 successor；冻结 M214 不在 filelist。M1609 的公开错误线严格为 `protocol_error = fault_q`，illegal header/raw 仅在时钟边界粘滞写入 `fault_q`。

逐分支追踪不是单纯 token 检查：K1(mode 0)、K8(mode 1) 和等带宽 K1×8(mode 2) 分别进入三个不同的根模块，但每条可达源码锥都最终到达 `m216 raw frontend → M1609/m214`。三条轴不存在旧 compactor 或旁路 compactor。

## Fresh synthesis 与公平性

runner 只在一个 `0,1,2` 循环中调用三次 DC；三轴使用同一 top、filelist、Tcl、SDC、slow/fast standard-cell library 和 3.000 ns 时钟，只改 `ARCH_MODE` 与输出目录。Tcl 从 RTL `analyze` 并参数化 `elaborate`，每轴仅一次 `compile_ultra`；没有 `read_ddc`、mapped-Verilog 导入或 M872 产物复制。旧 M872 只是身份/比较来源，不是任何新轴的综合输入。

物理标签闭合为：TSMC 28 nm 标准单元、3.000 ns、setup/hold uncertainty 0.200/0.050 ns、ideal pre-CTS clock、ZeroWireload、logic-only pre-macro、0 macro。hold 只是诊断报告，不是 closure。

## Hammer 与授权边界

CPython 3.6 和 3.10 各自通过 20 类静态检查，并各自拒绝 50/50 个变异：包括三分支单独绕过 M1609、旧 DDC 注入、轴专用 filelist/SDC/library、2.5 ns 偷换、propagated clock、false/multicycle exception、第二次/增量综合、提前 claim 与放行链倒置。`bash -n` 通过。

审阅时 M1636 release、attempt、result、PID work、lock 和 quarantine 全部不存在；本轮没有启动 DC/VCS/PT/PTPX/Formality。本回执只允许下一位作者建立精确绑定的 M1636 one-shot release；当前 DC 仍为 0。

## P2 与红线

M1634 的目标是刷新三轴面积/setup 证据，不会重跑五组合法等带宽 workload。因此 1913 vs 1945 cycle 与 1.0167× 仍属于冻结 M867/M903 directed component 证据，不得改写为 M1634 新测或系统加速。新面积也必须等 M1634 原始结果通过另一轮独立 result hammer 才能进入论文表。

`docs/359` 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`；未修改 `ucli.key`。
