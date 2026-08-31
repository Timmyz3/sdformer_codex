# M332：M329/M330/M331 独立统一打铁评审

结论：**94/100，P0=0、P1=1、P2=2**。M329 hold-guarded logic-only DC、M330 pre-layout/no-SPEF 同步 setup/hold、M331 RTL 到 mapped netlist 等价可以按限定口径准入。reset recovery/removal、物理 SRAM、post-route timing、功耗、系统加速比、paper PPA 和 headline 均 NO-GO。

本评审只读取既有证据，没有修改 RTL、合同、生产脚本或 docs/359，也没有启动新的 Synopsys 任务。

## M329：25 ps hold guard

M329 r1b 复算为：cell area **1997.981971 um²**、cells **2394**、FF **156**、logic levels **34**、critical path **1.57 ns**、setup WNS **+1.1141 ns**、DC hold WNS **+0.0252 ns**、macro=0，五类 constraint 全部 clean。

相对 M322：

- 面积 +55.440009 um²（**+2.854%**），cells -102、FF 不变。
- logic levels -2、critical path -0.01 ns、setup slack +0.0094 ns、DC hold slack +0.0251 ns。
- delay cell 数从 311 降至 192（-119），但改用更长的 DEL100/DEL150，delay-cell area 从 203.741995 增到 253.008004 um²（+49.266009，+24.181%）。因此“cell 更少”不能被解释成面积更小。

25 ps guard 只在 mapping 时将 hold uncertainty 从 0.100 ns 提至 0.125 ns。最终 report 与 mapped SDC 已恢复为 **0.100 ns**；mapped SDC 中不存在 0.125。该检查 PASS。

## M330：PrimeTime

PrimeTime W-2024.09-SP3 在 pre-layout、no-SPEF、ideal-clock、ZeroWireload、zero-macro 口径下得到：

- setup WNS **+1.1141 ns**；hold WNS **+0.0180 ns**。
- setup 156/156 MET、hold 156/156 MET，二者均 0 violated、0 untested；100 条最差 setup/hold 报告路径全部 MET。
- hold TNS 为 **0.0 ns**：由 156/156 MET、0 violated、global timing 无 hold violation，以及最差 100 条路径均非负共同确定。
- 相对 M326 的 hold WNS -0.0071 ns，提升 **25.1 ps**，说明 guard 在同一 PT flow 中确实修复了原 hold failure。

没有新增 waiver。原始 SDC、M322/M329 mapped SDC 与 M326/M330 PT flow 的唯一语义例外都是 `set_false_path -from reset_n`；M330 只是重复施加同一个 reset 例外。

## M331：Formality

Formality V-2023.12-SP3 返回 verify=1、Verification SUCCEEDED；**205 个 compare points 全绿**，构成为 49 ports + 156 DFF。failing、aborted、unverified、unmatched 均为 0。该结论只准入 M321 RTL 到 M329 hold-guarded mapped netlist 的逻辑等价，不外推 timing、SRAM、accuracy、功耗或系统性能。

## 剩余问题

P1 是 reset signoff：recovery 与 removal 各 156 项全部 untested，因为 reset_n 是异步 false path。加上当前仍是 ideal clock、no SPEF、零 SRAM 宏，因此物理 3 ns 仍不能准入。需要明确 reset release 协议，完成 recovery/removal 或同步释放设计，并做 propagated-clock/extracted-parasitic STA。

两个 P2 是证据工程问题：M330/M331 的 `output.sha256` 内容均重验通过，但源目录没有二级 seal；本次 M332 外层双层 seal 已锚定当前字节。其次，M330 只报告 ignored exceptions，未直接列 active reset waiver，receipt 也没写数值 TNS；本评审通过脚本静态扫描和 coverage/global timing 独立补齐，后续 runner 应直接封进 receipt。

可引用：M329 限定为 logic-only 的 DC 指标、guard 仅 mapping 生效且最终恢复 0.100 ns、M330 限定为 pre-layout/no-SPEF 的 setup/hold、唯一 reset 例外无新增、M331 的 205-point 等价。

不可引用：reset recovery/removal、post-route 333 MHz、物理 SRAM、完整 Conv、accuracy、功耗/能效、系统加速比、paper PPA、DATE headline 或 best-paper 结论。

复算入口：`python3 results/m332_m329_m330_m331_independent_unified_hammer_r1_20260825/audit_m332_independent.py`。
