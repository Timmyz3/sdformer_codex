# M1634｜M1609 C2 三轴 logic-only DC 源码作者回执

日期：2026-09-01

状态：`PASS_M1634_SOURCE_AUTHOR_RECEIPT__REQUEST_DIFFERENT_AUTHOR_M1635_HAMMER__NO_EDA_AUTHORIZED`

评分：96/100；P0=0，P1=0，P2=1。

## 结构结论

M1609 可以不改 wrapper 接入现有 C2 三轴顶层。它保留了旧 `m214_fc2_raw4_to_descriptor4_terminal_hint_compactor` 模块名和完整端口；新 filelist 只选 M1609，明确排除冻结 M214。源锥为 `M1609/m214 → m216 → m519 standalone → M803 top`，因而 K1、K8 和 K1×8 都会吃到同一 registered-fault 修正。

这也意味着不能只重综合 K8，却复用旧 M872 K1×8 网表：那会混用不同 fault 语义，可能偏置面积和时序。M1634 因此要求三轴同 filelist、同端口、同 3.000 ns SDC、同 slow/fast library 全部新跑。

## 物理产物合同

沿用 M872/M903 已准入 setup/area 流：每轴一次 `compile_ultra`，ideal clock，ZeroWireload，0 macro；setup 和 DRC 是门，hold 只报告不宣称闭合。每轴必须新产出 mapped Verilog/SDC、DDC、SVF、area/QoR 以及 setup/hold 报告，供后续 Formality、PT 和 mapped-SAIF/PTPX 使用。任一轴失败都整体 quarantine，不自动重试。

Python 3.6 和 3.10 均通过 12/12 项测试，并各自拒绝 24/24 类 source/runner/contract 攻击；`bash -n` 通过。

## P2 与红线

M1627 证明的是 compactor-local fault 修正；全 K8/K1×8 的五组合法 directed workload 未用 M1609 重跑。因此 1.0167× 仍是 M867/M903 冻结的 directed component 周期数，不是 M1634 新测周期，更不是系统加速。新面积只能在 M1634 结果独立准入后替换物理列。

当前仅请求不同作者 M1635 源码 hammer；M1636 release 不存在，未授权 DC。本轮没有运行 DC/VCS/PT/PTPX/Formality，没有修改 `docs/359` 或 `ucli.key`。
