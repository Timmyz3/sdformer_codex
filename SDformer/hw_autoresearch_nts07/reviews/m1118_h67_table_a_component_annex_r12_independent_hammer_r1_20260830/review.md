# M1118 Table-A component annex r12 异作者静态打铁

## 裁决

**PASS：准入 3 条带严格限定语的 component-annex 行；full-system Table-A production row 仍为 0。**

本轮未修改 config、builder、tests、contract 或任何 sealed authority；未运行 EDA、GPU、remote 或生产流程，`docs/359` 保持冻结。

## 独立复算

- **C1 / M1102–M1114：**10 sample、812,160 task 的 raw-CPU same-ledger replay，candidate `434,242,823` cycles，strongest-zero / same-coordinate-bit 均为 `763,908,050` cycles，独立相除为 `1.7591725401987818×`。容量账本 `122,880 + 49,152 + 42,880 = 214,912 B`，相对 `245,760 B` 预算余量 `30,848 B`。
- **C2 / M903：**五个 frozen directed workload 求和为 K8 `1,913` cycles、等带宽 K1×8 `1,945` cycles，独立重算为 `1.01672765×` cycle speedup、`4.541077998×` throughput/mm²、`77.6104%` logic-cell area saving。
- **C3 / M928：**TSMC 28 nm、3.000 ns ideal clock、ZeroWireload、logic-only pre-macro，面积 `62,433.503388 µm²`、71,898 cells、最低报告 setup slack `+0.0003 ns`、macro count 0；hold 未闭合。

Canonical builder 输出的 row ID 精确为 C1、C2、C3 三条，component rows=`3`，full-system rows=`0`，system speedup、power/energy、final-checkpoint binding 和 paper-PPA-ready 均为 false。

## 负向攻击

13 类攻击全部拒绝：duplicate JSON key、NaN、full-system claim 升格、C1 RTL claim 升格、C3 power claim 升格、M910/C1/C3 authority 漂移、C1/C2/C3 metric 变异、live-seal extra member、live manifest symlink。作者测试另行以隔离 Python 运行，21/21 通过。

## 可引用边界

- C1 仅可称为 frozen H67 四层 bottleneck Conv 的 raw-CPU same-ledger component opportunity；不是 RTL、mapped-gate、decoder-complete、final-checkpoint 或 system speedup，214,912 B 也不是物理 SRAM macro PPA。
- C2 仅可称为五个 directed component workload 上等带宽 K8-vs-K1×8 的 logic-only pre-macro cycle/area-efficiency 点；不是 full-network、trace-weighted、power 或 energy 结果。
- C3 仅可引用 Fixed-T10 component 的 logic-only pre-macro DC setup/area；不得推导 hold closure、PT STA、power、energy、throughput、speedup或 system evidence。
- 三个局部数字不得相乘，不得进入 full-system Table-A 行或摘要 headline。
