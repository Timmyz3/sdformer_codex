# M910｜M903 Table-A component annex r11 静态打铁

## 裁决

`PASS 100/100`。M903 可以成为第一条**真实 Synopsys component evidence
row**，但不能进入 M698/M706 的 full-system Table-A PPA 行。现有 Table-A
schema 要求十算子、17 个 SRAM macro、Formality/PT/PTPX/能量闭环；M903 是
五个冻结 directed workload、零 macro 的 logic-only pre-macro component。

因此 M910 不修改 M698/M706，只添加一个强类型 `component_annex`。扩展后：

- production component rows：`1`；
- full-system Table-A production rows：`0`；
- system speedup / power / energy / macro-inclusive PPA / paper headline：全部
  `false`。

## 第一条 component 行

| axis | cell area (µm²) | 最小 setup slack (ns) |
|---|---:|---:|
| K1 | 124,620.173180 | +0.0020 |
| K8 | 131,086.241193 | +0.0013 |
| K1×8（等带宽基线） | 585,479.153645 | +0.0012 |

冻结的五 workload 求和为 K8 `1913` cycle、K1×8 `1945` cycle。只在 K8
对等带宽 K1×8 的 component 口径下，可以引用 `1.01672765×` 周期、
`4.541077998×` throughput/mm²，以及 `77.6104%` logic cell-area saving。

## 静态验证

Python 3.6 的 builder/negative tests 为 `12/12 PASS`。独立静态检查器没有
导入 builder，重新校验 M706/M903 双封，并从 `1945/1913` 与两轴面积独立
复算三个派生量。没有执行 EDA、GPU、remote、license query，也未修改
`docs/359`。
