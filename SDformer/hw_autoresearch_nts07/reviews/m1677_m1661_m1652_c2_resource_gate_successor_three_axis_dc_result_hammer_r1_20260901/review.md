# M1677｜M1661/M1652 C2 fresh 三轴 DC 独立结果打铁

## 裁决

**PASS 100/100**：`PASS100_M1677_M1661_M1652_C2_RESOURCE_GATE_SUCCESSOR_THREE_AXIS_DC_RESULT_ADMITTED`，P0/P1/P2 均为 0。

结果目录和 attempt 目录的递归 manifest、outer seal、实际文件全集与 symlink/空目录边界均通过重算。K1、typed K8、等带宽 K1×8 均从同一 M1609 12-file source cone、相同 Tcl/SDC/library、3.000 ns ideal clock 和 ZeroWireload 依次执行了一次 fresh `compile_ultra`；三个 compile epoch 不重叠，incremental/hold optimization 均为 0。每轴新 DDC 与 SVF 均不同于旧 M872 对应产物，没有旧网表 byte reuse。

三个 `dc.log` 各且仅含一条已知的 Design Vision bootstrap 错误：

`Error: Error during sourcing of /opt/synopsys/syn/V-2023.12-SP3/auxx/gui/dv/.synopsys_dv.tcl`

除此之外无 `Error`、`Fatal`、unresolved reference 或 LINK 错误，三轴 `dc.rc=0`、终态 token、setup/DRC gate 和映射产物均齐全。因此该固定 GUI startup 诊断不否定本次批处理 DC 结果。

## fresh logic-only 结果

| 轴 | ARCH_MODE | cell area (µm²) | 最小 setup slack (ns) | setup WNS/TNS（QoR 四舍五入） | DRC nets | diagnostic hold min (ns) |
|---|---:|---:|---:|---:|---:|---:|
| K1 | 0 | 124,546.967176 | +0.0011 | 0.00 / 0.00 | 0 | −0.0190 |
| typed K8 | 1 | 130,476.905184 | +0.0002 | 0.00 / 0.00 | 0 | −0.0189 |
| equal-bandwidth K1×8 | 2 | 585,534.971643 | +0.0014 | 0.00 / 0.00 | 0 | −0.0177 |

相对旧 M903 同流结果，K1/K8/K1×8 面积分别变化 `−0.0587% / −0.4648% / +0.0095%`，方向和幅度均不改变公平结论。hold 报告明确是 diagnostic-only；负 hold slack 不得写成 hold closure，也不得写作 paper PPA ready。

## 公平指标

冻结 directed VCS 五 workload 的求和周期仍为 K8 `1913`、等带宽 K1×8 `1945`。本次 DC **没有刷新周期**；它只刷新同一 M1609 source cone 的三轴 setup/area。

- 等带宽 directed component 周期加速：`1945 / 1913 = 1.016728×`。
- fresh K8 相对 fresh K1×8 logic cell-area 节省：**77.7166%**。
- 结合上述冻结周期的等带宽吞吐/mm²：**4.562720×**。

论文中若使用 4.5627×，必须在同一句注明 1.0167× 周期和 logic-only pre-macro 面积；不得把它表述为 4.56× 稀疏周期加速。K8 对单 K1 的性能 headline 继续禁止。

## 边界与复核

可以引用：fresh M1609 三轴、3 ns、TSMC 28 nm 标准单元、logic-only pre-macro 的 setup/area，以及绑定冻结 directed VCS 周期的等带宽 K8-vs-K1×8 面积效率。

不能引用为：macro-inclusive PPA、hold signoff、power/energy、PTPX、全网/系统加速或 DATE headline。

`independent_hammer.py` 分别在 CPython 3.6 与 3.12 运行通过；归一化 runtime 字段后输出完全一致，并拒绝 20/20 个 seal、人口、指标、freshness、setup/DRC、hold 边界和 cycle-refresh 负突变。审阅者没有运行 EDA、VCS 或许可查询，也未修改生产 result、attempt、docs/359 或 `ucli.key`。
