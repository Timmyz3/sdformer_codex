# M903｜M872/M803 C2 R16 三轴 DC 独立结果打铁

## 裁决

**PASS 100/100**：`PASS100_M872_M803_C2_R16_THREE_AXIS_LOGIC_ONLY_DC_RESULT_ADMITTED`。

本次只读审阅确认 K1、K8、等带宽 K1×8 来自同一个 runner PID `2339146`、同一个已消费 attempt，三轴均满足 `TIM-209=0`、`OPT-150=0`、七件产物齐全、setup 无 violation，递归 manifest 与 outer seal 全量重算通过。审阅者未运行 DC/VCS/许可查询，也未修改 canonical 或 docs/359。

## 正式结果

| 轴 | 绑定 | cell area (µm²) | QoR setup WNS/TNS (ns) | 最小逐路径 setup slack (ns) | 资源快照 |
|---|---|---:|---:|---:|---:|
| K1 | `ARCH_MODE=0` 单 K1 诊断轴 | 124,620.173180 | 0.00 / 0.00 | +0.0020 | 11 |
| K8 | `ARCH_MODE=1` channel-split 候选 | 131,086.241193 | 0.00 / 0.00 | +0.0013 | 11 |
| K1×8 | `ARCH_MODE=2` 等带宽公平基线 | 585,479.153645 | 0.00 / 0.00 | +0.0012 | 34 |

K8 相对单 K1 的面积比是 `1.051886×`，但**不得**用 K8 对单 K1 的吞吐差作为 headline；公平吞吐比较只能是 K8 对等带宽 K1×8。

冻结的五个 directed VCS workload 周期为：

- K8：`[51, 131, 486, 1231, 14]`
- K1×8：`[53, 133, 499, 1246, 14]`

逐 workload 的等带宽 K8/K1×8 周期加速为 `1.039216× / 1.015267× / 1.026749× / 1.012185× / 1.000000×`；对应吞吐/mm² 比为 `4.641518× / 4.534555× / 4.585837× / 4.520790× / 4.466366×`。

五 workload 直接求和（K8 `1913` cycle、K1×8 `1945` cycle）得到：

- 等带宽 directed component 周期加速：**1.016728×**。
- 等带宽 directed component 吞吐/mm²：**4.541078×**。
- K8 相对 K1×8 的 logic-only cell-area 节省：**77.6104%**。

这些是五个冻结 directed component workload 的求和，不是 full-network、trace-weighted 或系统工作负载。

## 证据边界

可以引用：

- TSMC 28 nm 标准单元、3.000 ns ideal clock、ZeroWireload 下的 logic-only pre-macro DC setup/area；
- 冻结 directed VCS workload 上，K8 相对等带宽 K1×8 的周期与吞吐/mm²。

不可引用为：

- macro-inclusive PPA；
- hold signoff（本次 hold 只作 diagnostic）；
- power、energy、PTPX、系统或全网加速；
- K8 对单 K1 的性能 headline；
- DATE 摘要中的系统 headline。

完整机器可读证据见 `review.json`；独立重算入口为 `independent_hammer.py`。
