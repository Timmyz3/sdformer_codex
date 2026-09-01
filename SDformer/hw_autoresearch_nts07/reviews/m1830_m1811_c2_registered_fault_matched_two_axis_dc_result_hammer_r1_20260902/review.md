# M1830｜M1811 C2 registered-fault matched two-axis DC 独立结果打铁

## 裁决

**PASS，99/100，P0=0，P1=0，P2=0。** 本次只读审阅未运行 DC、VCS、PrimeTime、Formality、PTPX 或许可查询，也未修改 M1811 canonical、前序证据或 docs/359。

M1811 的 65 个 manifest member、manifest 自身及 outer seal 已递归、穷尽重算通过；无 symlink、绝对路径、`..`、重复项或未列文件。已消费 attempt 只记录 `k8,k1x8` 两轴、恰好两次 DC、`retry=false`。canonical 根目录也只有 `k8` 与 `k1x8` 两个轴目录；没有第三轴、旧 DDC/旧 mapped netlist 输入或自动重试。

## 可引用结果

| 轴 | ARCH_MODE | top | cell area (µm²) | setup 路径 | 最小 / 最大 setup slack (ns) | hold diagnostic |
|---|---:|---|---:|---:|---:|---:|
| K8 | 0 | `m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24_ARCH_MODE0` | 130,822.775176 | 100/100 MET | +0.0018 / +0.0033 | 100 条 violated，最差 −0.0190 ns |
| K1×8 | 1 | `m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24_ARCH_MODE1` | 585,534.971643 | 100/100 MET | +0.0014 / +0.0021 | 100 条 violated，最差 −0.0177 ns |

在完全相同的 51 个逻辑 public port、3321 个 scalar port bit、3.000 ns ideal clock、`ssg0p9v125c` target library 和 `ZeroWireload` 下：

- K8 相对等带宽 K1×8 的 logic-only cell-area efficiency 为 **4.475787727750473×**；
- K8 的 logic-only cell area 减少 **77.65756419144124%**；
- 两轴 macro/black-box 数均为 0，setup 均 MET。

这正式准入的是 **matched equal-bandwidth、logic-only、pre-macro 的 setup/area 证据**。它支持 C2 的“共享 typed-signed K8/Acc24 fabric 用远少于八份 K1 复制逻辑提供相同 public memory-port bandwidth”这一面积效率主张。

## 身份与结构核验

冻结 filelist 恰好 13 行、13 个源文件互异，当前源文件 SHA 与 runner 中 13 个 pin 全部一致。source wrapper 的 ANSI header 独立解析得到 51 个唯一逻辑端口；两份 mapped netlist 的 top header 也分别得到相同顺序、相同集合的 51 个端口，DC 报告均为 3321 scalar port bit。两个 mapped netlist 都只定义对应的单一 parameterized top，没有另一模式 top 或 legacy top。

两轴均从冻结 RTL 重新 elaborate；日志中没有 `read_ddc`、旧 mapped netlist 读入或旧结果路径复用。每轴恰好一次 `compile_ultra`、零 incremental compile、零 hold optimization；`TIM-209=0`、`OPT-150=0`，setup constraint 无 violation，macro/black-box 为 0。K8 compile wall 为 3565 s，K1×8 为 5685 s。

每份 `dc.log` 恰好一个 `Error:`，均为 `env -i` 未提供 HOME 时 DC GUI 启动脚本 `.synopsys_dv.tcl` 的同一非功能性 sourcing 错误；随后完整 elaboration、compile、report、netlist 写出及唯一 terminal 全部完成。未发现其他 Error、Fatal、Internal、未解析引用或 link failure。该白名单只适用于这一条精确 GUI startup 文本。

## 声明边界

可以引用：TSMC 28 nm 标准单元、3.000 ns ideal clock、ZeroWireload 下，M1809 K8 与 matched K1×8 的 logic-only pre-macro setup/area；上述 4.4758× area efficiency 或 77.6576% logic-area reduction 必须与 matched equal-bandwidth、logic-only、pre-macro 限定语同句出现。

不可引用为：cycle speedup、full-network/system speedup、power、energy、PTPX、macro-inclusive PPA、hold closure、post-layout/signoff、silicon 或 paper-PPA-ready。M1811 没有运行 VCS/Formality，mapped functionality 仍由前序 RTL VCS 链支撑，不应把本次 DC 结果写成 RTL↔netlist 等价证明。

下一门是对这两份确切 mapped top 做独立 Formality 与双角 PrimeTime；hold、power/energy 和系统周期仍须各自独立闭合，不能由 M1830 推导。
