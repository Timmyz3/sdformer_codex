# M1089 最终 checkpoint 重绑定只读审计

结论：**现在不是只能等 checkpoint。** 最终 c12 身份仍未闭合，但 C1 零工作任务、decoder 外部信任根、C2 mapped-gate 观测三个 checkpoint 无关的阻塞可以立即继续。最终 checkpoint 到位后，不需要把全部 RTL 与静态 DC 从零重做；需要重做的是所有含活动、稀疏、周期、流量、能量、精度和系统分母的数字。

## 身份现状

| 人口 | 已封身份 | 审计结论 |
|---|---|---|
| H67 ep35 | checkpoint `4f33e086...5158`；config `8be3f7bb...6c49` | 当前硬件证据人口；不得直接转给 c12 |
| c12 Motion | parent `7e8d524e...6cbb`；本地已封 ep14 `d51877d1...8654`、ep19 `024306e8...2cc8` | 两者都只是 candidate；未发现本地已封 ep24/ep29、valid825 或 selection receipt |
| Local5 ep44 | checkpoint `19820bec...4f57` | 只可作为独立、窄边界组件/control 证据；绝不与 c12 混分子或分母 |

“未发现 ep24/ep29”只描述本地 sealed ledger，不代表远端训练实时状态。

Local5 证据存在内部措辞冲突：`DATE_LOAD_AUDIT_APPENDIX_20260817.md` 表格说 ep44 无 RTL 继承、ep29 仍是 anchor；`docs/425` 及其 final audit 又封了 ep44 的窄组件级重绑定。保守处理是只承认 `docs/425` 的 100-group、OUT_DIM=2、非 foundry、非 full-encoder 边界，并禁止向 c12 转移。

## 哪些保留，哪些重放

| 模块 | 最终拓扑/精度完全相同时可保留 | 最终 c12 必须重放 | 当前先修阻塞 |
|---|---|---|---|
| C1 | one-port/parent/tie/dead-write 协议、SVA、M959 功能 VCS、同宏 datasheet、容量公式 | parent-hit/source/product/dead-write、`435,293,339` cycle 与 `1.7468x`、流量、SAIF/PTPX、M623 活动能量 | M1074 在 zero-work task 207 产生不可能 psum R/W；新 namespace 做 additive repair |
| C2 | typed signed K8/Acc24 协议；M903 参数相同 RTL 的 logic-only 面积；五例 directed `1913/1945`、`1.0167x`、`4.541x throughput/mm2` 仅保留为 directed component | 最终 raw4/value class、trace 加权周期/利用率、范围证明、SAIF/PTPX、energy/frame、系统贡献 | M1080 mapped VCS header 后停顿；新 namespace 先定位首个 X/state cone |
| C3 | Fixed-T10 协议、directed `17 cycle/tile`；M928 logic-only `62,433.50 um2` setup/area | 系数、threshold、数值范围、tile/activity、trace 周期、能量和系统份额 | hold 未闭合；T/rank/ROM/precision 改变则重综合 |
| Decoder | source-order/arbitration 概念、相同 shape 的 mapper/schema | D0-D3 payload、D1 value class/miter、周期、psum/地址流量、活动、多序列完整 aggregate | M1084 判 M1083 自签 side provenance 可伪造；须来自 lower-level witness 或独立 sealed producer 的 trust root |
| RQTB | exact quotient 协议、相同 shape/score precision 的组件 RTL/PPA | `1.1865x`、S10 `1.1764x`、energy `80.50→68.39 nJ`、等值类/K-zero/Q-empty、旧 `1.000911x` | 无主线 blocker；保持局部 attention 次贡献 |
| 系统表 | 十算子/17 宏 schema、baseline ladder、validators、表格式 | ordered trace、所有 payload、decoder-complete 分母、周期份额、地址计时 SRAM/DRAM、FPS、SAIF/PTPX、energy/frame、Table A/B/C | 旧 `620.3M/620.87M` 既绑定 ep35 又漏四层 ConvTranspose，不能再叫完整系统 |

静态证据也不是无条件继承。若 layer shape、T、rank、descriptor/Acc 位宽、memory depth、compiled coefficient、value class、overflow proof 或控制流改变，相关模块必须定向重综合/重验证。

## 最小重绑定顺序

1. checkpoint 前先修三项基础设施：C1 zero-work、decoder trust root、C2 mapped observation；同时冻结一次性统一 capture 方案。
2. 用 sealed valid825/selection 协议只选一个 ep24 或 ep29；锁 checkpoint/config SHA 与 `missing=0/unexpected=0` load audit。
3. 先做 topology/precision/parameter diff，逐模块判“静态复用”还是“定向重综合”。
4. 导出并封存 INT weight/bias/threshold/neuron constants，重做 overflow/saturation proof。
5. 对同一多序列 cohort **只抓一次** ordered full-network trace 和全算子 payload，所有模块共用同一 manifest roots。
6. 优先快杀 decoder D1 value class；随后依次重放 C1、C2、C3、RQTB。
7. 在同资源合同下生成组件周期，再生成 address-timed SRAM/DRAM。
8. 组装 decoder-complete unified replay；最后才做 SAIF/PTPX、memory energy、Table A/B/C 与有损 Pareto。

## 时间与风险

如果 checkpoint 前三个基础设施阻塞已清、最终 topology 基本不变，最终身份封存后约需 **4–6 个专注工作日**。若 D1 value class 或 C1 zero-work 修复引发结构重做，约 **7–10 个工作日**。训练等待时间无法从本地 sealed evidence 判断。

最大风险排序：最终 ep24/ep29 身份未封；decoder trust root；C1 full replay；D1 数值类；decoder-complete address-timed 分母；C2 mapped-gate；C3 hold/macro/power。

本审计采用数据质量流程，把“结构静态可复用”与“checkpoint/trace 统计必须重放”逐字段拆开，并对 Local5 冲突采取保守边界。它不新增任何性能、PPA、能量或系统准入主张。
