# M2C 96-lane 与 PrimeTime 收口（2026-08-21）

## 结论

M2C 将 M2B 的验证数据面从 16 个输出 lane 扩成网络常见的 96 个输出 lane，并在同一组
H67 ep35 / Local ep44 真实 bitmap 上完成 Synopsys VCS/SVA、DC、Formality 和 PrimeTime
prelayout STA。P4/P8 暂时只构成 logic-only Pareto 集；在 SRAM 端口、带宽、功耗与
full-system 目标函数齐全前不单选候选。本里程碑没有把 transaction 倍率写成全系统加速。

同时对 97,200 个 admitted tile 完成了可逆 XOR bank-remap DSE。所有 P4/P8、Local/Motion
组合都未达到 3% 生存门槛，因此不增加 remap RTL，也不为一个低于 2% 的 per-operator
bank-conflict 收益引入 weight 预排和映射选择状态。

## 96-lane VCS/SVA

96-lane 实例保持与 16-lane 相同的 256-source bitmap、per-bank 独立 word frontier、bank-local
地址和 decoupled request/response 合同；差异是每个有效 bank response 返回 96 个 INT8
weight，并更新 96 个 Acc32。20,000 个 tile、334,542 个 source 的测试逐 lane 建立确定性
weight oracle，并继续覆盖随机 request stall、随机 response delay、output backpressure、
unsolicited/mismatched response、reset/recovery、满载、Acc32 回绕和精确 `last`。

| P | issue beats | issue speedup | cmd→valid cycles | cmd→valid speedup | cmd→fire cycles | cmd→fire speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 334,542 | 1.000x | 1,407,188 | 1.000x | 1,432,699 | 1.000x |
| 2 | 189,430 | 1.766x | 802,335 | 1.754x | 827,939 | 1.730x |
| 4 | 113,298 | 2.953x | 484,966 | 2.902x | 510,529 | 2.806x |
| 8 | 72,005 | 4.646x | 313,652 | 4.486x | 339,200 | 4.224x |

周期与 16-lane 相同是预期行为：一个 transaction 的 source issue 次数由 bitmap 和 bank
冲突决定，而 96 个输出 lane 在同一 response 内并行。它证明控制器没有依赖 16-lane
小配置，但仍不是 checkpoint weight 或 network-output oracle。

## DC、Formality 与 PrimeTime

DC 使用 TSMC 28HPC+ NLDM slow `ssg0p9v125c`、3.0 ns clock。weight SRAM 位于显式外部
接口，面积仍是 premacro logic-only：

| P | DC path | DC setup slack | DC hold slack | logic levels | 96-lane area (um²) |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.85 ns | +0.95 ns | +0.0095 ns | 51 | 22,429.39 |
| 2 | 1.75 ns | +0.78 ns | +0.0100 ns | 51 | 24,288.26 |
| 4 | 1.91 ns | +0.62 ns | +0.0100 ns | 44 | 28,206.23 |
| 8 | 1.82 ns | +0.71 ns | +0.0094 ns | 43 | 37,328.13 |

四档均为 `macro/black-box=0`、timing `violating paths=0`，DC postcompile `check_design` 与
`check_timing` 无 warning/error。`constraint_violators.rpt` 只报告 timing/design-rule；库中
隐含的 `max_leakage_power=0` 是优化 sentinel，不再冒充项目功耗约束，真实 leakage 留待
SAIF/PTPX 与 SRAM macro 分账。DC 以 fast min-version library 做 hold 优化：100 ps 是
mapper guard，90 ps 是 DC 冻结网表报告 guard，PrimeTime 最终仍以 50 ps paper contract
独立检查。旧 16-lane 数据没有使用这组 hold guard，因此本版不再给出不等约束下的
`area/16-lane` 比值。相对上一版只做 50 ps 边界修复的面积增加，是为 fast-corner 留出
约 50 ps 可审计余量，而不是功能逻辑增加。

Formality 为：

| P | passing compare points | failing | unmatched |
|---:|---:|---:|---:|
| 1 | 6,512 | 0 | 0 |
| 2 | 6,528 | 0 | 0 |
| 4 | 6,558 | 0 | 0 |
| 8 | 6,614 | 0 | 0 |

PrimeTime W-2024.09-SP3 读取 DC mapped netlist/SDC；setup 使用 slow
`ssg0p9v125c`，hold 另用 fast `ffg1p05vm40c`，均为无 SPEF 的 prelayout 口径：

| P | slow setup slack | fast hold slack | setup violations | hold violations |
|---:|---:|---:|---:|---:|
| 1 | +0.9456 ns | +0.0495 ns | 0 | 0 |
| 2 | +0.7793 ns | +0.0500 ns | 0 | 0 |
| 4 | +0.6154 ns | +0.0500 ns | 0 | 0 |
| 8 | +0.7106 ns | +0.0494 ns | 0 | 0 |

`check_timing` 明确执行 unconstrained-endpoint 检查并成功。slow-corner analysis coverage 中 43% 的
library checks 是互补时钟/条件检查被 constant propagation 禁用，不是数据 endpoint 未约束。
这些数字没有 SPEF、SRAM macro 或布线拥塞，因此只作 prelayout closure，不作 signoff。

## Remap 负结果

| P | line | best global | global speedup | per-operator speedup | survives 3% gate |
|---:|---|---|---:|---:|---|
| 4 | Local | xor_shift_5 | 1.006433x | 1.017449x | NO |
| 4 | Motion hybrid | xor_shift_5 | 1.006547x | 1.015904x | NO |
| 8 | Local | xor_shift_4 | 1.006074x | 1.018489x | NO |
| 8 | Motion hybrid | xor_shift_5 | 1.006677x | 1.017754x | NO |

因此后续不实现 XOR remap。下一轮性能提升必须改变更大的执行对象，例如扩大 source issue、
跨输出/跨 tile 调度、activation/psum 驻留或 Local/Motion 与全网流水重叠，而不是继续微调
静态 bank 哈希。

## 声明边界与下一 gate

可声称：Local 96-lane 数据面上的真实 bitmap conservation/Acc32 miter、P1/P2/P4/P8 transaction
比率、premacro DC 面积/时序、RTL-to-gate Formality、无 SPEF PrimeTime STA，以及 remap
负结果。

不可声称：checkpoint INT8 weight bit-exact、真实 SRAM macro 面积/时序/能耗、name-mapped
gate SAIF/PTPX、post-layout PPA、full-network latency/energy 或 2x--3x accelerator speedup。

机器清单：`dual_line_m2c_l96_evidence_manifest_20260821.json`，绑定 26 个源码与 189 个
证据文件，状态
`PASS_M2C_LOCAL_L96_REAL_BITMAP_VCS_DC_FM_PTSTA_PREMACRO`。当前 banked engine 只实现
Local `command_current_bits`；Motion 仍是 selector/profile 与负 DSE，不把输入 H67 bitmap
偷换成 Motion RTL 完成。
下一 gate 是 checkpoint/grouped-Conv weight oracle、ordered full-network cycle 映射，以及
门级 SAIF/PTPX；只有系统模型证明扩大 P 或跨 tile 调度能显著推进 2x 目标后才新增大宽度 RTL。
