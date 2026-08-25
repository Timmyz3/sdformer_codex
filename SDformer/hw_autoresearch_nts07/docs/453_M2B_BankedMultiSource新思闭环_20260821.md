# M2B Banked Multi-Source 新思闭环（2026-08-21）

## 结论

M2B 把 Local 的单 source/cycle、片内 flop weight table 和 256:1 组合 weight mux，替换为：

1. 256-bit bitmap 先做 8 个 32-bit word 的分层前沿，并让每个 bank 独立推进 word；
2. source 按 `source_index mod P` 静态映射到 P 个 weight bank；
3. 每拍每 bank 最多选择一个 source，输出去掉固定 bank 位后的局部 SRAM 地址；
4. 外部同步 weight bank 通过 decoupled request/response 返回 `P × 16 × INT8`；
5. 每 lane 用最多 P 项 adder tree 更新 Acc32。

这使性能改进来自真实 multi-source issue，而不是重复计算 activation sparsity。p4 是当前稳健
候选；p8 的核内吞吐更高，但要付出 8-bank SRAM 端口和更宽的数据返回，暂只保留为上界。

## 真实 bitmap VCS/SVA

向量由冻结 H67 ep35、Local ep44 的 admitted NPZ 均匀确定性抽取，每线 10,000 个，共
20,000 个 256-bit tile。H67 popcount mean/P50/P95 为 `15.19/6/59`，Local 为
`18.26/9/67`。VCS 使用确定性 INT8 weight function 对每 tile、16 lanes 建立独立 Acc32
reference；它不是 checkpoint weight/network-output oracle。

四档都在 Synopsys VCS 2023.12 下通过随机 request stall、随机 response delay、output
backpressure、bank contract SVA，以及 unsolicited/mismatched response 两类 fail-closed 注入。
此外定向覆盖事务中 reset、故障后 reset 恢复、256-source 满载、Acc32 二补码回绕和精确
`last` oracle：

| P | issue beats | p1/beat | ideal-ready full cycles | p1/ideal | cmd→valid cycles | p1/latency | cmd→fire full cycles | p1/full |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 334,542 | 1.000x | 368,950 | 1.000x | 1,407,188 | 1.000x | 1,432,699 | 1.000x |
| 2 | 189,430 | 1.766x | 223,838 | 1.648x | 802,335 | 1.754x | 827,939 | 1.730x |
| 4 | 113,298 | 2.953x | 147,706 | 2.498x | 484,966 | 2.902x | 510,529 | 2.806x |
| 8 | 72,005 | 4.646x | 106,413 | 3.467x | 313,652 | 4.486x | 339,200 | 4.224x |

issue-beat 比率是同一组真实 bitmap 的精确 bank-conflict 结果；P2/P4/P8 bank utilization 分别为
`88.30%/73.82%/58.08%`。`cmd→valid` 包含 request stall 与随机 memory response delay，但在
output 出现时停止；`cmd→fire full` 才额外包含 output backpressure。三种周期都不含真实 SRAM
macro timing 或全网其他算子，只是 engine transaction 口径，不作为系统 latency。

## DC 与 Formality

四档均使用 TSMC 28HPC+ NLDM slow `ssg0p9v125c`、3.0 ns clock、0.2 ns setup uncertainty、
0.25 ns I/O delay。weight SRAM 位于明确的外部接口，因此表中只有 frontier/control/adder/Acc32
逻辑，不含 weight SRAM macro：

| P | critical path | slack | logic levels | cell area (um²) | area vs p1 | logic-only issue/mm² vs p1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.67 ns | +1.10 ns | 53 | 4,428.27 | +0.0% | 1.000x |
| 2 | 1.52 ns | +1.00 ns | 44 | 4,736.97 | +7.0% | 1.651x |
| 4 | 2.02 ns | +0.50 ns | 41 | 5,648.83 | +27.6% | 2.315x |
| 8 | 1.68 ns | +0.85 ns | 40 | 7,191.45 | +62.4% | 2.861x |

paper PPA wrapper 已完全移除 debug performance counter，而不是只依赖综合器常量传播；bank
接口也只保留有效的局部地址位。因此四档 `check_design_postcompile` 与
`check_timing_postcompile` 都没有 warning/error。P4 critical path 位于 response/adder 到
Acc32，仍满足 3 ns 目标；表中面积不含 weight SRAM macro。

Formality 结果：

| P | passing compare points | failing | unmatched |
|---:|---:|---:|---:|
| 1 | 1,392 | 0 | 0 |
| 2 | 1,408 | 0 | 0 |
| 4 | 1,438 | 0 | 0 |
| 8 | 1,494 | 0 | 0 |

四档 `check_timing_postcompile` 均检查 unconstrained endpoints 且无 warning；修正后的 DC Tcl
不再调用该版本不支持的 `report_timing -unconstrained`。

## 首轮打铁评审与修订

首轮评审为 `58/100，CONDITIONAL`。其中三个 P0/P1 问题已关闭：

1. 共享 selected-word 屏障改成 per-bank 独立前沿，P4 bank utilization 从 `52.82%` 提高到
   `73.82%`；
2. VCS 权重函数改成 source/lane 非短周期函数，并增加 active-source、duplicate-source、最终
   bitmap 三重 scoreboard；
3. paper PPA wrapper 删除 performance counter，bank request 改成无冗余的局部 SRAM 地址，
   重跑 VCS/DC/Formality 后获得本文件最终数字。

尚未关闭的是 96-lane 数据面、checkpoint INT8 weight oracle、真实 SRAM macro/能效和 ordered
full-network cycle 映射；这些进入 M2C，不在 M2B 中偷换成完成项。

修订版复评为 `64/100，CONDITIONAL`：correctness `22/25`、innovation `13/20`、performance
evidence `17/25`、DATE completeness `12/30`。复评指出旧 `randomized cycles` 没有包含
output backpressure；上表现已拆成三种互斥口径并由 VCS 直接计数。复评要求的 unsolicited
response、reset/recovery、满载、wraparound 和 `last` 定向验证也已补齐。它仍然否决把上述
倍率称作 accelerator speedup，因为 P1 是同一新 engine 的单发射配置而非冻结全系统基线。

## 声明边界与下一 gate

可声称：相同真实 bitmap、相同确定性 weight miter 下 p1/p2/p4/p8 的 bank-conflict issue
比率；bank-local SRAM 地址接口 RTL；VCS/SVA；3 ns premacro logic DC；RTL-to-gate Formality。

不可声称：checkpoint-weight bit-exact、真实 SRAM 端口/面积/能耗、全层或全系统 2x--3x、
PTPX energy、DRAM traffic 或 p8 已优于 p4。下一 gate 是连接目标 SRAM macro/DB、生成门级
name-mapped SAIF/PTPX，并把 p4/p8 映射回 ordered full-network trace。

机器可读证据清单：
`dual_line_m2b_evidence_manifest_20260821.json`，状态
`PASS_M2B_REAL_BITMAP_VCS_DC_FM_PREMACRO`；清单绑定 14 个源码和 79 个证据文件，并对
`date_dual_core.sdc`、3.0 ns、`ssg0p9v125c`、标准单元库 SHA、macro/black-box=0、
`check_design`、`check_timing` 和 Formality 状态执行 fail-close。
