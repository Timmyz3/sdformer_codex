# H67/H68 RTL 独立审阅与 DC 差距

> 2026-07-13更新：本文记录修复前findings。F01--F04、SDC、生产深度测试、SVA和映射网表回灌
> 已完成修复；F05完整加速器边界、F06 SRAM宏、正式工艺F07/F08和F10 Formality仍是有效缺口。
> 最新证据和逐项边界见`docs/49_H67H68逐位验证_占用类Shiftmax与DC交付结果.md`。

> 审阅日期：2026-07-13  
> 审阅角色：独立 RTL、验证与综合审阅员  
> 审阅标准：`rtl-design`、`functional-verification`、`logic-synthesis`、
> `erie-verilog-generator` 的 ASIC 检查要求  
> 审阅范围：`rtl_ttx/`、`rtl_h67/`、`tb_ttx/`、`tb_h67/`、`sim_ttx/`、
> `sim_h67/`，并只读核对 H67/H68 软件部署公式与配置  
> 修改边界：本轮未修改任何 RTL、testbench 或仿真/综合脚本，仅新增本报告。

## 1. Findings：按严重度排序

### F01｜阻断（P0）｜RTL gate 的定点编码与软件部署配置不一致

**位置**：

- 软件 `bsa_attention.py:173-183, 203-210, 2536-2544`
- H68 deploy 配置 `h68_*_dyadic_int8_deploy.yml:163-168`
- `rtl_ttx/ttx_row_engine.sv:164-174`
- `rtl_h67/h67_score_class_row_engine.sv:172-182`
- `tb_ttx/tb_ttx_row_engine.sv:218-224`
- `tb_h67/tb_h67_score_class_row_engine.sv:207-211`

软件在 `preserve_mean` 后按步长 `1/128` 做 gate 量化，配置允许范围为 `0..2`。这对应至少
9-bit 无符号 Q1.7，数值 `1.0` 的整数编码应为 `128`，`2.0` 应为 `256`。两个 row engine
却使用 8-bit `GATE_W=8`，按 `exp * 255 * n_tokens` 计算并饱和到 `255`。这更像 Q0.8/
归一化到 255 的编码，不是软件声明的 Q1.7。

确定性反例：

| 等分 row | 软件 Q1.7 gate | 当前 RTL gate |
|---|---:|---:|
| 8 个等分 token，`preserve_mean=1` | 128 | 255 |
| 162 个等分 token，`preserve_mean=1` | 81 | 161 |

现有 testbench 也复制了 RTL 的 `*255` 公式，因此 RTL 与 testbench 同时错误时仍会 PASS。这是
**语义阻断项**，在固定 gate 数制、位宽、饱和和舍入规则之前，当前输出不能接入软件权重或用于
DC 功耗/精度联合结论。

### F02｜阻断（P0）｜RTL Shiftmax LUT 尚未被 H67/H68 部署实验覆盖

**位置**：

- 软件 `bsa_attention.py:173-183`
- `rtl_ttx/ttx_exp2_lut_q8.sv:25-52`
- `rtl_ttx/ttx_row_engine.sv:124-169`
- `rtl_h67/h67_score_class_row_engine.sv:146-177`

软件 valid825 仍用浮点 `2**shifted`，只对输入 score 和最终 gate 做 `1/128` 量化。RTL 则先把
指数降成 16-entry fraction LUT 和 Q8，再在量化后的整数和上取 `ceil(log2)`。例如 Q7
`delta=-1` 时，软件指数约为 `0.9946`，RTL 因向上取到 `1/16` 桶而输出 `245/256=0.9570`。
量化误差还可能使 row sum 跨越 2 的幂边界，造成整个 row 的分母变化一档。

H67 dyadic valid825 的 AEE `1.4626` 和 H68 dyadic valid825 的 AEE `1.4715` 只证明了软件
score/gate 网格，不证明该 LUT、整数 row sum 和 RTL gate。当前没有真实 row 的逐项
`software hardware-model <-> RTL` 差分，因此不能声明 Shiftmax bit-exact 或精度已闭环。

### F03｜阻断（P0）｜`center -> quant` 与 RTL `raw quant` 的顺序不等价

**位置**：

- 软件 `bsa_attention.py:2536-2540`
- H67/H68 deploy 配置：`center_scores: true`
- `rtl_h67/h67_motionxor_score_q7.sv:51-63`
- `rtl_ttx/ttx_tx_score_q7.sv:44-47`

软件先按 row 减均值，再做 RNE score 量化；RTL 直接生成未中心化的整数 score，并依靠
Shiftmax 的平移不变性省略中心化。连续数学上平移不变，但 RNE 量化不具备该性质。已存在明确
反例：162-token row 中原始 Q7 类 `0/1` 各 81 个，软件 `center -> RNE` 得到 `0/0`，RTL
保留 `0/1`。

H67/H68 已完成的 dyadic valid825 仍保持 `center_scores=true`，所以没有关闭这个差异。必须先
冻结 hardware-order 软件参考顺序，再谈 RTL 一致性。

### F04｜阻断（P0）｜H68 只有叶级参数化意图，没有可综合、可回归的 H68 top

**位置**：

- `rtl_h67/h67_motionxor_score_q7.sv:3-7,45-48`
- `rtl_h67/h67_score_class_row_engine.sv:3-15,125-130`
- `rtl_h67/h67_attention_top.sv:3-8,79-85`
- `sim_h67/run_iverilog.sh:9-17`
- `sim_h67/run_yosys.sh:10-24`
- H68 deploy 配置 `:118-168`

当前叶模块新增了 `ENABLE_MOTION_XOR=0`，可表达 H68 部署期的无 Motion-XOR TTX。独立审阅
额外用 Icarus 参数覆盖运行了 score 和 row test，二者均 PASS。但是：

1. `h67_attention_top` 没有 `ENABLE_MOTION_XOR` 参数，也没有传给 row engine，top 永远综合成 H67；
2. 正式 `run_iverilog.sh` 只运行默认值 1，H68 模式没有进入回归；
3. `run_yosys.sh` 没有对参数 0 elaboration，也没有 H68 top/netlist；
4. 没有 `rtl_h68/`、H68 filelist 或明确的共享 top 模式合同。

旧 `rtl_ttx` 也不能直接充当 H68 frozen top：`ttx_tx_score_q7.sv:7` 默认
`ALPHA0_Q8=5`，而 H68 deploy 配置 `alpha0=1/64` 对应 Q8 整数 4；`ttx_attention_top` 又没有
覆盖该参数。故当前仓库不存在与 H68 deploy 配置一一对应的 top-level elaboration。

需要强调：H68 的 training-only matrix auxiliary 在 eval 中权重为 0，不应把训练分支做进芯片；
但必须提供“共享核心、Motion-XOR 关闭、alpha0=1/64、RNE”的正式部署 top 和等价性证据。

### F05｜阻断（P0）｜当前 `attention_top` 只是 row-score 子系统，不是完整加速器 top

**位置**：

- `rtl_ttx/ttx_attention_top.sv:17-43,77-110`
- `rtl_h67/h67_attention_top.sv:17-45,79-114`
- `rtl_ttx/ttx_late_gate_accum.sv:3-33`
- `rtl_ttx/filelist.f:5-8`
- `rtl_h67/filelist.f:1-7`

两个 top 只调度 descriptor、接收 Q/K、输出 `K bits + gate + threshold`。TTX filelist 虽包含
`ttx_late_gate_accum`，但 top 未实例化；H67 filelist 甚至未包含它。权重 SRAM、投影累加、输出
定点缩放、ATLIF 状态存储/更新、残差接口、跨层数据搬运、DMA/片上网络和配置寄存器均不在当前
层次中。

因此当前 RTL 可以作为“attention row engine 子块”送 DC 做早期探索，但不能把该面积、功耗、
吞吐称作 H67/H68 全加速器或全 encoder PPA。论文硬件结果若只综合当前 top，会系统性漏算主要
存储、互连和 ATLIF/投影开销。

### F06｜高（P0）｜存储编码不具备 SRAM macro 就绪性

**位置**：

- `rtl_ttx/ttx_row_engine.sv:55-75,124,157,184-185,211-213,236-238,250-255`
- `rtl_h67/h67_score_class_row_engine.sv:60-82,146,165,189-190,249-256,270-276,309-318`
- `sim_ttx/run_yosys.sh:18-25`
- `sim_h67/run_yosys.sh:17-24`

生产参数下每个 row engine 声明三组 256-depth active arrays 和一组 histogram，总逻辑存储约：

| 模式 | active arrays | histogram | 合计 |
|---|---:|---:|---:|
| TTX/H67 默认 top | 14,336 bit | 512 bit | 14,848 bit |
| H68 参数 0 叶级理论值 | 14,336 bit | 32 bit | 14,368 bit |

当前读端口是组合异步读，控制器按零读延迟工作。TTX 还在 reset 和每个 `cfg_start` 用 for-loop
单周期清 64 个 histogram word，无法映射普通单/双口 SRAM。H67 的顺序初始化更友好，但 load
阶段的 histogram read-modify-write 仍假设单周期组合读。

独立重跑当前源码的 Yosys 结果：

| top | memories | `$dffe` word cells | mux cells |
|---|---:|---:|---:|
| `ttx_attention_top` | 0 | 832 | 1,157 |
| `h67_attention_top` | 0 | 832 | 1,113 |

日志明确把三个 256-depth array 映射为每个 255:1 读 mux，并把 histogram 映射为 64 组寄存器/
写 mux。必须先确定 SRAM macro 的同步读延迟、端口数和 read-during-write 语义，再重排 FSM；否则
DC 的标准单元面积既不代表最终 SRAM 方案，也可能产生不可接受的 mux 关键路径。

### F07｜高（P0）｜没有 SDC、工艺库、DC 脚本和可用的时序/面积/功耗报告

**位置**：`sim_ttx/run_yosys.sh:18-20`、`sim_h67/run_yosys.sh:17-20`；仓库内未找到 `.sdc`、
DC Tcl、Liberty/operating-condition 绑定或约束 QA 报告。

现有 Yosys 流程只有 `proc; opt; memory; opt; check; stat`，没有目标时钟、IO delay、clock
uncertainty、max fanout、max transition、工艺角、wire load/RC、ABC technology mapping，也没有
`report_timing/report_area/report_power`。它生成的是通用结构网表，不是 DC-ready 约束包。

按 `logic-synthesis` 标准，目前缺少：

- `create_clock` 与目标频率；
- 所有输入/输出 delay 和 drive/load；
- uncertainty、fanout、transition；
- PVT/operating condition；
- area/power budget；
- unconstrained path 审计；
- worst-corner WNS/TNS、面积和功耗报告。

### F08｜高（P0）｜关键数据通路未流水，频率可实现性未知

**位置**：

- `rtl_ttx/ttx_tx_score_q7.sv:28-47`
- `rtl_h67/h67_motionxor_score_q7.sv:35-63`
- `rtl_ttx/ttx_exp2_lut_q8.sv:25-52`
- `rtl_ttx/ttx_ceil_log2_u32.sv:10-17`
- 两个 row engine 的 gate 组合逻辑

单周期路径可能包含 32-lane 多路 popcount、H67 XOR、score 比较、LUT/优先编码、两个乘法、
64-bit 可变右移和饱和。Yosys 对 H67 score 统计出 128 个 `$add`、32 个 `$xor`；gate 路径还
保留变量乘法和 shift。当前无 pipeline valid、无 target frequency、无关键路径报告。

功能上 row engine 又严格串行执行 `LOAD -> SUM_ACTIVE -> SUM_FOLD -> EMIT`，没有 row 间
双缓冲。最坏 dense row 约需 `N + N + N` 个数据周期，完整 6,720 rows/frame 的吞吐尚未在目标
频率、SRAM 延迟和外部 backpressure 下验证。需要先做 compile-explore，不能从“Yosys check=0”
推断时序可收敛。

### F09｜高（P1）｜验证环境未达到 ASIC coverage closure 标准

**位置**：

- `tb_ttx/tb_ttx_row_engine.sv:295-359`
- `tb_ttx/tb_ttx_scheduler.sv:41-101`
- `tb_h67/tb_h67_motionxor_score.sv:85-132`
- `tb_h67/tb_h67_score_class_row_engine.sv:216-335`
- `sim_ttx/run_verilator_lint.sh:7-14`
- `sim_h67/run_verilator_lint.sh:6-8`

当前优点是已有自检 testbench、row backpressure、fold on/off、score 定向和 scheduler row count。
但距离 `functional-verification` 签核仍有以下缺口：

- 无 V-plan、requirement-to-test 映射和接口 agent；
- 无 bind SVA，未证明 valid/ready 稳定、计数边界、deadlock freedom、histogram 生命周期；
- 无 top-level 功能仿真；
- H68 参数 0 不在正式 regression；
- score 随机只跑默认的 1,000 次单一模拟种子，row 只使用固定 8-token 向量；
- 未测试生产 `MAX_TOKENS=162`、cfg 边界、提前/缺失 `in_last`、运行中 reset、busy 时 start、
  长时间输入/输出 backpressure、全零 row 和满 active row；
- TB 无 watchdog，DUT deadlock 时回归可永久挂起；
- Icarus 明确提示忽略 `unique case` 语义；
- 无 functional/line/branch/toggle/FSM/assertion coverage 报告和 waiver；
- row scoreboard 复制 RTL LUT 与 `*255`，不是从冻结软件模型生成期望值。

本轮独立重跑 Icarus：TTX 两项、H67 默认两项以及额外 H68 参数 0 两项均 PASS。这只能记为
smoke regression 通过，不能记为 verification sign-off。

### F10｜高（P1）｜没有 LEC，且当前参考模型不足以承担 golden 角色

仓库内未找到 Formality/Yosys equivalence 脚本、mapping/compare point 报告或 RTL-to-gate LEC
结果。`write_verilog` 后只检查文件非空，没有把生成网表重新仿真，也没有证明 memory mapping、
参数 elaboration 和优化后网表等价。

正式 LEC 的 golden 也尚未冻结，因为 F01-F04 的软件/数制/模式合同仍在变化。正确顺序应是：
先冻结 executable fixed-point spec，再冻结 lint-clean RTL，之后才做 RTL-vs-DC-netlist 100%
equivalent；不能对当前错误数制做 LEC 后就宣称算法等价。

### F11｜中（P1）｜CDC 无已知内部跨域，但 CDC/RDC 仍不可签核

**位置**：两个 top 的 `clk/rst_n` 及全部外部控制/stream 端口；row/scheduler 的
`always_ff @(posedge clk)`。

当前 RTL 只有一个显式 clock，未发现内部 CDC，也没有 raw gated clock。`rst_n` 实现为同步、
低有效 reset，控制寄存器均有 reset；active memories 未 reset，但由有效计数保护，H67 histogram
在首 row 前顺序清理。

缺口是接口时钟归属和 reset 合同没有文档/约束：`start_frame`、cfg、row request、输入输出 stream
是否同域未说明，也无同步器/异步 FIFO。没有运行中 reset 测试、reset deassertion assertion、RDC
分析或 CDC waiver。TTX histogram 的同步 reset/单周期全清还构成大扇出和 SRAM 映射障碍。

结论应写成“单时钟设计，暂未发现结构 CDC”，不能写“CDC/RDC clean”。

### F12｜中（P1）｜稀疏输出协议缺少系统级零填充合同

**位置**：

- `rtl_ttx/ttx_row_engine.sv:177-185,249-257,301-311`
- `rtl_h67/h67_score_class_row_engine.sv:140-144,187-190,270-278,324-334`

K-zero token 被折叠后不会从 output stream 发出；`out_last` 表示最后一个 active entry，不是原始
row 的最后 token。数学上 `K=0` 的 gated-K 输出为零，省略是正确的，但系统必须保证目标输出
buffer 已清零，并按 `out_token_idx` scatter 写回；全零 row 则完全没有 output beat。

当前 top 没有输出 buffer/clear/scatter 实现，也没有接口标志说明稀疏语义。testbench 只把较少的
输出数量视为正确，没有验证最终 dense tensor。该合同未关闭前，不能把 row engine 直接接到普通
顺序 stream consumer。

### F13｜中（P1）｜H67 时间对数据搬运和容量开销未进入 RTL/PPA

**位置**：

- `rtl_h67/h67_temporal_pair_adapter.sv:6-18`
- `rtl_h67/h67_attention_top.sv:25-30`

当前 temporal-pair adapter 只是 64-bit `{K1,K0}` mux；真正的 K-pair SRAM、同空间地址生成、
T=2 布局转换、bank 冲突、带宽仲裁和 prefetch 均由外部环境承担。若直接综合 H67 top，报告会
计入两个 32-bit mux，却漏掉生成 64-bit pair 所需的数据搬运成本。

此外 H68 motion-off 模式理论上不需要 peer K；若复用相同 64-bit 接口，会高估输入带宽或引入
不必要切换。需要在共享微架构中定义 H67/H68 的 mode-dependent SRAM read policy。

### F14｜中（P1）｜参数接口宣称通用，但数学只对冻结参数成立

**位置**：

- `rtl_h67/h67_motionxor_score_q7.sv:3-7,17-21,51-59`
- `rtl_h67/h67_attention_top.sv:3-8`
- `rtl_ttx/ttx_tx_score_q7.sv:3-8,18,46`
- `rtl_ttx/ttx_exp2_lut_q8.sv:3-5,29-31`

H67 silence 的 `>>4`、motion 的整数 Q7 权重和 35 类范围只对 `HEAD_DIM=32`、alpha0=`1/64`、
motion=`1/4`、score step=`1/128` 成立，但模块仍暴露任意 `HEAD_DIM`，当前也没有 elaboration
guard。TTX normalization 用 `$clog2(HEAD_DIM)`，只对 2 的幂 head_dim 等价于除法；exp LUT 的
part-select 又隐含 `SCORE_FRAC>=5`。

应选择其一：把模块彻底冻结为常量接口，或增加 elaboration-time parameter legality check 和
合法参数回归。现在改变参数可能“可编译但算错”。

### F15｜中（P1）｜功耗、clock gating 和真实 activity 证据为空

当前没有 VCD/SAIF、真实 H67/H68 row replay、clock-gating coverage、ICG/test-enable、UPF 或
memory power model。RTL 依赖寄存器 enable，但没有明确 ICG 边界；按 `rtl-design` 标准，高/中
gating opportunity 的宽存储和 datapath 需要活动率分类及可测 ICG 方案。

现有 `spike_energy_proxy` 明确排除 attention 控制、归约和 memory，不能用作 RTL 总功耗。DC
前至少需要真实 valid825 profile 驱动的 SAIF，以及 H67 motion on / H68 motion off / fold off
三种 activity case。

### F16｜低（P2）｜lint 流程会放过 warning，Erie 规则缺少正式 waiver

**位置**：`sim_ttx/run_verilator_lint.sh:7-14`、`sim_h67/run_verilator_lint.sh:6-8`。

Verilator RTL top 在本轮严格重跑中无输出 warning，Yosys `check` 为 0 problem，这是正面证据。
但正式脚本使用 `-Wno-fatal`，TTX testbench 还批量关闭 width/unused/blocking 等告警；H67 lint
只检查 top，不检查 testbench。

Erie 静态检查对 parameter-bound for-loop 报 `MUST_LOOP_FOR_CONST_BOUNDS`，并对无显式宽度的
loop literal 报 warning。这里的 `HEAD_DIM`/memory depth 是 elaboration 常量，属于 Erie 正则
检查的保守误报，不是已证实的综合错误；但应写入 `lint_waivers.csv`，注明 rule、文件、理由和
批准人，而不是静默忽略。当前也没有统一 lint report/waiver 归档。

## 2. 当前可确认通过的证据

审阅基线工具：Verilator `5.020`、Icarus `12.0`、Yosys `0.33`。

| 检查 | 结果 | 结论边界 |
|---|---|---|
| Verilator `-Wall` lint 两个 RTL top | 无 warning 输出 | 仅语法/静态结构，不含 CDC/SDC |
| Icarus TTX row + scheduler | PASS | 固定小向量 smoke |
| Icarus H67 score + row | PASS | 默认 motion=1 smoke |
| 独立参数覆盖 H68 score + row | PASS | 叶级 motion=0；正式脚本未覆盖 |
| Yosys hierarchy/check 当前源码 | 0 problem | 通用逻辑网表，不是工艺映射 |
| H67 score Python 穷举/随机参考 | 中心化前 score 通过 | 不覆盖 Shiftmax/gate/整 row |

并发审阅期间其他 agent 更新了 H67 score/row/TB，移除了旧 `initial` 参数检查并新增
`ENABLE_MOTION_XOR`。本报告最终基于以下主文件哈希，不回退任何并发修改：

| 文件 | SHA-256 前 12 位 |
|---|---|
| `rtl_ttx/ttx_attention_top.sv` | `315ef595b534` |
| `rtl_ttx/ttx_row_engine.sv` | `c985f98e2368` |
| `rtl_h67/h67_attention_top.sv` | `a55afba59e63` |
| `rtl_h67/h67_score_class_row_engine.sv` | `5e6523b46fd6` |
| `rtl_h67/h67_motionxor_score_q7.sv` | `a867837e93dd` |

## 3. CDC、reset、memory、SDC、LEC、coverage 签核矩阵

| 检查域 | 当前状态 | ASIC 门槛 | 判定 |
|---|---|---|---|
| 软件定点等价 | score 前端部分通过 | 完整 row bit/cycle-accurate | **FAIL** |
| H68 top | 叶级参数 0 可编译 | 正式 top/filelist/regression | **FAIL** |
| RTL lint | top clean，Erie 有待 waiver 项 | 全文件 0 error、warning 审阅归档 | 部分通过 |
| CDC | 单 clock，未见内部 crossing | 时钟归属、约束、CDC report | 未签核 |
| RDC/reset | 同步低有效 reset | reset spec、运行中 reset、RDC report | 未签核 |
| memory | 14.8 Kbit 逻辑数组 | SRAM macro/端口/延迟合同 | **FAIL** |
| SDC | 无 | 全 clock/IO/PVT/DRC 约束 | **FAIL** |
| DC compile | 无 | WNS/TNS/area/power/no unmapped | **FAIL** |
| LEC | 无 | RTL-netlist 100% equivalent | **FAIL** |
| SVA/formal | 无 | P0 协议和安全属性 proven | **FAIL** |
| functional coverage | 无报告 | functional 100%、line 95%、branch 90% | **FAIL** |
| toggle/activity | 无 VCD/SAIF | 真实 workload activity | **FAIL** |
| DFT/UPF | 范围未定义 | 论文口径至少声明排除/假设 | 未定义 |

## 4. 修复顺序

### 阶段 A：先冻结可执行语义，禁止先跑正式 DC

1. 冻结统一 fixed-point contract：score/gate 位宽、Q 格式、RNE、饱和、center 顺序、exp LUT、
   denominator 和 preserve-mean。
2. 在软件中实现 RTL-exact Shiftmax hardware model，对 H67 epoch19、H68 epoch19 跑 valid825。
3. 用真实 row dump 做逐 row 差分，至少比对 score、row max、exp、row sum、denominator shift、
   gate 和稀疏输出 tensor。
4. 明确 H67/H68 模式：H67 `motion=1`，H68 `motion=0`；H68 不实现 training-only matrix branch。

**退出条件**：真实 valid825 上无未解释 RTL 差分，H67/H68 硬件顺序精度满足论文门槛。

### 阶段 B：形成真正可综合的共享 top

1. 将 mode 参数贯通 top、row engine、score 和 regression，生成独立 H67/H68 elaboration manifest。
2. 冻结 `HEAD_DIM=32` 等非法参数防护。
3. 定义稀疏 scatter/zero-fill、row completion 和异常协议。
4. 加入 K-pair buffer/address generator；H68 模式关闭 peer read 和 XOR toggling。
5. 明确本次 DC 是 row engine、attention tile 还是含 projection/ATLIF 的 accelerator top，禁止混用口径。

**退出条件**：两个模式都有独立 top-level smoke、netlist 和接口文档。

### 阶段 C：SRAM 化和时序化

1. 选定 SRAM macro 或可替换 wrapper，固定同步读延迟、端口和 read-during-write 语义。
2. 合并/重排 active arrays，修改 FSM 适配 SRAM latency；移除 TTX 单周期全 histogram 清零。
3. 对 popcount、exp/denominator、gate multiply/shift 加入必要 pipeline，并保持 ready/valid 对齐。
4. 建立 cycle model，报告每 row/frame cycles、带宽和 buffer stall。

**退出条件**：综合层次保留目标 memory macro，不再出现 832 个 word DFFE 和 255:1 读 mux。

### 阶段 D：验证闭环

1. 建 V-plan，添加独立 Python/C++ fixed-point reference 和 transaction scoreboard。
2. 增加 top test、H68 mode、生产 162-token、reset/backpressure/error/corner tests。
3. 添加 bind SVA：valid 稳定、索引范围、计数不溢出、hist 清理、请求最终完成、mode 不变。
4. 至少 10 seeds constrained-random；关闭 functional/code/FSM/assertion coverage。
5. 对 SRAM wrapper 和稀疏输出做 formal assist。

**退出条件**：P0 regression 100% PASS，coverage 达门槛，所有 waiver 有依据。

### 阶段 E：DC 与 LEC

1. 建立工艺库、PVT、top 和完整 SDC；先做 constraint QA，确保 0 unconstrained paths。
2. 对 H67、H68 和 fold-off baseline 使用完全相同约束做 compile-explore。
3. 关闭 WNS/TNS、max transition/fanout/capacitance，输出 area 和层次化关键路径。
4. 用真实 row replay 生成 SAIF，报告 dynamic/leakage 和 SRAM power。
5. RTL-vs-DC netlist LEC 100% equivalent；必要时再做 gate-level smoke。

**退出条件**：WNS>=0、TNS=0、无 unmapped/blackbox、LEC PASS，PPA 报告口径完整。

## 5. 最终判定

### 5.1 能否现在直接做 DC？

**可以把当前文件喂给 DC 做语法和早期结构探索，但不能把结果视为 DC-ready 或论文 PPA。**

原因不是单一脚本缺失，而是三个前置合同尚未冻结：

1. gate/Shiftmax/center 与软件不一致；
2. H68 没有正式 top elaboration；
3. memory 和完整 accelerator 边界尚未实现。

### 5.2 H67/H68 当前硬件关系

- H67 仍是当前精度更好的部署候选：dyadic valid825 AEE `1.4626`。
- H68 dyadic valid825 AEE `1.4715`，训练辅助在部署时删除，硬件本质是 Motion-XOR 关闭的 TTX。
- 两者适合做共享核心的 mode-on/mode-off 差分，而不是维护两套完全独立 datapath。
- 在 RTL-exact Shiftmax valid825 和同约束 DC 差分前，不能用现有结果决定最终论文硬件主线。

### 5.3 独立审阅结论

当前状态应标记为：

> **算法候选结果成立；score 叶级 RTL 有价值；row-engine smoke 可回归；但完整软件等价、H68 top、
> SRAM、SDC、coverage、LEC 和论文级 PPA 均未签核。**

最高优先级不是继续扩展新模块，而是依次关闭 F01-F04，再进行 SRAM/流水重构和正式 DC。
