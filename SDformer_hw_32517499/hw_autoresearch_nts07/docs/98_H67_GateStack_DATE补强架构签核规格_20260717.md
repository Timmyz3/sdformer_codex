# H67 GateStack DATE 补强架构签核规格

**日期**：2026-07-17  
**主线**：H67 GateStack  
**阶段**：Architecture  
**状态**：**架构规格条件冻结，未达到 Architecture Sign-off**

## 1. 文档目的与证据分级

本文档根据 `docs/97` 的 DATE 审稿意见，把已有 GateStack 原型收敛为可执行的架构签核合同。本文档不将模型预测写成 RTL 实测，不将 Yosys 结构写成工艺 PPA。

| 标记 | 含义 | 当前可支持内容 |
|---|---|---|
| `[prof]` | 真实软件 workload/profile 统计 | H67 profile100 的 K-zero、pair empty、term、event 和 gate-class 分布 |
| `[RTL]` | RTL 仿真、SVA、整数金参考或开放 LEC | 当前 single-context execution slice 的功能正确性 |
| `[model]` | 周期、存储、带宽或 DSE 模型 | 候选方向和趋势，不是芯片实测 |
| `[target]` | 本文档冻结的签核目标 | 后续 RTL、DC、SAIF 必须达到的门槛 |
| `[PPA]` | 目标库映射、STA、macro 和 mapped SAIF | **当前缺失** |
| `[open]` | 未冻结或未验证 | 不得进入论文结论 |

## 2. 架构阶段结论

### 2.1 论文主线

不再把 output-stationary、descriptor cache 和多格式共享后端分拆成三个“首次提出”。论文主架构冻结为：

> **GateStack：面向 H67 final-gate 等价类的、容量安全且跨 output tile 驻留的精确稀疏投影数据流。**

该核心数据流由下列不可分割的机制共同构成：

1. final Q1.7 gate 等价类作为精确合并键；
2. IPD32W 紧凑表示与 RAW41 无损越界 fallback；
3. 首次 IPD 解码同流 promotion，不复制第二解析器；
4. output-tile-stationary 外层与 head-stacked replay 内层；
5. Resident/IPD32W/RAW41 共享 TDR、multicast、product 和 AccTile；
6. payload/execution 双 tag、PLAN/COMMIT 和 last-use 分层生命周期保证精确性。

### 2.2 签核判定

| 项目 | 当前判定 |
|---|---|
| 架构问题定义 | 通过 |
| 主贡献收敛 | 条件通过 |
| 三个候选方案 | 已定义 |
| 推荐候选 | 平衡方案 C1，仅用于进入验证 |
| 性能签核 | 未通过 |
| 面积/功耗签核 | 未通过 |
| 真实 trace 签核 | 部分通过：一个样本、四stage首block首window真实RTL回放完成；100-frame/all-block主表仍缺 |
| 公平基线签核 | 部分通过：no-residency与RAW-only同顶层周期消融已完成；物理Direct/head-major/PPA仍缺 |
| full-encoder 系统闭环 | 未通过 |

## 3. 论文与实现的系统边界

### 3.1 双层边界

**A. 论文系统边界**

论文性能、能量和带宽结论必须覆盖 H67 encoder，至少包含：

- H67 Motion-XOR/SCS attention；
- GateStack projection；
- ATLIF 时间复用执行；
- S0/S1/S2 residual/skip 存储与数据搬运；
- stage 间张量和权重流量；
- 外部存储和片上 SRAM 带宽限制。

**B. 主要 RTL/PPA 测量边界**

当前允许作为目标库 PPA 主体的边界是 GateStack projection subsystem：

```text
group/payload/descriptor input
        |
        v
Output-Tile Scheduler
        |
PLAN -> Atomic COMMIT -> Dual-Tag Lifecycle
        |
Head-Slot + Descriptor Residency
        |
Resident / IPD32W / RAW41
        |
Shared TDR -> Multicast -> Product -> AccTile
        |
weight/bias/requant boundary -> final output
```

只有当 full encoder 的统一 schedule、存储和数据搬运进入可执行模型，且关键模块有 RTL/PPA 校准后，才允许使用“H67 encoder accelerator”。此前只能使用“GateStack projection subsystem”。

### 3.2 当前已实现与未实现

| 内容 | 状态 |
|---|---|
| single-context scheduler/control/three decoders/shared projection | `[RTL]` |
| payload/execution 双 tag、PLAN/COMMIT、abort | `[RTL]` |
| T162/L32/H3/6/12/24真实四stage回放 | `[RTL]`，真实Q/K/gate与checkpoint权重候选INT8码，单样本首window |
| 真实权重、bias accumulator整数闭环 | `[RTL]`，但候选量化尚未通过valid825 |
| 最终requant、BN folding、残差scale | `[open]` |
| dual context | `[model]`，未进入当前顶层 |
| full H67 attention/ATLIF/skip 集成 | `[open]` |
| target-library PPA | `[open]` |

## 4. 需求分解

### 4.1 Must-Have

| ID | 类别 | 要求 | 验收口径 |
|---|---|---|---|
| M-F01 | 功能 | 三种表示路径必须数值等价 | 真实 trace 上逐组、逐 token、逐 lane 零 mismatch |
| M-F02 | 功能 | 任何 class/capacity 越界必须 RAW41 无损 fallback | 不得截断、静默丢弃或近似剪枝 |
| M-F03 | 功能 | final-gate 等价类键冻结 | 与软件 RTL-exact 量化顺序一致 |
| M-F04 | 功能 | output-tile-stationary/head-stacked 语义不变 | 每 tile 仅 clear/bias/final 一次，所有 head 完整累加 |
| M-F05 | 功能 | 冻结舍入、饱和、负数右移、overflow 和 requant | valid825 部署精度与 bit-exact 报告 |
| M-F06 | 可用性 | 反压、非法 tag、missing slot 和 timeout 可有界退休 | 无死锁，错误 tag 可追溯 |
| M-P01 | 性能 | GateStack 相对最优公平 exact 基线周期提升至少 1.20x | 同 trace、同位宽、同 lane、同 SRAM 口径 |
| M-P02 | 性能 | full encoder 性能口径闭合 | 达成 30 FPS 目标，否则删除实时 claim |
| M-P03 | 功耗 | GateStack 子系统 EDP 至少改善 15% | 目标库、macro、mapped SAIF 同约束测量 |
| M-A01 | 面积 | 目标面积预算必须由工艺和论文边界冻结 | 绝对预算缺失时禁止架构最终签核 |
| M-W01 | 功耗 | 目标功耗预算和 PVT 必须冻结 | dynamic/leakage/clock/memory 分账 |
| M-M01 | 存储 | head slot 不得因压缩平均值而缩小 exact 容量 | 每 head 容纳 6642 bit RAW41 |
| M-M02 | 存储 | 所有容量报告同时给出 logical bits 和 rounded macro bits | 包含端口、bank、ECC/BIST 假设 |
| M-I01 | 接口 | 所有顶层流必须采用明确 ready/valid 或 SRAM 协议 | 禁止依赖未定义的零延迟存储 |
| M-V01 | 验证 | 主表使用 H67 四 stage 真实 bit trace | trace-shaped 只保留为控制回归 |
| M-V02 | 验证 | Direct、no-residency、head-major 三个基线必须存在 | 同约束 RTL/PPA 对照 |

### 4.2 Should-Have

| ID | 要求 | 验收口径 |
|---|---|---|
| S01 | dual-context build/execute 重叠 | 实测端口冲突、FIFO 占用和 overlap efficiency |
| S02 | product lane operand isolation | mapped SAIF 对照功耗 |
| S03 | empty destination 阻断 product/AccTile 宽总线 | 功能无差异且动态功耗下降 |
| S04 | AccTile bank enable/clock gating | 高门控机会寄存位覆盖率至少 60% |
| S05 | 按 stage 报告 p50/p95/p99/worst | 特别解释 S1 的低绝对 workload |
| S06 | unified 与 split backend 对照 | 面积、周期、利用率和 EDP |
| S07 | netlist LEC/Formality | mapped netlist 比较点全部闭合 |
| S08 | SRAM 端口冲突和 bank conflict 统计 | 真实 trace 均值与尾延迟 |

### 4.3 Nice-to-Have

| ID | 要求 | 说明 |
|---|---|---|
| N01 | 细粒度 abort/recovery | 当前整 context flush 可用，不是主贡献 |
| N02 | G=2 跨窗口精确合并 | 仅在子系统 EDP 额外改善过门槛时进入 |
| N03 | FPGA 原型 | 增强可复现性，不替代 ASIC PPA |
| N04 | DFT/UPF/physical-aware synthesis | 提高芯片完整度 |

## 5. 候选架构

### 5.1 C0：保守方案——单 Context 精确 GateStack

- `CONTEXTS=1`，`OUT_TILE=32`，`RESIDENT_TERMS=80`；
- 保留当前 Resident/IPD32W/RAW41 共享后端；
- 不引入双 context 仲裁或 G>1；
- 只补真实量化、SRAM wrapper 和功耗门控必要项。

**优点**：与已有 `[RTL]` 最接近，调度和验证风险最低。  
**缺点**：无 build/execute 重叠，难以支撑当前的系统实时叙事。  
**用途**：作为最小可交付实现和基线，不作默认论文主候选。

### 5.2 C1：平衡方案——双 Context 构建/执行重叠

- `CONTEXTS=2`，`OUT_TILE=32`，`RESIDENT_TERMS=80`；
- context A 构建 payload/descriptor 时，context B 使用单一 shared projection backend；
- head slot/cache 物理分 bank 或使用可证明的 1W1R 合同；
- 保留一套 TDR/multicast/product/AccTile，不复制 dense/sparse 核；
- 必须包含 operand isolation、empty-destination 阻断和 AccTile bank enable。

**定量依据**：

- Depth80 CSR 内命中 `99.9826%` `[prof+model]`；
- Stage3 双 context 非权重逻辑容量 `73.21 KiB` `[model]`；
- 85% delivery 假设下的速度 `1.400x` `[model]`；
- 当前双 context 未进入 execution top，上述数字不是 RTL 实测。

**优点**：对已有主数据流改动有限，有望隐藏 build 与 descriptor fill 开销。  
**风险**：端口冲突、context 隔离、双份 slot/cache 和新的尾延迟。  
**决策更新**：真实四stage消融显示 residency 仅带来 `1.024x~1.051x` 周期改善，C1 不再作为下一阶段第一优先级。先完成 FADC24 容量/RTL 迭代、物理公平基线和目标库 PPA；只有 cold build/commit 明确成为瓶颈时才重新晋级 C1。详见 `docs/101_H67真实四Stage消融与GateStack架构再冻结_20260717.md`。

### 5.3 C2：激进方案——跨窗口精确组合与增强投影

- 以 C1 为基础；
- 加入 G=2 跨窗口精确组合，容量越界时无损退化；
- 根据实测瓶颈在 multicast 宽度或 product issue 中二选一增强，不同时堆叠；
- 不默认采用异构 dense/sparse 双核或蝶形网络。

**优点**：吞吐上限更高。  
**风险**：状态、仲裁、容量和验证复杂度高，当前无 `[RTL]` 和 `[PPA]` 依据。  
**用途**：只作为 C1 无法达到吞吐或 EDP 目标时的备选。

### 5.4 候选权衡矩阵

| 项目 | C0 保守 | C1 平衡 | C2 激进 |
|---|---|---|---|
| context | 1 | 2 | 2+G2 |
| shared projection backend | 1 | 1 | 1 或有证据的定向增强 |
| 频率目标 | 500 MHz `[target]` | 500 MHz `[target]` | 500 MHz `[target]` |
| 当前 RTL 接近度 | 高 | 中 | 低 |
| 相对面积 | 1.0 参考 | 未实测，预期主要增加 slot/cache | 未实测，最高 |
| 相对功耗 | 1.0 参考 | 未实测，必须用门控对冲 | 未实测，风险最高 |
| 吞吐潜力 | 低 | 中 | 高 |
| 验证复杂度 | 低 | 中 | 高 |
| 新颖性潜力 | 中低 | 中高 | 高，但证据风险高 |
| 当前决策 | 基线/保底 | **条件推荐** | 暂缓 |

## 6. 公平基线与消融合同

### 6.1 必须实现的基线

| ID | 基线 | 保持不变 | 回答问题 |
|---|---|---|---|
| B0 | Direct/RAW41-only | output-tile schedule、lane、weight/bias、端口 | 等价类表示是否值得 |
| B1 | IPD32W no-residency | 表示、后端、存储口径 | promotion/residency 的独立收益 |
| B2 | Depth0/64/80 | 其余所有配置 | 命中、macro rounding 与 EDP 拐点 |
| B3 | head-major + partial-sum spill | 数值、lane、SRAM 带宽 | output-tile-stationary 的净收益 |
| B4 | split representation backend | 表示选择和算术位宽 | unified backend 是否节省面积/EDP |
| B5 | single/dual context | 投影后端和 trace | overlap 是否抵消端口冲突 |

### 6.2 公平性规则

所有主表对比必须同时满足：

1. 相同 H67 真实 trace 顺序与样本集；
2. 相同量化权重、bias、requant 和数值语义；
3. 相同 `OUT_TILE`、product lane 数和目标频率；
4. 相同 SRAM macro 家族、PVT、SDC、wire/load 假设；
5. 相同复位、warm-up、测量区间和输出反压；
6. 基线允许使用与 GateStack 相同的合理双缓冲和 clock gating；
7. 分开报告 logic、memory、clock、IO 动态功耗和 leakage；
8. 任何模型结果不得与 `[RTL]`/`[PPA]` 结果混在同一列。

### 6.3 功耗消融

至少包含：

- operand isolation 开/关；
- empty-destination 宽总线阻断开/关；
- AccTile bank enable 开/关；
- residency 开/关；
- unified/split backend；
- C0/C1 context 数。

## 7. H67 真实 Trace 合同

### 7.1 主 trace 范围

1. 覆盖 H67 四个 stage 的全部 attention block；
2. 主表至少使用 valid 集 100 frame，样本 ID 固定并可复现；
3. 保留 frame/stage/block/window/head/time/token/lane/output-tile 顺序；
4. 报告全局、分 stage、p50/p95/p99/worst；
5. `sample0/S3.B0/window0` trace-shaped 只作控制长回归，不进入论文真实活动主表。

### 7.2 每个 group 必须导出

- final gate code，lane ID，destination token ID/count；
- K event/token bitmap 及 IPD32W/RAW41 选择原因；
- payload bit/word count、term count、event count；
- 真实量化权重、bias、requant 参数；
- 软件金参考的 accumulator 和 final output；
- checkpoint、config、trace schema 和代码版本 hash；
- 冷启动/warm residency 边界和 context ID。

### 7.3 必须命中的覆盖类

- empty head、K-zero、pair-empty；
- IPD cache hit/miss；
- Depth64/80 越界；
- S=4 class overflow；
- RAW capacity overflow；
- 最大 term/event/fanout；
- 负权重、饱和边界和 requant 边界；
- 随机存储延迟与输出反压。

## 8. 存储和接口合同

### 8.1 GateStack 逻辑容量

| 存储 | C0 逻辑口径 | 证据/状态 |
|---|---:|---|
| Head slot | `24 x 104 x 64 = 159,744 bit` | RAW-sized exact slot |
| Descriptor cache D80 | `24 x 80 x 24 = 46,080 bit` | `[RTL]` 逻辑形状 |
| AccTile | `162 x 32 x 32 = 165,888 bit` | `[RTL]` 逻辑形状 |
| 三项数据小计 | `371,712 bit = 45.38 KiB` | 不含 metadata/macro rounding |
| 当前 Yosys memory bits | `378,208 bit = 46.17 KiB` | `[RTL]`，非物理 SRAM |

C1 Stage3 双 context 非权重逻辑容量为 `73.21 KiB` `[model]`。作为当前候选门槛，macro rounding 后应不高于 `80 KiB` `[target]`；若超过，先比较 Depth64，不得缩小 RAW exact slot。

### 8.2 当前 GateStack 接口

| 接口 | 协议 | 宽度/含义 | 状态 |
|---|---|---|---|
| group | ready/valid | tag、head count、first/count output tile | `[RTL]` |
| payload commit | ready/valid stream | 64-bit word + CSR/RAW metadata | `[RTL]` |
| descriptor fill | ready/valid stream | 24-bit descriptor 有效字段 | `[RTL]` |
| weight request/response | tagged ready/valid | 请求 input channel/tile，返回 `32x8 bit` | `[RTL]` 边界，存储未集成 |
| bias/requant | ready/valid | token + tile vector | `[open]` 数值合同未冻结 |
| final | 2-bank ready/valid | token ID/tag + `32x32 bit/bank` | `[RTL]` |

### 8.3 Full-Encoder 外部接口缺口

下列项目必须在系统签核前冻结：

- encoder 输入 event/voxel 与输出 flow tensor 的协议、位宽和带宽；
- weight/bias 存储层次与外存协议；
- S0/S1/S2 skip SRAM 位宽、bank、端口和 lifetime；
- attention/GateStack/ATLIF 之间的 producer-consumer FIFO；
- 帧级中断/完成与错误上报协议。

这些项目未冻结前，接口完整性不达到 architecture sign-off。

## 9. 吞吐、带宽和 PPA 目标

### 9.1 吞吐目标

| 目标 | 门槛 | 当前状态 |
|---|---:|---|
| 探索频率 | 500 MHz | `[target]`，非 STA 结果 |
| 30 FPS 拍数预算 | 16.667 Mcycle/frame | `[target]` |
| 含 1.25 保护系数的原始模型上限 | 13.333 Mcycle/frame | `[target]` |
| GateStack 子系统周期收益 | >=1.20x | `[target]` |
| GateStack 子系统 EDP 收益 | >=15% | `[target]` |

当前 full-encoder 模型的最优列出点仍仅约 `23.09 FPS` `[model]`，且包含理想化假设。因此 30 FPS 当前明确判定为 **FAIL**。后续若不达标，论文必须改为延迟-能量 Pareto，不得宣称实时。

### 9.2 带宽下界

| 事务 | 30 FPS 带宽 | 证据 |
|---|---:|---|
| ATLIF event 全物化的1-bit打包读写 | 3.945 GB/s | `[model]` |
| S0-S2 skip 8-bit 读写 | 0.697 GB/s | `[model]` |

以上不含权重、真实 bank conflict、控制和外存仲裁，不是总带宽签核。

### 9.3 PPA 合同

| 项目 | 签核要求 |
|---|---|
| 工艺 | 目标 node、standard-cell `.db`、PVT、Vt mix 冻结 |
| 时序 | worst setup corner WNS>=0、TNS=0；hold 不得有未解释 violation |
| 面积 | 绝对预算待定；最终估算必须低于预算 80% |
| 功耗 | 绝对预算待定；最终估算必须低于预算 80% |
| leakage | 不高于总功耗 15% `[target]` |
| SRAM | 使用 compiler/macro 实例，报告 rounded bits、面积、读写能量和时序 |
| 活动 | 真实 trace mapped SAIF，报告 annotation coverage |
| 等价 | RTL-to-mapped-netlist Formality/LEC 全部闭合 |
| 余量 | logic/memory/clock/IO 分账后再加 15%-20% 实现余量 |

由于面积与功耗绝对预算缺失，根据 architecture skill 规则，当前不允许最终签核。

### 9.4 Clock/Power Budget

当前顶层只有 `clk_core` 一个时钟域，但应按门控组分账：

| 时钟/门控组 | 频率 | activity alpha | clock power | 门控分类 |
|---|---:|---:|---:|---|
| scheduler/control | 500 MHz `[target]` | `[open]` | `[open]` | 待 mapped SAIF |
| head-slot/cache | 500 MHz `[target]` | `[open]` | `[open]` | 待 mapped SAIF |
| decoder/TDR/multicast/product | 500 MHz `[target]` | `[open]` | `[open]` | 待 mapped SAIF |
| AccTile banks/final | 500 MHz `[target]` | `[open]` | `[open]` | 待 mapped SAIF |

RTL VCD 中 projection 层次占 `89.84%` toggle `[RTL]`，这只能决定门控优先级，不能代替 activity alpha 或功耗。

## 10. 控制、复位、CDC 与安全边界

- 当前 GateStack slice 为单 `clk_core` 时钟域，CDC 暂为 N/A；
- `rst_core` 是同步复位，所有 SRAM wrapper 必须冻结复位后内容有效规则；
- 未来若 AXI/DRAM 或 SRAM 使用异步域，必须新增 CDC/RDC 签核；
- 安全认证要求为 N/A；数据完整性由 tag、PLAN/COMMIT、watchdog 和 abort 保证；
- DFT/BIST/UPF 当前未冻结，属于 physical handoff 缺口。

## 11. 风险登记表

| ID | 风险 | P | I | 分数 | 等级 | 缓解措施 | Owner |
|---|---|---:|---:|---:|---|---|---|
| R1 | 无四 stage 真实 bit trace | 5 | 5 | 25 | HIGH | 导出真实 gate/token/lane/weight/bias/requant trace | 算法/Profiling |
| R2 | 无目标库 PPA | 5 | 5 | 25 | HIGH | 冻结 node/PVT/macro，运行 DC/STA/SAIF/LEC | Synthesis/PPA |
| R3 | 绝对面积和功耗预算缺失 | 5 | 5 | 25 | HIGH | 项目负责人冻结预算 | Architecture/PI |
| R4 | 无公平基线和 RTL 消融 | 5 | 5 | 25 | HIGH | 实现 B0-B5 同接口对照 | RTL/Verification |
| R5 | 主贡献被视为常规机制组合 | 4 | 5 | 20 | HIGH | 以单一耦合数据流表述，用三类对照证明不可替代性 | Architecture/Paper |
| R6 | full encoder 未达 30 FPS | 5 | 4 | 20 | HIGH | 闭合空间引擎与存储瓶颈；否则删除实时 claim | System Architecture |
| R7 | 量化/requant/skip 数值合同未冻结 | 4 | 5 | 20 | HIGH | RTL-exact 软件部署和 valid825 验证 | Algorithm/RTL |
| R8 | SRAM macro rounding/端口使 C1 失去收益 | 4 | 5 | 20 | HIGH | Depth64/80 和 1W1R/banked macro DSE | Memory/PPA |
| R9 | projection 高翻转抵消压缩收益 | 4 | 4 | 16 | HIGH | operand isolation、bank enable、mapped SAIF 消融 | RTL/Power |
| R10 | dual context 端口冲突使模型 1.400x 失效 | 4 | 4 | 16 | HIGH | ordered trace cycle replay 和真实 SRAM latency | Architecture/Verification |
| R11 | full-encoder 边界过大导致工作量失控 | 3 | 5 | 15 | HIGH | PPA 聚焦 GateStack，其余系统用校准模型 | Project Lead |

所有 HIGH 风险均有缓解路径，但尚未关闭，因此不满足最终签核条件。

## 12. 候选淘汰门槛

### 12.1 通用硬门槛

任一条触发即停止该候选：

1. 真实 trace 上出现任何非预期数值 mismatch；
2. exact fallback 发生截断、丢失、重排或无界阻塞；
3. 相对最优公平基线周期收益 `<1.20x`；
4. 目标库含 SRAM/clock 的 EDP 改善 `<15%`；
5. WNS<0 或存在未解释 unconstrained path；
6. 面积或功耗超过最终冻结预算的 80%；
7. 论文主表仍只使用 trace-shaped 构造载荷。

### 12.2 C1 专用门槛

- dual context 相对 C0 没有可重复的净周期收益时，回退 C0；
- Stage3 非权重 macro rounded 容量超过 80 KiB 时，必须先评估 Depth64；
- 若 dual context 增加的功耗使 EDP 不达标，不得仅凭吞吐保留。

### 12.3 C2 专用门槛

C2 只有在以下条件同时满足时才允许开始 RTL：

1. C1 在真实 trace 上明确受 GateStack 本身吞吐限制；
2. C2 模型相对 C1 的子系统 EDP 预计额外改善至少 25%；
3. 新增状态不改变 exact fallback 和顺序语义；
4. 外存/空间引擎等 Amdahl 瓶颈不会将系统收益压到 5% 以下。

## 13. 架构签核清单

| 检查项 | 状态 |
|---|---|
| 所有 Must-Have 有对应架构块 | 已映射 |
| 三个差异化候选 | 完成 |
| 单一推荐候选 | C1 条件推荐 |
| 公平基线/消融合同 | 完成规格，尚未执行 |
| 接口完整 | GateStack 已定义，full encoder 未完成 |
| 存储 map/macro | 逻辑形状已定义，macro 未冻结 |
| 性能目标带余量达成 | 未达成 |
| 面积/功耗低于预算 80% | 预算缺失，未达成 |
| HIGH 风险全关闭 | 未达成 |
| 时钟域/CDC/reset | 单域与同步 reset 已定义，系统 CDC 待定 |
| DFT/UPF | 未定义 |
| 验证策略 | 已定义主合同，尚未执行 P0 |
| clock power budget | 表已创建，activity/power 待 mapped SAIF |

## 14. 最终决策记录

1. **冻结** H67 GateStack 为一个耦合的精确稀疏投影数据流，不再分拆三个弱创新。
2. **条件推荐 C1**：双 context、Depth80、32-lane、单共享后端与显式功耗门控。
3. **C0 作为保底和公平基线**，不删除。
4. **C2 暂缓**，未过 25% 额外 EDP 和 Amdahl 门槛不进入 RTL。
5. 下一阶段优先级固定为：真实 trace -> 公平基线 -> bit-exact -> 目标库 PPA -> full-encoder 分账。
6. 未关闭 P0 前，不增加新的架构名词或复杂互连。

**Architecture 终态**：

> `CONDITIONAL_ARCHITECTURE_FREEZE / SIGNOFF_NOT_ACHIEVED`

该状态表示候选、对照和门槛已冻结，可以进入 trace/基线/PPA 补强；不表示已达到 DATE 投稿、RTL handoff 或 ASIC sign-off 标准。
