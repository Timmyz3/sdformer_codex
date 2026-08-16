# H67真实四Stage消融与GateStack架构再冻结

> 2026-07-18 状态更新：本文中的“FADC24尚未进入RTL”已经被后续工作取代。流式FADC24 decoder、四stage同顶层真实trace回放和专属SVA均已完成；最新结论见 `docs/103_FADC24流式Decoder与四Stage同顶层RTL迭代_20260718.md`。本文其余内容保留为架构决策历史和GateStack-v1基线。

## 一、结论先行

本轮完成了 DATE 审稿补强中最关键的第一组真实证据：H67 epoch19 的真实 Q/K、真实 Q1.7 gate、checkpoint projection 权重候选 INT8 码和真实 bias 候选码，已经在 S0/S1/S2/S3 四个 stage 上完成 RAW41-only、IPD 无驻留和完整 GateStack 的同顶层 RTL 回放，所有模式均为 `mismatch=0`、`protocol_error=0`。

真实结果迫使架构贡献重新排序：

1. **主架构机制**是 final-gate 等价类驱动的 exact factorized projection，不是 descriptor cache；
2. **output-tile-stationary + head-stacked replay**是需要用 head-major spill 基线证明的数据流贡献；
3. **descriptor residency**当前只带来 `1.024x` 到 `1.051x` 的周期改善，应降级为数据移动与潜在能效优化；
4. 当前 IPD32W 在高扇出 head 上会发生 RAW fallback，S3 单个 overflow head 使理论 term 减少从 `80.13%` 降为 RTL 实际 `52.26%`；
5. 新候选 FADC24 能在该真实 overflow head 上把 `1078 byte` 降为 `815 byte`，并将每输出 tile 的执行 term 从 `1290` 降为 `537`，但它尚未进入 RTL，不能列为已完成贡献。

因此，当前架构状态是：

> **GateStack-v1 已完成真实四 stage 单窗口 RTL 功能消融；GateStack-FADC 是下一轮条件候选；系统级 DATE 签核仍被物理公平基线、部署精度和目标库 PPA 阻塞。**

## 二、证据等级

| 标记 | 本文含义 |
|---|---|
| `[prof]` | profile100 ordered workload 或真实位级 trace 的软件统计 |
| `[RTL]` | Icarus/Verilator/SVA 下的 cycle、计数器和整数输出结果 |
| `[model]` | 容量上下界、周期或存储模型，不是硬件实测 |
| `[PPA]` | 目标标准单元库、目标 PVT、SRAM macro 和 mapped SAIF；当前缺失 |

## 三、真实 Trace 数据闭环

### 3.1 采集内容

本轮实际采集了 H67 ep19 一个样本、四个 stage 的首个 attention block、每个模块首个 window：

- 二值 `Q/K`，布局为 `[T=2, window, head, spatial_token, lane]`；
- RTL Shiftmax 之后的 Q1.7 gate code；
- checkpoint 中 projection 浮点权重和 bias；
- 逐输出通道 dyadic INT8 候选权重；
- 与候选 scale 对齐的 32-bit bias accumulator code；
- 文件 SHA256、shape、stage 和量化合同 manifest。

加载审计结果为 ATLIF `105`、H60 attention `12`、checkpoint `missing=0`、`unexpected=0`。真实 trace 和四 stage 审计均通过。

主要路径：

- `results/h67_real_bit_trace_20260717/manifest.json`；
- `results/h67_real_bit_trace_audit_20260717/audit.md`；
- `results/gatestack_h67_real_trace_vectors_20260717/manifest.json`。

### 3.2 量化边界

当前 projection 权重采用逐输出通道对称 INT8 与 dyadic scale：

```text
scale[o] = 2 ^ ceil(log2(max_abs(weight[o,:]) / 127))
code[o,i] = RNE(weight[o,i] / scale[o])，饱和到[-127,127]
```

四 stage bias accumulator 均能装入当前 signed-17 候选范围；权重最大绝对量化误差约为 S0 `0.0039003`、S1 `0.0019531`、S2 `0.0019530`、S3 `0.0009766`。

这些只证明编码和本次整数回放成立。尚未完成 valid825 部署精度、projection BN folding、最终 requant、残差 scale 和饱和合同，因此仍禁止写“完整 INT8 部署已验证”。

## 四、四 Stage 同顶层 RTL 消融

结果文件：`results/gatestack_real_trace_ablation_20260717/report.md`。

| Stage | 路径 | 周期 | 相对 RAW 加速 | payload word | 相对 RAW 减少 | projection term | 相对 RAW 减少 |
|---:|---|---:|---:|---:|---:|---:|---:|
| S0 | RAW41-only | 4,873 | 1.000x | 936 | 0.00% | 537 | 0.00% |
| S0 | IPD 无驻留 | 2,455 | 1.985x | 186 | 80.13% | 186 | 65.36% |
| S0 | GateStack | 2,395 | **2.035x** | 110 | **88.25%** | 186 | **65.36%** |
| S1 | RAW41-only | 7,490 | 1.000x | 3,744 | 0.00% | 0 | 0.00% |
| S1 | IPD 无驻留 | 1,729 | 4.332x | 72 | 98.08% | 0 | 0.00% |
| S1 | GateStack | 1,677 | **4.466x** | 12 | **99.68%** | 0 | 0.00% |
| S2 | RAW41-only | 54,489 | 1.000x | 14,976 | 0.00% | 4,140 | 0.00% |
| S2 | IPD 无驻留 | 22,459 | 2.426x | 1,836 | 87.74% | 1,956 | 52.75% |
| S2 | GateStack | 21,374 | **2.549x** | 648 | **95.67%** | 1,956 | **52.75%** |
| S3 | RAW41-only | 524,854 | 1.000x | 59,904 | 0.00% | 64,848 | 0.00% |
| S3 | IPD 无驻留 | 259,122 | 2.026x | 15,264 | 74.52% | 30,960 | 52.26% |
| S3 | GateStack | 252,942 | **2.075x** | 8,663 | **85.54%** | 30,960 | **52.26%** |

所有十二种 stage/path 组合均完成 Icarus 和 Verilator+SVA 回放，32-bit accumulator 输出零 mismatch，协议错误为零。

### 4.1 正确解释

- S1 的 K 完全为零，`projection term=0`；其较高加速主要来自不再搬运和扫描固定 RAW payload，不能代表高计算密度场景。
- S0/S2/S3 的主要周期收益在 IPD 无驻留阶段已经出现，证明 exact factorized execution 是主机制。
- residency 相对 no-residency 的速度仅为 S0 `1.025x`、S1 `1.031x`、S2 `1.051x`、S3 `1.024x`。
- payload word 减少不是功耗减少。只有目标库 mapped SAIF 才允许给出能量和 EDP claim。
- 当前 RAW41-only 是同顶层运行路径，不能用于面积主表；physically-stripped Direct top 仍缺。

## 五、S3 RAW Fallback 的架构瓶颈

S3 的 24 个 head 中有一个真实 head 发生 IPD32W 容量回退：

| 指标 | 值 |
|---|---:|
| Head | 4 |
| term | 61 |
| event | 814 |
| 最大 fanout | 52 |
| fanout 大于 21 的 term | 15 |
| IPD32W 大小 | 1078 byte |
| RAW 物理槽 | 832 byte |

若所有 head 都能执行 factorized term，该 stage 每输出 tile 只需 `537` 个 term；由于该 head 回退 RAW，当前实际变为 `1290` 个 term。乘以 24 个输出 tile 后，投影 term 为 `30,960`，而理想值为 `12,888`。

这说明固定 token-list 格式会让少数高扇出 term 破坏大量投影复用，是比继续增加 cache depth 更值得解决的结构问题。

## 六、FADC24 条件候选

### 6.1 格式

FADC24 为每个 `(final gate code, K lane)` term 使用一个 24-bit descriptor：

```text
gate_code[8:0] | lane[4:0] | destination_count[7:0]
| bitmap_mode | reserved
```

目的集合逐 term 选择：

- fanout `<=21`：8-bit token ID list；
- fanout `>21`：162-bit token bitmap，按 21 byte 存储，最高 6 个 padding bit 固定为零。

该机制只改变目的索引表示，不改 gate、K、权重、乘积或累加顺序的数学结果。Python 金参考已完成逐 head encode/decode 往返等价检查。

### 6.2 单窗口真实结果

| Stage | 当前 RAW fallback | FADC24 RAW fallback | 当前每输出 tile term | FADC24 term | 额外减少 |
|---:|---:|---:|---:|---:|---:|
| S0 | 0 | 0 | 62 | 62 | 0.00% |
| S1 | 0 | 0 | 0 | 0 | 0.00% |
| S2 | 0 | 0 | 163 | 163 | 0.00% |
| S3 | 1 | 0 | 1290 | 537 | **58.37%** |

S3 head4 从 `1078 byte` 降为 `815 byte`，能够装入原 `832 byte` 物理槽。

实现时必须拆分两个原本混用的容量常量：`RAW_PAYLOAD_BITS=6642` 用于 RAW41 格式合法性检查，`SLOT_CAPACITY_BITS=104×64=6656` 用于 SRAM 物理写入上限。当前 head-slot adapter 把两者都绑定到 `HEAD_BITS=6642`；FADC RTL 不能直接放宽 RAW 合同，而应独立参数化，并继续要求 RAW payload 精确等于 6642 bit。

结果文件：`results/gatestack_fadc24_real_trace_20260717/analysis.md`。

### 6.3 Profile100 容量上下界

profile100 只有逐 head 的 term、event 和 max-fanout，没有逐 term fanout。因此本轮计算 guaranteed-fit、impossible 和 ambiguous 三类容量边界：

| Stage | head 实例 | 当前 fallback | guaranteed-fit | ambiguous | impossible |
|---:|---:|---:|---:|---:|---:|
| S0 | 264,000 | 5.713% | 95.428% | 4.548% | 0.023% |
| S1 | 144,000 | 0.062% | 99.957% | 0.040% | 0.003% |
| S2 | 216,000 | 0.321% | 99.836% | 0.148% | 0.016% |
| S3 | 48,000 | 1.862% | 98.654% | 1.056% | 0.290% |

相对当前 IPD32W，projection term 工作量的保守到乐观减少范围为：S0 `10.38%~74.12%`、S1 `2.75%~10.53%`、S2 `8.29%~17.85%`、S3 `8.79%~33.77%`。

结果文件：`results/gatestack_fadc24_profile100_20260717/analysis.md`。

FADC24 不能宣称全覆盖：四个 stage 都存在少量确定不可装入实例，必须继续保留 RAW 无损 fallback；ambiguous 实例还需要扩大位级 trace 才能消歧。

### 6.4 文献边界

FADC24 的“list/bitmap 按稀疏形态选择”本身不是可以独占的创新：

- [SMASH](https://arxiv.org/abs/1910.10776) 已用层次 bitmap 和硬件 bitmap 管理单元处理稀疏索引；
- [Cerberus](https://doi.org/10.1145/3653020) 已根据矩阵特征在 compressed sparse、bitmap 和 dense 模式间选择；
- [Spada](https://dl.acm.org/doi/10.1145/3575693.3575743) 已采用 profile-guided adaptive sparse dataflow；
- [ZeD](https://www.comp.nus.edu.sg/~tulika/PACT24.pdf) 已将 bit-tree 表示与稀疏矩阵数据流协同设计。

本工作可辩护的区别不是“提出 bitmap”，而是：

1. term 来自 H67 final-gate 与 K-lane 的精确等价关系，不是普通矩阵非零坐标；
2. list/bitmap 的选择粒度是一个可复用乘积的 destination set；
3. 编码选择直接决定一个乘积是否能够跨 token multicast，而不只是存储压缩；
4. 任意越界都回退 RAW41，不改变网络语义；
5. 编码、output-tile-stationary 累加和 shared projection backend 共同构成执行数据流。

论文中应将其写成 workload-semantic representation/dataflow co-design，不能写成通用稀疏格式的首次提出。

## 七、架构再冻结

### 7.1 冻结主线

当前冻结的论文硬件核心为：

> **GateStack：基于 final-gate 等价类的精确因子化投影架构，采用容量安全的多格式目的编码、output-tile-stationary 累加和共享 multicast/product backend。**

GateStack-v1 使用 IPD32W/RAW41；GateStack-FADC 只有在真实 trace 扩展、同顶层 RTL 和 PPA 均过门槛后才替换 IPD32W。

### 7.2 候选优先级变化

| 候选 | 新判定 | 原因 |
|---|---|---|
| C0 单 context GateStack-v1 | 当前 RTL 主线 | 四 stage 真实回放已通过 |
| C-FADC 单 context | 下一轮第一优先级 | 直接解决 profile100 fallback，潜在计算收益大于加深 cache |
| C1 双 context | 降为条件优化 | residency 周期收益只有 2.4%~5.1%，build/execute overlap 尚无 RTL 证据 |
| G>=2/G2 跨窗口 | 暂缓 | 增加状态和验证复杂度，当前收益证据不如 FADC |
| 异构双核/蝶形互连 | 不进入主线 | 当前 workload 与端口模型没有证明其 EDP 必要性 |

### 7.3 DATE 贡献表述候选

只有完成对应证据后，才可使用以下三条：

1. 面向 all-binary 事件光流的 final-gate 等价类精确投影架构，将相同 gate 与 K-lane 的 token 乘积合并为 multicast term；
2. 一种容量安全的 fanout-adaptive destination dataflow，在 token list、bitmap 和 RAW41 间无损选择，以减少高扇出 fallback；
3. output-tile-stationary 的 head-stacked 执行与共享多格式 backend，避免 head-major partial-sum spill，并通过真实四 stage trace、物理公平基线和目标库 PPA 验证。

第 2 条当前只有 `[prof+model]`，第 3 条仍缺 head-major 物理基线，因此尚不能放入论文摘要的完成时态。

## 八、下一轮最小验证清单

| 优先级 | 工作 | 晋级门槛 |
|---:|---|---|
| P0-1 | 扩大 FADC 位级 trace，覆盖更多样本、block 和 window | 消歧 profile100 ambiguous，分 stage 报 fallback 与 p95/p99 |
| P0-2 | 实现 FADC24 leaf decoder 与同顶层 no-residency 变体 | 真实四 stage零 mismatch；S3 周期优于 IPD32W no-residency |
| P0-2a | 拆分 `RAW_PAYLOAD_BITS` 与 `SLOT_CAPACITY_BITS` | RAW仍严格为6642 bit；压缩格式最多使用6656 bit物理槽 |
| P0-3 | 做 FADC24 24-bit 解包和 bitmap scan 结构消融 | decoder 面积/周期开销不能吞掉 term 收益 |
| P0-4 | physically-stripped Direct RAW41 top | 与 GateStack 相同端口、lane、SRAM 和反压 |
| P0-5 | head-major + partial-sum spill 基线 | 证明 output-tile-stationary 的净流量、周期和 EDP |
| P0-6 | projection valid825 部署验证 | 精度门槛冻结，完成 BN/bias/requant/residual 合同 |
| P0-7 | 获取 `.db`、PVT、SRAM macro 并跑 DC/STA/mapped SAIF | 500 MHz 时序闭合，EDP 与面积主表可复现 |

FADC24 的 RTL 晋级停止条件：真实扩展 trace 上 term 工作减少不足 `10%`，或同顶层周期改善不足 `5%`，或 mapped EDP 不改善。满足任一条件就保留 IPD32W，不为新格式增加控制复杂度。

## 九、当前仍不能回答“可以直接发 DATE”

本轮解决了“没有真实网络 trace”和“没有机制分账”的一部分拒稿理由，但尚未解决：

- RAW41 physically-stripped 面积基线；
- head-major partial-sum spill 数据流基线；
- valid825 部署精度；
- target-library area/timing/power/EDP；
- SRAM macro rounding、bank、端口与 mapped SAIF；
- full encoder 的 ATLIF、attention、skip/residual、stage 存储和外存分账。

因此当前可以称为“真实 workload 驱动的 projection subsystem RTL 原型”，还不能称为完成的 H67 encoder ASIC，也不能直接用 Yosys cell 数代替 DC/PPA 投稿。
