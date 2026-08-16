# TESSA 架构独立审阅与 RTL 准入条件

**审阅日期**：2026-07-13  
**审阅角色**：architecture-orchestrator  
**审阅对象**：TESSA（Temporal-pair Exact Sufficient-Statistics Accelerator）架构阶段  
**审阅范围**：架构需求、候选探索、性能与 PPA 风险、创新边界、RTL 交接条件  
**明确不包含**：本轮不修改或验证 RTL，不把分析模型结果写成 RTL、DC 或芯片结果

## 0. 独立结论

当前结论分成三个层级：

1. **TESSA 的研究方向成立。** H67/H68 各 100 个样本的聚合 profile 支持 temporal-pair 驻留、统一同构底座、block-aware descriptor 和共享 class-stationary SCS。H67/H68 的全网稀疏特征接近，不支持分别实例化两套核。
2. **当前架构尚未签核。** 目标工艺、频率、面积、功耗、端到端吞吐、SRAM 宏、外部接口和 ordered trace 均未冻结，architecture skill 的强制约束与 sign-off 条件未满足。
3. **RTL 只能“受控准入”。** 可以开始候选 A 的 fixed-bitmap pair 数据通路和候选 B 的可旁路 2-context 骨架；不得冻结 BMRF、4-context、row OOO、方向 bank mapping 或异构双核，也不得把当前周期代理写成实现收益。

本轮推荐的平衡候选不是“功能全开的 TESSA”，而是：

> **128-bit temporal-pair 驻留 + fixed-bitmap PESF + 两个 row context + 可旁路 PCCC + 共享 class-stationary SCS + block 静态 descriptor。**

`union-membership`、BMRF 和动态调度必须作为独立 feature gate，在 ordered trace 和同约束 PPA 后逐项晋级。

当前 architecture sign-off 判定：**NO-GO**。  
候选 A/B 探索性 RTL 判定：**CONDITIONAL GO**，仅限第 12 节明确的边界。

### 审阅后端口模型补充

后续端口感知三阶段重放确认了本审阅指出的双 commit 风险：H67 在 128-bit 供数、分 bank 单写口、无 PCCC 合并、2-context 下只比当前周期代理下降 22.33%；PCCC 全合并乐观上界可下降 48.85%。同时，4-context 在全合并上界下相对 2-context 仍可能改善 13.49%。因此正式结论进一步收紧为：实现接口参数化 `1/2/4`、首版启用 2，但最终物理 context 数不冻结；详见 `docs/57_TESSA供数提交端口与Context再评估.md`。该补充不改变完整架构 `NO-GO` 判定。

## 1. 审阅依据与证据等级

本审阅读取：

- `memory/architecture/run_state.md`；
- `docs/53_H67H68真实Workload统计与架构约束.md`；
- `docs/54_Transformer架构电路EDA论文深度调研与迁移边界.md`；
- `docs/55_TESSA统一时间对类驻留加速器架构方案.md`；
- `results/h67_h68_profile100_arch_features.json`；
- `docs/45-51` 中的架构签核、RTL/DC、验证、文献与创新缺口材料。

当前 run state 的 `last_stage` 虽为 `arch_signoff`，但它只表示流程曾到达该阶段，不等于签核通过。既有文档明确记录了 NO-GO 和未关闭的高风险项，本审阅沿用实际证据，不沿用状态名作结论。

### 1.1 证据等级

| 等级 | 定义 | 可支持的结论 |
|---|---|---|
| E0 | 概念、文献迁移或未测假设 | 只能列为候选 |
| E1 | 代数推导、局部软件检查或聚合机会统计 | 可冻结功能方向，不能声称系统收益 |
| E2 | 真实 workload 分布和可复现分析模型 | 可筛选架构，不能代替 RTL/PPA |
| E3 | RTL bit-exact、有限 FIFO 周期重放、同库 DC/SAIF/SRAM 结果 | 可形成论文硬件证据 |
| E4 | 布局后或硅后数据、完整系统测量 | 可进行最终物理签核 |

当前最高证据为 E2，而且 `profile100` 的 scope 明确不支持 ordered pair/row burst、union-index 流量、输入光流相关性和 SRAM conflict。

## 2. 需求分类与未决项

### 2.1 功能需求

| ID | 优先级 | 需求 | 映射模块 | 状态 |
|---|---|---|---|---|
| F01 | Must | H67 为功能超集，H68/TTX 为编译期特化 | descriptor、PESF、score/RNE | 已定义，未在 TESSA RTL 验证 |
| F02 | Must | 对冻结 hardware-order golden 逐位一致 | PESF、PCCC、SCS、gate | 数值合同已定义，TESSA 实现未开始 |
| F03 | Must | pair-empty 仍向分母提交两个 class-2 token | metadata、PCCC、SCS | 已定义 |
| F04 | Must | `K_t=0` 不发 gated-K，但精确参与 max/denominator | PCCC、histogram、SCS | 已定义 |
| F05 | Must | 12 个 attention block 通过 descriptor 时分复用 | descriptor scheduler | 架构定义完成 |
| F06 | Must | 105安装/93调用/81固定部署活跃ATLIF分列，PSN时间矩阵不按module数复制 | HIT-Flow HTT、DP-TME、参数ROM | 几何与吞吐下界已建模，RTL/PPA未定 |
| F07 | Must | 残差和三条长 skip 仅为 S0/S1/S2，S3 为 bottleneck-local | residual/skip SRAM | 语义已纠正，接口未冻结 |
| F08 | Must | backpressure 下无丢失、重复、串行和跨 context 污染 | FIFO、tag、completion | 验证要求已列，协议未冻结 |
| F09 | Should | H67/H68 使用同一物理底座，关闭无用 Motion/class 逻辑 | 编译期参数、门控 | 架构可行，PPA 未验证 |
| F10 | Nice | bitmap/event 表示可切换 | PESF/BMRF | 未获准冻结 |

### 2.2 性能需求

| ID | 优先级 | 需求 | 当前状态 |
|---|---|---|---|
| P01 | Must | 目标 encoder FPS、attention FPS 和端到端 latency | **未给出** |
| P02 | Must | 最坏 burst、p99 latency 和 backpressure 预算 | **未给出** |
| P03 | Must | 目标时钟频率和允许的 pipeline 深度 | **未冻结**；500 MHz 只是探索值 |
| P04 | Must | 输入、输出、SRAM 和上游 projection 的可持续带宽 | **未给出** |
| P05 | Should | 相对 162-token 基线至少有可复现净收益 | 有 E2 周期代理，无 E3 证据 |
| P06 | Should | 2-context 相对 pair 单 context 吞吐提升至少 20% | E2 模型满足，ordered trace 未验证 |

在 P01-P04 未冻结前，无法检查 skill 要求的“吞吐至少保留 10% margin”和“最坏情况满足 latency”。

### 2.3 面积与功耗需求

| ID | 优先级 | 需求 | 当前状态 |
|---|---|---|---|
| A01 | Must | 工艺节点、standard-cell library、PVT、VDD | **未冻结** |
| A02 | Must | 核面积预算和 SRAM 面积上限 | **未给出** |
| A03 | Must | SRAM compiler 的最小宏、端口、时序和能量 | **未给出** |
| W01 | Must | 总功耗、动态功耗和漏电预算 | **未给出** |
| W02 | Must | 真实 workload SAIF、clock tree 与 SRAM 活动 | **未完成** |
| W03 | Must | 高门控机会域至少 60% 寄存器位纳入 ICG 规划 | **无法评估** |
| W04 | Should | 集成预留 15%-20%，漏电低于总功耗 15% | 仅有规则，无估计 |

由于 `clk_mhz`、`area_um2` 和 `power_mw` 三项强制约束缺失，本轮不能形成正式 area/power margin。

### 2.4 接口、可靠性与验证需求

| 类别 | Must-Have 未决项 |
|---|---|
| 输入 | 128-bit/拍 pair read 是否真实可供；若由两个 64-bit bank 组装，延迟和冲突如何处理 |
| commit | 每 pair 最多两个 active 写和两个 histogram 更新；端口、队列、旁路和反压尚未定义 |
| 输出 | gated-K 的目的端、散写顺序、row completion、zero-fill 与 error 语义未冻结 |
| 控制 | CSR/descriptor 协议、启动/中止/flush、状态和异常寄存器未冻结 |
| 时钟复位 | 时钟域数量、同步/异步 reset、RDC/CDC 策略未冻结 |
| DFT | scan、SRAM MBIST、LUT ROM test 和测试模式未定义 |
| 安全/功能安全 | 当前项目未提出安全或 ISO 26262 目标，应显式标为不适用，而非默认已满足 |
| 验证 | TESSA 级 golden、formal、coverage、LEC、SDF 均未开始 |

## 3. Workload 事实与架构含义

### 3.1 全网事实

| 指标 | H67 | H68 | 可支持的结论 |
|---|---:|---:|---|
| pair-empty | 73.90% | 74.20% | 早到 metadata 有较高 payload/逻辑门控机会 |
| K-zero | 83.11% | 83.29% | active bank 写入稀少，class 路径重要 |
| motion-zero | 83.18% | 83.36% | H67 Motion 后级有门控机会 |
| Delta=0 | 74.00% | 74.30% | 几乎由 pair-empty 构成，非空时间复用不是主线 |
| active-entry/row | 18.38 | 18.40 | 仅约 11.35% token 进入 active replay |
| fold class/row | 2.27 | 2.24 | occupied-class scan 合理，但不能推出 PCCC merge 率 |
| 162-token 周期代理/帧 | 1387558 | 1372247 | 仅为 attention row 代理 |

H67 和 H68 的分布差异很小，支持统一同构底座。它不支持运行时同时保留两套完整 attention 核。

### 3.2 block 级差异

H67 中 S0B0 的 active-entry 约为 `59.89/row`，S0B1 约为 `3.05/row`，相差约 19.6 倍；S2B3 的 pair-empty 接近 99.98%。因此 block-aware 的“需求”证据较强，但当前只能冻结静态 descriptor，不能据此直接冻结 row OOO、动态密度阈值或异构双核。

### 3.3 当前周期模型的有效范围

| 模型 | 当前 serial | pair 1-context | pair 2-context | pair 4-context |
|---|---:|---:|---:|---:|
| H67 cycles/frame | 1387558 | 843238 | 607690 | 607489 |
| H68 cycles/frame | 1372247 | 827927 | 612195 | 612000 |

H67/H68 的 2-context 相对 pair single-context 分别改善 27.93%/26.06%；原两阶段模型中 4-context 相对 2-context 仅改善约 0.033%/0.032%。端口感知补充表明这一结论只适用于没有独立 commit 阶段的模型，因此当前决策是“首版启用 2、参数化 1/2/4、删除 8”，最终物理数量不冻结。

但模型把 pair 前端固定为 81 cycles，尚未计入：

- 128-bit pair SRAM 的供数和 bank conflict；
- 单 pair 两次 active/histogram commit 的写端口冲突；
- commit FIFO 堵塞、SCS 回放、输出 backpressure；
- row burst、block barrier drain 和上游/下游服务时间；
- projection、ATLIF、residual、skip 和 decoder。

按探索性 500 MHz 折算，H67 2-context 代理约为 823 attention-only frame/s，p99 代理约为 740 frame/s；这些数字不是 encoder 或端到端 FPS，也不是频率可实现性证明。

## 4. 五项关键机制的证据审阅

| 机制 | 正确性证据 | workload/性能证据 | PPA 证据 | 综合等级 | 审阅决定 |
|---|---|---|---|---|---|
| pair-resident | 七个充分统计量代数明确；hardware-order 合同明确 | 192 降至 128 data bit/pair，理论下降 33.3%；162 降至 81 issue 的 E2 上界 | 无 128-bit SRAM、时序和功耗 | E2- | 进入 fixed-bitmap 候选，但 81 cycles 不是保证值 |
| 2-context | 状态隔离和 tag 原则已定义 | 聚合模型显示相对 1-context 改善 26%-28%，4-context 额外收益低于 0.1% | 无 context SRAM 宏和控制面积 | E2- | 作为参数默认值进入骨架，必须支持退化为 1 |
| PCCC | 双 K-zero 同 class `+2` 语义正确 | K-zero 约 83% 只证明机会；同 class 比例、同拍 collision 和写事务下降未知 | 无 merge network、队列或 SRAM 能耗 | E1 | 可旁路实现，不能作为已晋级主贡献 |
| BMRF | 只有结构设想和 bitmap fallback 原则 | union-membership 真实 packet、压紧长度和冲突未得到 | 无综合、时序、SAIF | E0 | NO-GO，不进入第一版 RTL |
| block-aware | stage/block 异质性有 E2 强证据 | 静态 descriptor 合理；动态 admission/OOO 收益未知 | 控制、completion 和 reorder 成本未知 | 需求 E2，策略 E1 | 冻结静态字段；动态策略保持关闭 |

### 4.1 pair-resident 的隐藏带宽条件

每 row 的逻辑数据由 `192*81=15552 bit` 降为 `128*81=10368 bit`，每帧理论少 `34.84 Mbit`。但一拍一个 pair 需要 128-bit/拍的内部读取带宽；若上游只能提供单个 64-bit 读口，pair assembler 仍需两拍，81-cycle 前端收益不会自动出现。

此外，一对时间 token 最多产生两个 active entry 或两个 histogram update。若 context bank 只有单写口，最坏 active row 仍可能需要 162 个 commit cycle。候选 B 必须使用解耦 commit queue、明确的写合并/旁路，或证明双写端口的面积和能耗可接受。

### 4.2 2-context 的正确定位

2-context 的作用是让 pair front 与共享 SCS 重叠，不是增加计算 lane。它是当前最有定量支持的调度参数，但证据来自按 block 均值复制到 row 的粗粒度重放。ordered trace 返回前，RTL 必须允许 `NUM_CONTEXTS=1/2/4` 参数化，产品默认值和物理宏数量不得锁死。

### 4.3 PCCC 的准入边界

PCCC 的论文价值依赖“pair 内双 score 在存储前按 class 合并”，而不是 histogram 本身。正式晋级至少需要：

1. 双 K-zero 同 class 和不同 class 的真实比例；
2. 同拍多 update 的 class collision 分布；
3. 单/双端口 histogram 的 transaction、stall 和旁路次数；
4. merge 网络能耗小于节省的 SRAM/仲裁能耗；
5. 对 hardware-order golden 0 mismatch。

### 4.4 BMRF 的准入边界

BMRF 当前不能以“复旦蝶形网络迁移到 SNN”作为创新理由。它只有在动态 membership 压紧、统计归约和 PCCC segmented reduce 三者共享同一网络，并相对 prefix compactor/fixed bitmap 获得至少 15% EDP 改善时，才具备保留价值。

## 5. 四候选架构对照

所有候选都必须使用相同 hardware-order golden、trace、输入输出带宽、SRAM 假设、工艺库、频率约束和综合 effort。

| 候选 | 结构 | context 私有状态 | 性能证据 | 面积/功耗先验 | 风险 | 当前决定 |
|---|---|---:|---|---|---|---|
| A 保守 | 128-bit pair、fixed bitmap、1 context、简单双提交队列、共享 SCS；PCCC 可旁路 | 约 1.2 KiB | H67 843238 cycles/frame 代理 | 最少状态和控制；128-bit read 仍有风险 | 低到中 | 必做实现基线 |
| B 平衡 | A + 2 context + block 静态 descriptor + 可旁路 PCCC；event mode 保持关闭 | 约 2.4 KiB | H67 607690、H68 612195 cycles/frame 代理 | 多一份 active/hist 状态；前后端可重叠 | 中 | 推荐受控 RTL 主线 |
| C 激进 | B + union-event/BMRF + 4 context + exact row OOO + 方向 bank mapping | 约 4.8 KiB，另加 compactor/reorder | 4-context 单项相对 2 几乎无收益；其他无模型 | 路由网络、tag、completion 和切换显著增加 | 高 | NO-GO，逐 feature 晋级 |
| D 异构对照 | sparse index core + dense bitmap core + 双 FIFO/stratifier + 共享 SCS | 双前端及队列，容量未定 | 无有限 FIFO 模型 | 复制 datapath，失衡和数据搬运风险高 | 高 | 只作同面积 DSE 对照 |

### 5.1 推荐理由

候选 B 的推荐仅基于以下组合：

- pair-resident 能减少重复 K 搬运并统一 H67/H68；
- 2-context 在当前模型中有明显收益，而 4/8-context 边际收益消失；
- block 异质性足以支持静态 descriptor；
- 共享 SCS 避免按 12 个 block 复制后端。

推荐不包含“event mode 一定省能”“PCCC 一定两倍减写”或“BMRF 一定优于 popcount”的预设。

## 6. 与已有工作的差异和不可声称边界

| 既有工作 | 已有机制 | TESSA 可辨识差异 | 不可声称 |
|---|---|---|---|
| Bishop，ISCA 2025 | TTB、density stratifier、dense/sparse core、BSA/ECP | TESSA 主线为同构 pair 充分统计量与 exact class/SCS 语义，不使用 ECP，不删除 token | 首次 TTB、首次 SNN 异构核、首次密度分流 |
| LoAS，MICRO 2024 | 时间内层并行、single-bit 压缩、dual-sparse join | TESSA 固定 T=2，直接生成 H67/H68 score 所需七个统计量并接 class commit | 首次 temporal parallel、首次时间打包或事件索引压缩 |
| FuseMax，MICRO 2024 | attention 算子融合、片上数据流和负载均衡 | TESSA 融合对象是 pair score、K-zero class、SCS denominator 和 gated-K，且具有事件光流 block 分布 | 首次 fused attention、首次多 engine/context 负载均衡 |
| FLAT，ASPLOS 2023 | attention 数据流重排和中间流量压缩 | TESSA 的驻留对象是 `{Q0,Q1,K0,K1}` temporal pair，输出 class-stationary exact 语义 | 首次 attention dataflow fusion 或首次片上驻留 |
| 复旦 ISSCC 2023 | 面向静态剪枝权重的 butterfly zero skipper、CIM 数据分发 | BMRF 候选处理动态 4-bit membership，并尝试与统计归约/PCCC 共享，保留 bitmap fallback | 首次蝶形网络、把蝶形用于 SNN 即为原创、沿用其芯片收益 |
| C-Transformer，ISSCC 2024 | 同构可重构数据通路应对动态分布 | TESSA 统一的是 bitmap/event 到同一充分统计量合同 | 首次 homogeneous reconfiguration |
| Softermax/I-ViT | base-2 归一化、Shiftmax/整数 Softmax | TESSA 的增量是 K-zero 最终 score class multiplicity 与 gated-K 语义 | 首次 Shiftmax、首次 base-2 归一化 |
| ISSCC 2022 approximate OOO | 稀疏预测和乱序计算 | TESSA 仅允许独立 row 的 exact tag/completion 调度 | 首次 OOO，或把近似工作改成 exact 就视为完整新颖性 |

ANN 机制迁移到 SNN 可以构成设计空间来源，但不能单独构成 novelty。投稿 claim 必须落在新的数值合同、跨模块联合数据流、可证伪收益和完整实现上。

## 7. 性能风险

1. **供数瓶颈。** 81-cycle 前端依赖 128-bit/拍 SRAM 或等效双 bank；否则 pair 布局只省总 bit，不一定省周期。
2. **commit 瓶颈。** 每 pair 两个结果可能超过 active/hist 单写口能力，现有模型未计冲突。
3. **前端成为固定瓶颈。** H67 pair front work 为 544320 cycles/frame，已经高于 backend mean 298918；更多 context 无法突破固定前端。
4. **尾延迟未建模。** 旧 profile 只有 block 聚合值，无法证明 p99 context occupancy、FIFO 深度和 barrier drain。
5. **局部收益不能外推。** attention-only 代理不包含 projection、ATLIF、residual、skip 和 decoder，不能直接写整网 speedup。

## 8. 存储与接口风险

| 存储 | 逻辑容量 | 主要风险 |
|---|---:|---|
| active-entry/context | 162x56 = 9072 bit | 小宏利用率、单双写口、同步读延迟、replay 冲突 |
| histogram/context | 35x8 = 280 bit | 更适合 flop 还是 SRAM 未定；`+2` 和双 class 更新旁路 |
| occupied bitmap | 35 bit | 清零与 context 生命周期 |
| row state/tag/context | 约 128-256 bit + 64-96 bit | reset、flush、completion 一致性 |
| pair source | 容量未冻结 | 128-bit 宽口、ping-pong、bank mapping 和上游写入 |
| encoder state/skip | 容量未冻结 | 93 ATLIF 状态、block residual、S0/S1/S2 长生命周期 |

单 context 的主要私有状态约 1.2 KiB，2-context 约 2.4 KiB，4-context 约 4.8 KiB。这只是逻辑容量；SRAM 宏向上取整和外围电路可能主导面积。正式 memory map 还必须定义地址、端口、读写时序、初始化、ECC/parity、MBIST 和 backpressure。

## 9. 面积与功耗风险

### 9.1 面积

当前没有可报告的 mm2。面积必须分为：PESF/BMRF、score/RNE/PCCC、context/control、SCS、standard-cell memory、SRAM macro、clock/DFT 和 15%-20% 集成裕量。Yosys generic cell 与被打散的数组不能替代目标工艺面积。

高风险项：

- 128-bit SRAM 外围和宽 mux；
- 双写 active/hist 端口或多 bank；
- BMRF 五级网络和 event/bitmap 双格式转换；
- 4-context 的宏向上取整；
- OOO completion/reorder 状态。

### 9.2 功耗

当前 spike energy 不是芯片功耗，不能覆盖 SRAM、clock、Motion-XOR、PCCC、SCS 和控制。正式功耗必须使用真实 trace 产生 SAIF，并分列 clock、组合、时序、SRAM 动态和漏电。

以下只表示门控机会，不是 activity factor 或 mW 估计：

| 域 | workload 机会代理 | 初步门控判断 | 正式证据缺口 |
|---|---:|---|---|
| pair payload/read | pair-empty 73.90% | 高；仅早到 metadata 时可跳读 | metadata 生成与 SRAM SAIF |
| Motion-XOR 后级 | motion-zero 83.18% | 高 | motion metadata 到达位置和门控后切换 |
| active bank write | active token 约 11.35% | 高 | 写口、FIFO 和地址切换 SAIF |
| histogram/PCCC | K-zero 83.11% | 高活动，不应误判为可关 | collision、merge 和真实写事务 |
| shared SCS | backend/front 工作比约 0.55 | 中等机会，粗粒度 | ordered occupancy 和时钟 enable |
| 第二 context | occupancy 未知 | 未定 | ordered trace 与空闲周期 |

正式 `clock_power_budget` 仍缺每域频率、SAIF activity factor、clock mW 和 ICG 覆盖率，因此功耗签核为 NO-GO。

## 10. 风险登记表

| ID | 风险 | P | I | 分数 | 缓解措施 | 责任角色 |
|---|---|---:|---:|---:|---|---|
| R01 | 目标工艺、频率、面积和功耗预算缺失 | 5 | 5 | 25 HIGH | 冻结约束表、PDK/library/PVT 和预算 | PI/架构/综合 |
| R02 | 128-bit/拍和双 commit 端口假设使周期模型过于乐观 | 4 | 5 | 20 HIGH | ordered trace + 有限 FIFO/bank 模型 + SRAM 端口 DSE | 架构/存储 |
| R03 | 小深度 context memory 的宏向上取整抵消 2-context 收益 | 4 | 4 | 16 HIGH | SRAM compiler 与 flop/macro 分界 DSE | 物理/综合 |
| R04 | BMRF 复制既有蝶形思想且网络 PPA 无净收益 | 4 | 4 | 16 HIGH | 与 prefix/bitmap 同约束比较，不达 15% EDP 即删除 | 架构/综合 |
| R05 | PCCC merge 率不足或旁路控制造成频率下降 | 4 | 4 | 16 HIGH | 真实 collision profile、可旁路实现、单/双端口对照 | 架构/RTL |
| R06 | 只实现 attention row，却宣称 full encoder accelerator | 4 | 5 | 20 HIGH | 冻结论文 top 边界，补 projection/ATLIF/residual/skip 系统模型 | PI/架构 |
| R07 | 多 context、flush、barrier 和反压引发状态空间爆炸 | 4 | 4 | 16 HIGH | 从 1-context formal 建立，再做 2-context composition | 验证/RTL |
| R08 | 新软件候选改变 attention block 内部语义 | 3 | 4 | 12 | 保持 pair/PESF/SCS 接口稳定，新算子必须重新 golden/profile | 算法/架构 |
| R09 | block 静态策略对 row burst 无适应性 | 3 | 3 | 9 | ordered trace 后再决定是否 exact OOO | 架构 |
| R10 | 文献组合被视为显然拼接 | 4 | 5 | 20 HIGH | 以联合数值合同、消融和完整 PPA 证明非显然收益 | PI/全体 |

在 R01-R07、R10 没有缓解证据前，不允许正式架构签核。

## 11. 架构创新是否足够 DATE

### 11.1 当前状态

**当前还不够。** 现阶段的 TESSA 是有 workload 支撑的架构提案，但没有 TESSA RTL、目标工艺 PPA、SRAM 端口结果、ordered trace 或全 encoder 系统模型。它比既有 row-engine 微结构更接近架构贡献，但还不能作为“已经完成的 DATE 架构创新”。

### 11.2 达到 DATE 可辩护水平的最小组合

不要求强行加入异构双核或 BMRF。更克制且可能成立的贡献组合是：

1. **pair-resident exact sufficient-statistics dataflow**：在一个驻留周期内联合形成 H67/H68 两时间片 score 所需统计量，消除重复 peer-K 搬运；
2. **pair-to-class 融合提交**：将 K-zero 双 score 在 active SRAM 之前按最终 class 合并，并与 exact SCS denominator/gated-K 后端融合；
3. **block-aware homogeneous context execution**：用同一核覆盖 12 个 block 的强异质性，以 2-context 隐藏前后端不均衡，并证明优于同面积单 context 和双核对照；
4. **系统证据**：在同一 hardware-order golden、SRAM、频率和真实 trace 下给出 cycle、area、power、EDP、bit-exact 和 encoder 级影响。

这四项中的前三项必须形成一个不可拆散的联合数据流，而不是把 pair layout、histogram、多 context 分别包装成首次。若相对 fixed-bitmap pair 基线只有很小净收益，或者只在单个 block 有效，则 DATE 创新仍不足。

### 11.3 建议的可证伪门槛

| 结论 | 最低门槛 |
|---|---|
| pair-resident 成为贡献 | 相对 162-token 基线，真实 SRAM transaction 和 cycles 均下降至少 25%，Fmax 下降不超过 10% |
| 2-context 成为架构点 | 相对 pair 1-context，真实 trace 吞吐提高至少 20%，新增 context/control 面积低于 engine 15% |
| PCCC 成为微架构贡献 | histogram 写事务至少下降 2 倍，merge 率至少 40%，净能耗改善且 bit-exact |
| TESSA 成为主架构贡献 | 相对公平 serial/pair 基线，含 memory/control 后 attention EDP 至少改善 15%，并报告 encoder 级结果 |
| BMRF 晋级 | 相对 prefix/fixed bitmap EDP 至少改善 15%，满足目标频率；否则删除 |
| 异构核晋级 | 相对同面积 homogeneous TESSA EDP 至少改善 15%，且 p99 FIFO 失衡可控 |

## 12. 进入 RTL 的 GO/NO-GO 清单

### 12.1 当前决策

| 范围 | 决策 | 说明 |
|---|---|---|
| 候选 A fixed-bitmap pair datapath | **CONDITIONAL GO** | 可用于关闭接口、正确性和端口模型 |
| 候选 B 2-context 骨架 | **CONDITIONAL GO** | 必须可参数退化为 1-context，PCCC 可旁路 |
| union-event PESF | **NO-GO** | 缺真实 packet/转换/能耗统计 |
| PCCC 强制开启 | **NO-GO** | 缺同 class/collision 与写事务证据 |
| BMRF | **NO-GO** | E0，缺 profile 和 PPA |
| 4-context/row OOO/方向 bank | **NO-GO** | 无可测净收益，4-context 已显示边际收益极低 |
| 异构双核 | **NO-GO** | 只保留同面积模型对照 |
| 正式 architecture-to-RTL handoff | **NO-GO** | 接口、约束、memory map、clock/reset/DFT 未完整 |

### 12.2 探索性 RTL 开始前必须关闭

- [x] 冻结 TESSA top 为 encoder-attention subsystem；encoder模型完成前论文口径限定为attention accelerator。
- [x] 冻结 pair input 合同：128-bit直读、2x64-bit双bank和单64-bit fallback，valid/ready、row/pair tag以及第一版内部派生metadata。
- [x] 冻结每 pair 双结果 commit 合同：active/hist独立单写口、两个depth-2 queue、原子准入、bypass、反压和completion条件。
- [x] 冻结 hardware-order golden：score/RNE、35/3 class、exp2 LUT、denominator、Q1.7 gate、gated-K。
- [x] 冻结 1/2/4-context 逻辑 memory map、同步active-bank读延迟和生命周期；首版实现1/2，PCCC可旁路。
- [x] 冻结 64-bit block descriptor 最小字段；第一版只允许in-order，不允许动态OOO。
- [x] 冻结候选A/B的trace/cycle计数接口，覆盖供数、双commit、merge、FIFO、context、SCS、输出和barrier stall。

上述七项已由 `docs/58_TESSA模块接口存储与RTL前规格.md` 和 `spec/tessa_attention_subsystem_spec.json` 关闭，因此候选A/B可以进入探索性RTL module planning。它们不等于第12.3节的正式handoff，也不等于architecture sign-off。

### 12.3 正式 RTL handoff 前必须关闭

- [ ] ordered profile100 完成，得到 pair/row burst、PCCC collision、union packet、bank conflict 和 1/2/4-context occupancy。
- [ ] 目标 `clk_mhz`、`area_um2`、`power_mw`、工艺、PVT、VDD 和目标 FPS/latency 冻结。
- [ ] SRAM compiler 或可审计宏模型确定 active、pair、hist 和 encoder state 的端口/时序/面积/能量。
- [ ] 性能模型加入双 commit、有限 FIFO、SRAM wait、输出反压和 block barrier，并对目标保留至少 10% margin。
- [ ] clock/reset/CDC/RDC、ICG、scan/MBIST 和 error 处理策略冻结。
- [ ] 验证计划覆盖 1/2 context、任意反压、flush/reset、class 边界、无丢失/重复/死锁和 representation fallback。
- [ ] 所有 HIGH 风险有负责人、缓解方案和关闭证据。

### 12.4 architecture sign-off 前必须关闭

- [ ] 候选 A/B/C/D 在同库、同频率、同 SRAM、同 trace 和同接口下完成公平对照。
- [ ] DC 报告 Fmax/WNS/TNS/logic area，SRAM 报 macro area 和访问能量，SAIF 报动态/漏电分项。
- [ ] 面积和功耗分别低于预算 80%，漏电低于总功耗 15%。
- [ ] 高门控机会域的 ICG 规划覆盖至少 60% 相关寄存器位，并形成正式 `clock_power_budget`。
- [ ] RTL 对 hardware-order golden 0 mismatch，LEC/Formality 和 post-synthesis smoke 通过。
- [ ] 若论文称全 encoder，加上 projection、ATLIF、residual、S0/S1/S2 skip 的周期、存储和功耗；否则标题和 claim 限定为 attention accelerator。
- [ ] BMRF、OOO、方向 bank、异构核中未过门槛的功能从摘要、贡献和最终 RTL 配置删除。

## 13. 最终决策记录

1. 冻结 **H67 功能超集 + H68 编译期特化**。
2. 冻结 **temporal-pair 逻辑布局和 fixed-bitmap exact 合同**，不冻结“一拍一个 pair”的物理承诺。
3. 冻结 **首版启用 2-context、接口参数化 1/2/4 的骨架**，最终物理数量等待 ordered trace 和 SRAM DSE。
4. 冻结 **block 静态 descriptor**，关闭 row OOO。
5. PCCC 以可旁路 feature 进入规格，不作为已证明收益。
6. BMRF、event representation、方向 bank 和异构双核不进入第一版 RTL。
7. TESSA 当前可以开始受控原型，但 architecture sign-off、论文 PPA 和 DATE 架构贡献均保持 NO-GO，直到第 12.3 和 12.4 节关闭。
