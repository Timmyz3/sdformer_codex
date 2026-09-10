# 我理解的当前工作与缺口（一页）

- **调研落盘**：`survey_ab_fusion_20260910/`（本页只求通透对齐，**不开 A+B 融合脑暴**）
- **证据截止**：主仓 `sdformer_codex` 分支 `autoresearch/neuron-ops-20260507` HEAD=`22fc4aa1`；主战场文档以 README / CURRENT_LINES / lifting40 一页费用与 Prosperity 收口 README 为准
- **论文语境**：面向 **TCAS-II Express Brief** 的光流 SNN Transformer **软硬件协同**；生产树 `nts07`、主稿 `main.tex`、性能 RTL **只读禁止改**；尚未宣称可投稿的标题级 X

## 1. 主战场目标（一句话）

在敏感 patch 残差链（r1）上，用**可学习 lifting 结构化 T10 PSN**（借入 da4ml CSE / Gustav 式供数等底座 ≠ 自称 X）相对**普通 dense-source/raw**，在**同端口 / 同状态 / 同背压**下证明**同资源净服务**优势，并守住精度门；过门后再谈成对恢复与隔离 RTL，而不是先堆文献融合叙事。

## 2. 当前主线 vs 已收口线

| 地位 | 对象 | 状态 |
|---|---|---|
| **下一主实验（锁定）** | **lifting40 Stage B**：ordinary dense-source/raw vs `fast_raw_diagonal` 的 same-port/same-state/same-backpressure **schedule compare** | Stage A（一页净费用合同 + 编译/825/十帧代理账）**已完成**；Stage B **尚未开工**（见缺口） |
| **条件支线** | shared-Q（同一可逆时间坐标） | 仅允许**同框架消融**；未证明额外净硬件收益，**不强留标题** |
| **已收口→第二队列** | Prosperity / Gustav 本轮三探针（W 父关系 / 部分 lane 父 / 共同 mask 选择） | 负结果只停**本轮布局**；**家族保留、勿轻易杀线**；不挤掉 Stage B |
| **共同底座（非标题）** | Gustav NRV/供数、da4ml 整图 CSE、普通低秩/量化/剪枝、fixed-BN 等 | 可继承到同分母；**借入 ≠ X** |
| **旁路** | motion / 注意力 K=0 等 | 有限份额与费用，不排主岛 |

## 3. 关键证据（数字以合同页为准）

| 门闩 / 量 | 结果 | 含义 |
|---|---|---|
| 绝对 AEE ≤ **1.259**（valid825） | **过**：lifting raw `fast_raw_diagonal` **1.232979…**；shared **1.247809…** | 绝对预算过，≠ 全部准入通过 |
| 相对同预算强对照 ΔAEE ≤ **+0.005** | **失败**：vs ordinary dense/raw **1.219801…**，差 **+0.013178…**（上限对照约 1.22480） | 当前端点**不晋级**性能 RTL / 主贡献 |
| source CSE（完整合法图） | dense **260** 加减；lifting raw **159** + **35** 中间 RNE；shared **169** + 40 RNE | 节点数线索，**≠ 周期** |
| 同序十帧代理 | 加权项/H8 逻辑需求相对旧 dense 约少 **~15%** | **NOT schedule-closed**；不可冒充净服务% |
| shared vs raw（全 825 源账） | 加权项 −1.130%，H8 −4.442%，NRV 行 **+0.166%** | 共享增量薄，还欠 BZ/逆/状态账 |
| 一次轻探针 | `gate_collapse_probe` 全合并 241 vs 159+35 RNE → **停该全合并布局** | 负结果不杀纯门全家 |
| Prosperity 收口 | 产品费用选 mask 相对活动控制仅少 **0.09045%** 逻辑项 | 未证明标题级 X；保留未试联合图/有限执行接口 |

**同资源净服务 %：未知**（Stage B 未做）。≥10–15% 整段净服务潜力是晋级/杀门相关判据；在出分项表之前，**既不能宣称 lifting 过晋级门，也不能据此杀全家**。

## 4. 新颖性主张与禁止事项

- **主候选新颖性主张（待 Stage B/后续证明）**：可学习 lifting **结构化 T10 PSN** 作为执行对象与消费者接口差分；单纯快变换 / 常量编译 / PoT 化均为先验，**借入底座 ≠ X**。
- **shared-Q**：仅消融；无净收益则不留共享标题。
- **禁止**：改 `nts07` / `main.tex` / 性能 RTL；把节点比、十帧代理、局部加法减少写成 VCS/PPA；把单一布局负结果扩成家族失败；本调研阶段**抢跑 Stage B** 或开 A+B 融合脑暴。
- **负结果纪律**：只停列出的布局/端点，**不杀全家**。

## 5. 明确缺口清单

1. **未证明**：同资源净服务 ≥10–15%（无 same-port 时间线 / 服务分项表）；相对 ΔAEE≤0.005（已红灯，成对恢复尚未授权启动）。
2. **未开工**：`schedule_compare_same_port/`（主战场顶层与 lifting40 下均**不存在**可交付排程产物）；Stage C（train/quant）与 Stage D（隔离 RTL）正确未开。
3. **未知 / 未齐**：完整 r1→PED 有限服务所需的**两学生逐位置三源 gate 捕获**；RNE/端口/状态吃掉 159 节点优势后的真实瓶颈；PoT/DeepShift 等强对照训练（列为未试控制，非本阶段任务）。
4. **第二队列未试接口**（不抢 Stage B）：Prosperity 联合图/有限计划、完整 Gustav CPTB/NRV 物理链、敏感 patch 结构剪枝等——仅可在 Stage B 后或明确挂第二队列时推进。

## 6. 与「不抢 Stage B」的关系（本调研约束）

- 本目录后续若做 A+B / 文献融合，**只能进第二队列**，或 **Stage B 出净服务结论之后**再挂依赖。
- 当前唯一锁定主实验仍是：**ordinary dense/raw vs `fast_raw_diagonal` 的 schedule compare**；shared 仅同 harness 消融。
- 本页只固化「我理解的当前工作与缺口」，**不提出新融合实验设计**。

## 7. 目录速览（核实）

- 主战场顶层：`README.md`、`CURRENT_LINES_AND_PLAN_20260910.md`、`algorithm/`、`psn/`、`motion/`、`bn_state/`、`decoder/`、`literature/`、`prosperity_gustav_reopen_20260910/`、`sparsity_hardware_next_20260908.md`
- lifting40：`…/projection_chain/fast_temporal_recovery_lifting40/`（含 `net_cost_one_page.md`、`implementation_inputs.md`、825/十帧/编译/gate_collapse）
- **`schedule_compare_same_port/`：未见（未开工）**
