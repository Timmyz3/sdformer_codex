# A+B 融合候选（文献扩量后修订｜F1–F7）

日期：2026-09-10

文献基线：去重 **732**（paper 653 / open_source 60）。约束：不抢 Stage B；借入≠X；负结果只停布局。

## 文献对候选的增删改摘要

| 候选 | 变更 | 文献信号 |
|---|---|---|
| F1 | 保留优先#1；对照补开源 2:4/Wanda/SparseGPT | structured_pruning=16 |
| F2 | 保留优先#2；杀门补条件执行开源对照 | conditional_gate=11 |
| F3 | 降级措辞，联合图仍第二队列 | product_sparsity_psn=37 |
| F4 | 收紧，保持第二队列 | conditional_gate |
| F5 | 保持底座非 X | gustavson=14 |
| F6 | 维持低优先 Stage B 后可挂 | — |
| F7 | 显式旁路不进主岛 | sparse_transformer_hw=14 |
| F0 | 新增流程闸：设计页必须点名可跑开源对照 | open_source=60 |

## F1 — lifting 源活动 × 敏感 r1 消费者误差结构剪枝

| 项 | 内容 |
|---|---|
| **B** | 在 patch r1 上，按完整 T10 门 + PED/门双消费者损失，选广播域内可共同删的物理源字（H 重排同步）。 |
| **最强对照** | HiNM / 窄稠密 / 幅值 C16 / lifting 无剪枝；开源侧补 Wanda/SparseGPT/2:4 同槽控制（借入≠X）。 |
| **完整 A 继承** | lifting 结构化 T10 PSN、残差链 source→PED、Gustav NRV/源∩W、da4ml CSE；Prosperity mask 仅强控制。 |
| **一句话 X** | 相对 HiNM/窄稠密：用 lifting 改变后的源活动 + 双 PED 联合扰动定可共同删字（改损失名≠X）。 |
| **固定杀门** | 不胜 HiNM 或同服务窄稠密；AEE>1.259 或 ΔAEE>+0.005；并集读满无余量 → 停该布局，不杀全家。 |
| **与 Stage B 关系** | Stage B 后可挂；依赖净服务结论。 |
| **文献修订注** | 结构剪枝文献+开源工具链支持可复现强对照。样本：HighLight: Efficient and Flexible DNN Acceleration with Hierarchical S / Toward Efficient Permutation for Hierarchical N:M Sparsity on GPUs / FlexHiNM-GP: Flexible Hierarchical Pruning via Region Allocation and C |

## F2 — lifting 半步检查点 × 共享组有损共同完成

| 项 | 内容 |
|---|---|
| **B** | 在 lifting 半步/RNE 边界预测整组门字，一组接受/继续；接受发预测 θg（含非零）。 |
| **最强对照** | 静态窄层；独立预测+同组关闭；开源 early-exit/gated 作方法学对照（不搬 LLM serving 标题）。 |
| **完整 A 继承** | lifting 半步图与 35 RNE、Gustav 共享/NR4、SparseInfer+BitFair、θg/τ 分离与 finite 互斥。 |
| **一句话 X** | 相对旧 s2b3@C192：挂点改为 lifting 半步自然边界 × r1 多消费者并集费用。 |
| **固定杀门** | 不胜静态窄层与独立预测+同组关闭；预测费≈再算 PSN；AEE 破门 → 停该检查点布局。 |
| **与 Stage B 关系** | 第二队列；Stage B 显示瓶颈非并集则降级。 |
| **文献修订注** | 条件执行文献量大但本地负结果仍在，故不升主实验。 |

## F3 — Prosperity 产品稀疏 × 有限联合图

| 项 | 内容 |
|---|---|
| **B** | 第二队列试 Prosperity 产品 mask 与有限计划接口联合图，只对敏感 patch 计费。 |
| **最强对照** | 活动控制 mask；普通 2:4；无联合图 Prosperity。 |
| **完整 A 继承** | Prosperity 底座、本地 G4 负结果纪律、Gustav 供数分母。 |
| **一句话 X** | 仅当联合图同分母净服务稳定胜过活动控制才讨论标题；当前未证明。 |
| **固定杀门** | 相对活动控制逻辑项收益 <1% 或 AEE 破门 → 停该联合图（家族保留）。 |
| **与 Stage B 关系** | 第二队列；不抢 Stage B。 |
| **文献修订注** | 无新开源抬升标题证据。 |

## F4 — 门控生存期 / 末段证书式跳过

| 项 | 内容 |
|---|---|
| **B** | 对 r1 消费者尝试短生存期门控或末段证书跳过，费用记入并集。 |
| **最强对照** | 无证书全算；固定 mask；独立 per-consumer skip。 |
| **完整 A 继承** | 残差链消费者、finite 边界、历史 common3 负结果。 |
| **一句话 X** | 只有证书验证费+跳过在同端口净胜全算才算差分；证书本身≠X。 |
| **固定杀门** | 不胜无证书或验证费吞收益或 AEE 破门 → 停该布局。 |
| **与 Stage B 关系** | 第二队列 / Stage B 后可挂。 |
| **文献修订注** | arXiv 条件执行增量多，同分母未证，不升优先。 |

## F5 — Gustav CPTB/NRV 对齐打包（底座）

| 项 | 内容 |
|---|---|
| **B** | Gustav 供数/NRV 与 lifting 源字打包对齐，只追求执行接口一致。 |
| **最强对照** | 未对齐 Gustav；普通稠密供数。 |
| **完整 A 继承** | GustavSNN 全文与本地迁移、公开镜像。 |
| **一句话 X** | 无标题 X（显式借入底座）。 |
| **固定杀门** | 对齐不降并集费用或不改 Stage B 结论 → 停当标题企图，保留底座。 |
| **与 Stage B 关系** | 底座维护；不抢档期。 |
| **文献修订注** | gustavson 桶 + 开源镜像已盘点。 |

## F6 — clip / 取消边局部改造复测

| 项 | 内容 |
|---|---|
| **B** | 仅 Stage B 后，对曾失败 clip/取消边做最小复测，换 lifting 后源统计。 |
| **最强对照** | 无 clip；旧失败配置。 |
| **完整 A 继承** | 历史负结果纪律、lifting 新源活动。 |
| **一句话 X** | 换源统计后稳定胜过旧强对照才讨论；否则不升格。 |
| **固定杀门** | 再失败 → 停该局部改造，不杀 lifting 全家。 |
| **与 Stage B 关系** | Stage B 后可挂（低优先）。 |
| **文献修订注** | 无新文献强制翻案。 |

## F7 — 稀疏注意力 / Token 门（旁路）

| 项 | 内容 |
|---|---|
| **B** | 旁路调研：token/注意力稀疏是否能在解码器侧省并集费用。 |
| **最强对照** | FlashAttention/稠密注意力；vLLM 类比不作芯片证据。 |
| **完整 A 继承** | 本地 motion/注意力 K=0 有限份额。 |
| **一句话 X** | 解码器侧可检验省费用且不影响 r1 主岛精度；搬 LLM serving ≠X。 |
| **固定杀门** | 主岛无增益 → 停进主贡献。 |
| **与 Stage B 关系** | 不抢 Stage B；旁路。 |
| **文献修订注** | 开源极多，显式降权防诱拐课题重心。 |

---

## 优先 1–2

1. **F1**（Stage B 后可挂）
2. **F2**（第二队列）

主实验仍是 ordinary dense/raw vs `fast_raw_diagonal` same-port schedule compare。
