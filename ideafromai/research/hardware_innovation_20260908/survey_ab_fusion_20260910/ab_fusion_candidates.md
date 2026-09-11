# A+B 鼎汇/鼎勘式融合候选表（2026-09-10）

**性质**：第二队列调研设计；**不抢** lifting40 Stage B（ordinary dense/raw vs `fast_raw_diagonal` same-port/same-state/same-backpressure schedule compare）。  
**证据纪律**：绝对 AEE≤1.259 已过；相对 dense/raw ΔAEE≤+0.005 **已失败**（+0.013178）；同资源净服务% **未知**。融合不得假装晋级门已过；借入 Gustav / da4ml / 普通压缩 ≠ 自称标题级 X；shared-Q 仅消融；负结果只停列出的布局，**不杀** Prosperity/Gustav/lifting 全家；禁动 nts07 / main.tex / 性能 RTL；本文件不做实验。

**相对旧 AB_STACK（09-08）**：下列为**新组合或新杀门**（挂点改到 lifting/r1 残差链、半步 RNE、联合图挂源 DAG、末5部分合并等），非复读 s2b3 C16 / 旧 C192 清单。

---

## 总览表

| ID | 一句话 B | 与 Stage B 关系 | 建议排队 |
|---|---|---|---|
| F1 | 在 **敏感 patch r1** 上，按 lifting 改变后的源活动 + 双 PED/门消费者误差，选可共同删的物理源字 | **Stage B 后可挂**；**依赖 Stage B 净服务结论**（先确认 lifting 岛是否值得叠剪枝） | **优先试设计 #1** |
| F2 | 在 **lifting 半步/RNE 检查点** 上做共享请求组「接受/继续」有损共同完成 | **第二队列**；依赖 Stage B 是否显示并集/存活仍是瓶颈 | **优先试设计 #2** |
| F3 | 把 Prosperity **联合图/公共虚节点+部分 lane** 挂到 lifting **源 DAG**（非整 W 产品 mask） | **第二队列**；不抢 Stage B | 备选 |
| F4 | **仅末5门前推** 的部分合并 + 轻量误差证书（非已停的全合并） | **Stage B 后可挂**（若 RNE/端口吃掉 159 节点优势） | 备选 |
| F5 | 训练约束 **lifting 中间量/广播组** 的依赖生存期，使有限 RF 在完整 T10+PED 消费后周转 | **依赖 Stage B 净服务结论** | 条件 |
| F6 | **有界 clip 严格门完成** × lifting 可取消时间列（残差链多消费者并集） | **第二队列** | 备选 |
| F7 | Gustav CPTB/NRV 物理链上，按 **lifting 因子组** 对齐的源∩W 打包（借入 Gustav≠X） | **Stage B 后可挂**（Stage B 先用 Gustav 部件作同分母） | 条件 |

---

## F1　lifting 源活动 × 敏感 r1 消费者误差结构剪枝

- **B**：在 patch r1 残差链上，用完整 T10 门 + 真实 PED/门双消费者损失，在固定广播域内选可**共同删除**的物理源字（含 H 重排同步系数/阈值），使剪枝结果适合共享供数且保住隐藏通道表达。
- **最强对照**：同分母下打赢（1）ordinary dense-source/raw 同预算窄稠密/hidden50 式减宽；（2）完整 HiNM（含二阶）/VENOM-CRISP 两级结构；（3）幅值/活动感知 C16；（4）若 Stage B 后 lifting 仍在主岛，则同 lifting 无剪枝端点。**禁止**只回 s2b3 不敏感块打磨。
- **完整 A 继承**：lifting 结构化 T10 PSN 执行对象与常量编译图；残差链 source→sn1→Conv→sn2→PED 双消费者与 finite 端口/背压合同；Gustav NRV/源∩W/局部状态作供数分母；da4ml 整图 CSE 与普通 2:4/量化同等权限；Prosperity「产品费用」仅作 mask 选择强控制，不自称 X。
- **一句话 X**：相对 HiNM/窄稠密，差分是「按 lifting 改变后的源活动形状 + 非因果 T10×双 PED 消费者联合扰动」决定可共同删的物理字，而非只换损失名或砍不敏感 FC1。
- **固定杀门**：同槽同训练预算下不胜 HiNM 或同服务窄稠密；或只降 W² 不降物理源字/门/流误差；或 valid 探索 AEE>1.259 或相对同预算强对照 ΔAEE>+0.005；或并集仍读满且 Stage B 分母下无周期余量。**只停该 r1 剪枝布局**，不杀 lifting/Gustav/剪枝家族。
- **与 lifting Stage B 关系**：**Stage B 后可挂**；**依赖 Stage B 净服务结论**（若 lifting 相对 ordinary 无 ≥10–15% 整段潜力，则本叠层改挂 ordinary 敏感 patch 或降优先，仍不抢 Stage B）。

## F2　lifting 半步 RNE 检查点 × 共享组有损共同完成

- **B**：在 lifting 半步写回/必要 RNE 边界设固定检查点，用已付费的部分状态预测整组残差链 T10 门字，**一组接受则发预测 θg（含预测非零）并退休请求，否则继续原算**。
- **最强对照**：完整 Gustav 式共享请求同步（无预测）；SparseInfer/BitFair 式独立每神经元预测 + **相同**组关闭逻辑；同精度静态窄层；lifting raw 无预测端点；禁止用严格后缀界负结果冒充有损预测已死。
- **完整 A 继承**：lifting 半步图与 35 中间 RNE 语义；残差链真实消费者与 θg/τ 分离；Gustav 共享请求/NR4/有限 S；SparseInfer「预测后跳权→未跳完整执行→真实零补跳」与 BitFair 学习终止链作完整底座；finite_service 事件依赖与单资源互斥可复用边界。
- **一句话 X**：相对「s2b3@C192 独立/组预测」，差分挂在 **lifting 半步自然边界** 与 **r1 多消费者并集费用**，使共同完成对齐结构化写回点（借入预测链≠X）。
- **固定杀门**：净服务不胜同精度静态窄层与「独立预测+同组关闭」；或预测/摘要开销≈再做一次 PSN；或接受后 AEE 爆（探索门 1.259 / 相对 +0.005）；或组尾仍被最慢消费者钉死且无请求下降。**只停该检查点/组费用布局**。
- **与 lifting Stage B 关系**：**第二队列**；若 Stage B 显示瓶颈不在共享并集/状态存活而在纯算术/RNE，则降优先或改 F4。

## F3　Prosperity 联合图挂 lifting 源 DAG

- **B**：保留 Prosperity 原森林语义，在 lifting 已编译源 DAG 上同时选择**公共虚节点、部分 lane 继承与有限供权包**，做同端口计划选择（不是再扫整 W 产品 mask）。
- **最强对照**：ordinary dense 源图 + 同等联合图权限；纯 da4ml CSE / 空残差别名；本轮已收口的「产品费用选 mask 相对活动仅 −0.09045%」控制；lifting raw 无联合图。
- **完整 A 继承**：Prosperity 模式复用/部分和先验；Gustav 供数与有限计划接口；lifting 159+35 图与 ordinary 260 图同分母；残差链消费者到达与背压；本轮三探针（W 父关系 / 部分 lane / mask 目标）作负结果边界，不扩成家族失败。
- **一句话 X**：相对「在整 W 上选共同 mask」，差分是把产品式共享挂到 **lifting 结构化源图节点**，用公共虚节点吃重复子式与 lane 跳过，而非再证明 0.09% 级 mask 目标。
- **固定杀门**：同端口下计划服务不优于 da4ml CSE+普通部分 lane；或相对 lifting 无图端点净省 <5% 且精度不守门；或引入环/多父等待使背压变差。**只停该联合图挂源 DAG 布局**；Prosperity/Gustav 家族保留。
- **与 lifting Stage B 关系**：**第二队列**；不抢 Stage B（Stage B 仍是无 Prosperity 的 pure schedule compare）。

## F4　末5门前推部分合并 + 轻量证书

- **B**：仅对 lifting 已允许门前推的末5输出做跨 RNE 近似合并，并附**轻量误差证书/不确定则回退原半步图**；禁止复活已停的「全半步合成纯门图」。
- **最强对照**：分阶段 159+35 RNE 原图；已停全合并 241 加减布局（负对照）；ordinary 门出口 cutoff；同资源无证书的盲目合并。
- **完整 A 继承**：lifting 常量编译与末5 `write_time_indices` 门前推权限；ordinary 精确 cutoff 同等权限；gate_collapse 探针的负结果纪律；finite 回退路径与消费者一致性检查。
- **一句话 X**：相对全合并失败，差分是 **范围缩到末5 + 显式证书/回退**，只在证书费用后仍有算术与端口余量时成立。
- **固定杀门**：含证书/回退后加减或 issue 不低于 159+35 有效成本；或新网络函数未评即报 AEE；或回退率高导致净服务≤原图。**只停该部分合并布局**；不杀纯门/lifting 家族。
- **与 lifting Stage B 关系**：**Stage B 后可挂**——仅当 Stage B 分项表显示 **RNE/转发** 是吃掉节点优势的主因时升优先。

## F5　Avalanche 式依赖生存期 × lifting 中间量释放

- **B**：训练/掩码时约束沿公共供数顺序「同时未完成的广播组数」，使 lifting 中间量与 S 在**完整非因果 T10+PED 消费后**才释放，降低有限 RF 下的重读/译码。
- **最强对照**：同 F_cache/端口下固定双组配对；普通滚动择优（静态已见 ~0.07% 级薄增量）；hidden50；Avalanche 完成回收原样迁移无训练改图。
- **完整 A 继承**：Avalanche 排列/完成写出/复用保留；Gustav F_live 与 NR4；lifting 中间写回与 35 RNE 保活语义；残差链双消费者背压；静态两活探针上界先算再训。
- **一句话 X**：相对「SpMM last-use 回收」，差分是把 **非因果 T10+PED 延长占用的 lifting 中间生存期反向写入训练目标**（借入 Avalanche≠X）。
- **固定杀门**：乐观上界（源读/译码免费）已无 ≥5–10% 总服务余量则**不训练**；训后同 AEE 不胜固定配对/窄稠密；或只减局部字数不减整段 r1→PED 服务。**只停该生存期训练布局**。
- **与 lifting Stage B 关系**：**依赖 Stage B 净服务结论**（先看 RF/重读是否真瓶颈）。

## F6　有界 clip 严格门完成 × lifting 可取消列

- **B**：将 BN 后输入范围纳入学生定义，用严格区间证明门 0/1 提前完成；仅当请求组内依赖某 lifting/源时间列的消费者全部完成后，取消该列卷积生产。
- **最强对照**：同裁剪模型的普通逐门严格界；整字退休；PACT 式裁剪恢复；SpikeX 块占用费用训练；ordinary dense 列取消率；已有 clip 825 探索轴（勿把验证选 γ 当 held-out）。
- **完整 A 继承**：残差链 r1 捕获与 Conv1/2、BN、shortcut、粗头；lifting 或 ordinary 源时间结构（两轴都可挂，优先 lifting 后）；finite 请求组与逻辑向量计费边界；θ 幅值不参与「是否可取消」偷换。
- **一句话 X**：相对 SpikeX/逐门短路，待证的是训练范围后**取消是否在 lifting 列结构上足够集中**，使有限状态实现偿还比较/mux 费用。
- **固定杀门**：同精度下取消率/净服务不胜整字界+PACT；或范围收紧致 AEE 破 1.259 / 相对 +0.005；或比较+重算吃掉活项节省。**只停该 clip×取消布局**。
- **与 lifting Stage B 关系**：**第二队列**；与 F1 互斥并行时优先 F1（剪枝改供数面通常更大）。

## F7　Gustav 物理供数 × lifting 因子组对齐打包

- **B**：在补齐 CPTB/NRV/源∩W 物理链时，使源字/行打包边界与 lifting 因子组/半步写回对齐，减少跨组拆包与重复 NRV 行（Stage B 先用未对齐 Gustav 部件作公平分母）。
- **最强对照**：同一 Gustav 链 + ordinary dense 源序打包；逻辑 NRV 计费（非物理）；lifting 无对齐的通用双指针。
- **完整 A 继承**：Gustav GP/CPTB/NRV/局部膜与 2×4 叶边界；lifting matchings/半步图；implementation_inputs 中 finite_service / resident 可复用与不可继承清单；残差链真实 θg。
- **一句话 X**：相对「完整接入 Gustav」，差分仅是 **打包/行导航与 lifting 结构共设计**；Gustav 本身仍是底座 A。
- **固定杀门**：对齐后物理字/周期不优于 ordinary 序+同 Gustav；或只改善逻辑 NRV 行代理无端口闭合收益；或破坏同端口公平（给 lifting 多 bank）。**只停该对齐打包布局**；不杀 Gustav 家族。
- **与 lifting Stage B 关系**：**Stage B 后可挂**；Stage B 期间 Gustav 部件只作分母，不把本 F7 写成 Stage B 的 X。

---

## 明确不进本表主线的旧想法（避免复读）

| 想法 | 原因 |
|---|---|
| shared-Q 标题化 | 增量薄；仅消融 |
| Prosperity∪APEC、C2 静态共享、NR4 费用训练旧版、gate 全合并 | 已停布局 |
| 纯抄 Gustav / 纯 PoT 化系数 | 借入≠X；PoT 作强对照 |
| 只在 s2b3 再磨 C16/Gram | 挂点不敏感；已停打磨 |
| 注意力 K=0 整窗、用本网最终 flow 的 OP-STW | 旁路/已否因果版 |

---

## 优先试哪 1–2 个（仍不抢 Stage B）

1. **F1（首选第二队列设计）**  
   理由：唯一同时对准 **最大贵头 patch r1**、**lifting 已改源活动**、与 **双消费者可检验剪枝** 的交叉；杀门与 HiNM/窄稠密清晰；依赖 Stage B 只决定「叠在 lifting 上还是 ordinary 敏感 patch 上」，不占用 Stage B 档期。

2. **F2（次选第二队列设计）**  
   理由：把旧「共同完成」从 s2b3@C192 **迁到 lifting 半步边界**，形成新挂点与新杀门；若 Stage B 显示并集/存活非瓶颈则自然降级，机会成本低。

**最小下一步（调研设计，不开跑）**：为 F1/F2 各写一页「同分母对照清单 + 杀门数字 + 所需捕获/不改生产树」；等 Stage B 净服务分项表出炉后再决定挂 lifting 与否。

---

## 过夜文献回写（148 卡｜其中 arXiv方法 48｜unresolved 56）

- **F1**：补强「删字后索引/对齐税」分项（RISCSparse / Bit* / CFMP 线索）；开源对照仍要。
- **F2**：BitFair 早停/组关闭、LoAS/TASD 时间轴对照；**MP2 已粗验 PASS**。
- **F3**：Prosperity 仍第二队列；新增产品稀疏卡不自动抬标题。
- **F5**：Gustav/FlexSpIM/LoopTree 供数·驻留·占用语言巩固底座。
- **F7**：ConvFormer/Xpikeformer/Spike* Transformer 旁路主岛。
- **Stage B**：MP1 same-port 信用计数 PASS；仍非净服务%。
- **待定位**：56 条标 `unresolved_no_fulltext`，不虚报精读。

不抢 Stage B；不改生产 RTL/main.tex。

