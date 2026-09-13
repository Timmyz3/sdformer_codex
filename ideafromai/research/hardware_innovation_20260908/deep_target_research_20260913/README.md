# 昂贵算子的分解与任务稀疏执行

先读[综合报告](REPORT.md)，再看[三条目标与 RTL 工单](TARGETS_AND_RTL_QUEUE.md)。当前优先对象是 patch r0.conv2/r0.conv1 和 decoder2；lifting 保留为实现与质量对照。

| 材料 | 内容 |
|---|---|
| [综合报告](REPORT.md) | 实际昂贵位置、三条候选的 A/B/X、其余家族去向、后续执行顺序 |
| [RTL 队列](TARGETS_AND_RTL_QUEUE.md) | 完整函数和首性能范围；不将 CPU 或小叶功能当整层 RTL |
| [独立评审](INDEPENDENT_REVIEW.md) | 共享整字、普通负载平衡、既有粗细 mask 三类最强反例 |
| [分解专篇](decomposition/REPORT.md) | WINS 有限字母表、整数桶、Kronecker、支撑表，以及各自强 A |
| [稀疏专篇](sparse/mechanism_report.md) | N:M 支撑与 lane epoch、原生源字/halo 需求和实际事务 |
| [光流专篇](flow/flow_task_native_report.md) | 事件表示、可观测性、选择性细化、背景与跨窗口路线 |
| [decoder2 核查](integration/DECODER_TARGET.md) | 真实 sn/deconv/pred 顺序、已有粗流、去插零先验 |
| [文献清单](LITERATURE.md) / [CSV](LITERATURE.csv) | 去重 34 篇：32 篇方法/实现章节精读，FINEA 与 ASNA 两篇仅摘要/元数据 |
| [方法记录](WORKFLOW.md) | 研究范围、交叉评审与程序性方法来源 |

本轮新增执行为二值 Winograd 的 65,536 输入穷举；[结果](decomposition/winograd_binary_exhaustive.json)确认有限取值集合。没有新增训练、网络推理、RTL 或 EDA。三条主候选均未得到新颖性或净性能 PASS；工单已经明确，下一阶段以完整强控制和实际 RTL 逐项验证。

已有性能与质量数字来自[上一阶段](../open_fusion_execution/major_operator_fusions_20260913/README.md)，不得写成这三条候选已测。AT-LIF 使用 `{0,θ}` 可吸收身份；探索精度门为优于 NB0。论文全文缓存保留本地，Git 保存来源、分析和可复跑的代数程序。
