# 十一项新增融合的实际RTL筛选

本批十项均按[事先计划](PLAN.md)写成RTL并实际测量；上轮五项、强控制补齐、背压重跑与组合测试不重复计数。十项最终记录共1427条。随后另做[三项共同底座组合](joint_selected/README.md)，并实际追加第十一项[精确T10方向表示](temporal_direction/PLAN.md)：强控制补齐后，真实八块冷周期135102→136160，新增缩放方向未命中，停止此single-anchor布局。没有把参数扫描当新idea。

十一项最终1655条命令；另138条组合回放。第十一项的228条包含三种新增方向、非零残差与范围回退，[独立审阅](temporal_direction/INDEPENDENT_REVIEW.md)已完成。四组完整825评价均已结束，GPU评估进程已退出。

入口：[范围分开的性能表](PERFORMANCE_TABLE.md) · [机器可读结果](comparison.json) · [汇总脚本](collect_screen.py) · [阶段分析](REPORT.md)。所有周期为隔离Verilator测量；没有本批ASIC PPA、整网FPS或TCAS-II强接收证据。

| 执行方向 | 实际源码、对照和原始结果 |
|---|---|
| 三种算子分解 | [Q1位平面](decompositions/q1_bitplanes/REPORT.md)、[Q2 DA](decompositions/q2_da/REPORT.md)、[完整Q1列字典](decompositions/q1_dictionary/REPORT.md) |
| 三种有损函数 | [整组剪枝](algorithm_sparse/AS1_GROUP.md)、[原型与残差](algorithm_sparse/AS2_PROTOTYPE.md)、[时间保持](algorithm_sparse/AS3_TEMPORAL.md)、[总入口](algorithm_sparse/README.md) |
| 三种数据流 | [取消p存储](dataflow/d1_forward/REPORT.md)、[halo轮转](dataflow/d2_halo/REPORT.md)、[双上下文](dataflow/d3_interleave/REPORT.md) |
| 实际宽链复用 | [消费者宽ALU分段](phase_borrow/README.md) |
| 追加精确时间方向 | [三模式、残差与在线选择](temporal_direction/README.md) |
| 组合相容性 | [同halo/转发底座再开宽链](joint_selected/README.md)；不另计idea |

交叉审阅：[分解](REVIEW_DECOMPOSITIONS.md)、[数据流](REVIEW_DATAFLOW.md)、[有损执行](decompositions/REVIEW_ALGORITHM_SPARSE.md)、[宽链](decompositions/REVIEW_PHASE_BORROW.md)、[十项新颖性](decompositions/REVIEW_NOVELTY_TEN.md)。新颖性分数评价当前增量，不是接收概率；前三项明确标为实现者自评，后七项有独立意见。

[文献来源与未试接口](SOURCES_AND_NEXT_INTERFACES.md)明确实际阅读范围和迁入内容。本批没有复现作者整芯片，不把本地UCNN式归约、LUT-DLA式编码或普通稀疏执行称为作者完整架构。

算法身份仍为AT-LIF `{0,θ}`，本执行挂点θ可吸入权重。质量门采用同环境原NB0 valid825 AEE1.447936665574317；上一阶段R8＋真实整数消费者为1.3276350226079938。本批有损函数各自重新跑网络，不能继承精确模式AEE。统计窗口和硬件整帧回放都只对应所标记的r0组件，未形成整网硬件账本。

复现使用Python3.12，各子目录README给出prepare/build/run/check顺序。SV、C++ TB、数值参考与紧凑结果进Git；生成的fixture、模型权重和构建目录不进Git。生产nts07、主稿与历史冻结文件保持只读。
