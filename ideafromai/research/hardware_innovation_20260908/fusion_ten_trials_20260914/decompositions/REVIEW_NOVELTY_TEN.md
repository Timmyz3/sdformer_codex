# 十项接口的新颖性差分审阅

2026-09-14。评分只衡量**当前已实现、能与强A区分的方法贡献**，不是接收率、性能分或实现成熟度。1–2分表示主要迁入已知执行机制；3分表示有明确本地接口增量，但尚无足够一般化机制/消融支撑标题X；4–6分需要更强的差分证据；7分以上需要显著方法推进。本表前三项为**实现者自评**，后七项为**独立意见**；“A”是借入的已知机制与本地强控制，不声称复现整篇论文/整芯片。

| 项目、审阅身份 | A与实际X差分 | 分数 /10 | 是否主要为执行底座；当前证据 |
|---|---|---:|---|
| [1 Q1 bitplanes/bitmap](q1_bitplanes/REPORT.md)，自评 | AND-popcount、符号位减法属于bitserial A。完整原生源变bitmap及40位置调度是本地适配；未出现超出常规布局转换的新表示。 | 2 | 是。真实八块比最新双P/cachedOS慢18.62%；4320B bitmap与40路bit写已付。负点只停此固定布局。 |
| [2 Q2两组R4 DA](q2_da/REPORT.md)，自评 | 分组LUT、位枚举、最高符号位修正、最小有效宽度属于DA A。全R8驻留Q2下实际构表/查表是完整化证据，未建立新的X。 | 2 | 是。真实八块慢14.21%；消除乘法不等于减少周期。构表与位置展开成本已闭合。 |
| [3 Q1完整列32类字典](q1_dictionary/REPORT.md)，自评 | 重复权重sum-first/CSE为UCNN类A。24bit整列键、有限计数和组退休使合同可执行，但本身仍是既有方法的粒度选择。 | 2 | 是。真实八块慢2.99%；32类只覆盖69个K，状态税超过省下的42次Q1读。高重复反例保留家族潜力。 |
| [4 完成z整组关停](../algorithm_sparse/AS1_GROUP.md)，独立 | 结构剪枝为A；按真实I24变化拟合score并将剪枝单位对齐完成R8的Q2义务，是一个有界接口增量。尚无严格消费者误差界，不能借用Bishop的界。 | 3 | 主要是底座与任务校准。补齐共同全零旁路后，相对逐rank同预算省1.202%暖拍且十帧质量较差；不能把相对exact的8.746%全算X。 |
| [5 K4 latent原型＋单残差](../algorithm_sparse/AS2_PROTOTYPE.md)，独立 | LUT距离编码、pattern product与残差为LUT-DLA/Phi类A。迁到signed R8并截成一rank是新的本地有损工作点，未形成新的编码原理。 | 3 | 主要是表示适配。编码、表与残差均实装；补共同全零旁路后比exact省7.689%，十帧未过门。此负点不否定其他原型/残差接口。 |
| [6 T10整latent保持](../algorithm_sparse/AS3_TEMPORAL.md)，独立 | 阈值更新、持久参考与稀疏Δ为既有A。增量在于同帧T10把整向量保持绑定Q2 acc生命周期，并每T保留真实identity/I24义务。 | 3 | 主要是底座与有限策略。给rank控制相同Δ权限后，仅省1.743%暖拍且十帧质量较差；没有建立全面优势。不能用弱full-refresh rank分母。 |
| [7 完成p直接送消费者](../dataflow/d1_forward/REPORT.md)，独立 | STORE forwarding、holding/backpressure是A。强控制已能边STORE边送，候选只取消写p_mem；当前没有额外X。 | 1 | 是。64tile同为968747拍，只少30720个写词；这是事务证据，非已测面积/能耗收益。 |
| [8 横邻tile两列halo轮转](../dataflow/d2_halo/REPORT.md)，独立 | 卷积halo复用、环形地址为A；X没有超出原生几何映射与正确跨行退休。不能称运动/光流复用。 | 1 | 是。全帧省4.5196%，严格等于19080×768次装入；完整实现有工程价值，百分比不增加方法新颖性。 |
| [9 双tile共享算术交错](../dataflow/d3_interleave/RESOURCE_CONTRACT.md)，独立 | 多context、资源仲裁、阶段重叠为A。只有相对普通双context同时ready/RR仍有增量，才可能讨论Q1/Q2阶段选择的X；共享八ALU本身不够。 | 2 | 底座。最终full阶段错位260697004拍，比普通双ready/RR的233289744拍慢11.748%；候选X没有胜过强A，不沿用seq弱分母。 |
| [10 借用I24宽加法链做四P](../phase_borrow/README.md)，独立 | SWAR/carry分段与不重叠生命周期资源绑定为A。真实64位消费者链切换为四个signed13累加，随后恢复RNE前仿射，构成明确可审接口，但未显示新的通用共享机制。 | 3 | 是，保留较完整的接口证据。同416bit口全帧省4.077%；无自然双请求争用覆盖、无PPA/Fmax，不能直接叠加流水化收益。 |

十项均有实际RTL，功能/完整事务证据比纸面点明显成熟；这与标题新颖性是两件事。当前没有一项达到本表4分门槛，不代表候选家族无研究价值。整组关停/时间保持若要晋级，需要在相同消费者质量约束下进一步证明其“关闭整项后端义务”具有不可被普通逐rank执行解释的收益；本批尚未给出。phase借用的现实价值是保留一个经过完整消费者核验的资源绑定选择，不能因有4%收益就改称新算术。

判断复用已有primary阅读，不新增文献数量：[BISMO](https://arxiv.org/pdf/1806.08862) §II/III-A、[DA官方方法](https://www.mathworks.com/help/hdlcoder/ug/distributed-arithmetic-for-hdl-filters.html)、[UCNN](https://www.kartikhegde.net/media/UCNN_ISCA.pdf) §III-A/B的范围见[source_table](source_table.json)；[Bishop](https://arxiv.org/html/2505.12281v1)、[Phi](https://arxiv.org/html/2505.10909v1)、[LUT-DLA](https://arxiv.org/html/2501.10658v1)、[DeltaCNN](https://arxiv.org/html/2203.03996v2)的实际阅读段与未复现模块见[algorithm来源](../algorithm_sparse/SOURCES.md)。宽链先验沿用前阶段BitFusion/Envision记录与本阶段[来源边界](../SOURCES_AND_NEXT_INTERFACES.md)，不把只取得摘要的DeVSA当已排尽的碰撞。

静态独立审阅入口：[有损执行](REVIEW_ALGORITHM_SPARSE.md)、[宽链借用](REVIEW_PHASE_BORROW.md)。数据流逐实现审阅另见[root独立记录](../REVIEW_DATAFLOW.md)。本页只对当前差分评分，不代替新的质量收据，也不新增实验。

最终状态更新：AS1/AS2行使用全零encoder旁路后的SUMMARY（函数/质量未变）；D3行使用补齐普通RR的三臂full收据。所有评分保持方法差分尺度，不随新增小幅周期收益自动上调。
