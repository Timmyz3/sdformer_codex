# 点名文献补缺与共同活动打包的先验边界

日期：2026-09-13。范围：UNICORN、STELLAR、ELSA、DJP；补入 Column Combining、TensorDash，并复用本轮前段已经读过的 HighLight、S2TA、Bishop。只做文献与作者代码的静态阅读，没有运行训练、实验、RTL、EDA，也没有修改生产树。

最值得完整借入的是 **ELSA 2026 的目的行事件聚类/归约，加上 TensorDash 的受限互连调度**。但是，局部分组、有限队列、队列满后排空重试、非零操作前移都已有直接先验。研究增量只能落到：固定真实源词布局、权重端口、目的部分和集合和回放硬件以后，训练或选择共同活动分组是否仍减少完整事务。当前不能把这个增量写成已经成立。

## 1. 名称与证据状态

| 点名 | 核定身份 | 本轮实际读到 | 可以立即据此实现完整 A 吗 |
|---|---|---|---|
| UNICORN | *Unicorn: A Multicore Neuromorphic Processor with Flexible Fan-In and Unconstrained Fan-Out for Neurons*，DAC 2022，943–948，DOI 10.1145/3489517.3530563 | 官方 DAC 日程/出版元数据；Sensors 2022 映射论文中的目标平台描述；原论文全文未取得 | 不可以；机制细节仍缺 |
| STELLAR | *Stellar: Energy-Efficient and Low-Latency SNN Algorithm and Hardware Co-Design with Spatiotemporal Computation*，HPCA 2024，172–185；Xplore **10476421** | 官方出版元数据、作者学校成果说明；原论文全文未取得 | 不可以；FSBP/stRS 完整合同仍缺 |
| ELSA 2026 | *ELSA: An ELastic SNN Inference Architecture for Efficient Neuromorphic Computing*，ISCA 2026，arXiv 2605.20802v1 | 主文 §II–VI、相关消融与 artifact；作者卷积模拟器的输入队列/权重存储/PE/原生 im2col 代码 | 可以借入算法级完整机制；公开代码读到的是 Python 模拟器，不能称已取得作者 RTL |
| ELSA 2021 | *ELSA: Hardware-Software Co-design for Efficient, Lightweight Self-Attention Mechanism in Neural Networks*，ISCA 2021，DOI 10.1109/ISCA52012.2021.00060 | 作者全文 §III–IV，近似注意力筛选与专用硬件 | 可以作为 attention 强 A；不是通用 C1 卷积机制 |
| DJP | *Maximizing Energy Efficiency in Spiking Neural Networks: A Dynamic Joint Pruning Framework*，DAC 2025，DOI 10.1109/DAC63849.2025.11132570 | IEEE 官方摘要与出版信息；全文未取得 | 不可以把普通发放正则改名为完整 DJP |

以上身份由[DAC 2022 官方日程](https://www.dac.com/Portals/0/DAC%2059/59DAC%20Onsite%20Guide_v3.pdf?ver=GbBS5sBuhmEVJWVEz9CNIg%3D%3D)、[HPCA 2024 出版目录](https://www.proceedings.com/content/074/074054webtoc.pdf)、[ELSA 2026 作者论文](https://arxiv.org/abs/2605.20802)、[ELSA 2021 作者页](https://taejunham.github.io/)、[DJP 的 IEEE 页面](https://ieeexplore.ieee.org/document/11132570/)支持。全文、代码 URL 和逐篇阅读范围见 [source_master.csv](source_master.csv)。

**UNICORN 歧义保留。** Sensors 2022 的引文链确实指向上述 DAC 2022 SNN 处理器；不能将同名 CPU emulator、点云压缩 Unicorn 或 Unicorn-CIM 接到它的代码栏。针对“稀疏架构同名 UNICORN”的独立检索，尚未核到另一篇能与本地描述相符的 sparse-DNN 架构的唯一题名/DOI，因此没有自作唯一归并，也没有新增一个假条目。[Sensors 映射论文](https://www.mdpi.com/1424-8220/22/19/7248)是原始映射研究，但对 DAC 处理器细节仍属于转述，不能冒充处理器全文。

## 2. ELSA 2026：必须纳入的最新直接先验

### 2.1 已读的原作机制

主文 pp.3–7，§III-C、IV-A/B、V、VI：BAER 用同一个 spine/token 标识打包一组事件，256-bit flit 中容纳最多 17 组位置/符号；尾包填零。mini-batch Gustavson 将共享目的膜行的源事件一起处理，经 N-way 权重读取与加法树摊销膜行读写。列向切分权重/膜矩阵，并向各 PE 广播事件。路由器承担原生 im2col、flit 编解码、队列与完成调度。ST-BIF 的状态/发放和细粒度流水共同服务 elastic inference。[原文 §III–VI](https://arxiv.org/html/2605.20802v1#S3)

### 2.2 作者代码比主文更具体的队列证据

静态阅读了官方 artifact 的卷积路径，未执行：

| 代码 | 可观察机制 | 对本研究的直接影响 |
|---|---|---|
| [inputBuffer.py](https://github.com/Intelligent-Computing-Research-Group/ELSA/blob/main/ELSA_Simluator/convolution/processElement/inputBuffer.py#L58) | `CIdtoQdTLB/QIdtoCdTLB` 按目的 `column_id` 分配有限 spikeQueues；组满或队列满返回失败与待排空 queue ID | “局部目的分组＋有限队列”已经是 A |
| [processElement.py](https://github.com/Intelligent-Computing-Research-Group/ELSA/blob/main/ELSA_Simluator/convolution/processElement/processElement.py#L95) | 失败时排空指定队列；原 spike 尚未前进，之后重试；update 时排空指定目的组 | “容量冲突后 drain/retry”也已经是 A |
| [weightBuffer.py](https://github.com/Intelligent-Computing-Research-Group/ELSA/blob/main/ELSA_Simluator/convolution/processElement/weightBuffer.py#L43) 与 [SRAM.py](https://github.com/Intelligent-Computing-Research-Group/ELSA/blob/main/ELSA_Simluator/convolution/processElement/SRAM.py#L34) | 逐个读出权重，SRAM 统计访问次数；已读路径没有按当前周期记录 bank 占用 | 不足以证明已实现逐 bank 仲裁/回放 |
| [processElement.py](https://github.com/Intelligent-Computing-Research-Group/ELSA/blob/main/ELSA_Simluator/convolution/processElement/processElement.py#L132) | `weightLoadCycle` 以访问量除以总 SRAM 数估计；另一排空路径使用总数的一半 | 本地必须用实际同 bank 请求检查替代总体带宽平均值；这是一项验证要求，不等于论文结果错误 |

这里要严格区分两种冲突：**队列/目的组容量冲突**，ELSA artifact 明确已有；**同周期 W-bank 读端口冲突**，已读路径未见显式重放实现。未检查整个公开库的每个分支，因此不作“ELSA 全部代码绝无 bank 处理”的全称断言。公开论文声称 N-way 读取，不能据此假定它在本地一个廉价单读口 SRAM 上免费成立。

### 2.3 A/B/X 与完整借入要求

**A：**原生事件 im2col、目的行聚类、BAER header 摊销、有限输入队列、满后排空重试、批内源权重归约、膜/psum 一次读写、尾包/完成标记、消费者广播与反压。

**B：**普通 event AAC 2:4；同样容量与端口的行驻留批处理；同样 BAER 格式但按自然顺序入队；完整 ELSA 式目的聚类加容量冲突 drain/retry。不能只用“单事件一包”的弱 B。

**X 候选：**在上述全部硬件固定后，让分组/掩码训练感知实际 source-word、目的集合及 W-bank 冲突成本。至少比较自然顺序、随机等规模分组、只看权重支持的分组、只看边际发放率的分组、看真实共同活动的分组。若优势只是新增队列或更宽读口，应归入 A 的工程收益。

完整借入还需支付：目的到队列查找表及空闲表、每项源/目的/epoch 标签、尾包空位、重试占用、批形成等待、队列 SRAM 读写、权重广播/归约、psum RAW 旁路、原生 im2col 的 halo 重复和结束处理。将这些成本删掉后得到的理想包数不能称完整迁移。

**真实层适配（本地推断）：**r0 `sn2→conv2` 的二值源最直接；conv2 后仍输出连续值并接 norm/residual，不搬 ST-BIF 发放。continuous PED 的连续 V 不能直接变成 1-bit AER；仅能借行驻留/聚类，位宽和乘法成本保留。decoder2 若入口包含连续残差，同样不能套二值通信。非因果 T10PSN 必须完整读 T10；原作早响应/逐时间发放合同不应直接借到这条路径。

## 3. Column Combining × TensorDash × Bishop：剩余 X 的最强反例

### 3.1 三篇分别已经覆盖什么

**Column Combining，ASPLOS 2019，pp.4–7，§3.1–3.4、§4.1–4.2：**按列支持重叠和密度分组，以 α 限制组大小、γ 限制平均冲突；冲突行最终只保留一个权重并微调。MX cell 保存该权重的源选择，接收组内多路输入并转发。上游行置换让下一层分组输入物理连续，避免运行时 switchbox。因此“有限冲突分组＋训练＋输入重排＋小 mux”是很老的联合设计。[作者全文](https://www.eecs.harvard.edu/~htk/publication/2019-asplos-kung-mcdanel-zhang.pdf)；[作者训练/packing仓库](https://github.com/BradMcDanel/column-combine)已核README，未执行或逐模块迁入。

**TensorDash，MICRO 2020，pp.6–8，§3.1–3.5：**三行 staging buffer、每 lane 八选一，包含本位、时间前移与邻 lane 取数；两操作数同步移动，分级选择并清除已消费项，防止重复执行；源 scratchpad 需相应 bank 带宽。tile 可以共享调度器和某一侧 buffer，双侧独立调度则成本更高。因此“窗口内动态填泡＋有界跨 lane 调度＋消费位图”也是完整 A。[作者全文](https://arxiv.org/pdf/2009.00748)

**Bishop，ISCA 2025，pp.4–8，§3.2、4.1、5.1、5.3–5.4：**TTB/BSA、活动标签、stratifier 的原特征位置跟踪与权重对齐，以及稀疏/稠密分流已经将训练分组与真实执行结合。这里只复用前轮已经完成的阅读，不把它算本轮新增深读。[作者全文](https://arxiv.org/pdf/2505.12281v1)

### 3.2 与本地“权重不合并”的区别，以及为什么区别本身不足以成新

本地保留组内全部权重，并在同时活跃时逐个重放；Column Combining 会剪去冲突权重。两者函数与权重容量合同不同，必须分别标记。然而将“剪掉碰撞”改成“排队处理碰撞”，再使用 TensorDash 受限互连和 ELSA 有限目的队列，是自然且已有大量组成部件支持的组合。仅凭“无损”或“可重放”不能声称新的基本架构。

最强反例是：**一个普通实现已经保留所有权重，具有相同分组尺寸、相同事件位图、相同银行/端口、相同最大回放深度与相同归约能力，只把共同活动训练换成静态重排或普通发放率启发式。**若这个控制取得同样收益，则 X 为空；即使最终硬件比当前基线快，也只能报告强 A 的完整迁入收益。

可观察的新颖性对照句应写成：

> 固定原生源词格式、权重地址映射、lane 数、队列容量、每周期 bank 端口和全部目的提交规则，仅改变分组/掩码的学习目标；使用真实 T10 活动联合分布与消费者支持训练的分组，在同任务质量门限下，相对自然顺序、静态支持分组和边际活动率分组，减少了实际源词读取、bank 回放和尾部提交周期的净成本。

这句话仍是一项待检验假说。尤其当目标是“一组最多取一个活动项”，组内共同活动越强可能越糟；当硬件支持“一次源词读取服务多目的归约”，共同活动才可能有利。不能把同一个共现分数同时用来证明这两种相反的物理收益。

### 3.3 必须用物理集合定义目标

设运行时有效贡献集合为 `E={(source_word, source_bit, weight_word, bank, destination, epoch)}`。分组产生哪些事务应由硬件执行该集合后确定：

- 取消一个源词读，要求所有依赖它的存活消费者都不需要该词；只清除一个 lane 的事件不够。
- 取消一个权重词读，要求该词内所有需要的系数都不再参与；同词仍有活跃 lane 时，仅省 AAC 不能算省 SRAM 读。
- 一个 bank 的多个地址需多周期服务；同地址多目的广播必须显式实现，不能把不同地址当一次“包”。
- 多事件共用目的 psum 只有在归约树输入数、符号处理、累加顺序合同和 RAW 旁路都兑现时，才减少目的读写。
- 回放深度不能靠丢弃溢出贡献维持；必须有 backpressure/drain，或训练时明确付出有损代价并在任务上验收。

建议优化的是以上事务及提交尾部的组合目标，而非只优化 `nnz`、组内 OR 或理想最大事件数。本轮 root 的实测探针若没有这些地址与集合，应标记为“候选筛选”；它不是逐个写完 RTL，更不是 PPA 结论。

### 3.4 首个完整 RTL 范围建议

首模块可命名 `r0_conv2_grouped_event_engine`，但名字无贡献含义。固定 `C=96, K=3×3×96=864, N=96, T=10`，从原生 bit tensor 和静态折阈值权重开始，完整覆盖所有 K、全部输出通道与边界窗口，输出 conv2 连续累加结果。模块应包含 native im2col、source-word 解码、分组 metadata、真实 bank scheduler、有限队列/重放、psum RAW、epoch drain 和最终写回；TB 只提供原输入/权重/静态配置及参考输出，不提供 goldmask 或理想事件序列。

检查点先做每个 conv2 输出和源词/权重词/bank-stall/replay/psum 事务计数，再由现有 norm2/residual 桥接检查 r0 输出。完整 fullK/fullN 是算子级声明的底线；先实现 tile 可以调试，但不能用局部 tile 宣布整个 r0 的收益。B 与 X 必须共用同一 RTL，仅换合法静态分组/掩码配置。

## 4. 两项近年可完整借入的强 A

这两项已在此前 sparse 深读轮完成主文阅读，本轮将它们补入具体对照链；没有重新计算“新增论文数”。

| 工作 | A：必须一起借入的模块 | B：普通强控制 | X：本地尚可检验的耦合 | 适配层与边界 |
|---|---|---|---|---|
| **HighLight，MICRO 2023** | 两级 HSS/offset metadata、各 rank 的 skip、VFMU 的对齐宽读与变长移位；区分 B 侧压缩/gating 和真正周期跳过 | 相同层次 sparsity 与宽读格式、相同 buffer 的普通 HSS；不能用稠密搬运弱化 B | 把真实 source-word 与消费者支持的交集作为训练目标，只有跨完整物理词取消才计入 | r0 的权重/源词层次最直接；continuous PED 可借结构格式，不能假定连续值变事件；decoder2 需单独核入口 |
| **S2TA，HPCA 2022** | DBB 限界稀疏训练、DAP 运行时幅值选择、time-unrolled DP1M4/TPE、metadata、源 SRAM 和逐激活调度 | 普通 2:4/DBB 以及相同 DAP、相同 DP1M4；并给未剪的 task baseline | 允许组规模/掩码因实际端口与目的集合变化，但硬件固定以后仍须优于普通 DBB 选择 | continuous PED 的连续幅值可定义 DAP top-k，但任务有损必须重验；r0 二值幅值全同，top-k 大量平局，直接移植 DAP 没有自然重要性指标 |

HighLight 阅读范围为 pp.5–9、§4–6；S2TA 为 pp.4–7、§3–6。[HighLight 原文](https://arxiv.org/pdf/2305.12718)、[S2TA 原文](https://arxiv.org/pdf/2107.07983v2)。两篇本轮均未确认可直接迁入的作者 RTL；论文详细机制可复现不等于已有可运行公开实现。

## 5. 三项全文缺口：明确能借什么、还缺什么

### UNICORN：保留为 fan-in/fan-out 与网络映射底座，不伪装成局部打包新法

**已核定：**DAC 2022 的处理器身份和题名；Sensors 2022 使用它作为映射平台。映射论文区分实际处理器组织与用于比较映射器的扩展配置，不能把模拟扩到 8×8/16×16 的 core 网格写成原芯片规模。[Sensors §4 与参考文献 11](https://www.mdpi.com/1424-8220/22/19/7248)

STSM（spike train sliding multicasting）和 NMM（neuron merging）名称目前仅从作者条目所载摘要转述查到，未从本轮成功取得的处理器主文核验。因此以下是**待核借入清单**：spike train 格式、multicast 路由状态、fan-out 滑动步进、分片 neuron 合并/psum 完成条件、跨核时序与队列背压、存储地址布局。不能声称已经复现。

**A/B/X：**A 待全文补齐上述模块；B 应是等 fan-in/fan-out 的普通多播/分片累加，而非硬件不能表达大扇入的弱基线；X 若存在，应来自本网络实际 halo/残差消费者集合怎样取消多播事务。r0 的跨 H8/halo 消费者可能相关，continuous PED/decoder2 若是多核分片同样相关；对本轮片内分组 replay 它不是最先实现项。

### STELLAR：算法与数据流不能只借一个名字

**已核定：**作者学校说明包含极少脉冲/短时间窗学习、窗并行和时空编码稀疏架构；它是算法与硬件协同工作。[UESTC 作者单位成果页](https://news.uestc.edu.cn/info/1005/6567.htm) FSBP 和 stRS 的细节仍须原论文；本轮只读元数据/学校说明，不把 LoAS 对它的讨论升级成原文精读。

**待核完整 A：**Few-Spikes 神经元方程与幅值时间码、FSBP 训练和梯度、窗并行分配、stRS loop nest、事件格式、累加/发放分离边界、权重与状态 bank 组织、尾窗处理。

**B/X：**B 应包含普通时间展开/row stationary，且匹配同一神经元与同一编码；X 若把该思想移植到本地，必须重新建立 AT-LIF `{0,θ}` 和非因果 T10PSN 的合法时间合同。r0 二值 AAC 数据流可能借思想，continuous PED 不能直接走 Few-Spikes 编码；decoder2 也必须先核真实激活。没有全文时不把 STELLAR 列为“马上完整借入”工单，更不判它失败。

### DJP：联合剪枝是强算法先验，SOP 数不是端口模型

官方摘要明确联合权重与时空脉冲稀疏、多阶段 mask 控制发放阈值、TABN 可学习时间缩放，以及按实时计算影响调整重要性系数；任务包括 CIFAR 与 ImageNet。[IEEE 摘要](https://ieeexplore.ieee.org/document/11132570/) 尚未取得损失方程、mask 参数化、训练日程、精确推理展开或作者代码；第三方 AI 文章不能填补这些缺口。

**待核完整 A：**mask 对膜/阈值作用的位置和档位、TABN 与 BN 参数冻结/折叠规则、动态系数的精确更新、SOP 模型、剪枝与微调阶段、约束下的推理网络。B 至少应有普通发放正则、单独权重剪枝、普通联合正则，以及获取全文后的原版 DJP。X 候选是将其成本目标替换/补充为真实词、bank 和回放成本；“联合权重和脉冲剪枝”本身已经不新。

本地 r0 可研究发放/掩码联合训练，但 AT-LIF 的 θg、PSN 参数与 BN 处理要分开；continuous PED 没有直接 spike firing 指标，只能迁移权重/连续激活约束思想；decoder2 的入口合同同理。126.38× SOP 压缩不能当成本地能耗或速度，也没有证据据此构造 DJP 微架构。

## 6. ELSA 2021 的正确位置

§III–IV 的原作先计算 query/key 的二值随机投影 hash，以 Hamming 距离、key norm 和 LUT 筛选关系，再对候选执行精确 attention；硬件含 hash 计算、候选选择、队列和 attention 后端。算法的筛选成本与队列/内存访问必须一并借入。[作者全文](https://taejunham.github.io/data/elsa_isca21.pdf)

**A/B/X：**A 是完整近似关系筛选器；B 是精确 attention 和普通完整候选筛选；X 要来自本地真实 attention 的语义与成本耦合。r0 卷积、continuous PED 的一般矩阵乘法、decoder2 卷积都没有 query-key 关系这一对象，因此不能列为这些层的 generic C1/pruning 机制。它也不覆盖 ELSA 2026 的 BAER/目的行聚类，二者必须分开引用。

## 7. 建议推进次序与报告口径

1. 先把 **ELSA 式局部目的聚类＋TensorDash 式受限前移＋显式 bank 回放** 做成强 A。源词和权重不合并的合同保持清楚；逐周期 replay 和满队列反压由硬件产生。
2. 在同一个实现中，仅替换分组/掩码目标，测试真实共同活动和消费者支持是否给出净增量。Column Combining、Bishop、普通 2:4/DBB 都要成为训练/分组控制，而不是只在 related work 出现。
3. HighLight 用作宽词取消/metadata 控制，S2TA 用作结构稀疏/普通动态选择控制。UNICORN、STELLAR、DJP 保留为已核名但缺全文的后续借入项；不靠标题和摘要加速创新宣称。

本轮没有产生新的性能或任务质量结果，也没有宣称所有点名论文都已全文精读。已关闭的是 ELSA 同名歧义和关键队列先验缺口；仍开放的是后三篇全文、共同活动分组相对固定强 A 的增量，以及实际完整算子 RTL 的端口兑现。
