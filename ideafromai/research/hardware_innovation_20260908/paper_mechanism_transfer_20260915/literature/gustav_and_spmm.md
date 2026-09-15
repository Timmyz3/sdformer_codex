# GustavSNN 与非 SNN 稀疏乘法：问题是怎样逐层变具体的

2026-09-15。GustavSNN 主文 §§II–VII、评估和参考文献已逐段阅读；使用现有 `../../literature/GustavSNN_HPCA2026_public_mirror.txt` 及本轮规范化文本。GAMMA、MatRaptor、GROW 阅读下列具体方法段；TCAS-II 2023 量化/剪枝短文全文已读。前者的架构方法阅读不等于复现其完整实验。

## 1. 第一层：先比较乘法组织，再讨论 SNN 特性

GustavSNN 并非给 Prosperity 的 product-sparsity 加一个 Gustavson 模块。两者都借时间批处理，但选择不同的稀疏计算组织；论文用 Prosperity 官方模拟器作对照。主论文为 HPCA 2026，[DOI](https://doi.org/10.1109/HPCA68181.2026.11408587)。

§II/III 将内积、外积、Gustavson 放在同一个问题下：内积需要找交集，外积产生大量待归并 partial matrix，Gustavson 限定一个输出行的归约，减轻这两项代价，却引入不规则取数。它没有声称 GP 是新数学；其核心判断是 ANN 稀疏乘法的代价排序，在 SNN 上发生了变化。

二值 spike 很窄，W 和状态很宽，且有时间维与神经元更新。因此“保持 PE 忙”不是主要目标：如果更新一个宽膜值要反复出入共享存储，省掉的乘法未必抵过状态搬运。§IV-A/Fig.4 先用 flat-memory、无 bank 冲突、无限带宽的分析定位代价；这些曲线是架构动机，不能当成论文后续物理实验，更不能当本项目 PPA。

## 2. 第二层：宽状态搬运贵，但整个输出行又放不下

§IV-B/C 的解决方案是 column-parallel tick-batch（CPTB）。输出列分成每组 P 个，同一输入行的不同输出列组由不同 PE 处理；每个 PE 的 P 个神经元状态本地驻留。输出空间互斥，避免 PE 间汇总同一个结果；时间更新在局部完成。

这一步同时改变了**谁拥有结果、结果留多久、沿哪个维度并行**。单写“Gustavson 乘法”缺失了这三个真正决定费用的条件。论文的 P=8、K=8 等是其资源点，不能默认为任何网络最优。

§IV-D 再发现：按很长 T 合并位图，某行在任一时刻活过，就必须保留整个时间组。相反，在较小 P 的位置组里按时间逐片判断非空，可以保留更多时间稀疏。NRV 将行号与 P-bit 支持一起存储。P 过小使行号和访问开销上升，P 过大又使行并集变密；其稀疏率优势取决于这一权衡。论文的高稀疏阈值和图中最优 P 是数据相关观察，不是普遍常数。

## 3. 第三层：稀疏格式还需要真实的取数与归并电路

完整 A 至少包括以下组件，缺一项就应说明是部分适配：

| 组件 | 主文位置 | 必须承担的费用 |
|---|---|---|
| NRV 行号与位置 bitmap | §IV-D | 产生、压缩、解码、尾包及原始索引 |
| 最多 NR=4 行的供数与权重跳零 | §IV-E | 稀疏行与 W 的实际读入、zero-W 过滤、压紧 |
| 双缓冲和逐位置归并 | §IV-E，基于 GAMMA | 多条已排序位置流的最小索引选择、相同位置累加、背压 |
| 本地神经元状态 | §IV-E | 状态位宽、所有时间更新、阈值/复位和完整消费者 |
| tile 间共享 W 与 spike | §V-A | 每 tile 双口 W buffer、对应 PE 的广播、同步及不均衡等待 |
| 时间次序与列组次序 | §V-B | 时间稀疏与 W 重载的取舍，不是免费全时间并行 |
| 双稀疏的双指针交集 | §VII-B | NRV index 与 W index 的读取/比较；W 很稀时避免 skip 阶段先成为瓶颈 |

§VII-B 本身已讨论密 W 下双指针的额外代价，并提出选择不同策略的方向。因此我们补 dense bypass 有价值，却不能称新发现“W 密时压缩反而慢”。

§VI 使用 28 nm FDSOI 综合/布局、部分 SRAM 的 CACTI、周期仿真与功耗工具；与本项目 TS1N28 条件不同。本文不转引能效倍数作为本方潜力，也不从致谢推断流片。它实做的神经元为 LIF；“可支持其他神经元”的文字不能替代本非因果 T10 PSN 的状态时间线。

## 4. 沿关键引用回查，哪些东西已经是成熟 A

### GAMMA：不是只借一个最小值比较器

GustavSNN [49]，[ASPLOS 2021 作者全文](https://people.csail.mit.edu/sanchez/papers/2021.gamma.asplos.pdf)，[作者机构页面](https://research.nvidia.com/publication/2021-04_gamma-exploiting-gustavsons-algorithm-accelerate-sparse-matrix-multiplication)。本轮精读 processing element、调度、FiberCache 与预处理段。

GAMMA 从三个问题组织硬件：稀疏输入行长度差异大；高吞吐、低 radix 归并会反复搬 partial fiber；不规则 B 行复用和临时 C 行生命周期不同。它用高 radix、较低单 PE 输出率配许多并行 PE，配合有依赖的多轮归并调度。FiberCache 同时容纳可共享只读 B 与短命 partial C，但区分 fetch/read，计尚未消费的引用，消费完的临时输出可失效而不回写。

因此，“在生产/消费接口上记录最后一次使用”并非空白。若我们保留 gate 与 PED 两项义务，新增内容必须是**这两类消费者在不可跨越的 PSN/RNE 边界下怎样改变实际回收或服务**；两个 pending bit 本身不构成 X。GAMMA 的稀疏 fiber 与本方完整 T10 状态也不能直接等同：后者可能在前缀结束后仍然有未完成的时间混合。

### MatRaptor：先让格式与独立输出所有权一致

GustavSNN [39]，[MICRO 2020 作者全文](https://www.csl.cornell.edu/~albonesi/research/papers/micro20-2.pdf)。本轮精读 C²SR、PE 和 multiply/merge 队列。

普通 CSR 的一行跨多个 memory channel，会出现不需要的数据搬运；并行输出行长度又可能未定，维护传统 row-pointer 需要等前一行。C²SR 将行按固定 channel 分配，用 row length 与 pointer 定位，PE 各自持有输出行。PE 用多个有序主队列和一个 helper queue 逐步归并；helper 与主队列交换角色，避免把已有有序性丢掉再全量排序。

这里可借的是**格式、存储通道、输出所有权一起设计**。对我们，捕获包的 bank 地址是真实布局；不能用 `channel % 8` 假定它与物理 bank 一致。异步输出行、round-robin、有限队列是成熟 A，不宜单独再当新机制。

### GROW：先证明哪些行值得驻留，再隐藏不能驻留的行

GustavSNN [20]，[HPCA 2023 作者版](https://arxiv.org/abs/2203.00158)，[正式出版身份](https://pure.kaist.ac.kr/en/publications/grow-a-row-stationary-sparse-dense-gemm-accelerator-for-memory-ef/)。本轮精读 §V-C/D：high-degree-node cache、图划分、多行 runahead 与两个依赖表。

GROW 发现 GCN 邻接图存在少量高连接度节点，优先驻留其 RHS 行。缓存小于图时，全图高频行不能充分代表局部；先划分图，在各 cluster 内选高频行。剩余 miss 很普遍，故允许并发多输出行；一张 LDN 表记录缺失 RHS，另一张表记录谁在等待这条 RHS，以返回索引唤醒相应输出。

重要前提是邻接图静态，图划分成本可以跨后续推理摊销。本 SNN 的活动支持随输入变化；不能给每个事件帧免费 Metis，也不能从平均发放率假设 power-law。可试的迁移是利用**固定 W/消费者图决定的复用**加有界运行时队列，并用真实码字验证动态部分。GROW 的普通 runahead/合并请求应给原生基线相同权限。

### TCAS-II 2023：剪枝后为什么还可能更贵

GustavSNN [38]，[The Hardware Impact of Quantization and Pruning for Weights in Spiking Neural Networks](https://arxiv.org/abs/2302.04174)，[正式论文](https://doi.org/10.1109/TCSII.2023.3260701)。本轮全文精读；[作者代码](https://github.com/Intelligent-Microsystems-Lab/SNNQuantPrune)仅确认入口，未运行。

作者用 DVS Gesture SNN 比较量化、剪枝、先剪后量化、联合训练，并在 Eyeriss-like 的模型上计 metadata/DRAM/中间存储。其观察不是“剪枝不好”，而是高位宽稀疏 W 的索引税可能大于低位宽稠密/自然含零权重；ternary 在该任务有明显优势。论文使用 Timeloop/Accelergy 校准模型，非本项目 RTL 或硅测量；其能量和精度差不能迁给光流。

对我们，更有价值的问题是：**在 θ 已折入 W 后，量化等级、结构剪枝和实际 W 字打包共同决定什么数据真能少读？**强控制必须包含纯量化、纯 2:4/通道剪枝及普通 packed-W，不能只对比原始 FP32。该论文还强调 spike-time 对任务的影响，进一步说明 gesture 的 ternary 精度不能当光流的保证。

## 5. 本图迁移：哪些成立，哪些新增问题还在

| 本方条件 | 可直接沿用 | 需要重新解决 |
|---|---|---|
| `{0,θ}`，静态 θ 折权 | 二值选择 W、NRV、子集等式 | 折权后有限数值表示与既有舍入边界 |
| 非因果完整 T10 PSN | 完整输入窗口内的线性重排 | 不能只留 P 个 LIF 膜并每 tick 释放；须留全依赖或合法结构化系数状态 |
| 稠密/2:4/C16 混合 W | NRV∩W、dense bypass | 元数据何时取得、共享服务粒度与压缩 rank 补扫 |
| 门与连续值消费者 | 共享输入/可延后提交 | 门定了不代表连续状态可以释放 |
| 窄 N8、有限单服务口 | 有限队列、按所有权归并 | 原论文宽输出摊销/多核并行不能免费搬过来 |

本轮已把一项缺口写成真实 RTL：[生产接口试验](../PRODUCER_EXPERIMENT.md)。四个 64bit 单口行为存储从实际 accepted 源字开始，处理跨字 code3，再接现有 NRV∩W、局部 S 和完整 T10 判决。该工作使 A 更完整；完成 bank 的 seen-code 摘要与简单非空标志的性能相同，故摘要本身不升级为标题。它未覆盖前级量化器、完整层输出行、FC2/BN2/shortcut 和 ASIC 映射。

**值得保留的研究方法：先测完整 A 缺在哪里，然后改产生该代价的接口。**本轮结果支持继续研究有限 PWP 生成/消费、真实非因果消费者的驻留边界和剪枝后的物理字交集；不支持再把普通 row-OR、RR 或 dense bypass 各起一个新名字。
