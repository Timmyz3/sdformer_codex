# CICC 定向补检：先验、适配与尚未尝试部分

2026-09-09。本轮整理 **16 条 CICC 工作：15 条为旧 358 条表之外的新题名，1 条补正已有 Zhang 光流芯片的身份与阅读深度**。这不是 16 篇全文精读：Zhang 2026 的本地四页原文已通读；其余按一手摘要、作者介绍、官方节目题名分别标明。没有把题名推测写成原作机制，没有运行新实验。

原两份盘点保持独立。本补充的用途是改变强对照和待验证问题；论文自己的芯片数字不进入本地性能表。

## 检索范围

实际查看了 [CICC 2023 官方节目](https://www.ieee-cicc.org/wp-content/uploads/2024/10/CICC-2023-Program.pdf)中的相关计算场次、[2024 节目](https://www.ieee-cicc.org/wp-content/uploads/2024/10/CICC-2024-Program.pdf)中的 14/21/26 等相关场次，以及 [2025 节目](https://www.ieee-cicc.org/2025archive/wp-content/uploads/sites/21/2025/04/CICC-2025-Program-4-8-25.pdf)中的 11/18/37 等相关场次；另以 [2022 节目](https://www.ieee-cicc.org/wp-content/uploads/2024/10/CICC-2022-Program.pdf)回溯相关神经计算工作。没有通读这些年份的全部论文。

[2026 官方技术节目入口](https://www.ieee-cicc.org/technicalprogram/)指向动态页面，本轮未取得可完整检索的节目正文；2026 采用作者、机构和 IEEE 题名定向补检，覆盖不完整。未取得正文的论文保留为缺读，不能据此判失败。

## 直接改变当前对照的八项

### 1. Zhang 光流芯片 — CICC 2026，已有条目补正

**A 28-nm Optical Flow Estimation Accelerator with Redundancy Speculation, Bit-Width-Aware Compression and Similarity Detection**，DOI [10.1109/CICC65509.2026.11509564](https://doi.org/10.1109/CICC65509.2026.11509564)。[作者实验室入口](https://paicore.cn/)。

**阅读：完整原文。** 已读本地[四页 PDF](</home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/docs/Zhang 等 - 2026 - A 28-nm Optical Flow Estimation Accelerator with Redundancy Speculation, Bit-Width-Aware Compression.pdf>)。它针对 HybridSpike-FlowNet/MVSEC，并非 H67/DSEC。

- **A：** 按通道贡献排序，计算部分通道后推测 MaxPool/ReLU 结果并跳过后续工作；按实际位宽、非零位图和指针压缩搬运；先计算基础层，再用相邻帧特征相似性选择网络深度。PE 同时支持脉冲累加及多周期 ANN 算术。这些机制须连同排序、恢复和深度选择成本理解。
- **适配与边界：** 这是 patch 条件生产、宽状态压缩和跨帧跳过的直接强近邻。其推测不是本地非因果 T10 的严格证明；其 MVSEC 精度结果不能搬给 DSEC。当前粗头已删去的解码工作不能再算给动态深度。
- **纳入：** 完整强对照。旧 docs/228 基于更早身份和精度限制的否决不再足以永久排除 BWAC/DLSS；但不会因此恢复旧 FAED 为新标题。
- **尚未尝试：** 在当前固定 BN/量化学生上重建排序、推测、压缩、恢复和真实残差消费者。原文包含外存访问能量模型，不能把全表都称板级实测。

### 2. ROM-LTE — CICC 2026

**ROM-LTE: A 28nm 16.1TOPS/W ROM-Based LUT Tensor Engine with Complementary ROM and Switching-Suppressed Latch-Based PE for Fine-Tuning-Free ML Inference**，DOI 10.1109/CICC65509.2026.11509629。[一手机构摘要](https://snu.elsevierpure.com/en/publications/rom-lte-a-28nm-161topsw-rom-based-lut-tensor-engine-with-compleme/)。

**阅读：一手摘要及作者目录，未读完整电路。** 输出驻留、互补 ROM 和抑制切换的锁存 PE，面向稠密及不规则稀疏执行；不是必须重新训练码本的 LUT-DLA。

- **纳入理由：** PSN 常矩阵/常量计算不能只与通用 MAC 比较，还要比较查表面积、零模式切换和局部状态。
- **本地洞：** 10×10 非因果 PSN 的共享依赖、实际位宽和消费者完成，是否留下普通常矩阵加法图与这种零模式电路都未消除的成本，尚待测量。
- **尚未尝试：** 完整 ROM 编址、相位、时序及面积成本；不能把 ROM 宏数字迁给普通 foundry SRAM。静态 θW 的预计算也是基线，不是新的逐事件乘法机会。

### 3. OS-CIM — CICC 2025

**A 28nm 20.9–137.2 TOPS/W Output-Stationary SRAM Compute-in-Memory Macro Featuring Dynamic Look-ahead Zero Weight Skipping and Runtime Partial Sum Quantization**，DOI 10.1109/CICC63670.2025.10982878。[IEEE 摘要](https://ieeexplore.ieee.org/document/10982878/)、[作者芯片目录](https://seo.ece.cornell.edu/chip-gallery/)。

**阅读：一手摘要，未读全文。** 本地驻留部分和、向前查看零权并跳过、运行时部分和量化；使用定制 8T 存储结构。

- **纳入理由：** “局部部分和＋跳零权＋降状态成本”本身已有具体电路，必须作为 PSN/FFN 的近邻。
- **可借边界：** 调度和状态思路可研究；本地普通 1RW 宏不自动具备其读写和计算能力。运行时量化还须重新确认本地数值损失。
- **尚未尝试：** 全文的跳权窗口、控制、位宽及误差策略；当前 PSN 图中有符号和未来依赖的适配。暂不引入定制 SRAM 宏。

### 4. 非结构稀疏脉冲注意力/卷积阵列 — CICC 2024

**A 0.078 pJ/SOP Unstructured Sparsity-Aware Spiking Attention/Convolution Processor with 3D Compute Array**，Chaoming Fang 等。[作者机构记录](https://publications.polymtl.ca/65419/)、[作者伴随演示摘要](https://epapers2.org/biocas2024/ESR/paper_details.php?paper_id=2354)，CICC 2024 节目 14-6。

**阅读：官方身份＋作者伴随演示摘要，未读原 CICC 全文。** 已核到并行非零取数、三维阵列和注意力/卷积多模式调度。

- **纳入理由：** Gustav NRV∩W 之外还有真实非结构稀疏供数电路，不能只比较加法阵列。
- **本地洞：** θg 下有效权、实际 bank 冲突、规则 3×3 地址生成和非因果 T10 消费边界。不能把当前 θ 置 1 来取得适配。
- **尚未尝试：** 原作取数器、索引格式、bank 仲裁、累加宽度及全部模式。取得正文后先补强对照，不直接宣布新的三维阵列。

### 5. 样本自适应动态神经元剪枝 — CICC 2025

**A 40nm 0.05–1.4uJ/inference Sample-Wise-Adaptive Spiking Neural Network Processor with Dynamic Neuron-Pruning and Unstructured-Model-Aware Architecture**，Jinqiao Yang 等。[作者主页](https://fit.fudan.edu.cn/Data/View/6272)，CICC 2025 节目 11-5。

**阅读：一手题名和身份。** 动态神经元剪枝与非结构模型架构由题名确认，决策、训练、恢复和共享供数细节未核。

- **纳入理由：** 与 Grok 的“完整门字共同完成/少生产”直接相邻，应优先补正文。不能一边遗漏它，一边声称面向 SNN 的完成预测很新。
- **尚未尝试：** 全套原作策略及当前 T10 适配。缺正文不是机制失败，也不据题名认定它已覆盖我们的全部想法。

### 6. SparseTrim — CICC 2025

**SparseTrim: A Neural Network Accelerator Featuring On-Chip Decompression of Fine-Grained Sparse Model with 10.1TOPS/W System Energy Efficiency**，Jieyu Li 等。[作者履历论文表](https://www.ee.columbia.edu/~mgseok/cvs/cv_mgseok.pdf)，CICC 2025 节目 11-3。

**阅读：作者书目与官方题名。** 本轮未取得方法全文；不把二手评述里的具体编解码结构当已核原理。

- **纳入理由：** 细粒度剪枝后的瓶颈可能是权重编码和解压供数，不能继续只罚非零数量；当前已允许剪枝，新权重分布不等于旧稠密 FP32 分布。
- **尚未尝试：** 压缩表示、是否近似、解码吞吐、bank 和元数据整链。须与原始紧凑存储、bitmap/NRV、2:4 一起收费；“模型压缩”不能直接算周期。

### 7. Pro-Cache-CIM — CICC 2025

**Pro-Cache-CIM: A 28nm 69.4TOPS/W Product-Cache-based Digital-Compute-in-Memory Macro Leveraging Data Locality Pattern in Vision AI Tasks**，DOI [10.1109/CICC63670.2025.10983358](https://doi.org/10.1109/CICC63670.2025.10983358)，CICC 2025 节目 18-3。

**阅读：官方题名与出版身份，未得完整方法。**

- **纳入理由：** 若再考虑 C1 乘积缓存，必须查它，不能只与 Prosperity 的模式复用比较。
- **未证实部分：** 缓存对象、键、替换、一致性及读写代价仍需正文，不能直接把它等同父行森林，也不能凭名字说我们的机制已被做完。
- **处置：** 补为直接先验；当前已输的 Prosperity∪APEC 版本继续停止，不因新引用自动重开。

### 8. SP-IMC — CICC 2024

**SP-IMC: A Sparsity Aware In-Memory-Computing Macro in 28nm CMOS with Configurable Sparse Representation for Highly Sparse DNN Workloads**，DOI 10.1109/CICC60959.2024.10529009。[一手机构摘要](https://asu.elsevierpure.com/en/publications/sp-imc-a-sparsity-aware-in-memory-computing-macro-in-28nm-cmos-wi/)。

**阅读：一手摘要。** 支持列方向的 COO、游程和 N:M 等表示，以及精度配置和压缩权重消费。

- **纳入理由：** 运行时/分层换稀疏格式不是空白。NRV∩W 的比较必须付译码、路由、重写和冲突，不能把最小编码字节当实际吞吐。
- **尚未尝试：** 全文的格式选择、列路由和端口细节；本地真实稀疏权重下的映射。外围方法保留，宏 PPA 不移植。

## 保留为对照或待验证线索的八项

| 工作与 CICC 年份 | 一手来源/阅读深度 | 本地用途、保留或暂缓理由及未试部分 |
|---|---|---|
| **Quartet: A 22nm 0.09mJ/Inference Digital Compute-in-Memory Versatile AI Accelerator with Heterogeneous Tensor Engines and Off-Chip-Less Dataflow**，2024 | 官方节目 14-3，题名 | 连续算术与稀疏算术分工、片上阶段衔接的强近邻；完整引擎划分和状态生存期未读。先补方法，不引入其宏或将双引擎本身算创新。 |
| **A 38.5TOPS/W Point Cloud Neural Network Processor with Virtual Pillar and Quadtree-based Workload Management for Real-Time Outdoor BEV Detection**，2024 | 官方节目 21-3，题名 | 稀疏索引和负载管理可借；当前 DSEC 规则二维网格应先比较更便宜的固定偏移生成器。未迁四叉树、负载分配与 scatter；目前排在 HIRB 后。 |
| **A 1-TFLOPS/W, 28-nm Deep Neural Network Accelerator Featuring Online Compression and Decompression and BF16 Digital In-Memory-Computing Hardware**，2024 | 官方节目 26-3，题名 | 连续状态搬运的压缩对照；BF16 不等于当前整数合同。编解码、元数据和随机读取未核，暂缓实现。 |
| **iMCU: A 102-µJ, 61-ms Digital In-Memory Computing-based Microcontroller Unit for Edge TinyML**，2023 | 官方节目 7-6；[作者开放稿](https://par.nsf.gov/servlets/purl/10435540)搜索正文片段，本轮未完整读取 | 分层权重存储、较小计算缓存和多次 VMM 摊销可作容量/搬运基线；不能只借计算效率不借加载。未迁整层存储调度。 |
| **AI Processor with Sparsity-adaptive Real-time Dynamic Frequency Modulation for Convolutional Neural Networks and Transformers**，2023 | 官方节目 20-1，题名 | 输入相关时序/能量的电路方向；需要真实关键路径、时钟和 PVT 设计，不能由跳过率推合法频率。未读完整电路，当前先不实现。 |
| **A 22nm 0.43pJ/SOP Sparsity-Aware In-Memory Neuromorphic Computing System with Hybrid Spiking and Artificial Neural Network and Configurable Topology**，2023 | 官方节目 20-5，题名 | 混合 ANN/SNN 早有硬件；作为双轨分工近邻。精度、状态和通路切换未核，不把普通双轨直接恢复成标题。 |
| **DualLearn: A 4.686pJ/SOP-4.026pJ/FLOP SNN-ANN Neuromorphic Inference-Training Processor with Distributed and Memory-Adaptive Architecture**，2026 | [作者实验室](https://paicore.cn/)、[作者主页](https://ic.pku.edu.cn/szdw/zzjs/jcdlsjx1/zy/index.htm)介绍；DOI 10.1109/CICC65509.2026.11509522 | 分布式和可适配存储组织可借；当前固定推理无片上训练需求。未读全文，不为完整照搬而加无关训练硬件。 |
| **A Dynamic-Active Static-Sleep 18T Flip-Flop Supporting Retentive Clock Gating in 28nm CMOS**，2026 | [作者论文目录](https://mms.snu.ac.kr/?page_id=17)，题名 | 状态和时钟能量的相关电路；定制单元需晶体管/版图/库验证，普通 RTL 不能领取其能量。保留为电路近邻，暂不进当前实现。 |

## 从 CICC 光流原文追到的两项直接伴随先验

- **An Energy-Efficient Deep Convolutional Neural Network Accelerator Featuring Conditional Computing and Low External Memory Access**，**JSSC 2021**，56(3):803–813，DOI 10.1109/JSSC.2020.3029235。[一手机构摘要](https://asu.elsevierpure.com/en/publications/an-energy-efficient-deep-convolutional-neural-network-accelerator/)、[作者开放稿](https://par.nsf.gov/servlets/purl/10322669)。本轮核到按精度级联、先算高位再有条件细算，与跳零一起减少计算/搬运；原任务包含 FlowNet。已读一手摘要，未完整复现。其 MaxPool 条件不等于非因果 PSN 判决，但它是“光流＋少计算”的直接控制。
- **C-DNN: A 24.5–85.8TOPS/W Complementary-Deep-Neural-Network Processor with Heterogeneous CNN/SNN Core Architecture and Forward-Gradient-Based Sparsity Generation**，**ISSCC 2023**，DOI 10.1109/ISSCC42615.2023.10067497。[作者实验室介绍](https://ssl.kaist.ac.kr/bbs/board.php?bo_table=Neuromorphic&wr_id=3)。已核异构 CNN/SNN 分工及梯度驱动稀疏生成，未完整读方法。不能以“连续幅度＋发放标志”直接绕过这类混合执行先验。

## 补查后实际改变什么

1. **patch 预测/少生产的对照加重。** 除 TermiNETor、DynConv、CFMP 外，要补 Zhang 光流芯片和 CICC 2025 动态神经元剪枝。仍有待检验的差别是完整非因果 T10 的消费者如何取消尚未产生的昂贵前驱；不是“能早停”四个字。
2. **稀疏供数要完整迁移。** 非零取数阵列、SparseTrim、SP-IMC 加上另一份补检中的 HIRB/SIMO，说明格式、索引、解压与 bank 足以吃掉理想跳过收益。规则栅格普通索引必须同样优化。
3. **PSN 新候选不能一概按速度杀。** ROM-LTE、OS-CIM 提醒我们检查局部状态和切换。可预先定义一个新的同速/同面积能量问题，但不能把已失败速度版本临时改叫省电。真实能量仍须目标工艺和活动验证。
4. **数字 CIM 和模拟 CIM 分清。** 不因数字 CIM 宏无法直接采用而拒读它的调度与状态方法；也不把定制 8T、ROM 或 eDRAM 宏的面积/能量赋给 TS1N28 普通 SRAM。

本轮补查纠正了 CICC 覆盖明显不足的问题，尚未完成全部相关论文的全文与工件承接。后续先补与拟议 X 直接相交的正文，不靠继续增加题名代替机制验证。
