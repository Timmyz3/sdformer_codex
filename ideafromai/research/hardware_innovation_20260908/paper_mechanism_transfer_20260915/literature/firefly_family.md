# FireFly 家族：全文机制、开源边界与可检验迁移

2026-09-15。本文只读论文、作者公开代码和本方既有实验；没有新增训练、RTL 实验、EDA 或性能测量。**FireFly 的可借部分远多于“二值加法”，但各篇依赖不同的供数、状态和模型条件。** 本方静态 AT-LIF 幅值可折权，不能以“输出是 θ 而非 1”否定选择加法；也不能把已经完成的 bitmap7、普通 RR、端点重排、Q2 DA 或生产摘要重新列成新机制。

## 1. 身份与全文版本

| 论文 | 正式身份 | 本次读到的全文 | 作者代码证据 |
|---|---|---|---|
| **FireFly: A High-Throughput Hardware Accelerator for Spiking Neural Networks With Efficient DSP and Memory Optimization** | Jindong Li、Guobin Shen、Dongcheng Zhao、Qian Zhang、Yi Zeng；IEEE TVLSI **31(8), 2023, 1178–1191**；[DOI](https://doi.org/10.1109/TVLSI.2023.3279349) | [arXiv 2301.01905v5](https://arxiv.org/abs/2301.01905)，2023-06-06；[本地 PDF](firefly_original.pdf) | [作者 FireFly-v1](https://github.com/adamgallas/FireFly-v1)，MIT；硬件、仿真、部分板端材料均存在 |
| **FireFly v2: Advancing Hardware Support for High-Performance Spiking Neural Network With a Spatiotemporal FPGA Accelerator** | 同五位作者；IEEE TCAD **43(9), 2024, 2647–2660**；[DOI](https://doi.org/10.1109/TCAD.2024.3380550) | [作者接受稿](https://floyedshen.github.io/pdf/li2024fireflyv2.pdf)，14 页；[arXiv 2309.16158](https://arxiv.org/abs/2309.16158)；[本地](firefly_v2.pdf) | [作者 FireFly-v2](https://github.com/adamgallas/FireFly-v2)，MIT；硬件生成源和仿真存在，当前仓库无 v1 那样的板端 Python/bitstream |
| **FireFly-S: Exploiting Dual-Side Sparsity for Spiking Neural Networks Acceleration With Reconfigurable Spatial Architecture** | **Tenglong Li 第一作者**，随后上述五位；IEEE TCSI **72(8), 2025, 4007–4020**；[DOI](https://doi.org/10.1109/TCSI.2024.3496554) | [作者 IEEE 接受稿](https://floyedshen.github.io/pdf/li2024fireflys.pdf)及[arXiv 2408.15578v3](https://arxiv.org/abs/2408.15578)，后者修订于 2026-01-29；[接受稿](firefly_s_published.pdf)、[v3](firefly_s.pdf) | 论文、两位合作者主页与公开仓库检索中，**未定位到作者明确对应 S 的完整训练/硬件仓库**；不能据此断言作者从未公开 |
| **FireFly-T: High-Throughput Sparsity Exploitation for Spiking Transformer Acceleration With Dual-Engine Overlay Architecture** | 同 S 六位；IEEE Transactions on Computers **75(6), 2026, 2185–2199**；[DOI](https://doi.org/10.1109/TC.2026.3672901) | [arXiv 2505.12771v1](https://arxiv.org/abs/2505.12771)，2025-05-19，14 页；[本地](firefly_t.pdf)。正式 VoR 全文未取得，下面方法判断以该预印本为准 | 同样未找到作者明确对应 T 的完整实现仓库；不能把 v2 的公共 sparse 库当作 T 已开源 |

卷期页由出版社向 Crossref 登记的 DOI 元数据核对，原始返回保存在 `firefly_bibliography_*.json`。T 的[IEEE 条目](https://ieeexplore.ieee.org/document/11432885/)访问遇到脚本验证，正式 PDF 请求未成功；其正式身份与本次方法全文版本分别陈述。[Jindong Li 主页](https://adamgallas.github.io/)仍把 T 放在 preprint 栏，不能以主页未更新否定正式刊载。S 的 DOI 年号 2024 与卷期年份 2025 也不是冲突。原 FireFly 早期标题带 “Reconfigurable”，不是第五篇。

### 实际代码检查

没有编译或运行作者工程，以下是源码静态检查，不是复现实验通过。

- v1 当前归档有 159 个 Scala 文件、4 个 Python、2 个 `.bit` 和 2 个 `.hwh`；包含 `src/FireFly/source`、`src/FireFly/sim`、公共 `src/Lib`、Ultra96/ZCU104 测试和 CIFAR10 材料。**这些不能自动等价为论文所有网络的完整训练与板端复现包。** [源码副本](firefly_code_v1/)、[顶层](firefly_code_v1/src/FireFly/source/FireFly.scala)、[DSP 选择加法](firefly_code_v1/src/Lib/source/xilinx/DSP48E2/FOUR12MuxAdd.scala)。后者实际配置 `USE_MULT="NONE"`、`USE_SIMD="FOUR12"`，两路选择位控制 0/操作数。
- v1 的 [StreamCycleFifo](firefly_code_v1/src/Lib/source/fifos/StreamCycleFifo.scala)有 `pushPtr/popPtr/markPtr`、长度与复用计数、满空相位和 ready/valid；反复读取的区域在最后一次使用前不会被后续 push 覆盖。它是有生命周期的权重复用 RAM，不能简写成“一个免费 FIFO”。
- v2 当前有 217 个 Scala 文件，**0 个 Python/bitstream/hwh**。关键代码位于 `src/FireFlyv2_2`，不应因目录名 `_2` 再造一个论文版本。[BareMetal](firefly_code_v2/src/FireFlyv2_2/source/FireFlyFasterBareMetal.scala)实际连接输入转置、im2col、weight cycle、psum/neuron 和 shortcut；[SpikeWeightCalcInt8AddWrapper](firefly_code_v2/src/FireFlyv2_2/source/SpikeWeightCalcInt8AddWrapper.scala)将 im2col、weight FIFO 与下游 ready 三者联锁。
- [ReducePsum](firefly_code_v2/src/FireFlyv2_2/source/ReducePsum.scala)实际实施二/四/八位重建和 holding；[SewConnectAddTwo2Two](firefly_code_v2/src/FireFlyv2_2/source/SewConnectAddTwo2Two.scala)的 `sat` 是实际截幅。因此，“所有多位支持均等价无损”不是代码支持的结论。库中出现 sparsev1/sparsev2 也不足以证明它们就是后来 S/T 的论文实现。

## 2. 四篇分别改掉了什么假设

### FireFly：二值输入不应照搬 ANN 乘法，DSP 也不只可用乘法器

**问题和旧假设。** ANN FPGA 阵列通常围绕多位乘法和 MAC packing；SNN 原生 0/1 输入只需在权重与零之间选择。把乘法器留着做这件事浪费资源，但把全部累加搬到 LUT 也错失 DSP 内部的宽 ALU、寄存器与级联。见全文 §III–V、Algorithm 1、DSP 和 memory optimization 图。

**计算及并行。** 8 位有符号权重扩展到四个独立 12 位字段，DSP48E2 关闭乘法/预加器，两个 spike 分别控制两组权重，单 DSP 形成 **2 输入×4 输出**的选择加法。级联累加与 LUT 归约组合成更大的通道/核窗口阵列；12 位字段界限决定可级联规模，不能把 4×12 当作四个无限精度加法器。它是 dense 排程里的 mux/gating，**并非每个零都缩短周期的稀疏 decoder**。

**供数和布局。** Weight-stationary 阵列，输入横向广播、部分和级联；输出通道分组外层、T 再外于空间位置，复用权重。Partial-reuse FIFO 用同一 RAM 的受保护区段多次 replay，后续权重可以写入已释放空间；上下游通过宽度变换和 skid buffer 解耦。3×3 line buffer 与 MLP 路径有不同组织。将跨 Ci 分片的 psum 与跨 t 的膜电位共用存储，依据两种数据的存活期安排读取/更新/最后清零。

**能借与不能借。** 选择加法、位段独立、replay FIFO、存活期复用都是可以忠实迁移的 A；AT-LIF 的静态 θ 乘入 W 后仍成立。膜电位更新/阈值/复位是 IF/LIF 时间递推，不能替换本方任意稠密 T×T PSN。原论文的 300 MHz/5.53 TOP/s 是其 FPGA 和计数口径；v2 明确说明原 FireFly 的部分延迟比较未计 direct-coding 层，不能拿它当我们完整 I24 服务分母。

**引用链。** ANN 数据流 Eyeriss、低精度 FPGA/DSP packing，以及 Xilinx DSP48E2 手册是近邻；它的创新不是发现 `0×W=0`，而是器件映射和存储服务组织。[原文](https://arxiv.org/abs/2301.01905)

### FireFly v2：把非 spike 的位宽组织与时间组织拆开

**问题和旧假设。** 初代固定 3×3/kernel 并行、空间膜电位大缓存、仅二值主路；但真实 SNN 有直接编码首层、多位 ADD residual 和 average pooling。全部当 0/1 不成立，另建大通用 MAC 引擎又损失硬件复用。全文 §III–VI。

**运算表示。** 将多位输入分解成等效时间位，复用同一二值选择加法阵列，再用带权 shift-add 恢复原整数；S=4 时二位一次合成两个结果，四位和八位需要多轮及 holding。直接编码静态像素的卷积只需计算一次，再向实际 T 复制。**位分解/带权恢复可精确；将 ADD residual 限到二/四位的 saturate-or-shift 是另一个有损选择。** 平均池也要跟踪缩放，不能把拆位执行与丢位混成一个定理。

**调度和供数。** 并行轴为输出通道 M、输入通道 V、像素 N、等效时间 S；循环大致为 `(Co/M, Ho, Wo/N, T/S, Kh, Kw, Ci/V)`，阵列内展开 M/V/N/S。output-stationary 替代初代 weight-stationary，将因果神经元的 T 连续完成，只保留约 M×N 个残余膜电位；代价是输入 spike 的全时间组织，并非“完全没有状态”。可配置 im2col、banked 数据重排和 read-DMA 调度适应核大小/步幅；两条 128-bit 读 AXI 加一条写 AXI 是论文系统的真实供数条件。

**频率和神经元。** 继承 2×4/DSP，DSP/FF 域用 500–600 MHz、其余逻辑约半频；gearbox 两侧保持带宽，不能在本方单时钟周期表里凭空翻倍吞吐。两阶段 IF/LIF 更新借鉴 carry-lookahead：预先计算 spike 候选与后缀和，再选择正确状态，仍实现因果递推，**不是非因果 PSN 的矩阵变换**。IAND 和 ADD residual 有独立实现；IAND 的对象是真二值 shortcut，不能用于我们的 FP identity→J20→I24 相加。

**引用链。** §III 引用 Bit Fusion [21] 支持位分解动机；本文去读了原作，后者核心是空间可组合 BitBrick，不能被缩写为单纯 bit-serial。双频来自 Vitis AI DPU 的器件组织；空间/时间数据流对话 SpinalFlow、SATO。[作者全文](https://floyedshen.github.io/pdf/li2024fireflyv2.pdf)

### FireFly-S：稀疏率来自训练，空间层流水要求不同的布局

**问题和前代假设。** v1/v2 以通用 overlay 时间复用层，未充分利用权重稀疏；若直接把 v2 的输出通道最外层用于整网空间流水，每层必须等许多通道，输入 feature map 要长期保存。S 改为每层一个专用 pipeline stage，并联合剪枝量化以使整网权重容纳于片上。全文 §III–V。

**剪枝。** 借 gradient rewiring，参数化 `w=s·ReLU(ϑ)`，s 由初始权重符号确定，ϑ 带先验稀疏罚项；加入 weight decay/AdamW，覆盖 convolution、FC 和 bias。这里的训练参数 ϑ **不是本方 spike 幅值 θ**。权重全零不保证输出全零：还有 bias 与神经元。Algorithm 1 对该通道从零初态沿 T 检查 bias 驱动是否越阈，只有始终 silent 才可移除。算法伪码是累加 bias 的 IF 式形式，正文讨论 leak；不能据此声称已证明所有 LIF/PSN 的静默性。

**量化。** LSQ 原作学习权重/激活量化步长，S 扩展为每通道共享 scale 联合优化 W、bias 和神经元 threshold；训练有量化再反量化，推理保留整数神经元。二值输出不需携带普通 ANN 连续输出的反量化尺度。这个结论有神经元比较/状态共同缩放的条件；连续 Q2 或 FP residual 不能直接删掉 scale，也不能跳过我们规定的 RNE 边界。

**decoder。** 一组 spike bitmap 与 weight mask 先 AND；popcount 记录待处理数；每次用 `y=x & ~(x-1)` 取最低活动位，再清掉 y。因为 W 以非零值压缩存放，还需对该位之前的 weight mask 做 prefix-popcount 得到压缩地址，随后读 W、累加、bias/neuron。Fig.7 的路径从匹配到有效权重约 CLK0–CLK6，**交集成立不等于已经拿到权重**。多输出通道各有 detector；bias 可填入检测/取数 bubble，但没有消失。

**布局。** 将通道循环放入像素循环，同时保持一个神经元的 T 连续处理；保留当前卷积滑窗所需输入，而非每层完整 feature map。本文 Table I 给出 Sbuf 约 `((Kh−1)·Fwo+Kw)·T·Ci`，Vbuf 为 PCo。Orchestrator 是可部分 replay 的环形 FIFO，加多维计数器/stride，在 ready/valid 下沿 im2col 重用源。数据不是 TB 免费预转置。

**条件与证据。** 4-bit、约 85–95% W 稀疏是其训练后模型结果，不能套给现有 R8 或 FFN；COO/CSR 对比还固定单值解码吞吐，不能作为所有稀疏格式的最强控制。S 的 spatial 架构不支持把每层资源当作本方单份 8 ALU/8 mult。

**一个应保留的原文问题。** 接受稿和 arXiv 摘要把 10047/3683/2327 标成 FPS/W；arXiv Table VI 实际把它们列在 **FPS**，对应功率 1.301/1.821/3.799 W，FPS/W 是 **7722.52/2022.52/612.53**。本文采用表中列含义，并保留该单位矛盾。MNIST、DVS-Gesture、CIFAR10 分类精度不等于本方 AEE。[作者全文](https://floyedshen.github.io/pdf/li2024fireflys.pdf)、[arXiv v3](https://arxiv.org/abs/2408.15578)

### FireFly-T：解码、广播和专用注意力分别解决三个瓶颈

**问题和前代假设。** 单次只取一个 spike 的 decoder 无法使用更宽输入；简单增加稀疏 PE 会增加独立 W 请求和 bank 冲突；只支持 convolution 的主引擎又不适配 binary attention。T 回到层间时间复用的 overlay，并增加两个不同引擎，**不是给 S 空间流水换个 decoder**。全文 §III–IV。

**多路 decoder。** 从 carry-lookahead 得到跨 lane 的前缀“第几个 1”逻辑，单周期给出前 M 个非零位置，并清除这 M 个位；初始 popcount 与每拍减 M 的 tracker 控制换输入。组内/组间 carry 组织减少宽输入逻辑深度。它不是同时做 M 次 RAM 读取，更不是 M 倍免费 ALU。§V 对 PCi/M/worker 做不同配置比较，本方不能原样沿用其最优参数。

**供数与乱序。** 各输出通道有对应的宽权重 bank，宽 PCi 权重向量广播到空间 PFx×时间 PTs 的 worker；同一批解码索引跨 Co 复用。另设 PWo worker，在 Kh×Kw×Ci 的工作块之间向空闲 worker 发射，缓解工作不均；利用的是空间/时间邻近任务的稀疏度较接近、宽块能供给多 worker 的经验条件。消除了原先“各 PE 任意请求多个 bank”的路径，**并没有在任意窄口系统上消除带宽约束**。Table VI 中 balancer 占 sparse engine LUT 约 75.86%/84.20%，decoder 仅约 5.73%/3.62%；主要代价不在找 1。

**binary attention 与布局。** 稀疏引擎处理 binary-input convolution/linear，二值注意力引擎是独立 AND-popcount systolic array。对 QKᵀ 与后继乘 V 的布局需求，用 SRAM `(bank,address,byte)` 三维写地址、byte mask 和旋转，在写入时完成 `(L,d,T)→(T,d,L)`；读端逆旋转，省掉显式二次转置缓冲。它依赖分 bank、byte-write 能力、每拍输入/输出宽度匹配和数据尚未被读取的生命周期。不能拿“原地”当作零成本多写口。

AND-popcount 的 LUT6 优化先把三对 AND 与局部计数合入 6 输入逻辑，再用 6:3 compressor 归约；Fig.9 的 18-bit 示例由 50 降到 24 LUT。这是器件级 AND-popcount，不适用于任意连续权重乘法。注意力阶段的隐藏依赖 sparse/binary 工作与并行度的平衡，式(4)给出 `Pb≈(2/3)(FhFw/Ci)Ps`；它只是可重叠条件，不能自动把 `3TsLd²+2TsL²d` 的真实总服务变成第一项。

**残差及范围。** pre-neuron membrane shortcut 仍是连续相加，由单独的数据路径和第四条 AXI HP 通路服务；不是把 residual 改为 AND。主 sparse engine 重点利用**激活**稀疏，不能直接把 S 的 85–95% W 稀疏/压缩供数合并进 T 的成绩。T 论文给出相对 v2 的能效/DSP 效率提升，但模型、位宽、T、硬件和功能单元不同；这些比值不移入本方表。

**引用链。** T 引 SparTen [26] 讨论稀疏解码吞吐、引 Trapezoid [29] 讨论互连/冲突近邻；不能据引用就说 T 实现了整个 SparTen 或 Trapezoid。AND-popcount 归约近邻是 Wallace [35]；binary engine 对照 SpikeTA [24]；输入/空间组织也对照 Fang 等 3D SNN array [25]。[全文](https://arxiv.org/abs/2505.12771)

## 3. 反向追读的三篇非 SNN 原作

这三篇均已取得 primary 全文并读方法和评估限制；不是只抄 FireFly 引文。SparTen 的作者出版页和 DOI 已核对，但原文 PDF 请求受限，本次**未把 SparTen 算进全文完成数**；不据二手摘要写其具体电路。

| 原作与实际关系 | 原作的方法 | FireFly 选择了什么；本方不能省什么 |
|---|---|---|
| **Bit Fusion**, Sharma 等，ISCA 2018, 764–775；[作者 PDF](https://jongse-park.github.io/files/paper/2018-isca-bitfusion.pdf)、[DOI](https://doi.org/10.1109/ISCA.2018.00069)；v2 [21] | §II–IV：2-bit BitBrick 按 operand 位宽空间组合为 fused PE；16 个 brick 可形成不同 2/4/8-bit 组合，较宽16位用混合时间执行；配位宽相关 buffer 供数、shift-add、ISA 的循环与地址生成。原作明确区分空间融合与完全 bit-serial，纯时间方案的 shifter/accumulator 成本高。 | v2 借“不同精度可分解/重建”的动机，却主要复用二值阵列时间槽，并未移植 BitBrick。我们已有 Q1 signed3 三 plane、native4P/borrow 与 Q2 DA；不能再以“Bit Fusion”命名同样的拆位。若比较宽度可组合路线，必须把 signed13/19/32 的恢复和共同 SRAM 带宽算上。 |
| **Trapezoid**, Yifan Yang、Joel Emer、Daniel Sanchez，ISCA 2024, 931–945；[作者 PDF](https://yang-yifan.github.io/papers/isca24_trapezoid.pdf)、[DOI](https://doi.org/10.1109/ISCA59077.2024.00072)；T [29] | §III：dense IP；中等稀疏 TrIP 对多行 A×多列 B 同时交集，MFIU 用 bitmap AND/prefix 产生路由，两个 Benes network 分配 A/B，merge-reduction 支持多目的输出；高稀疏 TrGT/TrGS 使用不同 Gustavson 时空组织。多级 cache/local banks 在不同模式分别服务输入或输出。 | T 把它列为跨 bank 互连的近邻；T 的宽 W 广播不是 TrIP 多侧交集加双分配网络。Trapezoid 的优势依赖交集、归约、scatter、cache 和复用完整组合；“只做 bitmap AND”不等于迁入原作。其 ASIC 归一化性能/面积比较不能当本方 FPGA 同面积证明。 |
| **Learned Step Size Quantization**, Esser、McKinstry、Bablani、Appuswamy、Modha，ICLR 2020；[原文](https://arxiv.org/abs/1902.08153)、[IBM 论文页](https://research.ibm.com/publications/learned-step-size-quantization)；S [32] | §2：`q=round(clip(v/s))`；STE 得到对 step size 的分段梯度，内区为 `round(v/s)−v/s`；梯度规模按权重/feature 数和量化上界归一。原作每层 W/activation 各有 FP32 s，训练保存全精度权重，低精度前后传；推理整数 MAC 后仍需尺度恢复，可与 BN 合并。 | S 扩为 per-channel W/bias/threshold 共享 scale，使整数神经元闭合。它不是无需质量评估的精确量化定理。对本方应把静态 θ 折入 W、Q1/Q2 factor scale、BN、PSN 和最终 I24 看成一条实际数值链；不能删中间缩放或继承 S 的分类准确率。 |

补充：Bit Fusion 在作者 PDF 的作者列与 v2 参考文献的转录有差异；本文身份以原论文作者栏为准。T/S 的原作阅读不构成其代码复现。相关 PDF/text 均用 `firefly_` 前缀保存，供定位章节。

## 4. 对本方接口的核实，而非泛泛“SNN 相似”

1. **AT-LIF 输出支持为 `{0,θ}`，θ 静态。** 可预先构造 `W′=θW`，硬件消费二值 g。非零幅值本身不阻碍 mux/AAC；θ 的量化与折入顺序仍应由真实 gold 固定。
2. **R8 的两个线性层之间没有神经元/RNE。** `z=Q1g`，`p=Q2z`，Q1 signed3、z signed13，Q2 连续整数权重；不能把 z 当 spike。已有 shared R8 的 source/origin/FP identity 与空间版本对同一 r0.conv2 挂点，完整输出经 FP identity→J20→wide→I24，借 consumer 宽链必须和真实消费者争用。
3. **非因果 T10 允许线性子图内部重排，但不允许删依赖。** PSN 可能用完整 T×T 权重；FireFly 的递推 LIF neuron 不能直接替换。已有端点固定 T 排列+inverse 已实试；普通时间重排不再算新的候选。
4. **FFN 有额外真实屏障。** 本地 `psn/screen_psn_residual_bound.py::reconstructed_fc1` 重建的是 `fc1→bn1→sn2(PSN)→fc2`，BN 使用完整 T×P 域统计，PSN 用全 T。不能把 S 的“每像素做完立即下层”直接移来，假装统计已就绪；部分旧捕获是 float64 重建而非完整 FP32 producer，必须给候选重新固定 exact gold 边界。[本地重建源](../../psn/screen_psn_residual_bound.py)

已有最强共同 RTL 分母是 [shared R8](../../shared_execution_20260915/r8_reference/README.md)：bitmap7 的 held64/disjoint64/18seq36 cold 完整服务 **750650/792116/357261**，native4P 为 787604/831997/376567。两 context 共 4160 B Z、一个 416-bit Z 服务、单份 8×32 producer ALU 与 8×19×13 mult；consumer 是单份 8×64 链和 8×32×32 mult。bitmap 的 8 个 pop16、2592 B plane 与活动状态单列。普通控制享受相同容量许可，不声称等 Fmax/PPA。

[旧 Q2 DA 适配](../../transfer_adapt_20260914/da_adapt/REPORT.md)已经从 100043 改到 94823，仍慢同函数 MAC 的 87599。检测/构表税共 7992 拍，只少 768 次后端发射。**再提出“连续 Q2 拆位+小 LUT+lazy”会重复已试接口**。同理，[Gustav 原生源字生产/准入](../PRODUCER_EXPERIMENT.md)本轮已完成 1728 任务；root 回报 seen-code 与普通 flag 同拍，没有独立 X。本文不把另一张 producer nonzero 摘要当独立新候选；该条数字由主实验报告负责，本文未重复运行。

## 5. 三条 A→不适配洞→X 分析，其中仅两个优先未试接口

以下是待验证假设，不是已取得收益或新颖性。前两项是本轮建议保留的未试执行接口；第三项用于说明剪枝家族的迁移边界，不另开一个 RTL 任务。每项先迁入 A；A 有效才讨论组合增量，A 无效则记录实际失败依赖并只改一处。**其中普通广播、写时转置和组剪枝本身都属于成熟强控制。** 只有在对应强控制后仍有可解释增量，才值得把 X 写进贡献。

### 候选一：把宽 W 广播缩到单 W grant 下的跨 context 共同驻留

**A。** 迁入 FireFly-T 的“同一 W 向量服务多个空间/时间任务”与 v1 的受保护 replay 区。在本方先只针对连续 Q2：相同模型、相同输出 N8 的八个 rank 权重可供两个 context，避免两个 context 各自重复 VLOAD。它仍是 ordinary weight-stationary/broadcast 强控制。

**已经做掉与剩余部分。** bitmap7 已在每个 context 的一个 K16 内，将三 plane 经 W 口装入 qcache 前三行，随后供该块所有 P/T 重用；m7 的 z_hold 也已合并同 T 行提交。Q2 已按 N8 缓存八 rank，绝不能再将每个位置重读 W 作为弱对照。跨 K 的不同地址不能因调度相邻当作同 W；如果依靠相等权重合并，也必须先对照 count21 的 class/rep。实际剩余是两个 context 对**同一个模型、同一个地址**的重复装载，以及有界乱序下能否使这些请求相遇。

**洞。** T 假设宽 bank 足以覆盖多个 worker，而且邻近任务负载接近；本方仅一个 256-bit W 服务、一个 Z grant 和八乘法，两个 context 的 Q1 完成时间又不同。强制等齐可能增加首输出延迟，并阻挡另一个 context 的 I24 退休；每 context 的 bitmap Q1 还借用 qcache，不能在它未退出时覆盖。

**X 假设。** 只在已有 qcache 已被 Q2 接管、两边请求相同 `(model_epoch,N8,rank)` 时做显式共同驻留；设置每 context 的 pending/consumed 所有权，消费者独立按原单端口发射。未到齐者不得阻塞已有可执行 Q2；驻留到下一次真实替换而非按预测超时。这个“异步完成与缓存存活期相交”的收益必须和普通固定双 context cohort 比较，不能归给新的 RR。

**RTL 三臂。** A0 当前 bitmap7/count21/native4P/borrowRR；A1 固定 N8 cohort 共享装载；X 同地址机会广播与独立完成。三者都可使用同一 union cache/holding 容量，W 接受一次最多填一个既定物理 cache 写端；若需要把 payload 写两个 cache，必须两次收费写，不能假装一次写双副本。可改为一份共享 cache 两 context 按周期读，但必须计读取仲裁与状态。先只一处 2-context block，不添 worker、W grant、Z grant 或乘法。

**收益上界与判据。** Q2 每 tile 至多 96 个 N8/rank 向量；64 tile 两两共同装载，理想最多省 3072 次 W 接受，仅相当于 750650 拍的 0.409%（这还是把每次都当关键路径且未扣任何费用的乐观服务上界）。因此先核关键路径，不能为了微小理论上限扩控制器；Q1 同 K16/同 native-K 的跨 context 合并才可能提供额外机会，其收益必须独立计数，不能再次声称已有块内驻留。看减少的 W 接受/写入是否大于 cache 仲裁、等齐和回退费用；完整 raw/J/I24 cold/warm/BP、两64+18seq，含一方全空/长密、跨 mode、换源与模型换参失效。增加 `same_weight_hit`、`waiting_peer`、`cache_owner_stall`、Q1/consumer overlap 和最后 I24 退休时间。若现有 W 从未在关键路径，停止此放置，不能报“读次数减半即提速”。目前未有此实际跨 context 共同 cache 测量。

### 候选二：在真实 FFN 屏障之后用最终写入完成布局，而非借因果 LIF 删除屏障

**A。** 忠实迁入 T 的 `(bank,address,byte)` 写时重排，把一个已完成的 T10 spike tile 写成下游 FC2 可按 channel-block 一次读出 T 支持的布局；与“原布局写入、再显式 transpose”比较。**不改变 PSN 计算，也不改变 BN 统计服务。** 普通 bank-skew/写掩码转置是成熟 A。

**洞。** S 的每像素连续 T 依赖因果神经元与流式上下层；我们的 BN 跨 T×P、PSN 全 T，而 FC2 使用每位置/channel 的 T 门字。直接流送会过早读未完成门，或隐藏完整域统计。T 的 byte-write 也不等价为本方任意 bit-address SRAM；T10 尾部、通道尾部和 BP 会制造未写字段。

**X 假设。** 在“BN 已真实完成、PSN 全 T 结果最终确定”这一明确边界，复用既有源 bank 中已退休的存储行，通过最终门写入地址/掩码形成 FC2 原生布局；用完成向量而非部分非零摘要授权读。PSN 需要全部 T 是不变条件，改变的是**最后写入的顺序与 bank 所有权**。后级仍可用原生 direct 或已有时间共享归约，禁止把后级算法改动一起算入布局增量。

**RTL 三臂。** A0 完整 BN/PSN 产生原序门→收费 transpose→FC2；A1 独立同容量双缓冲写时转置→FC2；X 单物理 bank 的退休行复用/写时转置→同一个 FC2。需真实 RAM 接口，FP/整数数值边界固定；T10 的 6 个无效填充位不得产生事件。尾块若只能全字写，实际读改写收费；不提供额外 byte-write 权限。若 A1 已省掉同样搬运而 X 只省状态，应按状态取舍报告，不宣称更快。

**判据与现有缺口。** 从真实 fc1 源输入到 FC2 最后值计算，统计 mean/variance/rsqrt 和非因果 PSN 都必须在总账；先可用已产生连续输入的 suffix 单独测，但明确不是全 FFN。定向测试跨 T permutation、通道余数、读写同 bank、消费者迟到、连续位置覆盖。现有 C2 时间共享 RTL只收已生成门/诊断载荷，没有真实 producer transpose/BN 闭合；这项针对该缺口，而不是重做 C2 字典或 Gustav nonzero 摘要。[旧 C2 边界](../../../c2_temporal_shared_protocol_20260907/README.md)

### 第三条迁移边界：S 剪枝目标与本方发射组；暂不列为新执行接口

**A。** 迁入 S 的 gradient-rewiring+LSQ 联合训练机制，但保持本方模型图、静态 θ 折权、完整 BN/PSN/identity 路径；作为 algorithm A，先测自身质量，再在冻结函数上实现真实 dual-side bitmap/压缩取数。与普通幅值剪枝、普通 N8/block 组剪枝比较。**没有执行本轮训练，以下是后续提案。**

**洞。** S 的每输出 detector 可以利用单个 W 为零；本方一拍八 lane MAC/AAC，单 lane 零未必少一次 W、Z 或 ALU 发射。更稀疏 Q1 也可能让连续 z 仍然稠密；更小 Q2 参数可反而把邻域组合搬到连续侧。shared 空间实验已经出现 Q2 发射约为 R8 的 3.1 倍，因此“25%系数为零”不是系统目标。LSQ 的 scale 也不能跨连续残差任意消去。

**X 假设。** 冻结执行原语为“同一 rank→N8 的整个有效向量”和本方 K16/Q1 布局，训练/投影只改变一处：按真实服务组的联合支持而非孤立标量零数计代价；把 Q1 binary AAC 与 Q2 多位 MAC 的不同代价明确区分。候选不准凭标量稀疏率选择位宽或增大表。所有 scale、阈值、投影后 q2 与边界均需新的 CPU gold 与完整 AEE，质量门采用当前**优于 NB0**，不沿用旧 +0.005 gate。

**后续算法对照；本轮未运行。** 用户已授权算法训练，无需重新申请权限。算法 A1 标量剪枝+LSQ、A2 普通相同组 N8 剪枝、X 联合 Q1/Q2 服务代价；各自产生冻结函数后，都跑同一普通 native/bitmap7 以及同资源稀疏执行臂，禁止把不同函数的硬件数直接相除当纯加速。先已冻结函数的真实 N8 支持统计和小完整 RTL，只有净真实收益才两64/18seq/完整消费者；完整 825 AEE 单独做。元数据生成、压缩地址 prefix、group-live、模型加载、所有 clear 和消费者工作全收费。

**贡献上限。** 常见硬件感知/组结构剪枝已是很强 ANN 近邻；若 X 与 A2 的同质量前沿一致，归入普通组剪枝底座，不宣称新公式。比已有 moment/native_tap/free-U 剪枝多出的待验证部分是**面向二值前级与连续后级不同实际发射费用的联合约束**，不是再任意去掉一个 Winograd M。若训练无净质量/服务点，记录受限条件，不杀整个剪枝家族。

## 6. 本轮结论的可用范围

论文读法应是：v1 的 DSP/状态复用，v2 的多精度等效时间与供数，S 的训练压缩加空间布局，T 的多路解码、宽广播和注意力布局，各自都有真实条件。四篇作者都没有给出本方 R8+PSN+I24 的结果；本文也未声称迁移已成功。

优先未试接口仅保留两个：跨 context 同地址 W 服务，以及真实 BN/PSN 屏障后的 FFN 写时布局。前者应先看上述小收益上界，普通广播/cohort/异步 RR 均是 A；后者不能仅抓一份门字作整链。第三条组代价剪枝只保留边界分析，当前不建议追加任务，普通组剪枝很可能解释全部收益。bit-serial、bitmap、普通 RR、写时转置、组剪枝均不是本文新发明。S/T 代码与 T 正式版本全文仍是明确资料缺口；本轮交付的是来源可核的机制分析及对照合同。
