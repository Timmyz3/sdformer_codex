# 快速时间坐标的膜量化：先验迁移与剩余判别

**B：**当前 `fast_temporal_basis/control` 的四层蝶形满足 (F^TF=16I, B=F/4)，canonical (Q=BI) 同时供 source/proj 两个独立 gain、置换、bias、θ 读出，以及压到 R32 后的连续逆恢复。此前 dense As 的 cond≈24、逆矩阵最大行 L1≈33；这里 cond=1，但这只是条件数改善。已有 fixed 四轴的 24-bit 写回零饱和，不能外推为 fast B 的低位 AEE 已通过。

|完整先验 A（已读方法正文及相关作者代码）|可迁移部分与原作已覆盖的边界|
|---|---|
|[QuIP#，ICML 2024](https://proceedings.mlr.press/v235/tseng24a.html)；[官方代码](https://github.com/Cornell-RelaxML/quip-sharp)|校准 Hessian、随机符号/正交变换、block-LDL 误差反馈、E8P 权重码本与后续微调是一整套 **weight-only** PTQ。不能把其码本收益算成 Q 状态压缩；在线变换、解码与元数据仍收费。当前 T10 蝶形不是 10 阶等幅 Hadamard，不能直接套用其不相干性保证。|
|[QuaRot，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b5b939436789f76f08b9d0da5e81af7c-Abstract-Conference.html)；[官方代码](https://github.com/spcl/QuaRot)|已有残差隐藏状态旋转、norm gain/旋转离线折权、不能折叠处的在线 Hadamard、GPTQ 权重与逐 token 动态激活量化。应一并承接 scale 求取、量化/反量化及仍保高精度的操作；“在线旋转＋动态膜量化”不是新点。|
|[SpinQuant，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e5b1c0d4866f72393c522c8a00eed4eb-Abstract-Conference.html)；[官方代码](https://github.com/facebookresearch/SpinQuant)|已有量化任务损失驱动的 Cayley 正交旋转学习：可吸收到权重的旋转学习，不能吸收的在线 Hadamard 保留；GPTQ 配方先在 W16/A4/KV4 下学旋转。我们的 20 个符号 STE 是受限具体化，不能称已复现其连续旋转优化。|
|[IM-SNN，ICONS 2024 正文](https://par.nsf.gov/servlets/purl/10545833)；[SpQuant-SNN，Frontiers in Neuroscience 2024](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1440000/full)；[作者代码](https://github.com/SeoLabCornell/IM_SNN-SpQunat_SNN)|已有量化膜写回、负膜裁剪、stacked-gradient surrogate 与 PoT 权重；SpQuant 还包含空间/通道 mask 和阈值梯度路径。其 LIF 发放、复位后量化的膜状态，与我们仍需连续恢复的 Q 不同；不能把负 Q 当作可丢的静默膜，也不能把原阈值学习混同输出幅值 θ。|

**针对性迁移与 X：**完整借鉴“变换折叠＋实际状态量化写回＋量化任务恢复”，把量化器放在 Q 的真实保存位置和残差更新后；工作累加器另计。Q 不随 source 发放复位，两个门读出保持各自 θ、bias/center。静态 gain 才可按实际整数格点编入有向阈值（负 gain 用 ≤、零 gain 为常量）；动态 scale 必须付范围归约、scale 保存及阈值/残差对齐费用。当前蝶形每遍需要四个保护位，最后除 4；逐级舍入会定义新学生。QuIP# 的权重方法只有在同时压权重时才迁入，不能给状态免费套码本。

无裁剪、每坐标误差 ≤Δ/2 时，正交逆满足 ‖Bᵀe‖₂=‖e‖₂、‖Bᵀe‖∞≤√10·Δ/2；它不保证 Q 的峰值变小或门不翻转，更不覆盖 U/V、重复写回舍入与 clipping。与 QuaRot 的全精度可消去旋转不同，这里限制两个读出为单行 gain/置换，本身需要模型恢复。

**最强普通控制与最小判别：**fast raw 同样量化其保留的 I，享有相同 source B、静态阈值编译、混合精度、训练数据/GT 更新预算；另以固定 B 对照可学习 B，区分普通 QAT 和基学习。先固定同一位宽及校准规则，在真实写回处比较完整 AEE、两个门分歧、连续恢复误差、饱和与访问字节；动态量化若纳入，双方同享并收费。P2×H96×T10=1920 个状态数：24/16-bit 的裸载荷分别 5760/3840 B，raw 也有相同降幅，另须加 scale、R32 状态、保护位及 bank 字宽/packing。

**独立判断：**可检验增量仅限“同一低位时间状态跨残差服务两个 θg 门和连续消费者”，使同 AEE 下实际驻留/访问少于上述普通控制；不能仅以旋转、膜量化或参数共享申报。最强反对是它最终只等于既有旋转 QAT，而 raw 已能用相同位宽、Q 更新反多舍入/访问。概念潜力 **5/10**，当前增量证据 **2/10**，均非接收概率；尚无 fast B 低位网络或物理访问结果，不能据此停止整个家族。正文方法已读，未运行作者训练配方或复现其硬件。


**2026-09-10 定向补读：SmartExchange／DeepShift 与当前 lifting40。** 本节针对后来训练的非正交 lifting40，不能套用上文旧符号基的 cond=1 或精确除4合同。当前实际路径是同40个 Q12 系数正逆共用、半步 RNE24、累加48；普通源门与 shared 连续恢复的舍入需求不同。普通稠密 As 已有官方 CMVM（260加减、8606算术位、深7），新 lifting raw/shared 已有159/169加减的常量编译；这些应作为底座，不能只比100个一般乘法。新 raw 的十帧 AAC/H8逻辑向量较旧学生约少15%，支持继续研究时间结构，但不是节点比对应的速度，也未隔离训练与结构的因果贡献。

**SmartExchange：完整算法／硬件链可借，当前40系数不是其典型访存瓶颈。** Yang Zhao、Xiaohan Chen 等，*SmartExchange: Trading Higher-cost Memory Storage/Access for Lower-cost Computation*，**ISCA 2020，954–967，DOI 10.1109/ISCA45697.2020.00082**。已取得出版版全文，读了算法、硬件、评估范围；[作者主页](https://eiclab.scs.gatech.edu/pages/publication.html)及定向仓库检索未找到该作公开训练／RTL工件，不能把另作 SmartDeal 或文中引用的 VGG 基线代码称为 SmartExchange 工件。[出版版全文](https://par.nsf.gov/servlets/purl/10188410)

必须迁入的组合是：按输出卷积核／FC行重排，令 W≈Ce·Bs；Ce 列归一化、PoT投影、交替两次最小二乘拟合、通道／向量稀疏化，最后重新量化Ce和拟合Bs；再交替网络恢复与结构投影。硬件在PE附近以常驻小Bs和Ce移位加重建一般权重，双RE与输入FIFO采用双缓冲；配套向量索引求交、行内权重广播、局部psum、Booth位串行MAC、分开的输入／权重／索引／输出存储及编译调度。原评估使用8位激活、8位Bs、4位Ce，并实际计重建与存储；**重建后仍有MAC，不是全网无乘法**。这是完整可借基线，本文未复现其训练或硬件。[§III–V](https://par.nsf.gov/servlets/purl/10188410)

对本接口的判断：40个Q12系数仅80B，可整组驻留；再编码它们并在线重建通常不直接减少作用于连续状态的159/169个加减。若改为在激活上执行 Ce(Bs·x)，便新增中间状态及舍入位置，不再是原半步函数。对整个时间矩阵做低秩／行剪枝还可能破坏可逆性，不能独立分解正向和逆向后继续声称“同系数reverse”。可保留的适配是只约束单位三角shear中的a/b、保持对角1，并共同训练实际反向链；a或b置零仍保持实数可逆。稀疏因子、重建驻留本身已有先验，须与80B常驻、相同CSE/剪枝/广播的普通raw比较。此适配尚未试。

**DeepShift：更直接针对系数算术，但作者核不符合当前RNE合同。** Mostafa Elhoushi、Zihao Chen 等，*DeepShift: Towards Multiplication-Less Neural Networks*，**CVPR 2021 Workshops（MAI），2359–2368，非主会**。已读完整主文方法／实现／结果及作者 `modules.py`、`modules_q.py`、`ste.py`、`utils.py`、压缩与CUDA kernel路径，未运行。DeepShift-Q保留实值W并用PoT投影/STE训练；PS直接学习移位p与三值符号s，采用相应梯度和重构权值上的正则。完整迁移还包括预训练初始化、激活／bias量化、指数限幅、压缩码和真实移位核；普通对照须同优化器与恢复预算。作者GPU核用于推理，其比较对象是未优化卷积核，定制FPGA属未来工作；不能引用为本地28nm PPA。[出版方主文及身份](https://openaccess.thecvf.com/content/CVPR2021W/MAI/html/Elhoushi_DeepShift_Towards_Multiplication-Less_Neural_Networks_CVPRW_2021_paper.html)，[作者代码](https://github.com/mostafaelhoushi/DeepShift)

有三处必须适配而不能照搬。第一，默认Q/PS指数范围分别[-15,0]/[-14,0]，当前lifting系数幅值约4.65，需预先规定允许正移位的共同码本及零码，不能偷偷截到1。第二，`utils.round_to_fixed`用floor；`LinearShift`的非kernel路径还留有输入/bias定点化TODO，CUDA核以int32逐项右移累加。这些都不等于当前48位完整和→RNE24，作者现存训练／kernel路径不能直接当我们的精确参考。第三，同一量化a/b必须同时进入forward与反序reverse及GT梯度，不能给连续消费者另训逆矩阵。[量化与编码](https://github.com/mostafaelhoushi/DeepShift/blob/master/pytorch/deepshift/utils.py)，[PS实现](https://github.com/mostafaelhoushi/DeepShift/blob/master/pytorch/deepshift/modules.py)，[Q实现](https://github.com/mostafaelhoushi/DeepShift/blob/master/pytorch/deepshift/modules_q.py)，[CUDA累加](https://github.com/mostafaelhoushi/DeepShift/blob/master/pytorch/deepshift/kernels/cuda/shift.cu)

例如将a换成σ·2^p后，半步仍是 RNE24(old+σ·2^p·other)，不能改成old+RNE(σ·2^p·other)：old=other=1、a=1/2时两者分别为2和1。零shear可旁路；一般负指数仍需舍入信息，正指数还需处理溢出／饱和。门专用末端能否把RNE编入阈值，应给普通raw同样优化；shared中仍要更新或连续恢复的状态不能仅凭当前门已确定就删低位。θ幅度、两个读出的gain/bias/center和τ各自保留；替换系数及舍入是新学生，旧Q12的AEE不能继承。

**剩余可检验X及门槛。** 两篇没有提供“共同可逆状态同时服务两个门读出与降R后连续恢复”的数值/资源合同；但把DeepShift套在shear上也不足以构成新意。最直接的未试控制是：双方同样将a/b训练成含0的PoT值，保持当前半步合同和同一反向系数；以真实编译后的加减、RNE、状态位宽、源请求及AEE比较现有Q12+CSE。只有联合两个消费者需求后，能比这个普通PoT/QAT控制少执行或少保存，并偿还shared的BZ/逆/常量费用，才有剩余机制增量。若两轴都受益而共享仍不偿还，结论应落在结构化PSN底座，不必维持“共享Q”为标题。当前只完成补读，没有运行新训练、编译或硬件实验。
