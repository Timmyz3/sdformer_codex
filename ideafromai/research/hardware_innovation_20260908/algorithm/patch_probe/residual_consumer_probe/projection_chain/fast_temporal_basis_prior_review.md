# FABNet 与原生 T10 公共时间坐标：先验复核

2026-09-10。已读作者主文全部 17 页（含附录），并定向读作者工件的算法、蝶形单元、索引、S2P、双缓冲和服务模型；没有运行工件或修改实验代码。

**身份。** FABNet 是论文中的网络名；论文题名为 *Adaptable Butterfly Accelerator for Attention-based NNs via Hardware and Algorithm Co-design*，发表于 **MICRO-55，2022**，DOI `10.1109/MICRO56248.2022.00050`。作者 PDF、官方仓库引文一致；不是仅有 arXiv 的未发表工作。[作者全文](https://arxiv.org/pdf/2209.09570)，[SamsungLabs 官方工件](https://github.com/SamsungLabs/Butterfly_Acc)，[作者归档](https://zenodo.org/records/7010800)。

| FABNet 应完整继承的范围 | 对当前方案的约束和适配 |
|---|---|
| §III 的 ABfly/FBfly：前者保留 attention、用蝶形线性层；后者用二维 FFT 替换 attention，再联合选择层数、宽度和两类块 | “训练快变换＋加速 attention/FFN”已有。当前仍保留事件光流的 spike QK；没有迁入其完整网络，不能把算法全搬来作为已完成事实。 |
| §IV 的统一 FFT/一般蝶形计算引擎、BP/AP/PostP 分工 | 同一算术单元支持正反变换、残差加和读出，是应给双方的普通底座。原 BU 含乘法器；固定符号网络可专门化，不能继承其 FPGA 资源数字。 |
| §IV-C 的存储置换、配对 coalesce、写回 recover | “解决蝶形 bank 冲突”已有完整结构。原生 T10 四个固定匹配不直接满足其二次幂索引证明，需要重新给合法地址/端口排程；普通十字寄存器也应比较。 |
| §V 的实数/复数缓冲复用、双缓冲及 Q/K/V 分阶段流水 | 不能只搬一个 BU，再把补齐供数、跨阶段流水称为新意。共享状态与普通轴都应获得合法重叠，并计冷填充、临时结果和 shortcut。 |
| §VI 的完整算法/资源联合评价 | 原验证主要是 FP16 FPGA；其对照中的 FFT 有 dense-DFT 实现，不能把论文加速比当成本地快变换或 28 nm CMOS 的收益。 |

表中方法定位见[主文 §§III–VI](https://arxiv.org/pdf/2209.09570)。工件进一步明确了迁移边界：[`butterfly_unit_opt.v`](https://github.com/SamsungLabs/Butterfly_Acc/blob/main/hardware/npu_design/verilog/functionality/design/butterfly_unit_opt.v#L77) 实例化四个 FP16 乘法通道；[`bfly_accelerator.py`](https://github.com/SamsungLabs/Butterfly_Acc/blob/main/hardware/npu_design/simulator/bfly_accelerator.py#L18) 将维度向二次幂取整并显式计输入/系数/输出阶段；[`butterfly_indx_generator.v`](https://github.com/SamsungLabs/Butterfly_Acc/blob/main/hardware/npu_design/verilog/functionality/design/butterfly_indx_generator.v#L130) 的阶段配置没有 T10。S2P 第 67 行还留有背压待完善注释，因此“作者提供 RTL”不等于本链任意背压协议已满足。[S2P 源码](https://github.com/SamsungLabs/Butterfly_Acc/blob/main/hardware/npu_design/verilog/functionality/design/butterfly_s2p_opt.v#L67)。

**当前确实不同的接口。** [fast_temporal_basis.py](fast_temporal_basis.py:1) 是四层、每层五对的原生 T10 符号加减网络，令乘积为 (F)，则 (F^T F=16I)、(B=F/4)。它没有补到 T16，也不是十阶 Hadamard。source/proj 的排列、行增益、bias、center、θ 均在公共坐标外独立读出；source 增益为零只产生常量门，不会令 B 奇异。当前模块仅实现算术结构和初始化，尚未实现整链状态所有权/释放控制，也未免除读出增益乘法。

FABNet 的一般蝶形层可各不相同，也没有保证可逆；FFT 算法路径还取实部。其已有“共同运算格式”不能直接保证残差链中一个连续状态足以服务两个消费者。[作者 FFT 实码](https://github.com/SamsungLabs/Butterfly_Acc/blob/main/software/accuracy/code/fft_attention.py#L20)。这里值得保留的假说是训练并维持同一可逆 B：设 (I\in\mathbb R^{10\times96})、残差 (r=ZV_r^T+\mathbf1c^T)，则

\[
Q=BI,\qquad Q^+=Q+(BZ)V_r^T+(B\mathbf1)c^T,\qquad
(I+r)U^T=B^T(Q^+U^T).
\]

source 从 Q、proj 从 Q⁺ 分别产生各自的 θg；连续支路先将通道降至 R32，再逆变换恢复全部 T10。已有非 anchor 整支路删除若属于该学生，则其位置 r=0、无需逆变换；这项删除收益不能再次计为公共坐标贡献。共享与普通的 identity/Q 都是每位置 960 个数，当前没有裸槽数优势；机会是减少重复时间变换及两消费者之间的额外状态/重算。

**不能省掉的强控制与门槛。** 普通 raw-diagonal 必须获得同一个快 source B、相同训练预算、合法读出阈值归一化和最优 R16/R32 重排；另给 Q＋仅 anchor UI32，允许以额外状态免逆变换。T10 很小，普通十字寄存驻留不能被迫沿 FABNet 大尺寸流程写片外。每对两个输入/两个结果的读取、写回、跨层配对、有限输出背压、两个 3×3 halo 和 Q 的 last-use 均要计费；二读一写配置不能免费做到一拍完成双结果。精确整数路径须保留 F 的分子及 guard/fraction 位：存 (N=FX) 后 (F^TN=16X)；中间舍入或逐层截断除二会改变函数，须另做整网评价。θ 幅度和判决阈值不能合并为“1”。

补充先验只支持边界：[Dao 等 ICML2019](https://proceedings.mlr.press/v97/dao19a/dao19a.pdf) 已做可学习蝶形分解；[SWformer，ECCV2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09680.pdf) 已将空间小波正逆变换用于 SNN，不能声称“可逆快变换用于脉冲网络”首次出现。它们未在上述连续/门双消费者接口上证明单 Q 生命周期优势，这也不意味着这种组合自动新颖。

**判断：保留结构训练候选，不升为已证硬件创新。** 主观新颖性：单独“原生 T10 蝶形 PSN”约 **2/10**；“共同可逆坐标贯穿残差，独立 θg 读出，连续端降 R 后恢复，并完成有限状态调度”约 **5.5/10**，均不是接收概率。最强反对是：同样快的 source B 配 raw-diagonal 已省掉主要时间计算，候选只剩共享带来的门活动差，且 guard 位/状态寿命抵消供数收益。下一次有判别力的结果应是这两轴相同预算训练后的真实源请求、完整连续恢复精度和单 Q 生命周期；旧共享学生的稀疏度和 AEE 不能沿用。FABNet 应作为完整引擎/存储/流水底座，而不是仅作为“也用了蝶形”的相关工作引用。

**下一融合洞与强反对（2026-09-10，CPU 操作界）。** 对同一源时间词 (g_k\in\{0,1\}^{10})，实数恒等式为 (BZ=\sum_k(Fg_k)(\theta U_{:,k})^T/4)。当前 fixed 学生在 Z 完成后已有 RNE，此交换若删掉原 RNE 就是新数值函数；以下只评供数/算术机会，不继承其 AEE。应组合继承 Gustav 的私人 W 零过滤、NRV/L1D/局部归约和完成屏障，LoAS 的同 W 跨完整 T 复用，以及 Prosperity 的相同/子集支撑部分和复用；不把这些已知能力改名为 X。[GustavSNN](https://doi.org/10.1109/HPCA68181.2026.11408587)，[LoAS](https://arxiv.org/html/2407.14073v3)，[Prosperity](https://arxiv.org/html/2503.03379v2)。

对本文件 all+ 初始 F 枚举 1024 个词，单脉冲变成 7–10 个非零系数，Fg 范围为 [−6,10]。同 k、两 anchor 的非空 NRV 若含 n 个事件，其变换非零数至少 (9-2n)。故用现有逐 k 的事件数 E、NRV 数 R 和真实 U 非零数 (m_k)，可复算下界 (\sum_k m_k(9R_k-2E_k))：

| fixed diverse10，同十帧源/实际 U16 | 原始系数更新项 | 先展开 Fg 的更新项下界 |
|---|---:|---:|
| shared | 795,973,887 | ≥2,265,121,944（2.8457×） |
| raw diagonal | 772,569,456 | ≥2,234,588,208（2.8924×） |

这些含首次赋值，非真实加法数/周期；变换系数的移位加、ROM/译码还未计。计数来自 [fixed 源表](fixed_temporal_source_cost.json) 所指逐 k 数组及 `U_conv2_theta_q16`，不假定每行 16 个非零。普通后置 F 每 anchor/R16 向量仅 40 次加减，即全帧上限 12.288M 次；四帧旧 FP 的 shared Z（512 个 anchor、8,192 个 T10 向量）允许普通零旁路后，从 327,680 降为 232,800 次加减，另有 50,080 次复制/取负输出。这是旧 [Z 捕获](shared_temporal_recovery128x256_single_q_consumers_train4/shared_assigned_affine_A_capture/capture.json) 上的局部结构诊断，与 fixed 十帧分开；不能外推新快基学生。

仍可测的窄接口是：在有限 NR 窗口先按完整词或子词归约 W，再选择留原域部分和或直接送 F 域；但是 B 可逆，**精确同码关系完全不变**。若先造 (S_c=\sum_{k:g_k=c}\theta U_{:,k})，raw 也能用同一 S 按 c 散射，候选按 Fc 散射，不能将合并收益独占。完整查表须 1024×10×5bit=6,400B（仅此 all+ F），或付窄加减译码；一般符号配置另核范围。混合域若同时有未完成 Z 和 FZ，需要两份 R16×T10 部分和，或付原地换域/串行屏障；逆变换 R32 的费用仍在。同 T/P 共取权已给普通轴，因此减少事件不自动减少 W 请求。

本轮先停朴素逐词展开；有限子词归约仍是待证组合，当前没有足够理由提高 5.5 分。真正判别量是**同一有限窗口、同普通归约下剩余的额外更新/写回节省，能否超过后置 F 的 40 次加减及混合域状态税**。现有 fixed JSON 没有完整词直方图/NR 窗口顺序，不能用发放率猜这部分命中，更不能据此宣布净性能。

**本轮唯一保留的窄假说：跨舍入边界编译纯门 source（2026-09-10，尚未实验）。** 当前没有足够证据提出一个成立的新 X。最新 fixed raw/shared 的 825 帧 AEE 为 1.232979/1.247809，且分别有 13/21 次合同内饱和；共享既没有精度优势，也尚未偿还 BZ 与逆变换。下面只提出一个可先被 CPU 下界否决的假说，主观机制潜力 **4/10**、当前证据 **1/10**，不是接收概率。

**B 与完整普通控制。** raw source 已从 dense 的 260 加减降为 159 加减，但仍有 35 个不能直接用末端阈值消掉的中间 RNE；这些非线性边界阻止普通线性 CSE 跨步合并。raw 的变换结果只供 sn1 门，连续消费者保留原 I，因此这里不受旧 `F_pre+UF_tail` 混合消费者约束。强底座必须包括 FABNet 合法配对/供数、da4ml 实际字宽 CSE、Gustav 完整下游执行，以及 DeepShift/PoT QAT、普通固定低精度和末 5 个 RNE 阈值折叠。不能因为当前合同是精确 fixed，就排除同 AEE 更便宜的普通学生。

**候选 X 只落在一个编译边界：** 将一段“必须逐步产生舍入值”的图改为“先产生最终门的严格区间；只有未决门需要原舍入轨迹”。令第 i 个 Q12 shear 为 S_i，未饱和时其舍入误差 e_i 仅落在被更新坐标、幅度不超过半个整数 LSB，则

\[
q=BX+\sum_i(S_{39}\cdots S_{i+1})e_i,\qquad B=S_{39}\cdots S_0.
\]

先固定一个 Q12 的合并矩阵近似 \(\widehat B\)，按实际系数范围保留所需整数位，编译其 CSE；不把精确 B 的长分母当免费常量。若输入满足预编译安全范围 \(|X_k|\le M_k\)，每门半径可包含两项：传播的舍入误差，以及 \(\sum_k|B_{jk}-\widehat B_{jk}|M_k\)。对已按本 h 的 gain/bias/θ 归一化的正向整数门 \(q_j\ge\tau_{jh}\)，仅在 \(\widehat q_j-\rho_j\ge\tau_{jh}\) 或 \(\widehat q_j+\rho_j<\tau_{jh}\) 时结束；负 gain 的比较方向、零 gain 常量门分别编译。半径必须向外取整。安全范围外一律走原 fixed 路径，包含真实发生过的饱和，不用十帧零饱和作证明。

这不是把“先粗后精”当新意：[Precision Gating，ICLR2020](https://www.csl.cornell.edu/~zhiruz/pdfs/pg-iclr2020.pdf) 已有低精度结果与选择性修正，[MLSys2021 的精确推理](https://proceedings.mlsys.org/paper_files/paper/2021/file/b9799a12d683d136cc817f94b73a8938-Paper.pdf) 已有严格界与失败求值。这里**尚待证明的差别**只是：跨内部 RNE 合并后，原本不能共享的表达式能否形成足够便宜的门证书图。没有证据说明这个具体编译目标已构成新的研究贡献。

**首个小实验与固定杀门。** 先只编当前 raw 参数的这一份 Q12 合并图，按合法字宽列出加减、比较、常量与临时状态；若在“所有门一次确定、零 fallback”的乐观条件下仍被原 159 加减＋35 RNE 分项支配，立即停止，不开训练。只有这一步存在余量，才用已存 source 输入做一次证书回放：失败费用按所有未决门的**原图祖先并集**计，不能按门率线性缩放；合并结果通常不能作为原 RNE 轨迹的可复用前缀，重算、输入重读、界比较及宽状态全部计入。预设继续门为：与同资源的最强普通图相比，含这些费用的算术位工作至少少 15%，且未增加状态/常量读取；这只是 CPU 筛查阈值，不是周期或 PPA。若普通固定低精度/PoT 在同精度要求下已更便宜，则停止这条自适应 X；保留纯门 source 这一挂点，不用精确原合同替它制造需求。
