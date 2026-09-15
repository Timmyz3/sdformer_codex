# Bishop、Phi、Prosperity：从问题发现到可迁移机制

2026-09-15，独立全文阅读；本轮仅读论文与源码，没有运行 RTL、训练、性能实验或 EDA。这里的 A 是论文已有的完整方法，B 是本项目尚需解决的具体瓶颈，X 是待验证的接口增量。三篇的论文 PPA 均不作为本项目收益。

**阅读覆盖。** 实际逐段读完 Bishop v1、Phi v1、Prosperity v2 的正文、架构、评估、消融与参考文献，来源为本地 `survey_ab_fusion_20260910/p0_txts/{2505.12281,2505.10909,2503.03379}.txt`；又对 Bishop 第 5 页的公式做了 PDF 图像复核。读了本地 Prosperity 作者仓库的 README、`kernels/prosparsity_cuda.cu` 的关系生成及 `simulator/simulator.py` 的 FC 成本、调度与前缀选择。没有把旧 AI 卡当全文证据。

主论文：[Bishop v1 全文](https://arxiv.org/html/2505.12281v1)，[Phi v1 全文](https://arxiv.org/html/2505.10909v1)，[Prosperity v2 全文](https://arxiv.org/html/2503.03379v2)。Bishop/Phi 的作者主页本轮分别只找到论文条目、论文/幻灯片入口，未找到可核的作者训练与 RTL 仓库，故不声称复现了其实现：[Xu 主页](https://boxunxu.top/)，[Wei 主页](https://dubcyfor3.github.io/)。Prosperity [官方仓库](https://github.com/dubcyfor3/Prosperity)明确提供 simulator；这与论文声称实现过 SystemVerilog 是两个证据层级。

**本项目的共同边界。** 已定推理源是 `{0,θ}`，静态 θ 可折入权重；不能再用“非单位脉冲不能 AAC”否决迁移。非因果 T10 PSN 的整个时间字到齐后，可以在线性段重排执行并恢复索引；不能用这三篇的 causal LIF 更新/复位替换本 PSN。当前 R8 的 Q1 是二值源乘低位宽静态权重，Q2 是连续整数 Z；只有前者直接符合 spike-row 算法。R8 的强原生基线包括 bitmap/onehot、count/time-pair、borrow RR 和完整 I24 消费者，不能重用 PTB 或朴素逐位 AAC 为唯一分母。FFN 是另一个具体算子，需用它自己的真实源、形状、BN/PSN 边界核实，不借 R8 的周期或 AEE。

## 1. Bishop：从时间批处理不足，走到时空工作粒度和异构执行

**问题是怎样发现的。** §2.2/Fig.3 先按形状拆分投影/MLP 的 `O(TND²)` 与 attention 的 `O(TN²D)`，不是仅按全网发放率选硬件。§3.1 指出原 PTB 在长时间轴可共享 W，短 T 的 transformer 会失去这一摊销；§6.5/Fig.16 又实测 bundle 太大把空闲位置一并搬运/处理，故“打包越大越好”不成立。

**旧方法为何不足。** PTB 已有同 PE 内与跨 PE 的时间权重复用，Bishop 没有发明这两件事。新增问题是有限 T 下怎样加入 token 维度，以及输入 feature 的活跃 bundle 密度差异怎样落到合适执行核。稠密核处理稀疏输入会空转，稀疏核处理稠密输入会支付过多分发/规约开销；两核分配不均也会让快核等待慢核（§5.2–5.4、§6.4–6.5）。

**完整 A，不能缩成一个 OR-mask。**

| 已有组件 | 论文证据 | 完整迁移需要保留的接口 |
|---|---|---|
| TTB 表示与复用 | §3.2，Fig.4 | 一个 feature 的 `BSn×BSt` 原始时空位置、活跃 tag、同 W 的 bundle 内/间广播；最终输出位置保持 |
| feature 级 stratifier | §5.3，Alg.1，Fig.10 | 统计该 feature 的活跃 bundle 数，产生 sparse/dense feature 索引，并同步选择对应 W；其扫描、索引与读取不是免费 |
| dense 与 sparse 两核 | §5.4，Fig.9 | dense 是 output-stationary SAC（选择 W/0 后累加）；sparse 明确借 SIGMA 分发/规约；两核有局部和/输出缓冲，再在 Spike Generator 合并 |
| BSA 训练 | §4.1，式(9)(10)，Fig.5–6 | 惩罚应与活跃 bundle 对齐，含 surrogate/梯度定义、λ、校准与训练；公开公式存在下述待核点 |
| ECP 与 attention 核 | §5.1、5.5，Fig.7、9 | 从二值 Q/K 的支持量，在形成完整 QKᵀ 之前剪行/列；S-stationary 的 AND-accumulate 与 S×binary-V 的 select-accumulate，保留输出/缩放与 LIF 边界 |

**两处不能照抄的数学边界。**

1. BSA 文字说减少 active bundle，式(9)却写 `Z_b=||X_b||₀`，式(10)为 `Σ_b Z_b`。按通常 L0 定义，这只是总 spike 数：同样 2 个 spike 放在一个 bundle 或两个 bundle，损失都是 2。若要表达 active bundle，应再有 `1[||X_b||₀>0]` 或明确的另一 surrogate。PDF 第 5 页与 HTML 都是这一写法，[已保存并目视核对的公式页](bishop_phi_prosperity_bsa_equations.png)。这是作者实现口径待核，不能自行补正后称“忠实 BSA”。
2. ECP 约束的是被删 attention score 的局部界，不是完整 Y、下一层 spike 或任务误差证书。若 Q 某行的活跃 feature 集大小为 `a`，二值 K 给出 `0≤QKᵀ≤a`；删除后乘 V、缩放与神经元阈值仍须分析。Fig.7 的 `min(uQ,uK)` 也须区分：**Q、K 两侧均被删**的单元可受最小界约束；仅一侧被删只能使用该侧的界，不能把两种阈值的最小值推广到整个被删行列并集。这是独立推导，不是声称已发现作者 RTL 错误。

**最关键的 3 条来源。**

| 来源及在 Bishop 中的位置 | 本轮实际读到 | 从来源走到 Bishop 的改变 |
|---|---|---|
| PTB [27]，§3.1/Related Work | 作者公开稿 §4.2–4.4、式(7)(8)、Fig.6–8；[PDF](https://web.ece.ucsb.edu/~lip/publications/SparseSNNAccelerationIEEE-MICRO-Submitted2021.pdf) | PTB 的时间积分与后续逐时膜更新已分开；StSAP 已按不重叠 TB-tag 贪心配对（最多两 neuron）。Bishop 加入 token 维与 feature 分流。该 URL 页眉是 2021 投稿稿，未核为最终 HPCA22 排版版，不混用其 PPA |
| SIGMA [38]，§5.4 | 原作 §III-C、§IV、§VI-B/D，Fig.4–8；[作者 PDF](https://anands09.github.io/papers/sigma_hpca2020.pdf) | Benes 分发 + FAN 多个变长点积规约 + bitmap/controller 是被借的稀疏执行底座；Bishop 换为 TTB/SAC 并与 dense 核并用 |
| SpAtten [48]，Related Work | 原作 §III-A、Alg.2、Fig.4–5；[全文](https://arxiv.org/abs/2012.09852) | SpAtten 用累计 attention probability 做 token 重要性、跨层 cascade 删除；Bishop 利用二值 Q/K 在本次 score 形成前取得支持上界。不能将 SpAtten 简化成“只删算完的一个 score”，它还省后续 QKV/FFN |

**对当前系统的判断。** Q1/FFN 的 T10×P 权重复用与活跃 tag 可直接迁移；当前 single shared datapath 可以实现两种 dataflow，但这只是异构算法的时间复用适配，不能借双核并行收益。BSA 尚未完整借入。ECP 属于 attention 家族，应先保留一个真实 attention 层入口；用它跳过任意 FFN signed-weight 或连续 Q2 是换了定理条件。基本打包/分流为成熟 A，单独的新颖性约 2/10；若 X 能用同一生产者元数据降低完整共享执行器的付费服务，而非新增一套核，才有进一步研究价值。

## 2. Phi：从统计上的模式集中，到预计算、精确修正和执行汇合

**问题发现。** §2.3/Fig.1 用 t-SNE 展示 SNN row 的集中分布；§3.2 再按 K 分段做独立校准；§5.6/Table4 同时测随机二值矩阵，随机数据也有约 2.7× 的理论操作潜力。故“有模式”不全是 SNN 专有，真正问题是给定代码本预算下，真实数据是否有额外且可泛化的集中度。t-SNE 图不能代替实际查表命中、残差和供数成本。

**完整 A 的等式。** 对 K16 的二值 row `x`，选中心 `c`，令 `r=x−c∈{−1,0,1}¹⁶`，则 `xW=cW+rW`。`cW` 是静态 PWP；正残差加 W，负残差减 W，全部保留时精确。§3.1 还有关键原生旁路：若最好 pattern 的残差非零数比原始 popcount 更差，选零中心；zero 不算，onehot 不值得另存 PWP。PAFT 是**可选**有损训练，不是精确 Phi 的先决条件。

**完整 A 的工程组成。** §3.2/Alg.1：分区内过滤 zero/onehot，按 Hamming 聚类、均值后二值化；§4.2/Fig.4：运行时 matcher → 正负残差 → compressor → 多窗口 bank-conflict-aware packer；§4.3/Fig.5–6：把旧 psum 也打成 pack 项，8 输入可配置规约树和回写 crossbar；§4.4：L1 以 pattern-ID 读取 PWP，16 bank/16-to-8 crossbar，L1/L2 并行后按 output tile 同步；只预取实际使用的 PWP。§3.3 的 PAFT 损失以 N 加权 Hamming，目的为减少残差操作。

**旧方法不足和作者如何修接口。** 单纯保存高频结果没解决 outlier；Phi 用 signed 残差避免近似。单纯减少残差会让每行只剩一两项，PE 利用率变低；它连同旧 psum 跨行打包，并在打包前避 bank conflict。PWP 多到放不下，则利用前层正在产出的 pattern-ID 选择预取。这里的层间流水以输入/输出生产次序可重叠为前提，并非任何神经元模型都能隐藏 matcher。

**对我们最关键的容量推导。** 原配置 `k=16,q=128,n=32,m=256`（Table1）；含 64KB PWP、128KB psum 等，不能缩成“128 个 16bit pattern”。对本 R8 Q1 的 54 个 K16 分区，若照搬 q128，每个 N8 PWP 用能容纳 16 个 signed3 权重和的 signed7，PWP 本体就需 `54×128×8×7/8=48,384B`，还没算中心、ID、残差/psum；原 Q1 W 只有 `864×8×3/8=2,592B`。这是容量计算，不是新硬件测量。论文 §5.5/Fig.12 的 selective prefetch 已把额外 W/PWP 流量由约 9× 降到约 3×，并没有消灭这笔税。

**需要写清的适配边界。** N8 的 Hamming 查询摊销远弱于论文 N32（更远弱于跨完整 FFN 输出复用）；连续 Z 不能直接拿来匹配 0/1 pattern。PSN 必须先把依赖的 T10 数据到齐，再产生可消费的精确 source tag，不能免费前层预取。§4.2 称优化后不存在超过 8-unit pack 的 row，是其分布/实现假设；若声称支持任意合法 T10 输入，必须有长残差/psum 项的分拆或原生回退。现有频繁字典、DA、support codebook 试验只覆盖这个家族的局部，不等于 Phi 已完整迁移或已被否定。

**最关键的 3 条来源。**

| 来源及在 Phi 中的位置 | 本轮阅读证据 | 可借内容与不能借的结论 |
|---|---|---|
| JPEG-ACT [12]，Introduction | 原作 §III 的动机、SFPR/DCT/CDU、Fig.6/8/9；[作者 PDF](https://people.ece.ubc.ca/aamodt/papers/evans.isca2020.pdf) | 从数据统计寻找可压缩表示，再设计带宽/解码接口；原文说 dense conv 的频域熵更低，但 sparse ReLU 未观察到同趋势。它是训练 activation offload，有损变换；不是 Phi 的精确 LUT/CSE 算法 |
| SIGMA [46]，§4.3 | 上述原作 §IV-A/C/E、Fig.5–6 | Phi 明确借可变长规约，补上其 signed 残差、psum 项、pack 与 bank 约束。仅画一个 adder tree 不算完整借入 |
| Prosperity [60]，§2.2/Related Work | 本轮全文 + 作者 kernel/simulator | Prosperity 重用当前 tile 的已有 row；Phi 用离线中心和正负修正摆脱子集限制，同时换来 PWP 存储、查询与汇合税 |

额外最近邻：Phi [30] 是 Jégou 等的 Product Quantization（[DOI](https://doi.org/10.1109/TPAMI.2010.57)），本轮 INRIA 正文入口遇防爬/旧 URL 404，**未实际读到全文，不把它当已精读证据**。Phi [20] Transitive Array 已读到 §2–4 的 Hasse 图、静态/动态 scoreboard、prefix buffer 与两级 PE（见下面候选 3）；它证明 bit-sliced ANN 的结果重用已是强先验。

**当前判断。** 精确 Phi 值得按一个 K16 条带完成 A；首试应冻结代码本并带 zero/onehot/长残差回退，不先训练“更像 pattern”。如果 X 只是换代码本大小或给 Hamming 换 N 权重，新颖性约 2/10；若解决真实 production→PWP 供数→共享 psum 生命周期，使额外表示费用可由已有状态承载，才有候选增量。不能把论文 3.45× 当作 R8 的收益，也不能用旧字典税一次负例杀掉整个家族。

## 3. Prosperity：先约束重用图，再让图的顺序能被硬件执行

**问题发现与主动舍弃。** §III-A/B、Fig.2 从相同/包含的二值 row 找到可重用内积。它刻意只保留 exact match 和 proper subset；两个 row 的一般交集若不对应已有 row，会新增节点和执行状态，因此不做。§III-D/TableII 又比较一前缀/两前缀：第二前缀只帮助少量节点，其尝试的架构性能下降 30%。这解释它为什么选森林；没有证明所有双前缀/公共子表达式布局都不可行。

**完整 A。** `S_parent⊆S_child` 时，`y_child=y_parent+Σ_{k∈S_child\S_parent} W_k`。对相同行，父 index 必须更早，避免环；对子集行，popcount 更小天然先算。§V/Fig.5 的 TCAM 以 query 的 1 位设 don't-care 查所有子集；pruner 按 popcount 取最大父，过滤不合法 EM；稳定 popcount 排序生成顺序；dispatcher 保存 prefix/差分；processor 实读父结果作初值，再 pop 残余位、读 W/加、写结果，最后跨 K tile 规约。

**“近乎免费”成立的条件。** §VI/Fig.6 是 double-buffered TCAM/元信息、空间流水与前 tile 计算重叠，不是省掉检测。论文 m256、k16、n128 有独立 detector/pruner/sorter/dispatcher。§VII-G 明确承认 TCAM 总 bit 工作仍为 `m²k`；线性指完成时间，依赖并行比较面积。其 `ΔS>4.4%` 是特定 n128 和 TCAM/浮点加 45:1 代价的推导，不能带入本项目 signed3/8 lane AAC。§VII-F 的 5× 密度改善最终只成约 3.2× 同 Prosperity bit-only 加速，也已经展示 EM 一拍等退休税。

**源码实读的边界。**

| 入口 | 核到的行为 | 对“完整借入”的限制 |
|---|---|---|
| `third_party/Prosperity/kernels/prosparsity_cuda.cu`，`prosparsity_kernel` | 当前 row 至少 2 个 1 才选父；候选必须是子集；EM 父 index 更小；残余为原 row 减父 | 这是在 GPU 上生成原始 source 的 prefix/残余；不是 TCAM RTL，也未自己执行 W、数值累加和 output backpressure |
| `simulator/simulator.py`，`find_product_sparsity` / `get_prosparsity_cycles` | Python 同样排 EM 顺序；预处理成本计 non-onehot rows，FC 中另计 popcount；EM 仍计一次处理 | 上层按 sparse map 和数目算周期，不能自动证明我们原始输入→prefix→最终 I24 的数值/端口合法性 |
| 同文件 `run_fc` / CUDA 路 | 有 weights/source residency 分支，psum 读写记录，preprocess/compute/memory 取 max 模型 | 必须把实际争用/拒绝保持另落到本共享执行器；不能直接把 max 视为所有 pipeline 已实现 |

一个可复现口径差异：正文 §III-D 在最大子集并列时选最大 index；本地 Python `argmax` 与 CUDA 的 `>` 更新选择首先遇到的候选。二者都保持正确 partial order，但森林与局部代价可不同；报告应固定版本与 tie-break，而非强行称逐图复现。此处没有运行 CUDA，也不将本地 vendor 副本等同今日上游每一行。

**最关键的 3 条来源。**

| 来源及 Prosperity 引用 | 本轮实际读到 | 学到的接口变化 |
|---|---|---|
| CAM survey [68]，§V-B（TCAM 另引 [59]） | [作者原 PDF](https://www.pagiamtzis.com/pubs/pagiamtzis-jssc2006.pdf) §I-A/B、Fig.1–4；全文可取得，但本轮仅精读这些方法段与功耗讨论 | CAM 的并行 searchline/matchline、多个命中后的 encoder 是成熟硬件。Prosperity 需取得候选子集并用最大 popcount/EM 约束选父，不能只用普通 longest-prefix priority encoder 替代 |
| HAG [44]，§VIII-C | [KDD20 作者全文](https://www-cs-faculty.stanford.edu/people/jure/pubs/hags-kdd20.pdf) §3–4、Alg.2/3、Theorem1、cost function | HAG 显式建立共享 aggregation 节点，按 capacity 贪心减少重复；真实集合求和与有序 prefix 分开。Prosperity 为动态密集程度更高的 source 放弃昂贵全图搜索；“静态 GNN 不可直接原样用”不等于 CSE 思路禁用 |
| PTB [52]，§II-C/§VII-D | 上述作者公开稿 §4.2–4.4 | PTB 已有时间复用；Prosperity 的强消融还需对比同一 processor 上的 unstructured bit-only。两者差值才更能隔离 product reuse |

Batcher sorting network 是其 [4]（[原作 DOI](https://doi.org/10.1145/1468075.1468121)）；本轮读到了 Prosperity 如何使用它，没有取得并精读 1968 原作全文，故不把“稳定排序已读”写成“Batcher 原作已读”。

**当前判断。** T10 非因果完整门字提供合法重排窗口，θ 折权也不妨碍子集等式；但 R8 减小输出宽度后，检测/父读/退休可能吃掉加法收益。已有旧 Prosperity/APEC、parent、字典变体实验应保留为特定布局证据；本次不重新宣判家族。单纯森林/TCAM/popcount 排序为借入 A，新颖性 1–2/10；下面的容量与生命周期约束是候选 X，而非已经得到创新结论。

## 4. 优先限定接口：固定 Prosperity 父森林，只替换 root / parent-delta 的求值

**结论：代数和部分缓存探针已有；下述完整逐 K 求值协议未在这次核到的旧实现中运行。** 因而不重开“APEC 先删共同输入再重新建森林”，也不重开“Phi 先分解原行再另建动态残差图”。这次只保留原 source、父编号、tie-break、拓扑序、parent 存储生命周期和跨 K 目的提交，替换每个既定节点需要求出的那一项。不是换一棵更好看的森林。

**先对齐三份旧负例与一个已做正探针。** 本节实际读了下表中的报告及实现函数，未以目录名推断覆盖。

| 旧证据及准确位置 | 实际接口与数字 | 对本次限定接口的约束 |
|---|---|---|
| [complete_transfer / 完整合同](../../../complete_transfer_20260907/complete_path_contract.md)，§“Phi 根构造替换”；[报告](../../../complete_transfer_20260907/report-source.md) | 设想只替换 root，但若 Phi 等跨 K 完整行才返回，`K0:B←A / K1:A←B` 会产生假环；文中明确逐 K 返回仍待重构，没有该接口的 RTL 或周期负例 | 必须先发布当前 K 的 parent value，不能等完整行/I24。此处否决的是跨 K 完成边界，不是 signed residual 等式 |
| [same_workload 报告](../../../same_workload_c1c2_20260907/report-source.md)，[c1_cache_model.cpp](../../../same_workload_c1c2_20260907/c1_cache_model.cpp)：`build_data/common_phase/residual_tile` | 先对相邻 G2/G4 的 C768 输入求共同交集并删除，再 im2col、重建 residual 森林。G4 的 9KiB 系数缓存：48,944,824→56,992,980，**CPU 服务周期 +16.44%**；容纳全 K 仍 +2.964% | common 跨 C/tap 的 W 工作集与 K16 residual 共用同一 LRU/总线/源 ALU；原森林还被改变。本接口移除这个 common_phase，不代表已消除所有缓存/带宽税；旧 N96/M3000 数字不能代替 R8 N8 结果 |
| [prosperity_fusion 报告](../../../prosperity_fusion_20260906/README.md)，[screen_basis_residual.py](../../../prosperity_fusion_20260906/screen_basis_residual.py)：`assign_rows/simulate/WideCache`；[同批强对照](../../../prosperity_fusion_20260906/same_cohort_strong_baseline.json) | 先对**原行**做 Phi 分解，再建静态基图和动态 signed-residual 图，共用 16 槽 LRU；q8 是 4,317→5,788 次向量加法（**+34.07%**），W 读 5,684→8,145（+43.30%），**不是周期结果**。PWP 对照的请求有计数，选择性预取仅为 unique-touched 下界，非真实协议 | 该实现不保留原 Prosperity 森林；基/残差共抢宽槽的问题有证据，不能再次把“两类对象共用 LRU”称未试。固定父图后不持久缓存新修正结果，避免这一种双图挤占，但仍须付 PWP 缓存与两路汇合 |
| 同目录 [screen_joint_residual.py](../../../prosperity_fusion_20260906/screen_joint_residual.py)：`decide/choose/numeric_and_account` | **已做固定原父 + 最多 2 项正残差子集字典**：6,557→6,154 次加法，约 −6.15%；候选字典在当前 tile 上离线贪心选，构造加法和读写有计数，匹配/完整端口/周期未实现 | “固定森林＋残差查表”本身不是遗漏。尚未完成的是冻结代码本的正负修正、实际请求预取、两路就绪与 K-local 父发布；不能把该探针的 −6.15% 报作新实验预期 |

**B 与最强反对先写。** 既定 forest 的每个 root / delta 仍可能有多个非零源，需重复从当前 K16 求和。Phi 可以把这项改成 PWP 加少量修正；但 Prosperity 已把很多 delta 变成零或 onehot，R8 又只有 N8，matcher、PWP 读取和 merge 很容易比省掉的 AAC 更贵。原生 K16 W 若已驻留，少一次逻辑 W 读并不等于少一次外部传输；现有 P2/P4 打包还可能让少加法完全不省 issue。这是本试验最可能推翻 X 的解释，不是先假设旧负例已被解决。

**保持森林不变的精确接口。** 对既定节点 i、分区 k，令 `u_i=S_i`（root）或 `u_i=S_i\S_parent`（有父），两者均为 16bit 非负支持。对 u 选冻结的 Phi 中心 c：

`V_i,k = V_parent,k + PWP(c,k) + Σ_(u=1,c=0) W_k − Σ_(u=0,c=1) W_k`，root 的 parent 项为零。

Phi 的两条计算流是 PWP 与 signed correction；parent 可以作为 Phi L2 的旧 psum 项参与，而非另造一棵 residual 森林。只有 PWP/correction 都被接收并完成合并，才写当前 K 的 parent 槽并发布 `parent_ready`；随后仍按原合同累计到原目的跨 K psum。EM 的 u=0 直接继承，onehot u 直接 W，不先支付 matcher 或空域 merge；非平凡 u 若选择 c=0，则走完整原生残差。中间无 RNE，正负修正和所有 prefix 均需位宽证明。非因果 T10 只提供已到齐的调度窗口，不许可跨 K 等待环。

**有界 RTL 叶合同（待实施，不冒称完整 Phi 架构已经迁入）。** 首试一个 R8 Q1 的 K16、T10×P4=40 行、N8；同一固定森林分别送所有臂，检测/排序作为共同前缀单列。叶内求值、PWP/中心读入、父状态、输出和 BP 全收费；之后若进整核，再接原生 source 形成 forest 和所有 54 个 K16，不借叶的周期为完整层收益。

- 每个 K16 先固定 **q≤8** 中心，来自独立校准的 root/delta 分布，不用 held 输入选最优中心；使用 Phi 的 Hamming 聚类及 zero/onehot 规则，不训练。R8 signed3 W 的 PWP 范围为 [−64,48]，signed7 足够；packed signed3 W 为 48B，8 个 PWP 向量按 64bit 物理槽占 64B，中心 16B，共 **128B 数据/中心预算**。全 54 分区中心和 padded PWP backing 分别为 864B/3,456B，须与原 W 一同列静态容量和实际传输。PWP 与 W 不能免费共占同一物理地址。
- 临时总预算先封顶 **384B，所有强臂均可使用**；除上述 128B，还显式容纳两个 N8 signed32 向量 holding（64B）、至多两个已查询描述符/完成位和实际 tag。固定森林所需 parent/顺序/source 元数据为各臂共同且另外列账，不能用未声明 parent SRAM 挤出 PWP。这个容量是拟议叶合同，不是声称已映射进现有 SRAM 宏。
- `eval_req={context,epoch,K,row,parent_id,u_mask}` / `eval_rsp={same_tag,value}`，请求/响应 ready-valid；最多一个当前求值加一个有标签 lookahead。Hamming 查询一次比较一个中心的最小实现必须实收拍数；若改并行 matcher，独立报告比较器而不当免费。lookahead 仅对已接受、已查询描述符的实际 pattern 发 PWP 请求；原生 W/PWP 共用同一 256bit 数据请求口、一份在途信用/返回暂存。未被接受的预取、权重返回、输出受阻均保持标签和值，K/模型版本改变必须先排空再失效。
- **只用原 8×32 ALU** 仲裁 L2 与 L1/L2 join，复制、符号扩展、原生 first-term 初始化按实际实现计费。两流可重叠等数/查找，不能暗添两套算术。论文 8 输入规约树×32 SIMD 不能原样称同资源；本叶应允许同 8 ALU 下多行打包/普通 issue 合并，并完整记录哪些 Phi packing 能力尚未迁入。没有支持长 residual 的分拆/原生回退就不能叫合法全输入实现。

**必须保留的强对照。** F0：同 forest 原生求 root/delta，允许 onehot、零 residual、合法 parent forwarding、K16 W 驻留及现有 P2/P4 issue；F1：同预算的 **Phi-alone**，直接对原行 S 做 PWP+signed correction，保留零中心/zero/onehot、requested prefetch、相同 packed 服务/回退，不能削弱成每项都独占整条流水；F1 允许用同一校准样本和 q8 预算为原行 S 重新校准，不能强迫它用只为 u 选出的弱代码本。F2：同 forest + Phi 求 u（候选）；F3：F2 不 lookahead 的 demand-only 消融；F4：同 forest + 正子集小字典（已做代数的普通成熟控制）。F1/F2 各提供论文 Hamming 旁路和**有费的服务数旁路**：已知一次 PWP miss/merge 比 direct 剩余服务还贵就不用，查询税仍收；不可用先跑两路取最短的 oracle。评 complete layer 时再对无森林的 native bitmap/count 强臂，检测费用不隐去。

**判别量与 X 的边界。** 单位是实际 N8 向量 issue（packed 时记物理 issue）与最终退休周期，不是每个标量加法。必须逐节点输出 u 的 0/1/多项分布、query 拍、实际 PWP 请求/命中/未消费预取、W beats、parent 读写、join/ALU grant、ready 等待和冷/warm/BP 全端点；先全零/onehot/EM/长正负修正/K0-K1 反向父关系，再小真实与两套 held。该协议**结构上避开 APEC 跨 C 共同构造与“基/动态残差同 LRU”两种已见干扰**，但是否净赢仍未知。精确分解、预取、两路 join 都是 Phi 的 A；固定森林的 K-local 交付是一项工程适配，当前独立创新暂评 **3/10**。只有同资源完整服务证明确实解决原失败原因，且普通 F1/F4 做不到，才有理由提高；不据此宣称新代数或首创。

## 5. 三个非 SNN 方法迁入的可测接口，以及一条训练保留项

下列都是**待测建议**，不是本轮实验结果；“未试”仅指当前 shared-execution 合同下没有本轮收据，不声称已穷尽所有历史文件。次序按能否先用有限硬件费用判真伪安排。每项都先写 B 与强对照，再谈 X。

**非 SNN 候选 1：SIGMA 的生产者支持元数据 → 有费的 sparse/dense/原生路选择。补完整 A。**

- **B 与强对照：** 稀疏执行读取压缩 W 之前需要源 bank 是否为空、哪些源支持真正被消费；eager 元数据在空源上白付，lazy 元数据又有首行/压缩 rank 补扫。强对照是同源的原生 dense/bitmap、现 Gustav eager/lazy、同一阈值与同口配置，不是“所有元素都读”的弱 dense。
- **来源 A：** SIGMA §IV-C/E、Fig.5 已有 streaming bitmap 的 row-OR，与 stationary bitmap AND 生成只需加载的 stationary'；source/destination counters/table 后才 multicast/规约。其 §VI-B 还明确 no-local-reuse 虽算术满载却会输在带宽，不能把 PE busy 比当速度。
- **具体 X：** 从现有 producer 的已收费 word 写入/完成协议形成每 K16/bank 的支持摘要，带 context、generation、complete 位；consumer 只有在摘要覆盖完成后才选择 dense/bitmap/压缩行入口。使同一摘要同时服务格式选择和权重读取，避免另一遍 source 扫描；不是免费非空 flag，也不新增 SIGMA 全核。
- **最小实测：** 一个真实 FFN 或 Q1 bank，先保持函数与全部原始 T10 位不变；空、单 onehot、dense、跨命令覆盖及 BP 各一组，再两套 held tile。记录摘要更新/扫描/metadata/read/实际 W 请求、共享口冲突和最终 I24；输出 source reads/bytes 与冷服务周期。若只是把 tag 前移、全链无净收益，停止该布局，保留稀疏格式家族。
- **暂评：** 成熟 A 的迁移 2/10；只有证明摘要的生命周期同时解掉两处真实重复服务且不扩口，X 才值得提高。与 Bishop 的 feature stratification 区别在“何时知道支持、谁支付”，并非改个阈值。

**非 SNN 候选 2：HAG 的有限公共节点 → 当前 T10 K16 的有费合成父。保留接口。**

- **B 与强对照：** Prosperity 只能用现存 row 作父，`a=1100,b=1010` 的交集 `1000` 若没独立出现就不用；公共集合可能在很多 row 中存在但不呈严格包含。强对照必须同时有同资源 native bitmap/onehot、原生权重复用、Prosperity 一前缀，以及最小 DA/频繁 pattern 字典；相交位少于元数据/读写成本时原生应获胜。
- **来源 A：** HAG Alg.3 已有按重用次数建立新二元 aggregation 节点和 capacity 约束；Prosperity §III-B 明确将这个分支排除。任意创建新公共表达式绝非首创。
- **具体 X：** 固定每个已到齐的 T10×P、K16 条带最多 **2 个 synthetic 节点**；只对有 ≥2 个实际消费者且能覆盖付费 build+store+read 的节点入队。先算节点，再算残余并退休原始位置；节点可尝试占本 context Q1 期尚未归属 Q2 输出的少量 p_mem 字，但必须通过同 psum grant，并在阶段切换前释放。算法按“实际服务少几拍”而非 popcount 最大选节点，另一 context 的 I24/psum 争用也计入。
- **最小实测：** 先在当前真实 source 上导出一个固定容量的候选表和精确加法图；只写单 K16 完整有费叶，含节点未命中、单用、全同、一般交集、覆盖/BP；比较以上强臂再决定扩全层。必要量是合成节点数/有效复用次数/省 W 读/增父读写/依赖等待/总退休周期，全部整数端点精确。不能把 HAG 的静态全图搜索当运行时免费 oracle。
- **暂评：** 当前仅候选 3/10；若最后只是 greedy CSE 实现，则仍是 A。可能的研究点是动态完整 T10 条带中对**数据生命周期和共享服务**的可证明 admission，不是 HAG+SNN 组合名。旧“两前缀慢30%”不能杀它，但也不能预设它会赢。

**非 SNN 候选 3：Transitive Array 的静态权重图 → 连续 Q2 的零输入裁剪与有界前缀执行。较低优先级。**

- **B 与强对照：** Q2 的 Z 是连续有符号整数，Phi/Prosperity 的输入二值条件不成立。强对照是现 8 个共享乘法器的同函数 Q2、成熟 CSD/CMVM 公共表达式、直接 bitplane 与全 LUT/DA；不能用 CPU scalar multiply 或不同量化函数作分母。
- **来源 A：** [Transitive Array 原文](https://arxiv.org/html/2504.16339v1) §2–4、Fig.4–8、Alg.1/2 已将**权重**切为二值 TransRows，Hasse 图选择 parent；有静态/动态 scoreboard、缺失父路径、独立 prefix buffer、PPE 与 APE、符号位的有符号重建。不是“把 Prosperity 改名用在 ANN”。本轮实际精读这些方法段，未重做其全论文实验。
- **具体 X：** 对固定 R8 Q2，把 8 个 rank 的整数 Z 保留为加数，离线编译需要的权重 bitplane 子集 DAG；运行时只从真实 final-Z 支持删掉无贡献的叶，并对所需祖先做闭包，限制前缀工作集在声明容量。零裁剪不能删掉别的输出还需的祖先；符号 plane 用减法/符号权重，最后只在现 I24 处 RNE。若静态图的普通 CMVM 已同样实现，视为强 A 而非 X。
- **最小实测：** 一组 N8、全部 T10/P，记录所需节点/closure 节点/重复输出 bitplane/前缀 spills 与回写，完整 signed min/zero/BP gold；先比较共享 multiplier 的实际周期。不能复制原作并行 PPE+APE 后仍报“同8 ALU”。预期风险是原生 R8 的 8×乘法已很强，bitplane 重建反而慢；这只会关掉该图布局。
- **暂评：** 2/10 的成熟权重 CSE/bit-slicing 迁移；只有输入支持裁剪、容量和当前算术重用联合确有新结果才上调。

**保留项：Bishop BSA / Phi PAFT → 按实际服务数的训练目标。** B 是 `N×Hamming` 或总 spike 数不能刻画类表、稀疏格式、packed lane 退休与共享端口费用。先完成上述精确 A，冻结一个实际调度器，再让训练损失拟合其服务数；强对照需包含普通 activity、正确 active-bundle loss、Phi PAFT、相同幅度/同边际分布的置乱分组和等训练预算。历史 NR4 组损失负例限制特定目标，不能自动杀这一家族；新目标若只是多乘几个常数也不是新方法。本轮不训练，不把 q13 或其他学生的质量余量当误差预算；任何有损版本用 matched NB0 的完整 AEE 门独立验证。

**本轮最值得下一步做的一件事。** 先做 §4 的“**固定原森林、每 K16 root/delta 的 Phi 双路求值和就绪发布**”有费叶，原生 zero/onehot、Phi-alone、同父正子集小字典都保留。它直接核实旧 complete_transfer 留下的 K-local 接口，不重做已败的 APEC 先删除/重建图，也不把旧固定父缓存的代数探针换名重领。三个非 SNN 候选继续保留；尤其合成父改变共享图，应在这个更窄的执行对照完成后另作归因，不在第一轮同时叠加。

**材料与未闭合项。** 同名前缀 PDF/TXT 是本轮下载的 primary 原文副本（PTB 明确为公开投稿稿），公式 PNG 是 Bishop PDF 渲染。SIGMA、HAG、JPEG-ACT、SpAtten、Transitive Array 只对上述方法章节作实际精读，不能声称其全文全部核完；CAM 仅上述原文段。PQ/Batcher 原作全文、Bishop BSA 真实 surrogate、Phi 作者 matcher/packer 最坏情况实现、三篇针对本 PSN 的 full-chain 调度均未闭合。上述候选没有当前性能结论，也没有“未试所以无效”的结论。
