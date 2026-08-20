# D2 新颖性档案（Novelty Dossier）：跨窗语义（stride-12 重叠滑窗 + 滚动分母 + 跨窗 quotient 目录）

日期：2026-08-19。对象：D2 合同（CLAUDE_OPERATOR_CONTRACT_DRAFTS_20260818.md 的 D2 部分，J1-J6）与 D2 实现（D2_MOTION_SW12_IMPLEMENTATION_20260819.md，h89/motion_sw12_overlap）。
方法：WebSearch 中英多查询（8 轮，2026-08-19）+ 直接核对禁止粘贴清单（PADE 本次专项确认；其余沿用 D1/D3 档案的机制级精读结论）+ 内部文献核对（docs/433 门槛、CLAUDE_OPERATOR_CONTRACT_DRAFTS 排序、D2 实现说明）。所有引用标注 arXiv 号/会议名/DOI；检索不到即标注"未检索到"，无虚构文献。未碰 GPU、未改任何现有文件，只新建本档案。

---

## 0. 一句话结论

**D2 的两个机制组件（滚动分母 / 跨窗身份复用）各自都有近亲：滚动分母与 FlashAttention 系 online softmax（arXiv:1805.02867 / 2205.14135 / FlashDecoding 博客）同族，跨窗复用与卷积加速器的 overlap 复用（Eyeriss row-stationary / input-stationary dataflow / TrIM）同族——这两个都是拥挤战场；但"跨窗 quotient 目录（共享带身份码持久化）+ 门守恒约束下逐位精确的滚动归一化（J1-J3）+ 类集下界（J4，mean J=0.948 vs lag1 pooled 0.650）"这一组合未检索到占位。风险实质在**组合权重**：执行器组件（−4.8% 净 exp 流量）站不住"收益"叙事，唯一站得住的是目录/身份对象（J4/J5）。新颖性风险等级：高（D2 在合同排序垫底与本次检索结论一致；建议按 §7 降级为 side-note，或重构为"身份目录持久化"中心叙事后再升格）。**

---

## 1. D2 合同要素（对照审查用基线）

| 类别 | D2 要素 | 一句话 |
|---|---|---|
| 新算法算子合同 | 重叠滑窗分区 + 滚动分母恒等 | 1D 重叠分区 `window_partition_overlap(total, 15, 12)`（含尾窗 clamp）：stride-12 滑窗，36% token 重数 mult=2，每窗 90 token 共享带；滚动恒等 `Z_{i+1} = Z_i − Σ_leave + Σ_enter`（J1：增量式与全量重算**逐位相等**，16bit 块分解 int64 精确）；门集成恒等 Σ_t g_final(t) == #windows（J2+J3，Fraction 有理数精确）；类集下界 J(A,B) ≥ \|classes(shared band)\|/\|A∪B\|（J4：合成 Motion C 分布 300 窗链恒成立，mean J=0.948 vs 现网 lag1 pooled 0.650） |
| 新存储对象 | 跨窗 quotient 目录 | 共享带 Q7 类码 + 目录差分（Δcatalog = 进出带类码增量）；相邻窗目录交集中 **55.0% 类码由共享带携带**（J5）；共享带身份码按构造相同（J2：身份码 == 场 flat 下标，mult 180/900 双覆盖） |
| 新执行对象 | 滚动增量执行器 | 每新窗只重算 2×15×3=90 token 进入带的 exp 项（270/窗）而非 450 全量（J6：exp-add 总流量 −4.8% 净；窗口数 520→825，+58.7%）；leave/enter 项由**场坐标几何条带**直接给出（不能用成员掩码相减——实现发现 3：相邻窗 900 个 gather 位置布局不同） |
| 不动项 | Swin 窗口 (2,15,15) 与全部模型参数 | 窗口划分在算子内部完成（h89 纯追加，0 删除行/633 追加行）；checkpoint 与 Motion ep35 锚点直接兼容；**对比口径重建**：锚点 1.3297@ep35 基于 stride-15 稠密非重叠分窗，不可直接比较——合同验证以 h89 内部退化为准（stride=15：mult 全 1、窗口数不增 = 稠密非重叠基线），pass = AEE(stride12) ≤ AEE(stride15)·1.02 |

实测数字（check_d2 脚本 + 单测 38 例，ALL PASS）：J1 逐位 torch.equal + Python int 全量重算 + 16bit 块无溢出；J2 共享带身份；J3 Fraction 精确 Σ==#windows；J4 mean J=0.948 vs 0.650；J5 55.0%；J6 520/825/450/270/234000/222750（−4.8%/+58.7%）。

---

## 2. 文献检索记录（2026-08-19，WebSearch 8 轮）

| # | 查询主题 | 关键命中 | 结果 |
|---|---|---|---|
| 1 | hardware accelerator overlapping sliding window attention cross-window reuse | **SWAT（FPGA，arXiv:2405.17025，DAC'24）**、**SALO（arXiv:2206.14550）**、Focus（arXiv:2512.14661） | 重叠窗的**数据级复用**（K/V 驻留、FIFO、input-stationary）存在；**"注意力分数/分母/目录的结果级跨窗复用"未检索到** |
| 2 | incremental/online softmax sliding window | Milakov & Gimelshein online normalizer（arXiv:**1805.02867**，注意：检索修正了常被误引的 1807.04356）、FlashAttention（arXiv:2205.14135）、FlashDecoding++（arXiv 号未确认，中文解读文章确认内容） | online softmax 是滚动分母的算法先例——最大对抗面之一，见 §4.2 |
| 3 | flash-decoding parallel segments merge running denominator | FlashDecoding（**Stanford CRFM 博客 2023-10-12，非正式论文，arXiv 号未检索到**）、blocked softmax、Hybrid Tree Attention | 分段并行 + log-sum-exp 合并；是**合并式**滚动，不是减法式滚动 |
| 4 | convolution accelerator sliding window overlap reuse | Eyeriss row-stationary（ISCA 2016，经综述 arXiv:2012.11233 确认"maximizes sliding-window reuse"）、input-stationary dataflow、**TrIM（arXiv:2408.01254）**、shift buffer 专利体系 | **最大占位威胁**：overlap 输入复用是卷积加速器几十年的成熟技术，见 §4.1 |
| 5 | Swin transformer hardware shifted window | SWAT（ASP-DAC 2024，IEEE 10473931）、TRFPA（eScholarship，会议名未检索到）、Novella NPU（NCKU 学位论文） | Swin 硬件全部处理**cyclic shift/掩码**（保非重叠分区）；**无人改分区本身为重叠滑窗** |
| 6 | streaming normalization engine hardware | APCCAS 2024 流式 Transformer 片上归一化、**MXFormer（arXiv:2602.12480，deferred softmax 流水化）**、eMamba 范围归一化 | 流式/延迟归一化硬件存在；**对象是单窗归一化器流水，无跨窗增量分母** |
| 7 | attention score result reuse overlapping windows / sliding tile | **STA Sliding Tile Attention（ICML 2025，arXiv:2502.04507）**、**MAC-Attention（arXiv:2604.00235）**、**ReTopK（arXiv:2607.27692）**、DiTFastAttn（venue 未检索到）、LED/Longformer 重叠分块（transformers 源码确认）、vLLM PR #44584（window-align tile） | STA 的立场是**消除**重叠/混合块（硬件不高效）；D2 的立场是**拥抱**重叠并用滚动恒等消解其归一化代价——正反对照见 §4.4 |
| 8 | quantized attention integer softmax 2^s | **I-ViT ShiftMax（ICCV 2023）**、I-BERT（arXiv:2101.01321）、HCCS（arXiv:2604.02292）、P2-ViT（arXiv:2405.19915） | 整数 2^s 幂和分母存在，但（a）分数部分用线性插值**近似**（I-ViT Eq.15）；D2 的 s 是 Q7 整数，2^s 精确；（b）全部是**单窗全量求和**，无跨窗增量；（c）无窗分区重叠 |
| 9 | PADE（禁止清单专项） | PADE（HPCA 2026，arXiv:2512.14322，IEEE 11408448）| BUI-GF 位平面 guard filtering + BS-OOE + ISTA tiling；**无窗分区/重叠/跨窗对象**（本次确认） |
| 10 | spiking transformer window attention overlap | SpikeFET（NeurIPS 2025，全 spike 帧-事件跟踪）| SNN 窗口注意力硬件中"重叠窗 + 滚动分母 + 跨窗目录"未检索到 |
| 11 | streaming ASR overlapping frames reuse | Stateful Conformer（arXiv:2312.17279）、WhisperPipe（arXiv:2604.25611）、streaming-whisper（GitHub） | 重叠帧冗余消除在**时间流/软件侧**（K/V 缓存、chunk-aware look-ahead）；1D 序列、非空间窗分区、非归一化对象 |
| 12 | vision transformer overlapping patches hardware | SHViT（arXiv:2401.16456，重叠 patchify stem）、Trio-ViT（arXiv:2405.03882）、**HOPE（COOL CHIPS 2025，IEEE 11018597）**、Ouroboros（ACM 2025，运动感知缓存复用） | HOPE 的"overlap"是 **head 级执行调度重叠**，非空间窗重叠（本次确认）；重叠 patch 存在于 patchify 嵌入层（conv 实现），非注意力窗分区 |

**未检索到的占位**（诚实声明）：
- "注意力加速器中跨窗 **quotient 目录**（共享带身份码持久化，55% 目录交集类码由共享带携带，J5）"——未检索到（检索 #1/#5/#10 均空）。
- "2D 空间窗口分区上的**减法式滚动归一化**（Z_{i+1}=Z_i−Σleave+Σenter，无 max 重缩放、逐位精确）"——未检索到（online softmax 是乘法重缩放式；FlashDecoding 是 log-sum-exp 合并式）。
- "门守恒（Σ g_final == #windows）约束下重叠窗口分区的归一化无偏性证明（J3 Fraction 精确）"——未检索到。
- "Swin 分区本身从非重叠改为 stride-12 重叠滑窗的硬件算子"——未检索到（Swin 硬件只做 cyclic shift/掩码，SWAT ASP-DAC'24 / TRFPA / Novella NPU 全部保持非重叠 tile）。

---

## 3. 逐篇对照表（工作 / 会议 / 对象 / D2 边界）

| 工作 | 会议/arXiv | 复用/归一化对象 | D2 边界一句话 |
|---|---|---|---|
| **Eyeriss**（row-stationary） | ISCA 2016（经 arXiv:2012.11233 综述确认 RS "maximizes sliding-window reuse"） | 卷积滑动窗**输入像素**复用（operand/data 级，load-once） | **卷积 overlap 复用代表**：复用对象是**输入值**（同一像素供多输出位置），每个输出仍照算；D2 复用的是**计算完成的结果量**（归一化分母 Z 与目录身份码），且无身份语义、无内容相关判定 |
| **TrIM** | arXiv:2408.01254 | 三角形输入流动数据流（最大化输入利用率、去冗余） | 同 Eyeriss 维度：输入移动模式优化；无结果复用、无归一化对象、无目录 |
| **Input-stationary dataflow / shift buffer** | 综述 arXiv:2012.11233 + US 专利群 | 重叠窗输入值移位复用 | 数据级；且实现于**卷积**（线性算子），注意力加速器中的 IS 数据流（SWAT/SALO）复用 K/V **数据**而非**分数归约** |
| **SWAT（window attention）** | DAC 2024，arXiv:2405.17025 | FIFO 滑窗 + input-stationary + QK/Softmax/SV 融合：K/V 数据 load-once | 重叠窗的**数据驻留复用**；无滚动分母（每窗独立 softmax）、无跨窗目录、无身份码 |
| **SALO** | arXiv:2206.14550 | 稀疏注意力数据重排暴露滑动/膨胀窗 K/V 复用（systolic 阵列） | 数据重排型复用；对象是 K/V 装载，非归一化分母与类码 |
| **SWAT（Swin）** | ASP-DAC 2024，IEEE 10473931 | Swin shifted-window 掩码静态稀疏性（mask mode 编码 + SDDMM/SpMM） | 处理**非重叠**分区的 shift/掩码开销；**不改分区为重叠**；无滚动归一化 |
| **TRFPA / Novella NPU** | eScholarship（会议名未检索到）/ NCKU 学位论文 | butterfly NoC 做 cyclic shift；4D DMA/BMM 支持 shifted-window 布局 | 数据搬移/布局层；分区语义不动 |
| **Milakov & Gimelshein** | arXiv:1805.02867 | online normalizer：running max + **乘法重缩放** exp(m_old−m_new)，单遍软最大（浮点） | 滚动归一化之祖，但结构不同：D2 是整数 2^s 幂和、**无 max 相减、无重缩放**、纯加减（J1 逐位精确）；且不涉及窗分区 |
| **FlashAttention** | NeurIPS 2022，arXiv:2205.14135 | tile 化在线 softmax（O(N) 内存），KV 块流式滚动 | 沿 1D 序列维滑；浮点重缩放；无 2D 空间窗、无身份目录 |
| **FlashDecoding** | Stanford CRFM 博客 2023-10-12（arXiv 号未检索到） | KV 分段并行 + log-sum-exp 合并（第二阶段 reduce 核） | **合并式**滚动（两阶段）；D2 是**减法式**滚动（单累加器，leave/enter 条带），无第二阶段合并核 |
| **FlashDecoding++** | 2023（Infinigence-AI；本检索未确认 arXiv 号） | 预定义统一 max φ 消除重缩放依赖（**异步 softmax**） | 承认"固定上界免重缩放"思路存在；但 φ 是统计先验、浮点、1D 序列；D2 的免重缩放来自**整数网格构造**（s∈[0,162] 有界，2^s 精确），且滚动对象含目录 |
| **STA（Sliding Tile Attention）** | ICML 2025，arXiv:2502.04507 | tile 级滑动、**只保留稠密/空块、消除混合/重叠块**（FA 对齐） | **立场相反的对照**：STA 认为 overlap/mixed block 硬件不高效所以设计成非重叠 tile；D2 证明 overlap 的归一化代价可被滚动恒等消解（270/窗）。可写成 discussion 对照 |
| **MAC-Attention** | arXiv:2604.00235 | 相似匹配命中时复用历史注意力结果（band 修正） | 结果复用但**近似/预测器驱动**（band 修正、距离上限）；D2 的复用是**身份恒等**（J2：共享带类码按构造相同），零预测器 |
| **ReTopK** | arXiv:2607.27692 | 相似 query 的 top-k support 缓存复用 | 近似（query 余弦相似度门控）；LLM 长上下文；无窗分区 |
| **DiTFastAttn** | thu-nics（venue 未检索到） | 窗口注意力 + 残差缓存（校准期 cache 全局−局部残差） | 算法侧近似（阈值）；非硬件算子合同；无精确滚动分母 |
| **Longformer/LED** | transformers 源码确认（arXiv 号未在本次检索确认） | 重叠分块 + pad-and-diagonalize | 算法侧重叠窗先例（承认）；无归一化复用、无硬件对象 |
| **vLLM PR #44584** | vLLM 2025 | window-align KV tile 迭代（消减 masked-out tile） | GPU 内核优化；消除的冗余是"窗外的块"，D2 消除的是"窗内的重复 exp 项"——方向相反 |
| **I-ViT（ShiftMax）/ I-BERT / HCCS** | ICCV 2023 / arXiv:2101.01321 / arXiv:2604.02292 | 整数 2^s 幂和分母（近似分数指数 / 32bit 累加 / 倒数归一化） | 承认"整数幂和分母"不新；但均为**单窗全量求和 + 近似**；D2 是跨窗增量、精确、含目录；且项目自身 BSA/Shiftmax 基础同族（内部边界见 §5） |
| **Stateful Conformer / WhisperPipe** | arXiv:2312.17279 / arXiv:2604.25611 | 流式 ASR 重叠帧冗余：K/V 缓存 + chunk-aware look-ahead | 时间流 1D、软件/训练侧、缓存投影而非归一化；无空间 2D 窗分区、无精确滚动分母 |
| **Focus** | arXiv:2512.14661 | VLM 流式架构，stride-1 卷积式滑窗块相似检测 | 相似性**检测**（近似）；D2 是身份恒等（构造性） |
| **HOPE** | COOL CHIPS 2025，IEEE 11018597 | **head 级**执行重叠（调度） | 同名"overlap"但对象是 head 调度，与空间窗重叠零关系（本次确认） |
| **Ouroboros** | ACM 2025（DL 10.1145/3745756.3809212） | ViT 视频分析运动感知缓存复用（patch 位移仿射建模） | 帧间 token 特征复用（近似缓存）；无归一化对象 |
| **PADE** | HPCA 2026，arXiv:2512.14322 | 位平面 QK 稀疏跳过（BUI-GF）+ 乱序执行 + 稀疏 tiling | 禁止清单成员；无窗分区/重叠/滚动分母/目录对象（本次专项确认） |
| **SpAtten / Transitive Array / FuseMax / TeAAL / FLAT / Prosperity / Bishop** | 见 D1 档案 §8（ISCA'21 / ISCA'25 / MICRO'24 / MICRO'23 / MICRO'23 / HPCA'25 / ISCA'25） | token 剪枝 / 位片行前缀复用 / einsum 映射 / 数据流融合 / 空间行产品稀疏 / TTB 打包 | 禁止清单成员；机制级核对见 D1 档案 §3，全部无窗分区重叠、无滚动归一化、无跨窗目录。FLAT 本次补充确认内容（fusion+tiling、1.94× speedup、49% 能耗节省），与 D1 判定一致：数据流优化，无窗分区对象 |
| **RADiT** | DAC 2025 | ANN DiT 时间步相似块复用 | 时间步粒度近似；无窗分区 |
| **Zhang et al.** | CICC 2026，DOI 10.1109/CICC65509.2026.11509564 | 事件光流 U-Net 冗余推测 + 输入流相似跳过 | 同任务域最近作；无注意力、无窗、无归一化对象（D1 档案精读确认） |

---

## 4. 与最接近三篇的深边界（审稿人最可能引用）

### 4.1 vs 卷积 overlap 复用（Eyeriss row-stationary / input-stationary / TrIM）——"overlap 复用是卷积加速器的老技术"

这是 D2 **最大的占位威胁**，必须正面处理。边界：

1. **复用层级不同：operand/data 级 vs 结果级**。卷积 overlap 复用复用的是**输入像素值**（同一像素被多个输出窗消费 → load-once、移位缓冲），每个输出位置的计算**仍然全部执行**。D2 复用的是**计算完成后的结果量**：归一化分母 Z（450 项分数的归约结果）与跨窗 quotient 目录（共享带身份码）。卷积文献里"结果级复用"的对应物是**稀疏结果复用**（Prosperity 内积复用、Transitive Array 位片行前缀复用），而这两者的复用维度是空间行/位片，不是**窗口分区维**。
2. **语义内容不同**。卷积复用只有"位置关系"（哪些像素被哪些窗共享）；D2 的复用有**内容关系**：J4 类集下界 J(A,B) ≥ |classes(shared band)|/|A∪B|（300 窗链恒成立，mean J=0.948 vs 现网 lag1 pooled 0.650）与 J5（55.0% 的相邻窗目录交集类码由共享带携带）——"共享带携带了相邻窗目录的大部分类码"是**类码内容**的统计事实，卷积数据流没有可对应的陈述。
3. **复用成立的条件不同**。卷积 overlap 复用无条件成立（几何共享）；D2 的 Z 滚动是**有约束的恒等**——J1 要求 leave/enter 条带与场坐标几何严格一致（实现发现 3：不能用成员掩码相减，相邻窗 gather 布局不同），J3 要求门守恒 Σ_t g_final(t) == #windows（Fraction 精确）。复用必须满足**门守恒硬约束**，这是卷积数据流不存在的问题（卷积无归一化守恒）。
4. **精确性语义不同**。卷积数据流复用"相同值再装载"，精确性无感；D2 的 Z 是**归约量**，少加/多加一个 leave/enter 项就破坏归一化（J1 以 torch.equal + Python int 全量重算 + 16bit 块无溢出三重验证钉死）。"增量归约必须逐位等于全量重算"是 D2 的合同级精确性要求，卷积文献无此对象。

**结论**：卷积 overlap 复用是"输入数据的几何共享"技术；D2 是"归一化归约与身份目录的结果级跨窗复用"——层级（结果 vs 输入）、语义（内容相关 vs 位置相关）、约束（门守恒 vs 无）、精确性（逐位合同 vs 无感）四处不同。但必须承认：若 D2 的写作把重心放在"每窗少算 180 个 exp 项"，审稿人完全可以拿 Eyeriss 系数据流回答"这不就是数据复用"——因此 §7 的叙事建议（目录中心）是硬性要求。

### 4.2 vs online softmax / FlashAttention 系（Milakov arXiv:1805.02867、FlashAttention 2205.14135、FlashDecoding、FlashDecoding++）——"滚动分母是 online softmax 换皮"

这是**第二杀法**，且部分成立。边界：

1. **数值结构不同**。online softmax 的核心是 running max + **乘法重缩放** d_j = d_{j−1}·e^(m_{j−1}−m_j) + e^(x_j−m_j)——需要 max 维护与重缩放因子（浮点）。D2 的 Z = Σ 2^s 在 Q7 整数网格 [0,162] 上**无 max 相减、无重缩放**：Z_{i+1} = Z_i − Σ_leave + Σ_enter 是纯整数加减（16bit 块分解 int64 精确），增量与全量重算**逐位相等**（J1）。免重缩放不是统计先验（对比 FlashDecoding++ 的 φ 是分布统计），而是**整数网格构造**（s 有界、2^s 精确可移位）。
2. **滚动维度和对象不同**。online softmax 沿 **1D 序列（KV 块）**滑，每步只加新块（序列单调增长）；D2 是 **2D 空间窗口分区**（15×15 tile 在场坐标上 stride-12 滑），跨窗共享 36% token（mult=2），更新是"加 enter 条带 + **减 leave 条带**"的双向滚动（有进有出），条带由场坐标几何给出。FlashDecoding 的合并是分段**并行**后的 log-sum-exp 规约（两阶段、需要第二阶段 reduce 核）；D2 是**单累加器减法滚动**，硬件只有加减流水，无合并核。
3. **归一化的用途不同**。online softmax 的滚动是为了**输出 O 的流式累积**（O 在行内重缩放）；D2 滚动的是**行内归一化分母 Z**，用于门（gate）的 shiftmax 归一化，且受**门守恒**约束（J3：Σ g_final == #windows 精确）。门归一化的无偏性是 D2 的合同主张，online softmax 文献不讨论"Σ_t g_final(t) == #windows"这类全局守恒。
4. **D2 的存储对象在 online softmax 中不存在**：跨窗 quotient 目录（共享带身份码持久化、Δcatalog 差分）。online softmax 只维护 O(m) 的 running 统计量，从不维护**跨窗类码目录**——这是存储对象层的差异，不是数值技巧的差异。
5. **诚实边界**：若把 D2 的贡献写成"滚动分母省 exp 流量"，这就是换皮——FlashAttention 系已经把"流式分母"做透了。D2 的护城河只存在于**目录对象 + 2D 空间窗 + 门守恒**的组合（见 §7 叙事建议）。

### 4.3 vs 重叠窗注意力加速器（SWAT arXiv:2405.17025、SALO arXiv:2206.14550）——"重叠窗复用已有硬件先例"

1. **复用对象**：SWAT/SALO 复用 K/V **数据**（FIFO 驻留、input-stationary、数据重排）——省的是**装载带宽**；D2 复用**分数归约结果 Z 与类码目录**——省的是 **exp-add 计算项**（450→270/窗）。装载复用 vs 计算复用。
2. **分区语义**：SWAT/SALO 处理的是算法给定的滑动窗（1D 序列窗）；D2 改变的是 **Swin 2D 分区本身的 stride**（15→12），产生 mult=2 重叠重数，并以 J2 身份恒等（共享带类码按构造相同）为硬件目录的机制基板——这是**算子合同级**的改变（新存储对象 + 新执行对象，docs/433 三件套），不是数据流调度。
3. **精确性合同**：SWAT/SALO 无"滚动分母与全量重算逐位相等"的验证对象；D2 有（J1 单测硬约束）。

### 4.4 vs STA（Sliding Tile Attention，ICML 2025，arXiv:2502.04507）——直接对撞的立场

STA 与 D2 是**同一问题的相反回答**：重叠/混合窗口块在硬件上不高效（masked-out 计算冗余），STA 的解法是**把窗口设计成 tile 对齐的非重叠块**（只保留稠密/空块）。D2 的解法是**保留重叠，用滚动恒等把归一化代价消解**（每窗 exp 项 450→270）。这是 discussion 里最有力的对照：STA 用"消除重叠"回避问题，D2 用"增量归约"吸收重叠。但反过来说，STA 的存在也证明"审稿人知道重叠窗的归一化是公认的硬件负担"——D2 必须正面回答"为什么不让 overlap 消失"（答案：目录持久化的身份价值 + token 处理总量 +36% 的诚实账）。

---

## 5. 与项目内部贡献的边界（防"自我重复"杀法）

| 内部对象 | 出处 | D2 边界 |
|---|---|---|
| H67 Motion（T=2 pair 商，非重叠 (2,15,15) 分区） | 现网合同 | D2 是**分区语义**的改变（stride 15→12、mult=2、共享带），H67 要求 D%window[0]==0 非重叠零共享（round2 否决"跨窗 quotient 持久"的**身份死穴**）；D2 把共享写进合同（J2 身份按构造相同），是死穴的机制基板 |
| H82 class-major 目录 / H86 member-delta | docs/433 / docs/445 | H82/H86 是**窗内**目录（class-stationary descriptor、成员差分）；D2 的跨窗 quotient 目录是**跨窗持久化维**（Δcatalog = 进出带类码增量），对象维度不同（窗间 vs 窗内），且 J5 给出 55.0% 的共享带携带率实测——不是 H82 的换名 |
| D1（T=5 时间商 + RLE 广播）/ D3（方向场 stencil） | 同批草案 | D1 是时间维商，D3 是空间 stencil 分数偏移；D2 是**窗口分区维**语义——三维正交；D2 实现内 h87/h88 路径源码逐字节不变 |
| BSA/Shiftmax（2^s 整数软最大） | 项目基础（NeurIPS 2025） | D2 的 2^s 幂和是 BSA 网格的**跨窗滚动**扩展——单窗全量求和是现网，跨窗增量是 D2；I-ViT/HCCS 同为"整数幂和"家族（检索 #8），内部基础 + 外部同族双重承认，D2 的新颖性只在滚动与目录 |

---

## 6. 审稿人预答辩（DATE 最可能三条杀法）

### 杀法 1："overlap 复用是卷积加速器的老技术——Eyeriss row-stationary、input-stationary dataflow、TrIM 几十年前就做过滑动窗复用"

**反驳（引用 J1-J6 与检索记录）：**
1. **承认前件、拒绝结论**：卷积 overlap 复用（Eyeriss ISCA'16、IS dataflow、TrIM arXiv:2408.01254、shift buffer）确实成熟——但检索记录（§2 #4）显示其复用对象是**输入像素值**（operand/data 级 load-once），每个输出位置的计算仍然全部执行。D2 复用的是**计算完成的结果量**：归一化分母 Z（450 项分数归约的结果）与跨窗 quotient 目录（共享带身份码）。"结果级复用"在硬件文献中的对应物是 Prosperity/Transitive Array 的稀疏结果复用——其复用维度是空间行/位片，**没有窗口分区维**（专项检索 #1/#5/#10 均空）。
2. **卷积数据流没有内容语义**：卷积复用只由几何位置决定；D2 的复用有内容事实——J4 类集下界 J(A,B) ≥ |classes(shared band)|/|A∪B| 在 300 窗链恒成立（mean J=0.948 vs 现网 lag1 pooled 0.650），J5 相邻窗目录交集 55.0% 类码由共享带携带。"共享带=相邻窗目录的主要载体"是可证伪的**内容命题**（验证实验 3 可直接 dump 裁决），卷积数据流没有可对应的命题。
3. **D2 的复用有卷积没有的守恒约束**：J3 门集成恒等 Σ_t g_final(t) == #windows（Fraction 有理数精确）——滚动分母任何一位偏差都会破坏门守恒；J1 以 torch.equal + Python int 全量重算 + 16bit 块无溢出三重验证钉死"增量 ≡ 全量"。卷积数据流复用"相同值"无守恒问题，D2 复用的是**归约值**，逐位精确是合同硬约束而非实现细节。
4. **分层**：卷积 overlap 复用消除的是**装载**（memory 层级）；D2 消除的是 **exp-add 计算项**（450→270/窗，J6）并新增**目录持久**（身份码 × 重数 2 的下限账）。若审稿人追问"为什么不是数据流就能做"——因为 D2 的共享对象（Z 与目录）在**注意力算子内**，不在访存路径上，卷积数据流语言没有"归约结果跨窗复用"这个槽位。

### 杀法 2："滚动分母就是 FlashAttention 的 online softmax 换皮——Milakov 2018 就做完了"

**反驳（这是最诚实的对抗，部分承认）：**
1. **承认 online softmax 是滚动归一化之祖**（Milakov & Gimelshein，arXiv:1805.02867，注意检索修正了 arXiv 号：常被误引为 1807.04356，正确为 **1805.02867**；FlashAttention arXiv:2205.14135 使其成为主流）。
2. **数值结构三处不同**：(a) online softmax 需要 running max + **乘法重缩放** exp(m_old−m_new)（浮点修正）；D2 的 Z=Σ2^s 在 Q7 整数网格 [0,162] 上**无 max、无重缩放**，Z_{i+1}=Z_i−Σleave+Σenter 纯加减，16bit 块分解 int64 逐位精确（J1）。免重缩放来自**整数网格构造**而非统计假设（对比 FlashDecoding++ 的统一 max φ 是分布先验）。(b) online softmax 沿 **1D 序列维单调增长**；D2 是 **2D 空间窗分区**（15×15 tile、stride-12、mult=2、36% 重叠），滚动是**有进有出的减法式**（enter 条带加、leave 条带减，按场坐标几何给出——实现发现 3：不能用成员掩码相减）。(c) FlashDecoding 的跨段合并是**两阶段 log-sum-exp 规约**（需第二阶段 reduce 核）；D2 是**单累加器减法滚动**，硬件只增加减流水，无合并核。
3. **D2 的贡献重心不在分母滚动**：新存储对象（跨窗 quotient 目录：共享带身份码持久化 + Δcatalog 差分，J5 55.0% 携带率）与门守恒归一化（J3）在 online softmax 文献中不存在——它们维护 O(m) 统计量，从不维护**跨窗类码目录**。**防守策略**：论文中主动把"滚动分母"定位为机制组件而非贡献主张，贡献主张落在"身份目录持久化"（见 §7 叙事建议 3）——这同时化解本杀法与杀法 1。
4. **可证伪性**：J1 的逐位精确与 J3 的 Fraction 守恒是算子级硬约束（CPU 单测 38 例全绿），任何近似化实现（如 float 滚动）都会在测试中断言失败——"精确滚动"不是叙事而是可执行断言；online softmax 系（浮点重缩放）没有逐位合同。

### 杀法 3："+58.7% 窗口数是负收益——增量只省 4.8% 净流量，改 Swin 分区得不偿失"

**反驳（这是 D2 在合同排序垫底的真实原因，用对比口径方案正面回应）：**
1. **对比口径错误是审稿人视角的误导**：Swin 论文以非重叠 tile 报告，D2 的 AEE 与稠密基线**不可直接比较**——合同已冻结对比口径（实现说明 §7）：**h89 内部 stride=15 退化解**（mult 全 1、窗口数不增 = 稠密非重叠基线），pass = AEE(stride12) ≤ AEE(stride15)·1.02。+58.7% 窗口数是"以窗口为单位"的口径；以 **exp 项流量**为口径（J6）：450×520=234000 → 270×825=222750（**−4.8% 净**），以 **token 重叠**为口径：36% token 重数 2。三个口径必须并排报，审稿人才不会拿单个口径做文章。
2. **收益不在流量，在目录资产**：D2 的 433 价值是"身份死穴的机制基板"（round2 否决跨窗 quotient 持久就是因为非重叠零共享）。J4/J5 给出机制保证：mean J=0.948 的类集持久 + 55.0% 交集类码由共享带携带——**跨窗共享从统计机会变成合同保证**（J2 身份按构造相同）。这是"为什么值得改 Swin 分区"的唯一站得住的回答：stride-15 非重叠下相邻窗目录交集只能靠统计（lag1 pooled 0.650），stride-12 下由共享带构造性携带（J4 下界 0.948）。如果 DATE 只接受流量叙事，D2 应主动降级（见 §7）。
3. **与 STA（ICML 2025）的立场对照**：STA 因重叠/混合块在硬件上不高效而**消除**重叠（tile 对齐非重叠块）；D2 证明 overlap 的归一化代价可被滚动恒等消解（每窗 −180 exp 项，J6）。同一事实的两个答案——"重叠是负担"（STA）vs "重叠的负担可增量消除且带来目录价值"（D2）——是 discussion 的核心，也是回应"负收益"的最直接材料：重叠的**执行代价**已被 −4.8% 净账证明不增反减，重叠的**token 处理总量** +36% 是诚实成本（写入论文，不回避）。
4. **可证伪设计**：验证实验 2（fullres ft40 对比口径 AEE(stride12) ≤ AEE(stride15)·1.02）+ 实验 1（short loss 不塌）+ J1-J6 算子级单测（38 例）——若 fullres 未过，D2 合同自我否决；这个自证伪协议本身回应"负收益"质疑（对比口径重建是合同的一部分，不是补救）。

### 附加杀法（备用）："重叠窗就是 Longformer 的 sliding window 换名——算法侧早就有了"
**反驳**：承认算法侧重叠窗先例（Longformer/LED 重叠分块、Swin shifted window 跨窗连接、STA tile 滑动）——但 D2 的对象是**硬件算子合同**（docs/433 三件套）：新算法合同（stride-12 分区 + 滚动恒等 J1）+ 新存储对象（跨窗 quotient 目录 J4/J5）+ 新执行对象（滚动增量执行器 J6）。算法侧工作无归一化复用（Longformer 每窗全量 softmax）、无门守恒、无目录、无逐位精确合同。"算法有重叠窗"与"硬件有跨窗状态复用"是两回事（检索 #1/#5 确认：硬件侧重叠窗工作只做数据驻留复用）。

---

## 7. 诚实结论：新颖性风险等级与补救方向

### 风险等级：**高**（D2 在合同排序垫底与本次检索结论一致）

**降级风险因素（比 D1/D3 更重）**：
- **两个机制组件各自都有拥挤的近亲**：滚动分母 ↔ online softmax 系（1805.02867 / 2205.14135 / FlashDecoding / FlashDecoding++，"换皮"指控有真实文本支撑）；跨窗复用 ↔ 卷积 overlap 复用（Eyeriss/IS/TrIM，"老技术"指控有真实文本支撑）。**这是 D1（时间商）/D3（方向场偏移）没有的双线拥挤**。
- **执行账太弱**：−4.8% 净 exp 流量 vs D1 的 −78.3% 门流量——收益叙事在数量级上被同项目压过；窗口数 +58.7% 是显性负项，token 处理总量 +36% 是诚实成本。
- **对比口径必须重建**：valid825 锚点 1.3297@ep35 不可直接比较，Swin 基线论文口径是稠密非重叠——这是 D2 独有的验证成本（D1/D3 无此问题）。
- **STA（ICML'25）已把"重叠=硬件负担"写成论文共识**：审稿人的默认立场是"overlap 应被消除"，D2 要逆着共识论证"overlap 可被吸收且有目录价值"。

**支撑等级的因素**：
- "跨窗 quotient 目录（共享带身份码持久化）+ 55.0% 携带率（J5）+ 类集下界 0.948（J4）"这一**存储对象组合未检索到占位**（专项 #1/#5/#10 全空）。
- "2D 空间窗分区上的减法式精确滚动归一化（无重缩放、J1 逐位）+ 门守恒（J3 Fraction 精确）"未检索到占位。
- J1-J6 全部 CPU 单测验证（38 例）+ 注入式 forward 级复验——"增量≡全量逐位相等"是**可执行断言**，文献中的滚动归一化无此合同级验证。
- 禁止粘贴清单十项全部不触碰（PADE 本次专项确认；其余沿用 D1/D3 机制级核对）。

### 为什么值得改 Swin 分区（正面回答）——或降级建议

**改分区的唯一充分理由是目录/身份资产，不是流量**：
- round2 对"跨窗 quotient 持久"的否决根因是**非重叠零共享**（H67 要求 D%window[0]==0）；stride-12 重叠使共享带成为**构造性**身份载体（J2：身份码 == 场 flat 下标，mult=2 双覆盖），跨窗目录从"统计机会"（lag1 pooled J=0.650）升级为"合同保证"（共享带下界 J≥|classes(shared band)|/|A∪B|，mean 0.948）。
- 55.0% 的相邻窗目录交集类码由共享带携带（J5）——目录差分 Δcatalog 的写成本被共享带身份码 + 重数 2 的账覆盖，这是硬件跨窗目录（H82 窗内目录的窗间扩展）的机制基板。
- **但**：这份资产只有在"身份目录持久化"成为论文主线时才能兑现。若 DATE 论文主线是流量/能耗（本项目的其他线），D2 的价值无法承载独立合同。

**诚实建议（与合同排序一致）**：
1. **默认路径（推荐）**：D2 **降级为 side-note/第二轮候选**——身份机制叙述（跨窗目录持久化）并入 H82 线或作为 future work，不主张独立合同。理由：执行账（−4.8%）无法对抗 online softmax 与卷积 overlap 复用的双重"换皮/老技术"指控；验证成本（口径重建 + fullres ft40）与收益不匹配。
2. **升格条件（若坚持独立合同）**：必须完成三件事——(a) fullres 对比口径验证通过（AEE(stride12) ≤ AEE(stride15)·1.02）；(b) 把目录资产量化成硬件账（跨窗目录持久 = 共享带身份码 × 重数 2 的下限，J5 55.0% 携带率 → 目录写成本节省），与执行账分开报；(c) 论文把"滚动分母"降级为机制组件，主线改写为"**身份目录持久化**"（identity-catalog persistence），与 STA 的"消除重叠"形成正反对照 discussion。
3. **写作层红线**：related work 必须主动消化 online softmax 系 + 卷积 overlap 复用系（§3 对照表），逐条差分"结果级 vs 输入级、减法式 vs 重缩放式、目录 vs 统计量、2D 空间窗 vs 1D 序列"；禁止出现"我们首次提出滚动 softmax"类表述。

---

## 8. 引用清单（本次检索确认 / D1、D3 档案继承，全部真实）

1. Milakov & Gimelshein: Online normalizer calculation for softmax — arXiv:**1805.02867**（本次检索确认；注意常被误引的 1807.04356 为错误号）
2. Dao et al.: FlashAttention — NeurIPS 2022，arXiv:2205.14135
3. FlashDecoding — Stanford CRFM 博客（2023-10-12，Tri Dao 等；非正式论文，**arXiv 号未检索到**）
4. FlashDecoding++ — Infinigence-AI 等，2023（内容经中文解读文章确认；**arXiv 号未检索到**）
5. Eyeriss（row-stationary，sliding-window reuse）— Chen et al., ISCA 2016（经 arXiv:2012.11233 综述确认"RS maximizes sliding-window reuse"）
6. Hardware/software optimizations for DNN survey — arXiv:2012.11233
7. TrIM: Triangular Input Movement Systolic Array — arXiv:2408.01254
8. SWAT: Scalable and Efficient Window Attention-based Transformers Acceleration on FPGAs — DAC 2024，arXiv:2405.17025
9. SWAT: An Efficient Swin Transformer Accelerator Based on FPGA — ASP-DAC 2024，IEEE 10473931（与 8 同名不同作，注意区分）
10. SALO: An Efficient Spatial Accelerator Enabling Hybrid Sparse Attention — arXiv:2206.14550
11. STA: Fast Video Generation with Sliding Tile Attention — ICML 2025，arXiv:2502.04507
12. MAC-Attention — arXiv:2604.00235（2026）
13. ReTopK: Similarity-Guided Top-K Reuse — arXiv:2607.27692（2026）
14. DiTFastAttn — thu-nics（venue 未检索到）
15. Stateful/Cache-aware Streaming FastConformer — arXiv:2312.17279
16. WhisperPipe — arXiv:2604.25611
17. I-ViT: Integer-only Quantization（ShiftMax）— ICCV 2023
18. I-BERT — arXiv:2101.01321
19. HCCS: Taming the Exponential（int8 softmax 硬件映射）— arXiv:2604.02292
20. P2-ViT: Power-of-Two PTQ — arXiv:2405.19915
21. MXFormer: Microscaling FP CIM Transformer Accelerator（deferred softmax 流水）— arXiv:2602.12480
22. Streaming Transformer Accelerator with Efficient On-Chip Normalization — APCCAS 2024（Zhao et al.）
23. HOPE: Head-Wise Overlap Processing — COOL CHIPS 2025，IEEE 11018597（**overlap 为 head 级调度，非空间窗重叠**，本次确认）
24. TRFPA（butterfly NoC cyclic shift）— eScholarship 预印本（**会议名未检索到**）；Novella NPU — NCKU 学位论文
25. Ouroboros: Motion-Aware Cache Reuse for ViT — ACM 2025，DL 10.1145/3745756.3809212
26. Focus: Streaming Concentration Architecture — arXiv:2512.14661
27. SHViT（重叠 patchify stem）— arXiv:2401.16456
28. Longformer/LED（重叠分块）— 机制经 transformers 源码确认（**arXiv 号未在本次检索确认**）
29. vLLM PR #44584（window-align KV tile 迭代）、#24390（tile 剪枝）；sglang PR #8860（全掩码 tile 跳过）— GPU 内核层优化
30. PADE — HPCA 2026，arXiv:2512.14322，IEEE 11408448（禁止清单，本次专项确认无窗分区对象）
31. SpAtten — ISCA 2021，arXiv:2012.09852；Transitive Array — ISCA 2025，arXiv:2504.16339；FuseMax — MICRO 2024，arXiv:2406.10491；TeAAL — MICRO 2023，arXiv:2304.07931；FLAT — MICRO 2023（arXiv 号未确认；本次补充确认内容：fusion+tiling、1.94×/49%）；Prosperity — HPCA 2025，arXiv:2503.03379；Bishop — ISCA 2025，arXiv:2505.12281；RADiT — DAC 2025；Zhang et al. — CICC 2026，DOI 10.1109/CICC65509.2026.11509564（以上机制级核对继承 D1 档案 §3、§8）
32. 内部：docs/433（4.0 三件套门槛）、CLAUDE_OPERATOR_CONTRACT_DRAFTS_20260818.md（D2 合同与排序）、D2_MOTION_SW12_IMPLEMENTATION_20260819.md（J1-J6 实现与对比口径）、D1/D3_NOVELTY_DOSSIER_20260818.md（禁止粘贴清单机制核对依据）
33. 未检索到（诚实声明）：注意力加速器中"跨窗 quotient 目录/共享带身份码持久化"；2D 空间窗分区上的减法式精确滚动归一化；门守恒约束下的重叠窗归一化无偏性；Swin 分区本身改为重叠滑窗的硬件算子；FlashDecoding/FlashDecoding++ 的 arXiv 号；DiTFastAttn 与 TRFPA 的会议名；Longformer 的 arXiv 号（本次检索内）
