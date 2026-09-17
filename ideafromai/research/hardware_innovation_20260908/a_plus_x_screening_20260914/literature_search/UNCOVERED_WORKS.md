# 未覆盖工作检索记录（2026-09-14，任务5）

检索方式：arXiv 站内搜索（WebFetch）；OpenAlex/DBLP/SemanticScholar/arXiv API 均因共享代理出口限流不可用。已对照本地 775 实体 catalog 与各 README 近邻清单去重。

## 新发现（未入 catalog，与存活卡直接相关）

| 工作 | 出处 | 与本项目的关系 | 深读状态 |
|---|---|---|---|
| **MINT** (arXiv:2606.31514) | SOCC 2026 | MSDF 位串行 + 达到目标精度即终止；**逐层静态精度**（INT2–7 贪心、2% 精度预算）。C1 的最近邻：C1 差分=按**消费者**分级（同源双消费者不同深度）+ 动态逐判决裕度 + 精确等价（failure 标志），MINT 是逐层近似预算 | 摘要+结果表已读 |
| **ANNS-AMP** (arXiv:2606.07156) | 2026 | 位串行运行时精度自适应 + 轻量预测器提前终止（近似）。C1 近邻，同上差分 | 摘要级 |
| **The Sparsity Tax** (arXiv:2607.22790) | MCSoC 2026 | lockstep SIMD vs bitmap 门控 Sparse-SIMD vs SIMT 的稀疏开销：SIMD 近常数吞吐、门控元数据税、无架构线性缩放。**直接佐证 kill B/C 的粒度结论**（lane 门控有元数据税，块聚合吃掉收益）；GF22FDX RTL-to-gates，代码开源 | 摘要级，全文待取 |
| **APEX** (arXiv:2608.19046) | 2026 | LoAS 框架 + PASC-IF 神经元，输入脉冲×权重双稀疏，时间并行 dataflow，开销 1.3–5.4% 功率。LoAS 家族延伸，C2 近邻 | 摘要级 |
| **M-HySMap** (arXiv:2608.26223) | 2026 | 活动加权多播超图映射，多消费者脉冲路由（降 10.6–19.6% hop）。F1/C2 多消费者近邻 | 摘要级 |
| **SupraSNN** (arXiv:2606.13354) | 2026 | 超标量 SNN + Multi-Cast Tree 脉冲分发到并行突触。多消费者投递近邻 | 摘要级 |
| **ASTER** (arXiv:2511.06770) | 2025 | spike transformer 专用 PIM 加速器（~467× 能效 vs Orin Nano）。竞争性同域工作 | 摘要级 |
| **Lonic** (arXiv:2608.12500) | ICCAD 2026 | INT4 局部在线训练协设计。训练侧，非本卡直接近邻 | 摘要级 |
| **Sparsity Ceiling** (arXiv:2607.26648) | 2026 | 论证脉冲 Transformer 稀疏收益任务相关、主要在感知域。支持"供数义务"叙事动机 | 摘要级 |
| **Matterhorn** (arXiv:2601.22876) | 2026 | TTFS 编码；实测累积<3% 能耗、数据搬运主导——**直接支持"减少供数义务"的收益框架** | 摘要级 |

## 已知/已在 catalog 的确认

ExSpike（FPL 2026，本地已融合且更慢）、FlexSpIM（ISCAS 2025 全文已在 literature/）、SpiNNaker2、FireFly-T 系。

## 对卡片的更新动作

1. C1 近邻补：MINT（逐层精度预算 vs 消费者分级精确证书）、ANNS-AMP、BitFair、BISMO/BBS。
2. C2 近邻补：APEX（LoAS 双稀疏延伸）、M-HySMap/SupraSNN（多消费者路由）。
3. Sparsity Tax 的结论（lane 门控元数据税、无线性缩放）= C1 lane 证书设计必须正面回答"证书判定/元数据费用"，写进杀门3。
4. Matterhorn 数据搬运主导 = 所有卡的"供数义务"叙事引用支撑。

## 全文/深读补记（2026-09-14 第二轮）

- **MINT 全文深读**（HTML）：其"终止"就是**离线逐层精度查表后的固定周期数**（C(P)=2P+10），无数值证书、无运行时残差检查；误差界来自 MSD 顺序的结构性 bound（≤2^−P），精度由离线贪心（逐层单层扰动 + 全网复评 + 2% 预算）选定；面积固定 INT8 代价、只省周期与功耗。**C1 差分定案：运行时逐判决精确证书（margin>L1·2^−b）× 消费者身份驱动 × 样本内零翻转，vs MINT 离线逐层近似预算。** MINT 自认 PyTorch 模拟是"硬件性能上界"。
- **D-com 深读**（摘要级+结构）：运行时激活低秩分解（Lanczos 渐进），费用用 compute replication（访存→计算区搬移，6.2×）与 output shape-preserving 跨层复用摊销，multi-track 处理离群通道；对象是 Llama2-7b/A100 的 LLM 大激活。**与本地 T10/R8 小尺寸失配，C6 降为低优先**；"分解结果跨层保形复用"思想可与源字共享消费对照。
- **复用/缓存方向检索**（215 结果扫读）：主体是 LLM/diffusion KV/feature 缓存，与本项目机制不直接相关；仅 DSTAR（MICRO26，差分激活+稀疏复用）、VQVLA（运动感知 VQ+质心复用）可作二队列参考。

## 第三轮（2026-09-15，C1 形态定案后的近邻精查 + 2025-26 顶会顶刊扫描）

检索途径：arXiv 搜索/列表、Crossref API、Semantic Scholar 引文链、作者主页 PDF（DDG/Bing/IEEE 大多被反爬拦截）。

### C1 形态近邻精查（威胁排序）

| # | 工作 | 出处 | 机制 | 与 C1 差分 | 威胁 |
|---|---|---|---|---|---|
| 1 | **AEU 2026**（Sayadi/Moaiyeri/Timarchi, DOI 10.1016/j.aeue.2026.156548）"Activation-deterministic early termination … while preserving accuracy" | AEU 26, vol.217 | 二值/三值 CNN 加速器"激活确定即终止"且保精度（引 ECHO/BitSET，同一谱系） | 精确终止概念重叠最重；但域是 BNN/TNN 值级/popcount，非位平面区间证书、无 BFP、非门路判决。**Crossref 参考文献画像已核验**（29 条引文全为 BNN/TNN 值级谱系：XNOR-Net/FATNN/BCIM/O3BNN-R/ECHO/BitSET/Snapea，无 bitplane/BFP/区间界引文）；摘要与全文仍未取到（SD 403），投稿前仍需人工核验全文 | 高 |
| 2 | **ECHO** | Electronics(MDPI) 24, DOI 10.3390/electronics13101893 | MSDF 在线算术（最高位先行 digit-serial）+ 负输出精确符号识别提前终止 | 只判 ReLU 符号，无剩余位区间界、无逐判决证书、无 BFP，CNN 域 | 中 |
| 3 | **BitSET** | CASES23/TECS | bit-serial MSB 先行 + 阈值启发式终止 | 近似（容 1% 精损），非精确证书 | 中 |
| 4 | **BitStopper** (arXiv:2512.06457) | 25.12 | Transformer attention bit-serial 级融合 + 投机式 token 渐进终止 | 投机/启发式，无精确保证、无区间界与 BFP | 中 |
| 5 | **DSLOT-NN** | DSD 23 | MSD-first 在线算术 + 无效卷积终止，精度-功耗可调 | 近似、可调精度，非零差证书 | 中低 |
| 6 | **FlexSpIM** (arXiv:2609.08446) | 26.9，40nm CIM，Frenkel 组 | bit-serial/位并行可重构 operand shaping 的 event/SNN CIM 芯片 | 全文零处 "early termination"，无证书无 BFP；但同属 event+SNN+bit-serial 语境，**务必引用划界** | 中 |
| 7 | ICCD25 **SPoT** | ICCD 25 | 用最高幂次项估计激活符号提前终止（ReLU） | 符号估计、近似 | 低中 |
| 8 | **LeOPArd** | ISCA 22 | bit-serial attention + 可学习阈值位级提前终止 | 学习阈值，近似 | 低 |
| 9 | **ConvReflex** | SenSys 26 | 编译期捷径跳过必被 clamp 的卷积 | 受控精损、MCU 软件 | 低 |

**BFP×动态终止组合查空**：多组检索（block floating point + early exit/termination/adaptive）零命中；组级 BFP 与逐判决提前终止的组合未见先例。

**新颖性结论**：三要素组合（MSB-first 位平面串行供数 + 逐判决精确区间证书锁定即停 + 组级 BFP）在检索范围内仍为空白。现有 MSB-first+终止工作全部是近似/可学习/离线预算；"精确终止"仅 AEU26 在 BNN/TNN 值级出现（投稿前需全文核验并划界引用），无人与位平面区间界、BFP 共享指数、Transformer 门路消费结合。

### 新扫描到的未覆盖工作（2025-26，补录）

- **FlexSpIM**（arXiv:2609.08446，见上表 #6）；
- **NeuroFlex**（arXiv:2609.14092）：元素级 ANN-SNN 无损协同执行调度；
- 28nm Spiking ViT 加速器，双路稀疏核+免 EMA 自注意（TCAS 25）；
- Deformable Spiking Transformer 神经形态加速器（TCAS-II 25.12）；
- **SPAHRTA**（TCAS 26）：稀疏感知 SNN 在线训练加速器；
- **AIGOR**（arXiv:2607.03191）：模块化事件驱动 SNN 推理架构；
- **SpikON**（arXiv:2606.30926）：双并行在线学习 SNN 加速器；
- **UniPRE**（TCAS 25）：SNN-ANN 统一 max-pooling 预测+冗余消除。

### 第四轮（2026-09-15，闭源目录盲区补扫）

检索途径：Crossref REST + OpenAlex（全可用，含摘要）；Semantic Scholar 搜索端点 429（单 DOI 取回可用）；**DBLP 被 Anubis 反爬挡住全部失败**；SPIE/非英语仅一篇纯算法（JEI，非加速器）。JSSC/ISSCC 2025-26 无 bit-serial-ET 芯片；DAC/ICCAD 2026 覆盖仍不完全（仅 Crossref 索引标题）。

新增威胁排序（均需划界引用，非碰撞）：

| # | 工作 | 出处 | 机制 | 与 C1 差分 | 威胁 |
|---|---|---|---|---|---|
| 1 | Lyu/Liu/Xu, "Partial-Sum-Bound Early-Termination Convolution Accelerator" | EITCE 2026, DOI 10.1109/EITCE70137.2026.11634515 | **精确**早终止：通道累加中比较部分和 vs 剩余正贡献上界，可证 pre-activation ≤ 0 则无损跳过剩余 MAC（ReLU 当门） | bound 是 bit-parallel 通道序，非 MSB-first 位平面串行；无 BFP；CNN 域；中国区域会议 | 高 |
| 2 | "Bit-Level Loosely Coupled SNN Accelerator With Fast Inference and Hybrid Early Termination" | TVLSI 2026, DOI 10.1109/TVLSI.2026.3688394 | SNN 输出由输入 bit 逐位生成；混合 ET 跳冗余周期"无损"（LeNet-5 省 28.3%，28nm） | ET 是时间步/收敛级启发式，非逐判决精确区间证书；无位平面贡献字供数、无 BFP、无二值门消费者 | 中高 |
| 3 | DMP-BFP | ICCD 2025, DOI 10.1109/ICCD65941.2025.00012 | 运行时 BFP 精度调整（指数 vs 阈值比较），全精度乘法分解为 4 低精度乘法 | 启发式指数阈值，无早终止/证书，非 bit-serial MSB-first | 中 |
| 4 | AO-BFP | DATE 2026, DOI 10.23919/DATE69613.2026.11539329 | 离群感知自适应混合精度 BFP（LLM），利用诱导 bit 级稀疏 | LLM 域，无 ET/证书 | 中低 |
| 5 | Verifica | A-SSCC 2025, DOI 10.1109/A-SSCC67472.2025.11349404 | 65nm 存内**符号区间计算**加速器（NN 形式验证） | 区间算术硬件但用于验证负载，非跳过推理计算 | 中低 |
| 6 | DIET-PIM | APCCAS 2024, DOI 10.1109/APCCAS62602.2024.10808543 | 运行时重要性 ET 跳过低贡献特征区域（PIM） | 启发式、区域级、模拟 PIM | 低中 |
| 7 | BitL | MICRO 2025, DOI 10.1145/3725843.3756044 | 混合横/纵向 bit 级查表缩短 MSB→LSB bit-serial 关键路径 | 证实 MSB-first 范式；优化数据通路，无终止证书 | 低中 |
| 8 | BSViT | TCSI 2024, DOI 10.1109/TCSI.2024.3426653 | bit-serial ViT 加速器，动态 patch+权重 bit 组量化 | 无 ET、无 BFP 证书 | 低中 |
| 9 | bit-sparse 自适应 bit-serial（TCAD 2025, 10.1109/TCAD.2025.3560584）/ SmartBlock 自适应 BFP（ICPP 2025）/ IEICE ELEX SNN 预测停止（2024） | — | 各占一要素 | 无组合 | 低 |

**判读（对第三轮结论的收窄）**：三要素组合（MSB-first 位平面贡献供数 + 逐判决精确区间证书 + 组级 BFP + 二值门消费者）**仍未被占据**——没有任何工作在 bit-serial 语境下组合其中两个要素。但空白点声明需收窄措辞：EITCE 2026 独立占据了"精确 bound 证书早终止门控激活"（bit-parallel CNN 通道累加），TVLSI 2026 占据了"bit 级顺序生成+保精度 ET 的 SNN 硬件"——**均为必须引用的差分对象而非碰撞**（_operand 顺序不同、无 BFP、中小会议）。

## 检索边界

本轮为 arXiv 定向检索（4 组查询），未覆盖：ISSCC/JSSC/TCAS 2025-2026 闭源目录、DAC/ICCAD 2026 未上 arXiv 的论文、SPIE/非英语文献。上述"摘要级"工作未据摘要宣称适配或杀掉任何方向，全文精读列为各卡前置条件。
