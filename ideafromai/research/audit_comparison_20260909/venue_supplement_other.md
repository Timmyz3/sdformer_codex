# 薄弱电路会刊补检（2026-09-09）

对旧 358 条总表去重，新增主表 8 篇：6 篇取得一手方法或摘要，另 2 篇只核实题名与身份。下面明确分开阅读深度；没有正文不判机制失败。本轮没有实验，也没有把论文原指标写成本地性能。

优先价值在数字供数/索引与多阶段本地状态。HIRB、DNC、SIMO SPad 比单列更多 CIM 效率数字更值得完整承接；ALSCA 与压缩引擎是直接相关的缺读项。

## 1. HIRB sparse CNN — IEEE Symposium on VLSI Technology and Circuits 2022

**A Sparse Convolution Neural Network Accelerator for 3D/4D Point-Cloud Image Recognition on Low Power Mobile Device with Hopping-Index Rule Book for Efficient Coordinate Management**

来源：[一手原文/机构页面](https://users.eecs.northwestern.edu/~jgu/nu-vlsi/SCNN_VLSI2022.pdf)。作者全文及官方 C12-1 节目核实。阅读范围：两页作者全文：算法/硬件实现、Fig.2–4、测量结果均已读。

原机制：把稀疏坐标管理完整纳入芯片：octree 与部分距离跳过生成 HIRB；输入记录 end 指针，遍历目标与核索引；全通道共享核索引，经 LUT 取权，scatter 到目标累加。坐标管理和卷积复用同一可配置 PE 阵列。 65 nm 实测；原文坐标管理占运行时间从 67.5% 降至 14.7%。13.5–26.9× 权重存储压缩针对原文复制格式，不是相对现代普通权重驻留的优势。

挂点与剩余问题：patch 源事件/支持描述符到 3×3 目标的索引生产；也可作未来原生稀疏事件 stem 的完整底座。 当前规则 2D 栅格的邻居可由固定偏移生成，octree 未必值得；需解决跨 T10 消费者共享同一空间关系与目标状态释放，不能将普通核索引复用当新点。

必须保留的控制：固定 3×3 偏移生成器＋普通行缓冲/GP NRV；若改成稀疏坐标网络，再与完整 HIRB 的坐标生成和 scatter 成本同计。 尚未尝试：本地未迁移 HIRB 生成、端指针遍历及有限累加地址冲突；未训练点/体素稀疏 stem。

判断：纳入：完整索引底座及强对照，暂不直接移植 octree。

## 2. DNC near-memory — ESSCIRC 2022

**A Differentiable Neural Computer for Logic Reasoning with Scalable Near-Memory Computing and Sparsity Based Enhancement**

来源：[一手原文/机构页面](https://users.eecs.northwestern.edu/~jgu/nu-vlsi/DNC_ESSCIRC2022.pdf)。作者全文及出版社身份核实。阅读范围：作者全文 §II–V，尤其 Fig.3–7 和 §IV 已读。

原机制：八个本地存储 tile、每 tile 八 MAC；重配累加方向处理 similarity 与转置 recall 的映射冲突；同 PE 做多精度 MAC/逐元素乘法/加减。零输入译码同时免取权与 MAC；读写 head 另有阈值剪枝后总线压缩。 65 nm 芯片；原文报告多阶段平均 PE 利用率超过 90%，稀疏增强端到端约 30% speedup。head 阈值压缩和混合精度有准确率损失，不能标成精确跳零。

挂点与剩余问题：Conv/FC 生产与短 T10 矩阵消费共享算术、本地状态；3 bit 类别输入直接译码也已有其 one-hot 输入近邻。 原作是有状态 DNC，而非固定 T10 非因果神经元；本地必须补有限 Y/U 生命周期和消费者完成屏障，但普通本地驻留/转向累加本身已被覆盖。

必须保留的控制：同总 SRAM、端口、算术量下的可重配本地阵列；当前 one-shot32、完整十 Y 及 Gustav 多 W 行驻留共同对照。 尚未尝试：未复现原多阶段控制/转置累加；尚未比较我们的阶段切换和短矩阵利用率是否需要专用机制。

判断：纳入：多阶段本地执行强底座。

## 3. SIMO SPad — IEEE Transactions on Circuits and Systems II: Express Briefs 2023

**A Tiny Accelerator for Mixed-Bit Sparse CNN Based on Efficient Fetch Method of SIMO SPad**

来源：[一手原文/机构页面](https://repository.hkust.edu.hk/ir/Record/1783.1-126068)。作者所在机构知识库核实，70(8):3079–3083。阅读范围：一手机构摘要及元数据；未取得方法正文。

原机制：摘要明确 single-vector compressed sparse-filter 表示和 single-input multiple-output scratchpad：按稀疏权重取所需 activation，多 PE 共享 SPad。完整寻址/冲突处理正文未读，不能补想象实现。 摘要报告相对其实现 13.34% CLB LUT 与 46.24% CLB register 节省；混合位宽＋稀疏对 8 bit 稠密 VGG 的倍率不是单独供数电路增量。

挂点与剩余问题：FC1 多 H 行共同源 bank、3 bit source 多次重放，尤其剪一半 W 而物理源字读取未降的矛盾。 需要把共享 fetch 延伸到非因果 T10 状态完成及实际 64 bit 源字，不能只给新路线共享 SPad 而给基线 PE 私有重复读取。

必须保留的控制：完整 SIMO/CSF fetch＋普通 NR4 归约＋F_cache 驻留；同物理字、端口和剪枝精度。 尚未尝试：方法全文、CSF 生成与编码成本、SIMO bank 冲突和背压完整移植均未完成。

判断：纳入：优先补正文的直接供数先验。

## 4. FP4/FP8 shift-add training processor — ESSCIRC 2023

**A 4.27TFLOPS/W FP4/FP8 Hybrid-Precision Neural Network Training Processor Using Shift-Add MAC and Reconfigurable PE Array**

来源：[一手原文/机构页面](https://pure.skku.edu/en/publications/a-427tflopsw-fp4fp8-hybrid-precision-neural-network-training-proc/)。作者所在机构页面核实，pp.221–224。阅读范围：一手机构摘要及出版元数据；方法正文未读。

原机制：训练 GEMM 用 shift-add MAC；可配置 PE 阵列减少片上访问；片上 convolution decomposition 用统一路由支持多核尺寸。未取得正文，不假定其 exponent、舍入或重配置时序。 40 nm；摘要报告实际 ResNet-18 训练 2.61 TFLOPS/W；题名 4.27 是另一效率点，不混用。

挂点与剩余问题：已有短时间矩阵 INT 系数/CSD 消费与可配置共享加法链；也可为重新训练时的表示提供强底座。 当前主要张力是小矩阵状态/供数而非一般乘法器；换成二幂系数本身已是强先验，需要说明连续 θ、τ 和完整 T10 的表示/释放为何改变执行。

必须保留的控制：同资源 CSD 常量乘法、二幂量化和普通共享加法链；精度按新学生单列。 尚未尝试：完整 FP4/FP8 训练、卷积分解前后处理与本地寄存器布局未移植。

判断：纳入：位域算术及重配置强对照。

## 5. D3TA — ESSERC 2025

**D3TA: 38.9 TOPS/W Transformer Accelerator with Dual-Port 3T-eDRAM Digital Compute-In-Memory using HyperAttention and Triple-Sparsity-Handling**

来源：[一手原文/机构页面](https://www.esserc2025.org/_files/ugd/aa54ce_3ae2d7986b2f43d7bd4a3f3c9cf366f1.pdf)。官方会议四页论文核实；本轮未定位 DOI。阅读范围：官方原文 §I、总体架构、§II-A/HyperAttention 及 Fig.1–3；后段读取超时，未标全文精读。

原机制：head 级 Q/K/V 生成与行组 S/P/O 计算配流水，片上 attention buffer、互连和向量单元负责保留与交换；另设 activation/weight/output 三类稀疏 bitmask 与数字双口 eDRAM CIM。 摘要报告其 BERT 情形 2.6× latency 改善、2× 外存流量降低；不是本网络实测。三类 near-zero 策略是否精确及门限细节本轮未完成正文核验。

挂点与剩余问题：attention/FFN 跨算子片上保留，以及生产-消费流水与稀疏元数据同管理。 H67 K 当 V、Motion-XOR 和非因果 T10 均不能照搬其 QKV/softmax；可借外围流水和 metadata，而非假定有本地 3T 宏。

必须保留的控制：普通片上分块 attention/FFN 融合、双方同样的 fill overlap/向量单元/bitmask 成本；物理宏另界。 尚未尝试：全文后段、完整三稀疏判据、互连背压及 H67 适配未试；未采用定制宏。

判断：纳入：外围流水/掩码组织；宏不进入当前 PPA。

## 6. Winograd-Standard Fusion — IEEE Asian Solid-State Circuits Conference (A-SSCC) 2025

**A 28nm 244.45TOPS/W Winograd-Standard Fusion Accelerator with Symmetric Hybrid Domain CIM Groups for Edge AI Devices**

来源：[一手原文/机构页面](https://mn.cs.tsinghua.edu.cn/xinwang/PDF/papers/2025_A%2028nm%20244.45TOPSW%20Winograd-Standard%20Fusion%20Accelerator%20with%20Symmetric%20Hybrid%20Domain%20CIM%20Groups%20for%20Edge%20AI%20Devices.pdf)。作者三页稿标 A-SSCC 2025；Session/Paper 仍为占位，最终编号/DOI 本轮未核。阅读范围：作者原文方法、Fig.2–5 相关文字和结果已读；不是只读题名。

原机制：固定 F(2,3)，stride2 转普通卷积模式；离线变权、输入变换左右不同并行方式及 carry 预启动，输出变换共享中间项并提前部分计算，以融合前后处理降低存储。内核为混合域 CIM。 原文给出的前/内/后处理功耗份额约 7.3/20.5/72.2%，说明不能用 2.25× 乘法减少当整体优势。题名效率不迁成本地数字 PPA。

挂点与剩余问题：最贵 patch Conv3×3，以及时间变换前移导致源幅值连续化时的真实前后处理费用。 θg 加法型输入经变换会变成多位连续源，可能失去稀疏优势；必须同时承接普通模式和全部前后变换，不可只搬 Winograd 乘法数。

必须保留的控制：完整数字 Winograd 前/内/后处理，对同输入的直接 GP 稀疏卷积；共享 W、状态和精度成本同计。 尚未尝试：未训练适合 Winograd/时间变换的稀疏表示；未实现数字前后处理或使用论文定制 CIM。

判断：纳入：完整变换代价及复用手法，暂不选为新主线。

## 7. ALSCA — IEEE Transactions on Circuits and Systems II: Express Briefs 2024

**ALSCA: A Large-Scale Sparse CNN Accelerator Using Position-First Dataflow and Input Channel Merging Approach**

来源：[一手原文/机构页面](https://gr.xjtu.edu.cn/web/chyang00)。作者主页题名/刊物/年份核实。阅读范围：仅一手作者目录身份及题名；本轮未取得方法摘要/正文。

原机制：题名明确 position-first dataflow 和 input-channel merging；合并规则、硬件实现及是否数值合并尚未核，不能把题名扩写成已读机制。 不录入未核的一手性能数字。

挂点与剩余问题：P4/H8 供数、源列归并与有限多 F 驻留。 要先弄清原作如何归并和分配，才能判断是否覆盖当前公共源约束；目前只能列直接相关缺读，不能判无创新或直接可用。

必须保留的控制：完整原作＋普通 channel reorder/packing＋Gustav NR4/F_cache；不得用未实现当概念负分。 尚未尝试：原文方法尚待取得；分组/地址生产/合并后输出恢复均未试。

判断：保留为优先补全文项，尚不能晋级机制。

## 8. Broad-Spectrum Compression Engine — IEEE Transactions on Circuits and Systems II: Express Briefs 2024

**A Broad-Spectrum and High-Throughput Compression Engine for Neural Network Processors**

来源：[一手原文/机构页面](https://lvdongxu.github.io/)。作者主页及其出版社链接核实。阅读范围：仅一手作者目录；出版社页面拒绝抓取，未读方法正文/一手摘要。

原机制：题名指向跨分布的高吞吐压缩引擎；本轮不采用二手摘要中的自适应参数/多 lane 细节作为已核原理。 不录入未核的一手吞吐/面积数字。

挂点与剩余问题：3 bit 源码多次重放、θg 支持压缩以及稀疏索引带宽。 压缩存储不等于在压缩域直接消费；需核其解码并行度及随机取词/背压后才能判断能否代替 source bank。

必须保留的控制：EBPC/普通位图或 RLE、原始紧凑 3 bit 驻留、同 port 的解压再消费。 尚未尝试：完整压缩器正文、真实 code 分布与吞吐/端口匹配均未试。

判断：保留为优先补全文项，不冒充已读压缩创新。

## 检索覆盖与缺口

| 刊会 | 本轮检索年份 | 实际范围与缺口 |
|---|---|---|
| ESSCIRC/ESSERC | 2022–2026 | 2022/2023 关键词＋作者全文；2024 作者目录；2025 官方论文；2026 官方会议程序/工作坊检索。不是逐篇通读五年 proceedings。2026 程序页抓取受限，工作坊是交流活动，不充正式论文；本轮未纳入新 2026 正式篇目。 |
| A-SSCC | 2022–2026 | 逐年定向数字稀疏/SNN/卷积关键词；2024 官方 advanced program；2025 作者全文。2022 旧 Spike-CIM 已在总表；2023 CNN/SNN 超分 FPGA 仅身份线索；2026 本轮未定位可读目录，不等于无相关论文。 |
| VLSI Symposium（不含同名 TVLSI 期刊） | 2022–2026 | 2022 官方分日程序及作者全文；2023–2026 定向题名/作者目录/出版社检索。2024 FSNAP 正文未得；2025 NeuC-CIM 只读出版社摘要且属定制电荷域宏；当前首页已转下一届，未完成 2026 档案通读。 |
| TCAS-II | 2022–2026 | 逐年 sparse/spiking/compression/shift-add/neural accelerator 关键词及作者目录核验。2022/2025 未选到本輪可核且优先的新方法，不作无相关结论；2026 PredLM 仅作者索引身份，正文未得。旧 OPN 全名已在总表，未重复计数。 |

上述为逐年定向检索，并非逐篇通读所有 proceedings；没有纳入某一年不表示该年没有相关论文。ESSERC 2026 工作坊不充正式研究论文，TVLSI 期刊也不充 VLSI Symposium。

## 保留但未充主表的线索

- [FSNAP](https://ieeexplore.ieee.org/document/10631414)（VLSI Symposium 2024）：时间窗自适应直接相关，但本轮只核身份；不把二手摘要当完整机制，不和 JSSC 扩展重复计数。
- [PredLM](https://sangwoohong.github.io/research/)（TCAS-II 2026）：作者搜索索引列出题名，抓取正文未出现该条且原文未得；刊期细节待出版社确认。它可能是预测后免 W 请求的强近邻，不能凭旧 2021/2024 文献就断言覆盖充分。
- [SENNA](https://digitalcollection.zhaw.ch/items/a87eb225-0440-442f-9f24-fad108d33e99)（ESSERC 2025）：已读一手机构摘要：电路交换混合信号 SNN，确定延迟的连接组织可借；完整非因果 T10 与数字存储不直接适配，本轮不以宏数字凑主表。
- [NeuC-CIM](https://ieeexplore.ieee.org/document/11074927/)（VLSI Symposium 2025）：出版社摘要核验；电荷域/校准宏未移植，不因工艺不采用就否定其状态/事件触发机制。
- [heterogeneous CNN/SNN super-resolution](https://yonsei.elsevierpure.com/en/publications/a-resource-efficient-super-resolution-fpga-processor-with-heterog/)（A-SSCC 2023）：一手机构元数据；dense output 混合网络相关，未取得正文，不能说已读其上采样/核分工。
- conditional-computing CNN（JSSC 2021, 56(3):803–813）：根节点提供的直接光流芯片引用；超出本轮 2022–2026 和指定刊会，交根节点补全文/身份，不假装本轮已读。
- C-DNN（ISSCC 2023）：根节点新增线索，heterogeneous CNN/SNN 与 forward gradient sparsity；不在本轮指定会刊，完整题名/原文交根节点核验。

当前不能再称会刊覆盖完整。实际缺口是上述相关篇目的完整方法和强控制尚未移植，而不是欠缺更多名称。SIMO/ALSCA 应先核能否减少真实源字读取；DNC/HIRB 应核本地规则地址与有限状态的更便宜控制；这些普通原作行为迁齐之后，剩余差别才可作为本网络创新论据。
