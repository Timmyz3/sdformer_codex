# 文献与备选 idea 档案

日期：2026-09-05。

状态：候选库，未验证，未批准实施。当前优先意见见 [主假说](01_review_and_selective_shared_reduction.md)。

## 1. 为什么保存历史候选但不沿用历史排序

前序讨论曾在 9.20 限制下优先考虑小规模算术实验；取消截止后，曾将无损压缩存储排第一。随后独立评审发现，这些排序还没有真实数据机会和近邻差异支撑，因此撤回其确定性。

以下仍是可探索方向，但“能够移植已有机制”不等于“已经有新论文贡献”。不应把它们全部叠加到 C1/C2，也不应把本文当作多个并行实验的授权。

## 2. 小容量精确模式结果表

来源：[Phi，ISCA 2025](https://arxiv.org/html/2505.10909v1)。

原理：对二值输入 x 和参考模式 p，利用 xW=pW+(x-p)W；残差允许正负修正。Phi 正文明确区分无损分解和后续有损 pattern-aware fine-tuning，因此不能因论文包含微调就将无损部分整体排除。

候选：2/4/8 项局部模式结果表，比较离线固定模式与在线按需构造。表项绑定 weight-tile 身份，残差路径保持精确，模式选择若需要校准则与验证集分离。

风险：查表、结果存储、建表和单口排队可能比直接零跳过更贵。数学分解已有强 prior；较小表本身不构成创新。原文也讨论了模式结果带来的额外权重／结果读取成本。

淘汰条件：完整计入 metadata、建表、读写和 residual 后，不能超过强基线；或者区别仅剩参数变小。

## 3. 按 bank 独立解码的无损权重存储

来源：[EBPC，JETCAS 2019 论文](https://arxiv.org/html/1908.11645v2)，[出版信息](https://cris.unibo.it/handle/11585/728412)，[官方 SystemVerilog 编解码器与测试环境](https://github.com/pulp-platform/stream-ebpc)。

候选：将压缩块设计成可独立随机寻址的小单元，只解码被请求的 bank，随后沿用共享权重广播。对实际 frozen-derived INT8 权重作无损编码，不重新量化或剪枝；不可压缩块使用原始格式回退。

可能的研究问题：大块压缩偏好与小粒度稀疏读取之间的冲突，能否通过编码、布局、寻址和解码器协同解决？

风险：EBPC 的主要对象是 feature maps，其压缩率不能挪到 INT8 权重。原始流式解码器也不是 C2 宽并行接口的即插即用模块。目录、对齐、barrel shifter、临时缓冲、解码吞吐和宏颗数必须全部计入。

文件变小不等于面积变小。固定最大容量不变时，可能只改变访问能量或带宽；静态 checkpoint 压缩镜像是否能减少实际宏颗数，需要独立容量映射。没有相应 SRAM/RF .db 时不得宣布宏级 PPA 闭合。

淘汰条件：实际权重没有足够可压缩性，或解码／过量读取抵消收益，或只是给通用解码器接接口而无实质差异。

## 4. 融合式 signed 归约—累加器

来源：[Parallel Accurate Minifloat MACCs for Neural Network Inference on Versal FPGAs，TCAD 2025，作者全文](https://researchportal.tuni.fi/files/135714845/Parallel_Accurate_Minifloat_MACCs_for_Neural_Network_Inference_on_Versal_FPGAs.pdf)。借鉴符号处理与压缩树融合，不采用其 minifloat 数值格式，也不转用 FPGA 收益。

候选：将 signed 修正融入多操作数归约，严格限定组合中间位宽，最后更新 Acc24；inactive operand 提前隔离，保留原始 overflow、握手和 commit 语义。

对最多八个 ±INT8 有效权重的单拍归约，增量绝对值不超过 1024，12-bit signed 可容纳。此处只讨论中间增量，不缩减 Acc24 状态；-(-128)=+128 必须正确。

风险：综合器可能已经进行等价优化，显式压缩树不一定更好；更短 setup 路径也可能加重 hold。必须比较原 RTL 综合点、显式平衡树和融合结构，不用劣质串行链当唯一对照。

当前定位：可控的配套电路实验，不应仅因易做就成为主要创新。周期不变时只报告对应面积、频率或能量结果。

## 5. 有限窗口内重组独立 token 任务

来源：[Gamma，ASPLOS 2021，作者全文](https://www.guowei.zone/files/2021.gamma.asplos.pdf)，[Avalanche，ISCA 2025](https://doi.org/10.1145/3695053.3730990)。前者包含 affinity 重排和显式存储管理；后者包含矩阵重排、dead-product eviction 和 reuse-distance-aware caching。

候选：从已到达的 8 或 16 个独立 token 描述符中选择共享需求更高的四个组成 B4，最终按原 token ID 写回。这是执行次序变化，不是改变模型的 token 数学顺序。

必须计入窗口填充等待、选择器、描述符容量、尾部不足 B4、输出重排和公平性；不能跨越层边界、神经元时间依赖或 continuation commit 语义。不能借用未到达的未来任务信息。

风险：算法原理已有先例；增加窗口可能损害第一响应延迟，布局改进也可能只改善模拟服务条件而不改善真实硬件。

当前定位：架构备选，尚无实际机会统计。

## 6. 其他已检索工作及采用边界

| 工作 | 可借鉴对象 | 限制 |
|---|---|---|
| [Prosperity，HPCA 2025](https://arxiv.org/abs/2503.03379) / [官方实现](https://github.com/dubcyfor3/Prosperity) | product reuse、baseline 与 DSE | 官方仓库主要提供模拟器及相关代码，不能称为可直接使用的 ASIC RTL |
| [FEATHER，ISCA 2024](https://arxiv.org/abs/2405.13170) / [官方 RTL](https://github.com/maeri-project/FEATHER) | 计算与数据重排结合 | 整套可重构阵列不是局部改造；取消截止也不意味着自动批准重做阵列 |
| [ELSA，ISCA 2026](https://arxiv.org/html/2605.20802v1) / [官方实现](https://github.com/Intelligent-Computing-Research-Group/ELSA) | 成组事件、细粒度流水、mini-batch Gustavson | 不移植其神经元语义，不把局部优化升级为弹性全网络推理 |
| [SpikeX，arXiv:2505.12292](https://arxiv.org/html/2505.12292v1) | 空间／时间权重共享 | 此记录按预印本来源使用，不补造顶会身份；不采用需改变 frozen 模型的协同训练 |
| [Eyeriss v2，JETCAS 2019](https://people.csail.mit.edu/emer/media/papers/2019.04.jetcas.eyeriss_v2.pdf) | 不同复用与带宽需求下的互连 | 广播、可变数据流不是新概念；芯片指标不是我们的 baseline 数字 |
| [Sparseloop，MICRO 2022](https://sparseloop.mit.edu/documents/2022-micro-sparseloop.pdf) / [artifact](https://github.com/Accelergy-Project/micro22-sparseloop-artifact) | 稀疏优化分类与成本建模 | 统计／分析模型不能替代同 workload 的 RTL、宏和物理证据 |
| [UCNN，ISCA 2018](https://www.kartikhegde.net/media/UCNN_ISCA.pdf) | 重复权重与计算复用 | 现有 SNN 单项本已是加法，不能直接挪用减少乘法的收益，也不引入训练约束 |
| [LoAS，MICRO 2024](https://arxiv.org/abs/2407.14073) / [官方实现](https://github.com/RuokaiYin/LoAS) | 时间并行及双稀疏数据流 | 不改变 frozen ATLIF 语义；剪枝、静默化与微调内容不能混入 exact 路线 |
| [DeltaCNN，CVPR 2022 官方实现](https://github.com/facebookresearch/DeltaCNN) | 视频差分计算 | 不是 SNN 精确状态等价的现成证明；阈值抑制和长期状态代价需要另审 |
| [BBS / BitVert，arXiv:2409.05227](https://arxiv.org/abs/2409.05227) | 位级稀疏与编码 | 无需重训练不等于无损；bit-pruning 的精度变化不能移入 frozen exact 贡献 |

仓库链接只表示已找到作者／官方 artifact，不表示逐文件审计、可直接集成、许可已逐项核准或已复现实验。正式使用前须冻结版本、核对许可与引用，不能导入他人的 PPA 数字当作本项目结果。

## 7. 不推荐直接采纳的“idea”

- 再给 group-major order、broadcast、cache hit-before-read 改名。
- 加一张模式表就宣称超越 Phi，或做一次 AND/XOR 就忽略 Comperity。
- 增加队列／缓存后只报告周期，不计容量、端口和能量。
- 换压缩树后以未优化串行加法器作唯一 baseline。
- 以低 firing 推断高复用、低能量或可跳过神经元状态更新。
- 凭当前阈值决定跳过剩余输入，却遗漏未来膜电位、reset 和时间依赖。
- 以无损文件压缩率直接推导 SRAM 宏面积或系统加速。
- 未有新授权即引入 CIM、自定义 SRAM bitcell、整套 NoC、重训练或新模型贡献。

取消 9.20 截止后允许扩大研究深度，不改变“先查重与机会判断，再决定是否实现”的顺序。
