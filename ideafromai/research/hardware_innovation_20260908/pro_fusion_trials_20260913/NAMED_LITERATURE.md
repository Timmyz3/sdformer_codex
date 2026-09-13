# 点名论文、已借范围与融合状态

2026-09-13。本页合并本轮与此前已完成的定向阅读，不把复用旧全文再计算为“新读一篇”，也不把独立小叶称为原论文全系统复现。完整来源/章节及全文缺口见 [补缺报告](literature/REPORT.md)、[来源表](literature/source_master.csv) 和 [前轮去重文献表](../deep_target_research_20260913/LITERATURE.md)。

| 用户点名 | 核定会议/期刊与原作 | 应完整借入的 A | 现在真正做到什么；为何纳入或暂缓 |
|---|---|---|---|
| Prosperity | HPCA 2025；[作者论文](https://arxiv.org/abs/2503.03379)、[官方代码](https://github.com/dubcyfor3/Prosperity) | ProSparsity 检测/父依赖、有限 tile 执行、局部部分和、外层 loop/供数、调度与双缓冲；不是只有父行公式 | 此前已跑完整捕获算子的官方 CPU 外层与真实数值对照；本轮又在 RTL 保留两源部分和强控制。继续作为强 A；本轮配对 primitive 不冒充完整 PR，也不再把这份旧代数称 X。 |
| Phi | ISCA 2025；[作者全文](https://arxiv.org/abs/2505.10909)、[正式日程](https://www.iscaconf.org/isca2025/program/) | 按 K16 分块的 128 模式字典、聚类/校准与恢复训练、L1 pattern-weight products 及选择性预取、L2 有符号修正；多窗口 packer 同时检查容量和 psum bank，必要时排空最满窗口；8 输入可配置多行归约 | 方法和 §4 数据路已读。以前 Phi 融合的负结果不能算完整 Phi 实现失败。当前没有移植完整双层引擎及配套训练；保留为必须补齐的 A。其冲突检测/flush/归约已直接覆盖很多拟议 X，不能只摘“模式＋残差”后改名。 |
| FireFly-T | [arXiv 2505.12771](https://arxiv.org/abs/2505.12771)；本轮未核实正式期刊卷期，暂不填猜测的刊物 | 多 lane spike 解码、bitmap 消费追踪、稀疏引擎多维并行与供数解耦；二值引擎的布局变换/attention，以及 FPGA LUT6 算术 | 已读 §III–IV 架构和 sparse decoder/load balancing。多脉冲解码、受限队列、借空闲执行位置都应归入 A；目前没有完整迁入该双引擎，LUT6 优化不作为 28nm CMOS 新点。注意力叶只作旁路候选。 |
| Bishop | ISCA 2025；[作者全文](https://arxiv.org/abs/2505.12281) | token/time bundling、训练形成组稀疏、ECP、stratifier 保留原位置与权重对齐、稀疏/稠密核分工 | 前轮全文已读，本轮作为共同活动与剪枝的强对照。现有局部分组/8 槽 RTL 不等于完整 Bishop；仍可融合真实物理词消费目标，但“时间打包＋稀疏训练”不能单独作 X。 |
| ELSA | **两个不同工作**：ISCA 2021 [self-attention](https://taejunham.github.io/data/elsa_isca21.pdf)；ISCA 2026 [elastic SNN](https://arxiv.org/abs/2605.20802) | 2026 版：原生 im2col、BAER、mini-batch Gustavson、局部目的归约、队列/背压/完成流水；2021 版是 hash/Hamming attention 筛选 | 本轮新读 2026 正文及[作者模拟器](https://github.com/Intelligent-Computing-Research-Group/ELSA)。有限目的队列满后 drain/retry 已在代码；其已读路径不等于本地真实 bank 逐周期仲裁。优先补完整 A，不能把当前 pair RTL 宣称为完整迁移。ST-BIF 早发放不直接搬到非因果 T10 PSN。 |
| UNICORN | DAC 2022；[DOI 10.1145/3489517.3530563](https://doi.org/10.1145/3489517.3530563) | flexible fan-in/fan-out、spike train multicast、neuron 合并及跨核映射；精确格式/控制仍缺主文 | 已核题名、venue、引文链，原作全文未取得。保留而非否决；先补全文才能声称“借全”。不与同名 emulator/CIM/压缩项目合并。 |
| STELLAR | HPCA 2024；[IEEE 原作](https://ieeexplore.ieee.org/document/10476421) | Few-Spikes 算法/训练、窗并行、时空表示、存储/累加/发放合同 | 本轮取得正式身份和作者单位说明，主文仍未取得。时间码与本地 θg/PSN 不能只凭摘要互换；待全文补齐，不是性能失败。 |
| Dynamic Joint Pruning | DAC 2025；[IEEE 原作](https://ieeexplore.ieee.org/document/11132570) | 权重与时空发放联合剪枝、阈值 mask、TABN 时间缩放、动态重要性及完整训练日程 | 摘要已核，主文/代码尚缺。联合剪枝继续保留；普通 firing penalty 不冒充原版 DJP。后续 X 应比较真实物理词/回放费用与原 SOP 目标，而不把 SOP 减少当硬件能量。 |

## Pro 新建议：不是全杀，也不是全做完

| 建议 | 本轮尝试程度 | 去留边界与下一接口 |
|---|---|---|
| 连续 PED Kronecker＋局部中间量 | **实际拟合＋RTL**：真实 V96×24，Kron1/2、同函数展开直接、同参数 LR 数值控制 | 本地周期下降，免训近似误差大；算子名义份额约0.074%。保留可运行 A，不升主标题。算法恢复已授权，但当前优先大算子。 |
| T10 时间区间端点＋prefix | **实际统计＋RTL**：原 T、完整 K/N16、prefix/RNE/输出 | 当前几乎全长度1，变慢；停止这个原 T 布局。运动对齐后的真实跨帧接口没试，不按本结果杀掉。 |
| 共同活动低冲突打包 | **实际 RTL**：3配对×3模式，在 U16 与 r0 N96 上分别运行 | 本布局减少冲突却增加权重词，两源部分和更强。停止仅最小化共同活动的目标；原生词/消费者约束与有损训练仍未试。 |
| 原型 LUT＋残差 | 原作方法已读，**本轮没写 LUT/encoder RTL** | LUT-DLA/Phi 已覆盖基础；须给 prototype 查找和向量表带宽，不是被证明无效。连续小 V 暂低优先级。 |
| 运动对齐＋有限相位缓存 | 任务与先验已读，**本轮没运行运动参考、相位缓存 RTL** | MotionDeltaCNN/FluxShard 直接近邻；应使用前窗/已完成粗层参考，计 halo/相位缺失。不能借本次原 T 负结果否决。 |
| 稠密粗流＋稀疏细节 | 已有模型挂点核查，**新 mask/decoder 执行本轮未做** | 现有粗流头可作因果使能；WaveletVFI/CSSL 是强近邻。应先在真实 decoder2 输入依赖范围测试质量与工作量。 |
| stride-2 相位供数/剪枝 | 解析几何，**本轮未新增完整相位 RTL** | 1/2/2/4 fanout 和 polyphase 分解本身属于 A；不能把既有 anchor 输出删除重复算收益。 |

## 真正需要升级的基线

新增 FIFO、有限 replay、局部归约、两源共享本身不足以成为 X：本轮已经能在 Phi、ELSA、TensorDash 中指出直接机制来源。下一次应先给基线同样的这些权限，再比较表示或训练改变了哪些物理事务。

这并不要求等所有论文全文和整网 PPA 都齐了才动手。本轮已经先实现三个可运行执行器，并在贵的 r0 上补了真实输入测试。它要求的是：**把试出来的收益分清属于 A 的移植，还是 A 也做不到的 X**；未运行项保持“待试”，负结果只限定实际布局。
