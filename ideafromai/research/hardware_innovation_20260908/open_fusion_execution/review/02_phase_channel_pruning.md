# 立即执行：消费者空间相位 × 完整结构剪枝 × 真实残差误差

2026-09-12，独立审阅并已生成实际 mask；未经训练。**新颖性诊断 6.5/10，适配 8/10，性能未测。** 此分数是对差分清晰度的主观诊断，不是录用概率。

**B与贵部分。** 旧F1在updated I24边界不能删首读：gate与PED各需全部720个64位字，单纯换共同mask标签没有作用。改到真实 `sn2→anchor Conv2_U16` 边界：该分支只在even/even anchor执行，3×3邻域按源坐标产生四个空间相位。一个通道可只在某相位不再被Conv2使用，而在其余相位仍保留。这样可静态省掉该相位的preview V输出与完整T10 sn2生产，同时保留全部原始I24残差。现有ordinary角落局部链中 `preview_V32_BN1 + noncausal_T10_sn2` 为495744/2557022≈19.39%服务线索；不是全层份额。

**A与最近邻。** 完整HiNM/FlexHiNM-GP的区域分配、输入/输出排列和恢复应作强结构基线；DepGraph负责逻辑依赖；Bishop负责时间/空间打包及组稀疏基线。更直接的遗漏近邻是[PerforatedCNNs，NeurIPS2016](https://papers.neurips.cc/paper/6463-perforatedcnns-acceleration-through-elimination-of-redundant-convolutions.pdf)与[SACT，CVPR2017](https://openaccess.thecvf.com/content_cvpr_2017/papers/Figurnov_Spatially_Adaptive_Computation_CVPR_2017_paper.pdf)：空间少算、零填充及按后继膨胀回推生产域已存在，不能把棋盘式少算当首创。[FlexHiNM-GP正式页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9e33fdc35b68781132e836964a326bf3-Abstract-Conference.html)、[DepGraph作者代码](https://github.com/VainF/Torch-Pruning)、[Bishop原文](https://arxiv.org/html/2505.12281v1)。

**本次可编码的X。** 掩码 `D[phase=2*(y%2)+x%2, channel]` 为4×96，每相位删12通道。对even/even anchor，所有kernel位置满足 `phase=2*((ky-1)%2)+(kx-1)%2`，把对应 `U16[:,c,ky,kx]` 同时置零，编译器将该phase的preview V/sn2输出直接标为无消费者。完整非因果T10仍执行保留输出；gate本身只传二值，θ已在U内。真正待证差分是：这种**按消费者步长形成的phase-channel自由度**，比“所有空间同删12通道”保住更多任务信息，并在相同紧凑发射器下少服务。它不是无损剪枝，也不允许删raw I24或别的consumer。

**已经尝试。** [脚本](phase_channel_probe.py)在两学生同一实际帧固定64anchor网格选择mask，再以不重叠的64anchor网格评价。全K864、实际U16/F/PED U/V、每处RNE/sat与残差均重建；未剪基线与捕获的updated/gate/PED全部0差。单个phase-channel按固定 `PED相对平方误差+gate翻转率` 排名，未调权重；选12项后重算组合误差。不是完整HiNM或DuoGPT复现，不是AEE。

| 另格验证 | phase联合误差选择 | 普通全空间删12通道 | 同phase权重幅值 |
|---|---:|---:|---:|
| ordinary，固定score | 0.005362 | 0.006323 | 0.013250 |
| lifting，固定score | 0.004999 | 0.004538 | 0.010362 |

ordinary的PED误差较低，但gate翻转94对86；lifting当前phase输给普通减宽。因此只能说接口和源合同可跑，不能选赢家。

**下一步直接写。** 用[实际mask](phase_channel_masks.json)接现有P2机器：按phase压缩保留H通道列表，避免“读出后丢”；保留84通道的尾部、P2两相位不一致、索引/权重重排/后继scatter全部计费。先比较同mask的全扫与紧凑发射，再比较phase_joint与global_channel12、同phase幅值；另补2:4与完整FlexHiNM/普通perforation控制。现成12/96 mask是试验起点，允许后续一次同预算恢复，不能用当前局部MSE冒充任务精度。若打包税抵消收益，停这一布局，保留phase表示接口；不用等待整层15%才开始这个隔离原型。
