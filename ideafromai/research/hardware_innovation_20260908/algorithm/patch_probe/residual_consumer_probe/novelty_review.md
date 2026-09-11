# 残差后真实神经元的默认膜复用：独立概念审阅

2026-09-09；只读源码和主文，未训练或运行网络。**值得一次消费者机会检查；目前概念差异 5.5/10、该 X 的本地证据 2/10，均非录用概率。** R32 的完整 825 通过证明底座可用，尚不证明膜绑定、许可或 Conv2 净节省。

**B 与最强控制。** 普通 R32 的 FP/QDQ AEE 为 1.20109467/1.20304306，见 `../factor_completion_20260909/network_preview_only_valid825/`。**运行时修正：实际 proj 为 `SpikingPEDLayer`，不是此前据候选类假定的 `SpikingEmbeddingLayer`。** `Spiking_modules.py:817` 先执行 `conv_res(r1out)`（1×1、stride2、padding0），再执行 `sn(r1out)→conv→norm`，最后两路相加。故 r1 的 `Conv2→BN2＋identity` 有连续、门判决两种消费者；偶数行/偶数列 1/4 位置仍需连续值，其余位置才可能仅为 proj.sn 判决而生产。旧 Q16 终点没有测过这个边界。Conv2 有 96×96×3×3 权重，R32 下可取消的物理请求仍待测。强控制必须给完整稀疏 Conv2/残差融合、重算 `A_proj(identity)` 后相同界、同绑定矩阵普通 CSE＋逐门完成、SkipNet/BlockDrop 和直接删分支的精度控制；奇偶分块、H8/P4、T 广播、源零跳过及缓存同样授予普通编译器。

**A 必须完整继承，而非记作新意。**

| 已读主文与方法位置 | 已覆盖；没有据此自动否定的具体问题 |
|---|---|
| [Spike-driven Transformer，NeurIPS 2023](https://arxiv.org/html/2307.01694v1)，§3.1–3.2、式7–11 | 已有连续 membrane shortcut、残差后发放及保持后继脉冲运算。文中的膜 shortcut 不等于本实现 `A_sn1(x)` 的内部膜；原文没有展示绑定两套非因果 T10 算子、保存前者作为后者判决预览并取消 Conv2 的方法。 |
| [SkipNet，ECCV 2018](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Xin_Wang_SkipNet_Learning_Dynamic_ECCV_2018_paper.pdf)，§3–3.3；[BlockDrop，CVPR 2018](https://www.cs.utexas.edu/~grauman/papers/blockdrop_cvpr2018.pdf)，§3.1–3.3 | 已有前层输入驱动残差旁路、门控成本、策略训练/联合恢复；SkipNet 的 recurrent gate 也复用跨阶段计算。它们是有损路由，不提供当前学生逐个 θg 的严格证书。仅把 block 改成 H8 或称“复用预览”不够。 |
| [CompRRAE](https://arxiv.org/html/1906.03180v1)，§IV-A–C；[DPES，IEICE ELEX 2024](https://globals.ieice.org/en_publications/elex/10.1587/elex.21.20240206/_f)，§3.1–3.4 | 已有剩余范围/统计预测、比较反馈、停止累加及控制/访存费用；DPES 还有融合层和行广播。不能把界公式、负膜预测或融合后反馈当新增。DPES 的加权 phase 输入与本地非因果 T10 不同，其负号耐心计数不是严格证书。 |

待补全文的强对照：[ShortcutFusion](https://ieeexplore.ieee.org/document/9729106/) 官方摘要涉及 shortcut 复用感知存储；[Activation-deterministic early termination](https://www.sciencedirect.com/science/article/abs/pii/S1434841126003547) 摘要涉及二/三值激活及池化确定性提前结束。本轮没有取得二者全文，不能据摘要宣称完整迁移或精确同构；后续分别补存储、判决基线。

**X 的精确范围。** 设固定 BN2 为 `α₂·Conv2(θ₂g₂)+β₂`，下游判决裕量为

`m = m₀+r；m₀=Aₚx+Aₚβ₂+bₚ−centerₚ−τₚ；r=Aₚ[α₂ Conv2(θ₂g₂)]`。

sn1 已算 `u₁=A₁x+b₁−center₁`。只有联合训练满足 `Aₚ=M A₁` 且 M 确实便宜，才能由 u₁ 修正得到 m₀；一般的 `AₚA₁⁻¹` 仍是稠密矩阵，不能称免费复用。两处 θg 幅值均保留，τ 是判决边界；bias、center 和可能为负的 BN gain 都要入账。严格残差界 `[l,u]` 仅在 `m₀+l≥0` 或 `m₀+u<0` 时许可。sn1 门位不足：裕量 0.1 与 10 都发放，残差 −0.2 只翻转前者。

可保留的假说是：**把前面必算的时间投影约束成真实后继可直接消费的默认膜，省新预览的生产，并在没有连续消费者的位置撤销 Conv2 请求。** 本次主文未直接展示这一组合；但共享矩阵/CSE 和提前完成本身均已知。proj.sn 完成不能取消连续投影锚点，锚点也可能钉住共享 W 字。若同函数普通 CSE＋区间执行已取得相同净费用，就只剩常规权重共享，不能用“跨层”命名补创新。

**唯一最低费用判别与固定停门。** 对共同 R32 捕获一次 x、sn1 内部膜、sn2 门和 proj.sn 真实输出：先收费重算 m₀，用仅已知 sn2 支持/静态 W2 产生的一个固定廉价界，比较普通逐门和实际 P4/H8/T10 许可的 Conv2 请求；完整残差只作答案核验。再列同一许可下“保存 u₁＋廉价 M”替代重算的净成本，不能用最终一致率挑跳过位置。单个 P4/H96/T10 的 FP32 u₁ 就是 **15,360 B**，H8 为 **1,280 B**；identity 仍按真实生命周期保存，H8 分块不得同时免费获得 H96 广播。dense Aₚ 下一个未决 t 可保住全部 T10 生产，必须经真实依赖与 W/source 合并统计。若连重算控制都不能释放有效请求，或最有利复用收益仍覆盖不了膜读写、界、检查与广播损失，停止此具体 X；有正余量才训练一个固定绑定约束并与同训练预算普通共享矩阵/旁路比较 AEE，不扫门限。

## 补评：按真实消费者固定空间精度／秩

**B 与控制先行。** 新方向令全域产生便宜 Conv2 前缀，只有 even/even 连续锚点补尾；非锚点以此前缀经过原 BN、非因果 T10 PSN 产生 θg。这是另一个有损学生，保留 A/θ 不代表与父函数等价。diverse10：R32 为 1.12758346；非锚点 raw Conv2 置零、保留 BN2(0) 为 1.22019193；非锚点整个 BN 分支删除为 1.19536523。**最新完整825已区分这两种控制**：[整个BN分支删除](branch_control_valid825/nonanchor_norm2_branch_zero_summary.json) AEE **1.219633715**，通过1.259预算；[raw Conv2置零保BN2(0)](branch_control_valid825/nonanchor_conv2_raw_zero_summary.json) 为 **1.262666788**，未过。两者均为825帧／48,152,523有效像素，不能把门通过结果互相替换。普通空间删除已成为有效强底座，保留其成本／精度点，不因它有精度损失就排除。

**A 的直接近邻。** [Precision Gating，ICLR 2020，§3.1–3.4](https://arxiv.org/html/2002.07136v2) 已有低精度前缀、按输出选择补算、复用前缀及联合训练；[CADyQ，ECCV 2022，§3.3–3.4](https://arxiv.org/html/2207.10345v1) 已有空间／层精度选择、费用正则和蒸馏；[CFMP，ISSCC 2025，Fig.23.2.5](https://arxiv.org/pdf/2512.17555) 已有因子分解、空间 tile 中间掩码、两级稀疏取数与稠密恢复。CFMP 的不同 tile 保留不同因子数量可作本地不同有效秩的强迁移控制，这是方法推导，原文没有使用本模型的锚点语义。完整借入这些以后，静态 anchor mask、粗细相加或两类损失本身不能单列新机制。

**尚可检验的 X。** 本次原文没有直接展示由连续／仅判决消费者共同约束的表示与供数；该接口确实省掉旧动态逐门许可、表和 OR 的依赖。不过当前描述仍主要是给已知空间精度方案一个更具体的分配依据，概念差异暂 **5/10**，新增硬件收益证据 **2/10**，非录用率。特别是输入为静态幅值 θg：合法折权后的 g 已是一位，不能原样引用 PG 的激活高／低位节省；须明确改变有效权重或连续因子的表示。

**唯一下一比较。** 固定一个秩／位宽预算，对照普通 uniform rank16/rank32、nonanchor rank16＋anchor 原 W2，以及同 anchor mask 的完整 CFMP／精度分解；共同训练、真实双分支与 T10 消费者均保留。普通方案不必先算 anchor 前缀再算稠密 W2−P，若尾无更便宜结构，这种重复反而是候选的负担。只有前缀／尾表示在同精度下减少实际共享供数、宽状态或能量，才有结构共设增量；锚点尾可直接编成 stride2 卷积，奇偶调度双方共有。1/4 输出位置仍可能覆盖全部 3×3 输入 halo，不能推导出 3/4 源读或 W 流量减少。若该次比较仅得到普通混合秩收益，就归入强底座，停止给它另起机制标题。

## 补评：门专用近似，连续分支独立收缩

这次结构与上节不同：包括 anchor 在内的 proj.sn 全部用 rank16 门专用卷积；连续投影仍来自原 W2，两个分支都从真实 θg 出发。令 `z=θg`、`BN₂(v)=α⊙v+β`、`Conv2(z)=W₂*z+b₂`、`conv_res(v)=C·v+bᵣ`，则连续支路为 `C·x|anchor + K*z|stride2 + d`，其中 `K[o,c,k]=Σh C[o,h]α[h]W₂[h,c,k]`，`d=C(α⊙b₂+β)+bᵣ`。末级 1×1 没有空间扩展，合成核仍是 3×3、stride2、padding1；偏置必须完整保留。这是实数线性等式，FP 重排和重新定点不保证逐位一致。门支路改变后的整体是新学生。

**先验边界。** PG 的粗／细共享、CADyQ 的空间精度选择、CFMP 的双因子与物理掩码仍全部作为 A；它们没有在已读方法中给出这里“同一残差值的门／连续双消费者分离”。新增最贴的 [Collapsible Linear Blocks，MLSys 2022，§3.1、Fig.2、Algorithm 1–2](https://proceedings.mlsys.org/paper_files/paper/2022/file/3134f61af2136e249b0d8f190cbdc508-Paper.pdf) 已明确将 k×k→1×1 线性链及同输入残差折成一个核。因此 K 的形成、BN 折入、少存中间图均不能独占归因。可保留的研究句仅是：**训练一个门专用近似生产者，使连续分支不再受门消费者钉住，并考察分离后的两种表示能否在有限资源下比普通共享生产更便宜。** 目前概念差异 **5.5/10（判断区间 5–6）**、新增性能证据 **2/10**，并非已成立的电路贡献。

**最强反对是实际费用，不是“通用图能表达”。** 原来只算一次 `C(x+Y)`，现在仍须算一次 `Cx`，两者同为 96×96 连续投影，不能把前者记成净省 MAC。若 `Cx` 没有别处必算的可复用结果，不能用稀疏 K 取代稠密 C 的局部对比声称优势。对照 nonanchor rank16＋anchor 原 W2 在 anchor 用一份 W2 结果兼供门与连续支路；候选在同处新增 rank16 门卷积，并把 W2 换为同形状、可能更宽且更密的 K。故仅 Conv2／投影子账未显示必然优势。低秩门表示或可进一步减轻 T10 混合／状态，但合法的 A 前移、同函数 CSE、直接流式投影和奇偶分块必须同授普通轴，不能假定原轴一定写回完整幅值图。

**固定判别。** 保留简单删除、uniform rank16、nonanchor rank16＋anchor W2；另给全原 W2 门支路＋普通 K 收缩，以及“同一个 rank16 门学生＋未收缩原连续支路”，分别隔离近似和收缩贡献。只在一个固定 rank16 下核真实双分支 AEE与完整请求／状态／时间混合费用，包含 K 的系数宽度和取数、连续 Cx、低秩连续 MAC、halo。若收益全部等于普通门压缩加线性收缩，归入底座；若分离确实解除共享生产的物理限制，且净收益超过同权限轴，才值得继续联合表示／供数实现。当前不为代数恒等式开 RTL。

**状态问题仍留一个窄口。** rank16 门分支可能以 `T10×R16` 的 Z 驻留，连续 K 分支独立按 t 完成，不再共享原 anchor 的全幅 Y 生命周期；这是有限状态模型待判的问题，不能据此预言节省。普通轴必须获得 H8 分块、逐源时间 s 更新各输出 t、合法 A/V 换序和流式连续消费，不能强迫它保存 H96×T10。A/V 在实数中可交换的项也要通过同整数函数验证；identity 的时间变换仍在。四轴误差隔离控制保留，不把一个供数布局失利扩展成整个消费者分离思想失败。

## 待补强对照：Conv2 R16 的连续 V

**B 以完整消费者链重排。** [真实四帧费用](rank_service_opportunity.json) 中，普通 uniform R16 将 26.842M 源加权累加改成 4.474M U 源累加＋9.580M 连续 V 乘积；这些是采样算术、不是周期或全帧费用。[R16完整825](rank16_control_valid825/uniform_rank16_summary.json) 已完成，AEE **1.220460308**，825帧／48,152,523有效像素，通过1.259预算。与此同时，连续 PED 的96×96 W_res仍需 **23.59296M**乘积，约为V的2.46倍。故本节V缺口保留，但后续优先级转到真实W_res，不能只磨更小的V，也不能因各轴共有就把该项从分母删去。

**完整 A 未补到这个矩阵。** [本地 da4ml 结果](../../../psn/cmvm_20260909/README.md) 只覆盖 S0 的10×10 Aq、patch两份10×10 Aq及三列前缀；没有当前96×16 V的完整图。[现成 DeepShift-Q 控制](../factor_completion_20260909/shift_consumer_train16/README.md) 属先前 Conv1 因子V，不能给本学生直接记作免费移位。应继承 [DeepShift-Q 的符号／幂次量化及STE](https://openaccess.thecvf.com/content/CVPR2021W/MAI/html/Elhoushi_DeepShift_Towards_Multiplication-Less_Neural_Networks_CVPRW_2021_paper.html)，再给同一整数函数 [da4ml 完整分解、跨输出 CSE、位宽与延迟限制](https://calad0i.github.io/da4ml/cmvm.html)。强轴包括字MAC、逐常量移位／CSD、whole-V图及H8分块、合法先A后V的latent域计算；同授零Z、缩位、广播和有限寄存。若BN gain纳入V后破坏单移位格式，要明确系数格式或保留其额外乘法。

**唯一 A＋X 问句。** 完整编译若暴露“某些跨H8共享子式省加法，却因节点位宽、长寿命及真实端口引起额外RF读写”，能否在同rank／Q5码域／训练预算内，训练V的符号和幂次关系，使共享子图与真实T10／H8消费域对齐，减少这项可定位的物理损失？对照必须给普通Q5训练后完整da4ml、相同权重的最优合法分块与普通费用训练。只多一些CSE命中、AV换序或门完成后删节点均不是X；若完整普通图已消掉该损失，这个问句没有新增落点。当前尚无该矩阵的图、逐节点合法范围和有限端口证据，所以结论是**先补A，X未成立**，不据小样本AEE开硬件，不扫码宽／秩。

## 优先修正：真实连续 W_res 的完整强对照

截至本次 canonical 目录／9月9日 inventory 检索，没有 Monarch、DeBut、CirCNN 条目或真实 W_res 的尝试记录；FABNet 的蝶形线索不能代替它们。已有普通低秩实验改的是 Conv2，DeepShift改的是此前Conv1因素，da4ml改的是小A；**96×96 W_res 的普通低秩、DeepShift和完整常矩阵图均未补到**。这里输入是连续 `r1out`，输出也被作为幅值消费，不能套用一位g的加法费用或末端阈值截断。

一手线索仅作待完整迁入的控制：[Monarch，ICML 2022](https://proceedings.mlr.press/v162/dao22a.html)用两级带排列的块对角矩阵及解析稠密近似；[DeBut，NeurIPS 2021](https://papers.neurips.cc/paper_files/paper/2021/hash/86b122d4358357d834a87ce618a55de0-Abstract.html)用适配输入／输出尺寸的广义蝶形因子；[CirCNN，MICRO 2017](https://arxiv.org/abs/1708.08917)用块循环参数和FFT训练／推理。这些结构不是低秩的同义词，但也不自动保留原W_res函数、实际满秩或AEE。本轮只补主源条目和机制定位，不记作逐节复现；本地未试不能记作已失败。

**只排一项下一强基线：真实 W_res 的同一多位定点函数，以普通MAC对完整da4ml图。** 用原连续输入和真实后继核定输入／权重格式及完整AEE，再将同权重交官方两级分解/CSE，计节点位宽、移位、扇出、临时状态和有限端口；H8/空间分块、静态零与流式消费双方共有。原FP→定点是另一个学生；只有选定整数函数内的编译才要求精确。DeepShift-Q5／结构矩阵／低秩仍是可用的模型控制，不同时排成新实验清单。先回答这23.593M项经完整成熟底座后实际贵在哪里，再讨论X，不能把整图可编译本身称创新。

## 全残差链重投影的条件边界

**完整展开在实数域成立，连续投影支路没有遗留的第四个连续源。** 实际 [MS_PED 链](/home/zhumd/work/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py:1710) 是 `head θg→stem conv/BN→r0 ADD→r1 ADD→PED`；两个残差增量均来自各自 `sn2 θg→Conv2→BN2`。令三源为 `z_H=θ_Hg_H,z_0=θ_0g_0,z_1=θ_1g_1`，各线性支路为 `α_i⊙(W_i*z_i+b_i)+β_i`，则锚点投影可写成 `Σ_i K_i*z_i+d`，其中 `K_i=C diag(α_i)W_i`、`d=CΣ_i(α_i⊙b_i+β_i)+b_res`。这比只展开 r1 更完整：此前剩下的 `Cx` 可继续展开为 head、r0 两路。当前非锚点删除不改变这些锚点的定义；head 之前的事件输入及非线性不在可分配范围内。

**stem 核不能假定离线固定。** 实际配置虽为 `spike_norm=BN`，但 [load_system](/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/algorithm/nrv_cost_probe/run_probe.py:172) 经 [set_bn_mode](/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/algorithm/run_bn_probe.py:84) 清除了未选 BN 的运行统计；`patch_train_calibration.pt` 仅恢复 r0/r1 四处 BN。head、stem 仍依赖当前输入统计；head 的 BN 和发放必须先执行，stem 则须等当帧统计完成后生成 `K_H(f)` 和偏置。直接收缩需 `96×96×48×9=3,981,312` MAC／帧，另付缩放／偏置处理，不能称免费折 BN。

**几何与费用边界。** head 合成核为 `96×48×3×3,stride4,padding1`；两残差合成核各为 `96×96×3×3,stride2,padding1`，最后 1×1 不扩大 halo。实际带 BN 的卷积无偏置，但 BN offset 不为此消失，三处 θ 都按真实幅值保留。新增三核共 **207,360** 系数（若 FP32，829,440 B）；原 stem／残差门生产链仍需原核，不能整套扣除。head 支持须保留至动态 BN 就绪，或重读／重算；r0/r1 依次就绪，还要累计投影部分和。全帧支持的裸位图分别为 **18.432/9.216/9.216 MB**，全帧 FP32 投影部分和为 **73.728 MB**；这些是全图保存方案的量级，非最低 RF。分块可改变容量，但必须付 halo、供数、溢出或重算，锚点 1/4 不等于源读 1/4。

**不能直接继承现有定点学生精度。** 当前 X12 投影先对连续总和量化，即 `Cq·Q12(x_base+Y_0+Y_1)`；舍入／裁剪不满足分配律。原 FP32 的 Conv／BN／ADD 舍入点也不能通过实数等式消除。完整重投影须保留这些数值边界，或明确建立并验证另一数值函数。r0.sn1、r1.sn1 和 proj.sn 仍消费原连续残差链，除非另改其表示，三路重投影不会自动免掉这条链。线性分配、RepVGG／Collapsible Blocks 式重参数化均归入 **A**；本次只解除“必有不可展开 Cx”的疑问，不给代数、系数数或帧级摊销记创新／性能分。是否能在完整门链与有限状态下净省，是尚未计量的问题。
