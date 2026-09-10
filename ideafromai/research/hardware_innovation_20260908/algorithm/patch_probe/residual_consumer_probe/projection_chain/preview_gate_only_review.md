# Preview V32→96 的 gate-only 挂点

2026-09-10；只读图/参数与既有工件，未编新图、训练或运行 RTL。

**B：消费者条件确实改变。** 实际 [MS_ResBlock.forward](/home/zhumd/work/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py:907) 先保存 identity，另一支依次 `sn1→Conv1→norm1→sn2→Conv2→norm2`，最后才加 identity。因此 Conv1 的 V 输出没有额外连续旁路消费者；[TrainableLatentPair](../../factor_completion_20260909/latent_stage_train16/flow_backward_probe.py:41) 也只返回 θg，`shared_raw` 是内部预测/诊断暂存，sn2 后清除。不能用此前 PED 连续 anchor 保住 Conv2 的负结果否定此处。Conv2 的 3×3 halo 仍需要全空间 sn2 门，不能把 gate-only 误解成可无条件删除 75% 门。

**A：先补完整普通底座。** 实际 `flow_recovery64/preview_only/shared48_u8_vq5.npz` 有 32 条活 latent、V 的 3,072 项全部非零，A 有 34 项。[preview_temporal_coordinates.py](preview_temporal_coordinates.py:36) 已给所有轴普通重排 `Q=AZ`，使 A 的产品数由 250.6752M 降至 83.5584M/帧；V 仍有 2,359.296M 幂次系数项，均不是周期。固定 BN 可编到逐(t,h)阈值，读出为 $g_{t,p,h}=\theta\,1[\gamma_h\sum_j V_{jh}Q_{t,p,j}+\beta_h\sum_s A_{ts}+b_t\ge\theta]$。现有参数 gain 全正。应继承 [da4ml](https://calad0i.github.io/da4ml/cmvm.html) 的完整跨输出 CSE、位宽/深度权衡及普通 last-use；[BBS](https://arxiv.org/html/2409.05227v1) 的组和/常数分解必须连补偿一起计，进一步剪位或改 Q5 属另一学生；[TermiNETor 对应作者主文第5章](https://escholarship.org/content/qt58s6j4qq/qt58s6j4qq_noSplash_dde1f5459a65e508b012d6f254a34805.pdf?t=ros586) 的共同终止校准/供数值得继承，但其预测许可不能冒称严格界。

**X：有一个未测、可定义的融合接口。** 对每个 `(p,t)` 的 96 门保留未决位图；静态记录共享节点 v 的最终消费者集合 C(v)。只有 `C(v)∩未决门=∅`，且没有已发射的读者，才可取消尚未发射的节点/释放其值。严格门界必须来自当前已算量、合法输入范围和完整舍入包络，不读未来完整答案。潜在增量是**让门可确认的时点参与 CSE 分解和执行顺序，避免生成只供已确认门的宽公共值**；已算节点的普通最后使用释放、引用计数和动态死代码删除本身不是新机制。静态消费者表可共享，运行时位图/完成反馈、节点读写、范围计算与背压都收费。

**强对照与最强反对。** 同数值函数给完整 CMVM whole/H8 图、逐输出严格 early-stop、普通组级界/共同退出、零输入旁路和最佳 A/V 次序。高扇出节点可能被最后一个未决门保住，或已在门可判前算完；此时门率低、静态 CSE 节点少都不产生新增取消。当前未找到这份 V32→96 的 CSE/扇出文件；已有 `psn/cmvm_20260909` 是 T10 PSN，`projection_cmvm/bbs` 是 PED96×96，均不能顶替。下一步只值得先编这份真实 V 的普通图，在明确整数/FP 舍入函数后测“未发射节点取消”相对逐输出/组控制的增量；当前可保留挂点，尚没有硬件净收益或 ≥7 分新颖性的证据。

**三篇最近先验补核（2026-09-10；阅读深度分列）。**

| 先验与本次证据 | 已做过、应完整继承 | 对本挂点仍未获证的部分 |
|---|---|---|
| Cherati / Barzegar / Sousa，ISCAS 2025，*Early Termination of the MSDF Computations Towards Efficient Inference in Neural Networks*；[作者机构条目及 DOI 10.1109/ISCAS56072.2025.11043511](https://researchportal.ulisboa.pt/en/publications/early-termination-of-the-msdf-computations-towards-efficient-infe/) | 正式摘要确认 MSDF 串行 digit 的 MLP ASIC，以及达到目标输出精度后终止。 | 本轮未取得作者全文，不能断言其许可严格无损，也不能断言其没有跨输出共享/反向停止机制；不能拿摘要补成完整方法。 |
| Abdelhadi / Shannon，FGIE，FPT 2019，*Revisiting Deep Learning Parallelism: Fine-Grained Inference Engine Utilizing Online Arithmetic*；[作者全文](https://www.ece.mcmaster.ca/~ameer/publications/Abdelhadi-Conference-2019Dec-FPT2019-FGIE.pdf)，已读全文，重点 §III-C、图5–9、§IV–V | signed-digit 在线加法树、MSDF ReLU 的首个非零负 digit 检测及反向停止/复位；每神经元树、synapse 时间复用、同位显著度存储、缓存权重位重复使用均已实现。 | 图示是独立神经元树，共享 activation memory，不是实际跨输出 CSE DAG。全网络在线化在 §V 是未来工作。FPGA 的 BRAM/LUT 布局与频率、能效不能直接迁成 CMOS 收益。 |
| Pan 等，BitSET，TECS 22(5s):98 / CASES 2023；[作者条目](https://pyjhzwh.github.io/portfolio/BitSET/)及[作者报告](https://pyjhzwh.github.io/files/CASES23_BitSET_slides.pdf)，已读24页报告，重点8、17–20、24页 | 权重 MSB 预测负 ReLU、专门正负编码、OS 阵列、比较器、Skip Matrix Buffer、双缓冲均已有；预测与非预测位串行底座应分别给对照。 | 作者“PDF”链接实际指向 ACM，正文访问403，本轮未读24页期刊全文。报告明确预测式；实测“不掉精度”不是逐输入严格证书。报告未展示共享 CSE DAG 的 digit 需求传播，不能据此宣称正文也没有。 |

**剩余问题要落到实际图。** MSDF 早判本身已有完整实现。这里连续 `Q=AZ` 的数值 bit/digit 轴与原 T10 轴独立；VQ5 已是幂次移位，不能把 BitSET 权重逐位乘法的省项率直接套来。普通逐输出树也应允许将真实阈值对齐后做正/负两侧严格判决。da4ml 的 word 图则不能直接把每个节点换成一位加法器：须补 signed-digit 表示、在线延迟/配平、进位或残差状态、移位对齐，以及共享节点向不同消费者供给不同精度的协议。保守控制是“所有活消费者都不再需要某节点的后续 digit，且在途读取完成”才停止；最后一个难门可能保住整个共享前缀。

因此，尚可测的是**门确认时点是否足以改变真实 V 图的共享边界，使省下的未生产 digits/nodes 超过精度转换、反馈、缓冲和端口成本**。消费者引用计数和 last-use 仍是普通控制。必须同时给 word-parallel whole/H8 CSE、独立输出 MSDF 树与共享 MSDF 图，在同部署精度、明确面积/带宽预算下比较；完全展开的停钟通常先是活动量收益，折叠执行还须证明省掉实际发射。现有证据没有覆盖这一具体组合，也尚不足以证明它新颖或有净收益；两篇全文未获得的边界保留。

**实际 V 图与“选择共享/复制”复审。** 上述“尚无 V 图”是前一轮状态；现有 [完整编译结果](preview_gate_cmvm/result.json) 已给 1,711 节点、388 个多输出共享节点（376 个跨 H8）、最大深度6，独立输出归约为2,976次加减。合法接口为共同尺度 signed24，完整分子输出最多42位；355个验证向量是基向量/随机/定向域角点，尚不是部署 latent 的联合退休轨迹。388个共享节点不能当作可删除数量。

先验已覆盖目标函数的许多部分：[Potkonjak / Srivastava，TCAD 1998，§V、VI-B](https://web.cs.ucla.edu/~miodrag/papers/Potkonjak_TCAD_98.pdf) 已将时序变换、CSE/复制与执行单元、寄存器、连接费用共同考虑，不能把“优化实际费用而非节点数”单独叫新颖；[Li 等，ARITH 2018，§2–4](https://johnwickerson.github.io/papers/digitelision_ARITH18.pdf) 已按在线 digit 依赖和误差分析删除无用数字，并付内部残差保存；其保证针对特定迭代收敛/任意目标精度，不能直接当本 SNN 的逐门证书。这次定向核查没有确认它们做过“神经元联合严格退休轨迹驱动 CSE 共享边界”，但这个未确认不能充当首次提出的证明，前述两篇未获全文的限制也仍在。

**最强反对有一个逐样本不等式。** 若只复制完全相同的节点，表示、对齐及消费者需求不变，令 `d(v,h)` 为消费者 h 所需的该节点 digit 前缀长度（含在线依赖），则共享工作 `max_h d(v,h)` 不大于拆为组后 `Σ_G max_{h∈G} d(v,h)`。例如两个消费者需2和20位，共享20位，复制22位；普通逐消费者停止接收即可让快分支退休。非前缀需求同样是“并集不大于各份之和”。因此，纯复制无法靠早停省该节点的总 digit 算术。它只能由**实测有限端口/扇出/活状态阻塞的下降**抵偿复制税，或由进一步重关联/分解改变合法证书出现时刻；共享图应同样获得接收端门控、小缓冲、独立消费者完成及合理重排。

预先只需过一个判别门：在同完整数值函数上记录每个样本96门的**联合**证书时刻及反向 digit 需求，对原图、H8内CSE、独立树与选择复制图分别给相同早停；训练样本选静态图，验证样本只评价。先给候选最乐观的零反馈/零控制费用、但保留所有复制算术与必要输入读取的下界，再与强普通可实现排程比较；这个下界尚不能赢，就停止该布局。若只剩物理阻塞机会，必须指出具体哪个共享节点、哪条端口、多少等待或活状态可消除，并让普通加同预算缓冲/重定时后再比。改变图后要重新导出证书，不能把旧图的最早判门时刻无条件搬过去。当前这是值得做有界判别的编译—执行假说，尚无已成立的机制增量，不进入 RTL。

**共享粗值、按需细修正补查（五篇边界）。** 最贴近的是 [Precision Gating，ICLR 2020，作者全文§3、§4.3](https://www.csl.cornell.edu/~zhiruz/pdfs/pg-iclr2020.pdf)：已经把输入拆为高/低位，先算粗输出，只为选中的输出补低位卷积，并复用粗结果。这里是可训练的有损许可；[官方 `PGConv2d`](https://github.com/cornell-zhang/dnn-gating/blob/master/utils/pg_utils.py) 实际先密集算低位再乘mask，论文另有CPU SDDMM实验，专用硬件部署列未来工作，不能称它已实现共享CMVM的选择性供数。其余四篇的边界是：FGIE全文已给逐神经元在线树与反向停止；BitSET目前仅作者报告已读，预测与严格证书仍有区别；前述 ARITH 2018 全文的§2–4已给digit依赖、跳位和恢复残差存储；[SOAP，FPT 2013](https://cas.ee.ic.ac.uk/people/gac1/pubs/XitongFPT13.pdf) 作者原文首段/摘要确认联合改写表达式结构、精度及资源目标，本轮全文下载失败，不能据此核其全部CSE实现。**未确认这五篇完整实现了“多输出CSE＋严格门退休＋细修正请求”的组合；也没有据此证明该组合首次提出。** 粗细两阶段、运行时精度、CSE成本目标都应继承为底座。

可公平比较两种布局，均保持现有V的完整整数函数、同输入/输出精度、总算术单元、缓存容量和物理读写口；A前移生产与T10输出要求不删：

1. **单份渐进共享图。** 原1711图按digit或定宽chunk继续，每节点仅保存一份增量状态；粗片广播后，细片只送仍需它的消费者，按最大精度需求继续。给普通接收端门控、分片存储、合理队列/重定时；不强迫一个慢门让所有已退休分支继续收完整word。
2. **共享粗图＋选择性修正图。** 用同一静态切分 `Q=2^ℓ Q_H+Q_L`，粗图覆盖96输出，严格区间确认后仅给未决输出执行低位修正；修正可按H8局部CSE或共享图择优。两阶段时分复用相同算术资源，计粗结果/残差保存与修正输入重读，不能免费增设另一套树。普通独立输出早停和H8 CSE也获得同样切分。

一个必须先付的具体成本是**修正值的进/借位与位宽**。设 `K=2^15 V`，对负Q也采用精确floor分解，使 `0≤Q_L<2^ℓ`；则 `KQ=2^ℓ KQ_H+KQ_L`。后项范围按K的正负系数和传播，保守幅值界为 `(2^ℓ−1)Σ|K_hj|`，并非ℓ位；节点修正也要逐节点计算范围。若不重算粗阶段，必须保存足够的粗结果、在线内部残差，或未决输出的等价阈值余量，精确处理跨分界进/借位。统计报告要落到**实际物理字的细片读取并集、输出残差读写与峰值活状态**：若一个未决门仍需全部输入细片，源流量就可能不降；若细片与粗片同一个字且已全读，也不能再记带宽省益。只有相对布局1的最佳按需供数，布局2仍省掉足够端口服务/状态且覆盖这些费用，才构成这里尚待验证的增量。引用计数、背压、粗细切分本身不计新颖性。
