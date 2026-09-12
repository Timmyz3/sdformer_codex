# 证据后收敛：四个可直接做小试验的融合

2026-09-12。以下是提案/继续条件，不是已验证贡献；初始独立轮8条保留在原页。X必须是强普通对照已拿到同一先验A以后剩余的实测增量。每个模型新函数最终均以同DSEC本地NB0（valid825 AEE 1.445353）为精度门，并报告相对ordinary/lifting各自的变化；不再使用取消的+0.005门。几页的TCAS-II标题不能靠收集多个先验名凑成。

**阶段收口更新：** F-L2的免训权重rank0/4/8/16筛查已经执行，当前sign布局被W4/W8压过；普通W4/W8四臂随后真实diverse10均过NB0，详见[权重结果](../weight_compensation/README.md)和[算法结果](../algorithm/README.md)。F-L1已完成；F-L3完整PIT训练、F-L4跨帧状态仍未执行。下文“最低试验/优先级”保留提出时的方案，当前执行次序见[下一阶段](../NEXT_STAGE.md)，没有正在运行的训练。

## F-L1：精确连续传输＋动态BN默认值（已跑，降为公共底座）

**B。** 源为空不代表动态BN输出为零，连续PED也不能因为二值门已确定而删。真正可试的是保留所有必需的FP32/I24字，用精确codec减少传输；它与C1包含关系、源地址调度不同。

**A已借。** ZipServ的指数与尾数分开、精确逃逸；Finch的非零默认值思想；经典块位宽/BDI式差分/XOR作为普通控制。Atalanta与Shannonic已经把范围符号＋offset和紧凑状态当作核心方法，进一步压缩的概念不是空白。[ZipServ](https://www.cse.ust.hk/~weiwa/papers/zipserv-asplos26.pdf)、[Shannonic](https://proceedings.mlsys.org/paper_files/paper/2026/file/96f39c8de84678cb2a908cd52bfd7819-Paper-Conference.pdf)

**最强对照/缺项。** 当前所有普通mode都允许按块选、付元数据；它们仍不是完整ZipServ/EBPC/Atalanta原作。未来若接RTL，还须双方共有真实默认值、同字宽/端口、同双消费者缓冲、真实在线编码和随机读索引。离线同帧直方图不能当免费预测。

**实际X。** I24 all-modes仅比signed-width少0.1045%/0.1080%字节；不给新标题。FP32 fill+exp比fill少12.50%/12.56%，但普通all-modes还更好，属于加强普通底座。尤其本地已实现ordinary BN→PED融合不物化BN输出：对该当前强基线，BN输出codec的新增可避免流量是0，不能把离线capture大小当仍存在的DRAM事务。没有测周期X或PPA X。

**最低试验/已完成。** 单帧两臂全域与真实切片，FP32 43,008、I24 20,480个mode-block实际字节roundtrip为零错。结果与脚本见[一页结果](ONE_PAGE_RESULT.md)。继续触发只能是找到普通融合后仍必须spill的其他连续源，并证明codec端口/延迟预算足够；不能为codec人为恢复已删除的BN输出物化。当前停止主创新投入。

## F-L2：连续激活保真＋符号权重主体＋小残差补偿（优先新训练小试）

**B。** 当前昂贵连续PED/FFN有真实多位幅度，激活直接二值化会丢必要信息；其权重是否可以变简单，不能由这一失败推出。此前“连续必须普通多位MAC”的前提在允许新函数训练后可重开。

**A已借。** ReverB-SNN用实值发放与二值权重，把主体乘法变为加/减；MiLo把量化权重误差与低秩补偿联合优化。只取这两条已公开机制，不能把“二值权重＋低秩补偿”重新命名为X。[ReverB](https://arxiv.org/pdf/2506.07720)、[MiLo](https://proceedings.mlsys.org/paper_files/paper/2025/file/9032e5c9ec394ce768a2fa9bdc56af6c-Paper-Conference.pdf)

**具体新函数。** 在一个真实连续投影选择 `W ≈ diag(s)·sign(Wb) + U_r V_r`，保留原连续输入和所有输出消费者，保留已承诺RNE/sat边界；新的舍入必须明确写进训练前向。它改权重，不是另一agent的源激活`Dg+量化残差`。按实际相同输出向量评估符号主路径、残差乘加及scale成本。

**最强对照。** 同训练步数的普通W4/W8 A24投影、固定rank且同实测资源的纯低秩、MiLo式普通低位主体＋残差、ReverB式无残差主体。总rank/权重SRAM/源RF/累加精度相同，不能拿符号加法数与高位MAC数直接等价相加。CICC2026 BF16×1-bit只是可能的电路邻居：原digest未取得，不借它的TOPS/W。

**原作缺项。** ReverB全神经元训练、α折叠；MiLo完整HQQ交替优化、原INT3 kernel/量化compensator均未迁。尤其动态BN与双消费者下α不能未经证明跨边界搬移。

**待测X/反证。** 只有在相同精度门与共同A下，连续主体转换成可共享加/减供数后比普通低位＋残差减少真实完成周期/状态才是X。若二值主体残差rank逼近原rank，或W4同训练预算更优，则淘汰。

**最低试验。** 先从checkpoint单层权重做rank{0,4,8,16}残差谱/真实capture功能重放，再由root统一GPU队列做一层短恢复、diverse10筛查；过门才跑valid825。未执行，不能列为精度结果。

## F-L3：PIT的缩放/梯度先验＋双消费者可部署量化合同（做强训练对照）

**B。** 单看gate一致或连续MSE不能保证稠密光流；同一源同时服务阈值与连续消费者，其最合适的量化尺度/梯度方向可能不同。允许训练后，旧免训失败不等于这一优化无效。

**A已借。** PIT实际用逐时间/通道对角尺度、3σ初始化、修正surrogate与整数发放；其附录再参数化依赖具体相邻层。它不是稠密可逆T10变换，与已有静态θ折权高度相邻。[PIT原文](https://proceedings.iclr.cc/paper_files/paper/2026/file/0ac46bb0a72a7afe311d9b48b5088df8-Paper-Conference.pdf)

**具体试验。** 保留非因果T10 PSN，在同一现有有损源量化实验加一个PIT式time×channel尺度/梯度控制臂，显式传播给连续分支；不要把局部缩放折过动态BN。它应交给正在做新表示的算法agent统一训练预算，本代理不另开重复训练。

**最强对照/缺项。** 现有可学AT-LIF θ、普通per-channel QAT、same-D时I-LIF、同损失同训练预算的Dg+残差；PIT完整原作训练与原始folded推理尚未迁，不能声称现有source QAT已经复现PIT。

**待测X/反证。** 先问相同输出费用下有没有真实AEE收益；再问新尺度是否能减少被真实消费者要求的残差宽度/逃逸服务。如果普通θ/per-channel QAT已经解释改进，X=0，保留训练配方即可。单纯换初始化、加一项loss不是电路贡献。

**最低试验。** 将PIT diagonal-scale+rectified-surrogate作为现有量化恢复的一个小ablation；记录gate flip、连续误差、实际AEE和残差总字节，不能从任何前三者替代AEE。未执行。

## F-L4：跨推理窗口的差分尾部＋守恒神经元（只先做边界试验）

**B。** 现有MotionDeltaCNN迁移只测同一T10块内摘要参考，不能回答相邻推理窗口是否有可复用的尾部状态。当前PSN非因果且dynamic BN会改变连续默认值，因此原函数不能直接套流式守恒结论。

**A已借。** VISTREAM把差分输入、ST-BIF的跨推理状态与平衡点等价放在一起；平衡点等价不代表固定T10预算等价。EDCFlow给任务上的时间差分教师/细节目标，但已有EDCΔFuse/IF-TID提案，不是新发现。[VISTREAM](https://openaccess.thecvf.com/content/CVPR2025/papers/You_VISTREAM_Improving_Computation_Efficiency_of_Visual_Streaming_Perception_via_Law-of-Charge-Conservation_CVPR_2025_paper.pdf)、[EDCFlow](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_EDCFlow_Exploring_Temporally_Dense_Difference_Maps_for_Event-based_Optical_Flow_CVPR_2025_paper.pdf)

**具体新函数。** 先限制到T10 encoder已完整输出之后的一个可训练尾部，比较stateless/有状态差分，而不是把PSN改成因果或偷删BN。重新学习尾部归一化/阈值，明示这是新学生；若未来跨窗默认值更新仍需全域服务，也必须记账。

**最强对照/缺项。** stateless同宽尾部、简单前帧复用、完整DeltaCNN/MotionDeltaCNN控制、VISTREAM ST-BIF平衡点与相同延迟budget早退出两臂。其作者完整状态实现、checkpoint/所有任务没有迁，旧本地<2%不能写成完整原法失败。

**待测X/反证。** 只有相邻真实窗口中付清历史状态写回/运动变化/BN默认更新后，仍减少端到端尾部工作而不伤AEE，才继续。若额外状态成本或平衡迭代吞掉差分节省，或stateless普通量化同样好，淘汰。它可能完全不是本轮TCAS-II最短路径。

**最低试验。** 先在现有motion/capture连续帧上只量源变化、BN均值方差变化、尾部增量与最少状态字节，再写ST-BIF一层平衡/固定budget功能小例；零GPU即可先排除不合法等价。真正精度须重训/valid825，本轮未执行。

## 独立轮回看与评审日志

L-I01已被BNFF/FlexAcc/Finch强先验覆盖，不能当新归一化；L-I03落到F-L1并实际降级；L-I02/05/08由MiLo/PIT/ReverB约束后归入F-L2/3；L-I04与现有butterfly/Monarch/da4ml重叠，不另开；L-I06只保留为任务教师思想；L-I07无足够机制证据，停在假说。证据后新增F-L4，是旧“块内差分”前提改变后的跨窗口新函数小试，不代表已找到收益。

优先级是执行建议而非自动赢家：F-L2先做权重侧便宜筛查；F-L3并入正在运行的训练对照；F-L1主标题停止、共享底座保留；F-L4先用边界试验判断是否值得长期投入。不同意“压缩率大所以硬件必快”，也不同意“一个免训版本没过所以训练版都不能重开”。本页由同一AI作者自检，尚非独立反方评审；提交root进行最终交叉评审。
