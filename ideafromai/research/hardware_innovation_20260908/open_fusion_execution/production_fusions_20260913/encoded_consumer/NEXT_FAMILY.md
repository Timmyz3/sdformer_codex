# ONE 后继：跨推理帧的H8源门安全半径缓存

**选择B5的一个新参考边界，先零训练、零RTL。** 固定matched dense stage320父、两个同坐标源像素的24个H8组；`{64,256,1024}` I24单位是**ONE固定三级编码器**，不是三个配置扫描。从上一**已完成**参考取得I24、T10门词及等级。当前I24到达后付费认证，成功取消完整源DAG，失败执行原完整CSE。此具体接口未试；概念新颖性暂评**3/10**，不是录用概率。本页没有启动捕获或实验。

**B：为什么换边界。** [35条真实编码链](README.md)中，fixed/affine码内消费仍比raw慢0.70%–2.06%，full-D比同函数展开慢2.41%–3.18%；换码未删除源/Conv。旧phase/global H8剪枝、成对恢复、dense/34/lifting及两项PoT都已试。[连续四帧资料](../../../motion/capture/frames.json)只有native ep34门/flow，**没有matched dense I24/门前整数dot，必须新采**；旧约295MB宽历史方案也不是本缓存。[B3/B5/B8](../../breadth_20260912/coverage_owned/REMAINING_BATCHES.md)仍开放；水平P2 strict equal-a的38,220个SIMD8零命中不计新证据。

**A：最近邻已经覆盖安全半径，不能列X。** Pro两文的源结构/物理组建议及第二轮§6.1–6.2已给共同取消与存余量再检查；Grok的[F1](../../../survey_ab_fusion_20260910/f1_f2_paper_design.md)、[9/11复核](../../../grok_review_20260911/CODEX_FOLLOWTHROUGH.md)及[因果帧间条目](../../../../grok46_20260905/02_ranked_mechanisms.md)也已有源活动、完整剪枝对照、event-dirty/CBinfer方向。原始[CBinfer](https://arxiv.org/abs/1704.04313)覆盖历史结果复用；[DeltaCNN §3](https://openaccess.thecvf.com/content/CVPR2022/papers/Parger_DeltaCNN_End-to-End_CNN_Inference_of_Sparse_Frame_Differences_in_Videos_CVPR_2022_paper.pdf)覆盖非线性状态、累计变化和更新mask。**非零变化仍保持判定**的margin/dual-norm安全球至少已被[Hein–Andriushchenko定理2.1](https://proceedings.neurips.cc/paper_files/paper/2017/file/e077e1a544eec4f0307cf5c3c721d944-Paper.pdf)明确覆盖；线性门的`margin/||A||₁`是其直接特例。旧Pro证书也非只支持exact memo。

[SnaPEA §II-A](https://cseweb.ucsd.edu/~hadi/doc/paper/2018-isca-snapea.pdf)已有精确提前激活及按权重符号重排，但其非负输入/ReLU前提不能原样搬到signed I24；[LAWS](https://arxiv.org/abs/2605.04069)摘要还出现自认证区域缓存概念，这里只核身份/摘要，不背书其定理或硬件。本次未完整移植上述GPU后端、预测模式、全网历史管理；PIT/MFPSN、完整HiNM/VENOM、B8网络替换也未借全。跨帧换缓存名、证书以及普通请求取消都不是首次主张。

**候选X只剩付费执行接口。** 对实际整数源图和门/连续多消费者边界，认证能否在原源issue之前完成，并在同资源下留下净取消。用既有`compile_postprocess`保留RNE/sat的整数门原像；`L_t=sum|As_q16[t]|`静态。`m`定义为**包含端点的最大安全整数扰动**：对`S≥K`，真/假分别为`S−K`/`K−1−S`；对`S>K`分别为`S−K−1`/`K−S`，反向镜像。若改用“到最近翻门整数距离”`m_flip`，必须用`L_t*d<m_flip`。实际RF生成S/g时选全部80个门满足`L_t*d≤m`的最大固定等级，均不满足则0。当前付费比较`max|ΔI24|≤d`；通过返回实际门词，失败跑完整CSE并刷新。三级检查、H8/T10归约、历史读写及回送全收费；命中保留原参考以免累计漂移。原理全属A；这项X仍可能只成立为工程实例。

**最强普通对照。** 同父完整CSE原执行；同缓存/同地址/同cold fill的exact-I24 memo（Δ=0和静态常量命中单列）；普通CBinfer式先完成源再比较门；若接delta Conv，也给各臂同样的精确delta/缓存权限。候选独占收益只能来自**Δ≠0时，在源完成前认证的额外取消**。raw I24仍供残差，不能删首读；当前版后续K864、sn2、Conv2、merge、PED及动态BN/head全部照常，因此不会把门相同偷换成连续值相同。只要普通基线已同样消除该工作，就归A。

**真实入口与一次试验。** `BASE=ideafromai/research/hardware_innovation_20260908`，`OPEN=BASE/open_fusion_execution`。下列是核读过的实际入口；新certificate wrapper尚需实现，当前没有伪装成现成命令。

| 环节 | 现有可调用入口与固定参数 |
|---|---|
| 父/训练 | `OPEN/breadth_20260912/algorithm/train_matched.py --root BASE --structures dense`；本试验**0次更新**，只读`algorithm/matched_training/dense/stage320/deployed_constants.npz`。若以后恢复，才沿同seed912、64+256、GT与原head预算另立输出，不能覆盖已有训练。 |
| 连续捕获 | `algorithm/parent_network.py:ParentNetwork`＋`fixed_structure.py:LiteralForward(...,'dense')`＋`endpoint_observer.SmallCapture`＋`evaluate_axis`；按`BASE/motion/capture/frames.json`的Zurich0001–0004顺序采同父两halo，并在真实SOURCE_SN入口采I24、门前整数dot。现有`capture_endpoints.py`的CLI固定首帧，须用以上函数加四帧wrapper，旧门/flow资料不能补成I24。 |
| 编译/执行 | `breadth_20260912/source_execution/run.py:compile_graph/literal_gate`和已有`dense/program.json`；`hardware/matched_local_chain/run.py:install_source/source_gold/preview_gold`、其`windows.window`与`consumer.run`是同Machine骨架。需要新增有费缓存钩子与跨帧保留状态，原`run.py dense interior`仅是原首帧控制。 |

固定interior源窗中最先两个有效像素、全部12个H8，24项历史，每项240B I24＋16B门词，另等级/有效位/地址；**8KiB上限从原128KiB状态池内分配**。四帧冷启动＋三转移、完整局部链，ready/stress各一次；只执行上述一个固定三级编码。先查所有门、updated、PED及外送0差，再报exact/额外命中、取消源程序数与全费总服务。若放不下或不偿费，只停这24项/三级接口。精确函数可沿用该父NB0资格，不新增AEE、不恢复+0.005门、不外推全帧。

**排重与少数意见。** 本参考轴是独立推理帧的已完成状态，不是T10内位移、水平source pair/equal-a；未更改34项连接，也未复跑旧phase/global H8 mask，且不抢正在执行的码相位接口。少数意见是B3/B8改变生产函数可能有更大上限，而本安全球可能因帧差过大或H8最差lane几乎从不通过；因此先做这一项便宜、精确、能归因的否证。若只有通用memo/证书迁移的收益，保留工程结果并下调新颖性，不能承诺accept。
