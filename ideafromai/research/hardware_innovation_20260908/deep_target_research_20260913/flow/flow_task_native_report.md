# 事件光流任务结构与前端执行稀疏化

最值得先验证的任务原生方向是：**以已完成的 preds.1 粗流为先验，在 decoder2 研究由事件可观测性与不确定性决定的局部细化。**它比在 r0 前另建运动头更自然，但目标份额较小，且 WaveletVFI 已覆盖粗运动、动态阈值和稀疏细节这一强先例。r0 两个 96×96×3×3 大卷积仍适合建立“背景响应＋偏差支持”的无损对照及较大规模后续扩展；真正跨窗口缓存搬运保留为第三路线。三者都尚未得到本网准确率、覆盖率或硬件净收益证据。

“事件相机稀疏”本身不足以支撑方案。事件缺失既可能对应平坦区域，也可能对应孔径歧义、低对比度、遮挡或高速运动；网络中空输入经过 BN、PSN 和残差后也不必为零。这里的研究问题是：能否让昂贵空间计算的必要区域由任务可观测性决定，同时把低成本全图预测、时间依赖和真实存储事务闭合起来。

## 1. 本网证据与输入契约

当前定位分母来自 `root_owned/profile.json` 的一次实际前向，170 个 ATen 卷积/矩阵调用，合计 596.5464288 G 名义 MAC。普通卷积计数包含padding位置的名义乘加；源码对转置卷积实际按 `input.numel()×Cout×9` 的scatter范围计数，并未再数stride2插入零，边界裁剪尚未扣除。profile的泛化counts文案比代码更宽，故不能把decoder2的16.008G再凭“去插零”缩成1/4。分母未包括布尔Motion-XOR、NumPy onepass BN、访存和布局，不能转换为ASIC周期份额。

| 当前层 | 输入与卷积形状 | 名义 G MAC | 名义份额 | 已测左输入非零率 |
|---|---|---:|---:|---:|
| patch r0.conv1 | [10,96,240,320]；[96,96,3,3] | 63.7010 | 10.678% | 4.3602% |
| patch r0.conv2 | 同上 | 63.7010 | 10.678% | 3.6127% |
| patch.conv.conv.0 | [10,48,480,640]；[96,48,3,3] | 31.8505 | 5.339% | 9.8937% |
| patch.proj | [10,96,240,320]；3×3 stride 2 | 15.9252 | 2.670% | 0.9521% |
| decoder2 | [10,386,60,80]；转置卷积 | 16.0082 | 2.683% | 15.4003% |

r0 两个卷积合计 127.401984 G，只占这个算术分母的 21.3566%。即使假设完全去掉两层，名义 MAC 上限也只是约 1.272×；若两层各减半，则约 1.120×。这些仅是算术边界，实际 bit-skip 基线、BN、状态和通信会改变周期收益。不能把后文局部节约乘旧 ep34 份额，也不能从较小的注意力份额虚构整网速度。

当前任务输入契约为 T10 双极性 voxel，U-Net/Swin 主体、非因果 PSN，当前粗头出口为 `preds.2`；不是 RAFT 迭代网络。AT-LIF 输出为 {0,θ}，静态 θ 可吸入后继权重；Motion-XOR 不是 SDSA。本报告不使用旧“θ 是不可吸收事件载荷”解释。

### 已有代码能确认什么

`ParentNetwork.__init__` 对 `patch_train_calibration.pt` 所列 BN 置 `track_running_stats=True` 并装入校准统计；`install` 又单独将 `PROJECT.norm_layer` 覆写为由本次输入计算统计的 onepass BN。因此固定仿射 BN 的局部推导应限制在已确认覆盖的 patch 层，不能越过投影动态 BN 后继续声称全链局部。

仓库原模型的 `MS_ResBlock.forward` 是 `identity → sn1 → conv1 → norm1 → sn2 → conv2 → norm2 → +identity`。原 `PSN.forward` 用完整 T×T 权重乘时间序列，无未来掩码。生产分析所引用的远端 `code/SDformer` 快照、ep34 YAML 在本工作区并不齐全；这里用现有原模型确认结构语义，运行实例的层身份和形状以实际 profile 为准。未宣称逐文件验证了远端快照。

### 真正可获得的信息

| 信息 | 现有 voxel 接口是否可得 | 如何使用及限制 |
|---|---|---|
| 各时间 bin、空间位置、正负极性的聚合量 | 可得 | 可计算活动掩码、局部密度、时间重心、粗边缘方向；属于聚合后的信息 |
| 原始微秒时间戳、同像素事件先后顺序、事件个数精确分解 | 预处理 voxel 不能恢复 | 若需要精确 time surface / 原始事件匹配，须更改传感器到 voxel 接口并保留元数据 |
| 当前窗口各 bin | 窗口完成后可得 | 当前非因果 PSN 本来要求完整窗口；不能称逐事件零等待推理 |
| 上一连续窗口最终 flow、已缓存特征 | 只有加入按序连续处理和缓存后可得 | 可以成为当前窗口因果先验；diverse10 离散样本不能证明该接口有效 |
| 当前窗口最终 flow | 前端执行前不可得 | 不可反过来作为跳过同一前端的先验；需独立廉价头或上一窗口结果 |
| 无事件区域真实运动、遮挡真值、深度/IMU | 现输入不能直接保证 | 推断必须带不确定性，不以 GT 生成部署掩码 |

`VoxelGrid.convert_CHW_polarities` 将两种极性分开、经空间和时间三线性加权累积成 [T,2,H,W]；与有符号求和 voxel 不同，前者避免正负直接抵消，但仍丢失 bin 内细粒度次序。`DSECDatasetLite` 的预处理路径直接加载 `.npy`，并不会附带原始事件列表。当前配置原文件缺失，因此更细的归一化、窗口时长和重叠约定仍需从真实数据生成清单核验，不能凭 `T=10` 推断。[1]

## 2. 文献前的四个独立假说及复核结果

| 假说 | 数据/任务来源 | 拟改变的算子 | 精读后的判断 |
|---|---|---|---|
| H1 背景响应＋偏差支持 | 无事件区域可能服从同一确定性背景响应 | 区域级跳过卷积与非线性重复计算 | 保留为无损候选和强控制；halo、BN 全局依赖会吞噬收益 |
| H2 全图粗上下文＋运动支撑细化 | 光流内部可较平滑，边界/孔径区域更需高分辨率信息 | 用廉价全图分支加选择性 r0 残差代替两层处处等量计算 | 优先研究；任务质量和实际停工都需训练/验证 |
| H3 T10 时间束模式复用 | 边缘运动产生跨 bin 连续轨迹，局部通道发放模式可能重复 | 共享卷积贡献＋例外码 | 暂缓；重复率未测，且已接近 Phi/LoAS/公共模式复用，不因名字含运动就具新颖性 |
| H4 跨窗口运动搬运＋新生残差 | 已发生运动对下一窗口有预测价值 | warp 缓存，重算失配/新生区域 | 保留为第三路线；属于新接口，不能被单窗口未对齐差分失败淘汰 |

H3 若继续，无损模式必须比较实际发放位，而非比较近似运动标签。运动对齐后相同的源图案并不自动给出卷积输出可搬运的等价性：任意空间 warp 不与卷积交换，插值会把位图变成连续量，PSN 又混合全部 T。它更适合在已有单元上测模式熵和复用距离，不宜直接升格为主任务贡献。

## 3. 原论文给出的约束

### 3.1 真正的事件表示与稀疏网络

EST 将事件看作空间、时间、极性上的点集，通过测量、核聚合和采样形成张量；不同投影会丢失不同信息。论文还给出学习核及测试时 LUT 实现。因此“把事件换成稀疏编码”“保留极性”“可学习表示”都有明确先例。适合借用的是表示选择与下游任务联合评估，而不是声称原始事件天然等同于一个 bit。[1，§3，pp.5635–5637]

AsyNet 的关键边界很容易被摘要掩盖：§3.2 明确说明 Submanifold Sparse Convolution（SSC）只计算已有活动位置，**与普通卷积不等价**；其无损结论比较的是固定算子后的同步/异步执行；普通卷积可作为所有位置均活动的特例。不能援引其“same output”来证明把本网普通卷积裁成SSC、删去无事件输出也是无损迁移。它的 rulebook、状态缓存、新活动点初始化与失活点清零，则是必须正面对照的先例。[2，§3.2，PDF pp.5–8]

### 3.2 光流特有信息与误差传播

IDNet 直接利用连续事件轨迹中的模糊方向寻找运动，减少对相关体积的依赖；ID 是同一批事件反复去模糊，TID 是过去状态预测下一批事件并跨时间更新。这对跨窗口接口是强 A，但其 ConvGRU、warm-start 和训练过程与本网非因果 PSN 不同。不能把 IDNet 的相关体积删除收益拿来算本网 r0 收益。[3，§III-A–D，pp.14709–14711]

EEMFlow 把平滑稀疏 meshflow 与稠密光流区分开；EEMFlow+ 用 coarse-to-fine 和 Confidence-induced Detail Completion 修复上采样造成的运动边界混合。其 CDC 的 self-corrector 是五层稠密卷积，另有 self-attention。**置信度融合已有先例，软融合不等于省掉细分支计算。**能借鉴的是“全局/细节分工”，尚需新增的是在昂贵前端计算前决定真实执行集合。[4，§3.2，pp.19200–19202]

EDCFlow 将 1/4 分辨率的多时间尺度差分与 1/8 的相关体积结合。它明确指出差分提供细节却缺乏直接对应关系，其 warp 使用上次迭代的 flow；这证明时间差分与全局对应互补，也揭示直接移植进本 U-Net 的因果缺口。[5，§3.1–3.4，pp.1986–1987]

EMatch 用 temporal recurrence 和 spatial contextual attention 建立稠密对应空间，明确写到没有触发事件的像素仍需要上下文赋值。它反驳“无事件像素无须推理”，但并不证明每个像素都必须运行同样昂贵的 96 通道前端。[6，§3.2，p.5849]

STSC-Flow（CVPR 2026）不把所有事件仅投影到一张 IWE，而保留 bin 内相对时间构造 VWE，以局部结构一致性与轨迹一致性约束连续运动。其双向分支和当前窗口最后 bin 特征明确依赖完整窗口。它为有损稀疏化提供可考虑的训练正则与失败诊断，不能作为部署时免费的正确性证书；只有 voxel 时也无法复现任意原始时间精度。[7，§3.1–3.3，pp.15127–15129]

### 3.3 硬件与跨窗口最近邻

SpiDR 已有 65 nm 实测芯片，明确测 DSEC 光流、T=10、288×384、2→32→六个32→32→2卷积网络。其原始位图存储、spike-to-address、可变执行时间握手和权重/膜电位存内累积都是直接硬件最近邻；不是“已有芯片只做分类”。其 Fig.4 的 19-bit AER、94.7% 稀疏门槛仅适用于所示尺寸。本网更大源地址会改变阈值。[8，§II，PDF pp.2–5；§III/Table II，p.7]

hARMS 用最近局部流事件缓冲与多尺度窗口仲裁代替遍历整幅事件帧；边缘法向局部流不等于真实二维流，必须由足够邻域缓解孔径问题。其 FPGA 加速边界将“已有局部流”作为输入，验证系统里的局部流先在 PS 软件计算。不能把它的吞吐直接当成原始事件到稠密流的端到端硬件速度。[9，§II-B、§III–IV，pp.58183–58186]

MotionDeltaCNN 已处理相机运动下的对齐、环形缓存、新显露区域 bias 初始化和卷积边界补偿；丢失边界状态可能要求完整重置。其逐层累计/截断缓存与误差传播是跨帧路线强控制，不允许仅以“warp＋delta”定义增量。[10，§3.1–3.7，PDF pp.4–5]

Flow-Based Visual Stream Compression 已用前一发送阶段的 flow 预测后续事件，用置信度与周期更新控制重建；主文说明逐事件对应维护太贵，采用 sending/predicting 两阶段。它是有损事件流通信压缩，不是本网特征执行加速；不过“预测未来事件＋例外保留”的任务概念已存在。新方案应比较周期完整刷新与按失配刷新两种策略，而不是假设逐点校验免费。[11，§IV-A–D，PDF pp.4–6]

ASNA-Flow（TVLSI 2025，33(12):3409–3422）是必须核实的非常近邻，标题与摘要已指向光流空间局部性和异步稀疏硬件。本轮未取得主文，publisher 页面未返回正文；不能据摘要推测其具体编码、队列、神经元模型或创新空缺，也不将其计入 12 篇全文精读。[12]

## 4. 候选 C1：由事件可观测性决定的局部细化

### 4.1. 优先挂点 decoder2：复用本来就已产生的粗流

`preds.1` 在 `decoder2` 之前已完成，因而可用作本窗口内的合法粗运动先验；它不依赖尚未执行的 `decoder2/preds.2`。这里“因果”仅指计算图上的先后关系，T10内仍可非因果。原MS decoder是 `sn → deconv → norm`，包括此前flow拼接在内的输入均先经sn，所以deconv本身不是二电平与连续flow的混合乘法源。连续量费用主要出现在sn之前、deconv部分和、norm和后继。

**强A补充：WaveletVFI（TIP2023）已做廉价运动感知→动态阈值选择→逐尺度稀疏高频系数生成；其§III-C/Algorithm 1使用前尺度高频阈值、上采样mask、前置稀疏卷积dilate3和RGB mask并集。**因此“粗运动＋稀疏细节＋halo”已经不是空白。PSN/BN闭合也只是集成义务，不能独立承担新颖性。[13]

仍可能值得研究的X需要收窄到：**事件可观测性而非仅高频幅值决定保留/刷新；训练目标惩罚实际T10与多相输出请求组的服务费用，使多个消费者不再需要某一源组时才停工。**这是待检验的新组合假说，尚无证据表明比简单方案更优或足够新。判据必须包含无事件有效像素与运动边界的误差；纯group sparsity、group预算与mask膨胀本身都有先例。

一种明确有损的新算子是用已有粗流的上采样提供全图底图，并训练decoder2只预测选择集合内的细化残差：`f2 = U(f1) + Scatter_M(Δf2)`。这不是当前`preds.2`的无损重写，需要连同其特征接口微调或重训。另一种更贴现有网络的版本保留decoder2低成本背景特征近似，局部运行原细化链；必须比较它与直接输出上采样f1的简单早退控制。

M由已有f1的梯度/局部不一致、当前voxel支撑/极性/时间重心摘要、skip特征的低成本不确定性估计共同给出。无事件不能直接触发“可信跳过”；在遮挡、新生运动、低对比度或粗流不确定时应保留。部署mask不能使用GT、完整decoder2结果或oracle边界。若这些输入只是换一个普通网络产生mask，而没有更好的质量—实际服务折中，则X不成立。

转置卷积stride2存在不同输出相位及跨tile贡献。对目标保留集合M，必须反求真实输入支持和各相位贡献；不能直接把输出mask原样当输入mask。应联合所有T和输出相位定义请求组，以全组无需求作为事务停止条件。polyphase代数分解本身是成熟实现手段，不是这里的任务创新。decoder2当前BN的实例模式尚未由本分支核实，不能把patch.proj的动态BN机械迁移过来：固定BN时可仿射融合，若实际为动态BN则必须在完整统计与背景重判之后下结论。

| 对照 | decoder2挂点 | r0挂点 |
|---|---|---|
| 粗运动先验 | 已有preds.1，可免费使用其已完成结果，但读/缓存仍计费 | 无本窗最终flow，需要新增廉价头或上窗状态 |
| 结构改动 | 从后部细化开始，训练耦合较局部 | 改动前端会影响整个后继表征 |
| 名义分母 | 16.008192G，2.683% | 两层127.401984G，21.357% |
| 激活初筛 | 输入非零约15.4003%；MAC×density约2.465G | r0conv2约3.6127%；MAC×density约2.301G |
| 状态 | 空间较小；多相散射、连续部分和仍需计费 | 全T96通道240×320状态大 |
| 质量风险 | 粗流可能漏小物体/边界，需要保留低证据不确定区 | 可能过早删除使后继不可恢复的线索 |
| 推荐位置 | 任务原生有损执行首探针 | 无损背景支持与较大规模后续扩展 |

MAC×density只作需要重新查看decoder2的初筛，忽略padding有效tap、发放位置、端口和利用率，不是活跃AAC严格计数，更不是周期。仅消除整个decoder2在名义MAC上的上限约1.0276×；任何更大的系统收益必须来自额外消费者/存储实际消失的证据。

最强简单控制：上采样preds.1直接早退；均匀降宽decoder2；普通2:4/3:4；仅f1梯度mask；WaveletVFI式跨尺度高频阈值mask；仅事件密度mask；纯不确定性mask；普通T10/group预算稀疏训练。对照保持同样的BN、PSN、转置卷积相位排程与位宽，避免把工程修复只给新方法。

第一轮完整RTL应覆盖：preds.1和当前事件摘要读取→可部署mask/阈值→T10与各相位请求组闭包→decoder2 sn→实际deconv取数/部分和合并→norm→preds.2 sn/1×1读出→粗流底图合并和最终写出；还要包括mask失效时的dense fallback、FIFO背压、边界和整窗完成。只测一相deconv、跳过norm或离线mask均不能证明此路线净受益。

### 4.2. C1-r0：粗密细疏前端作为扩展挂点

**强 A：**EEMFlow 的稀疏运动场/细节补偿、EDCFlow 的粗细运动互补、hARMS 的孔径约束。**昂贵 B：**r0 两个高分辨率 96×96×3×3 对全部位置执行；低激活非零率不等同于低整块服务量。**待验证 X：**在进入这两层前，利用现有 voxel 的时间/极性支撑和一个廉价全图分支，生成“保留二维运动可观测性”的硬执行集合；所有支路与时间依赖闭合后再让物理事务消失。

### 算子改动

设 r0 输入为 X，原残差支路为 R(X)。新结构可写成

`Y = X + U(C(D(X))) + Scatter_M(R_local(X; halo(M)))`。

D 是空间降采样，C 是较小通道的全图上下文分支，U 为投影/上采样；R_local 为针对所选位置训练的细节残差。应先采用 24/48 通道、1/2 或 1/4 空间分辨率等少数结构控制，而不是同时搜索几十种网络。公式描述有损的新算子，并不等于原 R(X) 的代数分解。

M 不能只是事件密度阈值。建议廉价支路输出粗运动和误差预测，结合 voxel 时间重心变化、极性分布、局部方向一致性，形成三类优先集合：运动不连续/细边界、低证据但高不确定性区域、快速变化或新生支撑。低证据且低不确定性的平滑内部允许主要由粗分支承担。边界梯度来自部署可得的粗预测，不来自真实 flow。

局部时间重心平面只能作为特征，不能伪装成精确法向流：voxel 是事件计数/插值，不是亮度场；其导数不直接满足亮度恒常方程。若改用原始 time surface 拟合，必须把保留时间戳、拟合和置信度计算费用纳入接口。

### 掩码、状态与训练

先用跨 T 共享的 8×8 tile 掩码，避免 PSN 输入按 bin 断裂；允许晚些时候验证更细粒度。两个 3×3 的目标 tile 需两圈源 halo。独立处理一个 8×8 内核会对应至少 12×12 输入覆盖，面积比 2.25；相邻 tile 必须合并以减少重复取数。细分支在 halo 区需计算足够中间结果，不能只在输出 M 内算第一层。

在 240×320 上，8×8 tile 共1200个，跨T掩码150字节；逐T掩码1500字节。一个 [10,96,8,8] 的16-bit中间块为122880字节；[10,96,12,12] halo 源为276480字节。可以按通道/空间流式削减，但需保留PSN的全T输入或等价中间量，不能只报掩码很小。粗分支和最终合并仍可能是稠密连续张量，必须计入写回及后继读取。

本路线需要训练或至少充分微调。推荐直接监督最终 flow，同时约束硬掩码预算、事件边界/运动边界误差、无事件有效像素误差与时间连续性。对完整T的软代理训练可用于梯度，但部署必须执行硬选择。通过缩减训练误差不能证明 mask 避免了内存事务；硬执行轨迹应单独检查。

### 最强简单控制与首轮完整数据路

必须比较：原始 bit-skip r0；相同预算的均匀缩通道/降低前端分辨率；普通深度可分离或group卷积；仅事件密度mask；仅廉价不确定性mask；同宽同端口的2:4/3:4；下述C2无损支持对照。所有比较保持相同出口、相同flow协议与训练预算。GT/oracle mask 只允许诊断上限，不能作部署结果。

首个 RTL 验证边界应为：voxel摘要/粗分支输入读取→可部署mask计算→tile合并与halo地址生成→sn1→conv1→norm1→sn2全T→conv2→norm2→与粗分支/identity合并→输出封包和r1首消费者接收。包括mask旁路、dense fallback、FIFO背压、边界tile和全部完成条件；禁止预先从离线文件喂“正确mask”后只测MAC核。

判退条件不是某一次 AEE 变差，而是明确机制失败：在同质量下，mask/halo/粗分支/密集回填成本超过省下的bit-skip服务；或二维运动歧义区域必须扩到几乎全图。若只在固定小样本靠过拟合成立，也不足以推进硬件主张。

## 5. 候选 C2：背景响应＋偏差支持的无损前端

**强 A：**AsyNet 的活动更新/状态与 MotionDeltaCNN 的bias、非线性和halo处理。**B：**已有源仅约4%非零，仍可能为大量无变化背景重复执行卷积后处理和输出写回。**X：**以当前固定BN、非因果PSN、膜电位shortcut为明确契约，把隐式背景和偏差support一直传至真实消费者，而非强制所有背景实体化。

### 精确含义

对于某个已验证的背景输入 X₀，记 F 为完整算子链，预先取得 F(X₀)，实际输入写作 X₀+δX。卷积有

`Conv(X₀+δX) = Conv(X₀) + W*δX`。

固定BN给出背景 `a Conv(X₀)+b` 与偏差 `a(W*δX)`；非线性必须计算 `g(z₀+δz)−g(z₀)`，不能计算 `g(δz)`。PSN在每个位置把全部时间bin的偏差共同映射，空间支持不扩展，时间支持一般扩到全部T。残差输出的支持至少取 identity 支持与残差支路支持的并集。

**数值无损的稳妥起点是复用被证明输入完全相同的背景输出，受影响位置仍按原归约顺序计算。**浮点/定点中直接重排为“背景＋delta卷积”不自动bit-exact，存在舍入/饱和问题。可先只对整个感受野与背景完全一致的区域跳过，再逐步验证更激进的delta累加语义。

X₀不能不加说明就取r0输入全零。若从raw voxel空输入建立背景，必须经过前面的编码到达r0；背景可因padding出现边界类别，也可因时间偏置在各T不同。只缓存每通道一个常量可能不够。另一个较弱但简单的控制是直接以r0实际输入为起点检测全零/背景相同的感受野，省去对raw mask到r0的未经证明映射。

### 成本与瓶颈

一个完整 [10,96,240,320] 二电平源位图为9.216 MB（十进制）；16-bit完整连续状态为147.456 MB。背景码很小不能掩盖活动源、部分和、动态索引和残差状态。全局绝对地址至少27bit才能索引73728000个源元素，若每非零使用27bit地址，理论仅payload/地址比较已要求非零率低于约1/27；使用分层局部位图会改变结果。该算术说明格式需实测，并非建议用最差的全地址AER。

若以每空间位置“任一通道/T有偏差”为活动，4%逐元素非零仍可能使几乎所有空间位置活跃；support膨胀后更可能满图。必须测实际tile覆盖、run长度、halo重复率和bank冲突。此路线不能从profile单一density直接预测收益。

固定BN层可无损背景复用，动态BN则不行：全图均值/方差变化可能改变所有背景位置与阈值。可在下一阶段研究背景计数参与统计并更新背景码，但必须保留统计屏障及背景发放重判。第一版止于r0完整残差块，并把r1读取、隐式背景解码/实体化纳入消费者边界。

最强控制是普通bit-skip卷积＋零感受野/背景感受野检测＋dense输出；只有隐式背景贯穿消费者比这个简单版本更便宜，才有独立硬件价值。首轮RTL范围包含support扫描/更新、卷积halo、两级BN/PSN、shortcut、背景边界类别、输出协议和后继读取，不仅是一个mask扫描器。无需训练但需数值等价验证；若改变BN统计或删除非背景输出，则已转为有损路线。

## 6. 候选 C3：跨窗口搬运与按创新重算

**强 A：**IDNet-TID、MotionDeltaCNN、Flow-Based Visual Stream Compression。**B：**当前窗口整体重算r0且没有利用上一已完成窗口。**X：**把“连续事件窗口→粗运动缓存→当前证据检验→完整r0局部重算”做成明确的状态接口，针对voxel重分箱和非因果PSN设计刷新义务。

在窗口k完成后保存必要r0输入/输出或压缩特征、粗flow、时间边界和有效位。窗口k+1在前端前，仅用上一flow和当前voxel的廉价摘要预测缓存位置，检查支撑失配、极性/时间分布失配、遮挡/新显露、运动边界及缓存过期，决定重算区域。完整T10窗口仍需先到齐；本方案只有跨窗口因果性，不获得逐事件低延迟。

任意warp与卷积不交换。即使flow正确，细节插值、空间变化运动、PSN时间坐标重分箱也使“warp旧输出=当前输出”成为近似。严格无损版本只能在可证明的整数平移、相同时间映射、相同输入邻域、正确padding条件复用，并对其他区域重算；通用warp路线应标有损，需训练/蒸馏及定期刷新。对于新事件表示的学习，也需把传感器前端和T10时间原点一起训练或固定。

费用包括上一状态缓存、flow缓存、warp多点读取/插值、索引有效位、创新检验和重算halo。仅保存一层r0的16bit完整T特征就有147.456MB，若几层都存，片上存储很可能不可接受。应优先研究tile/cache-line驻留、低位宽或低分辨率上下文缓存；这些压缩本身也需质量与带宽对照。若使用本窗完整前端结果判断是否能跳过该前端，逻辑是循环的。

最强控制必须是相邻真实窗口的：无warp差分、全局整数平移补偿、固定周期keyframe复用、上窗flow搬运、独立廉价当前头；另比较“直接缩小网络”。统计warm-up、reset、急转弯/速度突变、昼夜变化和长序列漂移，给出平均及尾部代价。先前单时刻不对齐delta失败并未测试这些接口；相反，如果当前没有连续窗口和状态一致性协议，就不能把离散diverse10准确率作为跨帧证据。

首轮RTL范围应包含缓存写入/替换/有效位→flow及时间映射读取→warp地址/插值→当前摘要创新检测→halo与重算队列→r0两层和非线性→残差合并→下窗状态提交；至少跨多个窗口包含一次全刷新和一次新显露区域。该路线当前优先级低于C1/C2，因为新增状态费用最大且最近邻很强。

## 7. 尚缺的证据与可区分实验

| 证据 | 目前状态 | 能区分什么 |
|---|---|---|
| r0源逐元素非零率、调用形状 | 已有单帧profile | 说明大算子与bit-skip潜力，不说明tile/运动规律 |
| r0完整空间/T支持、halo后覆盖率 | 未测；NPZ仅640个采样patch行 | 决定C2是否还有整块可省；采样不能还原二维连通分布 |
| voxel事件密度与局部误差、边界/孔径关系 | 未测 | C1任务mask是否优于普通密度mask |
| 廉价粗分支的误差排序能力 | 未测 | 能否在完整前端之前选择细化位置 |
| 全T模式重复率/复用距离 | 未测 | H3能否胜过现有bit-skip和公共模式基线 |
| 连续窗口warp后差异与刷新率 | 未测 | C3是否值得支付缓存/warp费用 |
| 空事件背景沿真实部署链的响应 | 未测 | 固定BN覆盖、padding类别、PSN发放与残差义务 |
| C1/C2/C3最终AEE、边界AEE、无事件有效像素AEE | 未测 | 不能把已有r0低秩/N:M结果移植为这些候选结果 |
| 完整数据路周期/能耗/存储 | 未测 | 名义MAC不能替代物理事务和真实同资源比较 |

下一阶段应先取得跨场景的输入/激活空间支持与连续序列统计，再决定是否值得训练C1和实现RTL。评估必须固定有效像素mask与flow尺度，不把DSEC无标注区域当作零误差，也不把只在event pixels评价的分数与dense flow混用。C1的任务性增量由与均匀小网络、纯密度mask、纯不确定性mask的比较决定；C2的增量由与简单背景感受野检测比较决定；C3由与keyframe/普通warp缓存比较决定。

本报告只进行了文件阅读、原论文精读与形状/存储的代数推算；未运行训练、神经网络推理、RTL、综合或EDA。所有候选均为证据有边界的研究假说，不声称首创，也不预测录用。

## Sources

原论文主文已保存于本目录 `papers/`，精读位置及代码核验见 [source_table.md](source_table.md) 与 [sources.csv](sources.csv)。下列编号在正文就近引用；PDF页指下载文件物理页，印刷页单独注明。

[1] Daniel Gehrig, Antonio Loquercio, Konstantinos G. Derpanis, Davide Scaramuzza. [End-to-End Learning of Representations for Asynchronous Event-Based Data](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gehrig_End-to-End_Learning_of_Representations_for_Asynchronous_Event-Based_Data_ICCV_2019_paper.pdf#page=3). ICCV 2019，§3，pp.5635–5637。

[2] Nico Messikommer, Daniel Gehrig, Antonio Loquercio, Davide Scaramuzza. [Event-based Asynchronous Sparse Convolutional Networks](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530409.pdf#page=5). ECCV 2020，§3.2，PDF pp.5–8。

[3] Yilun Wu, Federico Paredes-Vallés, Guido C. H. E. de Croon. [Lightweight Event-based Optical Flow Estimation via Iterative Deblurring](https://pure.tudelft.nl/ws/portalfiles/portal/220695337/Lightweight_Event-based_Optical_Flow_Estimation_via_Iterative_Deblurring.pdf#page=4). ICRA 2024，May 13–17，§III，pp.14709–14711；下载件含前置页。

[4] Xinglong Luo, Ao Luo, Zhengning Wang, Chunyu Lin, Bing Zeng, Shuaicheng Liu. [Efficient Meshflow and Optical Flow Estimation from Event Cameras](https://openaccess.thecvf.com/content/CVPR2024/papers/Luo_Efficient_Meshflow_and_Optical_Flow_Estimation_from_Event_Cameras_CVPR_2024_paper.pdf#page=3). CVPR 2024，June，§3.2，pp.19200–19202。

[5] Daikun Liu, Lei Cheng, Teng Wang, Changyin Sun. [EDCFlow: Exploring Temporally Dense Difference Maps for Event-based Optical Flow Estimation](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_EDCFlow_Exploring_Temporally_Dense_Difference_Maps_for_Event-based_Optical_Flow_CVPR_2025_paper.pdf#page=3). CVPR 2025，June，§3.1–3.4，pp.1986–1987；arXiv上传2025-06-04。

[6] Pengjie Zhang, Lin Zhu, Xiao Wang, Lizhi Wang, Hua Huang. [EMatch: A Unified Framework for Event-based Optical Flow and Stereo Matching](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_EMatch_A_Unified_Framework_for_Event-based_Optical_Flow_and_Stereo_ICCV_2025_paper.pdf#page=5). ICCV 2025，October，§3.1–3.2，pp.5847–5849。

[7] Rui Hu, Song Wu, Wen Yang, Jinjian Wu. [From Contrast to Consistency: Rethinking Event-based Continuous-Time Optical Flow Estimation](https://openaccess.thecvf.com/content/CVPR2026/papers/Hu_From_Contrast_to_Consistency_Rethinking_Event-based_Continuous-Time_Optical_Flow_Estimation_CVPR_2026_paper.pdf#page=3). CVPR 2026，June，§3，pp.15127–15129；arXiv上传2026-05-25。

[8] Deepika Sharma, Shubham Negi, Trishit Dutta, Amogh Agrawal, Kaushik Roy. [SpiDR: A Reconfigurable Digital Compute-in-Memory Spiking Neural Network Accelerator for Event-based Perception](https://arxiv.org/pdf/2411.02854#page=2). arXiv:2411.02854，2024-11-05，§II–III，PDF pp.2–7。正式venue未核实，不能把模板页眉当期刊录用证据。

[9] Daniel C. Stumpp, Himanshu Akolkar, Alan D. George, Ryad B. Benosman. [hARMS: A Hardware Acceleration Architecture for Real-Time Event-Based Optical Flow](https://space.pitt.edu/sites/default/files/2024-10/hARMS_-A-Hardware-Acceleration-Architecture-for-Real-Time-Event-Based-Optical-Flow.pdf#page=3). IEEE Access 10，58181–58198，2022-05-13发表；DOI 10.1109/ACCESS.2022.3172396。

[10] Mathias Parger et al. [MotionDeltaCNN: Sparse CNN Inference of Frame Differences in Moving Camera Videos with Spherical Buffers and Padded Convolutions](https://openaccess.thecvf.com/content/ICCV2023/papers/Parger_MotionDeltaCNN_Sparse_CNN_Inference_of_Frame_Differences_in_Moving_Camera_ICCV_2023_paper.pdf#page=4). ICCV 2023，October，§3.1–3.7；arXiv初稿2022-10-18。

[11] Daniel C. Stumpp, Himanshu Akolkar, Alan D. George, Ryad Benosman. [Flow-Based Visual Stream Compression for Event Cameras](https://arxiv.org/pdf/2403.08086#page=4). 精读arXiv v1，2024-03-12，§IV，PDF pp.4–6；后续[IEEE IoT Journal 11(24):40229–40243](https://doi.org/10.1109/JIOT.2024.3450428)，2024-12-15，publisher提交的Crossref元数据已核；本轮未取得最终出版正文，技术判断依据v1。

[12] Jinghai Wang, Jilong Luo, Bo Li, Lingfeng Zhou, Zhiyi Yu, Shanlin Xiao. [ASNA-Flow: An Efficient Asynchronous Neuromorphic Accelerator for Real-Time Event-Based Optical Flow](https://ieeexplore.ieee.org/document/11142472/). IEEE TVLSI 33(12):3409–3422，2025-12，DOI 10.1109/TVLSI.2025.3600953。publisher提交的Crossref元数据已核，主文未取得；不计全文精读。


[13] Lingtong Kong, Boyuan Jiang, Donghao Luo, Wenqing Chu, Ying Tai, Chengjie Wang, Jie Yang. [Dynamic Frame Interpolation in Wavelet Domain](https://arxiv.org/pdf/2309.03508#page=6). IEEE TIP 2023，DOI 10.1109/TIP.2023.3315151；arXiv v1 2023-09-07、v2 2023-09-21。§III-C–D/Algorithm 1，PDF p.6；官方代码 [WaveletVFI](https://github.com/ltkong218/WaveletVFI)。
