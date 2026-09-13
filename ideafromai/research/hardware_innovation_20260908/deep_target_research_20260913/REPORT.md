# 昂贵算子的分解、事件光流结构与稀疏执行

## 1. 研究结论

目前最适合继续投入的对象是 **patch r0 的两个 96 通道 3×3 卷积，以及仍保留的 decoder2 细化链**。前者适合比较不同表示能否保住脉冲源的加法优势，后者已经拥有可用的粗流，适合研究任务引导的选择性计算。S2 FC1 已在当前执行路径中改造，不能继续拿旧网络的 FFN 份额说明它还是最大的未处理对象。

建议将三种不同问题分别推进，暂不合并成一座加速器：

| 方向 | 具体挂点 | 待验证的新增机制 | 当前判断 |
|---|---|---|---|
| 分解 | r0.conv2，随后迁移 r0.conv1 | 二电平输入的有限整数 Winograd 码，与变换域结构剪枝的消费者需求共同生成执行请求 | 三条中分解接口最具体；须击败已有范围优化与稀疏 Winograd，尚未证明新颖性净增量 |
| 剪枝、稀疏与打包 | r0.conv2 的 K4 权重组和真实源位字 | 联合选择剪枝支撑与 lane 执行，使尾部工作量、源字并集和部分和冲突一起减少 | 已有普通 2:4 质量对照，可最先做完整 RTL 比较；独立 lane 本身是先验 |
| 光流任务结构 | decoder2 → norm → preds.2 | 利用已有 preds.1 和事件可观测性，保留需要细化的区域，并使完整 T10 / 多相请求组停止执行 | 最贴任务，需训练；粗细分解、mask 和 halo 已有先例，任务选择与物理费用的共同收益才是候选 X |

lifting 保留为实现和质量对照。普通快变换、低秩、CSE、广播、缓存不再单独承担标题贡献。上述三条也不是已经成立的投稿贡献：当前新增结果是具体挂点、可实现的函数与强对照；没有新增网络性能或 RTL 周期结果。二值 Winograd 的有限取值集合做了完整小规模代数穷举，属于函数核验，不属于硬件加速测量。

## 2. 昂贵计算的实际位置

### 2.1 当前分母

当前 profile 来自 matched-dense stage320 的一次真实前向，输入为 `zurich_city_09_a_0001.npy`，出口为已有粗头 `preds.2`。170 条 ATen 卷积/矩阵调用合计 596.5464288 G 名义 MAC；没有计入布尔 Motion-XOR、非矩阵逐元素运算、NumPy BN、访存与布局。因此下面是**选取测量对象的算术表，不是芯片时间占比表**。[^1]

| 算子 | 名义 G MAC | 名义份额 | 实际左输入非零率 | 为什么值得做 |
|---|---:|---:|---:|---|
| patch r0.conv1 | 63.7010 | 10.678% | 4.3602% | 未改造的大卷积；分解要与原有非零常量加法比 |
| patch r0.conv2 | 63.7010 | 10.678% | 3.6127% | 已有真实 W、输入与低秩/N:M 质量对照，最容易形成公平比较 |
| patch 首个卷积 | 31.8505 | 5.339% | 9.8937% | 更靠近事件输入，空间支撑和表示改变的潜在收益更大 |
| decoder2 | 16.0082 | 2.683% | 15.4003% | 输入明显更密，且已有 preds.1 粗流可使用 |
| decoder1 | 15.9667 | 2.677% | 15.1700% | decoder2 路线成立后的迁移点 |
| patch.proj | 15.9252 | 2.670% | 0.9521% | 输入稀疏，但动态统计和后继状态容易主导净收益 |

r0 两个卷积合计占上述名义算术约 21.36%。当前 patch 合计 236.814336 G，FFN 合计 127.401984 G；它们不是旧 ep34 的周期信封。S2 FC1 已变为每块 `[1200,6,384]×[1200,384,1536]` 的调用，六块共 25.4803968 G，不能把原始 dense FC1 当作当前待消掉的工作量。[^1]

转置卷积的计数程序实际使用 `input.numel()×Cout×Kh×Kw`，已经按原始输入 scatter 范围计数，未把 stride 2 插入的零再算一遍；尚未扣除边界裁剪。decoder2 的 16.008 G 不能再凭“避免插零”变成四分之一。作为进一步定位的粗估，decoder2 的非零输入扇出约 2.465 G 项，r0.conv2 约 2.301 G 项；空间位置、有效 tap、权重零项、端口和利用率都还没有计入，不能把两者说成相同执行时间。[^1]

### 2.2 二电平为什么改变分解选择

AT-LIF 输出是 `{0,θ}`，静态 θ 可吸入下一层权重。r0 的一条输出可写成

\[
y_o=b_o+\sum_{k=1}^{864}\overline W_{o,k}s_k,\quad s_k\in\{0,1\}.
\]

原执行需要的是非零常量加法。普通低秩将其写为 `z=Vs, y=Uz`，虽然减少参数和名义乘法数，却把 z 变成较密的连续量，后因子重新需要一般乘法与较宽中间状态。既有真实捕获已经出现这种代价，不能用 dense MAC 数的下降替代与 bit-skip 的比较。[^2]

纯 SVD、空间分解和 Tucker 并未因为质量而整体出局：已有 diverse10 中，纯 SVD R8 为 1.347965、空间 R16 为 1.269130、Tucker R8 为 1.417015，均优于本地 NB0 的 1.454603。问题在于它们尚未给出强硬件增量，也不能把常见矩阵分解本身当作新颖性。NB0 使用原 final head 和 FP32 reduction，当前候选使用粗头和 FP64 reduction；这是同样本、同 GT 的任务质量比较，不是完全相同网络执行或位级协议。最终仍需完整 valid825 优于 NB0，十帧只是筛选。[^2]

## 3. 如何沿 GustavSNN 的研究路径推进

Prosperity 利用重复或子集 spike 模式共享部分和；GustavSNN 采用 Gustavson 型列并行时间批处理、稀疏格式与局部状态。它们共享部分 SNN 时间批处理背景，但不能把后者描述为在 Prosperity 的 product-sparsity 内核上直接升级。可借的方法论是：**完整实现一个强底座，找到它在具体工作负载上仍付出的费用，再改变表示或执行接口。**[^3][^4]

现有 Prosperity 官方模拟器已经跑过完整算子的迁移对照；这值得保留。先前父值提升、局部共享和静态融合的负结果只限制对应布局。GustavSNN 的本地工作包含 paper-guided 模型及功能切片，尚非原作完整系统复现；NRV、索引交集、供数、局部状态、跨 PE 同步不能只选其中广播一项便称“抄全”。[^2][^4]

这三条新方向中的 A 与剩余问题分别是：

| 强底座 A | A 优化后仍可能昂贵的 B | 候选增量必须改变什么 |
|---|---|---|
| WINS + 完整数字稀疏 Winograd | 变换把少量事件扩展成更多坐标；系数/中间码读取与逆变换仍在 | 同时利用二电平有限幅值和被剪消费者，改变实际请求与完成时间 |
| 普通 N:M + 有真实队列/部分和口的稀疏执行 | 权重稀疏不保证源字少读；公共遍历和最慢 lane 可钉住整组 | 剪枝支撑与具体执行队列共同选择，而不是只减权重个数 |
| WaveletVFI / EEMFlow 式粗细分工 + 去插零转置卷积 | 平滑位置仍可能执行同样细化；无事件区域又不能盲目删除 | 根据事件可观测性安排细化，并让完整物理请求组消失 |

上述 B 不是证明某篇论文“做得不好”，而是它的原任务和本网络之间需要实际测量的差异。借入 A 的已有贡献、原论文自己的性能、以及本地 X 的增量必须分开报告。

## 4. 分解方向：有限整数 Winograd 与变换域剪枝

### 4.1 完整 A

标准 F(2×2,3×3) 将一个 4×4 输入块变换到 16 个坐标，经各通道归约后逆变换出 2×2 输出。WINS 在各变换坐标的矩阵上剪整行或整列，并提供各坐标剪枝率均衡、逐层选择等策略。这些算法及融合权限都应保留；只比较未优化 dense Winograd 会制造弱分母。[^5][^6]

### 4.2 本网可利用的代数结构

对二电平支撑块 S，令 `V=BᵀSB`。标准 F(2,3) 中，15 个坐标只能取 `{-2,-1,0,1,2}`；另一个坐标只能取 `{0,1,2,3,4}`。穷举全部 65,536 个二值 4×4 块确认了这一集合。对预折常量 U，除值 3 需要 `U+2U` 外，非零乘积都可由一次带符号、带移位的常量累加表达。这个结论来自输入域与变换矩阵，不要求将连续值再量化成二值。[^7]

拟研究的执行为：从原始位块产生有效/符号/倍数码，与当前输出块的 WINS 消费者 mask 相交，再决定哪些系数字需要读取和更新哪些部分和。希望避免连续 V 张量物化，并消掉没有消费者或整数抵消后的请求。

**候选 X 不是“Winograd + SNN”或位宽缩窄。**完整基线也必须获得相同取值范围、零跳过、融合、码缓存与编译优化。如果普通有限域 WINS 获得这些权限后已经做出相同请求流，这个 X 就不成立。可能有意义的差异是：表示、消费者 mask 与物理系数字一起组织，减少基线仍支付的访问或串行阶段；目前未证明。

### 4.3 性能最强反对

在约 3.6% 的输入发放率下，无剪枝 Winograd 可能比直接 event AAC 多做加权项。独立 Bernoulli 模型只作直觉推算：每输入/输出通道的一个输出 tile，直接卷积约 `36p` 项，而变换域有限幅值展开约 2.1 项，相比直接约 1.3 项更高；还未计变换税。这个模型不是实测规律，但说明不能借用 dense 场景的 2.25× 倍率。[^7]

核表示还由每对通道 9 项变为 16 项；输入重叠 gather、mask 元数据、guard bits、逆变换和后继全部要支付。未经剪枝的实现失败，只能说明该端点没有收益，不能代替对 WINS 训练后模型的判断。

WINS 任意剪掉变换坐标后，所得函数通常不再等于某个普通 3×3 核。严格同函数的直接对照应展开成 4×16 的 tile 线性算子；原 3×3 bit-skip 和普通 2:4 则是同任务质量对照。两种对照缺一不可，否则会把函数变化误作数据流胜出。[^7]

### 4.4 第一份 RTL 应交什么

单坐标或 C8/O8 单元只作功能调试。**第一张可比较的性能表必须覆盖完整 C96、O96、T10，至少相邻输出 tile 的实际重叠取数、全部通道归约和完整输出**，并在相同源口、系数口、状态预算、背压下比较三条执行：普通有限域 WINS、候选联合请求流、同函数直接 tile 算子。全层回放再确认重叠缓存和边界没有被小 tile 隐去。

它首先回答“源表示和剪枝需求相交到底是否减少周期”，随后才回答“该模型是否在同 AEE 下优于普通 2:4 或低秩”。没有第一项，不能用参数下降或数值 PASS 代替硬件创新。

## 5. 稀疏方向：剪枝支撑与 lane 完成费用共同设计

### 5.1 为什么普通 2:4 是分母

已有 r0 的普通 2:4 diverse10 AEE 为 1.159183，质量并不弱；一些公共 pair 和 fill 方案在有限服务模型中反而比普通 2:4 更贵。公共源遍历、重复解码和元数据已经暴露出问题，但这些模型尚非 RTL。[^2]

Bishop 已研究时间打包与误差约束剪枝，LoAS 已有完整时间并行的双稀疏格式、索引交集、伪和与校正；HighLight、S2TA、Eyeriss v2 已处理结构化稀疏、压缩格式、局部存储和执行利用率。独立 lane、FIFO、优先编码器和 N:M 均不能重新申领为标题贡献。[^8]

### 5.2 拟改变的执行接口

以一个 K4 权重组为执行 epoch：实际权重载入后，8 个 lane 各自遍历其 P2×T10 的活跃事件，选择自己的输出部分和地址。每 lane 保持在已加载的权重和明确的 K4 范围内，结束后才能进入下一组。这样可以实际实现独立完成，而不是在 CPU 模型里免费将公共执行次数换成 `max(events)`。

剪枝支撑的候选目标同时考虑：输出误差、真实源字并集、最慢 lane 的事件数、系数请求、部分和 bank 冲突和尾部排空。先在**同一份独立 lane RTL**上比较传统幅值/Gram 选择、既有 request 选择、联合费用选择。若新目标只在公共 walker 的弱控制下有优势，不能算新增机制。

可以将训练/离线选择的目标概括为

\[
L=L_{flow}+\lambda_1\widehat{C}_{word}+\lambda_2\widehat{C}_{tail}+\lambda_3\widehat{C}_{conflict}.
\]

这些费用代理需要对应真实地址和调度，不把一般“硬件感知训练”当新。最重要的预测是：相同非零数量、相近 AEE 下，新支撑是否让**整层**完成时间下降；减少局部误差或加法个数并不够。

### 5.3 新颖性判断与首 RTL

这是三条中最容易先验证、也最容易只剩工程改进的一条。它有清楚的本地 B 和可用质量对照，但先验碰撞强。第一轮保留它，是为了把尚未实现的物理费用真正测出来，而非已经决定将它写为主创新。

第一份性能 RTL 应由原生源位字开始，包含真实 K4 支撑、系数请求、lane 选择、部分和读改写与背压，完成 r0.conv2 的 K864/N96/T10 全层输出。norm2 和 identity 合并是明确消费者检查点，不能让 C++ 偷做输出 Y 后仍称完整岛。若先分层实现，则各边界的周期与尚未接入部分分开列；对外不能称已完成完整层加速。

## 6. 光流方向：事件可观测性决定 decoder2 细化

### 6.1 这次可以用哪些任务信息

当前 voxel 提供时间 bin、空间位置和分离的极性聚合量，可导出局部活动、时间分布和方向线索；它不能恢复任意原始微秒时间戳。当前窗口内的最终 flow 不能反过来指导同一前端执行，但 decoder2 之前已经有 `preds.1`，可合法用于下一层选择。[^9]

“没有事件”不等于“运动为零”或“粗流可信”。EMatch 明确保留无事件像素所需的空间上下文；hARMS 的局部边缘流也面临孔径问题。新生运动、遮挡、低对比度与边界附近，单靠低活动率删计算可能恰好丢掉最需要的细节。[^9]

### 6.2 完整 A 与待试 X

WaveletVFI 已实现运动感知、动态阈值和逐尺度稀疏细节，包含前置 mask 膨胀；EEMFlow 已把较平滑的 meshflow 与细节补偿区分开。它们不是空白，也不能把软置信度融合直接算成少执行。转置卷积的四相位拆分及去插零同样已有硬件先例。[^10][^11]

可训练的新算子形式是

\[
f_2=U(f_1)+\operatorname{Scatter}_{M}(\Delta f_2),
\]

其中全图底图始终存在，M 由已完成的粗流、事件可观测性和廉价不确定性产生。M 不只追踪活动最多的位置，还要保留“事件少但预测不可信”的区域。训练既约束运动质量，也约束 M 扩展到完整 T10、各输出相位及实际请求组之后的服务费用。

**候选 X 是选择依据和执行粒度的共同设计**：事件可观测性是否比普通高频/flow 梯度 mask 更会分配细化预算，且这种分配是否真的消掉物理请求。仅给已有稀疏解码器接上 PSN、BN 和地址生成器，是集成义务，不足以证明创新。

### 6.3 消费者和最强反例

真实 MS decoder 为 `sn→deconv→norm`，后继 pred 为 `sn→1×1`。前一 flow 虽参与拼接，却已经经过 sn 才进入 deconv；不能构造“连续 flow 与二值 spike 双 MAC”的假对象。两段卷积间存在非线性，不能直接把 96 通道投影压成 2 通道并宣称无损。具体运行实例的 decoder BN 模式尚须核验。[^12]

最强简单控制首先是 **直接上采样 preds.1 早退**，其次是均匀缩窄 decoder2、普通 2:4、单用 flow 梯度、单用事件密度、单用不确定性、WaveletVFI 式阈值。若复杂控制比简单早退多付状态却没有更好的质量—费用曲线，应停该控制器。若有效 mask 经过多相位、邻域和 T10 并集后几乎满图，则必须修改执行粒度或表示，不能继续报告原始 mask 的稀疏率。

第一份 RTL 包含 mask 的实际生成、输入支持反推、各相位地址、所有源通道归约、norm、pred 神经元与最终 flow 写出。离线 GT mask、当前最终 flow mask、预先给出完成时刻的测试叶，均不能回答这一机制是否有效。该方向相对原网络有损，需要真实完整前向和训练；不能继承现有低秩模型的 AEE。

## 7. 其余适配家族的去向

| 家族 | 目前去向 | 不应混淆的理由或下一接口 |
|---|---|---|
| Prosperity / ProSparsity | 强执行对照继续保留 | 完整子集复用本身已有；旧融合失败不等于家族失败。后续只试它尚未消掉的源/表示/消费者费用 |
| GustavSNN | 供数和执行底座继续保留 | 完整 CPTB/NRV/局部状态/同步须逐项迁移，不能恢复为单纯广播标题 |
| lifting / CSE / PoT | 实现、质量与编译对照 | 已有进展可复用，但当前没有足够性能和差分支撑主标题 |
| 纯低秩、空间分解、Tucker | 质量过门的强控制 | 不是因 +0.005 被杀；主要要支付连续 latent 与后因子，常见分解本身新颖性低 |
| SmartExchange / StrassenNets / UCNN 融合 | 第二分解挑战者 | 源先变有限整数桶再消费常量；分桶和先加后乘已有，须证明全输出共享映射与真正请求减少。未完成新拟合/AEE/RTL |
| Kronecker + FastKron | 保留一个有界迁移点 | 支撑分层收缩可能免空/单事件 latent；大收缩块通常趋密。尚未试不能写不可行，也不把压参数直接当加速 |
| LUT / 差分表 / 支撑 DAG | 强先验与存储挑战者 | 小组时多事件合并有限，大组时表指数变大；必须计表容量和强直接事件对照 |
| 背景响应 + 偏差支持 | r0 无损控制及扩展接口 | 固定 BN 可复用相同背景响应；空间支持并集和 halo 未测，4% 位密度不说明整块空。动态 BN 不能免费越过 |
| 跨窗口运动搬运 + 创新重算 | 系统改造挑战者 | MotionDeltaCNN、IDNet-TID 和事件流预测均为强 A；需连续窗口、缓存、warp、刷新与漂移验证 |
| CGNet 式 prefix / 整组近似跳过 | 保留有损候选 | 必须由真实已执行前缀决定；r0.conv2 后是连续 norm+identity，不能只判某个 spike 门就声称输出无损完成 |
| H8 pair / fill / 静态时间共享 | 停已测无优势布局 | 不是整族稀疏被否决；目前未打赢普通同资源控制，不以新名续做同一端点 |
| Motion-XOR / K=0 / 行内 memo | 注意力旁路 | 可以继续利用合法输出跳过；分数分母和任务份额限制其主线价值，不代替大算子工作 |

分解挑战者的数学接口、各论文已覆盖部分及第一份验证范围见[分解专篇](decomposition/REPORT.md)；稀疏格式、剪枝及打包见[稀疏专篇](sparse/mechanism_report.md)；事件表示与任务路线见[光流专篇](flow/flow_task_native_report.md)。这些材料区分方法精读、仅摘要、未找到公开实现和未实验，不将已有大目录的全部条目冒充本轮全文复现。

## 8. 下一阶段的执行顺序

**先完成能公平比较的直接稀疏 RTL，同时准备分解与任务模型；不再让 CPU 服务表代替 RTL 筛选。**第一阶段保留三条候选，具体先后如下。

1. **r0.conv2 直接稀疏控制与联合支撑。**沿已有真实 W/输入和普通 2:4 质量对照，完成源字到全层输出的 RTL。比较公共执行与付费的 lane 独立执行，再在相同 lane RTL 上比较支撑选择。交付实际拍数、请求、状态和停顿；新颖性由最后一项差分判断。
2. **有限整数 Winograd。**复用直接控制、缓存和消费者；先完成准确的普通有限域 A 与同函数直接展开，再叠 WINS mask 的联合请求。并行只训练一组明确 WINS 表示，测完整前向；不能因无剪枝 Winograd 变慢就取消剪枝后的接口，也不在没有目标函数的情况下做端口扫参。
3. **decoder2 的任务细化。**先比较 preds.1 早退、普通阈值和事件可观测性选择，并保留均匀小解码器。同步建立实际多相支持生成与 finite-buffer 执行；较小功能模块可先写，但新候选的性能结论必须包含 mask 和完整消费者，不要求先有最终有效率才允许编写原型。

每条只保留一组明确候选函数和最强控制。失败记录指出失败的是表示、控制器、质量还是端口，不因单一布局负结果关闭未测试接口。出现明确差分后，才把该条扩到多层、多场景并推进同资源 VCS、DC/PT、Formality 与必要的物理实现。Verilator 周期只能写成相应仿真结果；逻辑综合不代替含真实 SRAM 的 ASIC PPA。

精度探索以优于本地 SDformerFlow NB0 为门，不恢复 +0.005 限制。最终验证还应报告运动边界、无事件有效像素和困难场景，防止平均 AEE 掩盖任务性删减的代价。候选比较不同网络函数时，同时列质量、服务和状态预算，不能要求所有有损模型位级等同原网络，也不能取消固定候选与硬件实现之间的数值核验。

## 来源

[^1]: 本地 [profile.json](../open_fusion_execution/major_operator_fusions_20260913/root_owned/profile.json)、[profile_current.py](../open_fusion_execution/major_operator_fusions_20260913/root_owned/profile_current.py)、[current_bottlenecks.md](../open_fusion_execution/major_operator_fusions_20260913/root_owned/current_bottlenecks.md)。单帧真实前向调用与形状，不是周期测量。
[^2]: 本地 [major_operator_fusions 阶段报告](../open_fusion_execution/major_operator_fusions_20260913/README.md)、[diverse10 汇总](../open_fusion_execution/major_operator_fusions_20260913/root_owned/aee_all.md)、[Prosperity 完整算子对照](../open_fusion_execution/major_operator_fusions_20260913/prosperity_owned/README.md)。
[^3]: [Prosperity: Accelerating Spiking Neural Networks through Product Sparsity](https://arxiv.org/html/2503.03379v1)，HPCA 2025；[作者公开工件](https://github.com/dubcyfor3/Prosperity)。
[^4]: [GustavSNN: Unleashing the Power of Gustavson's Algorithm on SNN Acceleration with Column-Parallel Tick-Batch Dataflow](https://doi.org/10.1109/HPCA68181.2026.11408587)，HPCA 2026；迁移边界见本地 [implementation status](../psn/gustavsnn_implementation_status_20260908.md)。
[^5]: Andrew Lavin, Scott Gray. [Fast Algorithms for Convolutional Neural Networks](https://openaccess.thecvf.com/content_cvpr_2016/papers/Lavin_Fast_Algorithms_for_CVPR_2016_paper.pdf)，CVPR 2016，§4.1、Eq.5–13、Algorithm 1。
[^6]: [WINS: Winograd Structured Pruning for Fast Winograd Convolution](https://openaccess.thecvf.com/content/ICCV2025/papers/Park_WINS_Winograd_Structured_Pruning_for_Fast_Winograd_Convolution_ICCV_2025_paper.pdf)，ICCV 2025，§4–5；原作者与阅读页码见[来源表](decomposition/source_table.csv)。
[^7]: 本地代数推导：[分解专篇 §4](decomposition/REPORT.md)、[穷举程序](decomposition/verify_winograd_alphabet.py)、[65,536 输入结果](decomposition/winograd_binary_exhaustive.json)。有限字母表及同函数展开是数学核验；不构成文献空白或硬件性能证据。
[^8]: [Bishop](https://arxiv.org/pdf/2505.12281v1)，ISCA 2025；[LoAS](https://arxiv.org/pdf/2407.14073v3)，MICRO 2024；[HighLight](https://arxiv.org/pdf/2305.12718)，MICRO 2023；[S2TA](https://arxiv.org/pdf/2107.07983v2)，HPCA 2022；[Eyeriss v2](https://arxiv.org/pdf/1807.07928v2)，IEEE JETCAS 2019。逐篇方法位置、公开实现和限制见[来源表](sparse/source_master.csv)。
[^9]: [EMatch](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_EMatch_A_Unified_Framework_for_Event-based_Optical_Flow_and_Stereo_ICCV_2025_paper.pdf)，ICCV 2025，§3.2；[hARMS](https://space.pitt.edu/sites/default/files/2024-10/hARMS_-A-Hardware-Acceleration-Architecture-for-Real-Time-Event-Based-Optical-Flow.pdf)，IEEE Access 2022，§II–IV；[事件表示 EST](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gehrig_End-to-End_Learning_of_Representations_for_Asynchronous_Event-Based_Data_ICCV_2019_paper.pdf)，ICCV 2019，§3。当前输入接口核对见[光流专篇](flow/flow_task_native_report.md)。
[^10]: Lingtong Kong et al. [Dynamic Frame Interpolation in Wavelet Domain](https://arxiv.org/abs/2309.03508)，IEEE TIP 2023，§III-C、Algorithm 1；[官方代码](https://github.com/ltkong218/WaveletVFI)。Xinglong Luo et al. [Efficient Meshflow and Optical Flow Estimation from Event Cameras](https://openaccess.thecvf.com/content/CVPR2024/papers/Luo_Efficient_Meshflow_and_Optical_Flow_Estimation_from_Event_Cameras_CVPR_2024_paper.pdf)，CVPR 2024，§3.2。
[^11]: K.-W. Chang, T.-S. Chang. [Efficient Accelerator for Dilated and Transposed Convolution with Decomposition](https://arxiv.org/abs/2205.02103)，ISCAS 2020，§II-C/D；Lois Orosa et al. [EcoFlow](https://arxiv.org/abs/2202.02310)，作者全文 §4；期刊版本 DOI [10.1109/TC.2023.3272282](https://doi.org/10.1109/TC.2023.3272282)，[sasiml 官方代码及分发模型限制](https://github.com/CMU-SAFARI/sasiml)。
[^12]: 本地 [decoder2 源码和挂点核查](integration/DECODER_TARGET.md)，包含真实 MS decoder / pred 的顺序与源文件链接。
