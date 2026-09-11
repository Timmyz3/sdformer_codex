# R3Q05 — Event-OF / patch r1 producer support（independent）

**问：** 事件相机二维光流 SNN-Transformer 上，哪一个机制能让 **patch r1 生产者支持集** 在硬件上付钱，且 **不用当前帧光流神谕**、**不宣称首个 event-OF 芯片**？

**身份：** AT-LIF \(\{0,\theta\}\)；推理 \(\theta\) 吸入下一层 \(W\)；脉冲路径是 0/1 GeMM。残差 / PED / I24 是另一条连续张量。双消费者 = **吸完后的二值门** ∪ **连续 PED**。不得写成“连续 AT-LIF 幅值被两个 MAC 共用”。

**门：** 同端口净服务 ≥15%；AEE 对 ordinary **1.219801338**（绝对 ≤1.259，相对增量 ≤0.005）。BN 是卫生项：全域人口。Delayed-V 等到 native BN 完成：integer 0-diff、`arithmetic_saving=0`、`native_BN_reduction_hardware_closed=false`；early_V96=55.3MB vs late_U32=18.4MB 的总线占用释放 **不是净服务**。Preview-V 微核 ordinary 角 10396 slot（8 FP32 FMA lane）。

**每卡只许一个机制。** Status: **candidate，无打分。** 不把 Prosperity/Gustav 当标题；不把 typed last-use 当标题；不把 BN reduction 当已闭合硬件。

**整题必须抄全的 A（不是本岛 X）：**

- SENECA FireNet 已是 event-OF 硬件。
- ASNA-Flow 摘要已宣称 spatial-locality sparse OF computing。
- EventShiftFlow 已是 occupancy-grid FPGA。
- ERAFT FPGA 是 **帧 RAFT**，不是 event-SNN。
- Spike-FlowNet 把残差当 SNN 会恶化 AEE → 禁止残差脉冲化。
- SciFlow 的 SCI 是 **同帧光流神谕** → 本岛禁用。
- BAT 只用过去事件、因果 → “只用过去事件估流”不是 X。
- 输入 event-count mask 是廉价 A。
- 非因果 T10 **不能** 套 SENECA-ANN 三行释放。
- DATE hybrid dense/sparse、FireFly-T overlay、Spike-IAND T-unroll、SpikePool、FlexSpIM 是映射/核形态 A；问它们是否覆盖 **吸 θ 之后** 二值 GeMM + 连续 PED 的双 last-use。

---

## R3Q05-I1 隐层 r1 支持钟控二值 GeMM，不碰输入占用栅

**一个机制：** 在 stem 之后、θ 吸入 \(W\) 之后，用 **patch r1 学到的脉冲支持集** 作为 0/1 GeMM tile 的 issue/clock-gate；输入平面仍按 event-count/occupancy 的廉价 A 处理，不把 r1 支持写成传感器占用。

**为何可能付钱：** 历史账上 patch 约 34.8%、r1 两卷积约 11%；若同端口分母落在 r1 子链，空 tile 不再上 0/1 阵列，有机会碰到 15%。付钱对象是 **隐层二值 GeMM**，不是输入事件计数。

**不是首个 event-OF HW：** SENECA/ASNA/EventShiftFlow 已存在。ASNA 的空间局部稀疏 OF、EventShiftFlow 的 occupancy grid 抄作输入侧 A。本机制若退化成“事件多为 0 就跳 MAC”，立刻降成 A。

**无神谕：** 支持来自 r1 AT-LIF 发放（或与发放同训的学生），不来自当前帧 flow / SCI。

**双消费者 / BN：** 连续 PED **保持稠密连续**（Spike-FlowNet 负结果）。跳过的只是二值 GeMM issue。全域 BN 仍对完整空间人口；不把 BN adder skip 算进 15%。Union：PED 活点不能被 GeMM 门误杀。

**杀死条件：** 二值 GeMM 在同端口链上份额 <15%；或 union 后可跳集合被 PED 填满；或 AEE 相对 1.219801338 超 +0.005。

---

## R3Q05-I2 用 r1 支持对 Transformer 做 token pack，而不是卷积 skip

**一个机制：** 把 r1 生产者支持集当成 SNN-Transformer 的 **变长 token 打包/解包**：只把活 patch 送进注意力/FFN，空 patch 不占序列位；T10 非因果，故一个 token 必须一次携带完整 10 行，禁止三行流式释放。

**为何可能付钱：** 注意力是 \(O(N^2)\) 或至少 \(O(N\cdot d)\)。\(N\) 从满 patch 降到 r1 活集，同端口 FMA/SRAM 流量可以过 15%，即使卷积干路因为 PED union 几乎不能跳。这是 OF 网上 Transformer 段的付钱点，不是 ASNA 式空间卷积稀疏。

**不是首个 event-OF HW：** FireFly-T 是 SNN-Transformer overlay A，不提供 OF 生产者打包。ERAFT FPGA 是帧 RAFT，不得写成 event-SNN 对照胜利。SENECA FireNet 仍是必须抄的 event-OF 硅。

**无神谕：** pack 谓词 = r1 事件/脉冲支持，不是当前帧光流幅度、不是 GT flow、不是 SCI。Event-count pack = EventShiftFlow occupancy A；必须证明 r1 学支持相对 count 仍改变 \(N\) 且过 AEE 门。

**双消费者 / BN：** 打包只作用于二值注意力/FFN GeMM。PED/残差在解包后的原网格上稠密写回。BN 全域。Pack/unpack 索引与对齐流量计入同端口分母。

**杀死条件：** pack 流量吃掉 15%；count-pack 已达同样 \(N\)；非因果 T10 令 token 过宽，抵消 \(N\) 下降。

---

## R3Q05-I3 过去事件只生产下一窗 r1 支持时刻表，不生产光流

**一个机制：** 只用 **已到达的过去事件** 预测 **下一计算窗** 的 patch r1 支持（issue 时刻表）。生产者本身因果；T10 PSN 仍可非因果，但 **不能** 用未来事件、不能用当前窗的完成光流去开下一窗的门。

**为何可能付钱：** 在事件到达前就决定哪些 r1 tile 不取 \(W\)、不上阵列，把空窗从同端口 issue 里拿掉。这是 **调度谓词**，不是 BAT 那种用过去事件直接估流。

**不是首个 event-OF HW：** BAT = 因果、只用过去事件做 OF，必须抄全；本岛若把“过去事件→flow”再写一遍则无 X。SENECA 三行释放仍不可用（T10 非因果）。A 是 BAT+SENECA+event-count；X 仅当谓词是 **支持时刻表** 且硬件少发 15% 同端口操作。

**无神谕：** 禁止 SciFlow 式“先看本帧 flow 再决定算哪”。监督是后继层真实 live-set 或任务 AEE，不是把 ordinary 的当前 flow 当免费先验。

**双消费者 / BN：** 时刻表必须覆盖 union（二值门 ∨ PED 需求）的上界，否则连续路径欠供。欠估计杀 AEE；过估计退回 count mask。BN 全域，不靠该时刻表改 reduction 硬件。

**杀死条件：** 不用当前事件就预测不准（退回 occupancy）；或与 BAT 骨干不可分；或同端口等待事件的气泡抵消 skip。

---

## R3Q05-I4 DATE 式双核：稀疏核只吃 r1 二值支持，稠密核专吃 PED

**一个机制：** 同一同端口资源点上拆成两条物理核：稀疏核只对 **吸 θ 后的 r1 支持** 做 0/1 GeMM；稠密核只做连续 PED/残差/I24。生产者支持集 **只调度稀疏核的 issue**，从不把 PED 改成脉冲。

**为何可能付钱：** DATE hybrid dense/sparse 已证明双核形态能存在；本网的洞是吸 θ 之后一条是二值 GeMM、一条是连续 PED，FireFly-T / FlexSpIM / SpikePool / Spike-IAND 是否覆盖这种 **类型分裂的双 last-use** 仍是未决议题。若稀疏核占同端口时槽 ≥15% 且其 issue 可被 r1 支持掐掉，净服务可以过门，即使稠密核满转。

**不是首个 event-OF HW：** 芯片叙事停在“把已有 hybrid 核接到 event-OF 学生”，SENECA/ASNA/EventShiftFlow 仍抄 A。禁止“首个 event-OF 加速器”。

**无神谕：** 稀疏核谓词 = r1 生产者支持，不是本帧 flow 图、不是 ERAFT 的帧相关。

**双消费者 / BN：** 这是 union 的硬件翻译：稀疏核可空，稠密核不可空。共享端口上稠密核背压计入分母。全域 BN 挂在稠密核之后；BN 未闭合 reduction 不得报 arithmetic_saving。Preview-V 10396 slot 若跑在稠密核，必须进同一张表。

**杀死条件：** 同端口被稠密 PED+BN 占满，稀疏核再省也不到 15%；或双核额外状态/系数预算上升。

---

## R3Q05-I5 光流头只在 r1 支持上解码，空穴插值；谓词仍是事件不是光流

**一个机制：** 骨干双路径仍算（或至少 PED 仍算），**flow head** 只在 r1 支持活点上做稠密读出，空 patch 用邻域插值/复制。谓词来自事件/r1 脉冲支持。这是 SciFlow 的 **反方向**：SciFlow 用同帧 flow 当神谕去省中间算；这里用事件支持去省 **读出**，中间不算神谕。

**为何可能付钱：** 若 head+H8/加权读出在同端口分母里够大，活点比例低到能净少 15%。ASNA 已宣称空间局部稀疏 OF，必须把 A 抄到 head 稀疏；本机制只在“骨干双消费者不能跳、只跳 head”时与 ASNA 卷积稀疏分开。

**不是首个 event-OF HW：** SENECA FireNet 已是端到端 event-OF 硅。不得写“我们先做了 OF 芯片”。FPGA 对照若用 ERAFT，必须标明那是帧 RAFT。

**无神谕：** 禁用当前帧 ordinary flow、GT flow、SCI 来生成 head 掩码。Event-count head-mask 是廉价 A；r1 学支持必须在 valid 上证明比 count 更贴 head 误差。

**双消费者 / BN：** 不把残差 SNN 化。Union 约束在骨干；head skip 不自动授权骨干 skip。AEE 对 1.219801338 极苛：插值会在运动边界裂开。

**杀死条件：** head 份额 <15%；count-mask 已够；AEE 裂开；或评委把 head 稀疏直接判给 ASNA。

---

## R3Q05-I6 占用 patch 内部：r1 支持与 T10 做 Spike-IAND 式时间位图与，不释放三行

**一个机制：** 空间上已占用的 patch **并不** 在 T10 上全活。θ 吸入后，把 r1 生产者的时间位图与 Spike-IAND 的 T-unroll 位图 **AND**，空时间行不上 0/1 GeMM。空间占用栅仍是 EventShiftFlow/event-count A。禁止把该 AND 解释成 SENECA 三行滑窗释放。

**为何可能付钱：** ASNA/occupancy 吃掉的是空间空穴；若本学生的空穴在 **活 patch 的时间行**，空间 mask 零收益，时间 AND 才有 15% 的来源。非因果 T10 只允许“十行到齐再 AND 折叠”，不允许三行提前 commit。

**不是首个 event-OF HW：** Spike-IAND 是 T-unroll A；SENECA/ASNA/EventShiftFlow 仍抄。X 只是谓词来自 r1 生产者时间支持，接到吸 θ 后的二值路径。

**无神谕：** 时间位图来自事件时间结构或 r1 发放，不来自当前 flow 的时间梯度。

**双消费者 / BN：** PED 在 T 上仍按连续全行（或证明某行 PED 也不需要——必须测量，不能默认）。Union 在时间轴上同样成立：某行只要 PED 需要，就不能因二值门空而丢源。BN 全域、全 T 人口。

**杀死条件：** 活 patch 内 T 行接近全 1；或非因果对齐代价 ≥ skip；或把三行释放写进 RTL 叙事。

---

## R3Q05-I7 用 r1 支持当 Preview-V 微核的 issue 过滤器，不把 BN-delay 当服务

**一个机制：** Ordinary Preview-V 微核一角 10396 slot / 8 条 FP32 FMA。r1 生产者支持集作为 **issue filter**：空 patch 不向该微核投 V 预览。BN 仍全域、仍按 native reduction 完成；明确 **不** 把 Delayed-V、总线占用从 55.3MB→18.4MB、或 `arithmetic_saving=0` 的 BN 等待改报成 15%。

**为何可能付钱：** SCOPE 已切断“晚发射 V / 少占总线 = 净服务”。还没切断的是 **少 issue 的 FMA slot**。若过滤后同端口完成链 slot 相对 10396 少 ≥15%，且 0-diff 到门和连续消费者，这是生产者支持直接打在已测量微核上的付钱方式。

**不是首个 event-OF HW：** 微核是本学生的 V 预览，不是 SENECA/ASNA 芯片替换。Event-count 过滤是 A，必须报 r1 支持相对 count 多滤掉的 slot。

**无神谕：** 不投 V 的原因是 r1 支持空，不是“当前 flow 已光滑/已零”。

**双消费者 / BN：** 过滤不得让 PED 缺操作数。G1 typed last-use 仍只是 revise-not-title：本卡标题是 **issue filter**，不是 last-use 类型学。BN reduction 硬件保持 unclosed。

**杀死条件：** 过滤只减少总线占用不减少 slot；或 8088 长背压类别未分裂就把等待当收益；或 AEE 动。

---

## R3Q05-I8 训练 r1 生产者去拟合双消费者 union 活集，硬件只跳补集

**一个机制：** 生产者不拟合光流、不拟合输入 count，只拟合 **运行时合法 skip 集** = 补集( 二值 GeMM 活 ∪ PED 活 )。一条 mask、两个消费者；硬件 issue 的 skip **仅允许 union 的补集**。损失函数是 live-set Hamming / 欠估计惩罚 + 任务 AEE，没有 SCI/当前帧 flow 项。

**为何可能付钱：** event-count 是 union 的廉价上界，往往过宽，15% 出不来。若 PED 并非空间稠密、只是与门错位，学 union 可以在不欠供连续路径的前提下收紧 mask。这是双消费者约束下 **唯一合法的稀疏谓词**，也是 DATE/FireFly-T/FlexSpIM 未按“吸 θ 后二值∪连续 PED”训练过的洞。

**不是首个 event-OF HW：** 稀疏 OF 计算已由 ASNA 宣称；occupancy FPGA 已由 EventShiftFlow 做。本岛不报芯片第一，只报谓词从 occupancy 换成 **union-live 学生**，并在同端口账上过 15%。

**无神谕：** 活集来自本网门与 PED 的前向，不来自 GT flow，不来自 ordinary 的完成 flow 场（那会把 SciFlow 从后门放进来）。

**双消费者 / BN：** Union 写进契约：任一消费者需要则必须算。全域 BN 对跳过位置填合法 0，不改 reduction 电路。欠估计（假阴性）直接杀 AEE；假阳性只是少省。

**杀死条件：** 测量后 PED 活集 ≈ 满域（union 无补集）；或学生相对 count 的 IoU 无差；或同端口 <15%。

---

## 共用对照与禁止句

**强对照（每卡都要，不能只比稠密）：** 输入 event-count mask；ASNA 式空间局部；occupancy-grid；FireFly-T / Spike-IAND / DATE hybrid 的普通映射；普通剪枝/量化；ordinary T10 源（AEE 1.219801338）。lifting CSE 只许当候选改写，不得当默认精度对照。

**禁止：** “首个事件光流硬件”；当前帧 flow / SCI 当推理谓词；残差改 SNN；event-count 当 X；非因果 T10 上写 SENECA 三行释放；把 BN-delay / 总线占用释放写成净服务；Prosperity/Gustav 标题化；连续 AT-LIF 幅值双 MAC。

**未做之前每卡都是 candidate。** 先测 union 活集覆盖、r1 支持相对 count 的额外 skip、以及该 skip 落在同端口哪一段（二值 GeMM / 注意力 pack / Preview-V issue / head）。哪一段拿不到 15%，停该卡布局，不杀“生产者支持”家族。
