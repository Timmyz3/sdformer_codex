# R3Q03 — 吸 θ 之后，哪一个 compiler/schedule 机制能当这封 TCAS-II 信

**Focal：** AT-LIF 输出 \(\{0,\theta\}\)，层共享 \(\theta\) 吸入下一层 \(W\) 之后，脉冲路径是 **0/1 × W 的二值 GeMM**。残差 / PED / I24 仍是另一条连续张量。问：在这个身份下，**单独哪一个编译器 / IR / 调度机制** 有资格当事件光流 SNN-Transformer 的信件主句。

**身份（只此一份）：** 层间传 0/1 脉冲，不是按事件保存的模拟幅值，也不是不可吸收 int8 payload。双消费者若仍在：其一是 **门（吸完后的二值）**，其二是 **残差连续路径**。禁止写成「连续 AT-LIF 幅值被两个 MAC 共用」。

**本文件元数据：** origin `ai-assisted`；stage `independent`；status `candidate`；**无打分**。每条只给一个机制。Prosperity / Gustav NRV-as-GeMM-psums / LoAS / FireFly-S / Phi / Bishop / FireFly-T overlay / DATE layer-split hybrid / Fang 3D AC / ESTU skip / PSN+da4ml / SDT MS 一律当 **A（必须抄全的对照）**，不得当标题。Spike-IAND 残差在保留 PED 时是负例。Live BN \(10\times96\times120\times160\) 是卫生。Delayed-V `arithmetic_saving=0` 不得当信。不得以 CIM、OpenROAD PPA、first-OF-HW、删除 35 次 RNE 当标题。

**已观测、只作 B 的材料（不是贡献）：** 长背压四臂均 8088，wait-class dump 缺失；源融合 6194→5354 属通用指令融合；serialized −6.4% 未闭合；FP preview ~49%；preview-V 普通角 10396 槽（8 条 FP32 FMA）；lifting AEE +0.013 未过 +0.005；同端口净服务 15% 与 AEE 仍是门。

---

## R3Q03-I1  Wait-class dump 作为一等 IR，把单条 8088 栅栏拆成类型化完成事件

- **id:** R3Q03-I1
- **origin:** ai-assisted
- **stage:** independent
- **status:** candidate
- **机制（一句）：** 编译器在 lowering 时不只发射「层完成」栅栏，而是把每次停顿打成 wait-class 令牌：`WAIT_GEMM_BIN`（二值 GeMM last-use）、`WAIT_PED`（连续残差 last-use）、`WAIT_BN`、`WAIT_PORT`（同端口供数），并只在同类 last-use 上设栅栏。
- **A：** 普通层屏障 / overlay 全局 sync（FireFly-T overlay 的层切）、SNE 的 UPDATE/FIRE 完成、SENECA 空间有序完成、Gustav 同 k-ID 屏障。这些都有「做完再下一步」，但没有把 **吸 θ 后二值 GeMM 完成** 与 **PED 完成** 分成可独立退休的 class。
- **B：** 两级写回资源点上 ordinary / lifting 长背压都是 8088；源程序少 22.8% 却推不动这条墙。缺 wait-class dump，就无法证明 CSE 或二值 GeMM 省下的是算术还是被同一条端口/BN 栅栏吃掉。Serialized −6.4% 未闭合，也说明「串一层」和「真依赖」没被 IR 分开。
- **X（候选增量句）：** 相对 A 的全局层屏障，本网在吸 θ 之后把完成事件按消费者类型拆栅栏，使二值 GeMM 的 last-use 不必等 PED/BN 的 last-use；**信件是 wait-class 调度 IR，不是新神经元。**
- **为何是 compiler：** 不改 AT-LIF、不改 \(W\leftarrow\theta W\)。改的是 SSA 上 last-use 标注与发射器插入的 fence 集合。硬件可以仍是同一 overlay 端口。
- **接点：** last-use 分析的产出是 **dump + 可拆栅栏**，不是「我们有类型」这句话本身。8088 必须先按 class 直方图拆开，才谈 15%。
- **明确不是：** 不是 G1「typed last-use」空类型系统当标题；不是 delayed-V；不是把 BN 卫生做成贡献；不是 Gustav 屏障改名。
- **杀门：** dump 后若 ≥90% 停顿落在同一 `WAIT_PORT`/`WAIT_BN`，拆栅栏净服务 <15%，或拆完 AEE 动了（非法放松依赖），停该布局。
- **缺证据：** wait-class 直方图、合法拆栅栏后的同端口时间线、与「仍用单条 8088」的对照。完整链仍未 PASS，本条不得据源融合百分比晋级。

---

## R3Q03-I2  T10 mix 的双分拣 CSE：布尔脉冲分拣 vs 仿射 mix 分拣，禁止跨吸收边界合并

- **id:** R3Q03-I2
- **origin:** ai-assisted
- **stage:** independent
- **status:** candidate
- **机制（一句）：** 把 T10 lifting/mix 图编成 **双分拣 IR**：Sort-S 上只允许 0/1 的 AND/OR/bitmap 运算（吸 θ 后的合法脉冲代数）；Sort-M 上保留仿射 mix 与现有 35 次中间 RNE。CSE 只在同一 sort 内合法；**禁止**把 Sort-M 节点 CSE 进 Sort-S 来「少一次 RNE」。
- **A：** da4ml 整图 CSE、PSN 前缀/滑动核、当前源 CSE（lifting 159 加减 + 35 RNE vs ordinary 260 加减）、通用指令融合 6194→5354。FireFly-S Bitmap AND、Prosperity 公共子组合只覆盖 Sort-S。
- **B：** 通用指令融合已经吃掉 6194→5354，不能再当 X。删除 35 RNE 被合同禁止当标题。若把 mix 与脉冲当同一 DAG，CSE 会把连续 mix 寿命拉到门的二值 last-use 上，或反过来强迫 PED 读被布尔化的值。Spike-IAND 残差在保留 PED 时是负例，说明「残差也二值化」不是合法 CSE。
- **X：** 相对 da4ml/通用 CSE，本网 CSE 的合法性由 **吸收边界 + 双消费者 last-use** 决定：只被门消费的 mix 前缀可降到 Sort-S；被 PED 触及的节点必须留在 Sort-M 并保留 RNE。信件是 **分拣 CSE 合法性**，不是更少加减。
- **为何是 compiler：** 这是 IR 类型与 CSE rewrite 规则。Prosperity/FireFly-S 只被允许出现在 Sort-S 的 lowering 里，本身仍是 A。
- **接点：** T10 mix 的公共子式按消费者着色；跨 sort 的 CSE 视为非法，即使算术上像。
- **明确不是：** 不是删 35 RNE；不是把 AT-LIF 再变回连续幅值；不是 IAND 残差；不是 MX3P 岛。
- **杀门：** 分拣后 Sort-M 工作量不降、同端口 <15%、或 Sort-S 降级改变门语义导致 AEE 相对对照 >+0.005。
- **缺证据：** 按消费者切过的 mix SSA、非法跨 sort CSE 的反例计数、与「单 DAG + 通用融合」同资源对照。

---

## R3Q03-I3  Dual-ready 反串行：用两个 last-use 谓词闭合 serialized −6.4%，而不是层串行

- **id:** R3Q03-I3
- **origin:** ai-assisted
- **stage:** independent
- **status:** candidate
- **机制（一句）：** 调度器不把一层当成一个 ready。每个 tile 维护两个谓词：`ready_bin = last_use(s∈{0,1} → GeMM)` 与 `ready_ped = last_use(residual → PED)`。合法反串行 = 在 `ready_bin[ℓ]` 之后立刻发射层 \(\ell+1\) 的二值 GeMM，即使 `ready_ped[ℓ]` 仍未到。
- **A：** DATE 层切 hybrid（稀密核按层拆）、FireFly-T overlay 的层间发射、LoAS FTP 时间并行、普通 double-buffer。它们的 ready 几乎都是「这层输出张量齐了」。
- **B：** serialized −6.4% 未闭合，说明当前至少有一种合法依赖被过度串行。吸 θ 之后，二值 GeMM **不再**需要模拟幅值到齐；继续用单 ready 会把已经合法的脉冲 GeMM 钉在 PED 后面。Delayed-V saving=0 又说明「把 V 往后拖」不是解；要拖的不是 V，是 **错误的层栅栏**。
- **X：** 相对 DATE/overlay 的层级 ready，本网用 **双消费者双 ready** 做 intra-layer 反串行，专门闭合 −6.4% 那一刀。信件是 schedule 谓词，不是 hybrid 核。
- **为何是 compiler：** 发射器上的 ready 函数与依赖边集合。不要求 DATE 那样拆两套核；同一端口上也可双 ready。
- **接点：** last-use 分析直接生成两条依赖边；CSE 的 mix 节点按其 last-use 挂到其中一条，禁止一条边误绑两条消费者。
- **明确不是：** 不是 LoAS 静态共享（加法 −8.96%、周期 +0.23% 已停该布局）；不是把 PED 串到 GeMM 后面当「省端口」；不是 first-OF-HW。
- **杀门：** 反串行后同端口净服务仍 <15%，或与「合法单 ready」比只复现 −6.4% 的另一面（功能错 / AEE 动）。重叠窗口若被 `WAIT_PORT` 填满，停。
- **缺证据：** 双 ready 时间线 vs 当前串行时间线的逐 class 差分；功能零差；完整链而非源核切片。

---

## R3Q03-I4  Last-use 驱动的 mapping pass：同一 overlay 还是 DATE 式拆核，由脉冲张量 vs PED 的寿命差决定

- **id:** R3Q03-I4
- **origin:** ai-assisted
- **stage:** independent
- **status:** candidate
- **机制（一句）：** 增加一道编译 mapping：对每个空间 tile 比较 `live(s_bin)` 与 `live(ped)`。寿命窗口短且重叠 → 降到 **FireFly-T 式单 overlay**（避免拆核搬运）；寿命窗口长且 `last_use(s_bin) ≪ last_use(ped)` → 降到 **DATE 式拆流**（二值 GeMM 核 vs 连续 PED 核），只搬 bitmap 不搬连续激活。
- **A：** FireFly-T overlay（整网映射到可重构 overlay）、DATE layer-split hybrid dense/sparse cores、Bishop 稀密分流、Phi 层次模式映射。这些是固定策略或按层类型拆，不是按 **吸 θ 后两个 last-use 的寿命差** 逐 tile 选。
- **B：** Overlay 把二值 GeMM 和 PED 绑在同一套端口上，正是 8088 一刀切的温床。DATE 按 CNN/Transformer 层类型拆，对不上「同一层里门是二值、残差是连续」。Bishop BSA 束是训练侧打包，不回答运行时寿命差。
- **X：** 相对「永远 overlay」或「永远按层拆核」，信件是 **last-use 寿命差选择 mapping**。硬件核仍抄 FireFly-T / DATE / Bishop，不自称新核。
- **为何是 compiler：** 这是 mapping/placement pass，输出是绑定与搬运边，不是新 RTL 原语。
- **接点：** 输入是 I1 的 wait-class 与 I2 的分拣寿命；输出是 overlay vs split 的二进制选择及 spill 边。
- **明确不是：** 不是 CIM 标题；不是 OpenROAD PPA；不是「我们做了第一份事件光流硬件」；不是再发明一套 hybrid 微结构。
- **杀门：** 所有 tile 都倒向同一侧（mapping 退化成常量）；拆流搬运 > 省下的 GeMM 等待；同端口 <15%。AEE 必须与不拆流同一函数。
- **缺证据：** 逐 tile `live(s_bin)`/`live(ped)` 分布、搬运字节、与纯 overlay / 纯 DATE 层切的三方同端口。

---

## R3Q03-I5  双消费者寄存器着色：T10 mix SSA 按 last-use 涂成 bitmap RF vs 连续 RF

- **id:** R3Q03-I5
- **origin:** ai-assisted
- **stage:** independent
- **status:** candidate
- **机制（一句）：** 在 mix SSA 上做 last-use 着色：只流向门的值进 **bitmap/比特 RF**（FireFly-S / Prosperity 产品稀疏的合法操作数）；任何流向 PED 的值进 **连续 RF**。同一 SSA 名若被双消费者读，必须在分叉点 **复制或降级**，禁止用一个物理寄存器延长两条寿命。
- **A：** 普通图着色寄存器分配、Phi 预取/模式字、Bishop TTB 打包、FireFly-S 双侧稀疏操作数、Gustav NRV 行缓冲。它们分配的是单一数值类型。
- **B：** 吸 θ 后操作数类型已经分裂，但当前源接口仍像单一 RF（两级写回共同 50B 流水态、两槽后可读）。若 mix 临时值因为 PED 还活着而占着本可当 bitmap 的槽，二值 GeMM 的 product sparsity 无从下手。预览 V 普通角 10396 槽 / 8 条 FP32 FMA，也像把脉冲路径留在 FP RF 里。
- **X：** 相对单 RF 着色，本网用 **消费者 last-use 决定物理文件**，并在双读点强制 copy/demote。信件是分配规则，不是第二套 MAC。
- **为何是 compiler：** 经典 RA 的类型扩展：颜色 = (寄存器文件, last-use class)。Prosperity/FireFly-S 只在 bitmap 颜色上启用。
- **接点：** CSE 公共子式若被两个文件都需要，公共计算只做一次，但 **物化两次**（bitmap 一份、连续一份），费用必须进账，避免把 CSE 次数当服务。
- **明确不是：** 不是「双文件硬件」当 ASIC 贡献；不是把 PED 也塞进 bitmap；不是 IAND 把残差涂成比特。
- **杀门：** copy/demote 流量吃掉 bitmap 收益；连续 RF 峰值不降；同端口 <15%；copy 引入的舍入使 AEE >+0.005。
- **缺证据：** 着色后两类 RF 峰值、copy 边字节、与单 RF + 通用融合的对照；功能零差。

---

## R3Q03-I6  一层 IR 降成双 kernel：K_bin（0/1 GeMM）与 K_ped（连续 PED），禁止用 preview-V 的 FP FMA 冒充脉冲路径

- **id:** R3Q03-I6
- **origin:** ai-assisted
- **stage:** independent
- **status:** candidate
- **机制（一句）：** 前端仍是一层；lowering 强制拆成两个 kernel 日历：`K_bin` 只含吸 θ 后的二值 GeMM（Gustav psums / Prosperity 子组合 / FireFly-S bitmap 皆可作实现），`K_ped` 只含残差连续与必要 BN。Preview-V / FP FMA 只允许出现在 `K_ped` 或编译期常数折叠，不允许出现在 `K_bin` 的内环。
- **A：** PSN+da4ml 整图编译、FireFly-T overlay 单 kernel 发射、DATE 按层选核、preview-V 微核（普通角 10396 槽，8 条 FP32 FMA）、FP preview ~49%。
- **B：** FP preview ~49% 与 10396 槽说明当前「一层」仍按 FP FMA 计价。吸 θ 之后这是错误的 lowering。Delayed-V `arithmetic_saving=0` 说明再动 V 的发射位置没有算术收益，必须动的是 **kernel 边界**：脉冲路径根本不该进 FP 槽。Live BN 体积大，但那是卫生，应留在 `K_ped` 侧，不写进标题。
- **X：** 相对 overlay 单 kernel 与 DATE 按层选核，信件是 **一层双 kernel 的 typed lowering**。日历上 `K_bin` 可在 `ready_bin` 后发射，`K_ped` 走自己的 last-use。
- **为何是 compiler：** kernel 拆分、操作数合法集合、禁止规则（FP 不得进 `K_bin`）都是编译器契约。
- **接点：** T10 mix CSE 的 Sort-S 进 `K_bin`，Sort-M 的输出若只供门则在 kernel 边界物化为 bitmap；供 PED 的留在 `K_ped`。
- **明确不是：** 不是 dual-core 芯片标题；不是 CIM；不是把 preview-V 延迟当贡献（saving=0 已杀）；不是 BN 折叠当贡献。
- **杀门：** `K_ped` 日历仍主导且总槽不降 15%；`K_bin` 仍残留 FP FMA；两 kernel 同步又退化成单条 8088。
- **缺证据：** 分 kernel 槽数 vs 10396；`K_bin` 内环指令 mix；同端口完整链；AEE 与单 kernel 函数一致。

---

## R3Q03-I7  Psum / PED 干涉图：Gustav NRV 伪和缓冲的 last-use 不得被连续 PED 寿命劫持

- **id:** R3Q03-I7
- **origin:** ai-assisted
- **stage:** independent
- **status:** candidate
- **机制（一句）：** 把 Gustav 式 GeMM psum 缓冲与 PED 连续张量放进同一张 **live-range 干涉图**。着色/spill 规则：psum 的 last-use = 该输出列/行的二值 GeMM 完成；PED 不得占用同一物理 psum 槽来「省一块 SRAM」。若 overlay 只有一块便笺，编译器必须插入 **typed spill**（spill 的是 PED 或已完成 psum，而不是把 GeMM 栅栏拉长去等 PED）。
- **A：** Gustav NRV-as-GeMM-psums / CPTB、LoAS 伪和与校正、Prosperity 公共部分和、普通 double-buffer。这些解决稀疏 GeMM 的 psum，不解决 **psum 与 PED 抢同一 overlay 便笺**。
- **B：** 身份锁定后，Prosperity/Gustav 对脉冲路径合法，但不得当标题。本地长背压 8088、两级写回共用流水态，很像 psum 槽被非 GeMM 寿命拖住。Serialized −6.4% 也可能是「等那块缓冲空」而不是真数据依赖。
- **X：** 相对把 NRV/psum 抄全，信件是 **psum vs PED 的干涉着色 + typed spill**。Gustav 仍是 A；增量在调度器不让 PED last-use 劫持 psum last-use。
- **为何是 compiler：** 这是缓冲分配与 spill 插入，经典编译问题，对象换成吸 θ 后的二值 psum 与连续 PED。
- **接点：** wait-class `WAIT_GEMM_BIN` 应在 psum last-use 处退休；若 dump 显示退休点总对齐 PED，即干涉未解。
- **明确不是：** 不是 Gustav 标题；不是 Prosperity 产品稀疏标题；不是再开一块无限 SRAM；不是 OpenROAD 面积故事。
- **杀门：** typed spill 字节 ≥ 省下的等待；干涉图着色后 8088 不变；同端口 <15%。功能必须与「拉长栅栏」的错误实现对比并证明零差。
- **缺证据：** psum/PED 干涉矩阵、spill 轨迹、与「共享便笺 + 单栅栏」对照。

---

## R3Q03-I8  Last-use skip certificate：只给「消费者集合 ⊆ 二值 GeMM」的通道发跳过证，PED 是不可跳过消费者

- **id:** R3Q03-I8
- **origin:** ai-assisted
- **stage:** independent
- **status:** candidate
- **机制（一句）：** 编译器对每个通道/tile 求消费者集合。若 `consumers ⊆ {K_bin}` 且 PED/BN 不读，则发射 **skip certificate**（ESTU / Fang 3D AC / ASTER 层跳的合法子集），允许二值 GeMM 静默。若 PED ∈ consumers，证书拒绝，即使脉冲全零也必须走连续残差合同（SDT MS **ADD**，禁止 IAND 改写）。
- **A：** ESTU skip、Fang 3D AC、ASTER 层跳/时步早退、FireFly-S 膜静默通道剪枝、Bishop ECP、SDT MS 残差。Spike-IAND 在本网保留 PED 时是 **负对照**。
- **B：** 事件光流硅（SNE UPDATE/FIRE、SENECA、ASNA-Flow、ERAFT FPGA）的 skip/静默默认假设「没有脉冲 ≈ 没有工作」。本网 PED 打破该假设：门可静默，连续残差仍在。训练侧 lifting AEE 已 +0.013，再靠 IAND/乱 skip 会把精度门彻底打穿。
- **X：** 相对 ESTU/Fang/ASTER 的静默，信件是 **由 last-use 集合签发的 skip certificate**，把 PED 写成不可跳过消费者。不是新 skip 电路，是证书 IR。
- **为何是 compiler：** 证书是静态/半静态分析产物，运行时只检查证。与 T10 mix CSE 的关系：Sort-M 上若仍有 PED 边，禁止把该节点证成 skip。
- **接点：** 双消费者调度的负空间——明确什么 **不能** 反串行、不能 overlay 掉。这补上 event-OF 先验在残差网上的洞。
- **明确不是：** 不是 first-OF-HW；不是删 T10；不是 IAND 残差；不是把 35 RNE 证成可 skip。
- **杀门：** 可证 skip 的通道份额太小，净服务 <15%；误证（PED 仍读）造成功能差或 AEE >+0.005；证书检查本身比一次许可还慢（已有持续检查 +0.6274% 负例）。
- **缺证据：** 逐通道消费者集合统计、合法 skip 份额、误证反例、与「无证书的 FireFly-S 静默」及「IAND 残差」两个对照。

---

## 使用约定（独立阶段）

- 八条都是 **candidate**，彼此不融合、不打分、不排序。后续若要留信，只许挑 **一条** 机制当 X，其余降为 A 的 lowering 细节或杀门工具。
- 任何一条一旦写进标题，必须同时带着：完整迁入的 A、同端口 15% 净服务、AEE 绝对 ≤1.259 且相对对照 ≤+0.005、wait-class dump、双消费者都接上。缺一项就仍是独立假说。
- 本文件不引用 round2 fusion / G1–G4 正文；I1 只借用「8088 未拆 class」这一观测，不把 typed last-use 整包当标题。
