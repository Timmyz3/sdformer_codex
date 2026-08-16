# Transformer 架构、电路、EDA 论文深度调研与迁移边界

**日期**：2026-07-13  
**目标**：从 ANN Transformer、SNN 加速器、稀疏架构、ISSCC/JSSC 电路和 EDA 框架中寻找可迁移机制，并明确哪些属于已有工作、哪些可能形成 H67/H68 的新组合  
**原则**：迁移 ANN 机制到 SNN 本身不是原创；论文必须引用来源，并用新的 workload、数据流、数值语义和实测 PPA 证明增量

## 1. 调研结论

当前不应把 Bishop 异构双核直接设为唯一主线。真实 profile 表明 H67/H68 之间差异很小，而 block 内部差异巨大，更适合研究：

> **统一同构、表示可切换的 temporal-pair 核 + 多 row context + class-stationary 后端。**

可迁移机制分为四层：

1. **架构层**：多上下文、精确乱序、表示切换、block-aware 调度、共享后端；
2. **数据流层**：temporal-pair 驻留、score/class/gate 融合、双缓冲和中间量不落片外；
3. **互连/存储层**：事件压紧、类更新合并、bank mapping、metadata-first；
4. **电路层**：数据不变时的切换抑制、细粒度时钟门控、可重构归约树。

其中最有机会形成论文增量的是它们围绕 H67 的精确充分统计量、K-zero 分母语义和 gated-K 输出形成的联合设计，而不是其中任何一个通用部件。

## 2. 架构类会议工作

| 工作 | 会议 | 原机制 | 可迁移内容 | 不能照搬或冒充原创 |
|---|---|---|---|---|
| Bishop | ISCA 2025 | TTB、density stratifier、dense/sparse core、BSA、ECP | TTB 描述符、密度统计、有限 FIFO 对照 | TTB、异构双核、ECP 都已有；ECP 改语义 |
| LoAS | MICRO 2024 | 时间全并行、spike compression、低成本 inner join | T=2 temporal-pair 连续布局、事件 membership packet | 不能宣称首次时间并行或首次 spike 压缩 |
| FuseMax | MICRO 2024 | attention 算子融合、负载均衡、与序列长度无关的 buffer | score-class-gate 融合、前后端负载平衡 | 标准 attention 融合不是我们的原创 |
| FLAT | ASPLOS 2023 | Fused Logit-Attend tiling、线性中间存储 | 中间 score 不落外存、pair/class 驻留 | FLAT 面向标准 QK-softmax-V，不能复用其结果数字 |
| ExTensor | MICRO 2019 | metadata-first、分层集合求交 | 事件 index 先读 metadata、精确集合运算 | metadata-first 和 intersection 已有 |
| SIGMA | HPCA 2020 | 灵活分发网络和 FAN 归约树 | class update 合并、可变长度归约 | FAN/树互连不是原创 |
| Flexagon | ASPLOS 2023 | 统一 merger-reduction 网络、多数据流 | 同一归约底座支持 bitmap/event/class 三种模式 | 统一 merger network 已有 |
| SpAtten | HPCA 2021 | 级联 token/head pruning、progressive quantization | 只借“分级 issue”组织方式 | pruning 和 top-k 会改变 H67 结果 |
| Energon | MICRO 2022 | 运行时弱相关 pair 过滤和 sparse engine | active metadata 与 elastic issue | weak-pair 过滤未经重训不能用 |
| SWAT | DAC 2024 | window attention 的 row-wise、input-stationary 和 kernel fusion | 9×9 window 输入驻留、block 间动态 pipeline | 面向普通 window attention，数值与本设计不同 |
| ASADI | HPCA 2024 | 利用 sparse attention 的对角局部性和 DIA 格式 | 检验运动边缘是否形成 diagonal bank locality | 未经本网络 profile 不能假设存在对角局部性 |
| FireFly-T | arXiv 2025/FPGA | 稀疏引擎+二值attention引擎、multi-lane decoder、OOO负载均衡、跨head延迟隐藏 | 作为异构双引擎与稀疏decoder强制对照 | 双引擎、multi-lane decoder、OOO和projection-attention重叠均非原创 |
| VESTA | IEEE 2024/2025 | 统一PE支持卷积/线性/dot-product，TFLIF、ZSC、WSSL、STDP | 作为统一SNN Transformer与列流融合强制对照 | 统一PE、多时间共同处理和不落地完整中间矩阵均已有 |
| STAR | arXiv 2025 | 跨stage协调tiling、排序与稀疏FlashAttention | 作为通用cross-stage fusion反证 | 跨阶段协同本身不能作为LR-HTT原创 |

关键来源：

- [Bishop 原文](https://arxiv.org/abs/2505.12281)明确使用 TTB、stratifier、稠密/稀疏核和 ECP；本设计不能再以这些通用名词主张首次。
- [LoAS 原文与开源代码](https://arxiv.org/abs/2407.14073)提出 fully temporal-parallel dataflow、spike compression 和 inner join；[GitHub](https://github.com/RuokaiYin/LoAS)主要提供模型 profiling、剪枝和 artifact，而不是可直接复用的完整 RTL。
- [FuseMax 原文](https://arxiv.org/abs/2406.10491)强调算子间负载失衡和随序列长度增长的片上 buffer；它支持我们把多 row context 和共享 SCS 作为系统问题，而不是只优化一个 XOR。
- [FLAT 原文](https://arxiv.org/abs/2107.06419)通过融合和 tiling 把 attention 中间存储从平方增长降为线性；本设计对应的是 pair-score-class-gate 融合。
- [ASADI 原文](https://www.comp.nus.edu.sg/~tulika/HPCA24.pdf)发现其 ANN sparse attention 呈对角而非行/列局部性。我们已经把 9×9 水平、垂直、双对角和 bank conflict 加入 profile，先验证再迁移。
- [VESTA 原文](https://arxiv.org/abs/2503.20246)已覆盖统一PE、四时间步融合、层间spike存储压缩和STDP列流；它是DP-TME、LR-HTT与CCSP必须共同面对的最近邻。
- [FireFly-T 原文](https://arxiv.org/abs/2505.12771)已覆盖稀疏/二值双引擎、多lane bitmap解码、worker维乱序和跨head延迟隐藏。因此Bishop式双路径不只是弱创新，而且已有更直接的SNN Transformer先例。
- [STAR 原文](https://arxiv.org/abs/2512.20198)已把cross-stage coordination和tiling用于稀疏attention；LR-HTT必须证明跨PSN、非标准class attention、FGP和RPI的语义增量。

### 2.1 SNN Transformer最近邻对主张的修正

[可重构并行时间步Spiking Transformer加速器](https://arxiv.org/abs/2503.19643)已经实现fully-parallel tick-batching和四时间步展开。DP-TME不能再主张首次时间并行或首次可重构timestep，只能主张T10/T2整除映射下的PSN矩阵slot复用。

[STEP](https://arxiv.org/abs/2505.11151)还指出Spiking Transformer能效必须同时计入稀疏度、位宽和存储访问，量化ANN可能达到相当甚至更高能效。本文必须报告真实memory/clock/control能量，不能只用spike operation数证明优势。

## 3. ISSCC/JSSC/TVLSI/ESSERC 电路与芯片工作

### 3.1 复旦 ISSCC 2023 蝶形 Zero Skipper

论文：**A 28nm 53.8TOPS/W 8b Sparse Transformer Accelerator with In-Memory Butterfly Zero Skipper...**，DOI `10.1109/ISSCC42615.2023.10067360`。

[复旦官方介绍](https://fics.fudan.edu.cn/70/b1/c22203a487601/page.htm)说明原设计：

- 对权重做局部细粒度、全局粗粒度剪枝；
- 用蝶形数据分配网络提取与稀疏权重对应的输入特征；
- 通过定制存储单元和传输门把数据分配通路并入 CIM；
- 另做动态局部注意力和 QK 共享。

原机制不能直接搬到 H67：

- 原网络为静态剪枝权重取数，我们是动态 `{Q0,Q1,K0,K1}` event；
- 原实现依赖 CIM 和定制传输门，我们当前目标是标准单元数字 RTL；
- D=32 很小，完整 butterfly/Beneš 网络可能比简单 bitmap-popcount 更贵；
- “蝶形网络用于 SNN”不是足够的新颖性。

可研究的新变体是：

> **Butterfly Mask-Reduce Fabric，BMRF**：把每个 feature lane 的四向量 membership 编成 4-bit mask，经稳定压紧后，由 16-entry mask LUT 直接产生 `q0/q1/k0/k1/overlap0/overlap1/motion` 的计数增量，再进入共享归约树；过密时无损回退到 128-bit bitmap。

其增量不是“提出蝶形网络”，而是：

1. 动态事件 membership 而非静态稀疏权重；
2. 压紧与 H67 充分统计量归约融合；
3. bitmap/event 双表示 exact fallback；
4. 输出直接服务 pair score 与 class commit，而非通用 MAC 阵列。

该候选只有在 union-event profile 和 DC PPA 同时通过时才能晋级。

### 3.2 C-Transformer：同构可重构优于固定异构

[ISSCC 2024 C-Transformer](https://iccircle.com/static/upload/img20240529102116.pdf)，DOI `10.1109/ISSCC49657.2024.10454330`，指出动态变化的数据分布会使专用异构架构利用率大幅波动，因此用 homogeneous reconfigurable HMAU 在乘法和多个累加器模式间切换。

对我们的启发不是复刻 HMAU，而是：

- H67/H68 共用同一功能超集；
- 同一 predicate-count-reduce lane 支持 bitmap mode 和 event-membership mode；
- block descriptor 决定模式，不实例化两套核；
- 两种模式必须生成完全相同的充分统计量。

这为“统一同构核而非 Bishop 式固定双核”提供了强先例，但 homogeneous reconfiguration 本身也不能作为原创。

### 3.3 MulTCIM：混合稀疏与利用率平衡

[MulTCIM JSSC 论文页面](https://researchportal.hkust.edu.hk/en/publications/multcim-digital-computing-in-memory-based-multimodal-transformer-/)，DOI `10.1109/JSSC.2023.3305663`，包含 long reuse elimination、runtime token pruning、modal-adaptive CIM network 和 bitwidth balancing。

可迁移的是“不同稀疏维度必须分别建模，并围绕利用率重排”，对应：

- pair-empty：representation/issue skipping；
- K-zero：active replay skipping，但 class/denominator 保留；
- fold class：class transaction reduction；
- block 差异：context 配额和发射顺序；
- bit 活动：时钟门控和数据切换功耗。

不能迁移 runtime token pruning，因为 H67 silent token 仍贡献 Shiftmax 分母。

### 3.4 D3TA：时间相同数据的切换抑制

[D3TA ESSERC 2025 原文](https://www.esserc2025.org/_files/ugd/aa54ce_3ae2d7986b2f43d7bd4a3f3c9cf366f1.pdf)采用 HyperAttention、三重稀疏处理和带 charge recycling 的双端口 eDRAM CIM。其电路通过相邻数据不变时复用全局读位线状态，降低切换。

可迁移为标准单元数字候选：

- pair bank 的 read-data hold/bypass；
- `K0=K1` 时关闭 Motion-XOR 后级寄存器和 popcount；
- 相邻 packet 相同 membership 时冻结 compactor route/control；
- SAIF 分别统计 clock、data、SRAM address 和 output toggle。

但当前 `Delta=0` 相比 pair-empty 只多约 0.1%，所以“非空完全复用”不应成为主贡献。是否有跨 pair/跨 window 相同 payload，必须等 ordered trace。

### 3.5 HARDSEA、SPRINT 和 ISSCC 2022 OOO

- [HARDSEA](https://ieeexplore.ieee.org/document/10367847/)，DOI `10.1109/TVLSI.2023.3337777`：ReRAM 做轻量相关性预测、SRAM-CIM 做精确稀疏 attention。可借“便宜前判定 + 精确后端”分层，但我们不使用模拟近似预测。
- [SPRINT](https://arxiv.org/abs/2209.00606)：ReRAM 近似筛选、数字重算。可借“筛选与精算解耦”，不能借其近似删除。
- ISSCC 2022 的 approximate Transformer processor，DOI `10.1109/ISSCC42614.2022.9731686`，已经使用 sparsity speculation 和 out-of-order computing。我们只能表述为“metadata 可证明、无误预测恢复的独立 row 精确乱序”，不能声称首次 OOO。

### 3.6 T-REX与ISSCC 2025层融合芯片

[T-REX](https://arxiv.org/abs/2503.00322)通过dynamic batching按长度并行处理1/2/4个输入，并用two-direction accessible register file支持矩阵按行或按列访问。其论文报告TRF改善12%到20%利用率。因此HTT不能把多方向bank或长度打包作为新颖性，必须落在冻结producer-consumer生命周期和event/residual精度路由。

ISSCC 2025的[Memory-Compute-Intensity-Aware CNN-Transformer Accelerator](https://ieeexplore.ieee.org/document/10904499)已经结合hybrid-attention layer fusion、KV/weight reuse和级联剪枝。LR-HTT必须与普通layer fusion分开消融，证明跨PSN-TESSA-FGP-RPI的额外事务收益。

## 4. SNN 与事件光流工作

| 工作 | 原贡献 | 对本项目的用途 |
|---|---|---|
| Bishop | SNN Transformer TTB/异构核 | 最直接对照基线和术语边界 |
| LoAS | dual-sparse SNN 时间并行 | temporal-pair 和事件压缩先例 |
| SpikeX | SNN 网络-硬件联合 DSE | 训练/架构闭环方法论 |
| hARMS | 异步事件历史上的光流硬件 | 事件 burst、局部历史和流式处理的 workload 背景 |
| Eventor | 事件视觉片上流式处理 | 片上缓冲和事件数据搬运参考 |

[hARMS](https://arxiv.org/abs/2112.06772)避免构造完整事件帧，只保存相关事件历史，使延迟不随传感器分辨率线性增长。用户当前不要求 voxel 前端 RTL，因此本项目不复刻其前端；可借鉴的是：

- 不能只报告平均事件密度，要报告 burst 和历史窗口；
- 光流活动集中于运动边缘，可能有方向性和局部连通；
- 延迟、吞吐和 buffer 深度要按事件到达顺序评估。

## 5. EDA 与开源实现

### 5.1 Sparseloop/Timeloop

[Sparseloop](https://sparseloop.mit.edu/)把稀疏优化分为 representation、gating、skipping，并计入压缩/匹配开销。[开源 Timeloop](https://github.com/NVlabs/timeloop)支持稀疏 tensor 的统计模型。

本项目必须采用同样的分账：

| 类别 | 本项目实例 |
|---|---|
| representation | bitmap、union-membership packet、class histogram |
| gating | pair-empty、K-zero active-bank gate、motion-zero clock gate |
| skipping | 不发 active replay、不读无效 payload、空 context 不进入前端 |
| overhead | metadata、compactor、format conversion、FIFO、tag、fallback |

不能只把跳过周期记收益，而忽略 metadata、packet 对齐和控制切换。

### 5.2 Stellar

[Stellar MICRO 2024](https://people.eecs.berkeley.edu/~ysshao/assets/papers/stellar-micro2024.pdf)将 functionality、dataflow、sparse structure、load balancing 和 private memory 分离，并生成可综合 Verilog。它的开源地址为 [hngenc/stellar](https://github.com/hngenc/stellar)。

我们不直接把 H67 塞入通用 sparse tensor IR，而是借其方法建立参数化 RTL：

- 功能：H67/H68 exact score/class/gate；
- 数据流：pair-resident、class-stationary；
- 表示：bitmap/union packet；
- 负载均衡：block-aware context admission；
- 私有存储：每 context active bank/histogram；
- 共享存储：pair source bank、SCS LUT、输出队列。

### 5.3 AccelTran

[AccelTran 开源仓库](https://github.com/jha-lab/acceltran)包含 cycle simulator、CACTI/NVMain、SystemVerilog 模块和 DC 脚本。其价值是评估组织方式：

- cycle simulator 与 RTL block PPA 分离；
- module-level DC 结果回填系统模型；
- 报告利用率和模块级功耗；
- 多配置 DSE 使用同一 workload trace。

其通用 Transformer MAC/softmax RTL不符合 H67 数值语义，不能直接复用为 golden。

## 6. 可形成新组合的机制

| 候选 | 组合来源 | H67/H68 特有增量 | 状态 |
|---|---|---|---|
| Pair-resident exact statistics | LoAS 时间并行 + FLAT 融合 | 两时间片共享 K pair/Motion-XOR，直接生成两个 Q7 score | 主线 |
| Homogeneous representation switching | C-Transformer + Flexagon | bitmap/event membership 都输出同一 7 个充分统计量 | 主线候选 |
| Multi-context exact row scheduling | FuseMax 负载平衡 + ISSCC OOO | 不预测、不删 token，按 block/metadata 成本调度独立 row | 主线 |
| Pair-coalesced class commit | SIGMA/Flexagon reduction | 双 K-zero score 同 class 时 `+2`，直接服务 SCS histogram | 主线候选 |
| BMRF | 复旦 butterfly + LoAS compression | 4-bit membership 压紧与 H67 统计归约融合、dense fallback | 条件晋级 |
| Direction-aware bank mapping | ASADI diagonal locality + 光流边缘 | 用实测水平/垂直/双对角分布选 bank 映射 | 条件晋级 |
| Exact data-toggle suppression | D3TA charge recycling | K motion-zero、相同 membership 和 inactive context 的数字门控 | 电路子贡献 |

## 7. 投稿时的原创边界

### 可以在实现和 PPA 后主张

1. 面向 all-binary event-flow attention 的 temporal-pair exact sufficient-statistics 数据流；
2. pair score、K-zero class commit、SCS denominator 和 gated-K 的 class-stationary 融合；
3. block/workload-aware 的同构表示切换与多 row context 精确调度；
4. 若通过淘汰门槛，动态 membership 压紧和充分统计量归约融合的 BMRF。

### 不能主张

- 首次 TTB、首次蝶形网络、首次 temporal parallel、首次 OOO；
- 首次 attention fusion、首次 sparse/dense reconfiguration；
- 把 ANN 方法换到 SNN 就算原创；
- 将未综合候选写成已有 PPA 收益；
- 隐去来源或用新名字包装已有机制。

## 8. 论文对照组要求

至少需要：

1. 当前 162-token serial row engine；
2. 81-pair fixed-bitmap single-context；
3. 81-pair fixed-bitmap + 2/4/8 context；
4. class-stationary + pair-coalesced commit；
5. homogeneous bitmap/event mode；
6. BMRF 开/关；
7. row-major/diagonal/XOR bank mapping；
8. 可选 Bishop 式双路径模型。

所有对照必须使用相同 trace、频率、库、SRAM 假设、输入输出带宽和 hardware-order golden。

## 9. 官方开源实现的源代码级复核

### 9.1 Prosperity不是“相同gate码乘相同权重”的直接先例

本轮检查了[Prosperity官方仓库](https://github.com/dubcyfor3/Prosperity)的
`simulator/simulator.py`、`simulator/accelerator.py`、`simulator/energy.py`和CUDA
ProSparsity kernel。其实际算法对一个二值activation tile的行做子集搜索：若prefix行是query行
的子集，就用XOR保留差分spike，并在计算query输出时先读取prefix输出partial sum，再累加差分
spike对应的weight row。硬件模型还包含popcount/搜索、stable sort、prefix table和issue顺序。

因此Prosperity与本项目的关系是：

- 强先例：SNN无损product/result reuse、在线metadata生成、row-wise processor和prefix结果回读；
- 差异：本项目不搜索脉冲行子集，不依赖前一个token的输出结果，直接使用SCS已经产生的最终
  Q1.7 gate code，将`gate × folded weight`向量多播到独立token accumulator；
- 强制基线：不能只比较bit sparsity，至少要与Prosperity式subset-prefix重用的搜索、状态和周期
  开销做概念及模型对照；
- 评价风险：官方`energy.py`以论文给定on-chip power乘cycle时间，并用固定DRAM比例估算能耗，
  不是本项目可直接继承的门级活动功耗。我们的最终功耗必须来自本RTL真实trace SAIF和同库分析。

### 9.2 FuseMax artifact给出的评价范式

本轮检查了[FuseMax官方artifact](https://zenodo.org/records/13377043)中的
`workspace/src/accel/proposal.py`和`cascade.py`。它把attention拆成多个Einsum，分别建模2D/1D
阵列、L3、register file和functional unit，再从Timeloop统计中提取scalar read/write/compute，
生成Accelergy action-count YAML。最终延迟由计算waterfall和memory latency取最大值，而不是将
每个operator周期简单相加。

对HIT-Flow的直接要求：

1. H60/SCS/NMF/projection/RPI必须有逐动作计数，不得只报乘法减少；
2. weight SRAM、destination bitmap、accumulator和metadata分别计read/write；
3. 流水重叠按明确stage依赖取critical path，不得把所有局部speedup相乘；
4. 系统模型、RTL综合和SRAM模型分层保存，DC结果回填系统DSE；
5. FuseMax已经覆盖attention级operator fusion，所以“不物化gated-K”只能作为本工作组合的一部分，
   新颖性必须落在最终gate码前推和窗口组目的多播。

### 9.3 SWAT对窗口数据流的最近邻约束

[SWAT，DAC 2024](https://arxiv.org/abs/2405.17025)已经提出window attention的row-major、kernel
fusion和input-stationary FIFO，并将QK、softmax numerator和SV组织成流水。因此本项目不能泛称
“首次窗口驻留或row-wise融合”。H67/H68使用固定9×9×T2的独立Swin窗口，并没有SWAT滑窗之间的
K/V overlap复用；可辩护增量只可能是同一block/head的独立窗口在**最终归一化gate code**上产生
运行时等价类，从而复用后续projection product，而非复用重叠K/V行。

### 9.4 Transitive Array进一步收紧乘法消除边界

[Transitive Array，ISCA 2025](https://arxiv.org/abs/2504.16339)已经对bit-sliced GEMM建立Hasse
偏序、动态scoreboard、负载均衡lane和Benes/crossbar distribution，并主张无乘法结果复用。
因此gate-code CSD shift-add、product table或Benes网络不能作为主贡献单独出现。它们只在同约束
DC证明对最终gate histogram有净EDP收益时，作为WG-GPS内部电路消融保留。

源代码级复核后的架构收敛见
`docs/68_最终Gate码驱动的窗口组数据流与架构创新边界.md`。
