# HIT-Flow最近邻反证与架构贡献修正

**日期**：2026-07-13  
**状态**：RTL前新颖性审查；用于约束贡献表述和对照实验  
**适用主线**：H67功能超集，H68编译期特化

## 1. 结论先行

最近邻检索表明，以下宽泛表述均已被已有工作覆盖，不能作为本文创新：

- 统一PE支持SNN Transformer的卷积、线性和attention；
- 多时间步并行、tick batching或可重构时间步；
- 稀疏引擎与二值attention引擎组成的双引擎；
- attention与projection流水重叠、中间矩阵不完整落地；
- attention算子融合、跨层融合、跨阶段协同tiling；
- 动态batch、双向可访问buffer、蝶形zero skipper；
- 仅凭“ANN方法首次迁移到SNN”主张原创。

因此，HIT-Flow不能靠换名包装通用结构。可继续争取的架构增量必须同时满足：

1. 对应H67/H68独有的可执行语义，而不是普通Spikformer或标准softmax attention；
2. 与最近邻实现使用同一workload、相同端口和工艺假设进行消融；
3. 有完整RTL、hardware-order逐位一致、真实trace周期和DC PPA；
4. 证明收益来自新组织方式，而不是位宽下降或训练模型本身。

当前判断仍为：**GO for architecture/RTL exploration，NO-GO for论文PPA与创新签核。**

## 2. 最接近的已有工作

### 2.1 VESTA：最强SNN Transformer统一执行先例

[VESTA](https://arxiv.org/abs/2503.20246)已经实现：

- 同一组PE支持卷积、线性和dot-product；
- 四个时间步共同处理，TFLIF把多位累加直接转为spike；
- ZSC利用连续时间步排列并避免卷积中间输出落地；
- WSSL对四时间步做weight-stationary线性计算；
- STDP在列完成后立即计算dot-product，只暂存一列而非完整中间矩阵；
- SystemVerilog、TSMC 28nm DC、500MHz、107KB SRAM和芯片级面积/功耗口径。

对本项目的直接约束：

- 不能写“首次SNN Transformer统一PE/统一数据流”；
- 不能把时间片共同处理或spike层间存储压缩写成首次；
- 不能把“不落地完整attention中间矩阵”作为LR-HTT或CCSP的唯一新颖性；
- VESTA-like固定T4统一PE和STDP列流必须进入基线。

我们的潜在差异只能收窄为：`T10/T2`同时存在的PSN时间矩阵、H60 Motion-XOR与K-zero分母精确语义、以及跨PSN-TESSA-FGP-RPI的冻结执行图生命周期路由。

### 2.2 可重构并行时间步Spiking Transformer加速器

[Hardware Efficient Accelerator for Spiking Transformer With Reconfigurable Parallel Time Step Computing](https://arxiv.org/abs/2503.19643)已经提出fully-parallel tick-batching和可重构时间步神经元，并并行展开四个时间步。

因此DP-TME不能主张“首次并行时间步”或“首次可重构时间尺度”。可争取的增量是：

- 目标算子不是递归LIF，而是`h=b+W[T,T]x`的PSN时间矩阵；
- `T_long=10`与`T_short=2`满足整除关系；
- 同一`32×10`slot阵列把T2时间slot重解释为五路空间位置；
- 输出直接形成H60的temporal-pair布局。

### 2.3 FireFly-T：双引擎、稀疏解码和延迟隐藏的直接先例

[FireFly-T](https://arxiv.org/abs/2505.12771)已经包含：

- sparse engine与binary attention engine双引擎overlay；
- 多lane bitmap稀疏解码；
- weight dispatch、worker维乱序和避免bank conflict的负载均衡；
- SRAM byte-write实现隐式数据布局变换；
- Q/K/V projection与QK、attention-V跨head重叠；
- 统一orchestrator适配不同网络拓扑。

所以“motion-density异构双核”“多lane稀疏解码”“OOO平衡”“稀疏与二值引擎流水”均不能作为本项目的主创新。双核只保留为对照架构。HIT-Flow若采用多context，必须强调独立row的**无预测精确发射**及K-zero denominator事务，而不能泛称首次乱序。

### 2.4 ULSeq-TA、STAR与ISSCC 2025层融合：跨算子/跨阶段融合先例

[ULSeq-TA](https://doi.org/10.1109/TCAD.2023.3329039)已经提出attention fusion架构、grouped sparse Softmax和dual-path sparse LayerNorm。VESTA的STDP也明确受其tile-wise fused linear-attention-linear启发。

[STAR](https://arxiv.org/abs/2512.20198)进一步提出跨阶段协调tiling、distributed sorting和sorted-updating FlashAttention。ISSCC 2025的[Memory-Compute-Intensity-Aware CNN-Transformer Accelerator](https://ieeexplore.ieee.org/document/10904499)则使用hybrid-attention layer fusion、KV/weight reuse和级联剪枝，并给出28nm硅后结果。

因此LR-HTT不能写成“首次跨算子融合”或“首次跨stage tiling”。它必须通过以下差异成立：

- 路由图覆盖PSN时间矩阵、event阈值、非标准class attention、selected-weight projection和多位residual岛；
- 每条边由冻结部署图计算`forward/resident/spill`，而非通用QK-softmax-V tiling；
- 统计对象是event-bank事务、RPI事务、stall与端到端周期，而不是只比较中间张量容量；
- 所有silent/K-zero路径保持Shiftmax分母语义，不做近似裁剪。

### 2.5 T-REX：动态batch和双向buffer的直接先例

[T-REX](https://arxiv.org/abs/2503.00322)的ISSCC 2025实现已经包含：

- 按输入长度把1/2/4个输入动态batch到同一计算资源；
- two-direction accessible register file支持矩阵按行或按列读写；
- 通过双向访问消除重复SRAM搬运，并报告12%到20%的利用率改善；
- 稠密矩阵核、稀疏矩阵核、全局buffer和辅助函数单元的完整芯片组织。

因此HTT不能仅凭“同一buffer支持不同访问方向”或“按长度打包多个输入”主张创新。LR-HTT的区别必须是编译期producer-consumer生命周期和event/residual精度边界；DP-TME的区别必须是T10/T2算子slot除数映射，而不是输入长度动态batch。

### 2.6 复旦ISSCC 2023蝶形zero skipper

[复旦官方介绍](https://fics.fudan.edu.cn/70/b1/c22203a487601/page.htm)确认该工作面向剪枝Transformer，使用in-memory butterfly zero skipper、动态局部attention和QK共享。

BMRF只能作为条件电路候选，不能进入当前主贡献。即使实现，也必须与原工作区分为：动态四向量membership、充分统计量归约、数字标准单元实现和exact bitmap fallback。若真实profile与DC不能证明净收益，直接删除。

## 3. 对当前三个主候选的威胁审计

| 候选 | 最近邻威胁 | 仍可能成立的窄贡献 | 必须新增的对照 |
|---|---|---|---|
| DP-TME | VESTA T4共同执行；并行时间步加速器；PTB/LoAS | T10/T2除数打包的PSN矩阵阵列，短时间slot转为空间并行 | 独立T10+T2；VESTA式固定时间组统一PE；通用可重构T阵列 |
| LR-HTT | VESTA ZSC/STDP；ULSeq-TA；STAR；ISSCC 2025层融合；T-REX TRF | 冻结部署图驱动、跨PSN-TESSA-FGP-RPI的精度感知生命周期路由 | 全物化；局部算子fusion；VESTA式列流；通用cross-stage tile |
| CCSP | VESTA STDP；FLAT/FuseMax；FireFly-T attention流水 | K-zero仍入分母但不读projection权重的class-stationary连续流 | dense gated-K；STDP式列流；TESSA与FGP分离；PCCC关 |
| RPI | 常见mixed-precision buffer | 保证ADD residual与长skip正确的必要边界 | 4/8/16-bit位宽DSE；不能单列创新 |
| BMRF | 复旦butterfly；FireFly-T多lane decoder | 动态membership与充分统计量归约融合 | prefix/bitmap/multi-lane decoder同约束DC |
| persistent-HTT | 视频/事件跨帧复用、T-REX batch复用 | 仅当同sequence精确稳定性足够时的tagged exact reuse | 关闭复用；读取判定开销；序列边界审计 |

## 4. 修正后的贡献表述

只有完成RTL和证据门槛后，建议使用以下克制表述：

1. **双时间尺度除数打包PSN执行阵列**：针对同一encoder内T10/T2 PSN时间矩阵，以slot整除映射把短时间尺度并行转为空间并行，并直接输出H60 temporal pair。
2. **部署图生命周期路由的异精度Head-Time Tile**：在binary event和multi-bit residual边界内，跨PSN、class attention和selected-weight projection选择局部转发、驻留或spill。
3. **保留zero-K归一化语义的类驻留稀疏投影流**：zero-K继续更新Shiftmax分母，但绕过K payload和projection权重读取；active K以稀疏tag流连续进入FGP。

不再建议把以下内容列为贡献：

- 统一PE、时间并行、双引擎、attention fusion、多context本身；
- RPI精度岛本身；
- 尚未通过profile和DC的BMRF或persistent-HTT；
- 仅有Yosys通用单元数或分析模型的PPA结论。

## 5. 收紧后的量化淘汰门槛

### DP-TME

- 对“独立T10+T2”面积或EDP改善至少10%；
- 对“通用可重构时间阵列/VESTA式固定组PE”EDP至少改善5%；
- T10、T2模式利用率均不低于70%；
- 所有PSN点逐位一致，且计入mode mux、slot控制和HTT packer。

### LR-HTT

- 相对全物化减少至少50%的event-bank事务；
- 相对仅局部算子fusion仍减少至少20%的总片上事务；
- 加入tag、mux、valid/ready、RPI和长线后，系统EDP改善至少12%；
- ordered trace最坏延迟满足30FPS并保留至少10%余量。

### CCSP

- 相对dense gated-K至少减少15%的attention/projection SRAM事务或能量；
- 相对VESTA/STDP式列流仍有至少8%的attention-projection EDP改善；
- PCCC若子系统EDP净收益低于5%，仅保留旁路功能，不列贡献；
- hardware-order整数参考与RTL逐位一致。

### BMRF与persistent-HTT

- BMRF含metadata、fallback和route control后，TESSA前端EDP至少改善8%；
- persistent-HTT相对block内LR-HTT额外改善系统EDP至少8%；
- 任一项不达标即淘汰，不用“有趋势”保留论文位置。

## 6. 论文必须增加的基线

| 基线 | 目的 |
|---|---|
| VESTA-like统一PE和固定T4/T-group映射代理 | 隔离DP-TME的T10/T2除数映射收益 |
| VESTA STDP式一列暂存与即时dot-product | 隔离CCSP/LR-HTT相对普通列流融合的收益 |
| FireFly-T式稀疏+二值双引擎周期模型 | 证明同构生命周期架构是否优于直接双引擎 |
| ULSeq-TA/FLAT式局部attention fusion | 证明跨PSN与RPI边界的额外价值 |
| T-REX式二维访问buffer代理 | 证明LR-HTT收益不是普通行列转置buffer带来的 |
| 复旦BMRF、prefix compactor和fixed bitmap | 淘汰不合算的复杂互连 |

跨论文芯片数字只能做归一化背景，不能直接作为本设计speedup分母。主要消融必须在同一RTL、同一SRAM模型、相同频率约束和相同H67/H68 trace下完成。

## 7. 真实workload统计对架构的决定关系

新ordered profile不是“锦上添花”，而是架构冻结条件：

| 统计 | 决定的结构 |
|---|---|
| T10/T2逐点调用、shape与活动率 | DP-TME阵列数、slot模式和时钟门控 |
| PSN输出binary/ternary/整数率与量化翻转 | event bank和RPI位宽 |
| pair-empty、K-zero、motion-zero的有序run length | exact issue层次、FIFO深度和门控收益 |
| 双K-zero同class率与commit burst | PCCC是否值得实现、hist端口数 |
| K lane数和active-entry分布 | CCSP decoder宽度、weight bank吞吐 |
| block/stage p50/p95/p99服务时间 | context数、descriptor调度和尾延迟 |
| 逐算子MAC、输入活动率和输出生命周期 | DP/Spatial比例、LR-HTT forward/resident/spill |
| 水平/垂直/对角局部性与bank conflict | 是否采用方向bank；无证据则保持row-major |
| 同sequence跨样本相等/翻转率 | persistent-HTT是否立项 |
| residual/skip数值域与分位数 | RPI 4/8/16-bit量化实验 |

当前已有profile足以证明稀疏性很高、block差异显著和中间event物化可能成为瓶颈，但不能决定BMRF、方向bank、persistent-HTT或最终context数。

## 8. 下一阶段实施顺序

1. 等待有序profile自动完成并执行字段完整性硬审计；
2. 生成上述十类统计、p99 FIFO和逐算子liveness；
3. 建立VESTA-like、FireFly-T-like和局部fusion三个同约束周期/事务基线；
4. 先实现DP-TME与LR-HTT最小RTL，不先实现复杂BMRF；
5. 实现CCSP时保留dense gated-K与STDP-style旁路，便于同RTL消融；
6. Icarus/Verilator逐位验证后再做Yosys结构检查和目标库DC；
7. 用真实trace生成SAIF，分别报告compute、SRAM、clock、routing和control功耗；
8. 只有达到本文件门槛后，才冻结DATE贡献和图表。

## 9. 最终风险判断

| 风险 | 等级 | 处置 |
|---|---|---|
| 统一/融合主张被VESTA、FireFly-T和ULSeq-TA覆盖 | 高 | 已收窄到H67精确语义和部署图生命周期 |
| LR-HTT退化为普通buffer bypass | 高 | 增加局部fusion和VESTA STDP基线，要求额外事务收益 |
| DP-TME只节省少量控制 | 中高 | 增加通用可重构时间阵列基线与5%增量门槛 |
| CCSP收益被K过稀导致前端开销抵消 | 高 | ordered trace与完整FGP RTL后硬淘汰 |
| BMRF复制复旦或FireFly-T稀疏decoder | 高 | 维持条件候选，默认不进主线 |
| 无目标库/SRAM宏导致PPA不可引用 | 高 | 保持NO-GO，获得库与宏后执行DC/功耗流程 |

这次反证没有否定HIT-Flow方向，但把可发表空间从“通用SNN Transformer架构”压缩到了三个可检验的窄问题。后续工作的价值取决于真实trace和同约束RTL对照，而不是命名。
