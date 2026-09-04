# HIT-Flow架构创新冻结候选与论文证据路线

**日期**：2026-07-13  
**状态**：RTL前架构候选；创新表述未签核  
**适用主线**：H67功能超集，H68编译期特化

> **2026-07-13最近邻反证修正**：VESTA已覆盖统一SNN Transformer PE、多时间处理和STDP列流；FireFly-T已覆盖稀疏/二值双引擎、多lane decoder、OOO负载均衡和跨head延迟隐藏；T-REX、ULSeq-TA、STAR及ISSCC 2025层融合芯片已覆盖动态batch、双向buffer和跨算子/跨阶段融合。本文三个候选只能按H67特有的T10/T2 PSN、K-zero分母精确语义和部署图生命周期路由主张增量。完整威胁矩阵与收紧门槛见文档62。

## 1. 核心判断

当前最值得做的不是照搬Bishop的稠密/稀疏双核，也不是把复旦ISSCC蝶形zero skipper换一个名字，而是建立一套由本网络真实执行图和光流workload共同约束的完整encoder数据流：

> **HIT-Flow-LR：带生命周期路由的Head-Invariant Temporal-Tile架构。** 统一时间矩阵、二值event、时间对attention、稀疏projection和多位residual的瓦片坐标，并根据生产者-消费者生命周期决定局部转发、bank驻留或长skip写回。

它包含三个可能成为主贡献的架构机制：

1. **DP-TME双时间尺度除数打包阵列**：同一`32×10`阵列执行T10和五路T2；
2. **LR-HTT生命周期路由瓦片**：跨PSN、TESSA、FGP进行局部驻留和直接转发，避免5.26亿event元素/帧全部写回再读出；
3. **CCSP精确类合并稀疏投影流**：PCCC/class-stationary SCS与FGP共享稀疏`{K,gate,tag}`流，不落地dense gated-K。

RPI多位残差岛是完整性所必需，但单独新颖性较弱。BMRF蝶形mask-reduce和跨样本persistent-HTT均为条件候选，不提前列为贡献。

## 2. workload到架构的因果链

| 实测或结构事实 | 硬件问题 | 架构响应 | 证据等级 |
|---|---|---|---|
| 四stage均为9×9窗口、head_dim=32 | stage间几何相同但C/head数不同 | 固定HTT坐标和按head流式复用 | A |
| 活跃PSN只有T10与T2，且10可被2整除 | 独立T10/T2阵列会有闲置和重复控制 | DP-TME五路T2打包 | A，PPA待补 |
| 81活跃点产生5.260亿event元素/帧 | 中间event物化占256-bit端口约411万拍 | LR-HTT local-forward/bypass | C，需liveness/RTL |
| 45个单消费者点占4.215亿event元素/帧，即80.13%静态上界 | 大部分event具备相邻消费者直通资格 | 弹性Forward，stall时无损退化Resident | A/C，静态合同已完成、实际比例待ordered |
| pair-empty约74%、双K-zero约83% | value/projection与class事务不对称 | TESSA class后端和K-zero投影门控 | B |
| K lane约1.17%，非零token/head仅约3.29 lane | dense gated-K与逐lane乘gate浪费 | FGP selected-weight accumulation | B，完整RTL待补 |
| S0B0与S0B1 active-entry相差约19.6倍 | 固定服务时间造成前后端阻塞 | 多context与block descriptor | B/C，ordered待补 |
| S0-S2 skip共1161万元素且旧张量近全非零 | 二值event bank不能承载ADD residual | RPI精度岛和生命周期管理 | A/C，位宽待补 |
| 光流活动可能沿运动边缘和相邻样本持续 | 可能存在方向bank或跨窗口复用 | profiler已补方向、bank和跨样本delta | D，等待结果 |

只有前三列形成闭环的机制才能进入架构图。没有workload证据的动态路由、跨帧复用或对角bank都只能作为候选。

## 3. 创新一：DP-TME

### 3.1 机制

物理阵列为：

```text
32 channel lane × 10 temporal-output slot
```

- T10：一个空间位置占据全部10个slot，10个输入时刻串行，10拍完成；
- T2：10个slot分成五组，每组2个slot，同时处理五个空间位置，2拍完成；
- 81位置窗口的T2五路纯计算为17组、34拍，尾组利用率`81/85=95.29%`；只有5个输入银行和足够宽的packet出口同时成立时才是系统下界。单32-bit出口为162拍，G4/128-bit为当前42拍平衡候选。

100组随机整数参考中T10比较2,592,000个hidden/event、T2比较518,400个hidden/event，均0 mismatch。

### 3.2 与已有工作的区别

PTB、LoAS和ISCAS 2025已经覆盖time batching、temporal parallel和可重构timestep。可争取的新点不是“首次时间并行”，而是：

- 网络同时存在`T_long=10`和`T_short=2`；
- 二者满足除数关系，且所有stage固定D=32；
- 用slot分组把短时间尺度转化为空间并行，不需要第二套阵列；
- 同一输出直接进入HTT bitpack和attention pair布局。

VESTA式固定时间组统一PE也必须作为基线；仅优于独立T10/T2阵列不足以证明架构增量。

### 3.3 晋级门槛

与“独立T10阵列+独立T2阵列”在同工艺、同吞吐和同SRAM端口下比较：

- 总面积或EDP至少改善10%；
- 相对通用可重构时间阵列或VESTA式固定时间组PE，EDP至少改善5%；
- 两模式真实利用率均不低于70%；
- bit-exact、溢出和饱和规则全部关闭；
- 若仅减少控制器而MAC阵列面积增加，降为实现优化。

## 4. 创新二：LR-HTT生命周期路由

### 4.1 为什么它比普通fusion更像系统架构

全物化ATLIF二值输出的最低流量为：

```text
526,046,400 element/frame × 1 bit × read/write × 30 FPS
= 3.945 GB/s
```

在256-bit单端口下约4.110M拍/帧。这个成本与双DP-TME的6.913M拍同量级，说明生产者-消费者数据路由会决定系统吞吐。

LR-HTT为每个瓦片维护：

```text
{stage, block, site, head, window, T_mode,
 representation, precision, next_consumer, lifetime, dirty}
```

根据下一消费者执行三种动作：

1. **Forward**：DP-TME阈值输出直接送TESSA或Spatial/FGP，不写L1；
2. **Resident**：消费者稍后到达，写binary HTT ping-pong；
3. **Spill**：跨block residual或S0-S2长skip写RPI/L2。

该选择由冻结执行图和descriptor决定，首版不做复杂在线预测。

代码静态合同已把81个活跃点分为：45个单消费者点、12个`proj_sn`双消费者fanout点和24个Q/K pair assembly点。对应元素分别为421,536,960、34,836,480和69,672,960/帧。80.13%仅是Forward资格上界，真实bypass必须由有序trace或RTL计数器给出。

### 4.2 与FLAT/FuseMax/RAWAtten的区别

已有工作已覆盖attention fusion、tiling和stage可重构。HIT-Flow必须证明的增量是：

- fusion跨越PSN时间矩阵、二值化、非标准class attention和selected-weight projection；
- 同一HTT同时携带二值event和多位residual的生命周期边界；
- 路由目标由本网络81个活跃PSN点和固定执行图生成，而不是通用QK-softmax-V；
- 量化收益用event-bank事务和端到端周期衡量，不只看算子中间tensor容量。

VESTA ZSC/STDP、ULSeq-TA、STAR、T-REX TRF和ISSCC 2025层融合均已覆盖部分融合或buffer重排。LR-HTT还必须相对“仅局部算子fusion”减少至少20%的总片上事务，否则降级为普通bypass。

### 4.3 晋级门槛

- 静态liveness证明每条forward不改变消费者顺序和数值；
- 相对“所有event物化”减少至少50%的event-bank读写；
- 相对“仅局部算子fusion”减少至少20%的总片上事务；
- 加入mux、valid/ready、buffer和长线后，系统EDP至少改善12%；
- 30 FPS候选必须在真实stall模型下保留至少10%吞吐余量；
- 若直接转发比例低于40%，降为局部bypass实现细节。

## 5. 创新三：CCSP精确类合并稀疏投影流

### 5.1 数据流

```text
128-bit pair
 -> exact sufficient-statistics
 -> pair-coalesced class commit
 -> shared class-stationary SCS
 -> sparse {head, token, Kbits, gate, threshold}
 -> selected-weight accumulation
 -> one shared scale per output partial sum
 -> projection residual
```

PCCC只在两个K-zero score确属同一最终class时执行`+2`，否则发两个commit；所有路径可旁路。K-zero仍贡献Shiftmax分母，但不进入projection权重读取。

FGP利用同一token/head的32个K lane共享gate：

```text
sum(W[d] * K[d] * gate) = sum(W[d] for K[d]=1) * gate
```

它与TESSA共享稀疏流，避免先生成dense gated-K再送projection。

### 5.2 新颖性边界

class histogram、归约网络、稀疏投影和共享scale分别都有先例。只有以下联合实现和证据可能形成增量：

- K-zero分母语义保持下的class commit合并；
- score/class/gate与selected-weight projection连续执行；
- 真实K lane分布下的权重事务和输出累加组织；
- 与普通dense gated-K、TESSA不融合、PCCC关闭三组对照。

还必须增加VESTA STDP式列流对照；如果收益只来自“中间K不完整落地”，则该点已被已有工作覆盖。

### 5.3 晋级门槛

- 软件整数参考、RTL和hardware-order模型逐位一致；
- PCCC真实同类率足以使attention子系统EDP净改善至少5%；
- CCSP相对dense gated-K物化减少至少15%的SRAM事务或能量；
- CCSP相对VESTA/STDP式列流的attention-projection EDP至少改善8%；
- FGP完整覆盖C=96/192/384/768、所有head拼接、BN folding和反压；
- 不能使用“理想全合并”数字作为论文结果。

## 6. 复旦蝶形网络如何使用

复旦ISSCC 2023的蝶形zero skipper服务静态稀疏权重和CIM输入分配。直接移植不构成原创，也可能在D=32下得不偿失。

唯一值得实现的候选是可旁路BMRF：

```text
4-bit动态membership/lane
 -> stable butterfly compaction
 -> 16-entry membership LUT
 -> sufficient-statistics reduce
 -> dense bitmap exact fallback
```

与原工作的差异落在动态四向量membership、压紧和充分统计量归约融合、以及exact fallback。其淘汰门槛为：

- union-event表示在真实block上的覆盖足够；
- 包含route control、metadata、fallback和FIFO后，TESSA前端EDP至少改善8%；
- p99时延不能比fixed bitmap恶化5%以上；
- 不通过就从论文贡献和最终RTL中删除，只保留文献调研记录。

## 7. 跨样本persistent-HTT候选

事件光流可能具有连续窗口重叠、运动边缘局部性和方向持续性，但当前没有结果证明stage输出可以跨样本精确复用。

profiler已新增同sequence相邻条目的：

- 精确相等率；
- active状态翻转率；
- 符号类别变化率；
- mean absolute delta及归一化值；
- 序列切换强制清空历史。

该候选只在以下条件满足时进入算法-硬件协同实验：

1. 至少一个高流量stage边界精确相等率超过70%，或active翻转率低于10%；
2. 复用判定无需读取与全计算相当的数据量；
3. exact reuse有严格tag和序列边界；近似delta更新必须重新训练并过full30/valid825；
4. 相对仅做block内LR-HTT，系统EDP额外改善至少8%。

在结果出来前，它不是贡献。

## 8. 三档物理候选

| 候选 | DP-TME | Spatial lane | ctx | HTT端口 | 定位 | 当前代理 |
|---|---:|---:|---:|---|---|---|
| HIT-L | 2×320 | 512 | 2 | 512-bit或双256-bit | 面积边界 | 75%bypass、512-bit时30.23 FPS |
| HIT-B | 4×320 | 512 | 2 | 256/512-bit DSE | 平衡主候选 | 75%bypass、256-bit时37.55 FPS |
| HIT-X | 4×320 | 1024 | 4 | 512-bit | 吞吐/面积上界 | 理想PCCC下56.10 FPS |

这些FPS来自旧全网event-operation代理和1.25倍保护系数，不是RTL结果。HIT-X依赖理想PCCC，只用于上界，不是推荐实现。

当前RTL优先实现HIT-L/HIT-B共同的参数化模块，先综合`N_DP=2/4`、`N_SPATIAL=512`和`BUS=256/512`，不为HIT-X单写一套控制器。

## 9. DATE论文的贡献层次

完成RTL、DC和真实trace后，建议按以下层次写：

1. **算法-架构协同**：H67 all-binary Motion-XOR/TTX形成可由class和selected-K精确执行的部署语义；
2. **系统架构**：LR-HTT按生命周期贯通PSN-attention-projection-residual，减少event物化和格式转换；
3. **计算架构**：DP-TME通过T10/T2除数打包统一两种PSN时间尺度；
4. **attention/projection微架构**：CCSP在保留K-zero分母语义下融合class commit、SCS和FGP；
5. **条件电路/互连贡献**：BMRF或persistent-HTT只有过门槛后加入。

不能以“ANN方法首次用于SNN”作为主要新颖性，也不能隐去PTB、LoAS、FLAT、FuseMax、C-Transformer和复旦ISSCC 2023的来源。

## 10. 论文图和实验

### 架构图

1. Fig.1：H67软件执行图、81活跃PSN点、两次block ADD和S0-S2 skip；
2. Fig.2：HIT-Flow-LR顶层，突出DP-TME、LR-HTT路由、TESSA/CCSP和RPI；
3. Fig.3：T10与五路T2在同一`32×10`阵列上的映射；
4. Fig.4：瓦片生命周期时间轴，标出forward/resident/spill；
5. Fig.5：CCSP从pair到projection的融合流水；
6. Fig.6：BMRF与fixed bitmap可旁路结构，仅在晋级后绘制主图。

### 必须对照

| 机制 | 基线 | 候选 |
|---|---|---|
| 时间阵列 | 独立T10 + 独立T2 | DP-TME |
| 中间存储 | 全event物化 | 40/50/75%及真实LR-HTT bypass |
| attention | 162-token serial | 81-pair、TESSA、PCCC开关 |
| projection | dense gated-K | FGP/CCSP |
| context | 1 | 2/4 |
| 表示 | fixed bitmap | BMRF开关 |
| 精度 | FP/16-bit参考 | ATLIF Q4/Q6/Q8、RPI 4/8/16-bit |

报告必须分开面积、动态功耗、漏电、SRAM能量、周期、p99 stall和外部带宽，不能只报一个总EDP百分比。

## 11. 当前架构签核

| 项 | 判断 |
|---|---|
| 功能规格 | H67/H68 attention和PSN语义已较完整 |
| 三档候选 | 已有，HIT-L/HIT-B为实现主线 |
| 性能模型 | 有代理模型，但置信度低到中 |
| 存储层次 | 生命周期与容量已定义，RPI位宽未冻结 |
| 接口 | TESSA已冻结探索规格，LR-HTT/DP-TME接口待写 |
| 功耗/面积 | 无目标库与SRAM宏，不能签核 |
| 风险 | ordered trace、量化、event liveness、DC均为高风险未关闭 |
| 最终结论 | **NO-GO for paper PPA，GO for parameterized RTL exploration** |

这一路线的关键不是继续堆名字，而是用真实trace证明LR-HTT事务减少，用RTL证明DP-TME和CCSP精确，用同约束DC证明这些组合在完整子系统中仍有净收益。

## 12. Ordered结果的自动决策接口

`scripts/analyze_hit_flow_ordered_profiles.py`将真实结果统一映射到以下门槛：

- persistent-HTT：高流量stage精确相等率至少70%，或active翻转率不超过10%；
- PCCC：双K-zero中的同class条件覆盖至少70%，之后仍需5%子系统EDP门槛；
- BMRF：自适应表示流量相对dense bitmap至少下降8%，之后仍需同约束DC；
- ATLIF Q8：采样event零翻转只允许进入valid825，不等于量化签核；
- RPI：三条长skip只有`binary01_ratio=100%`才允许讨论1-bit，否则继续4/8/16-bit部署验证；
- Spatial lane：用逐算子encoder活动率加权MAC回填HIT-L/B/X预算，替换旧2.235G全网代理。

输出固定为中文`results/hit_flow_ordered_profile_analysis.md`，机器JSON只用于后续脚本和守恒检查。
