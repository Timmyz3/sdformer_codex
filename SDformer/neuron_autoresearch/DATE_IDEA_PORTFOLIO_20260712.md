# DATE 投稿 Idea Portfolio：算法线与硬件线（2026-07-12）

## 1. 当前核心：H67 Motion-XOR TTX

### 1.1 它实际做了什么

H67 保留全网105个 one-sided binary ATLIF，以及12个完全同构的H60 attention block。对每个
window、head、token，Q/K均为32-lane二值事件。原H60纯TX分数为：

```text
S_TX(t,p) = sum_d 1[Q(t,p,d)=1 and K(t,p,d)=1]
          + alpha0 * sum_d 1[Q(t,p,d)=0 and K(t,p,d)=0]
```

H67在同一空间位置读取另一个时间片的K，增加运动证据：

```text
M_K(t,p) = popcount(K(t,p,:) XOR K(1-t,p,:))
S_H67(t,p) = S_TX(t,p) + (1/4) * M_K(t,p)
```

随后仍执行：

```text
S_center = S_H67 - mean_token(S_H67)
G = Shiftmax(S_center)
Y(t,p,:) = G(t,p) * K(t,p,:)
```

`preserve_mean=true`会把Shiftmax gate按token数恢复尺度。这里的`K*gate`是H60保留的value输出，
不是旧QKFormer的native `K*sn2(sumQ)` carrier。H67没有SC、Kmag、token-token矩阵、动态路由、
第二套attention或stage混合。

直觉：事件光流依赖时间变化。H60只判断同一时刻Q/K是否匹配；H67用相邻时间片K的XOR-popcount
显式奖励发生变化的位置，让Shiftmax更关注可能包含运动的信息token。权重固定为1/4，不是运行时
浮点乘法；在冻结Q7部署归一化下可并入整数score加法。

### 1.2 当前证据

| checkpoint | AEE | AAE | spikes | 结论 |
|---|---:|---:|---:|---|
| NB0 ep59 | 1.4872 | 9.93 | 44.05G | 原始baseline |
| H60 TTX | ~1.5003 float / 1.5016 deploy | 9.8431 deploy | 23.2439G | 当前硬件参照 |
| H67 ep19 | **1.4671** | **9.4155** | 26.3898G | 当前精度第一候选 |
| H67 ep4 | 1.4896 | 9.6386 | 24.2416G | efficiency Pareto点 |

H67 ep19相对NB0 AEE改善约1.35%、spikes下降约40.09%；相对H60 float AEE改善约2.21%，但
spikes比H60 deploy高约13.53%。因此它已通过算法门槛，但必须等统一dyadic valid825和包含
Motion-XOR/SRAM的PPA，才能宣称总能效优于H60。

## 2. DATE 算法侧 Idea 清单

| 优先级 | Idea | 核心贡献 | 部署数据流 | 成熟度 | DATE中的角色 |
|---:|---|---|---|---|---|
| A0 | **Motion-XOR TTX (H67)** | 用`K_t XOR K_1-t`把事件运动先验加入二值TX score | H60加一个temporal XOR/popcount；all12统一 | full30+valid825已成功，部署评估待完成 | 当前主算法候选 |
| A0 | **Castling-trained TTX (H68)** | 训练期full-matrix attention作为富教师，权重退火至0；部署只留TTX | 部署与H60相同，辅助矩阵完全删除 | full30运行中 | 若成功，作为“train rich, deploy simple”协同训练贡献 |
| A1 | **Dyadic-Temperature TTX (H69)** | 用固定`2^k`温度解决H60 gate近乎均匀 | score左移，无通用乘法 | 已排队 | H67的必要score-selectivity消融 |
| A1 | **Event-Selective TTX (H70)** | 由`popcount(Q OR K)`选择0--3位动态温度 | OR-popcount+LOD+shift | 已排队 | 事件密度驱动attention选择性 |
| A1 | **Match-Code attention (H73/H74/H75)** | 不再输出动态`weights@K`，把跨时位移匹配保留为descriptor，再用静态codebook映射 | 固定offset matching+Shiftmax+静态低比特投影 | 代码/配置/加载审计完成，full30排队 | 若超过H67，可成为新的统一attention主线 |
| A2 | **Window-Context TTX (H71)** | 在TTX输出后做无参数window context broadcasting | reduce+固定倒数+broadcast | 已排队 | 检验token mixing是否是精度瓶颈 |
| A2 | **Full/Pairwise accuracy oracle (H66a-e)** | 用full matrix、temporal pair、local5定位匹配范围与角度误差来源 | 成本从2/5邻域到N²不等 | full30排队 | 论文机制消融，不优先作为硬件主线 |
| A3 | **GT displacement supervision** | 用GT flow监督Match-Code offset descriptor，训练期增强、部署不变 | 无部署增量 | 仅预研；等待H73-H75结果 | Match-Code不聚焦时的单次补救实验 |

### 算法论文最推荐的组合

1. 主机制：Motion-XOR TTX，强调event motion prior、统一all12、二值匹配和固定dyadic增量。
2. 对照机制：H60 TTX去掉motion；H69固定温度区分“动态范围改善”与“运动信息改善”；H70区分
   普通event density与真正temporal change。
3. 精度上界：H66a full matrix或H74 MC49，说明更大匹配范围是否值得硬件成本。
4. 部署训练：若H68成功，把它作为正交的training-only增强；若失败，则作为完整负消融。

不建议把所有成功候选堆在一个最终模型中。DATE故事应保持一个统一attention公式，其余机制作为
消融和设计空间证据。

## 3. DATE 硬件侧 Idea 清单

### H0：Density-Stratified Token-Time Bundle（TTB）双路径

事件光流在运动边缘形成局部活跃区，大面积背景和静止区域低活跃；同时H60/H67的T=2、
head_dim=32天然适合把多个空间token与两个时间片组成固定bundle。TTB作为work descriptor，先由
stratifier统计Q/K活性与temporal change，再分三类：

```text
zero-change bundle  -> 复用上一score，不发射Delta计算
sparse-change bundle -> sparse core：changed-lane index + 小型串行popcount/update
dense-change bundle  -> dense core：固定32-lane并行TX/Motion-XOR
K-zero bundle        -> 独立关闭gated-K/value/projection路径
```

推荐的平衡版本是一套dense core加一套sparse core，共享packed Q/K SRAM、Shiftmax和输出投影。
不建议一开始复制多套异构core：必须先用真实`T=2 × token1/2/4/8 × 32 lanes` profile证明bundle
密度分布和路由命中率，再确定core比例与FIFO深度。TTB只改变调度；zero-change score reuse和
K-zero value gating可bit-exact。一般Q/K empty不能直接删除score，因为silent/silent项仍参与
window-wide Shiftmax。

三档微架构、路由规则、memory/compute/control账本与B1准入门槛见
`hw_autoresearch_nts07/docs/45_TTB异构双路径微架构评估.md`。

### H1：Motion-XOR/Delta 共享时间差分前端

H67需要`K_t XOR K_1-t`；Exact Delta-TTX需要检测`Q/K`相邻时间翻转。两者可共享temporal
buffer、XOR mask和popcount/changed-lane检测器。统一命名可用 **Temporal-Difference TTX
Front-End**。必须分别报告H67 motion score路径和Delta增量更新路径，不能把共享面积重复计算。

### H2：Exact Delta-TTX

冻结`alpha0=1/64`后，将TX整数化为：

```text
S64 = 64*n11 + n00
```

t1只更新`Q_toggle OR K_toggle`为1的lane，数学上与完整重算bit-exact。100-sample测得union
toggle为2.7832%，t1理想lane skip为97.2168%，折算整个T=2窗口的compare下降上限48.6084%。
这是目前最强硬件贡献候选，但最终收益必须扣除previous Q/K state、S64 accumulator、index
queue、scheduler和SRAM访问。

### H3：64-bit Temporal-Pair Packing

T=2、head_dim=32时，同一token/head的两个时间片Q或K恰好组成64-bit word。采用timestep-inner
layout可同时服务H60、H67和Delta-TTX。若baseline原本发两个独立32-bit请求，则transaction可
由324降至162；若baseline已经合并，则收益为0。论文claim必须由RTL地址trace决定。

### H4：Zero-Activity Folding

对严格`K=0` token，`gate*K=0`，可跳过late-scale、projection input read和对应切换。该优化
bit-exact，不需要重新训练。需要报告每stage的K-zero token率、bundle空闲率、实际clock-gating
覆盖率和控制开销。

### H5：Token-Time Bundle Scheduler

按4/8 token与两个timestep打包，使用changed-lane bitmap或run-length驱动已有popcount阵列。
只有profile显示高empty-bundle或明显changed-run时才实现稀疏queue；否则采用固定8-lane grouped
scheduler，避免稀疏控制反而耗能。

### H6：Shiftmax Integer Pipeline

将score max、`2^x`移位、power-of-two denominator和gate定点化组成流水。当前冻结网格实际需要
10-bit score/9-bit gate，不能误写成INT8。H67 motion项必须在原始score域合并后统一round-to-
nearest-even，不能先分别舍入。

### H7：Weight-Stationary Match-Code Engine

若H73/H74/H75胜出，固定9/17/49 offset address generator、halo buffer和per-head静态codebook
可做weight-stationary流水。DE9/AX17/MC49共享同一可配置匹配阵列，但论文最终只综合胜出规模，
不能把可重构支持全部候选当作免费能力。

### H8：Error-Bounded Gate Bundling（近似备选）

binary K下，跳过token的投影前L1误差上界是`abs(gate_i)*popcount(K_i)`；bundle上界低于epsilon
时允许跳过。它改变数值，必须给AEE-epsilon曲线，不能与Exact Delta或Zero Folding混写为无损。

## 4. 推荐的 DATE 贡献结构

### 最稳主线

**算法：Motion-XOR TTX**  
**架构：TTB density stratifier + dense/sparse paths + Exact Delta + temporal-pair packing**

这条线的优点是算法与硬件围绕同一个“事件时间变化”主题：算法用temporal XOR提高光流精度，
硬件利用temporal delta减少重复计算。两者不是拼接的两个故事。

### 论文可写成三项贡献

1. 提出统一all12的Motion-XOR TTX，在全二值ATLIF网络中用低成本时间变化证据恢复事件光流精度。
2. 提出density-stratified TTB异构双路径，并结合bit-exact Delta-TTX与64-bit时间对布局减少重复binary match。
3. 建立包含neuron spikes、attention logic、Shiftmax、state SRAM和traffic的可审计协同评估，
   而不是只用spike count声称总能耗。

### 必须补齐的投稿证据

- H67 epoch19统一dyadic valid825，以及至少一个复现seed。
- H60 vs H67去掉/加入Motion-XOR；最好补H69/H70区分温度和事件密度效应。
- attention operation、SRAM traffic、cycle和post-synthesis PPA，spike proxy单列。
- Delta-TTX bit-exact测试、toggle/bundle统计、加入state memory后的净节能。
- NB0、H60、H67的同口径AEE/AAE/PE1/PE2/outlier/spikes/energy/latency/area表。

## 5. 不建议作为最终 DATE 主线

- TX/SC stage混合、S2-only或partial replacement：公式和硬件数据流不统一。
- full N×N matrix部署：可作精度oracle，但SRAM与N²D成本过高。
- Mamba/SSM整网重构：会推翻当前硬件，不是attention增量。
- 同时组合Motion-XOR、动态温度、context broadcast和Match-Code：无法做清晰单变量消融。
- 只报告`total_spikes × pJ`：它不含新增attention、SRAM、NoC和control。


## True TTB profile100 自动结果

<!-- TRUE_TTB_TTX_H67_PROFILE100 -->
- artifact: `neuron_experiments/H9_bipolar_self_attention/results/ttb_true_density_ttx_h67_h68_profile100.md`

| model | tokens/bundle | Q-or-K density | empty | K-zero | no K-motion | active 1--4 | active 1--8 | active 1--16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TTX ep2 dyadic | 1 | 1.691499% | 72.539530% | 82.282540% | 82.369915% | 19.717835% | 24.120280% | 26.879439% |
| TTX ep2 dyadic | 2 | 1.691499% | 66.127210% | 78.106500% | 78.168351% | 20.017777% | 26.023788% | 30.592211% |
| TTX ep2 dyadic | 4 | 1.691499% | 60.089640% | 73.525184% | 73.570075% | 18.556144% | 26.025553% | 32.241234% |
| TTX ep2 dyadic | 8 | 1.691499% | 54.704775% | 68.838231% | 68.871861% | 16.705019% | 23.933644% | 31.997132% |
| H67 ep19 dyadic | 1 | 1.502114% | 73.897325% | 83.106384% | 83.175281% | 19.043564% | 23.382608% | 25.765469% |
| H67 ep19 dyadic | 2 | 1.502114% | 67.319149% | 79.544915% | 79.584164% | 20.094138% | 25.416482% | 30.060322% |
| H67 ep19 dyadic | 4 | 1.502114% | 60.963301% | 75.636288% | 75.667241% | 19.685629% | 26.506512% | 31.869218% |
| H67 ep19 dyadic | 8 | 1.502114% | 55.255939% | 71.416599% | 71.443439% | 18.047010% | 25.623377% | 32.758293% |
| H68 ep19 dyadic | 1 | 1.548900% | 74.201277% | 83.292383% | 83.355065% | 18.398058% | 22.791555% | 25.405763% |
| H68 ep19 dyadic | 2 | 1.548900% | 67.769127% | 80.035264% | 80.068906% | 19.527250% | 24.596737% | 29.303615% |
| H68 ep19 dyadic | 4 | 1.548900% | 61.517474% | 76.512280% | 76.536657% | 19.329514% | 25.838329% | 30.946131% |
| H68 ep19 dyadic | 8 | 1.548900% | 55.880384% | 72.718655% | 72.740138% | 17.770766% | 25.208699% | 32.022727% |

`empty` cannot by itself remove silent/silent score contributions to Shiftmax. Bit-exact skipping is limited to proven Delta score reuse and K-zero value/projection gating.

## 6. 2026-07-13 阶段选择：软件精度与硬件简洁分开

<!-- H67_H68_DYADIC_TTB_PORTFOLIO_DECISION_20260713 -->

- H67 dyadic ep19：AEE `1.4626`、AAE `9.3949`、spikes `26.3948G`，当前软件精度第一。
- H68 dyadic ep19：AEE `1.4715`、AAE `9.4517`、spikes `26.4311G`，部署期完全回到H60，当前硬件简洁第一。
- H67相对H68的AEE收益为约`0.00885`，是否足以支付额外Motion-XOR/popcount，要由完整attention PPA决定。
- 4-token true TTB上，H67/H68的K-zero分别为`75.64%/76.51%`，K-motion-zero分别为
  `75.67%/76.54%`。这支持dense/sparse双路径和Delta复用，但不支持仅凭Q/K empty删掉Shiftmax。
- DATE表述保持两条证据链：算法线比较H60/H67/H68及后续H69-H75；硬件线独立证明TTB
  stratifier、Exact Delta、K-zero gating和64-bit temporal packing的周期、流量与PPA收益。

## 7. Round3候选与硬件证据边界

<!-- ROUND3_PORTFOLIO_20260713 -->

算法线在H73-H75之后新增三项互斥候选，不堆叠机制：

| ID | 候选 | 核心表达 | DATE价值 | 首要风险 |
|---|---|---|---|---|
| H76 | PC9 | Omega9逐位移score plane加固定3x3 patch一致性 | 事件噪声下的局部运动一致性与score-plane复用 | line buffer与边界过平滑 |
| H77 | LC4 | 学习n11/n10/n01/n00的dyadic列联代价 | 利用one-sided事件的方向不对称 mismatch | 学成密度先验、novelty中高风险 |
| H78 | G4 | 四个8-bit组各自Shiftmax9，保留36维证据 | 比较位数不增而避免scalar cost bottleneck | 四路归一化与静态投影面积 |

硬件线不把true TTB empty直接写成skip收益。C0/B1必须用stage/row trace回放，分别计metadata、
payload、score constant injection、Delta state、FIFO、bank conflict、backend与wake-up。当前数据只
证明存在明显路由机会；PPA结论必须来自相同top、SRAM wrapper、SDC、工艺库和compile effort。

## 8. RTL-exact数值冻结证据

<!-- RTL_EXACT_NUMERIC_FREEZE_20260713 -->

H67/H68在当前RTL Shiftmax的16项Q8 LUT、Q7 score、Q1.7 gate与RNE路径下，valid825 AEE
分别为`1.4627/1.4727`，相对原dyadic模型仅退化`0.0001/0.0012`。数值格式已通过精度门槛；
后续硬件贡献应集中在Motion-XOR增量、TTB/Delta调度、SRAM traffic和同工艺PPA，不再把扩大
LUT当作默认优化方向。

## 9. 硬件 Round4：TTB exact 分级落地（2026-07-13）

<!-- TTB_HARDWARE_ROUND4_20260713 -->

深读与实现候选见`hw_autoresearch_nts07/docs/47_TTB稀疏异构架构文献映射与实现候选.md`。
硬件线不直接复制 Bishop 双核，而按以下证据门逐级推进：

| 级别 | 候选 | 数值关系 | 当前裁决 |
|---|---|---|---|
| E0 | metadata-first temporal-pair C0 | bit-exact | P0推荐；先实现 silent constant、u=0 reuse、K-zero/motion-zero局部门控 |
| E1 | dense/sparse exact Delta B1 | bit-exact | 条件晋级；必须由stage/row trace、有限FIFO、SRAM端口和同工艺PPA证明优于E0 |
| E2 | product-class exact reuse | bit-exact | trace-only；match/subset命中率、reuse distance和净SRAM能耗不过门则不做TCAM |
| A线 | Bishop ECP/DOTA/SpARC等剪枝聚类 | approximate | 与exact主线隔离，若采用必须另做full30+valid825和误差曲线 |

cycle-v2 collector已实现Delta `u=0..32`、theta12、conditional changed-lane sum，以及各TTB
bundle的完整`A_b` histogram/conditional payload；47项attention测试通过。实际profile100安排在
H76-H78之后，artifact生成前仍不能声明cycle/energy收益。H69/H70另做量化前score clipping
profile20，避免把H67/H68的Q7无损结论错误外推到固定/动态温度线。

## 10. 算法 Round4：局部 assignment 对立实验（2026-07-13）

<!-- ALGORITHM_ROUND4_ASSIGNMENT_20260713 -->

全文公式和官方代码审计见
`literature/idea_mining_20260711/notes/DEEP_IDEA_MINING_ROUND4_ALGO_20260713.md`。本轮不继续扩大
offset support，而在同一 Omega9 二值 Match-Code 上检验两个互斥 assignment 假设：

| 计划ID | 候选 | 部署机制 | DATE作用 | 裁决 |
|---|---|---|---|---|
| H79 | CF10 | row-only Shiftmax9加显式fixed-zero null，第10候选由top2 margin和query density生成 | 允许many-to-one并显式表示不可匹配/遮挡 | P0，实现并独立full30 |
| H80 | DN9 | 同一9条局部边同时做source-row与destination-incoming Shiftmax，Q1.7 gate相乘 | 检验目标端竞争能否消除事件局部歧义 | P1，实现并独立full30；不与CF10组合 |
| 孵化 | AMM9 | 训练期多模态offset监督，部署仍plain Omega9 | 运动边界精度正则 | 暂不插队，先看H79/H80及plain Omega9家族结果 |
| 孵化 | BSMR9 | 训练期block-shared masked score reconstruction | 部署零增量正则 | novelty/训练复杂度高，当前不授权full30 |

H79/H80均保持ATLIF105、all12统一公式、静态codebook输出和无native carrier，从冻结TTX ep2
独立起跑，不能叠加H67/H76-H78。短跑只作实现健康检查，不用于淘汰；结论来自共同full30和
valid825。晋级标准提高为击败当前H67 dyadic AEE`1.4626`且spikes不高于`26.3948G`，同时报告
null occupancy或destination collision/row mass，防止靠输出抑制伪改善。

## 11. Round4 Assignment 实现与实测加载状态（2026-07-13）

<!-- ROUND4_ASSIGNMENT_IMPLEMENTED_QUEUED_20260713 -->

H79/H80实现、52项公式测试、冻结TTX加载审计和优化器归属审计均已完成。H79 warm-start仅新增
12个codebook与12个CF10 beta，H80仅新增12个codebook；两项均为overlay210、unexpected0，
同模式严格重载0/0。它们已通过独立runner排在H78之后，TTB-v2和统一19项dyadic deploy/op
audit排在H80之后。该状态只证明实现与链路正确，不代表算法成功；主线裁决仍等待full30+
valid825及attention/control/memory同口径成本。
