# 窄latent改变可执行接口，但尚未形成独立X

以下是本实施者的差分自评；独立新颖性结论由root给出，不把作者自评冒充外部审阅。

本轮围绕已有完整R8：`z=Q1g`，Q1∈[-3,3]、K864，`p=Q2z`、Q2 signed16、R8、N96、T10。没有中间RNE；z的完整界为±2592，signed13足够。普通低秩、小整数累加、Q2缓存及OS都属于完整A。本报告记录本轮提出并推进的接口，不声称独立盲法；新近邻阅读与未取得全文边界见[source_table.json](source_table.json)。

|接口与昂贵余项B|最强控制/必须付费|本轮推进及可诚实留下的X|
|---|---|---|
|同(P,T)八rank存在相同绝对值：Q2反复乘同幅值|完整R8×N8 Q2缓存OS、最终位置/rank零支持；两侧同19×13乘。共享符号组描述符、prefix/start-count、abs、有限cache构造/epoch/tag均收费|[partition](../partition_rtl/REPORT.md)实际120runs通过，真实13比12慢3.91%。新本地接口是runtime signed支持到有限系数cache；UCNN/Phi已有大部分代数与pattern-reuse A。未证X。|
|signed z的时间/位置重复可用base+残差或共享乘积|不能把binary XOR冒充signed减法；前值/基值、14bit差、全N prefix/回写以及每条输出义务都收费。分母仍需cachedOS|诊断中z非零1463，时间Δ非零1897，原spike的区间失败条件未自动解除。root独立实现Δ/prefix，因此不重复。单纯乘积cache若MAC和加法本已同拍，减少乘法数不等于减少周期。|
|同k、同t相邻P同时活动，signed13状态只占32bit ALU一部分|同26bit bank/208bit向量口与carry-cut能力给scalar/dual两侧；共同160bit源窗消除旧gather税；第二级同cachedOS。不能与旧13bit口报同面积|[packed](../packed_rtl/REPORT.md)实际112runs通过。真实8tile14→15为91103→87599（−3.846%）。普通SIMD子字加法及窗口是A；本次测到了完整源义务到双位置状态事务合并的增量，尚无区别于既有精度可重构架构的论文X。|

本批第一可写合同是第一行的有限8项cache：320个29bit描述符上限、每N8一个epoch、无无限PWP。完成后其5376拍MAC减少被7920拍构造及描述符/abs/epoch抵消。按signed支持排序后跨P/T重放，只在当前构造方式下最多省708拍，少于3862负差；它还会丢失寄存器OS而新增psum scatter读写，所以没有晋级为另一个RTL点。这不是对所有共享表示的否定，也不是缓存大小扫描。

最后一行选择横向P0/P1、P2/P3，同t共享Q1。普通未pack的真实6413个消费者变成5245个非空pair；carry断开保证低半字段负数更新不污染高半。合并1168次完整读/ALU/写，实测减少3504拍；Q2的17556次MAC、完整3840向量psum读写不变。明确反例是real_2：只有3个活动消费者且没有双P重叠，14和15同为6410拍，位宽打包没有性能增益；若Fmax因更宽端口/地址选择而下降，周期改善也不足以推出延迟改善。

## 近邻差分与完整A边界

[UCNN ISCA2018原文](https://www.kartikhegde.net/media/UCNN_ISCA.pdf) §III-A/B给相同权重分组、间接输入/权重表及跨filter输入组复用；本轮的绝对值latent分组是交换权重和activation角色后的同类代数。§III-C提出partial-product reuse但未与前两项合为论文实现，因此不能说本核完整复现了UCNN，也不能反过来将同幅值分组单独算X。

[Phi](https://arxiv.org/html/2505.10909v1) §3.1–3.2、4.3–4.4的PWP、双向差分、L1选择/加法与L2打包构成更完整系统；本8项cache只借有限pattern-coefficient片段，没有PAFT/kmeans、完整两级processor及全部布局。它显著限制“signed support缓存”可声称的新颖范围。

[Prosperity](https://arxiv.org/html/2503.03379v1) III、V把binary subset/product reuse推进到TCAM发现、排序及依赖执行。signed z没有同样的二值subset/XOR等价；需要显式减法及父值生命周期。可检验的新接口必须同时承担父值存活、psum完成与非因果T10的消费义务，不能只沿用“差分”名称。[Comperity原DOI](https://doi.org/10.1145/3828526)本轮仍被403阻挡，只有已存primary Crossref摘要，不能排除其全文已有某个具体时域缓存设计。

[LoAS](https://arxiv.org/html/2407.14073v2) IV-C/D、V的快/慢prefix修正、FIFO及压缩fiber/cache依赖binary时间向量。窄latent允许减法和小标量乘，却不保证基向量更稀疏；本真实Δ增加29.67%非零是必须付费的反证。上一轮原spike interval端点的负结果只停止该端点，不能杀LoAS全A。

[SmartExchange](https://arxiv.org/pdf/2005.03403) IV-A/B强调近PE重建、basis驻留与稀疏PoT系数。当前Q1逐活动累加不是按系数码分桶/直方图，也未复现其完整重建引擎。把latent缩窄后塞入双P bank，是有实际服务收益的本地表示融合；是否能成为X仍需更强的架构差分证据，不以本阶段功能和周期通过代替新颖性结论。

这些记录不含新训练/AEE，不抢root的消费者/valid825口径。局部质量继承同一整数函数；完整网络质量与本模块周期是不同证据。没有PPA、频率或能耗结论。
