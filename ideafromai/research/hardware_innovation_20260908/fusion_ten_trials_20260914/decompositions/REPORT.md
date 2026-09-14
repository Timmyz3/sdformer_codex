# 三条完整代数接口的RTL筛选

2026-09-14。三个不同表示均完成原生source→完整Q1→完整Q2→rawp，C=N=96、K864、R8、T10、4×4→2×2。每条112runs/430080个整数输出通过，总336runs/1290240输出；三条真实八块均未胜过最新双P packed/native窗/cachedOS。以下为实现者的结果与差分自评，独立意见见[root审阅](../REVIEW_DECOMPOSITIONS.md)。没有重复旧delta、动态z幅值cache或同Ppair实验。

|固定接口|同资源强控制14|候选15|候选增加周期|背压控制→候选|
|---|---:|---:|---:|---:|
|[D1 Q1 bitplanes/popcount](q1_bitplanes/REPORT.md)|87599|103908|18.618%|91295→110023|
|[D2 Q2 R4×2 distributed arithmetic](q2_da/REPORT.md)|87599|100043|14.206%|91295→103717|
|[D3 Q1有限32类字典sum-first](q1_dictionary/REPORT.md)|87599|90217|2.989%|91295→93959|

数字为真实八块首命令核心总周期，包含完整本叶source加载/布局、计算、完整psum写出/drain及相应等待。配置另列：D1/D2每fixture3361拍；D3为4258=3361+897元数据，两侧相同。常量常驻、每块新送1537拍source/origin时，八块总量分别为99895→116204/112339/102513；不混入未执行消费者、全帧或AEE数据。

三条mode14分别与上一阶段最终mode15作逐记录比对：每条56行×16个标量指标=896项及56个完整状态数组完全复现；不是用旧scalar或重复取Q2作弱分母。每条额外单元/数组均同时给14/15权限，未综合，不能由同模块推断单独裁剪后等面积。尤其D1增加4320B源bitmap、40路bit写、128bit位平面权重读和8个16bit popcount树；这是真实资源条件，不是免费的预处理。

[verify.py](verify.py)只读独立重算所有14fixture的整数函数、位平面/DA/字典恒等、metadata、状态和与完整事务。每条53760个gold重算，所有112行验证通过；DA还检查43008个LUT系数值及最小signed宽度。计数核验不充当额外RTL实验。所有SV使用Verilator4.028 `-Wall --cc --exe`，TB仅供原生source/因子/metadata/gold、背压和无reset重启，动态bitmap/LUT/count均在SV。

三条停止的是本固定布局。D1在all-one 63103→24861，D3在极值同列控制63709→18049，说明密集输入或重复列分布可改变结果；这些只作机制反例，不替代真实性能。D2实际LUT消除了Q2乘法，却增加构表、位置展开与查询发射，当前未获得周期收益。没有缓存大小、组宽或位宽扫描。

## 先验与可能X

D1的AND-popcount、移位/符号和32bit累加已由[BISMO原论文](https://arxiv.org/pdf/1806.08862) §II/Algorithm1、§III-A/Fig4给出；此次只在完整原生卷积布局上实装该代数，不声称迁入BISMO全部fetch/execute/result指令流水。D2的分组LUT、逐位查表和最高符号位相减是[HDL Coder官方DA说明](https://www.mathworks.com/help/hdlcoder/ug/distributed-arithmetic-for-hdl-filters.html)明确支持的既有A；本点的最小signed宽度与整LUT词零过滤仍属普通执行优化。

D3的重复权重先求输入和属于[UCNN ISCA2018](https://www.kartikhegde.net/media/UCNN_ISCA.pdf)既有A（复用此前§III-A/B精读）；这里静态键为完整8rank signed3向量，32类容量、三块计数和组退休是新本地实施合同，不是未经证实的论文X。Q1真实862个非零K里有793种完整列，32类只覆盖69个K，故省下42次Q1读取却付出更多状态服务。来源和实际阅读范围列在[source_table.json](source_table.json)。

可能的X必须超出普通bitserial、DA或weight repetition，并证明额外布局/状态/消费者义务可被净省掉。本次三点都未做到，保留完整负控制，不因它们失败而否定三个家族。未训练、未新跑消费者或AEE、无PPA/Fmax/能耗结论。

复现：固定SV/TB为唯一实现源。先`/opt/anaconda3/bin/python3.12 prepare.py`只复制旧fixture并编静态字典；各子目录执行`run.py`；最后本目录执行`verify.py`。运行脚本不生成或替换RTL。生成期间的临时变体脚手架已移除。
