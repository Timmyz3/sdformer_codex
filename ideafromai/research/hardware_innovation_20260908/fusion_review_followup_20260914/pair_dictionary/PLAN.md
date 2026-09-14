# 固定四个 R2 字典：完整 native 线性核试验

2026-09-14。实现前固定：C96/K864/R8/N96/T10/P4，Q1[-3,3]、Q2 signed16，无中间舍入；对照为原 native-window dual-P packed z + rank/position/Q2-zero + cached R8×N8 OS。candidate 只改 Q1 累加表示；本轮不接 I24、不扩帧、不扫容量、不训练。

每个 rank pair 的6bit静态键建立至多32个频次≥2非零类，按频次降序/键升序固定选择。每 K 保存四个6bit code：0为零 pair、1..32为字典类、63为非零 direct fallback。类别由离线 Q1 得到；TB 只能送原 source/Q1/Q2/静态表和 gold。真实非零 R2 键数26/21/28/29，容量32覆盖这些键；高熵角落覆盖超过32类的fallback。

资源与调度先固定如下：

|资源|布局/端口|运行时费用|
|---|---|---|
|共享算术|原8条32bit加法、8个signed19×13乘法。计数态在bit10截carry，原dual-P z态在bit13截carry|每组占2条ALU，每条更新两个unsigned10计数；4组并行共8条，不新增数据ALU|
|计数|8bank×320row×20bit=6400B；每组2bank，组内共同地址，各组可独立地址；每bank每拍一读或一写|20个(P-pair,T)按两行分10块；每活跃K检查10块，非空块1读+1加写。每tile按max实际类数×10清全部8banks|
|K类别|864×24bit=2592B，一24bit读口|source全零时跳过；否则独立META_READ一拍。direct fallback先用该metadata屏蔽已计数rank|
|代表|32×24bit=96B，一24bit读口；同slot存4组各自两个signed3系数|每活动class-slot退休前读一次。各组slot含不同键是合法的，因为计数与乘法按rank独立|
|计数退休|一160bit count-hold；每slot每块读8bank，4bit pending取四个空间/时间位置中任一组非零的位置|选定位置时，四组各自取一个unsigned10 count，乘各自两个代表系数，共8乘法；一次208bit z读、一次masked-half写，无额外psum广播|
|元数据|32bit slot_live、4×6bit class-hold、maxclasses6bit、有限pending/address状态|clear/config/读写/控制状态和backpressure全部计拍|

四组独立地址意味着额外存储与地址选择是真实资源，不等面积。mode14/15同一个union SV、同八ALU/乘法/所有端口预算；没有单独综合后的面积/Fmax声明。control无需字典配置：控制3361拍，候选4258拍（+864类别+32代表+1尺寸），两张冷配置/运行表分别报告；warm只是本叶同源重复命令。

候选执行顺序：ZCLEAR→COUNT_CLEAR→native load/gather→source非空检查→META_READ→必要direct Q1 dual-P更新→10块count检查/读/写→所有K完成→按slot/class退休→原Q2完整输出。相同K的direct与dictionary rank不重叠，所有count在提交任何z增量前完成清零。所有 480×8 输出与独立 raw gold 比较。

验证：复用统一选定的旧分解14fixture，再加图外非零污染/随机稠密不同半字、全Q1极值同类count864、超过32类fallback。模式14/15×有无背压×两次无reset；无效metadata不属于配置合同，但Python必须逐rank核类别与代表相等。以模式14功能/核心计数复现原最强控制为门槛。负结果立即停止该布局，不扩大类别、不增加ALU、不挑样本。

本接口借用既有重复权重sum-first，未复现完整UCNN作者架构；待测的实际问题是局部键覆盖增加能否支付四个计数表、metadata和退休调度的费用。
