# Frontier joined：独立审阅

**原始X→部分时间源PSN→D展开→H384 FC1/完整PSN已实际接通，未见阻断数值或共享接口错误。** 给全部臂相同backend mode4普通内容去重并统一root参数bank后，992个非pair选择训练包上，新W″的普通resident-frontier code比static快11.157%/2.493%；early class在此之上仅再快**0.315%/0.301%（ready/BP，最后gate口径）**。主要收益归普通frontier/驻留，不能转给响应类别选择，更不构成新的算法或新AEE。

本审阅读取实际生成的 [joined_core.sv](joined_core.sv)、[frontier_source.sv](frontier_source.sv)、[TB](tb.cpp)、[prepare.py](prepare.py)、生成脚本和最终两份expanded CSV；只做Python3.12表核验，没有重跑RTL、训练或覆盖根报告。

## 数值、桥接与复用

源monitor以 `(P×96+c)×10+t` 索引原始X的独立完整10项点积，只检查 `producer_active[t]`，同时核唯一(P,c,t)、signed48 U与门。P由wrapper当前point提供，只有源DONE握手后才推进；部分U没有错配给下一个P。实际source MAC满足 `10×source_produced_pairs`，**不再等于100×source_channel_refs**；refs包含跨batch重复引用及驻留命中，不是唯一channel数或物理fetch数。

父joined TB正确记录refs，但它只从已核父source计数器读出，没有在父joined中独立重建每batch的refs。我已读 [active_prefetch/implement.py](active_prefetch/implement.py) 和其生成TB：该子链每个producer_valid清空96bit `batch_refs`，只对active t的channel去重累加nrefs，DONE再独立核 `count_source_channels==nrefs`，含义正确。这个新增assert不能倒记成父表已运行的检查。

实际每组code握手后，wrapper锁住十个标号，逐t真实读本地D并写一个16bit组；320×C96桥为3840B、D另192B。全部1920次展开/写被监测，每个H96消费者按顺序读取320个96bit桥字，H384共1280次读。源可在EXPAND期间继续下一组工作，但第二组输出会被sout_ready挡住；EXPAND不能覆盖未消费标号。backend直到32P源DONE和全部桥写完成才启动，四个H96顺序复用一个backend。源/消费者及外层top以DONE/start握手重用，测试仅初始reset。

source的A0/Q12、X/Q16、τ0/Q28与backend的A1/Q14、τ/sign/constant是**两套独立数值合同**。C++从原始X直接计算nearest-code、D展开、Y、完整T10 U和最终门；mode3另核D[class]与D[code]在当前完整H384 W上的响应逐值相同。prepare重算signed24 Y/signed48 U任意前缀界、INT10响应界；没有中途RNE或借入门答案。backend固定τ来自旧整数捕获，只定义此组件函数，不是动态BN供数或新整网质量。

## Root中性、资源及强普通消费者

源copy与冻结父叶的实际diff只有 `boot_addr` 去掉mode3额外+1。两图模式都读local222，wrapper映射为 `5952+222=6174`，仍落bank6；TB按模式把自己的六个root放同一物理word6174。模式镜像在任务前静态装入，不赠运行时root；均实际读一次。所有地址偏移5952、P×192、8192均为8的倍数，bank号不被wrapper悄悄改写。此前每P因222/223分处bank6/7的偏差已去掉，不以人工扣周期“校正”。

外部口确实只有8×128bit、每bank至多一个在途，dictionary/source/backend按wrapper状态独占grant，响应只交当前阶段；TB核请求保持、物理bank/初始化地址、在途所有权、最终pending清空。总256KiB是**16384×128bit外部共享X/参数池**，初始装池不在此组件服务内，每次实际读均计数。它不是全部片上存储：还包括桥3840B+D192B、源X128B+图cache128B及参数/控制，backend Y92160B、routes7680B、U5760B、τ5760B、payload384B、coeff192B、Y hold288B、乘法流水1920B及其它既存状态。[继承资源明细](../joined_chain/README.md)。

源10个16×24乘法单元与backend96个16×24单元**物理独立、串行活跃，共106个**，两者各自的累加器也仍存在。共享的是外部bank服务，不是把两个模块算作共96个MAC。没有EDA/同频/等面积证明。

同一P32内A0/τ0/D/rank/roots不变，当前仍逐P重装：static每tile960字、图臂672字。批内参数驻留尚未实现，故现表是明确收费的执行结果，不能称最小参数通信下界；若后续适配，应给普通static/code/class同样的批内复用权利，不能只去掉候选的启动税。

mode4强控已到齐：所有主臂都以同一实际W构造H96内容等价map，并实际装载每H96三字map；source exact-code与early-class具有相同普通去重权利。静态LUT/map只由D、W产生，动态输入只有X，DUT仍逐次读实际系数词。当前旧/新W均无非零D码的零响应，未发现因跳过code0类别而漏掉的有效普通合并。独立逐case核得，同函数各源模式的backend配置词、系数词、vector MAC和updates完全相同；ready下backend阶段周期也相同。BP周期差异仍可来自进入backend的全局日历相位，不能把它误称多做/少做消费者算术。

## 最终表核验：992训练P，完整H384

以下取frame1..31各固定32P；它们没参与此次pair选择，但仍是训练缓存，不称validation。主指标为 **`cycles_to_last_gate`＝最后一行gate实际握手时刻+1**，表示应用结果已交付。`cycles_to_done`另计回收握手，BP差值不是每帧固定2拍。

| 函数/臂 | 最后gate ready / BP | DONE ready / BP |
|---|---:|---:|
| W″ static64+PF + mode4 | 1,989,440 / 2,179,175 | 1,989,502 / 2,179,238 |
| W″ 整T10 code + mode4 | 1,879,645 / 2,232,487 | 1,879,707 / 2,232,562 |
| W″ resident-frontier code + mode4 | 1,767,487 / 2,124,840 | 1,767,549 / 2,124,911 |
| W″ resident-frontier class + mode4 | 1,761,921 / 2,118,445 | 1,761,983 / 2,118,513 |
| W′ resident-frontier code + mode4 | 1,767,487 / 2,124,910 | 1,767,549 / 2,124,981 |
| W′ resident-frontier class + mode4 | 1,763,758 / 2,122,178 | 1,763,820 / 2,122,249 |

W″的frontier code对static为−11.157%/−2.493%，BP仍有frame7负例。W″ class对该code仅−0.31491%/−0.30096%，BP仍有frame2、31负例；W′ class对其同函数code仅−0.21098%/−0.12857%。W′/W″是不同函数，不能把二者周期差直接视作同精度收益或混用AEE。[新表](expanded_new.csv)、[旧表](expanded_old.csv)。

两份expanded各340任务，全680任务核过H384；每任务122880个位置，合计**83,558,400个Y、同数U、同数最终门**，实际有效源U/门各11,263,822，最终code/class标号1,305,600。两份smoke各80任务，pass0/1的对应40条记录除pass外全部一致，覆盖无reset重复使用；这些重复任务不混入992P性能分母。表核包括 `cycles_to_done=Σwrapper state`、三阶段计时拆分、实际各类words之和、bytes=16×words、MAC=10×pairs、12字D、1920桥写/1280桥读、32次source配置与四次mode4参数装载；后端比较取实际运行，不相加叶周期。

## 归因与尚未闭合

这是普通多路径BDD、需求驱动供数、有限驻留与内容去重的完整组件迁移；根中性及相同mode4使“只早知道等价”的独立余量显著缩小。响应函数商类反推必要源判定仍可保留为候选接口，但当前**约0.3%**的整链余量、训练输入范围和未知新AEE不足以立独立创新标题，暂评维持3/10。父source仍只沿旧最小channel预取；active-prefetch子链在实际验证中，本报告只确认其nrefs监测增补，不提前引用子链收益。无需再扫布局救分，后续判断应依据该固定适配的真实整链收据及质量，保留成熟A底座。
