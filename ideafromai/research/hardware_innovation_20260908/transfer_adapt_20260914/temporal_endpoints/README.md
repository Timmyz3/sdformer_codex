已把旧端点负结果推进到**两域逐K混合＋一个固定校准时间顺序**。在与校准输入不重叠的同帧4000–4063共64tile，208bit z口下native2P13 direct冷服务1077632拍，同排列全endpoint单域mode5为1070536拍，混合mode4为 **1030068拍：比direct少47564拍（4.4138%），比同排列全端点少40468拍**。有BP时mode4比direct少4.1003%。全部计入输入/配置、start、Q1/Q2、两域clear、prefix/merge及逆地址raw退休，40bit排列首次配置1拍也已付费。

这不是胜过全部原生强控制：根代理补测的416bit z口四P8＋repair在两套64上仍更快，具体数值见下。当前正结果支持208bit口、额外520B z状态下的适配；尚未证明等面积优势或与强打包融合后的收益。

**2524条RTL命令、9692160个raw值全部通过**：26fixture六mode及跨mode共732条，held128–191四mode、两遍64及有/无BP共1024条，输入不重叠4000–4063三mode同协议共768条。原128集合与校准有source halo重叠，追加4000集合没有输入重叠，但二者都来自**同一帧**，不是跨帧或跨序列泛化。完整raw前已恢复原T顺序，所以该算术接口继承原质量；尚未接J/I24实际RTL，也未验证整网吞吐或新质量指标。

|held128–191，共64tile|mode0 direct|mode3 原序filtered mixed|mode4 固定序mixed|mode5 固定序全endpoint|
|---|---:|---:|---:|---:|
|核心拍，无BP|883426|888738|844199|866495|
|冷服务，无BP|983682|988994|944456|966752|
|暖服务，无BP|981858|987170|942631|964927|
|核心拍，有BP|918832|924445|881495|901957|
|冷服务，有BP|1019088|1024701|981752|1002214|
|暖服务，有BP|1017264|1022877|979927|1000389|

mode4对direct核心减少39227拍（4.4403%），冷服务减少39226拍（3.9877%）；少一拍收益是额外排列配置的实际成本。对同排列mode5再净省22296拍，隔离了付费混合两域的增量。每遍64的service均为各tile核心拍＋source/origin装载＋实际静态配置＋64个start拍；同一实例两遍之间不reset，第二遍不重载Q1/Q2或排列。

|输入不重叠4000–4063，共64tile|mode0 direct|mode4 固定序mixed|mode5 固定序全endpoint|
|---|---:|---:|---:|
|核心拍，无BP|977376|929811|970279|
|冷服务，无BP|1077632|1030068|1070536|
|暖服务，无BP|1075808|1028243|1068711|
|核心拍，有BP|1016089|970314|1008970|
|冷服务，有BP|1116345|1070571|1109227|
|暖服务，有BP|1114521|1068746|1107402|

校准只用同帧tile0–31。将每个K的两个空间位置在每个T表示成2bit门状态，以pair-state Hamming变化数加起始非零状态数作为端点更新费用，使用一次`10×2^10`子集DP求完整时间路径，得到唯一固定顺序 **[8,2,6,3,9,4,1,5,7,0]**。没有用两套held结果选择另一个排列或扫描配置。[calibrate_order.py](calibrate_order.py)来自根代理的原型，实际重算及cal/held机会账见[calibration.json](calibration.json)；这些机会数不是性能结论，性能来自后续RTL。离线DP和校准数据读取不在RTL服务中；当前测固定表部署，不能声称已计在线逐帧自校准成本。

独立审阅指出cal0–31与held160–191的source halo共享y1..2，合计12480个源字；原128集合仅输出tile不同。随后追加的4000–4063与cal输入交集为0，沿用相同排列，没有重校准；[输入准备](../audit/prepare_disjoint_stream.py)从原capture直接提取，并独立重算245760个raw与原fullgold完全相等，见[输入检查](../audit/disjoint_input_check.json)。[独立审阅](../audit/temporal_order_review.md)保留补测前原判断；本报告的mode5和输入分离结果是针对其缺项完成的后续实测，不改写审阅意见或上调其新颖性评分。

这里放开的是“执行必须沿原T”的接口假设。原生完整T10门字在本线性层开始前已经可读，Q1/Q2与θW不随T变化，而且raw之间没有跨T状态或中间RNE。mode4/5在RTL的L_GATHER用已配置40bit表重排四个源字的位，随后沿执行顺序做delta与prefix；全部Q2结果仍完整物化后，DRAIN_READ在原单psum口按逆排列读取，输出row仍是原T。TB既不重排source，也不重排gold。没有改变生成门字的神经元时间过程、跨神经元边界或消费者顺序。

|held64无BP的实际内部量|direct|原序filtered mixed|固定顺序filtered mixed|
|---|---:|---:|---:|
|Q1更新|77032|77032|57098|
|direct／endpoint更新|77032／0|77032／0|34115／22983|
|选择endpoint的K列|0|0|9167|
|付费SELECT|0|5312|14175|
|z clear写|1280|1280|2560|
|prefix ADD／merge ADD|0／0|0／0|1280／1280|
|z向量读|79592|79592|62218|
|z写|78312|78312|62218|
|Q1／Q2权重向量|25237／6144|25237／6144|25237／6144|
|Q2 MAC|198720|198720|198720|

无BP的净核心节省严格闭合：`3×(77032−57098)−14175−1280−2×1280−2×1280=39227`。其中14175是实际选择，1280是额外域clear，两项2560分别是prefix与merge的读/add写状态。Q2与权重读取保持不变；不是把CPU预测工作量直接减成周期。

同排列mode5实测70535次端点更新，恰好复现校准脚本预报的该集合全端点服务数；它只清1280行、执行1280次prefix读/add写，不做SELECT、第二域clear或merge。mode4对mode5的差值也闭合：`3×(70535−57098)−14175−1280−2×1280=22296`，共同prefix与排列配置抵消。这是同排列全端点的真实RTL消融。

|4000集合无BP的实际内部量|direct|固定序mixed|固定序全endpoint|
|---|---:|---:|---:|
|Q1更新|101030|77376|97811|
|direct／endpoint更新|101030／0|49284／28092|0／97811|
|选择endpoint的K列|0|9919|固定端点|
|SELECT|0|16997|0|
|z clear写|1280|2560|1280|
|prefix ADD／merge ADD|0／0|1280／1280|1280／0|
|z向量读|103590|82496|101651|
|z写|102310|82496|100371|
|Q2 MAC|210384|210384|210384|

4000集合mode4对direct的净核心节省为`3×(101030−77376)−16997−1280−2×1280−2×1280=47565`，减首次排列配置1拍得服务47564。对mode5为`3×(97811−77376)−16997−1280−2×1280=40468`。局部表示选择的节省超过双域和选择成本；没有把全部端点也能受益的部分都归给混合。

根代理另外保持旧`modular_core.sv`逐字不改，只更换原生hex raw TB，完成1776命令、6819840个raw值，包含同20fixture和两套64流。这个更宽端口强控制在本工作负载上更快：

|冷服务，无BP|held128–191|输入不重叠4000–4063|
|---|---:|---:|
|本叶208bit direct|983682|1077632|
|本叶208bit mode4 mixed|944456|1030068|
|旧416bit native dual|983042|1076992|
|旧416bit native triple|981218|1059511|
|旧416bit fourP8＋repair|933565|996457|

mode4比416bit four分别慢10891拍（1.1666%）和33611拍（3.3731%），因此不能称胜过全部原生强控制。208bit与416bit是不同z端口点，且本叶多520B状态，不能作等资源PPA排序，也不能将这个负比较扩大成端点家族无效。旧416bit dual每tile仅10拍clear，本叶208bit dual为20拍，64流640拍差恰由此产生。强控制原始结果见[SUMMARY_64.json](../strong_raw_control/SUMMARY_64.json)、[SUMMARY_disjoint.json](../strong_raw_control/SUMMARY_disjoint.json)及[独立核验](../strong_raw_control/verification.json)。

旧实验已先检查：[旧interval报告](../../pro_fusion_trials_20260913/interval/REPORT.md)在R24+onepass内部U16、P2×T10×K864的真实tile实现过全端点和整tile二扫选择，2404→3502拍，明确未做独立状态下的逐K混合。本轮不重新发明该端点公式，也不把其旧U16切口计作当前R8实验；实际新接口为独立direct/endpoint域的有费混合。实施前及条件补充见[PLAN.md](PLAN.md)。

[R8原报告](../../r8_consumer_fusion_20260914/REPORT.md)确认g是AT-LIF发放门、静态θ已吸收、`z=Q1g; raw=Q2z`之间无RNE。本轮保持对外原T10顺序、R8、K864、P4、N96以及原生96×4×4源gather，最终完整480个N8 raw输出。没有PSN时间截断、prev-anchor连续变换、跨非线性或提前舍入，也没有把独立consumer的J/I24纳入本叶输出。

|mode|实际执行|
|---|---|
|0 direct|原native2P13；相同K列一次取Q1向量，跨T/P复用|
|1 endpoint|`d_t=g_t−g_(t−1)`且`g_−1=0`，两位置端点合并更新，RTL按T前缀恢复后原Q2|
|2 mixed|每个非零Q1列付一拍比较direct和endpoint的成对更新次数，严格较少才选endpoint；两个域提前清零，最后prefix＋merge|
|3 filtered mixed|空源跳过；无相邻11时由必要条件直接判direct；其余才付SELECT。第二域仅在首次真选endpoint时付20拍clear，保持当前k/source/pending|
|4 calibrated mixed|mode3再使用唯一cal0–31固定执行排列；在原psum读口逆地址退休，恢复原始T raw|
|5 calibrated endpoint|与mode4相同排列、配置与逆地址；所有K固定用endpoint，只用单域及prefix，不付SELECT、第二域clear或merge|

mode3的必要条件是精确判定：当每个1前面都是0时，该1必为上升端，因此每个位置逐bit的endpoint支持包含direct支持，两个位置做OR后仍成立，`E_pair≥D_pair`。它只排除不可能便宜的端点，不预测后续背压或收益。固定模式控制与付费选择在同一RTL模块中，mode2保留作为选择税的对照。

|真实八块，冷命令合计|direct|全endpoint|逐K mixed|filtered mixed|
|---|---:|---:|---:|---:|
|核心拍，无BP|87599|101218|94655|87913|
|核心拍，有BP|91295|105006|98936|91713|
|相同冷配置拍|26888|26888|26888|26888|
|冷配置＋核心，无BP|114487|128106|121543|114801|
|Q1成对更新|5245|9678|5245|5245|
|SELECT拍|0|0|6896|314|
|选endpoint的活动K列|不选择|固定端点|0|0|
|z clear写|160|160|320|160|
|prefix ADD|0|160|0|0|
|merge ADD|0|0|0|0|
|Q1活动列／Q2权重向量|1958／756|1958／756|1958／756|1958／756|
|Q2 MAC|17556|17556|17556|17556|

完整全端点的增加量闭合为`3×(9678−5245)+2×160=13619`拍；权重读取不下降，因为从初值0开始的任何非零二值序列必有端点，跨T/P union-K不变。原mode2的7056拍额外成本就是6896次SELECT＋160次第二域clear，不能据此停在偏弱选择器上。

原序mode3跳过4938个空列，1644个无相邻11的活动列直接判direct，只有314列付比较；这些列也没有一个endpoint成对工作量严格少于direct。因此它没有触发第二域初始化、prefix或merge，只多314个核心拍。即使把这314次比较完全藏进原CHECK，原顺序逐K粒度的Q1更新仍是5245。随后mode4固定排列在同八块得到85669核心拍（direct87599）、有BP89552（direct91295），每次冷mode4另付1拍排列配置；这项真实小测正结果才触发held64执行。

|固定合成条件，核心拍无BP|direct|全endpoint|mixed|filtered mixed|
|---|---:|---:|---:|---:|
|全零|6299|6339|7181|6299|
|6拍长run|41071|20423|21345|21345|
|相邻交替脉冲|35563|61463|36445|35563|
|前半K长run、后半K交替|39403|41975|29997|29567|
|先交替、K中途转长run|39391|42071|30033|29601|
|Q1全−4并有下降端|35869|20357|21281|21281|

混合正例中mode3真实选择432个endpoint列及430个direct列，更新9484→6028次，支付432次SELECT、额外20次clear、20次prefix读/add写与20次merge读/add写，总周期39403→29567。全endpoint在同一个合成输入反而更慢（41975），说明有费两域适配实际修复了全端点失败条件。反向混合例在K中途第一次转endpoint，覆盖lazy clear保留当前工作；交替与全零经必要条件过滤后与direct同拍。以上都是有界功能/条件控制，不冒充真实性能样本。

算术、端口及状态详见[RESOURCE_CONTRACT.md](RESOURCE_CONTRACT.md)。六mode共用原8条32bit ALU与8个19×13乘法器，一个全局208bit z口；两域使z从原520B增至1040B，direct和全endpoint同样预留第二域，但不强制其初始化永不用的状态，不能声称与旧单域等面积。前缀保持复用原Q2 acc；混合prefix先读endpoint域、用同一2P13 ALU求和写回，再独立读direct域并用同一ALU合并，不能同拍读两域或免费相加。

下降端沿用13bit XOR＋carry-in，Q1=−4时正确形成+4；没有三位取负回绕或新增数据负权重加法器。任意端点部分K和的绝对界≤3456，前缀恢复与direct分区相加仍为完整二值dot product，signed13安全；Q2 signed32仍充足。本次独立模型与RTL边界例实际达到3456绝对值。

[verify.py](verify.py)从原生源重新形成事件、带符号端点、两域和、前缀、逆排列与最终raw。26fixture及两套64 tile合计591360个独立raw重算通过；再按每命令核Q1/Q2、选择、lazy clear、所有z读写、prefix/merge及每状态拍数。native direct与旧pair_sparse的80个去重fixture强控制和256个64流控制命令全部原计数逐项复现。C++ TB只装载原序source/Q1/Q2、静态活性和40bit排列，不提供中间z/delta；所有3840输出、地址、BP holding、无reset重启与跨mode均核对。专门`inverse_time_tag`令每个原T输出依次为1..10，错误逆序无法靠恒定输出蒙混通过；mode5也经过同功能fixture、逆映射、4/5及0/5跨mode检查。没有复制大原生capture，20旧fixture及两套64source/gold小窗直接引用，只新增6个小合成窗。

这里支持的结论是：旧interval明确未试的两域逐K混合已经实现；原序失败经必要条件过滤降至接近direct，再利用完整非因果门字接口允许的固定执行重排，在两套同帧64输入上比208bit direct获得3.99%和4.41%完整服务收益，并优于同排列单域全端点。后一个集合与校准输入不重叠。416bit原生四P仍更快；与强打包/RR及真实J/I24消费者融合、更细逐位置对选择、跨帧/序列泛化、不同真实时间关系、跨窗口/运动对齐输入、生产者直接提供端点及其格式成本仍未测试。

固定线性层差分/积分来自已有Sigma-Delta/Delta工作，全T权重复用也已有LoAS等近邻；旧报告对Comperity只取得摘要。时间重排、局部混合、必要条件过滤或carry切断本身没有建立论文新颖性，也没有声称复现完整外部作者系统。尚未跑全帧、EDA、面积/Fmax/能耗、训练或生产修改，没有commit。

结果：[SUMMARY_stream.json](SUMMARY_stream.json)、[results_stream.json](results_stream.json)、[SUMMARY_disjoint.json](SUMMARY_disjoint.json)、[results_disjoint.json](results_disjoint.json)、[SUMMARY.json](SUMMARY.json)、[results.json](results.json)、[verification.json](verification.json)。复现在本目录：

```
/opt/anaconda3/bin/python3.12 calibrate_order.py
/opt/anaconda3/bin/python3.12 prepare.py
/opt/anaconda3/bin/python3.12 run.py
/opt/anaconda3/bin/python3.12 verify.py
/opt/anaconda3/bin/python3.12 run_stream.py
/opt/anaconda3/bin/python3.12 run_stream.py --disjoint
/opt/anaconda3/bin/python3.12 verify.py
```

构建采用Verilator4.028 `-Wall --cc --exe`加独立`make`。小测正结果后`gate_64=true`，两套64和必要消融已完成；本轮收口，不追加排列搜索或其他机制。
