小位图已接入**双context、416bit z、唯一一组producer ALU／乘法器和实际FP32 identity→J20→I24消费者**。保留原生四P mode2、真实consumer64借链RR mode4、count21强控制。忠实mode8在4000集合冷启动比count21慢593拍；一次有界行复用适配mode7后，两套64tile冷周期为 **750649／792115**，比count21减少21962（2.8426%）／18673（2.3031%）拍。暖启动仍分别减少21064（2.7361%）／17775（2.1998%）拍。

这里迁入的是既有位平面、K16有限驻留、普通系数复用和RR；新增只验证现有416bit行holding如何减少重复z服务，不称新算法。没有将先前单context比例搬到本表。[实施前A/B与端口合同](PLAN_RESOURCE_CONTRACT.md)、[实测后唯一适配](ADAPTATION.md)。

|64tile完整经过时间|2 四P modular|4 borrow RR|21 count配对|8 忠实小位图|7 同T行复用|
|---|---:|---:|---:|---:|---:|
|128–191冷，无BP|787603|786421|772611|766786|750649|
|128–191暖，无BP|785755|784573|769865|764938|748801|
|128–191冷，有BP|858814|857282|844899|843242|823354|
|128–191暖，有BP|856904|855357|842204|841317|821429|
|4000–4063冷，无BP|831996|830886|810788|811381|792115|
|4000–4063暖，无BP|830148|829038|808042|809533|790267|
|4000–4063冷，有BP|909177|908472|888404|894014|870512|
|4000–4063暖，有BP|907252|906052|885709|892247|869184|

mode7相对borrowRR冷减少35772（4.5487%）／38771（4.6662%）拍；相对count21冷BP减少21545（2.5500%）／17892（2.0139%）拍。忠实mode8的失败仍可见：4000集合比count21暖慢1491拍、冷BP慢5610拍，不能因它胜borrowRR便省略更强count控制。所有行均为同结构完整consumer实测。

共同配置1848拍，包括864 Q1、96 Q2、864 k_live及24 consumer系数。count21首次另付class／rep／ngroups 897拍和固定排列1拍，暖命令保留同模型参数。plane副本从原Q1加载拍实际构造，每拍增加24个bit写和plane-live更新；这些写扇出和存储真实存在，未藏到TB预处理。每个非空K16的三个cache预取槽仍运行时付费，有效plane争用唯一W grant，空plane槽也占拍。每tile加载1536个source字和1个origin，BP覆盖外部加载、计算source／weight、FP32 identity和最终输出。

mode8把原80B位图、5B bitmap-live、48B qblock覆盖布局移入两个context。两context在top共享唯一八棵pop16，输入在同一producer ALU owner确定后选择；符号plane减法使用原32bit carry chain，不各自复制数据ALU或乘法器。plane系数2592B及162bit plane-live在top各一份，系数读取扩入原W-kind mux。每条带通过共同416bit z口读全52bit bank字，更新选定13bit字段后写完整字。原Q2、完整psum、count21逆映射和consumer退休路径沿用。

mode8仍按P后T枚举，已经读出的同T四P行反复进出z。mode7只将K16内部改为T后P：在原z_hold保留整行，BM_SCAN读当前P字段，原onehot／plane工作逐P完成；BM_STORE在同T还有P时付一拍holding提交，最后一P才申请z grant写回完整行。下一T重新读z。没有跳过holding提交拍，没有给holding添数据数组或bank端口，也没有改Q2／输出顺序。

|冷无BP的实际工作，8→7|128–191|4000–4063|
|---|---:|---:|
|活动P/T条带，两臂相同|41445|50403|
|Q1条带z整行读／写，各自|41445→14482|50403→17085|
|完整producer z grant，含clear／scan／Q2|284810→230884|314390→247754|
|BM_SCAN额外holding字段读|0→41445|0→50403|
|BM_STORE跨P holding提交|0→26963|0→33318|
|外部plane读，两臂相同|9711|10038|
|有效cache读／pop执行，两臂相同|70383|89370|
|onehot原Q1读／加，两臂相同|17984|20613|
|Q2 selected-bank z读／MAC，两臂相同|198720|210384|
|全部producer ALU grant，含proof，两臂相同|287951|321231|

holding为每context已有的416bit寄存器。上表`bitmap_hold_reads`只数BM_SCAN中8×13bit字段读；每次BM_STORE还实际读取整416bit holding，组成全字写数据。在26963／33318个非末P条带，该拍写回holding；在14482／17085个末P条带，该拍经z grant写回z。BM_POS还负责相应整行holding装载。字段读、整行提交和行装载处于不同状态；寄存器读改写及其mux是实际结构，不把它当成免费多端口SRAM。Verilator断言覆盖holding行所有权、未写回不得换行和进入Q2，拒绝grant时保持原状态和存储。

|冷无BP拥塞与完整窗口，8→7|128–191|4000–4063|
|---|---:|---:|
|条带z仲裁等待|9787→1583|12514→2182|
|总context仲裁等待|175136→169620|194344→189575|
|包含ALU请求的等待|144137→146714|158210→162041|
|包含W请求的等待|975→1288|894→1269|
|raw holding等待|259205→259410|266077→265632|
|两个context周期合计|1224242→1191968|1313432→1274900|
|完整consumer窗口|666537→650400|711132→691866|

周期闭合为：128集合少26963个BM_POS，仲裁少5516拍、raw holding多205拍，context合计少32274拍，实测窗口少16137拍；4000集合对应`−33318−4769−445=−38532`，窗口少19266拍。资源等待列可能重叠，例如Q2 MAC原子请求z和ALU，不能把这些列相加当总等待。行z争用下降并没有消除共享ALU瓶颈，ALU等待反增2577／3831拍；本轮没有追加第二种调度。

小工作负载也有残余负例：无reset换源测试中的128–129两tile冷BP，mode8为20024拍，mode7为20047拍，慢23拍；从4000–4001切回该集合的暖BP为18029→18037，慢8拍。原记录保留，mode7不是逐命令支配mode8。无BP对应两tile仍减少87拍。其他小fixture和跨row的数值见逐命令结果，没有按有利样本删选。

共同资源为top的一份8×32 ALU、8×19×13乘法器和一份真实consumer的8×64主链／8×32×32乘法器。每context source1920B、z520B、完整psum15360B及qblock128B不变。原Q1、Q2、class2592B、rep96B及40bit排列仍各一份；bitmap另有共享2592B plane系数、162bit静态plane-live寄存器、八棵pop16，每context80B bitmap、40bit live、40bit pending、16bit mask hold和256bit条带accumulator hold。qblock前三行48B仅在Q1阶段使用，Q2 VLOAD覆盖全部八行后再消费。mode7只增T/P选择和same-row控制，不增数据数组。详细数量及读写口径见[resource_contract.json](resource_contract.json)。

所有臂运行同一union硬件和端口权限；未主张分别裁剪后的native、count、bitmap等面积。plane-live是明确列出的静态寄存器广播／bit选择逻辑，不是另一个系数读取口。源bitmap仍为实际40路bit写寄存器实现；cache mux→pop→共享ALU、W地址选择和新增行选择路径均无Fmax／能耗实测，容量与周期不能直接推成同频率PPA。

**最终864条RTL命令，raw、J20、I24各13900800值全部通过。** 23个小fixture覆盖真实、全零／全一、−4和极值、padding poison、FP转换／乘积舍入和饱和、count边界、零Q2及原T签名；包含双context、无reset跨mode、159／19197跨row、两套64冷暖／BP，以及128↔4000换源和同时换mode。source和FP32 identity来自原数据，raw全部物化后进入实际consumer，TB没有赠送bitmap、z或J20。每批全部I24完成后才回收context并重载source，count也不能覆盖消费者仍拥有的psum。

[verify.py](verify.py)从154个去重tile／fixture独立重算raw、J20、I24，各591360值与原gold一致；同时重建signed3 plane计数、同T行数、缓存／holding／z义务和总周期，共1551719项检查。384条既有strong-control记录逐字段复现；原mode8的16条短／长关键周期与工作记录也复现。[verification.json](verification.json)、[comparison.json](comparison.json)。独立模型用于核对实际RTL计数，不用CPU工作量代替服务周期。

两个64集合仍为同一帧；4000集合与此前count时间校准的输入不重叠，count21排列固定继承、不重校准，bitmap不使用时间排列。已验证当前R8函数与原FP／RNE边界，未测跨帧、整帧19200tile、整网质量、EDA或论文新颖性。本次只闭合一种最终位平面布局及一次行复用适配。

复现：用`/opt/anaconda3/bin/python3.12`依次执行`implement.py`、`prepare_verify.py`、`run.py --stage small`、`run.py --stage short`、`run.py --stage held --skip-build`、`run.py --stage disjoint --skip-build`、`run_swap.py`、`verify.py`、`summarize.py`。Verilator4.028 `-Wall --cc --exe`及独立make。结果逐命令JSONL：[small](results_small.jsonl)、[short](results_short.jsonl)、[128集合](results_held.jsonl)、[4000集合](results_disjoint.jsonl)、[换源](results_swap.jsonl)。只写本目录，父树、生产和主稿只读，无EDA、训练或Git提交。
