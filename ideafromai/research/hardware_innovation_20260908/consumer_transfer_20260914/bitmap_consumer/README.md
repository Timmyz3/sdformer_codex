bitmap四臂已接到**实际FP32 identity→J20→I24单context消费者**，父目录[bitmap_pipeline/decomp_core.sv](../bitmap_pipeline/decomp_core.sv)原样复用。两套64tile冷启动无BP，原生dualP13 mode14为1138971／1232921拍，适配mode10为 **978610／1029806拍，减少160361（14.0795%）／203115（16.4743%）**。有source、weight、identity和输出BP时，分别减少114323拍（9.3780%）／144613拍（10.9516%）。

本叶仅验证消费者后还剩多少收益，没有修改bitmap算法、扩展RR或借用consumer主链。一个context、208bit z口、原bitmap/plane副本及pop树资源保留，不能把该表与416bit双context借链表当作同面积排序。[实施前计划](PLAN.md)

|64tile完整经过时间|14 native dualP13|15 原完整位平面|13 生成/流水/live块|10 再加单比特原生路径|
|---|---:|---:|---:|---:|
|128–191冷，无BP|1138971|1249559|1014578|978610|
|128–191暖，无BP|1137123|1247711|1012730|976762|
|128–191冷，有BP|1219057|1364752|1163569|1104734|
|128–191暖，有BP|1216912|1362732|1161517|1102739|
|4000–4063冷，无BP|1232921|1314971|1071032|1029806|
|4000–4063暖，无BP|1231073|1313123|1069184|1027958|
|4000–4063冷，有BP|1320477|1442467|1243789|1175864|
|4000–4063暖，有BP|1318472|1440454|1241802|1173869|

冷配置1848拍，包括864 Q1、96 Q2、864 k_live和24 consumer系数；每tile真实加载1536个原生source字和1个origin。两遍之间无reset且参数持久。bitmap及位平面生成继续由原RTL实际完成；TB不提供预排source、bitmap、中间z或预转换J20。所有臂输出原P/T/N顺序，identity来自FP32原输入，依次实际执行FP32→signed32 Q20、原64bit乘加、RNE26与signed24饱和。

|冷无BP，raw服务→完整consumer|128–191|4000–4063|
|---|---:|---:|
|native14|983682→1138971|1077632→1232921|
|原位平面15|1094270→1249559|1159682→1314971|
|适配13|859289→1014578|915743→1071032|
|适配10|823321→978610|874517→1029806|

无BP下每个臂新增相同155289拍：consumer导致raw holding等待154816拍、最后raw之后实际consumer尾部384拍、consumer系数24拍、每tile新增wrapper拍64拍及末尾1拍。它们全部在RTL发生，因此mode10的绝对节省160361／203115拍与raw阶段相同，收益比例从raw的16.30%／18.85%降到完整consumer的14.08%／16.47%。有BP不按这一固定差推算，表中数值均独立实测。

mode15原负迁移仍保留，比native完整consumer慢110588／82050拍。mode13只承接父叶已实现的格式生成重叠、plane读取/执行流水和live块枚举；mode10再利用已实际读到的单比特bitmap，读取该K原生signed3权重并加到同bm_acc。完整consumer下，13→10仍净省`2×17984=35968`／`2×20613=41226`拍，单比特路径读和加均付费，没有新增数据ALU。

|mode10冷无BP实际义务|128–191|4000–4063|
|---|---:|---:|
|bitmap读块|41445|50403|
|位平面pop/共享ALU执行|70383|89370|
|原生signed3单比特读取/加法|17984|20613|
|同拍plane读取与执行重叠|46922|59580|
|core周期，含raw等待|877881|929077|
|consumer周期|878265|929461|
|consumer join等待|661625|712821|
|raw output holding等待|154816|154816|

资源没有在接消费者时改变：producer为8条32bit ALU和8个19×13乘法器，source1920B、z520B与208bit口、完整psum15360B；bitmap额外4320B、位平面系数副本2592B、八棵pop16，以及270B live位、流水标签/holding与地址选择。八棵pop树的算术和位平面副本是真实增量，不能只列八条共享ALU就称等资源。consumer是一份原8×64主链和8×32×32乘法器、768B系数及原输入/输出holding；本叶没有consumer/producer宽链共享。

source/bitmap构造、plane读取和上一拍pop执行只按父叶原授权相邻服务重叠，没有增加plane读口。native Q1与plane权重读按原单读mux互斥。wrapper在当前tile完整I24退休后才重载source或启动下一tile，原psum中所有480个输出完整物化；没有持有中的输出被下一tile覆盖。

**504条RTL命令，raw、J20、I24各9922560个值全部通过。** 23个小fixture覆盖真实、全零/全一、−4、正负极值、padding poison、FP转换/乘积ties与饱和、count255/256及high127/128来源窗、零Q2、独特原T签名；包含无reset跨mode、159/19197跨row，以及两套64冷暖/BP。完整握手地址、tile/last和holding均检查。

[verify.py](verify.py)从154个去重fixture/native tile重算完整raw/J20/I24，各591360值，与原gold一致；同时重建signed3三plane加权pop和单比特路径，核对bitmap/权重/ALU工作、core/consumer及wrapper总周期。无BP的并发plane计数也独立核对。最终[verification.json](verification.json)确认接入时leaf内容保持父目录最终源原样；没有把独立模型工作量当成RTL性能。

两个64集合仍来自同一帧，4000集合与先前时间校准源不重叠；本叶没有使用或重校准时间排列。这里只支持真实消费者后的组件周期收益，未测跨帧、全19200tile、EDA、面积/Fmax/能耗或整网质量。本叶不是新的算法或完整BISMO系统复现，也不能由更大百分比推导论文新颖性。

结果为逐命令JSONL：[small](results_small.jsonl)、[short](results_short.jsonl)、[128集合](results_held.jsonl)、[4000集合](results_disjoint.jsonl)。源码与脚本只写本目录，父bitmap和其他目录只读；无EDA、训练、生产变更、main.tex或Git提交。

复现使用`/opt/anaconda3/bin/python3.12`：依次执行`implement.py`、`run.py --stage small`、`run.py --stage short`、`run.py --stage held --skip-build`、`run.py --stage disjoint --skip-build`、`verify.py`。Verilator4.028 `-Wall --cc --exe`后独立make。本轮四臂单context消费者验证到此收口。
