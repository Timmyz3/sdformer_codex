80B位图及其系数复用已经接到**实际FP32 identity→J20→I24单context消费者**。父叶[bitmap_block_resident/decomp_core.sv](../bitmap_block_resident/decomp_core.sv)最终显式读使能版本原样复用。两套64tile冷启动无BP，mode8完整经过时间为 **1046086／1109196拍**，比原生dualP13 mode14减少92885（8.1552%）／123725（10.0351%）拍；有source、weight、identity和输出BP时减少84473（6.9294%）／116005（8.7851%）拍。

本叶只做父叶mode14/9/8的完整消费者接入，沿用[先前单context harness](../bitmap_consumer/README.md)，透传实际cache读取／写入计数。[实施前接口计划](PLAN.md)定义了共同资源和所有权；这里没有追加机制、双context RR或consumer宽链借用。

|64tile完整经过时间|14 native dualP13|9 K16驻留位图|8 再复用Q2 qblock缓存系数|
|---|---:|---:|---:|
|128–191冷，无BP|1138971|1059836|1046086|
|128–191暖，无BP|1137123|1057988|1044238|
|128–191冷，有BP|1219057|1164339|1134584|
|128–191暖，有BP|1216912|1162342|1132469|
|4000–4063冷，无BP|1232921|1128948|1109196|
|4000–4063暖，无BP|1231073|1127100|1107348|
|4000–4063冷，有BP|1320477|1244442|1204472|
|4000–4063暖，有BP|1318472|1242517|1202477|

冷配置1848拍，包括864 Q1、96 Q2、864 k_live和24 consumer系数；每tile实际加载1536个source字和1个origin。暖命令在同实例无reset下复用同模型参数。cache按运行时K16重新预取，不由配置或TB免费注入。TB提供原source、FP32 identity和模型系数，完整3840个raw及J20、I24逐值与原gold比较；没有提供bitmap、z或预转换的J20代替实际计算。原T顺序、FP转换、64bit乘加、RNE26与signed24饱和边界均保持。

|冷无BP，raw服务→完整consumer|128–191|4000–4063|
|---|---:|---:|
|native14|983682→1138971|1077632→1232921|
|80B位图9|904547→1059836|973659→1128948|
|80B位图＋系数复用8|890797→1046086|953907→1109196|

父叶最终实测raw服务与本叶完整consumer在所有无BP冷臂相差155289拍：实际raw holding等待154816拍、最后raw后的consumer尾部384拍、consumer系数24拍、每tile新增wrapper拍64拍及末尾1拍。9→8的绝对节省13750／19752拍因此保留到完整消费者；百分比随完整分母变化。有BP行独立实测，没有套用固定尾部差。

mode9只驻留当前K16的40个16bit源字（80B）与40bit live（5B）。每个非空P/T条带通过原208bit z口读取全字，保留未选中的13bit半字，在同8条ALU执行onehot或三plane工作，最终写回完整208bit字。下一K16覆盖该位图。相比[完整位图mode10](../bitmap_consumer/README.md)，bitmap及live主数组合计减少4505B，却增加真实z服务：两64中，向量读由2560变为44005／52963，完整字写由3840变为42725／51683。mode8的完整consumer仍比完整位图mode10慢67476／79390拍，是存储与周期的不同取舍。

mode8把每个非空K16的三plane经原W口预取到Q1阶段闲置的Q2 qblock前三行48B。原qblock共128B，没有新增cache数据数组；Q2的VLOAD覆盖全部八行后才使用。三拍预取槽全部付费，空plane槽也占拍；有效plane受weight_allow背压。多活动位路径从唯一128bit cache读供给pop和共享ALU，单活动位仍走原Q1 signed3读。最终leaf中qblock读使能只在BASE_MAC或启用的BM_CACHED_POP；跳零plane不发生未计的cache访问。Q2 BASE_MAC的z读取也只使能selected_rank bank，向量阶段才使能全部八bank。

|冷无BP实际服务|128：9→8|4000：9→8|
|---|---:|---:|
|外部128bit plane读取|70383→9711|89370→10038|
|原生onehot读取／加法|17984→17984|20613→20613|
|有效128bit cache读取|0→70383|0→89370|
|cache填充槽／写入|0→9711|0→10038|
|位图读块，两臂相同|41445|50403|
|pop／共享ALU执行，两臂相同|70383|89370|
|z向量读，两臂相同|44005|52963|
|z完整字写，两臂相同|42725|51683|
|Q2 selected-bank z读／MAC，两臂相同|198720|210384|

9→8的无BP周期闭合为`3×非空K16数−多活动位P/T条带数`：128集合是`9711−(41445−17984)=−13750`，4000集合是`10038−(50403−20613)=−19752`。外部plane少读60672／79332次，对应工作转入cache，不能将其解读为同百分比能耗下降。预取也会失败：跨row短流first19197的三tile冷无BP，mode9为35862拍、mode8为35983拍，**慢121拍**；冷／暖BP也分别慢128／125拍。该原始记录保留，不因长流转正删除。

共同producer仍为8条32bit ALU、8个19×13乘法器、source1920B、z520B及208bit口、完整psum15360B、Q1 2592B和Q2 1536B。bitmap路径保留额外2592B plane系数副本及八棵pop16；这些是真实增量。consumer是一份原8×64主链、8×32×32乘法器、768B系数及原holding。wrapper等当前tile完整I24退休后才重载source和启动下一tile，所有480个输出字先完整物化，不覆盖消费者仍持有的输出。本叶没有与416bit双context借链对照等面积的主张。

**最终396条RTL命令，raw、J20、I24各7511040值全部通过。** 23个小fixture覆盖真实、全零／全一、−4和正负极值、padding poison、FP ties和饱和、零Q2、独特T签名等，包含无reset跨mode、159／19197跨row、两套64冷暖和BP。所有holding、输出地址、tile及last均检查。[verify.py](verify.py)从154个去重fixture/native tile独立重算raw/J20/I24，各591360值与原gold一致，独立核对cache／plane／onehot／z义务和总周期，共19091项检查。native mode14的108条既有记录逐字段复现。[verification.json](verification.json)确认leaf与父目录最终源内容相同；没有用CPU工作量代替RTL性能。

两套64仍为同一帧；4000集合与先前时间校准源不重叠，本叶不使用时间排列或重新校准。cache读mux→pop→共享ALU增加的组合路径尚无物理时序证据；容量与周期数字不能作为同频率PPA结论。未扩双context、全帧、跨序列、整网或EDA。本叶支持有限驻留／普通系数复用接入真实消费者后的组件结果，不作新算法主张。

逐命令JSONL：[small](results_small.jsonl)、[short](results_short.jsonl)、[128集合](results_held.jsonl)、[4000集合](results_disjoint.jsonl)。复现使用`/opt/anaconda3/bin/python3.12`依次执行`implement.py`、`run.py --stage small`、`run.py --stage short`、`run.py --stage held --skip-build`、`run.py --stage disjoint --skip-build`、`verify.py`。Verilator4.028 `-Wall --cc --exe`及独立make；全部改动只在本目录，无EDA、训练、生产变更、main.tex或Git提交。本次有界接入到此收口。
