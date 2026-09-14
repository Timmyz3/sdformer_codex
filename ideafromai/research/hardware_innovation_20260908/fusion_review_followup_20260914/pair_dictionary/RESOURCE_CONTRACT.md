# 共用资源、计数银行与数值合同

所有模式运行于同一SV实例。控制为完整native-window dual-P+final-support+Q2-zero-vector+cachedOS，不是稠密/逐标量弱控制。mode14不加载或读取字典表；mode15的额外配置、读写和清零自行付费。

|资源|布局|服务权限|
|---|---|---|
|source|1536×10bit=1920B；local16×10bit=20B|原source一10bit读；local每K四个10bit gather；图外零不发外部读，但local装入仍付拍|
|Q1 / Q2|8×864×signed3=2592B / 8×96×signed16=1536B|Q1一24bit common-K向量；Q2一128bit向量；source_allow/weight_allow只是本地读许可|
|Q2 cache|64×signed16=128B|一次选定rank的8系数，由flop/mux提供|
|z|8×20×26bit=520B；z_hold26B|一208bit读写；Q2可选bank/half标量读；分时；退休只更新一13bit half|
|psum / accumulator|8×480×32bit=15360B；8×32acc、256bit输出hold|STORE与DRAIN_READ各真实一拍，480写480读；输出ready阻塞保持|
|计数bank|8×320×20bit=6400B|每组2banks；行=class×10+block；C_READ各组可不同class地址；G_READ共同slot/block。每bank仅一个显式rd_addr/rd_data/enable；C_ADD/DCLEAR分时写|
|class|4×864×6bit=2592B|四bank同K读，共24bit，一拍；0zero/1..32group/63direct；其余码不在合同|
|代表|32×8×signed3=96B|同slot24bit读，四组的键可不同；padding代表为0|
|count_hold|8×20bit=20B|一160bit读后保持；pending4bit对同slot/block四个P/T位置作跨组OR|
|有限控制|class_hold24bit、slot_live32bit、ngroups6bit、count_active、cblock/slot/address、pending4bit|所有address arithmetic、比较和零归约是额外组合控制；不误称8条数据ALU包含这些控制|
|共同支持|k_live864bit、v_live96bit、position40×8bit、rank/block/remaining各8bit、source_masks40bit、Q1_hold24bit、pending40bit|由完整配置或实际运算产生；没有无限/免费元数据|

乘法表达式仍8个signed19×13→32bit。BASE_MAC的第二操作数为signed13 z；退休时是unsigned10 count零扩为signed13，每个rank pair可取不同count。八条32bit加法链：ZADD在bit13切carry；C_ADD在bit10切carry；其余整32bit。每C_ADD四组×两ALU，每ALU更新两unsigned10位置，总16个计数。读写计数按实际有效group的两banks统计；清零和slot退休读访问全部8banks。额外count银行与class地址能力不免费；没有单独PPA或等面积声称。

Q1限定[-3,3]，单count≤K864<1024，部分累加与最后z都在[-2592,2592]，signed13足够。分组退休与direct只改变加法顺序；任意前缀的每rank绝对和≤864×3，因而不依赖模数溢出。raw p绝对值≤8×2592×32768=679477248，signed32足够。G_MAC只改目标half，其余half保持。全部计数先清零、全部z先清零，每次start重置live/slot状态；运行中禁止cfg。

冷配置：控制1536 source+1 origin+864 Q1+96 Q2+864 liveK=3361拍；候选另32 representative+864 packed-class+1 maxclass=897拍，共4258。CPU静态建表不算运行时，但未给出离线编译时延或存储传输能耗。warm是同source同参数重复命令，仅用于无reset覆盖。

配置完整/有序；k_live必须等于OR_r(Q1!=0)；所有class必须与对应代表pair一致；每组≤32类，ngroups为组间最大。静态编译和独立verify执行这些验证。错误metadata与异步partial-config不在本合同。

BASE_MAC仍允许异步z选择+Q2 mux+乘法+加法在一拍完成；count地址选择/代表选择也是组合逻辑。本实验给的是所列状态及端口权限下的RTL周期，未测SRAM映射、Fmax、全帧吞吐或功耗。
