**给 Wz 两臂同样的普通 partial-frontier、X 驻留、active PF 和批内静态配置复用后，32 个训练帧的完整 H384 链仍净省 ready 1.7998%、BP 1.8390%。** 这替代父目录整 T10 单通道源臂作为本固定函数的最终强对照。权重/图/τ没有改变或重选，Wz 的网络 AEE 仍未知。[总账](SUMMARY.json)、[独立计数](PROFILE.json)。

输入是父目录已冻结的 Wz、全 H384 response canonical、固定 D 熵序；只把 X/case 列表扩大到既有32训练帧。`profile.py` 实际逐字节核过所有静态参数/图/root/rank与 [joined_zero.bin](../joined_zero.bin) 相同，前两帧和诊断 X 也相同。Wz仍按旧完整T10通道费用、只用frame0选pair，**没有按新frontier或frame1…31重新选择**。两训练帧上的局部Y相对RMSE 0.4561/0.4475、门翻1645/245760、权重平方改变量3682757沿原 [数值探针](../probe.json)；扩大输入是未用于选择的训练探针，不是留出测试或新AEE。

从已冻结 [retained_config](../../frontier_joined/retained_config/PLAN.md) 复制真实 [frontier_source.sv](frontier_source.sv)、wrapper与TB，仅用本地 zero-aware后端并让TB canonical搜索从0开始；构造过程保存在 [implement.py](implement.py)。本目录只跑 source mode2 exact-code / mode3 response-class，两臂均 `frontier=resident=activePF=1`、packed32、相同root全局word6174。每个P32命令的P0真实冷装A/τ/root/rank **21字**，其余31P复用同寄存器；每P仍清空X槽、graph cache和动态节点状态。跨新命令/模式重新冷装，没有借旧配置。

前沿臂可给十个时间lane分配不同channel，最多四个不同X通道驻留；只计算当前被请求的 `(p,c,t)`。故旧“完整T10通道任务”与本表的 `source_channel_refs` **不同单位**：后者计每次批发射中不同channel数，包含命中驻留后的再次引用；每个活 `(p,c,t)` 固定10个真实标量MAC。C++独立从实际active mask统计引用，Python另实现DAG路径、resident-first选取和FIFO替换，复核全部对数、引用数和X缺失读取，不拿RTL计数器自证。

| 固定范围，完整同Wz函数 | exact-code拍 | class拍 | 净省 |
|---|---:|---:|---:|
| 前两训练帧×P32，ready | 103136 | 99961 | 3.0785% |
| 前两训练帧×P32，BP | 122243 | 118415 | 3.1315% |
| 全32训练帧×P32，ready | **1763363** | **1731626** | **1.7998%** |
| 全32训练帧×P32，BP | **2079596** | **2041352** | **1.8390%** |
| 未用于选择的frame1…31，ready | 1712030 | 1682006 | 1.7537% |
| 未用于选择的frame1…31，BP | 2018706 | 1982420 | 1.7975% |

每帧均P32×T10×H384，主周期从top start接受到最后gate接受，所有池内配置、图/X读取、桥接、FC1/PSN和反压均在内。初始外存→池装填未建模。前两帧与32帧重叠，不能相加当独立样本；小范围第二遍只作验证。[小范围raw](small.csv)、[扩大范围raw](expanded.csv)。

| 全32帧实际工作/流量 | exact-code | class |
|---|---:|---:|
| 活 `(p,c,t)` / 标量MAC | 435787 / 4357870 | 414191 / 4141910 |
| 实际channel批引用 | 72028 | 67816 |
| X的128bit读字 | 115878 | 111930 |
| graph读字，ready / BP | 149318 / 148994 | 143654 / 143319 |
| 总读字节，ready / BP | 6311104 / 6305920 | 6157312 / 6151952 |
| 后端系数字 / 向量更新 | 77376 / 101260 | 77376 / 101260 |
| 后端96-lane PSN MAC issue | 316440 | 316440 |

双方都实际加载每H96的普通内容类map，canonical0在SOURCE路由阶段直接不建立job；D0没有LUT载荷。**exact-code同样享有所有静态零响应和内容去重。** ready源阶段1063019→1031282，后端阶段同699576拍，31737拍差全在源阶段；BP源阶段1270976→1232738，后端806956/806950的6拍差来自日历相位，不解释为少做后端计算。固定后端工作相同，不能把下游已享有的零消除再算成class独占收益。

资源仍是 **256 KiB共享八bank×128bit、每bank一个已接受在途请求**，按source/D/backend分阶段路由。源10和后端96个signed16×24乘法器是不同物理单元、串行活动，共106个；不是96-MAC同面积。较旧单通道源，普通frontier使用四个`X[2×128bit]`槽共 **128B**（原32B，增加96B），另保留128B graph cache；十lane slot索引、4×7bit tag/valid、FIFO、active/channel holding与mux均是真实逻辑，两臂共同拥有。静态寄存器复用不增A/τ/roots容量。桥接3840B+192B D、单H96后端Y/route/U/τ/cache等沿 [父资源表](../README.md) 保留；256KiB只是外部可见池，不包含所有局部数组。未做EDA或频率/面积推断。

小范围为2训练帧+全零X/交替signed24极值，两模式×ready/BP×两遍，共32个H384命令；扩大范围为32训练帧+同2诊断，两模式×ready/BP，共136命令。**合计168命令、20643840个Y/U/gate标量位置零差**，每次运行只初始reset，32P、4H96和所有后续命令都经握手重用。逐实际源MAC/U0/gate、部分时间lane、最低tie码/类、D展开、bridge、完整Y/U/gate、请求保持、唯一在途和输出保持全部检查。小例两遍与扩大范围对应例全部周期/计数相同；独立CPU核验168条记录通过。[small.log](small.log)、[expanded.log](expanded.log)、[profile.py](profile.py)。

该结果保留“消费者响应等价能反向减少源需判别状态”的固定执行证据，但它不是新的ROBDD、cache或预取算法。旧单通道源、不同root-bank、重复配置的数字在父目录继续保留为较弱控制；源单独32帧旧表也没有被此结果覆盖成同一种协议。Wz付出更大局部误差，未证明质量/服务共同占优；普通L2允许0的数值对照仍见父目录，本子目录没有另跑其frontier。当前只应保留这个有实测增量、AEE待证的候选，不继续挤剩余控制百分比。

复现：`bash run_all.sh`。只写本目录，复制冻结普通底座；旧文件不改，无新训练、选择或参数扫描。
