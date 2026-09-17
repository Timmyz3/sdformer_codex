# Active前沿child预取：有费扩大读取，小幅净收益

本叶关闭父实现已确认的一处接口缺口：child预取只跟旧最小rank channel，而不是本轮实际生产的多前沿。**同一冻结四槽核上新增一个PF范围开关，32帧固定输入全部RTL通过。** 31个未参与新类pair选择的训练帧（992个P1）上，resident exact-code ready **1064671→1060640（−0.379%）**，BP **1327565→1310925（−1.253%）**。这是普通exact-code也拥有的供数适配，不归为class特有机制。

图词ready130098→145268（+11.66%）；更多两侧分支读取换取LOOK等待减少。BP仍82/992病例变慢，最差+37拍，没有所有输入都改善的结论。没有改128B cache/128B X holding/10MAC容量、未扫描策略、无训练/AEE/EDA/PPA。

## B与X，准确到实际请求

B是[父冻结core](../frontier_source.sv)的旧谓词：

`node_id[t]>=16 && node_var[t]==channel && child>=16`

其中channel取全前沿最小D-rank；resident-first时它可能不在本批selected集合。父报告已经更正了“所有active前沿已获得child预取”的未实现表述，旧数据没有被改写。

X在 [frontier_source.sv](frontier_source.sv) 的 `start_active_prefetch=1` 时改为：

`node_id[t]>=16 && active_mask[t] && node_var[t]==batch_channel[lane_slot[t]] && child>=16`

因此只预取本批真实消费的当前节点的两个非终止孩子。对于旧单channel provider，虽然十个算术lane均active，仍必须匹配当前node变量，不能把其余t的未来节点当成本批已算。static next-X PF的覆盖分支保持原行为。参数0逐字保持旧相关性逻辑；1不读未来gate或分支答案，不更改DAG、source选择、四槽替换、算术或分类。

所有code/class均拥有同一0/1选择；普通single-channel/static实测**所有周期、流量、状态完全不变**。resident/plain前沿都测，主结果采用已完成供数适配的resident exact-code。固定32bit节点、D-only entropy rank、空槽优先/FIFO均不变。

## 资源和协议费用

继承父同核：唯一10个16×24mult＋10个48bitacc、X四槽128B、图cache128B、8bank×128bit共用服务、每bank1pending、同20bit pf_done。没有新增数据数组、图复制、乘加单元、外部口或在途容量。

新增一个start模式bit及active/slot匹配选择控制；slot匹配、mux、请求优先逻辑的综合复用和时序未测，不能声称绝对零面积。这个bit和其他模式位在同start事务传入，不另构造免费数据配置表；旧/新PF的实际配置word及启动服务逐病例一致。

每个非终止孩子是真实物理请求，两个分支都可能被读，无用分支、bank冲突、cache驱逐均保留。DRAIN继续等待被拒绝的有效请求接受，已经在途的请求可延后返回，LOOK在pending清空前不能启动下一X批；没有因prefetch范围扩大给更多返回端口。source填槽/按需MAC计划不变，所有source traffic、active pairs、批次、tag hits、refetch计数均与父独立CPU模型逐病例相同。

## 31帧主表：同函数新旧PF比较

输入仍来自[父expanded manifest](../../source_class_adapt/expanded_sources/manifest.json)，32个现有训练帧各固定32样本。frame0参与新W″类pair选择，frame1..31未参与；它们仍是训练缓存，不是独立验证集。旧W′/新W″class绑定各自whole-H384 response，不能把class当成原argmin code或同模型AEE。

| 992个P1执行臂 | 旧PF ready | active PF ready | 旧PF BP | active PF BP |
|---|---:|---:|---:|---:|
| static64＋next-X PF | 1286624 | 1286624 | 1367968 | 1367968 |
| 旧单channel exact-code＋PF | 1176829 | 1176829 | 1432418 | 1432418 |
| plain-frontier exact-code | 1079074 | 1075044 | 1367759 | 1345436 |
| **resident-frontier exact-code** | **1064671** | **1060640** | **1327565** | **1310925** |
| 旧单channel W′ class | 1173106 | 1173106 | 1422911 | 1422911 |
| resident-frontier W′ class | 1060942 | 1056578 | 1314381 | 1294111 |
| 旧单channel W″ class | 1158089 | 1158089 | 1408772 | 1408772 |
| resident-frontier W″ class | 1059105 | 1055455 | 1311467 | 1292189 |

旧新W镜像的static/code图和X/A/tau/D完全一致，全部static/code RTL记录逐字段一致。PF效果只在同mode/同函数的0/1列间相减；既有code/class根参数bank不同导致的BP启动差异不混入这个增量。root正在独立joined wrapper统一root物理地址；此处仍保持父接口和已测周期，不借其后续结果。

原64P小集 resident code为65584→65332 ready、81850→80755 BP；W′ residentclass为65414→65144、80947→79793。它们只作为小测保留，31帧为主。

## 更多物理读取为何仍变快，以及哪里变慢

| resident exact-code，992P | 旧PF ready | active ready | 旧PF BP | active BP |
|---|---:|---:|---:|---:|
| graph物理word | 130098 | 145268 | 130068 | 144953 |
| 其中MAC/DRAIN阶段预取word | 103724 | 127855 | 103720 | 127771 |
| 图LOOK状态周期 | 123849 | 119818 | 173725 | 154593 |
| XFETCH状态周期 | 142877 | 142877 | 308112 | 310611 |
| DRAIN状态周期 | 148677 | 148677 | 148682 | 148843 |
| graph节点消费者cache hits | 321547 | 341341 | 321563 | 341425 |

ready净省4031全部来自LOOK；X计划、MAC和排空不变。BP LOOK省19132，但XFETCH多2499、DRAIN多161，其余输出日历差抵消后总省16640。该成本账说明多预取不等于少流量；cache命中字段按节点消费者计，不能当物理字数。

逐病例：resident code ready 958快/24平/10慢（最差+2），BP824快/86平/82慢（最差+37）。新W″ residentclass ready945快/40平/7慢（最差+4），BP886快/66平/40慢（最差+23）。局部负结果保存在 [SUMMARY.json](SUMMARY.json) 的per_case_delta_statistics中；不能靠汇总正值隐去污染/等待个例。

此结果只证明当前有限cache/日历下，补齐实际active前沿的两孩子预取有小幅净收益；并不证明更深预取、更大范围或全图常驻会更快。没有再做第二种预取策略。

## 验证、复现与边界

[run.sh](run.sh) 使用Verilator4.028 `--cc --exe --unroll-count 512`＋make；Python3.12仅做统计核对，不生成训练数据或图。小集2680命令，两扩展各24648，共 **51976命令** 全PASS。实际partial-valid U/gate各检查 **26192500**，最终分类 **3118560** 个label。signed24正极值、交替正负极值和零输入诊断均通过；inactive U不当作完整source输出。

[verify.py](verify.py)核：PF0的25988条记录与父冻结结果所有字段相同；PF1所有逻辑source工作、X词、四槽命中/重读与父独立CPU验证计划一致；static/onechannel新旧PF整条记录相同。TB仍逐valid(t,c)核真实完整dot与gate、live node请求匹配、无重复(t,c)、完整Hamming/canonical、每bank单pending、请求拒绝保持、code输出背压和物理traffic。[verify.log](verify.log)、[expanded_old_cycles.csv](expanded_old_cycles.csv)、[expanded_new_cycles.csv](expanded_new_cycles.csv)为最终结果。

本叶已完成父缺失的active-frontier预取接口，但仍未完成joined FC1实接、无reset跨命令回收、跨P参数/图复用、独立验证集网络质量、时序/面积/功耗。这里的producer_valid是内部观察事件，没有任意外部ready协议；分类器内部立即消费active门，最终emitted code才有ready握手。没有把这一局部周期差外推为整网收益。
