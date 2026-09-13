# r0 C96/N96/T10：完整整数 Winograd 核与必要强控制

最终控制：共同有限64项静态支持后继选择器。**88条RTL命令、337,920个signed32输出逐值通过**；其中64条使用真实8块源，245,760个输出。无mask Winograd和原W16 AAC严格同函数；masked Winograd和phase-expanded E严格同函数。真实块上原direct更快：exact Winograd为其 **1.965×周期**，masked Winograd仍为 **1.630×**。本固定点没有证明新的X胜过原函数强A。

## B、完整强A与本轮X

B是当前r0.conv2.0：W96×96×3×3，T10输入二值，θ=1；整体profile名义63.701G MAC。这里真实执行一块4×4→2×2，全部C96/N96/T10；没有用C8子块外推。强direct以H16权重向量跨T10及4个输出复用，按真实源事件跳零，W紧密signed16。每个模式共享同一核的8条signed48加减链、局部数组预算、支持跳过、2行CR缓存和端口。

整数Winograd的完整**线性执行A**包括离线U4常量编译、SV源变换、全部C累加、付费3U准备、原位逆变换、RNE2、物理CR请求/响应及背压。WINS完整训练／恢复及论文系统没有复现；只借入坐标结构这一模块，不把论文标题当完成证据。

本轮固定X为每O8跨C96删4/16个ξ坐标，再由源有限取值与消费者支持取消实际半组发射／物理CR需求。坐标剪枝、有限字母表、±2移位、零跳过与缓存都已有A；本实验考察它们是否在真实昂贵对象形成新增执行收益。当前结果只支持该固定实现的功能与周期事实，尚不足建立新方法。

## 算子与边界

`Bt=[[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]`，`At=[[1,1,1,0],[0,1,-1,-1]]`，`G2=[[2,0,0],[1,1,1],[1,-1,1],[0,0,2]]`。

`V=Bt S Bt^T; U4=G2 Wq G2^T; Z=At(Σc U4⊙V)At^T; Y=RNE2(Z)`。无mask时Z=4·direct整数累加，所以RNE2不改函数。源S∈{0,1}，15个V坐标在−2…2，ξ=5在0…4；real_01/03/06等实际覆盖3、4，3U由同8链付拍形成。不是把±2称新。

任意U坐标mask改变函数。令`L=At⊗At`、`R=Bt⊗Bt`，强同函数直接控制为`E4[o,c,p,s]=Σξ L[p,ξ]·mask[o/8,ξ]·U4[o,c,ξ]·R[ξ,s]`，`Y[p]=RNE2(Σc,s E4[p,s]S[s])`；这是4×16相位核，不能当普通3×3。ξ=i*4+j，p=a*2+b，s=source_row*4+source_col。E的代数推导是本轮核验，不声称文献新定理。masked的43.8737%输出在/4前非整数；SV arithmetic-shift+ties-even处理正负数，覆盖计数在SUMMARY.json。原叶无bias，W指数16，输出为signed32 Q16线性域；未跨norm/residual/PSN。

对任何合法signed16 W，|U4|≤9·32768、|E4|≤81·32768，均可容signed24；3U可容signed32。|Mξ|≤96·9·32768·4，逆变换至多9项，48位不溢出。保守E累加界96·16·81·32768，经/4小于2^30，signed32输出安全。当前真实U4范围[-76576,95214]，E4[-76576,90645]。不在变换中加舍入。

## 实测周期（8个真实块求和）

| 模式 | 背压 | 总周期 | CR256请求 | CR字节 | 8路ALU有效周期 |
|---|---:|---:|---:|---:|---:|
| direct | 0 | 178,948 | 11,790 | 377,280 | 81,012 |
| wino | 0 | 351,567 | 34,362 | 1,099,584 | 123,892 |
| masked_wino | 0 | 291,639 | 28,026 | 896,832 | 105,500 |
| masked_expanded | 0 | 398,176 | 38,190 | 1,222,080 | 81,012 |
| direct | 1 | 219,807 | 11,790 | 377,280 | 81,012 |
| wino | 1 | 466,423 | 34,362 | 1,099,584 | 123,892 |
| masked_wino | 1 | 385,779 | 28,026 | 896,832 | 105,500 |
| masked_expanded | 1 | 525,294 | 38,190 | 1,222,080 | 81,012 |

正常CR响应延迟1拍；背压点响应延迟4拍、请求周期7的第2拍停、输出周期9的前3拍停、输入周期11的第3拍停。所有请求/响应、实际输入、配置、状态初始化、选择、变换、最后一笔输出及完成拍计入。不是CPU操作数换周期。每条命令冷缓存、完整配置和120拍原源装入；不假设跨tile权重驻留。

| 真实块 | direct | exact Winograd | masked Winograd | masked E-expanded |
|---|---:|---:|---:|---:|
| real_00 | 15,359 | 35,019 | 29,244 | 36,380 |
| real_01 | 25,811 | 52,352 | 43,325 | 53,690 |
| real_02 | 6,935 | 15,640 | 13,671 | 23,036 |
| real_03 | 10,811 | 23,570 | 20,190 | 29,588 |
| real_04 | 20,975 | 42,918 | 35,596 | 45,614 |
| real_05 | 15,395 | 33,281 | 28,052 | 37,010 |
| real_06 | 50,303 | 80,910 | 65,977 | 104,798 |
| real_07 | 33,359 | 67,877 | 55,584 | 68,060 |

正常点masked相对exact减少 **17.05%** 周期，但函数已变。相对**同一masked函数** E-expanded减少 **26.76%**；这只证明该函数用变换表示更省，不能据此包装成超过原direct。未与邻目录不同资源原生核跨核算倍率。全一合成源使V仅ξ5非零，可显示Winograd适合另一活动分布；它不是当前数据分母。

## 支持强控制与账目

初版逐项SCAN的结果完整保存在old_scan_control/，最终表已替换为真实RTL有限64项块内next-live控制。masked U有1920/9216个全零H16向量及768个单活O8半组；E有16128/36864全零向量。所有模式均从同一2bit/vector metadata选择后继，整空块最多一拍越过，不用离线周期oracle，也不把理论下界当测量。W原始无全零H16向量。共同权限还包括整源块零快速路径、事件零跳过、O8半组静态零跳过、实际紧密字节跨行与缓存复用。

W16静态165,888B；U24 442,368B；E24 1,769,472B。U的16/9系数及24/16位膨胀均付费。metadata配置分别1296/2304/9216B；原W没有被强制加宽。CR按32B行请求，只有需要的半组字节区间被取，2行缓存实际复用重叠词。

| 正常点状态账（8块） | direct | exact Wino | masked Wino | masked E |
|---|---:|---:|---:|---:|
| IDLE/config/start | 656 | 1,160 | 1,160 | 4,616 |
| LOAD | 960 | 960 | 960 | 960 |
| TR1 | 0 | 7,335 | 7,335 | 0 |
| TR2 | 0 | 6,800 | 6,800 | 0 |
| CLEAR | 3,840 | 15,360 | 15,360 | 3,840 |
| SCAN | 41,472 | 73,728 | 58,368 | 165,888 |
| LOOKUP | 11,790 | 42,240 | 32,891 | 41,088 |
| REQUEST | 11,790 | 34,362 | 28,026 | 38,190 |
| RESPONSE | 11,790 | 34,362 | 28,026 | 38,190 |
| DECODE | 11,790 | 21,120 | 16,965 | 20,544 |
| PRE3 | 0 | 348 | 348 | 0 |
| AAC | 77,172 | 83,064 | 64,672 | 77,172 |
| INV1A | 0 | 7,680 | 7,680 | 0 |
| INV1B | 0 | 7,680 | 7,680 | 0 |
| INV2A | 0 | 3,840 | 3,840 | 0 |
| INV2B | 0 | 3,840 | 3,840 | 0 |
| ROUND | 3,840 | 3,840 | 3,840 | 3,840 |
| SEND | 3,840 | 3,840 | 3,840 | 3,840 |
| FINISH | 8 | 8 | 8 | 8 |

每行状态之和等于总周期；完整88行在cycles.csv和rtl_results.json。真实源活动率从0.0130%到6.9466%，8块是同一帧四角+四内部，不能把它当全数据集平均或全图层速率。

## 资源公平性与尚未闭合项

[resource_contract.json](resource_contract.json)给精确数组/接口。共同8条signed48 carry-chain没有通用乘法器；地址计数和控制组合逻辑另有普通整数算术。source 1920B、V7680B、M15360B、temp64B、work48B、support9216B，加所列系数/cache等共34562B数组状态，另有FSM/地址计数/输出寄存器。direct可不用V/多余M，但共同预留不能证明面积相等。

内部为flop/mux：source有T10 gather及8通道transform读网；V为10读/8写；temp有两组8元素读；M在inverse-A需16×48bit并读，在AAC/inverse-B为8读8写；support为64项读取/后继选择。它们不是单口SRAM。共同CR256、input128、config128、output256端口已实跑；未测选择器—数组mux—48bit链路径Fmax，未综合、未EDA、无PPA或同面积结论。

入口是已捕获的连续4×4原二值与padding，TB只装入这些原值、静态系数/metadata和独立gold，未计算动态V/Y。整图line-buffer、tile重叠复用、图像源寻址、PSN及下游不在本核。静态模型常量离线编译双方同权，外部ROM容量及实际读取收费；未假装runtime权重变换免费完成。

## 独立审阅

sparse代理独立只读审阅未发现阻断：[REVIEW_WINOGRAD.md](../native_sparse/REVIEW_WINOGRAD.md)。有限核验包括65536二值源幅值域、144个核/源basis整数恒等、16个basis原位inverse、16385个signed RNE四分之一值、672768个存活密排系数解码，以及64真实AAC/SCAN和88行请求响应／背压／状态账。它没有重跑或修改本RTL。非阻断覆盖缺口：当前真实数据没有触发“仅低O8存活且vector_base mod32=16”的两行取数分支；各TB命令reset，未测不reset换模式。共同预算／多读口边界保持；masked模式也没有证明独立于普通WINS的新X。

## 质量与判决

本地mask相对原整数输出NRMSE **20.8638%**，无mask exact误差0。mask校准正是本8个tile，且与frame0/diverse10重叠；不是held-out。独立data_and_quality现已补齐diverse10、每臂516735个有效像素：frame-mean AEE parent FP32 **1.157520**，dense Q16 **1.163297**，coordinate25 **1.172116**；去掉校准帧的holdout9分别1.139374、1.137301、1.154187。coordinate相对dense十帧AEE增加约0.758%，低于历史NB0十帧门1.454603；当前环境fresh NB0十帧重跑为 **1.462941**（同516735有效像素、完整参数装入），也高于上述各臂；仍保留更严格历史门，不借新分母放宽。这里未恢复训练、使用TF32浮点CUDA消费者，phase卷积未模拟本RTL末级RNE2；这是十帧质量证据，不是整网bittrue或valid825精度PASS。此前缺第二帧的旧quality.json已由新[diverse10.json](../data_and_quality/diverse10.json)取代；数据协议见[protocol_audit.json](../data_and_quality/protocol_audit.json)，fresh NB0见[nb0_diverse10.json](../data_and_quality/nb0_diverse10.json)。

结论是停止把本固定F2/U24/H16/2行缓存点推进为获胜X，继续以r0昂贵真实对象为目标。负结果不杀Winograd家族：层级供数/驻留、恢复训练后更强支持结构可能改变结论，但本轮没有新增扫参或外推。小V可留功能试验平台，不替代r0目标。

## 复现与来源

运行`bash reproduce.sh`，要求本地Verilator4.028、g++、/opt/anaconda3/bin/python3.12及同级真实NPZ；脚本只写本目录。Verilator使用`--cc --exe`再make，未使用新版--build。prepare.py的原direct与变换整数恒等检查、prepare_real.py对flow U4/E4/gold逐数组核对独立于SV；tb.cpp仅合法握手/ROM/逐输出比较。合成3块、真实8块×4模式×2环境共88条。旧控制不覆盖。源码链接：[RTL](winograd_tile.sv)、[TB](tb.cpp)、[静态编译与gold](prepare.py)、[真实对齐](prepare_real.py)、[运行器](run.py)。

方法来源继承此前已读范围，不新增文献数量：[Lavin & Gray, CVPR2016, §4 F(2×2,3×3)](https://openaccess.thecvf.com/content_cvpr_2016/papers/Lavin_Fast_Algorithms_for_CVPR_2016_paper.pdf)给基本变换；[WINS, ICCV2025, §§2/4/5](https://openaccess.thecvf.com/content/ICCV2025/papers/Park_WINS_Winograd_Structured_Pruning_for_Fast_Winograd_Convolution_ICCV_2025_paper.pdf)给Winograd结构剪枝先验；全文阅读、代码状态和近碰撞见[此前研究报告](../../deep_target_research_20260913/decomposition/REPORT.md)及来源表。输入有限字母表65536穷举见该目录winograd_binary_exhaustive.json。完整A各模块边界如上，不把未复现论文系统称RTL已有。
