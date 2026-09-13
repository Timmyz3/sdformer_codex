# 固定整数因子链：独立函数与完整周期核对

2026-09-14。只读 [integer_factor.sv](../integer_factor/integer_factor.sv)、[tb.cpp](../integer_factor/tb.cpp)、fixture、[results.json](../integer_factor/results.json) 及定义。没有重新运行GPU或RTL，也未改rank、格式、掩码或训练。

**独立函数与账本核对通过，未发现本固定合法接口内的具体计费或数值错误。** 14个fixture、168条RTL命令、645120个输出通过作者回放。审阅从每套source/q1/q2/expanded重新计算53760个完整int64参考输出，与gold逐值一致；全部168条运行的2688项事务、配置和绝对周期检查相等。入口 [review_integer_counts.py](review_integer_counts.py)、收据 [review_integer_counts.json](review_integer_counts.json) 没有导入作者公式或供给RTL答案。

## 函数与共同静态零控制

独立解码q1.hex为 `[8,864]`、q2.hex为 `[96,8]`，expanded为 `[96,864]`，逐套验证 `expanded=Q2@Q1`、`k_live[k]=any(Q1[:,k]!=0)`。原生source先按signed origin实施图外0，再按四个输出phase取完整3×3/C96，复算 `z=patches@Q1.T`、`p=z@Q2.T` 和展开同函数，三者与gold的O8/phase/T/lane布局相符。

真实Q1有两个全零K列，expanded相应24个8lane行全零。mode6在每个真实几何目的用同一k_live过滤源/W请求；mode7/8在读四phase源之前跳过同一死K。它不是候选独占的稀疏控制。direct仍先读其C4原生源并枚举目的，若该目的所有源因k_live失效，仅付DEST_PICK/ADVANCE两拍；独立公式明确计入此路径。

zero_factor的全零函数、零源、全一源、边界和正负极值均作为功能fixture。zero_factor下direct没有额外全任务取消，而因子可逐K跳过；**该fixture不用于速度赢家判断**。

Q1的signed3域实际[-3,3]，完整z界±2592，signed13足够；Q2为signed16，产品13×16落signed29再符号扩展至32。完整p绝对界679477248，signed32足够。展开W需要signed21，父和signed22、四源scratch signed23；它与旧Q16原函数不是同一系数位宽/同一函数。审阅包含到±679477248的功能参考，未新增中间RNE。最终每输出scale位于此raw整数叶之后，不在RTL周期内。

## 独立第一阶段公式

对一个完整tile，Klive是活K列数；A是所有活K在P4/T10上的源活动总数；Q是其中至少一个P/T活跃的K列数；Z是最终8×40个z中的非零数。这里A已经包括空间消费者重复，不能直接用4×4原生源的popcount替代。

活K固定付四拍F_SOURCE、一拍F_CHECK、一拍F_KNEXT，图外置0也占source状态拍。死K付F_SOURCE判定和F_KNEXT两拍。每非空活K再付一拍Q1读取和一次F_TIME空集合结束检查；每活动项付F_TIME/F_ZREAD/F_ZADD三拍。因此：

`first_phase = 2*(864−Klive) + 6*Klive + 2*Q + 3*A`。

source bank事务只数图内且活K的四phase请求，不把图外状态拍叫SRAM读。Q1读取Q次；第一阶段z向量读A次、写A次，另40次z清零；first_issues=A。每次向量操作是8lane，即使某个Q1系数为0仍共用这一向量拍，未假定逐lane零值可免费取消整拍。

真实8tile合计Klive=6896、Q=1958、A=6413、Z=1463，第一阶段64563拍。实际源读23364，与direct原生读取9600不同：因子F_SOURCE按K/phase重新取源，重复和padding都已计费。

## 第二阶段、psum与完整周期

mode7按og/r驻留V，每tile固定96次V向量读；遍历每个rank的40个位置，一律付F_SCALAR/F_POSNEXT两拍，遇非零z再付F_RPS/F_MAC两拍。令 `M=12*Z`：

`second7 = 96 + 2*12*8*40 + 2*M = 7776 + 24*Z`。

mode8按og/位置保留本地psum，每个位置付F_ACLEAR和F_STORE，再对八rank付F_SCALAR/F_RNEXT。非零z付一次V读取和一次MAC：

`second8 = 12*40*(2+2*8) + 2*M = 8640 + 24*Z`。

mode7的V驻留减少了权重请求，mode8的本地累加减少了psum读写；这些权利和代价都真实执行。两者MAC数相同M，z scalar读相同3840。z_reads计数将第一阶段8lane向量访问与第二阶段单bank标量访问各计一次状态事务，**不是相同bit宽的能耗单位**。

完整绝对核心周期是：

`core7 = 1482 + first_phase + 7776 + 24*Z`

`core8 = 1482 + first_phase + 8640 + 24*Z`。

1482包含480拍psum清零、40拍z清零、一拍F_OUTPUT、960拍drain及一拍FINISH。二者每tile差恰为864拍，真实8tile差6912拍；此固定接口下mode8少psum事务但控制与重复V访问合计并不更快。该结论针对实际状态机，不外推所有输出驻留/融合因子实现。

| 八个固定真实tile，无背压核心 | direct mode6 | 因子mode7 V驻留 | 因子mode8位置累加 |
|---|---:|---:|---:|
| cycles | 370240 | 173739 | 180651 |
| source读 | 9600 | 23364 | 23364 |
| W/Q1/V向量读 | 40992 | 2726 | 19514 |
| 其中第二因子V读 | 0 | 768 | 17556 |
| z读 | 0 | 37133 | 37133 |
| z写 | 0 | 6733 | 6733 |
| 第一阶段8lane加法issue | 0 | 6413 | 6413 |
| 第二阶段8lane MAC issue | 0 | 17556 | 17556 |
| psum读（含drain） | 70692 | 21396 | 3840 |
| psum写（含clear） | 70692 | 21396 | 7680 |

mode7 psum读写均为M+480，mode8读480、写960（clear和最终store各480）；清零、最后连续输出没有由TB代做。TB每次完整14017拍配置全部source/origin/W/mask/Q1/Q2/k_live，共同权限一致；新tile源/origin1537拍另计。每fixture两次同参数无reset和两种背压的计数均已核对；背压先扣各实际stall再比无背压公式，不强迫不同调度具有同一stall相位。

## 可保留结论与限制

此相同raw整数函数下，mode7/8都比已给静态零控制的expanded direct少核心周期；mode7强于mode8。两个因子并未免费获得原生供数，source重复、z物化、V驻留/重复和psum都进入账单。

同一个SV为所有模式保留八条32bit数据加法链和八个13×16乘法器、expanded/Q1/Q2存储及z状态。direct也有这些共同资源，但这不等于原Q16无乘法叶同面积。不同向量位宽、内部memory/寄存器实现、乘法器延迟尚未综合/STA，不报ASIC PPA或把此tile周期直接外推整帧/全网络。

本轮真实新链质量见 [integer_factor_diverse10.json](integer_factor_diverse10.json)：A800十帧AEE1.3533257656301605，低于NB0十帧门；z/p精确FP64模拟后才乘scale并接原float32消费者。它是新Q1/Q2函数的实际评价，不继承旧FP32 SVD AEE；不是valid825或全网bittrue。普通因子化、驻留和中间局部累加仍属于执行A，不能凭这次加速自动称为新X。
