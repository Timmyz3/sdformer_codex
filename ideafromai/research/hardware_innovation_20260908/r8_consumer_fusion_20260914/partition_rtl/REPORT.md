# 全 R8 绝对值分组：有限缓存端点未胜过加强后的 OS

2026-09-14。**120 runs（112旧回放+8定向）、460800个完整raw输出通过；真实八块 mode13 比同资源强控制 mode12 慢3.91%。** 此处停止扩大支持缓存，不用旧mode8或expanded direct替代分母。已付费的完整分组端点保留为负控制。原生源/Q1→完整z→Q2→全部N96输出均由[SV](partition.sv)执行，TB未输入latent。[PLAN](PLAN.md)在实现前写定，[资源合同](resource_contract.json)、[实测](results.json)、[汇总](SUMMARY.json)可直接复核。

## 函数与强对照

固定Q1∈[-3,3]、Q2 signed16，`z=Q1g`，`p=Q2z`，无中间RNE。mode12先付40行最终z扫描得到位置/rank活性，再为每N8读取并驻留整个R8 Q2块（128B），输出psum保存在八寄存器中；每非零rank一拍MAC，完整480个向量写出。它不重复取Q2，也不再逐rank往psum数组读改写，故不是旧较慢mode8。

mode13把每个(P,T)的八rank按相同绝对值分组。最低rank为leader，标量`v=z_leader`，成员符号`sr=sign(zr)sign(v)`，贡献为`v·ΣsrQ2[:,r]`。所有组共享一次产生的320项上限描述符，跨全部N重放；每N8固定8项signed支持系数缓存，清epoch、roundrobin替换、miss构造全部在RTL。singleton直接走Q2，不挤占组缓存。

leader符号必须归一为+1。这样系数和最坏为[-262144,262143]，signed19充分；直接用未经归一的`Σsign(zr)Q2`可能得到+262144，不能错误塞进19位。任一组或全部psum绝对界均不超过679477248；两侧共用八个19×13乘法器及八条32位加减链。正负极值fixture实际达到±679477248。

静态k_live、Q2全N8向量零、最终rank零均给两侧同权限。本14套数据没有“有效组的整N8系数和刚好为零”而未跳过的弱分母，见[checks](checks.json)。所有完整输出均写入后drain；不以省略零输出或初始psum残留省周期。

## 同口、同状态权限的完整费用

两侧共同拥有8bank×40×13bit latent，第一阶段104bit向量读写、第二阶段标量读，互斥；source单10bit字口；Q1读24bit向量、Q2读128bit向量，受本地许可控制。这不是外部DDR响应模型。共同拥有128B Q2块寄存器、320×29bit描述符（1160B）、40×13bit start/count、8项16bit tag＋8×19bit系数的缓存、完整psum数组等。mode12不用部分状态，不证明分别裁剪后等面积。

descriptor仅在LOOKUP读一次。最终版本加了共同29bit hold，miss的BUILD/GROUP_MAC只读该hold；修正后原112 runs重新通过，真实汇总不变。mode12的BASE_MAC同样允许异步z标量reg/mux读、Q2块选择、19×13乘法及32位累加在一拍完成。组cache命中路径包含descriptor mux、有限tag比较、cache选择、乘法和加法，Fmax未测。所有数组均为显式寄存器/mux权限，未声称单口SRAM宏可免费实现。

|真实8块，首命令|mode12 cachedOS|mode13 partition|
|---|---:|---:|
|无背压核心周期|98895|102757|
|有背压核心周期|103582|107445|
|源10bit读取|23364|23364|
|Q1 / Q2向量读取|1958 / 756|1958 / 756|
|latent向量 / 标量读取|6733 / 17556|6733 / 0|
|完整psum读 / 写|3840 / 3840|3840 / 3840|
|八lane MAC拍|17556|12180|
|描述符产生 / 读取|0 / 0|1015 / 12180|
|abs / 组构造ALU拍|0 / 0|207 / 7920|
|cache hit / miss / eviction|0 / 0 / 0|372 / 3096 / 2400|
|epoch清理拍|0|96|

核心差值精确分解为：`(12180−17556)+7920+1015+207+96=3862`拍。乘法少30.62%，但组构造及前端税更大。8项缓存对multi组仅372次命中；这不是因表被设成无限容量后得到的乐观数。背压下差3863拍、慢3.73%。

每fixture首次真实配置3361拍（其中源/原点1537），第二命令不重装，JSON对应为0。常量驻留但每块新装源/原点时，八块为111191对115053拍，仍慢3.47%；这只是本叶的边界，并无整帧或消费者结果。

[verify.py](verify.py)只读重算15套57600个int64 gold及分组恒等、位宽、完整输出义务、120行状态和/主要事务；第二命令除setup外逐项复现第一命令。TB包含双背压、不reset重启、输出data/address保持；每行3840值全部一致。配置不一致、不reset换模式、任意epoch外改Q2未作为受支持测试。

## A/X与下一接口的边界

UCNN的重复值分组（此处交换权重/activation角色）、Phi pattern-weight缓存、SmartExchange基驻留都属于强A，primary阅读范围见[本阶段差分](../novelty/REPORT.md)。完整signed magnitude partition是本次新实现接口，不是已证论文创新。它尚未战胜同权限cachedOS，不能把旧R8对expanded的53%归给它。

按signed支持先排序、跨P/T消费虽然可减少cache miss，却失去mode12的寄存器psum驻留，还需scatter读写。旧8块每N8理想每种canonical multi-support只构造一次，仍需7212次构造发射；当前7920，最多省708，填不平3862负差，排序/重排费用尚未加入。因此不晋级该排序点，也不做8→16缓存扫描。这一计数界只约束当前构造方式，不冒充所有共享表示的下界。

全±3边界例则116529→113357拍，证明不同分组分布可改变结果；它只作功能/结构反例，不替代真实八块。用户另选双P窄latent packing作为不同表示接口，独立目录、共同更宽bank口重新设强控制；不得借其共同前端收益挪给partition。本端点无新训练/AEE/825、无消费者、无EDA/PPA/Fmax声明。

## 独立审阅后的必要定向覆盖

Root静态审阅要求补“原leader的Q2整N8向量为零”分支。原14fixture没有有意覆盖该条件。新增[q2_zero_leader生成脚本](prepare_targeted.py)只用Q1首列 `[1,-1,-1,1,0,0,0,0]` 与全1原生源，按N8设置负singleton、全负多成员、正负混合多成员及空组。原leader的标量继续留在描述符，删去leader后保留成员的符号不重新解释。

新增8runs（12/13×双背压×无reset第二命令），30720输出全部通过。每个mode13命令有120负singleton、6次multi miss、234次multi hit、120次空组、12次构造ALU发射；极值负Q2=-32768也覆盖。无背压12/13为4551/4535，有背压4795/4782。它只验证过滤后的符号/tag/epoch分支，不进入真实性能汇总，详见[targeted_results](targeted_results.json)。总120runs/460800输出。重建时先prepare.py，再prepare_targeted.py，随后run.py与verify.py；不需重跑拟合。
