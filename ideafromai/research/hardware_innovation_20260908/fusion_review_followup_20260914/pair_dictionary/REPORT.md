# 四个 R2 字典：Q1 读取减少，真实八块核心慢 29.539%

固定32类/组的完整 RTL 已完成。相对原 native-window dual-P/cachedOS 强控制，真实八块无背压核心 **87,599→113,475 拍（+25,876，+29.539%）**；计入各自冷配置后为 **114,487→147,539 拍（+33,052，+28.870%）**。停止这个布局，不扩容量、ALU或整帧。该结果不否定局部重复权重本身，也不把已有 sum-first 公式当作新理论。

|真实八块，首命令|mode14 原控制|mode15 R2 字典|
|---|---:|---:|
|运行周期，无背压|87,599|113,475|
|运行周期，有背压|91,295|116,571|
|实际配置周期，两种运行背压设置相同|26,888|34,064|
|配置+运行，无背压|114,487|147,539|
|Q1 / 代表24bit读取|1,958 / 0|25 / 171|
|Q2 128bit读取 / MAC|756 / 17,556|756 / 17,556|
|direct dual-P 更新|5,245|65|
|class metadata24bit读取|0|1,958|
|计数块检查|0|19,580|
|计数向量更新|0|4,872|
|计数清零向量|0|2,160|
|class-slot代表退休非零位置|0|2,675|
|计数bank20bit读 / 写|0 / 0|45,208 / 48,808|

配置未转嫁给控制：mode14每 fixture首命令3,361拍；mode15为4,258拍，额外864个四组class词、32个四组代表词和1个尺寸词。两者第二命令配置0，但只是同一source/config的无reset重复命令，不是多tile流式服务。[TB配置分支](/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/pair_dictionary/tb.cpp:23)与[汇总](SUMMARY.json)保留原始口径。

## 实际接口与边界

原来要求整个R8权重列完全相等的24bit键，改为四个R2的6bit键。真实四组分别有26/21/28/29个非零键；固定频次≥2规则后实际选23/20/27/24类，覆盖2,772个非零rank-pair/K，剩10个非零pair走direct，零pair直接省略。高熵随机角落四组各48键，固定32类后仍有829个非零pair走direct，确实执行容量fallback。[静态编译](/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/pair_dictionary/prepare.py:18)、[fixture表](fixtures.json)

source仍经完整C96×4×4原生窗口装入和K864 gather。每个有非零源事件的K先花一拍读取四组class。direct pair与字典pair互斥：direct用原Q1向量读，把已分组rank的系数置零；字典把同组同键的源事件累加至unsigned10计数，K遍历结束后按代表×计数恢复对应rank的z。没有来自TB的中间计数、z、动态mask或预先择优结果。[分类与direct](/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/pair_dictionary/decomp_core.sv:174)、[计数与退休](/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/pair_dictionary/decomp_core.sv:212)

为了保持**八条32bit数据ALU总数**，四组各占两条ALU，每条在bit10切carry，可同时加两个unsigned10空间半字计数。每K有20个(P-pair,T)位置，因此每次处理其中两行、共10个块。每非空块要先读后写；四组可以有不同class地址，必须有四组独立地址选择，不能用一个广播地址口冒充。退休时同一class-slot包含四组各自的键，四组各读自己的count，各自两个rank乘代表；全部八乘法与八ALU复用原数据通路，更新同一z位置的对应rank。没有额外数据减法器或第二套乘法器。

计数共8bank×320row×20bit=6,400B；class2,592B、代表96B。每bank读地址由C_READ/G_READ显式选成单一地址，同一读数据既生成pending又写count-hold，没有给支持检查偷偷加第二读口。C_ADD/DCLEAR写口分时。完整端口、控制、内存范围见[资源合同](RESOURCE_CONTRACT.md)。union控制拥有同一资源集合；没有分别综合后的等面积结论。

## 为什么负

原第一层Q1请求/更新开销为 `2×1958 + 3×5245 = 19,651` 拍。候选相同部分及新增义务为：

`2×25 + 3×65 + 1958 metadata + 2160 clear + 19580 block-check + 2×4872 count-RMW + 224 slot-scan + 21×171 slot-read/scan + 3×2675 retirement = 45,527`。

差值 **25,876**，与完整核心结果精确一致。共用source、Q2、输出与固定控制状态不变。局部键增加了共享覆盖，但source稀疏时每活跃K检查全部10块的费用、计数读改写、按非零位置退休的费用超过直接packed更新。这里比较的是实际服务状态，不是用省下的乘法数预测周期。[独立费用核验](/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/pair_dictionary/verify.py:102)

极密正例仍存在：all-one **63,103→41,106**；全−3或+3同类极值 **63,709→37,078**。但真实较稀疏real_2 **6,410→6,855**，real_6 **16,680→22,956**；全部零source也因270拍count clear与28拍slot scan而多298拍。这些说明分布与固定调度影响结论，不把极密反例替代真实八块。

## 验证与结论范围

16 fixture（旧14+新随机padding poison+count864）×2模式×2背压×2无reset命令，共 **128条、491,520个实际RTL输出值全部一致**。TB检查每一输出地址、数据和背压保持，以及状态拍数总和。Verilator4.028 `-Wall --cc --exe`无豁免构建。[运行脚本](run.py)、[原始结果](results.json)

独立脚本从原source/Q1/Q2重建61,440个gold，逐pair核静态类别/代表，再从原事件单独求全部计数与direct残差，重建完整z；最大计数确为864。所有运行事务、每bank读写、冷配置、总拍与数学义务一致；56条旧fixture的mode14记录连原状态数组一起与冻结强控制完全一致。[验证结果](verification.json)。没有把本轮初次构建与最终显式单读口修订的重跑重复计入128条。

本叶采用旧分解组fixture；独立审阅已确认其real_6 source与旧方向组有19词差异，因此不把本数值与T10或consumer叶直接合并。无新I24、质量、整帧、DDR/NoC延迟或EDA/Fmax/能耗主张。静态表编译在CPU离线执行，其时间未计作每tile服务；动态class读、清零、计数和退休全部在RTL收费。已有[UCNN §III–IV](https://arxiv.org/html/1804.06508v1)给出重复权重因式化及分组复用机制；本叶仅测试限容量rank-pair表和本地双P计数银行接口，未复现作者完整架构。
