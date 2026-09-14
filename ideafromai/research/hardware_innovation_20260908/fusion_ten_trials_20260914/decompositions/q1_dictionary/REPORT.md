# D3 静态重复列sum-first：省Q1词但慢2.989%

固定函数 `z=Q1g, rawp=Q2z`，Q1∈[-3,3]、Q2 signed16、无中间RNE。z signed13界±2592，rawp signed32绝对界679477248。全C96/N96/K864/T10/R8、完整480个256bit输出。共同前端native160bit窗和dual-P packed；共同后端最终rank/位置支持、Q2整向量零、128B全R8×N8缓存OS。mode14是上一阶段最强mode15，非旧scalar。各自额外资源两侧同权，不能称跨核或单独裁剪后等面积。


**B/完整A/接口：** 把Q1完整八rank signed3列作为24bit静态key；只选频次≥2的前32类，频次并列按key升序，其他列原样direct。类内 `count[g,p,t]=Σk∈g source[k,p,t]`，类退休 `z+=Q1_rep·count`，再执行完整Q2。它是静态重复权重/CSE/UCNN式sum-first的具体有限实现，不是上一轮动态z同幅cache。

真实Q1的862个非零K有793种完整列；32类覆盖69个K，见[dictionary_manifest](dictionary_manifest.json)。32class容量固定，所有K仍扫描，非字典列仍走最新双P。额外8bank×96row×20bit计数1920B，每字两个unsigned10字段，counter≤864；每k把20个P/T pair行分为三块，最多一次更新8bank的16个计数字段。共享32bit链在bit10断carry，原z更新在bit13断carry，其余模式完整32bit。

每实际类清3个八bank行；非空类退休读一次代表Q1、扫描3计数向量，再逐非零P/T用共享八乘法器及z读改写完成。额外class864×6bit=648B、代表32×10bit=40B、group-live32bit、计数hold160bit和有限pending16bit均声明，两侧同权。没有无穷字典、免费源计数或psum广播。

**首次cfg为4258=3361基础+32代表+864class+1尺寸，共897拍元数据；第二命令0。** definition.json与TB实际收据一致。静态编译只由Q1得到metadata，SV消费其完整加载；[verify](checks.json)确认每类均与代表列逐rank相等。

|真实8块首命令|14|15|
|---|---:|---:|
|无背压周期|87599|90217|
|有背压周期|91295|93959|
|Q1读取（包括代表）|1958|1916|
|非字典dual-P更新|5245|4863|
|计数块更新|0|257|
|计数clear向量|0|768|
|活动类代表读取|0|110|
|退休非零P/T count|0|432|
|Q2读取 / MAC|756 / 17556|756 / 17556|

被替换的152次Q1读取和382次packed更新所省状态，不足支付count块检查/读写和432次退休；核心净多2618拍。Q1读少42词，不将不同位宽计数字量变成能耗推断。极值全部Q1列相同形成一类，63709→18049；此机制反例说明重复分布很重要，只保留当前固定32类为负控制，不杀sum-first家族。

功能：14旧fixture（真实四角/四内部、zero/one/corner、正负极值/zero_factor）×两mode×双背压×两次无reset命令，112runs/430080输出全绿。每条mode14另与旧最终packed15逐记录验证896个标量指标和56状态数组完全一致，见[checks](checks.json)。只读[独立验证脚本](../verify.py)重算53760个gold与本表示，未把统计变成周期。配置与输出data/address保持在TB中实际检查；错误metadata/运行中改cfg/无reset换mode不属本合同。

本叶未新训练/消费者/AEE/整帧，未做Fmax/PPA。source_allow/weight_allow门控已配置本地flop/mux读，不是DDR/NoC响应模型。`weight_words`混合不同宽度格式，不能直接作为字节或能耗。所有资源口、半字写使能和组合路径详见[resource_contract](resource_contract.json)；bitstream/LUT/count是SV状态，TB不提供动态中间值。
