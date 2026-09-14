# D2 两组R4 DA：无乘法后仍慢14.206%

固定函数 `z=Q1g, rawp=Q2z`，Q1∈[-3,3]、Q2 signed16、无中间RNE。z signed13界±2592，rawp signed32绝对界679477248。全C96/N96/K864/T10/R8、完整480个256bit输出。共同前端native160bit窗和dual-P packed；共同后端最终rank/位置支持、Q2整向量零、128B全R8×N8缓存OS。mode14是上一阶段最强mode15，非旧scalar。各自额外资源两侧同权，不能称跨核或单独裁剪后等面积。


**B/完整A/接口：** 每N8用共享八ALU建立 `L[g,m]=Σr∈g m_rQ2[:,r]`，两组R4各16项；Q2仍只读一次驻留。输入z先读208bit并保留八signed13，找全部有效rank的最小公共signed宽度w。对于b<w−1做 `+L[g,mask_b]<<b`，最高b=w−1做减法。零mask/整LUT向量零不发射。DA/分组LUT/符号修正属于经典A，未形成独立X。

每N8构造2个零项和30个非零项，合32拍；整tile384拍。LUT为2×16×8×18bit=576B，系数界[-131072,131068]，signed18恰当。每项从去掉最低setbit的父项加一个Q2系数，使用同八32bit加法链；另给两侧8个32bit移位器、144bit LUT口和32bit LUT零metadata。最小宽度编码使用有限13bit×两组支持，没有无限查表。

位平面中间值保持signed32：即使先合所有低位的最坏同号项，绝对值也在signed32内；最后符号位修正得到原rawp。相同继承乘法器在候选Q2路径闲置，不把它们从候选资源删除再称同面积。DA_MAC是异步LUT读+shift+sign/add同拍，未测时序。

|真实8块首命令|14|15|
|---|---:|---:|
|无背压周期|87599|100043|
|有背压周期|91295|103717|
|Q2乘法MAC发射|17556|0|
|LUT写（含零项）|0|3072|
|构表ALU发射|0|2880|
|DA查表/shift/add发射|0|21960|
|额外z向量读取/展开|0|2484 / 2484|
|Q1/Q2读取|1958 / 756|1958 / 756|

无背压精确增量：`3072+2×2484+(21960−17556)=12444`拍。运行时构表、位置展开与多出的查表发射超过省下的乘法；减少乘法不等于减少服务周期。极值max_negative相对控制仅多384拍，但real_2多420拍。整块无有效z时当前仍构建LUT（zero多384拍）；这是固定首点的显式优化缺口，不用它证明DA家族无效，也不影响真实八块负结论。首次配置3361、第二次0。

[checks](checks.json)独立验证14fixture×12N8×2×16×8=43008个LUT系数、最小signed宽度、全部DA重建和运行事务。

功能：14旧fixture（真实四角/四内部、zero/one/corner、正负极值/zero_factor）×两mode×双背压×两次无reset命令，112runs/430080输出全绿。每条mode14另与旧最终packed15逐记录验证896个标量指标和56状态数组完全一致，见[checks](checks.json)。只读[独立验证脚本](../verify.py)重算53760个gold与本表示，未把统计变成周期。配置与输出data/address保持在TB中实际检查；错误metadata/运行中改cfg/无reset换mode不属本合同。

本叶未新训练/消费者/AEE/整帧，未做Fmax/PPA。source_allow/weight_allow门控已配置本地flop/mux读，不是DDR/NoC响应模型。`weight_words`混合不同宽度格式，不能直接作为字节或能耗。所有资源口、半字写使能和组合路径详见[resource_contract](resource_contract.json)；bitstream/LUT/count是SV状态，TB不提供动态中间值。
