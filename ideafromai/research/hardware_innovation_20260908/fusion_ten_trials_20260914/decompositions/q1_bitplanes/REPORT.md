# D1 Q1位平面：真实八块慢18.618%

固定函数 `z=Q1g, rawp=Q2z`，Q1∈[-3,3]、Q2 signed16、无中间RNE。z signed13界±2592，rawp signed32绝对界679477248。全C96/N96/K864/T10/R8、完整480个256bit输出。共同前端native160bit窗和dual-P packed；共同后端最终rank/位置支持、Q2整向量零、128B全R8×N8缓存OS。mode14是上一阶段最强mode15，非旧scalar。各自额外资源两侧同权，不能称跨核或单独裁剪后等面积。


**B/完整A/接口：** 对二进制源bitmap `b`，`z=pc(b&Qbit0)+2pc(b&Qbit1)−4pc(b&Qbit2)`。每次处理16K、八rank并行popcount。位串/AND-popcount/移位及符号已是BISMO类强A；此次实际接口包括原生source到全部40个bitmap的转换及完整Q2，不称新的乘法定理。

每k在BM_PACK向40个bank各写一个bit，包含dead k置零和边界零；全部864位覆盖，所以无需bitmap预清。每(P,T)扫描54个16bit字，非空字才逐个请求三个系数平面；整128bit系数平面零按已付cfg生成的bp_live跳过。普通Q1格式和位平面均真实存储并在同cfg以独立bit写权限形成，两侧同权。

额外源bitmap4320B（40×54×16），额外系数bitplanes2592B、162bit活性metadata；8个popcount16树共120个低位宽加法节点，另有共享8条32bit加减和移位。128bit系数读宽于普通Q1的24bit，mode14拥有同一额外格式/端口/树权限，但本结果不证明面积相等。BM_POP经过AND/popcount/shift/sign/add整条组合路径，未测Fmax。

|真实8块首命令|14|15|
|---|---:|---:|
|无背压周期|87599|103908|
|有背压周期|91295|110023|
|Q1 packed更新 / popcount发射|5245|9012|
|source bitmap16字读|0|17280|
|40bank bit写拍|0|6912|
|Q1读取|1958×24bit|9012×128bit|
|Q2读取 / MAC|756 / 17556|756 / 17556|

每fixture首次配置3361、第二次0。候选bitmap构造/扫描与三平面权重税使核心多16309拍，未得到净省。all-one反例63103→24861说明较密输入可能适配；real_2则6410→8655。该负结果只停16K/三plane这个布局。

功能：14旧fixture（真实四角/四内部、zero/one/corner、正负极值/zero_factor）×两mode×双背压×两次无reset命令，112runs/430080输出全绿。每条mode14另与旧最终packed15逐记录验证896个标量指标和56状态数组完全一致，见[checks](checks.json)。只读[独立验证脚本](../verify.py)重算53760个gold与本表示，未把统计变成周期。配置与输出data/address保持在TB中实际检查；错误metadata/运行中改cfg/无reset换mode不属本合同。

本叶未新训练/消费者/AEE/整帧，未做Fmax/PPA。source_allow/weight_allow门控已配置本地flop/mux读，不是DDR/NoC响应模型。`weight_words`混合不同宽度格式，不能直接作为字节或能耗。所有资源口、半字写使能和组合路径详见[resource_contract](resource_contract.json)；bitstream/LUT/count是SV状态，TB不提供动态中间值。
