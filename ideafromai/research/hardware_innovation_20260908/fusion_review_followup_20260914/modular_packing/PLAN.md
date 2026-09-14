本项在新目录实现完整原生 Q1/Q2→真实 FP32 identity→J20→I24，旧目录只读。

A 是子字打包及精确模数表示。B 是同源四个空间位置仍须按组更新。候选 X 是按实际进/借位义务启动精确高位修复，用现有 32bit 链容纳四个低位更新；这是执行候选，不预先认定新颖性。最近邻包括 ULPPACK、Bit Fusion、低精度累加；本项不进行位宽扫描或质量函数改变。

三臂采用同一源码、8×32bit producer carry 链、8个19×13乘法表达式、8bank×10row×52bit z（520B、416bit向量口）、完整 psum 和原独立 I24 后端。不会借消费者 64bit 链。source、Q1/Q2、局部窗、配置、消费者缓冲与输出背压权限相同。

|模式|更新接口|精确表示|
|---|---|---|
|0，诊断|两P×13bit，两个组|四个原生 signed13 z|
|1，强控制|前三P×10bit占30bit；第四P单独一组|每次结果 sign-extend 到对应13bit z字段；静态范围不合时精确回退mode0|
|2，候选|四P×signed8低位；溢出时修复四个5bit high|z = signed8(low) + 256×signed5(high)，Q1后付费规范化为标准13bit|

所有模式在冷配置 Q1 的每个已付费 beat 中，用现有八条加法链按双13bit段累加各 rank 的正和/负和，实计864次 proof issue。使用独立26B正负界寄存器，三臂共担；没有额外高位加法器。最后得到每rank全二值源/任意遍历前缀的安全区间，只有正和≤511且负和≥−512才启用三P10。真实Q1预期正界 `[248,119,290,95,222,338,236,263]`、负界 `[-419,-361,-211,-352,-367,-259,-236,-399]`；RTL自行计算，不由gold喂入。范围证明与Q1装载同拍重叠，额外配置beat为0，但算术占用明确为864次且无并发compute。warm相同参数不重算proof。

mode2低位链在8/16/24位断carry，逐byte用输入/结果符号判断signed overflow；仅同号输入变号结果产生义务。delta是旧低位为负时−1、否则+1，inactive字段无义务。检测不含另一加法链。任一lane/P有义务，初次写回low后进入额外 REPAIR_READ、REPAIR_ADD；读完整416bit z，再用同八条32bit链的四个5bit分段修复high并整字写回。每次修复多付2拍、1向量读、1向量写、1 ALU issue，逐字段义务另计，不能有32个隐藏high adders。

Q1全部结束后，mode2对10个T行各执行 NORMALIZE_READ、NORMALIZE_ADD：共享链按四5bit计算 h' = h − low[7]，保存标准13bit `{h',low}`。这一步固定多付20拍、10次z读、10次z写、10次ALU；随后原ZSCAN/Q2只需位拼接，不能在每个MAC前暗放高位decoder加法器。控制具有相同硬件能力但无需执行此无用转换。signed3 Q1、K864和二值源保证任意前缀z在[-3456,2592]，centered high5足以覆盖。

先使用原16fixture（含8真实、零/一、padding、正负极值、tie/saturation、FP边界），三模式、有/无BP、冷/暖；raw/J/I24逐值检查。角落Q1超10bit时mode1必须fallback并与mode0一致。再独立检查范围、低位修复和计数。只有真实8块候选相对三P10有明显净收益才继续64连续；否则记录失败并停止。此单context试验不替代D3强RR，后续多context组合仍须保留RR。
