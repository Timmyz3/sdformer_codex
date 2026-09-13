# C4 新核独立静态与记录审阅

2026-09-13。审阅范围：[c4_execution.sv](../rtl/c4_execution.sv)全部415行、[tb.cpp](../rtl/tb.cpp)全部78行、[PLAN](../rtl/PLAN.md)、最终[REPORT](../rtl/REPORT.md)与[资源合同](../rtl/resource_contract.json)、驱动与账本公式，以及最终[648条结果](../rtl/results.json)。仅涵盖封口的modes3/4/5，后续兄弟目录mode6不继承此结论。未修改对方文件、未重跑RTL、未执行GPU/EDA。独立脚本只读结果核对事务恒等式与汇总，11,016项检查通过，产物见[result_record_audit.json](result_record_audit.json)。RTL代理自己的原生几何/支持码公式核验见[ledger_checks.json](../rtl/ledger_checks.json)，两者不要合并为两次功能复现。

**结论：本范围无阻断发现；mode4支持码归约没有击败同驻留权限的mode5。** 648命令记录、2,488,320个标量输出由受审TB与完整卷积gold逐值比较；包括重复支持码、四项signed16极值、真实/边界/全零/全一/全mask/毒值padding及背压。审阅不等于形式证明、全网bittrue或PPA通过。

## 算术、覆盖与完成条件

- **全C96/N96/T10。** C4的`cp=0,2,…,46`，`c=2cp+{0,1,2,3}`覆盖96个通道；native_xy覆盖16个源位置。48bit目的枚举为12个O8组×4个输出位置；psum地址`(og*4+p)*10+t`在0…479，输出480拍×8 lane=3840值。没有以C8局部块代替全核。
- **精确整数范围。** 输入为二值，W为signed16。两项和signed17范围[-65536,65534]，四项和signed18范围[-131072,131068]。SV在32位公共加法链上显式符号扩展，再截至可精确容纳的17/18位。每输出最多864项，保守绝对和≤864×32768=28,311,552，signed32 psum充分；不同累加次序不溢出，不引入新的舍入边界。这里是Q16整数线性叶，不是整网RNE模拟。
- **按需码确实由RTL生成。** 四个10bit源词生成十个4bit码；15bit集合只标识实际非零码，不含十五个预计算和。每码付一拍首W复制、`popcount(k)-1`拍构造，再分别更新匹配t的psum；onehot直接复制，零码不构造。stale W暂存不会被未活动通道引用，因为码位只能来自`OR_t source`已请求的W。最后一次W加载后下一状态才读暂存，无同拍读新值假设。
- **源退休及psum义务分离。** `any_group=OR_og block_live[og,cg]`在读源前决定整Cg是否跳过；部分O8块死不误删仍有消费者的源。目的pending虽在选中时清位，但串行FSM仍先完成该目的所有pattern/t的读改写，才检查下一目的或移动源。因此没有未完成psum写回时提前覆写source/scratch。所有Cg结束后才drain，未实现非因果T10 PSN提前执行。
- **背压与重启。** source/weight allow为0时状态及待服务项保留；DRAIN_SEND只在ready时移动row，result_data寄存并保持。TB验证数据和地址稳定。每进程两命令不reset，验证的是同模式同配置重启；未测试无reset换mode/换mask、运行中取消或非法配置地址。合法合同是IDLE配置完整合法范围，start时配置已结束。

## 同资源与计费判断

三模式使用同一SV的最大资源权限：8条signed32数据加法链；source每拍单10bit词；W为8个signed16 bank同地址单读（合128bit）；psum为8个signed32 bank，读、加写分不同状态；输出256bit。码生成读取已寄存的40bit，不能计作额外source读口。block_live的并行OR和48bit目的/15bit码选择是实际组合逻辑，其延迟尚无物理结果。

共同主存状态为source 1,920B、W 165,888B、psum 15,360B、block mask 36B，另有原hold/psum寄存。新增四源40bit、四W512bit、单scratch144bit、15+4+4bit metadata、两个声明为integer的索引64bit及mode扩展1bit，合784bit功能状态，诊断计数器另32bit，与资源合同一致。mode4/5共享四源/四W暂存与一次目的枚举，mode5有相同首对/次对切换权限，无强塞空对扫描；因此主净差分4对5公平。mode5对3的控制循环收益单列。mode3不用新增状态，不代表独立综合后面积相同；同一RTL预算也不能推出同Fmax、能耗或SRAM映射。

本核读的是预装本地memory。allow表示服务背压，并非外部多 outstanding 请求/响应网络；DRAM、缓存一致性及跨tile流水未模拟。冷配置由TB实际执行：1536 source +288 mask +1 origin +10368 W = **12,193拍一次**。JSON在两条命令均重复打印该配置字段，但第二命令实际不重装。核心cycles含清零480拍、真实服务、控制、背压、最后480拍输出发送与480拍drain读，排除IDLE的start接受边沿。故冷进程总账为一次12193加两命令核心及两次start接受边沿，不能给warm命令再加12193。

SUMMARY的`cycles_with_fresh_source_origin=cycles+1537×tiles`是新tile源与origin装入的显式附加账，W/mask驻留；它不是TB中第二命令真的换源。独立记录审阅确认各命令`psum_read=psum_write=update_issues+480`；source/W请求在3/4/5完全相同；背压增量恰等三类stall相加；两次同模式命令计数一致。TB注释称global-time背压，其n实际上每命令归零，宜称“命令内固定时间波形”，不影响同模式对照的服务授权。

## 实际差分及其含义

下表为同一arm、同函数、同资源，真实8tiles的首命令核心周期总和；每项完整输出，均含清零/控制/末输出。冷装入和新源附加账三模式相同。

|mask arm|mode3 pair|mode5 C4驻留pair|mode4 C4归约|4相对5增时|背压4相对5增时|
|---|---:|---:|---:|---:|---:|
|dense|415024|391120|422008|7.8973%|7.6212%|
|block magnitude25|323132|303052|326364|7.6924%|7.4185%|
|Cin magnitude25|320116|302380|325960|7.7981%|7.4822%|
|Cin fullcost25|312796|294316|316660|7.5918%|7.3364%|
|mixed retirement25|321196|300848|323513|7.5337%|7.2860%|

dense的psum更新从73,428降至67,068（少6,360个八lane读改写），但构造加法从2,496升至7,572，并付40,596拍pattern复制，最终慢30,888拍。真正费用缺口是有限T10窗口中的实际支持数与重建税；“归约减少psum”本身成立，但不是完整job收益。mode5保留为更强A，停止这个固定单scratch逐码构造点；不由此否定Phi/Prosperity的前缀、压缩PWP或更完整排程。

整体质量另见[data/diverse10.json](../data/diverse10.json)：336块新校准与diverse10帧/序列不重叠；两种整Cin方案确实使source 9600→7200，但混合72块选择未退休完整Cg，source仍9600。该结构差异是结果。其AEE来自Q16 W解码后的原TF32浮点消费者，不是本RTL全网bittrue；不以“全低于NB0”替代同质量速度对照或825帧结论。
