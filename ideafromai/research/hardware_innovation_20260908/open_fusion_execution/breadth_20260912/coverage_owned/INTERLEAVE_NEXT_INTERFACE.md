# 下一接口：同96RF的源程序与PED交织（设计，未执行）

**先定义B。** 对同一组已冻结参数、相同真实输入和相同终点，先重跑最强普通串行：完整CSE原调度源程序，原resident PED P2/H32，以及既有 `new_interface_selection/resident_latent/run.py` 的普通P1留Z（R24/H48、R32/H32），按实测最低总服务选分母。dense、34、lifting都获得相同重排、16RF约束下重算/溢写权限；强原61/29/13临时RF调度保留，不能用较差DFS强制替换。重算保留必要RNE与饱和，增加的指令、SR/SW、完整signed48 spill（每向量48B）和ROM容量都付费；装不进512×128 ROM就记录该固定控制不可行，不扩大候选容量。新参数均重新执行，不能拼旧源时延、旧PED表或825。本文只设计下一次CPU接口，不改变当前matched局部绑定。

61/29只是当前whole-CSE/last_use_pressure调度的峰值，**不是dense/34的RF下界**。下一次仅固定一种小RF普通编译：逐输出行、仅行内CSE、至多13个源临时RF，必要时重新加载I24/计算并计全费用，不扫容量或策略。第一试仅用重算；当前源RTL未有通用48bit临时值spill指令，若后来增添须另列访存/ISA支持，不能用24bit截断代替。普通dense可选“小RF源＋同交织”，也可选“原61RF串行”或“原源＋较小消费者分组”；以同边界最低实测服务竞争，不能用13对61本身当X。

**现有代码能继承什么。** `stage_20260912/hardware/integrated.py` 继承的 `preview_sn2_chain/machine.py` 只有一个 `rf[96,8]`、一个ready表和一个pending写回日历，FP32与signed48共用float64容器；不是FP/INT各一套RF。容器仅保存确切整数/显式舍入FP32值，不代表64bit硬件。`source_program.run()` 和 `resident_mac()` 目前都整段串行，必须改成可暂停的操作流，所有操作仍经同一 `advance()`，单issue、2R/1W RF、SR64/SW64、CR256与单写回冲突不可各自计时后取max。禁止另造source_rf/consumer_rf或第二套ready/pending。

**直接16+80草图尚不合法。** NEW编译临时RF峰值dense61、34为29、lifting13，另有门RF95；resident PED还用RF88..91的4个供数向量。lifting直接拼接需13+1+80+4=98RF，不是96。下一次固定一个可实现的重排：每个k先收集位置ip0的T10，再收集ip1，只有两个供数RF；各位置完成所有H8组后才覆盖。P2的80个累加器持续驻留，增加的CR重访/调度必须真实执行；普通臂也能选择同供数方式或原4RF更快方式。

| 活值 | 物理RF | 可读/释放时刻 |
|---|---|---|
| P2×T10×H32 Acc48 | 0..79 | 各MAC真实写回后可读；全部K、原RNE/sat/bias与SW提交后逐组释放 |
| lifting源13个临时 | 80..92 | 编译器逐值last-use分配；load/ISOURCE两槽写回后可读，最后使用及待写回结束才复用 |
| PED当前一个位置的T10 | 93..94 | 全部值来自实际SR64响应、ILOAD写回后可读；该位置所有H8消费结束才复用 |
| 源门结果 | 95 | ISOURCE写回后由原collector读取；读完当前t才覆盖 |

源门另有既有16B打包collector，PED沿用24B gather/3B scalar collector及单SR/CR响应锁存；这些有限状态需单列并赋所有权，不把Python `values`/`words` 当免费跨上下文缓存。SR响应及跨字24bit收集期间不能被另一流偷换，加载数据完成后才允许切换其所有者；实际同槽SR/SW/CR与单条op的合法组合可用。源算术可填PED访存/RAW空槽，PED计算可填源访存/RAW空槽；同一issue或写回到期槽不能重叠，不能删必要MAC/RNE来制造空槽。只按ready与资源许可发射，不把原helper内整段 `drain()` 当全局阶段屏障，也不能忽略其真实依赖。

**最小合法数据边界。** 源输出是sn1门；PED需要 `Conv2 U→F→BN2常量+原I24→sat24` 后的updated，二者之间还有完整K864 preview与非因果T10 sn2。明确拒绝“源一出门就把未生成updated传给PED”。第一试验固定两个现有空间工作块A/B：在同一Machine先真实执行A的源、preview/sn2及Conv2/merge，到一个P2的updated已写入SRAM，再将该P2完整PED U→原RNE→V/bias/sat与B的源halo交织。B可以是另一既有corner/interior窗口，**这是同帧空间工作块，不是跨推理帧**。共同终点是A该P2的PED输出和B完整sn1门；共同前缀、双方冷填、B raw DMA及最终排空全部计费。后续B preview不纳入这次有界终点，不能称全层流水。

A沿用I[0,5760)、LAT16[16384,17344)、UPDATED[20480,26240)、PED_U[32768,34688)、PED_V[40960,46720)；B单像素raw缓冲为[90112,92992)，B sn1门重定位到[98304,121536)（最大11×11×96×2B），不覆盖A UPDATED或PROJ、也不碰122880起metadata。B源只用ROM/整数ALU，不替换A的PED系数；其后preview需等待A完成再替换系数和复用低地址。串行B采用同地址/输入终点，以免把额外物化税仅给普通臂。

**实现与验收门。** 对现有CPU ISA只需重映射ILOAD/ISOURCE/IMAC_INDEX和调度，不需新增数学指令；隔离source RTL尚无PED MAC/FP消费者，不能据CPU方案宣称无需RTL扩展。先逐次操作核RF活值/类型标签、真实ready、两源读/单写回、SR响应所有权及state区间；原35个实际norm、gate和PED全部输出须与各自新参数独立gold逐值一致。再报同一工作量的完整槽数/端口字节/溢写/重算及最强普通差额。若两RF供数税或串行P1留Z更快，停止此固定布局；当前没有测得服务收益、新机制名称或质量结论。

**后续已补普通编译控制，交织仍未跑。** 原未量化dense的固定两链CSD虽只需13工作RF＋门RF，但524字超ROM；新固定两项量化dense用原封不动的同生成器得到229字，在原RTL实际通过，ready472周期，比同函数完整CSE319慢47.9624%，见[低RF结果](../source_constant_probe/dense_low_state/README.md)。因此该新dense函数已有合法13＋1RF普通程序，配2供数RF可纳入上述容量；它的量化AEE仍须单独评估，不能替代主matched函数的质量或宣称交织已净赢。
