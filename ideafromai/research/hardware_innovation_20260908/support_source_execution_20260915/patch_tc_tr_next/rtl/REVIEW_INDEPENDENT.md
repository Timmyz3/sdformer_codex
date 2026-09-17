# patch U→Z→V/A→gate 独立审阅

审阅状态：数值/接口静态审阅已完成；资源表达式已修复，4,448命令回归仍在最后一个full文件中运行。负责人已纠正先前因旧日志尾部造成的完成误判。dense免TC_SCAN的2,224命令已完成；最终表须待资源回归实际退出后再核，本文不把正在写入的半表或部分profile的PASS当完成。

## 实际函数与参考独立性

实读本目录 README、PLAN、patch_core.sv、tb.cpp、prepare.py、profile.py 与冻结两份NPZ；另独立重聚合修复前已稳定的4,096条真实full记录。未运行RTL、重编参数、训练或EDA。

每命令是 P4×T10×K864 真实低10bit源门，第一因子 U 为 INT8；DUT真实请求source、U、V、A、tau和mask，自己形成Z、Y或Q及最终Uout。二进制中虽有gold，`memory()`只装参数和源词，gold不在DUT可请求地址中。C++从原source及参数逐层标量计算，与NumPy序列化gold互核；阶段末shadow实际Z/scratch写值，再核全部480个H8最终U/门及40个96bit有序输出，没有借旧capture的Y或门答案。

`Z=(S·U)M`，VA为 `A·(Z·V)`，AV为 `(A·Z)·V`。A只沿T10，V只沿rank→H；mask M 在同一P4跨T10保持，且中间没有RNE、截断、饱和或非线性，故两种顺序确实同一个整数函数。固定正BN gain与已冻结tau决定统一 `>=` 门方向；这不是任意BN符号或任意量化合同的通用实现。source从已有门词开始，不包含上游门的生产。

实际数组确认 ordinary为R32、8个TC全开，regional为R96、每区域24个J4 TC中12个有效；两者V都没有剩余全零rank。旧ordinary断开16rank已经裁掉，不拿空尾制造劣分母。两学生各自内部比较同函数；跨学生cycle或AEE不能互相替代。

global_rank把compact r还原为 `4*tc_map[r/4]+r%4`，U gather和V cache外读共同使用它；dense使用原rank并直接看mask。Z/Q以compact编号驻留，而V物理地址仍取global rank，未发现TC编号误当TR地址。源映射bit=t、pos=p*10+t、group/2400选八区域，与已审整数导出一致。

Z用16bit、Q/Y入口32bit、acc/scratch与最终U用48bit；已有任意归约静态界支持Z15、Q30、Y32、最终U47。RTL在Z、Y、Q写入和A/V入口检查范围，C++ int64不借相同截断掩盖溢出；最终U逐值对参考。当前测试不是任意INT8/A16矩阵的全域证明，admission限冻结参数。

## 资源与协议：已发现的两处问题及修复

初版条件 `acc-b : acc+b` 不能仅凭源码宣称物理唯一8个加减位置；Z有三处直接索引、scratch有两处直接索引，也不能用状态互斥直接当成显式单读口。已即时发根代理，并由实现负责人修改。

当前源码每lane使用48bit XOR/单carry链，U AAC、V移位加减、A乘加共享这一个数据加减位置。乘积表达式仍只有8路signed16×32；V另有8路变长移位及符号选择，地址生成、优先选择、mux和控制逻辑均不是免费硬件，未计入“8个数据ALU”的术语。未EDA，不能据此假定乘加或carry链能维持同Fmax。

当前Z/scratch均每bank只出现一个 `read_enable/read_addr/read_data` 表达式：U_READ和latent A_READ选各bank共同行，V_READ只使能v_r%8对应bank，输出域A_READ选scratch各bank共同行。写也各有一个入口，读写状态分开。修复没有增加缓存容量或新周期；最终4,448命令资源回归待实际结束。后述dense旁路仅改变其TC扫描状态。

外部是真正总128bit、最多一在途：MREQ持请求，握手后MWAIT等返回，再交ret状态。TB检查请求受阻保持、未初始化字、pending唯一、输出/done保持；返回提供者在pending接受前持有数据。没有8bank同时8×128返回的隐藏吞吐。256KiB是共同已存在存储图像的容量，起点不包括离线图像写入；冷命令读取参数费用真实，不能称外存全冷装。

Z共8×480×16=7,680B；scratch共8×480×48=23,040B，只放Y或Q；acc48B、operand32B；共同V cache768B每H8实装有效rank，source/U行cache各16B，holding/TC map/A/tau/gatepack均在源码。所有臂同预算，AV没有免费V重用或额外完整Q数组。dense保留死块/live-rank跳过、U/V cache和AV权利。每命令清除实际Z范围、重写scratch与所有gate；cache有效位start失效，无reset换源/region/mode不能借上一命令答案。

## 对照修正与最终结果

初版dense也执行并写其不用的TC map，regional每命令多24拍、ordinary多8拍。这是实质但很小的分母缺项，已要求修复。当前dense在A_GOT完成后直接设nlatent/nblocks并进入Z_CLEAR，仍使用实读mask进行死rank过滤；TC才进入TC_SCAN。TB相应要求dense没有TC译码事件。TC路径本身未改。

最终数字与记录组成待dense回归完成后补入；旧8.0465%/7.6625%暂不作为最后分母。仅ready解析可提前确定regional移除dense扫描后收益为8.0301%，BP因日历改变必须以实际新记录为准。

## 直接先验、新颖性与去留边界

直接A是ISSCC2025 CFMP：预训练tile mask决定第一因子有效TC，紧凑保存Z，再由同一TC对应TR恢复密集输出；作者也明确讨论TR片段在bank内交错。此机制在作者全文第1页CFMP段及Fig.23.2.5已有具体设计。因此“选择TC→紧凑Z→恢复TR”、普通H8字段排布及bank重排均不能作为本轮首创。[作者全文](https://arxiv.org/pdf/2512.17555)

本实现迁入了该局部执行语义与真实供数，但没有复现作者整片HAPU/LFS、mask分段流水/早停解码、原多bank/缓存规模或训练配方。当前把V重新排为每word同rank的H8，是向所有臂开放的普通接口适配；它闭合了旧参考的同bank冲突缺口，不能反过来借作者芯片PPA。

可交换线性因子、dyadic整数系数、output-stationary累加、零跳过和有界cache都是成熟手段。候选X是把非因果T10的A搬到latent域并实付Q扩位/驻留/恢复成本；本次实测允许检验这项增量，但没有因MAC减少自动得到收益。独立创新暂评 **3/10**：当前是可信的A迁移和一个有信息量的执行顺序消融，尚不足单独形成创新标题。

范围只到P4的sn2门，不含下游Conv2的18点/halo、完整网络质量、物理时序或与同函数expanded-W直接核的硬件对照。未发现会推翻同函数TC消融的参考同错；剩余这些对照限制总体架构/整网结论，不构成把CFMP家族全部停止的理由。
