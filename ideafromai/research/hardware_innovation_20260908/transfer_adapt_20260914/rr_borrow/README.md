缺失的 **RR＋既有consumer64链借用强控制已实际实现并测完**。128起64个真实tile冷启动无BP，原RR三P10为814321拍，原RR模数四P8为787603拍，忠实迁入的四P13借链为 **786421拍**。借链相对模数再省1182拍（0.1501%），相对三P10省27900拍（3.4262%）。原两mode全部176条旧命令的所有旧计数逐项复现。

忠实迁移还发现一个真实仲裁条件失败，并完成一次定向适配：异质邻接双tile使consumer与producer同时申请宽链，固定consumer优先的借链比modular慢112拍；增加一位宽链组RR后由36283降到36110拍，转为比modular快61拍。没有以该负结果否定借链家族，也没有把普通RR或通用功能单元共享当成新意。

|同资源模式|实现与作用|
|---|---|
|mode1|原普通RR＋证明有效的三P10；范围超出时精确回退dualP13|
|mode2|原普通RR＋四P low8/high5，实际修复与末尾规范化|
|mode3|忠实迁移四P13，用已有consumer64主求和链；consumer固定优先|
|mode4|只改宽链冲突策略：consumer与producer组轮转；组内仍是原两context资源RR|

根目录保留忠实迁移；[adapt_rr/interleave_stream.sv](adapt_rr/interleave_stream.sv) 同时支持四mode，已在该同一模块内重跑原三mode的12个64流命令，逐项复现根目录结果。mode4增加一位控制状态，全部四mode在适配模块中共担这一位与选择逻辑；没有增加数据ALU、乘法器、存储容量或端口。

|128起64tile|mode1 三P10|mode2 模数四P8|mode3 四P13借链|mode4 宽链组RR|
|---|---:|---:|---:|---:|
|冷，无BP|814321|787603|786421|786421|
|暖，无BP|812473|785755|784573|784573|
|冷，有BP|884397|858814|857282|857282|
|暖，有BP|882629|856904|855357|855357|

这64tile内consumer/borrow冲突实测为0，因此mode4不产生额外收益；它的条件适配效果由下面的异质pair验证，不能拿本64表声称改好了真实大流的宽链争用。两段小真实流也完整通过：159起3tile冷无BP为42284／41090／41036／41036；19197起3tile为32045／31884／31832／31832。

实际共享由[wide_phase_alu.sv](wide_phase_alu.sv)唯一8条64bit主求和carry链实现。消费者原来的`wide_hold + add_rhs`表达式已移出，改成operand/result总线；顶层在消费者与获准context之间先选操作数，之后只执行这一份ALU。借用时切断13/26/39位置carry，更新四个signed13字段，仍然装入原52bit z字。consumer仍有原8个32×32乘法器，producer仍只有原8个19×13乘法器和8条32bit链；FP32→J20及最终RNE/饱和的原有逻辑及其舍入增量保持原样。

每context保持原source、520B z、完整15360B psum、qblock、z_hold和correction状态。z仍每拍全局至多一份416bit服务，一个共同地址/银行读表达式；新增wide资源请求不会增加z口。mode3/4的ZADD原子申请`z+wide`，mode1/2的ZADD和所有BASE_MAC/修复/规范化原子申请`z+producer32`。只有收到整体grant才读取或提交；两context结果广播不会授权未获grant的一方写回。消费者在ADD_BIAS/ADD_IDENTITY无wide grant时也完整保持主和与状态。资源账见[resource_contract.json](resource_contract.json)。

|64冷无BP的实际服务量|mode1|mode2|mode3／4|
|---|---:|---:|---:|
|Q1更新issue|76424|60113|60113|
|Q2 MAC|198720|198720|198720|
|repair issue|0|0|0|
|NORMALIZE_ADD|0|640|0|
|producer32 grant，含864 proof|276008|260337|199584|
|借用wide grant|0|0|60113|
|全部wide grant，含consumer|61440|61440|121553|
|z grant|354768|323426|322146|
|context资源冲突拒绝|180553|174163|173079|
|借用ZADD的context RR拒绝|0|0|16691|
|consumer/borrow冲突|0|0|0|
|consumer join等待|497336|470618|469436|

这里把60113次Q1更新从producer32搬到已有consumer64，并没有删掉这些算术服务。与mode2相比，本64流主要删掉640次规范化ADD及640次READ，z grant少1280次；Q2仍198720次，共同z许可和完整后端使收益只有0.1501%。两份context活动周期总和由1265876降到1263512，其2364拍差包含仲裁变化，不能与wrapper时间相加。ordinary RR仍只是公共控制。

异质压力pair是6460与6461两个相邻tile，按原生source地址读取一个96×4×6的小输入窗：前四列全零、后两列全一。两个tile真实共享中间两列，先完成的零tile启动consumer时，后一个tile仍有Q1待更新。它不是两个互不相干的gold拼接；raw由完整K864、真实Q1/Q2重新计算，FP32 identity为零后仍经过原J20/I24消费者。模数臂在此真实执行80次repair与20次normalize，借链臂有5740次借用。

|异质双tile完整经过时间|mode1|mode2|mode3 固定consumer优先|mode4 宽链组RR|
|---|---:|---:|---:|---:|
|冷，无BP|53161|36171|36283|36110|
|暖，无BP|51313|34323|34435|34262|
|冷，有BP|55374|38214|38519|38214|
|暖，有BP|53377|36209|36502|36217|

无BP冷启动，mode3被consumer拒绝299拍；mode4把producer拒绝降到126拍，并实际支付126拍consumer等待。带BP冷启动是486→169拍producer拒绝，同时支付169拍consumer等待。两者总流量相同，受益来自等待发生的位置与可重叠区间，不能把减少的producer拒绝全部当净收益。mode4修复了该无BP失败条件、BP冷启动追平，但BP暖启动仍比mode2慢8拍；保留这一余差，不声称所有输入或相位下借链均更优。

功能与账目共 **508个RTL命令，raw、J20、I24各9707520个值全部通过**，来自396个常规/跨mode/同模块控制命令、96个逐拍断言命令、16个统一pair比较命令。包含16个既有fixture（真实、全零、全一、padding poison、正负极值、身份转换与乘积ties/饱和边界）、双context压力、跨mode冷暖重启、159/19197跨row三tile及64流。小测和断言通过后才执行各自64阶段。

逐拍断言检查grant只给eligible、共同资源不重叠、proof与context互斥、持续eligible的context不连续两次失败；拒绝时state、k/fp/zrow、pending、acc、holding、correction、全部z与当前psum都冻结，未授权z读总线为0。另检查wide与z绑定、consumer/producer不同时使用唯一宽链。mode4额外检查被拒consumer状态与完整512bit主和保持；断言副本仅用于验证，不计入候选资源。主接口TB检查输入请求保持、输出holding、tile/row/last、480beat退休及完整raw/J/I24。

[finalize.py](finalize.py)逐项比较176条原RR旧收据及12条适配模块内64控制收据，总188条旧控制完全复现；从67个去重native tile重新计算raw/J/I24，各257280值吻合。也按原生source重算Q1合并、Q2、repair/normalize与所有服务义务，包含异质pair，核对以下守恒式：

```
producer32_grants = proof + Q2_MAC + repair + normalize + (mode1/2 ? Q1 : 0)
wide_grants       = consumer_add + borrow_grants
borrow_grants     = mode3/4 ? Q1 : 0
z_grants          = z_vector_reads + z_scalar_reads + z_writes
core_arbitration  = context_conflict + borrow_consumer_stalls
consumer_cycles   = 3385*tiles + join_wait + output_stalls + consumer_wide_waits
window_cycles     = consumer_cycles + tiles + floor(tiles/2)
total_cycles      = window + launch + static + parameter_stalls
                  + source_load + origin + source_load_stalls + 1
```

ordinary RR、已有宽ALU借用、子字段carry切断或consumer/producer公平仲裁都不能单独作本项目新颖性标题。本轮提供的是缺失强控制、准确的失败条件、一次小幅但可复现的适配，支持继续做共同资源下的表示/调度组合研究；不能用本结果宣布modular或borrow整个家族无效。是否有独立论文新意仍未建立。物理单端口SRAM、面积/频率/能耗、完整19200tile与整网收益均未验证。全部是Verilator4.028组件周期，没有EDA、新训练、生产变更或commit。

结果：[final_checks.json](final_checks.json)、[results_64.json](results_64.json)、[adapt_rr/results_64.json](adapt_rr/results_64.json)、[adapt_rr/results_controls.json](adapt_rr/results_controls.json)、[adapt_rr/pair_results.json](adapt_rr/pair_results.json)。总计已写入final_checks。

在本目录复现，Python均用`/opt/anaconda3/bin/python3.12`，每个RTL构建均为Verilator4.028 `-Wall --cc --exe`再`make`：

```
/opt/anaconda3/bin/python3.12 run.py
/opt/anaconda3/bin/python3.12 run_stream.py
/opt/anaconda3/bin/python3.12 audit.py
/opt/anaconda3/bin/python3.12 run_extra.py
/opt/anaconda3/bin/python3.12 verify.py
/opt/anaconda3/bin/python3.12 run_stream.py --stage 64
/opt/anaconda3/bin/python3.12 build_adapt.py
cd adapt_rr
/opt/anaconda3/bin/python3.12 run.py
/opt/anaconda3/bin/python3.12 run_stream.py
/opt/anaconda3/bin/python3.12 audit.py
/opt/anaconda3/bin/python3.12 verify.py
/opt/anaconda3/bin/python3.12 run_stream.py --stage 64
cd ..
/opt/anaconda3/bin/python3.12 run_extra.py --adapt
/opt/anaconda3/bin/python3.12 finalize.py
```
