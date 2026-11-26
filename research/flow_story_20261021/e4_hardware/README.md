# E4：Q6 整数核与有限供数证据（2026-10-21）

本轮新增两项实证：**Q6 协议已真实通过 RTL 仿真和同约束 DC；四个完整仿射算子域的有限地址调度仍能兑现部分低位取消，但 K96 的 Q6 比普通并行 Q8 更慢。** 没有 GPU 前向、训练或 GT 读取，未改旧代码。下列服务结果不是整网 RTL 周期或芯片速度。

## 1. 实际 Q6 数据通路

[新 RTL](rtl/plane_pe.sv)由 R30 `plane_pe.sv` 复制另建，原 `mode_q8` 改为 `start_plane∈{0,2,4}`，分别执行 Q8/Q6/Q4；非法起始 plane 拒绝。每拍仍为 128 binary inputs × 16 outputs、2048-bit weight-plane、int32 accumulator，bit7 系数−128。普通对照仍为 binary16 × signed8 weight、16 outputs；激活接口不等宽。不是新增零组跳过或多上下文 PE。

[Verilator 原收据](results/RTL_INTEGER_V1.json)：261 个 row-group，165 个合成用例和96个旧真实操作数×精度事务。11种 K=1…3072、0/1/random 激活、负数/极值、所有 precision、有/无输入气泡及输出背压均核验。每种引擎/停顿组合比较4,176个int32，四种组合共16,704个；11个错误协议被拒绝。无停顿 binary 合计8,280个数据 beat、8,281周期；普通合计10,569个beat、10,570周期。测试流比例不可当部署速度比。

其中32组源操作数来自R30最终权重桥接，每组分别执行Q8/Q6/Q4。**这些是旧真实输入与旧权重的新精度协议测试；Q6整数答案由独立有符号整除golden产生，不是R27恢复Q6模型的真实层输出或新的全模型精度桥接。** 三种precision合计1,536个不同事务输出；其中Q6是512个派生整数值，不冒充512个新GPU输出。

[本次新 DC 原报告](results/dc_q6/area.rpt)、[摘要](results/dc_q6/summary.json)：TT28/0.9V/25°C，相同4ns、0.05ns uncertainty/transition、0.2ns I/O delay、0.01pF load，2 cores。

|核|本次cell area（库单位）|worst setup slack|
|---|---:|---:|
|新增Q8/Q6/Q4 binary plane核|9,930.829984|+1.78569ns|
|旧R30 Q4/Q8核，来源比较|9,930.829984|+1.78296ns|
|旧R30普通并行核，来源比较|9,890.062045|+2.00306ns|

新核实际重跑DC；报告面积恰与旧核相同，不是沿用旧结果。新核相对旧普通核面积约+0.4122%。这是局部整数资源参照，没有合成完整SRAM、供数、FP32、BN或PSN；没有功耗、布局布线和门级结果。原工具warning保存在`results/dc_q6/run.log`，不称零warning。

## 2. 地址级模型的范围与资源

[supply_replay.cpp](supply_replay.cpp) 使用冻结R27 inventory的实际几何，完整T10、H/W、全部通道和真实尾区，执行16个臂：四个算子分别跑ordinary Q8 parallel16、Q8 bitplane128、完整9×9域Q6/真尾Q8、完整9×9域Q4/真尾Q8。

- hot conv：240×320、K864、Cout96；3×3 halo/zero padding。
- stage2 FC1：30×40、K384、Cout1536。
- stage2 FC2：30×40、K1536、Cout384。
- stage0 FC1：120×160、K96、Cout384。

每个臂从其输入二值张量已合法产生开始，到该仿射算子及其后完整current-BN服务结束。**FC1/FC2分别建模；不能直接把两个数字相加当完整MLP延迟，因为中间PSN的全T生成尚未接入。** 相同限制适用于conv前置PSN和后续残差。没有用晚层信息控制早层，没有新评分器。

固定资源：4MiB驻留输入；两512KiB输入槽、两512KiB输出槽；两1MiB weight SRAM槽；32KiB weight RF；256KiB accumulator RF、256KiB stats RF。沿旧声明容量约8.53125MiB；当前单行PE只使用部分accumulator容量，不把空余RF算额外吞吐。

- 输入槽共用16×32-bit、1RW bank组（64B/tick）；输出槽同样共用一个16-bank组。两个槽没有免费倍增端口。
- weight SRAM每拍256B；weight RF为64×32-bit、1RW bank组，容量32KiB。虽然容量可分半，PE每拍读满全部64 banks，**RF refill与计算串行**，不假定另一个半槽可免费同时写。
- 共享DMA64B/tick，每个事务启动32tick。输出采用最多128行的完整Cout batch，写回连续全Cout布局，不假定免费逐64B scatter。
- 每个batch按16输出块，weight RF容纳一个完整块。conv块14,336B；stage2 FC2块24,576B，占超过半个RF，均在容量内。全部8权重plane仍需装载；Q6/Q4省RF请求，并不自动省weight SRAM/refill或外访。
- 真实halo源行搬入输入槽，packed行放在halo之后，避免覆盖还待消费的源。conv源输入按bank地址读取，真实边界不读虚构像素；padding K槽在硬件零填充，源Cin没有减少。
- 双缓冲DMA可与PE交叠，但输入DMA写占用共享输入banks时，PE暂停；输出DMA读占用输出banks时，退休写回暂停。输入gather/packing采用保守串行安排，避免免费读写端口。
- 每个PE beat生成实际RF及activation地址并校验范围、对齐、请求plane。权重256B对齐，对应64个不同bank；串行激活16B对应4个bank，普通16bit激活读取一个32bit word。完整beat序列都推进，输出仅保留有界trace前缀及每precision的样例beat，其他以总计和FNV地址摘要绑定。

这是**确定的、保守的可执行资源调度模型**，不是最优调度、不是逐tick全机RTL。完整源/打包数据值未在这个模型内重算；位移/转置packing逻辑、FP32标量资源及全部SRAM宏面积尚未DC闭合。旧R27聚合账与本模型调度不同，不能把本模型数字直接替换少数旧行后冒称新的整网精确周期。

## 3. BN、数值顺序和消费者义务

不继承未证明的producer-fused mean：本轮为每个臂付完整的**独立mean、variance、normalization三遍**，含全域DDR往返、16路向量服务、统计finalize、有限I/O槽和屏障。均值全部完成后才做方差，方差全部完成后才归一化。浮点归约树与GPU的数值一致性未证明；三个pass保证账本不把必要重读省掉，并不证明PyTorch逐位等价。

BN服务采用相同的保守向量资源假设：mean每16元素1tick、variance2tick、normalization3tick（含读/变换/写），统计finalize另计。浮点递归延迟及kernel schedule没有综合校准，所以这些是服务合同，不能称BN RTL实测。未来若用不同归约顺序，应对所有方法同等进行实际质量核验。网络PSN会混合完整T，不能在同一组全域BN未完成时提前宣称下游输出可用。

## 4. 完整域结果与负结果

[完整16臂结果](results/SUPPLY_REPLAY_V1.json)、[同资源摘要](results/SUMMARY_V1.json)。每个算子的Q8基线允许在普通parallel16与bitplane128中选服务较短者；当前四者均选parallel16（stage2两FC两者相同）。单位为上述抽象机器tick。

|完整算子域|strong Q8：仿射+BN|Q6：仿射+BN|Q6减少|Q4：仿射+BN|Q4减少|
|---|---:|---:|---:|---:|---:|
|hot conv K864/Cout96|311,970,246|259,342,002|16.8696%|197,425,968|36.7164%|
|stage2 FC1|43,110,614|37,512,908|12.9845%|31,915,202|25.9690%|
|stage2 FC2|31,649,378|26,052,674|17.6835%|20,455,970|35.3669%|
|stage0 FC1 K96|87,976,610|88,669,264|**−0.7873%**|80,079,722|8.9761%|

Hot conv的core beat减少Q6/Q4为21.1791%/46.0619%，实际有限仿射调度减少19.8040%/43.1029%，加BN后降到上表，不能只报core。conv保留761,640个源gather bank冲突额外tick；输入DMA造成约1.148M个PE阻塞tick。stage2 FC1还有约1.11M输出DMA阻塞tick。完整计数均在原结果中。

K96普通并行只需6个beat，而Q8串行需8；Q6完整区需6，真尾仍需8，加padding/refill后比普通Q8慢。**这是真负结果，不能用较慢Q8串行分母隐藏。** 未来可给K96保留普通执行或重新设计匹配数据路，但不能假称当前Q6每层都快。

conv Q8 DMA1,548,140,928B，Q6/Q4均1,548,144,000B，因串行K-padding略增；其余候选也不产生外访下降。本轮支持部分整数计算取消能在有限供数下留下局部服务优势，不能主张访存压缩或整体网络净收益。

这里的Q4是**共享Q8 scale、保留高四位的nested截断**，不是E0的独立scale/RNE4。E0双权重表示、双scale、分域切换和新增weight加载尚未放入本模型，因此不得把这些Q4费用当作E0的净费用，也不得据此宣称胜过RNE强普通对照。

## 5. 可复算核查与下一步

[check_results.py](check_results.py)独立按冻结inventory公式复算全部16臂的完整行/真尾、beat、源halo字节、packing、RF全部8plane refill、scale和容量；核验保存prefix的DMA独占、packing和accepted beat的bank所有权，以及真实RTL事务的独立beat总数和新DC原报告。[作者核查收据](results/ACCOUNTING_CHECKS_V1.json)全通过。它是作者内部不同路径计算，不冒充独立代理评审。随后 root 进行了[独立内部复核](../independent_review/E4_ROOT_REVIEW_V1.md)，全部16臂的独立beat重算吻合，并核对了资源边界与新DC原件。

同目录`reproduce.sh`可在新输出目录重建RTL和模型；`check_results.py`另行核验当前冻结原件。DC需按脚本说明在空目录另跑。当前版本源、实际输入、工具结果和DC库来源通过[冻结清单](FREEZE_V1.json)绑定。

下一步以E1实际精度定位的家族为准；只将存活动作接到其真实trace。需补动态位图/预测费用、实际层间PSN/残差依赖和全网统一时序；RNE若存活先补双表示供数。先关闭这些局部缺口，不扩大到完整top或把当前部分结果拼成整网加速。

草稿中曾发现双槽端口与halo暂存布局不够明确，已在封存前收紧并重新执行；`results/draft_shared_port_bug/NOT_VALID.json`明确为无效草稿，不能引用。一次检查脚本语法拼写已修正，不影响任何GPU或数值模型数据。
