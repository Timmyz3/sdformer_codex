# 本阶段算法执行结果

已完成两组合的真实 diverse10＋valid825、六个连续 PED 表示探针、四个普通权重量化对照，以及最后一个 source34 对照。没有本阶段训练、扫参、EDA 或生产修改。所有GPU任务已退出。汇总为 [stage_summary.json](stage_summary.json)，逐帧配对为 [paired_frames.csv](paired_frames.csv)；共15条候选/人口记录、1780行配对。

## 完整组合：两条均通过 valid825

| 实际新组合 | diverse10 AEE | holdout9 AEE | valid825 AEE | valid825 相对 NB0 降低 |
|---|---:|---:|---:|---:|
| ordinary 原排列 R24＋单遍 BN | 1.164806 | 1.140799 | **1.211716** | 16.16% |
| lifting activation-whitened R24＋单遍 BN | 1.191606 | 1.165713 | **1.232391** | 14.73% |
| 原 SDformerFlow 本地 NB0 | 1.454603 | 1.446426 | 1.445353 | — |

每个 valid825 结果覆盖825帧、18序列、48,152,523个有效像素；与NB0逐帧名字和valid像素数完全匹配。两组合的像素加权AEE分别为1.155485、1.173080，单独记录，不与帧平均混用。旧 original R32/CUDA 的全825分别是1.219801、1.232979，来自既有完整配对账本；新表没有相加R24与BN的单项AEE。

单项强对照仍保留：ordinary 的 R32/CUDA、R24/CUDA、R32/onepass 十帧分别1.161187、1.163094、1.166842；lifting 分别1.186586、1.190853、1.187835。每个原R32和R24首帧都在本轮复现后再换BN。R24辅助函数对既有CPU定点fixture、单遍函数对既有完整Engine以及新组合实际BN输出均核一致。

真实参数、逐帧与活动见 [combinations/run.json](combinations/run.json)。[hardware_exports/index.json](hardware_exports/index.json) 提供两轴相同首帧、原corner/interior窗口的I24/门/连续PED/BN输入输出、整域单遍BN统计以及实际部署常量。`deployed_constants.npz`包含实际R24，`student_parameters.npz`仍是原时间学生归档；不能从后者误取旧R32。完整张量留在A800本目录的ignored `captures/`，约几百MB/轴，不进入Git。

硬件代理已把两组合与现有同资源执行模型逐字段连接：四窗61,440个PED输出与signed24载荷相同，见 [final_combo_alignment](../hardware/final_combo_alignment/)。这里只继承已经确认的同一执行函数，R24 blocking和单遍BN仍是公共强底座，X=0；AEE表不构成RTL或PPA加速证据。

## 表示与量化：全部重新推理

以下均为diverse10，不继承组合的825。各臂排除既定校准帧后的9帧结论与10帧一致。

| 父网络 | 改动 | ordinary AEE | lifting AEE | 当前去留 |
|---|---|---:|---:|---|
| 各自R24＋onepass | 普通I24→Q8，步长2048 | 1.181770 | 1.213944 | 两臂过NB0，保留普通强对照 |
| 各自R24＋onepass | 同生产者 `Dg+c+Q8(r)` | **1.165295** | **1.206632** | 两臂过NB0，保留条件执行候选 |
| 各自R24＋onepass | 仅 `Dg+c`，丢弃残差 | 1.986498 | 2.009258 | 停当前固定参数表示 |
| 各自原R32/CUDA | U-only W8 | 1.177205 | 1.188024 | 两臂过NB0，保留普通权重量化强对照 |
| 各自原R32/CUDA | U-only W4 | 1.172908 | 1.190508 | 两臂过NB0，保留普通权重量化强对照 |

表示探针从真实 `updated_I24` 生成相同producer的proj门，逐帧与模块实际输出相符；没有把sn1/sn2、未来flow或另一组gate卷积权重借过来。原U/V两处RNE、sat24和bias保留。独立CPU/GPU整数fixture覆盖六臂各15,360个输出。`Dg_only`的时间/通道因式重排数值合法性另见 [REPRESENTATION_INTERFACE.md](REPRESENTATION_INTERFACE.md)，它的精度失败不等于这个代数身份失效。

预测残差的确有同码宽精度优势，但**尚未显示执行净优势**：十帧标量零码比例为4.57%/4.68%，完整P2×T10源字空率为0；普通Q8分别约2.37%/2.38%，完整源字空率约0.055%。预测残差把量化饱和数从2212→683、1765→661，改善可能来自重定心或裁剪边界，普通缩放/zero-point是下一次不同位宽比较必须有的对照。解码、减法、predicate就绪、18位D系数和临时状态均未计成硬件代价；不能从零率或字节上界推出加速。

四个权重量化臂只换原R32的U，使用低位码×q16行尺度的精确离线展开，V/e15、U的e16及bias都保持原值；未叠R24和单遍BN。actual-helper共12个fixture、184,320个输出零差。展开仅验证新函数，不是已实现低位压缩执行。W4十帧好于W8的局部次序不表示普遍单调关系，也不替代825。

原始结果见 [representations/aee/run.json](representations/aee/run.json)、[weight_controls/aee/run.json](weight_controls/aee/run.json)。权重包来源是同阶段 `weight_compensation`，未增加sign+残差GPU臂。

## 最后强对照：source34

在ordinary当前R24＋onepass父网络上，仅保留As的连续3/3/4时间组内34个原系数，exponent15、原阈值/θ及所有消费者不改。actual helper的 `As→RNE15/sat24→原predicate` 在两个真实窗口的193,920个Q24值及193,920个门上均与新CPU函数一致；因此不是只改FP shadow。

真实十帧AEE **1.686686**，排除首帧后 **1.651881**，均未过NB0。见 [source34/aee/run.json](source34/aee/run.json)。停止这一个没有恢复训练的固定遮罩；它不能否定结构稀疏家族，也不能证明lifting优于同预算训练后的稀疏source。普通34系数的硬件费用已由另一代理测出，但这次精度不允许把其省时列为可用网络收益。

## 后续可执行判断

1. 两组合当前具备完整825与真实硬件输入连接，可继续充当普通与lifting的共同强底座；还缺真正X的同资源净收益、完整RTL与ASIC证据。
2. 优先把普通U-only W4/W8计入完整消费者时间线，计入code/row-scale及中间状态，以此约束下一轮MiLo等权重侧融合。普通低位本身不作标题。
3. `Dg+Q8(r)`可保留为不同表示接口；若尝试更少位宽，只做一次事先固定的成对比较，并给普通scale/zero-point对照。当前没有多做低位扫描或自动825。
4. source34和仅Dg的负结果只停当前未恢复端点。若后续恢复训练，必须明确新身份及相同训练预算；本阶段不再追加GPU臂。

## 复算与口径

`/usr/bin/python3.12 summarize.py` 从已完成逐帧记录重新生成汇总与配对CSV，无需GPU。四个执行入口、顺序队列脚本与原始日志均在本目录；`PLAN.md`记录追加的唯一权重组和唯一source34臂。

NB0是本地复现的原SDformerFlow，不是官方作者checkpoint；NB0使用原final head，当前学生为真实coarse head，这是相同DSEC任务质量比较，不是相同网络功能等价。历史mask没有逐位归档，本表依据相同GT/有效mask定义、完整帧名与逐帧valid像素数对应，未声称保存过历史mask位图。

活动表是当前粗头路径实际调用的ATLIF模块输出，含dead-result调用，不用它重新定义105个安装模块或81个活消费者库存。AT-LIF身份始终是`{0,θ}`，连续PED源来自神经元前状态。没有使用旧+0.005门，也没有把CUDA运行时间写成硬件性能。
