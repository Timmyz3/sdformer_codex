# 扩大实际融合覆盖：2026-09-12

**仍没有把所有适配接口试完。本批已推进到匹配训练、新完整825、新参数源RTL、付费局部链和完整native后段。** 305条候选视图逐行关联到实测/未试接口；论文别名、同机制多来源和工具记录重叠，不是305个独立idea或已复现论文。逐项见[覆盖表](coverage_owned/COVERAGE.md)，未试范围见[剩余批次](coverage_owned/REMAINING_BATCHES.md)。

## 最新增量：两项源的完整825与真实共享资源交错

两项dense与lifting40各完成一次新的valid825：AEE **1.209053834 / 1.235243345**，均优于NB0。相对自己的未量化父分别为+0.000724/+0.009749；lifting十帧改善没有扩展到825，12项真实sat24饱和已计入评价。[逐帧质量](algorithm/source_constant_valid825/README.md)。

40个固定局部CPU执行已完成：首对A-PED P2＋完整B源的22例，以及完整A窗口消费者＋完整B源的18例。每次实际共享96RF、单issue/ready/pending、SR/SW/CR和单DMA暂存；不是合并两条独立时间线。

| 完整A局部窗口＋B源 | 最强普通串行槽 | 保存后交错槽 | 净减少 ready / 固定背压 |
|---|---:|---:|---:|
| dense 两项 | 2,372,848 | 2,356,692 | 0.681% / 0.545% |
| lifting40 两项 | 2,227,558 | 2,185,190 | 1.902% / 2.143% |
| 普通分组34 | 2,090,628 | 2,036,004 | 2.613% / 2.694% |

强普通立即完成PED并采用P1留Z；候选保存16个anchors的46,080B，迁移、读回和真实输出都收费。每例A全8×8门与4×4 PED、B全11×11源均数值通过。普通34也受益，故该布局保留为公共底座，**不作为lifting独占X，也不关闭整个结构/交错家族**。完整B消费者、native/全域BN与整层不在此表。[首P2](hardware/actual_interleave/README.md) · [完整A窗口](hardware/actual_interleave/full_window/README.md)。

## 新训练的时间源：质量与服务一起比较

三臂从同一ordinary R24＋onepass父出发，同train数据、同初始化预算、同新增320步GT恢复；没有继承旧lifting额外训练。推理加载实际整数常量、cutoff与RNE/sat。参数量不同，未声称等参数量。质量门为同协议优于本地原SDformerFlow NB0（valid825帧均1.445352535）。

| 新训练的源 | diverse10 AEE | 新valid825 AEE | 源RTL ready周期/H8 | 两真实halo局部CPU服务，相对新dense |
|---|---:|---:|---:|---:|
| 普通dense100 | 1.159737 | 1.208330 | 491 | 基线 |
| 普通分组34 | 1.414199 | 1.421461 | 303 | 少20.80% / 20.34% |
| 可学习lifting40 | 1.165343 | 1.225495 | 446 | 少6.07% / 5.78% |

新分组34已在完整825过门，旧免训遮罩失败不能否决该结构。lifting保留较好质量与小活跃集，但新源周期优势为9.16%，局部链被摊薄为约6%；**不能引用旧497→410或旧约8%局部服务**。两个halo不是整层，活动改变也参与服务变化，不能全归常量算术。[训练/AEE](algorithm/README.md) · [新源RTL](source_execution/README.md) · [完整局部CPU链](hardware/matched_local_chain/README.md)。新GPU两halo已对齐，三臂1,926,720项整数端点/全部参数0差；preview浮点归约差异仍保留，有限窗口不等于整网CPU/RTL等价。

按[逐序列配对](quality_by_sequence.csv)，新dense为18/18序列、765/825帧优于NB0；lifting为18/18序列、747/825帧；分组34为14/18序列、536/825帧更好，最大序列均值损失为0.19609。34符合用户整体AEE门，但不能描绘成各场景均不损精度，也不因这一分布重新收紧用户的准入门。

## 相同量化权限与低状态普通控制

固定一次“至多两项有符号2幂之和”投影，dense/lifting规则相同；不训练、不扫项数、不改下游或cutoff。

| 新函数/布局 | 源ready周期/H8 | 活跃工作RF，另加门RF | 结论 |
|---|---:|---:|---|
| 两项dense，完整CSE | 319 | 40 | 比其491周期父源少35.03% |
| 两项lifting，完整CSE | 290 | 10 | 比其446周期父源少34.98%，35次中间norm仍在 |
| 同两项dense，逐行低状态CSD | 472 | 13 | 229字ROM合法；普通源同样能低状态，需付时间 |

量化改变真实门位0.253%/0.519%，新diverse10为dense **1.165901**、lifting **1.152913**；自己的825现已完成，以上方新结果为准。[完整局部链四例](hardware/source_constant_local_chain/README.md)已实跑：dense对其父少8.20%/9.19%，lifting对其父少7.61%/7.94%；同量化权限下lifting相对dense仅再少5.47%/4.47%，不把源约35%推广整链。原未量化dense低状态布局需524字，超过共同512字ROM；量化后的同函数控制已经适配。**不能将dense的40/61RF当状态下界。** [两项接口及RTL](source_constant_probe/README.md)中保留完整对照；PoT、CSE、低状态重算都属于公共底座，单独不是X。

## 完整后段：普通底座继续补强

从旧ordinary R24＋onepass真实全图projection gate/PED开始，同一Engine完成native K864、192000×96输出、完整T-major单遍BN和最终PED相加；**尚未包含上游I24生产，不是整网。** 同96×8RF、128KiB状态/128KiB系数、SR64/SW64/CR256、32B/5槽DMA。

| 完整后段 | 服务槽 | 相对直接前一强控制 | 角色 |
|---|---:|---:|---|
| native＋物化BN＋PED | 533,087,251 | — | 普通参考 |
| native＋融合BN/PED | 457,055,251 | 少14.26% | 普通融合 |
| 两块目录复用＋同融合后缀 | 427,507,285 | 少6.46% | 普通分块 |
| 新W8函数，展开FP32＋上述复用 | 427,507,285 | 新函数另列 | 同权限参考 |
| 同W8，实际packed全权驻留 | 412,755,104 | 少3.45% | 解码收费后净收益；新十帧AEE1.149277 |

两块复用外读反增161,650,560 B，用W搬运换去目录重建。W8驻留将外读420,360,128降至221,377,856 B，解包和末尾scale MUL计费后净服务再少3.45%。这些是实际完整时间线，不能与局部链相加/乘倍率。[native链](hardware/native_bn_join/README.md) · [两块复用](hardware/native_bn_join/directory_reuse/README.md) · [W8驻留](hardware/native_bn_join/native_w8/README.md) · [独立审阅](coverage_owned/native_chain_review.md)。原未量化升序K和GPU归约不逐位相等；新W8两个输出窗口及全域统计已和CPU逐位相同，但W8改变权重，只有自己的十帧质量，不能继承旧GPU825。量化、分块和融合都是强底座，尚非标题X。

## 其他实际融合及停止的具体接口

| 接口 | 实测结果 | 去留边界 |
|---|---|---|
| NRV∩W＋同址合并/固定缓存 | 528例CPU前端正确；最优索引仍比最强dense慢42.93%–104.33% | 随机读/解码已计费；停该布局，不外推完整Gustav |
| 固定56字S＋F_live=2 | 352例正确；源读576→288，最强F2平均仍慢9.95%–18.26% | 128实际同配置对8胜120负；停该双包配置 |
| 真实PED W4/W8码＋尺度 | 25局部链正确；系数字节少，服务增0.56%–0.86% | 父为旧R32＋CUDA，和上述native W8不同 |
| 门条件预测＋Q8与普通量化 | 两旧R24学生×五表示实际diverse10；ordinary full-D 1.168703，lifting普通affine 1.180753优于full-D 1.203849 | 全部过十帧NB0；完整D无普遍优势。8种新增参数组合＋2个重复固定Q8控制 |
| Dg＋固定5+5查表 | CR表慢21.93%–27.96%；紧凑RF表慢44.16%–49.82%，压力点慢47.15% | 两付费放置停；保留其他响应/布局，查表不独占创新 |
| Pro水平P2舍入证书复放 | 305,760 lane/38,220 SIMD8零接受 | 9月11日已试；本批是复现，不计新覆盖 |

[硬件](hardware/README.md) · [表示来源与AEE](representation/README.md) · [共享索引](coverage_owned/NEW_INTERFACE_RESULT.md) · [双上下文](coverage_owned/flive2/README.md) · [重复实验纠正](rounding_margin/README.md)。Q8整字空率仍接近零；AEE改善不等于跳过。两查表放置均给普通条件加同等缓存权利。

## 候选X与后续

候选X须落在“结构时间源如何在真实二值门/连续消费者共同资源中改变可执行时间线”。本批补齐普通34源、普通两项常量、低状态dense、分块和W8驻留，避免把底座遗漏当创新；没有因一个负布局杀掉lifting、NRV、多上下文、剪枝或表示家族。

真实96RF交错已执行两种工作边界，小幅净收益保留为公共对照。下一步改接口：先在已执行的源→preview/后继中定位可删除的物化/重复首读，接结构源与native/全域BN完整生产链；当前学生因果帧间完整尾部、受限公共子图作为后续独立批次。注意力行memo、权重结构训练、固定stem/decoder替换仍列在[批次表](coverage_owned/REMAINING_BATCHES.md)，不假装全部启动。[原交错设计](coverage_owned/INTERLEAVE_NEXT_INTERFACE.md)保留设计到实测的链接；本轮不再扫当前保存/轮转布局。

AT-LIF按最后确认的`{0,θ}`、固定θ可折权；连续I24/PSN/PED单列。NB0门代替旧+0.005/1.259。原Grok错误身份材料不覆盖、不沿用其否决理由。生产nts07、docs359、H81、main.tex未改。源为Verilator RTL，其余性能主要为有数值执行的CPU服务模型；尚无新的VCS/DC/PT/Formality同负载闭环、ASIC PPA或TCAS-II强接收结论。
