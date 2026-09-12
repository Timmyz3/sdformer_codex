# 同预算结构训练与新源编译：独立代码审阅

**所审训练/编译路径暂未发现阻断功能的错误，但现在不能宣布 lifting 相对普通34结构的质量或性能优势。** 同父、同新增训练日程和字面硬前向已有具体证据；结构的有效参数空间不同。检查时只有dense完成新增320步和新源RTL重放，contiguous34仍在训练。未等待最终GPU、未重跑GPU/EDA，未改被审代码。本审阅由同团队另一个代理完成，不是外部盲审，不评录用概率。

所审主件为 `algorithm/fixed_structure.py`、`train_matched.py`、`initialization/run.json`、`smoke/run.json`、`source_execution/run.py`；为核实依赖，另读了 `parent_network.py`、`initialize_structures.py`、原 `FixedTemporalForward`、普通CMVM后处理、调度器/折门函数，以及已产生的dense端点与源结果。

## 已知实际缺陷与当前路径是否受影响

`algorithm/capture_train_parents.py` 的PED hook把 `helper.i` 存为 `source_I24`。旧 `FixedTemporalForward.finish()` 先执行 `self.updated=self.i`，随后原位写anchor，所以到PED hook时这个字段已经是updated值。**该 `train_capture.source_I24` 不能作为原source输入。** 此项由phase_aee先发现并报告；本审阅读代码独立确认，没有将它算作新的独立发现。

当前 `source_execution/run.py` 已改为直接读取旧 `stage_20260912/hardware/source_rtl_inputs/ordinary_*_inputs_i24.bin`，不再依赖大capture NPZ。root逐值核对它们与原 `hardware_exports/ordinary` 的FullCapture corner/interior halo导出的77760/116160个值完全相同；首次dense结果数据等价。原 `FullCapture.source_values` 在source hook先保存I24，更新发生之前且转为int32另存；因此当前新source输入路径不受上述字段错误影响。新增训练的上游生产者冻结，也使同位置旧ordinary I24可以继续作为源算术输入。**后续不要把它换成上述有别名问题的NEW字段。** root和phase_aee均已收到这一边界。

## 训练公平性：成立的部分与不能扩大的部分

我直接比较了三个初始化NPZ。contiguous34和lifting40的 `U_conv2_theta`、F、U_ped、V_ped的q16与指数，BN2/PED常量，以及consumer阈值/方向/常量/排列均与dense完全相同；差异位于源结构、源阈值/方向及相应元数据。三臂加载的是同一个ordinary original_ordered24+onepass父，不是把旧独立训练lifting端点给候选继承。

`train_matched.py:126–166` 每臂重新安装ordinary父、同seed，采用同64项与256项日程，各阶段新Adam，lr=1e-4；64步后保留同一套master参数值继续到累计320，重新开始优化器状态的规则三边一致。实际schedule长度为64和256，当前10评估文件与新增训练日程无交集。损失是有效像素上的真实coarse-head GT误差，未见teacher替代、验证选checkpoint或候选独享恢复步。dense当前日志已有64与256条实际更新；其余未完成不填最终数。

“同预算”在此指相同初始化/GT步数、数据日程、优化器设置，**不等于参数量或计算量完全相同**：

| 结构 | TRAIN矩拟合阶段的源变量 | GT阶段源矩阵有效系数 | 必须保留的说明 |
|---|---:|---:|---|
| dense | 100 | 100（最终99非零） | 原普通父源完整矩阵 |
| contiguous34 | tensor100、mask内34有效 | tensor100、仅34有效 | 66个槽被mask，smoke梯度也只有34非零，不能把100槽称100自由参数 |
| lifting40 | 40半步系数+10readout gain=50 | 40 | gain初始化后折入门cutoff/方向；不是34个参数的严格匹配对照 |

三个GT臂另有相同的下游参数和10个source cutoff。34结构是固定3+3+4块及共同输出排列，它不是所有可能34非零结构的最优代表；lifting具有不同混合拓扑和40次逐标量RNE/sat写入。若最终lifting胜出，只能说胜过列明的普通可训34结构及dense控制，不能说同参数量下压倒所有稀疏结构。

初始化MSE为dense0、contiguous34 **7.650555**、lifting **2.028684**；这是TRAIN4输入矩上的拟合目标，不是GT质量、RNE部署误差或硬件收益。还有一个具体初始化不对称：普通路径的均值补偿使用量化后的实际矩阵（`initialize_structures.py:70–74`），lifting补偿使用量化前effective/gain（63–66）。所以“解析匹配teacher均值”只对拟合域成立，不能声称部署后40次RNE/sat仍精确匹配均值。phase_aee认可将此限制写入最终说明；它不阻断当前明确使用真实硬前向的恢复。

## 硬函数和梯度

`QATForward` 不是将FP shadow用于损失的软替代。源矩阵、lifting系数、所有signed24写入实际执行round/clamp；卷积的整数bits由 `Bits` 给STE；两个门的硬比较按方向使用ceil/floor后的整数cutoff。`conv_forward` 返回的原conv值仅延续调用/shape，整数z/updated/PED与consumer都走保存的有费函数状态。source emit仍为 `{0,θ}`，θ固定；没有把它误设为不可折权的连续逐事件幅度。

三臂smoke都在同一真实TRAIN帧完成前向/反向，optimizer_updates=0。每臂分别比较source门、preview门、consumer门各 **73,728,000** 个值与PED **18,432,000** 个值，均0差；最终flow最大绝对差也为0。dense字面helper对原R24 helper也为0差。源有效梯度分别100/34/40项非零且有限，公共下游参数也都有梯度。因此当前证据足以排除明显断梯度、错接shadow或初始化字面函数不一致。

`Gate` 的方向符号和STATE_SCALE因子在value/cutoff梯度中一致；`Bits` 除以固定θ，与emit幅度相配。`OnepassSTE` 的前向调用同一个实际onepass统计函数，反向采用平滑BN公式，这是明确的替代梯度，**不是量化硬函数的真实导数，也不保证优化效果**。smoke只是一帧、尚无参数更新的检查；训练脚本另在每阶段导出后比较trained QAT与重新加载literal的flow，差异非0即停止。最终审阅需检查这些已有字段，不能只复用smoke的pass。

## 新源编译与RNE边界

`source_execution/run.py` 明确从**新stage320 deployed_constants**取系数、阈值，调用DAIS/da4ml0.6.0完整常量矩阵编译，再按共同RF/两阶段读写调度。普通dense/34分别对完整10×10矩阵做CSE；lifting对每半步5×10矩阵做CSE，RNE边界隔开，未非法展平为一个线性矩阵。

lifting编译在前7个半步层保留 **35个norm24**；最后半步的5个坐标只剩门消费者，才把RNE/sat和cutoff合成等价比较。前一半步的5个结果还被后一半步使用，仍保留norm24。普通路径10个纯门出口也获得同样的折门权限。源码原I24被保留给其余路径，source的q本身没有连续消费者，这一折叠边界在当前硬函数中成立。

折门函数处理正负方向、奇偶tie和越界常量。额外用Python `Fraction` 独立检查了当前lifting初始化cutoff及signed24两端附近，共 **378个RNE/tie/饱和门原像检查，0差**；未调用GPU。调度器还有596个向量的函数/寄存器标签/两拍RAW检查，新真实halo门gold由literal_gate重新计算。`compilation.json.full_halfstep_RNEs` 这个名字容易误读：它计的是**显式norm24节点数**，不是完整函数只有35次RNE，也不是5次RNE近似消失；展示时应解释“40次语义写入、35次物化、5次精确折门”。

检查时已生成的新dense结果是 **252 addsub、275条程序、61峰值活RF字**；两个实际halo的ready周期分别477252/972与712932/1452，都是 **491周期/SIMD8 tile**。这与旧ordinary497不同，说明确已重编新参数，未偷用旧source时延。该证据仅为同一帧9×9和11×11两个halo、ready/stress两环境的源RTL服务，不是四个独立帧或整链周期。

新dense stage320的10帧AEE为 **1.1597366283**、9帧为 **1.1351236837**。脚本使用新端点的quality.json，未继承旧825；对应NB0小集值1.45460286107/1.446425661411，采用严格小于，未加旧0.005硬门。新dense的99个As、13812个U_conv2、1522个F、2300个U_ped、2293个V_ped系数及多项bias/阈值确实改变，因此**全局或整局部链服务必须重放这些新参数和新门**。源RTL单独完成不能借旧下游费用推导总收益；脚本已明确 `inherited_downstream_or_frame_service=False`。

## 对候选X的判断与最小剩余证据

现在的有效新问题是：在共同普通父、同恢复预算下，lifting的受限参数化/逐半步RNE函数是否比普通可训34块结构更能保住任务质量，并在双方完整CSE、折门和共同RF/端口后提供净服务收益。公共lifting结构、普通CSE、静态折门和STE本身都不能重复算为X；初始化MSE优势也不能代替性能。

尚缺的最小闭环是三臂各自最终320步硬部署质量、实际参数改变量/裁剪范围、每阶段导出一致性，然后分别用**各自新常量**重编并执行同两个源输入。若需要整链主张，再接新门/权重/消费者，不能加回旧表。当前无需为收口等待或重复训练；正在运行的三臂把这些既定端点填完即可。现有材料支持继续执行这个具体对照，不支持强accept判断或已经成立的独立创新结论。
