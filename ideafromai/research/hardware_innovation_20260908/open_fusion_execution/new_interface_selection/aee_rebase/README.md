# PED 静态重定基的真实网络 AEE

**2026-09-12 新准入规则：用户取消相对学生+0.005硬门，改按同设置SDformerFlow baseline核验。原SDformerFlow本地复现NB0同diverse10为1.454602861，同九帧为1.446425661；本页八臂都优于这两个对照，可以继续融合。** [逐帧配对](../../accuracy_baseline/README.md) · [现行规则](../../ACCURACY_POLICY.md)。原数值、校准/留出分离保持不变，小集结果不等于候选valid825。

2026-09-12。**两学生×四臂×同十帧，共80次帧推理已完成。ordinary原排列固定R24是更简单的共同底座；lifting白化R24也可继续考虑。** 不训练，只替换两个既有固定学生的continuous PED U/V矩阵。原preview、source/sn2/proj门、BN2、dynamic proj BN、之前的非anchor整分支删除、原coarse出口和AEE公式全部保持。

执行原 R32、原排列 R24、权重 SVD R24、激活白化 R24。每学生每轴使用同一 diverse10；原 R32 在本批十帧全部复跑。校准仅用 `zurich_city_09_a_0001.npy` 的指定256 anchors，因此同时单列排除该帧的九帧变化。这不是 valid825，也不把低秩派生学生当冻结 ep34。

[ped_rebase_adapter.py](ped_rebase_adapter.py) 直接装入保存的 signed16 系数，U/V 指数仍16/15，保留原两次 RNE/sat24 与原 PED bias/sat。它不改原 helper 方法或公共工程文件。[check_cpu.py](check_cpu.py) 先对两学生四窗的四臂245,760值逐位通过，结果见 [cpu_check.json](cpu_check.json)。[evaluate.py](evaluate.py) 在 GPU 开始网络推理前再次对同一 NumPy fixture 逐位核对。

与同时开展的 onepass BN 评价串行使用 A800；本轮没有合并它的数值函数。初次启动因本目录 `adapter.py` 与既有 latent adapter 重名而在模型导入前退出；已更名为 `ped_rebase_adapter.py`，不把该次失败计作评价结果。

| 学生 | PED 表示 | diverse10 帧均 AEE | 对本学生 R32 的 Δ10 | 排校准后的 Δ9 | 同十帧/九帧NB0比较 |
|---|---|---:|---:|---:|---|
| ordinary | 原 R32 | 1.161187076 | 0 | 0 | 均更好 |
| ordinary | 原排列 R24 | **1.163094004** | **+0.001906928** | **−0.006367803** | 均更好 |
| ordinary | 权重 SVD R24 | 1.182310304 | +0.021123228 | +0.017948226 | 均更好 |
| ordinary | 白化 R24 | 1.164879694 | +0.003692618 | −0.005257619 | 均更好 |
| lifting | 原 R32 | 1.186585517 | 0 | 0 | 均更好 |
| lifting | 原排列 R24 | 1.196321902 | +0.009736385 | +0.013798852 | 均更好 |
| lifting | 权重 SVD R24 | 1.195021450 | +0.008435933 | +0.017135934 | 均更好 |
| lifting | 白化 R24 | 1.190852853 | +0.004267336 | **+0.010778473** | 均更好 |

同一10帧/516,735有效像素，每个R32均与旧对照逐帧AEE完全复现。CPU和GPU包装分别245,760值零差；所有模式逐帧source/consumer发放计数不变，PED三处裁剪均0。完整配对帧和数值记录在 [results/run.json](results/run.json)，精简表在 [summary.json](summary.json)，实际进度在 [run.log](run.log)。这里Δ的分母是**各自学生**，用于展示改动代价；当前任务精度基线另用NB0，不能将两者混称。

白化在四个小窗有更小PED NRMSE，却在ordinary真网AEE上输给原坐标截断，说明局部误差不是有效替代指标。lifting白化的校准帧Δ=−0.054332899，拉低了十帧均值；其余9帧仍+0.010778473。九帧也来自这份固定小集，不能宣称已证明跨序列泛化或统计稳健性。

ordinary 的原排列 R24 另补了 [service_original24.py](service_original24.py) 的四个原P2/压力点，7,680输出与实际候选整数参考零差。它得到与白化R24相同的服务：ready 38,344→32,192（−16.044%），固定背压42,285→35,853（−15.211%），见 [service_original24.json](service_original24.json)。这仍是已有8lane/96RF/SR64/SW64/CR256机器的**CPU真实载荷执行模型，只覆盖连续PED U→V**；不是VCS/PPA或整patch加速。因而目前性能增量可以由普通低秩底座解释，不能归因于白化或新控制器。

**后继用途。** ordinary原排列R24进入更强普通分母；lifting的原排列/白化R24可按真实AEE与执行成本参与结构源融合，不再限定为恢复训练初始化。完整消费者/更大评价给各臂同等编译与训练权限，不用较弱的R32分母衬托X，也不制造免费动态oracle。Maestro式有序/嵌套训练尚未执行；若尝试，应与这个便宜静态R24比较实际取消的请求及其控制成本。

新颖性诊断仍 **3/10**（迁移底座，未有X）；性能证据 **6/10**（已有小集AEE与同资源组件模型，缺完整范围/RTL/PPA）。本轮价值是“实际试出了可用普通底座，并发现局部误差排序误导”，不是找到可直接投稿的标题。A800已经释放给onepass BN后继复核。
