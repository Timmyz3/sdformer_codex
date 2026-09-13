# 整层真实源、固定三臂与质量扩展

## 已交付的完整数据

固定 `zurich_city_09_a_0001.npy`，在旧合法 matched-dense stage320 模型配置上重新捕获真实 r0 sn2→conv2。输入是 `[10,96,240,320]`，T10、完整C96；theta为单一静态1，全部输入严格属于0/1，bias不存在。原8个真实tile的源和FP32/Q16权重均逐值相同；本次原函数首帧AEE为1.3208328825980986，不把它替换成其他快照的旧数。

| 文件 | 数据合同 |
|---|---|
| [manifest.json](manifest.json) | 完整形状、原点、来源、三臂、幅值验证和旧数据一致性 |
| [source_bits.npy](source_bits.npy) | bool `[T10,C96,H240,W320]`，未拼接离散patch |
| [source_words.npy](source_words.npy) | uint16 `[C96,H240,W320]`，低10bit中bit t就是g[t] |
| `*_weights.npz` | 三臂固定 `weight_q16[96,96,3,3]`、`live[12,24]`、theta和指数；原FP32 W另存 |
| `gold_<arm>.npy` | int32 `[19200,10,96,2,2]`，完整K864/N96/T10；适合mmap按tile读 |
| `gold64_<arm>.npy` | 同函数前置跨行64tile，固定整层id128..191 |
| [original_output_fp32.npy](original_output_fp32.npy) | 原未量化卷积输出 `[10,96,240,320]`，函数身份单列 |
| [gold_progress.json](gold_progress.json) | 三臂完整gold及旧8tile、首64tile逐值核对记录 |

三臂名称为 `dense_q16`、`block_magnitude25`、`cin_fullcost25`；直接读取同名前轮固定mask，不训练或重选。整层有120×160个输出tile，行主序id；输出原点为 `(2*(id//160),2*(id%160))`，4×4源原点为输出原点减1，图外坐标置0。首次64tile id128..191包含输出tile行边界。导出完整输入不代表硬件已经完成整层执行，实际回放范围以stream_rtl收据为准。

整数函数是 `Σg·RNE(theta·W·65536)`，输出保留Q16整数，未包含后继BN/PSN。gold用FP64卷积计算精确整数：二值与signed16乘积及全部864项任意中间绝对和都远小于2^53，逐值断言无分数且signed32不溢出后转换。三臂全层gold各与旧8tile独立int64 gold完全相等、与首64tile参考相等。它不是原FP32卷积，也不声称全网bittrue。

## 新全帧统计

[source_statistics.json](source_statistics.json) 包含完整支持码频度、逐组记录和直方图；[source_statistics.py](source_statistics.py) 固定原通道顺序的连续Cin4/Cin8，不根据结果调整配对。

- 全源2,663,435个spike，密度3.6125149%；7,372,800个T10词中1,428,729个非零。
- T内1..9变化4,085,607次；包含起始D0的执行端点共4,884,960。末端归零D10另57,314，封闭端点共4,942,274；正区间起点2,471,137。起末边界定义分开，不假定跨帧连续。
- 连续Cin4每时刻支持popcount=0..4的频数为 `[16102334,2021160,284023,23703,780]`；两pair同t活跃H合计220,551。连续Cin8对应频数为 `[7247203,1421254,424104,102728,18058,2383,250,18,2]`；两个四源半组同t活跃H合计360,869。
- 不含halo的19,200个不重叠2×2空间区只有18个全零；spike数中位123、P99=431.01、最大637。这个分布是源活动统计，不是有halo的源读取或消费者周期。

端点多于spike是此帧当前源布局的实际属性；本表没有执行差分RTL或比较跨帧接口，不能从计数淘汰整个时间差分家族。支持冲突也不能直接换算整层收益。

这些端点定义在真实sn2输出的T10轴上；当前上游含非因果PSN，不能由该计数声明在线因果事件到达执行。

## 已完成的三臂valid825

合法A800 mux已实际连通，GPU起始空闲；复用远端Python3.12.3、Torch2.2.2+cu121、CuPy13.6环境。官方valid CSV确有825个唯一帧，全部2475个事件/GT/mask文件存在。固定三臂沿同一模型、同一Q16权重函数评价，不以diverse10代替valid825。

[protocol_checks.json](protocol_checks.json) 额外确认训练校准帧thun_00_a_0002不在官方825中，但thun_00_a序列在完整valid825内。因此这是官方帧级train/valid分离，不是825序列完全留出；此前diverse10的帧和序列均不重叠结论只适用于那十帧。

本环境dense diverse10协议探针AEE=1.1590088019062106，与前轮3090/Torch2.7的1.163296947981814略有不同；明确记录环境差，不调参数追旧结果。初10帧含暖机估计三臂45分钟，进入官方集合后估计约23分钟；最终三臂记录的wall_seconds合计1624.10秒（约27.1分钟），其中dense包含优先插入整数因子链时的暂停。这是评价耗时，不是硬件速度。

**三臂官方valid825均已完成，并通过历史NB0质量门1.44535253468097。** 没有在本轮重跑NB0，也未以新环境波动放宽门限。

| 固定函数 | 帧数 | 有效像素 | 帧等权AEE | 历史NB0门 |
|---|---:|---:|---:|---|
| dense Q16 | 825 | 48152523 | 1.2106890853803083 | 通过 |
| 普通块幅值25% | 825 | 48152523 | 1.2655970935545984 | 通过 |
| 完整Cin费用25% | 825 | 48152523 | 1.2992308516308682 | 通过 |

[valid825.json](valid825.json)、[三臂全部逐帧/汇总文件](quality825/)、[运行日志](valid825.log) 均已从A800同步。[validate_quality.py](validate_quality.py) 逐帧核对三臂825个唯一文件顺序、每帧有效像素完全一致，并从aee_sum/valid_pixels独立复算帧等权和像素等权均值；所有检查通过，见 [quality825_validation.json](quality825_validation.json)。dense协议十帧与官方825中相同帧AEE逐一差0，无运行中协议漂移证据。

便于直接使用的 [质量汇总CSV](quality825_summary.csv) 和 [825帧配对CSV](quality825_paired_frames.csv) 保留全精度数值。当前完整Cin费用控制在825上比普通块幅值AEE高0.033634，应与源/周期节省一起评估，不单因都过NB0便认为质量等价。

这是相同Q16系数解码后的原浮点/TF32消费者AEE，不是后继整数舍入全部闭合。新Q1/Q2因子链只做十帧，不能继承这里三臂的825结论。所有GPU进程正常退出，A800已释放，本阶段不再启动模型任务。

## 新固定整数因子链的十帧质量

根代理另在 [integer_factor/definition.json](../integer_factor/definition.json) 固定既有SVD rank8的唯一Q1/Q2格式：Q1为[-3,3]，Q2为signed16；`z=Q1*g, p=Q2*z, y=p*output_scale`，中间无RNE。未重选rank、格式或训练恢复。

[evaluate_integer_factor.py](evaluate_integer_factor.py) 实际在同一A800/env312上执行完整diverse10。两级conv使用FP64精确整数模拟，逐帧断言z/p无分数且分别满足2592/679477248界限；最终FP64乘scale再转原float32消费者。TF32设置仅继续影响其余原网络算子，未近似z/p。

结果 [integer_factor_diverse10.json](integer_factor_diverse10.json)：10帧、516735有效像素，**AEE=1.3533257656301605**，低于历史NB0十帧门1.45460286107。首帧完整两级p与展开整数权重conv逐值相同；[integer_factor_first8.npz](integer_factor_first8.npz) 额外保存真实source/z/p和scale，轴为tile,T,C或R或N,y,x。此新链没有valid825，不挪用旧浮点SVD结果。

跨环境边界有实测：A800首帧输入2663570spikes，3090本次capture为2663435；相同旧8tile中有19/122880个bit不同，spike总2812对2813。Q1/Q2函数和mask未改变，但不能说两GPU源逐位一致，也没有用A800源覆盖硬件固定的3090源。

新链评价期间只暂停owned825进程，保存状态并在dense已完成481帧后恢复，没有删帧或用新子集替代官方825。详见 [priority_pause.json](priority_pause.json)；dense的wall_seconds含暂停，不能作为纯前向速度。

所有新增文件位于本目录或远端同名owned目录；旧目录与生产只读。未新建环境、哈希、训练、重新选择掩码或扫描比例。

## 独立硬件审阅

[REVIEW_STREAM.md](REVIEW_STREAM.md) 核对新连续wrapper：24个跨行64tile job和6个完整19200tile job，共448266240输出；540项独立源/几何/事务/总周期检查全部一致。原leaf文本未变，完整Cin只减少leaf内读取，wrapper外部源仍各臂同样29276544次；没有隐含halo缓存或上游生产退休。

[REVIEW_INTEGER_FACTOR.md](REVIEW_INTEGER_FACTOR.md) 从14套fixture独立重建53760个int64输出，核对168条RTL记录的2688项计数全部一致。八真实tile的mode6/7/8核心分别370240/173739/180651；完整计入供数、z、V、psum后，当前mode8每tile必比mode7多864拍。共同静态零控制、宽系数/乘法器预算和最终scale未入叶周期的边界均已注明。
