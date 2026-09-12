# 新stage320端点已绑定完整局部同资源服务

**新dense、contiguous34、lifting40的两个原窗口全部执行完成，没有沿用旧服务或旧门值。** 新literal source、冻结preview的独立scalar FP32参考、完整K864整数消费者均逐值通过。此表是实际新参数下的CPU同机服务；新的GPU窗口捕获正在单独准备，尚不声称新GPU端点逐位一致。

| 新stage320端点 | corner局部服务 | 相对新dense减少 | interior局部服务 | 相对新dense减少 | 已测同literal十帧AEE |
|---|---:|---:|---:|---:|---:|
| dense | 2,055,566 | — | 2,743,658 | — | 1.159737 |
| contiguous34 | 1,628,098 | 20.80% | 2,185,570 | 20.34% | 1.414199 |
| lifting40 | 1,930,822 | 6.07% | 2,585,116 | 5.78% | 1.165343 |

这个对照没有删掉更快的结构稀疏控制。lifting40在当前十帧质量上接近dense，并取得约6%局部净服务；contiguous34服务更低，但质量损失更大。十帧数来自各自stage320既有GPU评估，此表没有运行新的AEE，也不继承任何旧学生或825。尤其冻结preview在GPU与scalar执行器上可能存在归约次序差别，待新GPU小捕获对齐后才能把接口逐位身份补全。

范围从同一普通父学生的真实I24 halo开始，实际走新编source程序→完整K864 preview U32/V与固定BN1/非因果sn2→驻留gate→新Conv2 U16/F、BN2常量和原I24相加→新projection gate和完整PED U24/V96。原I24再次供给消费者仍收费，source/sn2没有外部退出/重新灌入。三个端点均为96×8×48 RF、128KiB state/coef、同SR64/SW64/CR256、同32B/5槽、同8192B源ROM、同H4目录和驻留source MAC。每例总系数冷填均165,568B。

| 组成 | dense corner/interior | lifting40 corner/interior | contiguous34 corner/interior |
|---|---:|---:|---:|
| source及I24输入 | 510,786 / 763,026 | 467,046 / 697,686 | 328,050 / 490,050 |
| source到sn2前缀终点 | 1,465,518 / 2,115,214 | 1,350,010 / 1,961,646 | 1,079,034 / 1,586,086 |
| 后续完整整数消费者 | 590,048 / 628,444 | 580,812 / 623,470 | 549,064 / 599,484 |

每条时间线是同Machine接续，前缀终点等于消费者开始时刻。新源改变门活动，因而preview/消费者费用也改变；以上是三个实际共同训练端点的净服务比较，不能把所有差额独占归因于source加法个数。lifting的新程序及所有阈值已经重编，不能沿用旧410周期/17.5%或旧8%局部数字。小RF所允许的后续source16＋consumer80交织尚未实现，不在此表中预支。

身份处理使用所有新`U_conv2_theta/F/U_ped/V_ped`字面矩阵及指数，`source/consumer_threshold/direction/constant`，consumer permutation，BN2/PED字面bias。没有从历史bias、gain或readout重构新cutoff。lifting包仍保留父As字段，wrapper按显式structure选择四层q12 lifting、每半步RNE/sat与新permutation，避免“存在As就执行dense”的历史分支。

CPU参考不从Machine结果反填：source独立执行新整数函数，preview用独立C++逐项fmaf、TF32 completion、BN MUL/ADD与T10膜计算，然后以新sn2和新q独立重建updated/projection/PED；gold仅写入明确的新candidate view。原capture、integer_chain和生产文件不变。六个preview窗口最小膜阈值余量为约1.17e−4至7.91e−4，但不能据此推断GPU门必然一致。

不含native投影、全域BN、最终join或完整帧前级，不把该局部表加上其他输入身份的native表。不是VCS/DC/PT/Formality闭合的RTL加速比，更不是整网FPS。

实现：[run.py](run.py)、[preview_reference.cpp](preview_reference.cpp)；同参数表：[summary.json](summary.json)，单例`<structure>_<window>.json`保存完整阶段、端口与检查；新CPU端点`*_cpu_endpoints.npz`供随后GPU逐字段比较。复跑使用Python3.12及`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`，例如`run.py lifting40 corner`。

## GPU收口更新

实际新函数首帧的两个halo已由root接管队列捕获并逐项核对；源门、sn2门、updated-I24、投影门及PED整数端点和参数全部相同，浮点preview差异仍显式列出。见[gpu_alignment.json](gpu_alignment.json)。以上早期“待捕获”状态至此关闭；有限窗口核对不是整网等价，也未改变原服务表。
