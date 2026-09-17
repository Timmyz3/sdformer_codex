# 最终强控制：P32内静态参数只加载一次

真实源到完整FC1 H384/T10 PSN链；31个未用于pair选择的训练帧，每帧固定P32。所有模式P0冷加载，P1..31复用既有静态寄存器；每次top命令重装，X holding和图cache仍逐P清空。五臂均活动前沿PF、统一root bank；这项参数驻留是普通基线权限。

本表为cycles_to_last_gate，不是DONE、孤立叶或整网周期。每个函数分别比较，同W下code/class整数完全等价；W′/W″改变网络权重，不能跨表借质量。

## W′

| 同函数执行臂 | ready / BP周期 | 源标量MAC | ready / BP字节 |
|---|---:|---:|---:|
| static64 + next-X PF | 1931780/2059064 | 6348800 | 4046880/4046880 |
| one code | 1839283/2155990 | 5543800 | 5935872/5935168 |
| resident frontier code | 1723094/2029718 | 4224030 | 6134672/6129536 |
| one class | 1835560/2153684 | 5524400 | 5904144/5903744 |
| resident frontier class | 1719032/2023947 | 4213500 | 6088144/6084768 |

| 比较，减少为正 | ready | BP |
|---|---:|---:|
| frontier_code_vs_static | 10.8028% | 1.4252% |
| frontier_code_vs_one_code | 6.3171% | 5.8568% |
| frontier_class_vs_code | 0.2357% | 0.2843% |
| frontier_class_vs_one_class | 6.3484% | 6.0240% |

## W″

| 同函数执行臂 | ready / BP周期 | 源标量MAC | ready / BP字节 |
|---|---:|---:|---:|
| static64 + next-X PF | 1931780/2058873 | 6348800 | 4043296/4043296 |
| one code | 1839283/2155853 | 5543800 | 5932288/5931584 |
| resident frontier code | 1723094/2029589 | 4224030 | 6131088/6125952 |
| one class | 1820543/2135857 | 5447400 | 5919216/5918848 |
| resident frontier class | 1717909/2021488 | 4188830 | 6097216/6093904 |

| 比较，减少为正 | ready | BP |
|---|---:|---:|
| frontier_code_vs_static | 10.8028% | 1.4223% |
| frontier_code_vs_one_code | 6.3171% | 5.8568% |
| frontier_class_vs_code | 0.3009% | 0.3991% |
| frontier_class_vs_one_class | 5.6375% | 5.3547% |

## 判读与资源

普通前沿code共享全部供数与驻留优化。只有class相对这个强code的剩余差，才是消费者等价改变源义务的候选增量；参数驻留和预取本身不计创新。

680个完整H384命令通过；Y/U/gate各83558400次核对，partial(t,c)、批内独立channel refs、所有物理请求和反压均由TB检查。所有源实际MAC、后端系数/更新/PSN MAC和桥接语义计数与旧活动PF逐条相同，只有静态配置加载和服务时序改变。

维持106个独立乘法单元、256KiB外部参数/X池；另有Y90KiB、routes7.5KiB、U/tau各5.625KiB及其它局部状态、3840B门桥、192B D、128B X和128B图cache。复用已有参数寄存器，不增加容量；不称同面积或同Fmax。仍没有动态BN/FC2/shortcut、有效新AEE、VCS/DC/PT/Formality或PPA。
