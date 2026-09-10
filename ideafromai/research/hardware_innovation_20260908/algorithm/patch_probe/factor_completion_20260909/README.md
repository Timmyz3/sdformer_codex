# 单 Conv1 的两因子、共同掩码与连续消费参考

已完成多种固定结构融合、A800真实光流恢复、整数部署参考和真实权重字计数。本目录不改生产模型或已有结果。

**当前取舍：普通R32已经过完整精度门，固定R56私有补全尚不能作为主创新。** 把恢复后shared48的尾V直接删除、保留原共享32列，825帧AEE为FP **1.201095**、U8/VQ5解量化 **1.203043**，每轴48,152,523有效像素。数据在 `network_preview_only_valid825/`。这是未再训练的普通删除控制；软件保留原R槽方便数值比较，普通编译可删除断开的尾U/V。它不是新机制，但应作为后续更强底座。

实际完整首帧中，private56相对shared56同样条件执行的U请求优势，从J2的1.54836%降到64bit的0.40631%、256bit的 **0.03999%**；标量U更新少6.15237%，不等于速度。该表来自恢复后QDQ捕获，不能与后来的Aq14整数AEE拼接。[完整物理字比较](latent_stage_train16/physical_u_requests.md)。不再给这个固定私有结构追加RTL；因子化、移位消费与消费者完成思想仍保留，后续结构须修复实际剩余成本。

**真实网络已运行，当前仍未选赢家。** `network_diverse10_full/`、`network_diverse10_conditional/` 和 `network_diverse10_ablation/` 已回收，均是同10帧／516,735有效像素、真实Conv2/BN2/shortcut/粗头；不是valid825。未分解Conv1配相同A/b/θ的AEE为1.154261。第二阶段结果：

| 学生 | 完整执行AEE | 条件执行AEE |
|---|---:|---:|
| shared | 1.125389 | 1.134709 |
| grouped | 1.267051 | 1.279537 |
| hybrid | 1.123533 | 1.127649 |
| shared_compact | 1.117026 | 1.114425 |
| grouped_compact | 1.200369 | 1.205534 |
| hybrid，同预算λ=0 | 1.112868 | 1.123692 |
| shared_compact，同预算λ=0 | 1.113154 | 1.108202 |

更简单的compact仍是强控制，不能凭private启用或请求少1%宣称胜出。[潜变量分阶段结构试验](latent_stage_proposal.md)已经完成固定训练与真实网络10帧：共享预览与私有尾使用不相交的U系数，各段保持完整T10复用；它直接针对旧时间阶段重复取权的负结果。移位V权重的普通量化恢复控制也已完成，处理连续Z×V成本，移位本身不算X。

## 不相交潜变量与移位消费：实际结果

`latent_stage_train16/`保留普通shared56的32+24分阶段、private56的32共享+每H8两个私有（共24尾）以及普通shared48的32+16，三者均可提前完成与回退。两种R56同训练预算、相同原common3 A/b/θ与真实Conv2后继。原普通shared48的累计局部更新较少，明确保留这个便宜控制。

| diverse10，10帧／516,735有效像素 | 完整AEE | 条件AEE |
|---|---:|---:|
| shared56，λ=0 | 1.105539 | 1.103004 |
| shared56，λ=0.1 | 1.121857 | 1.125619 |
| private56，λ=0 | 1.136108 | 1.140348 |
| private56，λ=0.1 | 1.112135 | 1.137940 |
| 原普通shared48，同分阶段实现 | 1.128484 | 1.136489 |

数据在 `network_diverse10_latent/`；重建起点也已测并全部保留。全局换成分阶段V累加会改变FP舍入，因此不和旧单次V乘法的AEE当作位等价对照。局部valid4上，private56相对shared56的λ=0.1版本少约2.18%U请求、7.64%U加项；V连续乘加27,293,232→22,811,142。普通shared48的U仍更便宜。当前局部表没有计成完整服务步，FP连续MAC也不能与源加法等价相加。

`shift_consumer_train16/`按[DeepShift作者公开实现](https://github.com/mostafaelhoushi/DeepShift)的量化数学独立实现五bit V：符号和−15…0指数形成32种非零码，结构零另由连接掩码给出。出处是CVPR Workshops 2021，非主会。U/Z仍FP32，两结构各固定额外256步，FP32与Q5同预算，不加费用项。

| 同diverse10，旧时间完成器 | FP完整／条件AEE | Q5完整／条件AEE |
|---|---:|---:|
| 普通shared48 | 1.128476／1.115284 | 1.108159／1.127099 |
| hybrid | 1.107372／1.110306 | 1.109094／1.122280 |

见 `network_diverse10_shift/`。Q5局部教师门误差较高，却未导致本次10帧AEE过预算，因此不能只按局部代理杀掉量化融合。V载荷降低也不等于连续Z路径已经变成整数移位器；这里没有硬件速度结论。

**真实任务恢复六轴已完成。** `latent_stage_train16/train_flow_recovery.py`固定三结构×FP/U8+VQ5，各64步Adam1e-4，只更新U/V，用真实GT光流损失；不加请求或局部门损失，不扫描训练参数。显式train16全部属于DSEC train7345，与valid825无交集。训练后同train16完整空间校准，每轴每时间/通道1,228,800个观测，再评完整和条件执行。普通shared48累计320次更新、两种R56累计576次，恢复阶段预算一致，累计预算差异不隐藏。下游原有整数分支的梯度边界保留，已实测可由残差路径与原替代梯度传回U/V；不是全网络整数STE训练。恢复前的真实光流反传诊断在 `flow_backward_diagnostic/`：两结构各614,400输出值相对同参考0差，所有许可U/V梯度有限且非零，诊断无参数更新。

| 恢复后diverse10；FP/QDQ版本 | FP完整／条件AEE | U8/VQ5解量化完整／条件AEE |
|---|---:|---:|
| shared56 | 1.104834／1.090178 | 1.100674／1.104714 |
| private56 | 1.102896／1.100995 | 1.119326／1.103118 |
| shared48 | 1.107205／1.101525 | 1.116344／1.100278 |

完整12轴、各10帧/516,735像素见 `network_recovery64_diverse10/`，另有每轴首4帧真实门/接受/源空/尾需求及同坐标连续窗口。局部valid4六轴共5,898,240门，使用保存的新训练统计重算紧凑表及取消尾后的回退均0差；这是对各自完成函数的数值核对，不是对原教师零差。原教师门误差与对自身完整函数的改门另列在 `latent_stage_train16/flow_recovery64/cpu_valid4/`。

**整数算术另立学生，已真正跑网络。** `compile_integer_factors.py`将θ折U8、V幂次对齐、Aq14及固定BN/阈值接起来；不保留免费幅值乘法。共同保守规格为Zi16、Yi32、Ui48，三轴所有归约次序的Ui范围分别需47/45/47位，FP64可以精确承载。三个真实图区域的独立整数/原生适配共2,073,600门零差，另有非单位θ和阈值边界检查。A量化与消除中间FP舍入会改变学生，不能沿用QDQ成绩。

| 新整数版本，diverse10 | 完整AEE | 模式表条件AEE | 普通小表条件AEE |
|---|---:|---:|---:|
| shared56 | 1.084425 | 1.094601 | 1.110025 |
| private56 | 1.113727 | 1.097184 | 1.116050 |
| shared48 | 1.091660 | 1.118859 | 1.118451 |

见 `network_integer_recovery64_diverse10/`、`network_constant_predictor_diverse10/`。整数GPU源幅值/整数Z检查均通过；尚无这组的825或完整服务优势。普通小表将部分source-empty模式统一用已有code0阈值，只保留全空精确特判，各结构同享；裸INT48阈值载荷133,632→11,520 B。局部983,040门/轴最终输出与原模式表0差、U请求仅增0.117%–0.146%，但完整10帧AEE并非相同，不能外推全图位等价。均为统计提前函数；小表不是独立创新。

[CFMP 作者稿 Fig.23.2.5](https://arxiv.org/pdf/2512.17555)明确：训练两个因子和中间 tiled mask；FMS 在第一因子生产前解码 TC；紧凑存 Z；DRU 用同一 TC 转换为第二因子 TR 索引，累加恢复稠密输出。公开三页稿没有给齐扩展比、tile 维度、损失、排序准则和训练日程。这里迁入上述**可见功能链**；SVD 初始化、P4/J4 粒度、固定秩、STE 和损失是本地具体化，不称官方训练复现。

## 固定的一档与强控制

只分解 r1.conv1：`rawY=((theta_source*g @ U) ⊙ M) @ V`。原固定 BN 仿射、所选完整 T10 的 A/bias/θ、真实 Conv2/BN2/shortcut 保留。没有把已有两 Conv 之间的非线性吞成矩阵乘积。

| 轴 | 潜变量池 / 每区活跃数 | V 的许可连接 | 静态因素 FP32 预留 |
|---|---|---|---:|
| shared | 96 / 48 | 全 H96 | 368,640 B |
| grouped | 96 / 48 | 每组 8 个潜变量只到对应 H8 | 368,640 B |
| hybrid | 96 / 48 | 48 个共享；每 H8 另 4 个私有 | 368,640 B |
| shared_compact | 48 / 48，无 mask | 全 H96 | 184,320 B |
| grouped_compact | 48 / 48，无 mask | 每组 4 个潜变量只到对应 H8 | 184,320 B |

前三轴同最大 U/V 槽、同活 Z 宽度、同训练批次与步数；**实际非零 V、加项和最低压缩存储不同**，结果分别列出。后两项是更省静态参数的普通低秩强控制。R96 是相对 R48 活表示的双倍候选池，本身不比原 C96 更宽，不复述论文未知的扩展比。

mask tile 为 **P4×J4**，与物理输出 **H8** 分开。全图使用固定 8 个水平带 `region=y*8//240`，每带 24 位，T10 与横向共用；裸 mask **24 B**。每带约 8 个捕获 P4 组、16 帧用于学习，未采样位置也有固定定义。若学出的 8 份 mask 全相同，普通编译器可删掉未用因素列，结果显式报告这个退化控制。没有把全分辨率独立 mask 的 57,600 B 免费塞入 16 KB。

活 Z 一时刻 P4×48 FP32 为 **768 B**，十时刻全存为 **7,680 B**；全 T10 Y 为 **15,360 B**。这些是容量枚举，尚未排程，不能称适配原 2 KiB slice。V×Z 是连续乘加；θg 只使第一因子保持加法型源，第二因子的代价不免。

## 已跑功能结果

`check_reference.py` 已实际运行，记录在 `reference_checks.json`：

- 不同空间 mask、整零 mask、负因素、θ=2.5 的定向小例：稀疏生产/消费与稠密参考 **0 差**。
- TC 按列存到 7 bank 后逐 TR 恢复，所有地址和值一致；不把该索引映射叫一次 SRAM 事务。
- 两个真实原生 P4 源组、原 FP32 W 的五轴：Float64 稀疏/稠密最大绝对差 **1.34×10⁻¹⁵**。
- 私有/共享消费者的需求回推小例通过；这是逻辑功能，不是提前完成率或周期收益。

随后使用 `joint_completion_20260909/.venv_train312/bin/python` 的 PyTorch 2.8 CPU，前向/反向梯度和全图 adapter 形状检查通过，五轴各 256 步已完成（约 7 秒）。训练后的稀疏/稠密 TC/TR 也再次通过。`FactorConv1` 的全图适配器用于真实网络数值验证，它在 GPU 上密集形成 Z 后掩码，不把 GPU 墙钟当稀疏硬件速度。NumPy 参考才显式只生产掩码许可的列。

`fit_train16/result.json` 中，16 帧新同次 Y 与旧真实 Y 均逐值零差；valid4 从新完整位图提取的逐 p/k/T 源字与旧完整源捕获也逐值相同，沿用原真实 norm1 Y。CPU FP32 原 W 点积＋BN 仿射相对 GPU 捕获的最大 Y 差 0.001199、MSE 1.52×10⁻⁸、983,040 门中 6 门不同，不能称原 FP32 位等价。

| 第一阶段 valid4，983,040 门/轴 | 标准化 Y MSE | 总门差 | 漏掉教师非零门 | 不同区间 mask 数 |
|---|---:|---:|---:|---:|
| shared | 0.08734 | 1.1168% | 14.6604% | 4 |
| grouped | 0.38272 | 2.1749% | 42.7922% | 4 |
| hybrid | 0.07754 | 1.0459% | 13.1139% | 1 |
| shared_compact | 0.07754 | 1.0459% | 13.1139% | 1 |
| grouped_compact | 0.36685 | 2.1755% | 41.3118% | 1 |

**hybrid 只保留前 48 个共享潜变量，私有变量全部未启用；其活 U/V 与 shared_compact 逐值相同。** 本次只训重建和门恢复没有激活消费者分区 X。它是该初始化和 256 步损失下的具体退化，不是 CFMP 或可训练私有表示无效的结论。总门差因稀疏分母而较小，故同时列漏激活率；没有用这些局部指标代替网络 AEE。

## 数据与下一条命令

优先新采集目录下 `sampled_source.npz`：`frame_name, group_ids[64], source_gate_words[64,864,4], Y[10,96,64,4], theta_source`。k 顺序为 `((c*3)+kh)*3+kw`，bit t 对应时间 t，padding 为零。按 frame_name 与旧 `partial_completion/capture.pt` 对齐 A/θ；优先使用新同次 Y。也支持整包 PT/NPZ。

如 valid4 没有新同次小捕获，默认读取旧 `integer_valid10` 的逐 p 源字，仅在 P4-OR、每 p/T 计数与旧浮点 Y 捕获逐值一致时配对；**不读取旧整数 Yi 作为本次标签**。最好为新 valid4 也开 `--capture-sampled-source`。

在 A800 上，源码同步至同名相对目录后：

```bash
cd /root/private_data/work/hardware_innovation_20260908
env312/bin/python algorithm/patch_probe/factor_completion_20260909/train_factors.py \
  --source algorithm/patch_probe/joint_completion_20260909/full_capture4/capture \
  --temporal common3 --steps 256 --batch-groups 8 --device cuda
```

如果补采输出放在另一目录，仅替换 `--source`。默认顺序运行上述五轴，共同固定预算，没有 R、mask 密度或阈值扫描。损失为 train-only 标准化 post-BN Y MSE＋0.25 平衡门 BCE；不训练 A/bias/θ，不用 valid 选 mask/参数，不假装有四邻域 Conv2 蒸馏。每轴记录初始/训练/验证局部 Y 与门差，并保存 `.npz`；完整网络评价时调用 `install_conv1(module, parameter_file)`，仅替换原 conv1.0 的 forward，原 BN/神经元/Conv2 保留。

第一阶段只建立完整两因子对照。随后按根节点授权完成以下相同完成器的需求测量与第二阶段训练，没有抹掉第一阶段退化结果。

## 同一个完成器与第二阶段实际结果

`demand_completion.py` 对五轴自己的新 Y 应用同一许可：先算时间列 `[2,3,7]`，其余用该轴 train16 的均值/协方差给出 `gamma=3` 统计半径；普通 source-empty 和整通道结构常量给予所有轴。未来新 Y 只作该轴完整函数标签或回退值。测试实际改动非空未来源后，预测值保持不变；需求回推与独立 NumPy OR 一致。它是统计性许可，不是严格范围证明。

许可后的真实连接回推为 `need_Z[s,p,J4]=mask AND OR_{t,h,j∈J4}(unresolved[t,h,p] AND E[t,s] AND V[j,h]≠0)`。第一因子取权按源非零与需求在 **T/P4 上的并集**计，不用单门比例代替；prefix 与 tail 的请求分别统计。一个逻辑 U 请求对应 **J4 个 FP32 系数，即 16 B 未压缩载荷**，不能与旧 H8/INT8 请求的相同数字直接等价比较。

`demand_train16/stage1_completion.json` 给出完整第一阶段对照：全共享 U 加项保留 86.95%，普通分组保留 71.15%；但两阶段 U 请求相对各自完整一次扫描分别变成 **1.44846× / 1.30017×**。分区确实多释放生产，二次扫描仍增加请求。完整 V 连续乘加数为 shared 的 **33,675,264** 和 grouped 的 **2,806,272**，U 完整加项均 **19,572,672**，这两种算术不能当作等价的一次加法相加。

第二阶段预声明写在 `demand_train16/definition.json`，然后实际一次运行：五轴相同 256 步、batch8，λ 固定 0.10；损失为标准化 Y MSE＋0.125 完整门平衡 BCE＋0.125 许可/回退混合门平衡 BCE＋0.10 U 请求比。均值/协方差每 64 步只用当前学生 train16 重校并 detach，结束再次 train 校准；不训练 A/bias/θ/γ，不扫描 λ，不人为设置私有变量配额。五轴共约 54 秒 CPU，结果在 `demand_train16/result.json`。

| 第二阶段 valid4 | 完整学生门差 | 漏教师非零门 | 许可相对自身完整门差 | U 加项保留 | 分阶段 U 请求 / 完整一次 |
|---|---:|---:|---:|---:|---:|
| shared | 1.2299% | 16.9063% | 0.02014% | 86.7639% | 1.44619× |
| grouped | 2.1671% | 54.7094% | 0.03072% | 69.0728% | 1.27619× |
| hybrid | 1.1766% | 15.7328% | 0.02146% | 84.9180% | 1.43092× |
| shared_compact | 1.0728% | 13.3937% | 0.01841% | 86.6087% | 1.44716× |
| grouped_compact | 2.1755% | 40.5735% | 0.02228% | 71.0111% | 1.29778× |

**加入费用项的第二阶段中，hybrid 的私有潜变量开始参与执行**：8 区私有 J4 数为 `[1,1,2,0,2,1,2,1]`。相对它自己的第一阶段，U 请求少 1.1846%，但标准化 Y MSE 从 0.07754 上升到 0.09828。与普通 compact 相比也只有局部精度/工作折衷，尚不能认定净收益；grouped 的生产取消更多，同时漏激活更高。额外训练和混合门损失也可能使 private 上线，不能把这一个观测直接归因于费用项，固定 λ=0 对照另存。没有用该局部试验宣判整类 CFMP 无效，也没有把私有变量上线本身叫创新成功。

以上仍未计成完整硬件周期：V 连续乘加、原 BN/T10 阈值算术、均值/半径常量、控制比较、因子/源码跨阶段驻留、TC/TR bank 端口与真实 Conv2 消费仍需后续共同服务模型。首先用保存的完整因子学生测真实网络 AEE，再决定需要何种恢复训练；不能直接把一部分算项下降升格成加速比。

### 固定费用项因果对照

独立评审指出额外训练与混合门 BCE 是潜在混淆，因此从同 stage1 起点补了 **hybrid / shared_compact × λ=0** 两轴，其他损失项、256 步、batch、种子与 train 重校规则完全相同。只加这个指定对照，没有扫描 λ；文件在 `demand_ablation_train16/`。

| 控制 | λ | 完整门差 | 漏教师非零门 | U 请求 / 自身完整一次 | 8 区 private J4 数 |
|---|---:|---:|---:|---:|---|
| hybrid | 0 | 1.11023% | 13.3820% | 1.44928× | 1,1,0,0,0,0,0,0 |
| hybrid | 0.10 | 1.17655% | 15.7328% | 1.43092× | 1,1,2,0,2,1,2,1 |
| shared_compact | 0 | 1.05682% | 12.8458% | 1.45060× | 无私有结构 |
| shared_compact | 0.10 | 1.07279% | 13.3937% | 1.44716× | 无私有结构 |

λ=0 也能使两个 private tile 上线；其余训练变化确实有作用。指定费用项在 hybrid 中额外减少约 **1.267%** 请求，在 compact 中约 **0.237%**，同时两者都付出恢复误差。当前只支持“连接结构与需求目标有可测交互”，不支持“激活 private 已证明硬件收益”。尚未按相同网络 AEE、完整服务/状态和相同可编译优化闭合这项增量。

**独立概念判断仍为 5.5/10，不随实验跑通自动加分。** TC/TR 共 mask、分组低秩、共享/私有表示本身都应当作为继承底座。待证的窄差别是：训练消费者连接后，已完成的 T10 门在真实物理并集下多撤销第一因子生产，且完整费用有净优势。此次观察值得继续完整网络和服务对照；数值还不足以支撑新的主贡献句。

复跑固定第二阶段命令（无需 GPU）：

```bash
joint_completion_20260909/.venv_train312/bin/python \
  factor_completion_20260909/train_demand.py \
  --source joint_completion_20260909/full_capture4/capture \
  --temporal common3 --steps 256 --batch-groups 8 --device cpu
```

命令工作目录为 `algorithm/patch_probe/`。既有结果已齐，不需要为同一参数重复执行。

## 接入真实网络的调用

共同父继续使用 `run_patch_probe.probe.load_system`、同六整数源/粗头，并固定四处 patch BN 到原 train32 常量。单纯 `install_conv1` 不改神经元；从 native 检查点刚建的网络应调用下面的完整适配函数，安装 NPZ 内同一个 common3 A/bias，避免偷偷用另一份 A。

```python
from train_factors import install_factor_stack
adapter, original = install_factor_stack(
    modules[RES + '1.conv1.0'],
    modules[RES + '1.sn2.spiking_neuron'],
    factor_file,
)
```

`adapter` 只输出原形状的 raw Conv1 值。原 norm1 wrapper、完整 T10 `sn2`、θ 幅值、Conv2/norm2、shortcut 及粗读出均由网络原路径执行；不安装统计提前门作为第一次网络评价。`original` 保存原 forward 和 A/bias，方便轴间恢复。GPU Conv 因子与本地 dot 的舍入顺序不同，网络 AEE 必须实际重评；没有从局部门差外推 full825。

条件版本放在独立 `conditional_adapter.py`，不修改根节点的网络评价循环：

```python
from conditional_adapter import install_conditional_factor
pair, originals = install_conditional_factor(conv, neuron, second_stage_file)
# 完成该轴后：
conv.forward = originals['conv1_forward']
neuron.forward = originals['neuron_forward']
```

该配对先从 Conv 原输入产生每时刻 3×3 source-empty，真实 norm1 执行后，神经元使用 prefix Y、空源 BN 常量、该区间 `live_h` 与训练统计；失败组才读取完整 Y 回退。新函数保留连续 θ 输出。密集 GPU 可以先形成完整 Y，这仅是数值参考，不算生产取消的速度。

`conditional_checks.json` 已记录五轴完整 valid4：**4,915,200 门、gate 与 accept 均零差**，整图 T/C/P4 转换定向检查也零差。比较对象是各轴自己的已保存完成器，不是 native 或旧整数门。预编 source-empty 半径表有至多 128 个索引，GPU FP32 表为 480 KiB；当前 common3 每个 t 实际只有 1–2 种半径向量，可压紧成常量，尚未计 ASIC 端口/面积。不能把原样表免费计入 2 KiB 临时状态。
