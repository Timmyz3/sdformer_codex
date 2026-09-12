# full-D 输入侧有费预测核：可集成，D=0 无门读

`predictor.py` 已实现 `b[t,h]=c[t]+sum_j D[t,j]*g[j,h]`，输出十个H8向量到RF40..49，RF50保存本H8的八个uint16 T10门词。输入门必须已由真实生产者写入SRAM；预测函数不生成门、不编码Q8、不做RNE/sat，也不代替独立的PED U。

```python
from predictor import install, pack_constants, bind_predictor, predict_h8
install(EncodedMachine)  # 在该类的integer_op覆盖定义完成后调用
D_blob, c_blob = pack_constants(p['D'], p['c'])
m.dma_input(D_blob, D_base, True)  # root负责一次真实cold fill
m.dma_input(c_blob, c_base, True)
bind_predictor(m, D_base, c_base, p['D'], c=p['c'])
regs = predict_h8(m, pixel_gate_base + 2*h0, D_base, c_base)
# regs == tuple(range(40,50)); 返回时所有结果已完成原两槽WB。
```

`h0`取0,8,…,88；每pixel为96个连续little-endian uint16门词。`D_base`与`c_base`须32B对齐、不重叠，示例65536和65952。D载荷为**D.T按[j,t]排列的100个signed32，共400B**；c为10个signed32，共40B；`pack_constants`不自行padding，原DMA负责最后一字padding。D的实际值必须在signed19范围。

绑定只保存静态非零坐标，不保存D/c数值供算术使用；绑定时逐字节断言实际已填coef图像与传入该case的D（以及可选c）相同。这些直接内存读取仅用于参数一致性断言，不能供预测运算。该coef区若被改写，须重新绑定。

## 操作与费用

1. 每个c从真实CR256响应中选择，广播到八lane，再执行一次已有ILOAD写RF40+t。CR的原单字缓存权限保留；没有host常数预测数组注入。
2. D非零时，H8门经过两次SR64请求/响应，收集到既有`common_staging[0:16]`；再付一条collector控制issue及一次ILOAD写RF50，并等待两槽WB。其余`common_staging[16:64]`逐字节保持，未新加状态、端口或第二issue。
3. 每个有静态非零项的j，先付一次RF50读取/bit选择/整H8是否为空的判定issue。若八lane均为零，跳过整列所有D加法和该列系数读取，判定费用仍在。逐lane部分零只掩蔽该lane加数，仍执行整条SIMD8操作。
4. 新模型操作`IADD_GATE_CONSTANT`读取**accumulator RF与门RF两个向量**，从当前CR响应取得一个signed32 D常数，按门位选择D或0并加到acc。没有第三个RF读，也没有19bit乘法。结果通过原Machine单issue、两槽整数WB和原仲裁；每个实际移位/加法值都保持signed48。
5. 返回前只等待RF40..49完成，不擅自重置Machine、时间、计数、phase或其他RF。下一H8调用会覆盖40..50，因此encoder须先消费当前输出。

D=0时 `gate_address` 可为 `None`，不会做门地址计算、SR读取、collector、门ILOAD或门bit判定；fixed/affine仍按所需c执行初始化。c=0的普通fixed还可由caller采用静态零初始化/直接零基值路径，不能把本通用c加载费用作为fixed必需下界。

## 已有整数接口的独立核对

两个参数父来自`breadth_20260912/representation/parameters/{ordinary,lifting_raw}.npz`；U来自同父`stage_20260912/algorithm/hardware_exports/*/deployed_constants.npz`，形状24×96。它们是旧R24表示父，不是新320步或两项源学生。

full-D两父都是100个非零项。ordinary D范围−178330…221476，c范围−9011…3311；lifting_raw D范围−165914…221454，c范围−9533…3032。二者full-D step均为2048。`audit_integer_interface`按所有1024门码与所有signed8残差码范围检查，两个父的`Dg+c`和`Dg+c+step*q`都不越signed24；此证明只对既有参数成立，helper本身不插入饱和。

encoder仍必须执行部署定义：`q=clip_signed8(RNE((x-b)/step))`，重建为`sat24(b+step*q)`；不能先舍入/裁剪Dg再编码。完整U前累加的三角界也由helper输出，须小于2^47；U/V的原RNE/sat和bias由caller保留。

`Ug`实际是PED U乘T10门的结果，不等于其他projection卷积已经计算的量。其通用绝对界为ordinary555020、lifting_raw634283。latent侧`D(Ug)`仍是整数常系数CMVM，不能把signed19 D当现有signed16乘数；本文件的条件加法只适用于二值g，不能用在多位Ug上。

## 有界功能smoke

执行命令：`PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python3.12 predictor.py`。只输出JSON到stdout，不另写结果文件。两父各测试fixed/affine/full-D，四个定向H8门组包括全零、全一、分散one-hot及含最高两个时间位的混合门，共**1920个预测值全部0差**，与独立整数`D @ unpacked_gate + c`比较。

每父的四个full-D H8合计484个post-fill模型槽，8个SR64读、44个CR256读、4个collector、4个gate-ILOAD、40个有费bit判定、280个条件加；12个空列判定取消120个D加法。fixed/affine四组各64槽、门读/门load/判定均为0。这些数值只说明定向门smoke的真实操作计账，不代表真实halo命中或完整编码/PED服务。

所有检查保留原两槽WB及共享Machine计数，核对16B collector未覆盖余下48B staging。没有GPU、训练、EDA、生产RTL、新AEE或新函数。

## 普通强对照和借入完整度

当前是直接条件加法基线。只采用静态零D项删除、一次有费j谓词复用和已有单字CR缓存；**未声称执行完整da4ml CSE、Phi、Prosperity或旧5+5 LUT decoder**。常数RF缓存、相同列/公共子表达式、输入侧已测condition-add/LUT路径都应得到同资源权限；若提供缓存，必须计cold load、实际活RF和两读口实现，不能把acc＋gate＋系数三RF读藏进一条ADD。

latent侧现已由`latent_cse.py`补齐两个既有D的完整常量CSE图执行，下面说明借入程度、存储和实际费用。直接分解结果仍须同这条普通CSE控制比较，不能将未经普通CSE的直接分解当最终强控制。

## 完整D图的可执行普通CSE控制

```python
import latent_cse
plan = latent_cse.get_plan(axis, p['D'])  # exact deployed D binding
m.dma_input(latent_cse.control_image(plan), 73728, True)
latent_cse.bind_control(m, plan, 73728, spill_base=98304)
# Uq: RF0..29; Ug: RF30..59; Uc must already be added or reloaded later.
for hg in range(3):
    latent_cse.execute_latent_cse(m, plan, hg)
# Uq[t*3+hg] now holds its old value plus sum_j D[t,j]*Ug[j,hg].
```

控制和spill区由caller一次绑定；spill无需冷填，每次读取都有此前实际写入。控制高区73728起，SRAM98304起，最大使用的区间均不触及root的RECON110592和CODE114688。`get_plan`核对传入D与该父烘焙D完全一致。helper通过实际CR256响应解析每条128bit控制记录，不从host D或host中间值取得数值；同一CR256可服务两条记录，原单字缓存保留，所有miss计原请求/响应。控制冷填由root真实DMA计费。没有第二个免费ROM，原源ROM容量未扩充。

使用现有da4ml0.6.0环境和同源`ordinary_source_cmvm.solve/flatten/precise_domain`，固定`wmc/auto`、既有hard/decompose设置，对每父完整10×10 D求图；所有289/283个整数addsub节点、十个输出的移位和符号均保留，不添加RNE、不按行砍图、不重新训练。已有完整图在源码中烘焙，普通Python3.12即可运行，运行时无需da4ml。图的所有节点和移位操作数在完整signed24输入盒上的保守最大宽度均为43bit；每次真实运算和signed48存载再次检查范围。

初始32个scratch的圈法过于保守，已按root实际生命周期修正为RF60..95的36个寄存器，并允许当前hg的十个Ug输入在最后读取后回收；其余两个hg的Ug保留，Uq三十个结果保留。固定last-use/height优先顺序下，无spill分别需要127/120个总RF；**这是此顺序的需求，不是该D图的最低需求，也不是96RF无法执行的证明**。

为完成现有图，沿用完全相同的299/293条算术顺序，增加确定性的最远下一次使用spill分配。最多使用36个scratch加当前hg十个Ug物理位置，共46个工作RF，其他50个RF保留；不扫描排程、图分解或RF规模。每个spill值是完整8×signed48共48B，保留干净SRAM副本直到该值最后一次使用，允许再次驱逐时复用该副本。store付一次RF读取到既有64B staging的控制issue加六次SW64；reload付六次SR64请求/响应、一次48B collector issue、一次ILOAD及原两槽WB等待。使用`common_staging[0:48]`，保持末16B，失效此前packed系数latch。没有24bit截断或host spill缓存。

| 已有父 | 完整图addsub + Uq加回/H8 | spill写/读向量/H8 | 控制字节（DMA补齐） | SRAM spill字节 | 三个hg ready槽 | 三个hg stress槽 |
|---|---:|---:|---:|---:|---:|---:|
| ordinary | 289 + 10 | 36 / 38 | 5968（5984） | 1584 | 4617 | 5357 |
| lifting_raw | 283 + 10 | 31 / 31 | 5680（5696） | 1248 | 4104 | 4603 |

以上槽数从真实输入RF已加载、控制DMA已填完后开始，包含控制读取、所有spill和reload、算术及等待，排除caller之前的Uq/Ug计算和控制冷填。ready两父各240个广范围定向signed24输入结果全部0差；stress重复两父各240个结果仍0差，仅端口等待增加740/499槽。每个hg完成后逐组核对其余未执行hg的Ug保持；最终Uq与独立整数`initial+D@x`一致，staging末16B canary保持。此处smoke是完整D计算核的功能与计账检查，不是完整真实halo服务收益。

这条控制完整借入了da4ml对既有D的常量图，补了普通last-use回收及有费spill，但未声称最优排程、最小RF、最小spill、跨Ug生成与D图联合CSE，或完整复现其他动态复用论文。当前保留此可执行接口作为full-D的普通强对照；实际新接口是否值得保留，由root同函数、同真实编码前缀和U/V尾部的测量裁决。
