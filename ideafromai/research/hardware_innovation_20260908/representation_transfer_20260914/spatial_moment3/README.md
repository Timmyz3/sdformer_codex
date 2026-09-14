# Moment3：把已付不起的第四项删掉后的实际适配

本实现采用根已冻结的moment函数，不改变比例或系数。从general Winograd的失败出发：原Q11跨18序列仅0.268%净收益，每tile2832固定税几乎抵消MAC减少。moment的1536个scalar三元组（192个N8/rank组）全部满足 `g1=g0+g2`，因此直接采用已知约束FIR恒等式：`D=(z0−z2,z1+z2,z1−z3)`，`M_j=Σrank g_jD_j`，`y0=M0+M1, y1=M1−M2`。这是约束FIR/快卷积的实现适配，不是新公式。

[spatial_core.sv](spatial_core.sv) 从已锁定general core派生，只保留三个物理系数平面、三个M、三次D运算和两次输出恢复，无第四D/M、无恢复 `/2`。Q2就是原3tap表576个N8向量，未保留原g与general变换系数两份表。Z仍8bank×40row×32bit、同一个公共地址授权：先实际读两行保存四个原Z，用同8ALU做三次加减，原地写D0/D1及D2/零两行；跨stripe psum实际读取再用同ALU相加。额外holding均列于下文。

**同函数实际结果，均截至最后I24。**

| 冷启动一次、连续第一遍 | ordinary moment | general4 moment | moment3 | moment3对general4 |
|---|---:|---:|---:|---:|
| held64 ready | 2334322 | 1975186 | **1898706** | −3.87% |
| disjoint64 ready | 2510814 | 2107026 | **2030546** | −3.63% |
| 18序列36tile ready | 980690 | 908354 | **865250** | −4.75% |
| held64 BP | — | 2060295 | 1983266 | −3.74% |
| disjoint64 BP | — | 2195989 | 2118691 | −3.52% |
| 18序列36tile BP | — | 953063 | 909776 | −4.54% |

跨序列相对同函数ordinary减少11.77%，但本次删除第四项的独立收益应看general4→moment3的4.75%。[comparison.json](comparison.json) 对178个不同fixture逐字核对q1/q2/consumer配置、实际source/origin、raw/identity/J/wide/I24及原Z，全部相同；没有按名字假设同函数。每条ready命令精确减少 **1192 core周期**：少40次D运算、192次cache遍历、960次恢复运算；首配置另少192拍，所以64tile少76480、36tile少43104完整服务周期。general4的Q2有效读、MAC、实际Z变换读写、psum读写都与moment3相同，不能把192次cache遍历误报为192次有效W读减少。960个exact-half操作变成0。

**验证完整且保留负边界。** 15small含8真实、zero/one/random/tail/rank正负/图外poison，两64及36跨序列，ready/BP与无reset两遍连续换源；raw和完整consumer各716命令，共1432命令。每个完整命令实际比较3840个P、J、wide、I24，以及1280个原Z和1280个D物理字段（其中960个有效D、320个显式零padding）。全输出零差，[SUMMARY.json](SUMMARY.json) 的171856项独立计数/逐FSM/配置/重复核验全部通过。M没有逐值RTL monitor；其正确性由CPU三项计算、独立展开卷积、实际最终输出和每次M/恢复的signed32溢出断言共同支撑。

135个旧位置采用新的moment导出gold，另36个跨序列初次回放复用Q11模型在r0.conv2之前不变的实际source/FP32identity，按moment重新CPU生成raw/J/wide/I24。随后根代理在各新函数的完整825评估中重捕了这36个位置：source/identity与母体相同，各自Z/raw/J/wide/I24与独立gold完全相等，见[质量汇总](../quality/comparison.json)。没有借Q11 raw当新函数gold。静态 `D_abs≤17124`，M任意rank前缀≤106086078，恢复界≤154771993，均可无损放在D16/P32；从FP32 identity到J20和I24仍用原消费者的完整RNE/饱和合同。

**强控制也可使用同硬件。** [unconstrained/SUMMARY.json](unconstrained/SUMMARY.json) 将母U2固定为0，物理三个平面直接取U0/U1/U3，未除二raw命名为p2，output_scale减半后重新RNE生成aQ40；不要求它满足native-g约束，也不把它当非法近似排掉。使用同一个已编译RTL、同样178fixture和raw/full各716命令，全部D/P/J/wide/I24通过该控制自己的gold，171856计数核验通过。其每条实际周期、状态、配置和端口计数与moment3都相同，另经[57996项逐记录比较](phase3_control_comparison.json)确认。因而3M硬件收益不是moment投影独占，两个新函数的质量必须独立比较。

另一竞争函数native-tap普通控制的冷36tile为849931，快于moment3的865250；held/disjoint为1942296/2095294。它与moment函数不同，不能混成同函数加速比。根代理的[完整825质量](../quality/QUALITY_REPORT.md)现已完成：moment **1.296504**、native-tap **1.285953**、unconstrained **1.258343**，Q11母体1.254431，同环境NB0为1.447937。自由U控制同硬件却质量更好，因此当前优先保留自由U三项与native普通两个取舍点；moment降为约束消融，不能从早期十帧或局部L2提出抗混叠/光流特性优势。

独立评审要求的共同相邻和前移/普通两tap也已完整实现，见[Box2](../spatial_box2/README.md)。它与moment同函数，raw/full各716命令通过；冷36tile883130，比moment3慢17880拍，两64也慢7.24%/7.93%。前端更新与后端MAC增加抵消了免恢复及更小表/cache收益；保留其状态取舍，不再将此强控制标为未尝试。

**资源与仍存在的税。** producer保持8×32ALU、8个19×13乘法器、10bit源服务、256bit单权重服务、256bit向量/32bit单rank Z读、256bit psum行读写。Q1 4608B；Q2 7488B（物理signed13，moment实际±1023，unconstrained可达±2981）；Q2cache312B；source1920B/native窗口20B；Z1280B；psum15360B。M共96B，其中64B是相对原32B acc的新增；transform_tail32B、原z_hold32B、q1_hold8B；position支持80B、Q1/Q2 live各72B、block/remaining共48bit，另有控制和输出hold。对general4减少Q2表2496B、cache104B、一个M32B、Q2live24B及2B掩码。每tile仍付80次原Z扫描、80次transform读/写、120次D ALU、960次恢复、480次stripe加法；相对ordinary的固定税由2832减到1640，并非消失。

消费者实际保留原FP32→J转换、8×32×32乘法、8×64宽链、单context和同端口/背压。完整static配置是Q1 576+Q2 576+consumer24=1176拍，256bit总线37632B；每tile源1536+origin1+start1共1538拍另付。冷账为实际 `Σconsumer_cycles+static_once+1538×tiles`，第二遍权重驻留、source/origin重载；没有默认每tile重装权重。无EDA，不能从cycle宣称等Fmax或面积已证。

复现：`python3.12 -B implement.py`、`prepare.py`；`run.py` 和 `run.py --consumer` 先跑small，随后对held/disjoint/sequences加 `--skip-build --stage NAME`；最后 `verify.py`、`compare_controls.py`。unconstrained先 `prepare_control.py`，同run命令加 `--input unconstrained --skip-build`，最后 `verify.py --input unconstrained`。使用 `/opt/anaconda3/bin/python3.12`、Verilator4.028，最终raw/consumer build均无Warning/Error。旧树、生产、GPU、训练、EDA与Git均未修改或执行。
