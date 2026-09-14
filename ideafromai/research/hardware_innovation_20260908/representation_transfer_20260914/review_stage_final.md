# 本阶段最终审阅：三乘积、原生 tap 与 phase3 强控制

2026-09-14，已纳入最终825质量与Box2实测。结论：**本阶段完成了有价值的失败定位和完整实现适配，仍不足以把经典三乘法或 Winograd 结构剪枝包装为新算法投稿。** 当前保留的质量/周期取舍点是自由U-zero的3M与native_tap ordinary：前者精度更好，后者36个跨序列tile稍快。moment与自由U-zero在同一3M程序上逐记录同周期，但完整825质量更差，故moment降为约束/归因对照；同moment的Box2强控制也已实做，不能再列为待办。详见末尾闭环与唯一下一接口。

审阅范围为 [spatial_moment3 的SV](spatial_moment3/spatial_core.sv)、wrapper、prepare/verify及两个SUMMARY，[pruned_replay](spatial_winograd/pruned_replay/README.md) 的prepare/结果与父ordinary支持路径，以及 [phase3_endpoint.py](quality/phase3_endpoint.py) 和继承的 [整数消费者](spatial_r16_integer/torch_integer.py)。这是对实现、资源和归因的静态审阅；没有新增RTL、训练或跑回归。pruning数值导出由本代理完成，本报告不把作者自身读回称为独立第三方数值复验；实际RTL与整网验证由其它负责人完成。

| 对象 | 借入A及完整程度 | 本轮实际解决的B、执行增量 | 最强反对 / 暂评分 |
|---|---|---|---|
| moment→专用3M | 受约束三tap FIR、快卷积、Winograd域零组。已接完整Q1→raw→FP identity/J→wide→I24，不只是乘法叶 | 通用四M虽然跳过U2 MAC，仍付第四D、完整恢复和32词cache遍历。专用核确实改成3D/3M、24词cache、两次恢复，无中间除2 | 这些删项是固定零分量的直接代数简化；同样收益也给自由U-zero。算法新颖性2/10，具体系统接口3/10 |
| unconstrained→同3M | 成熟变换域直接置零；同192个m2整N8组，保留母U0/U1/U3。完整raw/I24与同一硬件已通过 | 独立两相位算子，保留p2并重定a，证明无需新增除法/舍入器；这是必要强控制的补齐 | 不能说它不合法、必然更贵或抢占额外运算资源。主要是成熟A与公平控制，2/10 |
| native_tap→ordinary | 原生结构tap剪枝及真实支持驱动MAC。每组一个共享tap零；ordinary与general Winograd均实际回放 | ordinary的q2_live与Z支持交集真跳MAC/W读，足以吃掉大部分通用快卷积机会；失败原因有逐状态数支持 | ordinary没有被故意限制成dense MAC；此臂是应保留的强普通基线，不是新标题，1–2/10 |

3M源码未见明确的功能/资源越权错误。Q1仍使用原双P15与唯一8条32bit carry链；两个原Z字先分别读到z_hold/transform_tail后，三个XD状态才计算和覆盖，低/高16位写回，第四高半显式清零。80项position_live在Q2前全覆盖；BASE_MAC仅选中rank bank读32bit并以signed16扩展到19，与唯一8个19×13乘法器、同ALU共用。M0复用acc，M1/M2各32B；恢复第一个输出后，第二个从尚保存的M1−M2重新起算，未误用已覆盖的M0。第二stripe真实p读、另拍ALU累加及写回，480行第二stripe全部完成才原序退休。raw拒绝时DRAIN_SEND保持，consumer完成前wrapper禁止换源/配置/重启。

实际3M存储合同为Q1 4608B、单Q2表7488B、cache312B、Z1280B、p_mem15360B、source1920B；M共96B，其中比ordinary原acc多64B，另transform_tail32B。20B窗口之外原本还有10B src_masks保持；Q1/Q2 live各72B、position支持80B等仍计。没有保留原g和变换系数双份；unconstrained使用同一表的物理3M系数。8×32ALU/8×19×13mult及256bit W服务相同，消费者另外8×32×32mult/8×64ALU也实接且统一计入，不能省略消费者称算术相同。相对父通用四M，数据/支持分配确实缩小；没有EDA或等Fmax/等面积证据。

专用3M相对**同moment函数**通用四M，每tile真实少1192拍：少192个VLOAD/cache写、40个第四D计算、960个恢复ALU状态；冷流再少一次192拍系数配置。它不是额外省下1192次MAC，也没有免费免掉全部Z变换/恢复。静态参数完整consumer首命令1176拍，其后逐tile source/origin1537拍、start1拍，所有输出直到最后I24均在服务数内。ordinary强控沿父更大的共享controller存储预算运行；独立ordinary实现还可去掉闲置general状态，故不能把下表直接当等面积比较。

| 完整消费者冷流 | held64 ready / BP | disjoint64 ready / BP | 18序列36tile ready / BP |
|---|---:|---:|---:|
| moment 专用3M | 1898706 / 1983266 | 2030546 / 2118691 | 865250 / 909776 |
| unconstrained 同3M | 1898706 / 1983266 | 2030546 / 2118691 | 865250 / 909776 |
| moment 通用四M | 1975186 / 2060295 | 2107026 / 2195989 | 908354 / 953063 |
| native_tap ordinary | 1942296 / 2023740 | 2095294 / 2180333 | **849931 / 892884** |

数字分别见 [moment SUMMARY](spatial_moment3/SUMMARY.json)、[unconstrained SUMMARY](spatial_moment3/unconstrained/SUMMARY.json)、[逐周期相同控制核验](spatial_moment3/phase3_control_comparison.json) 和 [pruned replay主表](spatial_winograd/pruned_replay/README.md)。两个3M函数各raw/full consumer均716命令、各2749440个输出；各171856项状态/费用检查通过，同源两函数57996项比较一致。pruned replay两函数/两模式/raw+consumer共5728命令通过。M仍是前缀断言和CPU算术，没有逐M RTL monitor；36记录是同Q11上游source/identity、按新函数重算gold，不等于36次新模型整网捕获。两64属于同帧区域，不能充作128条独立序列。

phase3端点合法性单独成立。tile模式4×4源经竖Q1得到2×4 Z；整图模式仅竖直pad，随后Z左右各pad1，按全局偶数x取4个滑窗位置并交织even/odd输出，和导出phase锚点一致。其物理系数字段明确为physical_coeff3，构造器拒绝q2，不能把该数组送ordinary三tap路径。代码输出p2=M0+M1、M1−M3，全程不除2；output_scale已减半且a_q40由新scale重新RNE，b和J不变，继承消费者仍只做末端一次RNE26/I24。母a有51个奇数，26通道新a不同于母a整数向下除2，这一差异已由配置真实支付。FP64直接点积只充当整数oracle：局部禁cuDNN、不靠round修饰输出；固定模型的D/M/p2全前缀界小于2^31、wide小于2^63，也远小于FP64精确整数范围。全图要求偶宽；当前宽320满足，未声称支持奇宽或任意相位裁剪。

这是带两相位的有损线性算子，绝不是伪造的平移不变1×3卷积。相对母函数，raw同尺度误差恰为−M2/+M2，一对输出和守恒，但单像素值改变；允许奇数p2。135+36的656640个raw与两相位展开全部一致，不能把“相位性”当成算不出来。另一方面，局部I24 L2更低也不证明AEE更低。

质量最终已直接读取三个新函数的完整825报告，frame AEE为 [moment 1.2965040894397415](quality/moment/deployed_valid/spatial_integer_summary.json)、[native_tap 1.2859525683870776](quality/native_tap/deployed_valid/spatial_integer_summary.json)、[unconstrained 1.2583431021574247](quality/unconstrained/deployed_valid/spatial_integer_summary.json)；[母Q11](quality/q11/deployed_valid/spatial_integer_summary.json)为1.2544305372701712，同门NB0为1.4479366656。相应pixel AEE为1.2280668950509641、1.2217265438507465、1.1916563897482657，母Q11为1.1930135413322505。moment十帧1.21015偏乐观，不能继续据此声称抗混叠约束获得质量优势。三个新函数的36条live捕获与导出gold也已由根核齐，见 [质量汇总](quality/comparison.json)；先前硬件记录仍准确称“复用同上游source/identity并重算gold”，现在增加的是整网捕获交叉证据。GPU wall time不是硬件加速分母。

最近邻边界：小tile双线性分解和F(2,3)是既有快卷积；本轮3M式由固定零分量直接简化而来，不能称新最少乘法公式。[Lavin/Gray，CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/Lavin_Fast_Algorithms_for_CVPR_2016_paper.pdf)。按Winograd子矩阵的行/列向量剪枝、平衡和速度/质量选择已经有WINS；本轮N8分组/全删一项是本机粒度的迁入，未完整移植其GPU平衡、训练或系统，也不是WINS的新首创。[WINS，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Park_WINS_Winograd_Structured_Pruning_for_Fast_Winograd_Convolution_ICCV_2025_paper.pdf)。空间约束使变换域稀疏与量化Winograd也已有先验；当前无中间RNE、固定界和二值前级/连续后级放置是具体差异，但未见证据足以说明该差异不可直接组合得到。[Spatial-Winograd](https://arxiv.org/abs/1901.02132)，[Wino Vidi Vici，WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Mori_Wino_Vidi_Vici_Conquering_Numerical_Instability_of_8-Bit_Winograd_Convolution_WACV_2024_paper.pdf)。先前定向核查没有找到primary明确覆盖“保二值竖Q1 AAC，只对连续横Q2快卷积”的完全相同接口；这既不能推出首创，也不能以一篇低秩/DSC摘要断言全部已被覆盖。

补核两条更早且直接的primary：2017论文已把普通卷积层改成Winograd层，直接学习和剪除变换系数，并实现稀疏执行；2018论文则把ReLU移入变换域并剪变换权重。[Enabling Sparse Winograd Convolution by Native Pruning，2017，核摘要](https://arxiv.org/abs/1702.08597)，[Efficient Sparse-Winograd CNNs，ICLR 2018，核摘要](https://arxiv.org/abs/1802.06367)。后者原站标ICLR，不能写成CVPR。这些覆盖自由U剪枝这一借入A，本轮固定U2、两相位p2和scale/2是本接口的数值/部署适配，未复现其训练或原硬件，也不另算新剪枝思想。2019 spatial→Winograd结构传递同样要求把“投影后保普通卷积表示”降为具体约束选择，不能自动视为X。

现在缺的是能在强控制之上成立、可复用到其它函数的执行机制及相应归因。现有证据完成了“通用迁入收益微弱→定位固定税→专用化消税”这条路线；只把它称作硬件感知、广播组或scale融合，不足以新增论文贡献。完整825现已补齐，仍不自动提高上述新颖性评分。保留快卷积/稀疏家族与已有正结果，当前低分针对现有贡献边界，不是家族否决。

本次提出的唯一补试“Q1前共同相邻和→ordinary两tap”已经实际完成为 [spatial_box2](spatial_box2/PLAN.md)。因g1=g0+g2，`g0+g1 x+g2 x²=(1+x)(g0+g2 x)`；令 `E_j=Z_j+Z_{j+1}`，用普通 `p_j=Σr(g0 E_j+g2 E_{j+1})`。它是成熟的同函数强控制，以下是独立最终增量，不把这条旧建议再留在未来工作中。

Box2静态核到实际 [SV](spatial_box2/spatial_core.sv)、wrapper、TB、[prepare](spatial_box2/prepare.py) 和 [费用核验](spatial_box2/verify.py)：source仍逐C从原1920B存储实读16个10bit字到20B窗口，按ky形成原10B src_masks保持；两行合计六个有效相邻门对的OR表示非零、AND表示count2，两个padding位置置零。TIMESEL只锁存当前两2bit count。每lane两个字段选0/W/2W，其中2W是16bit正确符号扩展的左移接线；仅ZADD在bit16切断carry，MAC状态恢复完整32bit进位。因此count2不需要第二次AAC，也没有新增乘法器或count SRAM。

边界与映射没有发现错误：L_LOAD按原source_origin检查240×320物理界，图外poison先归零；必要E2始终由原col2+col3生成，只有人工第四E列为0。低/高16bit分别承载E0/E1及E2/0；Q2输出x=0取E0/E1，x=1取E1/E2，权重地址`stripe*192+og*16+term`对应真实384个N8向量。E支持在80项扫描后全覆盖，第二stripe从原p_mem读回再累加，最终480行仍原序输出。静态E及所有前缀界[−17124,14854]在signed16，Q2任意前缀140756698在signed32；原a/b与I24保持moment函数。prepare实际验证count→E、原Z邻和→E、两tapP、原三tapP、expanded-W与J/wide/I24，并逐文件对齐已冻结moment3的178个fixture。

资源增加/节省均可指认：仍唯一8×32ALU/8×19×13mult、同256bit W、同1280B E与15360B p、原完整消费者。Q2表4992B/cache208B，比3M分别少2496B/104B；不用额外64B M和32B transform_tail。新增的是六组10bit OR/AND、16个0/W/2W字段mux和活动count多2bit；10B src_masks在20B窗口之外已经列入最终SUMMARY。实际完整consumer冷配置984拍，逐tile source/origin1537拍及start1拍不省。未做EDA，静态字节节省不能直接折算等频面积/功耗优势。

| Box2 同moment完整消费者 | held64 | disjoint64 | 18序列36tile |
|---|---:|---:|---:|
| 冷ready | 2036236 | 2191506 | 883130 |
| 冷BP | 2116792 | 2276349 | 925841 |
| 相对moment3冷ready变慢 | 7.243% | 7.927% | 2.066% |

结果见 [SUMMARY](spatial_box2/SUMMARY.json) 与 [同函数差分](spatial_box2/comparison.json)：raw/full各716命令全过，每种2749440个raw或I24输出，158968项状态/费用检查通过。ready逐命令精确式为 `Box2−3M = ΔQ2_MAC + 3ΔQ1_issue −1832`，冷配置再减192一次。36记录中Q1 issue 39550→45894增加6344，付19032拍；Q2 MAC 217296→282288增加64992；免固定65952与配置192，净**多17880拍**。新增Q1事件已从真实source独立定位为每行`t`的`s0=s1=0且s2=1`：原首更新组`s0∨s1`变成`s0∨s1∨s2`。source/Q1权重读不变，Q2读16092→10728、cache写20736→13824却不足以抵新增事件和Q2乘加。这是已经实测并定位的控制失败；Box2仍比同moment原三tapordinary快，不能据此否决共同因子或前移变换家族。

最终归因：在当前AEE与这些冷周期二维指标上，自由U3M同时优于moment3与Box2；Box2静态存储更小，未测PPA前不宣布在所有资源目标下劣于。主取舍点维持自由U3M（AEE1.258343、36冷865250/BP909776）与native ordinary（AEE1.285953、36冷849931/BP892884）。自由U的更好结果不增加新颖性：算法仍2/10、具体系统接口暂3/10，Box2成熟控制1–2/10。moment的平移不变表示只能保留为模型约束属性，不再声称质量或独占硬件收益。

**现在唯一最有价值接口：把上述空间候选接入与R8相同的双context资源仲裁和完整消费者。** 以现自由U3M与native ordinary各自冻结函数进入同一两context合同：source/weight/Z/psum的每个grant、拒绝保持、每context状态归属和输出退休均实建；唯一8ALU/8mult、相同外部供数/BP及I24资源，context和模型存储容量如实列出。然后用相同36序列流比较R8与空间候选的冷/暖完整服务，按各自完整825质量列取舍，不能把当前单context空间周期与旧双context R8直接相除。这个接口能回答现阶段最重要的未闭问题——空间方案在公平流水/并发条件下是否仍有价值。它本身也是需要完整迁移的成熟底座，不预先赋予高新颖性；其余未试接口保留，本报告不再追加第二个实验建议。
