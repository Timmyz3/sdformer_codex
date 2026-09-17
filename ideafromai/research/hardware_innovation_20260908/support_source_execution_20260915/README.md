# 支持码的真实源生产、完整消费者与失败适配

本轮把先前未计的源 PSN/最近码投影做成 RTL，再连接实际 FC1→后级完整 PSN；并在失配处实际测试响应约束选择、子集查表和证书。**尚不能称 TCAS-II 强接收证据或生产重构完成。** 子链周期来自隔离 Verilator；另有一组明确限定到单判决 lane 的28nm标准单元DC比较。没有完整核VCS/PT/Formality、完整PPA或整网周期，网络精度正用A800独立补测。

最新决策读 [STAGE_NET_RESULT.md](STAGE_NET_RESULT.md)。**证书PSN已接真实FC1、唯一Y和H96门接口，完整冷启动收益为4.32% ready / 3.90% BP**，见[joined_fc1](post_psn_subset/joined_fc1/README.md)；独立叶34.26%不是整链收益。[阈值坐标实现](post_psn_subset/normalized_certificate/README.md)完成496条命令及49bit边界验证，周期不变；[独立单lane DC实验](post_psn_subset/operator_timing/README.md)最坏到达时间降低6.44%、单元面积降低5.27%，不能外推完整核频率或面积。[距离界RTL](source_distance_bound/README.md)实测慢约7.9倍，固定比较器布局停止。[完整patch两因子RTL](patch_tc_tr_next/rtl/README.md)已有4,448条命令回放；区域紧凑执行约省8%，时间变换前移在该区域学生上更慢，最终资源表达和普通对照审阅另见其报告。

源端最终普通对照已经补齐 [P32内参数驻留](frontier_joined/retained_config/RESULTS.md)，因此下文较早的约3.3%或2.5% BP优势不再作为最终强控结论。普通frontier最终相对static省10.80%/1.42%，非零class只再省0.30%/0.40%。零响应强控为[同权限完整链](zero_response_reopen/frontier_retained/README.md)，约1.8%增量。下面保留各次适配的历史结果与原因，不把它们再当最终最强分母。

## 1. 源端：先加强普通对照，再判断增量

[source_classifier.sv](source_classifier.sv) 接受真实 X24、A16、τ48 和静态 D/图，从 RTL 的10路 MAC 产生源门，执行最近 Hamming 的 code/class 分类。TB 不提供门、码或证书。字典只有64个信息通道，另外32个公共距离项全部基线可删。

[source_table.md](source_table.md)、[完整 CSV](source_cycles.csv)、[汇总](source_summary.json)、[独立审阅](SOURCE_RTL_REVIEW.md)。67病例×40配置＝**2680任务**；160800个输出标签、1750820个实际已计算的U/门各自核对。真实部分64个包来自两训练帧各固定32个抽样位置，不是64个独立帧或连续P32空间tile。

| 强基线/适配，D-only熵序 | ready周期 | 背压周期 | 源T10任务 | 128bit字 ready/BP |
|---|---:|---:|---:|---:|
| static64＋下一X预取 | 83008 | 88256 | 4096 | 10112/10112 |
| 普通code图，64bit节点、无预取 | 74588 | 99672 | 3359 | 14731/14731 |
| 普通code图，32bit节点、双分支预取 | 71509 | 87055 | 3359 | 15636/15630 |
| 旧最近响应class，同压缩/预取 | 71385 | 86537 | 3353 | 15491/15491 |

同一核：10个16×24乘法器、10个48bit累加器，2级乘积流水；8bank×128bit每bank最多1笔在途；128B共同cache，静态源预取复用其中空间。当前通道另有32B X holding，图当前节点/根/rank、A/τ等均在RTL寄存器中。静态64不读图配置；图不读无用的D/mask参数。所有模式冷装自己实际需要的参数。图按物理128bit词读取，32bit节点非免费SRAM全表组合查找。

独立审阅发现并修正了普通static可提前读取下一X的遗漏。补齐后，普通图相对static的净减少为**13.85% ready / 1.36% BP**，不是先前未加强分母下的约16% BP。普通图的寄存器cache、排序和预取均归借入底座，不能写成响应语义的新贡献。

## 2. 失败适配：近响应不等于少生产

[source_class_adapt](source_class_adapt/EXPERIMENT.md) 改为按首训练帧的“整数W改变量/可省T10源任务”选每组至多一对响应约束；第二帧不参与选择。630对是离线编译枚举，不是630个硬件机制或训练试验。

真实重新投影的W″、完整H384响应类及全部16bit输入校验后，再用原RTL跑2680任务：源任务3359→3266，ready71509→69698（−2.533%），BP87055→84701（−2.704%）。这是相对普通code图的增量；不能把相对static的16.04%全部归它。实际只改五组，旧方法改六组；平方权重扰动较旧方法多4.33%，局部门翻转644→672/245760，没有新AEE。普通整数L2-only选择恰好与旧W′完全一致，完整输入/图二进制相同，复用旧控制结果，没有重复跑相同实验。

[独立审阅](source_class_adapt/REVIEW.md) 将当前创新暂评3/10：候选是“消费者响应等价反推源生产义务”，但特征获得费用、整数约束和图归约已有先验，现有增量尚薄。扩大输入与真实消费者结果分别见[扩展输入](source_class_adapt/expanded_sources/SOURCE_INPUTS.md)、[整链](joined_chain/PLAN.md)。

扩展到全部32个训练帧、固定P0..31后，这一判断进一步收窄。[实际扩展结果](source_class_adapt/expanded_sources/RESULTS.md)：31个未参与pair选择的训练帧上，新class相对code只省1.592%/1.651%，对static却在BP下慢2.983%，25/31帧为负。它们不是留出验证帧；两帧的小表不能作为代表性结论。

## 3. 改生产单位、修复重读，然后接完整消费者

[frontier_source/README.md](frontier_source/README.md) 已将“一次通道产出全部T10”改为按需 `(t,c)`，最多四个当前通道共用十个MAC。每个有效目标仍完整计算十个非因果源项；未计算U通过valid mask隔离。新增128B X holding相对原核多96B，普通code/static基线同享。

这版先测到重读：992个非pair选择训练包上，plain-frontier code虽然少算MAC，BP只比static少209拍。四槽驻留将额外重读channel从14512减到799，code相对static ready/BP减少17.251%/2.954%。[独立审阅](frontier_source/REVIEW.md)确认数值/端口，但创新仍暂评3/10：普通多路径图遍历和驻留解释主要收益；新pair选择相对旧class只再省约0.2%。

[frontier_joined/RESULTS.md](frontier_joined/RESULTS.md) 把部分生产接进真实D展开、FC1 H384与完整T10 PSN，同时将code/class根配置统一物理bank。840个完整命令，Y/U/gate各103219200次核对通过。在31训练帧固定P32上：

| 同W″函数，最强普通后端均内容去重 | ready周期 | BP周期 |
|---|---:|---:|
| static64+下一X预取 | 1989440 | 2179175 |
| 原整T10 code | 1879645 | 2232487 |
| resident-frontier code | 1767487 | 2124840 |
| resident-frontier class | 1761921 | 2118445 |

普通frontier对static省11.157%/2.493%；class相对同权限frontier code仅再省0.315%/0.301%。这是真实子链服务，不能移作完整层/整网或RTL准入加速比。物理资源为源10+后端96个独立乘法单元；256KiB是参数/X池，**还另有**后端Y90KiB、routes7.5KiB、U5.625KiB、3840B门桥及其它局部状态，不能把参数池写成全设计总存储。

另有一个明确未迁完接口已实际补试：[active-prefetch](frontier_source/active_prefetch/README.md) 将旧单通道child PF改为当前active前沿。容量和选择不变，普通code在源叶BP再省1.253%，却多读约11.7%图词；整链独立复测位于[frontier_joined/active_prefetch](frontier_joined/active_prefetch/PLAN.md)。普通code共同享有这一供数优化，不归class标题。

## 4. 后级最贵PSN的实际负结果

[post_psn_subset/README.md](post_psn_subset/README.md)：两组5bit普通distributed arithmetic＋同核门证书，296命令全部功能通过。32real的完整subset PSN为285126拍，cert为428572拍，**慢50.31%**；native96MAC为118344拍，是异资源参照。位平面虽少45.19%，新增判界244988拍超过省下的前缀/查表101542拍。这版H8×T10、80路ALU布局停止；没有将其当完整BitL迁移或否定所有DA/证书接口。

## 5. 数值、输入和复现

AT-LIF仍为{0,θ}，θ已在下游权重合同中吸收。源PSN的X是连续输入，本次单独定为X24 Q16、A16 Q12、τ48 Q28，无中间RNE。此新定点源在全32×512缓存上相对FP32改变1093/15728640原门、887个投影门，不能直接继承旧学生AEE。后级A是另一份真实Q14矩阵，不能混成源A。

[SOURCE_INPUTS.md](SOURCE_INPUTS.md) 给出训练缓存、形状和重建。默认再生已改成Python3.12纯NumPy；首次历史GPU核对用了3.10，已如实记录并完成3.12独立再生，不重标执行环境。

```bash
/opt/anaconda3/bin/python3.12 prepare_sources.py
/opt/anaconda3/bin/python3.12 build_prefix_tables.py --source-cases source_cases.npz
/opt/anaconda3/bin/python3.12 build_prefix_tables.py --source-cases source_cases.npz --order entropy
bash run.sh 67
/opt/anaconda3/bin/python3.12 summarize.py
```

融合未完成项按[FUSION_REMAINING.md](FUSION_REMAINING.md)保留具体接口：完整Phi外层/packer、完整BitL常数图、patch TC/TR双消费者、固定decoder2 BN后的细节分支等。没有以读过论文或做过局部叶子宣称整套作者方法已搬完。生产nts07、论文、docs359和H81未修改。
