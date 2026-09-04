# H67/H68 ASIC 自主研究日志

## 2026-07-12 启动

- H67 epoch19 dyadic valid825 已完成：AEE `1.4626`、AAE `9.3949`、spikes `26.3948G`。
- H68 epoch19 float valid825：AEE `1.4688`；dyadic valid825：AEE `1.4715`。
- H68 部署配置关闭 training-only matrix auxiliary，初步判断不需要新增矩阵硬件。
- 当前环境具备 Icarus、Verilator、Yosys，不具备 `dc_shell` 和标准单元 `.db/.lib`。
- 暂以 28nm、500MHz、0.5mm2、100mW 为探索约束；面积和功耗不是用户签核预算。
- 启动架构、RTL/验证、文献三个独立审阅任务；主线先建立统一冻结部署 RTL 与 DC 交接包。

## 2026-07-13 数值与RTL冻结

- H67 RTL-exact valid825：AEE `1.462688`、AAE `9.403994`、spikes `26.354366G`；相对原dyadic
  AEE变化`+0.000055`。
- H68 RTL-exact valid825：AEE `1.472654`、AAE `9.471391`、spikes `26.416394G`；相对原dyadic
  AEE变化`+0.001167`。H68部署图确认不含训练期矩阵分支。
- Gate修正为9-bit Q1.7和最近偶数舍入；100,000组独立整数向量0不一致。
- H67/H68 score分别完成35,937/1,089组合和各100,000随机向量审计，0不一致。
- row regression覆盖8/162 token、fold开关、反压、单token、全活动、H67全部35类和H68全部3类；
  Verilator lint、SVA、Yosys结构检查和行级映射网表回灌通过。

## 2026-07-13 架构迭代

- 新增占用类SCS-Shiftmax：只扫描真实出现的最终score类。H67用两拍类流水，H68编译期单拍。
- profile100加权周期代理：H67下降`12.86%`，H68下降`0.37%`。
- active-entry合并为`score+K+token`的56-bit bank并使用精确162深度。相对256填充，H67/H68
  Yosys通用单元分别下降`32.55%/31.51%`；该数值不是工艺面积。
- ATLIF第一轮覆盖解释：安装105、动态调用93，未调用12项均为`sn2_q`原carrier。2026-07-13进一步确认93个调用中12个`attn_sn`结果不进入正常推理输出，固定部署功能活跃口径修正为81。
- DC脚本、同步复位SDC、SVF、工件审计和Formality交接已生成；本机无DC/工艺库/SRAM宏，
  正式PPA、真实活动功耗和LEC未关闭。

## 2026-07-13 Workload外循环与TESSA架构转向

- 新增旧profile100架构分析器，在不抢占H69训练GPU的条件下完成H67/H68各100样本、
  12个attention block的stage/block分位数、每样本周期代理、TTB覆盖和Delta分桶。
- 81-pair两阶段粗粒度flowshop重放中，2-context相对pair单context再降H67 27.93%、H68 26.06%；
  当时未建模独立commit，4/8-context相对2-context额外收益低于0.1%，该结论随后由端口感知模型修正为“首版2、物理数未冻结”。
- H67/H68 pair-empty为73.90%/74.20%，K-zero为83.11%/83.29%，active-entry为
  18.38/18.40；模型间分布接近，不支持分别实例化物理核。
- block异质性显著：H67 S1B0/S2B3接近全静默，S0B0 active-entry均值59.89，
  S0B1仅3.05。主线由预设异构双核转为统一同构、block-aware、多row-context架构。
- collector新增时间对充分统计量、union-membership表示DSE、双K-zero class合并统计、
  sample-flow相关性，以及9x9水平/垂直/双对角局部性和4/8-bank映射冲突。
- 深度调研Bishop、LoAS、FuseMax、FLAT、ASADI、Sparseloop、Stellar，以及复旦ISSCC
  2023蝶形zero skipper、C-Transformer、MulTCIM、D3TA、HARDSEA等工作，冻结引用边界。
- 形成TESSA方案：128-bit temporal pair、PESF、PCCC、默认2-context、共享class-stationary
  SCS；BMRF、8 context、row OOO和异构双核均按可证伪门槛条件晋级。
- 新增中文文档53/54/55；`test_bsa_attention`与`test_binary_temporal_pair_arch`共56项通过。
- 新profile100已挂在软件串行队列末尾，未把ordered trace、BMRF或多context收益写成已完成结果。
- architecture skill 独立审阅将正式签核保持为`NO-GO`：81-cycle前端尚未计128-bit供数和双commit冲突；
  仅fixed-bitmap pair与可退化1/2-context骨架为受控准入，PCCC必须可旁路，其他候选继续等待trace/PPA。
- 随后补做端口感知三阶段重放：H67双K-zero/单K-zero/双active分别为83.11%/11.09%/5.80%；
  128-bit分bank单写口无合并、2-context仅降22.33%，PCCC全合并上界降48.85%。全合并上界下
  4-context相对2-context仍可改善13.49%，因此改为参数化1/2/4、首版启用2，最终物理数等待ordered trace。
- 冻结TESSA探索性RTL前规格：encoder-attention subsystem边界、128-bit pair与三种供数wrapper、
  active/hist独立单写口和depth-2原子commit、1/2/4-context逻辑memory map、completion、counter与SVA合同；
  机器JSON通过自动宽度/守恒校验，早期NTS07/ternary/98-token接口文档已标记废止。

## 2026-07-13 全Encoder存储与HIT-Flow架构迭代

- 代码复核确认当前ATLIF为`h=b+W[T,T]x`的PSN时间矩阵加阈值发放，不是递归LIF膜状态。
- H67/H68均为105安装、93动态调用；其中12个`attn_sn`结果不进入正常projection/Swin输出，固定部署功能活跃口径为81，构成为45个T10和36个T2。
- 活跃ATLIF每帧约44.244亿标量时间MAC，参数仅5247项；8-bit参数容量约5.12KiB。单个320-MAC时间阵列在500MHz下ATLIF-only约36.16FPS，完整encoder主候选至少从双阵列DSE开始。
- S0-S2长skip共11,612,160元素，旧profile非零率接近100%；1-bit容量1.384MiB只是理论下界，残差/skip位宽等待新stage value profile。
- collector新增stage边界数值域/整数率/binary/ternary统计，以及每ATLIF点首次调用的输入格式、阈值margin和4/6/8-bit参数量化事件翻转采样；CPU单元测试通过。
- 深入对照PSN、PTB、LoAS、ISCAS 2025 timestep-reconfigurable accelerator、Bishop和复旦ISSCC 2023蝶形zero skipper，确认时间并行、可重构timestep和蝶形网络均不能单独作为原创。
- 形成HIT-Flow全Encoder候选：Head-Time Tile、32x10 Divisor-Packed Temporal Matrix Engine、TESSA/factorized gated projection、binary-event/multi-bit-residual precision islands。
- DP-TME独立整数golden完成：100组随机输入下T10共2,592,000个hidden/event、T2五路打包共518,400个hidden/event均0 mismatch；81位置T2由162周期降至34周期，slot利用率95.29%。
- 新增可复现存储合同JSON/中文报告和中文架构文档59；完整RTL/DC签核仍等待ordered profile、ATLIF量化valid825、全encoder周期模型和目标工艺库。

## 2026-07-13 统计证据质量审计

- 按数据源、样本粒度和可复现程度将现有证据分为A到D四级；当前总体为“带限制内部使用”，足以推进参数化架构设计，但不具备论文最终PPA签核条件。
- 交叉核对通过ATLIF数量、三条长skip元素数、时间矩阵操作量、时间对类别守恒、K稀疏指标关系和DP-TME整数映射。
- 明确1-bit长skip容量只是理论下界，44.244亿是算法时间MAC而非芯片周期，81活跃ATLIF点只适用于固定正常推理部署图。
- 冻结新ordered profile的十项完整性检查，以及context、PCCC、蝶形压紧、RPI和ATLIF量化的量化晋级门槛；详见中文文档60。

## 2026-07-13 全Encoder预算与架构创新再冻结

- 建立可复现的全encoder周期/事务预算脚本，联合DP-TME、旧全网event-operation代理、TESSA端口模型、8-bit skip和event-bank读写，并采用1.25倍未建模保护系数。
- 全物化5.260亿ATLIF event元素在256-bit端口下需约411万拍/帧；跨PSN-attention-projection的HTT局部转发被提升为系统架构变量。
- 代理模型淘汰`2×DP-TME + 256 spatial lane`；`2×320 + 512 lane`为紧张面积边界，`4×320 + 512 lane`作为当前平衡候选。
- profiler新增同sequence相邻样本的stage精确相等、active翻转、符号类变化和归一化绝对差统计，序列切换自动清空；同时新增Linear/Conv2d/Conv3d逐算子运行时MAC和活动率代理分账，attention/profile相关60项测试通过。
- 架构创新收敛为DP-TME、LR-HTT生命周期路由和CCSP精确类合并稀疏投影流；复旦蝶形BMRF与persistent-HTT保持条件候选，详见中文文档61。
- 新增ordered profile后处理分析器及6项统计/预算守恒测试；支持用逐算子encoder负载替换旧全网SOPS代理。
- 独立postprofile watcher已启动，等待TTB-v2完成后自动生成中文ordered决策报告和第二版全encoder预算；缺少任一新字段时硬失败。

## 2026-07-13 最近邻反证与贡献收紧

- 深入复核VESTA：其统一PE已支持卷积、线性和dot-product，TFLIF/ZSC/WSSL覆盖四时间步共同处理与层间spike压缩，STDP已实现列完成即dot-product且不存完整中间矩阵。
- 复核可重构并行时间步Spiking Transformer加速器：fully-parallel tick-batching和四时间步展开已经存在，DP-TME只能围绕T10/T2 PSN矩阵的除数slot映射主张增量。
- 复核FireFly-T：稀疏/二值双引擎、多lane bitmap decoder、worker维乱序、bank-conflict规避和跨head attention延迟隐藏均已有，异构双核降为对照而非主线创新。
- 复核T-REX ISSCC 2025全文：dynamic batching按长度处理1/2/4输入；two-direction accessible register file按行/列访问并报告12%到20%利用率改善。LR-HTT不得以普通多方向buffer或动态打包主张原创。
- 补充ULSeq-TA、STAR与ISSCC 2025 CNN-Transformer层融合芯片作为cross-operator/cross-stage fusion强先例；补充STEP对SNN能效口径的警告。
- 新增中文文档62，建立最近邻威胁矩阵、修正后的三条窄贡献、强制基线和更严格淘汰门槛；同步修订文档54/61与研究状态HW16。
- 当前结论不变：可继续参数化架构和RTL探索，但在ordered profile、同约束最近邻基线和目标库DC前，不能签核DATE创新与PPA。

## 2026-07-13 ATLIF静态生命周期合同

- 逐条复核93个动态ATLIF调用及其代码消费者，新增可复现分析脚本、JSON和中文报告；2项合同测试通过，遇到未知模块路径或数量不守恒时硬失败。
- 12个`attn_sn`为固定部署调试死结果；81个活跃点中，45个为相邻单消费者、12个`proj_sn`为Q/K双消费者fanout、24个`sn_q/sn_k`需时间对同步。
- 对应活跃event元素为421,536,960、34,836,480和69,672,960/帧；单消费者点占80.13%，只作为direct-forward静态资格上界，不能当作真实bypass率。
- 代码确认ATLIF二值输出本身不承担长skip；Swin两次ADD、MS ResBlock ADD和S0-S2 skip保存多位算子输出/identity，必须与event bank分离。
- LR-HTT首版接口由此冻结为弹性Forward、Resident fallback、Q/K fanout保留和pair assembly；真实forward/resident比例继续等待ordered profile与RTL计数。

## 2026-07-13 HIT-Flow-LR RTL前规格冻结

- 新增中文文档63，将full encoder候选拆为descriptor scheduler、DP-TME cluster、event lifetime router、HTT bank、TESSA/SCS、CCSP/FGP、RPI和性能计数器。
- 根据静态生命周期冻结四类路由：single直通优先、`proj_sn`一写双消费者、Q/K pair assembly、dead descriptor删除；任何直通阻塞必须无损退化resident。
- 冻结多级exact issue语义：pair-empty和K-zero仍写class/denominator，只有payload、active bank和FGP权重读可门控；不允许把silent当作数学删除。
- 明确Swin两次ADD、MS ResBlock ADD与S0-S2 skip进入RPI，event bank只存1-bit ATLIF/Q/K；RPI位宽继续等待ordered value profile和valid825量化。
- 首版不实现异构双核、BMRF、persistent-HTT、近似pruning和复杂顶层控制；保留dense gated-K、STDP式列流与多context参数用于同RTL消融。

## 2026-07-13 Event Lifetime Router首个RTL切片

- 实现single弹性缓冲、Q/K独立fanout、Q0/Q1/K0/K1时间对组装和静态顶层路由；新增tag错配、重复slot和非法route显式错误输出。
- 代码审阅发现并修复fanout最后消费者与pair退休时的一拍接收气泡；两条路径均支持同拍retire/replace。
- 初次新增SVA时bind漏接四个端口，未把该轮误记为通过；修复后Icarus与Verilator自检、Verilator断言、Erie独立静态lint和Yosys结构检查全部通过。
- 定向测试覆盖独立反压、乱序pair、三类拒绝、同拍切换和事务计数；Yosys为113个generic cell、0 memory，只作结构检查，不作PPA引用。
- 中文文档64明确当前仅为单context寄存器前端，不含resident SRAM、多context、真实trace重放、DP-TME或DC，不能单独支撑DATE架构贡献。

## 2026-07-13 DP-TME端口感知修正与首版RTL

- 复核发现旧五路T2的34拍只计阵列计算：实际每拍要读5个不同32-lane输入，8-bit代理为1280-bit，并需至少约160-bit/拍持续event出口；单32-bit Router会把T2拉回162拍。
- 新增端口感知DSE脚本、JSON、中文报告和3项守恒测试；G4/128-bit为42拍平衡接口候选，G5/256-bit为34拍高带宽候选。
- 进一步纠正“G4可减少MAC面积”的错误推论：为保持T10的810拍，统一阵列仍需320个MAC；G3/G4只门控T2未用槽，裁剪阵列会令T10至少翻倍。
- 实现参数化`hitflow_dptme_array.sv`，强制T2两步/T10十步、命令tag和模式稳定、尾组slot mask、输出反压与协议错误拒绝。
- Icarus与Verilator断言仿真、Erie独立静态lint、Verilator RTL lint、Yosys零告警结构检查均通过；默认结构含320个乘法、321个加法和320个比较器。
- 中文文档65明确当前只是计算原语，尚缺5-bank SRAM、packet adapter、checkpoint定点差分、乘法流水和目标库DC；34拍不能作为系统实测引用。

## 2026-07-13 门类驻留多播投影与蝶形候选审计

- 恢复旧profile的K0/K1精确时间复用：H67为9.98%、H68为10.58%，且除S0外各stage仅约1%到2%，因此全局时间蝶形复用不再作为主贡献。
- 新增GCM-P统计：逐row活动K lane、唯一score-class/K-channel项、活跃投影class、最大fanout，以及M=1/2/4/8/16的精确多播交付事务；全部保存为压缩ordered trace。
- collector、聚合器、队列完成判定和postprofile完整性审计同步更新；H9完整CPU回归99/99、ordered分析器4/4通过。
- 新增GCM-P整数参考：200轮、1,036,800个输出0 mismatch；随机合成乘法减少79.15%只作代数验证，不作为H67真实收益。
- 文献反证确认复旦ISSCC 2023已采用蝶形特征分配、UCNN已利用重复权值、Prosperity已提出SNN product sparsity、Eyeriss v2已有分层多播；创新边界收紧为Shiftmax gate-class metadata与K-event bitmap驱动的attention-to-projection精确融合。
- 新增中文文档66，冻结统一HIT-Flow加GCM-P候选、direct fallback、class-slot/multicast DSE、同约束基线和定量淘汰门槛；真实profile、RTL、DC、SAIF前保持条件候选。
- 新增ordered周期DSE，扫描3种class-slot、5种多播宽度、3种输出lane和3种product-engine共135点；overflow逐row回退direct。postprofile watcher已更新并以新PID 2892386重启，profile完成后自动生成H67/H68中文DSE报告。

## 2026-07-13 最终gate码窗口组数据流与最近邻实现复核

- 代码级重新确认H67/H68 functional attention为`K乘token gate`，随后执行跨head拼接的C乘C
  projection与BN；12个`attn_sn`返回值为部署死结果，但block内attention ADD、MLP ADD及S0-S2
  三条长skip均保留在RPI边界。
- 修正早期GCM-P语义：score class只在同一Shiftmax row内等价，跨row/window必须使用最终RTL
  Q1.7 gate code。collector和DSE已按final gate统计G1/G2/G4/G8/G16窗口组。
- 排队脚本原误用普通dyadic H67/H68配置，已改为通过valid825的`*_rtl_exact.yml`并增加配置硬审计；
  watcher以独立session重启，等待软件队列释放GPU。
- 源代码级复核Prosperity官方simulator：其核心是二值activation行子集、XOR差分、prefix输出回读、
  stable sort和product table，不等同于final-gate乘权重多播；其能耗模型不能替代本项目SAIF/DC。
- 复核FuseMax artifact的Einsum cascade、Timeloop流量和Accelergy action-count生成；将逐SRAM、
  metadata、product、multicast、accumulator动作分账纳入HIT-Flow-WG评价合同。
- 新增中文文档68，冻结A/B/C三档候选、NMF、WG-GPS、分层分段多播、CSD条件候选、最近邻威胁、
  统计清单、基线和定量淘汰门槛；真实ordered profile和目标库DC前保持NO-GO签核。

## 2026-07-13 HIT-Flow-WG可综合RTL前规格

- 新增中文文档69，将SCS到projection后端拆为NMF目录、direct fallback、普通/CSD product engine、
  分段多播、accumulator bank、bias提交和group controller，冻结单时钟ready/valid接口及精确语义。
- 冻结复用键为`{block, final Q1.7 gate code, global input channel}`；不同token独立累加、不同block
  禁止复用、K-zero仍进入Shiftmax分母、bias每token/output tile恰好一次。
- 给出G/S/segment/M/output-tile/context参数集、普通乘法与CSD消融、简单分段与蝶形互连的准入条件，
  默认G1只作逐位验证点，不提前冻结论文参数。
- 建立D0到D6同RTL DC矩阵、SRAM/SDC/SAIF准入、逐位断言、活动计数器和高风险淘汰门槛；蝶形只在
  inter-segment stall和完整projection EDP同时达标时进入实现。
- 统一硬件脚本目录35项测试全部通过；H9算法/profile相关100项测试此前全部通过；修改Python文件
  `py_compile`通过，`git diff --check`返回0。
- H69及后续软件队列仍占用A800；TTB-v2和postprofile两个watcher正常等待。真实ordered profile、
  投影int8 valid825、G>1参数冻结、WG后端RTL和目标库DC仍未完成。
- 对跨窗口采集器追加边界审阅：修复沿扁平`batch_windows`可能跨样本组窗的问题，从block静态
  resolution/window恢复每样本窗口数；所有G新增有效窗口数ordered trace、尾组slot利用率和守恒检查。
- 新增跨样本边界相同gate码定向测试；修复后H9完整101项、硬件脚本35项全部通过。后续G>1收益
  将同时报告乘积复用和尾组利用率，不再使用隐含满组假设。

## 2026-07-13 NMF G1最终门码目录RTL

- 实现`hitflow_nmf_g1_builder.sv`、自检testbench和绑定SVA：按最终Q1.7 gate分配slot，生成
  32 lane的162-bit目的bitmap；gate=0/K-zero不发投影，slot溢出无损走direct fallback。
- 首版功能仿真发现跨group旧bitmap可见，修复为term输出必须受`slot_valid`保护；动态二维写口在
  生产参数Yosys产生663,552个write mux并180秒超时，因此被结构淘汰。
- 第二版固定slot/lane写口可综合，但162深度fallback memory-map产生4606项未驱动静态问题且容量
  不合理；改为单项弹性fallback，阻塞时反压上游，连续overflow测试证明不丢事务。
- 扫描地址由动态乘法改为每lane基址加162，Yosys generic cell由1469降至1379、mux由500降至409、
  `$mul`归零；生产参数完整memory-map后`check`为0问题。该数字仅作结构比较，不作PPA。
- Icarus三模块回归全部通过；NMF生产参数Verilator lint 0 warning/error；Verilator SVA通过；H9
  101项和硬件Python 35项此前通过。
- Erie本地lint的外部Verilator为0问题；内置Verilog-2001启发式对SystemVerilog parameter/genvar
  报4项常量循环误判，明确记录为方言适用性限制，未伪报Erie全绿。详见中文文档70。

## 2026-07-13 G1普通门码乘积后端RTL

- 实现`hitflow_gate_product_engine.sv`、定向testbench和绑定SVA，以9-bit无符号最终gate码乘
  signed int8权重，输出8路signed 17-bit乘积；错tag、输入通道或输出tile的权重响应均拒绝。
- 穷举全部65,792种gate/int8组合，证明乘积范围和17-bit二补码编码逐项正确；RTL定向覆盖
  `256*(-128)`、`256*127`、普通正负乘积、请求/输出反压、非法term与计数器。
- 生产RTL Verilator lint为0 warning/error，带SVA仿真通过；默认Yosys结构检查0问题、77个
  generic cells、8个`$mul`、0 memory，只作为普通乘法D1结构基线，不作PPA。
- 新增`run_projection_g1_checks.sh`统一回归NMF与product后端；两套Icarus、两套SVA、两套Yosys
  和Python参考全部通过。Erie仅对标准SystemVerilog genvar循环保留1项方言启发式提示。
- 中文文档71明确当前尚缺目的bitmap多播累加、bias、完整projection、真实trace、SRAM、DC和SAIF，
  下一切片进入分段多播accumulator，不将局部乘法复用提前写成论文结论。

## 2026-07-13 时空行分段多播RTL

- 实现`hitflow_segmented_multicast.sv`及逐bank SVA：共享一份宽product vector，每bank只发独立
  token ID并可独立反压；每个目的恰好一次，全部提交后才发done。
- 首版每拍扫描完整162-bit bitmap，虽功能和结构检查通过，但Yosys达到3,122个generic cells；
  结构淘汰后改成18-token当前段驻留和常数右移remaining bitmap，只扫描当前18位。
- 默认18-token/2-bank版本Yosys降为294个generic cells和3,596 wire bits，0结构问题；约90.6%
  generic cell下降只作结构比较，不作目标库面积结论。
- Icarus、生产RTL Verilator零告警lint、绑定SVA和Yosys全部通过；Erie仅保留2项参数化循环方言
  warning、0 error。中文文档72冻结简单网络为蝶形/Benes的公平准入基线。

## 2026-07-13 Banked Accumulator与Bias提交输出

- 实现2-bank同步read-modify-write accumulator；普通product写回，最后bias更新在final握手时同时
  写回并输出，暂称BCOD，避免bias后独立读出遍历，但保留传统流程作为后续公平对照。
- 定向逐token验证两bank并行、连续正负累加、bias-only token、final独立反压、重复bias拒绝和
  完整bias后finish；Icarus、生产RTL零告警Verilator、绑定SVA全部通过。
- 强制Yosys memory-map产生24,064项undriven，明确判失败；改用同步读模板并`memory -nomap`后，
  识别两块81×256-bit同步1R1W `$mem_v2`，check为0问题，外围258个generic cells。
- Erie内置Verilog-2001启发式对4个SystemVerilog参数循环误报，记录为方言限制。中文文档73明确
  当前尚缺G1集成、传统readout消融、真实trace、ACC位宽量化、SRAM宏、DC和SAIF。

## Ordered profile自动回填完成

<!-- HIT_FLOW_ORDERED_POSTPROFILE_20260713 -->
- 架构决策报告：`hw_autoresearch_nts07/results/hit_flow_ordered_profile_analysis.md`；
- 逐算子全encoder预算：`hw_autoresearch_nts07/results/hit_flow_full_encoder_budget_ordered.md`；
- H67 GCM-P DSE：`hw_autoresearch_nts07/results/gcmp_h67_multicast_dse.md`；
- H68 GCM-P DSE：`hw_autoresearch_nts07/results/gcmp_h68_multicast_dse.md`；
- H67跨窗口gate-product DSE：`hw_autoresearch_nts07/results/gate_window_group_h67_dse.md`；
- H68跨窗口gate-product DSE：`hw_autoresearch_nts07/results/gate_window_group_h68_dse.md`；
- PCCC、bank mapping、RPI、ATLIF量化、persistent-HTT和spatial lane按报告门槛重新冻结。
