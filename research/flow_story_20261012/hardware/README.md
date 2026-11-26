# MLP-only 有限资源周期模型与代表核 RTL/DC

2026-10-12。只在本目录新增研究原型，未改旧模型、未读取通用线、未用 GPU、未加载训练权重。

**本轮形成了一个可追踪的 stage0 单 MLP 块周期模型，以及经过独立 golden/Verilator 和 DC 验证的 4/16 lane 整数点积、残差、存储握手核。没有验证完整 MLP 数值、量化 AEE、完整引擎或整网收益。** 周期模型包含尚未 RTL 实现的 BN/阈值、SRAM、DMA、序列器和调度器；它的全部总周期不能写成实测 RTL 周期。[机器可读汇总](results/HARDWARE_EVIDENCE.json)

当前核的绝对吞吐很弱。即使采用下文同资源普通流水化对照，stage0 一个完整 MLP 块仍预测 **4,368.476 ms**，按假定 250 MHz、串行使用该资源合同，该块已将整网帧率上限限制在 **0.229 FPS 以下**。这说明首个执行原型不足以支持实时部署，不以局部相对削减掩盖绝对性能问题。该数字也不用于预测未来增加并行资源后的实现。

## 1. 与原生模型及既有证据的关系

原生 `MS_Spiking_Mlp.forward` 的顺序是：

`u → PSN1 → FC1 → frozen BN1 → PSN2 → FC2 → frozen BN2 → +u`。

外部 `u=x+Attention(x)` 已完整产生；关闭 tile 时输出必须为 u。两个 PSN 均对全部 T=10 做稠密时间矩阵乘，输入分别为 u 和 BN1 的连续有符号值，**没有把它们按二值 AC 计数**。FC1/FC2 的输入为脉冲；默认推理关闭 dropout。[原生类](../../../SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py)、[PSN.addmm](../../../SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_submodules.py)、[前轮原生 FP32/FP64 局部执行验证](../../flow_story_20261011/preflight/cpu_numerics/README.md)

本模型严格限定 B=1、T=10、C=96、hidden=384，覆盖 full 输入对应 stage0 的 H=120、W=160，以及 H=10、W=11 的边缘案例。每个 MLP tile 是独立 9×9 空间块、完整 T10，和 attention 窗口 9/15 无关。更高 stage 的权重/激活容量没有靠忽略溢出而套入本模型；当前接口直接拒绝 C≠96。[cycle_sim.py](cycle_sim.py)

前轮 `Pk=16,Pn=16` 是 **256 个标量乘积/issue** 的静态模板；本轮是 **16 个乘法 lane**，并有请求、响应、写回与有限缓存。两者不是同一个资源配置，不能混用前轮的全网分母或 23.8% 上限，也不能把本轮周期直接加到前轮静态 issue 数上。[前轮合同](../../flow_story_20261011/preflight/execution_contract/EXECUTION_CONTRACT.md)

## 2. 数值和存储合同

周期模型采用一个**尚未通过量化精度验证**的部署候选大小：输入 u、PSN 权重和 PSN 输入暂按 signed int8，FC 输入脉冲按每个脉冲 1 byte 存储，FC 权重 int8，点积累加/BN 前张量/输出残差为 int32。u 在旁路及残差相加时符号扩展到 int32。下一模块输入的重定标不在本块内，不能假装这一输出就能无缝替代整网 FP32 张量。

BN1 仍先完成冻结仿射，再经过一个显式计时的 int8 重定标接口供 PSN2；BN2 仿射后加 u。**没有定义完整 BN 的量化比例、舍入和饱和，也未实现完整整数 MLP 的数值输出**，所以这些只决定存储占用和待实现流水槽的成本，不是算法精度证据。训练中 current-stat BN 不能套用这个跳过合同；冻结部署引起的 AEE 成本由算法实验单独验证。

为避免把“部署优化”冒领为 gate 收益，dense 与所有 masked 对照都付 BN 的成本，均未使用免费融合。每 16 元素读 alpha/beta 共 128 byte、读 int32 激活 64 byte，再为仿射保留 2 周期。BN1 写 16 byte；BN2 另读 u、做残差加、写 64 byte。仿射 2 周期是设计假设，**没有其 RTL/时序支持**。共享层权重仅在首个 keep 请求时载入，之后 dense/candidate 都复用，不按 tile 重载。

| 固定资源 | 本轮容量/接口 |
|---|---|
| 点积 | 16 lane，8×8 signed 乘法、int32 累加，一个输出向量；默认一个在途读 |
| U 输入 bank | 2×80 KiB，各独立 128-bit、1RW；完整 tile 最大 77,760 byte |
| 输出 bank | 2×320 KiB，各独立 128-bit、1RW；完整 tile 最大 311,040 byte |
| 共享 scratch | 32 KiB、128-bit 1RW；按一个空间点保留全部 T10，保守峰值 27,840 byte |
| 权重 SRAM | 128 KiB、128-bit 1R，载入阶段使用写端；本层含 BN/PSN 表共 77,848 byte |
| SRAM 总容量 | **960 KiB**；外加 PSN 小表/偏置寄存器、累加寄存器及控制，未计宏面积 |
| 队列 | 2 条 descriptor、4 条 DMA request、2 个完整 tile slot |
| 共享外部 DMA | 128-bit、最多 32 beat/burst、一个 burst 在途；每 burst setup 6 周期，读延迟 12、写响应 2 周期 |
| 人为总线反压 | 绝对时钟模 17 等于 7 时拒收 beat；确定性压力条件，非真实 DRAM 测量 |

输入/输出 bank 的 slot 处于 loading/computing/output-ready/draining 时不可被重用；输入 DMA 只写 loading slot，PE 只读已 loaded slot；输出 DMA 只读已写完 slot。两个物理 slot bank 分离，因此“预取下一块”和“计算当前块”不争同一端口。scratch、权重读及向量序列器由单计算流显式顺序占用，不宣称多阶段神奇并行。

每个空间点完整执行 PSN1→FC1→BN1→PSN2→FC2→BN2+u 后，scratch 才复用到下一点；只有 tile 的所有有效点完成，才能排输出 DMA。该数据流较保守，但有限容量足以支撑它，而不是为一个完整 hidden tile 虚构无限 SRAM。

## 3. 如何得到周期

[pixel_microcode](cycle_sim.py) 逐 MAC 请求/响应执行 FSM：默认发出一对操作数读请求，下周期响应并累加；不能同周期再次发请求。输出向量保持到 128-bit 序列器排完。PSN 阈值、BN 仿射、残差、输入/输出端口另占周期。一个完整点微码得到：

| 阶段 | 默认周期/空间点 |
|---|---:|
| PSN1 | 1,380 |
| FC1 | 47,280 |
| BN1+重定标槽 | 3,600 |
| PSN2 | 5,520 |
| FC2 | 46,380 |
| BN2+残差 u | 1,200 |
| keep 合计 / bypass 合计 | **105,360 / 300** |

这不是将 MAC 总数除以 PE 数：MAC 响应、命令、两条 SRAM 请求、输出序列化、阈值和仿射都在微码中计时；另按原生形状独立断言 PSN 有 48,000 个有符号乘积/点，两个 FC 有 737,280 个二值加权累加/点。调度器仅把资源持续被同一微码占用的相同空间点循环压成一个事件区间，没有将计算和等待混为一次除法。

DMA 按每个 T、每个有效行发 burst；地址来自 THWC 连续布局，不把跨行、跨 T 的 tile 当作单段连续区域。每条记录保留 segment 数及首/末地址。边缘只执行真正存在的空间点；H=120,W=160 的 252 个名义 tile 有 1,212 个 padding 点，既不读也不执行，dense/candidate 同样裁掉。H=10,W=11 四 tile 的有效点分别为 81、18、9、2，覆盖集合不重不漏。MLP/PSN 仅耦合同一点的 T 维，故这里不需要 attention 的 padding 生产者闭包。

调度器在 DMA 完成、compute 完成、前端风险就绪、队列可以发射时推进；一个时间步内资源不变时才快进。额外以**同 FSM 逐时钟推进**核对一 keep 点，以及 9 tile 全 bypass 的 1/4 成员反压场景，周期、事件和计数完全一致。这是 event 快进的验证，**不是独立完整 MLP 算术 golden**。[cycle_results.json](results/cycle_results.json)

前端继承前轮实际 RTL 合同：1 个命令周期、N 个风险周期、N 个描述符周期，N=1 或 4；强制 dense 前端省掉风险收集。四成员按 base+k×stride 发非邻接 tile，取 max 风险，任一成员 keep 就整组 keep。本轮在周期模型中实现其费用和 descriptor FIFO 反压，**没有把前轮 RTL 与所有新模型部件联合做 RTL 仿真**。[既有前端证据](../../flow_story_20261011/preflight/rtl_issue_probe/RTL_ISSUE_PROBE.md)

风险值在本轮为 0/255 的合成输入，不来自 GT 或已训练 gate。默认预测器周期为 0，只对应“风险已提供”的执行评估。接口可以加 `predictor_cycles_per_tile` 的串行成本；其特征生成/读取和评分数值仍待接入，不能因此称完整动态算法成本已覆盖。反压等待周期与 PE/DMA 忙周期可能重叠，**不能相加当总周期**。

## 4. 合成 mask 与普通强对照

主几何 H=120,W=160，全部是 stage0 **一个 MLP 块**。uniform 是隔 tile 保留；clustered 保留前半 tile；adversarial 在同 tile 数约束下优先保留面积最大的 tile。后三者各保留 126/252 tile，但有效点不同，不能称为相同计算预算或相同精度。

| mask，单成员前端 | 保留有效点 | 模型周期 | 假定 4 ns 的 ms/块 |
|---|---:|---:|---:|
| dense / 强制全开前端 | 19,200 | 2,022,934,977 | 8,091.740 |
| uniform | 9,720 | 1,031,203,728 | 4,124.815 |
| clustered | 10,080 | 1,066,797,087 | 4,267.188 |
| adversarial | 10,206 | 1,080,172,534 | 4,320.690 |
| 全 bypass，仅作极限检查 | 0 | 10,012,759 | 40.051 |

所有存在 keep 的主表案例都传 **92,237,848 byte** 外部 payload，其中 77,848 byte 为一次性权重。输入 u 和输出张量在旁路时也必须读写，故该数据流中局部跳过几乎不省外部 payload；被省的是块内算术和私有 scratch 服务。全 bypass 不载入权重，但它没有精度意义。不能把没有 materialize 的中间特征重复计为外部访存节省。

四成员 max-risk 组在本轮这些半开 mask 下均变成 252/252 全开，周期与 dense 相同。这支持 MLP-only 使用天然独立的单成员 tile，不能用四窗控制前端的存在推断 MLP 也应该强行四窗绑定。

单 outstanding 是首核的保守状态机，不是强部署基线。为避免由它制造相对优势，额外在**相同 16 lane、相同 SRAM 大小/端口/权重复用**下建模常规流水化：允许请求与上一个响应重叠，最多 2/4 个在途读，仍逐时钟管理 credit、端口和 MAC 的 II。该控制尚未 RTL 实现。

| 同资源普通控制 | dense 周期 | uniform 周期 | dense / uniform ms/块 |
|---|---:|---:|---:|
| 1 outstanding，MAC II=1；代表核合同 | 2,022,934,977 | 1,031,203,728 | 8,091.740 / 4,124.815 |
| 2 outstanding，MAC II=1；仅模型 | 1,092,118,978 | 559,978,151 | 4,368.476 / 2,239.913 |
| 4 outstanding，MAC II=1；仅模型 | 1,092,118,978 | 559,978,151 | 4,368.476 / 2,239.913 |
| 4 outstanding，MAC II=2；仅模型 | 2,022,934,977 | 1,031,203,728 | 8,091.740 / 4,124.815 |

普通流水化本身几乎减半模型周期，绝不能计为风险调度贡献。风险 mask 必须与**同样流水化**的 dense 对照比较；该 uniform 案例的单块模型削减为 48.726%，仍非精度通过后的收益。该表没有证明这个普通数据流已经全局最优；跨点矩阵并行、更多乘法器、操作数缓存和适配更高 stage 仍可能改变比较。

另外以同一边缘 mask 扫过单 slot、64-bit SRAM/总线字宽、50% bus 反压；均改变预期周期，结果保留在 `resource_sensitivity`。本轮没有用任意带宽假设拟合一个预设加速比。

## 5. 实际 RTL、golden 与综合

[dot_residual.sv](rtl/dot_residual.sv) 接收 keep、K、bias、残差、输出 tag 和 lane mask。keep 路径逐项请求一个有符号标量和一个有符号向量，等待 SRAM 响应后做各 lane int8×int8→int32 累加，最后加残差；bypass 不发任何读，只输出残差。K=0 的 keep 输出 bias+残差。结果遵守 modulo 2³²，不声称已确定 SNN 的饱和/舍入行为。

该乘法可表示 PSN 的“时间权重标量×多个通道实值”和 FC 的“脉冲标量×多个输出通道权重”。PSN 的完整时间/地址控制、阈值、BN、tile 缓存、128-bit 写序列器与 DMA 均在模块外；RTL 自身的向量写端宽度为 LANES×32 bit。因此，RTL 的一个无阻塞 dot 是 `2K+2` 周期，**不能漏掉模型为了真实 128-bit SRAM 额外支付的写序列化周期**。

独立 [C++ golden](tb/dot_tb.cpp) 用 uint32 模运算累加数学点积，逐次核对数据、tag、lane mask、请求索引与读取数。覆盖 K=0/1/2/10/16/96/384、keep/bypass、符号和溢出、部分 lane、1–4 周期存储响应延迟、请求/写回反压、阻塞输出稳定、等待响应时复位。复位测试同时丢弃环境中的未返回响应；此前已经接受的写入不回滚。本轮没有形式验证或综合后门级等价测试。

| 实际 Verilator 4.028 | 4 lane | 16 lane |
|---|---:|---:|
| 完成案例 | 90 | 90 |
| 实际 read / write 握手 | 5,108 / 90 | 5,108 / 90 |
| 无阻塞公式检查 | 14 例 | 14 例 |
| 阻塞 write 周期 | 61 | 61 |
| 整个 TB 周期，含初始化/复位 | 19,010 | 19,137 |

两列随机 payload 的数量不同，推进了不同的响应延迟随机序列；TB 总周期不能当作 lane 数的吞吐比较。逐拍 trace 和汇总分别在 [4 lane](results/verilator_lanes4/summary.json)、[16 lane](results/verilator_lanes16/summary.json)。

DC V-2023.12-SP3、2 cores、TSMC28 HPC+ TT 0.9 V / 25°C，固定 4.000 ns 时钟；uncertainty/clock transition/input transition=0.050 ns，输入/输出 delay=0.200 ns，输出 load=0.010 pF。4 ns 是此次较保守的初始约束，不是扫频得到的最高频率。同步 reset 不设 false path。

| 代表核，实际 DC 预布局结果 | 4 lane | 16 lane |
|---|---:|---:|
| 标准单元面积 | 1,822.114 μm² | 6,796.986 μm² |
| 最坏 setup slack | +2.13584 ns | +2.13266 ns |
| `report_constraint -all_violators` | 无违例 | 无违例 |

面积范围只覆盖点积、残差寄存器和这一小 FSM，**不含 960 KiB SRAM 宏、BN、阈值/重定标、评分、路由/DMA、发射前端或完整控制器**。不能由它判整引擎面积增量≤10%，不能将 SRAM 容量乘一个未经核实密度来补面积，也不报功耗/TOPS/W。实际 mapped cell 的库面积与 LEF `SIZE` 逐项相符，支持 μm² 单位；LEF110a 与 timing DB180a 的版本差异仍记录，不宣称两个 release 完全等价。[汇总和哈希](results/HARDWARE_EVIDENCE.json)

DC 告警原样保留：有符号结果写入无符号 bit-vector、穷尽 FSM 的 default 不可达、DW carry/sign-extension 的未连接/feedthrough/同网多 pin，以及一个高扇出网的估算。RTL golden 覆盖了符号与模溢出，但这不替代门级等价；高扇出网还需要布局缓冲/CTS 后复查。此处无布局、布线、寄生、SRAM 宏时序或多 corner signoff。

## 6. 真实 mask 接口和复跑

[import_engineering_trace.py](scripts/import_engineering_trace.py) 已读取本轮 [整网工程执行 trace](../deployment_contract/whole_model_tile_probe.json)，导入 **stage0 第一个 MLP** 的四种策略。来源是一个 train 输入、36-step 工程 checkpoint、P-code 冻结 BN 的真实 FP32 执行观察；策略仍是人工 all-keep/all-skip/checkerboard/last-edge，**不是 learned gate**，也不是此 int8/int32 格式的精度证明。源 trace、checkpoint、input SHA 都进入生成的 `masks/engineering_*.json`。归档输入的原生编码等价问题正在数据线另行审计，本组工程权重不充当复现结果。

| 导入的 synthetic policy | 实际 FC1/FC2 行数，各自 | 模型保留有效点×T | 默认周期 / 同资源 2-outstanding 周期 |
|---|---:|---:|---:|
| all_keep | 192,000 | 19,200×10 | 2,022,934,977 / 1,092,118,978 |
| all_skip | 0 | 0 | 10,012,759 / 10,012,759 |
| checkerboard | 96,060 | 9,606×10 | 1,019,004,343 / 553,305,406 |
| last_edge_only | 210 | 21×10 | 12,224,558 / 11,206,479 |

四行均逐项断言与 hooks 观察一致，尤其尾 tile 的 3×7 空间点没有按名义 81 点收费。其余 11 个 MLP 的 GPU trace 没有被擅自乘一个系数映射到当前 C96 模型。结果为 [engineering_trace_cycle_results.json](results/engineering_trace_cycle_results.json)；GPU 的 wall time 不参与任何 PE 周期拟合。

[mask_example.json](mask_example.json) 是可运行的合成实例。`keep` 为按 y-major/x-major 排序的平面 Boolean 数组；一个值覆盖整个 9×9 的有效空间区域和全部 T10。当前只接 stage_index=0、block_index=0/1、C96；其他 stage 须先扩展容量/数据流合同。`origin` 必须明确是 `runtime_predictor`、`synthetic` 或 `gt_oracle_diagnostic`；后者只能诊断，不能写运行时收益。`provenance` 应包含 checkpoint SHA、frame/sequence、数据/BN 协议、评分器版本及特征可用时间。

`predictor_cycles_per_tile` 只是最简单的额外串行预算入口，**不能替代预测器 SRAM 读/写 trace**。真实 mask 接入后若缺少这些访问，应继续把结果标“给定 mask 的执行周期”；全成本和 AEE 未齐前不能通过主线硬件收益门。

```bash
python3 cycle_sim.py
python3 cycle_sim.py --mask mask_example.json --output results/real_mask_interface_example.json
python3 scripts/import_engineering_trace.py
bash scripts/run_verilator.sh
bash scripts/run_dc.sh
python3 scripts/collect_evidence.py
```

全部结果、源码/参考源哈希、RTL 逐拍 trace、DC SDC/netlist/DDC/log/report 在本目录。最窄的可用结论是：“已把 MLP-only 的整块开关落实为有限资源调度模型，证明同样 tile 数不等于同样计算预算，并实际验证了有符号点积、残差旁路和存储反压核。”**完整 SNN MLP 数值硬件、强 dense 执行器、端到端 predictor/存储成本、量化精度和整网周期门仍未关闭。**

## 7. 下一步：先形成可部署的普通 dense 引擎，再比较 A

**停止把当前 16 lane、整点串行循环当作拟部署基线。下一版选择 Pk=16、Pn=16 的 256 个标量乘积/issue 阵列作为可比较起点**，先做计算/端口下界及多 stage 容量设计，再决定是否增加 Pn 或复制引擎；256 也不自动等于满足部署吞吐。这是待实现的设计假设，不是本轮已综合核的性能。

阵列需要两种明确区分的模式：FC 用 16 个 spike bit 控制 16×16 个权重的选通/加和，属于二值加权 AC；PSN 用连续有符号激活与 T×T 权重做 real MAC，不能按 spike 位免费选通。只有在 PSN/BN/残差的统一量化合同通过精度门后，才能把 PSN 的 real MAC 落成 int8；否则须给浮点或其他精度单元另做面积、端口和周期账，不能借用本轮 int8 面积。以下容量仍暂按上一节 int8/int32 假设。

对一个空间点、T=10、C 为 16 的倍数，理想 II=1 的**计算发射下界**为：

`FC1+FC2 ≥ T·[ceil(C/16)·ceil(4C/16)+ceil(4C/16)·ceil(C/16)]`；

`PSN1+PSN2 ≥ T·ceil(T/16)·[ceil(C/16)+ceil(4C/16)]`。

C=96 时分别为 2,880 和 300 issue/点；stage0 的 19,200 点至少 61,056,000 issue。该下界尚无 BN、阈值、残差、warm-up、端口或 DMA 等待，**不得标为精确 latency**。PSN 的 K=10 只使用 10/16 个归约 lane；这部分利用率损失不能用总 MAC÷256 抹掉。

| 输入供给 | 无复用时每 issue 的需求 | 下一版普通优化，两种方案必须共用 |
|---|---|---|
| FC | 16 spike bit + 256 byte int8 权重；16 个 int32 累加器更新 | 每个 16×16 权重块在 M=8 行之间复用，权重 SRAM 平均下界降为 32 byte/issue；双权重寄存器块共 512 byte、M×Pn 累加器共 512 byte。至少两路独立 128-bit 权重读，尾部不足 M 行的气泡须计入 |
| PSN | 10×16 个 real 激活和 10 个时间权重 | 双缓冲完整 T10×Pn 操作数，在 10 个输出时间步复用；int8/FP32 两种情况下，RF 瞬时供给为 160/640 byte，摊销 SRAM 读下界为 16/64 byte/issue，双 RF 容量为 320/1,280 byte；不能把 RF 供给带宽当 SRAM 已实现带宽 |
| 输出/部分和 | 一个 Pn=16 的 int32 向量为 64 byte | 四路 128-bit bank 或明确计时的序列器；BN/PSN 中间数据写读、FC2 部分和更新和 residual u 读均须仲裁，不能仅给权重端口 |

多 stage 不能沿用当前 960 KiB 容量。按现有表示，整层权重/BN/PSN 表为 `8C²+40C+280 byte`；如果保留原来的完整 tile 双 U/输出槽及一个点 scratch，所需物化容量如下，尚未加 bank rounding、DMA FIFO 和阵列寄存器：

| C | 层权重/系数 KiB | 完整 tile 方案总活跃容量 MiB |
|---|---:|---:|
| 96 | 76.02 | 0.842 |
| 192 | 295.77 | 1.825 |
| 384 | 1,167.27 | 4.212 |
| 768 | 4,638.27 | 10.675 |

下一版先采用**整层权重驻留、内部空间条带执行**作为强 dense 起点：层权重预留约 5 MiB；每次处理最多 9 个空间点的全部 T10，隐藏通道按 128 为一组，BN1/PSN2 后累加 FC2 的 int32 部分和。这样 C768 的单条带 U/FC2 部分和分别为 69,120/276,480 byte；它们和双隐藏面板可放入约 1 MiB 的额外数据 SRAM，先按 **8 MiB 总 SRAM 上限**审查 banking 与宏配置。8 MiB 是待评估资源预算，未证明面积/功耗可接受；若必须压到约 2 MiB，就要分权重面板，并把跨条带重载或部分和 spill 的外部流量逐项加入下界。不能故意选一个重载很差的 dense 循环来放大 skip 收益。原子跳过仍是整 9×9×T10，内部条带只改变执行排布。

可执行顺序限定为三步：

1. 固定数值合同与目标吞吐，按四个 stage 的真实形状、SRAM 宏可用端口，完成上述计算/带宽/容量表；若下界已不满足目标，先扩大阵列或调整数据流，不进入 gate 收益比较。
2. 用同一资源描述实现普通 dense 的多在途请求、双缓冲、权重/激活复用和 partial-sum 数据流；将真实工程 trace 的 all-keep 和 synthetic mask 接入同一模拟器。两者共用普通优化、量化/BN 处理、时钟、banking、packing 和 DDR 条件。仍先报模型周期，阵列/存储控制完成 RTL 对照前不升级证据等级。
3. **A 专有改动**只落在风险特征/评分器、风险到达时间、整 tile admission/取消、描述符及地址生成、私有缓冲分配释放、u 旁路和输出有效性控制；取消 PSN/FC/BN 生产者必须真的撤销其 PE 与存储事务。保留 u 的传送/重定向成本和全部预测器访问，随后综合包含阵列、相关控制及存储接口的完整比较顶层。届时才判主计划的净周期、延迟和面积门，小核面积不再充当硬件贡献主体。
