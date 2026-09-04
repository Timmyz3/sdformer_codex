# 最终Gate码驱动的窗口组数据流与架构创新边界

**日期**：2026-07-13  
**对象**：H67 Motion-XOR主线、H68无Motion-XOR硬件简化线  
**状态**：RTL前架构候选；真实ordered profile和DC前不签核论文贡献

## 1. 先给结论

1. 当前工作不能只写成“row engine加一个稀疏投影单元”。DATE需要的是完整执行组织：数据如何
   从ATLIF流向attention、归一化元数据如何前推、窗口如何成组、权重和中间状态如何驻留、
   不规则目的如何多播、残差如何隔离，以及12个block如何在共享硬件上调度。
2. 不预设异构双核。旧profile证明block间密度差异很大，但尚未证明复制dense/sparse算术核后
   能抵消分流、FIFO和负载失衡开销。异构双核保留为基线，不作为默认主线。
3. 当前最有希望的架构增量是**归一化元数据前推加窗口组门码乘积驻留**：Shiftmax不再只输出
   dense gate张量，而是输出`final_gate_code + K-channel bitmap + destination bitmap`指令；
   projection对同一最终gate码和输入通道只生成一次权重乘积，并在多个独立窗口的token间精确多播。
4. 跨窗口复用键必须是RTL最终9-bit Q1.7 gate code。原始score class只能在同一Shiftmax row内
   代表相同gate；不同窗口分母不同，不能跨row按score复用。
5. 复旦ISSCC 2023/后续JETCAS工作已经覆盖蝶形zero skipper和local-attention reuse；Prosperity
   已覆盖SNN product sparsity；Transitive Array已覆盖无损GEMM结果复用；SWAT和FuseMax已覆盖
   attention的row-wise融合。因此不能声称首次蝶形、首次乘积复用或首次attention融合。
6. 真实profile尚未产出最终gate码的G=1/2/4/8/16统计。本文冻结的是参数化候选、接口、基线和
   淘汰门槛，不提前选择G、class slot、多播宽度或互连拓扑。

## 2. 代码级功能边界

### 2.1 H67/H68 attention真实路径

代码路径为：

```text
输入x
  -> proj_sn(x)                    二值ATLIF事件
  -> Linear_q + BN + sn_q          二值Q
  -> Linear_k + BN + position + sn_k
  -> H60 TX/SC score
       H67额外加入K0 XOR K1的Motion项
       H68部署时关闭Motion-XOR和Castling辅助
  -> RTL Shiftmax
       Q7 score
       16项Q8 exp2 LUT
       整数row sum
       上取整二次幂归一化
       Q1.7 RNE gate，范围0..256
  -> K * gate                      token-wise selector，不是N×N attention矩阵
  -> 拼接所有head
  -> C×C Linear projection + eval BN
  -> window reverse
  -> attention ADD residual
  -> MLP
  -> MLP ADD residual
```

`attn_sn(x)`在patched forward中被计算并作为第二返回值供调试，但functional projection使用的是
`x = proj(pre_attn_sn_x)`。因此固定正常部署图可删除12个`attn_sn`结果及其状态访问，但不能删除
projection后的两个block内ADD。

### 2.2 projection的精确整数式

设拼接head后的输入通道为`i`，token为`n`，输出通道为`o`：

```text
a[n,i] = K[n,i] * g[n,head(i)]
y[n,o] = bias[o] + sum_i a[n,i] * W_fold[o,i]
```

其中：

- `K`为1-bit事件；
- `g`为9-bit无符号Q1.7最终gate code；
- `W_fold/bias`是Linear与eval BatchNorm2d静态折叠后的权重和偏置；
- bias每个token/output只加一次；
- 不同head的局部lane映射到不同全局输入通道，不能按局部lane错误合并。

若两个token满足`g相同`且同一输入通道`i`上的K均为1，则二者需要完全相同的
`g * W_fold[:,i]`向量。硬件可只生成一次该向量，再向两个独立token accumulator多播。这一重排
不合并token结果，也不删除Shiftmax分母项。

### 2.3 residual和skip边界

- 每个Swin block有两次ADD：attention输出加shortcut、MLP输出再加block状态；
- encoder到decoder的长skip只有S0、S1、S2三条；
- S3是bottleneck/首个decoder输入，不是第4条长skip；
- ATLIF/Q/K event bank保存1-bit事件；ADD和S0/S1/S2必须进入独立多位Residual Precision Island；
- 当前没有证据支持把残差或长skip压成1 bit。

## 3. 主候选：HIT-Flow-WG

工作名为**HIT-Flow-WG**，其中WG表示Window-Grouped。它不是单独的projection小核，而是
full-encoder共享执行体系：

```text
Descriptor Scheduler
       |
       +--> DP-TME Cluster
       |      45个T10 ATLIF点、36个T2 ATLIF点时间复用
       |
       +--> LR-HTT Event Lifetime Router
       |      single直通、Q/K fanout、temporal-pair组装、resident fallback
       |
       +--> H60 Frontend
       |      TX/SC + optional Motion-XOR + exact issue hierarchy
       |
       +--> SCS Shiftmax
       |      silent multiplicity、active replay、最终Q1.7 gate
       |
       +--> NMF Instruction Builder
       |      final gate code、K channel、token destination bitmap
       |
       +--> WG-GPS Projection Backend
       |      gate-product stationary、weight-column stationary、分段多播
       |
       +--> RPI
              block内两次ADD、S0/S1/S2多位skip
```

### 3.1 NMF：归一化元数据前推

NMF（Normalization Metadata Forwarding）不把Shiftmax输出物化为`162×C` gated-K张量，而是
形成投影指令：

```text
{block_id, head_id, window_group_id,
 final_gate_code[8:0], global_input_channel,
 destination_bitmap[162*G-1:0]}
```

其关键点是复用SCS已经生成的最终gate，不增加Prosperity式通用子集检测，也不增加Transitive
Array式Hasse图在线scoreboard。SCS的归一化结果第一次直接成为后级projection的执行元数据。

NMF必须保留三类数学语义：

1. K-zero token不进入projection destination，但其score仍在Shiftmax denominator中；
2. gate code为0且K为1时可关闭权重读取和product，但仍计入gate histogram；
3. class slot溢出时逐row或逐group退化到direct active-entry路径，结果不得变化。

### 3.2 WG-GPS：窗口组门码乘积驻留

WG-GPS（Window-Grouped Gate-Product Stationary）把同一`block/head`的G个独立9×9×T2窗口组成
projection group。对每个`(final_gate_code, global_input_channel)`：

```text
1. 读取一次W_fold[:, input_channel]
2. 生成一次gate_code × weight vector
3. 将product保持在寄存器中
4. 扫描G个窗口的destination bitmap
5. 向每个token自己的output accumulator累加
```

窗口之间没有attention依赖；分组只改变执行顺序，不改变window partition、cyclic shift、token地址
或block残差。不同block权重不同，严禁跨block复用。G只能取能在状态、周期和bank端口上获益的值。

### 3.3 存储层次

```text
L0  pair register / LR-HTT queue
L1  active-entry bank + silent histogram + final gate directory
L2  destination bitmap bank
L3  folded projection weight SRAM
L4  token-output accumulator tile
RPI residual/skip SRAM，与event bank物理分离
```

设gate slot为S、每head 32个输入lane、每window 162个时空token：

```text
destination bitmap = S * 32 * 162 * G bit
```

例如S=4：

| G | destination bitmap/context |
|---:|---:|
| 1 | 2.53 KiB |
| 2 | 5.06 KiB |
| 4 | 10.13 KiB |
| 8 | 20.25 KiB |
| 16 | 40.50 KiB |

若同时保持G个窗口、L个输出通道、32-bit accumulator：

```text
accumulator tile = G * 162 * L * 32 bit
```

G=4、L=16时为40.5KiB。该状态成本很可能比product generator更大，所以G不能只按乘法减少率
选择。RTL前DSE必须比较完整accumulator SRAM、目的bitmap、读写端口和tail utilization。

### 3.4 分层分段多播

首版互连不是完整butterfly/Benes，而是两级结构：

```text
162*G destination bitmap
  -> 16-token local segment
  -> 每段bank-aware M-way selector
  -> token accumulator bank
```

产品向量在segment扫描期间驻留。简单结构若已达到理论交付下界的85%，不实现复杂蝶形。只有
当真实trace显示跨segment高fanout、多播长期阻塞product engine，且同约束DC后子系统EDP至少
改善15%，才将butterfly/Benes作为inter-segment网络晋级。

复旦蝶形zero skipper处理的是稀疏权重到CIM阵列的重排和利用率；本候选处理的是运行时
`gate_code × K-event`生成的动态token目的集合。二者用途不同，但蝶形拓扑本身已有明确先例，
不能作为原创点。

### 3.5 Exact Issue Hierarchy

前端执行分四级，所有跳过都保持H67/H68语义：

| 级别 | 条件 | 硬件行为 | 数学处理 |
|---|---|---|---|
| L0 | pair/QK元数据全空 | 关闭payload读取和部分popcount | 注入合法silent score/multiplicity |
| L1 | Kcurrent全零 | 不写active bank、不发projection | score仍进入SCS denominator |
| L2 | H67 motion-zero | 关闭Motion-XOR切换 | TX/SC照常 |
| L3 | active | 完整score、SCS、NMF、projection | 原始语义 |

这一级联可借鉴SpAtten的级联发射思想，但不做token/head pruning，不改变网络输出。

## 4. 三档架构候选

| 候选 | 关键配置 | 优点 | 主要风险 | 当前状态 |
|---|---|---|---|---|
| A 保守型 | G=1，S=4/8，M=2/4，direct fallback | 状态小、验证最容易 | 可能只有operator级收益 | 参数化准入 |
| B 平衡型 | G=2/4，S=4/8，M=4/8，L=16/32 | 跨窗口product和权重读复用，具备dataflow架构性 | accumulator和bitmap SRAM增大 | 主DSE候选 |
| C 激进型 | G=8/16，分层多播，CSD product，可选butterfly | 最大product复用和带宽合并 | 状态、布线、Fmax、尾部利用率 | 条件候选 |

推荐流程不是直接选B，而是让真实trace淘汰：A必须作为实现基线；B只有在投影后端和全encoder
Amdahl都受益时晋级；C中的CSD和butterfly分别做独立消融，不捆绑包装。

## 5. 电路级可选项

### 5.1 Gate-code CSD product generator

gate code只有0..256。可将每个gate code解码为CSD非零digit，用移位加减生成
`gate_code × int8_weight`，避免通用9×8乘法器。它只属于电路/微结构消融，原因是：

- Transitive Array已经提出乘法消除和结果复用；
- VLSI 2024已有basis-vector分解和LUT-assisted core；
- 常数乘法移位加本身不是新颖机制。

晋级条件：真实gate histogram下平均CSD非零digit不超过2.5；同库同频率下相对9×8 multiplier
的product-generator EDP至少改善10%；Fmax下降不超过5%。否则使用综合器生成的乘法器。

### 5.2 Product cache不作为首版主线

可以为`(block, gate_code, input_channel, output_tile)`保存product，但block内权重固定、gate码空间
只有257，cache实际接近显式表，容量和填充开销容易超过重算。只有真实trace显示跨group gate码
复用很高、weight SRAM能耗占主导，才评估小型victim product table。

### 5.3 Clock gating域

真实profile后需分别提取：

- Motion-XOR域：H68编译期关闭，H67按motion-zero活动率门控；
- active-entry域：K-zero时关闭写入；
- NMF/class目录域：direct fallback时旁路；
- product域：gate=0或无destination时关闭；
- multicast/accumulator域：按segment和output tile门控；
- DP-TME的T2/T10 unused slot域；
- RPI独立域，不能跟1-bit event域共用错误门控条件。

功耗报告必须用真实trace SAIF/VCD，而不是用spike count直接乘常数。

## 6. 最近邻论文与可迁移边界

| 工作 | 会议/期刊 | 已有架构机制 | 可迁移idea | 本项目不能声称 |
|---|---|---|---|---|
| 复旦Sparse Transformer | ISSCC 2023 | in-memory butterfly zero skipper、local-attention reuse | butterfly仅作互连消融；权重驻留 | 首次蝶形、首次local attention reuse |
| 扩展Sparse Transformer | JETCAS 2026 | butterfly feed-forward IMC、可变稀疏span、四芯片扩展 | 检查可变span和多芯片评价方法 | 用固定window包装成可变local reuse |
| Prosperity | HPCA 2025 | 二值脉冲行子集检测、prefix结果复用、dispatcher/product table | 无损复用基线、检测开销分账 | 首次SNN product sparsity/result reuse |
| Transitive Array | ISCA 2025 | bit-sliced偏序、Hasse scoreboard、Benes/crossbar、无乘法GEMM | 动态结果复用和互连强基线 | 首次无损GEMM复用或乘法消除 |
| SWAT | DAC 2024 | window attention row-major、kernel fusion、input-stationary FIFO | window级数据流和流水平衡方法 | 首次window attention融合 |
| FuseMax | MICRO 2024 | extended Einsum cascade、1D/2D阵列平衡、action-count评价 | 不物化中间张量、Accelergy动作分账 | 首次attention跨算子融合 |
| FLAT | ASPLOS 2023 | attention专用dataflow，缓解activation-activation流量 | Q/K与后端驻留顺序 | 首次attention dataflow |
| Bishop | ISCA 2025 | TTB、density stratifier、dense/sparse异构核、ECP | TTB描述符和异构基线 | 首次TTB或异构稀疏核 |
| SpAtten | HPCA 2021 | 级联token/head pruning和progressive quantization | 级联issue组织 | 把exact skip写成首次级联pruning |
| LoAS | ISCA 2024 | fully temporal parallel、spike compression、inner join | T=2 pair相邻布局 | 首次时间并行或spike join |
| Sparseloop | MICRO 2022 | representation/gating/skipping/reuse动作分账 | 评价口径 | 只用稀疏率推导能耗 |

主要来源：

- [ISSCC 2023官方program](https://www.isscc.org/s/ISSCC2023AdvanceProgram.pdf)
- [复旦ISSCC 2023官方介绍](https://fics.fudan.edu.cn/70/b1/c22203a487601/page.htm)
- [Prosperity论文与官方代码](https://arxiv.org/abs/2503.03379)
- [Transitive Array](https://arxiv.org/abs/2504.16339)
- [SWAT](https://arxiv.org/abs/2405.17025)
- [FuseMax论文](https://cwfletcher.github.io/content/research/2024.micro.fusemax.paper.pdf)
- [FuseMax artifact](https://zenodo.org/records/13377043)
- [FLAT](https://people.csail.mit.edu/suvinay/pubs/2023.flat.asplos.pdf)
- [Bishop](https://arxiv.org/abs/2505.12281)
- [Sparseloop](https://arxiv.org/abs/2205.05826)

## 7. 当前新颖性可以怎样写

只有真实profile、RTL、DC和系统周期模型通过后，建议将贡献收紧为：

1. **归一化到投影的软硬件协同接口**：针对all-binary H60 selector，前推RTL Shiftmax最终门码
   和K-event目的集合，使归一化元数据直接驱动后级projection，而不做通用模式搜索。
2. **窗口组门码乘积驻留数据流**：在保持每个token独立累加的前提下，对同一block/head的多个
   独立窗口联合构建gate-code/channel目的bitmap，跨窗口复用`gate × folded weight`和权重读取。
3. **全encoder事件/残差分离的共享执行体系**：DP-TME处理T10/T2 ATLIF，LR-HTT按生命周期路由
   1-bit事件，H60/SCS/NMF/WG-GPS处理12个attention block，RPI独立保存两次block ADD和三条skip。
4. **基于真实ordered trace的可证伪DSE**：同时计product、weight SRAM、metadata、multicast、
   accumulator和fallback，以同RTL direct/G1/G2/G4/CSD/butterfly模式完成消融。

不能使用以下表述：

- “首次将ANN idea用于SNN”本身不是贡献；
- “提出Shiftmax、butterfly、product sparsity、multicast或operator fusion”；
- “105个神经元模块缩成几套硬件”作为主要创新；这是正常时间复用；
- 用合成随机79.15%乘法减少代替H67真实workload；
- 用Yosys generic cell或spike energy proxy代替芯片PPA；
- 把H68训练期Castling矩阵分支写成部署硬件，H68部署分支为零。

## 8. 真实workload必须补齐的统计

### 8.1 已挂队列的P0统计

每个模型、stage、block、head、window和row记录：

- 最终Q1.7 gate code histogram、熵、非零码比例；
- G=1的唯一`(gate_code,input_lane)`项数、active K lane数和ratio；
- score-class到final-gate的额外合并比例；
- G=2/4/8/16的唯一项数、active class、最大fanout；
- 每个G的真实有效窗口数、尾group数量和slot utilization，且严禁跨样本边界组窗；
- M=1/2/4/8/16的精确destination交付事务；
- p50/p95/p99/max，而不只报全局均值；
- ordered trace，供有限FIFO、bank conflict和group tail重放。

H67/H68必须使用`*_rtl_exact.yml`。普通dyadic配置的浮点`2^x`后量化门码不能作为RTL架构
冻结数据。排队脚本已增加硬失败审计。

2026-07-13附加审计发现，早期实现沿扁平化`batch_windows`直接分组，理论上可能把一个样本的
尾窗口和下一样本的首窗口合并。采集器现已从block的`input_resolution/window_size`计算
`windows_per_sample`，每个样本独立切分G=1/2/4/8/16，并输出每组有效窗口数。专门构造边界两侧
相同gate码的回归测试已通过；后处理会对所有G检查有效窗口守恒和尾组利用率。

### 8.2 后续P1统计

1. **窗口空间相关性**：相邻window、shifted/non-shifted block、同sequence连续frame的gate码重用；
2. **bank冲突**：row-major、diagonal、XOR hash和简单mod映射的累加器读写冲突；
3. **投影Amdahl占比**：projection product、weight read、accumulator read/write占全encoder周期和能耗；
4. **LR-HTT真实比例**：direct-forward、resident、fanout、pair assembly及阻塞原因；
5. **RPI值域**：两个block ADD、S0/S1/S2的min/max/分位数和4/6/8/12-bit误差；
6. **ATLIF量化**：4/6/8-bit参数的事件翻转、margin、valid825精度；
7. **投影量化**：FP32、per-tensor int8、per-output-channel int8的逐block误差和valid825；
8. **动态活动率**：各clock-gating域的真实toggle/idle burst，供SAIF和ICG规划。

## 9. 参数晋级和淘汰门槛

| 指标 | 晋级 | 条件保留 | 淘汰 |
|---|---:|---:|---:|
| G=1 final-gate term ratio | <=0.70 | 0.70到0.85 | >0.85 |
| score到final-gate额外合并 | >=5% | 2%到5% | <2%，不写贡献 |
| G=2/4相对G=1 product周期 | 改善>=15% | 5%到15% | <5% |
| G>1完整projection EDP | 改善>=15% | 5%到15% | <5% |
| class/group overflow | <1% | 1%到5% | >5% |
| 多播交付效率 | >=理论下界85% | 70%到85% | <70% |
| 子系统面积增量 | <=10% | 10%到15% | >15% |
| Fmax下降 | <=5% | 5%到10% | >10% |
| CSD平均非零digit | <=2.5 | 2.5到3.0 | >3.0 |
| bit-exact | 0 mismatch | 无 | 任意mismatch |

G>1不能只看projection局部speedup。若projection在完整encoder只占10%，即使局部快2倍，端到端
上限也只有约5.3%；此时应优先缩小状态或把贡献重心放回DP-TME/LR-HTT。

## 10. 公平基线

| 编号 | 基线 | 目的 |
|---|---|---|
| B0 | dense gated-K materialize + dense projection | 不利用K-zero和融合 |
| B1 | active-entry sparse projection | 只利用K-zero |
| B2 | score-class GCM-P，G=1 | 证明final gate码而非score class的价值 |
| B3 | final-gate NMF，G=1 | 保守主基线 |
| B4 | WG-GPS，G=2 | 小状态跨窗口候选 |
| B5 | WG-GPS，G=4 | 平衡主候选 |
| B6 | G=8/16 | 检验状态和tail是否反噬 |
| B7 | CSD product | 电路消融 |
| B8 | butterfly/Benes inter-segment | 复旦/Transitive Array最近邻互连消融 |
| B9 | Bishop式dense/sparse dual-path | 证明是否真的需要异构双核 |

所有基线必须使用相同折叠权重、量化码、SRAM假设、时钟约束、累加器和输出舍入。只允许关闭或
打开被比较的数据流功能。

## 11. 评价动作分账

借鉴Sparseloop和FuseMax artifact，逐层记录：

```text
Q/K event bank read/write
temporal pair assembly
TX/SC popcount
Motion-XOR popcount
score histogram read/write
exp2 LUT access
denominator add/shift
final gate directory write/read
destination bitmap write/read
folded weight SRAM read
9x8 product或CSD shift-add
multicast selector和interconnect transfer
token accumulator read/add/write
bias/scale read
RPI residual/skip read/add/write
clock和leakage
```

周期报告至少给每frame、每block、每row的mean/p50/p95/p99/max和stall分解。面积分标准单元、
SRAM macro、clock tree估计；功耗分dynamic/leakage和各域；吞吐以完整encoder frame cycle为主，
row-engine FPS只能作为局部指标。

## 12. 论文架构图

1. **网络到硬件映射图**：完整encoder，标出12个H60 block、45个T10、36个T2、两次block ADD和
   S0/S1/S2；右侧画共享DP-TME/H60/WG-GPS/RPI，不画105套实例。
2. **Normalization-to-Projection图**：左侧传统gate张量物化，右侧SCS直接发NMF descriptor，
   强调K-zero仍进入分母。
3. **WG-GPS时空数据流图**：G个独立窗口、相同block/head权重、gate-code/channel目录、一次product、
   多token独立累加。
4. **存储层次图**：event bank、active/hist、destination bitmap、weight SRAM、accumulator、RPI。
5. **时序流水图**：META/SCORE/SCS/NMF/PRODUCT/MULTICAST/COMMIT，展示双buffer和反压。
6. **新颖性对照图**：Prosperity前缀行复用、Transitive Array偏序复用、复旦权重蝶形、本文最终
   gate码目的多播，避免文字上混淆。
7. **DSE Pareto图**：G、S、M、L、P对cycle/area/energy/p99的影响。
8. **端到端消融图**：B0到B9，分别报告product、weight read、accumulator、metadata和总EDP。

## 13. RTL前冻结接口

真实profile通过后，首个RTL切片建议包括：

```text
input  group_start/group_last
input  block_id/head_id/window_id
input  gate_valid/gate_ready
input  gate_code[8:0]
input  global_input_channel
input  token_destination_segment
output weight_req/weight_rsp
output acc_req {token_id, output_tile, product_vector}
output group_done
output overflow_fallback
```

必须具备：

- ready/valid全链路反压；
- group内block/head稳定断言；
- destination active-lane守恒；
- 每token bias恰好一次；
- overflow direct fallback；
- product在多播期间稳定；
- G=1与direct整数参考逐事务等价；
- G>1与逐window G=1逐token等价；
- 随机长反压、bank conflict、空group、全K-zero、gate=0/256和尾group覆盖。

## 14. 当前GO/NO-GO

### GO

- 继续收集最终RTL gate码ordered profile；
- 继续做G/S/M/L/P的周期、状态和动作DSE；
- 以A/B/C三档形成参数化RTL前规格；
- 以Prosperity、Transitive Array、SWAT、FuseMax和复旦蝶形作为强制最近邻基线；
- 保留H67功能超集、H68编译期关闭Motion-XOR的统一硬件。

### NO-GO

- 在真实final-gate profile前冻结G=4或任何多播宽度；
- 把score class跨窗口复用；
- 直接实现完整蝶形并把它写成原创；
- 把projection局部乘法减少直接换算为全芯片speedup；
- 在投影int8 valid825、SRAM端口、RTL、SAIF和DC前宣布可发DATE；
- 因为ANN机制尚少用于SNN就忽略其先例。

当前最值得验证的架构假设是：**H67/H68的最终gate码在同一block/head的多个窗口中是否具有足够
重复度，使一次`gate×weight`生成能够覆盖多个独立token，同时其额外bitmap和accumulator状态
不抵消权重读取与乘法收益。** 真实数据若否定这一假设，主线应退回G=1 NMF并把系统贡献重心放在
DP-TME、LR-HTT和event/residual precision isolation，而不是继续包装跨窗口复用。
