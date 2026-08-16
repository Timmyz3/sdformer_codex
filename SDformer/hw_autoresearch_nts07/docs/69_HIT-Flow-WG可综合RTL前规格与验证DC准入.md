# HIT-Flow-WG可综合RTL前规格与验证DC准入

**日期**：2026-07-13  
**状态**：RTL前规格v0.1；精确语义已冻结，窗口组和物理并行度待真实ordered trace冻结  
**适用主线**：H67功能超集、H68编译期子集  
**范围**：SCS最终门码到projection输出的融合后端，以及它与HIT-Flow encoder骨架的接口

## 1. 本规格解决什么问题

现有H67/H68 RTL已经覆盖row score、SCS Shiftmax和gated-K输出，但仅有row engine不足以证明
full-encoder架构收益。HIT-Flow-WG把后端扩展为可综合的数据流切片：

```text
H60/SCS最终Q1.7门码
        |
        v
NMF归一化元数据前推
        |
        +--> direct active-entry基线
        |
        v
WG-GPS门码乘积驻留
        |
        v
分层分段多播 -> token-output accumulator bank -> bias/BN折叠输出 -> RPI
```

首版RTL的目标不是证明`G=4`必然最优，而是让同一套RTL通过参数和feature gate公平实现：

- B1：逐active-entry直接投影；
- B3：最终门码目录，`G=1`；
- B4/B5：`G=2/4`窗口组；
- B7：普通乘法器与CSD生成器消融；
- B8：简单分段网络与可选跨段网络消融。

只有真实trace、逐位验证和同约束DC均通过，功能才可晋级为论文架构贡献。

## 2. 已冻结与未冻结边界

### 2.1 已冻结的精确合同

1. 每个attention row有162个时空token，即81个T2 token pair；
2. 每个head有32个K输入lane；
3. score是有符号Q7，最终gate是9-bit无符号Q1.7，码值范围0到256；
4. projection输入为`K bit × final_gate_code`，不是score class；
5. 复用键为`{block_id, final_gate_code, global_input_channel}`；
6. 不同token具有独立累加器，不能合并输出；
7. 不同block权重不同，禁止跨block乘积复用；
8. K-zero不发projection destination，但对应score仍必须进入Shiftmax分母；
9. 每个token、每个输出通道的folded bias只提交一次；
10. H68编译期关闭Motion-XOR，不能保留训练期Castling辅助路径；
11. projection后两次block residual ADD和S0/S1/S2长skip进入多位RPI。

### 2.2 等真实profile冻结的参数

| 参数 | 探索集合 | 冻结证据 |
|---|---|---|
| `WINDOW_GROUPS` | 1、2、4、8、16 | 完整projection EDP、状态、p99 tail |
| `GATE_SLOTS` | 2、4、8、16 | 每group唯一门码数和overflow率 |
| `SEGMENT_TOKENS` | 8、16、18、32 | bank冲突、扇出、时空行局部性和布线DC |
| `MULTICAST_WAYS` | 1、2、4、8 | 交付效率和accumulator端口 |
| `OUT_TILE` | 8、16、32 | SRAM宽度、Fmax和尾块利用率 |
| `N_CONTEXTS` | 1、2、4 | pipeline重叠与状态面积 |
| product实现 | 9x8乘法、CSD | gate histogram、DC EDP |
| inter-segment | round-robin、层次选择、蝶形 | 阻塞归因与同约束消融 |

首个可综合切片默认只用于验证：`G=1`、`S=4`、`SEGMENT_TOKENS=18`、`M=2`、
`OUT_TILE=8`、`N_CONTEXTS=1`。这些不是论文最终参数。

窗口组只能在同一样本、同一block、同一head内形成。采集器和RTL都必须显式携带
`grp_window_count`处理尾group；禁止把扁平化batch中相邻样本的窗口拼为一组。DSE必须报告
`valid_window_slots / (group_contexts × G)`，不能按100% slot利用率估算状态和吞吐。

## 3. 模块层次

```text
hitflow_wg_projection_top
  |- hitflow_nmf_builder
  |    |- hitflow_gate_directory
  |    `- hitflow_direct_fallback_elastic
  |- hitflow_product_engine
  |    |- hitflow_weight_tile_reader
  |    |- hitflow_mul_product_gen
  |    `- hitflow_csd_product_gen             可选
  |- hitflow_segment_scheduler
  |    `- hitflow_segmented_multicast
  |- hitflow_accumulator_bank
  |    |- hitflow_acc_address_mapper
  |    `- hitflow_bias_commit
  |- hitflow_group_controller
  `- hitflow_wg_perf_counters
```

顶层只连接子模块，不放置数据通路运算。控制与数据通路分离；首版所有模块位于单一
`clk_core`同步复位域，不引入CDC。

## 4. 参数和类型

建议的SystemVerilog参数如下：

```systemverilog
parameter int GATE_W          = 9;
parameter int WEIGHT_W        = 8;
parameter int ACC_W           = 32;
parameter int HEAD_DIM        = 32;
parameter int TOKENS_PER_WIN  = 162;
parameter int WINDOW_GROUPS   = 1;
parameter int GATE_SLOTS      = 4;
parameter int SEGMENT_TOKENS  = 18;
parameter int MULTICAST_WAYS  = 2;
parameter int OUT_TILE        = 8;
parameter int N_CONTEXTS      = 1;
parameter bit ENABLE_CSD      = 1'b0;
parameter bit ENABLE_INTERSEG = 1'b0;
```

内部乘积至少为17-bit有符号数，因为`256 × int8`需要覆盖`-32768..32512`。累加器默认32-bit，
最终宽度必须由折叠权重、active lane上界和valid825投影量化共同签核。所有乘法、符号扩展、
饱和和舍入都必须显式编码，禁止依赖上下文位宽推断。

## 5. 顶层接口

### 5.1 group descriptor

```text
grp_valid/grp_ready
grp_context_id
grp_stage_id[1:0]
grp_block_id[2:0]
grp_head_id[4:0]
grp_first_window_id[9:0]
grp_window_count            1..WINDOW_GROUPS，支持尾group
grp_input_channel_base
grp_output_channel_base
grp_last_head/grp_last_block
```

从`grp_valid && grp_ready`到`grp_done_valid && grp_done_ready`期间，context内的block、head和
窗口范围必须稳定。不同context可流水交叠，但同一block的projection barrier之前不能提交RPI ADD。

### 5.2 SCS到NMF的最终门码流

```text
gate_valid/gate_ready
gate_context_id
gate_window_offset
gate_token_id[7:0]
gate_code[8:0]
gate_k_bits[31:0]
gate_token_last
gate_window_last
```

一条gate事务表示一个token的最终门码和32个K lane。NMF逐个置位
`destination_bitmap[slot][lane][window][token]`。`gate_k_bits==0`时不产生destination，但仍必须
接收该token，以维持token计数和group结束守恒。

### 5.3 权重接口

```text
wreq_valid/wreq_ready
wreq_block_id
wreq_global_input_channel
wreq_output_tile

wrsp_valid/wrsp_ready
wrsp_weight[OUT_TILE][WEIGHT_W]
wrsp_last_tile
```

权重是Linear与eval BN折叠、量化后的block私有列。首版假设同步1拍SRAM；测试平台必须支持
0到16拍随机返回延迟。请求tag必须包含context，防止context切换后旧响应误提交。

### 5.4 输出和完成接口

```text
out_valid/out_ready
out_context_id
out_window_offset
out_token_id
out_output_channel_base
out_data[OUT_TILE][ACC_W]
out_tile_last/out_token_last

grp_done_valid/grp_done_ready
grp_done_context_id
grp_done_status              normal/direct_fallback/error
```

`out_data`应在bias提交并执行折叠后量化合同后输出。首版若只验证整数accumulator，可将
requantizer放在顶层外，但必须保留接口边界和逐token bias恰好一次的断言。

## 6. NMF微架构

### 6.1 gate directory

每个context维护`GATE_SLOTS`个目录项：

```text
valid
gate_code[8:0]
lane_present[31:0]
destination_bitmap[32][162*WINDOW_GROUPS]
```

处理一个token时，对每个置位K lane执行：

1. gate为0时只增加`gate_zero_kone`计数，不分配目录；
2. 命中现有gate slot时置对应lane/token目的位；
3. 未命中且有空slot时分配；
4. 未命中且slot满时，将该token的K bitmap和gate写入单项direct fallback弹性寄存器；
5. 所有置位动作完成后才接受下一token，或通过bank化写口并行化。

目录的slot仅按gate code组织；`global_input_channel`由`head_id × 32 + lane`计算，不为每个channel
复制gate slot。该组织避免`S × 32`个比较器，但bitmap仍按lane分离。

### 6.2 overflow语义

overflow不能丢弃、近似或替换门码。fallback项为：

```text
{context, window, token, global_input_channel, gate_code}
```

它进入同一product engine，但不使用目录多播。首版不设置162深度整帧fallback FIFO；若下游未
接收，单项弹性寄存器保持输出并反压SCS/NMF输入。这样把低概率overflow的容量成本转换为可统计
stall，避免为最坏情况复制整帧存储。目录和fallback完成后才能发`grp_done`。论文必须报告
overflow事务、反压周期和能量，不能只报告目录命中部分。只有真实overflow burst的p99超过1且
该反压成为瓶颈时，才把fallback深度从1提升到2/4。

## 7. WG-GPS乘积驻留流水

### 7.1 流水级

```text
P0 SLOT_SELECT
P1 LANE_SELECT / global channel address
P2 WEIGHT_REQUEST
P3 WEIGHT_RESPONSE
P4 PRODUCT_GENERATE
P5 PRODUCT_HOLD
P6 SEGMENT_SCAN
P7 ACCUMULATOR_COMMIT
```

`P5`中的`OUT_TILE`乘积向量必须保持不变，直到该lane的全部destination segment提交完成。
若accumulator反压，不能覆盖product或推进slot/lane。

### 7.2 普通乘法和CSD

普通模式并行生成`OUT_TILE`个`gate_code × weight`。CSD模式把gate code解码为最多若干个
`{sign, shift}`项并执行移位加减。两者必须对全部257个gate码、全部int8权重逐位相同。

CSD是feature gate，不得改变顶层接口、目录、累加器或调度。若DC后Fmax下降超过5%、
product-generator EDP改善不足10%，或真实平均非零digit大于2.5，则删除CSD论文贡献。

## 8. 分段多播和accumulator bank

### 8.1 首版网络

token地址先分为`ceil(162×G/SEGMENT_TOKENS)`段。每周期选择一个segment中最多
`MULTICAST_WAYS`个destination，向对应accumulator bank发：

```text
{context, window, token, output_tile, product_vector}
```

首版RTL已实现当前段驻留的bank-aware选择器，要求`SEGMENT_TOKENS % N_BANKS == 0`；宽product
vector保持共享，每个bank仅携带token ID并独立反压。bank映射至少比较：

- `token_id mod N_BANKS`；
- `(token_id xor window_id) mod N_BANKS`；
- diagonal映射。

选择标准是完整trace下的p99冲突和DC互连开销，不只看平均吞吐。

### 8.2 蝶形网络的准入边界

蝶形或Benes只允许替换inter-segment选择层，不能改变NMF和数学顺序。仅在下列条件全部成立时
进入RTL：

1. 简单分段网络的多播交付效率低于理论下界85%；
2. stall归因显示inter-segment阻塞占projection周期至少15%；
3. `G>=4`时仍存在高跨段fanout，而不是少数异常row；
4. 同库、同频率、同bank数DC后，完整projection子系统EDP改善至少15%；
5. 额外面积不超过子系统10%，Fmax下降不超过5%。

复旦工作已公开in-memory butterfly zero skipper，Transitive Array已有Benes/crossbar结果分发，
因此拓扑本身不是本文新颖点。本文最多贡献“由最终门码目的集合驱动的分段互连选择与淘汰方法”。

### 8.3 累加器端口合同

每个bank首版为1R1W同步存储，采用read-modify-write流水。对同地址连续更新必须具备旁路；同周期
多个请求命中同bank时保留在segment队列，不得丢失。每个token/output tile状态包含：

```text
acc_value[OUT_TILE][ACC_W]
bias_committed
pending_update_count或completion bitmap
valid/epoch
```

如果`MULTICAST_WAYS`超过可提交bank数，性能模型必须计入冲突，不能把M路选择等同于M路提交。

## 9. 控制状态机

每个context采用以下状态：

```text
FREE
 -> BUILD_DIRECTORY
 -> DRAIN_DIRECTORY
 -> DRAIN_FALLBACK
 -> COMMIT_BIAS
 -> EMIT_OUTPUT
 -> WAIT_OUTPUT_ACK
 -> FREE
```

异常状态包括`BAD_TOKEN_ORDER`、`BAD_GROUP_TAG`、`WEIGHT_TAG_ERROR`和`ACC_OVERFLOW`。ASIC正常
配置可选择只保留sticky error flag，但仿真必须使错误立即失败。

双context时，builder和backend可分别占用不同context。仲裁必须公平，且禁止backend读取仍在
BUILD的目录。context切换只发生在明确barrier，不能依赖固定延迟。

## 10. 必须实现的性能和活动计数器

```text
groups_accepted / groups_retired
tokens_accepted / active_k_lanes
gate_zero_tokens / gate_zero_kone_lanes
directory_hits / allocations / overflows
unique_gate_lane_terms
weight_reads / product_vectors_generated
multicast_destinations / multicast_issue_cycles
segment_empty_cycles / segment_bank_conflict_cycles
acc_reads / acc_writes / acc_bypass_hits
fallback_entries / fallback_cycles
cycles_build / weight_wait / product / multicast / commit / output_stall
max_directory_occupancy / fallback_backpressure_cycles
```

面积报告必须给带计数器和去计数器两个版本。用于SAIF的仿真保留计数器，但固定不相关debug总线，
避免把观测逻辑开销错误归入主数据通路。

## 11. Clock gating合同

真实活动率尚未完成，因此首版只预留综合可识别的同步enable，不实例化工艺ICG。ordered trace后按
以下独立域分类：

| 域 | 预期门控条件 | 冻结依据 |
|---|---|---|
| NMF目录 | 无gate输入或direct-only | directory写活动率 |
| 权重端口 | gate=0、无destination、等待segment | weight read活动率 |
| product | gate=0、无destination、product驻留期间 | product生成活动率 |
| multicast segment | 当前segment为空 | segment非空率和burst |
| accumulator bank | 当前bank无提交 | 各bank写活动率 |
| fallback | 无overflow | overflow率 |

若域活动率低于0.15，进入高门控优先级；0.15到0.40为中等；高于0.40视为常开候选。最终ICG必须
使用目标库可测试时钟门单元和registered enable，不能在RTL中用组合逻辑手工与门生成时钟。

## 12. 逐位验证计划

### 12.1 单元验证

1. NMF：全K-zero、全K-one、gate 0/1/255/256、slot命中、slot溢出、尾group；
2. product：257个gate码乘全部256个int8权重的穷举等价；
3. multicast：随机bitmap、随机反压、同bank冲突、最大fanout；
4. accumulator：连续同地址旁路、正负权重、32-bit边界、bias一次性；
5. controller：context交叠、空group、错误tag、输出永久反压后恢复。

### 12.2 组合验证

对随机折叠权重和随机token流比较：

```text
dense gated-K materialize
== direct active-entry
== NMF G=1
== WG-GPS G=2/4/8/16
== CSD模式
```

比较粒度是每个`{window, token, output_channel}`的整数accumulator，不能只比较checksum。

### 12.3 真实trace验证

1. H67/H68必须由`*_rtl_exact.yml`导出；
2. 每个12个attention block至少覆盖一个完整valid825样本，最终使用完整valid825；
3. 随机插入输入、权重和输出反压；
4. 校验计数器与离线profile一致；
5. 输出送入软件后续block，检查端到端AEE/AAE与部署参考一致。

### 12.4 断言

```text
accepted_groups = retired_groups + inflight_groups
accepted_tokens = 162 * accepted_windows
每个active K lane恰好进入directory或fallback一次
每个destination恰好提交一次accumulator更新
product_hold期间product、gate、channel、tile稳定
不同block不共用weight response
每个token/output tile的bias恰好提交一次
grp_done前directory、fallback、segment和acc请求均为空
任意输出反压期间valid、data和tag稳定
```

## 13. 综合与DC准入

### 13.1 进入DC前必须满足

- Verilator全文件lint error为0，warning逐条审计；
- Icarus或Verilator自检仿真全部通过；
- 参数组合至少覆盖G=1/2/4、S=4/8、M=2/4、OUT_TILE=8/16；
- Yosys无latch、组合环、未驱动、多驱动和不可综合memory；
- RTL与整数golden在随机和真实trace上0 mismatch；
- 所有SRAM替换点具有明确深度、宽度、端口和读延迟；
- SDC定义时钟、输入输出延迟、负载、max transition和max fanout；
- 保留direct模式作为同RTL公平基线。

### 13.2 DC对照矩阵

| 组 | 配置 | 回答问题 |
|---|---|---|
| D0 | direct active-entry | 最小稀疏基线 |
| D1 | NMF G1，普通乘法 | 最终门码目录是否值得 |
| D2 | WG G2 | 小窗口组收益 |
| D3 | WG G4 | 平衡候选收益 |
| D4 | WG G8 | 状态和布线是否反噬 |
| D5 | D3+CSD | 电路级product优化 |
| D6 | D3+inter-segment | 互连是否真是瓶颈 |

所有组使用相同工艺库、PVT、时钟、不确定度、SRAM宏、OUT_TILE、acc位宽、weight位宽和真实trace。
报告总面积、标准单元、SRAM、Fmax/WNS、动态功耗、漏电、每帧周期、每帧能量、EDP和各动作数。

### 13.3 晋级门槛

- D1相对D0完整projection EDP改善至少8%，否则NMF仅作接口优化；
- D2/D3相对D1完整projection EDP改善至少15%，且面积增量不超过10%；
- 完整encoder按Amdahl折算后的EDP改善至少8%，否则不能作为主贡献；
- D5相对D3 product子系统EDP改善至少10%，完整projection改善至少5%；
- D6只在互连stall条件满足时运行，完整projection EDP改善至少15%；
- 任意方案Fmax下降超过10%、overflow超过5%或出现bit mismatch即淘汰。

## 14. 风险登记

| 风险 | 概率 | 影响 | 分数 | 缓解措施 |
|---|---:|---:|---:|---|
| 最终gate码跨窗口重复不足 | 3 | 5 | 15 | 真实G统计；退回G1 |
| accumulator SRAM主导面积能耗 | 4 | 5 | 20 | output tile、context、G联合DSE |
| 多播写端口被bank冲突限制 | 4 | 4 | 16 | ordered bank trace和三种地址映射 |
| projection占全encoder比例过低 | 3 | 5 | 15 | 完整encoder动作和Amdahl分账 |
| int8折叠投影损失精度 | 3 | 5 | 15 | valid825逐block和端到端量化验证 |
| 蝶形网络重复已有工作且物理代价高 | 4 | 4 | 16 | 只作条件消融，不作为默认主线 |
| 真实SRAM宏端口不匹配RTL假设 | 3 | 4 | 12 | 先冻结wrapper和1R1W合同 |

当前所有高风险都有明确的可证伪实验，但在实验完成前架构不能签核。

## 15. 当前实施顺序

1. 等GPU队列完成后自动导出H67/H68 RTL-exact ordered gate trace；
2. 运行G/S/M/L联合DSE和bank冲突分析，选择G1及一个G>1候选；
3. NMF、单项弹性direct fallback、G1普通乘法、18-token分段多播和2-bank accumulator/BCOD已实现并通过统一回归；继续完成集成顶层和真实trace逐事务基线；
4. 在同一乘积、多播和累加后端上增加G2/G4窗口组，保持同一顶层接口；
5. 完成真实trace回放、断言、lint和Yosys结构审计；
6. 用同一SDC与SRAM合同执行D0到D4 DC；
7. 只有瓶颈证据成立时实现CSD或inter-segment网络；
8. 将胜出点接回HIT-Flow full-encoder周期、RPI流量和SAIF模型。

当前结论是：**可以开始编写参数化G1基线的RTL，但不能冻结G>1、多播宽度或蝶形网络。** 论文级
架构创新必须来自“最终门码直接驱动跨窗口投影执行”的真实收益，以及它在full encoder中经过
状态、互连、SRAM、控制和Amdahl开销后的净收益，而不是仅靠新命名或局部乘法次数下降。
