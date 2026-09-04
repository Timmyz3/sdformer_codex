# HIT-Flow-LR微架构与RTL前接口规格

**日期**：2026-07-13  
**状态**：RTL前冻结版本v0.1；算术位宽和物理SRAM宏未签核  
**主线**：H67功能超集，H68通过编译期descriptor特化  
**范围**：重排后的full encoder和decoder存储接口；不实现voxel前端、AXI顶层、DFT和复杂QoS

## 1. 设计目标

HIT-Flow-LR不是105套PyTorch module的硬件翻译，而是一组时间复用执行资源：

```text
Descriptor Scheduler
    -> Spatial/Projection Engine
    -> DP-TME temporal matrix + threshold
    -> Event Lifetime Router
       -> TESSA/CCSP
       -> Binary HTT resident bank
       -> Multi-bit RPI/skip bank
```

首版必须满足：

1. 固定部署只执行81个功能活跃ATLIF点，12个调试`attn_sn`不进入descriptor；
2. 同一DP-TME支持45个T10点和36个T2点；
3. H67 Motion-XOR、score、class、Shiftmax、gated-K与软件整数参考逐位一致；
4. K-zero仍进入class histogram和分母，但不读取projection权重；
5. 所有ADD residual和S0-S2 skip进入多位RPI，不混入1-bit event bank；
6. 任何直通失败均通过ready/valid无损退化为resident，不允许丢event或改变顺序。

探索目标仍为500MHz和30FPS。0.5mm²、100mW只是早期约束，不是已确认芯片预算。

## 2. 首版不做的内容

- 不做Bishop/FireFly-T式稠密/稀疏双物理核；它只作为同约束基线；
- 不做BMRF蝶形membership压紧，fixed bitmap/prefix先实现；
- 不做跨样本persistent-HTT，等待ordered profile；
- 不做在线预测、近似token pruning或ECP；
- 不把H68训练期matrix auxiliary放进部署RTL；
- 不在位宽验证前把ATLIF输入、bias、threshold和RPI写死为8 bit。

## 3. 顶层模块分解

| 模块 | 数量 | 功能 | 是否状态化 |
|---|---:|---|---|
| `hit_descriptor_scheduler` | 1 | 读取冻结执行图，发stage/block/site/head/window描述符 | 是 |
| `dptme_cluster` | 参数化2或4 | PSN T10/T2时间矩阵、bias、threshold、bitpack | 是 |
| `event_lifetime_router` | 1 | 根据consumer类别执行Forward/Resident/Fanout/Pair | 是 |
| `htt_event_bank` | 2 ping-pong | 保存无法直通的1-bit HTT和Q/K pair | 是 |
| `tessa_frontend` | 1 | pair sufficient statistics、H67/H68 score/class | 是 |
| `class_stationary_scs` | 1共享 | class histogram、exp2、denominator、gate | 是 |
| `ccsp_backend` | 1 | active K selected-weight projection和共享gate缩放 | 是 |
| `spatial_projection_engine` | 参数化512 lane代理 | Conv/Linear/selected-weight accumulation | 是 |
| `rpi_manager` | 1 | block ADD、S0-S2 skip、S3 bottleneck生命周期 | 是 |
| `perf_counter_bank` | 1 | 周期、stall、事务、活动与旁路计数 | 是 |

这里的“数量1”表示一套时间复用硬件，不等于一个组合逻辑实例可在一拍完成整个网络。

## 4. Descriptor合同

每个计算任务使用固定宽度descriptor：

```text
site_id          7b   0..80固定部署活跃ATLIF点
stage_id         2b   S0..S3
block_id         3b   stage内0..5
op_kind          4b   spatial/psn/qk/tessa/fgp/add/downsample/skip
t_mode           1b   0:T10, 1:T2
consumer_class   2b   single/fanout/pair/residual
precision_id     2b   event1/rpi4/rpi8/rpi16探索模式
head_id          5b   0..23
window_id       10b   最大S0窗口数
tile_id         16b   空间/通道瓦片序号
last_flags       4b   last_time/head/window/site
```

`consumer_class`由编译器根据固定代码图生成，不在硬件中预测。H67/H68差异通过`attention_mode`和常量表选择，综合论文版本应分别冻结以避免运行时双数据路面积。

Descriptor守恒：

```text
accepted_descriptors = retired_descriptors + in_flight_descriptors
site_id不得指向12个dead_debug点
同一RPI地址释放前不得被下一生命周期覆盖
```

## 5. DP-TME接口

### 5.1 输入

```text
cmd_valid/cmd_ready
cmd_site_id
cmd_t_mode
cmd_spatial_group_count   T10固定1，T2为1..5

x_valid/x_ready
x_time_index              0..9或0..1
x_group[PACK_GROUPS][32]  参数化有符号输入；T10只用group0，T2各组对应不同空间位置

weight_row[10]            当前输入时刻到各输出时刻的site权重
bias[10]
threshold[10]
```

权重按site存于小型parameter SRAM/ROM；同一时间权重广播到32个通道lane。T2模式只读取2×2有效权重，另外8个slot通过group映射服务其他空间位置。

### 5.2 输出

```text
event_valid/event_ready
event_tag = {site,stage,block,head,window,spatial_group,time}
event_packet[PACK_GROUPS][2][32]
event_group_valid[PACK_GROUPS]
event_last
```

T2的五路打包不是在一个32-lane输入上复制五次。`PACK_GROUPS=5`时每拍必须读取5个不同空间位置的32-lane输入，8-bit代理峰值为1280 bit/拍，并需至少约160 bit/拍持续event排空带宽。原34拍是计算阵列下界，不含单word序列化。端口感知DSE见`results/dptme_port_contract.md`；首版必须参数化`PACK_GROUPS=1..5`，不能默认五路一定最优。统一阵列仍保留10×32=320个MAC以维持T10吞吐；G3/G4只在T2时门控未用槽。若物理裁剪为6或8个输出槽，当前广播数据流下T10需分两遍并从810拍增至1620拍；更窄阵列惩罚更大。

### 5.3 算术规则

```text
acc[o,c] = bias[o] + sum_i weight[o,i] * x[i,c]
event[o,c] = (acc[o,c] >= threshold[o])
```

必须冻结：输入/权重/bias/threshold/acc位宽、乘法截断、舍入、溢出和比较符号。当前整数golden只证明映射正确，不证明最终定点格式；RTL首版全部参数化并禁止隐式截断。

## 6. Event Lifetime Router

### 6.1 静态分类

旧profile100与代码消费者得到：

| consumer类别 | 模块 | event元素/帧 | 占活跃输出 | 行为 |
|---|---:|---:|---:|---|
| single | 45 | 421,536,960 | 80.13% | 直通优先，阻塞时resident |
| fanout | 12 | 34,836,480 | 6.62% | 同一`proj_sn`event供Q和K两个消费者 |
| pair | 24 | 69,672,960 | 13.24% | Q/K按tag对齐形成128-bit pair |
| dead | 12 | 34,836,480 | 不计入活跃 | descriptor删除 |

### 6.2 路由状态机

```text
IDLE
 -> LOOKUP consumer_class
 -> SINGLE_FORWARD --downstream stall--> SINGLE_RESIDENT
 -> FANOUT_Q -> FANOUT_K/retain -> FREE
 -> PAIR_WAIT_PEER -> PAIR_ISSUE -> FREE
 -> ERROR_TAG_MISMATCH
```

首版不使用一个大crossbar。每类消费者使用固定出口和小型elastic FIFO，descriptor决定出口。fanout可采用一次写入、Q/K顺序双读，也可在Q/K engine可同时接收时广播；两种模式必须输出相同tag序列。

### 6.3 存储

`htt_event_bank`至少支持：

- 32-bit event word加tag；
- 一写一读基础端口；
- ping-pong bank避免producer和consumer争用；
- pair assembly的Q0/Q1/K0/K1存在位；
- 每项valid、epoch和sequence边界，防止旧数据误命中。

80.13%是静态直通资格上界。实际bank尺寸由ordered trace的最大连续stall和p99 occupancy决定，不按全帧526M元素配置片上SRAM。

## 7. TESSA/CCSP合同

### 7.1 Pair输入

```text
pair_valid/pair_ready
pair_tag = {stage,block,head,window,token_pair}
q0[32], q1[32], k0[32], k1[32]
```

前端每个pair生成两个token的exact sufficient statistics。H67 score使用Motion-XOR/TTX路径；H68部署模式关闭训练期aux，复用H67功能超集中的编译期子集。

### 7.2 多级精确issue

| 级 | 条件 | 精确行为 | 可门控单元 |
|---|---|---|---|
| L0 | pair empty | 写入冻结silent class计数，不读active payload | popcount后级、active bank |
| L1 | K-zero | 更新score class和分母，不写active K | K payload SRAM、FGP权重读 |
| L2 | motion-zero | 执行基础score，关闭Motion-XOR增量树 | motion branch |
| L3 | active | 完整score/class并写active stream | 无 |

这里的skip是计算/访存门控，不是删除数学项。L0/L1必须向class histogram注入与软件相同的常量或score类。

### 7.3 CCSP输出

```text
active_valid/active_ready
active_tag = {stage,block,head,window,token}
k_bits[32]
gate_q17
threshold
class_id
```

FGP只对`k_bits=1`的lane读取权重并累加，随后对partial sum乘一次共享gate。必须保留dense gated-K旁路和STDP式列流基线模式，以便同一RTL做消融。

## 8. RPI合同

RPI管理的不是ATLIF event，而是：

- 每个Swin block attention后的ADD结果；
- 每个Swin block MLP后的ADD结果；
- 两个MS ResBlock的identity与ADD结果；
- S0、S1、S2 encoder-decoder长skip；
- S3局部bottleneck。

接口：

```text
rpi_req_valid/rpi_req_ready
rpi_op = READ/WRITE/ADD/FREE
rpi_tag = {stage,block,lifetime_id,tile_id}
rpi_precision_id
rpi_data
```

首版支持4/8/16-bit参数化综合，但在ordered value profile和valid825量化前不选择最终物理位宽。长skip容量若超过片上预算，RPI manager必须显式发外部存储事务，论文不能隐藏该流量。

## 9. 调度顺序

一个Swin block的硬件顺序：

```text
1. RPI读取block shortcut tile
2. ATLIF proj_sn -> fanout给Q/K linear
3. ATLIF sn_q/sn_k -> pair assembly
4. TESSA score/class/SCS -> CCSP/FGP
5. projection完成 -> RPI ADD shortcut
6. ATLIF mlp.sn1 -> fc1
7. ATLIF mlp.sn2 -> fc2
8. RPI ADD block中间结果
9. block结束；按stage决定next block/downsample/skip write
```

Swin shifted-window的roll/partition/reverse/crop只改变地址，不复制整张量。block边界和RPI ADD是可见barrier；TESSA可在head/window粒度与Q/K producer重叠，但不能越过需要完整projection的ADD。

## 10. 三档候选与选择

| 候选 | DP-TME | Spatial lane | ctx | event端口 | 风险 | 用途 |
|---|---:|---:|---:|---|---|---|
| HIT-LR-L | 2 | 512 | 2 | T2 G3/128-bit | 吞吐余量小 | 面积边界 |
| HIT-LR-B | 2或4待DC | 512 | 2 | T2 G4/128-bit | 中 | 当前平衡RTL参数点 |
| HIT-LR-X | 4 | 1024 | 4 | T2 G5/256-bit | 输入银行/布线高 | 上界，不先实现 |

首版RTL采用参数化模块，同时跑`N_DP=2/4`、`PACK_GROUPS=3/4/5`、`N_SPATIAL=512`、`N_CTX=1/2/4`。不为每个候选复制控制器。`N_DP=4`不再作为默认平衡点，必须在端口感知全encoder周期和DC面积后晋级。

## 11. 性能计数器

至少实现：

```text
cycles_total
cycles_dp_busy / stall_input / stall_output
event_forward_words
event_resident_write_words / read_words
fanout_broadcast_words / fanout_second_reads
pair_wait_cycles / pair_issued
issue_l0_empty / l1_kzero / l2_motionzero / l3_active
class_commit_words / pccc_merged_updates
fgp_weight_reads / fgp_active_lanes
rpi_read_bits / write_bits / external_bits
fifo_high_watermark / p99由trace工具离线计算
```

这些计数器是论文PPA与workload因果链的证据，不是可选debug逻辑。综合时应分别报告带/不带计数器面积，功耗测试可冻结不活动计数器。

## 12. 正确性断言

1. accepted event最终必须forward或resident一次；fanout必须恰好消费两次；
2. pair只有四个存在位齐全且tag一致时发射；
3. K-zero可禁止active-bank写和FGP权重读，但不得禁止class commit；
4. `event_last`前不得切换site参数；
5. RPI `FREE`前所有消费者完成；
6. sequence/epoch切换必须清空pair valid和persistent状态；
7. backpressure任意持续时输出数据与tag保持稳定；
8. H68编译模式不得实例化或切换训练期aux路径。

## 13. 验证与综合顺序

1. DP-TME T10/T2定向与随机逐位测试；
2. router四类路径、随机反压、tag错配与bank溢出测试；
3. pair assembly加TESSA hardware-order差分；
4. CCSP dense/STDP/selected-weight三模式等价；
5. RPI ADD和4/8/16-bit模式边界测试；
6. block级H67/H68 trace replay；
7. full encoder descriptor replay和计数器守恒；
8. Verilator lint、Icarus仿真、Yosys结构审计；
9. 目标库DC、SRAM宏替换、Formality和真实trace SAIF。

## 14. 架构创新签核条件

- DP-TME相对独立T10/T2面积或EDP改善至少10%，相对通用时间组PE至少5%；
- LR-HTT实际forward比例至少40%，相对局部fusion减少总片上事务至少20%，系统EDP改善至少12%；
- CCSP相对dense gated-K减少事务或能量至少15%，相对STDP列流EDP至少改善8%；
- 30FPS最坏trace保留至少10%周期余量；
- 所有结果包含metadata、FIFO、router、RPI、SRAM和控制开销。

任何机制未过门槛都必须降级为实现细节或删除，不能仅凭命名进入DATE贡献。
