# H67 硬件学习路线与光流 Workload Profiling 设计

**日期**：2026-07-13  
**对象**：第一次参与数字硬件、但需要共同完成 H67/PHEA DATE 加速器的协作者  
**边界**：当前不要求实现 voxel 前端；先学习 encoder、H67 attention、ATLIF、残差/skip、存储和调度  
**配套架构文档**：`docs/51_H67架构创新缺口复核与精确异构方案修正.md`

## 0. 先回答两个问题

### 0.1 你需要学什么

你不需要先学完整本数字电路教材，也不应直接从 Verilog 语法开始。当前最有效的学习主线是：

```text
软件网络真实数据流
-> tensor shape / 数据格式 / 生命周期
-> H67 一个 temporal pair 和一整 row 的数值语义
-> SRAM、FIFO、ready-valid 和时分复用
-> workload profile 如何决定核、bank、并行度和调度
-> RTL、验证、综合和 PPA
```

最终目标不是“能读懂一段 Verilog”，而是面对任意一个模块都能回答：

1. 输入来自哪里，输出送到哪里？
2. 每拍或每个 transaction 搬多少 bit？
3. 哪些数据必须保存，保存多久？
4. 哪些逻辑可以并行，哪些有先后依赖？
5. 哪些硬件可以在12个block和81个固定部署功能活跃ATLIF点之间复用？注意105是安装、93是动态调用，另有12个调用结果死亡。
6. 哪个 profile 指标决定它是否值得实现？
7. 如何证明它对 hardware-order golden 逐位一致？

### 0.2 要不要做 workload profiling

**必须做。** 当前架构创新最可能来自“光流事件数据 + H67 网络内部结构”的联合特征，而不是凭经验画一套双核方框图。

没有 profile，下面的问题都没有答案：

- 稀疏核是否真的比稠密核省周期和能耗；
- temporal delta 与 event-set sparse 哪个机会更大；
- stage0、stage2 是否应该使用不同路由阈值；
- FIFO 多深才能覆盖连续运动边缘造成的 burst；
- K-zero class folding、motion gate 和 pair-empty 注入各自实际覆盖多少周期；
- 128-bit dense pair 与 sparse index packet 谁的真实 SRAM transaction 更少；
- attention 优化后是否被 projection、ATLIF、残差或 decoder 限制。

平均 firing rate 只能说明“网络稀疏”，不能决定硬件。架构需要的是联合分布、有序 trace、尾部行为和真实存储事务。

## 1. 先建立三个时间/空间尺度

### 1.1 输入序列时间：T=10

当前 H67 配置使用：

```text
num_frames = 10
num_bins   = 10
num_steps  = 10
```

这是整网脉冲/事件序列的时间长度。patch embedding 和 ATLIF/PSN 需要理解这一层时间维。

### 1.2 attention temporal window：T_window=2

Swin window 为：

```text
window_size = [2, 9, 9]
```

H67 Motion-XOR 比较的是同一局部 window 内的两个时间片 `K0/K1`，不是一次把全部 10 个时间步放进 H67 row engine。

### 1.3 token row：162 token

一个 attention row 为：

```text
2 time slices * 9 * 9 spatial positions = 162 tokens
head_dim = 32
```

PHEA 会把 162 个 token 重组为 81 个 temporal pair：

```text
pair_i = {Q0_i, Q1_i, K0_i, K1_i}, each operand is 32 bit
```

小白最容易犯的错误是把整网 `T=10`、attention pair `T=2` 和 row token 数 `162` 混成一个概念。学习和接口文档必须始终标注是哪一层时间维。

## 2. 网络数据流必须先学到什么程度

### 2.1 四个 encoder stage

当前静态结构为：

| stage | blocks | channels | heads | head_dim | HxW | windows/block | rows/frame |
|---|---:|---:|---:|---:|---:|---:|---:|
| S0 | 2 | 96 | 3 | 32 | 72x96 | 440 | 2640 |
| S1 | 2 | 192 | 6 | 32 | 36x48 | 120 | 1440 |
| S2 | 6 | 384 | 12 | 32 | 18x24 | 30 | 2160 |
| S3 | 2 | 768 | 24 | 32 | 9x12 | 10 | 480 |

按当前 descriptor scheduler，合计：

```text
6720 attention rows/frame
1,088,640 token scores/frame
544,320 temporal pairs/frame
```

S0 占 row 数约 39.3%，S2 占约 32.1%，二者合计约 71.4%。因此只看“深层 channel 更多”会误判真实 attention 调度压力；stage0 的大量 window 和 stage2 的 block 数都很重要。

### 2.2 每个 Swin block 内有两次局部残差

真实顺序：

```text
shortcut0 = block input
x = H67 attention(x)
x = sew_function(x, shortcut0)       # attention residual
shortcut1 = x
x = MLP(x)
x = sew_function(x, shortcut1)       # MLP residual
```

代码入口：

```text
third_party/SDformerFlow/models/STSwinNet_SNN/
Spiking_swin_transformer3D.py:824-847
```

硬件学习时必须为两条 residual 分别写出：位宽、buffer 生命周期、读写时刻和是否可以与主流水重叠。

### 2.3 跨 stage skip 只有 S0/S1/S2

BasicLayer 返回 downsample 后主路径和 downsample 前 stage output：

```text
x_out = downsample(x)   # 下一 stage 主路径
return x_out, x         # x 是 downsample 前输出
```

S0/S1/S2 的 downsample 前输出形成跨 stage 长生命周期 skip。S3 没有下一次 downsample：它是 bottleneck 主路径输出，并在 decoder0 再做 bottleneck-local fusion，不计作第四条跨 stage skip。

学习时要画两类线：

```text
短线：每个 block 内 attention residual / MLP residual
长线：S0/S1/S2 -> 对应 decoder
局部保留：S3 -> bottleneck -> decoder0 fusion
```

### 2.4 padding token 也属于真实 workload

空间尺寸不总能被 9x9 window 整除。软件执行 window padding、partition、reverse 和 crop。当前 H67 路径不能想当然地删除 padding token；profile 必须标注 valid/padded token，并确认 hardware-order golden 对 padding 的真实处理。

## 3. 十四个学习单元

这里不按自然日强制进度。每完成一个单元，必须有一个产出和一次口头自测。

### 单元 1：端到端网络图

学习：输入、patch embedding、S0-S3、bottleneck、decoder、prediction。  
阅读：`Spiking_STSwinNet.py:161-182`、`Spiking_swin_transformer3D.py:1223-1246`。  
产出：一张只画主路径的框图。  
通过问题：为什么 S0 和 S3 的输出生命周期不同？

### 单元 2：residual、skip 和 fusion

学习：block 内两次 residual、S0-S2 长 skip、S3 bottleneck-local fusion、prediction feedback。  
产出：每条旁路线的生产者、消费者、位宽、保存时长表。  
通过问题：为什么不能把所有旁路线都叫 skip SRAM？

### 单元 3：tensor shape 和 layout

学习：`T/B/C/H/W`、window、head、token、head_dim、padding。  
产出：S0-S3 shape 表，以及一次 `B,C,T,H,W -> Bwindow,head,162,32` 的手工变换。  
通过问题：162 和 81 分别代表什么？

### 单元 4：二值事件和定点数

学习：1-bit Q/K event、Q7 score、Q1.7 gate、累加器、饱和和 RNE。  
阅读：`rtl_h67/h67_motionxor_score_q7.sv`、`rtl_ttx/ttx_gate_quant_q17.sv`。  
产出：数据格式表。  
通过问题：为什么 all-binary 不表示 score、gate 和 residual 都是 1 bit？

### 单元 5：H67 一个 temporal pair

学习下面的充分统计量：

```text
o_t = popcount(Qt & Kt)
z_t = 32 - popcount(Qt) - popcount(Kt) + o_t
m   = popcount(K0 ^ K1)
N_t = 64*o_t + z_t + 16*m
score_q7 = RNE(N_t/16)
```

产出：手算三个例子：pair-empty、K-zero、motion-nonzero。  
通过问题：为什么“当前时间片 Q/K 空”不一定是固定 score？

### 单元 6：一整 row 和 SCS-Shiftmax

学习：162 个 score、row max、occupied class、active replay、denominator、gate、gated-K。  
阅读：`rtl_h67/h67_score_class_row_engine.sv`。  
产出：一张 row 状态机和 buffer 图。  
通过问题：K-zero 为什么不输出 gated-K，但仍必须进入 denominator？

### 单元 7：组合逻辑和时序逻辑

学习：AND/XOR/popcount、寄存器、计数器、FSM、同步 reset、组合关键路径。  
产出：把 H67 score 分成 2-3 级流水，并说明每级寄存什么。  
通过问题：多插一级寄存器为什么提高频率却增加 latency/area？

### 单元 8：ready-valid、FIFO 和 backpressure

学习：transaction 只有在 `valid && ready` 时发生；输出阻塞时必须保持数据稳定。  
产出：包含一次 `out_ready=0` 的时序图。  
通过问题：双路径为什么需要 tag、completion 和有限 FIFO？

### 单元 9：SRAM、bank 和生命周期

学习：容量、宽度、深度、同步读延迟、单/双端口、bank conflict、ping-pong。  
产出：metadata、Q0/Q1/K0/K1、active replay、class bank、PSN时间输入输出HTT、ATLIF参数ROM、S0-S2 skip的存储表。  
通过问题：为什么逻辑上 128 bit/pair 变少，不一定减少 SRAM energy？

### 单元 10：时分复用

学习：12个attention block和81个固定部署功能活跃ATLIF点不等于物理实例数；PSN时间矩阵按descriptor和HTT分时复用。  
产出：descriptor 如何选择 stage/block/head/window/site/state address 的图。  
通过问题：增加物理 lane 数会如何影响 frame cycles、SRAM bandwidth 和 area？

### 单元 11：稀疏和稠密两种算法

学习：dense bitmap AND/XOR-popcount 与 sparse index merge/intersection 如何生成同一充分统计量。  
产出：两条路径的 operation、payload、metadata 和服务周期表。  
通过问题：为什么 sparse core 不能简单忽略所有 silent lane？

### 单元 12：验证

学习：软件 golden、directed test、random test、assertion、formal、coverage、回归。  
产出：PAIR_EMPTY、35 个 K-zero class、u=0、FIFO full、fallback、reset/flush 的 test checklist。  
通过问题：为什么 valid825 精度好不能替代 RTL bit-exact test？

### 单元 13：综合和 PPA

学习：SDC、时钟、WNS/TNS、cell area、SRAM macro、SAIF、dynamic/leakage/clock power、Formality。  
产出：一张“Yosys proxy、DC logic、SRAM macro、post-layout”证据等级表。  
通过问题：为什么 row-kernel cycle reduction 不能直接写成整网 FPS？

### 单元 14：profile 到架构决策

学习：把分布、trace、周期和能耗映射成硬件选择。  
产出：本文件第 7 节的决策矩阵，填入真实数据。  
通过问题：双核相对 pair-fused single-dense 的净收益门槛是什么？

## 4. 哪些通用硬件知识现在必须学

| 知识 | 学到什么程度 | 当前用途 |
|---|---|---|
| 布尔代数 | 会化简 AND/XOR/OR、理解 popcount 输入 | H67 score、motion、empty predicate |
| 二进制与定点 | 会算补码、位宽、截断、RNE、饱和 | Q7 score、Q1.7 gate、累加器 |
| 组合/时序电路 | 能区分 combinational/register/FSM | RTL 分级和关键路径 |
| 流水线 | 会算 latency、II、throughput | dense/sparse core 和 SCS |
| ready-valid/FIFO | 会画反压时序、理解 full/empty | 双路径调度和共享后端 |
| SRAM | 会算 depth*width、端口和 bank conflict | event、state、skip、replay |
| 性能模型 | 会从 job service time 算 frame cycles | architecture DSE |
| 功耗基础 | 理解活动率、clock gate、memory access | profile 到 energy/frame |
| 验证 | 会区分功能、随机、形式、覆盖率 | bit-exact 签核 |
| 综合/STA | 看懂 timing/area/constraint report | DC 前后签核 |

现在不需要优先学习：晶体管版图、模拟神经元、电源完整性、完整 UVM 方法学、voxel 硬件和全 decoder RTL。后续真正进入物理实现再补。

## 5. 光流 workload 有哪些值得利用的特点

### 5.1 事件稀疏但有 burst

事件相机数据在静态区域稀疏，在运动边缘、纹理和快速运动处集中。全局平均很低，不代表每个 window 都低，也不代表连续到达的 bundle 不会把 sparse FIFO 打满。

硬件启发：

- metadata-first；
- finite FIFO 和 exact dense fallback；
- 用 p95/p99 和最长 burst 选 FIFO，不用平均 density；
- 时钟门控按 eligible/gated/wake cycle 分账。

### 5.2 光流具有多尺度运动

浅层分辨率高、window 多，深层空间尺寸小但 channel/head 多。小位移、边缘细节和大位移上下文在不同 stage 的表现可能不同。

硬件启发：

- route threshold 可以 per-stage 静态配置；
- stage0 和 stage2 应优先做 trace/PPA；
- 不应默认四个 stage 使用相同 sparse lane 数或 bundle 格式。

### 5.3 相邻时间片高度相关，但运动区域会破坏相关性

H67 的 `K0 XOR K1` 和 `u=(Q0 XOR Q1) OR (K0 XOR K1)` 正好测量局部 temporal change。

硬件启发：

- pair-fused co-compute；
- `u=0` exact reuse；
- 小 `u` 的 pre-RNE delta 候选；
- motion-zero 只作为正交 gate 属性。

这些机制是否值得实现必须看 `u` 的 stage/scene 联合分布和有序 burst，而不是只看 motion-zero 平均比例。

### 5.4 空间活动集中在边缘和局部结构

同一个 9x9 window 内，Q/K 事件可能以少量 index 形式出现，适合 sparse set intersection；纹理密集或快速运动区域则适合 dense bitmap。

硬件启发：

- sparse index 与 dense bitmap 双表示候选；
- route 根据真实 `w_set` 和服务时间，不根据单 bit 平均 density；
- 需要统计 index packet 对齐后的实际 SRAM transaction。

### 5.5 遮挡、边界、噪声和高速运动决定尾部情况

这些区域往往同时影响 event density、flow error 和 attention 分布。硬件不能只在容易样本上报告平均收益。

硬件启发：

- 按 flow magnitude、event count、AEE、scene 划分 profile；
- 单列最密 stage/window 和 p99 latency；
- exact 主线不因“平均稀疏”删除 denominator token。

注意：ground-truth flow 只用于离线分组和解释。芯片运行时不能依赖 ground truth；动态路由只能使用可观察 metadata，例如 event count、`w_set`、`u`、K-zero 和 FIFO watermark。

## 6. 必须收集的 profile

### P0：静态执行图和容量

不需要 GPU：

- stage/block/head/window/token/head_dim；
- 6720 rows/frame 和各 stage 占比；
- 105安装、93动态调用、12调用结果死亡、81固定部署活跃ATLIF点及`T=10/2` shape；
- 两次 block residual、S0-S2 skip、S3 local fusion；
- 所有 buffer 的最大 live range；
- padding token 数量和位置。

用途：冻结 descriptor、地址位宽、SRAM 上限和模块覆盖。

### P1：逐 temporal-pair 精确充分统计量

每条记录至少包含：

```text
sample/scene/stage/block/head/window/spatial_token
q0_count, q1_count, k0_count, k1_count
overlap0, overlap1, motion_count
w_set, u
pair_empty, kzero0, kzero1, motion_zero
N0, N1, score_q7_0, score_q7_1
valid_or_padding
```

用途：决定 PAIR_EMPTY、u=0/delta、sparse set、dense bitmap 和 K-zero commit 的联合覆盖率。

最重要的不是五个独立比例，而是：

```text
H(stage, w_set, u, kzero_mask, motion_zero, padding)
```

### P2：逐 row 的 SCS/Shiftmax 特征

- 35 类实际 occupied class 数；
- K-zero class multiplicity；
- active replay entry 数；
- row max class、denominator 范围；
- exp2 transaction 数；
- gate zero/saturation/entropy；
- gated-K output event 数；
- row 的 pair-empty/sparse/dense route 组成。

用途：决定 histogram 深度、occupied scan、active bank、denominator 位宽和 SCS 是否仍是瓶颈。

### P3：光流场景分层

每个 sample 至少统计：

- 输入 event 总数、正负极性比例、空间 occupancy；
- 每个 window event count 的 p50/p90/p99；
- ground-truth 或预测 flow magnitude 的 mean/p90/p99；
- x/y 方向占比和高速运动比例；
- 有效 mask/遮挡/边界比例，数据可得时记录；
- AEE/AAE；
- 各 stage 的 route mix、K-zero、u 和 FIFO 工作量。

用途：判断收益是否只来自低运动/低事件场景，并探索可观察 metadata 与硬件 route 的关联。

### P4：有序 trace 和 burst

必须保留原始执行顺序：

```text
arrival order
route request
consecutive run length
stage transition
row boundary
metadata/payload address
bank id
```

由 cycle replay 补出：

```text
FIFO enter/leave
service start/end
bank conflict
join wait
SCS credit wait
fallback
commit cycle
```

用途：选择 FIFO depth、bank 数、arbiter、stage threshold，报告 p95/p99 而不是理想下界。

### P5：存储事务

分别统计：

- dense Q0/Q1/K0/K1 四个 32-bit bank 的 reads；
- sparse packet logical bits、alignment bits、64/128-bit macro transactions；
- metadata read/write；
- active replay/class bank read/write；
- PSN时间输入输出HTT、ATLIF参数、block residual、S0-S2 skip traffic；
- off-chip 和片上数据搬运。

用途：防止“逻辑 bit 减少”被误写成“SRAM energy 减少”。

### P6：端到端 Amdahl 分账

至少分成：

```text
patch/projection
ATLIF/PSN
H67 score front-end
SCS-Shiftmax/gated-K
MLP
block residual
downsample
S0-S2 skip/S3 fusion
bottleneck/decoder/prediction
control/data movement
```

用途：把 attention 子系统收益换算成整网 cycle、power 和 energy/frame。

## 7. Profile 如何直接决定架构

| profile 证据 | 决定的硬件问题 | 证据不足时的默认选择 |
|---|---|---|
| pair-empty 联合比例和 run length | 是否做常量注入、metadata 粒度 | 保留 payload，先做正确性 |
| `w_set` 分布和 sparse packet transaction | 是否做 sparse set core | pair-fused single-dense |
| `u` 分布和 delta 服务时间 | 是否做 C1/C2 temporal reuse/delta | 仅 pair co-compute |
| K-zero per-token/class 分布 | class bank 和 active replay 深度 | 现有精确 35 类 |
| occupied class p99 | class scanner 并行度 | 保留完整35类扫描 |
| sparse/dense burst | FIFO depth、watermark、fallback | 小 FIFO + exact dense fallback |
| per-stage route mix | stage-aware threshold/lane 配置 | 单套保守参数 |
| bank address trace | 4x32 bank、端口和映射 | 不声称 selective read 节能 |
| row/backend stall | 是否复制 SCS 或加 credit queue | 单共享 SCS |
| ATLIF site activity/state traffic | `P_ATLIF` lane 数和 state bank | 时分复用，不按93实例化 |
| residual/skip live bytes | local buffer 与长周期 SRAM | 分开建模 |
| scene/flow 分层 | 动态路由是否稳健 | 只用 stage-static route |
| end-to-end Amdahl | attention 优化是否值得 | 限定论文口径为 attention tile |

## 8. 数据集和统计口径

### 8.1 三个规模

| 规模 | 样本 | 用途 |
|---|---:|---|
| smoke | 2-10 | 检查 hook、shape、tag、公式 |
| architecture | 100 | raw/ordered trace、DSE、FIFO/bank sweep |
| paper | valid825 | 聚合稳定性、scene/stage tail、最终表 |

不建议直接为 valid825 保存全部 raw bitplane，文件可能过大。合理方式是：

- profile100 保存可复算 raw sidecar 或压缩 bitplane；
- valid825 保存充分统计量、有序 route 和分层聚合；
- 对最密、最稀、最高 AEE、最长 burst 样本额外保存 raw trace。

### 8.2 不能只报全局平均

每项至少报告：

```text
mean, std, p50, p90, p95, p99, max
per-stage
per-scene/sample bin
best/typical/worst trace
```

### 8.3 可复现性

每个 artifact 必须记录：

- checkpoint、config、git revision/dirty 状态；
- dataset split、sample id、随机种子；
- module coverage、missing/unexpected；
- hook 的 tensor shape 和 dtype；
- hardware-order 参数、RNE、饱和和 padding 语义；
- profiler schema version。

输出统一包含 machine-readable JSON/CSV 和中文 Markdown 报告。

## 9. 当前已经知道什么，仍缺什么

### 已知

- H67 profile100 bit activity 约 `1.5021%`；
- TTB4 pair-empty 约 `60.9633%`；
- TTB4 K-zero 约 `75.6363%`；
- TTB4 motion-zero 约 `75.6672%`；
- H67 attention row hardware-order、K-zero class folding 和 SCS RTL 已验证；
- 12 attention blocks、93 executed ATLIF 逻辑点、S0-S2 skip 结构已审计。

这些是机会证据，不是双核收益。

### 仍缺

- v2 ordered trace 正式 artifact；当前 watcher 仍在等待软件队列；
- `w_set` 与 `u` 的二维联合分布；
- per-token K-zero 与 bundle K-zero 的严格分离；
- valid/padding token 对 route 的影响；
- 按 flow magnitude/event density/AEE 分层的 route 稳定性；
- sparse packet 的真实宏事务；
- 同步 SRAM、FIFO、join、SCS credit 的 cycle replay；
- pair-fused single-dense 与 heterogeneous PHEA 的强 baseline 对比；
- 全 encoder Amdahl 和 memory traffic 分账。

## 10. 当前 profiler 的最小新增 schema

在现有 collector 上最小增加：

```text
pair_stats:
  q0_count/q1_count/k0_count/k1_count
  overlap0/overlap1/motion_count
  w_set/u
  pair_empty/kzero_mask/motion_zero/padding_mask
  N0/N1/score_q7_0/score_q7_1

row_stats:
  occupied_classes
  class_multiplicity[35]
  active_entries
  route_counts[C0..C4]
  row_max/denominator/exp_transactions

sample_stats:
  input_event_count/spatial_occupancy
  flow_magnitude_bins/x_y_bins
  AEE/AAE
  stage_route_mix
```

ordered trace 可继续压缩为 zlib/base64，但必须带 shape、dtype、schema version 和完整 row tag。

## 11. 你现在从哪里开始

第一轮只做三个学习任务，不碰 RTL：

### 任务 A：画对网络旁路线

阅读：

```text
Spiking_swin_transformer3D.py:824-847
Spiking_swin_transformer3D.py:1065-1088
Spiking_swin_transformer3D.py:1223-1246
Spiking_STSwinNet.py:161-182
```

画出：

- block 内两次 residual；
- S0/S1/S2 downsample 前 skip；
- S3 bottleneck 主路径和 decoder0 local fusion；
- prediction feedback。

### 任务 B：手算一个 H67 pair

选择三个 8-bit toy vector，手算 `o/z/m/N/score`，再解释扩展到 D=32 时硬件需要哪些 popcount。

### 任务 C：做 profile-to-decision 表

从第 7 节选择五行，回答：

```text
我要测什么？
它决定哪个硬件模块？
什么结果会让我放弃这个模块？
```

完成这三项后，再开始 ready-valid、SRAM 和 PHEA 接口学习。这样学习路径始终和真实设计决策绑定。

## 12. 我们的协作方式

你的任务：

- 按代码画图并复述数据流；
- 手算小例子；
- 对每个硬件候选提出“依据哪个 profile”的问题；
- 检查论文图是否忠实于软件。

我的任务：

- 指定代码阅读范围并解释 tensor/接口；
- 审阅你的图和答案，纠正 residual、skip、padding、位宽和时序；
- 实现 profiler、cycle model、RTL、testbench 和综合脚本；
- 把每个架构决定绑定到可复现数据和退出条件。

每个阶段都采用同一个闭环：

```text
代码事实
-> 你画图/复述
-> profile 假设
-> 脚本验证
-> 架构选择
-> RTL/golden
-> cycle/PPA
-> 论文 claim
```

## 13. 开始异构 RTL 前的最低条件

- hardware-order golden 口径冻结；
- P0 静态图、接口和存储生命周期通过审阅；
- P1/P2 联合充分统计量和 SCS profile 完成；
- P4 ordered trace 能重建 route/burst；
- pair-fused single-dense cycle model 完成；
- sparse set service time 和 packet transaction 有实测/RTL 标定；
- 双路径相对 single-dense 在有限 FIFO/SRAM 约束下有净收益；
- 退出条件明确：若净收益不足，保留 pair-fused single-dense，不为了论文名字强上双核。

这套顺序的核心是：先学会描述真实工作，再测量真实工作，最后才设计执行真实工作的硬件。
