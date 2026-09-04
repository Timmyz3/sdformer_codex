# HIT-Flow全Encoder统一时间瓦片架构与创新边界

**日期**：2026-07-13  
**状态**：架构候选，尚未完成RTL/DC签核  
**软件主线**：H67功能超集，H68为编译期特化  
**范围**：事件重排后的encoder；voxel前端不做RTL；decoder保留存储和接口模型

## 1. 结论

当前最有依据的架构主线不是固定稠密/稀疏异构双核，而是：

> **HIT-Flow：Head-Invariant Temporal-Tile Flow Accelerator**，利用四个stage恒定的`head_dim=32`、`9×9`空间窗口，以及网络中`T=10/T=2`两个可整除时间尺度，统一PSN时间矩阵、二值事件瓦片、TESSA注意力和稀疏projection数据流。

核心由四部分组成：

1. **HTT，Head-Time Tile**：跨stage固定`9×9×32`空间/通道几何，只切换`T=10/2`。
2. **DP-TME，Divisor-Packed Temporal Matrix Engine**：`32×10`时间矩阵阵列；`T=10`处理一个空间位置，`T=2`把10列拆成5组并行处理5个空间位置。
3. **TESSA + FGP**：时间对充分统计量注意力与factorized gated-K projection融合，不落地dense gated-K。
4. **RPI，Residual Precision Island**：ADD残差和S0-S2长跳连使用独立多位bank；binary event与multi-bit residual不混存。

这四项组成完整encoder数据流。BMRF蝶形压紧、union-event模式、多context数量和方向bank mapping仍为条件候选，不预先写成论文贡献。

## 2. 代码和真实workload给出的新约束

### 2.1 ATLIF不是递归LIF

当前`ATLIFTernaryPSN`执行：

```text
h_seq = bias + W[T,T] × x_seq
event = (h_seq >= threshold)
```

它是PSN时间矩阵和阈值发放，不存在经典`mem[t-1]`递推。硬件应实现时间矩阵阵列，不能拿一个comparator或LIF膜寄存器冒充完整ATLIF。

H67/H68 checkpoint与profile100联合审计得到：

| 口径 | 数量 |
|---|---:|
| 安装ATLIF wrapper | 105 |
| 正常forward调用 | 93 |
| 结果死亡的`attn_sn`调用 | 12 |
| 固定部署功能活跃点 | 81 |
| 活跃`T=10`点 | 45 |
| 活跃`T=2`点 | 36 |

81个活跃点的时间矩阵、bias和threshold合计只有5247个标量，但每帧活跃时间矩阵约44.244亿标量MAC。结论是：**参数存储很小，时间矩阵吞吐很大**，真正需要复用的是阵列和数据瓦片，而不是为每个wrapper复制小ROM。

### 2.2 全二值不等于全存储一位

每个Swin block有两次`ADD`残差：

```text
x = attention(x) + shortcut
x = MLP(x) + x
```

旧profile中stage/block/skip张量非零率接近100%。S0-S2三条长跳连共11,612,160元素：

| 表示 | 容量 |
|---|---:|
| 1 bit理论下界 | 1.384 MiB |
| 4 bit | 5.537 MiB |
| 8 bit | 11.074 MiB |
| 16 bit | 22.148 MiB |

1 bit仅是容量换算，旧profile没有证明残差或skip为binary。HIT-Flow因此物理分离：

- ATLIF/Q/K输出：1-bit event bank；
- gated-K、projection、block residual、S0-S2 skip：multi-bit precision island；
- S3只作为瓶颈局部张量，不计为第四条长期encoder-decoder skip。

### 2.3 固定网络几何

| stage | C | heads | C/heads | 空间 | window |
|---|---:|---:|---:|---:|---:|
| S0 | 96 | 3 | 32 | 72×96 | 2×9×9 |
| S1 | 192 | 6 | 32 | 36×48 | 2×9×9 |
| S2 | 384 | 12 | 32 | 18×24 | 2×9×9 |
| S3 | 768 | 24 | 32 | 9×12 | 2×9×9 |

四个stage的head维和窗口不变，变化的只是head数、window数和block数。这是HTT和DP-TME成立的网络特有依据。

## 3. 最近先例与不能重复的贡献

| 工作 | 已有机制 | 对HIT-Flow的边界 |
|---|---|---|
| [PSN, NeurIPS 2023](https://papers.nips.cc/paper_files/paper/2023/hash/a834ac3dfdb90da54292c2c932c997cc-Abstract-Conference.html) | 用全连接时间权重并行产生神经元状态 | 不能声称提出PSN或首次时间矩阵神经元 |
| [PTB, HPCA 2022](https://doi.org/10.1109/HPCA53966.2022.00031) | 将多个时间点打包到systolic array | 不能声称首次time batching |
| [LoAS, MICRO 2024](https://arxiv.org/abs/2407.14073) | 时间维最内层、全时间并行、spike compression | 不能声称首次temporal-parallel或spike bitmask |
| [ISCAS 2025可重构timestep加速器](https://arxiv.org/abs/2503.19643) | tick-batching和可重构并行时间步LIF | 不能声称首次可重构T模式；其网络用IAND消除ADD残差，而本网络必须保留多位ADD residual |
| [Bishop](https://arxiv.org/abs/2505.12281) | TTB、stratifier、dense/sparse异构核 | 异构双核和TTB本身不是原创；本设计不依赖近似ECP |
| [复旦ISSCC 2023 Paper 16.2](https://doi.org/10.1109/ISSCC42615.2023.10067360) | CIM内蝶形zero skipper和local-attention reuse | 蝶形网络只能作为条件性membership压紧，不能作为主贡献 |
| FLAT/FuseMax | attention算子融合和中间量驻留 | 融合本身不新；必须证明H67充分统计量、class和gated-K特有增量 |
| Prosperity, HPCA 2025 | product sparsity与精确复用 | 不采用其TCAM匹配；FGP利用同一head内共享gate的确定性代数因式分解 |

HIT-Flow可能形成的增量不是任一通用名词，而是：

1. `T=10/T=2`除数关系与固定`D=32`共同决定的满利用时间阵列映射；
2. 同一HTT坐标贯通PSN、Q/K事件、TESSA和projection，不重复格式转换；
3. binary event与ADD residual的precision-island生命周期协同；
4. H67 class-stationary attention与共享gate projection因式分解的精确融合。

## 4. 顶层架构

```text
                 +---------------- Descriptor Scheduler ----------------+
                 | site, T-mode, stage, block, head, window, precision   |
                 +-------------+--------------------------+--------------+
                               |                          |
                    Multi-bit Feature/RPI SRAM            |
                               |                          |
                         Spatial Engine                   |
                   conv / linear / BN-folded MAC          |
                               |                          |
                               v                          |
                  +---------------------------+           |
                  | DP-TME x N (32x10 each)   |           |
                  | T10 mode / 5-way T2 mode  |           |
                  +-------------+-------------+           |
                                | threshold + bitpack      |
                                v                          |
                    Binary Head-Time Tile SRAM             |
                                |                          |
                Q/K pair ------+-----> TESSA PESF/PCCC/SCS|
                                             |             |
                                     sparse {K,gate,tag}    |
                                             v             |
                                  FGP Projection Backend --+
                                             |
                                   multi-bit residual merge
                                             |
                        S0/S1/S2 Skip Lifetime Manager
```

顶层只需要一个时钟域和descriptor队列；DMA、AXI、DFT和复杂QoS控制暂不作为第一版RTL内容。论文系统图必须画出外部/shared SRAM接口，不能把全部skip假装成小片上寄存器。

## 5. HTT统一存储层次

定义：

```text
HTT(T) = [T, 9, 9, 32]
```

| 瓦片 | 元素 | 1 bit | 8 bit | 16 bit |
|---|---:|---:|---:|---:|
| HTT(10) | 25,920 | 3.164 KiB | 25.312 KiB | 50.625 KiB |
| HTT(2) | 5,184 | 0.633 KiB | 5.062 KiB | 10.125 KiB |
| TESSA一行pair bank | 81×128 bit | 1.266 KiB | - | - |

建议三级存储：

1. **L0阵列寄存器**：32个当前输入、10列partial sum和site参数广播。
2. **L1 HTT双缓冲**：一侧供Spatial Engine/DP-TME，另一侧供TESSA或下一个算子。
3. **L2 RPI/Skip SRAM**：保存多位block residual和S0-S2长期skip；容量不足时显式连接外部DRAM。

bank规则：

- binary HTT按`{window,head,time,spatial}`连续存储，T2直接形成`{Q0,Q1,K0,K1}`；
- multi-bit HTT按32通道一个bank group，便于Spatial Engine和DP-TME共享；
- shifted window的pad/roll/partition/reverse/crop由地址发生器实现，不复制物理数据；
- 长skip只在stage结束时写，在对应decoder级读取；S3走瓶颈局部ping-pong。

## 6. DP-TME微架构

### 6.1 物理阵列

一个DP-TME含320个定点MAC位置：

```text
32 channel lanes × 10 temporal-output slots
```

权重为wrapper/site共享的`T×T`矩阵，因此每拍只需广播当前输入时间对应的一行/列权重，不需要为32通道重复读权重。

### 6.2 T=10模式

```text
输入：一个空间位置的32通道 × 10输入时刻
并行：32通道 × 10输出时刻
执行：输入时间串行10拍
产出：320个binary event，按时间/通道bitpack
```

每个空间位置完成`32×10×10=3200`个MAC，用10拍，阵列利用率100%。

独立整数golden已对100组随机输入验证2,592,000个T10 hidden值和对应event，均为0 mismatch。

### 6.3 T=2模式

```text
10 temporal slots = 5 spatial groups × 2 output times
每组处理一个空间位置的32通道
输入时间串行2拍
一次处理5个空间位置
```

五个位置完成`5×32×2×2=640`个MAC，用2拍，阵列利用率仍为100%。81个空间位置需要17组，最后一组4个位置，理论slot利用率为`81/85=95.29%`。

独立整数golden已对100组随机输入验证518,400个T2 hidden值和对应event，均为0 mismatch。五路阵列的纯计算下界为34拍，但这要求5个32-lane输入银行和至少约160 bit/拍持续event出口；单32-bit出口下界仍为162拍。端口感知修正与RTL证据见`results/dptme_port_contract.md`和文档65。

### 6.4 吞吐下界

活跃ATLIF时间矩阵为4,424,388,480 MAC/帧。在500MHz且无存储stall时：

| DP-TME数 | MAC/拍 | ATLIF周期/帧 | ATLIF延迟 | ATLIF-only FPS |
|---:|---:|---:|---:|---:|
| 1 | 320 | 13,826,214 | 27.652 ms | 36.16 |
| 2 | 640 | 6,913,107 | 13.826 ms | 72.33 |
| 4 | 1280 | 3,456,554 | 6.913 ms | 144.65 |

30 FPS每帧总预算仅16,666,666拍。因此：

- 单DP-TME不能作为完整encoder的30 FPS配置；
- 双DP-TME是面积优先候选，不是已签核答案；
- 四DP-TME用于吞吐/面积DSE；
- 8-bit MAC仅在ATLIF输入、参数和累加量化验证后成立，否则必须评估更宽阵列。

## 7. TESSA与FGP融合

### 7.1 TESSA保持的精确语义

- 128-bit `{Q0,Q1,K0,K1}` pair输入；
- H67/H68充分统计量与RNE score；
- K-zero仍进入class histogram和Shiftmax分母；
- pair-coalesced class commit可旁路；
- shared class-stationary SCS；
- gated-K只为K非零token产生sparse输出；
- 1/2/4 context参数化，物理数等待ordered trace。

### 7.2 FGP：Factorized Gated Projection

对一个token/head，gate和threshold在32个K lane间共享：

```text
y_j = Σ_d W[j,d] × K_bit[d] × threshold × gate
    = (Σ_{d:K_bit[d]=1} W[j,d]) × threshold × gate
```

硬件先执行spike-selected weight accumulation，再对每个输出部分和做一次共享缩放。H67 profile100显示：

- K lane密度约1.17%；
- K非零token/head约11.35%；
- 一个非零token/head平均约3.29个活跃lane。

因此FGP的价值不在近似剪枝，而在：

1. 不写回162×C的dense gated-K张量；
2. K-zero token不读projection权重；
3. 对同一输出通道，把约3.29次`weight×gate`改为若干weight加法加一次共享scale；
4. 直接消费TESSA的`{token,head,Kbits,gate,threshold}`流。

仓库已有`ttx_late_gate_accum.sv`证明单个32-lane部分和的代数形式，但它尚未接入top，也没有完整C×C projection、weight SRAM、BN folding或反压。论文贡献必须是完整FGP backend和端到端事务减少，不能把已有小组合模块包装成新架构。

## 8. RPI与残差生命周期

RPI至少包含：

| bank | 内容 | 生命周期 | 位宽状态 |
|---|---|---|---|
| `residual_a/b` | block内两次ADD的shortcut和结果 | 一个block | 待量化profile |
| `skip_s0` | S0 pre-downsample | 到decoder3 | 待量化profile |
| `skip_s1` | S1 pre-downsample | 到decoder2 | 待量化profile |
| `skip_s2` | S2 pre-downsample | 到decoder1 | 待量化profile |
| `bottleneck_s3` | S3到resblock/decoder0 | 局部 | 待量化profile |

调度采用tile-local ping-pong：

```text
load shortcut tile
 -> attention/projection tile
 -> ADD写residual_b
 -> ATLIF/MLP tile
 -> ADD写residual_a
 -> 下一个block或stage
```

但Swin shifted-window和MLP/linear的跨通道依赖可能要求halo或完整channel partial sum。正式RTL前必须逐算子冻结tile依赖，不能只凭HTT尺寸假设所有中间量都能在25 KiB内完成。

## 9. 三档架构候选

| 候选 | DP-TME | Spatial/FGP lane | TESSA context | 存储 | 定位 |
|---|---:|---:|---:|---|---|
| HIT-L | 2×320 | 512 | 2 | 双HTT、512-bit或双256-bit供数 | 面积边界；高bypass时才接近30 FPS |
| HIT-B | 4×320 | 512 | 2 | 双HTT + RPI | 当前平衡/吞吐主候选 |
| HIT-X | 4×320 | 1024 | 4 | 多bank双缓冲 | 吞吐上界；面积/功耗高风险 |

`Spatial/FGP lane`只是分析参数，不代表已经有512/1024套乘法器。binary输入可做selected-weight accumulation，multi-bit residual和gated path需要独立定点算术口径。

淘汰规则：

1. 单DP-TME和`2×DP-TME + 256 spatial lane`已由统一预算模型淘汰；其余候选在真实trace、SRAM stall和block barrier下达不到30 FPS时继续淘汰或增加并行度；
2. DP-TME在T2/T10实际利用率低于70%，重新分bank/调度；
3. HIT-B相对“独立T10阵列+独立T2阵列”若总面积或EDP改善低于10%，不得把DP-TME列为主贡献；
4. FGP相对dense gated-K materialization若SRAM事务或能量改善低于15%，降为实现细节；
5. RPI位宽若必须16 bit且长skip外存占主能耗，论文必须以memory-bound结果报告，不能只展示attention核PPA。

统一预算模型见`results/hit_flow_full_encoder_budget.md`。在旧全网event-operation代理、8-bit skip、1.25倍保护系数下：

- `2×320 + 512 spatial + 2ctx + 512-bit + 75% event bypass`为30.23 FPS，仅是面积边界点；
- `4×320 + 512 spatial + 2ctx + 256-bit + 75% event bypass`为37.55 FPS，作为当前平衡候选；
- 若所有ATLIF event全部写回再读出，256-bit端口需约411万拍/帧，因此HTT跨算子转发必须进入架构和RTL，不是可忽略的接口优化。

## 10. 论文创新表述

完成RTL/DC和端到端模型后，可争取以下三条：

1. **Dual-timescale divisor-packed temporal execution**：利用`T=10/T=2`和固定32维head，在同一`32×10`阵列上对两种PSN时间矩阵保持高利用率。
2. **Head-invariant cross-operator tile dataflow**：同一HTT坐标贯通PSN事件生成、temporal-pair attention和factorized projection，消除多次格式转换和dense gated-K中间量。
3. **Binary-event/multi-bit-residual precision islands**：针对ADD残差和三条长跳连，将事件稀疏执行与稠密多位生命周期分离，并用真实光流workload联合调度。

TESSA的pair/class/SCS可作为第2条内部的attention微架构贡献。BMRF只有通过ordered profile和同工艺PPA后才能成为第四条。

不能声称：

- 首次时间并行、首次tick batching、首次可重构时间步；
- 首次蝶形zero skipper、首次event compression；
- 首次attention fusion或首次共享projection scale；
- all-binary网络的所有激活和skip都是1 bit；
- 105、93或81套ATLIF物理实例；
- 未包含Spatial Engine、RPI和外存流量时的“full accelerator PPA”。

## 11. DATE图表结构

### Fig.1 软件到硬件执行图

```text
H67 block代码
 -> T10/T2 ATLIF分类
 -> 93动态调用/12死结果/81活跃点
 -> 两次ADD residual + S0-S2 skip
```

### Fig.2 HIT-Flow顶层

画DP-TME、binary HTT、TESSA、FGP、RPI和S0-S2 skip manager。S3画成bottleneck-local，不画成长skip。

### Fig.3 DP-TME双模式映射

左：T10，一个空间位置、32通道、10输出列。  
右：T2，10列拆成5个空间位置×2输出列。  
下方给出100%理论阵列利用和81/85尾组利用率。

### Fig.4 跨算子HTT数据流

```text
Spatial -> PSN threshold/bitpack -> QK pair -> TESSA -> sparse FGP -> residual
```

用红色标multi-bit，黑色标binary event；不得把残差画成binary。

### Fig.5 Memory lifecycle

时间轴展示block residual ping-pong、S0/S1/S2长驻留、S3局部释放，以及HTT双缓冲。

### Fig.6 DSE和Amdahl

- 1/2/4 DP-TME的ATLIF延迟；
- TESSA A1/B2/B4；
- dense gated-K vs FGP；
- encoder模块周期、SRAM流量、能量和面积分解。

## 12. 最小实施顺序

1. 等ordered profile完成，得到K-count、PCCC、burst、bank conflict和ATLIF量化采样；
2. 用新stage数值统计冻结RPI/skip候选位宽；
3. 建`DP-TME 32×10` Python cycle/bit-accurate模型，验证T10和5-way T2映射；
4. 建完整FGP reference，覆盖C=96/192/384/768和所有head拼接；
5. 用新profile的逐算子encoder操作分账回填HIT-L/B/X全encoder周期和带宽模型，复核30 FPS淘汰结果；
6. 按RTL skill写DP-TME、HTT wrapper、TESSA B2和FGP，不先写DMA/AXI/DFT；
7. Verilator/Icarus差分、lint、SVA、Yosys后，再用目标库DC比较独立阵列与DP-TME；
8. 有SRAM宏和真实trace后生成SAIF，报告logic/memory/clock/data动态功耗；
9. 最终只把达到量化、bit-exact、30 FPS和PPA门槛的模块写成DATE贡献。

## 13. 当前签核判断

| 项 | 状态 |
|---|---|
| 代码语义与执行点分类 | 已关闭到81活跃候选，仍需软件部署审计固化 |
| HTT几何与容量 | 已关闭 |
| DP-TME代数映射 | 整数golden双模式0 mismatch；浮点量化合同和RTL未关闭 |
| ATLIF 8-bit参数 | 仅参数误差初筛，未过事件翻转/valid825 |
| TESSA attention合同 | 探索性规格已关闭，ordered trace未关闭 |
| FGP单head代数 | 已有叶子证明，完整projection未实现 |
| RPI/skip位宽 | 未关闭 |
| 端到端30 FPS | 未关闭 |
| DC/SAIF/Formality | 未关闭 |

因此当前结论是：**HIT-Flow具备比“单attention行核”更完整的架构创新候选，但仍是RTL前架构，不可直接用于论文PPA或流片签核。**
