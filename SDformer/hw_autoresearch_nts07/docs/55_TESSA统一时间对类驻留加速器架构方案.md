# TESSA 统一时间对类驻留加速器架构方案

> **2026-07-13边界更新**：本文保留为attention子系统方案。全Encoder采用`docs/59_HIT-Flow全Encoder统一时间瓦片架构与创新边界.md`；ATLIF已复核为PSN `T×T`时间矩阵而非递归膜状态。口径为105安装、93动态调用、12个调用结果死亡、81个固定部署功能活跃点。

**日期**：2026-07-13  
**工作名**：TESSA，Temporal-pair Exact Sufficient-Statistics Accelerator  
**中文名**：时间对精确充分统计量加速器  
**当前主线**：H67 为功能超集，H68/TTX 为编译期特化  
**范围**：以 encoder 和 H67/H68 attention 为主；不实现 voxel 前端；decoder 只建接口和带宽模型

## 1. 为什么不直接冻结异构双核

profile100 给出：

- H67/H68 pair-empty 分别为 73.90%/74.20%；
- active-entry 分别为 18.38/18.40；
- fold class 分别为 2.27/2.24；
- 两模型 workload 接近，但 block 间 active-entry 可相差数十倍。

因此固定 sparse core + dense core 有三个风险：

1. 两套 datapath 面积和验证成本高；
2. block 服务时间变化会导致一侧 FIFO 堆积、另一侧空闲；
3. H67 的 silent/K-zero 不能简单删除，稀疏核仍要维护 class 和 denominator。

TESSA 的主方案是：

> 使用同一套 predicate-count-reduce lane，在 fixed bitmap 和 union-event 两种表示间切换；用多个 row context 隐藏前端与 SCS 后端的不均衡；H67/H68 用编译期参数冻结功能，而非运行时保留两套完整核。

异构双核保留为 DSE 对照，不预设它一定失败。

## 2. 冻结数值语义

对一个空间 token/head 的时间对：

```text
o_t = popcount(Q_t & K_t)
q_t = popcount(Q_t)
k_t = popcount(K_t)
z_t = 32 - q_t - k_t + o_t
m   = popcount(K0 ^ K1)

H68/TTX: score_t = RNE((64*o_t + z_t) / 16)
H67:     score_t = RNE((64*o_t + z_t + 16*m) / 16)
```

必须保持：

- `PAIR_EMPTY=(Q0|Q1|K0|K1)==0` 时，两个 score 都为 2，但两个 token 仍进入 row max 和 Shiftmax 分母；
- `K_t=0` 时不产生 gated-K 输出，但 score class 仍精确依赖 `Q_t` 和 `K_peer`；
- H67 K-zero 可达 35 个 class，不能压成单一 silent class；
- gate 为 9-bit unsigned Q1.7，使用冻结 exp2 LUT、整数 denominator 和 RNE；
- RTL 对 hardware-order golden 逐位一致，算法到 hardware-order 的误差单独报告。

## 3. 顶层架构

```text
                    +---------------- Encoder Descriptor ----------------+
                    | stage/block/head/window, mode, context quota, banks |
                    +--------------------------+--------------------------+
                                               |
 ATLIF/QK pair SRAM -> Pair Fetch/Metadata -> Row Admission Queue
          |                                    |
          |                          +---------+---------+
          |                          |  Row Context Pool  |
          |                          | C0 C1 (可扩C2 C3) |
          |                          +---------+---------+
          |                                    |
          +-> Pair-Resident Exact Statistics Fabric (PESF)
                    | bitmap mode / union-membership mode
                    | pair-empty / K-zero / motion metadata
                    v
              Pair Score + RNE
                    |
          Pair-Coalesced Class Commit (PCCC)
              |                         |
        context histogram         context active bank
              |                         |
              +------ Shared Class-Stationary SCS ------+
                                      |
                          tagged {token,K,gate,threshold}
                                      |
                              ATLIF/residual consumer
```

系统层：

```text
Input feature SRAM
    -> projection/ATLIF lane cluster
    -> Q/K temporal-pair bank
    -> TESSA attention engine
    -> block residual SRAM
    -> MLP/projection/ATLIF lane cluster
    -> block residual merge
    -> next block/stage

S0/S1/S2 pre-downsample outputs -> three long-life skip SRAMs -> decoder
S3 output -> bottleneck-local fusion -> decoder0
```

12 个attention block和81个固定部署功能活跃ATLIF点通过descriptor复用物理单元；93是PyTorch动态调用口径，其中12个`attn_sn`结果死亡。硬件不按PyTorch module数实例化。

## 4. 核心模块

### 4.1 Pair Fetch 与输入接口

推荐逻辑接口：

```systemverilog
pair_valid / pair_ready
pair_last
pair_tag     = {context, stage, block, head, window, spatial_idx}
q0_bits[31:0], q1_bits[31:0]
k0_bits[31:0], k1_bits[31:0]
pair_empty, k0_zero, k1_zero, k_motion_zero  // 可选早到 metadata
```

当前 H67 每个时间片输入 `Q32 + Kpair64`，一对时间片重复传输 K pair：

```text
当前：2 * (Q32 + Kpair64) = 192 data bit/pair
pair-resident：Q0+Q1+K0+K1 = 128 data bit/pair
理论输入 data bit 下降 33.3%
```

若上游只能提供 64-bit `{Q_t,K_t}`，Pair Assembler 用两个 bank 在两拍组成 128-bit pair；只有 SRAM 支持 128-bit/拍或等效双 bank 无冲突供数时，81 个 pair/row 才能每拍一个。否则 pair-resident 只能保证减少逻辑传输 bit，不能保证把前端从 162 拍降到 81 拍。

H68 不需要 peer-K 计算，但保留同一 pair 物理布局可以复用 fetch 和 context；综合时可选择窄 64-bit token 接口或 128-bit pair 接口作为 PPA 对照。

### 4.2 PESF：Pair-Resident Exact Statistics Fabric

PESF 只输出充分统计量，不保存中间逐 lane 结果：

```text
{q0,q1,k0,k1,o0,o1,motion}
```

两种 exact mode：

#### 位图模式

- 直接消费 4×32-bit bitmap；
- 32 lane 并行 AND/XOR；
- 共享分级 popcount tree；
- 适合高密度或 packet 转换不划算的 block。

#### 并集成员事件模式

- payload 为 `{index[4:0], membership[3:0]}`；
- membership 表示该 lane 是否属于 Q0/Q1/K0/K1；
- 16-entry 小 LUT 产生七个计数器的 0/1 增量；
- 归约树累加事件，不执行 32 个全宽位运算；
- 适合 union lane 很少的 pair。

两种 mode 的输出进入同一 score/RNE 单元，必须逐 pair 0 mismatch。

### 4.3 BMRF：条件性的蝶形 Mask-Reduce Fabric

BMRF 不是主线必选项。其五级 2×2 网络对最多 32 个 lane 的 `{valid,membership,index}` 做稳定压紧，并把压紧后的 membership 直接送入统计量归约。

```text
32 lane membership
 -> 5-stage stable compaction
 -> 4/8 event issue groups
 -> 16-entry membership LUT
 -> segmented counter reduction
 -> seven sufficient statistics
```

与复旦 ISSCC 2023 的区别：

- 原设计按静态剪枝权重提取 feature；本设计压紧动态时间对事件；
- 原设计服务 CIM MAC；本设计直接形成 H67 充分统计量；
- 本设计有 bitmap exact fallback；
- 网络还可在 PCCC 中复用为同 class update 的 segmented reduce。

若综合表明 BMRF 面积/切换超过节省的 SRAM 和 popcount 能量，则删除，只保留固定 bitmap 和简单 prefix compactor。

### 4.4 Pair Score 与精确早停层次

精确 issue 层次：

| 层级 | 条件 | 行为 | 数值影响 |
|---|---|---|---|
| L0 | pair-empty | 不读 128-bit payload，向 class 2 提交两项 | 无 |
| L1 | K0/K1 zero | 计算精确 score，只写 histogram，不写 active bank | 无 |
| L2 | K motion-zero | 关闭 motion popcount/加法后级；H68 恒关闭 | 无 |
| L3 | active | 完整 score，写 active bank | 无 |

注意：若 metadata 是由读取 payload 后才生成，只能节省后级组合/寄存器切换，不能把 SRAM 读能耗记为节省。论文必须分别报告 metadata 生成位置。

### 4.5 PCCC：Pair-Coalesced Class Commit

一个 temporal pair 同时产生两个 score。第一版必须保留 PCCC bypass，并用 2-entry commit queue 吸收同一 pair 的双结果；不能假定 histogram 或 active bank 天然具有双写口：

- 两侧 K 均非零：写两个 active entry；
- 一侧 K-zero：一个 histogram update + 一个 active entry；
- 双 K-zero 且同 class：一次 `hist[class] += 2`；
- 双 K-zero 且不同 class：两个 update 进入 2-entry commit queue；
- 多 lane 并行时，同 class update 先做 segmented reduce，再写 bank。

这一结构把 pair 并行和 class-stationary SCS 结合起来。创新点不是“做 histogram”，而是时间对产生的双 score 在进入存储前按 H67 K-zero 语义合并。

晋级门槛：

- 双 K-zero 同 class 或多 lane 同 class 的可合并比例至少 40%；
- histogram SRAM 写事务至少下降 2 倍；
- merge network 能耗低于所省 SRAM/仲裁能耗。

### 4.6 Row Context Pool

每个 context 保存：

| 状态 | H67 容量 | 说明 |
|---|---:|---|
| active-entry bank | 162×56 bit = 9072 bit | `{score16,K32,token8}` |
| score histogram | 35×8 bit = 280 bit | K-zero token count |
| occupied bitmap | 35 bit | 只扫描真实 class |
| row max/sum/counters | 约 128-256 bit | SCS 状态 |
| descriptor/tag | 约 64-96 bit | 独立 row 标识 |

单 context 约 1.2 KiB，不含 pair source SRAM。首版 2-context 的主要私有存储约 2.4 KiB，参数化 4-context 约 4.8 KiB；最终物理数量尚未冻结，正式面积必须用目标 SRAM macro，而不是把 RTL 数组综合成触发器后外推。

状态：

```text
FREE -> FILL/COMMIT -> READY_SCS -> SCS_ACTIVE -> EMIT -> FREE
```

共享 SCS 处理 C0 时，前端可填 C1；4-context 参数化复核时才启用 C2/C3。context 数并不增加算术 lane 数，主要用存储换流水重叠和尾部平滑。

### 4.7 Block-Aware Row Admission

descriptor 静态字段：

```text
stage, block, heads, windows, n_tokens
representation_mode
pair_issue_width
context_quota
bank_mapping
motion_enable
class_depth/class_pipeline
```

动态 metadata：

```text
pair_empty_count estimate
kzero count estimate
union event count
FIFO watermark
free context bitmap
```

第一版只做 block-aware in-order admission；第二版在同一 block 内对独立 head/window row 做 exact OOO：

- 不预测 score，不删 token；
- 只改变独立 row 的执行顺序；
- tag 控制散写和 completion；
- block barrier 前必须收齐；
- residual/MLP 依赖不跨越 barrier。

这与 ISSCC 2022 的 speculative approximate OOO 不同，但不能声称首次乱序。

### 4.8 Shared Class-Stationary SCS

每个 context 完成前端后：

1. 扫 active bank，求 row max 和 active exp sum；
2. 只扫描 occupied K-zero class；
3. 计算 class count×exp2；
4. 得到 denominator；
5. replay active entry，生成 Q1.7 gate；
6. 输出 `{token,K,gate,threshold}`，按 token index 散写。

H67 class 35、每 class 两拍流水；H68 class 3、编译期单拍。SCS 是共享后端，不为 12 个 block 分别实例化。

## 5. 片上存储层次

```text
L0 lane registers:
  pair bitmap/event group, popcount partial, score/RNE

L1 context-private:
  active-entry bank, histogram, occupied bitmap, row state

L2 engine-shared:
  pair source ping-pong SRAM, descriptor queue, completion table,
  exp2 LUT, output reorder/scatter FIFO

L3 encoder-shared:
  projection/PSN时间输入输出HTT、ATLIF参数ROM、block residual SRAM,
  S0/S1/S2 skip SRAM, stage ping-pong feature SRAM
```

关键原则：

- `{K0,K1}` 物理相邻或同一 64-bit word；
- `{Q0,Q1,K0,K1}` 可在 128-bit pair read 中共驻留；
- score 不写全 162-entry dense SRAM；K-zero 进入 class histogram，active 才写 replay bank；
- SCS 与前端不同时访问同一 context bank，降低端口数；
- 每个 context 采用独立 bank，避免多端口大 SRAM；
- skip 只有 S0/S1/S2 三条，S3 是 bottleneck-local 保留。

## 6. 性能模型

当前 H67 row 周期代理：

```text
C_serial = 162 + max(A,1) + 2F + A + 3
```

其中 `A` 为 active-entry，`F` 为 occupied fold class。profile100 全网均值：

```text
A = 18.38, F = 2.27
C_serial ≈ 206.3 cycles/row
```

理想 128-bit pair input 的单 context：

```text
C_pair ≈ 81 + conflict_stall + max(A,1) + 2F + A + 3
```

忽略供数、双 commit、bank conflict 时约 125.3 cycles/row，相对当前 row proxy 的理论下降约 39%。这不是 RTL 或芯片结果，只是决定“pair-fused 值得实现”的上界依据。若只有 64-bit 单读口，或者同一 pair 的两个 active/histogram 更新在单写口排队，实际前端/提交周期可能重新接近 162 拍。

多 context 的稳态下界：

```text
C_steady >= max(pair_front_work, SCS_backend_work, memory_service)
```

现有 block 粗粒度重放结果：

| 模型 | pair 1-context | 2-context | 4-context | 8-context |
|---|---:|---:|---:|---:|
| H67 周期代理/帧 | 843238 | 607690 | 607489 | 607089 |
| H68 周期代理/帧 | 827927 | 612195 | 612000 | 611618 |

2-context 相对 pair 单 context 再降低 H67 27.93%、H68 26.06%；在只建模 `pair front -> SCS` 的两阶段条件下，4/8-context 相对 2-context 的额外收益低于 0.1%。该结论不足以冻结最终物理 context 数，因为它没有独立建模每 pair 双结果 commit。

加入 `fetch -> commit -> SCS` 三阶段和端口约束后：

| 模型 | 128-bit 分 bank 单写口 | 2-context | 4-context | 4 相对 2 |
|---|---|---:|---:|---:|
| H67 | 无 PCCC 合并 | 1077711 | 1037358 | -3.74% |
| H67 | PCCC 全合并上界 | 709702 | 613965 | -13.49% |
| H68 | 无 PCCC 合并 | 1081856 | 1041900 | -3.69% |
| H68 | PCCC 全合并上界 | 702601 | 618279 | -12.00% |

真实 PCCC 位于两条边界之间。因此 RTL 应支持 `NUM_CONTEXTS=1/2/4`，首版配置为 2；是否物理实例化 4 个由 ordered trace、同 class 合并率、SRAM 宏取整和 DC EDP 决定。8-context 仍不准入。

仍必须用 ordered trace 评估 1/2/4 context，报告：

- total cycles/frame；
- front/backend utilization；
- context occupancy p50/p90/p99；
- bank conflict；
- output backpressure；
- block barrier drain；
- 最长连续高密度 row。

## 7. 三档候选

| 候选 | 结构 | 预期 | 风险 | 当前结论 |
|---|---|---|---|---|
| A 保守 | 81-pair bitmap、1 context、PCCC、共享 SCS | 低风险完成 pair dataflow | 前后端串行 | 必做基线 |
| B 平衡 | fixed bitmap PESF、首版 2 context、block-aware、可旁路 PCCC | 主架构贡献最完整 | commit 端口和多context控制 | 推荐受控主线 |
| C 激进 | BMRF、4 context、row OOO、方向 bank mapping | 可能进一步降流量/冲突 | 网络面积、验证和时序 | 条件晋级 |
| D 异构对照 | sparse index core + dense bitmap core + shared SCS | 可覆盖极端密度 | 双核面积和 FIFO 失衡 | 仅 DSE 对照 |

## 8. 淘汰门槛

| 机制 | 晋级条件 | 失败处理 |
|---|---|---|
| 128-bit pair dataflow | 输入事务下降至少 25%，row proxy 周期下降至少 25%，bit-exact | 回到 token stream |
| 2 context | 相对 pair single-context 吞吐提高至少 20%，context/control 面积低于 engine 的 15% | 保留 1 context |
| 4 context | 相对 2-context 吞吐再提高至少 8% | 保留 2 context |
| event representation | 包含 metadata/对齐后 SRAM bit 至少下降 25%，转换能量有净收益 | fixed bitmap |
| PCCC | histogram write 至少下降 2×，merge 率至少 40% | 简单双 entry queue |
| BMRF | 相对 prefix/bitmap EDP 至少改善 15%，时序达到 500MHz | 删除 BMRF |
| diagonal/XOR mapping | bank cycle p99 至少下降 20%，平均不恶化超过 5% | row-major |
| row OOO | frame cycles 至少下降 10%，completion/control 面积低于 10% | block-aware in-order |
| 异构双核 | 相对同面积 homogeneous 设计 EDP 至少改善 15% | 不实例化双核 |

## 9. PPA 与吞吐评估口径

### 面积

分开报告：

- PESF/BMRF 组合逻辑；
- score/RNE/PCCC；
- context control/tag/completion；
- active/hist/pair/skip/ATLIF SRAM macro；
- SCS LUT 和 arithmetic；
- clock tree margin 和 15-20% 集成裕量。

Yosys generic cell 只能用于结构趋势，不能当 28nm mm²。

### 功耗

```text
Ptotal = Pclock + Plogic_dynamic + Psram_dynamic + Pleakage
```

SAIF/VCD 需来自至少：静态场景、典型运动、快速运动/高纹理、最坏 burst。分别报告：

- pair payload/read；
- bitmap/event conversion；
- popcount/compactor；
- class bank；
- active bank；
- SCS；
- FIFO/context/control；
- clock tree。

### 吞吐和延迟

- attention row/s、frame cycles、attention-only FPS；
- encoder cycles、encoder FPS；
- 端到端模型 FPS 只在 projection/ATLIF/residual/skip/decoder 模型全部计入后报告；
- p50/p90/p99 latency 和 worst burst；
- 不把局部 row speedup 直接写成整网 speedup。

### 存储和带宽

- logical bits、macro rounded capacity、端口数、bank 数；
- 每帧 read/write transaction；
- 平均和 p99 bank conflict；
- off-chip/encoder-shared/context-private 分账；
- bitmap、event packet、metadata、padding 分账。

### 控制复杂度

- FSM state、tag width、completion entries、FIFO depth；
- context switch 和 barrier cycle；
- deadlock/backpressure assertion；
- control area/power；
- verification state-space 和 coverage。

## 10. 验证计划

### 功能

1. 软件 hardware-order pair golden；
2. bitmap PESF 与 union-event PESF 逐 pair相等；
3. H67 35 个 K-zero class、H68 3 个 class；
4. pair-empty 双 class-2 注入；
5. PCCC 同 class `+2`、双 class 两写；
6. 1/2/4 context 配置下的任意合法完成顺序；
7. output reorder/scatter；
8. backpressure、FIFO full、flush、reset；
9. block barrier、三条 skip 生命周期。

### 形式与覆盖

- 两种表示 score 等价；
- context 间状态不串扰；
- 每个 accepted pair 恰好提交两个 token 语义；
- 每个 active token 最多发射一次；
- K-zero 不发 gated-K 但计入 denominator；
- completion 前 row token 数为 162；
- no drop/no duplicate/no deadlock；
- class、mode、context、fallback、stall cross coverage。

### 综合

- Verilator lint；
- Icarus/Verilator random regression；
- Yosys 结构趋势；
- DC 同库同约束；
- SRAM macro 替换；
- SAIF 功耗；
- Formality/LEC；
- post-synthesis SDF smoke；
- 关键候选 post-layout 或至少带线延迟估计。

## 11. DATE 论文可画的图

1. **Figure 1：软件-硬件协同总览**  
   H67/H68 all-binary encoder、部署数值合同、TESSA、AEE/spikes/energy。
2. **Figure 2：真实 workload 图**  
   S0-S3 和 12 block 的 pair-empty、active-entry、fold class 热力图；突出 block 异质性。
3. **Figure 3：TESSA 顶层架构**  
   descriptor、pair bank、PESF、context pool、PCCC、共享 SCS、ATLIF/residual/skip。
4. **Figure 4：temporal-pair dataflow**  
   当前 192-bit 重复 K pair 与 128-bit pair-resident 对比；七个充分统计量。
5. **Figure 5：PESF/BMRF 微结构**  
   bitmap mode、membership mode、mask LUT、归约树、exact fallback。
6. **Figure 6：class-stationary PCCC+SCS**  
   双 K-zero 同 class 合并、active replay、occupied class scan。
7. **Figure 7：多 context 时序**  
   首版 C0 后端与 C1 前端重叠，另示 4-context 参数化对照、block barrier 与 tag completion。
8. **Figure 8：DSE/PPA**  
   single-token、pair-A、TESSA-B、BMRF-C、异构-D 的 cycle/area/energy/EDP。

## 12. 论文命名与贡献点

候选标题：

> **TESSA: A Temporal-Pair Exact Sufficient-Statistics Accelerator for All-Binary Event-Flow Transformers**

中文工作标题：

> **面向全二值事件光流 Transformer 的时间对精确充分统计量加速器**

实现并通过门槛后，可写三项主贡献：

1. **软硬件协同**：把 H67/H68 全二值 attention 冻结为 pair-local 七个充分统计量、K-zero class 语义和 gated-K 输出，在保持 hardware-order 逐位一致的同时减少 spike 和输入搬运。
2. **架构**：提出 pair-resident、class-stationary、multi-context 的统一同构执行体系，用 block-aware 配置适应事件光流网络内部强烈异质性，而不为 12 个 block 或 H67/H68 实例化独立核。
3. **微架构**：提出 temporal pair 的 PCCC；若 PPA 通过，再加入 bitmap/event 双表示的 BMRF，将动态 membership 压紧与 H67 充分统计量归约融合。

论文中必须克制：BMRF 未通过前不能写进摘要；pair dataflow 和多 context 的 cycle/energy 未完成 RTL/DC 前只能叫 proposal。

## 13. 下一步实施顺序

1. 完成新 profile100 和 ordered trace；
2. 用 trace 冻结 pair format、context 数和 bank mapping；
3. 先冻结 128-bit 直读与 `2x64-bit` assembler 两种供数合同，以及双结果 commit queue、旁路、反压和 completion 合同；
4. 实现候选 A：fixed-bitmap pair input + pair score + 可旁路 PCCC + 1 context；
5. 建 bitmap 模式 bit-exact 回归和含供数/双 commit 的 cycle model；
6. 扩到首版 2-context、block-aware admission 和共享 SCS，并保留 1/2/4 参数化复核；
7. 综合简单 prefix compactor 与 BMRF，按 EDP 淘汰；
8. 同约束比较 homogeneous 与 dual-path；
9. 接入 encoder skeleton、ATLIF/residual/S0-S2 skip 端口模型；
10. DC/SAIF/Formality 和论文图表。

当前仅对候选 A 的 fixed-bitmap pair 数据通路和候选 B 的可退化 1/2-context 实现给出受控 RTL 准入；接口和模型必须预留 4-context 参数，但不预先实例化 4 份物理状态。PCCC 必须可旁路。正式架构签核仍为 `NO-GO`，在 ordered trace 返回前不冻结 event mode、BMRF、4-context 物理配置、row OOO、方向 bank mapping 或异构双核。独立准入审阅见 `docs/56_TESSA架构独立审阅与RTL准入条件.md`。

探索性 RTL 的模块、端口、存储、completion、counter 和 SVA 合同已冻结在 `docs/58_TESSA模块接口存储与RTL前规格.md`，机器可读版本为 `spec/tessa_attention_subsystem_spec.json`。
