# TTB 异构双路径微架构评估

**日期**：2026-07-12  
**对象**：SDformer DATE 的 H60 TTX / H67 Motion-XOR TTX 主线  
**范围**：评估引入 Bishop 风格 Token-Time Bundle（TTB）、density stratifier，以及 dense/sparse heterogeneous cores 的可行性；不修改模型语义，不给出未经综合和存储宏标定的 PPA 数值。

## 0. 执行结论

1. **立即可冻结的是 TTB 数据布局和单路径精确门控，不是 Bishop 式双核。** `T=2`、`head_dim=32` 允许把同一空间 token/head 的两个时间片装入 64-bit word；Exact Delta-TTX 的 `Q_toggle OR K_toggle` 为零时可直接复用前一分数。二者都是数值等价优化。
2. **现有 100-sample profile 支持继续做 Exact Delta。** 全局 temporal union-toggle density 为 `2.7832%`，对应 t1 lane skip 理想上限 `97.2168%`、整个 `T=2` TX compare reduction 理想上限 `48.6084%`。这些是 compare 数上限，不是周期、功耗或能效结果。
3. **现有 profile 不足以批准 dense/sparse 双核。** 当前 JSON 有逐 block Q/K activity、旧 TTB1/TTB2 代理和总体 toggle 计数，但没有已落盘的 locality-v2 raw count：zero-update token/head、changed-lane histogram、bundle4/8 empty、changed-token run length。没有密度分布和队列行为，就无法证明第二个 core 能覆盖其 SRAM 端口、队列、仲裁和空闲功耗。
4. **推荐采用“两阶段平衡方案”。** 先实现/标定保守单路径基线 C0；只有当 locality-v2、cycle model 和同工艺综合共同证明交叉点成立，才升级为 B1 的一密一疏双路径。A2 多队列多核方案不进入当前 DATE 主线。
5. **Bishop 只能作为机制来源。** 本项目不得引用 Bishop 的 speedup、area、power 或 energy 数值作为 SDformer 结果；本文件也不提供虚构 PPA。

## 1. 证据基线与边界

### 1.1 已读取材料

- `docs/22_AllBinary_NTS_H60_P0与量化验证结果.md`
- `docs/22_AllBinary主线硬件规划.md`
- `docs/24_AllBinary_UniBinH60_RTL启动与DATE硬件设计.md`
- `docs/36_TTX主线硬件方案与RTL差距分析.md`
- `docs/42_H67运动XOR与有界TTX硬件增量.md`
- `neuron_autoresearch/DATE_IDEA_PORTFOLIO_20260712.md`
- H60 valid40 P0 JSON：`.../allbinary_nts_h60_ft_ep2_valid40/nts11_hardware_p0_profile.json`
- TTX 100-sample Exact-Delta JSON：`.../date11_ttx_ep2_delta_profile100_exact_20260711/nts11_hardware_p0_profile.json`
- H67 epoch19 valid825 `spike_profile.json`
- `results/ttx_temporal_pair_layout_model.{md,json}`

### 1.2 项目内已测事实

| 项 | 项目数据 | 允许的解读 |
|---|---:|---|
| 网络形状 | all12 H60-family；`T=2`；`head_dim=32`；每窗总 token `162=2x81` | 可设计固定 temporal-pair 数据布局 |
| TTX 100-sample Q toggle | `0.7983%` | 仅为 lane activity |
| TTX 100-sample K toggle | `1.9946%` | 可与 H67 Motion-XOR 的 K XOR 前端复用 |
| TTX 100-sample union toggle | `2.7832%` | 支持继续评估 sparse delta update |
| t1 理想 lane skip | `97.2168%` | 不是整块 attention 或整芯片节能 |
| 全 `T=2` 理想 TX compare reduction | `48.6084%` | 未扣 state SRAM、queue、control、Shiftmax |
| TTX stage TTB2 empty 代理 | S0 `28.96%`、S1 `74.24%`、S2 `63.36%`、S3 `60.93%` | 只支持 work-issue/gating 探索，不证明双峰或双核收益 |
| K-zero token | S0 `78.67%`、S1 `97.28%`、S2 `93.55%`、S3 `89.05%` | 支持 exact late-scale/output folding；不能据此跳过 Shiftmax |
| H67 epoch19 valid825 | AEE `1.4671`、AAE `9.4155`、spikes `26.3898G` | 算法候选已达门槛；`energy_uj` 明示为 spike-only proxy，不能替代硬件能耗 |
| temporal-pair 事务模型 | 未合并基线时请求/传输可从 `324/20736 bit` 到 `162/10368 bit` | 仅当 RTL address trace 证明基线未合并；逻辑容量不下降 |

### 1.3 尚不存在的证据

- 无目标工艺、频率、面积和功耗预算，不能做 architecture sign-off。
- 无 SRAM macro/CACTI、DC/Genus、PT-PX/Joules 或 SAIF 标定结果。
- 现有 Yosys `24313 cells, memories=0` 是旧 H60/TTX 兼容原型，不代表真实 SRAM 或本方案面积。
- 无双路径 RTL、queue trace、bank conflict、backpressure、利用率或 post-synthesis PPA。
- 无 H67 统一 dyadic 部署后的完整 attention logic + memory + control 能耗。
- profile 代码已经定义 locality-v2 字段，但本次读取的 100-sample JSON 未包含这些 raw fields，不能把“代码能统计”写成“数据已测得”。

## 2. 必须保持的数值语义

H67 保持统一 all12 TTX 数据流：

```text
S64(t,p) = 64*n11(t,p) + n00(t,p)       # alpha0 = 1/64 部署图
M_K(p)   = popcount(K0(p) XOR K1(p))
S_H67    = S_TX + M_K/4                 # 在原始 score 域合并后统一舍入
center -> Shiftmax -> gate*K
```

Exact Delta 在 t1 只更新发生翻转的 lane：

```text
U = Q0 XOR Q1 OR K0 XOR K1
if popcount(U) == 0:
    S64_1 = S64_0
else:
    S64_1 = S64_0 + sum(delta_contribution[lane] for lane in U)
```

实现约束：

- 必须保存 previous Q/K 共 `64 bit/token/head`，或等价的 2-bit contribution class；仅保存 match bit 不够。
- `S64` accumulator、10-bit score、9-bit gate及饱和/舍入规则必须独立审计。
- H67 的 `K0 XOR K1` mask/popcount 与 Delta 的 K-toggle 可以共享前端，但 Motion-XOR score 路径和 Delta 更新路径必须分别计数，不能重复申报共享收益。
- `K=0` 只保证 `gate*K=0`。除非已证明 score/center 的精确折叠规则，否则不能在 Shiftmax 之前删除该 token 的 score。
- 旧 TTB1/TTB2 是 Q-token activity 的调度代理，不等同于 Delta locality-v2 的 4/8-token changed-lane bundle。

## 3. 候选微架构

### 3.1 C0 保守：Temporal-Pair TTB + 单路径分组引擎

```text
64-bit Q/K temporal-pair SRAM
  -> shared XOR/nonzero front-end
  -> exact empty/reuse detector
  -> fixed 8-lane grouped TTX engine (single issue path)
  -> score accumulator
  -> shared center/Shiftmax/gated-K backend
```

设计要点：

- 保留一个物理 TTX row engine 和一个有序 work queue；不增加第二套 compute core。
- t0 做完整 32-lane TX；t1 用同一阵列按固定 8-lane group 扫描。全零 group clock-gate，非零 group 在原阵列更新。
- K-toggle mask 同时供 H67 Motion-XOR popcount 和 Delta 更新使用；Motion count 每个 spatial token/head 计算一次并供两个时间片消费。
- 采用静态 in-order schedule，无跨 bundle reorder buffer。
- TTB4/8 只用于组织 address、valid 和 clock-enable，不把低密度 bundle编码成可变长 index stream。

**优点**：最接近现有 row engine；控制、验证和 SRAM 端口增量最小；即使密度不是双峰也不会留下长期低利用率的第二 core。  
**缺点**：固定分组仍为稀疏 t1 支付 group scan/issue 周期，不能充分接近 `2.7832%` lane-density 的理想上限。

### 3.2 B1 平衡：一密一疏双路径，共享后端

```text
64-bit temporal-pair SRAM (banked)
  -> shared Q/K XOR + popcount + density stratifier
       |-> dense FIFO  -> 32-lane full-recompute core --|
       |-> sparse FIFO -> 4/8-lane delta-update core  --|-> tagged score accumulator
       `-> reuse/empty bypass --------------------------|-> shared center/Shiftmax/gated-K
```

设计要点：

- 一个 32-lane dense core 处理 t0 和高密度 t1；一个窄 sparse core 消费 changed-lane bitmap/index。
- 两条前端只共享一个 center/Shiftmax/gated-K 后端，不复制最昂贵且每行必须执行的归一化路径。
- 每个 score 带 `{stage, block, window, head, spatial_token, timestep}` tag；row-complete bitmap 保证全部 score 就绪后才启动 center。
- sparse core 优先采用 bitmap + fixed group，不默认采用 RLE。只有 changed-run profile 显示平均 run 明显大于 1 且 burst 访问可合并时才改为 RLE。
- dense/sparse FIFO 深度、仲裁权重和 SRAM bank 数由 trace sweep 决定，不在微架构阶段拍定。

**优点**：能把 t0 稳定交给 dense core，同时让低密度 t1 在窄 core 上减少无效 lane switching；与 H67/Exact-Delta 的共享 XOR 前端一致。  
**缺点**：增加第二读路径或 operand staging、双 FIFO、tag/completion、仲裁和两套局部时钟门控；若 sparse arrival 呈 burst 或 dense core 已有足够空隙，双核可能只增加面积与静态功耗。

### 3.3 A2 激进：多 sparse cluster + dense core 的动态异构调度

```text
multi-bank pair SRAM
  -> hierarchical stratifier
       |-> empty/reuse bypass
       |-> 2x sparse lane clusters (bitmap/RLE selectable)
       `-> 1x dense vector core
  -> out-of-order completion table
  -> shared or duplicated score banks
  -> center/Shiftmax/gated-K
```

设计要点：

- 4/8-token bundle 先按 empty、low、high 分级，再由每 token update count 二次路由。
- 两个 sparse cluster 并行吸收低密度 burst；dense core 可与其并行处理 t0 或高密度 t1。
- 支持 per-stage threshold、work stealing、可变长 index/RLE、out-of-order completion 和 credit backpressure。
- 若单一 Shiftmax 后端形成瓶颈，可能需要双 score bank 或双 row context；不建议在 profile 前复制 Shiftmax。

**优点**：提供最高的稀疏并行上限，适合 locality 很强且不同 stage 到达率差异大的情形。  
**缺点**：最容易被 SRAM 端口、队列、reorder、负载不均和 backend serialization 抵消；验证状态空间显著增加。当前没有证据支持其进入 DATE 主线。

## 4. 精确路由规则

### 4.1 路由输入

每个 spatial token/head 由共享前端生成：

```text
q_mask      = Q0 XOR Q1             # 32 bit
k_mask      = K0 XOR K1             # 32 bit
update_mask = q_mask OR k_mask      # 32 bit
u           = popcount(update_mask) # 0..32
m           = popcount(k_mask)      # H67 motion evidence
```

bundle metadata 至少包括：

```text
stage/block/window/head/bundle/timestep
token_valid_mask
per-token update_count or grouped update bitmap
row_context_id
```

### 4.2 C0 规则

```text
t0                         -> SINGLE_FULL
t1 and update_mask == 0    -> REUSE_S64
t1 and update_mask != 0    -> SINGLE_GROUPED_DELTA
K == 0 after gate available -> suppress gated-K/projection read/write
```

该规则完全由 exact condition 驱动，不使用低/高密度阈值。

### 4.3 B1 规则

初始候选阈值仅作为 sweep 参数，不是冻结常数：

```text
t0                          -> DENSE
t1 and u == 0               -> REUSE_S64
t1 and 1 <= u <= theta      -> SPARSE_DELTA
t1 and u > theta            -> DENSE_RECOMPUTE
```

`theta` 必须通过以下交叉点选择：

```text
latency_sparse(u) + queue_wait_sparse + energy_sparse(u)
    versus
latency_dense + queue_wait_dense + energy_dense
```

实现可以先 sweep `theta in {2,4,8,12,16}`，但最终阈值必须来自同工艺 post-synthesis 单元成本与 workload trace，不得只按直觉选择 `8`。

bundle 级规则：

- bundle 内所有 token `u=0`：整包 bypass，不分配 compute core。
- mixed bundle：按 token bitmap 发往对应 FIFO；不得因 bundle 平均密度低而把高密度 token 塞入 sparse core。
- FIFO 超过高水位：只允许把本可 sparse 的 token 回退到 dense full-recompute，不能丢弃或近似更新。
- dense/sparse 均产生 bit-exact `S64` 后才能进入共享 row completion；调度顺序不能改变 center/Shiftmax 的输入集合。

### 4.4 A2 规则

在 B1 规则上增加 per-stage `theta_s`、run-length/bitmap 选择和 work stealing。只有以下条件同时满足才启用 RLE：

```text
average_changed_run_length > measured_RLE_crossover
and index_bytes_RLE < bitmap_bytes
and SRAM trace shows burst coalescing
```

否则固定 bitmap，避免 RLE decoder、变长 FIFO 和碎片化访问。

## 5. Memory / Compute / Control 成本账本

以下为结构性成本和计量公式，不是面积或功耗估值。

### 5.1 Memory

| 项 | C0 | B1 | A2 | 必须计入的量 |
|---|---|---|---|---|
| temporal-pair Q/K | 1 份 | 1 份，多 bank/operand staging | 多 bank，可能双读端口 | 逻辑数据 `10368 bit/window/head`；物理 bit、bank、端口分别报 |
| previous Q/K 或 contribution state | 必需 | 必需 | 必需 | `64 bit/spatial-token/head` 或等价 class state；不得与 pair buffer 无条件重复/抵消 |
| S64 accumulator | 1 context | 多 in-flight context | 更多 context | 深度、位宽、读改写次数 |
| queue metadata | 单有序队列 | dense+sparse FIFO、tags、completion bitmap | 多 FIFO、RLE/index、ROB/credit | entry width x depth、实际 occupancy、读写次数 |
| score/row buffer | 单 bank 可行 | 至少仲裁或双 staging | 可能双 bank | row completion 前 lifetime 与端口冲突 |
| skip/event SRAM | 不变 | 不变 | 不变 | 仍需与 attention state 分列；不能用 1-bit skip 收益掩盖新状态成本 |

物理容量报告应至少分成：payload bits、ECC/valid/tag bits、bank padding、macro peripheral。temporal-pair packing 只改变布局；在 baseline 已 coalesced 时不减少 traffic，更不减少逻辑容量。

### 5.2 Compute

| 项 | C0 | B1 | A2 |
|---|---|---|---|
| t0 TX | 原 32-lane/full grouped engine | 32-lane dense core | 32-lane dense core |
| t1 Delta | 同一 engine 的 8-lane group scan | 4/8-lane sparse core，必要时 dense recompute | 2 个或更多 sparse cluster |
| H67 Motion-XOR | 与 K-toggle mask共享，popcount 分时 | stratifier 前端共享；计一次 | 分层前端共享；需审计扇出 |
| popcount/add | 单套复用，周期较多 | dense/sparse 各自局部树 + accumulator | 多局部树 + merge |
| center/Shiftmax/gated-K | 单套 | 单套共享 | 默认单套；复制需单独证明瓶颈 |

理论 compare 量可以用：

```text
C_full = N_spatial * heads * windows * (32_t0 + 32_t1)
C_delta_ideal = N_spatial * heads * windows * (32_t0 + sum(u_t1))
```

但实际周期还要加 mask generation、dispatch、FIFO、SRAM、accumulator RMW、row completion 和 backend cycles。`48.6084%` 只能作为 TX compare reduction ceiling。

### 5.3 Control

| 控制项 | C0 | B1 | A2 |
|---|---|---|---|
| FSM/descriptor | 小增量 | 双 FIFO + arbiter + completion | hierarchical scheduler + ROB/credit |
| ordering | in-order | core 内可并行、row 边界有序 | out-of-order completion |
| backpressure | 单链路 | dense/sparse/backend 三方 | 多队列、多 bank、多 producer |
| clock gating | group/core 粒度 | front-end、dense、sparse、backend 分域 | cluster + queue + bank 粒度 |
| verification | exact bypass/group delta | 加路由交叉点、并发和饥饿 | 加 OOO、RLE、work stealing、死锁 |

控制能耗必须记录 `bundle_seen/empty/sparse/dense/fallback`、FIFO occupancy、stall、bank conflict、core active cycles 和 backend wait，不能只报告 skipped compares。

## 6. 必补 Profile 指标与数据格式

### 6.1 P0：决定是否值得做 sparse path

必须按 `checkpoint x sequence x stage x block x head` 保留 raw count，再做 element-weighted 汇总：

| 指标 | 用途 |
|---|---|
| `token_heads`, `zero_update_token_heads` | exact bypass 覆盖率 |
| `update_count_hist[0..32]` | 选择 dense/sparse crossover `theta` |
| `bundle4/8_total`, `bundle4/8_empty` | 决定 bundle 粒度和整包 gate |
| `changed_token_runs`, run-length histogram | bitmap 与 RLE 选择 |
| Q/K/union toggle raw elements | 复核现有 `2.7832%`，避免均值偏差 |
| K-toggle histogram | H67 Motion popcount 的真实活动 |
| K-zero token/bundle | gated-K 与 projection folding |
| consecutive empty bundles | ICG amortization 与 wakeup 成本 |
| per-sample p50/p90/p99 | 防止总体均值隐藏 burst |

仅有全局平均 toggle density 不足以判断双峰。双路径准入要求分布在 `u` 接近 0 和接近 dense-crossover 两侧均有足够质量；若绝大多数非零 token 都集中在 `u=1..4`，可能只需 sparse-only 分组引擎；若集中在中间且无明显两群，单一可门控 core 更合理。

### 6.2 P1：决定双核是否有净吞吐/能耗收益

- cycle trace：arrival timestamp、route、service start/end、queue wait、backend wait。
- 每 core issued token、lane utilization、active/idle/gated cycles。
- dense/sparse FIFO occupancy 分布、overflow/fallback、starvation。
- SRAM 每 bank read/write、bit count、conflict、stall、row-buffer hit。
- bitmap/index/RLE metadata bytes 和访问次数。
- score accumulator RMW 次数与 row context lifetime。
- H67 Motion、Delta、Shiftmax、gated-K 分项 operation count；禁止并入 neuron SOP。
- 64-bit temporal-pair address trace，用于判断相对 baseline 是否真的减少事务。

### 6.3 P2：PPA 与 sign-off 所需

- 固定工艺、corner、V/F、SDC 和相同 SRAM macro 下的 C0/B1/A2 leaf + subsystem 综合。
- SAIF/VCD workload activity；memory macro read/write energy；clock tree/ICG 单列。
- 面积分解：dense core、sparse core、stratifier、FIFO/tag/ROB、state SRAM、score SRAM、Shiftmax、controller。
- 功耗分解：dynamic/leakage/clock/memory；报告空闲但未 power-gate 的第二 core 成本。
- worst-case latency、frame throughput、p99 queueing；同时报告全芯片 frontend/MLP/downsample/decoder 边界。
- H60、H67、H67+Delta C0、H67+Delta B1 同口径结果。Bishop 数字只能列 related work，不进入归一化分母。

## 7. 候选比较

| 维度 | C0 保守 | B1 平衡 | A2 激进 |
|---|---|---|---|
| 数值语义 | exact | exact | exact，调度更复杂 |
| 新 compute | 最少 | 1 dense + 1 narrow sparse | 1 dense + 多 sparse |
| 新 memory/control | 低 | 中 | 高 |
| 对 bimodality 依赖 | 无 | 高 | 很高 |
| SRAM 端口压力 | 低 | 中到高 | 高 |
| backend bottleneck 风险 | 可直接测 | 中 | 高 |
| RTL/验证风险 | 低 | 中高 | 高 |
| 当前证据充分度 | 足够开始基线实现/模型 | 不足，待 locality-v2 + crossover | 明显不足 |
| DATE 定位 | 稳健硬件基线 | 条件成立后的主候选 | 探索/附录，不进当前主线 |

## 8. 风险登记

评分为 Probability x Impact；分数不是 PPA。

| ID | 风险 | P | I | 分数 | 缓解/退出条件 | Owner |
|---|---|---:|---:|---:|---|---|
| R1 | 平均 toggle 很低但非零工作不呈双峰，第二 core 无利用率 | 4 | 5 | 20 | locality-v2 histogram；无清晰 crossover 则停在 C0 | Architecture/Profiling |
| R2 | state/FIFO/metadata SRAM traffic 抵消 compare savings | 4 | 5 | 20 | address trace + macro energy；memory-aware 净收益不过门则不升级 B1 | Architecture/Physical Design |
| R3 | dense/sparse 并发造成 bank conflict和 backend serialization | 4 | 4 | 16 | trace-driven bank/queue sweep；以 frame cycles 而非 core cycles 验收 | Microarchitecture |
| R4 | 路由或 completion 改变 row score 集合/顺序，破坏 bit exact | 3 | 5 | 15 | token tag、row bitmap、随机 backpressure golden；任何 mismatch 阻断 | RTL/Verification |
| R5 | 历史 TTB proxy 被误当 Delta bundle4/8 数据 | 4 | 4 | 16 | 图表分开命名；只引用 raw locality-v2 字段 | Profiling/Paper |
| R6 | H67 Motion 与 Delta 共享前端后重复计算或重复申报收益 | 3 | 4 | 12 | operation/PPA 表明确 shared/exclusive block | Architecture/Paper |
| R7 | sparse FIFO burst、dense fallback导致 p99 latency退化 | 3 | 4 | 12 | per-sequence p99 trace；设 bounded fallback 和 starvation counter | Microarchitecture/Verification |
| R8 | 多 core 拉长时序或增加 clock power，频率下降 | 3 | 5 | 15 | leaf + top timing、clock power分列；达不到 C0 频率则不晋级 | RTL/Physical Design |
| R9 | 近似 error-bounded bundling 混入 exact 主线 | 3 | 5 | 15 | 独立 B-class 配置和 valid825；当前三方案禁止近似丢弃 | Algorithm/Verification |
| R10 | 缺目标预算却提前宣称 sign-off/PPA 优势 | 4 | 5 | 20 | 在工艺、V/F、area/power budget、SRAM macro齐备前统一标“未签核” | Project Lead |

当前所有高风险都有退出条件，但 R1/R2/R3/R10 的证据尚未完成，因此不能签核 B1/A2。

## 9. 推荐与准入门槛

### 9.1 当前推荐

**冻结 C0 为实现和测量基线；把 B1 设为条件式目标；不实现 A2。**

理由：

- C0 利用已测的低 temporal toggle，且不要求密度双峰。
- C0 能建立 state SRAM、64-bit pair layout、mask generation、accumulator RMW 和真实 cycle/energy 的基准，这些恰好是当前缺失项。
- B1 的核心价值不是“有两个不同 core”，而是其净收益超过第二路径的 memory/control/clock 成本；该判断必须相对 C0，而不是相对理想 full compare。
- A2 在只有 `T=2`、单一共享 Shiftmax 后端且尚无 queue trace 时过早，会扩大验证和 PPA 不确定性。

### 9.2 B1 晋级必须全部满足

1. locality-v2 在统一 H60 TTX 与 H67 候选上完成至少 100-sample raw count，并补 valid825 或代表序列稳定性检查。
2. `update_count_hist` 显示 sparse 与 dense 路径都有非边缘工作量；阈值由综合后的 latency/energy crossover 得出。
3. trace-driven 模型显示 B1 相对 C0 的 frame cycles 改善，且 p99 不因 queue/bank/backend 冲突退化。
4. 同工艺、同 V/F、同 SRAM macro、同约束下，B1 的 total energy/frame 或目标 PPA 指标优于 C0；必须包含 idle core、clock、FIFO/tag 和 state memory。
5. randomized backpressure + golden 验证证明 H60/H67 score、center、gate、output 在冻结定点规则下 bit-exact 或满足事先定义的 LSB 容差。

若 2 或 4 不满足，正式结论应是：**本项目采用 Bishop 启发的 TTB work unit，但不采用其 dense/sparse heterogeneous cores。** 这是有效的负结果，不应为追求形式上的异构而增加硬件。

## 10. 交付判定

本轮仅完成 architecture exploration 和风险/准入定义，不构成 PPA 或 architecture sign-off。下一轮最小闭环应是：

```text
locality-v2 raw profile
  -> C0/B1 trace-driven cycle model
  -> leaf synthesis + SRAM macro cost
  -> exact golden/backpressure verification
  -> B1 go/no-go
```

在该闭环之前，DATE 主张应保持为“temporal-pair TTB + exact delta-aware scheduling 的候选微架构”，不能写成“已实现 Bishop 式异构核并获得某项 PPA 收益”。
