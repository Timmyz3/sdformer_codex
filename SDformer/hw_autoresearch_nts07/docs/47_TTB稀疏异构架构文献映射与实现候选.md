# TTB 稀疏异构架构文献映射与实现候选

**轮次**：DATE 硬件侧 Round4

**对象**：H60 TTX / H67 Motion-XOR TTX，`T=2`，`head_dim D=32`

**输入证据**：`docs/45_TTB异构双路径微架构评估.md`、`docs/46_TTB真实分布周期模型与综合协议.md`、`neuron_autoresearch/DATE_IDEA_PORTFOLIO_20260712.md`、`results/ttb_true_density_ttx_h67_h68_profile100.json`
**边界**：只讨论不改变冻结软件数值语义的硬件候选；不把 Bishop 或其他论文的 PPA、speedup、energy 数字当作 SDformer 结果；本文不提供未经本项目 RTL、同工艺综合和 SRAM 宏标定的 PPA。

## 0. 执行结论

1. **推荐主线仍是 exact C0 -> B1 的分级推进，而不是直接复制 Bishop。** 先实现 metadata-first temporal-pair、精确常量注入和 `u=0` 复用；只有 trace-driven replay 证明双路径相对 C0 的 frame cycle、p99 和 energy/frame 都有净收益时，才进入一密一疏 B1。
2. **true TTB profile100 已证明 bundle 级空组很多，但没有证明双核有净 PPA 收益。** H67 的 bundle1/2/4/8 empty ratio 分别为 `73.8973% / 67.3191% / 60.9633% / 55.2559%`；这是特定 profile100 上整组 payload/issue 可被审查的候选比例，不是周期、功耗或 SRAM 节省结果。
3. **文献中可迁移的是机制，不是结果。** Bishop 提供 density stratifier 与负载平衡问题；LoAS 提供时间维内层和一位 spike 连续访问；Prosperity 提供无损 exact/partial-match 复用；Sparseloop 提供 representation/gating/skipping 成本分类；Stellar、STONNE、AccelTran 和 Prosperity artifact 可用于生成叶子模块或建立模拟参考。它们都不能直接替代本项目的 bit-exact row replay 和同工艺综合。
4. **近似路线必须隔离。** Bishop BSA/ECP、DOTA weak-attention omission、Sava pruning/low-bit mask、SpARC row clustering、AccelTran DynaTran、SNN STELLAR 的 FS-neuron co-design 都改变训练、激活、attention mask 或数值结果，不能进入“软件语义不变”的 H60/H67 主线。
5. **Round4 推荐候选为 E0。** E0 只增加早到 metadata、常量注入、per-token exact gate 和可审计计数器，风险最低。E1 双路径是条件晋级项；E2 exact product-class reuse 先做 trace-only 价值筛选，未达到命中率/复用距离/净 SRAM energy 门槛前不做 TCAM。

### 0.1 与当前实现队列的对齐（2026-07-13）

本文提出的第一批 profile 缺口已经进入只读 collector 和串行 watcher：新增 Delta `u=0..32`
histogram、`theta=2/4/8/12/16` 的 conditional lane sum，以及 bundle1/2/4/8 的完整 `A_b`
histogram和`kappa=12`统计。`run_ttb_cycle_profile_v2_after_round3.py` 将在 H76-H78 结束后对
TTX/H67/H68 各重放100样本并写入独立 v2 artifact，不覆盖本文引用的旧 profile100。代码和47项
attention测试已通过，但在 artifact 实际生成前，G0仍只算“采集器就绪”，不算“trace证据完成”。

H69/H70 另有量化前 score-clipping profile20，用于检查固定/动态左移在 Q7 `[-2,2]` 下的裁剪率；
它只决定温度线位宽风险，不替代 E0/E1 的 cycle、SRAM 或 PPA 证据。

## 1. 本项目事实与语义合同

### 1.1 true TTB profile100 的已测分布

下表直接来自 `ttb_true_density_ttx_h67_h68_profile100.json`。`activity_density` 对同一模型的 bundle1/2/4/8 相同，因为它是原始 bit activity；bundle 变大只改变整组 empty、K-zero 和 motion-zero 的概率。

| 模型 | bit activity | bundle1 empty | bundle2 empty | bundle4 empty | bundle8 empty |
|---|---:|---:|---:|---:|---:|
| H60 TTX profile | 1.6915% | 72.5395% | 66.1272% | 60.0896% | 54.7048% |
| H67 ep19 | 1.5021% | 73.8973% | 67.3191% | 60.9633% | 55.2559% |
| H68 ep19，部署回 H60 | 1.5489% | 74.2013% | 67.7691% | 61.5175% | 55.8804% |

H67 的 bundle1/2/4/8 K-zero ratio 为 `83.1064% / 79.5449% / 75.6363% / 71.4166%`，motion-zero ratio 为 `83.1753% / 79.5842% / 75.6672% / 71.4434%`。这些值的合法解释如下：

- `empty_ratio(b)`：若 metadata 在 payload 之前到达，且该组每个被省略 token 的 frozen score 可由精确常量注入恢复，则是整组 Q/K payload fetch、局部 compare 和 issue 的**候选上限**。
- `kzero_ratio(b)`：是 `gate*K`、相应 value/projection 数据通路的**候选 gate 上限**；它不允许删除 score、center 或 Shiftmax 项。
- `motion_zero_ratio(b)`：只允许 gate H67 的 Motion-XOR 增量分支；不能 gate H60 TTX score 主路，也不能与 K-zero 重复申报同一开关活动。
- `active_1_x_ratio`：只是某密度区间的 bundle 数量，不是该区间的周期占比。服务时间、SRAM 事务、FIFO 等待和共享后端会改变最终贡献。

### 1.2 冻结的 exact 语义

对 spatial token/head 的两个时间片：

```text
U       = (Q0 XOR Q1) OR (K0 XOR K1)
u       = popcount(U), 0 <= u <= D=32
S_TX(t) = frozen bipolar/equality score in the current fixed-point domain
S_H67   = S_TX + M_K/4
M_K     = popcount(K0 XOR K1), shared observation only
output  = frozen center -> Shiftmax -> gate*K path with current widths and RNE
```

硬件可以改变数据布局、执行顺序和选择 dense/sparse 算法，但必须满足：

1. `t0` 建立完整、正确的 score state；`t1,u=0` 才可复用。
2. sparse delta 与 dense recompute 生成相同宽度、相同符号扩展、相同溢出处理的 `S64`。
3. 一个 row 的全部 score 项完成后，才按冻结 token 顺序进入 center/Shiftmax；不能改变归约结合顺序而引入 LSB 差异。
4. H67 motion 项在原始 score 域合并后统一舍入；不能先舍入两个分支再相加。
5. `empty=(OR(Q|K)==0)` 只有在硬件注入该模式的冻结 `S_silent` 时才可 skip；“输入为空”不等于“Shiftmax 中没有这个 token”。
6. FIFO fallback、work stealing 或重排只能改变位置和时间，不能丢 descriptor、合并不同 tag 或使用不完整 hash 命中。

## 2. 文献深读映射

### 2.1 检索和判定方法

检索范围为 2022-2026 的 ISCA、MICRO、HPCA、ASPLOS、DATE、DAC、CICC。优先阅读全文、artifact README、RTL/模拟器仓库和评估方法，不以摘要中的 speedup 作为可迁移证据。2026-07 截止时，未找到已在这些会议正式发表、且能改变本项目 exact 推荐的 2026 工作；未确认 venue 的 2026 预印本不纳入主证据。

### 2.2 机制、语义和可复用性

| 工作 | 正文机制与成本观察 | 对 H60/H67 的合法迁移 | 不可迁移/必须隔离 | 开源状态判定 |
|---|---|---|---|---|
| Bishop, ISCA 2025 | TTB density stratifier 将 workload 分给 dense/sparse core；正文 DSE 明确阈值偏高或偏低都会造成双核失衡，且数据移动主导 energy；sparse core 使用 STONNE 建模 | TTB descriptor、按工作量分层、每层阈值 sweep、双 FIFO 失衡审计 | BSA 改训练分布，ECP 删除 Q/K/S/V/Y 活动，是 approximate；论文阈值、核规模、PPA 不能外推 | 未发现公开 Bishop RTL；STONNE 是通用 cycle-level 模拟器，不是 Bishop RTL |
| Sparseloop, MICRO 2022 | 将稀疏优化分成 representation、gating、skipping；gating 保留 cycle，只抑制无效切换，skipping 还需定位下一个有效元素并减少访问/周期 | 建立 metadata、IneffOps、payload、index、gated cycle、skipped cycle 的分账模型 | 统计密度模型不能替代 stage burst、FIFO、bank conflict 的离散事件 replay | 公开 Timeloop/Accelergy artifact；适合早期能耗账本，不是本项目 cycle-accurate RTL |
| LoAS, MICRO 2024 | fully temporal-parallel 把 time 放在内层，减少跨 timestep refetch；一位 spike 的 bitmask/坐标压缩与 inner join 仍有 metadata 和 join 开销 | `T=2` temporal-pair 连续布局、固定 bitmap、metadata-first fetch；用于比较 bitmap 与 index 交通 | LoAS 面向 activation/weight 双稀疏 spMspM；本项目当前权重不是该 workload，不能继承其 speedup | MIT artifact 可运行；未把它当作可直接综合的 H67 RTL |
| Stellar design framework, MICRO 2024 | DSL 将功能、dataflow、格式、load balance 和 memory 分离，可生成 spatial array/buffer 的 synthesizable Verilog；细粒度 work shifting 会增加连线、regfile 和拥塞风险 | 生成 dense/sparse lane、bitmap loader、buffer 的叶子 microbenchmark；比较固定与可配网络 | 不作为 H67 top-level；生成 RTL 仍需 lint、时序、等价和物理实现 | 公开 Chisel/Scala 仓库，可独立生成 array/buffer RTL，也可接 Chipyard |
| ASADI, HPCA 2024 | 以 DIA 类格式利用 sparse attention 的对角局部性；bubble/row index 和恢复位置都有 metadata 成本 | “先量 locality 再选格式”的方法；给 dilated/Match-Code 候选设计 index/decompress 计账 | H60 是每 token 的固定 TTX score，不存在已测 `N x N` 对角 attention 稀疏性；不能直接采用 DIA/ReRAM 数据流 | 未发现可直接复用的开源 RTL |
| SNN STELLAR, HPCA 2024 | 通过 FS neuron 与 spatiotemporal computation 获得结构化稀疏；LoAS 正文指出 FS neuron 解耦 accumulation/firing，时间并行条件不同于 LIF | 只借鉴 spatiotemporal 组织和“神经元语义决定可并行性”的审查方法 | FS neuron 是算法/神经元语义变化，不能替换当前 H60/H67 | 未找到可用于本项目的公开 RTL；不能与 MICRO 2024 的 Stellar generator 混淆 |
| Prosperity, HPCA 2025 | Product Sparsity 对 binary rows 做 exact match/partial subset，复用 prefix 内积；TCAM detector、popcount、排序、dispatcher 和结果表均有显著 control/area/power 成本 | 先 profile 相同/子集 Q/K mask 与复用距离；只允许 full-tag exact hit，候选 E2 使用 residual mask 做精确增量 | 不在命中率未知时实例化 TCAM；spiking GeMM 的收益和 tile 参数不能外推到 TTX score | 公开 cycle-accurate simulator、CUDA kernel、CACTI 流；论文实现过 SV，但公开仓库未见 RTL，不能称为可复用 RTL |
| DOTA, ASPLOS 2022 | learned detector 先预测 weak attention，再省略 attention；token parallelism 增加 reuse，但 scheduler/buffer 随并行度增长 | scheduler/buffer 参数 sweep、检测开销单列、token 并行度与 memory traffic 的权衡 | weak-attention omission 需要训练且改变 attention 结果；H60 也不是标准 `N x N` softmax attention | 未作为本项目 RTL 来源 |
| Flexagon, ASPLOS 2023 | 根据矩阵尺寸/模式选择 IP/OP/Gustavson；MRN 同时支持 merge/reduce；不同 dataflow 需要 FIFO/cache/psum SRAM，mapper 在论文中仍是 future work | 对固定 bitmap 与 index 路径做 memory-aware 而非 compute-only 比较；可借鉴 merge/reduce 网络的 leaf DSE | 本项目 `D=32,T=2` 规模很小，不支持直接引入三层通用 SpMSpM 网络 | 有 cycle-level/相关代码入口，正文有主要 block RTL 比较；不是 H67 可直接集成 RTL |
| FEASTA, ASPLOS 2024 | SFT 统一 dense/compressed sparse 表示，loader/intersector/dispatcher/compute 分层，通用性来自 ISA 和多模式控制 | loader/intersector 的接口拆分、格式 metadata 成本和多模式控制计账 | 对固定小维度 TTX 使用完整 SpTA ISA 很可能控制过重；不在 trace 前引入 | 正文提供架构和综合方法；未确认有适合直接集成的公开 RTL |
| Sava, DATE 2024 | 对相似 sparsity 的 rows 重排以改善 unstructured sparse 负载；正文也使用 pruning 和低比特 mask | 仅迁移“按 measured work 分桶”和重排索引/写回成本；E1 可做 stage-static ordering | pruning、mixed precision mask 有已报告精度损失，属于 approximate | 未发现可直接复用 RTL；论文 RTL/PPA 只属于 Sava |
| SALO, DAC 2022 | scheduler 对 sliding/dilated window 做 splitting/reordering；相邻 query 复用 K/V，分割后通过严格重归一化恢复结果 | 固定 descriptor、重排后 tag 恢复、共享 K/V 的审计方法 | workload 是长序列结构化窗口 attention，不是 H60 TTX；其专用对角连接不可直接复用 | 论文描述 Chisel/ASIC 评估，未找到公开可集成 RTL |
| SWAT, DAC 2024 | 固定长度 FIFO 保存滑窗 K/V、row-major kernel fusion 降低中间 score 存储 | FIFO 指针、输入驻留和“每项加载一次”的 address-trace 验证方法 | 512-window 的 K/V reuse 不存在于当前 `T=2,D=32` TTX，不能继承 traffic 结论 | FPGA 架构论文；未作为本项目 RTL 来源 |
| SpARC, DAC 2024 | 对相似 attention rows 聚类，每簇只算一次 score，是明确的 approximate attention | 只借鉴 cluster tag、写回索引和流水调度成本项 | row clustering 改变结果，不能进入 exact 主线 | 未发现可直接复用 RTL |
| SparseTrim, CICC 2025 | 三页 silicon paper 强调 fine-grained sparse model 的片上解压；可公开获得的材料不足以重建其完整控制和 SRAM 协议 | 只作为“芯片级结果必须连同压缩/解压、memory 和 system energy 报告”的评估纪律 | 标题中的 TOPS/W、工艺和解压收益不能作为本项目估计或目标 | 未发现公开 RTL/模拟器；不依据会议简介推断内部实现 |

正文与 artifact 入口见文末参考资料。特别注意两个同名工作：HPCA 2024 SNN STELLAR 是 algorithm-hardware co-design；MICRO 2024 Stellar 是 dense/sparse accelerator RTL 生成框架。

### 2.3 文献到本项目的证据等级

| 等级 | 定义 | 本轮条目 |
|---|---|---|
| L0，可直接复用思想 | 不改变算术语义，且与 `T=2,D=32` 接口可对齐 | temporal-pair、bitmap、metadata-first、exact delta、bounded FIFO fallback |
| L1，可复用工具/叶子模块 | 需重写 wrapper、位宽和验证，不能直接作为 top | Stellar generator、AccelTran SV leaf、STONNE、Sparseloop、Prosperity/LoAS artifact |
| L2，仅作 DSE 方法 | workload/格式不同，只能迁移计账或调度原则 | Flexagon、FEASTA、SALO、SWAT、ASADI、Sava row grouping |
| L3，approximate 隔离 | 改训练、神经元、mask、score 或输出 | Bishop BSA/ECP、DOTA、DynaTran、SNN STELLAR FS、Sava pruning、SpARC |

## 3. 共同符号和可审计模型

### 3.1 工作量符号

对 stage `s`、row `r`、token/head temporal pair `i`：

```text
D            = 32
b            = token bundle volume in {1,2,4,8}
N_b          = number of bundle descriptors
E_b          = measured empty bundle count
n_s(u)       = number of t1 token/head items with update count u
P_D, P_S     = effective dense and sparse lane width
theta_s      = exact routing crossover for stage s
Wscore       = frozen score storage width
Wtag         = full row/token/head/time tag width
```

`n_s(u)` 不在本文引用的 legacy true TTB JSON 中；cycle-v2 collector 已实现该字段，但 artifact
仍在软件全量队列后等待生成。因此当前任何使用 `theta` 的定量结果仍必须标为待 trace 填充。
平均 bit density 不能生成 `n_s(u)`，bundle empty ratio 也不能替代 update histogram。

### 3.2 叶子服务时间

以下都是待 RTL 标定的符号，不是实测周期：

```text
c_D       = c_D_fetch + ceil(D/P_D)*c_D_lane + c_D_reduce + c_D_write
c_S(u)    = c_idx(u) + c_S_fetch(u) + ceil(u/P_S)*c_S_lane
            + c_delta_reduce + c_state_RMW
c_R       = c_reuse_check + c_score_state_read
c_E       = c_meta_check + c_constant_inject
c_BE(r)   = c_center(r) + c_shiftmax(r) + c_gateK(r)
```

每个 `c_*` 必须由相同频率目标下的 RTL latency/II 和 SRAM 宏端口时序填充。若 pipeline 可重叠，cycle replay 使用 initiation interval；若不可重叠，使用完整 latency。不能把 `ceil(u/P_S)` 单独当作 token latency。

### 3.3 队列、冲突和控制方程

对路径 `x in {D,S,BE}` 的离散周期：

```text
Q_x[k+1] = min(F_x, max(0, Q_x[k] + A_x[k] - G_x[k]))
overflow_x[k] = max(0, Q_x[k] + A_x[k] - G_x[k] - F_x)
G_x[k] <= service_capacity_x[k]
```

其中 `G_x[k]` 还受 operand SRAM bank、score-state RMW、tag completion 和 downstream credit 限制。frame makespan 必须从 trace replay 的最后一个 committed row 得到：

```text
C_frame = max_r(commit_cycle[r]) - min_r(arrival_cycle[r]) + 1
L_row   = commit_cycle - arrival_cycle + 1
I_DS    = abs(W_D - W_S) / max(1, W_D + W_S)
```

`I_DS` 只是工作量失衡指标，不能替代 p95/p99 queue wait。Bishop 的阈值 DSE 和 Sava 的 row grouping 都说明平均分配比例不足以预测尾部延迟。

## 4. Exact 候选微架构

### 4.1 E0 保守：metadata-first temporal-pair C0

```text
pair SRAM:  [Q0,Q1,K0,K1] + early metadata
                         |
metadata -> exact gate -> fixed 32-lane score/delta engine
                         |
             tagged S64 / completion
                         |
       frozen center -> Shiftmax -> gate*K
```

**数据布局**：每个 token/head 的 Q/K temporal pair 为 `4D=128 bit` payload；bundle4/8 只合并地址和 descriptor，不使用可变长 RLE。早到 metadata 至少包含 `bundle_empty`、per-token empty bitmap、K-zero bitmap、motion-zero bitmap、`u` 或 update mask 的位置。

**bit-exact route**：

```text
if bundle_empty && all silent constants are derivable:
    inject S_silent for every covered token
elif token_empty:
    inject S_silent for that token; other tokens continue
elif t == 0:
    full exact compute
elif u == 0 && score_state_valid:
    exact state reuse
else:
    exact delta on the single fixed engine
```

若 early metadata 不是由同一 payload 在写入端生成并受 parity/ECC/tag 保护，则不能据此省略 payload read。

**cycle equation**：

```text
C_E0 = C_meta_read
     + N_empty_token*c_E
     + N_t0*c_D
     + n(0)*c_R
     + sum(u=1..D) n(u)*c_S0(u)
     + C_bank_stall + C_state_stall + C_BE + C_drain
```

这里 `c_S0(u)` 是单路径引擎的 exact delta 服务时间，不假设存在第二 sparse core。

**SRAM bits**：

```text
S_pair_payload = N_resident_pair * 4D
S_group_meta   = N_resident_bundle * (1 + b + b + b)
                 + N_resident_pair * (D_update_mask_if_stored)
S_score_state  = N_live_pair * (Wscore + 1 valid)
S_completion   = N_live_row * (Ntoken bits + Wtag)
```

metadata 的三个 `b` 分别代表 token-empty、K-zero、motion-zero bitmap；若字段可由 update mask 无组合关键路径地导出，可删除重复存储，但必须比较增加的 read/logic energy。

**合法 gate**：empty payload/score leaf、`u=0` delta leaf、K-zero 后端、motion-zero H67 分支、inactive lane/register clock。**不能 gate**：缺少常量注入时的 score 项、整行 center/Shiftmax、未完成 row 的 score state、仅凭平均 density 判定的 mixed bundle。

**定位**：Round4 推荐。它直接消费现有 profile 的 empty/Kzero/motionzero 信息，同时避免第二端口、双 FIFO 和跨核 completion 的主要风险。

### 4.2 E1 平衡：exact Delta B1 一密一疏

```text
metadata/update mask -> bit-exact stratifier
        |                     |
   dense FIFO             sparse FIFO
        |                     |
  32-lane recompute      P_S-lane delta
        |                     |
        +---- tagged score-state/join ----+
                                          |
                               shared frozen backend
```

**bit-exact route**：

```text
t0                                  -> DENSE_FULL
t1 && u==0 && state_valid           -> REUSE
t1 && 1<=u<=theta_s && S_FIFO_ready -> SPARSE_DELTA
t1 && u>theta_s                     -> DENSE_FULL
t1 && sparse_high_water             -> DENSE_FULL fallback
otherwise                           -> stall, never drop
```

`theta_s in {2,4,8,12,16}` 是初始 sweep，不是预选答案。只有 `c_S(u)+queue/memory/control < c_D` 的交叉点才可使用。fallback 仍做 full exact recompute，因此只影响效率，不影响结果。

**cycle lower bound 与真实模型**：

```text
W_D(theta) = N_t0*c_D
           + sum(u=theta+1..D) n(u)*c_D
           + N_fallback*c_D

W_S(theta) = sum(u=1..theta) n(u)*c_S(u)

C_front_LB = max(W_D, W_S)
C_E1       = trace_replay(route, FIFO, bank, RMW, join, BE credits)
```

`C_front_LB` 明确只是无冲突、无限 FIFO、完美并行下界。最终结果必须用 `C_E1`；共享 backend 可与前端流水重叠的部分由真实 credit trace 决定，不能简单相加或简单取 max。

**SRAM/control 增量**：

```text
S_FIFO = F_D*(Wtag + D + Wroute_D)
       + F_S*(Wtag + D + Wroute_S)
S_join = N_live_row*(completion_bits + destination + valid)
S_stage_cfg = Nstage*(ceil(log2(D+1)) + watermarks + arbitration bits)
```

`D` 项是 update bitmap；若 FIFO 只存 pointer，则必须另计 descriptor SRAM 的读端口、bank conflict 和 pointer lifetime。E1 的关键成本不是 sparse ALU 本身，而是第二供数路径、score-state 原子 RMW、FIFO/tag/completion 和 idle/clock power。

**晋级条件**：见第 8 节。只要 E1 在包含 SRAM 与 idle core 后不能优于 E0，就保留 TTB work unit 而不采用异构双核。

### 4.3 E2 激进研究支线：full-tag exact product-class reuse

该候选受 Prosperity 启发，但不复制其 TCAM 规模。先在 trace 中查找同一 stage/head 内 temporal-pair mask 的 exact match 和 proper-subset；只有重复存在且 reuse distance 可被小表覆盖，才设计硬件。

```text
class detector -> prefix/result table -> residual exact delta
       miss ---------------------------> E1 route
```

对当前 mask `X` 和候选 prefix `P`：

```text
exact_match  = (X == P) && full_context_tag_match
partial_match = ((P & ~X) == 0) && full_context_tag_match
R            = X & ~P
u_R          = popcount(R)
```

只有 prefix 与当前项使用完全相同的 frozen weight/score context，且表项保存完整 tag 和精确 `S64` 时可命中。hash 只能选 set，不能作为命中依据；任何 alias 都必须 full compare 后转为 miss。

**cycle equation**：

```text
C_E2 = N_lookup*c_lookup
     + N_EM*c_table_read
     + sum_PM c_S(popcount(R))
     + N_miss*c_E1_service
     + C_dependency_wait + C_table_conflict + C_E1_common
```

**SRAM/TCAM equation**：

```text
S_class = Nentry*(Wfull_tag + D_mask + Wscore + valid + replacement_state)
S_dep   = Nlive*(prefix_id + ready + consumer_count)
```

若使用 TCAM，必须单列每次 lookup 激活的 cell 数、lookup energy、时序和面积。Prosperity 正文指出 detector TCAM 是主要片上功耗项之一，因此本项目默认采用小 set-associative full-compare 表做价值验证，而不是一开始上全并行 TCAM。

**定位**：非主线，先 trace-only。若 exact-match/partial-match 命中率、reuse distance、residual `u_R` 和净 SRAM access 没有同时达标，E2 结论应为负。

### 4.4 E3 不推荐的通用多模式核

把 Stellar/FEASTA/Flexagon 的可配格式、multi-dataflow、work shifting 全部放入 H67 top，理论上可覆盖 dense、bitmap、index、RLE 和 product reuse，但对 `D=32,T=2` 会增加 crossbar、ISA/descriptor、intersector、psum storage 和验证状态空间。除非后续 DATE 主线出现多个尺寸和多个 sparse operator 共享该核，否则不进入 RTL 候选；这些框架只用于 leaf DSE。

## 5. Memory、compute、control 统一成本账本

### 5.1 payload 和 metadata traffic

对 bundle `b`，Q/K 的未压缩 temporal-pair payload 为：

```text
B_payload_raw(b) = N_b * 4*b*D bits
B_meta(b)        = N_b * Wbundle_meta + Ntoken * Wtoken_meta
```

若仅整 bundle empty 才跳 payload，则：

```text
B_payload_saved_ceiling(b) = E_b * 4*b*D bits
B_payload_fetch_floor(b)   = (N_b-E_b) * 4*b*D bits
```

这两个式子都是**仅 payload、完美 early metadata、无 bank granularity 浪费的上下界**。实际 SRAM 以 macro read width 计费：一个非空 bit 导致整行 read 时，saved bytes 可能为零；metadata 额外 read、ECC、address 和 arbitration 也必须加入。

若按 per-token bitmap 允许 mixed bundle 局部 fetch：

```text
B_payload_actual = sum(read_transactions) macro_read_width
```

必须由 address trace 计算，不能由 empty ratio 乘逻辑位数得到。

### 5.2 compute 与 clock gate

```text
N_dense_lane_ops = N_dense_items * D
N_sparse_lane_ops = sum_sparse u
N_index_ops = sum_sparse (index_decode + mask_scan)
N_state_ops = score_state_reads + score_state_writes
```

以下比例只允许作为 operation/gating ceiling：

- `n(0)/N_t1`：t1 compare 的理想 skip ceiling，不是全 attention cycle reduction。
- `1-activity_density`：bit 为零的比例，不等于 lane、word 或 SRAM access 可跳比例。
- `empty_ratio(b)`：整 descriptor 的 issue/fetch ceiling，受常量注入和 SRAM 粒度约束。
- `kzero_ratio(b)`、`motion_zero_ratio(b)`：各自局部分支的 gate ceiling，不能相加。

真正合法的 clock gate enable 必须来自 registered、bit-exact predicate，并记录 `eligible_cycles`、`gated_cycles`、`wake_cycles`。组合数据门控不能被当作 ICG 节省；插入 ICG 后要检查 enable timing、test enable 和 scan。

### 5.3 control 状态

| 状态 | E0 | E1 增量 | E2 增量 |
|---|---|---|---|
| descriptor | fixed bundle + token bitmap | route、FIFO pointer、fallback reason | class/set、prefix id、residual mask |
| queues | 单 issue queue | dense/sparse FIFO、credit、高低水位 | lookup/miss/dependency queue |
| completion | row bitmap | 跨核 tagged join、原子 RMW | prefix ready/consumer tracking |
| errors | metadata/payload tag mismatch | FIFO overflow、starvation、duplicate completion | alias、stale prefix、dependency cycle |
| performance counters | gate/fetch/service/backend | occupancy/stall/fallback/imbalance | hit type/reuse distance/residual work |

## 6. 必须新增的 trace

### 6.1 P0：决定 E0/E1/E2 工作量

每个 record 必须带 `sample_id, stage, layer, head, row, token, time, sequence_len`，并新增：

| 字段 | 目的 |
|---|---|
| `q0_bits, q1_bits, k0_bits, k1_bits` 或可复算 hash+raw side file | golden 重放和 predicate 审计 |
| `empty_b1/b2/b4/b8`、per-token empty bitmap | 验证 JSON 聚合和 mixed bundle |
| `q_toggle_mask, k_toggle_mask, update_mask, u` | `n_s(u)` 与 exact route |
| `kzero_mask, motionzero_mask` | 分离合法 gate，避免重复计数 |
| `score_t0_golden, score_t1_golden, motion_golden` | dense/delta/motion bit-exact 比较 |
| `silent_score_constant_id/value` | 证明 empty skip 后仍保留 Shiftmax 输入 |
| `exact_class_id, subset_prefix_id, residual_u, reuse_distance` | E2 价值筛选；没有候选时填 invalid |
| `bundle_address, bank, macro_word` | 从逻辑 empty 转换为真实 SRAM transaction |

聚合至少输出 per-stage `u_hist[0..32]`、run-length、相邻 descriptor 相关性、bundle burst length、exact/subset hit rate 与 reuse-distance CDF。profile100 之外还需要 valid825 或代表性长序列稳定性检查。

### 6.2 P1：cycle-accurate replay

每个 descriptor/row 记录：

```text
arrival_cycle, metadata_ready_cycle, payload_ready_cycle
route_requested, route_committed, fallback_reason
fifo_enter/leave, fifo_occupancy_before/after
service_start/end, core_id, effective_u
operand_bank_conflict_cycles, state_RMW_stall_cycles
join_wait_cycles, backend_credit_wait_cycles
backend_start/end, commit_cycle
```

全局记录每周期 dense/sparse/backend valid-ready、SRAM port grant、ICG enable、idle/wake 和 reset/drain。必须可重建 `Q_x[k]`，否则不能审计 FIFO 深度和 p99。

### 6.3 P2：PPA activity

- block 级 SAIF/VCD：metadata front end、XOR/popcount、dense lanes、sparse lanes、state SRAM wrapper、FIFO/tag/join、center/Shiftmax、gate*K、clock tree。
- SRAM macro counters：每 bank read/write/idle/retention、bit width、depth、port、byte/write mask 使用。
- control counters：route compare、bitmap scan、index decode、full-tag compare、fallback、flush、replay、ECC/parity error。
- workload 切片：平均、最密 stage、最稀 stage、最高 FIFO burst、最长 backend wait、p99 sequence；不能只用全局平均 density。

## 7. 可复用模块与使用边界

| 来源 | 可复用资产 | 建议用途 | 使用前门槛 |
|---|---|---|---|
| 本项目 H60/H67 | frozen fixed-point score、Motion-XOR、Shiftmax/gate backend、golden profile | 所有候选的唯一语义 oracle | 先冻结宽度、RNE、溢出、token 顺序 |
| Stellar MICRO 2024 repo | Chisel DSL、spatial array、buffer、sparse-dense 示例、Verilog 生成 | 生成 `P_S={2,4,8}` lane/intersector 叶子，比较 wiring/control | license 审查；生成 RTL lint、formal、同 SDC 综合；不直接接 top |
| AccelTran repo | BSD-3 SV 的 MAC、softmax、DMA 等 leaf 和 cycle simulator | 参考模块边界、综合脚本和 cycle-state 组织 | 算术/softmax 与 Shiftmax 不同，禁止直接替换 golden backend |
| STONNE | C++ cycle-level sparse accelerator simulator、SRAM/FIFO/access counters | 快速搭建稀疏 lane/network sensitivity；校验解析模型趋势 | 自定义 TTX op、真实 trace 和本工艺表；默认表不能当本项目 PPA |
| Sparseloop artifact | representation/gating/skipping analytical model | 早期 traffic/energy 分账和格式 sweep | 用实测 density/hist；尾部 queue 用独立 replay |
| LoAS artifact | temporal-parallel/dual-sparse evaluation code | 验证 time-inner layout、bitmask/index overhead 的方向 | workload 改成 `T=2,D=32`；不使用双稀疏权重结论 |
| Prosperity repo | cycle-accurate simulator、CACTI integration、reference data | E2 detector/table 的模拟骨架和成本项清单 | 公开仓库无可直接复用 RTL；TCAM/FP add 参数必须重新综合 |
| Flexagon/FEASTA | merge/intersect/format 设计 | 只做 leaf 结构备选 | 固定小维 workload 下先证明通用控制不是负收益 |

开源代码的“可运行”不等于“可综合”，论文的“实现过 RTL”也不等于公开仓库含 RTL。每个外部模块都需要 license、版本 commit、test、位宽和工艺依赖清单。

## 8. PPA 与验证准入门槛

### 8.1 G0 语义门

- 对 H60、H67 的所有 trace，E0/E1/E2 的 score、motion、center、Shiftmax、gate 和最终输出逐 bit 等于冻结 golden；容差若非 0，候选自动转入 approximate 分支。
- randomized valid/ready、FIFO full/empty、bank conflict、reset/flush 下无丢失、重复、stale state、tag alias。
- formal/assertion 覆盖：`u=0 reuse`、sparse/dense equivalence、fallback equivalence、row complete before backend、full-tag-before-hit。

### 8.2 G1 cycle 门

- 使用同一输入 trace、同一 SRAM latency/port、同一 backend，报告 frame cycles、rows/s、平均/p95/p99 latency、FIFO occupancy、fallback 和 bank/backend stall。
- E1 必须相对 E0 而不是理想 dense baseline 改善目标 workload 的 frame cycles，且 p99 不退化；否则不晋级。
- E2 必须相对 E1 在计入 lookup/dependency/table conflict 后仍改善；只报告 reduced adds 或 hit rate 不过门。

### 8.3 G2 同工艺综合门

所有候选必须使用：

```text
same technology/library/VT set
same PVT, voltage, frequency target, uncertainty and IO constraints
same arithmetic widths and pipeline contract
same SRAM compiler/macro or same characterized wrapper
same synthesis/P&R effort, utilization and clock-gating policy
same trace-derived SAIF window
```

报告 logic area、macro area、total area、WNS/TNS、achieved Fmax、dynamic/leakage/clock/memory power、energy/frame。没有目标工艺、SRAM 宏和 post-synthesis/netlist activity 时统一标记 `TBD`，不得用 Bishop、Sava、Prosperity 或 AccelTran 的工艺数字补空。

### 8.4 G3 memory/control 净收益门

- E0：`payload+metadata+state+backend` 的 energy/frame 优于无 metadata baseline；若只减少 compare、总 energy 不降，保留 gate 但不声称能效收益。
- E1：包含 idle sparse core、第二供数路径、FIFO/tag/join、state RMW、clock tree 后，总 energy/frame 或项目冻结的 PPA objective 优于 E0。
- E2：包含 full-tag table/TCAM、lookup、dependency 和额外 SRAM 后优于 E1；TCAM 若成为关键功耗或时序路径，回退 E1。
- 任何候选都必须给出 SRAM port 峰值需求不超过 macro ceiling 的证明；靠无限端口得到的 cycle 结果无效。

### 8.5 同工艺比较表模板

| Metric | H67 reference | E0 metadata C0 | E1 exact B1 | E2 exact reuse | 证据 |
|---|---:|---:|---:|---:|---|
| bit-exact mismatches | TBD | 0 required | 0 required | 0 required | golden/formal |
| frame cycles avg/p99 | TBD | TBD | TBD | TBD | trace replay |
| dense/sparse/backend util | N/A | TBD | TBD/TBD/TBD | TBD | cycle trace |
| FIFO p99/max/fallback | N/A | TBD | TBD | TBD | cycle trace |
| SRAM reads/writes/bank stalls | TBD | TBD | TBD | TBD | address trace |
| logic area | TBD | TBD | TBD | TBD | same-tech synth/P&R |
| SRAM macro area | TBD | TBD | TBD | TBD | compiler report |
| WNS/TNS/Fmax | TBD | TBD | TBD | TBD | STA |
| dynamic/leakage/clock power | TBD | TBD | TBD | TBD | SAIF + signoff tool |
| total energy/frame | TBD | TBD | TBD | TBD | cycles + power |
| objective improvement vs prior | baseline | TBD | vs E0 | vs E1 | frozen objective |

## 9. Approximate 隔离表

| 方法 | 改变内容 | 进入 exact 主线？ | 若未来单独研究所需证据 |
|---|---|---|---|
| Bishop BSA/ECP | 改训练分布并按误差界删除 Q/K/S/V/Y 活动 | 否 | 独立训练、valid825 AEE/AAE、阈值-误差曲线、approx RTL |
| DOTA | learned detector 省略 weak attention | 否 | H60 专用 detector、任务精度与最坏误差，不可复用 DOTA 结果 |
| AccelTran DynaTran | runtime activation pruning | 否 | pruning threshold、再训练/精度和硬件开销 |
| SNN STELLAR | FS neuron 和模型 co-design | 否 | 重新定义软件模型，不属于本轮“语义不变” |
| Sava | spatial/value pruning、低比特 mask | 否 | 模型级精度和 mask sensitivity |
| SpARC | attention row clustering 共用近似 score | 否 | cluster error、任务精度、最坏序列 |
| Error-Bounded Gate Bundling H8 | 项目内误差有界 gate | 否，单独 H8 | valid825 + hardware error monitor + exact fallback |

“论文报告 negligible/no accuracy loss”不等于 bit-exact，也不等于本项目合法。只要 score 或输出有一 bit 差异，就必须从 E0/E1/E2 表移到 approximate 表。

## 10. 风险登记与缓解

评分为 Probability x Impact，不是 PPA。

| ID | 风险 | P | I | 分数 | 缓解/退出条件 |
|---|---|---:|---:|---:|---|
| R47-1 | empty bundle 被错误当成可删除 Shiftmax token | 4 | 5 | 20 | silent constant trace + bit-exact assertion；缺证据则只 gate payload 后的局部逻辑 |
| R47-2 | bundle empty 上限被写成 cycle/energy 节省 | 4 | 5 | 20 | 报表固定标 ceiling；最终只引用 address/cycle/SAIF |
| R47-3 | E1 双路径受 SRAM bank/backend 串行化 | 4 | 4 | 16 | 有限端口 replay；相对 E0 不改善即退出 |
| R47-4 | sparse burst 导致 FIFO overflow/p99 退化 | 4 | 4 | 16 | stage trace、watermark、exact dense fallback、starvation bound |
| R47-5 | H67 Motion 与 Delta 共享 XOR 后重复申报收益 | 3 | 4 | 12 | block counter 分 shared/exclusive；同一 toggle 只计一次 |
| R47-6 | variable-length index/RLE 控制超过 lane 节省 | 4 | 3 | 12 | 默认固定 bitmap；run/traffic 证据不过门不启用 RLE |
| R47-7 | E2 full/partial match 表造成 TCAM 功耗热点 | 4 | 4 | 16 | trace-only -> 小表 -> TCAM 分级；净能耗不过门退出 |
| R47-8 | 重排破坏 row completion 或 RNE 顺序 | 3 | 5 | 15 | tag scoreboard、canonical commit、formal equivalence |
| R47-9 | 外部 simulator 默认能耗表被当作本工艺 PPA | 4 | 5 | 20 | 只导入本项目综合/宏表；外部数字仅 related work |
| R47-10 | 通用 sparse framework 对固定小维过度设计 | 4 | 4 | 16 | 只生成 leaf；top-level E3 默认不立项 |
| R47-11 | profile100 对 valid825/长序列不稳定 | 3 | 4 | 12 | 分 stage/sequence bootstrap、valid825 或代表集复测 |
| R47-12 | 目标工艺/面积/功耗预算未冻结却宣称 sign-off | 4 | 5 | 20 | 全部 PPA 留 TBD；预算和同工艺 G2 前不签核 |

## 11. 推荐执行序列

1. **冻结 exact contract 与 trace schema。** 先补 `u_hist[0..32]`、per-token mixed bundle、silent constant、address/bank 和 row golden。
2. **实现 E0 的 cycle model/leaf RTL，不改训练代码。** 只做 metadata、constant inject、fixed bitmap、exact gate 和计数器；与 reference 做 bit-exact 和同工艺比较。
3. **离线 sweep E1。** 对 `theta_s in {2,4,8,12,16}`、`P_S in {2,4,8}`、FIFO depth、bank 数和 backend credit 做有限资源 replay，先决定有没有 B1，再写 top RTL。
4. **并行做 E2 trace-only。** 输出 EM/PM 命中率、reuse distance、residual `u` 和表容量曲线；没有明显净工作量下降就关闭该支线。
5. **PPA 只按 G0-G3 晋级。** 论文数字不进入归一化分母；最终 DATE 表至少包含 H67 reference、E0、E1，E2 只有过门才出现。

当前推荐可以写成：**“SDformer 采用 Bishop 启发的 TTB descriptor 与 density-aware exact routing，首先落地 metadata-first C0；异构 B1 和 product-class E2 由真实 trace 与同工艺 PPA 条件晋级。”** 不能写成“已获得 Bishop 式 speedup/energy saving”。

## 13. H79/H80 软件候选对硬件线的接口边界（2026-07-13）

H79 CF10与H80 DN9已进入软件full30队列，但不得直接套用H60/H67的`gate*K`后端PPA。两项都采用
Omega9局部二值匹配加静态codebook，因此若胜出，硬件接口保持TTB metadata、Q/K bitplane与
Shiftmax流水不变，替换的是attention backend：H79增加top2、query-popcount、两个dyadic beta
乘法和fixed-zero null；H80增加destination incoming edge重排、第二套Shiftmax9与Q1.7 gate
product。最终operation audit已分别计数，不能把H79的null第10行当作存储或投影MAC，也不能把
H80第二次归一化视作免费。

TTB E0/E1 exact线仍以H67冻结语义为主，不因H79/H80排队而修改。只有候选valid825同时超过
H67精度与spikes门槛后，才为其建立独立golden trace、SRAM address replay和同工艺PPA；在此之前
H79/H80属于软件算法候选，不进入H67 exact RTL的面积/能耗分母。

## 12. 主要参考资料与 artifact

- Bishop, ISCA 2025，全文与架构/DSE：<https://arxiv.org/html/2505.12281>
- Sparseloop, MICRO 2022，全文：<https://sparseloop.mit.edu/documents/2022-micro-sparseloop.pdf>；artifact：<https://github.com/Accelergy-Project/micro22-sparseloop-artifact>
- LoAS, MICRO 2024，全文：<https://ruokaiyin.github.io/papers/loas.pdf>；artifact：<https://figshare.com/articles/software/LoAS_Fully_Temporal-Parallel_Dataflow_for_Dual-Sparse_Spiking_Neural_Networks/27012058>
- Stellar automated design framework, MICRO 2024，全文：<https://people.eecs.berkeley.edu/~ysshao/assets/papers/stellar-micro2024.pdf>；RTL generator：<https://github.com/hngenc/stellar>
- ASADI, HPCA 2024，全文：<https://www.comp.nus.edu.sg/~tulika/HPCA24.pdf>
- SNN STELLAR, HPCA 2024，DOI：<https://doi.org/10.1109/HPCA57654.2024.00023>
- Prosperity, HPCA 2025，全文：<https://arxiv.org/html/2503.03379>；artifact：<https://github.com/dubcyfor3/Prosperity>
- DOTA, ASPLOS 2022，全文：<https://par.nsf.gov/servlets/purl/10357543>；DOI：<https://doi.org/10.1145/3503222.3507738>
- Flexagon, ASPLOS 2023，全文：<https://arxiv.org/abs/2301.10852>；DOI：<https://doi.org/10.1145/3582016.3582069>
- FEASTA, ASPLOS 2024，全文：<https://nicsefc.ee.tsinghua.edu.cn/nics_file/pdf/2d9c78a8-7279-4e15-9538-0664d04de93f.pdf>；DOI：<https://doi.org/10.1145/3620666.3651336>
- Sava, DATE 2024，全文：<https://past.date-conference.com/proceedings-archive/2024/DATA/368_pdf_upload.pdf>
- SALO, DAC 2022，全文：<https://arxiv.org/abs/2206.14550>；DOI：<https://doi.org/10.1145/3489517.3530504>
- SWAT, DAC 2024，全文：<https://www.comp.nus.edu.sg/~tulika/DAC_2024_DIA.pdf>
- SpARC, DAC 2024，DOI：<https://doi.org/10.1145/3649329.3655936>
- SparseTrim, CICC 2025，DOI：<https://doi.org/10.1109/CICC63670.2025.10982861>；会议 program：<https://www.ieee-cicc.org/wp-content/uploads/2025/02/CICC-2025-Program-2-14-25.pdf>
- AccelTran paper：<https://arxiv.org/abs/2302.14705>；公开 SV/simulator：<https://github.com/jha-lab/acceltran>
- STONNE cycle-level simulator：<https://github.com/stonne-simulator/stonne>
