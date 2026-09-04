# DATE 第十六轮 FCSR 复审整改与 QFSA 双侧关系架构

> **口径更新（2026-07-30）**：第十八轮已将 direct/residual 相序冻结为独立后端并行，并将 Stripe 强基线改为行内增量建表的 nonblocking 双 row ping-pong。有关周期、公平基线和贡献收敛，以 `docs/205_DATE第十八轮架构创新收敛_QFIT双向关联数据流_20260730.md` 为准。

## 1. 独立复审结论

本轮使用两个独立 subagent：

| 角色 | 评分 | 裁决 |
|---|---:|---|
| DATE 严格审稿人 | 2.7/5，新颖性 3.2/5 | Weak Reject；仅建议最小可证伪 RTL |
| 微架构/RTL 评审人 | 新颖性 3.8/5、物理 2.3/5、证据 1.7/5 | Conditional Go |

两份意见一致：

1. FCSR 不是普通 Q/K line buffer 的简单改名；
2. 但最强的 row/stripe fused baseline 可能实现相近的 `O(W)` 状态；
3. 当前没有 post-G0 ordered source workload；
4. 底边 burst、双时间平面和 gate/K snapshot 端口未闭合；
5. FCSR 单独作为 DATE 主贡献仍偏薄。

因此本轮不直接写 FCSR RTL，而先修正模型并提升架构抽象。

---

## 2. 已完成的强制纠错

### 2.1 两个时间平面公平复用

Local5 两个时间平面没有跨时间邻接。首版物理相序冻结为：

```text
T0完整空间平面 -> drain -> plane epoch切换 -> T1完整空间平面
```

B1 与 FCSR 都允许逐平面复用 K/gate 状态。修正后：

| 窗口 | B1两阶段状态 | FCSR状态 | 降低 |
|---:|---:|---:|---:|
| 2×9×9 | 7,965 bit | 3,663 bit | 54.0% |
| 2×15×15 | 20,205 bit | 5,633 bit | 72.1% |

此前 74.2%/85.0% 使用了不公平的 B1 `2HW` 与 FCSR `HW` 口径，正式撤销。

### 2.2 退休突发序列化

固定五点拓扑的最大几何退休 burst 为 3，但首版不实现 15 路 gate read、
3 路 K snapshot 和 3-enqueue FIFO。

冻结：

```text
one source snapshot/cycle
bottom burst 2/3 -> hold current destination bundle -> serialize retire
```

模型已将序列化 stall 计入。这样 FIFO 只需普通 1-in/1-out，当前 destination
在 retirement bundle 排空前阻塞下一项。

### 2.3 当前 gate/K 旁路

最后消费者产生的 gate 与 source K 可能来自当前 destination 的
self/north/west 位置。RTL 必须：

- 当前五 gate bundle 直接 bypass 到 snapshot mux；
- 当前 Direct-5 tap 中的 K 直接 bypass；
- 其他历史 gate 从 direction ring 读；
- snapshot 提交后才允许覆盖行状态。

否则同拍读写会读到旧 gate。

### 2.4 真实 source trace

ordered trace 已升级为 `et3_ordered_term_trace_v2`，新增：

- `source_term_count`；
- `source_gate_count`；
- `source_k_popcount`；
- `source_retire_destination`；
- `destination_direct_score_cycles`；
- `destination_delta_total`；
- `destination_qfsa_w2/w4/w8_score_cycles`。

回放入口：

```text
scripts/replay_local5_frontier_trace.py
```

它将输出四个同一时间线消融：

1. Direct 两阶段；
2. QFSA 两阶段；
3. Direct + FCSR；
4. QFSA + FCSR。

禁止把 QFSA-only 与 FCSR-only 的独立加速比相乘。

---

## 3. 为什么继续只做 FCSR 仍不够

即使 FCSR 做到：

- 三行 gate ring；
- 静态 last-consumer；
- source snapshot；
- score/projection overlap；

审稿人仍可将它解释为：

> 固定 stencil 的方向延迟线、row-wavefront 和 fused producer/consumer。

这可以是一项好的 memory/dataflow 机制，但较难独立支撑 DATE 架构主线。

Local5 还有第二个更强、且已由真实 profile 支持的结构特征：

| 特征 | pre-G0 profile100 |
|---|---:|
| 四方向 `Kneighbor XOR Kself` lane density | 1.8916% |
| 邻边 K 完全等于 self K | 86.0097% |

当前 TARE 逐邻居处理没有把该特征转化为高吞吐架构：

- 每个 neighbor 单独 issue；
- 每个候选单独 classifier；
- 五个 candidate 串行；
- window16 TARE 比 Direct 更慢。

问题不在 exact residual 公式，而在执行粒度仍是“一个 neighbor 一次”。

---

## 4. QFSA：Quotient-Frontier Stencil Architecture

QFSA 由两个互补方向组成：

```text
输入侧：Topology Quotient Score Fabric
  self K anchor + 四方向联合 residual
  -> 5个exact score

输出侧：Frontier Relation Transpose
  destination-major gate
  -> bounded-lifetime source relation
  -> source projection
```

共同原则是：

> 固定 stencil 图不只提供邻接地址，还提供可在硬件中静态利用的关系商与
> last-consumer frontier。

### 4.1 输入侧：拓扑商 score

对 direction `i`：

```text
K_i = K_self XOR Delta_i
```

alpha-XNOR 的整数 raw score 可写为：

```text
raw(Q, K_i)
  = raw(Q, K_self)
  + sum(delta_contribution(Q[l], K_self[l], K_i[l]))
      for l in changed_lanes(Delta_i)
```

现有 TARE RTL 已验证该 raw16 累加后单次 RNE 与 direct score bit-exact。
QFSA 不改变公式，只改变四方向 residual 的组织。

### 4.2 CDRP：跨方向残差打包

旧 TARE：

```text
for direction in N/S/E/W:
  classify Delta[direction]
  ZERO / LIST4 / DIRECT
```

QFSA：

```text
Delta[N/S/E/W] -> 128-bit tagged event mask
               -> hierarchical count/compact
               -> W4 {direction,lane,old,new,Q} events/cycle
               -> four direction accumulators
```

每个 residual event 带 `direction` tag。W4 后端同拍可更新同一或不同方向：

- 同方向多个 event 在小型 tagged reduction tree 中先求和；
- 不同方向送往四个 raw accumulator；
- accumulator 初值均为 self anchor raw；
- 每个方向完成后独立标记 ready；
- 五个 score 全部 ready 后进入 exact Shiftmax5。

该机制暂称：

> **CDRP（Cross-Direction Residual Packing）**

### 4.3 Exact direct/residual cost router

四方向只有 16 种 direct/residual 分配。对每个 destination，硬件已知四个
delta popcount `c_i`，可选择共享 direct popcount 或 W-lane residual：

```text
extra_cycles(W)
  = min over direct subset D:
      |{i in D | c_i > 0}|
      + ceil(sum(c_i for i not in D) / W)
```

总 score 周期：

```text
1 anchor cycle + extra_cycles(W)
```

这是 exact cost routing：

- zero direction 直接复用 anchor；
- 多个低密方向跨 direction 合并到同一 W4 wave；
- 单个高密方向可由共享 32-lane direct engine 重算；
- 不剪枝、不近似、不使用预测命中率。

profile 脚本已加入每 destination 的 W2/W4/W8 exact cycle 统计。当前尚未生成
fullres post-G0 数字，不能先宣称加速。

独立整数参考：

- `scripts/qfsa_exact_reference.py`
- `results/qfsa_exact_reference_20260730/report.{md,json}`

10,000 个随机五候选 case、50,000 个 score 比较得到 0 mismatch。该证据只
证明 anchor raw + changed-lane residual + 单次 RNE 与五路 direct 整数等价，
不证明 tagged compactor 或周期/PPA。

### 4.4 蝶形网络的正确使用位置

复旦 ISSCC/FABNet 类 butterfly zero-compaction 不再放在普通 term NoC。
QFSA 中更合理的位置是 128-bit 四方向 residual mask 到 W4 tagged event 的
压紧：

```text
4 x 32 direction masks
  -> local 8-lane selectors
  -> direction-tagged butterfly/prefix merge
  -> first 4 events + remaining count/state
```

本工作不能声称提出 butterfly。可写差分是：

- 被压紧的不是稀疏 weight 或普通 activation；
- event 带 direction tag 并更新四个独立 exact score accumulator；
- 密集方向由 cost router 转到共享 direct engine；
- compactor 与固定五点 topology quotient 联合设计。

必须比较：

- priority encoder；
- linear prefix/LIST4；
- tagged butterfly；
- area/Fmax/toggle；
- 多 wave 和 direct fallback。

只有同工艺 EDP 为正才采用 butterfly 实现。

---

## 5. QFSA 完整流水

```text
Plane-serial paired-time row stream
        |
        v
Direct-5 streaming taps
        |
        +--> Kself anchor raw (1 x 32-lane alpha-XNOR)
        |
        +--> 4 x XOR mask + popcount
                  |
                  v
        exact cost router
          | residual directions
          v
        tagged W4 compactor/reducer ----+
          | direct directions           |
          +--> shared direct raw engine |
                                       v
                              4 score accumulators
                                       |
                          single-RNE x 5 / Shiftmax5
                                       |
                         direction gate ring + bypass
                                       |
                           static source frontier
                                       |
                       one-snapshot/cycle retire queue
                                       |
                     source term builder / product directory
                                       |
                         bank-local destination Acc
```

关键状态：

- 三行 Q/K；
- 四个 direction delta mask 或 compact cursor；
- 五个 raw score accumulator；
- 三行 direction gate ring；
- 当前 gate/K bypass；
- depth-8 source descriptor FIFO；
- plane epoch、row/column 和 retirement hold。

---

## 6. 外部机制借鉴与本土化

| 工作 | 借鉴机制 | QFSA 本土化 | 不能声称 |
|---|---|---|---|
| Prosperity | exact/partial relation reuse | self K 是拓扑静态 anchor；不做 TCAM matcher | product-sparsity/reuse 原创 |
| Phi | pattern + residual 两级执行 | 固定 self pattern + 四方向 exact delta；无 learned codebook | Phi 复现 |
| Bishop | density stratification、TTB | exact cost router；frontier-complete STT 控制元数据 | ECP、异构双核原创 |
| FireFly-T | 多 lane event decoder | W4 direction-tagged residual events | decoder 原创 |
| 复旦 ISSCC/FABNet | butterfly compaction | 四方向 tagged mask 的可选物理实现 | butterfly 原创 |
| streaming stencil | line reuse | 在线 relation transpose 与 source last-use retirement | line buffer 原创 |
| sparse outer-product | source scatter | bounded stencil source descriptor 与 exact destination Acc | 通用 outer-product 原创 |

Prosperity/Phi/FireFly/butterfly 不再作为互相独立的“已采用贡献”。它们只解释
QFSA 输入侧某个机制的来源。论文创新落在：

> 固定 stencil 图的 topology quotient 与 consumer frontier 被同一个流水同时
> 利用，从输入 score 到输出 projection 双侧消除冗余关系工作和全窗口 barrier。

---

## 7. 可单列的 DATE 贡献

若真实 profile、RTL 和 PPA 全部过线，建议只列三条：

### C1：Topology-Quotient Multi-Relation Score Fabric

一个 self anchor raw 加跨四方向 tagged residual packing，同时生成五个 exact
score；高密方向由 exact cost router 回退到共享 direct engine。

### C2：Bounded-Lifetime Online Relation Transpose

利用静态 last-consumer frontier，把 destination-major gate 在线转换为
source-major relation，并将中间状态从完整 window 降为行有界状态。

### C3：Quotient-to-Frontier Elastic Pipeline

在同一 ordered schedule 中把 variable-latency quotient score producer 与
source projection consumer 连接，使用真实 source work、retirement hold 与
backpressure，不物化完整 K/gate plane。

以下不再单列贡献：

- 三行 line buffer；
- 双时间平面复用；
- W4 compactor；
- butterfly；
- FIFO；
- MFEP；
- DCTF bank；
- 定点格式。

---

## 8. 强基线与消融

### 8.1 Score 前端

| 编号 | score结构 | 目的 |
|---|---|---|
| S0 | 1×32-lane direct，五候选串行 | 等面积强基线 |
| S1 | 5×32-lane direct，并行 | 等吞吐面积上界 |
| S2 | 旧逐邻居 TARE-4 | 证明联合打包而非 residual 公式带来收益 |
| S2a | 4×W1 独立方向 residual + 共享 direct | 与 QFSA-W4 总 residual lane 数相同的最强隔离基线 |
| S3 | Phi-like pattern/residual | codebook/matcher强基线 |
| S4 | Prosperity-like matcher | 在线关系发现成本对照 |
| A1 | QFSA W2/W4/W8 | 新架构 |

### 8.2 Relation/投影

| 编号 | 结构 |
|---|---|
| R0 | full-plane two-phase |
| R1 | row/stripe fused double buffer |
| R2 | DiSEP full-plane inverse addressing |
| A2 | FCSR serialized frontier |

### 8.3 组合

必须在同一 trace 报：

```text
S0+R0  Direct two-phase
A1+R0  QFSA-only
S0+A2  FCSR-only
A1+A2  QFSA combined
```

主结果是 `A1+A2` 相对 `S0+R0/R1/R2` 中最优者，不是各子模块收益相乘。

第17轮审稿后已将 `S2a` 和 `R1` 写入 trace/replay：

- `S2a`：四个 W1 方向 lane 独立推进，空闲 lane 不能跨方向借用；与 QFSA-W4
  共享一个 direct engine 和四个 accumulator；
- `R1`：整 source row 在下一 consumer row 结束后进入双 row descriptor
  buffer；与 FCSR 比较逐 row barrier 和逐 source frontier；
- combined 强基线：`4×W1 + Stripe`；
- combined 候选：`QFSA-W4 + FCSR`。

当前只有回放合同和单元测试，等待 post-G0 trace 后才产生结果。

---

## 9. 评估口径

### 周期

- destination score cycles；
- residual/direct route；
- compactor waves；
- Shiftmax latency；
- retire hold；
- FIFO stall；
- source builder setup/scan；
- product compute；
- destination delivery；
- Acc bank conflict；
- plane drain；
- mean/p95/p99 window latency。

### 存储与流量

- Q/K line state；
- full K/gate plane；
- gate ring；
- delta mask/cursor；
- descriptor FIFO；
- product directory；
- weight/Acc SRAM；
- K/gate/delta/term/delivery bit traffic；
- macro 对齐后面积，不只报逻辑 bit。

### PPA

- 同一工艺库、同一 SDC；
- S0/S1/S2/A1 同输入吞吐；
- R0/R1/R2/A2 同 SRAM 端口；
- DC/STA；
- post-G0 trace SAIF；
- energy/window、EDP、area-normalized throughput；
- Fmax 下降；
- score 与 projection 分项。

---

## 10. RTL 前证据门

QFSA score RTL 进入实现前必须由 fullres post-G0 profile 给出：

- joint delta events/destination mean/p95/p99/max；
- active direction count；
- W2/W4/W8 score cycle reduction；
- direct fallback 比例；
- residual wave burst；
- 四 stage 分布；
- invalid boundary 分布。

FCSR RTL 进入实现前必须给出：

- source term/destination ordered trace；
- real retirement burst；
- source FIFO depth/stall；
- builder term 与 delivery 周期；
- row/stripe strong baseline。

若 Local5 fullres accuracy 不具竞争力，QFSA 仍可作为硬件探索，但不能替代
Motion 算法主线。

---

## 11. 双线架构地位

### Motion

当前优势：

- full30/valid825/fullres 结果更成熟；
- TTB、Motion-XOR、SCS-Shiftmax、term projection 已有较完整 RTL。

当前弱点：

- ACRT 输给强 all-class replay；
- DCTF-2C 偏工程；
- 需要 DC/STA/SAIF 而不是继续堆机制。

### Local5/QFSA

当前优势：

- 固定 stencil 同时提供 input quotient 和 output frontier；
- pre-G0 delta 结构非常强；
- 有机会形成从 score 到 projection 的完整架构原则。

当前弱点：

- fullres accuracy 尚在训练；
- joint ordered profile 尚无结果；
- QFSA/CDRP 只有模型合同，没有 RTL/PPA；
- FCSR 尚未对比 row/stripe 强基线。

因此当前裁决仍是：

> **Motion 保持算法/部署证据主线；Local5/QFSA 成为架构创新优先线。**

是否切换论文唯一主线，等待 Local5 fullres valid825、post-G0 ordered profile
和 QFSA/FCSR 同约束 PPA。

---

## 12. 本轮新增与验证

新增/修改：

- `scripts/model_local5_frontier_retirement.py`
- `scripts/replay_local5_frontier_trace.py`
- `scripts/profile_local5_hardware_features.py`
- `scripts/et3_ordered_trace_replay.py`
- 对应三组测试。

已通过：

```text
32 tests PASS
```

覆盖：

- frontier 几何；
- retirement burst 序列化；
- 公平逐平面存储；
- v1/v2 trace 兼容；
- source work/retire 导出；
- joint-direction W2/W4/W8 exact cost 枚举；
- 50,000 score 的 QFSA 整数参考零失配；
- Direct/QFSA/FCSR/combined 回放接口。
- 4×W1 同总 residual 宽度基线与 row/stripe 强基线。

未覆盖：

- QFSA RTL；
- gate/K write-through snapshot；
- row/stripe baseline RTL；
- fullres post-G0 trace；
- DC/STA/SAIF。

本轮状态：

> **QFSA 已形成比 FCSR 单机制更完整的架构候选，但仍处于
> PROFILE_CONTRACT_READY / RTL_NOT_STARTED。**
