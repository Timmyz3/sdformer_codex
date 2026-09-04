# 被否定 Idea 重开、ET3 原生 Set/Multiset 切片与第一轮 DATE 整改

## 1. 本轮回答的问题

本轮没有把过去否定的结构全部重新堆回架构，而是用 Motion 与 Local5 的现有 workload 证据重新过筛。结论如下：

1. Bishop 式 dense/sparse 双核继续否定。两条线的 exact-or-LIST4 覆盖均超过 94%，当前没有证据证明复制 dense core、双 FIFO 和 stratifier 后仍有净 EDP 收益。
2. Phi 式学习 codebook 继续否定。Motion 的 temporal peer 与 Local5 的 self stencil 已提供静态 exact anchor，尚无证据支持额外 matcher、codebook SRAM 和 residual decoder。
3. 独立蝶形网络不能作为贡献。它只保留为高 fanout compaction/multicast 的物理消融候选，必须在同 SDC、同 SRAM 下胜过 central/prefix 才能恢复。
4. ARST 锚点驻留 tile 暂缓。它需要真实 payload fetch/hold/release 与 SRAM latency 证据。
5. ET3 被选为唯一立即落地候选。原因不是“再做一个 term buffer”，而是 Motion 的 SET 与 Local5 的 bounded MULTISET 可以共享同一 exact typed term 合同。

机器可重跑的审计结果见：

- `scripts/reconsider_rejected_dual_line_ideas.py`
- `tests/test_reconsider_rejected_dual_line_ideas.py`
- `results/dual_line_reopened_ideas_20260728/reconsideration.md`
- `results/dual_line_reopened_ideas_20260728/reconsideration.json`

## 2. 第一轮独立 DATE 评审

第一轮评审给出的分数为：

| 维度 | 分数 |
|---|---:|
| 当前 DATE 完整度 | 2/5 |
| 当前创新性 | 2/5 |
| 目标创新性 | 3/5 |
| 证据完整度 | 2/5 |

评审的核心否决点不是 RTL 行数，而是：

1. Local5 结果只能称为 pre-G0 离线理想 MPET 聚合机会，不能称为在线硬件收益。
2. 缺少 post-G0、全分辨率、有序 destination trace。
3. 必须实现有限目录、分段 destination、overflow fallback 与原生 multiplicity backend。
4. 必须同时比较 dense per-edge、MFEP+EXPLODE 和原生 ET3，证明 exact 语义及工作量差异。
5. CATF/蝶形和 ARST 不应在 ET3 闭环前扩张。

因此本轮只实现一条审稿人可检查的完整切片：

```text
Motion SET / Local5 MULTISET source
        |
        v
bounded key directory
        |
        +-- same key and segment not full --> append destination
        |
        +-- same key but segment full -----> allocate next segment
        |
        +-- directory full ----------------> exact fallback FIFO
        v
segmented typed term stream
        v
native multiplicity-aware executor
        v
integer accumulator
```

## 3. ET3 的架构合同

### 3.1 Typed key

每个目录项的 key 为：

```text
{mode, group_tag, gate_code, lane_id, multiplicity}
```

- Motion：`mode=SET`，强制 `multiplicity=1`。
- Local5：`mode=MULTISET`，允许 `multiplicity=1..5`。
- `group_tag` 隔离窗口/head 生命周期，禁止跨 group 合并。
- destination 在同一 group 内必须唯一，重复 destination 被视为协议错误。

### 3.2 有限目录与分段

目录不是离线无限 `unique()`：

- `KEY_CAP` 限制同时驻留的 segment 数。
- `SEG_DEPTH` 限制每个 segment 的 destination 数。
- 同 key 的 destination 超过 `SEG_DEPTH` 时创建同 key 的下一 segment。
- 目录耗尽后，每个输入 item 进入有限 fallback FIFO，并作为单 destination exact term 发射。
- fallback 只损失复用机会，不改变累加结果。

### 3.3 原生 multiplicity 执行

原生 executor 在 term 首 beat 计算一次：

```text
product[out] = multiplicity * gate_code * weight[lane][out]
```

后续 destination beat 复用该 product。与 EXPLODE 相比，它不会把 multiplicity 为 `m` 的 item 展开为 `m` 条重复命令。Motion 的 `m=1` 是同一数据通路的严格子集，而不是旁路特例。

### 3.4 原子边界与反压

typed stream 显式携带：

- `term_first`
- `term_last`
- `head_last`
- `fallback`

当下游 `ready=0` 时，`valid` 与全部 payload 必须稳定。只有 `valid && ready` 才能推进 destination、term 或 group 边界。

## 4. RTL 与验证

新增 RTL：

- `rtl_et3/et3_bounded_term_directory.sv`
- `rtl_et3/et3_native_multiset_executor.sv`
- `rtl_et3/et3_native_slice_top.sv`

新增验证：

- `tb_et3/tb_et3_native_slice.sv`
- `sim_et3/run_et3_native_slice_checks.sh`

测试配置故意设置为 `KEY_CAP=2, SEG_DEPTH=2`，同时触发：

1. Motion SET 的同 key 多 segment。
2. Local5 MULTISET 的同 key 多 segment。
3. 两个 group 各一次目录 overflow fallback。
4. 有效输出反压，并检查 stalled payload 稳定。
5. dense 整数金参考逐 destination、逐 output lane 比较。
6. term first/last、head_last、SET/MULTISET、fallback 计数审计。

结果：

```text
Python profile/evidence tests: 5/5 PASS
Icarus functional simulation: PASS
Verilator RTL lint: PASS，无 RTL warning
Verilator functional simulation: PASS
Yosys synthesis-readiness check: PASS
```

小规模测试的审计计数：

| 项目 | 数值 |
|---|---:|
| source/destination beat | 9 |
| typed term / product compute | 6 |
| native command | 9 |
| EXPLODE baseline command | 15 |
| fallback item/term | 2 |
| SET term | 3 |
| MULTISET term | 3 |
| 人工反压 stall | 3 |
| accumulator mismatch | 0 |

这些数字只证明 RTL 行为，不是 workload 加速比。

## 5. Workload 指导意义

Local5 pre-G0 profile100 的离线统计为：

| 指标 | 数值 | 证据边界 |
|---|---:|---|
| exact-or-LIST4 覆盖 | 94.6010% | pre-G0 profile |
| 理想 MPET term | 13,732,741 | 离线全局 unique |
| 逐 destination term | 153,748,435 | 离线 profile |
| 理想 fanout mean/p95/max | 11.20 / 45 / 110 | 离线 profile |
| 原生 multiplicity destination command 减少 | 18.3810% | 相对 EXPLODE 的离线上界 |
| 理想 MPET product compute 减少 | 92.7098% | 未计目录和 fallback |

因此，ET3 的硬件假设获得了比“Local5 也能用 term”更具体的支持：

1. multiplicity 必须在 executor 原生消费，否则先损失约 18.38% 的命令机会。
2. 跨 destination term 复用潜力很大，但实际收益完全取决于在线目录容量、顺序和 fallback。
3. fanout p95=45 表明固定两目的 PPDI 不是最终形态；应以 segmented destination 为主，pairing 只做 delivery 微结构消融。
4. w15 用 w9 fanout 外推时 adaptive list/bitmap 相对全 list 仅省 2.6947%，因此暂不把 CATF 自适应格式列为独立贡献。

Motion 已有更成熟的 ordered profile100；Local5 仍缺 post-G0 ordered trace。因此当前不能因为 Local5 的离线上界更大就直接切换主线。

## 6. 当前可写与不可写的创新点

当前唯一新增、可进入候选贡献列表但仍需真实 trace/PPA 支撑的架构点是：

> ET3：一种面向 all-binary attention-to-projection 的在线有界 exact tile-to-term transduction。它以统一 typed IR 保留 Motion SET 与 Local5 bounded MULTISET 的不同代数语义，通过 segmented destination 和无损 fallback 避免物化 gated-K tensor，并由原生 multiplicity executor 消除重复 edge command。

当前不能写成已完成贡献的内容：

- 全分辨率 ET3 加速或节能。
- post-G0 Local5 的 92.71% product reduction。
- 自适应 list/bitmap fabric。
- 蝶形网络优于 prefix/central。
- 目标工艺面积、频率、功耗或 EDP。

## 7. 下一阶段硬门槛

1. Local5 post-G0/full-resolution ordered destination trace。
2. Motion 多样本 ordered trace 对齐同一 schema。
3. `KEY_CAP × SEG_DEPTH × fallback` 的 overflow、周期、p95/p99 与存储 DSE。
4. 真实尺寸、SRAM latency、随机反压下的三方 bit-exact 回放。
5. dense、EXPLODE、ET3 在同 SRAM macro、同 SDC、同 trace 下的 DC/STA/SAIF。
6. 第二轮 DATE 复审后，只迭代评分最低且可实证的部分。
