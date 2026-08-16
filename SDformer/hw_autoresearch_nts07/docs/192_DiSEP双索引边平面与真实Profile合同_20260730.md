# DiSEP 双索引边平面与真实 Profile 合同

## 1. 本轮结论

Local5 不再以“source-major 少读 K”作为架构贡献。公平计算 Q/K 总读取后，
query gather 与 source scatter 都是每条五邻域边读取一对 Q/K，单纯交换循环
顺序并不减少总向量访问。

本轮保留的 Local5 架构候选是：

> **Dual-Indexed Stencil Edge Plane（DiSEP）**：将 Local5 的五方向
> final-gate 存入方向银行，同一份数据由 destination 地址写入、由
> source 与逆方向地址读取，从而在 destination-major normalization 和
> source-major projection 之间切换所有权，但不执行实体 transpose/copy。

DiSEP 当前只通过 projection 整数等价，不是 RTL，也没有真实 workload 收益
和 PPA。Motion 的 LS-COTT 因逐 lane 容量和 exact spill 未闭合，继续暂停。

---

## 2. 为什么这是架构问题

Local5 的两个相邻阶段要求不同的数据所有权：

```text
score/Shiftmax5:
  destination d owns {self, north, south, east, west} candidates

projection:
  source s owns all destination edges that consume K[s]
```

普通实现有两个选择：

1. 一直保持 destination-major，投影阶段重复生成 source/product；
2. 归一化后执行一次 destination-to-source transpose，再做 source-major 投影。

DiSEP 用固定五点 stencil 的双射关系取消第二种实现中的复制阶段：

```text
Phase A
Q/K line buffer
  -> destination-major 5-score row
  -> exact Shiftmax5
  -> G[direction][destination]

Phase B
source generator
  -> inverse-direction address
  -> G[direction][destination(source, direction)]
  -> {source, final-gate, lane} grouping
  -> one product, multiple destination deliveries
  -> integer Acc
```

架构主张不是“五个 SRAM bank”，而是：

> 利用固定 stencil 的方向可逆性，使同一 edge-gate plane 同时具有
> destination-major 写语义和 source-major 读语义，跨阶段不搬运数据。

---

## 3. 地址合同

设 `G[r][d]` 表示 destination `d` 对角色 `r` 的 final gate。Phase B 对
source `s` 的读取为：

| source 在 destination 中的角色 | 读取 |
|---|---|
| self | `G[self][s]` |
| north | `G[north][south(s)]` |
| south | `G[south][north(s)]` |
| west | `G[west][east(s)]` |
| east | `G[east][west(s)]` |

边界外地址必须由真实 candidate-valid mask 屏蔽。不能把 invalid candidate
当作 gate=0 的普通边，因为它不属于 Shiftmax5 denominator。

时间维不发生跨时刻邻接；`T=162` 和 `T=450` 分别是两个独立的
`9x9`、`15x15` 空间平面。

---

## 4. 存储与流水建议

### 4.1 第一版物理组织

```text
5 x direction gate bank
  depth: spatial tokens per time plane
  width: final gate code
  write: one completed destination row writes five directions
  read: source engine reads up to five inverse-direction entries

source term builder
  key: {source, final-gate, K lane}
  value: destination mask/list

product lane
  input: {gate, K lane, weight tile}
  output: product resident across destinations

integer Acc
  destination-indexed, sufficiently wide, no per-add saturation
```

第一版不得假设五个读口免费。应比较：

- 五个单口方向 bank 并行读；
- 一个或两个 bank 的序列化读；
- bank 复制；
- 与 B2 transpose buffer 相同总 bit、相同 macro 端口的实现。

### 4.2 可重叠调度

建议以 stripe 为粒度双缓冲：

```text
buffer A: Phase A 写当前 stripe
buffer B: Phase B 读上一 stripe
```

halo 行必须计入容量和等待周期。若 Phase A 必须等完整窗口结束才允许 Phase B
读，则 DiSEP 只取消拷贝，不一定改善吞吐。

---

## 5. 整数等价证据

参考模型：

- `scripts/local5_disep_reference.py`
- `tests/test_local5_disep_reference.py`
- `results/local5_disep_reference_20260730/report.json`

覆盖：

- `T=1`；
- `T=162=2x9x9`；
- `T=450=2x15x15`；
- 500 个随机几何、边界 mask、随机 invalid edge；
- gate=0、K 稀疏、INT8 权重；
- destination gather 与 DiSEP source projection 的最终整数 Acc。

结果：

| 指标 | 数值 |
|---|---:|
| cases | 503 |
| compared accumulator | 183,937 |
| mismatch | 0 |
| delivery 守恒 | 143,388 / 143,388 |
| synthetic product reduction | 0.61% |
| synthetic max gate fanout | 2 |

0.61% 来自随机 gate，只说明：

> 关系转置本身不会自动产生复用；DiSEP 必须由真实 Local5 final-gate 的
> source-centric 重复度证明。

该结果不能写成 Local5 加速收益，也没有覆盖 Shiftmax hardware-order、
ready/valid、SRAM latency、周期或 PPA。

---

## 6. 真实 Profile 合同

已扩展：

- `scripts/profile_local5_hardware_features.py`
- `tests/test_local5_source_gate_lane_stats.py`

新增统计以 `(window-head, source, final-gate, lane)` 为 exact term key：

| 字段 | 含义 |
|---|---|
| `source_gate_lane_terms` | source-major 只生成一次的 product term |
| `source_gate_lane_delivery` | 原始 active edge-lane delivery，必须守恒 |
| `source_gate_lane_max_fanout` | 单 term 最大 destination fanout |
| `source_gate_lane_fanout_histogram` | fanout 完整分布 |
| `source_gate_lane_terms_per_window_head_histogram` | 容量和尾延迟依据 |
| `source_gate_cardinality_histogram` | 仅 active source 的 gate 基数 |
| `source_gate_cardinality_all_histogram` | 包含空 source 的 gate 基数 |
| `source_active_instances/source_instances` | source 调度占用率 |

运行时强制：

```text
source_gate_lane_delivery == naive_active_edge_products
source_gate_lane_terms <= source_gate_lane_delivery
```

只有 Local5 fullres、真实 invalid mask、Q7 score、Q1.7 gate、
hardware-order Shiftmax 后的 ordered profile 可以进入论文主表。pre-G0 与随机
参考只用于检查方向。

---

## 7. 公平基线

| 编号 | 数据流 | 是否复制 gate plane | 投影计算 |
|---|---|---:|---|
| B0 | 当前单口 query gather | 否 | destination逐edge |
| B1 | 等带宽 streaming query gather | 否 | destination逐edge |
| B2 | query normalize + 实体 transpose | 是 | source逐edge |
| B2a | 普通方向bank + 逆地址读取 | 否 | source逐edge |
| B2b | 普通方向bank + 标准五项比较/CSE | 否 | source内局部gate合并 |
| A0 | DiSEP 双索引方向bank | 否 | source-gate-lane term |

统一约束：

- 相同 score lane 和 projection lane；
- 相同 Q/K SRAM 总端口、复制和 line-buffer 容量；
- 相同 Shiftmax5 数值顺序；
- 相同 gate storage 总 bit 与 macro 规则；
- 相同 Acc 端口和整数宽度；
- 相同 SRAM latency、ready/valid 与 final backpressure；
- fullres `2x15x15` 几何。

必须分别报告：

1. score/normalize 周期；
2. gate-plane 写读流量；
3. transpose/copy 周期和 bit traffic；
4. product generation 数；
5. destination delivery 数；
6. Acc 冲突与 stall；
7. mean/p95/p99 window latency；
8. DC/STA/SAIF 的面积、频率、功耗和 EDP。

只有 A0 相对 B1、B2a 和 B2b 中最优者的改善才属于 DiSEP。B2只用于
量化实体transpose成本，不能作为唯一强基线。

---

## 8. 晋级与淘汰门槛

DiSEP 进入 RTL 的最低条件：

- fullres post-G0 `source_gate_lane_term_ratio <= 0.80`；
- p95 source gate cardinality足以形成稳定复用，而不是仅均值有效；
- 相对 B2，消除 transpose 后片上 traffic 至少下降 20%；
- 相对 B2a/B2b，source-gate grouping 仍有可测周期或能耗收益；
- 相对 B1，含方向 bank、term builder 和 Acc 冲突后的周期不退化超过 5%；
- 整数 Acc 全量等价；
- T450 地址、边界和双缓冲无容量溢出。

进入 DATE 主贡献的最低条件：

- 相对 B1、B2a 和 B2b 中最优者的同约束 EDP 至少改善 15%；
- 多 sample、多 window 的 mean/p95/p99 均成立；
- 收益不能只来自更多 SRAM 端口或复制；
- 目标工艺 DC/STA/SAIF 闭合。

若真实 term ratio 接近 1，DiSEP 应淘汰为负结果，不以“无拷贝”单独充当贡献。

---

## 9. 双线当前排序

| 路线 | 成熟度 | 架构新意 | 当前决策 |
|---|---|---|---|
| Motion SCS/ACQN + GCM-P | profile与RTL较成熟 | 以 exact class/gate term 数据流为主，系统新意仍偏弱 | 保留强基线 |
| Motion LS-COTT | 仅模型 | class-owned normalize-to-term 有差分，但 spill 不闭合 | 暂停 RTL |
| Local5 DiSEP | projection 整数等价 | 双索引跨阶段所有权切换，有成为数据流贡献的可能 | 等真实 profile |

当前不能宣布 Motion 或 Local5 已成为 DATE 架构主线。下一决策点是：

1. H67 fullres hardware-order profile；
2. Local5 fullres hardware-order profile；
3. DiSEP source-centric term/fanout；
4. Motion 逐 lane class/event 容量与 exact spill。
