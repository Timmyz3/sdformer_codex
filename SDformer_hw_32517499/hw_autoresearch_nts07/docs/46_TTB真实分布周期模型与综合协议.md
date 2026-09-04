# TTB 真实分布周期模型与综合协议

**日期**：2026-07-13  
**对象**：H60 TTX、H67 Motion-XOR TTX、H68 deployment H60  
**目的**：用 profile100 的真实 token-time bundle 分布建立可复算的 C0/B1 cycle/traffic 模型、路由阈值 sweep 和同工艺综合协议。本文不提供或推断未经工具测量的 PPA。

## 0. 结论

1. 三条线的 true TTB 都很稀疏，但 bundle 数量和周期收益不能由平均 density 直接推出。bundle1 empty 为 `72.54%--74.20%`，bundle8 empty 仍为 `54.70%--55.88%`；这是可用于路由的实测覆盖率，不是 speedup。
2. H67 的 stage 差异很大。bundle8 empty 在 S0 只有 `26.51%`，在 S1 为 `87.30%`。模型必须按 row/stage 顺序回放；用全局 empty ratio 乘总周期会掩盖 FIFO burst、bank conflict 和尾部延迟。
3. `Q OR K` empty 不能删除 Shiftmax 中的 token。它只能在能够从 metadata 提前识别时，旁路 Q/K payload 和 popcount，并向 score row **注入精确 silent/silent 常数**。K-zero、no-K-motion 和 Delta zero-update各自只能 gate 对应子路径。
4. B1 Exact-Delta 的阈值变量是每 token/head 的 `u=popcount(Q_toggle OR K_toggle)`；true-density 的 `A_b=sum(Q OR K)` 是 bundle payload/activity 指标。两者语义不同，必须在 trace 和表格中分列。
5. 现有数据可精确重放 `theta={2,4,8,16}`；没有 `<=12` raw count。`theta=12` 只能报告 `theta=8` 与 `theta=16` 之间的严格区间，补 profile 后才能给单点结果，禁止插值。
6. 当前数据支持 C0 作为可审计基线，并支持实现 B1 trace simulator；仍不支持宣称 B1 的 PPA 或净能效优于 C0。

## 1. 数据源和字段语义

用户给出的 `results/ttb_true_density_ttx_h67_h68_profile100.json` 在本仓库实际路径为：

```text
neuron_experiments/H9_bipolar_self_attention/results/
ttb_true_density_ttx_h67_h68_profile100.json
```

同时读取：

- `docs/45_TTB异构双路径微架构评估.md`；
- 三个 profile100 的 `nts11_hardware_p0_profile.json/.md`；
- `_token_time_bundle_stats()`、`_delta_locality_stats()` 和 profiler aggregation 字段。

### 1.1 True TTB 定义

固定 `T=2`、`D=32`。对连续 `b in {1,2,4,8}` 个 spatial token：

```text
L_b       = T * b * D = 64b                 # Q/K lane-pair positions
P_QK,b    = 2 * L_b = 128b bit              # dense Q+K payload，不含 metadata
A_b       = sum over bundle of (Q OR K)      # 0..L_b
E_b       = [A_b == 0]
KZ_b      = [sum(K) == 0]
MZ_b      = [sum(K0 XOR K1) == 0]
F_b(k)    = count(1 <= A_b <= k) / N_b
```

`activity_density` 是 `sum(A_b)/sum(L_b)`。`active_1_K_ratio` 的分母是全部 bundle，包括 empty 和大于 K 的 bundle；它是候选 sparse route 覆盖率，不是条件概率，也不是已实现 skip。

### 1.2 Exact-Delta 定义

对单个 spatial token/head：

```text
U         = (Q0 XOR Q1) OR (K0 XOR K1)
u         = popcount(U)                     # 0..32
E_delta   = [u == 0]
S_delta(theta) = [1 <= u <= theta]
D_delta(theta) = [u > theta]
```

`delta_bundle4/8_empty` 表示 bundle 内所有 token 的 `u=0`。它与 true TTB 的 `A_b=0` 不同：前者表示跨时间不变，后者表示两个时间片 Q/K 全静默。

## 2. Profile100 实测分布

### 2.1 全局 true TTB

| model | b | Q-or-K density | empty | K-zero | no K-motion | `1<=A<=4` | `1<=A<=8` | `1<=A<=16` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TTX ep2 dyadic | 1 | 1.691499% | 72.539530% | 82.282540% | 82.369915% | 19.717835% | 24.120280% | 26.879439% |
| TTX ep2 dyadic | 2 | 1.691499% | 66.127210% | 78.106500% | 78.168351% | 20.017777% | 26.023788% | 30.592211% |
| TTX ep2 dyadic | 4 | 1.691499% | 60.089640% | 73.525184% | 73.570075% | 18.556144% | 26.025553% | 32.241234% |
| TTX ep2 dyadic | 8 | 1.691499% | 54.704775% | 68.838231% | 68.871861% | 16.705019% | 23.933644% | 31.997132% |
| H67 ep19 dyadic | 1 | 1.502114% | 73.897325% | 83.106384% | 83.175281% | 19.043564% | 23.382608% | 25.765469% |
| H67 ep19 dyadic | 2 | 1.502114% | 67.319149% | 79.544915% | 79.584164% | 20.094138% | 25.416482% | 30.060322% |
| H67 ep19 dyadic | 4 | 1.502114% | 60.963301% | 75.636288% | 75.667241% | 19.685629% | 26.506512% | 31.869218% |
| H67 ep19 dyadic | 8 | 1.502114% | 55.255939% | 71.416599% | 71.443439% | 18.047010% | 25.623377% | 32.758293% |
| H68 ep19 dyadic | 1 | 1.548900% | 74.201277% | 83.292383% | 83.355065% | 18.398058% | 22.791555% | 25.405763% |
| H68 ep19 dyadic | 2 | 1.548900% | 67.769127% | 80.035264% | 80.068906% | 19.527250% | 24.596737% | 29.303615% |
| H68 ep19 dyadic | 4 | 1.548900% | 61.517474% | 76.512280% | 76.536657% | 19.329514% | 25.838329% | 30.946131% |
| H68 ep19 dyadic | 8 | 1.548900% | 55.880384% | 72.718655% | 72.740138% | 17.770766% | 25.208699% | 32.022727% |

这些数是 raw-count weighted profile100 结果。不同 `b` 的 density 相同是因为 numerator/denominator 对同一批 lane 重新分组；empty、K-zero 和 CDF 会随分组变化。

### 2.2 H67 分 stage：empty / activity density

| stage | b=1 | b=2 | b=4 | b=8 |
|---:|---:|---:|---:|---:|
| S0 | 56.484% / 2.648% | 45.490% / 2.648% | 35.101% / 2.648% | 26.513% / 2.648% |
| S1 | 94.919% / 0.186% | 92.893% / 0.186% | 90.418% / 0.186% | 87.300% / 0.186% |
| S2 | 82.490% / 0.891% | 78.258% / 0.891% | 74.175% / 0.891% | 70.210% / 0.891% |
| S3 | 67.937% / 1.898% | 61.433% / 1.898% | 55.388% / 1.898% | 49.915% / 1.898% |

每格为 `empty / Q-or-K density`。TTX 与 H68 也呈同样的强 stage 偏斜：S1 最稀，S0 最密。因此不能只按全局比例配 FIFO 或决定 core 宽度。

### 2.3 Exact-Delta 实测

| model | union-toggle density | zero-update token/head | empty bundle4 | empty bundle8 | mean changed run |
|---|---:|---:|---:|---:|---:|
| TTX ep2 dyadic | 2.785797% | 72.649925% | 60.136104% | 54.739989% | 3.9776 |
| H67 ep19 dyadic | 2.508996% | 73.999091% | 61.008022% | 55.290287% | 3.7017 |
| H68 ep19 dyadic | 2.552265% | 74.295967% | 61.558461% | 55.911905% | 3.7412 |

注意：`mean changed run` 是 changed token 数除以 run-start 数，不是 index byte reduction；是否用 RLE 仍要看完整 run histogram、边界和实际 burst trace。

## 3. 可审计模型输入

对每个 model/stage/row/bundle size 建立 exact histogram：

```text
n_b(a)       = count(A_b == a), a in [0, L_b]
n_delta(u)   = count(update_count == u), u in [0, 32]
N_b          = sum_a n_b(a)
N_token      = sum_u n_delta(u)
```

硬件参数必须由 RTL/综合填写，不能用论文或经验值代入：

| symbol | 含义 |
|---|---|
| `W0` | C0 每周期处理的 lane-pair 数 |
| `Wd`, `Ws` | B1 dense/sparse core lane 宽度 |
| `Bbus` | Q/K SRAM 有效数据位宽/周期 |
| `c_cls` | metadata/classification 周期 |
| `c_const` | empty constant score 注入周期 |
| `cD0`, `cS0` | dense/sparse 每 work item 固定 setup 周期 |
| `c_acc` | score accumulator read-modify-write 周期 |
| `c_join` | row completion/join 周期 |
| `C_backend(row)` | center + Shiftmax + gated-K 实测周期 |
| `H_meta`, `H_idx` | bundle header、index/FIFO tag bits |

所有方程输出至少分为：理想 core service、无冲突流水、trace-replayed actual。只有最后一项可用于吞吐表。

## 4. C0 单路径周期模型

### 4.1 Full-score / true-density 前端

C0 使用一条固定宽度、有序路径。若没有 group occupancy trace，只能采用保守 full scan：

```text
c0_active(b) = c_cls + ceil(P_QK,b / Bbus)
             + cD0 + ceil(L_b / W0) + c_acc

c0_empty(b)  = c_cls + c_const

C0_front(row,b) = n_b(0) * c0_empty(b)
                + sum_{a=1..L_b} n_b(a) * c0_active(b)
```

如果实现 write-time active-group bitmap，才可使用：

```text
G_b,w(j) = number of nonzero W0-lane groups in bundle j
c0_active(j) = c_cls + c_payload(j) + cD0 + G_b,w(j) + c_acc
```

仅有 `A_b` 不能推出 `G_b,w`；合法边界是：

```text
ceil(A_b/W0) <= G_b,w <= min(A_b, ceil(L_b/W0))
```

因此不能把 `ceil(A_b/W0)` 当实测周期。

### 4.2 C0 Exact-Delta 路径

与 `docs/45` 一致，t0 完整计算，t1 在同一 core 上做 grouped delta：

```text
C0_delta(row) = N_token * c_t0_dense
              + n_delta(0) * c_reuse
              + sum_{u=1..32} n_delta(u)
                  * (c_delta0 + ceil(u/W0) + c_acc)
              + C_backend(row) + C_stall(row)
```

这里的 `ceil(u/W0)` 仍假设 changed-lane bitmap 可直接压缩发射；若硬件必须扫描 32-bit mask，则要增加 mask scan/priority-encode 周期。

### 4.3 C0 总周期

按 row 顺序累计：

```text
C0_frame = sum_rows (
    C0_front_or_delta(row)
  + c_join(row)
  + C_backend(row)
  + C_bank_conflict(row)
  + C_backpressure(row)
)
```

如果 frontend、core、backend 跨 row 流水，应由离散事件 trace 求 makespan；不能简单相加后再假定 100% overlap。

## 5. B1 双路径周期模型

### 5.1 Exact-Delta 主路由

冻结 bit-exact 路由：

```text
t0                  -> DENSE_FULL
t1, u == 0          -> REUSE
t1, 1 <= u <= theta -> SPARSE_DELTA
t1, u > theta       -> DENSE_RECOMPUTE
```

理想 service work：

```text
W_D(row,theta) = N_token * c_t0_dense
               + sum_{u=theta+1..32} n_delta(u)
                   * (cD0 + ceil(32/Wd) + c_acc)

W_S(row,theta) = sum_{u=1..theta} n_delta(u)
                   * (cS0 + ceil(u/Ws) + c_acc)

W_R(row)       = n_delta(0) * c_reuse
```

在 dense/sparse 真并行、没有 bank/FIFO/backend 冲突时，core lower bound 为：

```text
C_core_LB(row,theta) = max(W_D, W_S, W_R)
```

可审计总式为：

```text
B1_row(theta) = C_dispatch
              + max(W_D + Q_D, W_S + Q_S, W_R)
              + C_join + C_backend
              + C_bank_conflict + C_fifo_stall
              + C_dep_stall + C_wakeup

B1_frame(theta) = trace_makespan(all rows)
```

`Q_D/Q_S` 必须来自 queue trace，不能设为零后称为实际周期。由于 t0 总是进入 dense path，单看 t1 的 dense 比例会低估 dense core 负载。

### 5.2 True-density 可选 sparse fetch/score 路由

如果另做 event-index sparse score core，可定义：

```text
A_b == 0             -> CONST_INJECT
1 <= A_b <= kappa    -> SPARSE_EVENT_SCORE
A_b > kappa          -> DENSE_EVENT_SCORE
```

该 `kappa` 与 Delta `theta` 是两个参数。sparse core 必须逐 token/time 重建：

```text
n11 = popcount(Q AND K)
q1  = popcount(Q)
k1  = popcount(K)
n00 = D - q1 - k1 + n11
S64 = 64*n11 + n00
```

只要上述计数完整、padding valid 正确，使用 event indices 不改变 score。若只处理 `Q AND K` 而丢失 `q1/k1/n00`，则不是 bit-exact TTX。

## 6. Traffic 模型

### 6.1 两种 metadata 时序必须分开

**read-after-classify**：stratifier 先读完整 Q/K 才知道 `A_b/u`。这种实现可以减少 compute switching，但不能声称减少源 Q/K SRAM read traffic。

```text
T_QK_scan = N_b * P_QK,b
```

**write-time sidecar**：上游写 Q/K 时同时生成 empty/count/bitmap/index，后续先读 sidecar 再决定 payload。只有这种实现能合法旁路 empty payload 或选择压缩格式，但要计 sidecar 写读和容量。

### 6.2 C0 traffic

存在提前 metadata 时：

```text
T_C0_QK = (N_b - n_b(0)) * P_QK,b
T_C0_meta = N_b * H_meta
T_C0_score = N_valid_scores * W_score
T_C0_total = T_C0_QK + T_C0_meta + T_C0_score
           + T_state + T_acc + T_backend
```

`empty` 仍要产生 valid score entry，因此 `T_C0_score` 不随 empty ratio同比下降。

### 6.3 B1 traffic

对 true-density sparse sidecar，lane-pair有三种 active state `{01,10,11}`，payload code 至少需 2 bit。两种候选格式：

```text
T_bitmap(j) = L_b + 2*A_b(j)
T_index(j)  = A_b(j) * (ceil(log2(L_b)) + 2)
```

加上 header/tag 后：

```text
T_sparse(theta) = sum_{1<=A_b<=theta} n_b(A_b)
                  * (H_idx + chosen_encoded_bits(A_b))

T_dense(theta)  = D_b(theta) * (H_meta + P_QK,b)
T_empty          = n_b(0) * H_meta
```

不能直接取 `min(T_bitmap,T_index)` 当结果；必须综合两种 decoder、用同一 SRAM macro，并由地址 trace确认 transaction padding。现有 JSON只有 CDF count和总 active lanes，没有 `sum(A_b | A_b<=theta)`，因此尚不能精确计算各阈值的 sparse traffic。

Delta B1 还必须单列：previous Q/K state、update bitmap/index、S64 accumulator RMW、FIFO/tag、score completion。H67 Motion-XOR 与 Delta 共享 K-toggle读取时，只计一次共享 read；两条功能的逻辑操作仍分列。

### 6.4 合法的 value/projection traffic gate

```text
KZ_b == 1 -> gate*K 恒为 0
```

因此可以 gate 对应 K value read、late-scale、projection input和零输出写（若消费者支持 implicit zero）。不能据此删除 Q/K score、center 或 Shiftmax。是否省去写零必须在接口协议中定义 implicit-zero valid，不得默认。

## 7. 阈值 sweep `{2,4,8,12,16}`

### 7.1 H67 true-density bundle route

下表的 E/S/D 都以全部 bundle 为分母：`E=P(A=0)`，`S=P(1<=A<=kappa)`，`D=P(A>kappa)`。

| b | empty E | kappa | sparse S | dense D |
|---:|---:|---:|---:|---:|
| 1 | 73.8973% | 2 | 14.1190% | 11.9837% |
| 1 | 73.8973% | 4 | 19.0436% | 7.0591% |
| 1 | 73.8973% | 8 | 23.3826% | 2.7201% |
| 1 | 73.8973% | 12 | `[23.3826%,25.7655%]` | `[0.3372%,2.7201%]` |
| 1 | 73.8973% | 16 | 25.7655% | 0.3372% |
| 2 | 67.3191% | 2 | 14.1042% | 18.5766% |
| 2 | 67.3191% | 4 | 20.0941% | 12.5867% |
| 2 | 67.3191% | 8 | 25.4165% | 7.2644% |
| 2 | 67.3191% | 12 | `[25.4165%,30.0603%]` | `[2.6205%,7.2644%]` |
| 2 | 67.3191% | 16 | 30.0603% | 2.6205% |
| 4 | 60.9633% | 2 | 13.1275% | 25.9092% |
| 4 | 60.9633% | 4 | 19.6856% | 19.3511% |
| 4 | 60.9633% | 8 | 26.5065% | 12.5302% |
| 4 | 60.9633% | 12 | `[26.5065%,31.8692%]` | `[7.1675%,12.5302%]` |
| 4 | 60.9633% | 16 | 31.8692% | 7.1675% |
| 8 | 55.2559% | 2 | 11.8318% | 32.9123% |
| 8 | 55.2559% | 4 | 18.0470% | 26.6971% |
| 8 | 55.2559% | 8 | 25.6234% | 19.1207% |
| 8 | 55.2559% | 12 | `[25.6234%,32.7583%]` | `[11.9858%,19.1207%]` |
| 8 | 55.2559% | 16 | 32.7583% | 11.9858% |

`kappa=12` 只给严格上下界。表中 S/D 是 route item 数，不是 lane work、cycle、traffic 或 core utilization。

### 7.2 Exact-Delta B1 token route

| model | zero/reuse | theta=2 S/D | theta=4 S/D | theta=8 S/D | theta=12 S/D | theta=16 S/D |
|---|---:|---:|---:|---:|---:|---:|
| TTX | 72.649925% | 14.706432% / 12.643643% | 20.612697% / 6.737379% | 25.687818% / 1.662257% | bounded by theta 8/16 | 27.321462% / 0.028614% |
| H67 | 73.999091% | 14.537763% / 11.463147% | 20.009647% / 5.991262% | 24.789253% / 1.211657% | bounded by theta 8/16 | 25.980506% / 0.020403% |
| H68 | 74.295967% | 14.068539% / 11.635494% | 19.376670% / 6.327363% | 24.385115% / 1.318917% | bounded by theta 8/16 | 25.687417% / 0.016615% |

这里 S/D 只描述 t1 token route。B1 dense core还承担全部 t0，因此不能从 `D=0.02%` 推断 dense core空闲。相反，theta较大可能让 sparse FIFO承担几乎所有非零 t1，形成 sparse bottleneck。

### 7.3 Sweep 执行矩阵

cycle/traffic simulator必须运行：

```text
model in {TTX, H67, H68}
b in {1,2,4,8}
theta in {2,4,8,12,16}
architecture in {C0, B1}
FIFO depth in agreed sweep
```

对每点输出：mean/p50/p90/p99/max cycles、payload/meta/state/score traffic、dense/sparse issued work、lane utilization、FIFO high-water、stall breakdown、bit-exact pass。阈值选择依据是目标 PPA/cycle objective，不是 sparse route 比例最大。

## 8. Bit-exact 路由条件

| 条件 | 可合法 gate/bypass | 仍必须执行 |
|---|---|---|
| `A_b=0` | Q/K payload read、TX popcount、H67 motion branch；注入配置推导的 silent score | token valid、score entry、center、Shiftmax；不能删 denominator 项 |
| `KZ_b=1` | gated-K、late-scale、projection input；implicit-zero协议成立时可省零写 | TX score、center、Shiftmax |
| `MZ_b=1` | H67 Motion-XOR/popcount/add branch | 基础 TTX score与后端 |
| `u=0` 且 previous state valid | t1 Delta update，直接复用 `S64_0` | t1 score entry、H67 motion合并、center、Shiftmax |
| `1<=u<=theta` | 在 sparse core只处理 changed lanes | 所有 changed contribution、accumulator和舍入 |
| `u>theta` 或 sparse FIFO fallback | dense full recompute | 与 sparse结果同一固定点语义 |
| padding token | 不产生 score | valid mask必须阻止常数注入 |

额外条件：

- `A_b=0` 时 dyadic `alpha0=1/64` 的每 token/time `S64=D=32`；实现应从 descriptor推导，不把 32 写死为跨配置常量。
- previous Q/K、S64 accumulator和 row context必须同属一个 `{stage,block,window,head,token}`，reset/first-timestep时禁止 reuse。
- H67 Motion项在原始 score 域合并后统一 round-to-nearest-even；不能在 dense/sparse 路径分别提前舍入。
- H68 deployment不含 training-only matrix branch；路由器不能因 H68 名称实例化额外 attention路径。
- dense/sparse completion顺序可以不同，但 row-complete前 score集合、valid mask和定点值必须与 reference一致。
- queue overflow只允许 backpressure或 dense fallback；不得 drop、merge、近似截断 active lane。

## 9. 哪些比例只是上限

| 指标 | 分类 | 合法表述 |
|---|---|---|
| `activity_density` | 实测输入统计 | lane activity，不是 cycle/power reduction |
| true TTB `empty` | 实测 route coverage | 具备提前 metadata和常数注入时可 bypass payload/popcount；不是 Shiftmax skip |
| `K-zero` | 实测 route coverage | value/projection gate 上限，仍需接口/traffic验证 |
| `no K-motion` | 实测 route coverage | H67 motion branch gate 上限，不影响基础 TTX |
| `active_1_K` | 实测候选分流比例 | sparse route覆盖率，不是 skip或加速 |
| Delta `zero-update` | 实测 exact reuse覆盖率 | state有效时可合法复用 score |
| `97.x% t1 ideal lane skip` | compare upper bound | 未扣 mask/state/SRAM/FIFO/control |
| `48.x% full-T2 compare reduction` | TX compare upper bound | 不是 attention、frame或energy reduction |
| temporal-pair `50% transaction` | 条件上限 | 仅当地址trace证明baseline未合并；容量不减 |

## 10. 必须新增的 trace/profile 字段

本轮不修改 profiler；以下是后续实现协议。

### 10.1 Identity / ordering

```text
trace_version, model_id, checkpoint_id, sequence_id, sample_id
stage_id, block_id, window_id, head_id, row_context_id
bundle_size, bundle_id, spatial_token_base, valid_token_mask, timestep
```

### 10.2 Density / exactness

```text
L_b, q_active_count, k_active_count, union_active_count(A_b)
n11_count, n00_count, k_motion_count
per_token_update_count[0:b], update_bitmap[0:b][31:0]
empty, kzero, motion_zero, delta_reuse
active_group_bitmap_w4/w8
active_le2, active_le4, active_le8, active_le12, active_le16
active_lane_sum_le2/4/8/12/16
```

必须增加 `active_le12` 和 exact histogram `A_b=0..L_b`；仅有 CDF count不足以算 conditional sparse traffic。`active_lane_sum_leK` 用于计算路由后实际 index payload，不可用 bundle count替代。

### 10.3 Route / FIFO

```text
theta_delta, kappa_event, selected_route, selected_format
fallback_reason, enqueue_cycle, dequeue_cycle
dense_fifo_occ_before/after, sparse_fifo_occ_before/after
fifo_high_water, fifo_full, backpressure_cycles, starvation_cycles
route_sequence_no, completion_sequence_no
```

### 10.4 Cycle timestamps / stalls

```text
arrival_cycle, metadata_ready_cycle
payload_req_cycle, payload_rsp_cycle
service_start_cycle, service_end_cycle
score_write_cycle, row_complete_cycle
shiftmax_start/end_cycle, output_commit_cycle
stall_bank, stall_fifo, stall_backend, stall_dependency, stall_wakeup
dense_active_cycles, sparse_active_cycles, gated_cycles, lane_active_cycles
```

### 10.5 Traffic

```text
memory_id, bank_id, address, transaction_id
read_or_write, requested_bits, transferred_bits
payload_bits, metadata_bits, bitmap_bits, index_bits
state_bits, accumulator_bits, score_bits, padding_bits
bank_conflict, transaction_coalesced, implicit_zero
```

### 10.6 Bit-exact checker

```text
reference_S64_t0/t1, routed_S64_t0/t1
reference_motion, routed_motion
reference_score_q, routed_score_q
reference_gate_q, routed_gate_q
reference_output_q, routed_output_q
rounding_mode, saturation_flag, mismatch_stage
```

## 11. FIFO 与负载不均风险

1. **stage burst**：S1 高 empty、S0 高 density。全局平均可能让 sparse FIFO 在 S0 突发积压、在 S1 长期空闲。
2. **theta 增大导致单边拥塞**：H67 t1 在 theta=8 时 sparse/dense token为 `24.79%/1.21%`；theta=16 为 `25.98%/0.02%`。这不是平衡，必须把全部 t0 dense work和两核service rate一起计算。
3. **bundle size权衡**：H67 b=1、kappa=8只有 `2.72%` dense bundle；b=8则为 `19.12%`，但 b=8 empty更低、payload更大、mixed bundle拆分成本更高。
4. **head/window同步 burst**：同一 stage 的多个 head可能在相近周期产生同一路由，平均 arrival rate不能保证FIFO不满。
5. **共享 SRAM端口**：dense full read和sparse state/index read可能争用同一 bank，使理论双核并行退化为串行。
6. **共享 backend**：center/Shiftmax必须等待完整 row；一个慢 sparse尾项会造成 head-of-line blocking。
7. **fallback反馈**：sparse FIFO满时回退dense会加重dense拥塞，必须记录 fallback chain并设置有界仲裁。
8. **wakeup成本**：长 empty run支持clock gate，但短脉冲可能被ICG/wakeup开销抵消；需要 consecutive-empty和active-burst histogram。
9. **RLE风险**：mean changed run约3.7--4.0不等于RLE必胜。可变长decode、run跨bundle和随机bank访问必须单列。
10. **容量不是性能**：加深FIFO可以减少丢压但增加面积、leakage、读写动态功耗；深度必须通过p99 occupancy sweep确定。

必须报告每个 FIFO 的 occupancy CDF、max、full events、producer/consumer idle、starvation、fallback和对 frame makespan的贡献。

## 12. 同工艺综合比较协议

### 12.1 公平性约束

C0 与 B1 必须使用完全相同的：

- 工艺库、PVT、V/F target、SDC、IO delay、clock uncertainty；
- `T=2`、`D=32`、score/gate/accumulator位宽和舍入/饱和规则；
- SRAM compiler/macro family、ECC、bank padding和memory accounting方法；
- workload trace、bundle size、threshold、FIFO depth和valid mask；
- synthesis/physical effort、wire-load或placement假设；
- SAIF/VCD窗口与clock-gating enable条件。

先综合 leaf，再综合包含 SRAM wrapper、FIFO、stratifier、join和shared backend的 subsystem。只比较leaf会遗漏B1主要控制成本。

### 12.2 Leaf 表模板

| item | C0 grouped core | B1 dense core | B1 sparse core | B1 stratifier/FIFO/join | evidence |
|---|---:|---:|---:|---:|---|
| RTL commit/hash | TBD | TBD | TBD | TBD | source |
| target clock | TBD | same | same | same | SDC |
| achieved slack | TBD | TBD | TBD | TBD | timing report |
| combinational area | TBD | TBD | TBD | TBD | synthesis |
| sequential area | TBD | TBD | TBD | TBD | synthesis |
| memory bits/macros | TBD | TBD | TBD | TBD | compiler/report |
| leakage | TBD | TBD | TBD | TBD | same corner |
| dynamic logic | TBD | TBD | TBD | TBD | same SAIF |
| clock power | TBD | TBD | TBD | TBD | same SAIF |
| critical path | TBD | TBD | TBD | TBD | timing report |
| bit-exact tests | TBD | TBD | TBD | TBD | regression |

### 12.3 Subsystem / workload 表模板

| metric | C0 | B1 theta=2 | B1 theta=4 | B1 theta=8 | B1 theta=12 | B1 theta=16 |
|---|---:|---:|---:|---:|---:|---:|
| bundle size / FIFO depth | TBD | TBD | TBD | TBD | TBD | TBD |
| total logic area | TBD | TBD | TBD | TBD | TBD | TBD |
| total SRAM area | TBD | TBD | TBD | TBD | TBD | TBD |
| achieved frequency / WNS | TBD | TBD | TBD | TBD | TBD | TBD |
| leakage | TBD | TBD | TBD | TBD | TBD | TBD |
| dynamic: compute | TBD | TBD | TBD | TBD | TBD | TBD |
| dynamic: memory | TBD | TBD | TBD | TBD | TBD | TBD |
| dynamic: clock/control | TBD | TBD | TBD | TBD | TBD | TBD |
| cycles/frame mean / p99 | TBD | TBD | TBD | TBD | TBD | TBD |
| dense/sparse utilization | N/A | TBD | TBD | TBD | TBD | TBD |
| FIFO max/full/stall | TBD | TBD | TBD | TBD | TBD | TBD |
| Q/K payload traffic | TBD | TBD | TBD | TBD | TBD | TBD |
| metadata/index/state traffic | TBD | TBD | TBD | TBD | TBD | TBD |
| score/backend traffic | TBD | TBD | TBD | TBD | TBD | TBD |
| energy/frame | TBD | TBD | TBD | TBD | TBD | TBD |
| bit-exact result | TBD | TBD | TBD | TBD | TBD | TBD |

所有数值必须附 report/trace路径。百分比差异只在绝对值、单位、corner和统计窗口完整后计算；表中不得填入 Bishop 或其他项目的PPA。

## 13. Go / No-Go

B1 相对 C0 晋级必须同时满足：

1. `theta=12` 和 exact hist/conditional active-lane sums补齐；
2. trace replay在所有代表序列上bit-exact，queue overflow只触发backpressure或exact fallback；
3. mean和p99 frame cycles均达到项目目标，且不是只改善core-local service；
4. 同工艺、同SRAM、同约束的subsystem综合显示目标PPA指标优于C0；
5. 把metadata/state/FIFO/clock/backend与idle第二core成本全部计入；
6. H67 Motion-XOR与Delta共享项没有重复计数。

在此之前的合法结论是：**真实 TTB 分布支持常量注入、value/projection gating和Exact-Delta reuse的进一步实现评估；B1异构双路径是否优于C0仍待trace与同工艺综合。**

## 14. Cycle-v2 profile 实现与队列（2026-07-13）

<!-- TTB_DELTA_CYCLE_V2_PROTOCOL_20260713 -->

`profile_nts11_hardware_p0.py` 与 attention collector 已新增本协议10.2要求的关键原始量：

- Delta `u=0..32` exact histogram；
- `theta=2/4/8/12/16` 的 sparse token count 和 conditional changed-lane sum；
- bundle1/2/4/8 的 `A_b=0..L_b` exact histogram；
- `kappa=2/4/8/12/16/32` 的 active bundle count 和 conditional active-lane sum。

这些字段仅在挂载 `_h9_profile_collector` 时计算，不进入训练或部署 attention datapath。当前90项
attention、加载与周期回放单元测试已通过，其中包含 theta12、histogram 和 lane-sum 的确定性样例。自动 runner
`run_ttb_cycle_profile_v2_after_round3.py` 在软件 full30 队列结束后对 TTX/H67/H68 各重放100
样本，写入独立 v2 目录并检查 ATLIF105、attention12、overlay210/210、missing0/unexpected0。
下游 Delta/deploy watcher 等待 v2 完成标记，因此不会与 profile 并发。

该补丁关闭了第13节门槛1中的“theta12/exact hist/conditional sum”采集缺口；第15节又加入
stage/block有序trace与有限FIFO回放。两者仍须在v2真实artifact生成后才视为证据完成；SRAM
address/bank transaction、共享backend credit与端到端stage依赖仍是B1对C0作定量裁决的剩余P0。

此外，H69 固定 x8 与 H70 event-selective shift 会扩大 score 动态范围。collector 已加入量化前
`score_clip_low/high/total`，边界 `-2/+2` 本身不计 clip。最终 watcher 在全候选 deploy 前调用
`run_temperature_score_clip_audit.py`，对 H69/H70 各自 valid825 rank-1 checkpoint 各 profile20，
据此决定当前 Q7 range 是否可冻结；在该表生成前不得仅凭 H67/H68 的 RTL-exact 结果外推温度线。
## 15. Ordered trace与有限FIFO自动回放（2026-07-13）

<!-- TTB_ORDERED_TRACE_FINITE_FIFO_REPLAY_20260713 -->

软件profiler现支持`--ordered-trace`，按实际forward顺序压缩保存Delta逐token更新lane数，以及
TTB4/8的active、K-active和K-motion计数。编码为int16 little-endian加zlib/base64，shape保留
`[batch_windows,heads,tokens-or-groups]`，避免只留下全局histogram后无法恢复burst和stage差异。

`entrypoints/replay_ttb_dual_path_cycles.py`实现有限资源C0/B1前端代理：metadata每拍最多接收一个
bundle；empty走精确constant/reuse；`0<active<=kappa`进入Ps-lane sparse队列，其余进入Pd-lane
dense队列；FIFO满时对metadata前端施加真实backpressure。先扫kappa与Ps的解析下界，再对每种
route前三名执行FIFO4/8/16逐周期回放，按attention调用边界排空并报告stage、stall、busy和最大
occupancy。

同一解析sweep还输出三种traffic口径：E0 dense Q/K为每lane两bit；B1 bitmap格式为固定union
bitmap加每active lane的Q/K两bit；B1 index格式为count header加`ceil(log2(capacity))`坐标与Q/K
payload。所有路径都计相同的count/tag route metadata，再报告bitmap/index相对E0的bit reduction。
回放同时给出逐bundle保守64-bit transaction：每个descriptor固定占一个metadata word，三种payload
均在bundle边界独立向上取整到64 bit，不允许跨bundle理想拼接。该数能暴露短稀疏payload的padding
损失，但仍未模拟transaction coalescing、地址到bank映射、bank conflict或端口grant，因此不等于
真实SRAM transaction或energy。

共享Shiftmax/backend另以每window/head row=`1/4/8/16` cycles做工作量下界敏感性，计算
`max(metadata,dense,sparse,backend)`，用于识别backend-bound区间。该表不模拟bundle完成后的row
join顺序、backend FIFO或credit，因此只作为下界；finite FIFO cycle仍严格标记为front-end replay。

该回放只签核route/front-end可行性，不签核整核性能。共享Shiftmax/backend credit、SRAM
bank映射/conflict、port grant、transaction coalescing、projection、decoder和NoC尚未进入模型；在这些项加入前，输出固定标记为
`row-kernel proxy`，不得换算端到端FPS或energy/frame。当前90项软件/链路/周期单测通过，真实
TTX/H67/H68 profile100 replay已排在H80软件队列之后。


## H69/H70 deployment score-clipping profile20 自动结果

<!-- H69_H70_TEMPERATURE_SCORE_CLIP_PROFILE20_20260713 -->
- artifact: `neuron_experiments/H9_bipolar_self_attention/results/temperature_score_clip_profile20_20260713.md`

| candidate | best epoch | score elements | clip low | clip high | clip ratio |
|---|---:|---:|---:|---:|---:|
| H69 | 19 | 21772800 | 0 | 0 | 0.000000% |
| H70 | 19 | 21772800 | 0 | 0 | 0.000000% |

裁剪按量化前 score 严格小于 -2 或大于 2 计数，边界值不计入；该表用于判断固定/动态左移是否需要扩大 score 位宽，不替代 valid825 精度。
