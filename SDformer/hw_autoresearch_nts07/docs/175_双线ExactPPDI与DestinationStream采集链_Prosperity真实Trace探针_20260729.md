# 双线 Exact PPDI / Destination Stream 采集链与 Prosperity 真实 Trace 探针

> **2026-07-29 二次勘误：** 初版 Motion PPDI 采集按空间 token ID 取奇偶，
> 在 `T=2,N=81` 时会把展平 ID `0/81` 错判为同奇偶；同时未排除
> `gate=0`，ordered count trace 也可能在 `int16` 溢出。三项均已修复并增加
> 定向测试。本文后续出现的 `59/59` 已更新为 `62/62`，真实 profile100 数值
> 仍须重跑，旧 profile 不含这些修正。

## 1. 本轮结论

本轮完成的是“下一次真实 profile 必须产出什么”的采集链，不宣称新收益已经获得。

双线新增：

- Motion：逐 `{gate,lane}` 的 even/odd destination 计数，得到 exact PPDI delivery；
- Local5：unique term、destination delivery、PPDI、destination delta 和 escape 的守恒统计；
- Prosperity：真实 Motion sample0/window0 K support 已送入官方 CPU simulator；
- Phi：仍无公开官方 simulator，继续保持 clean-room 边界。

这轮的架构意义是把两个此前依赖假设的候选变成可证伪机制：

1. PPDI 只有 profile100 exact parity work 通过才保留；
2. ISHD 只有真实 delta/escape 和 metadata 通过才保留。

---

## 2. Motion exact PPDI 统计

### 2.1 旧统计为什么不够

旧 profile100 有：

```text
projection_gate_multicast_delivery_m1
projection_gate_multicast_delivery_m2
projection_gate_multicast_delivery_m4
```

其中 M2/M4 只限制每拍 destination 数，不限制偶/奇物理端口。PPDI 每拍只能带：

```text
最多一个 even destination
最多一个 odd destination
```

所以：

```text
unconstrained M2 <= exact PPDI <= scalar M1
```

sample0/window0 的 `30.27%` 不能代表 profile100。

### 2.2 新实现

在 gate projection profile 内，原有：

```text
class_channel_counts[row, gate, lane]
```

扩展为：

```text
parity_counts[row, parity, gate, lane]
```

exact PPDI 命令数：

```text
sum_row_gate_lane max(even_count, odd_count)
```

新增字段：

```text
projection_gate_ppdi_delivery_exact
projection_gate_group_ppdi_delivery_g{1,2,4,8,16}
projection_gate_ppdi_delivery_exact_ordered_trace
projection_gate_group_ppdi_delivery_g*_ordered_trace
```

聚合器强制检查：

```text
M2 <= PPDI <= M1
```

### 2.3 验证

新增定向测试构造同一 gate/lane 的两个偶 destination：

```text
M1 = 2
无约束 M2 = 1
exact PPDI = 2
```

说明统计没有把普通双发错误当作偶/奇双端口。

完整 `test_bsa_attention`：

```text
62/62 PASS
```

Motion 的真实 profile100 PPDI 数值仍需重新运行 inference/profile 后获得；在此之前模型继续禁用 PPDI。

---

## 3. Local5 exact destination stream

### 3.1 新守恒边界

Local5 明确区分：

```text
unique product term = mfep_multicast_terms
destination delivery = mfep_scalar_delivery
```

`mfep_scalar_delivery` 必须等于已有：

```text
destination_gate_lane_groups
```

不相等时 profiler 立即失败。

### 3.2 新统计

对真实 `(term_key, destination_id)` 组合键排序，在 term 内得到单调 destination：

```text
term header
destination continuation delta
```

新增：

```text
mfep_scalar_delivery
mfep_ppdi_delivery_exact
mfep_ppdi_command_reduction
mfep_destination_continuations
mfep_destination_delta_histogram
mfep_destination_delta_escape_b4/b6/b10
mfep_destination_delta_escape_ratio_b4/b6/b10
```

escape 定义：

```text
delta > 2^bits - 1
```

### 3.3 验证

单测覆盖：

- 多 term；
- even-only、odd-heavy parity；
- delta `2/3/5/20`；
- 4-bit escape；
- 6-bit 不 escape。

结果：

```text
2/2 Local5 sink/stream tests PASS
```

现有 pre-G0 JSON 没有逐 term destination，因此不能离线补算这些字段；必须由下一次 post-G0/fullres profile 产生。

---

## 4. Prosperity 官方真实 Motion K-support 探针

输入来自：

```text
results/h67_real_bit_trace_20260717/
```

Q/K 和 gate 均来自真实 Motion 网络 sample0/window0。将每个 stage 的多头 K 拼接为：

```text
[batch=1, time=2, sequence=81, input_dim=heads×32]
```

再调用官方 `Simulator.run_fc` 的 product-sparsity 和 bit-sparsity 路径。

| Stage | K density | gate code 数 | product cycles | bit-sparse cycles | product/bit 周期比 |
|---|---:|---:|---:|---:|---:|
| S0 | 0.01151 | 2 | 174 | 193 | 1.109× |
| S1 | 0.00000 | 1 | 498 | 348 | 0.699× |
| S2 | 0.00555 | 3 | 1,716 | 1,333 | 0.777× |
| S3 | 0.02172 | 3 | 10,212 | 16,230 | 1.589× |

解释：

- Prosperity product-sparsity 不是“只要更稀疏就一定更快”；
- S1 全零、S2 极稀疏时，在线 preprocess 成本使官方 product 路径反而更慢；
- S0 小幅受益，S3 受益明显，说明 stage 间 pattern 价值不均匀；
- 这支持按 stage 冻结静态 schedule 或 bypass，而不是全 encoder 统一开启在线 matcher。

严格边界：

- 输入只是 binary K support；
- 没有表达 Q1.7 gate；
- 因此不是 gated-K 投影等价基线；
- 只有一个 sample/window，不能报 profile100 分布。

---

## 5. 对双线架构的直接指导

### 5.1 Motion

优先机制仍是：

> **静态 temporal anchor + exact residual + stage-conditioned bypass。**

原因：

- ordered temporal delta 已有真实证据；
- 官方 Prosperity 真实 K-support 探针显示在线 product discovery 在 S1/S2 可能负收益；
- exact PPDI 采集已就绪，可作为 projection 后端条件增强，不作为主创新。

下一次 profile 后：

1. 计算每 stage PPDI reduction；
2. 计算 TARE W2/W4/W8 raw 与 lane 归一；
3. 对 online matcher 做 stage-wise bypass；
4. 若 PPDI 小于 15% 或 PPA 不赢，删除 PPDI。

### 5.2 Local5

优先机制仍是：

> **固定 self-anchor + directional exact residual + stencil schedule。**

原因：

- 五点拓扑是算法定义，不需要在线识别；
- 当前 1.583× 仍只有 histogram 下界；
- metadata 为 11.96%，ISHD 未过门槛；
- 新 delta/escape 采集将直接决定 compact descriptor 是否值得做 RTL。

下一次 profile 后：

1. 检查 destination 守恒；
2. 检查 delta6 escape；
3. 检查 exact PPDI；
4. 生成 ordered FIFO trace；
5. 只有 metadata ≤10% 且 escape/fallback 可控才实现 ISHD。

---

## 6. 当前创新点分级

| 机制 | Motion | Local5 | 当前定位 |
|---|---|---|---|
| 静态拓扑锚定 exact residual | temporal pair | self + four directions | 主架构候选 |
| stage-conditioned online-discovery bypass | 有真实 Prosperity 微 trace 动机 | 固定拓扑天然 bypass | C1 的调度扩展 |
| exact PPDI | 采集链已就绪 | 采集链已就绪 | 条件后端优化 |
| ISHD delta/escape | 待 ordered destination | 待 post-G0 destination | 条件编码优化 |
| Prosperity/Phi pattern | 官方 K-support 微 trace；Phi-like | 等待真实矩阵 | 竞争基线，不是本工作贡献 |

当前仍不能把 PPDI、ISHD、TTB/STT 分别列成独立 DATE 贡献。论文主张应收敛为一条：

> 利用事件光流网络固定时空拓扑，把运行时稀疏关系发现改写为可提前编排的 exact anchor/residual 数据流，并按 stage 关闭负收益的在线发现和不规则执行。

---

## 7. 复现

```bash
cd /root/private_data/work/sdformer_codex/SDformer/hw_autoresearch_nts07

/opt/conda/envs/sdformerflow/bin/python -m unittest \
  tests.test_local5_ordered_trace_sink -v

cd /root/private_data/work/sdformer_codex/SDformer
/opt/conda/envs/sdformerflow/bin/python -m unittest \
  neuron_experiments.H9_bipolar_self_attention.tests.test_bsa_attention -v

cd /root/private_data/work/sdformer_codex/SDformer/hw_autoresearch_nts07
/opt/conda/envs/sdformerflow/bin/python \
  scripts/run_prosperity_motion_bittrace_probe.py
```

产物：

- `results/prosperity_motion_bittrace_probe_20260729/report.json`
- `results/prosperity_motion_bittrace_probe_20260729/report.md`
