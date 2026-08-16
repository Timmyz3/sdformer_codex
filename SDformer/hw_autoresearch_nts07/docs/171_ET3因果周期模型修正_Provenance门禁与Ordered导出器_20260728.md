# ET3 因果周期模型修正、Provenance 门禁与 Ordered 导出器

## 1. 第四轮 DATE 评审

对 `docs/170`、native-m baseline 与首版 CPU replay 的评分：

| 维度 | 分数 |
|---|---:|
| Recommendation | Reject |
| 完整度 | 3/5 |
| 创新性 | 2/5 |
| 证据完整度 | 2/5 |
| 目标潜力 | 3/5 |

评审确认：

- `60%` synthetic reuse retention 算术正确。
- native-m RTL 的算术边界基本公平，未发现合法协议 P0。
- `25/35` 是旧公式内部可复现值，但不是 RTL cycle。
- `24` cycles 的双 context 数字违反因果关系，必须撤销。

## 2. 旧双 context 公式为什么错误

旧公式对同一个 chunk 计算：

```text
max(collect(chunk_i), emit(chunk_i))
```

这等价于 chunk 尚未收集完成就开始发射自身，违反：

```text
collect(chunk_i) 完成
    才能
emit(chunk_i)
```

正确的双 context overlap 只能发生在：

```text
emit(chunk_i) || collect(chunk_i+1)
```

并必须支付首块填充、末块排空以及 context 被前两块占用时的等待。

## 3. 新因果递推

CPU replay 现在对每个 chunk 记录：

```text
(collect_cycles, emit_cycles, transition_cycles)
```

然后维护：

```text
context_free[2]
collect_engine_free
emit_engine_free
```

对 chunk `i`：

```text
ctx = i mod 2
collect_start = max(collect_engine_free, context_free[ctx])
collect_end   = collect_start + collect + transition
emit_start    = max(collect_end, emit_engine_free)
emit_end      = emit_start + emit
context_free[ctx] = emit_end
```

最终周期为最后一个 `emit_end + final_commit`。该递推显式包含 fill/drain，不能发生 same-chunk overlap。

## 4. Latency 与 II 分离

旧 native 模型使用：

```text
items * weight_read_latency
```

它把 latency 错当作不可流水化 issue interval，天然抬高 native 基线。

新配置分离：

```text
weight_read_latency
destination_issue_interval
```

在默认 `latency=2, II=1` 下：

```text
native cycles =
items * II + (latency - 1) + final_commit
```

ET3 emit 对每个 chunk 同样只支付一次 pipeline fill，不再按 term 数重复乘 latency。这个模型仍是参数化因果调度模型，不是 RTL 实测 cycle；下一阶段必须用 microbenchmark 标定 latency、II 和 backpressure。

## 5. 修正后的 synthetic 结果

原 `24` cycles 已删除。重新生成：

- `results/et3_synthetic_protocol_replay_20260728/report.md`
- `results/et3_synthetic_protocol_replay_20260728/report.json`

结果：

| 指标 | 修正前 | 修正后 |
|---|---:|---:|
| reuse retention | 60% | 60% |
| native-m 参数化周期 | 25 | 16 |
| ET3 single-context 参数化周期 | 35 | 31 |
| ET3 dual-context | 24（错误） | 29（因果递推） |

结论更负面：

```text
native-m = 16
causal dual-context ET3 = 29
single-context ET3 = 31
```

它仍是 synthetic，不代表 Local5 workload；但已经证明不能用 product/weight-read 数量下降推导周期加速。

## 6. Post-G0 provenance 强门禁

旧版只检查：

```text
evidence_level == post_g0
```

新版本在加载 trace 时强制：

1. `config_sha256` 为 64 位十六进制。
2. `checkpoint_sha256` 为 64 位十六进制。
3. `cohort_sha256` 为 64 位十六进制。
4. `resolution.full_resolution == true`。
5. `sampling.groups_per_block_sample > 0`。
6. payload hash、group item hash 与 schema 继续全部通过。

任一条件不满足时直接拒绝加载，不会输出 `performance_claim_allowed=True`。

## 7. Local5 ordered trace 导出器

在现有 profiler 中新增可选参数：

```bash
--ordered-groups-per-block-sample N
--ordered-evidence-level pre_g0|post_g0
```

实现位置：

- `scripts/profile_local5_hardware_features.py`
- `tests/test_local5_ordered_trace_sink.py`

导出器直接复用 profiler 已计算的：

```text
first_stack
gate_code
multiplicity_stack
```

对每个 sample、每个 block，在所有 `(window,head)` 中均匀抽取 N 个 group，并保持 group 内原始：

```text
destination -> candidate -> lane
```

顺序。空 group 会被显式保留。

输出：

```text
ordered_term_manifest.json
ordered_term_items.npz
```

CPU roundtrip 单元测试已覆盖：

- 两个均匀抽样 group。
- gate/lane/multiplicity/destination 顺序。
- payload/ordered item hash。
- post-G0 provenance。
- full-resolution 标志。

当前 GPU 利用率约 96%，仍由 full-resolution 训练占用，因此本阶段没有运行网络导出，不能声称已经有 post-G0 trace。

## 8. Native-m baseline 进一步公平化

native FIFO 在满且同拍 pop 时现在允许 push，消除了原版不必要 bubble：

```text
space_available = count < depth || queue_pop
```

baseline 也接入与 ET3 同一套动态 SVA，并增加 Verilator functional+assert 回归。

统一回归现为：

```text
Python tests: 11/11 PASS
Torch ordered sink test: 1/1 PASS
Icarus ET3/native-m: PASS
Verilator lint ET3/native-m: PASS
Verilator functional+SVA ET3/native-m: PASS
Yosys check ET3/native-m: PASS
```

## 9. 当前决策

创新性仍为 2/5，Motion 仍为唯一硬件主线。

Local5/ET3 下一步只能做真实 kill test：

1. GPU 空闲后导出冻结 checkpoint/cohort 的 post-G0 full-resolution trace。
2. 用真实 trace 扫有限容量与存储 bit budget。
3. 用 RTL microbenchmark 标定 latency/II/backpressure。
4. 同时满足 `retention >= 70%` 和相对 native-m 净周期改善 `>= 15%` 才继续。
5. 否则淘汰 ET3，保留 native-m 或回到 Motion 主线。
