# Ordered Trace Schema、Native-M 公平基线与 ET3 负结果

## 1. 阶段目标

第三轮 DATE 复审已给予：

> ET3 小规模、单 outstanding、输入唯一性由上游保证、下游最终恢复 ready 条件下的协议生存性阶段签核。

但 DATE Recommendation 仍是 Reject，创新性仍为 2/5。下一决定点必须来自同一 ordered trace 下普通 per-destination native-m queue 与 ET3 的公平比较。

本阶段只完成 GPU 空闲前可做的 CPU/RTL 前置工作：

1. 冻结 ordered term trace schema。
2. 实现普通 native-m queue RTL 基线。
3. 实现 native-m queue、单 context ET3、理想双 context ET3 的 CPU replay。
4. 用 synthetic trace 验证协议与模型，不形成 workload 性能主张。

## 2. Ordered trace schema

新增：

- `scripts/et3_ordered_trace_replay.py`
- `scripts/make_et3_synthetic_trace_fixture.py`
- `tests/test_et3_ordered_trace_replay.py`

trace 由 `manifest.json + ordered_items.npz` 组成。

### 2.1 Manifest 必要字段

```text
schema = et3_ordered_term_trace_v1
evidence_level = synthetic | pre_g0 | post_g0
payload_file
payload_sha256
config_sha256
checkpoint_sha256
cohort_sha256
resolution
groups[]
```

每个 group 记录：

```text
tag
sample/stage/block/window/head
empty
ordered_item_sha256
```

### 2.2 NPZ 必要数组

```text
group_offsets
group_tags
item_mode_multiset
item_gate_code
item_lane_id
item_multiplicity
item_destination
```

验证器强制检查：

1. schema 与 evidence level。
2. payload SHA256。
3. group offset 单调性与数组长度。
4. group tag、empty 标志与 ordered item hash。
5. gate 非零、multiplicity 在 1..5。
6. Motion SET 必须 `multiplicity=1`。
7. 同 group 上游 `{mode,gate,lane,m,destination}` 唯一性。

任何 hash、合同或 duplicate 错误都会拒绝 replay。

## 3. Native-m RTL 公平基线

新增：

- `rtl_et3/et3_native_m_queue_baseline.sv`
- `tb_et3/tb_et3_native_m_queue_baseline.sv`

公平边界：

- 与 ET3 使用同一个 `et3_native_multiset_executor`。
- 相同 source item 接口与 group epoch。
- 相同原生 multiplicity，不把 Local5 展开成错误的 SET。
- 每个 destination item 是一个独立 term。
- 不做跨 destination aggregation/product reuse。
- 深度为 2 的 FIFO 在测试中产生真实输入反压。

Local5 小例：

```text
source item=6
native command=6
product compute=6
EXPLODE work=15
fallback=0
accumulator mismatch=0
```

同一 Local5 item 集在 ET3 小例中为：

```text
destination command=6
product compute=4
EXPLODE work=15
```

这只证明 ET3 的 product reuse 与 native-m baseline 的差异，不代表周期或能耗收益。

## 4. CPU replay 模型

每个 frozen group 同时计算：

### 4.1 Native-m queue

```text
每 item 一次 product compute
每 item 一次 weight read
每 item 一次 destination write
```

### 4.2 单 context ET3

```text
collect 所有 item
+ 按有限 KEY_CAP/SEG_DEPTH/FALLBACK_DEPTH 发射
+ partial drain transition
+ final commit
```

### 4.3 理想双 context ET3

按 chunk 对 collect 与 emit 取 `max()`，表示一个 context 收集时另一个 context 发射。该结果是 overlap 上界，不是现有 RTL cycle。

### 4.4 关键指标

- online term/product compute
- fallback
- partial drain
- peak directory/fallback occupancy
- native/ET3 weight read
- mean/p95/p99/max cycles
- online product reuse retention

其中：

```text
retention =
  (native_products - online_ET3_products)
  / (native_products - unlimited_ideal_terms)
```

## 5. Synthetic 协议负结果

结果：

- `results/et3_synthetic_protocol_trace_20260728`
- `results/et3_synthetic_protocol_replay_20260728/report.md`
- `results/et3_synthetic_protocol_replay_20260728/report.json`

配置：

```text
KEY_CAP=2
SEG_DEPTH=2
FALLBACK_DEPTH=1
weight_read_latency=2
```

结果：

| 指标 | 数值 |
|---|---:|
| item | 11 |
| unlimited ideal term | 6 |
| online ET3 term | 8 |
| fallback item | 2 |
| partial drain | 2 |
| native product/weight read | 11 / 11 |
| ET3 product/weight read | 8 / 8 |
| online reuse retention | 60.0000% |
| native-m queue 模型周期 | 25 |
| 单 context ET3 模型周期 | 35 |
| 理想双 context ET3 模型周期 | 24 |

这是 deliberately-small synthetic 协议例，`performance_claim_allowed=False`。不能据此判断 Local5 workload 是否淘汰 ET3。

但它揭示了必须正视的架构事实：

1. product compute/weight read 减少不保证周期减少。
2. 单 context collect-then-emit 可能比流式 native-m queue 更慢。
3. 只有 collect/emit overlap 后，聚合收益才可能覆盖相序开销。
4. 旧的 DCTF-2C/双 context 机制可以作为 ET3 内部 overlap 实现候选重新评估，但不能单独列新贡献。
5. 真实 trace 若 retention 低于 70%，或 dual-context 后净改善低于 15%，ET3 应淘汰。

## 6. 当前验证

统一入口：

```bash
sim_et3/run_et3_native_slice_checks.sh
```

当前包括：

```text
Python evidence/replay tests: 9/9 PASS
Icarus ET3: PASS
Icarus native-m baseline: PASS
Verilator ET3 lint: PASS
Verilator native-m lint: PASS
Verilator ET3 + SVA: PASS
Yosys ET3 check: PASS
Yosys native-m check: PASS
```

Yosys 仍将小型目录、product 与 accumulator 展开为寄存器；这些是综合可读性结果，不是 SRAM macro 或目标工艺 PPA。

## 7. 下一步

1. 给现有 Local5 profiler 增加可选的 post-G0 ordered NPZ 导出。
2. 导出时保留原始 destination 顺序、empty group 和 cohort/hash。
3. 先用少量 full-resolution group 跑 schema/唯一性审计，再扩大样本。
4. 用冻结 trace 扫 `KEY_CAP × SEG_DEPTH × FALLBACK_DEPTH × SRAM latency`。
5. 只有通过 70% retention 与 15% 净改善门槛，才实现 ET3 双 context RTL 并进入 DC/STA/SAIF。
