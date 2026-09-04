# M103 correction/fallback 主项机会前置独立打铁 r1

日期：2026-08-24  
范围：只读审计 M40/M43/M72/M78/M83/M88 和实际二进制；独立重建 cap11 heldout correction/fallback multiset。未修改生产文件，未运行 producer、VCS 或 Synopsys。

## 结论

评分 **80/100**，`P0=1 / P1=4 / P2=3`。

**GO：推进 phase-local weight-stationary correction batching / full-vector hold 方向，并先补最小 ordered semantic trace 与 scheduled bank trace。**

**NO-GO：当前把 last-vector cache、run/broadcast、bank multicast 的机会率写成 cycle reduction、physical speedup、system speedup 或 headline。** 原始证据能精确重建每个 phase 的 correction/fallback 事件多重集合、source、block 和 add/sub 方向，但没有冻结硬件 issue order、destination tag、accumulator bank/address、依赖或 queue backpressure。

最重要的发现有两个：

1. 8,640 个 phase 中，每个 phase 的全部 128 个 `(source,block)` weight identity 都至少使用一次，所以按 key “整项跳读”机会为 **0**；M88 也已经把完整 12,288 B weight phase image 放在双 buffer 中。
2. 重复使用非常强：每个 phase 只有 128 个 weight key，却总计产生 188,148,490 个 correction/fallback destination event。每个 phase-weight key 平均服务 170.13 个 destination，p50/p95/p99 为 152/344/409，最大 695。真正的机会是将同 key destination 成批调度，在三拍装入一次 96 B weight vector 后保持并连续复用。

这是一项强硬件创新候选，但不是已经实现的性能结果。

## 现有证据究竟包含什么

| evidence | 保留内容 | 缺失内容 |
|---|---|---|
| M40 packed-source binaries | sample/operator；C-order timestep/y/x activity；16-bit partition/source mask 可导出 | correction 选择、block、硬件 tag/bank、issue order |
| M72 JSON | 每 operator/partition 的 16 个 center；aggregate heldout correction count | event row/source/block/tag/order |
| M78 analyzer/JSON | cap11 eligibility、correction/fallback/PWP aggregate、width use、service accounting | `phase_metrics` 输入是 `Counter`，row order 已丢失；无事件 trace |
| M83 records/offsets | 1,728 phase 的 catalog PWP，pattern-major/block-major | heldout use、correction source、destination tag/order 全无 |
| M88 JSON | per-sample bounded phase duration、preparation、buffer ledger | correction/PWP issue sequence、cache、accumulator bank/conflict |
| M85/M99 actual-record replay | catalog 221,184 entries 各读取一次 | 不是 58,969,374 次 ordered PWP use，更不是 correction trace |

因此：底层 M40 mask 顺序仍在，可以提出一个 raster expansion；但没有任何冻结 artifact 证明最终硬件会采用哪种 block/source/PWP/correction 排序。M103 的 order-independent 分组统计是 exact；cache hit/run/multicast 只能给 fail-closed 区间。

## 独立重建结果

审计脚本不 import 生产 analyzer，直接解包 20 个 heldout M40 binaries，按 M72 frozen centers、M78 tie-break、beneficial rule 和唯一 cap11 outlier 独立展开。

- reconstructed partition vectors：25,920,000；
- correction/fallback events：**188,148,490**，与 M78 `correction_ops_all_blocks` 完全一致；
- phase-weight groups `(sample,op,partition,source,block)`：**1,105,920 = 8,640 x 128**；
- signed groups（再区分 add/sub）：**1,900,560**；
- 每 phase event 数：3,184–48,592；
- 每 phase signed group 数：128–256，均值 219.9722。

### Weight-key group

| 指标 | 数值 |
|---|---:|
| groups | 1,105,920 |
| singleton groups | 0 |
| mean events/group | 170.1285 |
| p50 / p95 / p99 | 152 / 344 / 409 |
| maximum | 695 |
| first 之后的重复 events | 187,042,570 |

### Signed group

| 指标 | 数值 |
|---|---:|
| groups | 1,900,560 |
| singleton groups | 33,528（1.764%） |
| mean events/group | 98.9963 |
| p50 / p95 / p99 | 59 / 291 / 364 |
| maximum | 681 |
| first 之后的重复 events | 186,247,930 |

方向分组仍然很大；如果 held vector 可以每个 destination 携带 add/sub bit，则同一 weight key 不需要因方向变化重新读取。

## 四类机会判断

### 1. Last-vector cache

现有硬件 issue order 未冻结，所以 exact hit count 不存在。fail-closed 区间是：

```text
0 <= last-vector hits <= 187,042,570
```

上界来自每 phase/key 首次事件外的全部重复，只有把相同 key 排成连续 run 才能达到。

若定义一个建议但未冻结的 `phase -> raster row -> block 0..7 -> source ascending` 顺序，相邻相同 `(source,block)` 命中为 **0**：同一 row 内 source 不重复，block 边界改变 block，常规 row 末尾是 block7、下一 row 起点是 block0；唯一 outlier 的 exact-center fallback 也不会产生相同边界 source。于是一个不带 reorder 的单行 cache 不是主线。

### 2. 按 `(source,block)` 驻留

所有 128 个 key 每 phase 都使用，因此不存在 unused-key skip。M88 的 24,576 B 双 weight buffer 已经让 weight phase resident；问题是它按 32 B/cycle 读取一个 96 B vector，驻留本身不会自动把三拍变一拍。

有效硬件必须增加以下至少一种能力：

- 三拍装载一个 96 B last-vector hold register，随后保持；
- 96 B/cycle 宽读或三 bank 并行 read；
- source-major correction bitplane/descriptor queue，使相同 key destination 连续到达。

这些会增加 mux、register/SRAM read port、tag queue 和 accumulator pressure，必须单独综合。

### 3. Run-length / broadcast batching

分组规模足以支持此方向。条件模型中，若每个 weight group 第一次仍需三个 32 B token、其余 destination 每个只需一个 held-vector delivery token，则 correction accounting envelope 为：

```text
188,148,490 + 2 x 1,105,920 = 190,360,330 tokens
```

若 add/sub 必须分 run，则为：

```text
188,148,490 + 2 x 1,900,560 = 191,949,610 tokens
```

若 128-vector full-width resident array 在 phase preparation 时已填好，纯 delivery floor 是 188,148,490 tokens。

这些是基于明确假设的 service-token envelopes，**不是 cycle result**。它们尚未计入 descriptor transpose、有限 queue、PWP/correction dependency、accumulator conflict 和 delivery datapath Fmax。

### 4. Bank multicast

无法从现有证据计算。M40 可导出 raster row，但 M78 没有冻结 destination tag，更没有 accumulator bank/address mapping、每 bank write ports 或同拍冲突。fail-closed 可消除 event 区间只能写为：

```text
0 .. 186,247,930
```

上端是假设每个 signed group 可以无界广播的纯组合上界，不具有可执行性。任何 B-way multicast 数字必须来自带 bank address 的 scheduled trace。

## 2.0x / 2.5x 目标硬约束

service-island bit-sparse denominator 为 1,114,383,288，当前 candidate 为：

```text
correction/fallback 564,445,470
+ PWP              226,222,255
= candidate        790,667,725
```

| 目标 | candidate 必须不超过 | 当前总计至少减少 |
|---|---:|---:|
| 2.0x | 557,191,644 | 233,476,081 |
| 2.5x | 445,753,315 | 344,914,410 |

### 为什么 PWP-only 不够

- 即便把 PWP 226,222,255 **全部删除**，candidate 仍有 564,445,470，只有数学比值 1.9743x，达不到 2.0x；
- 若每个 58,969,374 PWP event 至少保留一个 output delivery token，PWP-only floor 后 candidate 仍为 623,414,844，对应 1.7875x 的模型比值。

因此 correction 必须下降：

| PWP 假设 | 2.0x correction 至少省 | 占当前 correction | 2.5x correction 至少省 | 占当前 correction |
|---|---:|---:|---:|---:|
| PWP 不变 | 233,476,081 | 41.36% | 344,914,410 | 61.11% |
| PWP 降到 one-token/event | 66,223,200 | 11.73% | 177,661,529 | 31.48% |
| PWP 被数学删除 | 7,253,826 | 1.29% | 118,692,155 | 21.03% |

这证明 correction 是 2x/2.5x 的必改主项；不能再只优化 PWP metadata/map/unpack。

## 最小 trace schema 与采集点

需要两层 fail-closed trace。

### A. Semantic multiset + natural order trace

采集点：M72 nearest-center/cap eligibility 完成之后、M78 `Counter` 聚合之前。每一 event 至少含：

```text
schema_version, input_manifest_sha
global_event_seq, phase_event_seq
sample_id, sample_key, operator_id, operator_name
partition, timestep, y, x, raster_row
pattern_index, center_hex, beneficial, route
route = PWP | CORR_ADD | CORR_SUB | FALLBACK_ADD
source[0..15 or null], output_block[0..7]
weight_key = operator/partition/source/block
destination_tag, vector_hash, phase_first, phase_last
```

必须保留零事件 row/phase boundary，避免只存事件后无法重建 run gap。回执应精确守恒 188,148,490 correction/fallback 与各 width PWP populations。

### B. Scheduled hardware trace

采集点：最终 correction/PWP scheduler 与 accumulator-bank mapper 之后、service issue 之前。新增：

```text
scheduled_seq, issue_cycle
queue_id, queue_occupancy, reorder_window
cache_key, cache_hit, cache_fill
run_id, multicast_group_id
accumulator_bank, accumulator_address, bank_port
dependency_group, pwp_ready, commit_seq
stall_reason, accepted, retired
```

还需冻结 accumulator width、overflow/saturation policy，证明 add/sub/PWP 重排不会改变有限位宽中间语义。若只在无限精度数学上可交换，不得 admission。

## 推荐硬件里程碑

建议 M104 做 `phase-local correction transpose + held-weight broadcaster`：

1. 在 phase 内把每 row 的 add/remove correction mask 转成 source-major bitplane或有限 descriptor runs；
2. 每个 `(source,block)` 三拍读入 96 B weight 后保持；
3. 连续流出 destination tag + sign，按 accumulator bank 做 bounded conflict queue；
4. PWP route 与 correction route 共享 commit sequence，escape 走真实 fallback；
5. VCS 对 semantic multiset、tag、sign、最终 accumulator bit-exact miter；
6. 冻结 cache/run/multicast counters 后，再将真实 scheduled count送入周期模型；
7. DC A/B 比较 hold register、transpose storage、tag queues、bank fabric 的面积/Fmax，继续保持 module-only scope。

建议先实现 **held-weight batching，不先赌多播**：它只需一条 accumulator delivery/cycle，而 multicast 需要证明多个 destination 同拍写端口，风险更高。

## Findings

### P0

1. **M103-P0-01：ordered correction/tag/bank trace 缺失。** 无法 admission last-cache hit、run length、multicast 或任何实际 service reduction。

### P1

1. **M103-P1-01：自然顺序与硬件顺序均未冻结。** M40 raw order 可重建，但 M78 用 Counter 丢弃；任何排序都是新假设。
2. **M103-P1-02：resident 不等于宽服务。** 全 weight image 已驻留且每 key 都用，收益必须来自宽读/hold/reorder，不能把 buffer residency 当省拍。
3. **M103-P1-03：重排的精确中间语义未证。** signed correction、PWP base、fallback 与有限位宽 accumulator 的依赖/overflow policy缺失。
4. **M103-P1-04：accumulator bank/port 冲突未知。** multicast 和一拍一 destination 的 sustained delivery 均未闭合。

### P2

1. **M103-P2-01：简单 last-vector cache 在建议 canonical order 下命中为零。** 需要 transpose/reorder，不能把 group reuse 当自然 locality。
2. **M103-P2-02：M83/M99 actual records 只是 catalog。** 它们无法替代 ordered heldout-use trace。
3. **M103-P2-03：仍为 valid825-internal。** 后续硬件结果只能是 module opportunity，不能扩成 accuracy/system/headline。

## GO / NO-GO

- **GO**：采集上述两层 trace；order-independent group strength足以支持继续投入。
- **GO**：先做 one-held-vector、source/block batching、one destination/cycle 的 bounded实现。
- **GO**：bank multicast 只做 trace DSE，等 bank conflict 证据后再决定 RTL。
- **NO-GO**：当前 last-vector hit rate、run/broadcast saving、multicast saving 或 correction cycle reduction claim。
- **NO-GO**：把条件 token envelope写成 physical/module/system speedup。
- **NO-GO**：仅靠 PWP-only 优化冲 2.0x 或 2.5x。

机器审计见 `m103_correction_reuse_preflight_audit.json`，机器结论见 `m103_correction_service_reuse_preflight_independent_hammer_review.json`。
