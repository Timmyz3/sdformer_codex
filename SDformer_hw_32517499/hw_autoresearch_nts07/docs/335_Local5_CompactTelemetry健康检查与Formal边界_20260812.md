# Local5 Compact Telemetry 健康检查与 Formal 边界

> 日期：2026-08-12  
> 最终包：`results/local5_compact_telemetry_samples3_6_v3_reviewfix_20260812/`  
> 证据：`[rtl汇总遥测]+[父级数值证据引用]`  
> formal G0：**DENY**

## 1. 结论

sample3-6 的 48 个真实 RTL 窗口已建立 compact telemetry 健康检查索引。它把每个
窗口的 `actual_receipt`、`window_complete`、Verilator PASS telemetry、Acc32 archive
slice 和 `actual.memh` 连接为同一条可审计链，并独立重算结构计数与 service delay。

这个包可以用于快速检查 1,200-window 数值回放是否具有完整来源和非退化遥测；它
**不能**替代 462,600-phase 逐事件 ledger，也不能提供 C0-C4、resource conflict、
候选性能或 ASIC PPA。

## 2. 为什么需要 compact 健康检查

H24 单窗完整 trace 有 47,941,735 行、Phase Array Store 约 1.23 GB。直接将逐行 CSV
扩展到 1,200 窗会产生不合理的验证存储和重放成本。另一方面，只保存一行 PASS scalar
又会形成同源自报。

本轮折中不是取消 formal phase ledger，而是在它之前增加一个轻量 G0 前置层：

```text
sealed numeric shard
  -> window_complete SHA
  -> actual receipt / raw log / actual.memh
  -> Acc32 archive expected/actual slice
  -> compact telemetry health row
```

## 3. 独立检查内容

每个窗口执行以下检查：

1. canonical 12-block `(stage,block,H)` 拓扑和 sample/window identity；
2. `window_complete` 冻结的 receipt、raw log 和 actual SHA；
3. receipt 内 executable、run argv、release provenance 和当前文件 SHA；
4. 从父 Acc32 archive 按 offset 提取 expected/actual slice；
5. archive actual 与 RTL `actual.memh` 逐元素一致；
6. 独立重算 identity-bound miter digest；
7. `token/partial/final/readout/RMW/release/scheduler` 的 H 闭式计数；
8. 独立 uint32 实现重算 token/weight/result transaction delay；
9. `weight_cycles=2*weight_count+weight_delay_sum`；
10. `drain_cycles=3*final_count+result_delay_sum`。

H3 的真实锚点为：seed=17828 时 token、weight、result delay sum 分别为
10,108、23,156、108,297；单元测试与 RTL telemetry 一致。4/4 测试通过。

## 4. Fail-closed 负例

首次集成目录 `results/local5_compact_telemetry_samples3_6_v1_20260812/` 在读取第一个
真实 receipt 时拒绝。原因是生成器假设 `provenance_level=complete`，而 sealed v5
实际冻结值为 `exact_argv_sealed_release`。该错误被修正为严格接受真实 schema 后，
才生成 v2。

这不是 RTL 失败，但说明工具不会在 provenance 枚举不一致时静默降级。

## 5. 独立复审与修复

初轮复审为：

```text
2.8/5 Weak Reject
P0=0, P1=3, P2=3
```

主要问题是当前 receipt 未与父 `window_complete` 闭环、miter 摘要只复制未重算、
cycle/phase 仍同源自报，以及缺少逐 phase 重放。修复后：

- receipt/log/actual 与 `window_complete` 三 SHA 闭环；
- archive expected/actual、`actual.memh` 和 miter digest 独立重算；
- canonical topology/sample/head 强制检查；
- transaction delay 和 weight/drain cycle 部分独立重算；
- 证据标签从过强的 provenance/miter 说法降为
  `[rtl汇总遥测]+[父级数值证据引用]`；
- `cycles/frontend_cycles` 明确保留为同源验证遥测。

最终针对性复审为：

```text
4.3/5 Accept（仅作为 compact telemetry 健康检查）
P0=0
P1=0（在健康检查声明范围内）
```

外部不可变 trust root 仍缺，作为 P2 保留。

## 6. 不能替代 formal phase ledger

本包只有 48 个逐窗口汇总 row，没有以下正式内容：

- 13,800 个 input-head phase 记录；
- 462,600 个 phase 的 start/end/duration；
- term/source/command offset 和 event identity；
- resource 占用、相序和冲突；
- output-tile patch；
- `rtl_observed/rtl_derived/schedule_model` 来源分类；
- 独立 scheduler 重算 C0-C4 和 tail cycle。

因此 `cycle/frontend_cycles` 不能用于 candidate latency、speedup、吞吐或能效。
formal G0 继续为 `DENY`。

## 7. Formal 下一步

紧凑正式 phase 表示至少需要保存：

```text
identity = sample/stage/block/window/input_head/output_tile/phase_id
phase_type + event_type + resource
start_cycle + end_cycle + duration
term/source/command offsets
origin = rtl_observed | rtl_derived | schedule_model
raw trace/template/tile patch/window complete/miter SHA
```

独立 parser 必须能从参数化 template+patch 无损展开，并在 H3/H6/H12/H24 真实 trace
上逐事件 SHA 一致；独立 scheduler 再从底层事件重算 C0-C4。只有完成这条链，才有
理由讨论用紧凑表示替代旧的数十 GiB v4 archive。

本包是验证基础设施，不是 DATE 架构创新，不进入贡献列表。
