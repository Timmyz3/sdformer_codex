# M586｜M579 r2 source author handoff

日期：2026-08-28  
状态：**AUTHOR_SOURCE_ONLY；请求 fresh source-static hammer；没有 execution candidate/release。**

## 交付物

- analyzer：`system_simulator/scripts/analyze_m579_paft_control_single_port_product_capture_r2.py`
  - SHA256：`70eb07465bb008569967f69ae0ea0d51057d64dd0d51669b604a8f1cd4d4b471`
- immutable future runner：`system_simulator/scripts/run_m586_m579_paft_control_single_port_product_capture_r2_exact_sha.sh`
  - SHA256：`8e0efbb6c9f1e188f45fe37f4ae15b4f60f9b8cff9c533a0e822f3549aecd45e`
- r2 source contract：`contracts/m586_m579_paft_control_single_port_product_capture_source_contract_r2_20260828.json`
  - SHA256：`319d1c895fd2327f0320c4277cc6f853d2fe8536d20406110784dc04a5fa44ec`
  - member sidecar 与 outer sidecar 已双封。

## M584 阻塞项关闭

1. **Python/spawn P0**：冻结 `/opt/anaconda3/envs/python310/bin/python3.10`，二进制 SHA `4cd88f...6b0f`，Python 3.10.16；NumPy 2.0.1 的绝对 init 路径和 SHA `c09e25...7c2b` 同时绑定。runner/analyzer 均 fail-closed 检查。轻量 spawn probe 已实际通过，子进程成功 import M43/M504/M505 并执行八 row dead-write-only recurrence；正式 trace records=0。
2. **task-order P0**：r1 worker 的 `(partition,chunk)` 数组在进入周期模型前显式 reshape/transpose 成冻结 M528 的 `(chunk,partition)` C-order。anchor 为 `[0,47,94,141]`，每算子 20,304 tasks；末 chunk 56 rows。结果 schema 写入 `sample_operator_row_chunk_partition`。
3. **M504 P1**：直接绑定 SHA `9a7586...30a5e`，每个 worker import 前重验。
4. **M255 P1**：strict-parse M255；输出同时携带 valid825、相同十帧及完整 64 帧三个 accuracy scope。完整 `zurich_city_09_a` 明确为 PAFT AEE 退化 1.0189020311889285%，并写死 `accuracy_performance_pareto=false`、single-seed、无共同 evaluator runtime SHA。
5. **atomic attempt/output P1**：runner 默认只做 preflight。未来 `--execute` 必须有另行评审的 launch contract；流程为 pre-attempt 全输入/80 payload 检查、原子 attempt、same-filesystem staging、terminal rehash、member/outer 双封、拒绝 racing target、rename；任一错误 quarantine staging/attempt。
6. **terminal rehash P1**：正式发布前重新验证 execution contract、r2 analyzer、M43/M504/M505/r1 base、M247/M255/M528、容量账本、docs359、两个 manifest 和 80 packed payload。

## 额外关闭

- 容量不再只用 hard-code：strict-parse M528 hammer、M528 result JSON、M528 capacity CSV；候选九行 macro-rounded sum 必须等于 213,376 B，预算 245,760 B，余量 32,384 B。
- 结果同时披露 `generated_macro_integration_ppa_energy=OPEN_NOT_ADMITTED`，容量不能当 integrated macro PPA。
- 每 record 显式检查 sample/operator 映射、shape/output_shape、2,304,000 elements、三个 288,000-byte plane 的 offset/extent、packing string、negative_count=0、timestep support sum、payload basename/unique/size/SHA。
- bit/product/PAFT-control 三种 ratio 继续分字段；绝不相乘，绝不升格 system/RTL/headline。

## 已执行的唯一测试

五次迭代过程中的轻量 preflight；最终 source tuple 上 runner preflight 输出：

```text
PASS_LIGHTWEIGHT_IMPORT_SPAWN_RECURRENCE_ONLY
synthetic rows=8, parent_edges=5, ideal issues=6, liveness cycles=8
task order=sample_operator_row_chunk_partition
formal_trace_records_processed=0
result_or_attempt_created=false
```

未运行 80-record 正式 CPU、GPU、EDA 或远程任务。未来 execution contract、正式 result、正式 attempt 均不存在。docs/359 SHA 仍为 `dedde7ce...bdfc4`。

## 评审要求

fresh hammer 必须独立攻击 source/runner/contract，不得运行正式 80-record。只有 P0=0、P1=0、score>=95 才允许 root 另建 execution candidate；source review 自身不得授权 execution release。
