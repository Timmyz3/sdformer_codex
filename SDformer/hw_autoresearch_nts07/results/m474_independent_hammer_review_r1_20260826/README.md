# M474 独立 post-run hammer（2026-08-26）

## 裁定

**GO，96/100，无 P0。允许进入 3.0 ns TSMC28 pre-macro DC/STA，但仅限冻结的 M474 micro logic。**

该裁定只证明独立 96-lane fused-parent pipeline 在冻结环境合同下的 directed functional correctness，以及“每个 residual issue 一拍、无额外 completion/parent-read bubble”这个微架构假设。它不把 M473 的 full performance、物理 scratch macro、PPA、全网或系统倍速改成 admitted。

## 独立检查结果

- 生产结果目录 `results/m474_fused_parent_dual_update_vcs_r1_20260826` 的 manifest 与 outer seal 均通过；合同副本与源合同逐字节相同，12 个执行输入 SHA 全部匹配。
- 正式流程是 Synopsys VCS V-2023.12-SP1；compile/sim 均通过。10 个 directed 计数精确匹配，9/9 SVA cover 非空，无 assertion/fatal/error marker。
- 另起 clean exact-SHA replay，计数与 cover 和正式结果完全一致；正式 compile/sim 日志 SHA 与开发 preflight 日志不同，排除了把开发回归冒充正式结果的情况。
- clean-room arithmetic oracle 检查 96 lanes × 5 rows：480 个 scratch 和 480 个 psum 值；scratch 范围 `[-2, 10]`，psum 范围 `[-100, 296]`。K16 signed-INT8 理论 residual 范围 `[-2048, 2032]` 可由 signed13 prefix 容纳。
- one-ahead registered-Q、ID correlation、same-address RAW forward、consume+nonmatching prefetch、final same-cycle scratch+psum dual write 均有 directed/SVA 证据。
- final overflow attack 命中 1 次，`issue_ready`、scratch write、psum write、row completion 被原子阻断，并锁存 protocol fault；对应 cover 非空。

审计器共执行 **1,139 checks，0 mismatch**。

## 仍为 false 的边界

- `m473_full_controller_rtl=false`
- `m473_performance_admitted=false`；M473 状态仍为 `PASS_M473_CPU_DSE_NO_GO`
- `physical_scratch_macro=false`
- `macro_timing/power/ppa=false`
- `full_network/system_speedup/date_headline=false`

因此，3.0 ns DC/STA 结果必须标注 **pre-macro**。外部 64-row × 96-lane signed12、1R1W scratch 的真实 timing/energy/area 不得计入或推断。若逻辑 setup 不收敛，或实现要求新增 issue bubble，立即推翻 fused timing 假设；即使逻辑收敛，也还需目标 144-byte 1R1W macro gate。

## 可复核入口

```bash
python3 results/m474_independent_hammer_review_r1_20260826/audit_m474_independent.py \
  --root .
```

传入另一次 exact-SHA run 目录可同时核对重放：

```bash
python3 results/m474_independent_hammer_review_r1_20260826/audit_m474_independent.py \
  --root . --replay-dir <fresh-m474-run-dir>
```

`docs/359_DATE终局冻结_20260813.md` 未修改，SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
