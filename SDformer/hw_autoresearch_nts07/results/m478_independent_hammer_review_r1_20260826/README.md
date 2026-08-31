# M478 M476r2 full-wrapper regression 独立 hammer（2026-08-26）

## 裁定

**99/100，P0=0，P1=0；`P1_R2_TARGETED_REGRESSION_BREADTH` 已真正关闭。**

正式 verdict：`PASS_P1_R2_TARGETED_REGRESSION_BREADTH_CLOSED_MICRO_FUNCTIONAL_ONLY`。

该裁定只准入 M476r2 的 directed micro-functional regression breadth。DC/STA、Formality、physical macro、M473 performance、power/energy、full-network、system speedup 和 DATE headline 全部仍为 false。

## Receipt-blind 复核

没有把 producer receipt 当作唯一真值。独立脚本直接解析 producer 的正式 compile/sim log，恢复出冻结的 14 项计数：

```text
issues=6 rows=5 forward=1 reads=4 responses=4 dual_enqueue=1
full=2 fullconsume=2 stalls=9 b2b=2 exact=2 partialbeats=2
id_attacks=1 overflow_attacks=1
```

九条 base cover 全部非空且计数与 sealed r1 suite 一致：forward 1、macro read/response 4/4、dual enqueue 1、queue full/full-consume 2/2、back-to-back 2、output stall 3、overflow atomic block 1。无 assertion/error/fatal。

另做 fresh exact-SHA VCS replay，14 个计数与 11 个 cover 记录全部一致。

## Coverage composition 为什么成立

M478 full suite 中两条新 hazard cover 为 0：

- `cp_stalled_same_address_prefetch = 0`
- `cp_release_to_new_value_forward = 0`

这是预期结果，因为原 r1 full suite 本来不包含独立评审后来发现的 `output stall × same-address prefetch` 交叉。不能把 M478 单个 run 的 0 冒充 P0 closure。

但 separately sealed M476r2 targeted run 的正式 sim log 直接给出：

- `cp_stalled_same_address_prefetch = 3`
- `cp_release_to_new_value_forward = 1`

因此组合证据完整：M478 负责所有九条原 base path 通过 wrapper 的全量重跑，targeted suite 负责两条 hazard path 的非空闭环。两套 run 都有 manifest + outer seal，且 prior independent P0 hammer 也继续有效。

## TB 与 RTL 身份

独立规范化比较证明，M478 TB 与 sealed M476 r1 full TB 的测试主体逐字相同，差异仅为：

- module/top 名称；
- 新 debug signal；
- DUT 从 r1 core 实例改成 `m476r2_backpressure_safe_parent_queue_pipeline`；
- assertion wrapper 改为 r2；
- PASS/timeout 标签。

TB 没有直接实例化 r1 core。compile log 也确认 r2 wrapper 被解析且 top 为 `tb_m478_m476r2_full_regression`。底层 sealed r1 core SHA 仍为 `c5aa9d0c...`，未修改。

## 剩余边界

本次关闭的是上一轮唯一指定的 regression-breadth P1，不是 Formality，也没有产生任何 DC/PPA 或性能数字。99 分中保留 1 分，原因只是 directed simulation + compositional coverage 不等于穷举形式化等价；该项属于明确的后续门，不构成 M478 目标内的新 P1。

docs/359 未修改，SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 复核

```bash
python3 results/m478_independent_hammer_review_r1_20260826/audit_m478_independent.py \
  --root .
```

加入 `--replay-dir <fresh-m478-run>` 可同时验证 fresh exact-SHA replay。
