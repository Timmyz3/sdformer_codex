# M476r2 backpressure-safe parent queue 独立 hammer（2026-08-26）

## 裁定

**93/100，无 P0；允许启动 same-constraint 3.000 ns pre-macro DC compare，但仅用于成本/时序评估。**

正式 verdict：`GO_SAME_CONSTRAINT_3NS_PREMACRO_DC_COMPARE_ONLY`。

M476r2 在不修改 sealed r1 core 的前提下，关闭了 r1 独立评审复现的 `old=5/new=1` stale-parent P0。这个门不准入 M473 performance、power、system speedup 或 DATE headline；这些字段继续为 false。

## P0 复测结果

独立 VCS 负测保持与 r1 P0 相同的核心攻击：scratch 旧值为 5，final 要提交新值 1，最后一个 issue 因 `psum_write_ready=0` 连续停顿，同时持续施加同地址 prefetch。

结果：

- 连续停顿 5 拍，同地址 prefetch 0 次握手、0 次普通 SRAM read；
- 释放后走 sealed r1 的 exact RAW forward，96 个 queue lane 全为 1；
- scratch 提交后的 96 lanes 全为 1；
- child 对该 parent 的 96-lane 计算全部消费新值，无 stale mismatch；
- 另一次复位后，final 停顿期间对不同地址 9 的 prefetch 仍可握手并读回 9，说明 guard 没有粗暴冻结所有无关流量。

```text
PASS M476r2 independent attack closed old=5 new=1 stalled=5 same_reads=0 forward=1 child_checks=96 diff_read_value=9
```

Producer 正式 run 也满足冻结合同：stall 3、read 0、forward 1、write 2、child check 96、stale mismatch 0；两个 r2 新 cover 和三个相关 r1 cover 均非空。fresh exact-SHA replay 的 counts/covers 完全相同。

## 封存与身份审计

- sealed r1 core SHA 仍为 `c5aa9d0c...`，没有被 r2 修改；
- r1 VCS、r1 独立 P0 hammer 与 r2 producer 三套 manifest/outer seal 全部有效；
- r2 contract copy 和 17 个 frozen/run input SHA 全部通过；
- receipt-blind 重解析正式 compile/sim log，无 assertion、error 或 fatal；
- 独立审计共 246 项检查，0 mismatch；
- docs/359 未修改，SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 并发审计

对 `prefetch_valid × final × same-address × psum_ready × core_ready` 的 32 个布尔组合穷举：

- 同地址 final 在 psum backpressure 下：prefetch valid/ready 同时 fail-close；
- psum release 后：请求进入 sealed core 的既有 RAW forward；
- 不同地址：只要 core ready，wrapper 完全透明；
- queue full 或 response reservation 令 core not-ready 时：wrapper 不绕过 core backpressure；
- 未发现内部 ready combinational cycle。

## P1 与 DC 门禁

1. wrapper 新增 `prefetch_valid/issue payload/psum_write_ready -> prefetch_ready` 的组合路径。协议要求 source 的 valid/payload 不得依赖 ready；full-controller 集成也必须检查不会形成外部 valid-ready loop。r2 top 必须在与 r1 完全相同的 3.000 ns DC 约束下实测该路径成本，时序失败则不得准入性能。
2. r2 producer run 是定向 P0 closure：macro read/response、dual enqueue、queue full/full-consume 和 overflow cover 为 0。这些在未修改的 sealed r1 core 中有动态证据，但尚未经 wrapper 全量重跑。Formality/full-controller admission 前，应把 r1 全套并发回归接到 wrapper，或形式化证明 `stalled_raw_hazard_w=0` 时透明。
3. guard 只覆盖可恢复的 `psum_write_ready=0`；不能包装成通用 pending-write coherence。ID/overflow 等非法 final 继续使用 sealed core 的 terminal fault 合同。

因此 DC 的 GO 是“可测成本”，不是“已经有 PPA 或性能优势”。M473 `PASS_M473_CPU_DSE_NO_GO` 也没有被推翻。

## 复核

```bash
python3 results/m476r2_independent_hammer_review_r1_20260826/audit_m476r2_independent.py \
  --root .
```

加入 `--replay-dir <fresh-r2-run>` 可同时核对 fresh exact-SHA replay。
