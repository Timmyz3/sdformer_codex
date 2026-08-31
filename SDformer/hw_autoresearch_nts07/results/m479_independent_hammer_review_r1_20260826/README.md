# M479 lane-local enable 独立 hammer（2026-08-26）

## 裁定

**88/100，P0=0、P1=2；允许 same-constraint 3.000 ns DC，但只作为 DRC/area diagnostic。**

正式 verdict：`GO_SAME_CONSTRAINT_3NS_DC_DRC_DIAGNOSTIC_ONLY`。

功能证据足以放行一次 DC 决策实验；但 `lane-local enable` 在 RTL 中仍只是 global `issue_accept_w` 的 kept aliases，不能在综合前宣称已经关闭 M477 的 fanout/transition/capacitance 问题。

## 独立 VCS

本 review 直接在 16 个 frozen input SHA 上用 Synopsys VCS V-2023.12-SP1 重编译、重跑，没有依赖 producer receipt。

Full suite：14 项计数与 M478 完全一致；九条 base cover 全非空：forward 1、macro read/response 4/4、dual enqueue 1、queue full/full-consume 2/2、back-to-back 2、output stall 3、overflow block 1。

Targeted stalled-RAW：stall 3、ordinary read 0、forward 1、write 2、96 child lanes、stale mismatch 0，old=5/new=1；两条 hazard cover 为 3/1。Full suite 中这两条为预期的 0/0，因此 coverage composition 完整。两次运行均无 assertion/error/fatal。

## RTL 与 ready/握手

规范化差分显示，M479 core 相对 sealed M476 core 只有：module rename、两组 96-bit lane enable aliases、以及 accumulator 更新前的 lane enable 判断；wrapper 与 M476r2 的行为不变。

32 个 `prefetch_valid × final × same-address × psum_ready × core_ready` 组合穷举确认：

- stalled same-address RAW 继续 fail-close；
- release 后继续使用 exact RAW forward；
- different-address prefetch 在 core ready 时透明；
- queue full/reservation backpressure 不被绕过；
- 没有内部 ready combinational cycle。

此前的外部 `valid/payload/psum_ready -> prefetch_ready` 组合路径仍存在；source 必须遵守 valid/payload 不依赖 ready 的协议。

## 为什么只给 diagnostic DC GO

M479 写法本质是：

```text
lane_enable[0:95] = issue_accept_w
outer if (issue_accept_w) {
  if (lane_enable[lane]) update_lane;
}
```

内层条件在外层 accept 分支内逻辑冗余。`keep` 可以保留 net identity，但不会自动实例化受约束的物理 buffer tree；DC 仍可能折叠条件、保留 global source 的高 fanout，或者用大量 buffer 修复。M477 的三个匿名 fanout nets 也不能仅靠 RTL 观察证明都来自这一个 enable。

因此 DC 必须完全复用 M477 的 3 ns SDC、slow/fast library、ideal clock、ZeroWireload 和 hold-fix flow，并满足：

- setup/hold path 均非负；
- max-delay、min-delay、max-capacitance、max-transition、max-fanout 五组全部 clean；
- mapped netlist/report 确实出现预期 lane-local fanout tree；
- 任一 DRC 仍违反即 M479 NO-GO，不能放宽 max_fanout 或改约束。

即使 DC clean，也只能说明该 DRC repair 可实现。M479 没有移除两个 1152-bit response slots，不能预写面积下降，更不能准入 M473 performance、system speedup 或论文 PPA。

## P1

1. Lane aliases + `keep` 不保证物理复制/缓冲，且嵌套条件语义冗余；是否解决 M477 只能由 mapped DRC 证明。
2. M476r2 的 external valid/payload-to-ready 路径仍在；无内部 loop，但 DC 与 full-controller 集成都必须继续检查。

docs/359 未修改，SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 复核

```bash
python3 results/m479_independent_hammer_review_r1_20260826/audit_m479_independent.py \
  --root .
```
