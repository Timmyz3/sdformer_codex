# M109-r2 dual-timeline window/storage frontier：独立打铁复审

日期：2026-08-24

结论：**90/100，P0=0 / P1=5 / P2=5。r1 prior-drain serialization P0 已在 r2 软件递推层关闭；当前准入 corrected same-clock precompacted service-island software schedule bound，不准入 RTL-measured、physical、system 或 headline。**

## 独立闭环

- M109-r1 scheduled frontier 已由独立 revocation 合同撤销。
- r2 加入独立 `controller_free`：`dispatch=max(fill_end, controller_free)+1`，correction 完成或 empty dispatch 后更新 controller timeline。
- 用此前从冻结 M40/M72/M41 raw 数据重建的四窗口 ordered ledger 逐项对照，W43/W64/W294/W384 每点的 **22 个公开 recurrence 字段全部一致**；work/group/storage 也全部一致。
- W64 对 M108-r3 的 22 个共同 schedule 字段全部一致：`521,264,186 cycles / 2.138770047785×`。
- 五项 r2 manifest 全部重新校验通过；`docs/359` SHA 仍为 `dedde7ce...`。

## 四窗口结果

| W | groups | candidate cycles | ratio | storage lower bound | geometry |
|---:|---:|---:|---:|---:|---|
| 43 | 46,867,834 | 556,942,442 | 2.001759255× | 101,864 B | projection |
| 64 | 35,140,002 | 521,264,186 | 2.138770048× | 151,592 B | M106 geometry VCS |
| 294 | 10,395,056 | 446,212,276 | 2.498504788× | 696,232 B | projection |
| 384 | 8,271,296 | 439,708,199 | 2.535462042× | 909,352 B | projection |

四点共同 `events=188,148,490`、`PWP tokens=226,222,255`。W294 仍低于 2.5×，W384 仍高于 2.5×；该 crossing 仅准入为 software schedule bound。

## 尚未关闭的硬件边界

1. 没有 integrated commercial M106/PWP/M104/accumulator/flush/commit cycle miter，所以 2.13877× 不是 VCS-measured speedup。W64 只有 controller geometry 的 directed VCS。
2. signed24 full-lane accumulator 仍是 port cut；没有 macro latency、RDW、clear/epoch、commit hazard 或 finite-width final-state miter。
3. shared weight SRAM 的地址、bank、端口、延迟与争用未建模。
4. 输入已假定 losslessly precompacted；scan、有限队列与 delivery bandwidth 不在模型中。
5. fixed8 denominator 的 raw service tokens 与共同 commit/flush 完全可复现，但没有 matched controller/descriptor ingress，不能叫 equal-controller end-to-end baseline。

## 两个证据硬化缺口

- M109-r2 没有携带 ordered descriptor digest；W64 digest `a011720a...` 由 M108-r3 与独立 raw rebuild 共同确认，但 frontier 自身不自包含。
- producer 对 W64 明确检查 18 个整数状态字段，却没有 fail-closed 比较 ratio、headroom、两个 maximum 字段与 digest。本评审已确认全部 22 个共同字段相等，建议下一版把完整 key-set 检查并入 producer。

## Claim boundary

- r1 revocation、r2 controller-free 修复、raw work/group/storage：**GO**。
- W64 对 M108-r3 common-field identity：**GO**。
- corrected same-clock precompacted service-island software bound：**GO**。
- W294<2.5×、W384>2.5×：**GO（software bound only）**。
- actual combined RTL cycle measurement、non-W64 RTL measured、macro area、physical/equal-area/system/full-network/headline：**NO-GO**。

机器证据见 `m109_r2_window_storage_dual_timeline_frontier_independent_audit.json` 与 `m109_r2_window_storage_dual_timeline_frontier_independent_hammer_review.json`。本评审只写本目录，未修改 production、contracts/results 或 `docs/359`。
