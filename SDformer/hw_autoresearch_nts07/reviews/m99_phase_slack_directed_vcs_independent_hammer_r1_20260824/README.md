# M99 phase-slack directed VCS 独立打铁

## 结论

评分 **79/100**，`P0=0 / P1=4 / P2=5`。

dev_r2 的 VCS 编译和仿真结果本身可信，可以作为历史 directed evidence 封存：requested RTL/TB/SVA
SHA 全匹配，VCS V-2023.12-SP1 编译完成，仿真正常 `$finish`，唯一 PASS 行与 TB frozen counter
一致，八个 cover 全部命中，未发现 assertion/fatal/error signature，也没有 assertion 被禁用。

但是，**当前 RTL/TB/SVA 三个 SHA 不应冻结为 directed exact-SHA contract**。核心原因不是 RTL
已经证明错误，而是 simultaneous-load/lookup 的 SVA 与预定接口语义冲突，同时现有场景只覆盖了
unloaded 状态，恰好绕过冲突。修复 SVA 和 directed scenario 后 SHA 会变化；当前 run 也没有把
编译时 source SHA 写入日志，不能从日志自身证明 executed bytes。

在 full actual-record VCS replay 和同流 3 ns DC 完成前，M99 仍不得 admission；更不得声称 PPA、
module/system speedup 或 DATE headline。

## exact SHA 与 VCS 结果

| artifact | SHA256 / result |
|---|---|
| RTL | `93c638f69a2a50f4d020f4a2d0b974d620574e80b05f96c0a0358008c8883353` |
| TB | `9afa3a4d5d948fea695bd109f0ea6a12a08e9c49b7e0647bac869d1822dbbc99` |
| SVA | `1f5cee2e0e31b287794b50cda2e6087ee89fc311ba89021ffb738dea0a6528c0` |
| compile log | `7b9f87c2e691c59f4536d2f4542b4ef96c7e3eb7cf6b943d60696f74a224de86` |
| sim log | `3f130a2ea749b42e6547ec3ebfde3047565c906797415f37d60bf64c0a8a6e20` |
| assertion report | `cee8082aa50f8aa22f465708bf483f3bc7a28c533b65d923cd43cbe174b7e3da` |

PASS counters 独立复算为：128 entries、436 beats、512 parser cycles、10 stall cycles、3 poison、
1 early lookup、1 simultaneous attack。synthetic metadata 中 code 0/1/2/3/4 分别出现
26/26/26/25/25 次，对应 beat 数 `26×3 + 26×4 + 26×4 + 25×5 + 25×1 = 436`。

assert report 有八个 cover，均为 1001 attempts：

| cover | matches |
|---|---:|
| phase load | 5 |
| simultaneous load+lookup | 1 |
| lookup stall | 1 |
| escape | 27 |
| width9 / width10 / width11 | 28 / 28 / 27 |
| metadata error | 5 |

cover 数与 sim log 完全一致。`assert.report.disablelog` 没有 compile-time、hierarchical 或 dynamic
disable 条目。

## 128-cycle 边沿

这一项 directed 证据较强。`load_and_wait_for_parser` 在 E0 接受后，以 post-edge `#1` 观察
`phase_loaded`，早于 128 或晚于 128 都会 fatal；一份 legal 加三份 poison metadata 共执行四次，
所以 512 恰为 `4×128`。legal case 从 task 返回后先等 negedge 驱动 lookup，再在下一 posedge
握手，行为对应 E129。

但 SVA 端口没有暴露 `parse_active/index/cursor/poison`，因此没有 assertion 直接证明 E1..E128
恰好依次处理 index 0..127、index 不 wrap、cursor delta 正确和 poison 单调。当前 TB 证明端到端
固定 latency 和若干最终结果，不是完整的 parser micro-sequence proof。

## simultaneous load+lookup 的未闭合 seam

RTL：

`phase_load_ready = !m82_busy && !parse_active_q && !lookup_valid`

因此在一个已加载、无 poison、M82 idle 的合法旧 phase 上，同时拉高 load 与合法 lookup 时，load
ready 必须为 0，但 `mapper_valid=1` 且 `m82_beat_ready=1` 时 lookup ready 可以为 1。这正是 direction
定义的“旧 lookup 优先、load 等待”。

当前 SVA 却无条件断言：

`phase_load_valid && lookup_valid |-> !phase_load_ready && !lookup_ready`

它会把上述合法优先级行为判错。现有 simultaneous test 紧跟 reset、phase 尚未 loaded，故
`mapper_valid=0`、两边 ready 都是 0，cover 命中而错误断言没有暴露。

修订应把 assertion 改成“两个 accept 不可同时发生”，并分别 cover：

1. unloaded/poisoned simultaneous：两者不接收且 lookup fault；
2. loaded+idle simultaneous：load backpressure，旧 lookup 正常接收；
3. lookup/M82 stall simultaneous：两者等待，payload 稳定；
4. lookup 完成后 held load 最终接收新 phase。

## M85 differential 的有效边界

当前 legal campaign 对一个合成 metadata image 覆盖 128 个 entry、436 beat、全部 width/escape；
differential bundle 包括 ready、八个 bank-row address、output tag/width/escape/values/accept、错误和
busy。输出在五个 entry 后各 stall 两拍，共十拍。这足以证明该 synthetic phase 在 M99 commit
之后与 M85 cycle-aligned 一致。

它没有覆盖：

- 1,728 个 actual phases、221,184 entries、835,383 beats；
- 多种真实 code/base/terminal 分布和随机 backpressure；
- metadata capture 后修改 live input；
- audit 中 held second load；
- reset 在 index 0/63/127 的 abort；
- early lookup 在 index 63/127，以及 sticky fault 后由新 load 恢复。

所以这仍是 directed synthetic differential，不是 actual-record replacement evidence。

## poison 覆盖边界

三个固定攻击都以 128 cycles 完成，并与 M85 的最终 metadata/protocol blocking 对齐：reserved code5
at entry0、wrong base at pattern4、pattern15 base=8191。第三项主要覆盖 late wrong-base；它不是
独立的 cursor/fetch overflow 证明。

尚缺 code6/code7、reserved mid/final entry、独立 fetch overflow、cursor overflow、all-escape zero
terminal，以及每类 poison 对应的定位/单调/final-commit assertion。`cp_metadata_error=5` 只能说明
错误状态被观察五拍，不等于五种 poison class。

## 冻结与 admission 决策

- 当前 dev_r2 run：**GO 作为带边界的历史 directed evidence**。
- 当前 RTL/TB/SVA exact-SHA contract：**NO-GO freeze**。
- 修复 simultaneous SVA、补 loaded-old-phase 场景并用 fail-closed SHA runner 重跑后：可重新申请
  directed contract freeze。
- full actual-record replay 前：**NO-GO RTL replacement admission**。
- 同一 TSMC28/3.000 ns/ideal-clock/ZeroWireload 流程 DC、resource-family collapse 和 same-flow area
  gate 前：**NO-GO performance/PPA admission**。
- actual-record 与 DC 即便通过，也只允许 standalone/loader-integrated logic-island claim；系统、能耗、
  SRAM macro 和 DATE headline 仍需各自证据。

机器结论见 `m99_phase_slack_directed_vcs_independent_hammer_review.json`；独立日志/算术审计见
`m99_directed_vcs_independent_audit.json`。评审没有修改生产 RTL、TB、SVA、合同或结果。
