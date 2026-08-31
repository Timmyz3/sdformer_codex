# M180 双窗 K4 reservoir 独立打铁评审 r1

结论：**83/100，`PASS_AS_EXECUTABLE_REFERENCE_REJECT_AS_PRIMARY_K4_ARCHITECTURE`，P0/P1/P2 = 2/5/3。**

M180 是可信的可执行模块参考，但不应继续作为 FC2 主架构。它把 M179 的双窗、跨
descriptor top-4 bank 调度落实成 RTL，并通过 exact-SHA VCS/SVA 与 3 ns Synopsys DC；
但相对 M177 只带来 `1.129841x` analytic schedule opportunity（少
`11.491993%` cycles），逻辑面积却达到 `10.966839x`，conditional
opportunity-per-area 只有 `0.103023`。建议冻结 M180 作为 global-top4 K4 对照，下一主线
转向 K8 fixed-bank 实现筛选。

## 独立验证结果

- M180 RTL/SVA/TB、两个 contract、M179 correction/review、VCS/DC evidence 的 44 个
  SHA entry 全部重验，0 mismatch；`docs/359` 仍为
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
- sealed VCS 的 17/17 cover、3 个攻击、1 次 release+refill、3 次 window replace 和 62
  个 cross-descriptor group 均可复核，0 assertion failure。用新 seed `180913` 直接重跑
  sealed binary，17/17 仍全过。
- 原 sealed population 的 descriptor 数都是窗口深度的整数倍，未真正测试非零 partial
  final window。本评审另写独立 VCS TB，以 stage0/1/2/3 的 `3/6/10/9` descriptors
  测试 D2/D4/D8/D8 partial close：source multiset、replay、bank identity、unused-lane
  zero 全为 0 mismatch。
- 不导入主 scoreboard/analyzer，独立穷举 1,679,616 个 bank-count 向量
  `[0,5]^8`，再随机检查 100,000 个 `[0,96]^8` 向量。top-4-largest greedy 每次都构造性
  达到 `max(max-bank, ceil(events/4))`，0 mismatch。另有 10,000 个随机 descriptor
  window、164,262 个 source term 的 identity/conservation 检查，0 mismatch。

## 必须修正的协议边界

RTL 的 header legality 只检查 `output_blocks in {1,2,4,8}`，没有检查
`descriptor_count <= stage beat extent`。独立 VCS 实测
`output_blocks=1, descriptor_count=5` 会被接受且当拍 `protocol_error=0`，但 stage0 只有
beat index 0..3。生产者不继续发送时会留下不可完成 token；发送第 5 个 descriptor 时才
会 fault。

因此现有三种 attack 只能证明其命名的 malformed cases，不能扩写为完整 protocol
fail-close。应在 header 阶段加入 4/8/16/32 的 count bound，并给四个 stage 都加
over-extent attack。

## Synopsys DC 复核

| 项 | 独立复核 |
|---|---:|
| Cell area | 14,417.928053 um2 |
| Cells / sequential | 22,209 / 1,882 |
| Logic levels | 161 |
| Critical path | `group_ready` -> `group_source_channel_q_reg[3][10]` |
| Setup / hold slack | 0.0000 / 0.0003 ns |
| 五类 constraint | 5/5 clean |
| TIM-209 / OPT-150 loop | 0 |
| Macro | 0 |

max-delay、min-delay、max-capacitance、max-transition、max-fanout 五类
`report_constraint` 确实全部 clean，且没有结构 timing loop。但这是 ideal clock、
ZeroWireload、0 macro 的 pre-macro 结果，setup 已四舍五入为 0，selector 路径 161 levels，
没有工程余量。`dc.log` 仍把 `clk_core` 和 tie-high `n31756` 报为 TIM-134 high-fanout
informational nets；安全措辞是“五类显式约束 clean”，不能说所有物理 net 都已低于 16。

## 架构决策：K8 取代主线，M180 保留作基线

M180/M179 K4：`127,581,198` cycles；M177/D1 K4：`144,146,504`，所以只省
`16,565,306` cycles。M180 面积 `14,417.928053 um2` 对 M177
`1,314.684003 um2` 为 `10.966839x`，不能以此作为高效硬件卖点。

独立评审已复核的 M181 same-depth K8 为 `97,607,807` cycles，相对 M180 K4 还有
`1.307079853x` analytic gain，并通过 fixed bank ownership 去掉 global top-4 sorter。
所以应把 K8 作为下一 implementation screen，同时保留 M180 作为 matched K4 参考。K8
仍需八 bank weight response、八 accumulator lane、bounded-D8 VCS/SVA 和同约束 DC；在
这些完成前，`4.344533568x` K1/K8 仍只能叫 analytic frontend ratio。

## P0 / P1 / P2

P0：

1. 冻结 M180 为可执行 K4 reference，不再投入下一轮 top-4 sorter 微调；改做 bounded-D8
   K8 fixed-bank RTL，并在相同 library/clock/SDC 下与 M180 做 VCS/DC 对比。
2. producer、directory、weight response、arithmetic、accumulator/commit 未组成前，禁止把
   M179/M180/K8 ratio 称为 physical、complete-FC2、system 或 headline speedup。

P1：

1. 修复 header descriptor-count extent fail-close，并补四档攻击。
2. 把本评审的四档非零 partial-window case 纳入 exact-SHA 主回归，新增 partial-close cover。
3. 若继续保留 K4 实现，需 pipeline/hierarchy 化 selector；K8 需预提交更浅 critical path。
4. release+refill 目前只有 1 hit，补重复边界、stall sweep 和 buffer ownership assertions。
5. 用 frozen payload 的 materialized groups 接 numeric miter/accumulator，再谈 executable
   end-to-end schedule。

P2：

1. 增加直接的 unused-lane-zero SVA。
2. maxfanout16 只表述为五类 constraint clean，并披露 clock/tie-high TIM-134 信息。
3. 如需统一 admission JSON，新增 keyed-to-RUN_COMPLETE 的 overlay，不修改 precommit contract。

机器可读裁决见 `m180_independent_hammer_review_r1.json`，独立重算见
`independent_audit_result.json`。本评审只写当前目录，未修改 M180 主线或 `docs/359`。
