# M184 双窗 K8 fixed-bank frontend 独立打铁评审 r1

结论：**89/100，`PASS_AS_PRIMARY_FIXED_BANK_FRONTEND_WITH_P0_COMPOSITION_GATE`，P0/P1/P2 = 2/5/3。**

M184 standalone frontend 是可信的，而且相对 M180 的硬件简化不是纸面故事：在完全相同的
TSMC 28nm、3 ns、max-fanout16、ideal-clock、ZeroWireload 条件下，总 cell area 从
`14,417.928053` 降至 `10,026.828029 um2`（`-30.455833%`），combinational area
`-42.305743%`，cell count `-33.968211%`，logic levels 从 `161` 降至 `136`。同时 M182
bounded K8 的解析周期相对 M179 K4 为 `1.307079853x`，所以 matched same-clock 的条件性
schedule-throughput/frontend-logic-area 为 `1.879496010x`。

这个结果足以让 M184 取代 M180 成为 frontend 主 Pareto 点，但还不能叫 physical、complete
FC2、system 或 headline speedup。

## 独立验证

- M184 RTL/SVA/TB、VCS/DC contract、sealed run、M182 和 `docs/359` 的 SHA 全部复核，
  0 mismatch；`docs/359` 仍为
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
- sealed VCS：21/21 cover，320 个 bitmap event 与 320 个 unique source term 完全守恒，
  283 个 held replay result、1757 个 replayed source term，0 assertion failure。
- fresh seed `8451841`：21/21 cover 再次全过，0 assertion failure。
- review-only 独立 VCS（未导入主 scoreboard）：40 tokens、307 descriptors、3852 events、
  4487 results、21736 replayed terms；29 个 partial-final windows、1493 stall、6 次
  release/refill、26 次 window replacement、4 档 header extent attack 与 3 档 descriptor
  extent attack 全过，earliest-event/bank identity/conservation 均 0 mismatch。
- 独立 recurrence probe 覆盖 zero token 与四档非零 partial token。M182 recurrence 在非零
  token 上精确对应 `first descriptor_accept -> token_done_valid`；若把真实
  `token_done_accept` 也算进去，还要多 1 cycle。零 token 的 2 cycles 对应
  `header_accept -> token_done_accept`。所以 `97,607,807` 不是完整 header-to-header 模块
  throughput，必须保留 analytic 口径。

## M180 对比裁决

| 项 | M180 K4 | M184 K8 | 变化 |
|---|---:|---:|---:|
| Cell area (um2) | 14,417.928053 | 10,026.828029 | -30.455833% |
| Combinational area (um2) | 10,687.698016 | 6,166.187967 | -42.305743% |
| Cells | 22,209 | 14,665 | -33.968211% |
| Sequential cells | 1,882 | 1,915 | +1.753454% |
| Logic levels | 161 | 136 | -25 levels |
| Critical path (ns) | 2.53 | 2.52 | -0.01 ns |
| Analytic wall cycles | 127,581,198 | 97,607,807 | 1.307080x |

两者都有两份 D8-capable bitmap window；M184 虽然输出八个 structural bank slot，而 M180
只输出四个 packed slot，但去掉 global top-four ranking、bank IDs 和 bank-to-lane packing
后，mapped combinational area 和 logic levels 仍显著下降。因此这项 trick 的硬件收益真实。

不过 timing 余量几乎为零：setup `+0.0023 ns`、hold `0.0000 ns`，critical path 仍是
`group_ready -> group_source_channel_q_reg[5][11]`。此外 DC 报告 1924-load tie-high net 的
TIM-134；五类显式 constraint clean 不等于已经完成 tie-cell/CTS/route。

## 最关键组合缺口

M184 的输出是 arbitrary `group_bank_valid` mask，slot 本身就是 bank，没有 bank ID。现有
M183 arithmetic 则要求 prefix-packed `issue_slot_valid`、显式 `issue_bank_id` 和 unique-bank
检查，所以 **M184 与 M183 不能直连**。如果加 compactor/bank-ID adapter，会把 M184 刚去掉的
部分成本带回来。下一步应直接做真正的 structural fixed-bank K8 accumulator/weight-response
接口，再进行组合 VCS/DC。

## P0 / P1 / P2

P0：

1. 做 M184 mask-native fixed-bank accumulator/weight-response composition；在完成前不承认
   complete FC2。
2. `97,607,807`、`1.307080x`、`4.344534x` 均保持 exact-payload analytic 口径；
   `1.879496x` 只能写成 matched same-clock conditional frontend screen，禁止 physical、
   system、paper-PPA-ready 或 headline。

P1：

1. pipeline/register `group_ready` 到 selector/load 的路径，预提交非零 timing margin 后重跑。
2. 把四档 multi-window partial-final case 和 explicit partial-close cover 纳入主 exact 回归。
3. 把四档 header extent 和 descriptor extent attack 纳入主 exact 回归。
4. 明确定义 M182 cycle interval，并做 frozen representative/all-payload materialized-group RTL
   replay。
5. 接八 bank weight response，包含 SRAM address/read stall 与 backpressure，再报告组合周期和
   宏面积。

P2：

1. 增加 window event conservation、buffer ownership、partial close 的 ghost/SVA invariant。
2. 清理 5 个 VER-318 signed/unsigned warning，并披露 tie-high/CTS 尚未处理。
3. 如需统一 admission，新增 keyed-to-RUN_COMPLETE overlay，不修改 precommit contract。

机器可读裁决见 `m184_independent_hammer_review_r1.json`。本评审只写当前 review 目录，未修改
M184 主线或 `docs/359`。
