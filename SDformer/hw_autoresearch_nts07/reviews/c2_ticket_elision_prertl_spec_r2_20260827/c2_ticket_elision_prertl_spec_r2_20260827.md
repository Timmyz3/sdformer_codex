# C2 typed-ticket tag-elision pre-RTL 规格 r2

日期：2026-08-27  
对象：Motion/H67 FC2 signed-source C2 内部 tag transport 物理消融  
模式：仅修订规格/合同；未修改 RTL，未运行 VCS/DC/PT/PTPX/Formality，未修改 `docs/359`  
裁定：`REVISED_SPEC_READY_FOR_INDEPENDENT_STATIC_HAMMER__RTL_BLOCKED_UNTIL_M519_R5_THREE_AXIS_DC`

## 一句话结论

r2 仅授权对一个问题做后续物理 A/B：在当前 **single-token** H67 语义下，
`token_tag_q` 已是唯一可观测语义 owner，独立 backpressure 的 weight-bank leaf 是否可以只携带
`{implicit_bank,epoch16,generation32,slot3}` 而不再复制 `tag24`。候选的目标是 cell area 或
dynamic power，**周期必须为 0 变化**。

`27.5346%` 仍只是对候选有利的 local metadata movement 上界，不是 measured area/power/
energy。r2 修复 r1 的五个 P1，但不授权现在开 RTL：必须先等 M519 R5 K1/K8/K1x8
三轴 3 ns DC 结果完成且经独立双封评审准入。

## 1. 范围锁定：当前 single-token，不假装 multi-token

### 1.1 中央 owner 只能是 `token_tag_q`

冻结 M218 一次只有一个 `token_active_q`；合法 `group_tag` 必须等于 `token_tag_q`，
`result_tag` 和 `token_done_tag` 也都直接来自 `token_tag_q`。因此 r2 锁定：

- candidate 的语义 tag owner 是既有 `token_tag_q`，它有真实 result/done 消费者；
- tag-elided service 中 `sb_tag_q[0:7]` 应删除或任其因无功能消费者被综合自然删除；
- 合法 response 先通过 ticket/bank-mask 校验，然后其语义 output tag 为 `token_tag_q`；
- 不得用 `dont_touch`/`keep`/dead state/仅 assertion 消费者人工保留 `sb_tag_q`，也不得将其冒充物理节省的分母。

未来 multi-token/per-slot tag owner 明确不在本轮范围。若以后需要，必须先定义两个不同 tag
同时 outstanding 的真实架构消费者，并重新立项；不能用 r2 做授权。

### 1.2 保留与删除边界

tagged 与 elided 两点必须保留完全相同的 address18、epoch16、generation32、slot3、
expected/arrived bank mask、signed INT8 weight128/bank/cycle、L4、O8/FIFO4、K8、Acc24、stall 和
result/done tuple。候选只删除：

- service↔adapter↔leaf 的 request/response `tag24` transport；
- adapter `pending_tag_q` 与 `slot_tag_q`、tag mux 及 equality；
- tagged leaf shell 中的 per-slot tag state/echo；
- candidate service 中只用于 memory-response tag 比较的 `sb_tag_q`。

r1 的 216-bit adapter state 与 1536-bit TB leaf state 不得直接写成 production savings。只有下文的
matched synthesizable leaf shell 经 DC/PTPX 后，才能写 mapped 结果。

## 2. production 与 transport-local flush 严格分列

| 边界 | `soft_flush` | 允许覆盖 | 禁止主张 |
|---|---:|---|---|
| production M519 standalone/K8/K1x8 | 恒为 0 | reset、wrong epoch/generation、stale、reorder、duplicate、slot reuse、R5 同拍 fault precedence | 不得声称 runtime soft-flush |
| transport-local attack harness | 显式可控 | M218 原有 flush/drain/ack、old epoch、pending leaf、ack timeout | 只能称 local protocol stress，不得外推 production feature |

tag-elision A/B 不得给 M519 增加 production flush 端口。transport-local tagged/elided harness 使用相同
flush 激励与 leaf drain/ack 规则；full-M519 物理点仍将 `soft_flush=0`。

## 3. 唯一变量 A/B：先冻结 tagged R5-M490 reference

M490 是 K8 cut-through 数据路，M499 R5 是 channel-local request-fault/response-retirement precedence 的已验证
来源。不能直接拿旧 M490 与新 candidate 比，否则 fault precedence 与 tag transport 同时改变。

必须按以下顺序：

1. clone M490 为 **tagged R5-precedence reference**，仅移入 M499 R5 的
   `response_channel_open=!fault_q&&!illegal_response` 与
   `request_channel_open=response_channel_open&&!illegal_request` 及同拍 state-mutation precedence；
2. 用 M519 R5 attack oracle 证明该 tagged reference 对 legal traffic/cycles/tuples 与冻结 M490 一致，
   对 15/16 号同拍 fault 与 M499 R5 一致；然后双封冻结此 reference；
3. 从这个已冻结 reference clone elided adapter/service/leaf，仅删 tag transport/state/compare/mux；
4. A/B 的算术、scheduler、queue、backpressure、fault precedence、参数、约束、trace 和工具 SHA 全部相同。

不得原地修改 M218/M490/M499/M519。不得将 R5 precedence 修复的面积或功耗混入
tag-elision 收益。

## 4. matched 可综合 leaf shell，不再用 TB 的 1536 bit

建立 tagged/elided 两棵完整的 synthesizable scalar-leaf shell：

- 同一 L4 response latency、O8 live slots、1R1W 调度、128-bit data、18-bit address 和 hard-SRAM/black-box 边界；
- 同一 `valid/due/epoch/generation/block/slice/channel` 元数据和 arbitration/hold 逻辑；
- tagged shell 对每个 O8 entry 存储/回显 `tag24`，elided shell 无 tag 端口/状态/回显；
- 两点由同一 matched leaf wrapper 选择，macro data array 完全相同且分账；metadata shell 不得当免费；
- 禁止把 `tb_m349` 的 `integer cycle_q/due_q`、程序化 `weight_value` 或 1536-bit `tag_q`
  直接当作 production leaf/PPA 证据。

物理 A/B 必须使用这对 shell；不允许 tagged 一侧用 TB model，elided 一侧用可综合 shell。

## 5. 可执行 ghost-tag shadow scoreboard

ghost-tag miter 必须是 TB/SVA 中可执行的 reference model，不是文字假设：

1. 定义 `ghost_valid[bank][slot]`、`ghost_epoch`、`ghost_generation`、`ghost_tag`；
2. 每个 `bank_req_accept[bank]` 在该 bank/slot 记录完整 ticket 和 tagged-reference request tag；
3. 每个 `bank_rsp_accept[bank]` 用 **pre-edge** `implicit_bank+epoch+generation+slot` 查找，必须唯一命中；
4. 同拍 final response + same-slot new request 按 retire-before-write 观测、new write wins 更新，与 RTL nonblocking 语义一致；
5. bundle retire 时，expected mask 中每个 bank 的 ghost tag 必须相同且等于 `token_tag_q`；
6. legal traffic 下 tagged/elided 的 cycle、accept、bank traffic、Acc24、result/done multiset 全相等；
7. 另建 tagged-reference-only negative test：ticket/data 不变、仅翻转 response tag，tagged reference 必须
   fault 且零算术 side effect。candidate 无 wrong-tag port，此 negative test 不得算 candidate 额外 coverage。

## 6. 18 类攻击与 scope

| ID | 攻击 | scope |
|---|---|---|
| A01 | response-before-request | production + local |
| A02 | invalid/out-of-range slot | production + local |
| A03 | wrong epoch | production + local |
| A04 | wrong generation | production + local |
| A05 | old generation after same-slot reuse | production + local |
| A06 | old epoch after soft flush | transport-local only |
| A07 | duplicate response from one bank | production + local |
| A08 | response from bank outside expected mask | production + local |
| A09 | partial-bank response reorder within one slot | production + local |
| A10 | newest/oldest slot reorder across bundles | production + local |
| A11 | all eight banks complete together | production + local |
| A12 | final-beat cut-through plus same-slot request presentation | production + local |
| A13 | held response under backpressure plus reuse attempt | production + local |
| A14 | partial request fanout plus bank backpressure | production + local |
| A15 | legal response plus malformed request on same edge | production + local; R5 precedence |
| A16 | illegal response plus otherwise legal request on same edge | production + local; R5 precedence |
| A17 | pending leaf state during control restart | production reset subcase + local soft-flush/drain/ack subcase，分别计数 |
| A18 | two successive different header tags reuse same slot | production + local; ghost owner 不串 tag |

每个 ID 都要 assertion 和非零 cover。A06 与 A17 soft-flush 子例只证明 transport-local harness；
production 只记 A17 reset 子例。所有 assertion failure、ghost mismatch、Acc24/tuple mismatch 均必须为 0。

## 7. M519 R5 三轴 DC 是 author RTL 的硬前置

本 r2 规格通过静态评审也不等于可以开 RTL。必须先有新的独立双封回执证明 M519 R5
K1/K8/K1x8 三点：

- DC V-2023.12-SP3，TSMC28 HPC+ `ssg0p9v125c` / `ffg1p05vm40c`，3.000 ns；
- 每点 precompile `TIM-209=0` 且 `OPT-150=0`；
- setup/hold WNS clean，五类 constraint 零 violation，terminal PASS 且原始树双封；
- K1/K8/K1x8 面积、时序与同资源分母可用。

在此前置成立前，`rtl_authorized=false`、`vcs_authorized=false`、`dc_authorized=false`、
`ptpx_authorized=false`。

## 8. matched VCS→DC→mapped-gate SAIF/PTPX 顺序

1. clone-only author tagged R5 reference 和 elided candidate，独立 static hammer P0/P1=0；
2. 运行 legal ghost-tag miter、18 类攻击、tagged wrong-tag negative test；独立 VCS receipt hammer P0=0；
3. 一次 paired DC 同时报 transport-local 与 full-K8；只有所有 setup/hold/五类约束/
   `TIM-209/OPT-150` 全 clean 才进入下一步；
4. **无论 DC area 是否达到 8%/15% 门，只要功能与时序 clean，必须无条件完成一次
   matched mapped-gate VCS SAIF + PTPX**；不允许因 area 小而跳过 power 轴；
5. 两点使用同一 contiguous measurement window、reset exclusion、stall/traffic trace 和
   `tt0p9v25c`；exact net/leaf SAIF annotation=100%，同时报 nonzero-toggle net/leaf coverage；
6. PTPX 报 internal、switching、dynamic=`internal+switching`、leakage、total；只能称 selected transport
   slice/stdcell power，macro 未入账时不得称 full-FC2 energy。

DC 面积分母固定为同一 transport hierarchy 的 total mapped cell area；sequential 使用
noncombinational area，不用 FF count。full-K8 必须检查 setup/hold/五类约束二值 clean，
不得用“频率回退不超 1%”代替。

## 9. 三档物理门（面积与功耗两轴均可达）

以 `A=(tagged_area-elided_area)/tagged_area`、
`P=(tagged_dynamic-elided_dynamic)/tagged_dynamic` 计算：

| 状态 | 必须条件 | 处理 |
|---|---|---|
| `PROMOTE_C2_SUBMECHANISM` | 功能/时序/cycle/traffic 全 clean，full-K8 面积/功耗无 >1% 回退，sequential area reduction `>=10%`，且 `A>=15%` **或** `P>=20%` | 作 C2 子机制与局部物理消融；不新增 C4 |
| `KEEP_C2_IMPLEMENTATION_DETAIL` | 上述 guardrail clean，未达 promote，但 `A>=8%` **或** `P>=10%` | 正文实现细节或附录，不进摘要数字 |
| `NO_GO_PHYSICAL` | guardrail clean 且 `A<8%` **并且** `P<10%` | 双封负结果，不再扩 RTL |

任一 ticket/tag/Acc24/result/done mismatch、cycle/traffic 变化、setup/hold/约束违规、
`TIM-209/OPT-150` 非零都是 `P0_FAIL`，不能用面积/功耗数字救回。若局部点达门但
full-K8 改善很小，只能写 local protocol ablation。

## 10. K8/K1x8 公平性

- 首先可做 K8 local tagged-vs-elided A/B，因为它只用来裁决该子机制；
- 在没有给 equal-bandwidth K1x8 对称提供同一 tag-elision 优化前，不得更新
  K8-vs-K1x8 throughput/mm2 主表；
- 若 K8 local 至少达 `KEEP`，后续 K1x8 只能用同一 leaf shell、相同时钟/带宽/工具/
  trace 与同一 owner 范围对称实现；
- tag-elision 周期必须不变，不得与 C2 K1 周期倍率相乘。

## 11. DATE claim 边界

只有物理门通过后，允许写入 C2：

> Within C2's single-token FC2 transport, a typed epoch-generation-slot
> ticket lets independently backpressured weight-bank leaves elide replicated
> tag state and switching, while the existing token-level owner preserves
> semantic result identity under an unchanged signed-Acc24 schedule.

必须引用 ELSA 和 FireFly-T，不得写 `first`、不得把 bundled AER/bank dispatch 当我方原创。
`27.53%` 只能写 static local upper bound，不能写 measured energy。

本规格当前标签固定为：`headline=false`、`cycle_speedup=false`、`system_speedup=false`、
`C4=false`、`paper_ppa_ready=false`。

## 12. 本 r2 的授权语义

本目录只是 author pre-RTL 规格。下一步只能由不同 reviewer 做 source-only static hammer；
该评审即使 P0/P1=0，也只表示规格可作为 **M519 R5 DC 之后** 的 RTL author 合同，
不直接授权 VCS/DC/PTPX。
