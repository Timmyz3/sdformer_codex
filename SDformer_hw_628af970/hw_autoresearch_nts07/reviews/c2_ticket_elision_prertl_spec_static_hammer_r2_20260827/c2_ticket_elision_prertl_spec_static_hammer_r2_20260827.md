# C2 typed-ticket tag-elision pre-RTL r2 独立静态打铁

日期：2026-08-27  
被审对象：`reviews/c2_ticket_elision_prertl_spec_r2_20260827`  
状态：`STATIC_SPEC_PASS__WAIT_FOR_M519_R5_THREE_AXIS_DC_BEFORE_RTL_AUTHORING`  
评分：**99/100**；P0/P1/P2 = **0/0/1**。

## 裁决

r2 已关闭 r1 的五个 P1，可以作为**未来 RTL author admission 的规格输入**；但它本身不授权
RTL、VCS、DC、PTPX 或任何性能/物理 claim。当前 M519 R5 三轴 DC canonical 与独立 receipt
仍不存在，因而必须继续等待该硬前置条件。

本评审独立核实：当前 H67 是 single-token；`token_tag_q` 是 result/done/legality 的真实语义
owner；`{implicit_bank,epoch16,generation32,slot3}` 在既有 stale/reorder/duplicate 威胁模型下足以
唯一识别 leaf response。`27.5346%` 仍只是静态、局部、对候选有利的 movement 上界，不是
area、power、energy、cycle 或 system speedup。

## r1 五个 P1 回归

### P1-01｜owner 死状态：关闭

M218 只在 `token_active_q=0` 且 FIFO/outstanding/skid/emit/done 全空时接收新 header；合法
group/frontend-done tag 必须等于 `token_tag_q`，result/done tag 也直接由它驱动。r2 因此正确
锁定 single-token owner=`token_tag_q`，明确删除或自然综合掉 candidate 的 `sb_tag_q`，禁止
`dont_touch`、`keep`、debug/assertion-only consumer，也禁止 multi-token claim。

### P1-02｜production/local flush 混用：关闭

源码确认 M519 standalone 的 `svc_soft_flush=1'b0`，K1x8 八条 lane 也均绑 0。r2 将 A06 和
A17 soft-flush/drain/ack 限于 transport-local harness；production 只覆盖 reset、wrong
epoch/generation、stale/reorder/duplicate/reuse 和 R5 precedence，不新增 production flush port，
也不宣称 runtime soft-flush。

### P1-03｜R5 precedence 混入 A/B：关闭

r2 要求先从 M490 clone tagged R5-precedence reference，仅移入 M499 R5 的 request/response
channel-local precedence；其 legal traffic/cycle/tuple 对冻结 M490，A15/A16 对 M499 R5 oracle，
双封后才派生 elided 点。A/B 唯一变量明确为 tag transport/state/equality/mux，原
M218/M490/M499/M519 均不得原地修改。

### P1-04｜TB leaf 冒充物理 leaf：关闭

r2 明确 `tb_m349` 的 integer `cycle_q/due_q`、程序化 `weight_value` 与 1536 logical tag bit
都不能进入 production PPA 分母。tagged/elided 必须使用同 L4/O8/1R1W/128-bit/18-bit address、
同 arbitration/hold/scheduler、同 hard-SRAM black-box boundary 的两棵可综合 leaf shell；metadata
shell 不免费、macro data array 分账。

### P1-05｜功耗门不可达：关闭

r2 明确规定：功能和 DC timing clean 后，无论 area 是否达到 8%/15%，都必须完成一次 matched
mapped-gate VCS-SAIF/PTPX。这样 `A<8% && P<10%` 的 NO-GO、`A>=8% || P>=10%` 的 KEEP、
以及 seq-area>=10% 且 `A>=15% || P>=20%` 的 PROMOTE 都可被实测裁决。

## 验证合同审计

JSON 与 Markdown 中 A01--A18 **恰好 18 个、无缺号、无重复**。A06 为 local-only；A17 将
production reset 与 local soft-flush/drain-ack 分列；A15/A16 明确使用 R5 precedence。
每类都要求 assertion 与非零 cover，所有 assertion/ghost/Acc24/tuple mismatch 必须为 0。

ghost-tag miter 已达到 pre-RTL 可执行规格：

- 每 bank/slot 保存 valid、epoch、generation、tag；
- bank request accept 写影子 ticket/tag，bank response accept 用 pre-edge ticket 唯一查找；
- same-edge final response + reuse 使用 retire-before-write、new-write-wins；
- bundle retire 时 expected banks 的 ghost tag 全等且等于 `token_tag_q`；
- legal A/B 比较 cycle/accept、bank traffic、Acc24、result/done；
- tagged-only wrong-tag negative test 独立，candidate 因无 tag port 不获得 coverage 加分。

唯一 P2：后续 RTL author contract 应把 ghost entry 的普通 clear 时点和 expected-mask 来源写成
逐边事件（建议 bundle retire 清 shadow、core request accept 保存 expected mask），避免 TB 作者对
“retire-before-write”的文字产生不同实现。这不改变 r2 的身份、安全或物理门，不阻断规格 PASS。

## 物理与公平性

- paired top 要求同端口或 normalization shell、同非 tag hierarchy、同 leaf scheduler/macro
  boundary、同 trace/stall/window；
- area 分母是同 transport hierarchy 的 total mapped cell area，sequential 分母是
  noncombinational area，FF count 不得代替；
- 每点 `TIM-209/OPT-150=0`，setup/hold 和五类 constraint 二值 clean；
- PTPX dynamic=`internal+switching`，同时报 leakage/total；exact net/leaf annotation=100%，还要
  报 nonzero-toggle net/leaf coverage；
- K8 local A/B 只可裁决 C2 子机制。K1x8 未对称实现前，不得更新 K8-vs-K1x8
  throughput/mm2；cycle delta 必须为 0，不得与 K1 倍率相乘。

三档门逻辑无死区：PROMOTE、KEEP、NO-GO 均以同一 A/P 分母和 common guardrail 判定；任何
功能、cycle、traffic、timing、constraint 或 loop gate 失败直接 P0，不得用物理数字救回。

## M519 R5 前置条件与授权边界

launch admission 与 runner 身份均通过 source SHA 检查，但审阅时不存在 M519 R5 K1/K8/K1x8
三轴 DC canonical。r2 已明确要求三点均有 3.000 ns、双角、loop gate、setup/hold/五类约束、
PASS terminal、原始树双封及不同 reviewer 的 receipt review。

因此本次只准入：**规格可在上述前置完成后用于创建一份新的 exact-SHA RTL author
admission。** 当前固定：

- `rtl_authorized=false`
- `vcs_authorized=false`
- `dc_authorized=false`
- `ptpx_authorized=false`
- `cycle_speedup=false`
- `system_speedup=false`
- `paper_ppa_ready=false`

`docs/359` SHA-256 保持
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

