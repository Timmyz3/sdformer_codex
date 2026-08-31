# C2 typed-ticket tag-elision pre-RTL r2 独立静态打铁请求

日期：2026-08-27  
被审对象：`reviews/c2_ticket_elision_prertl_spec_r2_20260827`  
请求状态：`AWAITING_DIFFERENT_INDEPENDENT_SOURCE_ONLY_HAMMER__NO_RTL_OR_EDA_AUTHORIZED`

## 评审边界

本请求只允许 source/contract 静态审计。不得修改 RTL，不得运行 VCS/DC/PT/PTPX/
Formality 或任何开源 RTL 工具，不得修改 `docs/359`。r2 静态 PASS 也不直接授权
RTL；它只证明此规格在 M519 R5 三轴 DC 独立双封通过后，可用于创建一份新的
exact-SHA RTL author admission。

## 必查 P0/P1

1. 中央 owner 必须是既有 `token_tag_q`；当前范围必须是 single-token。`sb_tag_q`
   在 candidate 中可删/可自然综合掉，严禁 `dont_touch`/dead state/assertion-only consumer 抬高收益。
2. production M519 必须明确 `soft_flush=0`；只有 transport-local harness 可暴露 flush/drain/ack。
   A06/A17 的 flush 覆盖不得写成 production runtime feature。
3. 必须先从 M490 clone **tagged R5-precedence reference**，仅继承 M499 R5 channel-local
   precedence；对 legal oracle 与 R5 attack oracle 双封后，才能从它 clone elided 点。A/B
   唯一变量必须是 tag transport/state/compare/mux。
4. 必须建立同 L4/O8/1R1W/128-bit/18-bit 地址/macro 边界/调度的两棵可综合 leaf
   shell。`tb_m349` 与其 1536 logical tag bits 不得当 production PPA 分母。
5. 功能与 DC 时序 clean 后，无论 area 是否达 8%/15%，必须运行一次 matched
   mapped-gate SAIF/PTPX。检查 area 与 dynamic-power 两轴均可到达 PROMOTE/KEEP/NO-GO。
6. 验证 ghost-tag scoreboard 定义可执行：每 bank/slot 在 request accept 存完整影子
   ticket/tag，response accept 用 pre-edge ticket 唯一查找，same-edge reuse 新 write wins，bundle retire
   时 expected-bank tag 全等且等于 `token_tag_q`。
7. tagged-reference wrong-tag negative test 必须单列；candidate 不得因无 wrong-tag port 获得 coverage 加分。
8. 机械统计 A01–A18 恰好 18 类，每类有 scope；production/local flush 不得混用。
9. K8 local A/B 不得更新 K8-vs-K1x8 主表；只有 K1x8 对称提供同一优化后才能更新
   throughput/mm2。周期 delta 必须为 0，不得与 K1 倍率相乘。
10. M519 R5 K1/K8/K1x8 三轴 DC 独立双封结果必须是 RTL author 的硬前置；
    本规格与本评审请求都不得绕过它。

## 必查口径

- local metadata `27.5346%` 是 static favorable upper bound，不是 area/power/energy；
- area 分母是同一 transport hierarchy total mapped cell area，sequential 是 noncombinational area；
- clean timing 是 setup/hold/五类 constraint 全部二值 clean，同时 `TIM-209/OPT-150=0`；
- SAIF 要求 exact net/leaf annotation=100% 且额外报 nonzero-toggle coverage，同一 contiguous window；
- `PROMOTE`: clean + seq area >=10% + (total area >=15% 或 dynamic >=20%)；
- `KEEP`: clean + 未 promote + (area >=8% 或 dynamic >=10%)；
- `NO_GO`: clean + area <8% 并且 dynamic <10%；
- 任一功能/cycle/traffic/timing/constraint 失败都是 P0，不能用物理数字抵消。

## 预期独立产物

若且仅若 P0=0 且 P1=0，请创建并双封：

- `reviews/c2_ticket_elision_prertl_spec_static_hammer_r2_20260827/`
- JSON：`c2_ticket_elision_prertl_spec_static_hammer_verdict_r2.json`
- MD：`c2_ticket_elision_prertl_spec_static_hammer_r2_20260827.md`
- `SOURCE_SHA256SUMS`、`mechanical_checks.txt`、`RUN_COMPLETE`、`SHA256SUMS`、
  `SHA256SUMS.seal.sha256`
- 成功 status：
  `STATIC_SPEC_PASS__WAIT_FOR_M519_R5_THREE_AXIS_DC_BEFORE_RTL_AUTHORING`

成功评审仍必须写：`rtl_authorized=false`、`vcs=false`、`dc=false`、`ptpx=false`、
`cycle_speedup=false`、`system_speedup=false`、`paper_ppa_ready=false`。

## 身份

- spec JSON SHA256：`2f44f2600295d57b96114dd1a3c622eeae1bba1604929fd06634235188d99b80`
- spec MD SHA256：`a647e80bcd64ababdbbb40bcf123e0ab4920dacc39830a5864020f065bed22d6`
- SOURCE evidence SHA256：`30225b689b2e28f495d46a295300995d80a8e50ad986e6783146006e402c24d3`
- spec inner manifest SHA256：`dc6cb7b8608eebbd5c594153f966614240d76d64be79589f7843fb67143a0a3f`
- `docs/359` SHA256：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`
