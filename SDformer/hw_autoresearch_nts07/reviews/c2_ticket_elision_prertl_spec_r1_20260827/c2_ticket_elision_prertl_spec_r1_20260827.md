# C2 tag-elision 条件性物理优化 pre-RTL 规格 r1

日期：2026-08-27  
对象：Motion/H67 FC2 signed-source C2；M519 R5 完成后的低风险物理优化  
模式：只读 RTL/合同审计；未修改 M519 R5，未运行 VCS/DC/PT/PTPX，未修改 `docs/359`  
裁定：`CONDITIONAL_GO_AFTER_M519_R5__PHYSICAL_ABLATION_ONLY`

## 技术摘要：27.53% 不应直接杀，但不能被写成周期加速

前一轮的 `27.5346%` 是对候选有利的 **局部 metadata movement 上界**，不是全 FC2
或全网节省。它没有通过人为设定的 30% fast-kill 门，只能说明不能靠 bit 账本直接升格，
不能说明物理优化无价值。本规格把状态由“默认停止”改成 **条件性物理候选**：M519 R5
闭合以后允许一次 clone-only RTL/VCS/DC/SAIF/PTPX A/B；是否保留由 matched cell area、
sequential area 和动态功耗裁决。

该机制预期 **周期严格不变**。它的目标是从八个独立 weight-bank leaf 和响应 join 中删除
重复 `tag24`，保留中央 scoreboard 中唯一的 tag owner，并用
`ticket={epoch16,generation32,slot3}` 恢复 tag。任何后续表述只能是 C2 的协议压缩/物理效率
子机制，不能成为 C4、不能称新的稀疏机制、不能称 cycle/system speedup。

## 已知账本与物理假设

| 项 | tagged baseline | tag-elided candidate | 已知变化 | 证据边界 |
|---|---:|---:|---:|---|
| leaf request metadata / active bank | 93 bit | 69 bit | -25.81% | 删除 tag24；address18 与 ticket51 保留 |
| leaf response metadata / active bank | 75 bit | 51 bit | -32.00% | 删除 tag24 |
| request+response interface movement | 168n | 120n | -28.57% | `n` 为 active banks/bundle |
| 含 leaf/central entry 写读的上界，平均 `n=5.6268` | 2136.13 bit/transaction | 1547.96 bit/transaction | -27.5346% | 未计不变 scoreboard/mask/due，因此仍是有利上界 |
| M490 pending+8 slot tag state | 216 bit | 0 bit | -216 logical state bit | `pending_tag_q` 24 + `slot_tag_q` 8×24 |
| 八个 O8 scalar leaf tag entry | 1536 bit | 0 bit | -1536 logical state bit | `8 banks × 8 slots × 24 bit` |
| tag equality width | 216 bit | 0 bit | -216 equality-input bit | adapter 8×24 + central response 24 |
| M218 central `sb_tag_q` | 192 bit | 192 bit | 不变 | tag 的唯一 owner，8 slots×24 |

以上最后三行是 **pre-RTL logical ledger**，不是 cell area、功耗或能量。综合可能共享、删除或
重构逻辑；只有 matched DC/PTPX 才能裁决物理价值。

## 最小 RTL delta：只移动身份所有权，不碰算术或调度

实施必须 clone 新 namespace；不得原地改 M218、M490、M499、M519 R5。建议内部候选名
`C2A_TAG_ELIDED_TYPED_TICKET`，正式 M 编号由主线分配。

### 1. 中央 service clone

从冻结 M218/M519 K8 service clone，只做这些变化：

- 保留 header/group/result/done 的 `TAG_BITS=24`；保留 `sb_tag_q[0:7]`。
- memory request/response 内部协议删除 `mem_req_tag` 和 `mem_rsp_tag`。
- response legality 从
  `{slot-valid,epoch,generation,tag,bank-mask}` 改成
  `{slot-valid,epoch,generation,bank-mask}`。
- 合法 response 的 tag 必须由 `sb_tag_q[mem_rsp_slot]` 恢复；不得从 leaf 猜测、常量化，
  也不得直接用未验证的当前 token tag 代替 per-slot owner。
- group FIFO、block/slice context、Acc24、response skid、O8/FIFO4、generation/epoch 规则均不变。

### 2. K8 bundle-to-bank adapter clone

从 M490 语义 clone；同时继承 M519 R5 已证明的 channel-local fault precedence，但 A/B 两点
必须使用完全相同的 precedence，避免把故障修复混入 tag-elision 收益。

- 删除 core/leaf 两侧 request/response 的 tag24 transport。
- 删除 `pending_tag_q`、`slot_tag_q[0:7]`、bank response tag comparator 和 response tag mux。
- 保留 `pending/slot {epoch,generation,slot}`、expected/arrived mask、weight store、cut-through、
  response hold 和所有 bank/channel/block/slice 范围检查。
- request address `output_block3+slice3+source_channel12` 不得删除或压缩。
- request fault 不得撤销同拍独立合法 response；illegal response 同拍关闭 request/response；
  sticky fault 后所有 side effect 为零。

### 3. scalar leaf protocol clone

从 M349 的协议状态语义提取可综合 leaf shell；weight SRAM/data array 仍为外部 macro/black box。

- 每 bank、每 live slot 保留 pending/due、epoch16、generation32、block3、slice3、channel12；
  删除 entry `tag24`。
- response 回显 `{epoch,generation,slot}` 和 weight128，不回显 tag。
- L4、1R1W、八 bank、每 bank O8、ready/valid/backpressure 行为不变。
- 不得把 weight SRAM、due scheduler 或外部 response latency 当免费资源。

### 4. 顶层保持不变的部分

M214/M216 compactor/frontend、signed INT8 weight、K8 bank mask、六个 16-lane slice、Acc24、
result/done tuple 均保持 bit-exact；不增加 matcher、router 或第四条 Conv 支线。

## 安全不变量：ticket 必须足以唯一拥有 response

| 编号 | 必须成立的不变量 | 失败后处理 |
|---|---|---|
| I1 | live request 的 ticket 为 `{epoch,generation,slot}`，三字段全部比较 | response 不 accept，sticky fault+stale |
| I2 | generation 单调递增且禁止 wrap；slot 复用时 generation 必须变化 | generation exhausted 时 fail-closed |
| I3 | soft flush 增 epoch、清 scoreboard，并在正确 ack 前禁止新 token | old epoch response 只 drain/stale，不更新 Acc24 |
| I4 | leaf response bank 必须在 expected mask，且 arrived bit 尚未置位 | duplicate/unexpected bank 零 side effect |
| I5 | bundle 仅在 `arrived|incoming == expected` 时完成一次 | 不得 early-complete 或 double-retire |
| I6 | held/cut-through response 的 slot owner 在 core accept 前不可复用 | payload/ticket 在 backpressure 下稳定 |
| I7 | 恢复 tag 必须等于 matched request 写入的 `sb_tag_q[slot]` | 任何 tag mismatch 为 P0 |
| I8 | request partial fanout 不重发已 accept bank，也不丢 pending bank | request/bank ledger 精确守恒 |
| I9 | legal response+illegal request 同拍时，response 可恰好 retire 一次；request 零 side effect | 继承 R5 channel-local fault contract |
| I10 | illegal response+legal request 同拍时两通道均零 side effect | sticky fault/stale |
| I11 | bank request/response、Acc24 write、result/done multiset 与 tagged baseline 完全相同 | 任一 mismatch 立即 NO-GO |
| I12 | 合法 workload 的 cycle/accept 序列逐周期相同 | 不允许用 tag-elision 声称 cycle gain |

ticket 足够的前提不是“slot3 单独够用”，而是：slot live ownership、不可 wrap 的 generation、
flush epoch 三者共同构成唯一身份。tag 是 workload 语义，不再由不可信/分布式 leaf 回显，
而由合法 ticket 索引中央 owner 恢复。该协议不是密码学校验；伪造完整 ticket 的威胁模型
与 tagged baseline 同样不受保护，不能写成安全增强。

## 必须覆盖的 VCS/SVA 攻击

在 tagged/elided legal-traffic miter 之外，至少覆盖以下 18 类；assertion fail 必须为 0：

1. response-before-request；
2. invalid/out-of-range slot；
3. wrong epoch；
4. wrong generation；
5. old generation after same-slot reuse；
6. old epoch after soft flush；
7. duplicate response from one bank；
8. response from bank outside expected mask；
9. partial-bank response reorder within one slot；
10. newest/oldest slot reorder across bundles；
11. all eight banks completing together；
12. final-beat cut-through plus same-slot request presentation；
13. held response under core backpressure plus reuse attempt；
14. partial request fanout plus bank backpressure；
15. legal response plus malformed request on the same edge；
16. illegal response plus otherwise legal request on the same edge；
17. reset/soft-flush while leaf responses remain pending；
18. two successive, different header tags reusing the same slot，证明中央恢复不串 tag。

删除 tag port 后不得把“wrong tag attack 不存在”算作覆盖提升；应由 wrong
epoch/generation/slot、flush 和 reuse 攻击替代。另需 ghost-tag 参考 miter：在合法 ticket
约束下，candidate 恢复的 tag 必须等于 tagged baseline 的 response tag。

## matched DC/SAIF/PTPX 实施方案

### 前置门

只有 M519 R5 完成以下闭环后才开始：独立 static P0=0、三阶段 VCS P0=0、K1/K8/K1x8
三点 precompile `TIM-209=0/OPT-150=0`、3 ns DC clean、原始树双封存。当前 R5 合同仍是
author draft；本规格不授权 EDA。

### A/B top 与资源

建立同一个参数化 matched top，`ELIDE_TAG=0` 为 tagged reference，`ELIDE_TAG=1` 为
candidate。两点必须满足：

- 完全相同的 top-level port list；candidate 的 normalization shell 可保留 tag fault-injection
  pin 但内部不得消费，防止 pin 数差污染 stdcell 比较；生产 top 再删除这些 pin。
- 相同八个 leaf shell、八个 1R1W macro black-box 边界、L4、O8/FIFO4、Acc24、K8、
  128-bit/bank/cycle payload、input/output delay/load。
- 相同合法请求/响应/stall/flush trace；request、response、Acc24、result、done tuple 逐拍相同。
- transport-local top 保留相同 leaf hierarchy，反映 hard-SRAM 边界；另跑 full M519 K8
  flattened top，防止局部好看、全 top 无收益。
- 若只优化 K8 而未给 K1x8 应用等价 tag-elision，结果只能作 K8 local ablation，不能更新
  K8-vs-K1x8 throughput/mm² 主表。更新主表前必须给三轴对称地提供该优化或明确排除它。

冻结工具口径：Synopsys DC V-2023.12-SP3；TSMC28 HPC+ slow
`ssg0p9v125c` + fast min `ffg1p05vm40c`；3.000 ns；ideal clock；ZeroWireload；setup
uncertainty 0.200 ns、hold 0.100→0.090 ns；0.250 ns I/O delay；0.100 ns input transition；
0.010 pF output load；max fanout 32；macro count=0 的 stdcell 账与 macro-shell 账分列。

DC 必须分别报告 total/combinational/noncombinational area、leaf/sequential cell count、logic
levels、setup/hold WNS、五类 constraint、TIM-209/OPT-150、每层 hierarchy area。不得用
port count 或 logical bit 账替代 cell area。

功耗用相同 VCS trace 生成 mapped-gate SAIF，PTPX 在 `tt0p9v25c` 报 internal/switching/
leakage/total；两点 exact net/leaf annotation 均要求 100%，SAIF duration 与 reset window 相同。
只可称 selected transport slice/stdcell power；0 macro 时不得称完整 FC2 energy。

## 物理裁决门：保留中间档，不再用 30% 一刀切

| 状态 | matched 结果 | 论文处理 |
|---|---|---|
| `PROMOTE_C2_SUBMECHANISM` | transport-local cell area `>=15%`，或 transport-local dynamic power `>=20%`；同时 sequential area `>=10%`、full K8 area/power/3ns 均无 `>1%` 回退、cycle/traffic 完全相同 | 写进 C2 微结构与消融；可报告 throughput/mm² 或 energy/op 改善，不作新贡献编号 |
| `KEEP_C2_IMPLEMENTATION_DETAIL` | area `8–15%` 或 dynamic power `10–20%`，且无功能/时序/traffic 回退 | 正文实现细节或附录；不进摘要数字 |
| `NO_GO_PHYSICAL` | area `<8%` 且 dynamic power `<10%`，或 setup/hold/traffic/cycle 任一越界 | 封负结果，不再 RTL 扩展 |
| `P0_FAIL` | ticket/tag/Acc24/result/done mismatch，stale/reuse/flush 攻击漏检，TIM-209/OPT-150 非零 | 立即停止，不能引用物理数字 |

sequential cell count 只能解释 area/power 来源，不能单独升格。若 isolated transport 过门但
full K8 总功耗/面积变化不足，也只能作为 C2 局部实现优化，不得包装成整个 accelerator
energy efficiency。

## 实施风险与工期

M519 R5 后属于低到中风险：clone 两个协议模块、一个 leaf shell、一个 matched wrapper，
不碰算术、不换 checkpoint、不改 trace。预计 author RTL+SVA/TB 约 4–6 小时，独立 static
与一次 VCS 约 2–3 小时，matched DC 受队列影响约 3–6 小时，SAIF/PTPX 再约 3–5 小时。
若只做 VCS+DC 决策，约一个工作日；过 DC 门后才投入功耗闭环。

## DATE claim 边界

合法写法（只有物理门通过后）：

> Within C2, a central typed-ticket owner reconstructs token identity while
> tag-elided, independently backpressured weight-bank leaves carry only address
> and stale-safe epoch/generation/slot state, reducing replicated control area or
> switching energy under an unchanged K8 schedule.

必须引用 ELSA 的 bundled AER 公共身份摊销和 FireFly-T 的 bank-aware dispatch；我方差异是
signed analog ATLIF、八个独立 1R1W leaf、O8/flush/stale/reorder fail-closed 协议与 Acc24。
不得写 `first`，不得把 `27.53%` 写成 measured energy，不能把该数字与 C2 的 K1 倍率相乘。

本机制永远属于 C2 子机制：`system_speedup=false`、`cycle_speedup=false`、`C4=false`、
`paper_ppa_ready=false`，直到各自的显式证据门改变对应标签。

## 局限与下一步

当前没有新 RTL、VCS、DC 或功耗结果；27.53% 仍只是静态上界。最合理的下一步是先完成
M519 R5，随后做一次 exact-SHA clone-only implementation 和独立 static hammer。若 VCS
bit-exact，再运行同一 one-shot runner 内的 paired DC；面积不过 8% 才真正物理淘汰，
而不是因为差 2.47 个百分点于任意 metadata 门就提前杀掉。

