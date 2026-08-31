# C2 ticket/tag-elision pre-RTL 规格独立打铁 r1

日期：2026-08-27  
被审对象：`reviews/c2_ticket_elision_prertl_spec_r1_20260827`  
模式：只读源代码、合同和封存证据审计；未修改 RTL，未运行 VCS/DC/PT/PTPX，未修改 `docs/359`  
裁定：`REVISE_THEN_CONDITIONAL_GO_AFTER_M519_R5_DC`  
评分：**88/100**；P0/P1/P2 = **0/5/4**。

## 结论

`27.5346%` 没有证明面积或功耗差，因此旧的 30% metadata fast-kill 确实是**过严的代理门**，
不能据此永久杀掉本候选。现有冻结 RTL 也支持机制的核心正确性：每个物理 bank 独立，
`{epoch16,generation32,slot3}` 加隐含 bank identity 能唯一匹配一个 live leaf request；M218 的
generation 每次 request accept 后递增、到全 1 即 fail-closed，slot 在 held/cut-through
response 完成前不可复用。因此删除 leaf/adapter 的 `tag24` 在当前信任模型下是可行的。

但 r1 **还不能直接授权 RTL**。它混合了当前单-token M519、未来多-token per-slot tag owner、
未暴露的 soft-flush，以及尚不存在的可综合 leaf shell。另有一个裁决死路：DC 面积不足 8%
时按实施顺序不会运行 PTPX，却又需要知道动态功耗是否不足 10% 才能 NO-GO。先出 r2 修订规格，
并等待 M519 R5 三轴 DC 闭合，再允许一次 clone-only A/B。

最终判定是：**机制值得一次物理 A/B，但不是现在直接开 RTL；它永远只是 C2 子机制，周期与
系统倍速均为 false。**

## 已独立核实的事实

### 1. ticket 足以替代分布式 tag

- M218 对 response 已检查 slot valid、epoch、generation、tag 与完整 bank mask；候选删除的仅是
  其中由 leaf 回显的 tag 比较，slot/epoch/generation/mask 仍在。
- M218 的 `generation_q` 是全局 request 序号：每次 `mem_req_accept` 加一；到全 1 时阻止 issue
  并报错。soft flush 增 epoch、清 scoreboard，且 ack 前禁止新 token。
- M490/M499 每个 bank 的 response 还检查 expected bank、未到达 bit、epoch、generation 和
  slot。bank identity 由独立物理端口隐含，无需再编码。
- 同拍 final-beat response + same-slot request 的 nonblocking 更新顺序让新 request ownership
  覆盖旧 clear；held response 时 slot 仍 valid，复用关闭。删除 tag 不改变这条数据依赖。

所以 ticket 不是只有 `slot3`；它是 `{bank implicit,epoch,generation,slot}`。在不考虑能伪造
完整 ticket 的恶意 leaf、只处理 stale/reorder/duplication/protocol fault 的既有威胁模型下，
它与 tagged baseline 等价。该机制不是安全增强。

### 2. 27.53% 的口径重算正确，但只是上界

冻结字段宽度重算一致：

| 边界 | tagged | elided | 静态变化 |
|---|---:|---:|---:|
| active-bank request metadata | 93 bit | 69 bit | -25.806% |
| active-bank response metadata | 75 bit | 51 bit | -32.000% |
| request+response interface | 168n | 120n | -28.571% |
| 加选定 entry write/read，`n=5.6268169` | 2136.1323 bit/txn | 1547.9579 bit/txn | -27.5346% |

M490 明确含 `pending_tag_q` 24 bit 与八个 `slot_tag_q` 共 192 bit；M349 仿真 leaf 含每 bank
八个 `tag_q`。逻辑账本的 `216 + 1536 = 1752 bit` 算术正确。它没有计入不变的 weight store、
mask、due、valid 和 M218 其他状态，因此仍是对 candidate 有利的局部上界，不是 measured
area、switching、energy 或 accelerator traffic。

### 3. 冻结证据边界

M519 R5 三阶段 Synopsys VCS 已由独立 receipt-blind review 以 98/100、P0/P1=0/0 准入：
12 类 fault/recovery、10 个 gated cover 与 K1/K8/K1x8 冻结周期全部通过。三轴 DC 只有静态
launch admission，尚未完成；因此 r1 写的“R5 完成后”前置门当前仍未满足。

## P1：r2 必须修复后才准许 author RTL

### P1-01｜中央 tag owner 与当前单-token RTL 冲突

M218 同一时刻只允许一个 `token_active_q`，所有 live request 的 tag 都等于 `token_tag_q`；
`result_tag` 和 `token_done_tag` 也直接来自 `token_tag_q`。候选删除 `mem_rsp_tag` 比较后，
`sb_tag_q[0:7]` 没有自然的功能消费者，flattened DC 很可能把 192 bit 全删掉。因此 r1 同时
声称“保留 192 bit per-slot owner”和“用它恢复 tag”并不是一个可综合、可观察的合同。

r2 必须二选一：

1. **当前 H67 单-token 范围（推荐）**：明确 tag owner 是既有 `token_tag_q`，允许 dead
   `sb_tag_q` 自然消失，并重算逻辑账本；不要声称多-token per-slot recovery。
2. **未来 multi-token 范围**：先给 tag recovery 一个真实、非 `dont_touch` 的架构消费者，
   并单独验证多 tag 同时 outstanding；这已超出本轮最小实现。

禁止用 `dont_touch` 人工保留 `sb_tag_q` 来兑现旧账本，也禁止把仿真 assertion 当综合消费者。

### P1-02｜soft-flush claim 与 M519 production top 不一致

M519 standalone 和 K1x8 都把 `soft_flush` 常量绑为 0；M490/M499 也没有 flush request/ack
端口。r1 却要求 full candidate 覆盖 flush pending leaf，并写成端到端安全性质。两者不能同时
成立。

r2 应把范围拆开：transport-local tagged/elided miter 可以显式暴露 M218 flush 并使用同一
drain/ack leaf harness；full M519 flattened top 只能声称 reset、wrong epoch/generation、stale、
reorder 与 slot reuse，不能称 production runtime soft-flush。若要给 M519 增加 flush 端口，
那是另一项功能变更，不能混入 tag-elision A/B。

### P1-03｜A/B 不能把 R5 fault precedence 混进唯一变量

冻结 K8 使用 M490；R5 channel-local precedence 实际落在 M499。r1 要求候选继承 R5 语义，
但未冻结一个 R5ized tagged M490 reference。直接拿旧 M490 对比新 candidate 会同时改变 tag
和 fault precedence，物理差无法归因。

正确顺序是先 clone 一个 **tagged + R5 precedence** reference，并用 M519 R5 attack oracle
闭合；然后只从该 clone 删除 tag，A/B 其余 SHA 必须一致。原 M490/M499/M519 不得原地修改。

### P1-04｜1536 bit leaf state 来自 TB，不是已冻结生产 leaf

`m349_fc2_scalar_bank_memory_model` 位于 `tb_m349`，含 integer cycle/due 与程序化
`weight_value`，不能直接当 paper DC leaf。r1 的“可综合 leaf shell”尚不存在，因此 1536 bit
不能先写成 production sequential-state reduction。

r2 必须冻结一对真正可综合的 matched leaf shells：相同 L4/O8/1R1W、相同地址与 128-bit
macro data 边界、相同 scheduler；唯一差异是 tag port/state/comparator。宏 data array分账，
但 metadata shell 不得免费。tagged/elided 两点均由新 shell 构成，不能一边用 TB model、
一边用 RTL shell。

### P1-05｜功耗裁决路径当前不可达

r1 的实施顺序说“只有 DC 物理门通过才做 PTPX”，NO-GO 却要求 area `<8%` **且** dynamic
power `<10%`。若 area=5%，流程不会跑 PTPX，也无法证明 power<10%。

r2 必须改成：功能/时序 clean 后无论 area 是否过门，都运行一次 matched mapped-gate
SAIF/PTPX；或者删除 power 作为独立 promotion 轴。推荐保留功耗轴并总是跑一次，因为本候选
本来就是 metadata switching 优化。只有 area 与 dynamic power 两项都实测低于下门，才能
永久 NO-GO。

## P2

1. ghost-tag miter 需要定义成可执行 shadow scoreboard：每个 bank/slot 在 request accept 时
   保存 reference tag；legal response accept 时以完整 ticket 查找；bundle retire 时证明所有
   shadow tag 一致且等于中央 token owner。另做独立 negative test 翻转 tagged reference 的
   response tag；不能把 candidate 没有 wrong-tag port 计作额外 coverage。
2. matched DC 的 15%/8% denominator 必须精确指定为同一 transport hierarchy 的 total mapped
   cell area；sequential area 用 noncombinational area，不用 FF count。full-K8 timing guard 应写
   成所有 setup/hold/五类约束 clean，而不是含糊的“3 ns 回退不超过 1%”。
3. K8 local A/B 可以先做；但它不能更新 K8-vs-K1x8 throughput/mm2 主表。若 local 点过 8%
   area 或 10% dynamic 门，再决定是否给 K1x8 八条 lane 对称 elide；未做则只报 K8 local
   implementation ablation。
4. 100% exact SAIF annotation 是本仓库已有可达门，但还应同时报告 nonzero-toggle coverage，
   并冻结同一 contiguous measurement window；仅报 annotation 100% 不能说明 tag nets 真在切换。

## 修订后最小 clone 清单与实现顺序

前置：M519 R5 K1/K8/K1x8 三轴 DC 独立封存，`TIM-209/OPT-150=0`，3 ns clean。

1. clone M490 为 tagged R5-precedence reference；只闭合同拍 request/response fault 语义，
   不改算术、reuse、queue 或带宽；
2. 从该 reference clone elided adapter，删除 `pending_tag_q`、`slot_tag_q`、bank tag ports、
   comparator 与 mux；
3. clone M218 service 为当前单-token candidate：response legality 保留 slot/epoch/generation/mask，
   tag 来自 token-level owner；不要用 `dont_touch` 保留死 `sb_tag`；
4. 新建 tagged/elided synthesizable scalar leaf shell 对，macro data 边界相同；
5. 新建同端口 matched wrapper；`ELIDE_TAG` 只能选择上述两棵完整层级，所有其他源码 SHA
   与参数一致；
6. 先跑 legal-traffic cycle/tuple/bank/Acc24 ghost-tag miter，再跑 stale/reuse/reorder/flush-local
   与 R5 同拍 fault attacks；
7. VCS 独立锤审 P0=0 后跑 paired transport-local + full-K8 DC；功能/时序 clean 后无条件跑
   一次 mapped-gate SAIF/PTPX；
8. local 物理门通过才考虑 K1x8 对称实现；否则不更新 C2 三轴主表。

## 物理门修正版

- `PROMOTE_C2_SUBMECHANISM`：transport-local total cell area `>=15%` 或 dynamic power
  `>=20%`，且 sequential area `>=10%`、full K8 area/power不回退超过 1%、所有 timing/traffic/
  cycle/tuple 门 clean；
- `KEEP_C2_IMPLEMENTATION_DETAIL`：area `8–15%` 或 dynamic `10–20%`，其余 guardrail clean；
- `NO_GO_PHYSICAL`：area `<8%` **且** dynamic `<10%`，或任一功能/traffic/cycle/timing P0；
- 若 full K8 总面积/功耗变化很小，即使 local promotion 也只能写局部协议消融。

## DATE claim 边界

只有物理门通过后，允许写进 C2：

> A typed epoch-generation-slot ticket removes replicated tag transport from
> independently backpressured FC2 weight-bank leaves while preserving the
> signed-Acc24 transaction schedule; the token-level owner retains the semantic
> output tag.

需引用 ELSA 与 FireFly-T。不得写 `first`，不得把 `27.53%` 写成 energy，不能声称 cycle/system
speedup，不能与 K1 倍率相乘，不能升为 C4。当前 `paper_ppa_ready=false`。

