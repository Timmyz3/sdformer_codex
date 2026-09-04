# M107 W64 ping-pong schedule replay 独立打铁评审 r1

日期：2026-08-24  
评分：**72/100**  
严重度：**P0=1，P1=4，P2=2**  
`docs/359` SHA256：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`，评审前后未变。

## 结论

M107 的 raw work reconstruction 和发布的 fluid ping-pong recurrence 可以独立复现，但“two-bank fill/drain cycle schedule”尚不能按当前 M106 RTL 的 cycle-exact schedule 准入。

独立审计没有 import 或运行 M105/M107 producer。它从冻结 M40 packed binaries、M72 centers 和 M41 INT8 weights 重新构造 20 个 heldout records 的 cap11 W64 event bitmap，精确得到：

- 406,080 windows，其中 35,309 empty、370,771 nonempty；
- 188,148,490 correction/fallback events；
- 35,140,002 active `(source, block)` groups；
- `188,148,490 + 3 × 35,140,002 = 293,568,496` correction service tokens；
- PWP uses-by-width 与 `226,222,255` tokens 全部重合。

按 M107 analyzer 的 fluid recurrence，partition-major 与 window-major 的每个发布字段均精确复现。但冻结 M106 RTL 在 READY→DRAIN 前有一个 controller dispatch edge；bank 在 drain 完成后若 producer 正被阻塞，也需要一个 EMPTY→FILL reacquire edge。M107 允许这些边界零延迟，因此 selected window-major 少算 **384,097 cycles**。

加入可由 RTL 状态机直接证明的 edge 后，仍然只在相同的 service-island 边界内：

```text
window-major combined = 546,844,543
same-clock service-island ratio = 2.037842934093246
2x headroom = 10,347,101 cycles
```

它仍大于 2×，但余量很薄；accumulator、shared-weight SRAM 和 PWP dependency schedule 都未接入，所以不能认为 2× 已稳。

## 证据与独立性

| 证据 | SHA256 |
|---|---|
| M107 contract | `84ce5aac54075b431988995b9ceaec592bfe21eabbefce594069ef7d3abb3002` |
| M107 analyzer | `2c655e2e907d046761de2ce44563ae1f0505798d40fdaba5faca98c192d741b8` |
| M107 result | `0c613c2da4eb3e860ccc33dfbbe3fba0bb1424c19ab12af339e54178d03333db` |
| M107 RUN_COMPLETE | `83234c3e626278a0bff2cf7bd5c481a08504b7448a33802870e1fdf186027fc7` |
| M107 manifest | `8857341cc2263bc1d75cef3b9d6815e68b9153a8b8329ae15654c9bd05519678` |
| frozen M105 auditor | `5e5c07631dd8c4bb328cd234da5c04fde8eb9800d1516b3fe462124b2b661ed5` |
| frozen M105 result | `3348b6c02ad97be5b61ffb6f8d5f79578f4551e037097c4f74ac598d2842767b` |
| M106 contract | `881491f58543f2c6b0b5b3c1d07d7b170cdbfb4190153a18929bdddd83a39999` |
| M106 production RTL | `0abc1adf612788bbfdd2f26ff847234ee7efaaa2addcc7f28f03ddac22cd68e7` |
| M106 sealed RUN_COMPLETE | `fc118089b84ea99c1ed72077171539bd113aa85e0d07f709d35800ae23b5b1d4` |

M107 manifest 8/8 通过。所有 JSON 采用 duplicate-key/non-standard-constant 拒绝模式读取。独立 raw decoder、center tie-break、PWP width catalog、cap11 eligibility、W64 window aggregation和两个 schedule recurrence都在本 review 脚本内实现。

## Raw work 守恒

独立重建的 PWP width histogram 为：

| signed width | catalog entries |
|---:|---:|
| 8 | 52,248 |
| 9 | 128,893 |
| 10 | 37,144 |
| 11 | 2,898 |
| 12 | 1 |

heldout PWP uses 与收费：

| width | uses | tokens/use | tokens |
|---:|---:|---:|---:|
| 8 | 11,164,284 | 3 | 33,492,852 |
| 9 | 32,360,036 | 4 | 129,440,144 |
| 10 | 13,936,011 | 4 | 55,744,044 |
| 11 | 1,509,043 | 5 | 7,545,215 |
| total | 58,969,374 | — | 226,222,255 |

20 个 record 的 event 总数、group 总数、empty window 总数与 M105/M107 精确一致。partition-major 与 window-major 只改变 window sequence，不改变 `(events,groups)` multiset 或总 work。

## 发布 fluid schedule 的精确复现

| 指标 | partition-major | window-major |
|---|---:|---:|
| correction tokens | 293,568,496 | 293,568,496 |
| descriptor fill cycles | 188,554,570 | 188,554,570 |
| producer bank stall | 122,011,062 | 131,683,621 |
| service idle | 16,997,647 | 26,669,695 |
| correction schedule | 310,566,143 | 320,238,191 |
| + PWP combined | 536,788,398 | 546,460,446 |
| published ratio | 2.0760196982 | 2.0392752964 |
| headroom to 2× | 20,403,246 | 10,731,198 |

这些数逐字段匹配 M107 JSON，说明 aggregate、empty-window population、bank modulo-two recurrence和两个 loop ordering 的 producer 计算没有随机漂移。

## P0：fluid boundary 不是当前 RTL 的 cycle-exact boundary

M107 recurrence 使用：

```text
drain_start = max(previous_drain_end, fill_end)
fill_start = max(previous_fill_end, selected_bank_free)
```

这等价于 READY bank 在 `fill_end` 边界立即发出第一项 service，以及 drain bank 在 `bank_free` 边界立即接受下一 window 的第一项 fill。

冻结 M106 RTL 的 nonblocking state transitions 不支持这两个零延迟边界：

1. close accept 将 bank 写成 `BANK_READY`；下一 edge 才观察 READY 并设置 `drain_active_q`；第一项 `service_accept` 再在后续 edge 出现。因此每个 window 有一个 controller dispatch edge，empty window 也需要这个 edge 才能检查 `active_key==0` 并释放。
2. drain 最后一项 service 将 bank 写成 `BANK_EMPTY`；当 producer 因两 bank 均占用而 stalled 时，下一 edge 才观察 EMPTY 并设置 `fill_available_q`，随后 edge 才能 accept 新 fill。因此同边界 reuse 需要 reacquire edge。

本评审基于冻结 RTL 状态转换建立 event-driven recurrence，always-ready service、无 producer grace，并保持所有其他 M107 port cuts。结果：

| 指标 | partition-major edge-aware | window-major edge-aware |
|---|---:|---:|
| controller dispatch edges | 406,080 | 406,080 |
| blocked-fill reacquire boundaries | 309,325 | 290,527 |
| correction schedule | 310,944,503 | 320,622,288 |
| + PWP combined | 537,166,758 | 546,844,543 |
| service-island ratio | 2.0745574282 | 2.0378429341 |
| headroom to 2× | 20,024,886 | 10,347,101 |

部分 dispatch edge 替代了原 fluid model 的 fill-wait idle，因此净差不是完整的 406,080；selected window-major 净少算 384,097 cycles。

这不是大幅推翻比值，但会推翻 `two_bank_fill_drain_cycle_schedule=true` 的 cycle-exact 含义。当前应将发布结果降级为 **fluid software schedule**，直到 actual M106 controller replay 或与 RTL 对齐的 r2 recurrence 关闭该差异。

## Empty window、bank reuse 和 loop order

- Empty window 共有 35,309 个。`events=groups=0` 守恒正确，producer 的 close cycle 也已计；缺的是 consumer 观察 READY/empty 并 toggle `next_drain_bank_q` 的 dispatch edge。
- Fixed modulo-two bank sequence与正常 M106 ping-pong 生命周期一致；漏洞不是 bank index，而是 blocked reuse 的同钟 release/acquire 时序。
- partition-major 顺序是 `(sample,operator,partition,window)`；window-major 是 `(sample,operator,window,partition)`。独立重建确认总 work 相同。
- contract 选择 window-major 来把 accumulator 限制为 64-row tile。不能用更高的 partition-major `2.076×` 作为 selected 实现数字；它规避的是 6.912 MB full-raster accumulator footprint。

## PWP 串行计费审计

PWP 没有 aggregate 重复收费或漏计：

- eligible rows 的 PWP seed 与 correction delta 是不同 work；
- fallback rows 在 correction event term 中收费，不产生 PWP use；
- 58,969,374 uses 按 width 计成 226,222,255 tokens，combined 中只加一次；
- PWP 不包含在 `events + 3×groups` 中。

但 M107 把全部 PWP aggregate 简单追加在 correction schedule 之后。这不是可执行 dependency order：实际 eligible destination 需要 seed-before-correction；PWP 与 correction 还可能争用 shared weight SRAM、accumulator banks和commit path。串行相加可作为保守 service work charge，却不能证明时序可执行，也不能证明没有新的 dependency idle。

## 2× 余量是否足够

不够稳健。selected edge-aware window-major 离 2× 只有 10,347,101 cycles：

- 仅为当前 combined 的 1.892%；
- 相当于每个 correction/fallback event 只允许 0.05499 个额外 cycle；
- 即平均每 18.18 个 event 多一个无法隐藏的 cycle，就会跌破 2×。

accumulator bank/RMW、finite-width miter、shared-weight SRAM、PWP seed dependency、macro latency和physical frequency全部尚未进入。尤其 contract 需要每 event 同拍完成一组 96-lane vector read+write；若实际宏是 1RW 且没有 forwarding/dual-port，这个缺口远大于现有余量。

因此 2.076× 与 2.039× 都只是四个 heldout bottleneck operators 的 same-clock service-island 指标，不是 full-network/system，也不是已经完成的 module scheduled speedup。2.039× 甚至应先纠正为 edge-aware 2.03784× 后再继续集成。

## Port cuts 与 claim boundary

以下缺口在 contract/result 中写为 false，边界是明确的；本评审维持该边界：

- accumulator schedule；
- 24-bit finite-width equivalence miter；
- accumulator bank/address/1R1W or 1R+1W replay；
- shared-weight SRAM address/port/contention schedule；
- per-row PWP seed-before-correction dependency replay；
- physical memory latency、equal area、macro-inclusive PPA；
- full-network/system/headline。

## Findings

### P0

- **M107-P0-01-FLUID-BOUNDARY-NOT-M106-CYCLE-EXACT**：发布 recurrence 省略 M106 READY dispatch 与 blocked-bank reacquire edge，selected window-major 少算 384,097 cycles。必须发布 edge-corrected r2 或 actual-controller replay，当前 `two_bank_fill_drain_cycle_schedule=true` 不准入。

### P1

- **M107-P1-01-ACCUMULATOR-RMW-NOT-SCHEDULED**：188,148,490 个 event 的 bank/address/read-modify-write没有执行模型；现有 10.35M headroom 只能容忍约 5.5% 的一拍 stall。
- **M107-P1-02-PWP-DEPENDENCY-NOT-EXECUTABLE**：PWP aggregate 只收费一次且数学正确，但被整体放在 correction 之后，未证明 seed-before-correction、commit和shared SRAM arbitration。
- **M107-P1-03-SHARED-WEIGHT-SRAM-PORT-CUT**：correction 三拍 weight load 与 PWP payload service 的地址、port、latency和冲突未重放，same-clock token不能提升为 physical cycle。
- **M107-P1-04-SERVICE-ISLAND-NOT-SYSTEM**：cohort 仅为 valid825 heldout 的四个 bottleneck Conv3x3 operators；baseline denominator 也是该 service scope，不是全网端到端。

### P2

- **M107-P2-01-PARTITION-MAJOR-NOT-SELECTED**：2.076× 对应 partition-major，未满足选择 window-major 所需的 bounded accumulator footprint；禁止 cherry-pick 更高比值。
- **M107-P2-02-ORDER-DIGEST-MISSING**：result 保存 aggregate schedule，没有冻结每 record/window 的 context/order digest。独立重建能复现当前顺序，但未来 analyzer 漂移时 manifest 不会直接显示是哪段顺序改变。

## GO / NO-GO

| 项目 | 决定 |
|---|---|
| exact raw events/groups/PWP work ledger | GO |
| published fluid recurrence 可复现性 | GO |
| published cycle-exact M106 schedule | NO-GO，P0 |
| edge-aware software service-island bound | GO，必须注明非 accumulator/physical schedule |
| `>2×` robust admission | NO-GO，余量不足且 port cuts 未闭合 |
| physical/equal-area/system/headline | NO-GO |

## 下一里程碑 admission

建议下一里程碑为 **M108 actual-controller + accumulator/shared-SRAM dependency replay**，至少满足：

1. 使用 actual W64 window sequence，逐 window 保留 context、events、groups、empty flag和bank id；
2. cycle model 与 M106 RTL 对齐，包含 READY dispatch、EMPTY reacquire、close、fill/drain同钟优先级；在小型 directed/random streams 上与 commercial RTL 逐 cycle miter；
3. PWP seed 在相应 correction 前完成，shared-weight SRAM 有地址、port和冲突仲裁；PWP总 charge 仍必须恰好 226,222,255；
4. 24-bit accumulator 做 finite-width bit-exact miter，八 bank 的 address/RMW/forwarding 与目标宏端口匹配；
5. selected window-major combined 必须不超过 `557,191,644`，且所有新增 stall 有收据。当前 edge-aware 后最多只剩 10,347,101 cycles；
6. 即使通过，先只准入 scheduled service-island，不准入 physical/system/headline，直到 macro/PPA和全网账本完成。

本评审未修改 production RTL、脚本、contracts、results 或 `docs/359`。
