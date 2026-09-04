# M129 row-admission bubble and descriptor-cost independent hammer

## Verdict

**88/100；P0=0、P1=2、P2=2。数值与模型边界通过，身份闭包需要修正。**

冻结 production analyzer 已 exact-SHA 重跑，输出 JSON 与 sealed result byte-identical。独立实现没有导入 M129 analyzer，也没有复用 M122 `FoldSchedule`；它分别用 M122 source-count histogram 代数、raw held-out trace 重建和独立二 bank/controller recurrence 三条路径复算，工作量与三个 candidate-cycle 总数全部一致。

`1.269300343948×` 可以作为 **frozen held-out、same-clock、module-cycle model A/B**：比较的是 M125/M127 row-mask interface 和 M128 conservative descriptor-startup interface。它不是 VCS 实测吞吐、physical speedup、system speedup 或 headline。

## Independent numeric reproduction

| Quantity | Independent result | Production | Result |
|---|---:|---:|---|
| events | 188,148,490 | 188,148,490 | exact |
| active `(row, output-block)` | 94,735,083 | 94,735,083 | exact |
| K4 groups/descriptors | 99,847,888 | 99,847,888 | exact |
| nonempty `(partition, window)` | 68,820 | 68,820 | exact |
| M122 ideal candidate cycles | 351,410,711 | 351,410,711 | exact |
| M125/M127 row-mask candidate cycles | 446,132,870 | 446,132,870 | exact |
| M128 conservative descriptor cycles | 351,479,358 | 351,479,358 | exact |
| M128 / row-mask module-cycle A/B | 1.269300343948× | 1.269300343948× | exact |

Histogram algebra is independently sufficient to recover the two main work counts:

```text
active row-blocks = Σ[n>0] histogram[n]
                  = 94,735,083

K4 descriptors   = Σ ceil(n/4) × histogram[n]
                  = 99,847,888
```

Raw trace reconstruction independently obtains the same values and additionally counts 68,820 nonempty partition-windows.

## Exact admission charges

这三个模型共享同一 M122 descriptor fill、PWP、weight load、two-bank producer、controller、commit recurrence，只替换 correction service 中的 interface charge：

```text
M122 ideal folded cycles
  = K4 descriptors
  = 99,847,888

M125/M127 row-mask folded cycles
  = K4 descriptors + one admission per active row-block
  = 99,847,888 + 94,735,083
  = 194,582,971

M128 conservative descriptor folded cycles
  = K4 descriptors + one startup per nonempty partition-window
  = 99,847,888 + 68,820
  = 99,916,708
```

这里的“一拍”是 cycle-model charge，不是从完整 producer RTL 测得。M128 外部 descriptor generation、canonical partitioning、buffer/storage、memory energy 和跨接口调度均没有计费；因此不能把差值 94,653,512 cycles 外推为系统 latency 或 FPS。

## Descriptor cost

| Format | Payload fields | Bits/item | Held-out total bits | vs row-mask total |
|---|---|---:|---:|---:|
| M125/M127 row mask | block3 + row9 + mask16 + negate16 | 44 | 4,168,343,652 | 1.0000× |
| M128 descriptor | block3 + row9 + valid4 + IDs16 + negate4 + selected16 + last1 | 53 | 5,291,938,064 | 1.269554× |
| proposed M130 compact | block3 + row9 + count2 + IDs16 + negate4 + last1 | 35 | 3,494,676,080 | 0.838385× |

M128 单条 payload 从 44 增至 53 bit，增加 20.45%；结合 held-out 上 K4 descriptor 数量多于 active row-block，total payload 增加 26.9554%。所以 M128 只能主张 cycle-model 改善，不能主张 descriptor bandwidth reduction。

35-bit 只是 M129 中的 successor what-if。即使工作区随后出现 M130 RTL，M129 result/contract 没有绑定其 RTL、VCS、producer 或 physical evidence；不能用后续未绑定文件倒推 M129 已验证 35-bit bandwidth reduction。

## Identity hammer

### Direct frozen identity：PASS fail-closed

对 M122 result 做一个 JSON-valid 的 whitespace SHA drift，production analyzer 在创建输出前精确拒绝：

```text
frozen input identity drift: m122_result
```

### Transitive replay identity：FAIL open

M129 校验 M122 script/result SHA 后，加载 M122 module 并直接执行它指向的 M109/M108/M105 replay helpers；但它没有执行 M122 `main()` 中的 M109 SHA checks，也没有自行 pin 这些传递依赖。

负测向已通过直接 M122 SHA check 的 module 注入一个只追加注释、SHA 已变化的 M109 copy。production M129 仍完整运行并生成与 sealed result 相同 SHA 的 JSON：

```text
production M109 SHA = 4eed1e1e...
drift M109 SHA      = fbc33f50...
analyzer rejected   = false
numeric output      = exact match
```

这不推翻当前数字——独立复算已经确认它们——但说明现有 exact replay 的 provenance 不是传递闭包。

### M128 correction overlay：未绑定

active overlay SHA `e646cc71...` 没有出现在 M129 analyzer、result 或 contract frozen identity 中。M129 的合同已经正确写出 producer unimplemented、bandwidth false、physical/system/headline false，但没有同时封存 overlay 中更具体的限制：

- complete row-partition canonicality 未证明；
- untagged `row_done` 不能作为安全 completion token；
- idle `group_ready` 依赖 semantic payload，source 不能组合依赖 ready；
- 53 bit 只是 payload，不含完整接口、buffer 或 producer traffic。

因此后续引用 M129 必须携带 overlay，或发布一个新 correction overlay 将其 SHA 和传递 replay identities 一并冻结。

## Scorecard

| Dimension | Score | Evidence |
|---|---:|---|
| Exact-work algebra/reconstruction | 25/25 | histogram 与 raw trace 双路径 exact。 |
| Independent cycle recurrence | 25/25 | 三个 candidate totals 与完整 recurrence exact。 |
| Admission/descriptor model boundary | 18/20 | charge 与 exclusions 已写清；仍是未含 producer 的 model。 |
| Identity and reproducibility | 10/20 | direct identity fail-closed；transitive dependency 与 overlay 未封。 |
| Claim discipline | 10/10 | 1.2693×、53-bit 反增、35-bit proposed 边界正确。 |
| **Total** | **88/100** | **numerically citable with module-cycle qualifier and identity correction.** |

## P0

**0 个。** 冻结数据范围内没有工作量、周期 recurrence、ratio 或 descriptor bit arithmetic mismatch。

## P1

### P1-1 — Transitive trace-replay identity fail-open

M109/M108/M105 helper 或底层 manifest/profile 即使漂移，M129 的三项 frozen identity 也不保证拒绝。建议新增一份 transitive manifest，至少冻结实际执行的 M109/M108/M105 scripts、M40 manifest、M72 result 和 M41 bridge result，并在 replay 前逐项校验。

### P1-2 — M128 correction overlay 未进入 M129 claim lineage

M129 依赖 M128 cross-row II1 作为模型动机，却只 pin sealed VCS receipt，没有 pin active correction overlay。建议发布 M129 correction contract，把 `e646cc71...` 纳入 identity，并继承 canonicality/completion/ready/producer 四项限制。

## P2

### P2-1 — Admission bubble 是接口假设，不是完整 producer measurement

Row-mask 每 active row-block +1、M128 每 nonempty partition-window +1 的模型写得清楚且计算正确；但外部 descriptor 构造与存储未计费。论文中必须写“candidate module-cycle model under explicit interface charges”。

### P2-2 — M128 bandwidth 反增，35-bit 不能回填

M128 payload total 是 row-mask 的 1.269554×，没有 bandwidth advantage。M129 的 35-bit 数字只有代数 what-if；只有绑定 compact producer/consumer RTL、canonicality、traffic 和 memory energy 后，才能升级。

## Safe claim

> Independent histogram algebra and raw-trace recurrence reproduce 94,735,083 active row-blocks, 99,847,888 K4 descriptors and all three candidate-cycle totals. On this frozen held-out same-clock module model only, charging one row-mask admission per active row-block versus one M128 startup per nonempty partition-window gives 1.269300344×. M128's 53-bit payload increases modeled traffic versus the 44-bit row mask; the 35-bit successor remains proposed.

## Prohibited claims

- 不得把 `1.269300344×` 写成 physical、system、end-to-end 或 headline speedup；
- 不得把 `3.171917× vs fixed8` 乘上尚未匹配的频率；
- 不得声称 M128 降低 descriptor bandwidth；
- 不得把 35-bit 当成 M129 已实现、已 VCS 或已物理验证；
- 不得从 payload bits 推导 SRAM/DRAM energy；
- 不得在不携带 M128 overlay 的情况下声称 complete canonical producer 或安全 completion protocol。

## Artifacts

- `independent_recompute_m129.py`：独立 histogram/raw trace/recurrence 复算。
- `independent_result/m129_independent_recompute.json`：独立数值回执。
- `production_rerun/`：production analyzer exact-SHA byte-identical 重跑。
- `run_m129_identity_negative_tests.py`：direct fail-closed 与 transitive fail-open 负测。
- `m129_identity_negative_tests.json`：身份测试机器回执。
- `audit_m129_independent.py` 与 `m129_row_admission_bubble_and_descriptor_cost_independent_audit.json`：最终 fail-closed 审计。
- `manifest.sha256`：review 文本/source/receipts 清单。

`docs/359_DATE终局冻结_20260813.md` 未修改，SHA256 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
