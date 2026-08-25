# M132 dual-row 512-bit PWP compact-K4 schedule independent hammer

## Verdict

**87/100；P0=0、P1=2、P2=2。Cycle DSE 数值通过，512-bit 物理端口与完整身份闭包未准入。**

Production analyzer 已 exact-SHA 重跑，输出与 sealed result byte-identical。独立实现没有调用 production M122 `FoldSchedule`，而是从冻结 width placement 重建 raw descriptor 序列并自行实现 two-bank producer/controller/service recurrence；256-bit compact baseline 与 M129 完整 recurrence 逐字段相同，512-bit candidate 也逐字段复现 M132。

可引用的最强结论是：在 frozen heldout W384、same-clock、service-island cycle DSE 中，将 PWP service 从每拍一个 256-bit logical row 提升为每拍两个 logical rows，把 candidate cycles 从 351,479,358 降到 245,485,910，即 1.431769986×。`4.541455956×` 也是相对 frozen fixed8 的同频 service-island ratio，不是 physical/system/headline speedup。

## Independent width/token algebra

| Signed PWP width | Frozen uses | 96-lane vector bits | 256-bit cycles | 512-bit cycles |
|---:|---:|---:|---:|---:|
| 8 | 11,164,284 | 768 | 3 | 2 |
| 9 | 32,360,036 | 864 | 4 | 2 |
| 10 | 13,936,011 | 960 | 4 | 2 |
| 11 | 1,509,043 | 1,056 | 5 | 3 |

纯整数代数得到：

```text
PWP256 tokens
  = 11,164,284×3 + 32,360,036×4
    + 13,936,011×4 + 1,509,043×5
  = 226,222,255

PWP512 tokens
  = 11,164,284×2 + 32,360,036×2
    + 13,936,011×2 + 1,509,043×3
  = 119,447,791

reduction = 47.198921256%
```

独立 raw heldout width-placement replay 得到完全相同的 width uses 与 token totals；不是只从 production result 抄数。

## Independent descriptor recurrence

| Metric | Independent | Production | Result |
|---|---:|---:|---|
| compact-K4 PWP256 cycles | 351,479,358 | 351,479,358 | exact |
| dual-row PWP512 cycles | 245,485,910 | 245,485,910 | exact |
| cycles removed | 105,993,448 | 105,993,448 | exact |
| M132 / compact256 | 1.431769986310× | 1.431769986310× | exact |
| fixed8 same-clock service-island | 4.541455955660× | 4.541455955660× | exact |

Independent recurrence preserved the same descriptor fill, PWP/correction overlap, two producer banks, controller dispatch, weight loads, W384 commits and flush edges。`compact_k4_pwp256` 与 M129 `m128_descriptor_conservative_startup` 的完整 JSON recurrence 相同，不只是 candidate-cycle total 相同。

## “Free 512-bit port” attack

Cycle DSE 把每拍可用 PWP read bandwidth 从 256 提到 512 bit，瞬时 bandwidth 是 2×。这不是把参数从 256 改成 512 就自动获得的硬件能力。

至少需要以下一种真实结构：

- 两个可以同拍独立读取的 256-bit logical rows；
- 等价 true dual-read/duplicated macro 组织；
- 或 16 个 conflict-free 32-bit word banks 同拍供数，而 baseline 只需 8 个。

当前没有提供：

- PWP address → 16 bank mapping；
- heldout cycle 的双行 bank-conflict trace；
- collision arbiter、stall/replay protocol；
- dual-row/16-bank RTL 与 VCS conservation；
- foundry macro、外围、mux/wire area；
- SAIF/PTPX energy；
- matched DC/PT frequency。

47.20% token reduction 低于 50%，来自 96×width 向 512-bit beat 取整；它不等于 SRAM access energy 减半。更宽的 macro output、bank selection、crossbar 和布线也可能降低频率或增加动态功耗。

物理吞吐近似为：

```text
throughput gain = 1.431769986 × (f512 / f256)
```

所以只有 `f512/f256 > 0.698436208` 时，same-clock cycle advantage 才仍是正的。没有 matched period sweep 前，不能把 1.4318× 叫 physical speedup。

## Compact-K4 predecessor boundary

M132 pin 了 M131 VCS receipt，但该 receipt 明确写着：

```text
complete_row_partition_losslessness=false
descriptor_producer_implemented=false
synopsys_dc_elaboration_clean=false
physical_speedup=false
```

因此 M131 当前只支持 accepted compact descriptor 的局部 numeric/protocol 与 II1 证据，尚未证明完整 sparse producer 能无损地产生每个 descriptor，也没有干净 DC。M132 的 35-bit compact correction stream 是 cycle contract，不是完整 integrated hardware。

M115r2 的 signed19 prefix-bound 数学证据同样不能替代 512-bit read organization；它约束累加数值范围，不证明 PWP macro ports、bank conflicts、area 或 energy。

## Identity hammer

M132 已修复 M129 大部分 lineage：直接 pin M109/M108/M105 scripts、M40/M72/M41、M129 result/correction/review 和 M131 receipt。对直接 M129 result 做 SHA drift，analyzer 在产生输出前 fail-closed。

仍有一处 exact-SHA 缺口：M132 通过 `m122.M109_RESULT` 读取 fixed8 baseline service tokens，却没有把 M109 result 放入 `frozen_paths/EXPECTED_SHA256`。负测在通过 M122 script SHA 后注入一个 JSON-valid、语义不变但 SHA 漂移的 M109 result，production analyzer 仍 PASS 且输出与 sealed result byte-identical。

这个缺口不会推翻当前数字：independent review 已直接 pin M109 result，且 compact256 与 M129 full recurrence equality 会拒绝影响相关 baseline 字段的语义漂移。但 production analyzer 的“完整 exact-SHA lineage”仍不成立，应补 pin `ee61b90e...`。

## Scorecard

| Dimension | Score | Evidence |
|---|---:|---|
| Width-use/token algebra | 20/20 | 纯代数与 raw trace exact。 |
| Independent descriptor recurrence | 25/25 | 未调用 production FoldSchedule；两模型 full recurrence exact。 |
| Production reproducibility | 15/15 | exact-SHA rerun byte-identical。 |
| Identity closure | 10/15 | 主要 transitive inputs 已 pin；M109 result 漏 pin。 |
| Physical port realism | 7/15 | Requirements 写清，但没有 RTL/conflict/macro/frequency。 |
| Claim discipline | 10/10 | same-clock service-island 与 physical/system 边界正确。 |
| **Total** | **87/100** | **强 cycle-DSE milestone；不是 hardware PPA admission。** |

## P0

**0 个。** 冻结 heldout 范围内没有 width count、token、recurrence、cycle、ratio 或 M129 baseline mismatch。

## P1

### P1-1 — 512-bit candidate 没有可执行硬件端口证明

缺 dual-row/16-bank RTL、address mapping、conflict trace/arbiter、macro 与 matched Synopsys。任何 physical throughput、area 或 energy 主张都被阻断。

### P1-2 — M109 result exact identity 漏 pin

Production 对直接 inputs fail-closed，但 `m122.M109_RESULT` 的 SHA 漂移可以通过。应将 M109 result 加入 frozen identity，并新增负测。

## P2

### P2-1 — M131 producer/integration 尚未闭合

35-bit compact consumer 局部 VCS 已有，但 complete row partition、producer implementation 和 clean DC 都是 false。需要 producer→descriptor→dualrow service 的 conservation replay。

### P2-2 — Same-clock gain 仍需 macro/frequency/energy A/B

至少做 256-bit 8-word-bank 与 512-bit 16-word-bank 同工艺、同容量、同约束对照，给出 conflict stalls、area、Fmax、read energy 和 PTPX。`f512/f256` 低于 0.698436 时，cycle gain不会形成 throughput gain。

## Safe claim

> Independent width algebra and descriptor recurrence reproduce the frozen M132 result: two logical 256-bit rows reduce modeled PWP tokens from 226,222,255 to 119,447,791 and same-clock heldout service-island cycles from 351,479,358 to 245,485,910 (1.43177×). The 4.54146× fixed8 comparison is also same-clock service-island only. A conflict-free 512-bit read implementation, macro cost, frequency, energy, physical and system speedup are unadmitted.

## Prohibited claims

- 不得把 1.4318× 写成 RTL-measured、physical、system 或 headline speedup；
- 不得把 4.5415× 写成 full-network/end-to-end speedup；
- 不得将两个 256-bit rows/cycle 当作免费 SRAM port；
- 不得假定 16-bank conflict-free、无 arbitration stall；
- 不得从 token reduction 推断 macro area/energy；
- 不得乘任何未匹配 frequency ratio；
- 不得把 M131 局部 VCS 当作完整 producer losslessness。

## Artifacts

- `independent_recompute_m132.py`：独立 width placement、token algebra 和 recurrence。
- `independent_result/m132_independent_recompute.json`：独立数值回执。
- `production_rerun/`：production exact-SHA byte-identical 重跑。
- `run_m132_identity_negative_tests.py` 与 `m132_identity_negative_tests.json`：direct fail-closed / M109-result fail-open。
- `audit_m132_independent.py` 与 `m132_dualrow512_pwp_compact_k4_schedule_independent_audit.json`：最终审计。
- `manifest.sha256`：review source 与文本证据清单。

`docs/359_DATE终局冻结_20260813.md` 未修改，SHA256 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
