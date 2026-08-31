# M518 matched Fixed T10 ATLIF baseline RTL specification r1

Date: 2026-08-27  
Verdict: `GO_IMPLEMENT_AS_NEW_SIBLING_FILE__NOT_YET_RTL_OR_PPA_ADMITTED`  
Score: **94/100**  
Findings: **P0=0, P1=4**  
Execution in this review: **read-only source/evidence inspection; no VCS, DC, Formality, PTPX, GPU, or production-file edit**

## 1. Literal decision

Implement a new sibling top named `m518_matched_fixed_t10_atlif` in a future
`rtl_m518/` directory. Copy the proven M273r2 transport/lifecycle pattern, but
write a new dense scheduler and accumulator datapath. Do **not** add a Fixed
mode to `m273_integrated_rank3_atlif`, and do not fork M31 as the top.

This is the minimum faithful materialization of M265's
`FIXED_DENSE_T10_EXACT96_TILE_CLOSED` model:

- the external port names and widths are exactly those of M273r2;
- Fixed configuration is five ordered 256-bit beats carrying 1,064 payload
  bits, not M273's six-beat rank3 frame;
- each raw tile is five ordered 256-bit beats carrying `X[10][16]` signed INT8;
- exactly 96 dynamic signed INT8 multiplier slots execute 1,600 products in 17
  tile-closed issue cycles (`16*96 + 64`);
- issue cycles 12--16 atomically complete and push result beats 0--4 while
  multiplication remains active;
- two raw banks, oldest-ready selection, 16-beat registered result FIFO,
  full-FIFO simultaneous pop/push, tags, release, and sticky registered
  fail-closed protocol match M273r2;
- there is no Fixed intermediate bank and no extra product register.

The five-beat configuration is an intentional M265 match. "Same config
boundary" means the same 256-bit valid/ready/accept/last transport and the same
ready trace, not equal payload length. Padding Fixed to six beats would change
the admitted clean formula from `17*N+12`, add 45 cycles to the frozen
45-context population, and no longer reproduce 124,412,490 cycles.

## 2. Frozen arithmetic

For output temporal row `r in [0,9]` and lane `l in [0,15]`:

```text
wide[r,l] = signext25(bias[r])
          + sum(t=0..9, signed8(X[t,l]) * signed8(W[r,t]))
q24[r,l]  = saturate_signed24(wide[r,l])
event[r,l] = (q24[r,l] >= signed24(threshold))
```

There is no requant shift, rank intermediate, membrane carry, stochastic
rounding, or cross-tile accumulation. Threshold equality fires.

Configuration bit layout, LSB first:

| Bits | Meaning | Indexing |
|---|---|---|
| 0--799 | 100 signed INT8 weights | `W[r,t] = frame[((r*10+t)*8)+:8]` |
| 800--1039 | ten signed Q24 biases | `bias[r] = frame[800+r*24+:24]` |
| 1040--1063 | one signed Q24 threshold | `frame[1040+:24]` |
| 1064--1279 | required zero padding | any one is a config fault |

Raw mapping is `raw_data[word*8+:8] -> X[2*beat + word/16][word%16]` for
`beat=0..4`, `word=0..31`. Result beat `b` contains rows `2*b` and `2*b+1`:

```text
result_data[row_in_beat*16 + lane] = event[2*b+row_in_beat,lane]
result_data[47:32]                 = 0
result_valid_bits                  = 48'h0000_ffff_ffff
```

### Width and overflow contract

One signed INT8 product is in `[-16,256, 16,384]`. Ten products are in
`[-162,560, 163,840]`. Adding any signed Q24 bias gives
`[-8,551,168, 8,552,447]`, which fits signed 25 bits
(`[-16,777,216, 16,777,215]`). Therefore:

- multiplier products are signed 16 bits;
- accumulator registers are exactly signed 25 bits, one per `(row,lane)`, 160
  registers total;
- each update is evaluated in signed 26-bit combinational precision, then may
  be stored in 25 bits only after an assertion proves bit25 equals bit24;
- no wrap, per-step saturation, or truncation is allowed;
- only the completed 25-bit sum is saturated to signed Q24 before comparison.

## 3. Exact 96-slot product schedule

Let `c` be the held dense issue-cycle register (`0..16`) and `s` the physical
multiplier slot (`0..95`). A cycle advances only on `dense_issue`.

### Cycles 0--11: beats 0--3 prologue

```text
b    = c / 3
sub  = c % 3
j    = s / 3                  // output scalar 0..31 within beat b
k    = s % 3
r    = 2*b + j/16
l    = j % 16
t    = 3*sub + k              // taps 0..8
```

All 96 slots are active. Every addressed accumulator adds three products. On
`sub==0`, the old accumulator is ignored and the base is the sign-extended
bias, making stale data unobservable.

### Cycles 12--15: close beats 0--3 and precompute beat 4

For slots 0--31:

```text
b = c-12; j=s; r=2*b+j/16; l=j%16; t=9
```

Those 32 products close result beat `b`. In the same issue, slots 32--95 do:

```text
q = s-32; j=q/2; k=q%2
r=8+j/16; l=j%16; t=2*(c-12)+k
```

Thus beat4 receives taps 0--7, two per accumulator per cycle. At `c==12`, its
old accumulator is ignored and the base is `bias[8 or 9]`. All 96 slots remain
active.

### Cycle 16: close beat 4

- slots 0--31: scalar `j=s`, rows 8/9, tap8;
- slots 32--63: scalar `j=s-32`, rows 8/9, tap9;
- slots 64--95: inactive and driven to signed zero operands.

Each beat4 result uses `acc_q + product(tap8) + product(tap9)` in the same
26-bit expression. It must not compare the pre-edge accumulator. Active-mask
population is 96 on cycles 0--15 and 64 on cycle16. Per tile the ledger is:

| Work | Products |
|---|---:|
| beat0--3 taps0--8 | `4*32*9 = 1152` |
| beat0--3 tap9 closures | `4*32 = 128` |
| beat4 taps0--9 | `32*10 = 320` |
| Total | **1600** |

No product crosses a tile boundary, no slot computes two products in one
cycle, and no second multiplier pool exists.

## 4. Ports

The future top shall use the exact M273r2 list below so that an A/B harness can
instantiate either top without an adapter. Frozen parameters are `TAG_W=48`,
`FIFO_DEPTH=16`; any other value is an elaboration fatal.

| Direction | Port | Width / meaning |
|---|---|---|
| in | `clk_core`, `rst_core` | synchronous active-high reset |
| in | `config_valid`, `config_data`, `config_last` | 1, 256, 1 |
| out | `config_ready`, `config_accept` | accept is valid && ready |
| in | `raw_valid`, `raw_data`, `raw_last`, `raw_tag` | 1, 256, 1, 48 |
| out | `raw_ready`, `raw_accept` | accept is valid && ready |
| out | `result_valid`, `result_tag`, `result_beat` | 1, 48, 3 |
| out | `result_valid_bits`, `result_data` | 48, 48 |
| in | `result_ready` | output backpressure |
| out | `result_accept` | valid && ready |
| in | `release_valid` | context release request |
| out | `release_ready`, `release_accept` | drain-qualified handshake |
| out | `tile_done_valid`, `tile_done_tag` | pulse when beat4 enters FIFO |
| out | `context_retire_valid`, `context_retire_cycles` | registered pulse/count |
| out | `config_loaded`, `protocol_error`, `busy` | lifecycle/status |
| out | `stage1_issue` | alias of actual `dense_issue` |
| out | `stage2_issue` | tied zero |
| out | `product_push`, `fifo_push` | both equal direct dense-to-FIFO push |
| out | `product_replace` | tied zero; Fixed has no product register |
| out | `fifo_pop` | equals result_accept |
| out | `result_fifo_occupancy` | 5 bits, range 0--16 |
| out | `raw_bank_occupancy` | 2 bits, range 0--2 |
| out | `intermediate_bank_occupancy` | tied zero |
| out | `debug_config_beats`, `debug_raw_beats`, `debug_tiles_loaded` | 32 each |
| out | `debug_stage1_issues`, `debug_stage1_done` | dense issue cycles / tiles |
| out | `debug_stage2_issues`, `debug_stage2_done` | tied zero |
| out | `debug_product_pushes`, `debug_result_departures` | FIFO pushes / pops |
| out | `debug_product_replacements` | tied zero |
| out | `debug_context_cycles` | from first config accept |

An internal 96-bit `multiplier_active_mask` and 5-bit `dense_cycle_q` are
mandatory bind targets even though they are not public ports.

## 5. Orthogonal state machines and ownership

A single enum cannot safely describe overlapping raw fill, dense execution,
and FIFO drain. Implement these orthogonal registered machines:

1. **Config/context:** `config_beat_q` 0--4, 1280-bit candidate/capture,
   `config_loaded_q`, decoded weights/bias/threshold, context counters.
2. **Raw fill:** `fill_active_q`, `fill_bank_q`, `fill_beat_q`, `fill_tag_q`.
3. **Raw ownership:** two 1280-bit banks, two tags, two 32-bit order values,
   `raw_owned_q[1:0]`, `raw_ready_q[1:0]`.
4. **Dense:** `dense_active_q`, `dense_raw_bank_q`, `dense_cycle_q[4:0]`, 160
   signed25 accumulators.
5. **FIFO:** 16 entries of `{tag[47:0], beat[2:0], valid_bits[47:0],
   data[47:0]}`, read/write pointers and 5-bit count.
6. **Fault:** one sticky `protocol_error_q`; reset is the only recovery.

Raw beat0 reserves a free bank. Beats1--4 continue in that bank even if both
banks are owned. A bank becomes compute-ready only after legal beat4. It stays
owned until dense cycle16 issues. If both banks are ready, the lower unsigned
monotonic completion order wins; fixed bank priority is forbidden. The frozen
admitted context domain must stay below the 32-bit order-counter wrap point.

`raw_ready` is true only when configured, not quarantined, and either a packet
is already filling or a bank is free. Completion-to-new-ingress same-cycle
bypass is forbidden. A legal fifth raw beat can make a bank ready only after
the edge; dense can start it on the next edge.

Dense start atomically removes the selected bank from `raw_ready_q` but retains
ownership. Cycles 0--11 issue regardless of FIFO credit. Cycles 12--16 issue
only with:

```text
fifo_credit = (fifo_count_q < 16) || result_accept
```

If a closing cycle has no credit, `dense_cycle_q`, all accumulators, raw
ownership, tag, FIFO write pointer, and debug issue/push counters hold. On an
issued closing cycle, accumulation and exactly one FIFO push are atomic.
Cycle16 also releases raw ownership and pulses tile_done for that tag.

The FIFO is registered with no empty fallthrough. A push into empty storage is
not visible for departure until the next cycle. At full count, simultaneous
head pop and tail push are legal and leave count 16. FIFO head payload remains
stable while `result_valid && !result_ready`.

`work_empty` is exactly:

```text
!fill_active_q && raw_owned_q==0 && !dense_active_q && fifo_count_q==0
```

Release requires reset low, config loaded, no fault, at least one completed raw
tile in the context, `work_empty`, and `!raw_valid`. Release never overlaps a
new context. Stale data bits need not clear on release, but every stale config,
raw, accumulator, and FIFO payload must be overwritten before its valid/owned
state can expose it.

## 6. Protocol and fault-edge semantics

Legal configuration has last low on beats0--3, last high on beat4, and zero
padding. Legal raw has last low on beats0--3, high on beat4, and a constant tag
across all five accepted beats.

Fault causes are:

- accepted config early-last or missing beat4 last;
- accepted final config with nonzero padding;
- accepted raw early-last or missing beat4 last;
- accepted continuation raw beat with tag drift;
- zero-tile release attempt after legal configuration when work is empty.

The candidate frame including the current accepted beat is checked at the
edge. An offending config/raw payload is not committed. `fault_event` may only
set `protocol_error_q`; it must not combinationally qualify `protocol_error`,
`result_valid`, issue, push, or ready. Transfers already advertised before the
fault edge (including a dense FIFO push and result pop) commit consistently.
From the next cycle, sticky quarantine suppresses every config/raw/release
accept, result_valid, dense issue, FIFO push, and FIFO pop until reset.

Held release while busy is backpressure, not a fault. If `raw_valid` and
zero-tile `release_valid` coincide, raw has priority: release stays unaccepted
and the zero-tile fault is not raised on that edge.

## 7. Clean cycle equality and M265 reconciliation

Measurement is inclusive from the first configuration-beat acceptance through
release acceptance. For `N>=1`, gap-free five-beat config/raw input and
continuously high `result_ready`:

```text
cycles = 17*N + 12
N=1 -> 29 cycles
N=4 -> 80 cycles
```

The fixed overhead is five config beats, five beats to make the first raw bank
ready, one registered-FIFO tail departure, and one release cycle; subsequent
raw fills and result departures overlap the 17-cycle dense service. Across
7,318,350 tiles and 45 contexts:

```text
17*7,318,350 + 12*45 = 124,412,490
```

The matched M265 rank3 value remains 36,592,605 and the isolated analytical
ratio remains 3.3999353148x. M518 does not admit that as RTL speedup until the
VCS and matched physical gates below pass.

## 8. Same-resource fairness

The fair statement is narrow:

- both tops have the same raw/result/tag/release transport widths, two raw
  banks, 16-entry registered FIFO, clock/reset, downstream ready trace, and
  no cross-context work;
- configuration uses the same 256-bit transport/ready trace, while each design
  pays its own payload length (Fixed five beats; rank3 six);
- Fixed has exactly 96 signed INT8 multiplier slots and never packs leftover
  products across tiles;
- rank3 has 96 signed INT8 stage1 slots **plus distinct** CSD stage2 resources;
- each synthesized top must include all of its own config and working state.

This is resource-schedule and boundary matching, not equal silicon area. Only
matched DC reports can establish area; only area plus frequency and energy can
support throughput/area or energy claims.

## 9. Existing RTL search and reuse decision

No equivalent Fixed top exists in the inspected tree.

| Source | Why it is not M518 |
|---|---|
| `rtl_m273/m273_integrated_rank3_atlif.sv` | complete rank3, 6-beat/1349-bit config, intermediate banks, CSD stage2 and product register; comment explicitly says Fixed is absent |
| `rtl_m31/qfit_atlif_unified_t10_t2_stream_core.sv` | T10 is rank3 and T2 is dense; parameter and raw interfaces differ; no M273 frame/release/fault contract |
| `rtl_m27/qfit_atlif_rank3_exact96_core.sv` | rank3 wide request core, not dense T10 and not the matched streaming boundary |
| `rtl_m30/*` | rank3 resident/T2 helpers with different protocols |
| `rtl_m37_r10/qfit_atlif_csd_reconstruct_t10.sv` | stage2-only CSD reconstruction, not a complete Fixed candidate |
| `rtl/atlif_unified_encode_unit.v`, `rtl_allbinary/*` | comparator/legacy membrane leaf units, no dense matrix engine or matched boundary |

Safest implementation strategy:

1. Create new `rtl_m518/m518_matched_fixed_t10_atlif.sv`.
2. Port the M273r2 config/raw/FIFO/release/fault structure by inspection while
   changing only the config length/layout and deleting rank3 intermediate and
   product-register state.
3. Add the schedule above as new code and bind-visible internal ledgers.
4. Create new `tb_m518/`, `verif_m518/`, contract, exact-SHA runner and result
   directories. Do not edit M273/M285/M289 evidence.

Adding a mode parameter to M273 is higher risk: constant-mode optimization,
six-versus-five-beat framing, stage debug semantics, Formality compare points,
and rank3 regression become coupled. Forking M31 is also a rewrite because its
top boundary is incompatible. Common transport factoring may be considered
only after both sibling tops have independent equivalence receipts.

## 10. VCS attack/defense matrix

Use Synopsys VCS V-2023.12-SP1 with exact-SHA preflight. Wrong/missing SHA must
stop before compile. A passing run has zero assertion failures and zero X/Z on
accepted payload/control.

| ID | Attack / positive test | Required result |
|---|---|---|
| V01 | N=1 and N=4 gap-free, ready-high | exact 29 and 80 cycles |
| V02 | independent integer oracle over zero, random, alternating-sign, and all extreme signed8 operands | every tag/beat/mask/event exact |
| V03 | bias+sum just below/at/above both Q24 saturation rails; threshold equality | exact saturate-then-compare |
| V04 | bind slot ledger on every issue | masks 96 for c0--15, 64 for c16; 1600 unique `(r,l,t)` products/tile |
| V05 | back-to-back tiles filling both raw banks | no issue bubble beyond the 17-cycle tile service and oldest tag first |
| V06 | deliberately make bank1 older than bank0 | bank1 issues first; fixed bank0 priority forbidden |
| V07 | hold result_ready low until FIFO full, then pulse | closing phase stalls atomically; prologue may progress; no loss/duplication |
| V08 | full FIFO simultaneous result pop and closing push | both commit, occupancy stays 16, pointers each advance once |
| V09 | stall FIFO head for multiple cycles | result tag/beat/mask/data stable |
| V10 | config early-last on beats0--3 and missing beat4 last | registered sticky fault; offending beat not committed |
| V11 | toggle each padding bit 1064--1279 | every nonzero padding case faults |
| V12 | raw early-last beats0--3, missing beat4 last, tag drift beats1 and4 | registered sticky fault; offending raw beat not committed |
| V13 | sustained valid around every legal config/raw phase with two half-cycle probes | zero protocol pulses and zero issue/result retractions |
| V14 | malformed raw acceptance coincident with FIFO full pop+push | bad raw payload suppressed; established pop/push commit; next-cycle quarantine |
| V15 | legal config then zero-tile release held eight cycles | no release/retire; one registered fault; sticky reset-only quarantine |
| V16 | release held during partial raw, compute cycles0/12/16, and FIFO drain | no early accept; exactly one accept only after full drain |
| V17 | simultaneous raw_valid and release_valid at empty configured context | raw wins, no zero-tile fault on that edge |
| V18 | reset during partial config/raw, every dense phase, FIFO stall, and quarantine | deterministic empty reset state and clean next context |
| V19 | two legal contexts with opposite data, no payload clearing between releases | no stale config/raw/accumulator/FIFO influence |
| V20 | wrong RTL/TB/SVA/contract/filelist/docs identity | launcher fails before VCS starts and creates no positive receipt |

Required SVA includes handshake identities, FIFO/head stability and bounds,
registered-fault stickiness, post-fault fail-closed behavior, release drain,
raw ownership conservation, oldest-order selection, phase hold on close stall,
push iff issued phase12--16, beat=`phase-12`, exactly 17 issues/five pushes per
tile, active-mask population, 26-to25 overflow assertion, result shape, and
departure/push/tile conservation. Covers must hit dual-ready arbitration,
phase12 and phase16 stalls, full pop/push, fault-edge pop/push, all five result
beats, zero-tile fault, release, and reset recovery.

## 11. DC, Formality and PTPX gates

### Matched DC

Run Fixed and the exact M273r2 source with the M289 settings:

- DC V-2023.12-SP3, TSMC28 HPC+ `tcbn28hpcplusbwp35p140`;
- max `ssg0p9v125c`, min `ffg1p05vm40c`;
- 3.0 ns clock, 0.2 ns setup and 0.05 ns hold uncertainty;
- 0.25 ns input/output delay, 0.01 pF output load;
- synthesis max fanout24, paper screen fanout32;
- flattened standard-cell logic, ideal clock, ZeroWireload, zero macros.

Fail unless source hashes pass before `dc_shell`, exit is zero, mapped Verilog,
SDC/DDC/SVF are nonempty, `check_design` and `check_timing` pass, setup/hold are
MET, and max delay/min delay/capacitance/transition/fanout have zero violations.
Seal cell/comb/sequential area, leaf/sequential cell count, logic levels,
critical path and start/end points. Report M289's rank3 anchor separately:
102,852.287739 um2, 9,639 sequential cells, 2.78 ns path, zero macros. Do not
call areas matched before both exact runs pass; ideal-clock/ZeroWireload area is
still not paper physical PPA.

### Formality

Use the exact M518 RTL and elaborated parameters as reference, its own DC mapped
netlist as implementation, and the DC SVF. Compare every architectural and
debug output, not only result_data. Fail on any unmatched/aborted/inconclusive
compare point, black box, undriven reference, unresolved array, or non-equivalent
reset state. `verify` must succeed after a clean read; a deliberately modified
weight bit mapping must produce a negative control failure. Formality proves
synthesis equivalence, not the high-level 17-cycle arithmetic theorem; the
slot-ledger SVA/oracle remains independently required.

### SAIF/PTPX

Generate gate-level SAIF from a VCS campaign only after VCS/Formality pass.
Fixed and rank3 must use the same clock, reset convention, raw tags/data,
result-ready schedules, context measurement window, library/corner, and output
load. Measure both (a) full first-config-accept through release-accept energy
and (b) explicitly labelled steady service; never mix them. Fail if clock/reset
activity is missing, any sequential output is unannotated, annotated leaf-pin
coverage is below 95%, time windows differ, or PTPX reports unresolved cells.
Seal internal, switching, leakage, total power, window time, energy/context,
energy/tile and annotation coverage. Until checkpoint-derived Fixed/rank3
inputs and trained-rank3 accuracy are admitted, PTPX is engineering evidence,
not an efficiency or paper-energy claim.

## 12. Findings and admission boundary

P0 findings: none in this specification.

P1 findings:

1. **P1-M518-RTL-ABSENT:** the executable RTL/TB/SVA and exact-SHA VCS receipt
   do not yet exist. This document authorizes implementation, not functionality.
2. **P1-SCHEDULE-NOT-INDEPENDENTLY-PROVED:** the explicit map closes M265's
   ambiguity, but it still needs frozen-contract review, oracle VCS, slot SVA,
   and an independent hammer.
3. **P1-MATCHED-PHYSICAL-OPEN:** no Fixed DC/Formality/SAIF/PTPX result exists;
   M289/M302 only admit rank3 logic-only DC under ideal/ZeroWireload assumptions.
4. **P1-ALGORITHM-WORKLOAD-OPEN:** trained rank3 accuracy, checkpoint-derived
   factors/range, representative matched switching, macro-inclusive physical
   PPA, and system speedup remain unadmitted.

Admitted now: a concrete implementable Fixed RTL specification and a justified
new-file decision. Not admitted: M518 RTL correctness, area equality, RTL
speedup, throughput/area, power, energy, accuracy, system speedup, paper PPA,
or headline.

## 13. Evidence identity

The review inspected the current exact M273r2 RTL SHA
`11d5c6c4f5f0c44ea0a8c2b815683a2e1ab2dbb007bd3afdca0d8ae9e901067d`,
TB `4c7d11e...`, SVA `b5909fd6...`; M265 contract `88b77bea...` and result
`7fa5d46a...`; M273 contract `e1d21925...`; M285 contract `0f0ebe41...`;
M286 review `da3fb7ad...`; M289 contract `07efe93c...`, area report
`88edb124...`, QoR `435b2041...`; and M302 contract `e8ca72d3...` and review
`aed54de6...`. `docs/359` was not read for content or modified; its evidence
identity remains the frozen `dedde7ce...` reported by those contracts.
