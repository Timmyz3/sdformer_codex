# Actual hardware progress, 2026-09-05 afternoon

## Completed: matched TSBG power (M2248)

The six PTPX jobs completed using mapped M2018 ordinary/B4 logic, unchanged
3-ns constraints and the same three preselected reuse-density windows.

| Window | Ordinary / B4 energy (nJ) | Energy reduction | Average power change |
| --- | ---: | ---: | ---: |
| low | 1101.81 / 1101.61 | approximately zero | -0.018% |
| median | 1570.48 / 1042.02 | 33.65% | +0.054% |
| high | 5200.13 / 1775.33 | 65.86% | +0.757% |

This is post-load **logic execution energy**, TT 0.9 V/25 C, ideal clock with
0.1-ns slew, ZeroWireload, verification INT8 weights. It excludes SRAM banks,
CTS/SPEF, preload, and full-network execution. Three selected windows are not
a population energy estimate. Clock-pin internal power is 97–98% of the total:
the principal current benefit is finishing sooner, not lower average power.
Nonclock dynamic execution energy separately decreases 1.60/21.94/44.87%; this
report decomposition is not a clock-gated hardware measurement.

Both axes have 1712 input nets and 74460 sequential output nets mapped for
toggle and static probability. VCS's enum SAIF naming issue was fixed by a
plain-vector state probe; main ns and supplemental ps windows agree. The 122
zero-slew warnings are constant-tie inputs, and the four default-activity
aliases have no connected pins. Independent reviewer recomputation agrees.
There is no new hashing or authorization ladder.

Evidence: `results/m2248_matched_power/summary.json`, six `power.rpt` files,
and `system_simulator/scripts/summarize_m2248_matched_power.py`.

## Completed: consumer-scoped bank fill RTL (M2249)

Actual M803 + complete C2 frontend, four-row cache, independent Acc24 contexts.
Both ordinary and B4 support bank masks and partial refill; B4 forms the union
of the four consumer masks before reading. It shares weights, never signed
products. Reset is required before changing weight identity.

| Window | Ordinary / B4 cycles | Ratio | Ordinary / B4 bank reads |
| --- | ---: | ---: | ---: |
| low | 2044 / 2044 | 1.000x | 312 / 312 |
| median | 2904 / 2189 | 1.327x | 384 / 300 |
| high | 12844 / 5848 | 2.196x | 2442 / 1098 |

VCS passes all committed lane values and request counts. Two additional
four-bundle sequences pass partial low/high-bank fills, warm zero-read reuse,
fifth-row eviction and negative-source times -128. Earlier compile-order and
floating-point TB-time failures remain separate; the final run uses integer
picosecond comparison and `-no_save`.

Evidence: `results/m2249_bank_selective_3xd7_3fp/result.json`, `rtl_m2249/`,
`tb_m2249/`, and `dc_handoff/scripts/run_m2249_bank_selective_vcs.py`.
This new variant has no area/timing/power result yet; M2248 belongs to M2018.

## Completed: broader CPU replay (M2252)

Six M2249 pilot axes calibrate with zero cycle/read mismatch. The literal FSM
model then counts 4320 independently reset G48 chunks once:
14,508,203 / 8,052,073 cycles = **1.8018x**; reads 2,623,644 / 1,604,430,
**38.85% fewer**, versus mask-aware ordinary LRU4. The memory-ready equation
uses the actual clean-reset phase, not a fitted latency or scale factor.

This is a calibrated CPU experiment, not 4320 RTL runs, full-FC latency, or a
system result. Output-tile multiplicity and continuation overhead are absent.
Evidence: `results/m2252_masked_c2_cycle_model/result.json`.

## Active / next

- Existing ordinary/B4 mapped areas reproduce 249710.45 / 249739.81 um2.
  Setup +26.4/+68.8 ps; hold -16.4 ps. First hold-only optimization inserted no
  cells; a hold-priority attempt also inserted no cells. A common clock-gated
  comparison is running with unchanged clock/I/O limits. The first ICG style
  was unavailable in the library; the corrected style uses its existing
  precontrol ICG cells, not an invented library primitive.
- Compare common clock gating on both axes before using the ungated energy
  numbers as the best implementation comparison.
- Three real ep34 FC weight candidates exported on CPU (M2251), without
  training or changing the checkpoint. Both RTL axes now pass all three
  real-weight numerical/SAIF windows (M2253); this is an activity sensitivity,
  not an admitted FC quantization/AEE result.
- C1 is unchanged. No new fourth contribution or full-system headline is added.
- Strong Accept is not certified by these results. The useful improvement is
  actual component power evidence plus a functioning stronger-baseline RTL
  extension; mapped closure, fair gating and manuscript integration remain.

## September 5, 16:02: ordinary clock-gated timing and preservation close

The original gated DC reduced area to 223190.85 um2 but left -16.496 ps hold.
A data/reset endpoint buffer ECO closed hold, and sizing three shared-control
cells recovered setup without weakening constraints. The final ordinary axis
is `results/m2255_hold_buffers_wq4h9t91`: 234537.41 um2, setup +122.717 ps,
hold +0.004 ps, still ideal-clock/ZeroWireload, no routing. Hold uncertainty
remains 50 ps; the tiny positive reported margin is not a post-CTS guarantee.

`results/m2255_mapped_preservation_x62ps2cz` proves preservation against the
original ungated mapped ordinary netlist: 77180 compare points PASS, no failing
or aborted points. The 1163 inserted clock-gating latches are recognized by
Formality's latch-based clock-gate analysis, not cut as functional data paths.
This is mapped-to-mapped preservation, not an added RTL-to-gate campaign.

The TSBG gated axis is running. Do not transfer the
ordinary result to TSBG, change the paper's matched area pair yet, or use
M2248 ungated power for this netlist. The consumer-union causal third RTL mode
has passed; its population model and the three actual RTL anchors are
separated in M2254.

## September 5, 16:31: direct mapped activity and first gated power point

The old RTL map is not valid after register merging during clock insertion;
the first attempted mapped-ECO/old-map PTPX was stopped and is not used.
Instead, M2256 runs the actual gated netlist with the same port-level workload,
scalar arithmetic scoreboard, request ledger, and candidate ep34 FC weights.
DC's flattened port order is explicitly reconstructed; no state is forced.
The middle window passes at the unchanged 6733 cycles. Low and high also pass
4717/22294 cycles using the same compiled mapped design.

The first direct-SAIF PTPX point is 21.9369202 mW, ordinary/median, at TT
0.9 V/25 C. All 257197 reported nets, including 1163 clock-gate outputs, obtain
toggle and probability directly from the gate SAIF; none use default activity.
The report includes 74349 sequential output nets and 1712 primary input nets.
This is zero-delay mapped activity plus prelayout library power, not routed
glitch power or external SRAM energy. It must not be compared directly with
M2248 as a pure gating improvement: M2248 used verification weights and RTL
activity propagation. The fair TSBG gated/real-weight counterpart is pending.

Independent milestone review confirms unchanged SDC, the 30016 buffer ECO and
three cell sizes, and the mapped-preservation result. The unfiltered min-path
report and full timing constraint report support reported hold MET; an empty
slack-filter query by itself would not. There is still a zero-valued leakage
optimization target violation, so this is not "all constraints PASS".
