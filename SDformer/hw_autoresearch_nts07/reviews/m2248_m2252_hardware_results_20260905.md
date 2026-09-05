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
  cells; a hold-priority attempt is running with unchanged clock/I/O limits.
- Compare common clock gating on both axes before using the ungated energy
  numbers as the best implementation comparison.
- Three real ep34 FC weight candidates exported on CPU (M2251), without
  training or changing the checkpoint; power sensitivity is not run yet.
- C1 is unchanged. No new fourth contribution or full-system headline is added.
- Strong Accept is not certified by these results. The useful improvement is
  actual component power evidence plus a functioning stronger-baseline RTL
  extension; mapped closure, fair gating and manuscript integration remain.
