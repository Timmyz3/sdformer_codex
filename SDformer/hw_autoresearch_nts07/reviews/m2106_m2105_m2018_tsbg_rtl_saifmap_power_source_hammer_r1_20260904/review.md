# M2106 independent source hammer: M2105 matched TSBG RTL-SAIF/saif_map/PTPX

## Verdict

**PASS, 98/100; P0/P1/P2 = 0/0/0.**  The final frozen M2105 source identity is
authorized for exactly one serial matched campaign: one license query, one VCS
compile, two `simv` runs, two DC runs, two PTPX runs, and two SAIF files.  There
is no automatic retry and no reuse of previous failed activity, netlist, map, or
power artifacts.  This review invoked no EDA, `lmstat`, or GPU tool.

The review initially observed the PTPX Tcl changing while the source author was
still adding pre-power gates.  No verdict was issued on that moving identity.
The hammer was restarted from the final contract SHA
`64a5b888aa7519f5a69c9b141314e1cfb9c2be46b96bcf9a1b2b4aa435acb4fa`;
all identities remained stable throughout the final 45-check run.

## Matched experiment

Both axes instantiate the same
`m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend` RTL and execute frozen ep34
global slot 42: sample 0, layer 28 FC1, token 0, 48 source groups.  The only DUT
parameter difference is `SCHEDULE_MODE=0` for ordinary LRU4 versus
`SCHEDULE_MODE=1` for TSBG-B4.  Ports, cache capacity, clock constraint, library
flow, workload payload, and the separately declared 294,912-byte weight SRAM
are common.

The testbench fail-closes on the complete workload ledger.  Ordinary must
measure 20,292 execute cycles and 14,304 scalar 128-bit weight reads; TSBG must
measure 7,569 cycles and 4,608 reads.  Both must conserve 149 rows, 1,278 issue
events, 29,472 signed products, and 24 commits, with the axis-specific exact
cache hit/miss/eviction counts.  Descriptor preload is excluded from both
measurement windows.

## Activity and mapping closure

- The UCLI files record only `core.dut_base.implementation` or
  `core.dut_tsbg.implementation`; the testbench, assertions, and directed SRAM
  model are excluded.
- Each RTL SAIF must have zero aggregate `TX`, exact per-record
  `T0+T1+TX=DURATION`, and duration equal to the frozen cycle count times 3 ns.
  At least 100 signal records, 20 toggling records, and nonzero activity on all
  eight request/response/bridge/commit valid-or-accept cones are required.
- DC starts native `saif_map` before analysis, then emits both default and
  essential PTPX transformation maps from the freshly synthesized axis.
  Intra-class contradictions fail; intersection, union, and cross-class target
  differences are retained for inspection rather than silently discarded.
- PrimeTime sources the fresh default map followed by the essential map, then
  reads the corresponding RTL SAIF using the exact hierarchy strip path.
  Before any `report_power`, the Tcl itself requires at least 95% direct-net
  annotation, at least 95% fully annotated leaf cells, at least 20% nonzero-net
  coverage, zero inconsistent annotation rows, nonzero activity on all eight
  critical public cones, and a successful `check_power`.  The Python result
  parser repeats these gates and checks switching/internal/leakage/total power
  and power-times-duration energy arithmetic.

Mechanical mutation tests independently demonstrated that a nonzero SAIF
`TX`, a one-nanosecond duration drift, an intra-class RTL-to-gate conflict, or
an inconsistent power-component sum is rejected.

## SRAM and publication boundary

The PTPX number, if the campaign and the later independent result hammer pass,
is **pre-layout standard-cell logic energy** driven by transformation-mapped
RTL DUT-only activity.  It is not mapped-gate VCS activity.  The common 288 KiB
weight SRAM has neither area nor dynamic energy inside the PTPX total.  The
exact request counts are intentionally emitted separately; a per-read SRAM
energy may be applied only in a separately labelled foundry-QRT/model column.

This source does not authorize claims of hold closure, post-route PPA, Fmax,
energy/frame, system speedup, or paper-ready PPA.  No measured number is
admitted until M2107 completes and a separate independent result hammer seals
the raw reports.

## Execution authorization

Authorized once, serially, on the exact identities in `review.json`:

- license queries: 1
- VCS compiles: 1
- `simv` runs / SAIF files: 2 / 2
- DC runs: 2
- PTPX runs: 2
- automatic retry: false
- reuse of old artifacts: false

The execution owner should launch only after the existing same-UID EDA queue is
empty.  This is operational scheduling, not permission to create another
attempt if the one-shot coordinate is consumed.
