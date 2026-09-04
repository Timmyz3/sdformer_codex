# M2143 independent M2142 source hammer

## Verdict

**FAIL, 84/100; P0/P1/P2 = 1/0/0.** M2144 is **not authorized**.
This independent review performed no license query, VCS compile, `simv`, SAIF
acquisition, DC, PT/PTPX, ICC2, or GPU work.

The acquisition mechanics are otherwise disciplined: the runner contains one
license query, one compile, one `simv`, at most one raw/admitted SAIF, no retry
loop, and no downstream EDA.  The UCLI ordering and parser gates also withstand
the independent mutations listed below.  Those facts do not cure the P0
contract contradiction.

## P0: the claimed ordinary-only simulation executes TSBG

M2142's contract says `ordinary_only=true` and `tsbg_axis_run=false`.  However,
the new wrapper instantiates `tb_m2051_ep34_tsbg_full40_cycle` as `core`.  That
parent testbench unconditionally:

1. instantiates `dut_tsbg` with `SCHEDULE_MODE=1`;
2. asserts `load_valid_tsbg` for every frozen descriptor;
3. drives a separate TSBG memory model and assertions; and
4. records and waits for `tsbg_done_cycle` alongside the baseline.

The `+M2142_AXIS_ORDINARY` plusarg is checked only by the wrapper.  The parent
testbench never consumes it, so it cannot disable the TSBG instance or its
stimulus.  The UCLI scope correctly reports only
`core.dut_base.implementation`, but a SAIF reporting scope is not an execution
gate: the schedule-mode-1 TSBG datapath still elaborates and runs during the
sole simulation.

Therefore a passing raw log/SAIF would still violate the explicit zero-TSBG
budget and would encode a false `tsbg_axis_run=false` claim.  This is a source
authorization failure, not a request to relax the parser or reinterpret
"ordinary-only" as "ordinary-only SAIF output."

## What independently passed

- The six-entry M2142 source inventory, contract sidecar/outer seal, exhaustive
  M2142 selfcheck seal, and exhaustive M2140 failure-hammer seal all verify.
  `docs/359` remains exactly `dedde7ce...`.
- M2139 remains consumed with no retry authority.  M2144 result, attempt, and
  lock were absent during review, but freshness alone grants no authority.
- The fixed launch surface is one `lmstat`, one VCS compile with
  `+vcs+initreg+random`, one runtime with `+vcs+initreg+0`, and one possible
  SAIF; DC/PTPX/ICC2/GPU paths and automatic retry are absent.
- UCLI enables observation before the first run, reaches the first stop only
  after the 228-element five-family observational census, resets activity
  history before the measurement run, then reports the ordinary DUT scope.
- Runtime gates retain the exact 20,292-cycle / 14,304-read / 60,876-ns
  ledger.  SAIF gates retain exactly 93,971 records, TX=0 for every record,
  per-record conservation, at least 20 toggled records, and activity in all
  eight critical request/response/bridge/commit valid/accept families.

## Independent mutations

Fourteen mutations failed closed: six runtime mutations (phase, census,
cycles, reads, duration, marker order), five SAIF mutations (TX, record count,
duration, conservation, critical cone), and three UCLI mutations (late enable,
late reset, TSBG scope substitution).  A clean synthetic runtime and a clean
93,971-record SAIF both passed their intended parser paths.  These tests used
only Python and temporary text fixtures; no simulator or license was invoked.

## Only safe successor

Do not execute M2144 from M2142.  Author a fresh additive identity with a
genuinely single-axis testbench: instantiate only the schedule-mode-0 ordinary
frontend, ordinary memory model, and ordinary assertions.  It must contain no
`dut_tsbg`, no `load_valid_tsbg`, and no TSBG completion dependency; static or
elaboration evidence must prove those paths absent.

Keep every existing causal and activity gate: enable before reset/preload,
observation-only 228-element census, first-stop `power -reset`, exact ordinary
ledger and duration, exact record count, all-record TX=0 and conservation, and
critical toggles.  That fresh source requires a new exhaustive independent
source PASS before one new VCS attempt.
