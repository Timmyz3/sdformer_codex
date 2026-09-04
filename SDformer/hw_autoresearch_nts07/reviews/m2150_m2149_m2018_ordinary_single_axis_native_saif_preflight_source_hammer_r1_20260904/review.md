# M2150 independent M2149 source hammer

## Verdict

**PASS, 98/100; P0/P1/P2 = 0/0/0.** Exactly one fresh M2151
ordinary-only causal native-RTL-SAIF preflight is authorized. The review itself
performed zero license queries, VCS compiles, `simv` runs, SAIF acquisitions,
DC, PT/PTPX, ICC2, or GPU work.

This authority is deliberately narrow. M2151 may make one license query, one
compile, one ordinary `simv` run, and write/admit one fresh DUT-only SAIF. Any
failure consumes M2151; there is no retry. A raw pass still requires an
independent M2152 result hammer and is not paper-citable power, energy, PPA, or
speedup evidence.

## The M2143 topology defect is removed

The four-source filelist ends in a new self-contained testbench. That
testbench directly instantiates exactly one
`m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend`, explicitly at
`SCHEDULE_MODE=0`, plus eight ordinary memory-bank models and one ordinary SVA
instance. It neither compiles the old parent dual-axis testbench nor compiles
the public-name adapter. There is no schedule-mode-1 instance, `dut_tsbg`,
TSBG load-valid, TSBG completion wait/counter, or TSBG hierarchical execution
path. Thus UCLI scoping is no longer being mistaken for execution isolation:
only one execution axis exists in the elaborated source set.

## Causal SAIF protocol and functional gates

- UCLI selects only `dut_ordinary` and enables the native observer before the
  first run, so reset and all 383 descriptor-preload cycles are observed.
- Before the first stop, the testbench performs an observation-only census of
  all 228 named internal elements: 192 row-live, four cache-valid, eight
  adapter slot-valid, 16 bridge-overflow, and eight response-shape elements.
  There is no DUT force or deposit.
- UCLI resets activity history only after that first stop, then runs the exact
  20,292-cycle / 60,876-ns ordinary measurement window before disabling and
  reporting the single DUT scope.
- The frozen slot-42 fixture identity and expected ledger are checked before
  the pass marker: 149 rows, 1,278 issues, 29,472 signed products, 24 commits,
  1,788 weight bundles, and 14,304 scalar reads/responses. All 24
  context/slice accumulators are checked against the independent INT8
  arithmetic scoreboard; reordering and independent bank stalls must both
  occur.
- The parser admits only one occurrence of every causal marker in strict
  order, the exact ledger and duration, exactly 93,971 SAIF records, TX=0 on
  every record and in aggregate, per-record `T0+T1+TX` conservation, at least
  20 toggled records, and nonzero activity in all eight request, response,
  bridge, and commit valid/accept cones.

## Integrity, isolation, and one-shot semantics

The 11-entry source inventory, contract sidecar and outer seal, exhaustive
four-member author receipt seal, and exhaustive five-member M2143 failure seal
all verify byte-for-byte. The pinned VCS and `lmutil` binaries are regular
files with the contracted hashes. `docs/359` remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
M2151 result, attempt, and lock were absent during review.

The runner validates all source and review identities before consumption,
rejects active same-UID VCS/simv/DC/PT/ICC2 processes, creates the lock and
sealed attempt before the license query, uses one isolated temporary build and
staging directory, and atomically publishes only a fully parsed and sealed
result. Its fixed launch has one absolute pinned VCS binary, one compile top,
`+vcs+initreg+random` at compile, `+vcs+initreg+0` at runtime, no SDF or unit
delay, and no loop or retry path. Failure is sealed into quarantine while the
attempt remains consumed.

## Independent fail-closed attacks

Fifty-four valid mutations were rejected with zero unexpected passes:

- 10 topology/filelist attacks: old parent, second direct frontend,
  schedule-mode 1, second DUT/load/wait/path, old parent filelist, public-name
  adapter, and bad source order/cardinality;
- seven UCLI attacks: missing or late enable/reset, substituted second scope,
  duplicate report, and extra run;
- 21 runtime attacks spanning phase uniqueness/order, census completeness and
  bounds, frozen identity, every ledger field, duration, arithmetic marker,
  second-axis marker, and fatal/assertion/mismatch/timeout tokens; and
- 16 SAIF attacks spanning under/over-count, duration, TX, conservation,
  insufficient toggles, duplicate header, negative fields, and each of the
  eight critical public cones.

All attacks used Python and temporary text fixtures only. No EDA or license
tool was invoked.

## Authorized next action

Run the committed M2149 one-shot runner exactly once to create M2151. Do not
edit or retry M2142/M2144, change the M2149 source identity, or add another
execution axis. If M2151 passes, seal it and obtain M2152 independent result
review before drawing even the narrow causal-preflight conclusion.
