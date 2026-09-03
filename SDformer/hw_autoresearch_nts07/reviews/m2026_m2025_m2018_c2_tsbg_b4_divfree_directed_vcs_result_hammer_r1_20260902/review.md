# M2026 independent M2025 VCS-result review

## Verdict

**PASS, 99/100; P0/P1/P2 = 0/0/0.** The sole consumed M2025
attempt and sole published result both verify against their double seals. No
work, lock, retry, failure, or incomplete sibling exists. The receipt and
attempt ledger agree on one license preflight, one VCS compile, one `simv`
execution, and no retry.

The compile log contains one VCS V-2023.12-SP1 banner, parses the exact five
frozen filelist rows in order, selects the expected top, and builds all seven
modules. There is no compile error, unregistered-SVA warning, or ignored
`global_finish_maxfail` warning. Its 24 keyword warnings are confined to task
variables in the frozen M1984 testbench; neither M2018 nor the M2020 adapter is
their source location.

## Functional and protocol result

The simulation contains exactly one complete PASS line:

```text
PASS_M1880_C2_TSBG_B4_REAL_M803_TYPED_SIGNED_DIRECTED rows=48 issues=576 products=9216 commits=24 bundles_base=576 bundles_tsbg=144 scalar_base=4608 scalar_tsbg=1152 stale=1 retired_replay=1 replay_accept=0 reset=2 recovery=1
```

All ten M1970 begin/complete phase pairs occur once. There are 52 load begins,
52 load completions, zero load timeout, and no assertion/error/fatal/watchdog
match. All eleven TSBG-side SVA cover rows have nonzero matches, including bank
backpressure/reorder, positive and negative bridge values, bridge and commit
stall, terminal commit, eviction, weight bundle, stale attack, and reset
recovery. The exact result also proves the directed ledger's 576-to-144 bundle
and 4,608-to-1,152 scalar-request reductions, both 75%.

The independent result hammer passed under Python 3.6 and 3.12 and rejected
8/8 corrupted compile/simulation variants. This reviewer ran no EDA tool and
made no license query. docs/359 remains unchanged.

## Paper boundary

This closes M2018 as **directed G12 functional/protocol/recovery evidence**.
The 75% reduction is citable only as a directed microarchitectural regression,
not as an ep34 or production-G48 performance result. Production G48 dynamics,
same-area comparison, exact component cycles, system speedup, timing, energy,
and headline performance remain unadmitted.
