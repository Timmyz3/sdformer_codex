# M2126 independent source-hammer request

Review the exact M2125 source identity frozen by contract
`m2125_m2018_tsbg_rtl_saif_window_diagnostic_source_contract_r1_20260904.json`.
Do not invoke a license query, VCS, `simv`, DC, PT, PTPX, ICC2, or any other
EDA executable.

The review must independently and exhaustively establish all of the following:

1. The M2119 quarantine and M2120 failure review are exhaustive double-sealed
   and match the pinned duration (60877.5 ns), record count (93,971), nonzero-TX
   record count (58,277), consumed/no-retry disposition, and source-successor
   boundary.
2. M2125 is additive.  M2117/M2119, M2051, M2018, the fixture, and docs/359 are
   byte-identity pinned and unmodified.
3. The only future EDA path is one M2127 license query, one shared VCS compile,
   two serial `simv` runs, and two fresh DUT-only SAIF files.  DC/PT/PTPX/ICC2,
   reuse, retry, caller-selected path/hash/axis/workload, and a second launch
   path must all be absent.
4. The compile command contains exactly one `+vcs+initreg+random`; each runtime
   command contains exactly one `+vcs+initreg+0`, fixed slot42, and exactly one
   fixed axis selector.  No UNIT_DELAY, SDF, force/release, assertion
   suppression, or X coercion is permitted.
5. The first activity stop occurs one explicit settled negedge after observing
   `full_execute_start_cycle`; the second occurs one explicit settled negedge
   after observing the selected completion.  The UCLI scripts select only the
   corresponding DUT implementation, enable after the first stop, disable at
   the second, then emit exactly one SAIF.
6. The parser requires, for each axis, the exact frozen functional ledger,
   duration=`cycles*3 ns`, exactly 93,971 SAIF records, TX=0 in every record,
   global TX=0, exact T0+T1+TX conservation, and nonzero critical-cone activity.
7. Tool files, all source files, the contract/sidecars, review identity,
   attempt/result/lock freshness, and same-UID collision protection are
   fail-closed and exact-pinned.
8. Passing M2127 would remain a VCS-only diagnostic.  It cannot support power,
   energy, mapped activity, speedup, silicon initialization, or a paper claim;
   a later DC/saif_map/PTPX attempt needs a new independently reviewed source.

Required M2126 machine-readable outputs:

- `review.json` with schema
  `m2126_m2125_m2018_tsbg_rtl_saif_window_diagnostic_source_hammer_r1_v1`;
- status beginning `PASS_M2126` only if P0/P1/P2 are all zero;
- `severity_counts={"p0":0,"p1":0,"p2":0}` and score at least 95;
- `authorization` exactly equal to the future M2127 budget in `selfcheck.json`;
- `identity.runner_sha256=6021c4a9b4297e5527f09006f21dd3a06d98b2a7ad76ffc55ca259029e658815`
  and `identity.contract_sha256=5fadf923093797c8734e1aa54044cd2292e745ec541983254cea7a4c4ce4457e`;
- independent positive checks and mutation rejections for the eight groups
  above;
- `review.md`, mechanical-check evidence, `RUN_COMPLETE.txt`, exhaustive
  `SHA256SUMS`, and `SHA256SUMS.seal.sha256`.

Any uncertainty is a fail-closed review, not an authorization.
