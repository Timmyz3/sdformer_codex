# M518 matched Fixed T10 ATLIF independent static hammer r8

Date: 2026-08-27  
Verdict: `STATIC_GO__EXACT_SHA_ONE_SHOT_R8_VCS_AUTHORIZED`  
Score: **98/100**  
Findings: **P0=0, P1=1, P2=1**

This was an independent, receipt-blind, source-only review. I did not execute
the candidate runner, VCS, DC, Formality, PT/PTPX, or any open-source EDA
tool. I did not modify the author RTL, SVA, TB, contract, runner, or
`docs/359`.

## Decision

The r8 testbench-only post-combinational-settle repair is statically admitted.
Exactly one invocation of this frozen runner is authorized:

```text
dc_handoff/scripts/run_vcs_m518_matched_fixed_t10_atlif_r8_exact.sh
SHA256 fe457d7bbf93e72e913c55427696fb782dcc00dee80c74b1f4dba9c3edd01a52
```

The operator must set both
`M518_EXPECTED_RUNNER_SHA256=fe457d7bbf93e72e913c55427696fb782dcc00dee80c74b1f4dba9c3edd01a52`
and
`M518_EXPECTED_STATIC_ADMISSION_SHA256=e28022f96b6f0026905c796d977568e5ca69bd9c6d9ec9882be7bd3dc768f5ff`,
leave `M518_RUN_DIR` unset, and use the absent canonical path
`results/m518_matched_fixed_t10_atlif_vcs_r8_exact_20260827`. The exact launch
admission and both of its seals are at
`contracts/m518_matched_fixed_t10_atlif_vcs_launch_admission_r8_20260827.json`.

## Minimal repair and preserved campaign

- The unique functional source delta is exactly `#0.2` after the V08 phase16
  negedge drives `result_ready=0` and before the combinational `fifo_credit`
  sample. With a 10 ns clock it leaves 4.8 ns before the next posedge.
- Removing only that unique delay reconstructs frozen r7 TB SHA256
  `a2de78ac5a3c537e03113f06552a09808426170d188d39e462b500b0c865eb12`.
  Therefore every other TB byte, including V01--V20, V06, numeric and
  conservation oracles, the PASS signature, and all cover requirements, is
  unchanged.
- RTL and SVA remain byte exact to r7. The SVA contains exactly 51 assertion
  labels and 25 cover labels; the runner's nonzero-cover gate is the identical
  25-name set.
- There are zero `$deposit`, DUT `force/release`, hierarchical DUT-state LHS,
  or writing binds. The DUT retains one `always_ff` block and no
  `always @(posedge)` downgrade.

## Provenance and runner

The author request member and outer seals pass. All **28/28** runner SHA-map
paths exist, are unique, and match. The sealed r7 static review and r7 failure
diagnosis both pass member and outer seal verification. The r7 result remains
an immutable diagnostic failure: its failure marker is present,
`RUN_COMPLETE.txt` is absent, and the independent classification is a TB
active-region settle P1 rather than an RTL P0.

The runner is mode 0755 and passes `bash -n`. All three embedded Python blocks
parse under the installed Python 3.6.8. It rejects any defined `M518_RUN_DIR`
and checks the exact runner SHA plus the exact double-sealed static-admission
SHA before canonical result creation. Atomic `mkdir` prevents a second attempt.
The wrong-TB control must exit 10, create no tool or positive receipt, and seal
only its disjoint negative directory before any VCS command. The positive path
contains exactly one Full64 identity query, one Full64 compile with one
`M518_VCS_V06_HARNESS` define, and one fixed-seed simulation. No open-source
EDA command exists in the runner.

## Findings and boundary

The P1 is inherited publication atomicity: `RUN_COMPLETE.txt` is written just
before the final member and outer seals. The active failure trap keeps a late
sealing failure non-citable, but it can leave contradictory markers; the
mandatory independent post-run receipt hammer must reject such topology. The
P2 is the normal static residual: compilation, execution, assertion and cover
closure, and the scoreboard cannot be proved without the authorized run.

Admitted now: exact r8 source identity, the one-delay reverse proof, unchanged
campaign/oracle structure, and one-shot VCS authorization. Not admitted:
SystemVerilog compilation, VCS behavior, V01--V20 runtime, numeric or
production equivalence, RTL cycles, DC, Formality, STA, power, energy,
speedup, PPA, system speedup, or a headline claim.
