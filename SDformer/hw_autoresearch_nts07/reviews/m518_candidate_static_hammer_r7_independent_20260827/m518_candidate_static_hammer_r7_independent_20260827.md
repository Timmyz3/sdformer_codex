# M518 matched Fixed T10 ATLIF independent static hammer r7

Date: 2026-08-27  
Verdict: `STATIC_GO__EXACT_SHA_ONE_SHOT_R7_VCS_AUTHORIZED`  
Score: **98/100**  
Findings: **P0=0, P1=1, P2=1**

This was a receipt-blind, source-only independent review. I did not execute
the candidate runner, VCS, DC, Formality, PT/PTPX, or any open-source EDA
tool. No author input or `docs/359` was modified.

## Decision

The r7 simulation-only legal-fill repair is statically admitted. Exactly one
invocation of the following runner is authorized:

```text
dc_handoff/scripts/run_vcs_m518_matched_fixed_t10_atlif_r7_exact.sh
SHA256 971c61367d84eea7f6dec73fa4a6efa79cc82fe308e00ffa800409c68b9edce0
```

The operator must set
`M518_EXPECTED_RUNNER_SHA256=971c61367d84eea7f6dec73fa4a6efa79cc82fe308e00ffa800409c68b9edce0`,
leave `M518_RUN_DIR` unset, and use the absent canonical path
`results/m518_matched_fixed_t10_atlif_vcs_r7_exact_20260827`. The exact launch
admission is sealed at
`contracts/m518_matched_fixed_t10_atlif_vcs_launch_admission_r7_20260827.json`.

A separate M519 DC campaign was active at review time. This reviewer did not
launch M518; queue the authorized one-shot until M519 DC exits, then recheck
the runner SHA, 46 inputs, canonical absence, and unset override.

## Legal-fill and production boundary

- The macro-absent RTL has exactly 50 ordered ports and is byte-for-byte the
  M273 public signature. The macro-present image has 52 ports; the only
  additions are `v06_hold_dense_issue` and
  `v06_first_empty_fill_bank1`.
- There are zero `$deposit`, hierarchical `force/release`, hierarchical DUT
  LHS assignments, or writing binds. The DUT retains exactly one `always_ff`
  owner and no `always @(posedge)` downgrade.
- Mechanical removal of the three RTL harness deltas reconstructs frozen r6
  RTL SHA256
  `90e0304bd8fa5bae5f4cf523d8ab7c62b42878a0ce17b75bd62f8b9288600a6a`.
  Removing the bounded V06 TB stimulus/closure deltas reconstructs frozen r6
  TB SHA256
  `e7973a91d04b9f20542b04c58a213e2c8929259768701664dda76721720d2888`.
- V06 now holds dense issue, sends payload/tag 601 to bank1 through five legal
  raw beats, then payload/tag 600 to bank0 through five legal beats. The
  resulting state is ready/owned `11/11`, order1 < order0, with exact tags.
  Releasing the hold crosses a real issue edge and checks bank1 selection,
  post-edge ready `01`, ownership `11`, tag preservation, two-tile
  conservation, and the unchanged numeric scoreboard.
- No DC, Formality, PT, or PTPX script contains the harness define. The
  simulation harness is not paper hardware; macro-off production equivalence
  still requires Formality before a physical claim.

## Provenance and runner

All **46/46** runner SHA-map paths exist, are unique, and match. Six historical
member manifests and six outer seals pass, including the sealed r6 ICPD
diagnosis. The r6 result remains diagnostic: compile rc=255 and the independent
classification is TB instrumentation P1, not RTL P0.

Both embedded Python blocks parse under Python 3.6, and the source-static block
executes cleanly under Python 3.6. The runner rejects any defined
`M518_RUN_DIR`, gates exact runner identity before atomic canonical `mkdir`,
runs a wrong-TB SHA control that must exit 10 before VCS, then verifies the
positive inputs and history. It contains exactly one Full64 VCS identity
query, one Full64 compile, one harness define, and one fixed-seed simulation.
The exact PASS line, 51 assertions, all 25 nonzero covers, finite receipt, and
inner/outer result seals are mandatory.

## Findings and boundary

The only P1 is publication atomicity: `RUN_COMPLETE.txt` precedes the final
seals, so a late sealing failure could leave contradictory markers. The
mandatory independent receipt hammer must reject that topology. The P2 is the
normal static residual: compilation and runtime closure remain unproven until
the one-shot actually passes.

Admitted now: exact r7 source identity, legal-fill isolation, campaign
preservation, and one-shot VCS authorization. Not admitted: SystemVerilog
compile, VCS behavior, V01--V20 runtime, numeric or production equivalence,
cycles, DC, Formality, STA, power, energy, speedup, PPA, system speedup, or a
headline claim.
