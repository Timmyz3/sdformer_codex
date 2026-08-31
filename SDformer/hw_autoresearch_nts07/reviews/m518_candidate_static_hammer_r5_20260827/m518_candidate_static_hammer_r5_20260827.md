# M518 matched Fixed T10 ATLIF independent static hammer r5

Date: 2026-08-27  
Verdict: `STATIC_GO__EXACT_SHA_ONE_SHOT_R5_VCS_AUTHORIZED`  
Score: **97/100**  
Findings: **P0=0, P1=5**

This was a receipt-blind, source-only review. The reviewer did not execute the
candidate runner, VCS, DC, Formality, PT/PTPX, or an open-source RTL tool. No
production file, old failed result, or `docs/359` was modified.

## Decision

Exactly one invocation of this runner is authorized:

```text
dc_handoff/scripts/run_vcs_m518_matched_fixed_t10_atlif_r5_exact.sh
SHA256 854f152ad23bcc3e353953dee93d0b88f24eab2b4f34261bd88c3c3560a7312a
```

The operator must set `M518_EXPECTED_RUNNER_SHA256` to that exact digest, leave
`M518_RUN_DIR` unset, and use the currently absent default result path
`results/m518_matched_fixed_t10_atlif_vcs_r5_exact_20260827`. This authorizes
only the runner's wrong-RTL negative control, exact preflight, one Full64 VCS
identity query, one Full64 VCS compile, and one fixed-seed simulation. It does
not authorize DC, Formality, PT/PTPX, or any performance, energy, PPA, system,
or headline claim. A separate independent post-run receipt hammer is required.

## Exact six-token repair

The r5 RTL has zero `within` word tokens and exactly six `tap_within` tokens:
one local declaration, one initialization, two assignments, and two
time-index uses. Replacing those six word tokens with `within` in memory gives
SHA256 `09b1d976...a93412a`, the exact frozen r4 RTL. This proves there is no
other RTL byte change. SVA `977f9565...`, TB `e7973a91...`, and filelist
`09e43560...` are unchanged; the filelist still contains only RTL, SVA, and TB
in that order. No public port, arithmetic, schedule, or verification campaign
change was introduced.

## Evidence and fail-closed checks

- All 24 runner SHA-map inputs exist and match. They bind the r5 sources,
  M273 reference, r4 contract and failed-result evidence, the double-sealed r4
  failure audit, the double-sealed baseline specification, and `docs/359`.
- Both evidence manifests and both outer seals verify. The r4 audit remains
  diagnostic only and authorizes neither r5 execution nor DC.
- The external runner-SHA gate precedes result creation, the automatic negative
  campaign, and any tool side effect. The wrong-RTL control must return 10,
  cannot create compile/simulation/positive artifacts, uses a disjoint output
  directory, and creates a member manifest plus outer seal.
- Positive preflight mechanically reconstructs the r4 RTL SHA, checks the
  exact 50-port M273-compatible interface, enumerates the complete 1,600-product
  row/lane/time bijection, and preserves ordered V01--V20.
- The only tool-launch path contains one `vcs -full64 -ID`, one
  `vcs -full64 -sverilog -assert svaext` compile with runtime SVA enabled, and
  one simulation with seed 51820260827. An exact PASS line and all 25 required
  nonzero covers are mandatory.
- Receipt JSON serialization forbids non-finite values and is strictly reparsed
  before publication. The failure trap remains active through member-manifest
  and outer-seal creation.

The unchanged campaign includes independent signed-INT8/Q24 numerical checks,
four randomized contexts, six arithmetic rail points, frame/padding/protocol
attacks, five release-state attacks, nine reset attacks, FIFO-full same-edge
pop/push, phase-12/16 stalls, all five output beats, and a complete 96/96/.../64
slot ledger. None of those results is admitted until VCS actually completes
and an independent reviewer validates the receipt.

## Retained limitations

1. `M518_RUN_DIR` must remain unset; the override can redirect output.
2. The publication set is sealed but not checked against an exact hard-coded
   whitelist; the post-run reviewer must reject any missing or extra member.
3. Static authorization is out of band rather than a runner SHA-map member.
4. Publication is not an atomic directory rename. A late failure can leave a
   positive member plus `RUN_FAILED_OR_INCOMPLETE`; that contradictory topology
   must be rejected by the mandatory post-run review.
5. r4 stopped at its first parser error. The rename closes that exact defect but
   cannot guarantee that VCS will not expose a later compile or runtime issue.

The first four are provenance/operational hardening items; the fifth is normal
residual execution risk. None creates a path for an unreviewed identity or
allows static evidence to be cited as a VCS result.
