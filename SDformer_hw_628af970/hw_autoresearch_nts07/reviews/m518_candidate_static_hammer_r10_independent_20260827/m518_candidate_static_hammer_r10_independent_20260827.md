# M518 r10 independent source-only static hammer

Date: 2026-08-27  
Verdict: `STATIC_GO__ROOT_MAY_CREATE_DOUBLE_SEALED_EXACT_SHA_ONE_SHOT_R10_VCS_ADMISSION`  
Score: **99/100**; P0/P1/P2 = **0/0/2**

## Decision

The r10 runner-only schema repair is statically closed. Root may now create the
separate, double-sealed r10 launch admission for exactly one invocation of
`dc_handoff/scripts/run_vcs_m518_matched_fixed_t10_atlif_r10_exact.sh` at
SHA256
`ca1873f864891b6da02260b10feedbdf7d18b9e1ecca52dbcc77e0bed8c91ef6`.
The admission must use JSON integer `1` for `authorized_invocations`, authorize
VCS only, and bind the canonical r10 result path. This review is not itself the
launch admission and executed no runner, negative control, or EDA.

## Mechanical findings

1. The r10 author request member manifest and outer seal both validate. The r9
   pre-tool failure review, r8 release-failure review, r9 static review, and r9
   launch admission also pass their member and outer seals.
2. The sealed r9 pre-tool diagnosis points exactly to
   `decision.r9_vcs_authorized_by_this_review`. The sealed r8 `decision` is a
   JSON object, that exact key is present, and its value has JSON boolean type
   and is strictly `false`. Missing, `null`, numeric zero, string `"false"`,
   and `true` do not satisfy the r10 source check.
3. The sealed r9 pre-tool review's exact
   `decision.r10_vcs_authorized_by_this_review` key is likewise present and
   strictly boolean `false`. Neither historical diagnostic object can act as
   r10 launch authority.
4. All 31 paths in the runner SHA map are unique and independently hash with
   zero mismatch. RTL, SVA, TB, and filelist remain byte exact to r9. The
   preserved source has 51 assertion labels, 25 cover labels, the exact
   V01--V20 PASS signature once, and unchanged 29/80-cycle and numeric oracle
   identities.
5. Runner SHA, admission SHA, both admission seals, and admission semantics are
   checked before canonical `mkdir`; `M518_RUN_DIR` is forbidden. The isolated
   wrong-TB control must exit 10 before VCS and is forbidden from producing a
   compile log, `simv`, positive receipt, or `RUN_COMPLETE.txt`.
6. The positive semantic source check ends before `PREFLIGHT_COMPLETE.txt`,
   which precedes the first VCS identity command. At completion,
   `RUN_ARTIFACTS_COMPLETE.txt` is included in the member manifest; the run
   manifest, run outer seal, copied manifest, and copied outer seal are all
   positively checked before `RUN_COMPLETE.txt` is written. This closes the
   r9 publication-order P1 in the requested topology.

## Residual P2 findings

- **Static/runtime boundary:** source review cannot prove VCS compilation,
  V01--V20 execution, assertion/cover closure, numeric equivalence, or cycles.
  A different reviewer must hammer the completed one-shot receipt.
- **Admission count type hardening:** Python treats boolean `true` as equal to
  integer `1`. The reviewer-controlled admission must therefore encode an
  actual JSON integer `1`; the exact-SHA and double-seal gate makes this a P2,
  not a launch blocker. A later runner can additionally reject `bool`.

## Admission root is allowed to create

```text
path=contracts/m518_matched_fixed_t10_atlif_vcs_launch_admission_r10_20260827.json
authorized_runner_sha256=ca1873f864891b6da02260b10feedbdf7d18b9e1ecca52dbcc77e0bed8c91ef6
authorized_invocations=1 (JSON integer, not boolean)
vcs_authorized=true
dc_authorized=false
required_result_path=results/m518_matched_fixed_t10_atlif_vcs_r10_exact_20260827
```

After root double-seals that admission, the invocation must set its exact SHA
in `M518_EXPECTED_STATIC_ADMISSION_SHA256`, set the runner SHA above in
`M518_EXPECTED_RUNNER_SHA256`, and leave `M518_RUN_DIR` unset.

DC, Formality, PT/PTPX, power, energy, speedup, PPA, system, and headline
claims remain unauthorized.
