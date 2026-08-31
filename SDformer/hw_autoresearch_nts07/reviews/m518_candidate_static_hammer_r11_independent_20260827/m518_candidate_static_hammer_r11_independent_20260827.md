# M518 r11 independent source-only static hammer

Date: 2026-08-27  
Verdict: `STATIC_GO__ROOT_MAY_CREATE_DOUBLE_SEALED_FIXED_WRAPPER_ONE_SHOT_R11_VCS_ADMISSION`  
Score: **99/100**; P0/P1/P2 = **0/0/1**

## Decision

The r11 fixed-wrapper launch-plumbing repair is statically closed. Root may
create the separate, double-sealed r11 launch admission for the exact runner
SHA256
`4e50a78cae0a4a05cad50865468e8321897d7ce74d851212551d5ccfa4d660a8`
and exact wrapper SHA256
`798f433ff0ee790058b86b781e01de9fd021c0947cdf49c8bfcc0e95480c3650`.
The admission must authorize exactly JSON integer `1` (not Boolean `true`),
VCS only, the fixed wrapper path, and the canonical r11 result path. The only
permitted operator entry point after that admission exists is:

```text
dc_handoff/scripts/launch_vcs_m518_matched_fixed_t10_atlif_r11.sh
```

This review neither creates that admission nor authorizes manual SHA
assignments. It executed no wrapper, runner, negative control, or EDA.

## Mechanical findings

1. The r11 author request member manifest and outer seal validate. The sealed
   r10 failure review also validates both seals and says exactly: r10 is
   consumed, r10 retry is false, r11 authoring is true, and that historical
   review does not authorize r11 VCS.
2. RTL, SVA, TB, and filelist are byte exact to r10 at the frozen SHAs. The
   preserved source contains 51 assertion labels, 25 cover labels, and exactly
   one unchanged V01--V20 PASS signature with the 29/80 clean-cycle and
   17-issue-cycle identities. This is source preservation, not runtime proof.
3. All 42 unique runner input paths independently hash with zero mismatch.
   Contract, result identity, runner, and wrapper also match the request SHAs.
   The canonical r11 result and all three r11 admission files were absent at
   review time. `docs/359` remains
   `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
4. Before result creation the runner separately checks presence, length,
   lowercase-hex-64 syntax, live runner/wrapper identities, admission SHA,
   both admission seals, and admission semantics. Its pre-`mkdir` diagnostics
   expose only constant variable name, presence, length, and regex pass; no
   supplied SHA value is printed.
5. Both runner and wrapper require Python
   `type(authorized_invocations) is int` and value exactly `1`; therefore JSON
   Boolean `true` is mechanically rejected. The wrapper validates both
   admission seals before computing live admission, runner, and self SHAs,
   then binds those identities plus VCS-only policy, canonical result path,
   and required wrapper path.
6. A present `M518_RUN_DIR` is rejected. Caller-provided SHA overrides are
   cleared; the wrapper exports the exact three SHA environment names from
   live values and removes `M518_RUN_DIR` from the runner environment. No
   operator SHA re-entry is required or allowed.
7. The isolated wrong-TB control still precedes any VCS command. The semantic
   source check and `PREFLIGHT_COMPLETE.txt` precede the first VCS identity
   command. `RUN_COMPLETE.txt` remains after the run-manifest member/outer
   checks and copied `SHA256SUMS` member/outer checks.

## Residual P2

- **Static/runtime boundary:** this review cannot prove VCS compilation,
  V01--V20 execution, assertion/cover closure, numeric equivalence, or cycles.
  A different reviewer must hammer the completed one-shot receipt before any
  behavior claim is admitted.

## Admission root is allowed to create

```text
path=contracts/m518_matched_fixed_t10_atlif_vcs_launch_admission_r11_20260827.json
authorized_runner_sha256=4e50a78cae0a4a05cad50865468e8321897d7ce74d851212551d5ccfa4d660a8
authorized_launch_wrapper_sha256=798f433ff0ee790058b86b781e01de9fd021c0947cdf49c8bfcc0e95480c3650
authorized_invocations=1 (JSON integer, not Boolean)
vcs_authorized=true
dc_authorized=false
required_launch_wrapper_path=dc_handoff/scripts/launch_vcs_m518_matched_fixed_t10_atlif_r11.sh
required_result_path=results/m518_matched_fixed_t10_atlif_vcs_r11_exact_20260827
```

DC, Formality, PT/PTPX, power, energy, speedup, PPA, system, and headline
claims remain unauthorized.
