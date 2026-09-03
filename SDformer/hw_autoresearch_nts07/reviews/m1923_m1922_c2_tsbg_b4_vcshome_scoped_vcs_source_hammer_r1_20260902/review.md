# M1923 — M1922 TSBG scoped-VCS_HOME runner source hammer

## Verdict

**PASS, 99/100; P0/P1/P2 = 0/0/0.** Exactly one attempt of the exact M1922 runner is authorized. This review performed no license query, attempt creation, VCS/simv invocation, or EDA run.

## Identity and the two M1915 regressions

- Runner: `run_m1922_m1914_c2_tsbg_b4_vcshome_scoped_directed_vcs_one_shot.sh`
- SHA256: `7e391c7ab50b56b9d9ea383686c40e44414f4baf0ad3e5b3025210880d66a667`
- The runner's `RUNNER` constant resolves to that exact existing executable. This closes M1915-P1-1.
- Frozen M1907 review SHA and double seal verify, and the runner matches its exact status including `__DO_NOT_AUTHORIZE_ATTEMPT`. This closes M1915-P1-2.
- Frozen M1915 FAIL review SHA, exact status, and double seal also verify.

## Static closure

- `bash -n` passes; the runner is executable and starts through a clean fixed shebang.
- The filelist and its adapter, RTL, SVA, and testbench are separately exact-SHA pinned and match the M1881 source-reviewed identities.
- `VCS_HOME` is supplied only to the unique VCS compile and unique `simv` run; it is absent from the sole `lmutil` license preflight.
- Fresh M1922 attempt/result/failure/work/lock namespaces, same-UID EDA exclusion, memory/commit gates, durable attempt-before-license ordering, explicit signal exit codes, no-replace publication, double seals, and no retry are present.
- Success requires one exact directed PASS token and no assertion/error/fatal token. A raw PASS remains `paper_admitted=false`, `same_area=false`, and `system_speedup=false`, and explicitly requires a different-author result hammer.
- `docs/359` remains SHA256 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

## Authorization boundary

Invoke the exact M1922 executable directly with the exact runner SHA above and the final SHA256 of this double-sealed `review.json`. The budget is one license query, one VCS compile, one simv run, no automatic retry. This source review does not itself establish RTL execution, timing, area, energy, speedup, or paper admission.
