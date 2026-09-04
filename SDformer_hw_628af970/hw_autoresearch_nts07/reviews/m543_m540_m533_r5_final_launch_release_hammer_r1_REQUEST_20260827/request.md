# M543 / M540 / M533 r5 final launch-release independent hammer request

Perform one fresh independent, read-only hammer of the exact double-sealed `launch_now=true` final release bound in `request.json`. This review must execute no runner, VCS, simv, HDL simulator, Synopsys EDA tool, CPU/GPU experiment, or remote/network job, and must not create or reserve the unique result directory.

Verify the final release and both sidecars strictly. A PASS requires exact schema `m540_m533_m528_dead_write_only_1rw_vcs_launch_release_v1`, exact status `AUTHORIZED_ONE_M533_M528_DW1RW_R5_VCS_RUN`, and `launch_now=true`. Reject duplicate JSON keys, non-standard numeric tokens, symlinks, a missing or invalid seal, or any identity drift.

Compare authorization as a closed dictionary, not a subset: exactly ten keys, `vcs_runs=1`, and all nine other counters zero. Compare the resource policy literally against both the release candidate and the r5 runner: 3 samples, 2-second spacing, 128 GiB `MemAvailable`, 32 GiB swap, 32 GiB commit headroom, cgroup v1, stable failcnt, zero `under_oom` and `oom_kill`, and fail-closed missing counters.

Recompute the three-stage chain exactly: candidate SHA `70628023...` -> independent candidate-hammer review SHA `e05602c5...` -> final release SHA `2528d19e...`. Confirm the candidate hammer is a double-sealed 100/100 PASS with P0/P1/P2 all zero. Confirm the final release binds the live r5 runner, repair contract, r5 runner-static review, M536 failed launch review, M537 failed r4 static review, frozen source-static review, and frozen source contract exactly.

Confirm the sole result identity is `results/m533_m528_dead_write_only_1rw_vcs_r3_20260827`, that it remains absent, and that authoring/reviewing the release has not consumed the attempt. Confirm the runner, TB, core RTL, SVA, macro assets, docs/524 baseline, and docs/359 remain unchanged.

A PASS output must use schema `m543_m540_m533_r5_final_launch_release_hammer_v1`, status `PASS_M543_M540_M533_R5_FINAL_LAUNCH_RELEASE_HAMMER`, verdict `PASS`, score exactly 100, and P0/P1/P2 all zero. It must bind the exact final-release, candidate, candidate-hammer, runner, repair-contract, and runner-static-review member SHA256 values listed in `request.json`, and it must be double-sealed.

Even a passing final-release hammer does not itself run VCS. Root must independently confirm the live same-UID collision and resource gates immediately before invoking the immutable runner. Any audit, identity, collision, resource, or result-path failure forbids execution and leaves the unique attempt unconsumed.
