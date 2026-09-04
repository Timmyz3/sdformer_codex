# M540 / M533 r5 launch-release-candidate independent hammer request

Perform one fresh independent, read-only hammer of the exact `launch_now=false` release candidate bound in `request.json`. This review must execute no runner, VCS, simv, HDL simulator, Synopsys EDA tool, CPU/GPU experiment, or remote/network job, and must not create the unique result directory.

Verify the candidate and both sidecars strictly. A PASS requires the exact schema `m540_m533_m528_dead_write_only_1rw_vcs_launch_release_candidate_v1`, exact status `READY_FOR_INDEPENDENT_RELEASE_CANDIDATE_HAMMER`, and `launch_now=false`. Reject duplicate JSON keys, non-standard numeric tokens, symlinks, any missing seal, or any candidate drift.

Compare the authorization object as a closed dictionary, not a subset: exactly ten keys, `vcs_runs=1`, and all nine other counters zero. Independently compare the resource policy against the r5 runner's literal `expected_policy`: 3 samples, 2 seconds, 128 GiB `MemAvailable`, 32 GiB swap, 32 GiB commit headroom, cgroup v1, stable failcnt, zero `under_oom` and `oom_kill`, and fail-closed missing counters.

Recompute and check the candidate's exact bindings to the live r5 runner, r5 repair contract, 100/100 r5 runner-static review, M536 failed launch review, M537 failed r4 static review, frozen M533 source-static review, and frozen M533 source contract. Confirm the exact unique result path `results/m533_m528_dead_write_only_1rw_vcs_r3_20260827` is absent and the final `launch_now=true` release path is absent. The candidate must not itself authorize a launch or consume the attempt.

A PASS output must use schema `m540_m533_r5_functional_vcs_launch_release_candidate_hammer_v1`, status `PASS_M540_M533_R5_FUNCTIONAL_VCS_LAUNCH_RELEASE_CANDIDATE_HAMMER`, verdict `PASS`, score exactly 100, and P0/P1/P2 all zero. Its `identity` object must bind the exact candidate, runner, repair-contract, and runner-static-review member SHA256 values. Its `decision` object must set `release_candidate_pass=true`, `closed_authorization_pass=true`, `source_static_100_p0_p1_p2_zero=true`, `final_launch_release_required=true`, and `vcs_launch_authorized_now=false`.

Double-seal the review directory. Do not create the final launch release. Even a passing candidate hammer does not authorize VCS; a separately authored, double-sealed `launch_now=true` final release and fresh collision/resource preflight remain mandatory.
