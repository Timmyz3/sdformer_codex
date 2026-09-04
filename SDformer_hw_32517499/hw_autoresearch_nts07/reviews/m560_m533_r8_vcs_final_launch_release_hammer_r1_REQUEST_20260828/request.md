# M560 / M533 r8 final launch-release hammer request

Perform a fresh independent source-only hammer of the exact double-sealed `launch_now=true` release in `request.json`. Run no runner, VCS, `simv`, DC, Formality, PT/PTPX, other HDL/EDA tool, CPU/GPU experiment, or remote/network job. Do not create or reserve the result identity or any attempt marker.

A PASS must strictly parse every JSON member, verify every member manifest and outer seal, and recompute the one-way chain: immutable r8 runner and source contract -> M561 source-static PASS100 -> `launch_now=false` candidate -> M564 candidate-hammer PASS100 -> final release SHA `41cfcbc8...`. Verify the release binds the live VCS/foundry artifacts, TB r4, core r2, SVA r2, macro adapter/binding plan, failure provenance, and all eight exact old-partial SHA256 values.

Compare authorization and resource policy as exact closed dictionaries. The release intent is exactly `launch_now=true`, `run_vcs=true`, `max_attempts=1`, with one VCS and one `simv` run and every other run counter zero. The result and attempt marker must remain absent. Release authoring and review must consume no attempt.

The shared host had a different-UID `simv` process (UID 1909, PID 580855) at authoring. This observation is not a frozen execution admission and must not cause the review to run or kill anything. A passing hammer still requires root to repeat a full live shared-host collision check and every immutable-runner collision/resource gate immediately before any invocation; release is not execution.

A PASS output must use schema `m560_m533_r8_vcs_final_launch_release_hammer_v1`, status `PASS_M560_M533_R8_VCS_FINAL_LAUNCH_RELEASE_HAMMER`, verdict `PASS`, score 100, and P0/P1/P2 = 0/0/0. Its `identity` must bind final release SHA `41cfcbc8...`, runner SHA `176c14d3...`, candidate SHA `eaacfd44...`, and candidate-hammer SHA `e13f0898...`. Its `decision` must contain `exactly_one_vcs_attempt_authorized_now=true` and `all_other_runs_authorized=false`, while also stating that a fresh root live preflight remains mandatory. Double seal the review package.
