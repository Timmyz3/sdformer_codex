# M540 / M533 r5 launch-release-candidate independent hammer

## Verdict

**PASS — 100/100, P0/P1/P2 = 0/0/0.**

This was a fresh, independent, read-only audit of the exact `launch_now=false` M540 release candidate. No runner, VCS, `simv`, other HDL simulator, Synopsys EDA tool, CPU/GPU experiment, or remote/network job was executed. The unique result directory and final `launch_now=true` release remain absent.

## Candidate and authorization

- Candidate SHA256: `70628023b391ee8c3ff3ee749405e9383b12c8dff0924ca96e2bfd788695349a`.
- The member sidecar and its outer seal verify exactly. Candidate JSON is strict: no duplicate keys, non-standard numeric tokens, symlinks, missing members, or seal drift were found.
- Schema and status are exact; `launch_now` is the Boolean `false`.
- `authorization` is a closed ten-key dictionary: `vcs_runs=1`; all nine other run counters are zero. It equals the r5 runner's literal `expected_auth` dictionary, not merely a subset.
- The candidate cannot launch the runner, create the result, or consume the unique attempt. The separately authored final release remains mandatory.

## Identity and upstream chain

The candidate binds the live r5 runner (`24c833dc...`), repair contract (`968cdad1...`), 100/100 r5 runner-static review (`fc231300...`), M536 failed launch review (`f1b55799...`), M537 failed r4 static review (`bc6413ba...`), frozen M533 source-static review (`0e0b3890...`), and source contract (`3e50884b...`) exactly. All applicable member manifests and outer seals pass.

The r5 runner-static review is exactly `PASS_M539_M533_R5_RUNNER_SOURCE_STATIC_HAMMER`, with score 100 and P0/P1/P2 = 0/0/0. It binds this live runner and repair contract and explicitly does not authorize VCS. The earlier M536 and M537 failures remain immutable predecessor evidence rather than being overwritten.

## Resource policy

The candidate resource policy exactly equals the r5 runner's literal `expected_policy`: three samples, two-second interval, 128 GiB `MemAvailable`, 32 GiB swap, 32 GiB commit headroom, cgroup v1, stable session and user fail counts, zero `under_oom` and `oom_kill`, and fail-closed treatment of missing counters.

## Attempt state and immutability

- `results/m533_m528_dead_write_only_1rw_vcs_r3_20260827`: absent.
- `contracts/m540_m533_m528_dead_write_only_1rw_vcs_launch_release_r1_20260827.json`: absent.
- Frozen runner, TB, core RTL, SVA, macro adapter/binding plan, docs/524, and docs/359 were not changed by this review. docs/359 remains `dedde7ce...`.
- This review makes no functional, recurrence, speedup, PPA, energy, system, or paper-headline claim.

## Decision

The release candidate passes and the closed authorization passes. Root may now author a separate, double-sealed `launch_now=true` final release that binds the exact candidate and this review member SHA256. Even then, the runner must perform a fresh collision/resource preflight before consuming the sole VCS attempt. This review itself authorizes no run.
