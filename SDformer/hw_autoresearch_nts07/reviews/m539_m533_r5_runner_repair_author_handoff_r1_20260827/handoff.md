# M539 / M533 r5 bounded runner-repair handoff

This source-only package repairs exactly the two P1 and one P2 findings in the sealed M537 r4 runner hammer. It does not execute the runner or any HDL/EDA tool, create a result directory, modify functional sources, or authorize a VCS launch.

The r5 runner invokes every embedded validator as `python3 -I`, rejects nonempty `PYTHONOPTIMIZE` before validation, and uses explicit conditional exceptions for all schema, status, `launch_now`, exact closed ten-key authorization, SHA, score, and P0/P1/P2 checks. No security predicate depends on Python `assert`. It also hard-requires the frozen source-static review member SHA `0e0b38901c2c1f380e4500a4253b9d2174424d2e6881295b1f66a226bf1caf4c` after verifying that review's double seal.

The runtime resource monitor now has a periodic atomic heartbeat. The parent requires a live and fresh heartbeat before compile, after compile, and before finalization. Once the compile or simv child has ended, the parent creates a final-request marker; the monitor performs one additional synchronous cgroup-v1 sample, checks both session and `user.slice` failcnt/OOM/usage fields, writes an exact final acknowledgment, and exits. The parent admits only monitor exit code zero, one exact final ack, one final-sample record, and no violation marker. It never discards the success-path `wait` status.

The r4 three-stage non-circular release chain, same-UID collision scans, cgroup/prelaunch thresholds, resource-before-result order, and consumed-attempt failure semantics remain. Four future M539/M540 paths are absent, so r5 currently fails before preflight/result creation. The only next action is the sealed fresh independent read-only source-static hammer request.

Frozen core r2, SVA r2, TB r3, macro/binding plan, r3/r4 runners, `docs/524`, and `docs/359` were not modified. This package establishes no functional correctness, recurrence, speedup, PPA, energy, full-network result, or paper headline.
