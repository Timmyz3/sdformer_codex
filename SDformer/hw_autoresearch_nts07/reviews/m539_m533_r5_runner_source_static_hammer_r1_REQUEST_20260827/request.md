# M539 / M533 r5 runner source-static hammer request

Perform one fresh independent read-only static hammer of the exact r5 runner and repair contract named in `request.json`. Do not execute the runner, create a release object or result path, run VCS/simv or any HDL/EDA tool, launch CPU/GPU experiments, or perform remote work.

First reproduce the sealed M537 r4 findings. Confirm every embedded Python invocation uses isolated mode, nonempty `PYTHONOPTIMIZE` is rejected before semantic validation, and no schema/status/`launch_now`/closed-authorization/SHA/score/P0/P1/P2 decision relies on Python `assert`. Confirm the runner hard-checks the exact frozen source-static review member SHA `0e0b38901c2c1f380e4500a4253b9d2174424d2e6881295b1f66a226bf1caf4c` as well as its double seal and PASS semantics.

Audit monitor fail-closure mechanically. A PASS requires an atomic periodic heartbeat; liveness and freshness checks before compile, after compile, and before finalization; a post-child final request; a synchronous final sample that validates both cgroup-v1 scopes' numeric failcnt/OOM/usage fields; an exact acknowledgment; monitor exit code zero; one final sample; and no discarded success-path wait status or violation marker. Check compile and simv failure paths remain consumed-attempt failures, while release/collision/prelaunch-resource failures precede result creation.

Also verify the preserved three-stage non-circular release chain, exact ten-key authorization, same-UID tool collision classification, cgroup-v1 prelaunch thresholds, current absence of all future M539/M540 release paths, absence of the r3 result identity, frozen functional hashes, and frozen `docs/359` hash.

A PASS must score exactly 100 with P0/P1/P2 all zero, bind the live runner and repair-contract SHA values, use schema `m539_m533_r5_runner_source_static_hammer_v1` and status `PASS_M539_M533_R5_RUNNER_SOURCE_STATIC_HAMMER`, remain source-only, and set `vcs_launch_authorized_now=false`. Double-seal the review. Static PASS alone does not authorize VCS.
