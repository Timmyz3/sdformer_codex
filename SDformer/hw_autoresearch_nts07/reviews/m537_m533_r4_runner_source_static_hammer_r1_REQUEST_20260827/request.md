# M537 M533 r4 fail-closed runner source-static hammer request

Perform one fresh, independent, read-only static hammer of the exact runner and repair contract in `request.json`. Do not execute the candidate runner. Do not run VCS, simv, Icarus, Verilator, DC, Formality, PT, PTPX, CPU/GPU experiments, or remote work.

The review must reproduce all M536 blockers and establish that r4 closes them mechanically before the unique result path is created. In particular, verify that the runner cannot pass today because its fixed future runner-static review, release candidate, candidate hammer, and final launch release paths are absent. Audit the non-circular digest chain: the candidate freezes the closed ten-key authorization with `launch_now=false`; an independent 100/100 hammer binds its exact SHA; only then may a final `launch_now=true` release bind both the candidate and hammer SHA values plus the exact r4 runner.

Inspect the `/proc` same-UID collision classifier. It must reject direct and `common_shell_exec` forms of DC, Formality, PrimeTime/PTPX, VCS, and simv while excluding only the scanner process and its direct parent runner. Inspect cgroup-v1 resolution from `/proc/self/cgroup`, the mandatory session-scope and `user.slice` counters, three prelaunch samples, the exact 128/32/32-GiB thresholds, stable `memory.failcnt`, zero OOM state, and the runtime monitor. Missing fields must fail closed.

A PASS must be exactly score 100 with P0/P1/P2 all zero, use schema `m537_m533_r4_runner_source_static_hammer_v1`, status `PASS_M537_M533_R4_RUNNER_SOURCE_STATIC_HAMMER`, bind the runner and repair-contract SHA256 values, and remain source-only with `vcs_launch_authorized_now=false`. Double-seal the review. Static PASS is not a launch release.
