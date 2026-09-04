# M537 / M533-r4 fail-closed runner source-static hammer

Verdict: **FAIL, 86/100, P0/P1/P2 = 0/2/1.** This was a fresh independent read-only static hammer. It executed neither the candidate runner nor VCS, simv, Icarus, Verilator, DC, Formality, PT, PTPX, CPU/GPU experiments, or remote work. It created no result/attempt directory. Static FAIL authorizes no launch.

## What r4 closes mechanically

1. The fixed future sequence is structurally non-circular: a `launch_now=false` candidate is hashed, an independent 100/100 hammer binds that exact candidate member, and a final `launch_now=true` release binds both the candidate and hammer members. Candidate and final authorization objects are compared against an exact ten-key dictionary with `vcs_runs=1` and every other class zero.
2. At the review snapshot, the runner-static review, release candidate, candidate hammer, final release, and new result path were all absent. Missing any future artifact fails before the temporary preflight directory and before the sole result-directory `mkdir`. After this review is written, the other three release artifacts remain absent and the runner remains unusable.
3. The same-UID `/proc` classifier examines executable basenames plus the first wrapper tokens and the complete `common_shell_exec -shell` argument string. It covers direct DC, Formality, PrimeTime/PTPX, VCS front ends, and simv. Only the scanner PID and its direct parent bash runner are excluded; the r4 filename's `vcs` substring does not self-match.
4. The cgroup-v1 resolver selects the memory-controller session path from `/proc/self/cgroup` and also requires `/sys/fs/cgroup/memory/user.slice`. Both paths exist on this host with readable `memory.failcnt`, `memory.oom_control`, and `memory.usage_in_bytes`. Three prelaunch samples, two seconds apart, enforce exactly 128 GiB MemAvailable, 32 GiB SwapFree, 32 GiB commit headroom, stable failcnt, and zero `under_oom`/`oom_kill`. A second collision scan precedes the atomic result creation.
5. Bash syntax, strict JSON parsing, request/contract/handoff/prerequisite member and outer seals, current frozen functional hashes, and `docs/359` all pass. The new result path is absent.

## P1 blockers

### M537-R4-P1-01 — Python optimization can erase every semantic release interlock

The runner invokes ordinary `python3` and expresses the existing source-review checks, r4 static-review checks, candidate/hammer/final status checks, exact authorization comparison, `launch_now`, SHA bindings, scores, and P0/P1/P2 requirements exclusively as Python `assert` statements (runner lines 165–323). The runner neither rejects `PYTHONOPTIMIZE` nor uses isolated mode. Under Python optimization, those assertions are removed. Once the four future paths exist and self-seals verify, a nonconforming candidate, hammer, or final release can therefore pass the semantic stage and reach collision/resource gating and attempt creation.

Minimum repair: replace security-relevant assertions with explicit conditional failures, and invoke the validator in an environment-isolated mode that cannot be changed by `PYTHONOPTIMIZE`. The repaired runner must be a new exact identity and receive another fresh static hammer.

### M537-R4-P1-02 — runtime monitor death and the final sampling window fail open

The background monitor writes a violation marker for values it successfully reads, but the parent kills it after simv, discards the wait status with `wait ... || true`, and checks only whether the marker file exists (runner lines 487–542). An unexpected monitor exit before writing the marker, a read/awk failure after the readability precheck, or a failcnt/OOM transition after the last one-second sample but before the kill can leave no marker and be admitted. This does not satisfy the frozen requirement that missing runtime counters fail closed and failcnt/OOM invariants hold through the whole attempt.

Minimum repair: make monitor liveness and exit status mandatory, share one explicit counter-validation routine, take a synchronous final sample after simv and before stopping the monitor, and fail if the monitor died unexpectedly or any final field is missing/non-numeric/drifted.

## P2 finding

### M537-R4-P2-01 — the frozen source-static review member SHA is not consumed

The repair contract freezes the source-static review member as `0e0b38901c2c1f380e4500a4253b9d2174424d2e6881295b1f66a226bf1caf4c`, and that hash currently recomputes. The runner verifies the review's self-seal and selected semantics, but never compares its member SHA to the frozen value. This leaves the explicit `frozen ... source-static SHA` machine requirement incomplete even though the currently inspected prerequisite is valid.

Minimum repair: hard-check the exact source-static review member SHA before semantic parsing, as already done for the failed M536 member.

## Decision

The r4 design is materially closer to fail-closed than r3: release ordering, collision scans, cgroup-v1 policy, and result-directory ordering are present and inspectable. It is not launch-admissible because the release semantics have an environment-controlled bypass and runtime resource monitoring can fail open. Do not author a release candidate and do not execute r4. Author a new runner identity containing only the three bounded repairs above, then repeat this static hammer.

This review establishes no RTL functional correctness, trace recurrence, speedup, PPA, energy, full-network result, or paper headline.
