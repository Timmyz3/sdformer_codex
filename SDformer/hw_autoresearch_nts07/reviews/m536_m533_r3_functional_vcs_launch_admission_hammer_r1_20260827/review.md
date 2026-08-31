# M536 / M533-r3 functional-VCS launch-admission hammer

Verdict: **FAIL, 82/100, P0/P1/P2 = 0/3/2.** This was a fresh read-only hammer. It ran no VCS, Icarus, Verilator, DC, Formality, PT, PTPX, CPU/GPU experiment, or remote job, and it did not create the future result directory.

## Two-part launch decision

- **A — current launch: FORBIDDEN.** At the fresh sample `2026-08-27T22:11:26+08:00`, same-UID M519 PIDs `4165439`, `4165666`, and `4165667` were alive; PID `4165666` was `/opt/synopsys/.../common_shell_exec -shell dc_shell`. The candidate also freezes `launch_now=false`. The unique result path remains absent.
- **B — after collision clear: runner/wrapper r4 is still required.** A root/operator-only recheck is not fail-closed because the exact r3 runner neither reads `launch_now`, nor consumes this independent hammer verdict, nor scans same-UID tool processes, nor evaluates a frozen resource policy. Do not release r3 after merely observing M519 exit.

## What passed

1. The request, candidate admission, source contract, source-static review, and their member/outer seals all verify. Strict JSON parsing rejects duplicate keys and non-standard constants.
2. Every frozen source/model identity in the candidate currently recomputes exactly, including top r2, macro adapter and binding plan, SVA r2, TB r3, foundry manifest/views, and frozen `docs/359`.
3. The sealed source-static review is exactly PASS 100/100 with P0/P1/P2 `0/0/0`.
4. The authorization object has exactly ten keys: `vcs_runs=1`; Icarus, Verilator, DC, Formality, PT, PTPX, CPU, GPU, and network/remote are all explicit zero. The runner's set equality rejects unknown and missing keys.
5. The exact future path `results/m533_m528_dead_write_only_1rw_vcs_r1_20260827` is absent.

## P1 blockers

### M536-LAUNCH-P1-01 — the frozen runner ignores all release interlocks

The candidate says `launch_now=false` and requires a passing fresh M536 hammer. The runner validates only schema, status, authorization, and a subset of identity fields; it never parses `launch_now`, never binds a M536 review path/hash/verdict, and proceeds from admission checks directly to `mkdir -p`, `vcs`, and `./simv`. Invoking r3 now would therefore consume the attempt despite both explicit release blocks.

Minimum repair: create a new exact-SHA runner/wrapper identity that requires `launch_now=true` in a newly sealed admission, binds a sealed M536-successor review with P0/P1 zero, and refuses to create the result directory before both pass. The current sealed candidate must remain immutable and false.

### M536-LAUNCH-P1-02 — same-UID collision exclusion is operator-only

The runner contains no same-UID scan for `dc_shell`, `dc_shell-t`, Synopsys `common_shell_exec`, `vcs`, or `simv`. The fresh scan proves this is not hypothetical: M519 `common_shell_exec -shell dc_shell` remains alive. A manual promise not to invoke the script cannot prevent an accidental invocation and is not fail-closed.

Minimum repair: before result-directory creation, the new wrapper must scan the launching UID, exclude only its own known process tree, classify full command lines, and hard-fail on every frozen forbidden class. The scan and decision must be written into a prelaunch receipt that is sealed after the attempted launch outcome, not supplied as an unbound operator note.

### M536-LAUNCH-P1-03 — resource gate has neither an executable policy nor host-compatible counters

The runner reads none of `MemAvailable`, `CommitLimit`, `Committed_AS`, `SwapFree`, or cgroup memory state and binds no reviewed thresholds. The candidate requests cgroup-v2 `memory.current` and `memory.events`, but this host is cgroup v1; the applicable session exposes `memory.usage_in_bytes`, `memory.max_usage_in_bytes`, `memory.failcnt`, and `memory.oom_control` instead. At the sample, `MemAvailable=408465444 kB`, `CommitLimit=541608164 kB`, `Committed_AS=487358524 kB`, and `SwapFree=57219580 kB`, but no sealed rule maps these values to PASS/FAIL.

Minimum repair: freeze and independently review a deterministic policy that supports the detected cgroup version, validates all required files, computes the stated headroom/emergency conditions, and hard-fails before result creation on missing counters or policy failure.

## P2 findings

1. The admission has 28 identity keys, but the embedded semantic check compares only five (`runner`, source contract, static review, foundry manifest, and slow Verilog). The source/model files are separately hard-hashed, so this is not a present functional-input escape; r4 should nevertheless bind the exact admission member SHA and validate every security-relevant identity field instead of accepting a merely self-resealed JSON.
2. The runner verifies static-review schema/status/P0/P1/source-contract binding, but not `score_100=100`, `p2_count=0`, or verdict. The current review does satisfy all three; r4 should enforce the full frozen prerequisite to prevent semantic drift after resealing.

## Required next gate

Author a separately admitted r4 wrapper/runner with machine-enforced release, collision, and resource checks. Subject it to a fresh independent read-only hammer. Only after that PASS and a fresh clear prelaunch sample may the sole VCS attempt run. A future VCS result still requires an independent post-run receipt hammer and establishes only functional RTL/model behavior—not trace recurrence, speedup, PPA, energy, full-network performance, or a paper headline.

