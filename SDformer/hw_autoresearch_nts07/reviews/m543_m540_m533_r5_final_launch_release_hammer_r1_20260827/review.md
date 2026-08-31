# M543 / M540 / M533 r5 final launch-release independent hammer

## Verdict

**PASS — 100/100, P0/P1/P2 = 0/0/0.**

The exact double-sealed final release is structurally and cryptographically valid: schema `m540_m533_m528_dead_write_only_1rw_vcs_launch_release_v1`, status `AUTHORIZED_ONE_M533_M528_DW1RW_R5_VCS_RUN`, and `launch_now=true`. It authorizes exactly one VCS attempt and no other tool, experiment, GPU, or remote execution.

This review was read-only with respect to all frozen inputs. It executed no runner, VCS, simv, HDL simulator, Synopsys EDA tool, CPU/GPU experiment, or network/remote job. It did not create or reserve the unique result path.

## Frozen release chain

The acyclic three-stage chain recomputes exactly:

1. candidate `70628023b391ee8c3ff3ee749405e9383b12c8dff0924ca96e2bfd788695349a`;
2. fresh independent candidate-hammer review `e05602c545c943c74b503588f8f2828024af90c5dcb338082c12645263193b70`;
3. final `launch_now=true` release `2528d19e9edf14b54f6b470ac978efa806a0680b82e68cd40034cdf04f36db55`.

The candidate hammer is double-sealed, PASS, 100/100, and P0/P1/P2 zero. The final release binds the exact candidate and candidate-hammer members. Neither upstream member contains the final-release member digest, so no circular hash dependency exists. Filesystem timestamps also preserve candidate -> hammer -> final authoring order.

## Closed authorization and resource policy

The final release, candidate, and r5 runner use the same closed ten-key authorization dictionary: `vcs_runs=1`; `iverilog`, `verilator`, `dc`, `formality`, `pt`, `ptpx`, `cpu`, `gpu`, and `network_or_remote_jobs` are all zero. Missing, extra, or changed keys fail closed.

The resource policy is also literal-identical across the final release, candidate, and runner: three samples, two seconds apart, MemAvailable >=128 GiB, SwapFree >=32 GiB, commit headroom >=32 GiB, cgroup v1, stable session/user `memory.failcnt`, zero session/user `under_oom` and `oom_kill`, and failure on missing or non-numeric counters. The runner performs initial and final same-UID collision scans and all release/resource checks before its sole atomic result-directory `mkdir`.

## Provenance and immutability

All exact upstream members and both levels of their seals pass: repair contract `968cdad...`, r5 runner-static review `fc231300...` (PASS 100/0/0/0), M536 failed launch review `f1b557...`, M537 failed r4 static review `bc6413...`, source-static review `0e0b389...` (PASS 100/0/0/0), and source contract `3e50884...`.

The runner (`24c833...`), TB, core RTL, SVA, macro adapter/binding plan, foundry macro artifacts, docs/524 baseline (`4f3ffe...`), and docs/359 (`dedde7...`) all retain their frozen SHA256 identities. Scoped `git diff --check` passes.

The sole result identity is `results/m533_m528_dead_write_only_1rw_vcs_r3_20260827`; it remains absent and unconsumed.

## Live read-only snapshot and release decision

At `2026-08-27T23:22:42+08:00`, a one-point read-only snapshot found no same-UID Synopsys/VCS/simv collision. MemAvailable was 412,474,500 KiB, SwapFree 57,218,812 KiB, commit headroom 73,833,564 KiB, and cgroup-v1 session/user failcnt, under_oom, and oom_kill were all zero.

This snapshot is informative only; it does **not** replace the runner's two collision scans and three prelaunch resource samples. Root may invoke only the immutable r5 runner after recomputing this review/final-release double seal and confirming the unique result path is still absent. Any live identity, collision, resource, or result-path failure forbids execution. A post-run independent receipt hammer remains mandatory.

No functional, speedup, PPA, energy, full-network, or paper-headline claim is admitted by this release review alone.
