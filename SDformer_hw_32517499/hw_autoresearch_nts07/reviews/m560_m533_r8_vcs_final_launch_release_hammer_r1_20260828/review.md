# M560 / M533 r8 final launch-release independent hammer

## Verdict

**PASS — 100/100, P0/P1/P2 = 0/0/0.**

The exact double-sealed release is structurally, semantically and cryptographically valid. It is the one-way successor of the M561 source-static PASS and M564 launch-candidate PASS, binds the immutable r8 runner and full source/tool/foundry/provenance identity, and carries exactly `launch_now=true`, `run_vcs=true`, `max_attempts=1`.

This hammer ran no runner, VCS, `simv`, HDL/EDA tool, CPU/GPU experiment or remote/network job. It did not create the new result or attempt marker. Release and review are not execution.

## Frozen one-way chain

The independently recomputed chain is:

1. M561 source-static review `8131cd0c...`, PASS 100 with zero findings;
2. `launch_now=false` candidate `eaacfd447...`;
3. M564 candidate hammer `e13f08988...`, PASS 100 with zero findings;
4. `launch_now=true` release `41cfcbc8...`;
5. M567 author handoff `830db09b...` and the sealed final-hammer request.

Every member manifest and outer seal passed. Strict JSON parsing rejected duplicate keys and non-finite/non-standard numbers. Upstream candidate/review members do not contain the final-release digest, so the chain has no cryptographic back-edge.

## Closed authorization and immutable identity

The release, candidate, request and runner literal agree on the exact 11-key budget: one VCS compile and one resulting `simv` run; Icarus, Verilator, DC, Formality, PT, PTPX, CPU, GPU and network/remote jobs are all zero. The resource policy also agrees exactly: 3 samples spaced 2 seconds, MemAvailable at least 128 GiB, SwapFree and commit headroom at least 32 GiB, cgroup-v1 failcnt stability, zero OOM state, missing-counter failure and zero same-UID Synopsys/VCS/`simv` collisions.

Live SHAs match the frozen r8 runner (`176c14d3...`), TB r4, core r2, SVA r2, parent-scratch macro adapter/binding plan, VCS binary and TSMC28 foundry manifest/slow Verilog/DB. The M544/M551/M558 failure provenance and all supplied package seals remain valid. `docs/359` remains `dedde7ce...`.

The consumed old partial still contains exactly eight plain regular files and all eight required SHA256 values, with no extra member. The new result path and its `.attempt` marker remain absent.

## Live-host block

At the final read-only observation, another user's `simv` was still running: UID 1909 (`fangyl`), PID 580855, started 2026-08-24 22:25:28 +08:00. This does **not** invalidate the static release or lower this hammer score, but it **does block actual launch now**. Root must not invoke the runner while that global collision remains.

After it disappears, root must freshly verify this review/release double seal, result/attempt absence and the complete live shared-host collision state, then allow the immutable runner to repeat all identity, provenance, old-partial, resource and collision gates. Only one exact r8 attempt may be considered; any live gate failure forbids execution. A post-run independent receipt hammer remains mandatory.

No functional, speedup, PPA, energy, system or paper-headline claim is admitted by this source-only final-release review.
