# M560 / M533 r8 VCS launch-admission candidate hammer

Verdict: **PASS, 100/100, P0/P1/P2 = 0/0/0.** This was a fresh independent read-only candidate review. It did not invoke the runner, VCS, `simv`, any HDL/EDA tool, any CPU/GPU experiment, or any remote job. It did not create the r8 result identity, a `launch_now=true` release, or a final-release hammer, and it did not modify or seal the consumed r3 partial.

## Candidate identity and binding

- The candidate SHA is `eaacfd447b257316854eacbf0bef2e4800f46c32609bdd46bca48848ccb9c15b`; its inner manifest and outer seal both verify.
- Runner r8, source contract r6, the fresh 100/100 source-static review, its member manifest and outer seal, and the atomic author handoff all match their declared live SHA256 identities.
- TB r4, core r2, SVA r2, macro adapter, and macro binding plan are byte frozen. The VCS executable, foundry manifest, Verilog model, and DB match the declared live identities; the foundry member manifest verifies.
- M544, M551, and M558 failure provenance remains present and double sealed. The old consumed partial is still exactly eight plain regular files with the eight frozen SHA256 values and no extra member.

## Closed authorization

The candidate's one-VCS/one-`simv` numbers are a prospective closed budget, not a live authorization. At this gate `launch_now=false`; the separate true release and final-release hammer were absent before review authoring. Therefore the effective policy is `run_vcs=false`, `max_attempts=0`. The runner also requires the future candidate-hammer, true-release, and final-hammer chain before it can reach result creation.

The resource and collision schema matches the runner: three two-second-spaced prelaunch samples, 128 GiB `MemAvailable`, 32 GiB swap and commit-headroom thresholds, fail-closed cgroup-v1 counters, and zero same-UID DC/FM/PT/PTPX/VCS/`simv` collisions with PID/starttime/exe/argv evidence. This is a schema audit only, not a live resource admission.

## Decision and claim boundary

Another agent may author a separate `launch_now=true` release that binds this sealed review. Even then, no attempt is authorized until a fresh final-release hammer independently scores 100/100 with P0/P1/P2 = 0/0/0. This review establishes no functional VCS result, RTL verification, speedup, PPA, energy, full-network result, or paper headline.

`docs/359` remains frozen at `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
