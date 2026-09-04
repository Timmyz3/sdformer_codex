# M782 / M533 r14 pre-mkdir launch-boundary failure audit

Verdict: **PASS failure audit; P0=0, P1=1, P2=0**.

The released r14 command exited rc=1 with `RuntimeError: M770 launch boundary`. This is a deterministic fail-closed source predicate bug. The sealed M770 review contains `decision.r14_launch_authorized_now=false`; the sealed r14 runner instead asks for the nonexistent `decision.vcs_launch_authorized_now`. `dict.get()` returns `None`, so the runner's `is False` test fails at its first M770 validation call (runner line 861 invoking the predicate at line 449).

This failure occurs before the clean-environment gates, preflight temporary directory, live VCS identity/license-status probe, atomic result `mkdir`, VCS compile, and simv. The prospective r14 result and failed-result identities remain absent, and no matching preflight temporary directory exists. Therefore r14 produced no functional, timing, PPA, energy, cycle, speedup, or paper evidence. Under the release's explicit atomic-mkdir definition, the attempt was **not consumed**.

The exact r14 release is nevertheless **permanently withdrawn**. Its sealed runner cannot pass the predicate, and changing the runner changes the released identity. The unconsumed atomic attempt is not permission to invoke the known-defective release again. Preserve all r14 seals as historical evidence; do not edit, rerun, relabel, or cite them.

M779 missed the defect because it checked the intended M770 field (`r14_launch_authorized_now`) in its own semantic mirror but did not validate the actual field name used inside the runner's executable Python heredoc. This is `M782-P1-01`, a blocking validation-coverage omission. It is not P0 because fail-closed ordering prevented result creation and all live license/HDL/EDA side effects. M782 supersedes M779 for launch purposes.

Only one additive **r15 source package** is authorized. Its executable change is limited to replacing the bad lookup with `decision.r14_launch_authorized_now`, plus additive identity/hash/path rebinding. RTL, TB r7, SVA r2, macro adapter, foundry `UNIT_DELAY`, environment, license, protocol, coverage, watchdog, resource, collision, timing-bypass, and claim boundaries remain frozen. No r15 launch, VCS, simv, license query, or other EDA run is authorized by this review; a fresh source/candidate/release/final-hammer chain is mandatory.

`docs/359_DATE终局冻结_20260813.md` remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
