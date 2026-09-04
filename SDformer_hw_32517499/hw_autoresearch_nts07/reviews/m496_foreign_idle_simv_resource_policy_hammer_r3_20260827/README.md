# M496 foreign-idle simv resource-policy static hammer r3

Verdict: **STATIC GO, exact-literal launch only**. Score: **98/100**.
P0: **0**. P1: **1**.

This was a source-only final review. DC, VCS, simulation, Formality,
PrimeTime, and DSE were not launched. No production file or docs/359 was
modified.

## Final state-machine verdict

- The canonical run remains absent throughout all preflights and all three DC
  points. Work is accumulated only under a PID-qualified hidden work root.
- Before first DC, a same-filesystem attempt staging directory is populated with
  exactly two payloads (`ATTEMPT_CONSUMED.txt`, `identity.sha256`), an inner
  manifest, and outer seal. `mv -T` publishes it at the canonical attempt path;
  the exact four-member population, both seals, both identities, status, and
  canonical-run field are immediately checked.
- The attempt receipt is rechecked after every point and at both final identity
  boundaries. Its `canonical_run` field names the public canonical directory,
  not the PID work root.
- Full runner/input identity is checked initially, before and after every point,
  before final receipt construction, and after final receipt construction. The
  initial input manifest is rehashed at both final boundaries.
- The runtime monitor's OOM latch and return code are hard gates. Each point also
  requires a clear cgroup after monitor completion; final receipt construction
  and final publication both require another clear cgroup.
- Root PASS files and the evidence seal are first created only inside the hidden
  work root. The last input, attempt, cgroup, and seal checks all complete before
  a single same-filesystem `mv -T` publishes the entire directory at the
  canonical path. Relative evidence-manifest paths are rooted below the run and
  remain valid after the directory rename.
- On any ordinary failure before publication, the EXIT trap writes a failure
  marker where applicable and moves the work root to a PID-qualified failed or
  prelaunch quarantine. No canonical PASS remains.

## SIGKILL analysis

SIGKILL cannot be trapped, but all windows are fail-closed:

- During preflight: only a temporary preflight directory may remain; canonical
  and attempt stay absent.
- After work creation but before attempt publication: only a hidden work root
  may remain; it contains no launched-DC admission.
- After attempt publication and before/during DC: attempt remains consumed and a
  hidden partial work root may remain; canonical stays absent, so retry is
  conservatively blocked.
- After all hard gates but before final rename: a sealed hidden work root and
  consumed attempt may remain, but no canonical PASS is visible.
- During the final rename: rename is atomic on the same filesystem, so either
  hidden work or the fully checked canonical run exists. If canonical exists,
  all hard gates and seal verification already passed.

The small marker-to-process-spawn window can consume the one-shot without an
actual DC child if SIGKILL lands there. That is conservative loss of an attempt,
not false admission or an unauthorized retry.

## Residual P1

`m485_complete=1` is assigned immediately before the final `mv -T`. If the
rename itself returns an ordinary error, the EXIT trap sees `complete=1` and
does not quarantine the hidden work root. Canonical remains absent, the runner
returns failure, and attempt remains consumed, so admission safety is intact;
only cleanup/forensic organization is weaker. A future cleanup may introduce a
separate `publish_started` state or set complete after a successfully verified
rename with a trap branch covering both paths. This is not a DC launch blocker.

## Literal execution authorization

Static GO applies only to this literal SHA and canonical absolute path:

```text
M496_EXPECTED_RUNNER_SHA256=35ab6393fe533f514a1353d13c3c59a6731f103cbf3a355dadc4a64d56c4529e \
  /home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/dc_handoff/scripts/run_dc_m496_fc2_three_axis_matched_logic_only_exact_sha.sh
```

Dynamic SHA command substitution is not authorized. All internal resource,
process, output-absence, exact-identity, attempt, OOM, DC-report, timing, and
seal gates must pass. At review time canonical run, PID work root, and canonical
attempt were absent; `bash -n` passed. The observed foreign `simv` remained
`fangyl / Sl / 0.0% / 110032 KiB` and is only sampled, never signaled.

Frozen docs/359 remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
This authorizes one exact resource-gated logic-only DC replay, not PPA, power,
system speedup, or a DATE headline.
