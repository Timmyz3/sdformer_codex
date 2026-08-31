# M509 exact-runner narrow static hammer r3

Verdict: **STATIC GO, exact-literal launch only**. Score: **99/100**.
P0: **0**. P1: **0**.

This was a narrow source-only review of the r2 residual cgroup issue. The
runner, exporter, verifier, checkpoint loader, GPU workload, VCS, DC,
Formality, PrimeTime, and DSE were not executed. No reviewed production file
was modified.

## r2 P1 closure

- Each of the three preflight resource snapshots now reads and records the
  canonical cgroup-v1 `oom_kill` counter alongside failcnt and `under_oom`.
- Immediately before the pre-attempt identity ledger and atomic attempt
  consumption, the runner samples failcnt, `under_oom`, and `oom_kill` start
  values. All three are written into `initial/ATTEMPT_CONSUMED.txt`, which is
  included in the initial inner and outer seals.
- After export, independent verification, artifact rehash, frozen-input rehash,
  and start/end identity verification, it samples the three end values. PASS
  requires failcnt unchanged, `under_oom=0`, and `oom_kill` unchanged.
- All three end values are written into `POSTEXPORT_PASS.txt`, which is included
  in the final attempt inner and outer seals. A changed counter exits before any
  final PASS receipt exists.
- The fixed server path used by the runner currently exposes a numeric
  `oom_kill` field in `memory.oom_control`; this review does not authorize moving
  the runner to a different cgroup layout.

## Identity and launch authorization

The verifier remains exactly
`660f9a28056350a558e48ea3bdcfd8420c062686047c52b2ca96bf8ba2ffcf7b`;
the source-recomputed twelve-target quantizer proof from r2 is unchanged. The
runner changed and is authorized only at
`bc6e3c174058b83e4219444c84d07b967385253b55bcdcc4128e2128064da0fd`.

The only authorized launch form is an explicit literal assignment followed by
the canonical absolute runner path:

```text
M509_EXPECTED_RUNNER_SHA256=bc6e3c174058b83e4219444c84d07b967385253b55bcdcc4128e2128064da0fd \
  /home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/system_handoff/scripts/run_m509_h67_ep35_fc2_only_int8_ptq_export_r2_exact_sha.sh
```

The obsolete r2 literal `081b9c23...` must not be used. Dynamic command
substitution or recomputing the SHA at launch is not authorized. All internal
resource, process, canonical-path, output-absence, and identity gates must also
pass.

`bash -n` passes. Frozen docs/359 remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
No accuracy, BN2, RTL, cycle, energy, PPA, system-speedup, or DATE-headline
claim is admitted by this static review.
