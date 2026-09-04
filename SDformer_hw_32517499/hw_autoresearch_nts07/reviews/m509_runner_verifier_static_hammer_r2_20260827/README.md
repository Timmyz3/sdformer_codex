# M509 exact-runner and post-export verifier static hammer r2

Verdict: **STATIC GO, exact-literal launch only**. Score: **97/100**.
P0: **0**. P1: **1**.

This was a source-only review. The runner, exporter, verifier, checkpoint loader,
GPU workload, VCS, DC, Formality, PrimeTime, and DSE were not executed. No
reviewed production source was modified.

## r1 P0 closure

1. **Relative `BASH_SOURCE` burn is closed.** The runner resolves
   `${BASH_SOURCE[0]}` to `m509_runner_abs` before changing directory, requires
   that resolved file to equal the canonical runner path, and uses the absolute
   path in both the pre-attempt identity ledger and final `sha256sum -c`.
   Repository-root, hardware-root, and absolute invocations therefore resolve to
   the same identity. A symlink/non-canonical layout fails before consumption.
2. **Output-parent burn is closed.** The absent canonical outgoing parent is
   created before attempt consumption. Both export and verifier parents must be
   writable directories, and both filesystems must have at least 2 GiB free,
   before any attempt directory exists.

## Independent verifier closure

The verifier now derives every one of the twelve FC2 targets directly from the
frozen source tensor in float64: per-row `max(abs(w))/127` (one for zero rows),
ties-to-even `torch.round`, clamp to `[-127,127]`, INT8 conversion, row sumabs,
and dequantization back to the source dtype. It requires exact equality of the
sealed NPZ scale/code/sumabs, serialized candidate tensor, shape, and manifest
statistics to that source-derived result. Thus a mutually consistent but wrong
candidate/NPZ pair no longer passes.

## Mandatory launch authorization

Static GO applies only when the caller supplies the reviewed SHA as a literal,
not a dynamically recomputed value:

```text
M509_EXPECTED_RUNNER_SHA256=081b9c23ebb10fa661b0b7391590a031cd5a7172686743621450ab946f79940c \
  /home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/system_handoff/scripts/run_m509_h67_ep35_fc2_only_int8_ptq_export_r2_exact_sha.sh
```

The runner's environment comparison is deliberately an external authorization
hook; it would accept any self-consistent caller value. Therefore a value
obtained at launch with `sha256sum`, command substitution, wildcard, or copied
from an unsealed artifact is **not authorized by this review**.

## Residual P1

- The runner seals three preflight cgroup snapshots and enforces zero failcnt
  plus `under_oom=0` at start/end, but does not explicitly read and seal the
  `oom_kill` counter exposed by this cgroup-v1 `memory.oom_control`. A successful
  export is still strongly guarded because failcnt must remain zero; adding an
  explicit start/end `oom_kill` equality would make the contract wording fully
  literal and improve auditability.

## Verified execution boundary

- Exporter, verifier, contract, preflight-review seal, checkpoint, M51/M160
  ledgers, compatibility sources, and docs/359 are exact-SHA checked before and
  after the one-shot.
- The three-sample resource/process gate and final process gate run before
  attempt consumption; failure leaves the one-shot unconsumed.
- A pre-attempt identity ledger is constructed and successfully checked before
  the atomic attempt `mkdir`; its final check includes the absolute runner path.
- The passive candidate must contain exactly `model_state_dict`; all state keys
  match the source; non-target tensors are bit exact; all twelve targets and 36
  NPZ arrays pass frozen-source recomputation.
- Export and verifier artifacts are staged, atomically published, inner/outer
  sealed, and rehashed before a final PASS attempt seal is emitted.
- Accuracy, BN2 bridge, RTL, cycle, energy, PPA, system-speedup, and DATE-headline
  admissions remain false.

`bash -n` and verifier byte compilation passed. ShellCheck is unavailable.
Frozen docs/359 remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
