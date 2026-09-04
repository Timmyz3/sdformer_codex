# M509 exact-runner and post-export verifier static hammer r1

Verdict: **STATIC NO-GO (2 P0)**. Score: **82/100**.

This review was source-only. It did not execute the runner, exporter, verifier,
checkpoint loader, GPU workload, VCS, DC, Formality, PrimeTime, or DSE, and it
did not modify any reviewed production file.

## P0 findings

1. The runner changes directory to the hardware root but later hashes the
   original `${BASH_SOURCE[0]}` string. When launched from the repository root
   using the natural relative path
   `hw_autoresearch_nts07/system_handoff/scripts/run_...sh`, that string no
   longer resolves after `cd`. The failure occurs only after the attempt
   directory has been created, so it irreversibly consumes the one-shot before
   exporter launch. Resolve the runner to an absolute path before `cd`, use that
   path for both identity snapshots, and demonstrate the repository-root,
   hardware-root, and absolute-path invocation cases statically.
2. The canonical `system_handoff/outgoing` parent is currently absent, yet the
   runner creates it only after writing the consumed attempt seal. A permission,
   race, or filesystem failure can therefore burn the one-shot before exporter
   launch. Create/validate the parent and check writability plus the contract's
   2 GiB free-disk gate before consuming the attempt. The verifier output parent
   should receive the same pre-consumption validation.

## P1 findings

1. The verifier proves that the serialized target tensors equal the sealed NPZ
   dequantization, but it does not independently recompute the contract's
   row-wise scale, ties-to-even INT8 code, `[-127,127]` clamp, shape, and sumabs
   from the frozen source tensor. A consistently wrong NPZ and candidate can
   pass. Add this recomputation if the receipt is to mean independent PTQ
   correctness; otherwise narrow `export_integrity` to cross-artifact
   consistency.
2. The runner checks `under_oom` and cumulative `memory.failcnt` before launch,
   but does not record a start/end `memory.events`/OOM-kill delta. This does not
   admit a failed export, but adding it would fully match the contract wording.
3. The runner itself is captured in the initial identity ledger but is not
   externally authorization-pinned by the earlier preflight review. Launch must
   be conditioned on the reviewed runner SHA (or a later sealed review/launcher)
   and not merely on a self-observed start/end hash.

## Verified strengths

- Contract, exporter, verifier, preflight-review seal, frozen checkpoint,
  manifests, compatibility sources, and docs/359 are pinned before and after
  execution.
- The three-sample resource/process gate precedes attempt consumption, and a
  failed resource gate leaves the one-shot unconsumed.
- Attempt directory creation is concurrency-safe: only one competing launch can
  win the atomic `mkdir`.
- Export and verify outputs are canonical, overwrite-refusing, staged, atomically
  published, and rehashed through inner and outer seals.
- The verifier passively reloads the candidate, requires a one-member
  `model_state_dict`, compares the complete state-key set, proves non-target
  tensor bit equality, reconstructs all twelve FC2 targets from 36 declared NPZ
  arrays, and keeps accuracy/RTL/cycle/energy/PPA/system claims false.
- `bash -n` passes; ShellCheck was not installed. Frozen docs/359 remained at
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

Re-review gate: repair both P0 findings without changing the exporter/contract
identity, add or explicitly narrow the P1 quantizer claim, then request a new
static hammer. Do not execute this r1 runner.
