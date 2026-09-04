# M808｜M798 decoder production precompute failure hammer

## Verdict

**PASS100 failure audit; M798 is permanently consumed and may not be rerun.** The exact authorized command consumed the sole canonical attempt, then the production driver rejected the runner-created attempt directory before staging allocation and before any record/timestep/config scheduling. No raw cycle, speedup, decoder-complete, full-network or Table-A claim exists.

Target defects: **P0=1, P1=1, P2=0**.

## Exact identity and one-shot state

- Candidate: `db787cb6d2af42ff61ae4b0827c405747f71571ad4c798a1036e8b11f841fed7`
- Driver: `44b7deca2791e7a47bde9413a8acbb6f13aa7da6938699b98ccc080cdcf428a3`
- Runner: `daf55986d22c7c0250cb39c0dfd9cf4f35fa01f654bcafbc00743a7a12886fb5`
- True release: `cebcfe1f65d49396fb8ea08c01dd0027cb313c957a98c00d4836b1d4b813b892`
- Final-launch hammer review: `a32e4e2e556b23fa9af7940c344d5768ed88704a33b1ea8c4374d486395dae95`
- Attempt receipt: `541f32d65681cf1f0c459a0b965617aa57828c47ab53ff426945a909c0cc0dc0`

The source-hammer, true-release sidecars and final-launch-hammer seals independently recompute. The final hammer's future command exactly names the runner and release SHA values observed in the attempt receipt. The canonical attempt now exists with status `CONSUMED_IMMEDIATELY_BEFORE_M798_PRODUCTION_REPLAY` and `max_attempts=1`. The runner refuses an existing attempt. Therefore this M798 identity is permanently spent; deleting, moving or rewriting the sentinel is prohibited.

## Root cause

The exact failure path is:

`run_production → validate_true_release(attempt_required=True) → _validate_candidate(attempt_required=True) → verify_sealed(canonical_attempt) → M785/M768.verify_sealed_directory`.

The driver verifies the correct canonical directory:

`results/.m798_m785_h67_decoder_physical_residency_cycles_r1_attempt_consumed`

Its root manifest expects exactly:

- `initial/SHA256SUMS.seal.sha256`

The generic recursive verifier actually sees:

- `initial/SHA256SUMS`
- `initial/SHA256SUMS.seal.sha256`
- `initial/attempt.json`

Thus `missing = {}` and `unexpected = {initial/SHA256SUMS, initial/attempt.json}`. The inner `initial/` population is internally complete (`expected = actual = {attempt.json}`), and all four manifest/outer-seal hashes are intact. This is not external directory pollution, not an incorrect `attempt` variable, and not release-semantics drift. It is a deterministic mismatch between a hierarchical runner seal and a verifier that demands a flat manifest listing every recursive non-root-seal file.

## Execution boundary

The failure occurs before `output.mkdir`, contract/oracle loading and every schedule loop. Consequently:

- staging directories: 0
- canonical result directories: 0
- scheduled population records: 0
- detailed rows: 0
- computed cycles or ratios: 0
- citable raw results: none

The driver entered production mode only far enough to validate identity. It did not execute the decoder cycle model.

## Missing failure quarantine

At failure the runner had `started=1` and `success=0`, but neither a published result nor a stage existed. Its EXIT trap only moves one of those two objects to quarantine. With neither present, it exits without generating `failed_or_incomplete.*`.

This preserves claim safety—there is no partial result to mistake for evidence—but loses post-consumption auditability. A post-consumption nonzero exit must always create an atomic no-clobber, double-sealed failure receipt containing the return code, attempt outer seal, exact source/release identities and phase flags, explicitly with no cycle fields. This is M808-P1-1.

## Minimal legal repair

M798 cannot be recovered. The additive repair requires a new identity and new canonical result/attempt paths, plus a new candidate, driver, runner and tests. The frozen M785 analyzer, contract and oracles may remain exact-SHA parents.

The smallest safe design is:

1. Flatten the attempt seal so root `SHA256SUMS` directly names `attempt.json`, or define one exact hierarchical verifier. Do not mix formats.
2. In a temporary sibling, construct the future runner's byte-exact attempt tree and execute the full `attempt_required=True` validation path before release. It must reach the pre-staging boundary without scheduling a row.
3. Add a sealed failure receipt for every nonzero exit after attempt consumption, including validation failures before stage creation.
4. Bind the new candidate/driver/runner/tests/canonical paths by exact SHA; obtain a fresh source hammer, then a new true release and a fresh final-launch hammer.

Authorized now: **source-only additive repair under a new identity**. Not authorized: M798 rerun, production replay, result hammer, Table-A insertion, cycle/speedup claims, VCS/EDA/license/GPU/remote work.

`docs/359_DATE终局冻结_20260813.md` remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
