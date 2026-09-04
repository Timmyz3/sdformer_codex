# M700 | M699 multi-sequence decoder capture fresh static hammer

## Verdict

**GO_M699_GPU_ONE_SHOT__CAPTURE_ONLY__P0_0_P1_0**. Score **98/100**; severity **P0=0, P1=0, P2=3**.

This authorizes exactly one capture through the reviewed M699 runner after the review has been externally pinned by its `review.json` SHA and outer-seal-file SHA. It admits no payload or density before a fresh result hammer, and no accuracy, cycle, speedup, RTL, EDA, energy, PPA, system, or DATE-headline claim.

## Frozen evidence

- Producer, runner, contract, author tests, M511 producer, M686-r6 helper, checkpoint/config dependencies, and docs/359 all match their contract-pinned exact byte roots.
- The author package has a complete two-level seal. Its manifest SHA is `029d298276256a0318866f51e9df49f4a50b5b85730ab2b7ba70369c8f26c35a`; outer-seal-file SHA is `e47f50a29f393c51cce4ac6b7668193f78bd8e06892296dfbd814260a5f93af9`.
- All 30 selected NPYs (368,643,840 bytes total) independently match exact relative path, byte count, SHA256, `(10,480,640)` shape, and FP32 dtype. Selection is the declared endpoint-covering `round(i*(N-1)/9)` lattice over 108/75/75 source populations.
- Runtime requires exact H67 ep35 load audit `missing=0, unexpected=0`, four ordered ConvTranspose hooks per sample, 120 final records, and M686-r6 deterministic controls with native `cudnn.allow_tf32=true`.
- Canonical output and one-shot attempt were absent throughout review. Author tests pass 9/9 and runner syntax passes.

## Attacks

Fresh attacks rejected content-SHA replacement, cross-sequence/sample swapping, missing-hook acceptance, scaled values being mislabeled binary, nextafter-to-theta coercion, performance-claim upgrade, an old review with a stale runner root, and removal of the post-consumption fail-closed receipt. The exact scaled route accepts only `{0,runtime theta}`; a near-theta value is rejected and its candidate payload removed.

The runner reverse-binds the externally pinned review roots to the reviewed runner and contract, consumes the one-shot before entering Python, rehashes the review twice before consumption, quarantines any post-publication failure, and rehashes the runner/producer/contract/docs roots after capture.

## Boundaries

The three cohorts contain no labels or masks and therefore cannot produce AEE. A D1 scaled mask is not a folded-weight equivalence proof. Pre/post hashing is not advertised as protection against a privileged concurrent replace-and-restore race. These are P2 scope boundaries, not launch blockers.

This review ran no GPU, model, or EDA workload and did not modify docs/359. The unique launch command is emitted only after this directory is double-sealed, because its two externally supplied review roots cannot be self-embedded in a sealed member.
