# M672 — M660-r2 fresh independent static hammer

## Verdict

**NO-GO, 88/100, P0=0 / P1=1 / P2=0. Do not run the GPU capture and do not consume the one-shot.**

The repaired producer closes the four M666 P1 implementation defects. The
real H67 checkpoint also reconstructs exactly on CPU: `missing=0`,
`unexpected=0`, wrapper `Spiking_neuron`, leaf `ATLIFTernaryPSN`, exact
parameter `sttmultires_unet.decoders.1.sn.spiking_neuron.thresh`, theta bytes
`b3ff7f3f`, and no forward. The combined M660-r2/M665 CPU suite passes 39/39.

One runner admission defect still blocks execution.

## P1-1 — reviewed CPU-preflight identities are not enforced

M670 freezes two exact preflight identities:

- `preflight.json`: `8dbab013ed5099b699eed0a1d7e085e6afdd9f873f73d53006c027338b37af3a`
- `SHA256SUMS.seal.sha256` file: `adbc96005afa1126567f2c2ce70283b1db37d7cb6c81f6590d5cffff132b05ae`

The runner accepts externally reviewed SHA values only for itself and the
contract (lines 45–56). For a pre-existing CPU preflight it checks the nested
seal and a small semantic subset—status, contract SHA, named parameter, and
zero missing/unexpected keys (lines 186–208)—then creates the attempt at line
211. It never compares the receipt or outer-seal file with the two identities
frozen by M670.

The independent attack copied the preflight privately, added a field to the
receipt, regenerated both seals, and replayed the runner's literal predicates.
The changed receipt and changed outer-seal identity both passed. No canonical
file was modified and no attempt was consumed.

This is a fail-closed admission defect: a review-time CPU topology preflight
can be substituted after review by a different consistently resealed receipt.
Even if the later producer rejects a real topology fault, the one-shot has
already been consumed.

## Minimum repair

1. Require reviewer-supplied expected SHA values for both `preflight.json` and
   `SHA256SUMS.seal.sha256` before resource checks.
2. Compare both digests after nested-seal verification and once more
   immediately before creating the attempt.
3. Record expected and observed values in the initial double-sealed attempt
   receipt.
4. Update the runner identity in the contract, regenerate the CPU preflight
   against that new contract SHA, freeze both new preflight identities, and
   request a fresh independent review.

## What passed

- M670 request, M669 author handoff, CPU preflight and M666 review double seals.
- Every contract input and predecessor evidence root.
- Independent real-checkpoint CPU exact-load reconstruction; no forward/GPU.
- Exact wrapper/leaf path and non-aliasing theta clone.
- The source's 62-check sample/order-bound stability lattice and 30-or-40
  payload / 40-hook lattice, plus the author attacks rerun in the 39/39 suite.
- Raw FP32 `uint32` comparison attacks: signed zero, adjacent ULP at a chunk
  boundary, NaN payload bits, hashes, and the all-ten conjunctive gate.
- Post-finalization-style scrub of candidate masks, folded weight, sidecar,
  manifest, success marker and both seal levels.
- Deterministic algorithms, cuDNN deterministic/benchmark modes, both TF32
  controls and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- M665 schema/packing/route tests.
- Canonical output and attempt remain absent; docs/359 remains
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

M658 P2 deliberately remains `PENDING_POST_RESULT_INDEPENDENT_HAMMER`; this
static review does not pre-close it. No payload, performance, RTL, EDA, energy,
PPA, system-speedup, or DATE-headline claim is admitted.
