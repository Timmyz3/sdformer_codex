# M660 H67 layer-static decoder payload｜author handoff r1

Date: 2026-08-28  
Status: **STATIC AUTHOR HANDOFF ONLY — GPU / one-shot / canonical output remain absent**

## Frozen candidate

| Item | Path | SHA256 |
|---|---|---|
| Producer | `neuron_experiments/H9_bipolar_self_attention/entrypoints/capture_m660_h67_layer_static_decoder_payload.py` | `2e1ea26b5293ba1063e7be0056cebd2b25e09903bb528c31427c032df8b73acc` |
| One-shot runner | `hw_autoresearch_nts07/system_handoff/scripts/run_m660_h67_layer_static_decoder_payload_one_shot.sh` | `ae9902b42331f3e88e94b11d9c5a5f6f3bdfc3e2b473939a7569af38f2396281` |
| Contract | `hw_autoresearch_nts07/contracts/m660_h67_ep35_layer_static_decoder_payload_contract_r1_20260828.json` | `38200ef4db5795d8be70e6e776aabf09dad10818344b972add535900a95f2cb4` |
| CPU static tests | `hw_autoresearch_nts07/system_simulator/tests/test_m660_h67_layer_static_decoder_payload.py` | `0dc63c88349dec0ecc77d2fb4aa51f0df82316d1c435a73f1d760ae50fb54cc0` |

The contract binds the final M658 verdict (`review.json` SHA
`3edece8a...`, member seal `aed109ab...`, outer-seal file `5d235106...`),
the conservative M659 fallback plan, and the later M662 conditional
threshold-fold audit (`review.json` SHA `58b105f4...`, member seal
`90e2a7a...`, outer-seal file `b3d48002...`).  M662 is treated only as a
conditional measurement prerequisite; it contributes no internal performance
claim.

## Exact typed decision

The producer preserves M511's exact H67 ep35 model, checkpoint, dataset,
preprocessing, no-running-BN protocol, ten samples and sample-major four-hook
order.  It has two independent decisions:

1. D0/D2/D3 must remain exact `{0,1}` and always form the unique 30-cell
   payload lattice.  Their little-bit-first bitpacks total exactly
   **75,480,000 bytes**.
2. D1 must have one finite, strictly positive, stable FP32 scalar official-ATLIF
   threshold.  Only if every one of its 92.4M S10 elements is bit-exactly
   `0` or that runtime scalar `theta` may ten D1 masks totaling
   **11,550,000 bytes** be promoted, yielding **87,030,000 bytes** across all
   four modules.  Any other value selects the full-770-channel common FP32
   fallback and publishes no D1 bitpack, folded weight or output-scale
   sidecar.

Passing `{0,theta}` admits only the exact scaled-binary representation.  The
producer separately runs a streamed original-output versus
`ConvTranspose2d(mask, float32(theta*W))` miter.  Folded-weight deployment is
admitted only when all ten output hashes are byte-identical and every mismatch
count is zero.  A nonexact miter leaves the folded payload explicitly marked
`DIAGNOSTIC_CANDIDATE_NOT_ADMITTED`; the separately sealed original-weight +
output-scale sidecar is also `UNMITERED_CANDIDATE_NOT_ADMITTED`.  Neither is
silently called lossless.

All four original FP32 weights, the conditional folded candidate and the
sidecar are independently sealed under `weights/`.  Every module output is
stream-hashed.  Raw D1 activation values are never serialized, rounded,
thresholded or coerced.

## M658 P2 closure

The canonical candidate contains an independently double-sealed
`runtime_receipt/` with hostname, exact Python executable and SHA, Python,
torch/numpy/spikingjelly, compiled CUDA/cuDNN, nvidia-smi binary and SHA,
driver, GPU UUID/name/memory/compute capability, exact argv, and the complete
sanitized environment.  The runner uses `/usr/bin/env -i`.

The externally reviewed contract and runner are independently rooted.  The
runner requires both nonempty caller values, verifies both files before
one-shot consumption, rechecks the contract in Python preflight, and passes
both roots through the sanitized environment.  The producer then requires the
contract-root environment value to equal the running contract SHA.

## Only candidate command after fresh hammer explicit GO

```bash
M660_EXPECTED_CONTRACT_SHA256=38200ef4db5795d8be70e6e776aabf09dad10818344b972add535900a95f2cb4 \
M660_EXPECTED_RUNNER_SHA256=ae9902b42331f3e88e94b11d9c5a5f6f3bdfc3e2b473939a7569af38f2396281 \
hw_autoresearch_nts07/system_handoff/scripts/run_m660_h67_layer_static_decoder_payload_one_shot.sh
```

Do not run this command until a fresh independent static hammer reports
P0=0, P1=0 and an explicit GO.  The runner consumes a new M660-only one-shot
immediately before Python capture; failure does not restore authorization.

## Claim boundary

This handoff authors source only.  It does not authorize or report GPU
execution, a captured payload, cycle simulation, speedup, RTL, VCS, DC,
Formality, PT/PX, energy, PPA, system speedup or a DATE headline.  The M511
one-shot and failed staging, M649/M658/M659/M662 evidence, and `docs/359` are
read-only; `docs/359` remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
