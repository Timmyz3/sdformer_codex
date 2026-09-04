# M678 — M660-r4 final fresh independent hammer

## Verdict

**GO_ONE_M660R4_S10_GPU_CAPTURE_ONLY — 100/100, P0=0 / P1=0 / P2=0.**

M660-r4 closes M675 P1. The runner verifies the reviewed receipt and outer-seal
file immediately after nested-seal verification, runs the semantic checker,
then repeats the same two comparisons immediately before `mkdir(attempt)`.
The second mismatch exits 42 without creating attempt or output.

The authorization is for exactly one S10 payload capture. It does not admit
the resulting payload, performance, RTL, EDA, energy, PPA, system speedup, or a
DATE headline. A post-result independent hammer remains mandatory.

## Independent evidence

- Every M677 target SHA and all nested seals independently match.
- Runner is an executable, non-symlink regular file (`0775`).
- M660-r2 + M665 + M676: **44/44 tests passed**.
- Wrong receipt with correct outer: frozen runner exits 41; attempt/output
  remain absent.
- Correct receipt with wrong outer: frozen runner exits 41; attempt/output
  remain absent.
- Private-copy between-gate attack: the reviewed pair passed the first gate and
  semantic check; a consistently resealed replacement was then installed. The
  second pair rejected it with exit 42 and no attempt/output. No author artifact
  was modified.
- Fresh H67 CPU exact-load: missing/unexpected `0/0`, exact wrapper
  `Spiking_neuron`, leaf `ATLIFTernaryPSN`, theta `b3ff7f3f`, exact frozen
  receipt match, no forward and no GPU.
- Canonical output and attempt remain absent; docs/359 remains
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

## Exact five launch identities

1. Producer: `53b91b9ec8be00e60a5e029c63c392f5fe5e4773de92b440c6d4561dc1ab0116`
2. Runner: `047540d002f1812ed20097a03705d67f9260d10244d37401ed9a11c7643f631b`
3. Contract: `099f27d16892f633ff5c0847c1e5958d9ba805668942c8d4e76f6d30692606aa`
4. Preflight receipt: `89381b8a8ecf8b9b3b8194fd5b77815b79cd1642ac2be2fd08412fa7ca54c78d`
5. Preflight outer-seal file: `8b1c4c817a94a3c1fe438d8bdc5c8513a7852e2dd90b12f16638e1c13cf83966`

## Unique authorized command

Run only from `/home/zhumd/work/sdformer_codex/SDformer`:

```bash
M660R2_EXPECTED_CONTRACT_SHA256=099f27d16892f633ff5c0847c1e5958d9ba805668942c8d4e76f6d30692606aa \
M660R2_EXPECTED_RUNNER_SHA256=047540d002f1812ed20097a03705d67f9260d10244d37401ed9a11c7643f631b \
M660R3_EXPECTED_PREFLIGHT_RECEIPT_SHA256=89381b8a8ecf8b9b3b8194fd5b77815b79cd1642ac2be2fd08412fa7ca54c78d \
M660R3_EXPECTED_PREFLIGHT_OUTER_SEAL_SHA256=8b1c4c817a94a3c1fe438d8bdc5c8513a7852e2dd90b12f16638e1c13cf83966 \
hw_autoresearch_nts07/system_handoff/scripts/run_m660r4_h67_layer_static_decoder_payload_one_shot.sh
```

The attempt is non-restorable after consumption. There is no second candidate
command under this review.
