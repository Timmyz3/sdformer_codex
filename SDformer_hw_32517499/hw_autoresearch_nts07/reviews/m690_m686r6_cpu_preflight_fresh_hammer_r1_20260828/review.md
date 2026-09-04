# M690｜M686-r6 CPU preflight fresh hammer

## Verdict

**GO_GPU_ONE_SHOT — capture only.** Score **100/100**, severity **P0=0, P1=0, P2=0**.

The preflight is a valid CPU-only exact-load/topology receipt. This approval authorizes exactly one launch through the reviewed runner and four exact environment SHA roots. It does not admit the future payload and authorizes no cycle, speedup, RTL, EDA, energy, PPA or paper claim. Any success or failure consumes the attempt; a successful result still requires a fresh independent result hammer.

## Preflight evidence

- Exact file population: `RUN_COMPLETE.txt`, `preflight.json`, `SHA256SUMS`, and `SHA256SUMS.seal.sha256`; no symlinks.
- Sealed population is exactly `RUN_COMPLETE.txt` and `preflight.json`; both seal levels independently verify.
- Receipt SHA: `bfd05b9bc6ef9e6b66c336b4392082ec72c096b5c40e7a89f17b5227a7b0d78f`.
- Manifest SHA: `a257caede04fcbbf0b43990e849d9118ad4ff331dd8e02703c8fb754486eb28c`.
- Outer-seal-file SHA: `deb66bd505802ee7b5e46953ff93477d657300845429ae041293879880f6cb5c`.
- Contract and checkpoint identities match; checkpoint and overlay load audits are all 0/0.
- The real topology is `sttmultires_unet.decoders.1.sn` (`Spiking_neuron`) → `spiking_neuron` (`ATLIFTernaryPSN`) → scalar `thresh`.
- Device is CPU; no forward, GPU, capture or one-shot was executed.
- Deterministic controls are exact, including `cudnn_allow_tf32=true`.
- Updated combined static tests pass **28/28**.

## Independent attacks

Two attacks were replayed on temporary copies:

1. Changing `RUN_COMPLETE.txt` without resealing leaves both external roots unchanged, but the repaired second member verification returns failure before attempt creation.
2. Changing `preflight.json` and consistently regenerating both seals passes full seal verification, but both independently reviewed external roots mismatch before attempt creation.

The real attempt and canonical output remained absent after both attacks.

## Only authorized command

```bash
/usr/bin/env -i \
M660R2_EXPECTED_CONTRACT_SHA256=cd17f141c2e7dc26b6b9093251ebe98b793e3e3436c7ea1f598dc2b4e1959b04 \
M660R2_EXPECTED_RUNNER_SHA256=a9b9644a410f46b0c3b241fcf9691442865646c5a73c51c22fd82de19edc9c39 \
M660R3_EXPECTED_PREFLIGHT_RECEIPT_SHA256=bfd05b9bc6ef9e6b66c336b4392082ec72c096b5c40e7a89f17b5227a7b0d78f \
M660R3_EXPECTED_PREFLIGHT_OUTER_SEAL_SHA256=deb66bd505802ee7b5e46953ff93477d657300845429ae041293879880f6cb5c \
/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/system_handoff/scripts/run_m686r6_h67_layer_static_decoder_payload_one_shot.sh
```

M690 itself executed no GPU workload or EDA action and did not modify the preflight, runner, contract, tests, attempt, output or docs/359.
