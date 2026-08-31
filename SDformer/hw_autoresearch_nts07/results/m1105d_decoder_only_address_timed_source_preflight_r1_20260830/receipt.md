# M1105D source preflight receipt

Status: `PASS_SOURCE_AND_FULL_IDENTITY_PREFLIGHT__PRODUCTION_NOT_RELEASED`.

- Frozen population: H67 ep35, three sequences, 30 samples, 120 D0-D3 calls, 261,090,000 packed bytes.
- Global call ordinals are contiguous and every sample is ordered D0, D1, D2, D3.
- All sealed payload identities passed.
- All 30 D1 bitpacks were reconstructed as exact little-endian FP32 words `{0, 1065353139}` and matched the frozen raw-content SHA256: 0 mismatch.
- D1 weights were not folded and theta was not coerced to one.
- The receipt contains per-call address regions plus the common resource and transaction/timestamp schemas.
- No production transaction enumeration, cycle, traffic, speedup, RTL, EDA, energy, or PPA result is admitted.
- A different-author source hammer must pass before a later production runner is released.
- If a final checkpoint replaces ep35, activity, theta, weights, identity, miters, and every derived result require rebinding.

M700/Prosperity is not an input and none of its external opportunity metrics are claimed as ours. docs/359 was not modified.
