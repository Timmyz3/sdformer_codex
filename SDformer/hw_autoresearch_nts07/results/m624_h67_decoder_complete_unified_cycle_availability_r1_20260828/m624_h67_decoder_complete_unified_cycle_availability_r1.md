# M624 H67 ep35 decoder-complete unified cycle simulator availability

## Verdict

`FAIL_CLOSED_INPUT_AND_EXECUTABLE_SCHEMA_GAPS__NO_CYCLE_RESULT`. No decoder-complete cycles, traffic, stall, Fixed numerator, speedup or headline were generated.

The frozen ordered trace is useful but partial: 1840 rows = 790 operator + 930 ATLIF + 120 attention rows across 10 samples, and it contains zero ConvTranspose rows. M51 declares 310 binary records but only 160 payloads are locally present; 150 payloads (564480000 bytes) are missing.

## Configuration matrix

| Row | Current executable path | Blocking gate | Unified metrics |
|---|---|---|---|
| B0_Dense96_Fixed_T10 | 10-sample ordered trace: shapes/order for 79 Conv2d/Linear modules plus ATLIF/attention<br>M22 logical traffic and M23 bank-port envelope are partial inventory only<br>M518 Fixed-T10 directed VCS component | zero ConvTranspose rows/bitpacks<br>no complete operator-scope/fixed numerator<br>no common compute+memory completion schedule | null |
| B1_PTB_like_structured_K1x8 | M527 project-defined PTB-like semantics only | no executable configuration manifest<br>no per-group full-population scan ledger<br>no charged dense fallback/decoder schedule | null |
| B2_exact_bit_sparse_K1 | M216 FC2 aggregate K1 CPU component<br>M519 directed K1 VCS component<br>M51 exact-binary manifest | 150 of 310 M51 payloads locally absent<br>decoder bitpacks absent<br>no all-operator K1/common-memory schedule | null |
| B3_exact_bit_sparse_K1x8 | M519 directed K1x8 VCS component<br>M51 exact-binary manifest | 150 of 310 M51 payloads locally absent<br>decoder bitpacks absent<br>no all-operator replicated-state/control/resource manifest | null |
| Ours_C1_C2_C3_exact | C1 M528 exact CPU same-ledger four-bottleneck-Conv candidate<br>C2 M216 FC2 + M522 decoder mapper + M523 tap bundler<br>C3 M518 Fixed-T10 directed VCS component | components are disjoint and cannot be summed/multiplied<br>M590 r6 failed static review P0=3/P1=2<br>decoder inputs/weights/result absent<br>no non-overlap shared SRAM/DRAM schedule | null |

## Minimum data handoff

| Item | Action | Exact population |
|---|---|---|
| R1_M511_DECODER_INPUTS | Run the superseding independently reviewed local M511 capture once; then run the sealed payload verifier. | 40 records = 10 samples x 4 ConvTranspose2d; 87,030,000 packed bytes |
| R2_M578_DECODER_WEIGHTS | Export and seal four checkpoint-bound signed-INT8 COUT_CIN_KY_KX tensors; no synthetic weights. | 4 tensors; 7,140,096 int8 bytes; shapes 384x1536x3x3, 192x770x3x3, 96x386x3x3, 96x194x3x3 |
| R3_M51_MISSING_PAYLOAD_TRANSFER | Transfer only manifest-listed missing members and verify each existing SHA; do not recapture or regenerate. | 150 records; 564480000 bytes; operator counts {'Linear': 140, 'Conv2d': 10} |
| R4_DECODER_ORDER_EXTENSION | Capture/seal global execution ordinal for each decoder call or emit a new complete ordered trace; module-local order alone is insufficient for a unified schedule. | 40 metadata rows = 10 samples x 4 decoder calls |
| R5_OPERATOR_SCOPE_AND_FIXED_NUMERATOR | Create a complete operator-scope manifest and M527 fixed-numerator receipt, explicitly charging normalization/state/update/control/fallback work. | one frozen 10-sample population; included/excluded partition must be exhaustive |
| R6_SAFE_UNIFIED_CPU_SOURCE | Repair/supersede M590 r6 or implement a new common scheduler; require fresh static hammer before any production CPU run. | B0/B1/B2/B3/Ours in one 96-lane, 240-KiB, 192-byte-per-3ns-cycle resource schema |

M511 is marked pending a superseding local launch review; this audit does not authorize or run it. M590 r6 remains unusable because M596 found P0=3/P1=2 and forbids execution.

All M510 decoder ranges remain analytic projections only. M522/M523 prove mapper/bundler support, not decoder cycles. M22/M23 remain partial logical/envelope ledgers, not executable total cycles.

`docs/359` remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
