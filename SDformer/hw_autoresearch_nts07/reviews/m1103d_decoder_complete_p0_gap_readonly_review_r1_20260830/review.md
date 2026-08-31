# M1103D decoder-complete P0 gap read-only review (2026-08-30)

## Verdict

**STOP_CURRENT_EVIDENCE_NO_SAME_RESOURCE_ADDRESS_TIMED_DECODER_COMPLETE_BASELINE.** Existing M699/M705 data are sufficient to freeze a decoder-only H67 ep35, three-sequence workload without waiting for a newer checkpoint, but they do not contain an executable common-resource address/timestamp schedule. No decoder-complete cycle, traffic, energy, FPS, or speedup claim is admitted.

This is a read-only availability review. It launched no EDA, RTL, simulator, GPU capture, or full replay and does not modify docs/359.

## What is already usable

- M699/M705 contains 30 samples (three DSEC sequences x 10), 120 D0-D3 hook calls, 261,090,000 packed bytes, and checkpoint-bound module/weight identities. D0/D2/D3 are exact `{0,1}` payloads.
- D1 is exactly `{0, theta}` with runtime `theta=0.9999954104423523` (IEEE-754 word 1065353139), not `{0,1}`. The payload is exact scaled-binary, but folded-weight deployment and decoder numerical equivalence are not admitted.
- The selected cohorts show stable density (maximum per-module spread 0.4478 percentage points), but this is density evidence, not performance evidence.
- These facts remain valid only if the paper freezes H67 ep35. Selecting a newer/final checkpoint requires rebinding weights, theta, payloads, identities, densities, numerical miters, and all derived cycle/traffic results.

## Why the current baseline is still incomplete

M624 correctly remains fail-closed. Its old full-network trace contains zero ConvTranspose rows and its local M51 bundle is missing 150/310 records (140 Linear and 10 Conv2d; 564,480,000 bytes). M699 repairs the decoder activation population for decoder-only analysis, but does not repair the full-network scope.

For a decoder-only same-resource baseline, the remaining P0 gaps are:

1. no global call ordinal and no address-timed SRAM/DRAM transaction stream for all D0-D3 calls;
2. no common executable schedule with frozen 96 lanes, 240 KiB SRAM, 192 B per 3 ns cycle, bank/port/commit/spill rules, and a fixed numerator;
3. no admitted D1 numerical policy. D1 must retain its typed theta multiplier with original checkpoint weights, or a separately verified weight-folding miter must prove equivalence;
4. no same-schedule dense/bit/K8 cycle, stall, and byte rows.

For a full-network decoder-complete baseline, the M51 missing 150 records, exhaustive operator scope, decoder global order, and fixed-numerator receipt remain additionally mandatory.

## Minimum one-day source-only closure

1. Freeze ep35 and consume the sealed M699 manifest/calls; derive and seal the actual D0-D3 model order from the producer/model rather than assuming it.
2. Add a CPU-only decoder adapter that expands each call into source descriptors plus explicit SRAM/DRAM addresses, timestamps, reads, writes, stalls, and final commits under the frozen common resource.
3. Keep D1 as a typed scaled-binary source (`bit * theta`) using the original numeric weights. Run an exact/reference numerical miter for all 120 calls; do not silently coerce theta to one or fold it into weights.
4. Replay dense, bit, and K8 modes on the identical transaction schedule and report cycles, stall breakdown, SRAM/DRAM bytes, and fixed numerator separately.
5. Seal the result and run an independent hammer. If any required weight, order, numerical oracle, or memory rule is absent, STOP rather than synthesize it.

Required inputs are the M699 `manifest.json`, `calls/*.bitpack`, `SHA256SUMS` and outer seal; M705 review/seals; the M672/M670 polyphase mapper and M677 review; checkpoint-bound decoder weights/identity (M686 or the frozen ep35 source); the M699 capture producer and model module order; and the M520/M527 plus M22/M23 resource/memory definitions.

## Prosperity boundary

M700/M739 is only a Table-C external opportunity result. On D0/D2/D3 binary-support records, the unmodified official Prosperity CPU simulator reports 465.918M product cycles versus 1,438.563M bit cycles, 3.087586x (per-call geometric mean 2.944887x). D1 is excluded from the exact subset. This is phase-summed support work, not our hardware, not same-resource local execution, not monolithic ConvTranspose latency, and not decoder/full-network speedup. It must not be multiplied by C1/C2 results.

## Frozen evidence identities

- M624 result SHA256: `d213e9ed57d5ccdd869b17ea64ca6ba6920d2e6b962451a4bbc00468990a5614`
- M705 review SHA256: `6af48fb271254ef20f6baa1e435acfe51fdf38b457fe9782d6cac0b0e2883bd3`
- M739 review SHA256: `9ef7dc5390ca2c1717dded1ea89d787336bd01296da6aed674536981f0f9caa2`
- M699 manifest SHA256: `e2d7c92a038c213b590603ff534a33f3579bf1224cc3f56c11629e1d4c813dc0`
- docs/359 SHA256 remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
