# M40a amplitude-codebook event milestone, revision 1

Date: 2026-08-22

Status: `PASS_M40A_EXACT_AMPLITUDE_CODEBOOK_SOURCE_TRACE_BUT_REAL_PRODUCT_SCHEDULE_BLOCKED_ON_WEIGHT_QUANTIZATION_PHYSICAL_LAYOUT_AND_ACCUMULATORS`.

M40a establishes a new H67 ep35 trace identity for the four M39 bottleneck
Conv3x3 inputs.  It does not inherit the M36 profiler SHA: M36 retained only a
patch-embed census.  The M40 tracer ran the exact frozen checkpoint, config and
first ten validation samples on the local RTX 3090, with checkpoint load
`missing=0/unexpected=0` and the configured CuPy SNN backend.

## Frozen artifacts

- contract: `hw_autoresearch_nts07/contracts/m40_conflict_aware_event_schedule_contract_r1_20260822.json`, SHA-256 `1eeeea8f1778f45305226dbccf31a920586dff3eb14ee0bf684ef833728f9018`;
- tracer: `hw_autoresearch_nts07/system_simulator/scripts/trace_m40_bottleneck_packed_sources.py`, SHA-256 `b02ac10fb95e68fa2871b74330d6f39d7d3d8cbfa6440990d43ec832e943bf19`;
- real trace manifest: `hw_autoresearch_nts07/results/m40_h67_ep35_bottleneck_packed_sources_s10_r6_20260822/m40_bottleneck_packed_source_manifest.json`, SHA-256 `e743364bb599214dc13ad2591bf96dbf6091d95f8cc5a585ddc86370ccc514d3`;
- analyzer/reference scheduler: `hw_autoresearch_nts07/system_simulator/scripts/analyze_m40_conflict_aware_event_schedule.py`, SHA-256 `dd6dc32f773d8aa8c095173d51b4b182cb7cead3e8d0e8e3076ed7cb76fba372`;
- Python 3.6 regression: `hw_autoresearch_nts07/system_simulator/tests/test_m40_conflict_aware_event_schedule.py`, SHA-256 `e85e877448641dca1cd1acde8e87f96c4e0a4688cb65e45affd1291bc02dcb0f`;
- result: `hw_autoresearch_nts07/results/m40_conflict_aware_event_schedule_r3_20260822/m40_conflict_aware_event_schedule.json`, SHA-256 `419ea51faabda4c2f45b9fa535d1a0fa8142bb4c8b8258468e88a1dc99c310e7`.

The thirty DSEC event/ground-truth/mask files were hashed independently on the
remote source and after local transfer.  The two sorted per-file SHA-256 lists
have the same digest, `216a7ee39eb46665b269fafa18e688e8ec39c86907435e9ff687682265109f19`.
Every individual file receipt is retained in the real-trace manifest.

The r4 directory is deliberately non-authoritative.  It completed 40 bitmap
files but the loader fetched an unavailable eleventh sample before the old
loop-body stop condition; it has no manifest and carries an immutable
`ABORT_DO_NOT_CITE.json` marker, SHA-256
`74cef3d8ad176a7838b59254dac4043ae714741972f016d73be6f225663e708d`.

## Exact amplitude-code discovery

Every one of the 40 real input tensors has exactly two float32 bit patterns:
zero and one operator-static positive amplitude.  Each code remains constant
across all ten samples and equals the preceding ATLIF checkpoint threshold:

| Conv input | float32 code | UQ0.24 raw | M35 delta | M35 producer |
|---|---:|---:|---:|---|
| resblocks.0.conv1.0 | `3f7fff87` | 16,777,095 | 121 | resblocks.0.sn1 |
| resblocks.0.conv2.0 | `3f7fff70` | 16,777,072 | 144 | resblocks.0.sn2 |
| resblocks.1.conv1.0 | `3f7fff9f` | 16,777,119 | 97 | resblocks.1.sn1 |
| resblocks.1.conv2.0 | `3f7ffdb4` | 16,776,628 | 588 | resblocks.1.sn2 |

The analyzer decompresses every retained float32 payload, verifies its payload
SHA, reconstructs it from the positive bitmap plus the corresponding layer
code, and compares the complete bytes.  The result is exactly 92,160,000 values
and zero bit mismatch.  All four convolutions are bias-free.  Therefore the
following exact-arithmetic identity is admitted:

`sum_i ((bitmap_i * theta_layer) * weight_i) = theta_layer * sum_i (bitmap_i * weight_i)`.

This yields the `LAYER_STATIC_AMPLITUDE_CARRY_EVENT_ACCUMULATION` candidate:
carry one activity bit through event accumulation, conditionally accumulate a
weight for each active source, and apply the already-proven M35 complement-CSD
layer amplitude once per output.  The dense activation representation falls
from 368,640,000 float32 bytes to 11,520,000 activity bytes, exactly 32x,
excluding four static threshold words.  This is a representation and exact
algebra result, not an integrated speedup or energy result.

## Real structured work, without uniform-density substitution

The analyzer replays the actual bitmaps, applies the exact stride-1/pad-1/
dilation-1/groups-1 geometry, and excludes padding pseudo-events.  It expands
valid source/destination multiplicity without materializing 7.64 billion CSV
rows.  Across the four operators, exact 96-lane product lower-bound cycles per
sample are:

| Line | Mean | min | p95 nearest-rank | p99 nearest-rank | max |
|---|---:|---:|---:|---:|---:|
| Local | 74,112,377.6 | 73,417,496 | 74,995,872 | 74,995,872 | 74,995,872 |
| Motion numeric transition | 109,914,755.2 | 108,971,224 | 110,962,768 | 110,962,768 | 110,962,768 |

These are exact product-count lower bounds, not executable schedule cycles.
They close M39's uniform-density P2 for source support and show that Motion is
1.483x worse than Local on this bottleneck trace.  They do not validate the
old 12.8--13.3 million-cycle conditional M4 projection.

## Executable reference boundary

The Python 3.6 scheduler has explicit 4-byte bank mapping, lanes, queue credits,
conflict deferral, coalesced same-word service, contiguous flush boundaries and
LRU weight-tile residency.  Its exact five-event oracle retires 5/5 events with
zero loss in three cycles, exercises one bank-conflict deferral, one credit
stall, two tile evictions, and four zero-mismatch M35 integer checks.  This
admits the reference semantics only.

M22/M23 contain 540 call-level rows per target operator, but lack source,
destination, kernel, operand, accumulator and physical product-address fields.
The real M40 schedule therefore remains fail-closed until an address-run/RLE
trace freezes int8 weight and bias quantization, capacity-valid tiling and
base addresses, accumulator ownership/order, weight load/eviction order, exact
bank/row multiplicities, and a target SRAM macro DB.  A 7.64-billion-row
product CSV is explicitly neither required nor permitted as a substitute for
the compressed conservation ledger.

## Strict claim boundary

M40a admits only the exact ten-sample source trace, the 40/40 two-code result,
the 92,160,000-value bit miter, four M35 UQ0.24 mappings, padding-aware structured
product counts, the 32x representation identity, exact bias-free algebra and
the small executable scheduler oracle.  Real integer Conv/M35 output
equivalence, real Local/Motion cycle mean/p95/p99, SRAM macro service, physical
address schedule, integrated RTL/VCS/DC/STA/Formality, system speedup, PPA,
power, energy, external comparison, DATE headline and best-paper claims remain
false.
