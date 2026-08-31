# M178 independent hammer review

Status: **PASS as a negative performance result; 91/100.**

The independent audit did not import the M172, M176 or M178 analyzers. It
rescanned all 120 frozen H67 FC2 payloads with a different chunk size, checked
each payload SHA/size/popcount, rebuilt the 96-bit bank/group ledgers, and
compared explicit-EOT and directory-fused cycles for every token.

Recomputed results:

- 24,449,376 explicit-stream descriptors become 18,869,376 nonzero
  descriptors: a 22.8226683577% serialized descriptor-count reduction.
- K1 is 424,060,394 cycles for both encodings.
- K4 is 144,146,504 cycles for both encodings.
- K1/K4 is 2.9418708205x.
- Aggregate and all four stages have exactly 1.0x explicit-EOT/directory-fused
  K1 and K4 ratios; per-token mismatches are zero.
- Forty-four directed recurrence attacks cover header-only length zero,
  explicit zero EOT, single and long replay, multi-descriptor sequences and
  compaction gaps. All matched. A length-zero token is two cycles under the
  declared recurrence.

There is no latency gain. The two-cycle empty-token result remains a conditional
protocol assumption because directory/header transfer, finite ports and RTL are
not implemented. Likewise, removing 5,580,000 EOT entries does not yet prove a
total memory-bit or energy reduction: the replacement directory also has
5,580,000 start/count/tag/output-block entries whose widths and accesses have
not been charged.

Findings: no P0; two P1 boundary items (unmodeled directory cost/ports and the
header-availability assumption); two P2 documentation/population items. M178
must remain `NO-LATENCY-GAIN`, `headline=false`, and only a future memory/energy
ablation option. The performance mainline should proceed to finite weight-bank
response and accumulator-context overlap.

`docs/359` remains unchanged at
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
