# M39-r2 threshold-carry integration evidence specification

Date: 2026-08-22

## Status and version boundary

M39-r2 is `BLOCKED_BY_STALE_M38_R2_EXPLORATORY_ONLY_REANCHOR_REQUIRED_AFTER_M38_R3`.
It is an evidence-gate repair and a preserved numerical DSE draft, not a current recursive
admission, final freeze, measured speedup, or headline artifact.  The hashed M38-r2 result is
auditable, but a fresh fail-closed M38-r2 build currently stops at
`receipt live source drift for unified_core_rtl`.  M39-r2 therefore records the M38-r2 numbers
without promoting them.  M31-r4/M37-r8 must first close into M38-r3, after which M39 is to be
mechanically reanchored.

M39-r1 remains unchanged and is classified
`NO_GO_DRAFT_SUPERSEDED_DO_NOT_CITE`.  It accepted insufficient VCS log evidence and anchored
superseded M38/M35 artifacts.

## Frozen artifacts

- Contract: `contracts/m39_remaining_bottleneck_input_contract_r2_20260822.json`, SHA-256
  `c52ebd87995e3e8ab7d20459b344dbf4e3cb33545340bc90f797f5e29a870894`.
- Analyzer: `system_simulator/scripts/analyze_m39_remaining_bottleneck_r2.py`, SHA-256
  `6c5efc9a7e5b74fbfe637c6952499bcd096d7c11884e680b968cb58d4790319d`.
- Result: `results/m39_remaining_bottleneck_r2_20260822/m39_remaining_bottleneck.json`,
  SHA-256 `94937be87fbbb6fe9ebf7b495b49236681f5955f12a9f6c92c631efba13abfb4`.
- M33 final receipt: `contracts/m33_output_receipt_r1_20260822.json`, SHA-256
  `9d670a6e950c3d0a1d934004901b9380a021b6d2375d3c96cc139bac96aa766e`.
- M35 final input receipt for this milestone remains immutable r2, SHA-256
  `63b61a88213e3882a0ad3a67c3e74047291c920d80a237bcdd44a8e84dcb5d5e`.

## Recursive evidence gate

M33 flat-r2 is admitted only as a standalone Synopsys result:

- VCS input/output manifests are recursively verified as 6/6 and 3/3 files.  The source has
  eight assertions and four covers; all four cover matches are nonzero.  The run reports 2,048
  packets, 4,608 valid scalar products, and 8,192 reconstruction checks.
- DC live/sealed ledgers verify 31/31 and 33/33 files.  The zero-wire ideal-clock 2.000 ns result
  is 12,997.403898 um2 with setup/hold slack 0.0006/0.0107 ns.
- The self-contained Formality snapshot verifies 21/21 files and 655/0/0
  passing/failing/unmatched compare points.

M35 r7 is likewise standalone only:

- VCS r6 input/output manifests verify 7/7 and 3/3 files.  The source has nine assertions and
  four covers, all with nonzero matches.  The run reports 5,120 packets, 23,680 valid products,
  II=1, and eight results per accepted packet.
- DC r7 live/sealed ledgers verify 30/30 and 32/32 files.  The zero-wire ideal-clock 2.000 ns
  result is 19,633.571938 um2, zero integer multipliers, and setup/hold slack 0.0000/0.0102 ns.
- The self-contained Formality snapshot verifies 20/20 files and 2333/0/0 compare points.  The
  live Formality wrapper has drift and is explicitly ignored; the sealed snapshot is the sole
  authority.

The analyzer rejects a forged one-line M33 or M35 `sim.log` even when its receipt, output
manifest, and M39 outer contract hashes are consistently rewritten.  It also rejects contract
top-level population, identity, and claim-boundary drift.

## Conserved exploratory DSE

The fixed compute baseline remains 620,868,243 cycles.  The selected M30 candidate is
`dual256b_independent_output_packed24`: Local 305,047,198 and Motion 303,376,924 cycles.  The
384-bit alternative is 24 cycles slower on each line.  Replacing the 73,183,500-cycle T10 bucket
with the M38 conditional 36,591,750-cycle, II=5 ledger gives the unadmitted ideals Local
268,455,448 and Motion 266,785,174 cycles.

Four-bottleneck alternatives, with zero overlap credit:

| Line | Sidecar | Event | Late | Control | Replacement | Conditional total |
|---|---:|---:|---:|---:|---:|---:|
| Local | M33 | 13,282,495 | 2,304,000 | 1,484,515 | 17,071,010 | 205,895,501 |
| Local | M35 | 13,282,495 | 1,152,000 | 1,484,515 | 15,919,010 | 204,743,501 |
| Motion | M33 | 12,836,419 | 2,304,000 | 1,524,011 | 16,664,430 | 203,818,647 |
| Motion | M35 | 12,836,419 | 1,152,000 | 1,524,011 | 15,512,430 | 202,666,647 |

Ten-consumer alternatives, also with zero overlap credit:

| Line | Sidecar | Event | Late | Control | Replacement | Conditional total |
|---|---:|---:|---:|---:|---:|---:|
| Local | M33 | 17,662,220 | 7,614,000 | 1,974,013 | 27,250,233 | 189,817,484 |
| Local | M35 | 17,662,220 | 3,807,000 | 1,974,013 | 23,443,233 | 186,010,484 |
| Motion | M33 | 17,069,055 | 7,614,000 | 2,026,532 | 26,709,587 | 187,606,564 |
| Motion | M35 | 17,069,055 | 3,807,000 | 2,026,532 | 22,902,587 | 183,799,564 |

These rows are alternatives, never additive.  M38 changes only the T10 ATLIF bucket, while the
consumer replacement is drawn from the M7/M25 noneligible bucket.  No sidecar overlap or double
subtraction is credited.

The 2.7x gate is evaluated with `Fraction(27,10)`, never floating point.  Its exact common cycle
ceiling is 2,069,560,810/9.  Exact maximum ten-consumer replacement is 606,455,551/9 cycles for
Local and 621,488,017/9 for Motion.  The exact 3x ceiling is 206,956,081 cycles; its Local/Motion
replacement ceilings are 44,388,830 and 46,059,104 cycles.

## External-adapter boundary and next gates

Prosperity real-domain statements are limited to the exact M32 evidence.  Fixed-point behavior,
accuracy, product density, subset forest, detector schedule, and physical traffic are unadmitted.
The official repository HEAD observed by `git ls-remote` on 2026-08-22 is
`6ee1c6f1cb419fcf942f2eda63db84ca28248f4b`; no repository file SHA is claimed.  Phi-like RTL,
calibration, accuracy, and traffic remain unimplemented and unadmitted.

Future admission requires an integrated one-pool VCS/DC/STA/Formality top, address-timed memory,
and accuracy closure.  The energy gate is explicitly future-to-future:
`FUTURE_INTEGRATED_SAME_TRACE_PTPX_PLUS_MACRO_ENERGY_NOT_WORSE_THAN_FUTURE_M38_BASELINE`.
Local5 ep44 has missing attention coverage and is `UNKNOWN NONZERO`; neither Local5 full-system
cycles nor any system speedup is admitted.
