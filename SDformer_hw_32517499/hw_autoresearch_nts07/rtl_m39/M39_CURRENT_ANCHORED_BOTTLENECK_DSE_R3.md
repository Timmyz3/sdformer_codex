# M39 current-anchored bottleneck DSE, revision 3

Date: 2026-08-22

Status: `PASS_M39_R3_CURRENT_ANCHORS_CONDITIONAL_BOTTLENECK_DSE_ONLY`.

M39-r3 is a current evidence-chain and analytical-DSE milestone. It admits the
four frozen H67 Conv3x3 source rows, exact work counts, analytical resource and
traffic lower bounds, and the conditional-DSE arithmetic. It does not admit an
implemented bottleneck schedule, Local/Motion full-system cycles, system
speedup, integrated RTL or Synopsys results, physical memory, PPA, energy,
trained coverage, external-accelerator comparison, or a paper headline.

## 1. Exact artifacts

- contract: `hw_autoresearch_nts07/contracts/m39_remaining_bottleneck_input_contract_r3_20260822.json`, SHA-256 `bf58fbbc852f10a6f7695585ecbb2cc01e14ed06f59e2ec4a38f912e33ebc5e2`;
- analyzer: `hw_autoresearch_nts07/system_simulator/scripts/analyze_m39_remaining_bottleneck_r3.py`, SHA-256 `9bbe19beecb55b1b3495352081309370d8909ad1001aa05efea147f7e645b470`;
- regression: `hw_autoresearch_nts07/system_simulator/tests/test_m39_remaining_bottleneck_r3.py`, SHA-256 `53ec41f5a1babcf6bbe41b6fb4091a2fb53be48fcd1190f3efc0ddda9c867c34`;
- result: `hw_autoresearch_nts07/results/m39_remaining_bottleneck_r3_20260822/m39_remaining_bottleneck.json`, SHA-256 `8923bbf5b1e630ad8e940ffa967f18ae9e59176c3f2dd6b29af2c1d696fbdcbb`.

Two clean builds produced the same result bytes and SHA-256.

## 2. Current recursive anchors

The sole M38 authority is the independently admitted M38-r5 model-only chain:

- model-only admission SHA-256 `2d231c4a88d616158bcac0e867ec166a109fe8df55f10fc81182fc8ec01f08fe`;
- independent GO review SHA-256 `36bb10294a209bd32ad4131d8b0171749aa50535083166dc38b5de5b28d2d529`;
- independent validator SHA-256 `ce34da7dd759c0b43efc147a9b8f22f700414e17f7a8a9f1a3336c4afb64b445`.

M39-r3 imports and executes that validator, requires exact type-strict equality
with its rebuilt admission and review payloads, and preserves every forbidden
M38 scope. It does not directly anchor an older M38 contract or result.

The late-scale authority is M35 output receipt r3, SHA-256
`d088daa8e51a40eb26ee07624f2c6a3b06f95bd0d1395c4bb91bdd1532195b84`.
M39 recursively verifies its M35-r2 predecessor and M33 final receipt, including
VCS, DC/STA, and sealed Formality evidence. The strict flat comparison is only
at common 28 nm, 2.000 ns, zero-wire ideal-clock qualification: M33 provides
four results/cycle at 12,997.403898 um2 and M35 eight results/cycle at
19,633.571938 um2, or exactly the receipt's 1.3239978888x standalone result
rate per area. Integrated density remains false. The receipt itself records
that an independent r3 review is required; this M39 artifact does not erase
that boundary.

## 3. Why M39-r2 is superseded

M39-r2 remains unchanged and is classified
`M39_R2_STALE_ANCHOR_AND_UNTYPED_DSE_SUPERSEDED_DO_NOT_CITE`. R3 rehashes its
contract, analyzer, regression, result, and specification.

The audit found four material deficiencies:

- r2 used stale M38-r2 and M35-r2 anchors;
- its JSON loading accepted duplicate keys and Python-equal boolean/integer or
  integer/float substitutions;
- its aggregate Local/Motion event-cycle values were hard-coded rather than
  derived from the four real operator rows;
- it did not expose the work reduction, bank bandwidth, SRAM capacity, or
  lane/bank/density sensitivity required by that projection.

R3 uses duplicate-key-safe and non-standard-number-rejecting JSON loading. The
complete contract and rebuilt anchor payloads use recursive type-strict
equality. Every input and superseded artifact is SHA-bound.

## 4. Four Conv3x3 bottleneck work ledger

All four operators have T10/C768/H15/W20 input and output, 3x3 kernels, and 768
output channels. Their exact frozen work is:

| Operator | Active product terms | 96-lane activity cycles | Local conditional event projection | Motion conditional event projection | M35 CSD terms |
|---|---:|---:|---:|---:|---:|
| resblocks.0.conv1.0 | 2,630,357,176 | 27,399,554 | 4,570,264 | 4,416,777 | 3 |
| resblocks.0.conv2.0 | 947,018,995 | 9,864,782 | 1,645,452 | 1,590,192 | 2 |
| resblocks.1.conv1.0 | 2,898,921,692 | 30,197,101 | 5,036,896 | 4,867,738 | 3 |
| resblocks.1.conv2.0 | 1,168,273,912 | 12,169,520 | 2,029,884 | 1,961,713 | 4 |

The aggregate is 63,700,992,000 dense terms, 7,644,571,775 active terms,
79,630,957 separately rounded 96-lane cycles, and 9,216,000 outputs. The exact
product density is `305782871/2548039680`, approximately 12.0007%.

The distinction between the two event numbers is essential:

- without a proven coalescing/reuse schedule, the 96-lane/24-bank lower bound
  is still 79,630,957 cycles;
- projecting the M25 effective M4 reduction onto each operator gives
  13,282,496 Local cycles or 12,836,420 Motion cycles.

Thus the conditional path needs effective work reduction greater than 5x and
close to 6x. The projection is not an executable schedule and receives no
cycle admission.

## 5. Resource and sensitivity model

The analyzer evaluates 90 exact rows: Local/Motion, observed plus four
hypothetical density points, three lane counts (48/96/192), and three bank
counts (12/24/48). Each bank contributes four bytes per read or write, derived
from the nominal 96-byte row across 24 banks. Event service width is
`min(lanes, 4*banks)`.

For every row, r3 reports separately:

- the uncoalesced event lower bound;
- the conditional M4-projected event cycles;
- the M35 late-scale compute/read lower bound;
- proportional frontend/control cycles;
- the effective work reduction required by the conditional projection.

No sensitivity row reports system speedup. Distinct operators retain their own
ceilings; r3 does not borrow the one-cycle aggregate rounding credit used by
r2.

The four weights occupy 21,233,664 bytes. After the existing 52,032-byte fixed
resident allocation, only 193,728 preferred or 365,760 hard-cap bytes remain,
requiring at least 110 or 59 weight tiles. The complete Q24 intermediate is
27,648,000 bytes and also cannot fit.

Compulsory fused traffic is at least 31,601,664 bytes. Materializing and then
reading Q24 increases that lower bound to 86,897,664 bytes. These counts exclude
partial sums, indices, metadata, replay, bank conflicts, and DRAM timing. The
no-reuse one-byte-per-active-product stream is 7,644,571,775 bytes and again
shows why the projected 6x coalescing must be implemented and proven. No
address-timed memory claim is made.

## 6. Conditional compute DSE, not system speedup

The fixed compute reference remains 620,868,243 cycles. Replacing only the
73,183,500-cycle T10 bucket with the admitted M38-r5 conditional model value
36,591,750 gives Local/Motion ideals of 268,455,448/266,785,174 cycles. This is
a model substitution, not integrated cycle evidence.

With zero overlap credit, the four-bottleneck alternatives are:

| Line | Sidecar | Event | Late | Control | Replacement | Conditional total | Conditional compute ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| Local | M33 | 13,282,496 | 2,304,000 | 1,484,515 | 17,071,011 | 205,895,502 | 3.015x |
| Local | M35 | 13,282,496 | 1,152,000 | 1,484,515 | 15,919,011 | 204,743,502 | 3.032x |
| Motion | M33 | 12,836,420 | 2,304,000 | 1,524,011 | 16,664,431 | 203,818,648 | 3.046x |
| Motion | M35 | 12,836,420 | 1,152,000 | 1,524,011 | 15,512,431 | 202,666,648 | 3.063x |

The exact rational 2.7x and 3x gates pass in these conditional rows. They do
not constitute admitted system speedup because event coalescing, shared-pool
integration, memory timing, and end-to-end execution remain absent. The legacy
ten-consumer calculation is retained only as a separately reconciled
alternative; it is never added to the four-bottleneck result.

## 7. Regression and next gate

The Python 3.6 suite passes 17/17 tests. It covers exact anchors, per-operator
work, M35 thresholds, 90-point sensitivity, SRAM/traffic lower bounds, exact
rational gates, Local/Motion/Local5 scope separation, repeat-build identity,
duplicate keys, `NaN`/`Infinity`, recursive bool/int and float/int forgeries,
claim drift, anchor rebind attempts, and overwrite refusal.

The next independent hammer should attack the M38 and M35 rebuild gates,
per-operator rounding, density scaling, bank-width assumptions, compulsory
traffic definitions, and every admission flag. After a GO, the next hardware
milestone is an executable event-coalescing schedule and fixed-point sidecar
integration, followed by integrated Synopsys VCS/DC/STA/Formality/SAIF/PTPX and
address-timed memory evidence.
