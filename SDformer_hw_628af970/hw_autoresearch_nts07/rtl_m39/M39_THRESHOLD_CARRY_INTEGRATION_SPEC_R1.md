# M39 threshold-carry integration and remaining-bottleneck contract, revision 1

Status: fail-closed ledger and conditional design-space milestone. This document
does not admit integrated RTL, executable system cycles, Local5 full-system
performance, accuracy, PPA, power, energy, FPS, or a headline speedup.

## 1. Correct system anchor and remaining work

M39 uses M30 r3's best `dual256b_independent_output_packed24` row, not the
24-cycle-slower 384-bit row:

| H67 mechanism | M30 cycles | M38 conditional ideal cycles | Conditional ratio versus 620,868,243 |
|---|---:|---:|---:|
| Local | 305,047,198 | 268,455,448 | 2.312742x |
| Motion | 303,376,924 | 266,785,174 | 2.327222x |

The M38 column substitutes the 73,183,500-cycle T10 bucket with the
36,591,750-cycle conditional II=5 theory bucket. M38 is not integrated RTL and
these ratios are not measured system speedups.

The remaining shared non-ATLIF ledger is:

| Component | Local cycles | Motion cycles |
|---|---:|---:|
| M4-accelerated eligible operators | 54,565,804 | 52,733,277 |
| Noneligible operators plus Q/K | 162,059,820 | 162,059,820 |
| M21 frontend/control | 6,098,531 | 6,260,784 |
| Registered bubbles | 738 | 738 |
| H67 RQTB attention anchor | 3,090,731 | 3,090,731 |
| Total | 225,815,624 | 224,145,350 |

The 162,059,820 cycles are exactly 132,987,740 noneligible operator cycles plus
29,072,080 Q/K cycles. The noneligible operator split is bottleneck Conv
79,630,957, patch embed 27,099,543, FFN expand 17,474,490, downsample 8,691,053,
and prediction 91,697. Q/K plus the H67 attention anchor is only 32,162,811
cycles. Therefore only the four bottleneck convolutions form a single remaining
bucket that can independently save at least 50 million cycles.

`Local` and `Motion` above are two mechanisms evaluated on the same frozen H67
profile100 ledger. They are not Local5 ep44. Local5 has zero attention rows in
its current ordered trace because attention is missing unknown nonzero, not
because it costs zero; at least 120 calls, full-system cycles, and speedup remain
unknown.

## 2. Scalar-carried bitplane consumer path

For an ATLIF producer output `x = theta*b`, where `b` is binary and `theta` is a
layer-resident scalar, the downstream affine operator is evaluated as:

```text
W*(theta*b) + bias = theta*(W*b) + bias
```

The proposed consumer path carries `b` and `theta` instead of materializing a
Q24 tensor, performs source-owned signed-INT8 weight accumulation into Acc32,
late-scales completed Acc32 values, then applies the original bias, RNE,
saturation, and consumer semantics. Local issues current `b` sources. Motion
may issue signed `+W` for 0-to-1 and `-W` for 1-to-0 changes only when the
previous-state key, refresh, reset, and output identity are preserved.

The four bottleneck operators are 3x3 Conv2d with per-invocation im2col shape
`M=3,000`, `K=6,912`, `N=768`; the observed input activity range is 5.95% to
18.20%. A 768-channel, 15x20 one-timestep bitplane is 28,800 bytes and a
three-row binary window buffer is 5,760 bytes. Together with the existing
52,032-byte resident footprint, one buffer needs 86,592 bytes; a double or
Motion previous/current buffer envelope needs 115,392 bytes. The existing
24-bank, 96-byte-row, 1R1W/bank organization must sustain the same 96-byte
weight issue per cycle used by the 96-lane source-owned engine.

## 3. Two late-scale alternatives

### 3.1 M33 generic shared-pool client

M33 decomposes four Acc32 values and one UQ0.24 threshold into balanced
radix-128 signed digits. Twenty products per output use 80 lanes of the sole
96-lane signed-INT8 pool and produce four outputs per cycle. It supports generic
UQ0.24 thresholds, but it competes with M38 stage 1, T2, and other clients.

Frozen standalone evidence is VCS r2 with 2,048 packets and 8,192 digit checks,
plus exploratory flat DC r2 timing MET at 2 ns and 12,997.403898 um2. The flat
top contains its own 96 multipliers, so this area is not the incremental area of
a shared-pool integration. A same-top integrated A/B and Formality are required.

### 3.2 M35 complement-CSD parallel sidecar

All ten frozen H67 thresholds have `theta_raw = 2^24 - delta`, with delta from
1 through 588 and at most four canonical signed-power terms. M35 evaluates:

```text
Acc*theta_raw = (Acc << 24) - sum(sign_k * (Acc << shift_k))
```

It uses zero integer multipliers and accepts eight outputs per unstalled cycle.
Frozen evidence is the M35 r3 ten-threshold math audit, VCS r6 II=1x8, and the
latest fair DC r7 area 19,633.571938 um2 at 2 ns with timing MET. Compared with
flat M33 r2, its standalone result throughput density is 1.323998x. The latest
r7 Formality run is pending, and the H67 threshold complement property cannot
be generalized to Local5.

M35 has no arithmetic conflict with the 96-lane pool, but M39 grants zero
overlap credit. Only an integrated design with independent ports and
accumulator buffers, plus VCS and system-scheduler evidence of simultaneous
service, may replace a serial sum by an overlapped maximum.

## 4. Conserved conditional DSE

Every M39 row uses this equation:

```text
replacement = event_accumulation + late_scale + frontend_control - overlap_credit
after = M38_ideal - scope_before + replacement
overlap_credit = 0
```

The M38 T10 bucket and the noneligible consumer bucket are disjoint. The
four-bottleneck and ten-consumer scopes are alternatives, not additive
optimizations. No row may subtract M38 T10 twice or apply the ten-consumer output
population to the four-bottleneck scope.

| Scope / implementation | Line | Before | Event | Late | Control | Replacement | Savings | Conditional after / ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 4 bottleneck / M33 | Local | 79,630,957 | 13,282,495 | 2,304,000 | 1,484,515 | 17,071,010 | 62,559,947 | 205,895,501 / 3.015453x |
| 4 bottleneck / M33 | Motion | 79,630,957 | 12,836,419 | 2,304,000 | 1,524,011 | 16,664,430 | 62,966,527 | 203,818,647 / 3.046180x |
| 4 bottleneck / M35 | Local | 79,630,957 | 13,282,495 | 1,152,000 | 1,484,515 | 15,919,010 | 63,711,947 | 204,743,501 / 3.032420x |
| 4 bottleneck / M35 | Motion | 79,630,957 | 12,836,419 | 1,152,000 | 1,524,011 | 15,512,430 | 64,118,527 | 202,666,647 / 3.063495x |
| 10 consumers / M33 | Local | 105,888,197 | 17,662,220 | 7,614,000 | 1,974,013 | 27,250,233 | 78,637,964 | 189,817,484 / 3.270870x |
| 10 consumers / M33 | Motion | 105,888,197 | 17,069,055 | 7,614,000 | 2,026,532 | 26,709,587 | 79,178,610 | 187,606,564 / 3.309416x |
| 10 consumers / M35 | Local | 105,888,197 | 17,662,220 | 3,807,000 | 1,974,013 | 23,443,233 | 82,444,964 | 186,010,484 / 3.337813x |
| 10 consumers / M35 | Motion | 105,888,197 | 17,069,055 | 3,807,000 | 2,026,532 | 22,902,587 | 82,985,610 | 183,799,564 / 3.377964x |

All table entries are conditional compute DSE, not executable or measured
cycles. The ten-consumer M33 event/control terms come directly from M32. The
four-bottleneck terms are independently recomputed from the four operators'
79,630,957 cycles, 9,216,000 outputs, the line-specific M4 speed, and the
proportional M21 control charge.

For the ten-consumer scope, 2.7x is impossible if replacement exceeds
67,383,950 cycles on Local or 69,054,224 on Motion. The 3x limits are
44,388,830 and 46,059,104. For bottleneck-only 3x, the limits are much tighter:
18,131,590 and 19,801,864 cycles. Any integrated startup, tail, contention, or
memory cost must fit inside those limits.

## 5. Prosperity and Phi adapters

Prosperity uses exact runtime subset/prefix reuse among binary activation rows.
The official simulator explores M/K tiles and uses eight popcount units; its
default is M=256, K=16, N=128. An M=256, K=16, N=96 probe matched to this
design's 96 lanes would need about 106,880 incremental bytes including a
98,304-byte Acc32 tile and Conv line buffer, or 158,912 bytes with the frozen
resident footprint, which fits 240 KiB.

This is only an RTL shape. The current evidence contains aggregate activity,
not exact binary bottleneck im2col rows, subset forests, product density,
detector latency, or metadata. `selected_binary_tile_vectors.npz` is explicitly
unavailable. Bit density must not be used as product density. A bottleneck-only
Prosperity adapter is NO-GO unless the complete replacement including search,
issue, and memory is at most 29,630,957 cycles, the threshold for saving 50
million cycles.

Phi partitions K into 16-bit vectors, uses 128 calibrated patterns, precomputes
pattern-weight products, and evaluates a signed sparse residual. Its partition
shape fits these operators, but pattern coverage, residual density, precomputed
product traffic, and Local5 calibration are unavailable. Phi changes compute
into buffer/DRAM traffic and normally uses pattern-aware fine-tuning. It is
NO-GO until calibration/test coverage, address-timed traffic, and valid825
accuracy are closed. Q/K plus attention has a perfect-elimination ceiling of
32,162,811 cycles and cannot justify a standalone >=50M-cycle adapter.

Primary sources:

- Prosperity: <https://arxiv.org/abs/2503.03379> and
  <https://github.com/dubcyfor3/Prosperity>
- Phi: <https://arxiv.org/abs/2505.10909>

## 6. Admission sequence and NO-GO gates

VCS must cover the full producer-to-consumer fixed-point miter, Conv padding and
window order, bias/RNE/saturation, Local and signed-Motion state identity,
sequence reset, long stalls, FIFO hazards, and single-pool arbitration. M35 may
claim overlap only after simultaneous-service coverage with independent ports
and buffers.

DC/STA/Formality must use a single integrated top, one and only one 96-lane
INT8 pool, identical hierarchy and 3 ns constraints, and same-top M33/M35 A/B.
Setup and hold must meet, unintended multipliers must be zero, all Formality
compare points must pass, and integrated area delta must be no more than 15%.
M35 r7 is not admitted before its pending Formality run passes.

SRAM should remain at or below 240 KiB and is a hard NO-GO above 408 KiB under
the frozen 24-bank organization. PTPX must use SAIF from the same ordered H67
and Local workloads and include SRAM macro/CACTI plus address-timed DRAM energy;
same-trace energy must not worsen versus the M38 baseline.

The primary accuracy gate is bit-exact equality to the frozen integer reference.
If re-quantization is unavoidable, the fallback gate is valid825 delta AEE at
most 0.02 under the identical evaluator. Local5 additionally requires its ep44
threshold census, attention trace, fixed-point miter, and independent valid825;
H67 evidence cannot substitute.
