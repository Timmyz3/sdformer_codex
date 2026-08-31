# M143r2 independent hammer review

Verdict: **82/100, conditional accept as a heldout module-cycle DSE only**. P0=0, P1=3, P2=5. M143r1 is superseded by the M142 combinational-loop fix; this review pins and audits M143r2.

## What independently passed

The review rebuilt all 20 heldout records in record/window/partition/row/block order and obtained exactly:

- 25,920,000 raw 128-bit rows;
- 14,078,105 all-zero rows (54.3137%);
- 99,847,888 canonical K4 descriptors;
- 113,925,993 producer cycles, with the exact identity `99,847,888 + 14,078,105`;
- 188,148,490 source events and 119,447,791 PWP512 tokens.

The row formula is independently checked as

`sum_rows(max(1, sum_blocks(ceil(popcount(mask16) / 4))))`.

The independent recurrence reproduces the full-materialized reference B2/B3/B4 values `188,168,131 / 144,690,917 / 133,991,596`, the raw128 values `193,972,510 / 147,637,375 / 135,461,009`, and the reported B4 ratios `2.594690240x` versus compact256 and `1.812225612x` versus dualrow512. A clean production rerun is byte-identical at SHA256 `8b5821d747e653ac9053a4cfe94fe9eb40c78ce0eaaca4c9af4fdf8073b5bd19`.

Bank-lifetime attacks pass for B2/B3/B4: no PWP starts before full fill, no bank refills at/before release, no refill precedes prior PWP completion, peak occupancy stays within the bound, and no PWP/correction service crosses any of the 160 modeled outer barriers. Producer lookahead is bounded by 2/3/4 descriptors respectively.

## Blocking gaps before RTL-cycle admission

1. **P1 — zero-work endpoint mismatch.** The Python recurrence releases 300 zero-correction units without a correction completion and models 1,332 zero-PWP units at zero duration. The contract says release only after matching correction completion, and M142 always requires the endpoint state transitions. A one-cycle endpoint-floor sensitivity changes final B4 by only 18 cycles, but that sensitivity is not an RTL proof.

2. **P1 — outer barrier missing from M142.** The model inserts 160 flush/commit barriers. M142 has no outer-boundary or commit-done interface, so exact behavior requires an unstated external ready-gating wrapper.

3. **P1 — sequence identity not closed.** The model assumes 69,120 ordered units. M142 does not enforce row count/order, and completion echoes only bank plus a default 16-bit tag, not the internal 32-bit sequence. The frozen extent exceeds the 65,536-value tag space.

Secondary findings are: Python ring allocation versus RTL lowest-free allocation (no heldout cycle impact in the independent sensitivity), B2 is model-only although listed with B3/B4, negate masks are not reconstructed, top-level `exact_work` omits correction tokens, and production imports its recurrence from a prior review artifact.

## Claim boundary

The two ratios are trustworthy only as frozen-heldout, same-clock, module-cycle-model comparisons. They are not M142 RTL throughput, matched-frequency, physical, full-network, system, or paper-headline speedups. SRAM capacity/energy and B4 physical cost remain unpriced.

## Reproduction

Run with the Python environment that provides NumPy:

```bash
/opt/anaconda3/bin/python \
  results/m143_independent_hammer_review_r1_20260824/independent_recompute_m143.py \
  --output /tmp/m143_independent_recompute.json
```

The audit script refuses to overwrite an existing output. See `independent_recompute_and_attack.json` for exact schedule counters, attacks, stream digests, and sensitivity points; see `review_score_and_findings.json` for the scored P0/P1/P2 disposition.
