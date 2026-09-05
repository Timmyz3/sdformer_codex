# Consumer-aware co-fill: measured incremental claim

The TCAS-II hook is **equal bytes need not mean equal service time**. Once
ordinary sector-valid caching and group-major TSBG have removed repeated bank
reads, a sparse consumer can still leave a partially filled row. Its next
consumer pays another six-slice request/response round. All four masks are
already available: unioning them before the first read coalesces those refill
rounds, while each signed product and Acc24 context remains private.

This is an extension of TSBG, not a third accelerator or a newly discovered
sparsity law. Prior mechanisms are sector-valid caches, sparse broadcast and
request coalescing (FireFly-T/ELSA/SSR and ordinary cache design). No `first`
claim is needed; the claim is a specific circuit/transaction tradeoff on this
typed, backpressured, eight-bank interface.

## What actually survives the stronger control

CPU replay of 4,320 independently reset G48 chunks:

| Axis | Cycles | Accepted scalar bank reads | Refill beats |
| --- | ---: | ---: | ---: |
| token-major, demand | 14,508,203 | 2,623,644 | 1,657,104 |
| group-major, demand | 10,545,945 | 1,604,430 | 1,165,050 |
| group-major, union co-fill | 8,052,073 | 1,604,430 | 743,718 |

The **incremental** modeled result is 23.65% less execution time (1.3097x)
and 36.16% fewer refill beats, at identical bank-read counts. The 38.85% read
reduction versus token-major includes existing TSBG locality and must not be
attributed to union co-fill alone. Six original VCS pilot axes calibrate the
model exactly; a third group-demand RTL axis is prepared, not yet run.
An independent reviewer reproduced all table totals without calling the
production cycle-model function, plus all eight warm/directive rounds.

The 1/2/4/8-cycle uniform-response sensitivity gives 1.232/1.289/1.351/1.423x
incremental ratios. This changes only the response-delay assumption and keeps
the same readiness and output backpressure; it is not extra RTL evidence.
There are 21 slower chunks, 1,867 ties, and 2,432 wins. The worst is 210 to 214
cycles. No additional reads are prefetched over a complete B4 group, but the
first consumer can wait longer for the union.

## Circuit cost still to measure

- Four 16-bit bank-valid sets, union OR and missing-bank selection.
- Actual M803 adapter, cache data and independent signed accumulation retained.
- Full-row identity must remain immutable until reset; this is not a globally
  coherent cache or cross-layer persistence mechanism.
- Compare all three modes with the same clock gating, SDC, cache and ports.
- M2248 power belongs to the original M2018 full-row design, not this variant.

## Manuscript decision

Keep C1 + C2/TSBG as the two contribution bullets. If mapped cost is acceptable,
replace repetitive C2 implementation prose with the co-fill equation
`missing = union(consumers) & ~valid` and the three-axis table. Report requests,
refill transactions, execution cycles and energy separately. Do not multiply
1.310x by old 1.8345x or mix the 4,320-chunk model with the 2,880 fixed-region
VCS population. Until its physical test completes, this note is candidate
evidence outside the submitted brief, not a Strong Accept certificate.

## Direct-prior check (primary sources, September 5)

- [OuterSPACE silicon follow-up, JSSC 2020, Sections III-B/C](https://www.bsg.ai/papers/JSSC_OuterSpace_Park_2020.pdf)
  coalesces identical addresses at the crossbar and cache, records requesters
  in bit vectors, and multicasts returned data. This is direct prior for
  coalescing/broadcast; its measured silicon ratios are not our baselines.
- [NetSparse, MICRO 2025, Sections 5–6](https://doi.org/10.1145/3725843.3756076)
  distinguishes duplicate-request removal from concatenating different fine
  requests to amortize transaction headers. Its setting is a distributed
  cluster, not an on-chip SRAM frontend. The useful analogy is the separation
  of saved data bytes from saved transaction overhead, not importing its
  128-node performance ratio.

Our constrained object is four already-present 16-bank masks. A bitwise union
co-fills distinct needed banks without discovering request matches in a new
associative structure. The M803 response protocol is unchanged. This is a
testable specialization of known coalescing ideas, not a priority claim.
