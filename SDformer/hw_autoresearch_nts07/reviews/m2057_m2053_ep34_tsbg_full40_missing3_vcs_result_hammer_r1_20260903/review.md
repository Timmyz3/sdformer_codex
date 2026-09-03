# M2057 / M2053 ep34 TSBG full40 missing-3 VCS independent result hammer

## Verdict

**PASS, 98/100; P0/P1/P2 = 0/0/2.** The M2057 canonical result is ready for an ISCAS component table and a carefully scoped abstract statement. It is not a system or full-FC result.

The unusual evidence lineage is legitimate because it is explicit and independently checkable: 1,917 canonical logs are byte-identical to the valid raw outputs of failed attempt M2053, while slots 86, 893, and 1755 are byte-identical to the three M2057 successor runs made with `-no_save` and the exact same compiled `simv`. M2053 remains `FAILED_OR_INCOMPLETE_DO_NOT_CITE`; only the double-sealed M2057 composite is citable, and only as a **1917+3 same-image cross-attempt population**.

## Independent result

| Metric | Independently recomputed value |
|---|---:|
| Fixed VCS component workloads | 1,920 |
| Nonempty / empty retained | 1,634 / 286 |
| Improved / equal / slower | 1,343 / 570 / 7 |
| Ordinary-LRU4 post-load cycles | 12,522,876 |
| TSBG-B4 post-load cycles | 5,124,365 |
| Weighted cycle speedup | **2.443790792×** |
| Time reduction | **59.0799669%** |
| Scalar weight requests | 8,774,304 → 3,136,608 |
| Scalar-request reduction | **64.2523441%** |
| Geomean workload speedup | 1.765376568× |
| Conventional median | 1.837302877× |
| Min / max workload speedup | 0.993527508× / 3.245787909× |

The worst workload is slot 1694 (`zurich_city_12_a`, sample 35, FC1 layer 12, last token region, token 47996, 12 source groups): 307 → 309 cycles, or 0.993527508×. Therefore “every workload improves” is prohibited.

The conventional even-sample median in `result.json` averages the two center values. The independently reported nearest-rank p50 is 1.837013875×; this is a definition difference, not a discrepancy.

## Distribution

### Nearest-rank workload speedup percentiles

| Population | p0 | p10 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All 1,920 | 0.9935 | 1.0000 | 1.0000 | 1.8370 | 2.8602 | 3.0894 | 3.1156 | 3.1615 | 3.2458 |
| Nonempty 1,634 | 0.9935 | 1.0000 | 1.2643 | 2.2563 | 2.9663 | 3.0956 | 3.1221 | 3.1651 | 3.2458 |

### Recomputed stratification

| Dimension | Group | Workloads | Empty | Base cycles | TSBG cycles | Speedup |
|---|---|---:|---:|---:|---:|---:|
| Sequence | interlaken_01_a | 480 | 83 | 3,120,326 | 1,225,652 | 2.5458× |
| Sequence | thun_01_b | 480 | 102 | 2,921,236 | 1,182,417 | 2.4706× |
| Sequence | zurich_city_09_a | 480 | 50 | 3,204,120 | 1,345,486 | 2.3814× |
| Sequence | zurich_city_12_a | 480 | 51 | 3,277,194 | 1,370,810 | 2.3907× |
| Target | FC1 | 1,440 | 125 | 10,731,228 | 4,213,970 | 2.5466× |
| Target | FC2 | 480 | 161 | 1,791,648 | 910,395 | 1.9680× |
| Token | first | 640 | 133 | 4,314,614 | 1,722,660 | 2.5046× |
| Token | middle | 640 | 67 | 3,718,962 | 1,598,910 | 2.3259× |
| Token | last | 640 | 86 | 4,489,300 | 1,802,795 | 2.4902× |
| Source groups | 6 | 240 | 32 | 383,697 | 171,837 | 2.2329× |
| Source groups | 12 | 240 | 67 | 402,730 | 188,454 | 2.1370× |
| Source groups | 24 | 960 | 99 | 6,987,577 | 2,682,531 | 2.6048× |
| Source groups | 48 | 480 | 88 | 4,748,872 | 2,081,543 | 2.2814× |

All strata reconcile exactly to the row population and `result.json`.

## Integrity, SVA, and attack audit

- Canonical census: 1,927 top-level files, 1,925 manifest members, and exactly 1,920 unique `sim_slot*.log` files. Inner and outer seals pass.
- The frozen JSON, descriptor `memh`, and stats `memh` hashes agree. The upstream M1707 capture inner and outer seals pass.
- Every log contains exactly one `PASS_M2051_EP34_TSBG_FULL40_CYCLE`; no fatal/error pattern or provenance, preload, fixture, row, aggregate, breakdown, attack, or recovery mismatch was found.
- Each workload reports 24 commits, one stale attack, zero replay accepts, two resets, and one recovery. Retired replay is exactly one for nonempty workloads and zero for empty workloads.
- All mandatory candidate covers are nonzero in all 1,920 logs. Per-log min–max values are: bridge-negative 6–6, stale-attack 2–4, reset-recovery 3–6, bank-response-reorder 12–588, bridge-stall 5–221, commit-stall 3–4, terminal 8–8, and weight-bundle 200–26,827.
- Optional independent-backpressure/cache-eviction covers can naturally be zero in low-pressure subsets. Base-side attack covers are intentionally zero because attacks target the candidate; neither condition violates the exact PASS contract.

## ISCAS admission

A legal table/abstract sentence is:

> Across 1,920 fixed ep34 component workloads (40 samples, four DSEC sequences, all 12 FC1 and four G48-supported FC2 layers, first/middle/last B4 quartets), TSBG-B4 reduces post-load VCS cycles from 12.523M to 5.124M (2.444×; 59.08%) and scalar weight requests from 8.774M to 3.137M (64.25%).

The same table or nearby method text must state:

1. Component VCS post-load cycles; the common 383-cycle descriptor preload per workload is excluded.
2. Real ep34 activity/sign descriptors with deterministic directed INT8 verification weights.
3. All 12 FC1 layers but only four FC2 layers with at most 48 source groups; not the full token, FC, network, or system population.
4. Empty workloads are retained, and 7/1,920 workloads are slightly slower (worst 0.9935×).
5. The result is a double-sealed 1,917 inherited + 3 successor same-`simv` cross-attempt population; M2053 remains failed.

M2030 uses the exact same M2018 and M803 RTL hashes, so its matched logic-only area-overhead result may be placed beside M2057 as a separate physical axis. The VCS cycle result must not be multiplied by, merged with, or relabeled as same-area silicon performance.

## P2 findings

- **P2-1:** `result.json` says preload is excluded but omits the measured constant 383-cycle value. Put it in the paper table note or methodology.
- **P2-2:** `result.json` says `real_weights=false` but does not spell out that the weight values are deterministic directed INT8 verification data. Use that exact characterization in the paper.

## Prohibited upgrades

- M2053 passed, one successful attempt, or automatic retry.
- Every workload improves.
- All FC2, full FC, full token, decoder, full-network/system, FPS, or end-to-end speedup.
- Captured/real weights or natural bipolar descriptor coverage.
- Same-area speedup, macro-inclusive PPA, hold closure, power, or energy.
- Multiplication with M2030, C1, C3, Prosperity, Phi, or any system factor.

No EDA, GPU job, license query, result mutation, predecessor mutation, paper edit, or `docs/359` edit was performed by this review.
