# TSBG ep34 same-I/O/cache B2/B4/B8 milestone self-review

Verdict: **88/100, CPU premodel milestone passes; continue to a separate RTL/physical gate.** This is not yet a paper-admitted component result.

The 40-sample live93 replay covers 960 FC1/FC2 pairs, 11,040 binary frames and 44.64M tokens. The baseline is an ordinary persistent same-capacity LRU-B weight-row buffer, not an uncached K1 stream. Both arms use the same B-row cache, eight 16-B/cycle banks, 128-B/cycle aggregate weight service, eight-source compute service and Acc24 commit work. TSBG shares only the weight fetch; signed values and accumulators remain private.

| B | Serialized cycles, ordinary -> TSBG | Speedup | Weight bytes, ordinary -> TSBG | Reduction | Max incremental state | Min sequence speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 29,460,829,487 -> 18,485,149,286 | 1.593757x | 3,128,063,239,680 -> 1,875,549,238,272 | 40.0412% | 3,388 B | 1.585008x |
| 4 | 29,410,200,623 -> 11,607,113,795 | 2.533808x | 3,123,767,270,400 -> 1,087,141,069,824 | 65.1978% | 10,164 B | 2.517177x |
| 8 | 29,249,468,207 -> 7,512,634,972 | 3.893370x | 3,107,008,367,616 -> 615,326,513,664 | 80.1955% | 23,716 B | 3.879682x |

All three points pass the local `>=1.15x` cycle gate and the energy-branch gate. B8 is the preferred continuation because the existing M1794 source is B8; B2 is the lowest-state fallback and B4 is the middle Pareto point.

The B4/B8 access, fetch, compute and commit ledgers exactly match M1763. INT8 weight bytes and weight-service cycles are exactly one quarter of the older FP32 screening ledger. This validates the new B2 extension and the design-point normalization without reusing the old 2.904x/5.122x roofline figures as the new result.

Open boundaries are material: equal physical area is not established; candidate context state is priced but larger; INT8 is a design point rather than checkpoint quantization authority; M1827 blocked production VCS on source-governance semantics; and no DC or mapped energy exists. Accordingly, the legal paper statement remains “CPU same-I/O/cache premodel opportunity,” not “RTL speedup,” “same-area speedup,” or system acceleration.
