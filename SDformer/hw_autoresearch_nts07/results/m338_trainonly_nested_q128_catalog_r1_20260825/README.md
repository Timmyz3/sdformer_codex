# M338 train-only nested q128 exact-work catalog

M338 preserves every admitted M77 q16 entry and deterministically appends
train-observed patterns to nested q32, q64 and q128 prefixes. All 128 packed
mask payloads and all 128 compressed value payloads are rehashed before use.

The calibration exact signed vector-operation speedups over bit-sparse work are
1.541232x, 1.694615x, 1.863533x and 2.058328x for q16, q32, q64 and q128.
These are train-only work observations. They are not runtime-trace results,
cycles, energy, system speedup, PPA or a DATE headline. Exact runtime replay and
selective PWP cache/DMA accounting are deferred to M339.

