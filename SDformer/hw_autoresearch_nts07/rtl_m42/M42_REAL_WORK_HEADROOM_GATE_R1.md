# M42 real-work performance headroom gate r1

M42 reconciles the M39 conditional Local/M35 cycle model with the
independently reviewed M40 ten-sample product-count workload.  It is a target
contract, not an executable scheduler or a speedup result.

The frozen resource model has 620,868,243 fixed-reference compute cycles,
188,824,491 cycles outside the four bottleneck convolutions, and 2,636,515
late-scale plus front-end cycles inside the replacement.  M40 contributes a
Local mean of 74,112,377.6 and p95 of 74,995,872 product-count-div-96 work
quanta.  These work quanta are deliberately not called executable cycles.

The frozen event engine is P8-L96: at most eight conflict-free sources are
issued per cycle and each source is broadcast to 96 output lanes.  Its peak is
therefore 768 product additions per cycle.  The factors below are required
effective source-issue widths, not reductions in the number of logical
products.

The resulting gates are:

| Target | Maximum product-engine cycles | Required effective issue width from Local mean | Required effective issue width from Local p95 |
| --- | ---: | ---: | ---: |
| 2.5x | 56,886,291.2 | 1.3028x | 1.3183x |
| 2.7x | 38,490,195.1 | 1.9255x | 1.9484x |
| 3.0x | 15,495,075 | 4.7830x | 4.8400x |

The single-source diagnostic sensitivity is 2.3378x at the Local mean and
2.3301x at Local p95.  Pure Motion remains worse than Local by 1.4830822969x
in the reviewed cohort and is not a headline path.

No target is admitted until an exact-weight, physical-address, finite-bank
scheduler reports per-sample executable cycles, conflicts, stalls, memory
traffic, integer output equivalence, and same-resource baseline cycles.
