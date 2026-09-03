# M2050/M2048 ep34 TSBG multilayer/token VCS independent hammer

**PASS, 98/100; P0/P1/P2 = 0/0/4.** M2050 admits a scoped
multi-layer, multi-token component-cycle distribution claim. It does not admit
full-FC, all-FC2, real-weight, same-area, macro-inclusive, power, energy, or
system claims.

## What independently closes

- The canonical result and single consumed-attempt directory both pass their
  inner manifests and outer seals. The attempt records one license query, one
  VCS compile, 192 simulations at parallelism four, and no retry.
- Every one of the 192 simulation logs has exactly one M2050 PASS line, no
  fatal/assertion/timeout signature, and nonzero signed-bridge, stale-attack,
  and reset-recovery cover. VCS is `V-2023.12-SP1_Full64`; the compile contains
  seven modules and no compile error.
- Every entry in the sealed M1707 manifest verifies. All 192 selected
  descriptors were decoded again from `fc_frames.bin`; the reconstructed
  36,864 fixture words, 192 packed stat words, and 192 metadata rows are
  byte-for-byte identical to the frozen fixture.
- Selection is fixed without consulting performance: global samples 0, 10,
  20, and 30 are the first captured samples from four sequences; all 12 FC1
  layers and the four FC2 layers with at most 48 source groups are selected;
  each contributes first, middle, and last aligned B4 token quartets. Smaller
  layers are zero-padded onto the same G48 hardware.
- An independent model reproduces the RTL's nonblocking LRU age semantics and
  lowest-cache-row tie break for all workloads. Ordinary LRU4 has
  10,084 misses / 85 hits / 9,420 evictions; TSBG has 3,493 / 6,676 / 2,829.
  The TB fail-closes unless these packed expected counters match both RTL
  instances before PASS.
- The previous M2047 layer-28 first-quartet anchors for all four samples
  reproduce exactly in rows, issues, products, cycles, bundles, and scalar
  requests.

## Independently recomputed result

| scope | workloads | empty | ordinary cycles | TSBG cycles | speedup | scalar-request reduction |
|---|---:|---:|---:|---:|---:|---:|
| FC1 | 144 | 10 | 1,175,236 | 448,945 | 2.617773x | 67.3088% |
| supported FC2 | 48 | 9 | 206,468 | 102,398 | 2.016328x | 54.3779% |
| total | 192 | 19 | 1,381,704 | 551,343 | **2.506070x** | **65.3610%** |

The aggregate is **60.0969% less component execution time**. The ordinary and
TSBG paths issue 121,008 versus 41,916 bundles, or 968,064 versus 335,328
scalar weight-bank requests. These are external component-interface requests,
not DRAM bytes or energy.

All 19 empty workloads are retained in weighted, geometric, median, minimum,
maximum, and breakdown statistics. They contribute 513 cycles to both axes.
The independent target/layer/sequence/token-role/source-group breakdowns match
the result exactly. Sequence speedups span 2.3152x--2.6012x, token-role
speedups 2.4122x--2.6316x, source-group speedups 2.2159x--2.6459x, and
layer-aggregate speedups 1.3077x--2.9163x. One nonempty workload is marginally
slower at 0.998322x, so an every-workload improvement claim is forbidden.

## Signed and arithmetic boundary

The measured selected ep34 descriptors contain 25,045 nonzero codes and zero
negative codes. Signed negative sources and the exact `-(-128)=+128` 9-bit
corner are exercised after the measured interval in the directed recovery;
both accumulators must match the exact scoreboard. This proves the signed
hardware path, but it does not make the natural measured interval bipolar.
Hardware weights are deterministic directed INT8 values, not captured real
weights; scheduling and cycle count are weight-value independent.

## Legal relationship to M2030

M2050 and M2030 pin the same M2018 SHA
`96fb355750d50a2f1944f9d27123eef1fc70525a8146b08856884fe09c4bec21`
and M803 SHA
`cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156`.
They may be placed side by side: M2050 provides exact VCS component cycles;
M2030 separately reports 3 ns logic-only DC areas of 249,710.451846 and
249,739.809848 um2, or 0.0117568% schedule-area overhead, with setup met.

They must not be multiplied or described as same-area evidence. M2030 is
pre-macro with ideal clock, ZeroWireload, and standard-cell state arrays. Both
hold diagnostics are -0.0164 ns, and no power or energy was measured.

The quarantined M2049 attempt remains `FAILED_DO_NOT_CITE`; no M2049 metric is
used here.

## Residual P2 boundaries

1. The selection covers all FC1 but only four of twelve FC2 layers and three
   B4 quartets from the first captured sample of each sequence, not the full
   sample/token population.
2. Natural production descriptors are unipolar in this selection and weights
   are directed; signed/minus-128 correctness is a separate recovery test.
3. Aggregate performance is strong, but one workload is 0.998322x.
4. The physical companion is logic-only, not macro-inclusive, hold-closed, or
   power-characterized.

## Permitted paper wording

> Across a fixed, performance-independent 192-workload ep34 distribution
> spanning four sequences, all 12 FC1 layers and four G48-supported FC2 layers,
> and first/middle/last four-token groups, exact post-load VCS execution on the
> same physical B4/G48/LRU4 RTL reduced component cycles from 1,381,704 to
> 551,343 (2.506x; 60.10% less) and scalar weight-bank requests from 968,064 to
> 335,328 (65.36% fewer) by changing token-major scheduling to
> source-group-major reuse. The 19 empty workloads are included. Separately,
> matched 3 ns TSMC-28 logic-only DC of the exact same RTL source measured
> 0.0118% schedule-area overhead with setup met on both axes; macros, hold
> closure, power, energy, and system performance are excluded.

## Estimated ISCAS impact

M2050 directly closes the prior M2047 review request for a predeclared
cross-layer and cross-token distribution. If incorporated with the exact scope
above, the earlier ISCAS artifact-open estimate moves approximately from
4.1/5 to **4.2/5**, with Evaluation around 4.4/5 and an 80--90% Accept tendency.
It still does not independently establish Strong Accept; macro-aware,
hold-closed power/energy or a broader system closure remains the material gap.
