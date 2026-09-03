# M2047/M2046 ep34 TSBG G48 S4 VCS independent hammer

**PASS, 97/100; P0/P1/P2 = 0/0/4.** The result admits a narrow
four-token component-cycle statement. It does not admit full-FC1, full-capture,
same-area, real-weight, macro-inclusive, power, energy, or system claims.

## What independently closes

- Both the canonical result and the single consumed-attempt directory pass
  their inner manifests and outer seals. The attempt receipt records one
  license query, one VCS compile, four simulations, and no retry. There are
  exactly four simulation logs and exactly one
  `PASS_M2046_EP34_TSBG_G48_CYCLE` line in each.
- The capture's sealed `sample_order.json` independently confirms that global
  samples 0, 10, 20, and 30 are the first sampled entries of four distinct
  sequences. Layer 28 is the first 48-source-group FC1. The fixture was decoded
  again in memory from the sealed `fc_frames.bin`; it is byte-identical to the
  frozen 768-word fixture. The builder never evaluates performance.
- The two DUTs are elaborations of the exact same M2018 source at B4/G48/LRU4,
  with the same M803 adapter, public ports, cache capacity, memory models, and
  wall-clock ready functions. `SCHEDULE_MODE` affects only the static bitmap
  ordering and the corresponding selected-row clear index: token-major for the
  baseline and group-major for TSBG.
- Both modes are checked against the same independently constructed exact
  accumulator scoreboard. Rows, issues, signed products, 24 commits and four
  terminal commits are conserved. The candidate covers independent bank
  backpressure, reordered responses, bridge/commit stalls, negative unit
  sources, a retired legal-identity replay, a bogus stale response, two reset
  recoveries and a complete post-reset legal service. The directed recovery
  includes the `-(-128)=+128` corner; both commit scoreboards must remain exact.

## Independently recomputed result

| slot / sample | ordinary cycles | TSBG cycles | speedup | ordinary bundles | TSBG bundles |
|---|---:|---:|---:|---:|---:|
| 0 / 0 | 20,292 | 7,569 | 2.680935x | 1,788 | 576 |
| 1 / 10 | 21,706 | 7,595 | 2.857933x | 1,908 | 564 |
| 2 / 20 | 23,898 | 8,023 | 2.978686x | 2,088 | 576 |
| 3 / 30 | 20,817 | 7,588 | 2.743411x | 1,836 | 576 |
| total | 86,713 | 30,775 | **2.817644x** | 7,620 | 2,292 |

The weighted result is 64.51% fewer component execution cycles. Bundle reads
fall 69.9213%, and eight scalar banks per bundle give 60,960 versus 18,336
scalar requests, the same 69.9213% reduction. These are external weight-bank
interface requests in this component environment, not DRAM bytes or energy.

## Legal relationship to M2030

M2047 and M2030 pin the same M2018 SHA
`96fb355750d50a2f1944f9d27123eef1fc70525a8146b08856884fe09c4bec21`
and the same M803 SHA
`cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156`.
Therefore the two results may be placed side by side: M2047 supplies the
four-token exact VCS cycle measurement, while M2030 supplies the matched
logic-only physical schedule ablation. At 3 ns, M2030 reports 249,710.451846
versus 249,739.809848 um2, only 0.0117568% TSBG schedule-area overhead, and both
setup slacks are positive.

They must not be multiplied or relabeled `same area`. M2030 uses ideal clock,
ZeroWireload and standard-cell state arrays; macros and power are absent, and
both hold diagnostics are -0.0164 ns.

## Residual P2 boundaries

1. The measured ep34 bundles contain no negative codes. Signed unit-source and
   INT8 `-128` behavior are covered in a separate directed recovery phase, not
   in the measured real-activity interval.
2. Four tokens from one FC1 layer and four first samples are a microbenchmark,
   not full FC1, the full 40-sample capture, or a system workload.
3. M2047 pins the fixture but not the builder, sealed sample order, or layer
   inventory. This review pins and rechecks them; a successor should reverse-pin
   them directly.
4. M2030 remains pre-macro logic-only and not hold-closed or power-characterized.

## Permitted paper wording

> On a fixed four-sequence, first-sample, layer-28 four-token G48
> microbenchmark using real ep34 activity masks, exact VCS execution of the same
> B4/LRU4 parametric RTL reduced component cycles from 86,713 to 30,775
> (2.818x; 64.51% fewer) and scalar weight-bank requests from 60,960 to 18,336
> (69.92% fewer) by changing token-major scheduling to source-group-major
> weight-row reuse. Separately, matched 3 ns TSMC-28 logic-only DC of the exact
> same RTL measured 0.0118% schedule-area overhead, with setup met on both
> modes; macros, hold closure, power and energy are excluded.

Do not shorten this to “FC1 speedup” or “system speedup,” and do not call the
selected real descriptor interval bipolar or real-weight execution.
