# TTX transferable-idea deep-read review (2026-07-11)

This note reviews the first mining report against full paper mechanisms, available
reference code, the corrected deployment constraints, and the H66 measurements.

## Frozen deployment constraints

- Neuron: one-sided binary ATLIF is allowed. A two-sided ATLIF must use symmetric
  positive and negative thresholds.
- Attention: all 12 encoder blocks must deploy one formula. `gate*K` or
  `weights@K` is allowed; the prohibited carrier is the old native
  `K*sn2_q(sum_channel(Q))` followed by another attention gate.
- DSEC is the first screen. Partial-stage and TX/SC mixed deployment are excluded.
- Accuracy is the first screen. Final evidence must still report AAE, spikes,
  attention operations, memory traffic, and standard valid825 metrics.

## Corrections to the first mining report

### Bishop (ISCA 2025)

The paper's ECP is derived for binary matrix attention `S=QK^T`. It prunes a Q or
K row when its active-bundle count makes every resulting score bounded by a
threshold. This is not directly equivalent to H60 dyadic TTX: TTX gives a small
positive reward to silent/silent channel pairs, and produces one scalar selector
per token rather than an `N x N` score matrix.

A valid TTX adaptation is therefore one of:

1. Post-Shiftmax bounded gate/value pruning. For binary K, the omitted output L1
   contribution of token `i` is exactly `gate_i * popcount(K_i)` before projection.
   Bundles of 4 or 8 tokens can be skipped when the sum is below epsilon.
2. Progressive channel evaluation. After each 8-channel TX group, use lower and
   upper score bounds to stop tokens that cannot exceed a Shiftmax relevance
   threshold. This transfers SpAtten-style progressive refinement to binary
   channel groups rather than pretending channels are quantization bits.
3. Bundle-sparsity training on the actual TTX input/output bundles, followed by
   explicit cycle and SRAM-access accounting.

Calling a plain score threshold "Bishop ECP" would be inaccurate. The paper is an
architecture and bounding template, not evidence that arbitrary TTX thresholding
is error bounded.

### EDCFlow (CVPR 2025)

EDCFlow combines high-resolution adjacent temporal feature differences with a
lower-resolution robust cost volume. Its difference path is not just a scalar
gate: it uses multi-scale difference extraction and adaptive fusion because the
difference is sharp but noisy, while correlation is robust but spatially blurred.

The clean binary adaptation is **Motion-XOR TTX**:

```text
motion_i = popcount(K[t,i] XOR K[1-t,i])
score_i  = TX(Q[t,i], K[t,i]) + 2^-2 * motion_i
output_i = Shiftmax(score)_i * K[t,i]
```

It preserves all12 H60 dataflow, uses one temporal K buffer plus XOR/popcount and
one shift-add, and does not introduce SC, a native carrier, or a cost volume. H67
pre-registers only weight `1/4`; no weight sweep is permitted before valid40.

### FlowFormer (ECCV 2022)

FlowFormer first constructs a full 4D cost volume, patchifies each cost map with
three stride-2 convolutions, and uses learned latent codewords to cross-attend to
cost patches. It then alternates intra-cost and inter-cost aggregation and uses a
dynamic cost-query decoder. It does not simply pool K inside an existing attention.

Consequently, "M=4/8 pooled latent K TTX" is only a loose inspiration. It adds
codewords, reconstruction, and a new dataflow, so it is P2 and must not be claimed
as a faithful FlowFormer transplant.

### Castling-ViT (CVPR 2023)

Castling-ViT trains a cheap linear-angular attention together with a masked
quadratic softmax branch; regularization drives the auxiliary mask to zero and the
quadratic branch is removed at inference. Its key transferable principle is
training-time capacity that disappears at deployment.

For TTX this suggests **Castling-TTX distillation**: retain H60 as the only deployed
path, add H66a full alpha-XNOR matrix only during training, and distill its output or
score structure into H60 while annealing the auxiliary coefficient to zero. This is
more defensible than deploying H66a: H66a valid40 AEE was 1.6554 and its explicit
matrix operations are missing from the current profiler. The experiment should be
run only after H67, with an inference audit proving the auxiliary branch is absent.

### SwiftFormer (ICCV 2023)

The official `EfficientAdditiveAttnetion` implementation normalizes Q and K in L2,
projects Q onto a learned global vector, normalizes token weights, forms one global
query, multiplies it elementwise with K, then applies learned projections. It is
linear in token count but is not immediately multiplier-free or spike-native.

A binary version would require a learned binary global vector, approximate norms,
global reduction, and a new projection path. It also resembles a global Q-derived K
carrier. This is P2, not a direct replacement for H60.

### SpAtten (HPCA 2021) and progressive TTX

SpAtten first computes with fetched MSBs and requests LSBs only for low-confidence
cases. TTX has binary channels, not multibit operands, so the literal MSB/LSB method
does not apply. The transferable principle is confidence-triggered refinement:
evaluate TX in fixed 8-channel groups and stop when exact remaining-channel bounds
prove irrelevance. Any proposed early stop must include a bound under centered
Shiftmax and must account for control divergence.

### MEET (CVPR 2025), DeltaCNN, and exact Delta-TTX

MEET's central warning is important for DATE: temporal delta execution can reduce
dynamic computations yet consume more energy when the required activation states
spill beyond on-chip SRAM. Approximate delta truncation can also accumulate error;
MEET retains delta-sigma linearity and jointly optimizes dynamic cycles and state
memory rather than reporting sparse operations alone.

Binary TTX admits a smaller, exact adaptation for its `T=2` window. With
`alpha0=1/64`, maintain the integer score `S64=64*n11+n00`; at `t=1`, only channels
where Q or K toggles can change their contribution. Updating those lanes yields the same
TX score and Shiftmax output as full recomputation, so no new accuracy experiment is
needed. The required evidence is instead:

- measured `Q_toggle`, `K_toggle`, and union toggle density for all 12 blocks;
- TX compare/popcount cycles saved after scheduler/control overhead;
- previous Q/K (`2D=64` bits) or 2-bit contribution-class state per lane, plus
  the S64 accumulator and SRAM banking; a 1-bit match state is insufficient because
  active-active and silent-silent matches have different weights;
- proof that state stays on chip and that the exact full-recompute result is matched.

This **Delta-TTX** path is more hardware-native than approximate token pruning and
can share the XOR/toggle detector introduced by Motion-XOR TTX. It remains a P0
hardware optimization only if measured union toggle density is materially below 1.

The TTX epoch2 checkpoint was profiled on 100 DSEC validation samples. Summing raw
lane counts over every block, head, window, and sample gives `1,741,824,000` t1
lanes: Q toggle `0.7983%`, K toggle `1.9946%`, and Q-or-K union toggle `2.7832%`.
Therefore t1 can ideally skip `97.2168%` of lane contribution updates. Since t0
still requires a full score, the upper bound over the complete two-slice window is
`48.6084%` fewer TX lane comparisons, before state-memory and scheduling overhead.
This element-weighted result supersedes the earlier head-only estimate.

## Measured evidence that changes ranking

| Candidate | Evidence | Decision |
|---|---|---|
| H60 dyadic TTX | valid825 AEE 1.5016, AAE 9.8431, spikes 23.2439G | deployed reference |
| H66a full matrix | valid40 AEE 1.6554, AAE 15.5025 | training oracle only |
| H66c TP-TTX | valid825 AEE 1.6567, AAE 10.4283, spikes 24.4950G | fails 5% AEE window |
| H66d LR-TTX | 120-step valid10 AEE 1.2026, AAE 15.0169 | stop; weaker than TP |
| H66e TP self-bias | 120-step valid10 AEE 1.1949, AAE 14.7047 | stop; worsens TP |

The H66 results show that pairwise K aggregation improves small-sample AEE but has
an AAE/generalization failure. Local or temporal neighborhoods should not be promoted
merely because valid10 looks strong.

## Revised experiment priority

1. **H67 Motion-XOR TTX**: one 1/4-weight point, 120-step then valid40 only if it
   beats the pre-registered gate. This changes the deployed attention minimally.
2. **Castling-TTX**: training-only H66a auxiliary, deployed H60 unchanged. Audit the
   serialized model and runtime module count to prove zero inference overhead.
3. **Exact Delta-TTX profiling**: measure temporal toggle density and include state
   SRAM; no model retraining or accuracy approximation.
4. **Bounded dyadic gate bundling**: frozen-checkpoint inference ablation first,
   then short fine-tuning only if a useful skip ratio exists inside the AEE window.
5. **Attention operation accounting**: TX comparisons, XOR, Shiftmax, K late-scale,
   projection input activity, SRAM reads/writes, and control cycles.
6. Hamming/STAtten/SwiftFormer only if the first five fail. They require `D x D`
   state or a materially different projection/dataflow.

## Sources and inspected code

- Bishop, ISCA 2025: `papers/bishop_isca25_2505.12281.txt`
- EDCFlow, CVPR 2025: `papers/edcflow_cvpr25_2506.03512.txt`
- FlowFormer: `papers/flowformer_2203.16194.txt`
- Castling-ViT, CVPR 2023: CVF paper, equations 4-8 and training-only auxiliary path
- SwiftFormer, ICCV 2023: `repos/SwiftFormer/models/swiftformer.py`, lines 141-180
- SpAtten, HPCA 2021: project paper/site sections on progressive quantization
- MEET, CVPR 2025: CVF paper, delta-sigma state-memory and dynamic-cycle analysis
- TaskFusion, ISCA 2023: dual delta activation/weight sparsity and sparse-dense reuse
