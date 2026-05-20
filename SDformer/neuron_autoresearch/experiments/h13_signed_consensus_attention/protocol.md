# H13 Signed-Consensus Attention Protocol

Date: 2026-05-19

## Goal

Reach H9a-level accuracy while keeping SOPs near 3.0G by making ATLIF ternary
Q/K attention more compatible with BSA/TSN-style signed spikes.

Reference target:

| Run | AEE | AAE | SOPs | Firing |
|---|---:|---:|---:|---:|
| H9a valid40 | 1.504 | 7.636 | 3.085G | 0.0724 |

## Hypotheses

### H13a: tuned symmetric target-rate ATLIF with legacy attention

Change only the H12c neuron hyperparameters:

- `threshold_init: 0.2`
- `max_threshold: 0.4`
- `target_rate: 0.10`
- `target_rate_eta: 0.01`
- `activity_eta: 0.5`
- `lambda_ang: 0.1`
- attention mode remains `compat_qk_product`

Prediction: stronger threshold feedback should avoid H12c's dense negative
firing and establish a safer neuron baseline before attention changes.

### H13b: signed consensus plus Shiftmax

Replace the Shiftmax score with ternary sign consensus:

```text
score = sum(sign(Q) * sign(K)) / head_dim
gate = Shiftmax(score)
output = K * gate
```

Prediction: keeping Shiftmax preserves BSA-style normalization, while the score
now reflects signed ternary agreement instead of theta-weighted real products.

### H13c: signed consensus plus ShiftNorm

Use the same sign-consensus score, but normalize nonnegative scores with a
next-power-of-two denominator:

```text
score = relu(sum(sign(Q) * sign(K)) / head_dim + bias)
gate = score / 2^ceil(log2(sum(score)))
output = K * gate
```

Prediction: this is more hardware-friendly than Shiftmax, but may be less
expressive. It should run only if smoke metrics are not clearly worse than H13b.

### H13f: bias-centered ternary calibration

Root-cause fix after H13a-e: original PSN has a learned/baseline bias near
`-1`, which is silent in binary PSN but was interpreted as a negative event by
zero-centered ternary firing. H13f subtracts the copied PSN bias before ternary
thresholding for Q/K only:

```text
h = PSN_affine(x) - PSN_bias
out = {-theta, 0, +theta}
```

This preserves the original positive spike boundary and allows negative spikes
only when the signal moves below the silent center. Binary FFN/downsample groups
remain zero-centered.

### H13j: stronger Q/K sparsity

Same mechanism as H13f, but Q/K target firing is lowered:

- `target_rate: 0.05`
- `target_rate_eta: 0.03`
- `activity_eta: 0.6`
- `max_threshold: 1.8`

Purpose: keep H13f's AAE benefit while moving SOPs toward the H9a range.

### H13m: H13j plus all FFN/downsample scope

Combines H13j's bias-centered bipolar Q/K with H9g's broad binary replacement:

- all 12 Swin block FFN `mlp.sn1/sn2`
- all existing downsample neurons in stages 0/1/2

Purpose: Q/K-only sparsity is not enough; FFN/downsample contribute a large
fraction of global firing. H13m is the first H13 candidate aimed directly at
the `~3G` SOP target.

### H14: strict BSA matrix attention

H14 is the follow-up requested by the paper logic: instead of H13's token-wise
signed consensus gate, it builds a true ternary attention matrix:

```text
Qe = sign(Q), Ke = sign(K)
Score = Qe @ Ke^T
Gate = Shiftmax(Score)
Output = Gate @ V
```

The SDFormerFlow QK block has no separate V projection, so K is reused as V.
H14a uses threshold-scaled K to preserve ATLIF amplitude; H14b uses sign(K)
for the most hardware-friendly event-only path; H14c uses a milder head-dim
normalization variant for stability.

### H13r: H13n plus angular protection

H13_DEEP_ANALYSIS highlights that AAE can drift even when AEE remains acceptable.
H13r keeps the H13n replacement scope and signed-consensus Shiftmax attention,
but enables angular loss:

- `lambda_ang: 0.2`
- `use_angular_loss: true`

Purpose: test whether a mild direction constraint can preserve the strong AAE
behavior of early H13 checkpoints during full-parameter finetuning.

### H13s: H13n plus ShiftNorm

H13s keeps the H13n replacement scope, but swaps Shiftmax for power-of-two
ShiftNorm:

```text
score = relu(signed_consensus(Q, K) + bias)
gate = score / 2^ceil(log2(sum(score)))
output = K * gate
```

Purpose: reduce the hardware cost of exponent/LUT-style Shiftmax while keeping
a signed-consensus normalization mechanism.

### H13t: H13n plus exact L1 popcount normalization

H13t is the strongest hardware-friendly ablation from the review docs:

```text
score = relu(signed_consensus(Q, K) + bias)
gate = score / sum(score)
output = K * gate
```

It removes Shiftmax completely. If H13t holds accuracy, the paper story can
lean on signed ternary popcount plus simple accumulation/division rather than
copying Shiftmax directly.

### H13u: independent negative-event target

H13u keeps H13n attention and scope, but adds a separate negative firing target:

- `negative_target_rate: 0.025`
- `negative_target_eta: 0.02`
- asymmetric threshold scale bounded to `[0.7, 1.3]`

Purpose: preserve negative ternary events as a first-class signal instead of
letting positive and negative firing balance only indirectly through a shared
threshold.

## Screening Rules

Smoke runs use 12 training steps and one validation sample. A candidate is
discarded immediately if:

- train loss diverges or becomes NaN
- `activity_mean > 0.35`
- `neg_mean > 0.25`
- validation loss is much worse than H12 smoke behavior

Guard runs use 120 training steps. A candidate proceeds to longer training only
if it approaches:

- `AEE <= 1.55`
- `AAE <= 8.2`
- SOPs roughly `2.9G-3.15G`
- firing roughly `0.07-0.09`

## Implementation Boundary

All code lives in experiment-local overlays/configs. The baseline folder under
`third_party/SDformerFlow` is not modified.

## Results So Far

| Run | Scope | Valid40 AEE | Valid40 AAE | SOPs | Firing | Decision |
|---|---|---:|---:|---:|---:|---|
| H13f guard120 | bias-centered Q/K + H9a FFN/downsample scope | 1.502 | 7.233 | 3.746G | 0.0879 | accuracy strong, sparsity weak |
| H13j guard120 | H13f with Q/K target rate 0.05 | 1.527 | 7.222 | 3.659G | 0.0858 | accuracy acceptable, still sparse gap |
| H13m guard120 | H13j + all FFN/downsample | 1.573 | 7.564 | 3.628G | 0.0851 | promoted to full; short SOPs still high but scope should improve with long training |
| H13i guard120 | bias-centered Q/K + ShiftNorm guard | 1.554 | 7.516 | 3.785G | 0.0888 | normal bipolar firing but weaker than Shiftmax |
| H13n guard120 | H13j + partial high-SOP FFN and downsample stage0/stage2 | 1.500 | 7.365 | 3.651G | 0.0856 | best precision candidate; promoted to full |
| H13p guard120 | H13n with stronger Q/K target rate 0.02 | 1.541 | 7.797 | 3.590G | 0.0842 | sparse pressure hurts accuracy too early |
| H13q guard120 | H13n with mid Q/K target rate 0.035 | 1.579 | 7.524 | 3.649G | 0.0856 | no gain over H13n, not promoted |

## Queued From H13 Review Docs

After the active H13n full run finishes, the controller profiles H13n
checkpoints and then screens these guard120 variants before H14:

| Run | Added idea | Config |
|---|---|---|
| H13r | angular-loss protection for AAE | `configs/h13r_ang02_h13n_guard120.yml` |
| H13s | signed-consensus ShiftNorm | `configs/h13s_shiftnorm_h13n_guard120.yml` |
| H13t | signed-consensus popcount L1 norm, no Shiftmax | `configs/h13t_popcount_l1_h13n_guard120.yml` |
| H13u | independent negative firing feedback | `configs/h13u_negtarget_h13n_guard120.yml` |
