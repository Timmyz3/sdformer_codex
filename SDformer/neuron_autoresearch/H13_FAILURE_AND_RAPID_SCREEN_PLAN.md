# H13 Failure Diagnosis And Rapid Screen Plan

Date: 2026-05-19

## Diagnosis

H13n did not fail like a simple bad learning-rate run. The evidence points to a
module/objective mismatch plus hyperparameter weakness.

Observed valid40 trajectory:

| checkpoint | AEE | AAE | SOPs | firing |
|---|---:|---:|---:|---:|
| H13n epoch0 | 1.5445 | 7.6520 | 3.5992G | 0.08443 |
| H13n epoch7 | 1.5825 | 7.4031 | 3.7777G | 0.08862 |
| H13n epoch29 | 2.5752 | 13.5773 | 3.7052G | 0.08692 |
| H9a reference | 1.5044 | 7.6365 | 3.0847G | 0.07236 |

Key points:

- Early checkpoints can keep AAE competitive, but SOPs/firing are worse than
  H9a and baseline.
- Later checkpoints degrade AAE badly, so train loss is not a safe proxy.
- Threshold statistics during H13/H14 runs show firing stays around or above the
  baseline range; target-rate feedback is not producing the intended sparse
  pressure.

Interpretation:

1. **Attention design risk**: H13 signed consensus gate is carrier-preserving,
   but it still changes Q/K token weighting in a way that can conflict with
   optical-flow direction. AAE degradation is the symptom.
2. **Sparse objective risk**: target-rate and activity regularization do not
   reliably lower global firing/SOPs under full fine-tuning.
3. **Training horizon risk**: good early profiles can disappear by epoch29.
   Therefore full training must be gated by early profile checkpoints.

## Current Rapid Screen Batch

Entry:

```bash
neuron_experiments/H9_bipolar_self_attention/entrypoints/rapid_screen.py
```

Batch:

| Candidate | Purpose |
|---|---|
| H18a alpha-XNOR Shiftmax | CVPR 2025 alpha-XNOR ternary similarity with carrier preserved |
| H18a alpha-XNOR L1 | Same similarity without exponent-like Shiftmax |
| H18b A2OS2A gate | CVPR 2025 binary-Q/nonnegative-K/ternary-V inspired gate |
| H13v lower LR | Test whether AAE drift is optimizer speed |
| H13w stronger sparse feedback | Test whether SOPs/firing can be forced lower |
| H13x threshold frozen | Diagnose whether threshold update destabilizes training |

Screen:

- steps: 40 and 120
- initial profile: valid10
- promote: valid40 if AEE/AAE/SOPs pass loose thresholds

## Decision Rules

- If a variant improves valid10 but fails valid40, it is not promoted.
- If AAE is good but SOPs/firing are worse than H9a, it is not a sparse-story
  mainline.
- If SOPs improve but AAE exceeds about `8.2`, it is only a sparsity ablation.
- If H18a/H18b do not beat H13/H9a in short screen, move next to H21 Hamming-QK
  and TIM temporal prefilter.
- Do not reject direct replacement from a 40-step collapse alone. H18c/H18d/H18e
  show that direct attention can have terrible 40-step AAE while recovering by
  120 steps. For this branch, 120-step valid10 is the minimum useful screen.

## Expanded Direct-Replacement Screen

The user explicitly allowed non-conservative replacement. Therefore the next
batch also tests direct attention replacement, not only auxiliary gates:

| Candidate | Form | Hyperparameters |
|---|---|---|
| H18c | direct alpha-XNOR matrix + Shiftmax | `alpha0=0.02`, `mismatch_penalty=0.25` |
| H18d | direct alpha-XNOR matrix + L1 | same base values |
| H18e | direct A2OS2A matrix + L1 | binary Q, nonnegative K, thresholded K as V proxy |
| H18c sweep | direct alpha-XNOR matrix + Shiftmax | `alpha0 in {0,0.01,0.05}`, `mismatch_penalty in {0,0.25,0.5}` |

Reason:

- A failed single setting should not rule out a paper mechanism.
- Alpha-XNOR is sensitive to silence reward (`alpha0`) and opposite-polarity
  penalty.
- Direct replacement may reveal whether the paper mechanism works better than
  the carrier-preserving gate, but it remains under short-screen control before
  any full training.

Observed first-pass signal:

| Candidate | steps | valid10 AEE | valid10 AAE | SOPs | firing | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| H18c direct alpha-XNOR + Shiftmax | 40 | 7.2732 | 93.2726 | 2.4476G | 0.05742 | collapsed too early |
| H18c direct alpha-XNOR + Shiftmax | 120 | 1.0876 | 6.7315 | 3.8138G | 0.08946 | strong accuracy, sparse pressure still weak |
| H18d direct alpha-XNOR + L1 | 120 | 1.1248 | 6.9325 | 4.2328G | 0.09929 | accurate but too dense |
| H18e direct A2OS2A + L1 | 120 | 1.0467 | 6.2924 | 4.3253G | 0.10146 | best early accuracy, not a sparse mainline yet |
| H13v low LR repair | 120 | 0.9609 | 5.9033 | 3.8293G | 0.08983 | strong early accuracy, needs valid40 |
| H13w stronger sparse feedback | 120 valid40 | 1.5350 | 7.5568 | 3.5815G | 0.08401 | valid40 survives, but SOP gap to H9a remains |

This supports the user's hypothesis that failed runs can be hyperparameter or
training-horizon failures, not necessarily module failures.

## H22 Hyperparameter Separation

H22 keeps the H18c module fixed and varies only the hyperparameters. It is
queued after H21 with 120-step screens only.

| Group | Knobs | Purpose |
|---|---|---|
| H22a-d | `target_rate`, `target_rate_eta`, `activity_eta` | lower SOPs/firing without changing attention semantics |
| H22e | `score_scale` | test Shiftmax sharpness and AAE stability |
| H22f-h | `alpha0`, `mismatch_penalty` | test alpha-XNOR silence reward and opposite-polarity penalty |
| H22i | `consensus_score_norm=active` | make attention score normalization sparse-aware |
| H22j | `value_mode=sign` | test hardware-friendly sign values versus ATLIF threshold amplitude |
| H22k | optimizer LR | decide whether recovery/sparsity is optimizer-speed limited |

## H23 Low-LR Sparse Combinations

First-pass H13v shows that lowering LR to `1e-5` can dramatically improve
120-step valid10 AEE/AAE while still leaving SOPs high. H23 therefore tests
combined low-LR plus stronger sparse target feedback after the one-axis H22
screen.

| Candidate | Base | LR | Sparse target | Extra attention change |
|---|---|---:|---|---|
| H23a | H18c | `1e-5` | target `0.040`, eta `0.05`, activity `0.8` | none |
| H23b | H18c | `1e-5` | target `0.035`, eta `0.08`, activity `1.0` | none |
| H23c | H18c | `1e-5` | target `0.040`, eta `0.05`, activity `0.8` | `score_scale=0.75` |
| H23d | H13v | `1e-5` | target `0.040`, eta `0.05`, activity `0.8` | none |
| H23e | H13v | `1e-5` | target `0.035`, eta `0.08`, activity `1.0` | none |

## H21 Hamming Follow-Up

SpikeVideoFormer official Hamming attention uses:

```text
x = (2K - 1)^T V
x = (2Q - 1) x / (2 * dim)
```

This is a direct non-QKFormer replacement and avoids a token-token softmax
matrix. Because SDFormerFlow's QK attention has no V branch, H21 reuses K as the
value stream.

Planned candidates:

| Candidate | Form |
|---|---|
| H21a | official binary mapping: `0/1 -> -1/+1` |
| H21b | ternary-active mapping: silence stays `0`, active polarity is `-1/+1` |
| H21c | official binary mapping with sign-only K as value proxy |

Rationale:

- H21a is closest to SpikeVideoFormer.
- H21b is safer for sparse event data where silence should not always be a
  negative vote.
- H21c tests whether threshold-valued multiplication is part of the hardware
  cost/AAE problem.
