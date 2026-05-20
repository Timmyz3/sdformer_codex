# Fast Screening Protocol

Date: 2026-05-19

Goal: reject bad neuron/ternary/attention variants before full training. A full
run should only start after it survives a cheap proxy screen and at least one
valid40 profile.

## Why Full Training Was Wasteful

Several variants looked acceptable in training loss but failed in profile:

- H13n full was best around epoch7, then degraded by epoch29.
- Strict/direct attention variants can preserve loss while making AAE worse.
- SOPs/firing are not reliably predicted by train loss alone.

Therefore the first decision signal must be **profiled AEE/AAE/SOPs**, not only
train loss.

## Three-Stage Gate

### Stage 0: No-Train Profile

Use the baseline checkpoint after module replacement, before updates:

- Purpose: catch broken wiring and catastrophic attention semantics.
- Samples: valid10 first; valid40 if valid10 is close.
- Reject immediately if AEE or AAE explodes.

### Stage 1: Short Train

Train from the same baseline checkpoint with:

- `loader.n_epochs: 1`
- `runtime.max_train_steps: 40`, `80`, or `120`
- `runtime.skip_state_save: true`
- `optimizer.use_amp: true`
- `batch_size: 8` or the largest stable value
- profile checkpoint epoch0 on valid10

Promotion rule for valid40:

```text
AEE <= 1.62
AAE <= 8.20
SOPs <= 3.90G
```

These thresholds are intentionally loose. They should catch candidates that can
approach H9a without spending full training time. A candidate with excellent
AEE/AAE but slightly high SOPs should still get valid40 once, because a later
hyperparameter sweep can target sparsity.

### Stage 2: Truncated Multi-Epoch

Only for candidates that pass Stage 1:

- Train 3-8 epochs or save checkpoints at short intervals.
- Profile early checkpoints, not only final checkpoint.
- Stop if valid40 AEE/AAE worsen while SOPs does not drop.

Full training starts only if the early checkpoint has a credible reason to beat
H9a or provide a clearly better sparsity story.

## Command

Use the new rapid screener:

```bash
/opt/conda/envs/sdformerflow/bin/python -u \
  neuron_experiments/H9_bipolar_self_attention/entrypoints/rapid_screen.py \
  --config h13n_biascenter_shiftmax_target05_halfffn_down02_guard120.yml \
  --config h14c_strict_bsa_thetav_mild_guard120.yml \
  --steps 40 \
  --steps 120 \
  --valid-samples 10 \
  --promote-samples 40 \
  --batch-size 8 \
  --workers 8 \
  --amp
```

The script writes:

```text
neuron_experiments/H9_bipolar_self_attention/results/rapid_screen_<stamp>/summary.md
neuron_experiments/H9_bipolar_self_attention/results/rapid_screen_<stamp>/summary.csv
```

It creates temporary configs under that result folder, so the baseline and
canonical experiment configs remain unchanged.

## Override Examples

Try a lower target firing rate:

```bash
--set atlif_ternary_psn.target_rate=0.035
```

Try stronger activity penalty:

```bash
--set atlif_ternary_psn.activity_eta=1.0
```

Try a different attention normalization, assuming the code supports it:

```bash
--set bsa_attention.mode=signed_consensus_popcount_l1
```

Try lower learning rate:

```bash
--lr 1e-5
```

## Current Recommended Search Order

1. H9a-compatible carrier-preserving attention modes.
2. H18a ternary alpha-XNOR auxiliary gate.
3. H18b A2OS2A-style hybrid gate.
4. H21a Hamming-QK gate.
5. Temporal gates from TIM/STAA only after Q/K semantics are stable.

Do not promote direct carrier replacement unless Stage 0 and Stage 1 profiles
are already competitive. The earlier H10/H14 behavior shows direct replacement
can destroy AAE while train loss still looks plausible.
