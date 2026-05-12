# E2 ATLIF Porting Root Cause

Date: 2026-04-25

## Official Reference

Source checked from `https://github.com/putshua/Activity-Pruning-SNN`.

Key official files:

- `/tmp/Activity-Pruning-SNN/models/submodules/layers.py`
- `/tmp/Activity-Pruning-SNN/utils/utils.py`
- `/tmp/Activity-Pruning-SNN/main.py`

## Findings

The `spike * threshold` behavior is correct for ATLIF. The earlier suspicion that output scaling itself was wrong was incorrect.

The actual porting issue was that E2 only moved the forward neuron shape into SDFormerFlow. It did not move the training mechanism that makes ATLIF sparse:

- official ATLIF returns `out * thresh`;
- official surrogate computes a separate `thre_updates` value;
- official training calls `threshold_update(model, lr)` after `optimizer.step()`;
- official training optionally adds `regularize_spike(model) * eta2`;
- E2 previously had none of the threshold-update training hook;
- E2 also used a generic surrogate whose threshold gradient summed over all elements, while the official ATLIF threshold gradient uses a mean.

This explains why E2 epoch59 produced high firing rates. The threshold was simply being optimized through the flow loss and generic surrogate gradient, without the official threshold-increase pathway.

## Code Changes

Changed files are all under `neuron_experiments/E2_exp_atlif`, not under `third_party/SDformerFlow`.

| File | Purpose |
|---|---|
| `overlay/models/STSwinNet_SNN/experimental_neurons/single/atlif.py` | Replaced generic spike function with official-style ATLIF surrogate, preserved `out * thresh`, added `update_value`, `r`, `s`, and `act_value`. |
| `overlay/models/STSwinNet_SNN/experimental_neurons/training.py` | Added experiment-local `threshold_update()` and `regularize_activity()` helpers. |
| `overlay/models/STSwinNet_SNN/experimental_neurons/factory.py` | Passes `threshold_eta` into ATLIF. |
| `overlay/models/STSwinNet_SNN/Spiking_modules.py` | Allows `threshold_eta` through the SDFormerFlow spiking-neuron wrapper. |
| `entrypoints/train.py` | Patches the baseline training source at launch time to call activity regularization and `threshold_update()` after optimizer steps. |
| `configs/full_bs12w8.yml` | Adds explicit ATLIF threshold-update config. |
| `configs/smoke.yml` | Adds explicit ATLIF threshold-update config. |

## Current E2 Config Knobs

```yaml
spiking_neuron:
  threshold_eta: 0.0001

experimental_neuron:
  threshold_update: true
  threshold_lr_scale: 1.0
  activity_eta: 0.0
  min_threshold: 0.001
```

`threshold_eta` maps to the official `eta` / `sp` term. `activity_eta` maps to the official `eta2` activity regularizer and is currently disabled.

## Verification

Commands run:

```bash
/opt/conda/envs/sdformerflow/bin/python -m compileall -q \
  neuron_experiments/E2_exp_atlif/overlay/models/STSwinNet_SNN/experimental_neurons/single/atlif.py \
  neuron_experiments/E2_exp_atlif/overlay/models/STSwinNet_SNN/experimental_neurons/training.py \
  neuron_experiments/E2_exp_atlif/overlay/models/STSwinNet_SNN/experimental_neurons/factory.py \
  neuron_experiments/E2_exp_atlif/overlay/models/STSwinNet_SNN/Spiking_modules.py \
  neuron_experiments/E2_exp_atlif/entrypoints/train.py
```

```bash
/opt/conda/envs/sdformerflow/bin/python -m unittest tests.test_profile_sops
```

```bash
SDFORMER_USE_MLFLOW=0 SDFORMER_MLFLOW_MODEL_LOGGING=0 \
/opt/conda/envs/sdformerflow/bin/python neuron_experiments/E2_exp_atlif/entrypoints/train.py \
  --config neuron_experiments/E2_exp_atlif/configs/smoke.yml \
  --path_mlflow ''
```

Results:

- compile check passed;
- `tests.test_profile_sops`: 5 tests passed;
- E2 smoke training completed one epoch and validation without crashing.

Temporary smoke checkpoints created under `third_party/SDformerFlow/results/` by the baseline script were removed after the smoke run.

## Next Step

The previous E2 epoch59 checkpoint should be treated as invalid for judging ATLIF. The next fair experiment should retrain E2 with this corrected ATLIF port, then rerun `tools/profile_sops.py` to compare AEE, AAE, firing rate, and estimated SOPs against PSN.
