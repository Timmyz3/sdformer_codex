# E2 ATLIF official debug short train - 2026-04-26

## Problem

The previous corrected E2 ATLIF full run did not match the Activity-Pruning-SNN behavior. It produced worse flow accuracy and worse sparsity than PSN:

| run | AEE | AAE | firing rate | estimated SOPs |
| --- | ---: | ---: | ---: | ---: |
| PSN baseline epoch59 valid40 | 1.5848 | 7.5012 | 0.08496 | 3.6219G |
| previous corrected ATLIF epoch59 valid40 | 8.6602 | 67.8866 | 0.37876 | 16.1464G |

## Root Cause

The implementation/config did not preserve the official ATLIF training scale.

1. `tau` was mapped as `1.0 / tau` in the earlier `exp_atlif` path, while the official ATLIF code uses `mem = mem * self.tau + x[t, ...]` directly.
2. `v_th` was set to `0.1`, while the official ATLIF default threshold is `1.0`.
3. `activity_eta` was `0.0`, so the official activity regularization equivalent was disabled.
4. The largest issue: official Activity-Pruning-SNN calls `threshold_update(model, optimizer.param_groups[0]["lr"])` with typical classifier LR around `0.1`. SDFormerFlow uses AdamW LR `1e-4`, so copied threshold updates were 1000x too small. The previous full run thresholds stayed near `0.100` for 60 epochs.
5. AMP diagnostics produced `inf/nan` threshold gradients, so this short run disables AMP for ATLIF.

## Code Changes Used

No baseline files under `third_party/SDformerFlow` are changed for this experiment path.

| purpose | file |
| --- | --- |
| official tau path | `neuron_experiments/E2_exp_atlif/overlay/models/STSwinNet_SNN/experimental_neurons/factory.py` |
| ATLIF neuron | `neuron_experiments/E2_exp_atlif/overlay/models/STSwinNet_SNN/experimental_neurons/single/atlif.py` |
| activity regularization and threshold update | `neuron_experiments/E2_exp_atlif/overlay/models/STSwinNet_SNN/experimental_neurons/training.py` |
| diagnostic entrypoint | `neuron_experiments/E2_exp_atlif/entrypoints/diagnose_atlif.py` |
| short-train config | `neuron_experiments/E2_exp_atlif/configs/diagnostic_official_eta1e-3_eta2_1e-4_lrs1000_noamp.yml` |

Important config values:

```yaml
model:
  spiking_neuron:
    neuron_type: exp_atlif_official
    v_th: 1.0
    tau: 0.9
    threshold_eta: 0.001
experimental_neuron:
  threshold_update: true
  threshold_lr_scale: 1000.0
  activity_eta: 0.0001
runtime:
  use_amp: false
```

## Short Train Verification

Command:

```bash
/opt/conda/envs/sdformerflow/bin/python neuron_experiments/E2_exp_atlif/entrypoints/train.py \
  --config neuron_experiments/E2_exp_atlif/configs/diagnostic_official_eta1e-3_eta2_1e-4_lrs1000_noamp.yml \
  --path_models neuron_experiments/E2_exp_atlif/results/short_official_eta1e-3_eta2_1e-4_lrs1000_noamp_checkpoint_epoch{}.pth
```

Log:

`neuron_experiments/E2_exp_atlif/results/short_official_eta1e-3_eta2_1e-4_lrs1000_noamp_train_20260426_231144.log`

| epoch | train loss | valid loss | lr | sec/epoch |
| --- | ---: | ---: | ---: | ---: |
| 0 | 8.4039 | 7.1098 | 1.0e-4 | 43.83 |
| 1 | 7.3078 | 6.7920 | 5.0e-5 | 43.17 |
| 2 | 7.7943 | 6.3522 | 2.5e-5 | 43.54 |

Thresholds from saved checkpoints:

| checkpoint | ATLIF modules | min threshold | mean threshold | max threshold |
| --- | ---: | ---: | ---: | ---: |
| epoch0 | 105 | 0.999882 | 1.062975 | 1.620348 |
| epoch1 | 105 | 1.000263 | 1.092422 | 1.782736 |
| epoch2 | 105 | 1.000439 | 1.105388 | 1.832972 |

This confirms that the threshold update is now active and increasing.

## Short Inference/Sparsity Check

ATLIF short checkpoint epoch2 on valid8:

```bash
/opt/conda/envs/sdformerflow/bin/python tools/profile_sops.py \
  --config neuron_experiments/E2_exp_atlif/configs/diagnostic_official_eta1e-3_eta2_1e-4_lrs1000_noamp.yml \
  --checkpoint neuron_experiments/E2_exp_atlif/results/short_official_eta1e-3_eta2_1e-4_lrs1000_noamp_checkpoint_epoch2.pth \
  --split valid --num-samples 8 --batch-size 1 --num-workers 0 \
  --dense-ops 42.63G --metric AEE --metric AAE
```

Output:

`neuron_experiments/E2_exp_atlif/results/profile_sops_short_official_eta1e-3_eta2_1e-4_lrs1000_noamp_epoch2_valid8_20260426_231518/sops_summary.json`

PSN baseline epoch59 on the same valid8 sample count:

`neuron_experiments/E0_psn_baseline/results/profile_sops_epoch59_valid8_20260426_231631/sops_summary.json`

| run | samples | AEE | AAE | firing rate | estimated SOPs |
| --- | ---: | ---: | ---: | ---: | ---: |
| PSN baseline epoch59 | 8 | 1.0116 | 6.2213 | 0.08548 | 3.6439G |
| ATLIF official short epoch2 | 8 | 6.3675 | 69.3083 | 0.13952 | 5.9478G |

## Interpretation

The corrected ATLIF path is now mechanically closer to the official source: thresholds increase, validation loss falls during short training, and sparsity is much better than the broken full run. It is still not better than PSN after only 3 short epochs on a 128-sample train subset, so this is not enough evidence to launch another full run blindly.

## 2026-04-27 Official-Copy Revision

The ATLIF core was changed again to avoid a hand-rewritten implementation. The file below now contains the upstream `zif_backward`, `ZIF`, `Surrogate`, and `ATLIF` blocks copied from `Activity-Pruning-SNN/models/submodules/layers.py`, with only a thin `ATLIFNode` wrapper for SDFormerFlow argument names:

`neuron_experiments/E2_exp_atlif/overlay/models/STSwinNet_SNN/experimental_neurons/single/atlif.py`

The first short-train launch used `--path_models`, which the baseline entrypoint does not support, so it exited before training. A second launch reached epoch0 but failed during `mlflow.pytorch.log_model(model)` serialization. The verified short run below disables MLflow with `SDFORMER_USE_MLFLOW=0` and saves local checkpoints.

Command:

```bash
SDFORMER_USE_MLFLOW=0 /opt/conda/envs/sdformerflow/bin/python neuron_experiments/E2_exp_atlif/entrypoints/train.py \
  --config neuron_experiments/E2_exp_atlif/configs/diagnostic_official_eta1e-3_eta2_1e-4_lrs1000_noamp.yml \
  --save_path /root/private_data/work/sdformer_codex/SDformer/neuron_experiments/E2_exp_atlif/results/short_official_copy_eta1e-3_eta2_1e-4_lrs1000_noamp_20260427_012326_checkpoint_epoch{}.pth
```

Note: because that run used a relative `--save_path`, the baseline `chdir` initially wrote checkpoints under `third_party/SDformerFlow/neuron_experiments/...`; those generated checkpoint files were moved back to the experiment results directory. Future launches should use an absolute `--save_path`.

Short-train log:

`neuron_experiments/E2_exp_atlif/results/short_official_copy_eta1e-3_eta2_1e-4_lrs1000_noamp_20260427_012326.log`

| epoch | train loss | valid loss | lr | sec/epoch |
| --- | ---: | ---: | ---: | ---: |
| 0 | 8.4416 | 7.3578 | 1.0e-4 | 46.59 |
| 1 | 7.2099 | 6.7949 | 5.0e-5 | 45.37 |
| 2 | 7.0020 | 6.1672 | 2.5e-5 | 45.18 |

Thresholds from copied-official checkpoint files:

| checkpoint | ATLIF modules | min threshold | mean threshold | max threshold |
| --- | ---: | ---: | ---: | ---: |
| epoch0 | 105 | 0.999717 | 1.063039 | 1.623900 |
| epoch1 | 105 | 0.999632 | 1.092164 | 1.784561 |
| epoch2 | 105 | 0.999682 | 1.105335 | 1.834236 |

Short inference valid8:

`neuron_experiments/E2_exp_atlif/results/profile_sops_short_official_copy_eta1e-3_eta2_1e-4_lrs1000_noamp_epoch2_valid8_20260427_012703/sops_summary.json`

| run | samples | AEE | AAE | firing rate | estimated SOPs |
| --- | ---: | ---: | ---: | ---: | ---: |
| ATLIF official-copy short epoch2 | 8 | 6.1818 | 65.4762 | 0.13829 | 5.8955G |

Next tuning should stay in short-run mode and test one variable at a time:

| trial | change | reason |
| --- | --- | --- |
| A | keep current config, train 10 short epochs | see whether AEE/firing continues improving |
| B | reduce `threshold_lr_scale` from `1000` to `300` | current pruning may be too aggressive early |
| C | keep `threshold_lr_scale=1000`, reduce `threshold_eta` from `1e-3` to `3e-4` | decouple threshold growth from optimizer LR scale |
| D | pretrain from PSN baseline if compatible | preserve flow accuracy while learning ATLIF thresholds |

## 2026-04-27 Baseline Integration And Sweep

Integration checks:

| check | result |
| --- | --- |
| copied official ATLIF selected by baseline model construction | pass, `Spiking_neuron(... ATLIFNode())` appears throughout the model |
| ATLIF threshold parameters in optimizer | pass, `count=105 requires_grad=105 in_optimizer=105` |
| threshold gradients | pass, finite values in two diagnostic batches |
| official `update_value` path | pass, threshold mean changed `1.00000 -> 1.00216 -> 1.00349` over two batches |
| checkpoint path handling | fixed in `entrypoints/train.py`; path-like args are converted to absolute paths before baseline `chdir` |
| MLflow model pickle | disabled for ATLIF runs with `SDFORMER_USE_MLFLOW=0`; local checkpoints are used |

Short sweep, 3 epochs each, valid8 profile:

| tag | threshold_eta | threshold_lr_scale | valid loss epoch2 | threshold mean epoch2 | AEE | AAE | firing rate | SOPs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| official-copy base | 1e-3 | 1000 | 6.1672 | 1.1053 | 6.1818 | 65.4762 | 0.13829 | 5.8955G |
| eta1e-3_lrs300 | 1e-3 | 300 | 6.7055 | 1.0340 | 6.7293 | 75.5997 | 0.14594 | 6.2214G |
| eta3e-4_lrs1000 | 3e-4 | 1000 | 6.7927 | 1.0339 | 6.7901 | 80.5703 | 0.14673 | 6.2549G |
| eta3e-4_lrs300 | 3e-4 | 300 | 6.5975 | 1.0097 | 6.5966 | 74.9607 | 0.14704 | 6.2682G |

Selected full-run params:

`threshold_eta=1e-3`, `threshold_lr_scale=1000`, `activity_eta=1e-4`, `v_th=1.0`, `tau=0.9`, `use_amp=false`.

Full run started:

```bash
SDFORMER_USE_MLFLOW=0 /opt/conda/envs/sdformerflow/bin/python neuron_experiments/E2_exp_atlif/entrypoints/train.py \
  --config /root/private_data/work/sdformer_codex/SDformer/neuron_experiments/E2_exp_atlif/configs/full_official_copy_eta1e-3_eta2_1e-4_lrs1000_bs8w8_noamp.yml \
  --save_path /root/private_data/work/sdformer_codex/SDformer/neuron_experiments/E2_exp_atlif/results/full_official_copy_eta1e-3_eta2_1e-4_lrs1000_bs8w8_noamp_20260427_021118_checkpoint_epoch{}.pth
```

Runtime:

| field | value |
| --- | --- |
| pid | 596367 |
| log | `neuron_experiments/E2_exp_atlif/results/full_official_copy_eta1e-3_eta2_1e-4_lrs1000_bs8w8_noamp_20260427_021118.log` |
| config | `neuron_experiments/E2_exp_atlif/configs/full_official_copy_eta1e-3_eta2_1e-4_lrs1000_bs8w8_noamp.yml` |
| checkpoint pattern | `neuron_experiments/E2_exp_atlif/results/full_official_copy_eta1e-3_eta2_1e-4_lrs1000_bs8w8_noamp_20260427_021118_checkpoint_epoch{}.pth` |

## 2026-04-27 AMP And Pin-Memory Speed Check

`pin_memory` was hardcoded to `True` in the baseline training script. The experiment entrypoint now patches it to:

```python
bool(config["loader"].get("pin_memory", True))
```

This keeps baseline files untouched and allows experiment configs to set `loader.pin_memory`.

AMP diagnostics:

| setting | result |
| --- | --- |
| AMP without threshold grad sanitize | ATLIF `thresh_grad` contained `nan` |
| AMP with `sanitize_threshold_grads=true` | finite sanitized threshold grads; official `update_value` still updates thresholds |

Short speed/quality checks:

| setting | train samples/sec | short valid trend | max memory |
| --- | ---: | --- | ---: |
| AMP bs8 pin=true | 4.0332 | epoch0 valid 6.7302 | 50.866 GiB |
| AMP bs8 pin=false | 4.2200 | epoch0 valid 6.7627 | 50.858 GiB |
| AMP bs10 pin=false | 4.56 by epoch2 | bad, valid `8.4120 -> 8.6934 -> 8.9269` | 63.440 GiB |
| AMP bs8 pin=false, 3 epochs | 4.40 by epoch2 | bad, valid `6.7460 -> 6.9566 -> 7.1953` | 50.874 GiB |
| AMP bs4 pin=false, 3 epochs | 3.94 by epoch2 | worse than no-AMP, valid `7.4672 -> 7.1153 -> 6.9577` | 25.908 GiB |
| AMP bs12/bs16 | OOM | not usable | 78-79 GiB before OOM |

Conclusion: `pin_memory=false` is slightly faster in this setup, but AMP hurts ATLIF short-train quality. The active full run therefore uses no-AMP with `pin_memory=false`.

Active full run:

| field | value |
| --- | --- |
| pid | 612483 |
| config | `neuron_experiments/E2_exp_atlif/configs/full_official_copy_eta1e-3_eta2_1e-4_lrs1000_bs8w8_noamp_pinfalse.yml` |
| log | `neuron_experiments/E2_exp_atlif/results/full_official_copy_eta1e-3_eta2_1e-4_lrs1000_bs8w8_noamp_pinfalse_20260427_023609.log` |
| checkpoint pattern | `neuron_experiments/E2_exp_atlif/results/full_official_copy_eta1e-3_eta2_1e-4_lrs1000_bs8w8_noamp_pinfalse_20260427_023609_checkpoint_epoch{}.pth` |

## 2026-04-28 Mid-Training Full Validation (Epoch 49)

To determine if `ATLIF` resolves the baseline performance issues, the active full run was checkpoint-profiled on `valid40` at `epoch30` and `epoch49`.

### 1. Performance vs Baseline (`valid40`)
* **PSN Baseline final (Epoch 59)**: AEE = `1.5848`, SOPs = `3.6439G` (valid8 proxy 3.62G)
* **Broken ATLIF previous run**: AEE = `8.6602`, SOPs = `16.1464G`
* **Current ATLIF (Epoch 30)**: AEE = `3.6035`, SOPs = `3.0059G`
* **Current ATLIF (Epoch 49)**: AEE = `3.8743`, SOPs = `2.8245G`

**Conclusion (涨点效果)**: The bug in `ATLIF` is successfully fixed. Performance improved drastically compared to the broken version (8.66 -> 3.87, SOPs 16G -> 2.82G). **However, it has not surpassed the PSN Baseline (1.5848)**.

### 2. Threshold & Sparsity Evolution
Threshold means extracted directly from checkpoint files show strict monotoic growth:

| Checkpoint | Mean Threshold | Total SOPs | Valid AEE |
| --- | ---: | ---: | ---: |
| Epoch 0 | 1.8867 | - | - |
| Epoch 10 | 3.4087 | - | - |
| Epoch 20 | 3.7987 | - | - |
| Epoch 30 | 3.9631 | 3.0059G | 3.6035 |
| Epoch 40 | 4.0406 | - | - |
| Epoch 49 | 4.0752 | 2.8245G | 3.8743 |
| Epoch 59 | 4.0926 | 2.8692G | 3.7574 |

**Analysis**:
The continuous growth in threshold parameters pushes the network to become overly sparse (SOPs dropping to 2.86G, much lower than the 3.64G of PSN). In the later epochs, this over-pruning actually harmed the AEE (rebounding from 3.60 up to 3.87, slightly settling to 3.75). 
The `threshold_eta` or `activity_eta` penalties are currently too strong, suffocating the model's ability to learn the optical flow.

## 2026-04-28 Final Training Evaluation (Epoch 59)

The training successfully completed at Epoch 59. Here is the final comparison between the custom ATLIF implementation and the PSN baseline on the `valid40` subset:

| Metric | PSN Baseline (Epoch 59) | ATLIF (Epoch 59) | Difference |
| :--- | :--- | :--- | :--- |
| **AEE** | 1.5848 | 3.7574 | +137% (Worse) |
| **AAE** | 7.5012 | 18.6163 | +148% (Worse) |
| **Firing Rate** | ~0.0849 | 0.0673 | -20% (Sparser) |
| **SOPs** | ~3.6439G | 2.8692G | -21% (Sparser) |
| **Threshold (mean)** | 1.0 (Fixed) | 4.0926 (Adaptive) | Higher Activation Barrier |

**Conclusion**:
The ATLIF modification effectively reduces the firing rate and Synaptic Operations (SOPs) by ~20% compared to the baseline, achieving a very sparse network. However, the performance degradation is severe (AEE increases from 1.58 to 3.75). The final mean threshold of `4.09` verifies that the `activity_eta` penalty is still overly dominant, aggressively increasing thresholds throughout the entire 60 epochs without plateauing correctly. Future steps should focus on easing this penalty or adding a dynamic decay to `threshold_lr_scale` to prevent over-pruning.
