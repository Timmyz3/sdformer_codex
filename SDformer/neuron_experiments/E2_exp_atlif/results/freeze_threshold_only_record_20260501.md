# Freeze Threshold-Only ATLIF Run

Date: 2026-05-01 UTC

## Goal

Test whether ATLIF can recover sparsity by freezing the pretrained ATLIF backbone and updating only the learnable ATLIF thresholds.

## Setup

| field | value |
| --- | --- |
| config | `neuron_experiments/E2_exp_atlif/configs/freeze_threshold_only_from_atlif_epoch59_lr1e-4_lrs1000_act1e-4_bs12.yml` |
| init checkpoint | `neuron_experiments/E2_exp_atlif/results/full_pretrained_eta3e-4_eta2_3e-5_lrs300_bs12w8_amp_pinfalse_checkpoint_epoch59.pth` |
| freeze mode | `threshold_only` |
| trainable parameters | 105 |
| frozen parameters | 54,913,928 |
| epochs | 5 |
| batch size | 12 |
| workers | 8 |
| AMP | true |
| optimizer lr | `1e-4` |
| threshold lr scale | `1000` |
| activity eta | `1e-4` |
| weight decay | `0` |

## Running Command

```bash
setsid env SDFORMER_USE_MLFLOW=0 /opt/conda/envs/sdformerflow/bin/python neuron_experiments/E2_exp_atlif/entrypoints/train.py \
  --config /root/private_data/work/sdformer_codex/SDformer/neuron_experiments/E2_exp_atlif/configs/freeze_threshold_only_from_atlif_epoch59_lr1e-4_lrs1000_act1e-4_bs12.yml \
  --prev_runid /root/private_data/work/sdformer_codex/SDformer/neuron_experiments/E2_exp_atlif/results/full_pretrained_eta3e-4_eta2_3e-5_lrs300_bs12w8_amp_pinfalse_checkpoint_epoch59.pth \
  --save_path /root/private_data/work/sdformer_codex/SDformer/neuron_experiments/E2_exp_atlif/results/freeze_threshold_only_from_atlif_epoch59_lr1e-4_lrs1000_act1e-4_bs12_20260501_014255_checkpoint_epoch{}.pth
```

## Runtime

| field | value |
| --- | --- |
| PID | `1606360` |
| log | `neuron_experiments/E2_exp_atlif/results/freeze_threshold_only_from_atlif_epoch59_lr1e-4_lrs1000_act1e-4_bs12_20260501_014255.log` |
| checkpoint pattern | `neuron_experiments/E2_exp_atlif/results/freeze_threshold_only_from_atlif_epoch59_lr1e-4_lrs1000_act1e-4_bs12_20260501_014255_checkpoint_epoch{}.pth` |

## Baseline Targets

| run | AEE | AAE | firing rate | SOPs |
| --- | ---: | ---: | ---: | ---: |
| PSN baseline epoch59 valid40 | 1.5848 | 7.5012 | 0.08496 | 3.6219G |
| previous ATLIF full-pretrained epoch59 | 2.5128 | 12.5417 | 0.12212 | 5.2062G |

The useful signal for this freeze run is whether firing rate drops below the previous ATLIF `0.12212` without making AEE much worse than `2.5128`.
