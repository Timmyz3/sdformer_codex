# E2 ATLIF Full-Pretrained Inference Report

Date: 2026-04-30 UTC

## Experiment

This report evaluates the ATLIF full-pretrained run against the PSN baseline using the same valid split profile setting:

| field | value |
| --- | --- |
| experiment | `E2_exp_atlif` |
| neuron | `exp_atlif_official` |
| config | `neuron_experiments/E2_exp_atlif/configs/full_pretrained_eta3e-4_eta2_3e-5_lrs300_bs12w8_amp_pinfalse.yml` |
| training log | `neuron_experiments/E2_exp_atlif/results/full_pretrained_eta3e-4_eta2_3e-5_lrs300_bs12w8_amp_pinfalse_20260428_231544.log` |
| evaluated checkpoints | epoch45, epoch59 |
| split | `valid` |
| samples | 40 |
| batch size | 1 |
| dense ops | `42.63G` |
| metrics | `AEE`, `AAE`, firing rate, estimated SOPs |

The epoch45 checkpoint is included because it had the best observed validation loss during training. The epoch59 checkpoint is the final saved model.

## Commands

Epoch45:

```bash
SDFORMER_USE_MLFLOW=0 /opt/conda/envs/sdformerflow/bin/python tools/profile_sops.py \
  --config neuron_experiments/E2_exp_atlif/configs/full_pretrained_eta3e-4_eta2_3e-5_lrs300_bs12w8_amp_pinfalse.yml \
  --checkpoint neuron_experiments/E2_exp_atlif/results/full_pretrained_eta3e-4_eta2_3e-5_lrs300_bs12w8_amp_pinfalse_checkpoint_epoch45.pth \
  --output-dir neuron_experiments/E2_exp_atlif/results/profile_sops_full_pretrained_epoch45_valid40_20260430_010301 \
  --split valid --num-samples 40 --batch-size 1 --num-workers 0 \
  --dense-ops 42.63G --metric AEE --metric AAE
```

Epoch59:

```bash
SDFORMER_USE_MLFLOW=0 /opt/conda/envs/sdformerflow/bin/python tools/profile_sops.py \
  --config neuron_experiments/E2_exp_atlif/configs/full_pretrained_eta3e-4_eta2_3e-5_lrs300_bs12w8_amp_pinfalse.yml \
  --checkpoint neuron_experiments/E2_exp_atlif/results/full_pretrained_eta3e-4_eta2_3e-5_lrs300_bs12w8_amp_pinfalse_checkpoint_epoch59.pth \
  --output-dir neuron_experiments/E2_exp_atlif/results/profile_sops_full_pretrained_epoch59_valid40_20260430_010343 \
  --split valid --num-samples 40 --batch-size 1 --num-workers 0 \
  --dense-ops 42.63G --metric AEE --metric AAE
```

## Outputs

| run | summary | layer firing CSV |
| --- | --- | --- |
| PSN baseline epoch59 | `neuron_experiments/E0_psn_baseline/results/profile_sops_epoch59_valid40/sops_summary.json` | not regenerated in this run |
| ATLIF epoch45 | `neuron_experiments/E2_exp_atlif/results/profile_sops_full_pretrained_epoch45_valid40_20260430_010301/sops_summary.json` | `neuron_experiments/E2_exp_atlif/results/profile_sops_full_pretrained_epoch45_valid40_20260430_010301/layer_firing_rates.csv` |
| ATLIF epoch59 | `neuron_experiments/E2_exp_atlif/results/profile_sops_full_pretrained_epoch59_valid40_20260430_010343/sops_summary.json` | `neuron_experiments/E2_exp_atlif/results/profile_sops_full_pretrained_epoch59_valid40_20260430_010343/layer_firing_rates.csv` |

## Metrics

| run | AEE | AAE | AEE_PE1 | AEE_PE2 | AEE_PE3/outliers | firing rate | estimated SOPs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PSN baseline epoch59 | 1.5848 | 7.5012 | 0.4839 | 0.1855 | 0.0896 | 0.08496 | 3.6219G |
| ATLIF epoch45 | 2.6196 | 13.4010 | 0.7375 | 0.3782 | 0.2048 | 0.12156 | 5.1819G |
| ATLIF epoch59 | 2.5128 | 12.5417 | 0.7203 | 0.3587 | 0.1895 | 0.12212 | 5.2062G |

## Delta Versus Baseline

Positive delta means worse for AEE/AAE/outlier metrics and more active for firing/SOPs.

| run | AEE delta | AEE ratio | AAE delta | AAE ratio | firing delta | firing ratio | SOPs delta | SOPs ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ATLIF epoch45 | +1.0348 | 1.653x | +5.8998 | 1.787x | +0.03660 | 1.431x | +1.5601G | 1.431x |
| ATLIF epoch59 | +0.9280 | 1.586x | +5.0405 | 1.672x | +0.03716 | 1.437x | +1.5843G | 1.437x |

## Threshold And Firing Notes

| checkpoint | ATLIF modules | min threshold | p25 | mean threshold | p75 | max threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| epoch45 | 105 | 1.046052 | 1.121329 | 2.366585 | 2.697813 | 12.885397 |
| epoch59 | 105 | 1.046639 | 1.124383 | 2.382501 | 2.719119 | 12.933386 |

Top firing layers from epoch59:

| firing rate | layer |
| ---: | --- |
| 0.2723 | `sttmultires_unet.encoders.swin3d.layers.0.downsample.sn` |
| 0.2683 | `sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.1.mlp.sn1` |
| 0.2642 | `sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.5.mlp.sn1` |
| 0.2626 | `sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.4.mlp.sn1` |
| 0.2622 | `sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp.sn1` |
| 0.2606 | `sttmultires_unet.encoders.swin3d.layers.2.downsample.sn` |

## Files Used By This Evaluation

| role | file |
| --- | --- |
| profile/eval entry | `tools/profile_sops.py` |
| experiment train entry | `neuron_experiments/E2_exp_atlif/entrypoints/train.py` |
| experiment config | `neuron_experiments/E2_exp_atlif/configs/full_pretrained_eta3e-4_eta2_3e-5_lrs300_bs12w8_amp_pinfalse.yml` |
| baseline model modules via overlay | `neuron_experiments/E2_exp_atlif/overlay/models/STSwinNet_SNN/Spiking_modules.py` |
| neuron factory | `neuron_experiments/E2_exp_atlif/overlay/models/STSwinNet_SNN/experimental_neurons/factory.py` |
| official ATLIF wrapper/core | `neuron_experiments/E2_exp_atlif/overlay/models/STSwinNet_SNN/experimental_neurons/single/atlif.py` |
| ATLIF training helpers | `neuron_experiments/E2_exp_atlif/overlay/models/STSwinNet_SNN/experimental_neurons/training.py` |

## Result

This ATLIF full-pretrained run does not beat the PSN baseline. The final checkpoint improves accuracy over epoch45, but it is still worse than baseline on AEE and AAE, and it is also less sparse. The global firing rate is about 43.7% higher than baseline, so estimated SOPs increase from 3.6219G to 5.2062G.

The threshold values did grow, but the learned thresholds did not translate into lower global firing on this SDFormerFlow integration. The likely next debugging target is not just the scalar threshold value, but the interaction among ATLIF output scaling, residual/attention blocks, and where activity regularization is applied.
