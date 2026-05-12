
# Plan A ATLIF Conservative Baseline-Param Transfer

Date: 2026-04-30 UTC

## Goal

Test ATLIF again with a conservative transfer setup: keep the PSN baseline checkpoint, restore baseline-like optimizer and loader settings, and make ATLIF threshold/activity tuning much weaker than the previous full run.

This is a short 20-epoch run, not a full 60-epoch run. It should be profiled before deciding whether to continue.

## Config

| field | value |
| --- | --- |
| config | `neuron_experiments/E2_exp_atlif/configs/planA_pretrained_baselineparams_lr1e-5_bs4_lrs100_eta3e-4_act1e-6.yml` |
| init checkpoint | `experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth` |
| neuron | `exp_atlif_official` |
| epochs | 20 |
| batch size | 4 |
| workers | 4 |
| AMP | true |
| pin memory | true |
| optimizer lr | `1e-5` |
| scheduler milestones | `[5, 10, 15]` |
| ATLIF threshold eta | `3e-4` |
| threshold lr scale | `100` |
| activity eta | `1e-6` |
| threshold grad sanitize | true |

Note: PSN's `v_th=0.1` and `tau=2` were not copied into ATLIF. The official ATLIF wrapper keeps `v_th=1.0` and `tau=0.9`, because those parameters are not semantically equivalent to PSN's threshold/tau in this integration.

## Running Command

```bash
setsid env SDFORMER_USE_MLFLOW=0 /opt/conda/envs/sdformerflow/bin/python neuron_experiments/E2_exp_atlif/entrypoints/train.py \
  --config /root/private_data/work/sdformer_codex/SDformer/neuron_experiments/E2_exp_atlif/configs/planA_pretrained_baselineparams_lr1e-5_bs4_lrs100_eta3e-4_act1e-6.yml \
  --prev_runid /root/private_data/work/sdformer_codex/SDformer/experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth \
  --save_path /root/private_data/work/sdformer_codex/SDformer/neuron_experiments/E2_exp_atlif/results/planA_pretrained_baselineparams_lr1e-5_bs4_lrs100_eta3e-4_act1e-6_20260430_011426_checkpoint_epoch{}.pth
```

## Runtime

| field | value |
| --- | --- |
| PID | `1331800` |
| log | `neuron_experiments/E2_exp_atlif/results/planA_pretrained_baselineparams_lr1e-5_bs4_lrs100_eta3e-4_act1e-6_20260430_011426.log` |
| checkpoint pattern | `neuron_experiments/E2_exp_atlif/results/planA_pretrained_baselineparams_lr1e-5_bs4_lrs100_eta3e-4_act1e-6_20260430_011426_checkpoint_epoch{}.pth` |

## Decision Rule

After training, run `tools/profile_sops.py` on the best validation checkpoint and the final checkpoint with `valid40`, `AEE`, `AAE`, global firing rate, and SOPs.

Baseline target:

| run | AEE | AAE | firing rate | SOPs |
| --- | ---: | ---: | ---: | ---: |
| PSN baseline epoch59 valid40 | 1.5848 | 7.5012 | 0.08496 | 3.6219G |

Continue this ATLIF branch only if it moves toward both:

| metric | desired direction |
| --- | --- |
| AEE | close to baseline, ideally below `2.0` in valid40 |
| AAE | close to baseline |
| global firing rate | below previous ATLIF `0.122`, preferably near or below baseline `0.085` |
| SOPs | below previous ATLIF `5.206G`, preferably near or below baseline `3.622G` |

## Training Result

| epoch | train loss | valid loss | lr |
| --- | ---: | ---: | ---: |
| 0 | 8.4757 | 6.9979 | 1.0e-5 |
| 5 | 6.3751 | 6.2700 | 5.0e-6 |
| 10 | 5.9045 | 5.7415 | 2.5e-6 |
| 15 | 5.6730 | 5.7454 | 1.25e-6 |
| 19 | 5.5395 | not run | 1.25e-6 |

Best observed validation checkpoint: epoch10.

## Inference/Profile Commands

Epoch10:

```bash
SDFORMER_USE_MLFLOW=0 /opt/conda/envs/sdformerflow/bin/python tools/profile_sops.py \
  --config neuron_experiments/E2_exp_atlif/configs/planA_pretrained_baselineparams_lr1e-5_bs4_lrs100_eta3e-4_act1e-6.yml \
  --checkpoint neuron_experiments/E2_exp_atlif/results/planA_pretrained_baselineparams_lr1e-5_bs4_lrs100_eta3e-4_act1e-6_20260430_011426_checkpoint_epoch10.pth \
  --output-dir neuron_experiments/E2_exp_atlif/results/profile_sops_planA_epoch10_valid40_20260430_121156 \
  --split valid --num-samples 40 --batch-size 1 --num-workers 0 \
  --dense-ops 42.63G --metric AEE --metric AAE
```

Epoch19:

```bash
SDFORMER_USE_MLFLOW=0 /opt/conda/envs/sdformerflow/bin/python tools/profile_sops.py \
  --config neuron_experiments/E2_exp_atlif/configs/planA_pretrained_baselineparams_lr1e-5_bs4_lrs100_eta3e-4_act1e-6.yml \
  --checkpoint neuron_experiments/E2_exp_atlif/results/planA_pretrained_baselineparams_lr1e-5_bs4_lrs100_eta3e-4_act1e-6_20260430_011426_checkpoint_epoch19.pth \
  --output-dir neuron_experiments/E2_exp_atlif/results/profile_sops_planA_epoch19_valid40_20260430_121249 \
  --split valid --num-samples 40 --batch-size 1 --num-workers 0 \
  --dense-ops 42.63G --metric AEE --metric AAE
```

## Inference/Profile Outputs

| run | summary | layer firing CSV |
| --- | --- | --- |
| Plan A epoch10 | `neuron_experiments/E2_exp_atlif/results/profile_sops_planA_epoch10_valid40_20260430_121156/sops_summary.json` | `neuron_experiments/E2_exp_atlif/results/profile_sops_planA_epoch10_valid40_20260430_121156/layer_firing_rates.csv` |
| Plan A epoch19 | `neuron_experiments/E2_exp_atlif/results/profile_sops_planA_epoch19_valid40_20260430_121249/sops_summary.json` | `neuron_experiments/E2_exp_atlif/results/profile_sops_planA_epoch19_valid40_20260430_121249/layer_firing_rates.csv` |

## Inference/Profile Metrics

| run | AEE | AAE | AEE_PE1 | AEE_PE2 | AEE_PE3/outliers | firing rate | estimated SOPs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PSN baseline epoch59 valid40 | 1.5848 | 7.5012 | 0.4839 | 0.1855 | 0.0896 | 0.08496 | 3.6219G |
| previous ATLIF full-pretrained epoch59 | 2.5128 | 12.5417 | 0.7203 | 0.3587 | 0.1895 | 0.12212 | 5.2062G |
| Plan A epoch10 | 5.6600 | 27.6559 | 0.9145 | 0.7298 | 0.5544 | 0.16096 | 6.8619G |
| Plan A epoch19 | 5.6760 | 29.2109 | 0.9067 | 0.7151 | 0.5466 | 0.16163 | 6.8903G |

## Delta Versus Baseline

Positive delta means worse for AEE/AAE/outlier metrics and more active for firing/SOPs.

| run | AEE delta | AEE ratio | AAE delta | AAE ratio | firing delta | firing ratio | SOPs delta | SOPs ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Plan A epoch10 | +4.0752 | 3.571x | +20.1547 | 3.687x | +0.07600 | 1.895x | +3.2401G | 1.895x |
| Plan A epoch19 | +4.0912 | 3.582x | +21.7097 | 3.894x | +0.07667 | 1.902x | +3.2684G | 1.902x |

## Threshold Notes

| checkpoint | ATLIF modules | min threshold | p25 | mean threshold | p75 | max threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Plan A epoch10 | 105 | 0.998902 | 1.000000 | 1.081431 | 1.042216 | 1.696176 |
| Plan A epoch19 | 105 | 0.998902 | 1.000000 | 1.095459 | 1.050871 | 1.748445 |

## Result

Plan A did not work. It is worse than both the PSN baseline and the previous ATLIF full-pretrained run. The conservative baseline-like optimizer preserved neither accuracy nor sparsity:

| comparison | result |
| --- | --- |
| accuracy | AEE stayed around `5.66`, much worse than baseline `1.58` and previous ATLIF `2.51` |
| sparsity | global firing stayed around `0.161`, worse than baseline `0.085` and previous ATLIF `0.122` |
| SOPs | estimated SOPs increased to about `6.89G`, worse than baseline `3.62G` and previous ATLIF `5.21G` |

The weak threshold update left most thresholds close to `1.0`, so the intended activity pruning did not activate. This suggests that a simple conservative transfer is not enough for this ATLIF integration. The next useful check should be a staged run: first freeze or nearly freeze the backbone and tune thresholds/output behavior only, or explicitly warm start from the previous better ATLIF full-pretrained checkpoint rather than directly from PSN.
