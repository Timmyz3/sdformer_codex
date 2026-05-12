# H2 Adaptive Ternary PSN

This experiment fuses three ideas while keeping the baseline skeleton intact:

- PSN temporal mixing: reuse baseline PSN `weight` and `bias`.
- Ternary spike: official Ternary-Spike-style `{-1, 0, +1}` activation with clamp STE.
- AT-LIF-style adaptive threshold: learn a positive `theta` and emit `spike * theta`.

The first implementation is attention-only. It replaces `sn_q` and `sn_k` inside
Swin attention blocks, not every spiking neuron in the network.

## Files

| role | file |
| --- | --- |
| train entrypoint | `entrypoints/train.py` |
| eval entrypoint | `entrypoints/eval.py` |
| SOP/profile entrypoint | `entrypoints/profile_sops.py` |
| neuron | `overlay/models/STSwinNet_SNN/adaptive_ternary/adaptive_ternary_psn.py` |
| installer | `overlay/models/STSwinNet_SNN/adaptive_ternary/installer.py` |
| smoke config | `configs/smoke.yml` |
| short config | `configs/short.yml` |
| full config | `configs/full.yml` |
| tests | `tests/test_adaptive_ternary_psn.py`, `tests/test_entrypoint_patch.py` |

## Commands

Smoke:

```bash
SDFORMER_USE_MLFLOW=0 SDFORMER_MLFLOW_MODEL_LOGGING=0 \
/opt/conda/envs/sdformerflow/bin/python neuron_experiments/H2_adaptive_ternary_psn/entrypoints/train.py \
  --config neuron_experiments/H2_adaptive_ternary_psn/configs/smoke.yml \
  --prev_runid experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth \
  --save_path neuron_experiments/H2_adaptive_ternary_psn/results/h2_smoke_checkpoint_epoch{}.pth
```

Short:

```bash
SDFORMER_USE_MLFLOW=0 SDFORMER_MLFLOW_MODEL_LOGGING=0 \
/opt/conda/envs/sdformerflow/bin/python neuron_experiments/H2_adaptive_ternary_psn/entrypoints/train.py \
  --config neuron_experiments/H2_adaptive_ternary_psn/configs/short.yml \
  --prev_runid experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth \
  --save_path neuron_experiments/H2_adaptive_ternary_psn/results/h2_short_checkpoint_epoch{}.pth
```

Full:

```bash
SDFORMER_USE_MLFLOW=0 SDFORMER_MLFLOW_MODEL_LOGGING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/opt/conda/envs/sdformerflow/bin/python neuron_experiments/H2_adaptive_ternary_psn/entrypoints/train.py \
  --config neuron_experiments/H2_adaptive_ternary_psn/configs/full.yml \
  --prev_runid experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth \
  --save_path neuron_experiments/H2_adaptive_ternary_psn/results/h2_full_checkpoint_epoch{}.pth
```

Previous full run:

```bash
tail -f neuron_experiments/H2_adaptive_ternary_psn/results/h2_full_bs4w8_amp_20260509_022110.log
```

| item | value |
| --- | --- |
| start time UTC | 2026-05-09 02:21:10 |
| pid wrapper | `346812` |
| pid python | `346814` |
| config | `configs/full.yml` |
| init checkpoint | `experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth` |
| save pattern | `results/h2_full_bs4w8_amp_20260509_022110_checkpoint_epoch{}.pth` |
| setting | batch 4, workers 8, AMP on, TF32 on, pin_memory false |

This run was stopped at user request before epoch 0 completed so larger batch
speed tests could be run.

## Speed Sweep

Date: 2026-05-09

All runs used H2 full model, AMP, TF32, 8 workers, `pin_memory: false`, and
`runtime.max_train_steps: 120`.

| batch | train samples/sec | max GPU GiB | decision |
| ---: | ---: | ---: | --- |
| 8 | 9.3662 | 38.736 | stable |
| 12 | 9.8531 | 57.943 | stable fallback |
| 16 | 10.0783 | 77.098 | fastest |

Active full run:

```bash
tail -f neuron_experiments/H2_adaptive_ternary_psn/results/h2_full_bs16w8_amp_20260509_024107.log
```

| item | value |
| --- | --- |
| start time UTC | 2026-05-09 02:41:07 |
| supervisor PID | `356535` |
| primary config | `configs/full_bs16.yml` |
| primary log | `results/h2_full_bs16w8_amp_20260509_024107.log` |
| primary save pattern | `results/h2_full_bs16w8_amp_20260509_024107_checkpoint_epoch{}.pth` |
| fallback config | `configs/full_bs12.yml` |
| fallback log | `results/h2_full_bs12w8_amp_fallback_20260509_024107.log` |
| supervisor log | `results/h2_full_best_supervisor_20260509_024107.log` |
| fallback policy | if bs16 exits nonzero, auto-start bs12 |

Completed: the bs16 run naturally stopped after the configured 20 epochs
(`epoch0` through `epoch19`) and saved
`results/h2_full_bs16w8_amp_20260509_024107_checkpoint_epoch19.pth`.

## Inference Result

Date: 2026-05-09

Report:

`results/h2_epoch19_inference_report_20260509.md`

| run | samples | AEE ↓ | AAE ↓ | firing ↓ | SOPs ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| PSN baseline epoch59 | 40 | 1.5848 | 7.5012 | 0.08496 | 3.6219G |
| G1 best valid40 | 40 | 1.6056 | 7.2452 | 0.06365 | 2.7134G |
| H2 adaptive ternary PSN epoch19 | 40 | 1.7949 | 8.8486 | 0.20373 | 8.6849G |

Conclusion: H2 epoch19 is worse than PSN baseline on both accuracy and
sparsity. The all-Q/K ternary replacement makes many attention branches nearly
always nonzero, so this version should not be selected as the sparse-efficient
neuron story.

## Design Notes

`H2a` is the default config with `reg_lambda: 0.0`, because the first question is
whether attention Q/K ternary PSN preserves accuracy. `H2b/H2c` can be made by
turning on `target_rate` and `reg_lambda` after short-run accuracy is acceptable.

## Smoke Result

Date: 2026-05-08

Command: see `results/h2_smoke_20260508.log`.

| item | value |
| --- | ---: |
| installed modules | 4 Q/K modules (`layer0_only`) |
| train steps | 2 |
| train loss | 1.0975 |
| validation samples | 1 |
| validation loss | 0.8036 |
| max GPU memory | 10.789 GiB |
| mean theta | 1.0000 |
| mean ternary activity | 0.0168 |

Checkpoint:

`results/h2_smoke_checkpoint_epoch0.pth`
