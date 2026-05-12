# H2 Epoch19 Inference Report

Date: 2026-05-09

## Run

H2 was evaluated with the same valid40 SOP profiling protocol used for the
PSN baseline and G1. The profiler hooks `Spiking_neuron` wrappers only, so H2
Q/K AdaptiveTernaryPSN outputs are counted once through their SDFormerFlow
wrapper, matching the baseline counting convention.

```bash
SDFORMER_USE_MLFLOW=0 SDFORMER_MLFLOW_MODEL_LOGGING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/opt/conda/envs/sdformerflow/bin/python neuron_experiments/H2_adaptive_ternary_psn/entrypoints/profile_sops.py \
  --config neuron_experiments/H2_adaptive_ternary_psn/configs/full_bs16.yml \
  --checkpoint neuron_experiments/H2_adaptive_ternary_psn/results/h2_full_bs16w8_amp_20260509_024107_checkpoint_epoch19.pth \
  --output-dir neuron_experiments/H2_adaptive_ternary_psn/results/profile_sops_h2_bs16_epoch19_valid40_baselinehook_20260509 \
  --split valid \
  --num-samples 40 \
  --batch-size 1 \
  --num-workers 4 \
  --dense-ops 42.63G \
  --metric AEE \
  --metric AAE \
  --module-pattern Spiking_neuron
```

Outputs:

| file | path |
| --- | --- |
| summary | `results/profile_sops_h2_bs16_epoch19_valid40_baselinehook_20260509/sops_summary.json` |
| layer firing | `results/profile_sops_h2_bs16_epoch19_valid40_baselinehook_20260509/layer_firing_rates.csv` |
| log | `results/profile_sops_h2_bs16_epoch19_valid40_baselinehook_20260509.log` |

## Metrics

| run | samples | AEE ↓ | AAE ↓ | firing ↓ | SOPs ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| PSN baseline epoch59 | 40 | 1.5848 | 7.5012 | 0.08496 | 3.6219G |
| G1 best valid40 | 40 | 1.6056 | 7.2452 | 0.06365 | 2.7134G |
| H2 adaptive ternary PSN epoch19 | 40 | 1.7949 | 8.8486 | 0.20373 | 8.6849G |

Compared with PSN baseline:

| metric | delta | ratio |
| --- | ---: | ---: |
| AEE | +0.2101 | 1.133x |
| AAE | +1.3474 | 1.180x |
| firing | +0.11877 | 2.398x |
| SOPs | +5.0630G | 2.398x |

## H2 State

Loaded H2 summary:

```text
theta_mean=0.9976
theta_min=0.9773
theta_max=1.0033
activity_mean=0.8703
pos_mean=0.0152
neg_mean=0.8551
```

Top firing layers show the main issue is that many attention Q/K paths are
almost always nonzero after the ternary replacement:

| firing | layer |
| ---: | --- |
| 1.000000 | `sttmultires_unet.encoders.swin3d.layers.1.swin_blocks.0.attn.sn2_q` |
| 0.999964 | `sttmultires_unet.encoders.swin3d.layers.1.swin_blocks.0.attn.sn_k` |
| 0.998831 | `sttmultires_unet.encoders.swin3d.layers.1.swin_blocks.1.attn.sn2_q` |
| 0.996717 | `sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.2.attn.sn2_q` |
| 0.980938 | `sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.3.attn.sn_q` |
| 0.969726 | `sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.3.attn.sn_k` |

## Conclusion

H2 epoch19 does not support the sparse-efficient story. It loses accuracy and
raises firing/SOPs substantially. The learned threshold stayed near 1.0 and did
not create sparsity; in fact the ternary Q/K branches are mostly negative
nonzero spikes. This suggests the next H2 variant should not simply use
`sign(x)` with a fixed dead zone. It needs either a much stronger adaptive
dead-zone/threshold schedule, a target-rate regularizer, or a narrower local
replacement instead of all attention Q/K blocks.
