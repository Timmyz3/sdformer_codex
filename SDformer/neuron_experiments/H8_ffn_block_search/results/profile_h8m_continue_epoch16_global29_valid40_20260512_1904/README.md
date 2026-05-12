# H8m continuation epoch16/global29 valid40 inference record

Run time: 2026-05-12 19:04 UTC

## Experiment

- Experiment: H8m, promoted full run continuation
- Local checkpoint: `checkpoint_epoch16.pth`
- Global interpretation: original H8m epoch12 + continuation epoch16 ~= global epoch29
- Config: `neuron_experiments/H8_ffn_block_search/configs/generated_full/h8m_stage3_block0_continue_from_epoch12_20260512_034520.yml`
- Checkpoint: `neuron_experiments/H8_ffn_block_search/results/h8m_stage3_block0_continue_from_epoch12_20260512_1210_nomlflowmodel_setsid/checkpoint_epoch16.pth`
- Entry: `neuron_experiments/H8_ffn_block_search/entrypoints/profile_sops.py`
- Samples: valid40
- Batch size / workers: 4 / 4
- SNN backend: auto -> cupy

## Mechanism

- Attention Q/K: PSN + ATLIF + ternary
- stage0 FFN + selected high-SOP FFN/downsample: PSN + ATLIF + binary
- H8m added stage3 block0 FFN binary replacement beyond the H6a core.
- Trainable mode: all parameters

Loaded ATLIF summary after checkpoint restore:

- modules: 32
- threshold mean: 0.1021457533
- threshold min/max: 0.1000000015 / 0.1082660332

## Metrics

| Run | Samples | Global firing rate | Est. SOPs | AEE | AAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| PSN baseline epoch59 | 40 | 0.08496059 | 3.6219G | 1.584776 | 7.501204 |
| H6a frozen epoch11 | 40 | 0.07743447 | 3.3010G | 1.553494 | 8.200176 |
| H6a all-params epoch29 | 40 | 0.06492980 | 2.7680G | 1.594698 | 21.636660 |
| H8m continuation epoch16/global29 | 40 | 0.069890 | 2.9794G | 1.596608 | 22.794633 |

## Delta vs baseline

| Metric | Delta | Relative |
| --- | ---: | ---: |
| Global firing rate | -0.015071 | -17.74% |
| Est. SOPs | -0.6425G | -17.74% |
| AEE | +0.011832 | +0.75% |
| AAE | +15.293429 | +203.88% |

## Read

The final H8m continuation keeps AEE close to the PSN baseline and reduces SOPs, but AAE remains severely worse. This matches the H6a all-params failure mode and supports the hypothesis that Q/K ternary replacement without the full BSA normalization path, such as Shiftmax, destabilizes direction-sensitive optical flow estimates.

## Output files

- Summary: `sops_summary.json`
- Layer firing rates: `layer_firing_rates.csv`
