# H6a-all epoch29 valid40 inference record

Run time: 2026-05-12 03:50 UTC

## Experiment

- Experiment: H6a-all
- Config: `neuron_experiments/H6_attention_ternary_binary_highsops/configs/h6a_qk_ternary_mlp_down_binary_allparams_full.yml`
- Checkpoint: `neuron_experiments/H6_attention_ternary_binary_highsops/results/h6a_qk_ternary_mlp_down_binary_allparams_full_20260511_171228_setsid/checkpoint_epoch29.pth`
- Entry: `neuron_experiments/H6_attention_ternary_binary_highsops/entrypoints/profile_sops.py`
- Samples: valid40
- Batch size / workers: 4 / 4
- SNN backend: auto -> cupy

## Mechanism

- Attention Q/K: PSN + adaptive-threshold LIF + ternary output
- Stage0 FFN and stage0/stage2 downsample: PSN + adaptive-threshold LIF + binary output
- Trainable mode: all parameters
- Loaded ATLIF threshold summary after checkpoint restore:
  - modules: 30
  - threshold mean: 0.1118358394
  - threshold min/max: 0.1000000015 / 0.1299999952

## Metrics

| Run | Samples | Global firing rate | Est. SOPs | AEE | AAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| PSN baseline epoch59 | 40 | 0.08496059 | 3.6219G | 1.584776 | 7.501204 |
| H6a-all epoch29 | 40 | 0.06492980 | 2.7680G | 1.594698 | 21.636660 |

## Delta vs baseline

| Metric | Delta | Relative |
| --- | ---: | ---: |
| Global firing rate | -0.02003080 | -23.58% |
| Est. SOPs | -0.8539G | -23.58% |
| AEE | +0.009921 | +0.63% |
| AAE | +14.135456 | +188.44% |

## Output files

- Summary: `sops_summary.json`
- Layer firing rates: `layer_firing_rates.csv`

## Notes

This checkpoint gives a clear sparsity/SOP reduction while keeping AEE close to the PSN baseline on valid40. AAE is much worse than baseline, so this run should not be described as globally better without addressing angular error.
