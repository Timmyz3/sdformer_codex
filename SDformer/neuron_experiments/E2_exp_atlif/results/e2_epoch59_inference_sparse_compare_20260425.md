# E2 ATLIF Epoch59 Inference and Sparse Profile

Date: 2026-04-25

## Scope

This report compares E2 ATLIF against the local PSN baseline using the same profiler, split, checkpoint epoch, and sample count.

- Split: `valid`
- Samples: `40` (`test.sample` is `40` in both configs)
- Dense ops reference: `42.63G`
- SOPs estimate: `dense_ops * global_firing_rate`
- Firing-rate definition: non-zero output ratio of hooked `Spiking_neuron` modules
- Profiler script: `tools/profile_sops.py`

## Commands

E2 ATLIF:

```bash
CONDA_PREFIX=/opt/conda/envs/sdformerflow \
PATH=/opt/conda/envs/sdformerflow/bin:$PATH \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/opt/conda/envs/sdformerflow/bin/python tools/profile_sops.py \
  --config neuron_experiments/E2_exp_atlif/configs/full_bs12w8.yml \
  --checkpoint neuron_experiments/E2_exp_atlif/results/full_bs12w8_checkpoint_epoch59.pth \
  --num-samples 40 \
  --batch-size 1 \
  --num-workers 0 \
  --dense-ops 42.63G \
  --metric AEE \
  --metric AAE \
  --output-dir neuron_experiments/E2_exp_atlif/results/profile_sops_epoch59_valid40
```

PSN baseline:

```bash
CONDA_PREFIX=/opt/conda/envs/sdformerflow \
PATH=/opt/conda/envs/sdformerflow/bin:$PATH \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/opt/conda/envs/sdformerflow/bin/python tools/profile_sops.py \
  --config third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_bs4_resume_epoch15_to60.yml \
  --checkpoint experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth \
  --num-samples 40 \
  --batch-size 1 \
  --num-workers 0 \
  --dense-ops 42.63G \
  --metric AEE \
  --metric AAE \
  --output-dir neuron_experiments/E0_psn_baseline/results/profile_sops_epoch59_valid40
```

## Same-Profiler Comparison

| Model | Config | Checkpoint | AEE ↓ | AAE ↓ | PE1 ↓ | PE2 ↓ | PE3/outliers ↓ | Global firing rate ↓ | Estimated total SOPs ↓ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| PSN baseline | `third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_bs4_resume_epoch15_to60.yml` | `experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth` | 1.5848 | 7.5012 | 0.4839 | 0.1855 | 0.0896 | 0.08496 | 3.6219G |
| E2 ATLIF | `neuron_experiments/E2_exp_atlif/configs/full_bs12w8.yml` | `neuron_experiments/E2_exp_atlif/results/full_bs12w8_checkpoint_epoch59.pth` | 4.0057 | 21.4918 | 0.8671 | 0.6006 | 0.3840 | 0.38560 | 16.4381G |

## Delta Against PSN Baseline

| Metric | E2 ATLIF - PSN | Relative change |
|---|---:|---:|
| AEE | +2.4210 | +152.76% |
| AAE | +13.9906 | +186.51% |
| PE1 | +0.3832 | +79.18% |
| PE2 | +0.4150 | +223.72% |
| PE3/outliers | +0.2944 | +328.43% |
| Global firing rate | +0.30064 | +353.86% |
| Estimated total SOPs | +12.8162G | +353.86% |

## Previous Inference Metrics

These are prior baseline inference outputs kept for reference. They were produced by the earlier inference pipeline, not by the profiler run above.

| Run | Metrics file | AEE ↓ | AAE ↓ | PE1 ↓ | PE2 ↓ | PE3/outliers ↓ |
|---|---|---:|---:|---:|---:|---:|
| PSN bs4 epoch59 previous inference | `third_party/SDformerFlow/results_compare_bs4_epoch59_20260425/98d161a3f7144441a60fa79083e0fffd/metrics_0.yml` | 1.3307 | 7.8132 | 0.4266 | 0.1526 | 0.0728 |
| Original full previous inference | `third_party/SDformerFlow/results_compare_original_full_20260425/66d1fc5322004d59a03c8ab132b11830/metrics_0.yml` | 2.3923 | 12.0129 | 0.5333 | 0.2492 | 0.1581 |

## Output Files

| Run | Summary JSON | Layer firing CSV |
|---|---|---|
| E2 ATLIF | `neuron_experiments/E2_exp_atlif/results/profile_sops_epoch59_valid40/sops_summary.json` | `neuron_experiments/E2_exp_atlif/results/profile_sops_epoch59_valid40/layer_firing_rates.csv` |
| PSN baseline | `neuron_experiments/E0_psn_baseline/results/profile_sops_epoch59_valid40/sops_summary.json` | `neuron_experiments/E0_psn_baseline/results/profile_sops_epoch59_valid40/layer_firing_rates.csv` |

## Conclusion

Under the current E2 ATLIF settings, E2 is worse than PSN on both accuracy and sparsity. Its AEE increases from `1.5848` to `4.0057`, and its estimated SOPs increase from `3.6219G` to `16.4381G`. This version should not be selected as the sparse/efficient replacement without further changes to ATLIF parameters, spike thresholding, initialization, or training schedule.
