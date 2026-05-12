# E2 Corrected ATLIF bs16w2 Epoch59 Inference and Sparse Profile

Date: 2026-04-26

## Scope

This report evaluates the corrected ATLIF full training run:

- Config: `neuron_experiments/E2_exp_atlif/configs/full_corrected_bs16w2.yml`
- Checkpoint: `neuron_experiments/E2_exp_atlif/results/full_corrected_bs16w2_checkpoint_epoch59.pth`
- Split: `valid`
- Samples: `40`
- Dense ops reference: `42.63G`
- Profiler: `tools/profile_sops.py`

## Command

```bash
CONDA_PREFIX=/opt/conda/envs/sdformerflow \
PATH=/opt/conda/envs/sdformerflow/bin:$PATH \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/opt/conda/envs/sdformerflow/bin/python tools/profile_sops.py \
  --config neuron_experiments/E2_exp_atlif/configs/full_corrected_bs16w2.yml \
  --checkpoint neuron_experiments/E2_exp_atlif/results/full_corrected_bs16w2_checkpoint_epoch59.pth \
  --num-samples 40 \
  --batch-size 1 \
  --num-workers 0 \
  --dense-ops 42.63G \
  --metric AEE \
  --metric AAE \
  --output-dir neuron_experiments/E2_exp_atlif/results/profile_sops_corrected_bs16w2_epoch59_valid40_20260426_221119
```

## Results

| Model | Config | Checkpoint | AEE ↓ | AAE ↓ | PE1 ↓ | PE2 ↓ | PE3/outliers ↓ | Firing rate ↓ | Estimated SOPs ↓ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PSN baseline | `third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_bs4_resume_epoch15_to60.yml` | `experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth` | 1.5848 | 7.5012 | 0.4839 | 0.1855 | 0.0896 | 0.08496 | 3.6219G |
| E2 ATLIF old | `neuron_experiments/E2_exp_atlif/configs/full_bs12w8.yml` | `neuron_experiments/E2_exp_atlif/results/full_bs12w8_checkpoint_epoch59.pth` | 4.0057 | 21.4918 | 0.8671 | 0.6006 | 0.3840 | 0.38560 | 16.4381G |
| E2 corrected ATLIF bs16w2 | `neuron_experiments/E2_exp_atlif/configs/full_corrected_bs16w2.yml` | `neuron_experiments/E2_exp_atlif/results/full_corrected_bs16w2_checkpoint_epoch59.pth` | 8.6602 | 67.8866 | 0.9732 | 0.8999 | 0.8019 | 0.37876 | 16.1464G |

## Output Files

| File | Path |
| --- | --- |
| Summary JSON | `neuron_experiments/E2_exp_atlif/results/profile_sops_corrected_bs16w2_epoch59_valid40_20260426_221119/sops_summary.json` |
| Layer firing CSV | `neuron_experiments/E2_exp_atlif/results/profile_sops_corrected_bs16w2_epoch59_valid40_20260426_221119/layer_firing_rates.csv` |
| Run log | `neuron_experiments/E2_exp_atlif/results/profile_sops_corrected_bs16w2_epoch59_valid40_20260426_221119.log` |

## Notes

The corrected ATLIF run reduces the global firing rate only slightly versus the previous E2 ATLIF run, from `0.38560` to `0.37876`, and the estimated SOPs from `16.4381G` to `16.1464G`. Accuracy is worse than both the PSN baseline and the previous E2 ATLIF checkpoint on this 40-sample validation profile.
