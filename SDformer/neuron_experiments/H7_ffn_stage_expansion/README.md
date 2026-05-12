# H7 FFN Stage Expansion

H7 extends H6a by replacing FFN/MLP spike nodes in one additional encoder
stage at a time. The goal is to test whether later FFN stages can reduce SOPs
with less optical-flow accuracy loss than expanding attention projection.

## Rationale

Baseline valid40 sensitivity shows that stage0 FFN is a strong first target:
the four `layers.0.*.mlp.sn{1,2}` nodes contribute about 14.38% SOPs, while a
hard zero-ablation only raised AEE by about 3.02%. H6a already replaced those
stage0 FFN nodes with binary ATLIF-PSN and kept Q/K ternary.

H7 keeps that H6a core and adds one extra FFN stage per config:

| config | Q/K | FFN binary targets | downsample binary targets | trainable |
|---|---|---|---|---|
| `h7_stage01_ffn_binary_80.yml` | all Q/K ternary | stage0 + stage1 FFN | stage0 + stage2 | `atlif_only` |
| `h7_stage02_ffn_binary_80.yml` | all Q/K ternary | stage0 + stage2 FFN | stage0 + stage2 | `atlif_only` |
| `h7_stage03_ffn_binary_80.yml` | all Q/K ternary | stage0 + stage3 FFN | stage0 + stage2 | `atlif_only` |

All FFN expansions use `output_mode: binary`, so they emit only `0/+threshold`.
This avoids H5's signed ternary failure mode where negative spikes made
high-SOP layers dense.

## Why Not Replace Every FFN Yet

The experiment is staged so each run has an interpretable answer:

- stage1 tests the next highest-resolution FFN block after stage0.
- stage2 tests the deepest, largest block group, but with weaker sparsity.
- stage3 tests late semantic FFN nodes with the smallest spatial size.

If one extra stage is clearly stable, the next experiment can combine the best
two stages and optionally switch `trainable` from `atlif_only` to `all` so the
backbone can adapt to the new sparse activation distribution.

## Commands

Train a short 80-step probe:

```bash
SDFORMER_USE_MLFLOW=0 python neuron_experiments/H7_ffn_stage_expansion/entrypoints/train.py \
  --config neuron_experiments/H7_ffn_stage_expansion/configs/<config>.yml \
  --prev_runid experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth \
  --save_path neuron_experiments/H7_ffn_stage_expansion/results/<name>_checkpoint_epoch{}.pth
```

Profile:

```bash
python neuron_experiments/H7_ffn_stage_expansion/entrypoints/profile_sops.py \
  --config neuron_experiments/H7_ffn_stage_expansion/configs/<config>.yml \
  --checkpoint neuron_experiments/H7_ffn_stage_expansion/results/<checkpoint>.pth \
  --output-dir neuron_experiments/H7_ffn_stage_expansion/results/<profile_dir> \
  --split valid \
  --num-samples 10 \
  --batch-size 1 \
  --num-workers 4 \
  --dense-ops 42.63G \
  --metric AEE \
  --metric AAE \
  --module-pattern Spiking_neuron
```

## Results

Short probe date: 2026-05-11. All three probes start from
`experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth`,
train for `max_train_steps: 80`, then run valid10 SOP/metric profiling on the
epoch0 checkpoint produced by the short probe.

Reference rows:

| run | valid split | firing | SOPs | AEE | AAE |
|---|---:|---:|---:|---:|---:|
| H4h Q/K ATLIF-PSN reference | valid10 | 0.087457 | 3.7283G | 1.027874 | 6.088951 |
| H6a Q/K ternary + stage0 FFN/downsample binary | valid10 | 0.086604 | 3.6919G | 1.049870 | 6.106936 |

H7 short probes:

| run | added FFN stage beyond H6a | installed modules | trainable params | train step sec | firing | SOPs | AEE | AAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `h7_stage01_ffn_binary_80` | stage1 | 34 | 1,278 | 1.7225 | 0.083854 | 3.5747G | 1.074384 | 6.388056 |
| `h7_stage02_ffn_binary_80` | stage2 | 42 | 2,166 | 1.7375 | 0.082757 | 3.5279G | 1.100928 | 6.760951 |
| `h7_stage03_ffn_binary_80` | stage3 | 34 | 1,278 | 1.5976 | 0.084142 | 3.5870G | 1.102599 | 6.741123 |

Layer-group firing rates from valid10:

| run | Q/K | proj | stage0 FFN | stage1 FFN | stage2 FFN | stage3 FFN | downsample |
|---|---:|---:|---:|---:|---:|---:|---:|
| `h7_stage01_ffn_binary_80` | 0.038001 | 0.103367 | 0.131414 | 0.061587 | 0.101837 | 0.092950 | 0.189695 |
| `h7_stage02_ffn_binary_80` | 0.038239 | 0.101746 | 0.130239 | 0.081411 | 0.086906 | 0.091683 | 0.190773 |
| `h7_stage03_ffn_binary_80` | 0.038013 | 0.104252 | 0.130329 | 0.080176 | 0.104044 | 0.076447 | 0.194834 |

Current read:

- Stage1 expansion is the least damaging H7 extension, but still loses against
  H6a on valid10 accuracy while saving another 0.1172G SOPs.
- Stage2 expansion saves the most SOPs, but the short probe hurts AEE/AAE too
  much to justify a full run without weakening the FFN sparse penalty or
  allowing full-network adaptation.
- Stage3 expansion does not support the initial hypothesis in this 80-step
  probe: it sparsifies stage3 FFN locally, but gives less global SOP reduction
  than stage1/stage2 and worse AEE than stage1.

Recommended next step: run a stage1-only FFN expansion with `trainable: all`,
or keep `trainable: atlif_only` but reduce the added FFN stage activity
regularization. The frozen-backbone H7 variants are useful as sensitivity
probes, but none of the three should be promoted to a full run unchanged.
