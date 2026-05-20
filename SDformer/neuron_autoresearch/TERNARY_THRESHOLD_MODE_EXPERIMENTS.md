# H12 Ternary Threshold Mode Experiments

Date: 2026-05-18

Goal: separate the old asymmetric negative-threshold control from the BSA/TSN
symmetric ternary firing paradigm, while keeping the SDFormerFlow baseline
untouched.

## Shared Entry

All three schemes use:

- Train entry: `neuron_experiments/H9_bipolar_self_attention/entrypoints/train.py`
- Profile entry: `neuron_experiments/H9_bipolar_self_attention/entrypoints/profile_sops.py`
- Neuron code: `neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/atlif_ternary_psn/atlif_ternary_psn.py`
- Installer/config parser: `neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/atlif_ternary_psn/installer.py`

The baseline under `third_party/SDformerFlow` is not modified.

## Schemes

| ID | Config | `threshold_mode` | Firing rule | Purpose |
|---|---|---|---|---|
| H12a | `h12a_asymmetric_scale_ablation_full.yml` | `asymmetric_scale` | `pos: mem >= theta`, `neg: mem <= -theta * scale` | Old negative-scale control, kept only as ablation |
| H12b | `h12b_symmetric_bsa_tsn_full.yml` | `symmetric_bsa_tsn` | `sign(mem) * I(abs(mem) >= theta) * theta` | BSA/TSN-style symmetric ternary firing |
| H12c | `h12c_symmetric_target_rate_full.yml` | `symmetric_target_rate` | Same as H12b, plus total firing-rate threshold feedback | Symmetric ternary with ATLIF sparsity control |

Smoke configs with the same names ending in `_smoke.yml` are provided for fast
connectivity checks.

## Commands

Smoke:

```bash
/opt/conda/envs/sdformerflow/bin/python \
  neuron_experiments/H9_bipolar_self_attention/entrypoints/train.py \
  --config neuron_experiments/H9_bipolar_self_attention/configs/h12b_symmetric_bsa_tsn_smoke.yml \
  --prev_runid experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth \
  --save_path neuron_experiments/H9_bipolar_self_attention/results/h12b_smoke/checkpoint_epoch{}.pth
```

Full:

```bash
/opt/conda/envs/sdformerflow/bin/python \
  neuron_experiments/H9_bipolar_self_attention/entrypoints/train.py \
  --config neuron_experiments/H9_bipolar_self_attention/configs/h12c_symmetric_target_rate_full.yml \
  --prev_runid experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth \
  --save_path neuron_experiments/H9_bipolar_self_attention/results/h12c_full/checkpoint_epoch{}.pth
```

## Interpretation

H12a is not a proposed method. It is the old control that shows why the large
negative threshold scale suppresses negative spikes.

H12b is the clean BSA/TSN fusion with ATLIF: BSA/TSN supplies symmetric
ternary polarity, ATLIF supplies adaptive threshold magnitude.

H12c adds a total firing-rate target so sparsity is controlled by threshold
growth rather than by killing only the negative branch.
