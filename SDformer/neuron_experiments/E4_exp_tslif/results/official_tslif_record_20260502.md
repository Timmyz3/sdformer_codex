# E4 Official TS-LIF Record

## Source

| item | value |
| --- | --- |
| official repo | `https://github.com/kkking-kk/TS-LIF` |
| local source | `/root/private_data/work/optimization_sources/neuron_optimization/TSLIF_TS-LIF` |
| commit | `a59826a6c7f62d0f16edbafdbb28db65bebd9f69` |
| official neuron file | `TS-LIF/SeqSNN/network/snn/TSLIF.py` |
| official surrogate file | `TS-LIF/SeqSNN/network/snn/surrogate.py` |

## Code Files

| role | file |
| --- | --- |
| official TS-LIF core and SDFormer wrapper | `neuron_experiments/E4_exp_tslif/overlay/models/STSwinNet_SNN/experimental_neurons/single/official_tslif.py` |
| stable import path | `neuron_experiments/E4_exp_tslif/overlay/models/STSwinNet_SNN/experimental_neurons/single/tslif.py` |
| experiment neuron factory | `neuron_experiments/E4_exp_tslif/overlay/models/STSwinNet_SNN/experimental_neurons/factory.py` |
| shared experiment neuron base | `neuron_experiments/E4_exp_tslif/overlay/models/STSwinNet_SNN/experimental_neurons/base.py` |
| overlay spiking modules | `neuron_experiments/E4_exp_tslif/overlay/models/STSwinNet_SNN/Spiking_modules.py` |
| train entrypoint | `neuron_experiments/E4_exp_tslif/entrypoints/train.py` |
| eval entrypoint | `neuron_experiments/E4_exp_tslif/entrypoints/eval.py` |
| unit test | `neuron_experiments/E4_exp_tslif/tests/test_official_tslif.py` |

No files under `third_party/SDformerFlow` were edited.

## Porting Notes

The previous E4 `tslif.py` was a hand-written approximation. It has been replaced by an official-source implementation:

- official two-state update: `v1 = d0 * v1 + d1 * x - yy * v2`, `v2 = d2 * v2 + d3 * x - kk * v1`
- official two-branch firing with TS-LIF `atan` surrogate
- official soft reset mapping: short state reset by long spike, long state reset by short spike
- trainable `decay_factor`, `kk`, `yy`, `alpha_s`, `alpha_l`

SDFormer-specific adaptation:

- official TS-LIF uses fixed `alpha_s/alpha_l` shapes such as `[1, 128]`; SDFormer creates neurons before feature channel sizes are known, so this experiment uses trainable scalar `alpha_s/alpha_l` broadcast over the feature tensor. This keeps the official weighted two-branch output and guarantees the parameters exist before optimizer construction.
- wrapper accepts SDFormerFlow `[T, B, ...]` tensors and resets state at the start of each independent forward call.
- wrapper subclasses spikingjelly-compatible memory modules to avoid backend/reset warning spam.

## Verification

Unit test:

```bash
/opt/conda/envs/sdformerflow/bin/python neuron_experiments/E4_exp_tslif/tests/test_official_tslif.py
```

Result: 2 tests passed.

Clean smoke command:

```bash
SDFORMER_USE_MLFLOW=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/opt/conda/envs/sdformerflow/bin/python neuron_experiments/E4_exp_tslif/entrypoints/train.py \
  --config neuron_experiments/E4_exp_tslif/configs/smoke.yml \
  --prev_runid /root/private_data/work/sdformer_codex/SDformer/experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth \
  --save_path /root/private_data/work/sdformer_codex/SDformer/neuron_experiments/E4_exp_tslif/results/e4_official_smoke_clean_20260502_checkpoint_epoch{}.pth
```

Smoke result from `e4_official_smoke_clean_20260502.log`:

| metric | value |
| --- | ---: |
| train loss | 9.8472 |
| validation loss | 6.2777 |
| train step time | 2.5093 sec |
| train samples/sec | 0.3985 |
| max GPU memory | 18.874 GiB |

## Tuning Results

All tuning runs used:

- PSN baseline init: `experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth`
- split: `train_backend_benchmark_split_seq.csv` and `valid_backend_benchmark_split_seq.csv`
- AMP enabled
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `runtime.allow_tf32=true`
- `runtime.cudnn_benchmark=true`

| config | status | train samples/sec | train step sec | epoch sec | max GPU GiB | note |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `tune_bs2w4_pinfalse_amp.yml` | pass | 1.9077 | 1.0484 | 67.10 | 37.956 | slower |
| `tune_bs4w2_pinfalse_amp.yml` | pass | 2.5879 | 1.5457 | 49.46 | 75.854 | stable |
| `tune_bs4w4_pinfalse_amp.yml` | pass | 2.5925 | 1.5429 | 49.37 | 75.854 | stable |
| `tune_bs4w8_pinfalse_amp.yml` | pass | 2.6057 | 1.5351 | 49.12 | 75.854 | selected |
| `tune_bs4w4_pintrue_amp.yml` | pass | 2.5419 | 1.5736 | 50.36 | 75.854 | pin slower |
| `tune_bs5w4_pinfalse_amp.yml` | OOM | - | - | - | ~79.22 | no margin |
| `tune_bs6w4_pinfalse_amp.yml` | OOM | - | - | - | ~79.24 | no margin |
| `tune_bs8w4_pinfalse_amp.yml` | OOM | - | - | - | ~79.18 | no margin |
| `tune_bs10w4_pinfalse_amp.yml` | OOM | - | - | - | ~79.20 | no margin |
| `tune_bs12w4_pinfalse_amp.yml` | OOM | - | - | - | ~79.24 | no margin |
| `tune_bs16w4_pinfalse_amp.yml` | OOM | - | - | - | ~79.13 | no margin |

## Full Run

Selected config:

`neuron_experiments/E4_exp_tslif/configs/full.yml`

Selected full setting:

| setting | value |
| --- | --- |
| batch size | 4 |
| workers | 8 |
| AMP | true |
| pin_memory | false |
| TF32 | true |
| cudnn benchmark | true |
| weight decay | 0.0 |
| v_th | 1.0 |
| v_reset | 0.0 |
| detach_reset | false |

Launch command:

```bash
setsid env SDFORMER_USE_MLFLOW=0 SDFORMER_MLFLOW_MODEL_LOGGING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/opt/conda/envs/sdformerflow/bin/python neuron_experiments/E4_exp_tslif/entrypoints/train.py \
  --config neuron_experiments/E4_exp_tslif/configs/full.yml \
  --prev_runid /root/private_data/work/sdformer_codex/SDformer/experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth \
  --save_path /root/private_data/work/sdformer_codex/SDformer/neuron_experiments/E4_exp_tslif/results/e4_official_tslif_full_bs4w8_amp_pinfalse_tf32_20260502_checkpoint_epoch{}.pth \
  > neuron_experiments/E4_exp_tslif/results/e4_official_tslif_full_bs4w8_amp_pinfalse_tf32_20260502.log 2>&1 < /dev/null &
```

Full training status checked on 2026-05-04:

| checkpoint | train loss | note |
| --- | ---: | --- |
| epoch59 | 1.406976 | final saved checkpoint |

Final checkpoint:

`neuron_experiments/E4_exp_tslif/results/e4_official_tslif_full_bs4w8_amp_pinfalse_tf32_20260502_checkpoint_epoch59.pth`

## Epoch59 Inference And Sparsity

Command:

```bash
SDFORMER_USE_MLFLOW=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/opt/conda/envs/sdformerflow/bin/python tools/profile_sops.py \
  --config neuron_experiments/E4_exp_tslif/configs/full.yml \
  --checkpoint neuron_experiments/E4_exp_tslif/results/e4_official_tslif_full_bs4w8_amp_pinfalse_tf32_20260502_checkpoint_epoch59.pth \
  --output-dir neuron_experiments/E4_exp_tslif/results/profile_sops_official_tslif_epoch59_valid40_20260504 \
  --split valid \
  --num-samples 40 \
  --batch-size 1 \
  --num-workers 4 \
  --metric AEE \
  --metric AAE
```

Output directory:

`neuron_experiments/E4_exp_tslif/results/profile_sops_official_tslif_epoch59_valid40_20260504`

| run | checkpoint | samples | AEE | AAE | firing | SOPs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PSN baseline | epoch59 | 40 | 1.5848 | 7.5012 | 0.08496 | 3.6219G |
| E4 official TS-LIF | epoch59 | 40 | 2.1816 | 9.8193 | 0.09417 | 4.0146G |

Delta vs PSN baseline:

| metric | delta |
| --- | ---: |
| AEE | +0.5968 |
| AAE | +2.3181 |
| firing | +0.00921 |
| SOPs | +0.3927G |

Conclusion: E4 official TS-LIF epoch59 is worse than the PSN baseline on both accuracy and sparsity in this valid40 profile.

## Post-run Audit

Checked on 2026-05-04 against official TS-LIF source commit `a59826a6c7f62d0f16edbafdbb28db65bebd9f69`.

Neuron dynamics:

- The transplanted core keeps the official two-state charge equations, soft reset, learnable `decay_factor`, `kk`, `yy`, `alpha_s`, and `alpha_l`.
- The main intentional deviation is `alpha_s/alpha_l`: official `TSLIFNode` hard-codes feature-shaped alpha tensors such as `[1, 128]`, while the SDFormer overlay uses scalar `[1]` alpha tensors so one wrapper can run on arbitrary convolution/attention feature shapes.
- `detach_reset` support was added in the overlay, but the E4 full config uses `detach_reset: false`, so this does not change the current run.
- The official hard-reset branch references `spike_d`; E4 uses a safe hard-reset implementation, but `hard_reset` is false in the current run.

Training and integration paradigm:

- Official TS-LIF experiments train their own architectures with TS-LIF inserted in specific modules, commonly using Adam, higher learning rates such as `5e-4` or `1e-3`, `weight_decay: 0.0`, gradient clipping, and early stopping.
- E4 instead performs a blanket replacement of SDFormerFlow `Spiking_neuron` modules and initializes from the PSN baseline checkpoint with `strict=False`; therefore all new TS-LIF parameters start random while the rest of the network is PSN-pretrained.
- E4 full training used AdamW, `lr: 1e-4`, multistep decay, and by epoch59 the LR was `3.125e-6`. This is a baseline-continuation protocol, not the official TS-LIF training protocol.

Parameter sanity after epoch59:

| parameter | shape | min | max | mean | negative fraction |
| --- | --- | ---: | ---: | ---: | ---: |
| `alpha_s` | `[1]` | -2.3037 | 2.5902 | 0.2084 | 0.3905 |
| `alpha_l` | `[1]` | -2.6222 | 2.8946 | -0.0718 | 0.5524 |
| `decay_factor` | `[4]` | 0.0477 | 1.0186 | 0.5167 | 0.0000 |
| `kk` | `[1]` | 0.4713 | 0.9517 | 0.7885 | 0.0000 |
| `yy` | `[1]` | -0.1225 | 0.2066 | 0.0905 | 0.0476 |

Audit conclusion:

The current E4 result should not be treated as a faithful official-TS-LIF training result. It is a mechanically valid SDFormerFlow transplant of the TS-LIF dynamics, but it does not fully follow the official usage/training paradigm. The most likely next correction is an E4b protocol with channel/feature-shaped alpha where possible and official-style optimizer settings, or a staged fine-tune with a higher learning rate for TS-LIF parameters and a lower learning rate for the PSN-pretrained backbone.
