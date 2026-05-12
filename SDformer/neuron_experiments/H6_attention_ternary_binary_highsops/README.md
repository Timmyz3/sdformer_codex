# H6 Attention Ternary + Binary High-SOPs

H6 tests the hypothesis that ternary output should stay in attention, while
non-attention high-SOPs layers should use binary ATLIF output. This directly
addresses the H5 failure mode: signed ternary output made high-SOPs layers
dense because negative spikes became frequent.

## Variants

| config | attention output | high-SOPs output | target paths |
|---|---|---|---|
| `h6a_qk_ternary_mlp_down_binary_80.yml` | Q/K ternary | stage0 MLP + stage0/stage2 downsample binary | no attention proj replacement |
| `h6a_qk_ternary_mlp_down_binary_full.yml` | Q/K ternary | stage0 MLP + stage0/stage2 downsample binary | 30-epoch full run from baseline epoch59 |
| `h6a_qk_ternary_mlp_down_binary_allparams_full.yml` | Q/K ternary | stage0 MLP + stage0/stage2 downsample binary | H6a full run, but train all parameters |
| `h6b_attn_ternary_mlp_down_binary_80.yml` | Q/K + attention `proj_sn` ternary | stage0 MLP + stage0/stage2 downsample binary | literal "attention ternary" variant |

The binary groups use `output_mode: binary`, so they emit only `0/+threshold`
and never produce negative output events.

## Commands

Train:

```bash
SDFORMER_USE_MLFLOW=0 python neuron_experiments/H6_attention_ternary_binary_highsops/entrypoints/train.py \
  --config neuron_experiments/H6_attention_ternary_binary_highsops/configs/<config>.yml \
  --prev_runid experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth \
  --save_path neuron_experiments/H6_attention_ternary_binary_highsops/results/<name>_checkpoint_epoch{}.pth
```

Profile:

```bash
python neuron_experiments/H6_attention_ternary_binary_highsops/entrypoints/profile_sops.py \
  --config neuron_experiments/H6_attention_ternary_binary_highsops/configs/<config>.yml \
  --checkpoint neuron_experiments/H6_attention_ternary_binary_highsops/results/<checkpoint>.pth \
  --output-dir neuron_experiments/H6_attention_ternary_binary_highsops/results/<profile_dir> \
  --split valid \
  --num-samples 10 \
  --batch-size 1 \
  --num-workers 4 \
  --dense-ops 42.63G \
  --metric AEE \
  --metric AAE \
  --module-pattern Spiking_neuron
```

## Reference

| experiment | samples | global rate | SOPs | AEE | AAE |
|---|---:|---:|---:|---:|---:|
| H4h q/k reference | 10 | 0.087457 | 3.7283G | 1.027874 | 6.088951 |
| H5a q/k + proj ternary | 10 | 0.108943 | 4.6442G | 1.061316 | 6.269898 |
| H5b + stage0 MLP ternary | 10 | 0.171391 | 7.3064G | 1.064775 | 6.315249 |
| H5c + downsample ternary | 10 | 0.180286 | 7.6856G | 1.137244 | 6.704098 |

## Results

80-step short training from
`experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth`,
then valid10 SOPs profile:

| experiment | installed modules | train loss | valid loss | step sec | max GPU GiB | global rate | SOPs | AEE | AAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H6a q/k ternary + MLP/downsample binary | 30 | 8.0682 | 1.0306 | 4.2326 | 44.736 | 0.086604 | 3.6919G | 1.049870 | 6.106936 |
| H6b q/k/proj ternary + MLP/downsample binary | 42 | 8.9467 | 1.0382 | 4.9496 | 48.669 | 0.106058 | 4.5213G | 1.061870 | 6.567753 |

Full run:

| experiment | config | run dir | status |
|---|---|---|---|
| H6a q/k ternary + MLP/downsample binary | `configs/h6a_qk_ternary_mlp_down_binary_full.yml` | `results/h6a_qk_ternary_mlp_down_binary_full_20260511_120542_setsid` | finished, 30 epochs |
| H6a all-params continuation | `configs/h6a_qk_ternary_mlp_down_binary_allparams_full.yml` | pending | trainable `all`, lr `2e-5` |

Full-run checkpoints, valid loss, and valid10 profile:

| checkpoint | train loss | valid loss | global rate | SOPs | AEE | AAE |
|---|---:|---:|---:|---:|---:|---:|
| epoch11, best valid loss | 2.7532 | 1.0098 | 0.079970 | 3.4091G | 1.021146 | 5.930885 |
| epoch29, last | 1.8045 | 1.1147 | 0.073425 | 3.1301G | 1.100111 | 6.754824 |

Main valid40 comparison against PSN baseline:

| experiment | train loss | valid loss | global rate | SOPs | AEE | AAE | SOPs vs base | AEE vs base | AAE vs base |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PSN baseline epoch59 | - | - | 0.084961 | 3.6219G | 1.584776 | 7.501204 | 0.00% | 0.00% | 0.00% |
| H6a epoch11, best valid loss | 2.7532 | 1.0098 | 0.077434 | 3.3010G | 1.553494 | 8.200176 | -8.86% | -1.97% | +9.32% |
| H6a epoch19 | 1.8604 | 1.0661 | 0.075049 | 3.1994G | 1.614310 | 8.376308 | -11.67% | +1.86% | +11.67% |
| H6a epoch29, last | 1.8045 | 1.1147 | 0.071159 | 3.0335G | 1.628533 | 8.709481 | -16.24% | +2.76% | +16.11% |

Valid40 layer-group mean firing rates:

| experiment | Q/K | attention proj | stage0 MLP | downsample |
|---|---:|---:|---:|---:|
| PSN baseline epoch59 | 0.042556 | 0.119917 | 0.149582 | 0.235088 |
| H6a epoch11 | 0.003175 | 0.101365 | 0.102602 | 0.193181 |
| H6a epoch19 | 0.000094 | 0.101043 | 0.077216 | 0.179498 |
| H6a epoch29 | 0.000006 | 0.100678 | 0.062639 | 0.171124 |

Epoch11 is the best trade-off so far. Later checkpoints keep reducing SOPs,
but Q/K becomes almost silent and optical-flow accuracy degrades.

Full-run layer-group mean firing rates:

| checkpoint | Q/K | attention proj | stage0 MLP | downsample |
|---|---:|---:|---:|---:|
| epoch11, best valid loss | 0.003695 | 0.105275 | 0.106652 | 0.196315 |
| epoch29, last | 0.000007 | 0.104650 | 0.065094 | 0.170950 |

Layer-group mean firing rates:

| experiment | Q/K | attention proj | stage0 MLP | downsample |
|---|---:|---:|---:|---:|
| H6a | 0.039871 | 0.107404 | 0.132644 | 0.198864 |
| H6b | 0.038808 | 0.450847 | 0.130406 | 0.192483 |

## Takeaway

The mixed ternary/binary idea works better than H5's direct ternary expansion
to high-SOPs layers. H6a slightly improves SOPs over the H4h q/k reference
while keeping the high-SOPs binary groups sparse. H6b confirms that making
attention `proj_sn` ternary is still expensive: Q/K stays sparse, but `proj_sn`
becomes dense and raises total SOPs from 3.69G to 4.52G.

The next graded replacement should keep ternary output only on Q/K and use
binary adaptive-threshold output for high-SOPs non-attention modules. If we
still want to test attention projection, it should use a much stronger
projection-specific sparsity target or a positive-only ternary gate instead of
the current signed ternary surrogate.
