# H5 High-SOPs Fusion

H5 extends H4 from attention Q/K only to a graded set of high-SOPs spiking
wrappers. The baseline and `third_party/SDformerFlow` remain untouched; training
still enters through this experiment's wrapper entrypoint and calls the baseline
code after installing the overlay modules.

## Goal

Validate whether PSN + ATLIF adaptive threshold + ternary output can reduce
total SOPs more meaningfully when applied beyond Q/K, while avoiding the most
risky output-side modules first.

## Graded Replacement Design

| config | replaced modules | reason |
|---|---|---|
| `h5a_qk_proj_80.yml` | all attention Q/K + all attention `proj_sn` | adds attention output spikes, a medium-risk high-SOPs tier |
| `h5b_qk_proj_stage0_mlp_80.yml` | H5a + stage0 `mlp.sn1/sn2` | targets the largest transformer MLP spike contributors |
| `h5c_qk_proj_mlp_downsample_80.yml` | H5b + stage0/stage2 `downsample.sn` | tests a stronger high-SOPs cut, still avoiding decoders/preds |

The Q/K modules keep the H4h strength:

- `activity_eta: 2.0`
- `max_threshold: 0.13`
- `negative_threshold_scale: 30.0`

High-SOPs target groups use gentler path-specific settings via
`atlif_ternary_psn.target_groups`, for example:

- attention `proj_sn`: `activity_eta: 0.05`, `max_threshold: 0.11`
- stage0 MLP: `activity_eta: 0.03`, `max_threshold: 0.105`
- downsample: `activity_eta: 0.02`, `max_threshold: 0.105`

This keeps the experiment from turning every high-activity layer into the same
over-sparse Q/K behavior.

## Entry Points

- Train:

```bash
SDFORMER_USE_MLFLOW=0 python neuron_experiments/H5_highsops_fusion/entrypoints/train.py \
  --config neuron_experiments/H5_highsops_fusion/configs/<config>.yml \
  --prev_runid experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth \
  --save_path neuron_experiments/H5_highsops_fusion/results/<name>_checkpoint_epoch{}.pth
```

- Profile:

```bash
python neuron_experiments/H5_highsops_fusion/entrypoints/profile_sops.py \
  --config neuron_experiments/H5_highsops_fusion/configs/<config>.yml \
  --checkpoint neuron_experiments/H5_highsops_fusion/results/<checkpoint>.pth \
  --output-dir neuron_experiments/H5_highsops_fusion/results/<profile_dir> \
  --split valid \
  --num-samples 10 \
  --batch-size 1 \
  --num-workers 4 \
  --dense-ops 42.63G \
  --metric AEE \
  --metric AAE \
  --module-pattern Spiking_neuron
```

## Baseline Reference

| experiment | samples | q/k rate | global rate | SOPs | AEE | AAE |
|---|---:|---:|---:|---:|---:|---:|
| PSN baseline epoch59 | 40 | 0.045224 | 0.084961 | 3.6219G | 1.584776 | 7.501204 |
| H4h fusionverify epoch14 | 40 | 0.002076 | 0.081691 | 3.4825G | 1.538182 | 7.967144 |

H5 should beat H4h epoch14 on total SOPs without letting AAE degrade as badly as
the over-sparse H4h epoch24/29 checkpoints.

## Short-Run Results

All short runs inherit from:

`experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth`

Each run uses `max_train_steps: 80`, then profiles `valid10`.

| experiment | modules | q/k rate | proj rate | stage0 MLP rate | downsample rate | global rate | SOPs | AEE | AAE | result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| H4h q/k reference | 24 | 0.037218 | - | - | - | 0.087457 | 3.7283G | 1.027874 | 6.088951 | reference |
| H5a q/k + proj | 36 | 0.038516 | 0.453555 | 0.148991 | 0.205956 | 0.108943 | 4.6442G | 1.061316 | 6.269898 | worse SOPs |
| H5b + stage0 MLP | 40 | 0.039556 | 0.461897 | 0.664543 | 0.220953 | 0.171391 | 7.3064G | 1.064775 | 6.315249 | much denser |
| H5c + downsample | 42 | 0.039959 | 0.466664 | 0.664847 | 0.573813 | 0.180286 | 7.6856G | 1.137244 | 6.704098 | much denser |

Conclusion: direct ternary ATLIF replacement does not transfer from Q/K to
high-SOPs layers. The extra high-SOPs modules become dense rather than sparse:
`proj_sn` fires around `0.45`, stage0 MLP around `0.66`, and downsample around
`0.57`. The H5a/H5b/H5c checkpoints are therefore not suitable for full
training.

The likely reason is that high-SOPs layers have a much broader signed activation
distribution than Q/K. The ternary negative branch creates many negative output
events, so replacing high-SOPs layers with the same signed ternary neuron
increases activity before the adaptive threshold can compensate.

Next candidate: keep ternary ATLIF only on Q/K, but use a positive-only
ATLIF-PSN gate for high-SOPs layers, or add a target-rate controller for these
groups. That would preserve the Q/K fusion story while making high-SOPs layers
actually sparse instead of signed-dense.
