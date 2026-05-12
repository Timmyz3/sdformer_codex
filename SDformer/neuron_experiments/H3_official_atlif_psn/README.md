# H3 Official ATLIF-PSN

H3 isolates one question: can PSN temporal mixing benefit from the official
Activity-Pruning-SNN adaptive threshold mechanism without introducing ternary
negative spikes?

## Design

- Keep baseline PSN `weight` and `bias` for temporal mixing.
- Replace only attention `sn_q`/`sn_k` by default.
- Use official ATLIF `Surrogate`: output is `0` or `thresh`.
- Accumulate official `update_value` during forward.
- Call `threshold_update()` after every optimizer step.
- Do not use ternary spikes in this experiment.

## Files

| role | file |
| --- | --- |
| neuron | `overlay/models/STSwinNet_SNN/atlif_psn/atlif_psn.py` |
| installer/training helpers | `overlay/models/STSwinNet_SNN/atlif_psn/installer.py` |
| train entrypoint | `entrypoints/train.py` |
| profile entrypoint | `entrypoints/profile_sops.py` |
| smoke config | `configs/smoke.yml` |
| tests | `tests/test_atlif_psn.py` |

## Smoke

```bash
SDFORMER_USE_MLFLOW=0 SDFORMER_MLFLOW_MODEL_LOGGING=0 \
/opt/conda/envs/sdformerflow/bin/python neuron_experiments/H3_official_atlif_psn/entrypoints/train.py \
  --config neuron_experiments/H3_official_atlif_psn/configs/smoke.yml \
  --prev_runid experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth \
  --save_path neuron_experiments/H3_official_atlif_psn/results/h3_smoke_checkpoint_epoch{}.pth
```

## Verification

```bash
/opt/conda/envs/sdformerflow/bin/python -m unittest \
  neuron_experiments.H3_official_atlif_psn.tests.test_atlif_psn
```

Result on 2026-05-09: `Ran 3 tests ... OK`.

## Short Run

This run continues from the PSN baseline checkpoint:

`experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth`

Only layer-0 attention `sn_q` and `sn_k` are replaced, so four modules are
installed:

- `layers.0.swin_blocks.0.attn.sn_q`
- `layers.0.swin_blocks.0.attn.sn_k`
- `layers.0.swin_blocks.1.attn.sn_q`
- `layers.0.swin_blocks.1.attn.sn_k`

```bash
SDFORMER_USE_MLFLOW=0 SDFORMER_MLFLOW_MODEL_LOGGING=0 \
/opt/conda/envs/sdformerflow/bin/python neuron_experiments/H3_official_atlif_psn/entrypoints/train.py \
  --config neuron_experiments/H3_official_atlif_psn/configs/short_bs16.yml \
  --prev_runid experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth \
  --save_path neuron_experiments/H3_official_atlif_psn/results/h3_short_bs16_120steps_checkpoint_epoch{}.pth
```

Training result on 2026-05-09:

| item | value |
| --- | ---: |
| max train steps | 120 |
| batch size | 16 |
| workers | 8 |
| AMP | true |
| train step time | 1.5560 s |
| train samples/s | 10.2830 |
| max GPU memory | 73.353 GiB |
| train loss | 1.716811 |
| valid loss | 1.738627 |
| threshold mean after train | 0.100052 |
| firing mean from H3 modules | 0.042124 |

Checkpoint:

`results/h3_short_bs16_120steps_checkpoint_epoch0.pth`

## Valid40 SOPs Profile

```bash
SDFORMER_USE_MLFLOW=0 SDFORMER_MLFLOW_MODEL_LOGGING=0 \
/opt/conda/envs/sdformerflow/bin/python neuron_experiments/H3_official_atlif_psn/entrypoints/profile_sops.py \
  --config neuron_experiments/H3_official_atlif_psn/configs/short_bs16.yml \
  --checkpoint neuron_experiments/H3_official_atlif_psn/results/h3_short_bs16_120steps_checkpoint_epoch0.pth \
  --output-dir neuron_experiments/H3_official_atlif_psn/results/profile_sops_h3_short_bs16_120steps_epoch0_valid40_20260509 \
  --split valid \
  --num-samples 40 \
  --batch-size 1 \
  --num-workers 4 \
  --dense-ops 42.63G \
  --metric AEE \
  --metric AAE \
  --module-pattern Spiking_neuron
```

Result:

| experiment | AEE | AAE | firing rate | SOPs |
| --- | ---: | ---: | ---: | ---: |
| PSN baseline epoch59 valid40 | 1.584776 | 7.501204 | 0.084961 | 3.6219G |
| H3 short bs16 120-step valid40 | 1.642847 | 8.717479 | 0.090827 | 3.8720G |

Artifacts:

- `results/profile_sops_h3_short_bs16_120steps_epoch0_valid40_20260509/sops_summary.json`
- `results/profile_sops_h3_short_bs16_120steps_epoch0_valid40_20260509/layer_firing_rates.csv`
- `results/profile_sops_h3_short_bs16_120steps_epoch0_valid40_20260509.log`

## Current Decision

H3 proves that the modular replacement path, official ATLIF-style threshold
update path, checkpoint load path, and SOPs profile path are all connected.
However, this exact setting is not a good full-training candidate yet: after
120 continued-training steps, accuracy is lower than baseline and global firing
rate/SOPs are higher than baseline.

The likely issue is not the entrypoint wiring. The threshold did increase, but
only from `0.100000` to `0.100052`, which is too small to create the expected
ATLIF sparsification effect. The next H3 variants should sweep stronger
threshold growth and optional differentiable activity regularization before any
full run:

| next variant | change | goal |
| --- | --- | --- |
| H3b | increase `threshold_eta` or `threshold_lr_scale` | make threshold growth measurable within short training |
| H3c | add small `activity_eta` | directly penalize firing if threshold-only update is too weak |
| H3d | clamp or schedule threshold growth | avoid accuracy collapse while forcing sparsity |

## Hyperparameter Diagnostics

Official Activity-Pruning-SNN uses these training-side knobs:

| official arg | meaning | default in source |
| --- | --- | ---: |
| `eta` | threshold update scale passed as `sp` into ATLIF surrogate | `1e-4` |
| `eta2` | activity regularization multiplier | `0` |
| `lr` | optimizer lr; also used by `threshold_update(model, lr)` | `0.1` |
| `epochs` | full training schedule | `300` |

H3 maps these into SDFormerFlow as:

| H3 config | meaning |
| --- | --- |
| `threshold_eta` | official `eta` / surrogate `sp` |
| `threshold_lr_scale` | compensates SDFormerFlow's much smaller AdamW lr |
| `activity_eta` | official `eta2`-style differentiable activity penalty |
| `trainable` | `all`, `atlif_only`, or diagnostic `threshold_only` |
| `stage_selection` | `layer0_only`, `stageN`, or `all` attention stages |

The short 120-step H3 run used:

```yaml
optimizer.lr: 0.00005
atlif_psn.threshold_eta: 0.001
atlif_psn.threshold_lr_scale: 1000.0
atlif_psn.activity_eta: 0.0
atlif_psn.stage_selection: layer0_only
atlif_psn.trainable: all
```

This was too weak for sparsity. The threshold mean only grew from `0.100000`
to `0.100052`.

### 40-step threshold-scale sweep

All runs below continue from PSN baseline epoch59, use batch size 16, AMP on,
and evaluate the same 10 validation samples. The PSN baseline row uses the same
profile config but no H3 overlay.

| experiment | replaced modules | trainable params | threshold lr scale | threshold mean | Q/K firing | global firing | SOPs | AEE | AAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PSN baseline valid10 | 0 | 54.9M | - | - | 0.042371 | 0.087981 | 3.7506G | 1.064138 | 6.151228 |
| layer0 Q/K 40-step | 4 | 54.9M | 1k | 0.100017 | 0.040879 | 0.094252 | 4.0179G | 1.100843 | 6.537345 |
| layer0 Q/K 40-step | 4 | 54.9M | 10k | 0.100176 | 0.043258 | 0.096319 | 4.1061G | 1.072094 | 6.305673 |
| layer0 Q/K 40-step | 4 | 54.9M | 50k | 0.100882 | 0.042962 | 0.096538 | 4.1154G | 1.025642 | 6.213643 |
| all Q/K threshold-only 40-step | 24 | 24 | 50k | 0.100701 | 0.038784 | 0.092033 | 3.9233G | 1.079688 | 6.395789 |

Interpretation:

- The first H3 setting was underpowered for threshold growth.
- Increasing `threshold_lr_scale` improves short-run adaptation but does not
  automatically reduce global SOPs.
- The clean `all Q/K + threshold_only` diagnostic proves local ATLIF sparsity:
  Q/K firing drops from `0.042371` to `0.038784`.
- Global SOPs still rise because reduced Q/K activity changes downstream
  activations; downstream PSN layers can fire more even when Q/K fires less.

Next feasibility tests should therefore use wider replacement and an explicit
activity objective, not just a larger threshold update:

| next variant | config direction | reason |
| --- | --- | --- |
| H3e | `stage_selection: all`, `trainable: threshold_only`, stronger/longer threshold schedule | isolate ATLIF threshold pruning without changing PSN weights |
| H3f | `stage_selection: all`, `trainable: atlif_only`, small `activity_eta` | let PSN mixer adapt while directly penalizing firing |
| H3g | target high-SOP downstream PSN layers as well as Q/K | prevent sparsity from simply moving downstream |

## H3f Sweep

H3f keeps the same neuron implementation and changes only the training
protocol:

```yaml
atlif_psn:
  target: qk
  stage_selection: all
  trainable: atlif_only
  threshold_eta: 0.001
  threshold_lr_scale: 50000.0
```

This replaces all attention Q/K spiking modules, 24 modules total, but trains
only the ATLIF-PSN parameters inside those modules: 168 parameters.

### Valid10 activity sweep

All rows continue from PSN baseline epoch59 and train for 40 steps.

| experiment | `activity_eta` | threshold mean | Q/K firing | global firing | SOPs | AEE | AAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PSN baseline valid10 | - | - | 0.042371 | 0.087981 | 3.7506G | 1.064138 | 6.151228 |
| H3f valid10 | 0.1 | 0.100696 | 0.037670 | 0.092335 | 3.9362G | 1.052968 | 6.088702 |
| H3f valid10 | 0.5 | 0.100694 | 0.037928 | 0.089924 | 3.8335G | 1.094040 | 6.483778 |
| H3f valid10 | 1.0 | 0.100698 | 0.035220 | 0.088080 | 3.7549G | 1.068813 | 6.277256 |
| H3f valid10 | 2.0 | 0.100699 | 0.035497 | 0.087757 | 3.7411G | 1.061202 | 6.279803 |

Valid10 interpretation:

- Higher `activity_eta` pushes global firing down.
- `activity_eta=2.0` is the first H3f short run that beats PSN baseline valid10
  SOPs, although the margin is tiny.
- Q/K firing is consistently lower than baseline, so the local ATLIF-PSN
  sparsity mechanism is active.

### Valid40 check for `activity_eta=2.0`

| experiment | samples | Q/K firing | global firing | SOPs | AEE | AAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PSN baseline epoch59 | 40 | - | 0.084961 | 3.6219G | 1.584776 | 7.501204 |
| H3f act2.0 40-step | 40 | 0.033408 | 0.085358 | 3.6388G | 1.558476 | 8.319466 |

Valid40 interpretation:

- H3f act2.0 improves AEE slightly, but does not yet beat baseline SOPs.
- Q/K firing is substantially lower, but global firing is still slightly higher
  than baseline. This confirms the same pattern as earlier diagnostics:
  sparsity is created at Q/K, but not yet enough to dominate the global SOPs
  proxy.
- The next useful test is not a full run yet. It should either train longer
  with the H3f recipe or add downstream high-SOP modules to the ATLIF-PSN target.

### Stronger activity penalty and longer short run

After the first H3f sweep, the activity penalty was increased slightly and the
short run was extended from 40 to 80 steps:

```yaml
atlif_psn:
  stage_selection: all
  trainable: atlif_only
  threshold_eta: 0.001
  threshold_lr_scale: 50000.0
runtime:
  max_train_steps: 80
```

| experiment | `activity_eta` | steps | threshold mean | H3 module firing | valid loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| H3f act3.0 | 3.0 | 80 | 0.101398 | 0.035026 | 1.029256 |
| H3f act4.0 | 4.0 | 80 | 0.101396 | 0.034931 | 1.027467 |

Valid10 comparison:

| experiment | Q/K firing | global firing | SOPs | AEE | AAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| PSN baseline valid10 | 0.042371 | 0.087981 | 3.7506G | 1.064138 | 6.151228 |
| H3f act3.0 80-step valid10 | - | 0.088423 | 3.7695G | 1.055892 | 5.931479 |
| H3f act4.0 80-step valid10 | 0.033354 | 0.086345 | 3.6809G | 1.021265 | 6.216580 |

Valid40 comparison:

| experiment | Q/K firing | global firing | SOPs | AEE | AAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| PSN baseline epoch59 | - | 0.084961 | 3.6219G | 1.584776 | 7.501204 |
| H3f act2.0 40-step | 0.033408 | 0.085358 | 3.6388G | 1.558476 | 8.319466 |
| H3f act4.0 80-step | 0.031791 | 0.084246 | 3.5914G | 1.552034 | 8.435783 |

This is the first H3f point that beats the PSN baseline on both valid40 AEE and
the SOPs proxy:

- AEE improves from `1.584776` to `1.552034`.
- SOPs decrease from `3.6219G` to `3.5914G`, about `0.84%`.
- Global firing decreases from `0.084961` to `0.084246`.
- AAE worsens from `7.501204` to `8.435783`, so angular accuracy remains a
  concern.

Current recommendation: H3f is now feasible enough for a longer short run, but
not yet strong enough for a full run. The next check should be `activity_eta=4.0`
for 120 or 160 steps, or target downstream high-SOP spiking modules to make the
SOP reduction larger than 1%.

### Activity penalty follow-up: 5.0 and 6.0

The regularization penalty was increased again while keeping the same H3f
recipe:

```yaml
atlif_psn:
  target: qk
  stage_selection: all
  trainable: atlif_only
  threshold_eta: 0.001
  threshold_lr_scale: 50000.0
runtime:
  max_train_steps: 80
```

Training comparison:

| experiment | `activity_eta` | steps | threshold mean | H3 module firing | valid loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| H3f act4.0 | 4.0 | 80 | 0.101396 | 0.034931 | 1.027467 |
| H3f act5.0 | 5.0 | 80 | 0.101394 | 0.035198 | 1.019206 |
| H3f act6.0 | 6.0 | 80 | 0.101396 | 0.035018 | 1.024313 |

Valid10 comparison:

| experiment | Q/K firing | global firing | SOPs | AEE | AAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| PSN baseline valid10 | 0.042371 | 0.087981 | 3.7506G | 1.064138 | 6.151228 |
| H3f act4.0 80-step valid10 | 0.034515 | 0.086345 | 3.6809G | 1.021265 | 6.216580 |
| H3f act5.0 80-step valid10 | 0.036907 | 0.088797 | 3.7854G | 1.040733 | 6.181826 |
| H3f act6.0 80-step valid10 | 0.036020 | 0.087199 | 3.7173G | 1.022939 | 6.238154 |

Follow-up conclusion:

- Increasing `activity_eta` from `4.0` to `5.0` or `6.0` does not improve the
  current short-run sparsity/accuracy trade-off.
- `activity_eta=6.0` recovers most of the AEE but still has higher SOPs and
  higher Q/K firing than `activity_eta=4.0`.
- `activity_eta=4.0` remains the best H3f point so far, so `5.0` and `6.0`
  were not expanded to valid40.

## H3h: Q/K plus high-SOP target paths

H3h tests whether the ATLIF-PSN mechanism can be extended beyond attention Q/K
to layers that dominate the global SOPs proxy. The installer now supports exact
module paths through `atlif_psn.target_paths`, while keeping the old Q/K target
path intact.

Config:

`configs/h3h_qk_highsop_act0p5_80.yml`

Targets:

- all attention `sn_q` and `sn_k` modules
- `sttmultires_unet.decoders.3.sn`
- `sttmultires_unet.encoders.swin3d.layers.2.downsample.sn`
- `sttmultires_unet.encoders.swin3d.layers.0.downsample.sn`
- `sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp.sn1`
- `sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.1.mlp.sn1`
- `sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.attn.proj_sn`
- `sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.1.attn.proj_sn`
- `sttmultires_unet.decoders.0.sn`
- `sttmultires_unet.resblocks.1.sn1`
- `sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.5.mlp.sn1`

Short-run result:

| experiment | modules | steps | threshold mean | H3 firing | valid loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| H3h qk+highSOP act0.5 | 34 | 80 | 0.102127 | 0.077320 | 1.226941 |

Valid40 profile:

| experiment | Q/K firing | global firing | SOPs | AEE | AAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| PSN baseline epoch59 | 0.042556 | 0.084961 | 3.6219G | 1.584776 | 7.501204 |
| H3h qk+highSOP act0.5 80-step | 0.033636 | 0.082442 | 3.5145G | 1.706355 | 8.907388 |

Layer-local effect on the explicit high-SOP targets:

| layer | baseline firing | H3h firing | change |
| --- | ---: | ---: | ---: |
| `decoders.3.sn` | 0.281865 | 0.247242 | -12.3% |
| `layers.2.downsample.sn` | 0.298979 | 0.206454 | -30.9% |
| `layers.0.downsample.sn` | 0.236217 | 0.203449 | -13.9% |
| `layers.0.block0.mlp.sn1` | 0.221972 | 0.199371 | -10.2% |
| `layers.0.block1.mlp.sn1` | 0.231007 | 0.188496 | -18.4% |
| `layers.0.block0.attn.proj_sn` | 0.182469 | 0.187881 | +3.0% |
| `layers.0.block1.attn.proj_sn` | 0.107325 | 0.090913 | -15.3% |
| `decoders.0.sn` | 0.243735 | 0.173638 | -28.8% |
| `resblocks.1.sn1` | 0.242144 | 0.164126 | -32.2% |
| `layers.2.block5.mlp.sn1` | 0.281117 | 0.171390 | -39.0% |

Interpretation:

- The path-based installer works and can sparsify high-SOP modules.
- This first setting is too aggressive structurally: replacing Q/K and many
  high-SOP downstream layers together hurts accuracy.
- The next useful test should be grouped and staged: Q/K first, then one family
  at a time (`decoder-only`, `downsample-only`, `MLP-only`, `proj-only`) with a
  target firing floor or lower activity penalty.
