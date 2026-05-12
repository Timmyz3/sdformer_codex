# H4 ATLIF Ternary PSN

H4 fuses the H3 official ATLIF threshold-update path with ternary output:

- baseline PSN temporal `weight` and `bias` are copied;
- spike output is `{-threshold, 0, +threshold}`;
- threshold update follows the ATLIF accumulator idea but uses `abs(input)` so
  both positive and negative spikes contribute to sparsification;
- `threshold_max` is used in the first sweep to avoid the H3 full-run failure
  mode where Q/K firing collapses to zero.

The first target is attention Q/K only.

## Files

- `overlay/models/STSwinNet_SNN/atlif_ternary_psn/atlif_ternary_psn.py`
  defines `ATLIFTernaryPSN`, including per-module activity strength and
  optional target-rate feedback.
- `overlay/models/STSwinNet_SNN/atlif_ternary_psn/installer.py`
  replaces selected baseline `Spiking_neuron.spiking_neuron` modules at runtime
  and applies scalar or per-stage sparsity settings.
- `overlay/models/STSwinNet_SNN/atlif_ternary_psn/training.py`
  hooks threshold updates after optimizer steps and prints H4 statistics.
- `entrypoints/train.py`
  calls the baseline train entry after installing the overlay.
- `entrypoints/profile_sops.py`
  calls the shared SOPs/firing-rate profiler with the H4 overlay installed.
- `configs/h4h_asymneg30_cap0p13_act2p0_full.yml`
  is the selected full-run config.

## Selected Full Config

- Previous checkpoint:
  `experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth`
- Target modules: all attention `sn_q` and `sn_k`, 24 modules.
- Trainable parameters: H4 neuron parameters only, 168 parameters.
- Ternary rule: positive if `input >= threshold`, negative if
  `input <= -threshold * 30.0`.
- Threshold cap: `max_threshold=0.13`.
- Activity pressure: `activity_eta=2.0`.
- Optimizer: AdamW, `lr=5e-5`, AMP on.
- Loader: batch size 16, workers 8, `pin_memory=false`.

## Short Sweep

| config | neg scale | activity eta | q/k rate | global rate | SOPs | AEE | AAE | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| h4a | symmetric | 1.0 | 0.977292 | 0.208027 | 8.8682G | 1.164812 | 6.423452 | negative spikes too dense |
| h4e | 10 | 0.5 | 0.197547 | 0.112383 | 4.7909G | 1.076342 | 6.212028 | still too dense |
| h4f | 20 | 0.5 | 0.054586 | 0.090503 | 3.8582G | 1.021093 | 6.285384 | close |
| h4g | 30 | 0.5 | 0.038430 | 0.087967 | 3.7500G | 1.023835 | 6.113209 | balanced |
| h4h | 30 | 2.0 | 0.037218 | 0.087457 | 3.7283G | 1.027874 | 6.088951 | selected |
| h4i | 40 | 1.0 | 0.036278 | 0.087839 | 3.7446G | 1.037485 | 6.108155 | slightly worse than h4h |

Short profiles used valid10. The naive symmetric ternary rule was rejected
because baseline PSN has substantial negative outputs, which become dense
negative spikes if the threshold is symmetric.

## Follow-Up Sparsity Sweep

These runs tested weaker sparsity, per-stage sparsity, and target-rate adaptive
threshold feedback. They were run for 80 training steps and profiled on valid10.

| config | idea | q/k rate | stage q/k rates | global rate | SOPs | AEE | AAE | decision |
|---|---|---:|---|---:|---:|---:|---:|---|
| h4g | scalar act0.5 | 0.038430 | s0 0.04445, s1 0.01226, s2 0.04135, s3 0.04982 | 0.087967 | 3.7500G | 1.023835 | 6.113209 | baseline scalar reference |
| h4h | scalar act2.0 | 0.037218 | s0 0.04413, s1 0.01187, s2 0.03970, s3 0.04821 | 0.087457 | 3.7283G | 1.027874 | 6.088951 | best short balance |
| h4j | weaker cap0.11 act0.5 | 0.039446 | s0 0.04483, s1 0.01258, s2 0.04264, s3 0.05134 | 0.088958 | 3.7923G | 1.036873 | 6.102337 | preserves AAE but does not save SOPs |
| h4k | weak early, strong deep | 0.037455 | s0 0.04831, s1 0.01232, s2 0.03907, s3 0.04690 | 0.087135 | 3.7146G | 1.049844 | 6.330054 | lower SOPs, worse AAE |
| h4l | target-rate adaptive | 0.037236 | s0 0.04306, s1 0.01230, s2 0.04006, s3 0.04789 | 0.086512 | 3.6880G | 1.049980 | 6.348640 | lowest SOPs, accuracy tradeoff too high |
| h4m | scalar act1.0 | 0.038629 | s0 0.04373, s1 0.01256, s2 0.04160, s3 0.05069 | 0.087759 | 3.7411G | 1.033355 | 6.171578 | not better than h4h |
| h4n | rate guard from h4h targets | 0.038183 | s0 0.04350, s1 0.01244, s2 0.04121, s3 0.04953 | 0.088266 | 3.7628G | 1.049575 | 6.188853 | stabilizer idea works, but short result not better |

Follow-up conclusion: simply lowering sparsity intensity increases SOPs. Stronger
per-stage or target-rate feedback lowers SOPs a little more, but the AAE penalty
is not worth it. H4h remains the best short-run H4 variant. H4n is still useful
as a mechanism if long training needs a guard against over-sparsifying q/k, but
it should not replace H4h based on short-run metrics alone.

## Full-Run Checkpoint Profiles

The full run was intentionally stopped after epoch3 because validation did not
improve while activity continued to shrink. Epoch0 is the best H4 checkpoint.

| experiment | samples | q/k rate | global rate | SOPs | AEE | AAE |
|---|---:|---:|---:|---:|---:|---:|
| PSN baseline epoch59 | 40 | 0.045224 | 0.084961 | 3.6219G | 1.584776 | 7.501204 |
| H3 ATLIF-PSN epoch29 | 40 | 0.000000 | 0.081477 | 3.4734G | 1.585315 | 8.433778 |
| H4h epoch0 | 40 | 0.028139 | 0.084207 | 3.5897G | 1.572934 | 8.338563 |
| H4h epoch2 | 40 | 0.018968 | 0.084185 | 3.5888G | 1.599631 | 8.402593 |
| H4h epoch3 | 40 | 0.016730 | 0.086718 | 3.6968G | 1.655399 | 8.717447 |

Conclusion: H4h epoch0 is viable as a ternary-fusion proof of concept. It keeps
AEE slightly better than the PSN baseline and lowers SOPs by about 0.9%, but
AAE is worse and the SOPs reduction is weaker than H3. Continuing the same
training past epoch0 is not worthwhile: q/k firing drops, but total SOPs do not
continue to improve because other layers dominate the total.

## Fusion-Verify Full Run Profiles

Clean full run from the same PSN baseline epoch59 checkpoint:

`results/h4h_fusionverify_full_20260510_checkpoint_epoch{}.pth`

Profile command pattern:

```bash
python neuron_experiments/H4_atlif_ternary_psn/entrypoints/profile_sops.py \
  --config neuron_experiments/H4_atlif_ternary_psn/configs/h4h_asymneg30_cap0p13_act2p0_full.yml \
  --checkpoint <checkpoint> \
  --output-dir <profile_dir> \
  --split valid \
  --num-samples 40 \
  --batch-size 1 \
  --num-workers 4 \
  --dense-ops 42.63G \
  --metric AEE \
  --metric AAE \
  --module-pattern Spiking_neuron
```

| experiment | samples | q/k rate | global rate | SOPs | AEE | AAE | profile dir |
|---|---:|---:|---:|---:|---:|---:|---|
| PSN baseline epoch59 | 40 | 0.045224 | 0.084961 | 3.6219G | 1.584776 | 7.501204 | `neuron_experiments/E0_psn_baseline/results/profile_sops_epoch59_valid40` |
| H4h fusionverify epoch14 | 40 | 0.002076 | 0.081691 | 3.4825G | 1.538182 | 7.967144 | `results/profile_h4h_fusionverify_full_epoch14_valid40_20260511` |
| H4h fusionverify epoch24 | 40 | 0.000097 | 0.083700 | 3.5681G | 1.561874 | 8.393927 | `results/profile_h4h_fusionverify_full_epoch24_valid40_20260511` |
| H4h fusionverify epoch29 | 40 | 0.000022 | 0.080117 | 3.4154G | 1.560816 | 8.364778 | `results/profile_h4h_fusionverify_full_epoch29_valid40_20260511` |

Fusion-verify conclusion: epoch14 is the best checkpoint for validating the
fusion idea. It cuts estimated SOPs from `3.6219G` to `3.4825G` while improving
AEE from `1.584776` to `1.538182`, but AAE worsens from `7.501204` to
`7.967144`. Epoch24 and epoch29 are more aggressively sparse in q/k, but q/k is
almost zero and angular accuracy degrades, so they are less suitable as the
main proof-of-concept checkpoint.
