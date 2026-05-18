# H6a-all AEE/AAE gap analysis

## Metric definitions from baseline

Source: `third_party/SDformerFlow/loss/flow_supervised.py`

### AEE

For each valid pixel:

```text
pred = network_output * flow_scaling
AEE_pixel = sqrt((pred_x - gt_x)^2 + (pred_y - gt_y)^2)
AEE = mean(AEE_pixel over valid pixels)
```

AEE measures endpoint distance in flow space. It is mostly sensitive to vector magnitude error.

### AAE

For each valid pixel:

```text
pred = network_output * flow_scaling
cosine = dot(pred, gt) / (|pred| * |gt|)



AAE_pixel = acos(clamp(cosine)) * 180 / pi
AAE = mean(AAE_pixel over valid pixels)
```

AAE measures direction error. It can become large even when AEE is small, especially for low-magnitude flow pixels where a small endpoint error can still rotate the vector direction substantially.

## Observed metrics

| Run | Firing rate | SOPs | AEE | AAE |
| --- | ---: | ---: | ---: | ---: |
| PSN baseline epoch59 | 0.084961 | 3.6219G | 1.584776 | 7.501204 |
| H6 frozen epoch11 | 0.077434 | 3.3010G | 1.553494 | 8.200176 |
| H6 frozen epoch29 | 0.071159 | 3.0335G | 1.628533 | 8.709481 |
| H6 all-params epoch29 | 0.064930 | 2.7680G | 1.594698 | 21.636660 |

## Likely root cause

The AAE blow-up is not caused by the H6 replacement idea alone. The frozen H6 run keeps AAE near baseline while reducing SOPs. The issue appears mainly in the all-parameters fine-tune.

Evidence:

- Training optimizes endpoint loss only: `lambda_ang: 0`.
- H6 all-params is much more sparse than frozen H6.
- Several early/high-impact paths become much quieter than the frozen version:
  - `layers.0.downsample.sn`: baseline 0.236217, frozen 0.192959, all-params 0.095514
  - `layers.0.swin_blocks.1.mlp.sn1`: baseline 0.231007, frozen 0.144312, all-params 0.072686
  - `layers.2.downsample.sn`: baseline 0.298979, frozen 0.225690, all-params 0.119124
- Q/K activity is almost zero in both frozen and all-params H6, but only all-params has severe AAE degradation. So Q/K sparsity alone is not enough to explain the gap; the stronger all-params sparsification of FFN/downsample paths is the more suspicious difference.

Interpretation:

The all-params run likely learned to preserve endpoint magnitude enough to keep AEE close, while losing direction consistency in many valid pixels. Since angular loss is disabled, the optimizer has little direct pressure to keep vector direction aligned.

## Metric caveat

`AEE_PE1`, `AEE_PE2`, `AEE_PE3`, and `AEE_outliers` in the baseline metric implementation are not reliable percentages when profiling with batch size larger than 1. The numerator is summed across the batch while the denominator is per sample, so values can exceed 1. This does not directly affect AEE, AAE, firing rate, or SOPs.

## Recommended next H6 direction

1. Keep the H6 mechanism, but avoid full all-parameter fine-tuning as the default.
2. Use a two-stage schedule:
   - Stage A: train ATLIF thresholds/adaptation only.
   - Stage B: unfreeze only BN/projection/late-stage FFN with a smaller LR.
3. Add a small angular or cosine consistency term, for example `lambda_ang > 0` or direction distillation from the PSN baseline.
4. Protect early/downsample paths with weaker sparsity:
   - lower `activity_eta`
   - lower `max_threshold`
   - target-rate floor for stage0/downsample
5. Report both original AAE and a diagnostic masked AAE for `gt_mag > 0.5` or `gt_mag > 1.0` to separate true direction failure from low-motion angular sensitivity.
