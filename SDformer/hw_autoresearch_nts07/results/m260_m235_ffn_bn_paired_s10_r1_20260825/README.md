# M260 M235 FFN dynamic-BN downstream gate

Status: **paired first-ten experiment executed, M235 Q16 downstream accuracy gate failed**.

The exact frozen H67 ep35 evaluator was run twice on the same ten validation
frames with `bn_policy=no_running`.  The reference keeps PyTorch FFN BN affine
outputs; the candidate replaces only the 12 BN1 and 12 BN2 affine outputs with
the exact M234/M235 64-entry LUT plus one-Newton Q16 coefficient recurrence.
PyTorch still finalizes current-batch moments in both runs.

## Exact paired result

| Metric | Reference | M235 approximation | Relative delta | Paired wins/losses |
|---|---:|---:|---:|---:|
| AEE (lower is better) | 0.9501519361 | 0.9665978201 | +1.730869% | 3 / 7 |
| DSEC Fl (lower is better) | 2.3105949034 | 2.4698466338 | +6.892239% | 4 / 6 |
| Total spikes per frame | 101,920,461.1 | 101,932,344.7 | +0.011660% | 6 / 4 |

Across 4,377,600,000 FFN BN output values, the coefficient path has mean
absolute error `4.82407e-5`, RMSE `6.33373e-5`, maximum absolute error
`0.0017414093`, and zero numeric rails.  Those small local errors still alter
recurrent threshold events enough to worsen the flow output.  A local BN error
bound is therefore not an SNN accuracy admission.

## Decision

- Do not run valid825 for the current M235 Q16 configuration.
- Do not claim this BN configuration as accuracy-safe, system speedup, PPA, or
  a standalone paper contribution.
- Next, sweep mixed invstd/alpha/offset precision and optional second Newton
  iteration while recording per-module event flips and threshold margins.
- If mixed precision cannot restore the paired gate cheaply, use exact-recurrence
  quantization-aware fine-tuning rather than hiding the degradation.

The remote payload manifest is preserved as `SHA256SUMS` with its original
absolute server paths; the analyzer relocates each entry under this directory
and verifies all 11 payloads.  Analyzer SHA256:
`eacc7550254eb57694f5ff6cdcbb82ef2c9057fa844ca678c348cafd34127257`.

This result does not modify `docs/359`; its SHA256 remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
