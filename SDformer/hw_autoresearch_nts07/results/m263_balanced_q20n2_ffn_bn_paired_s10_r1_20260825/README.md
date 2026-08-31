# M263 balanced-Q20 two-Newton FFN BN downstream gate

Status: **paired first-ten executed; local-error and DSEC-Fl gates pass, AEE
gate fails**.

The candidate widens dynamic moments/parameters/coefficients, uses a 32-entry
Q20 LUT and performs two Newton iterations.  It replaces all 12 FFN BN1 and 12
FFN BN2 affine outputs while PyTorch still finalizes current-batch moments.

| Metric | Reference | Candidate | Relative delta | Candidate wins/losses |
|---|---:|---:|---:|---:|
| AEE | 0.9501519361 | 0.9606827700 | +1.108332% | 3 / 7 |
| DSEC Fl | 2.3105949034 | 2.3268252290 | +0.702431% | 5 / 5 |
| Spikes/frame | 101,920,461.1 | 101,935,268.6 | +0.014528% | 5 / 5 |

The coefficient path evaluates 220,800 channel pairs and 4,377,600,000 BN
outputs with zero rails.  Mean absolute output error is `2.97206e-6`, RMSE is
`3.85667e-6`, and maximum absolute error is `9.91821e-5`.  This recovers
`0.62254` percentage points of the M235 Q16 AEE regression, but uniform
approximation of all 24 FFN BN modules remains downstream-unsafe.

Decision: do not run valid825 and do not build Q20 RTL yet.  First ablate by
Swin stage and BN1/BN2, then allocate high precision or fallback only to the
threshold-sensitive modules.  Uniform further widening is not justified.

Analyzer SHA256:
`b85c2d3533f5649176d1d0aad2abcc07afb0180cd4f45f15f1ea851ce1984edc`.
The original remote absolute-path `SHA256SUMS` is preserved and relocatably
verified.  No system-speedup, PPA, energy, valid825, or headline claim is made.
`docs/359` remains unchanged at
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
