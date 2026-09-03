# M2045 canonical valid825 independent result review

## Verdict

**PASS, 95/100; P0=0, P1=2, P2=3.** The paired task-accuracy
result may be cited with the scope below. The independent hammer passes 25/25
checks without GPU, EDA, or license access.

The canonical result and reviewed derived bundle both have exhaustive inner
manifests and outer seals: every directory member is a regular non-symlink,
every data member is listed exactly once, and there are no unsealed extras.

## Citable result

The candidate is evaluated on exactly the same 825-frame, 18-sequence,
48,152,523-valid-pixel population as the frozen baseline. The validation-list
SHA, evaluation protocol, metric contract, per-sequence frame counts, and
per-sequence valid-pixel counts are equal.

| Metric | Frozen baseline | Candidate | Candidate − baseline |
|---|---:|---:|---:|
| AEE | 1.1995140134 | 1.1973673040 | **−0.0021467094** |
| AAE | 5.4006410839 | 5.4128083761 | +0.0121672922 |
| AAE_Benchmark | 5.1063634050 | 5.1216190149 | +0.0152556099 |
| DSEC_Fl | 5.3133596618 | 5.3288341660 | +0.0154745042 |

The contracted AEE gate is `candidate − baseline <= +0.02`; the observed
`−0.0021467094` passes. All four headline metrics and deltas were independently
reparsed from the two sealed profiles. Frame-equal, pixel-global, and
sequence-balanced aggregates were independently reconstructed from all 18
per-sequence rows and match the recorded aggregation audit.

The paper may therefore cite the candidate valid825 accuracy and its delta
against this frozen baseline. A safe sentence is:

> On the same local DSEC valid825 population, the hardware-order attention plus
> eight-operator QDQ deployment candidate obtains 1.1974 AEE versus 1.1995 for
> the frozen checkpoint baseline (delta -0.0021), satisfying the +0.02 AEE
> deployment gate.

## Identity and execution audit

- M2045 wrapper SHA:
  `890dfd6bac5ddd2696af41ecfbc1a98cc1284d64ef6fbdbf993d485274dd17e1`
- M2045 contract SHA:
  `4c3222055a7fa7b8b246ab43caf7b37a7eeb8554021f3556d9998942d302bdb0`
- Frozen M2044 engine SHA:
  `edc5df9ce9debbb28863abf26426b7504c16552f7c47865b3a31a091b6cb9b20`
- Evaluator SHA:
  `84daee48291d8ab2ee644f43458b909e96190c0dce7f5ff4d4179b61be30faac`
- Reviewed bundle manifest SHA:
  `ef2b502f7e17e2a28b11c4a627c8bc6f16ef78b5782b2636ace5a743544bdd8c`
- Canonical result manifest SHA:
  `c25a4857b5cd40616aa94324b396ed9a96d457a1453307a29eb99918fadf59fa`

The sealed execution log records
`/opt/conda/envs/sdformerflow/bin/python` as the evaluator interpreter and a
single zero exit code. The checkpoint load audit has zero missing, unexpected,
overlay-missing, and overlay-unexpected keys. The backend audit records TF32
disabled and cuDNN benchmark disabled. All 12 attention blocks and 105 ATLIF
modules were configured. Each of the four C1 Conv3x3 and four decoder
ConvTranspose targets executed exactly 825 times and produced nonzero output
populations. The prior tensor audit checks all 921 checkpoint tensors: 913
untouched tensors are exactly equal and all eight target QDQ tensors are exact.

## Adversarial claim boundary

The admitted object is only:

- the hardware-order attention path over the full valid825 run; and
- QDQ deployment of exactly four bottleneck Conv3x3 and four decoder
  ConvTranspose weight tensors.

It is **not** full-network INT8, whole-network hardware-order equivalence, an
SV-equivalent network, hardware cycles, speedup, system speedup, power, energy,
or PPA. The separate M2043 integer bridge is not made a full-network claim by
this result.

## Findings

### P1

1. The frozen baseline configuration has `allow_tf32=true` and
   `cudnn_benchmark=true`, while the candidate has both disabled and records the
   resulting backend audit. Thus the candidate accuracy and delta against the
   frozen reference are citable, but the delta must not be presented as a
   strict backend-matched causal effect of only attention/QDQ transforms.
2. The sealed profiles contain per-sequence aggregates rather than raw
   per-frame prediction/error arrays. This review independently recomputes all
   aggregate levels available in the profiles, but it cannot rederive optical
   flow errors from raw predictions without a new evaluation.

### P2

1. The output intentionally retains the M2044 result schema and engine producer
   SHA. This double-sealed review supplies the missing M2045 successor lineage.
2. The result does not capture complete package versions, remote host identity,
   GPU model, driver, or CUDA version.
3. The population is the explicitly named local DSEC validation list, not an
   official hidden test set.
