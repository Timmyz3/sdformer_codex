# M269 independent hammer review of M263

Verdict: **the M263 evidence is reproducible, but the uniform balanced-Q20N2
candidate is rejected.** Evidence quality scores `92/100`; candidate readiness
scores `32/100`. There are no P0 findings, five P1 findings and two P2
findings.

## Independently reproduced result

The audit imports no M263/M263-DSE production analyzer. From frozen M233 arrays
it independently reimplemented the integer Q20 two-Newton path and recomputed
all `220,800` coefficient pairs across the exact 24 FFN BN targets. It obtained
zero rails and the same captured-interval maximum output-error bound,
`1.1332509529893287e-4`.

The exact paired ten-frame CSVs and spike profiles independently reproduce:

| Metric | Reference | Candidate | Relative delta | Wins/losses |
|---|---:|---:|---:|---:|
| AEE | 0.95015193607 | 0.96068277002 | +1.108331578% | 3 / 7 |
| DSEC Fl | 2.31059490344 | 2.32682522903 | +0.702430598% | 5 / 5 |
| Spikes/frame | 101,920,461.1 | 101,935,268.6 | +0.014528486% | 5 / 5 |

The predeclared AEE limit was `+0.25%`; therefore this is a firm no-go for
uniform downstream admission and for valid825. The M263 author made the correct
fail-closed admission decision.

## Semantic and population audit

The target set is exactly 12 BN1 plus 12 BN2 modules and matches normalized M233
NPZ keys. There are 240 hook calls, `220,800` channel-coefficient pairs and,
from network geometry, `4,377,600,000` BN output values. Hooks see
`[T,N,C,H,W]`, compute biased current moments over `T,N,H,W`, replace the affine
BN output before downstream consumers, and run after the `no_running` policy.
The SNN is reset before every frame. PyTorch still performs moment finalization.

Runtime output-error receipt arithmetic is internally exact: mean absolute
error `2.9720649150e-6`, RMSE `3.8566667235e-6`, maximum
`9.918212890625e-5`. The raw 4.3776-billion-value population is not archived,
so those runtime aggregates are not independently reconstructible from raw
values; only their arithmetic and the independent local-pair model are audited.

## Hammer findings

P1 findings:

1. AEE regresses `1.108331578%`, over four times the admitted limit, and loses
   on seven of ten frames.
2. Raw or independently reducible runtime delta shards are missing.
3. Moment reduction/finalization cost and schedule are absent; no RTL, VCS, DC,
   SRAM, timing or energy evidence exists.
4. All frames come from only `zurich_city_09_a`; no sequence transition or
   cross-sequence holdout is exercised, and captured ranges come from this same
   population.
5. The visible lower bound already includes 1,766,400 coefficient multiplier
   operations in ten frames, a 672-bit LUT and 10,598,400 bits if alpha/offset
   are materialized, but there is no demonstrated avoided-work numerator.

P2 findings: spike count is not hardware energy; the frozen remote and local
manifests are both needed, although relocation succeeds.

## Reproducibility and boundary

Clean replay passes. A deeply relocated source tree produces byte-identical
JSON. Mutating one payload while retaining its frozen manifest exits nonzero,
emits no result and reports payload SHA drift. `docs/359` remains unchanged at
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
M269 invoked no DC flow.

Next action is not wider uniform precision. Use a preregistered stage/BN1/BN2
sensitivity ablation with exact fallback, held-out multi-sequence frames and
explicit sequence boundaries. Only then schedule current-moment reduction and
coefficient reuse before committing RTL or Synopsys resources.
