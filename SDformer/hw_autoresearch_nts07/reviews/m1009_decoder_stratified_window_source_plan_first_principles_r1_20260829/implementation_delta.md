# M1010 minimal additive source delta authorized by M1009

Do not edit M785, M890, M896 or M946. Add one wrapper only.

## Required source units

1. `FrozenAuthority`: verify exact M785/M890/M896/M946 and M1008 identities.
2. `BlockIndexer`: traverse compressed metadata without scheduling expanded
   requests; emit complete block identities, counts, stratum features and a
   sealed population-census hash.
3. `DeterministicSelector`: sort by SHA256 of the frozen seed and canonical
   block identity; choose pilot windows before any cycle result exists.
4. `BlockResetAdapter`: construct one common-charged boundary-ready marker,
   explicit cold fills, original body transactions, commit when applicable,
   and explicit drain. Remap cross-block dependencies only to the boundary
   marker. Record original and reset transaction-ID censuses separately.
5. `ExactWindowMiter`: instantiate fresh M896 RUN-GTLS and M890 reference
   schedulers on the identical block; compare all 14 frozen exact fields plus
   M1009 conservation fields.
6. `WindowSealer`: seal index first, then each selected result independently;
   never aggregate an unsealed window.
7. `Estimator`: reuse
   `check_m1009_decoder_stratified_window_source_plan.py` finite-population
   formulas; preserve paired candidate/baseline covariance.

## Fail-closed source modes

- permitted before release: `--validate-source`, `--self-test`, and synthetic
  estimator tests only;
- forbidden: real window execution, full row, output publication, production,
  EDA, GPU and remote work;
- D1 selector or generator calls must raise immediately;
- any attempt to label continuous-M785 cycles or transaction ratio as speedup
  must raise immediately.

## Pre-execution hammer gates

- exact source hashes and sealed M1008 identity;
- block-index census on a tiny synthetic fixture;
- stratum priority and deterministic selection invariance;
- boundary token, fill/drain and transaction-ID conservation attacks;
- exact M890/M896 synthetic window miter;
- estimator census and partial-sample CI tests;
- proof that no real payload/window function is reachable without a later
  release identity.
