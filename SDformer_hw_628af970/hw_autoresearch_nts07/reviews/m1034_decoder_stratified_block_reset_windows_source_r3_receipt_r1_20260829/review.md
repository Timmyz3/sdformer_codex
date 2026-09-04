# M1034 decoder stratified block-reset source r3 receipt

Status: `PASS_M1034_R3_SOURCE_ONLY__M1035_INDEPENDENT_HAMMER_REQUIRED`.

M1034 fixes the sole M1024 P0 without modifying M1023 r2. Instead of mutating the raw estimator result, it constructs one exact-schema publication envelope. In the CI hard-stop state, `point_estimates` is `null`; recursive numeric walking finds only CI bounds, CI widths, the critical value, population/sample counts and finite-population coverage. Per-stratum candidate/baseline means are not copied.

The original M1024 example now exposes zero `candidate_mean_cycles` or `baseline_mean_cycles`. Injecting either a nested cycle mean, speedup, FPS, throughput, or a non-null hard-stop point container fails the recursive schema validator. The 5–10% state retains explicitly diagnostic points with `point_estimate_admitted=false`; the ≤5% state remains only a later-release candidate and is not paper-citable here.

Eight tests, source self-test, static validation, source validation and Python compilation passed. No real payload/window, full row, EDA, GPU or remote task ran. M1035 independent hammer remains mandatory.
