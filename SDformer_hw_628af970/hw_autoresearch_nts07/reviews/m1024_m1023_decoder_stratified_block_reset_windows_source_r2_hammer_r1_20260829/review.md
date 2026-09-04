# M1024 — independent hammer of M1023 decoder stratified-window r2 source

**Verdict: STOP execution release; author an additive r3 CI-redaction repair.** Score 86/100; P0/P1/P2 = 1/0/0.

M1017’s three-P0 FAIL authority and the M1023 source receipt verify against their manifests and outer seals. Source, checker, tests, contract, frozen M1014/routing dependencies, and `docs/359` identities match. The author’s 11/11 tests, source self-test, source validation, static checker, and isolated Python compilation all pass.

Two original P0s are closed under stronger independent attacks:

- Fourteen cycle/latency/time/runtime/speedup aliases spanning case, punctuation, and nested containers are rejected before selection. Unknown fields and nested values under allowed keys also fail closed.
- Every one of 21 canonical reset/service fields was independently changed for boundary, fill, and drain. All 63 asymmetric cases are rejected. Normal paired reset remains exact.

The CI repair is incomplete. For an above-10% case, the three top-level estimates are null, but the returned `strata` object still contains `candidate_mean_cycles=50.5` and `baseline_mean_cycles=50.5`. These are point-cycle estimates. Their presence contradicts both `HARD_STOP_REPORT_BOUNDS_AND_COVERAGE_ONLY` and the required rule that all cycle/speedup point values be suppressed above 10%. The static checker and 11 tests only inspect the three top-level fields, so they miss the nested disclosure.

The required r3 repair is narrow: recursively redact point-cycle and point-speedup fields in the above-10% publication object while retaining CI bounds and coverage/sample-count metadata; then add nested-output tests. The 5–10% diagnostic state and ≤5% later-release candidate state behave correctly.

D1 remains strict common-charge only; D0/D2/D3 frozen routing is intact. Zero-commit `COMMIT_TAIL`, windows above 10,000 requests, selectors above 32, and non-pilot geometry fail closed. No real payload, real window, EDA, GPU, or remote execution occurred.
