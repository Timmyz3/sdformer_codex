# M1326 shadow hammer of M1325

Verdict: `FAIL_DO_NOT_CITE__M1327_FRESH_NAMESPACE_VALIDATOR_REQUIRED`.

M1325 correctly derives the frozen four direct M1227 keys, sets capture 100,
deep-copies the exact cohort, propagates the new result path in its delegated
capture, and exposes no production CLI or direct attempt consumer.  Author
tests pass 10/10.

The release path is nevertheless blocked.  Its identity projection calls
`M1319.validate_exact_m1313_m1314`, which calls
`M1249.validate_production_launch`, which unconditionally calls
`M1249.ensure_fresh_namespaces`.  The old M1249 attempt is already consumed.
The later delegate rewrites only `CANONICAL_RESULT`, so new M1325 constants do
not repair the earlier old-attempt/log freshness rejection.

M1327 must retain M1319's extended identity and exact M1313/M1314 validation
without invoking old M1249 namespace freshness or consumption.  Only a future
release may check and consume the three new namespaces exactly once.

No remote access, GPU, capture, attempt consumption, or production occurred.
