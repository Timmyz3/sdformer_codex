# M1023 decoder stratified block-reset source r2 receipt

Status: `PASS_M1023_R2_SOURCE_ONLY__M1024_INDEPENDENT_HAMMER_REQUIRED`.

This additive wrapper leaves M1014 r1 unchanged and closes the three P0 findings from the M1017 negative hammer.

1. Selection now accepts only an explicit scalar metadata schema. A recursive, case- and punctuation-normalized semantic scan rejects cycle, latency, runtime, elapsed-time, throughput and speedup fields at any nesting depth. Unknown fields and nested values also fail closed.
2. Candidate and baseline now compare the complete canonical boundary/fill/drain request sequence. Equality includes kind, address sequence, width, banks, dependency roles, port/resource, beats, initiation interval, latency, service-cycle charge, return distance and outstanding limit. Count equality alone is insufficient.
3. The public estimator has three states. Above 10% worst relative CI half-width, candidate cycles, baseline cycles and speedup point estimates are all `null`; only bounds and coverage survive. Between 5% and 10%, points are diagnostic/adaptive and not admitted. At or below 5%, a point may become eligible only for a later independently released run and remains non-citable here.

Eleven synthetic/unit fault-injection tests passed. They cover timing aliases and nested paths, reset kind/service/address/byte/issue/bank mutation, and all three CI states. The normal synthetic pair remains 649 versus 649 cycles with exact reset-sequence SHA `a1a45bca...`.

No real payload/window/full row, EDA, GPU or remote task ran. M1023 does not authorize execution. A receipt-blind M1024 independent hammer is required before any separate execution release can be considered.
