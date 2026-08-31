# M415 independent hammer of M414

Result: **95/100, P0/P1/P2 = 0/0/2, GO M416 DC and Formality.**

This directory independently reruns all three exact-SHA Synopsys VCS jobs, including all 17,280 H67 phases and 51,840,000 ordered rows.  It also audits the old M405 recurrence against the M414 balanced `(distance, local-ID)` minimum, the two-pass global-ID tie rule, fallback predicates, sequential state machine, compatibility wrapper, task ledger and claim boundary.

M414 is accepted only as an exact timing repair.  It does not change the frozen cycle ledger, accuracy, system speedup or DATE headline, and it has not yet passed DC, Formality or PrimeTime.

`exact_sha_three_job_rerun/RUN_MANIFEST.sha256` and its own outer seal protect the independent VCS rerun. `SHA256SUMS` and `SHA256SUMS.seal.sha256` form the M415 review-level second seal.
