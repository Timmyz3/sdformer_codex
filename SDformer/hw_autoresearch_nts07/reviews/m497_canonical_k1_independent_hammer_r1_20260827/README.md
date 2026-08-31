# M497 canonical K1 independent hammer r1

Verdict: **93/100, CONDITIONAL GO**.

The exact-SHA Synopsys VCS bundle is internally consistent and supports one
narrow claim: with the same scalar-bank memory model and directed work, the
replicated K1x8 endpoint is 5.863399625x faster geometrically than canonical K1.
K1x8 has eight times the peak bank/service bandwidth, so this is a
resource-performance Pareto point, not a same-resource or system speedup.

M499 is a reasonable conservative K1 repair: it removes the only direct
`core_rsp_accept` dependency from the adapter request-ready cone and requires a
slot retirement to become registered before reuse.  Exact VCS then completes
all directed cases with zero numeric, transaction-multiset, and weight mismatch.
This establishes a functional repair.  It is not a formal proof that every
possible ready loop has been eliminated.

The downstream gate is conditional.  At the audit snapshot, M495/M496 still
instantiates and pins M494/M490 for `ARCH_MODE=0`.  The current M496 runner must
not execute as-is.  Rewire the K1 point to M499, re-lock every affected SHA, and
retain one identical top/SDC/library/port shape across all three elaborations.

See `independent_hammer_report_r1.md` for findings and claim boundaries.
