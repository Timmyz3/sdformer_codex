# M1739 source author audit

M1739 does not add or modify RTL. It binds the M1701 mapped top and its nine
parent SRAM macros. The mapped activity TB drives only public inputs and checks
only public outputs/counters. M863 was not reused verbatim because its attack
closure contains hierarchical `force`/`release` operations.

The 64-row task is an ep34-density-conditioned directed component workload,
not a captured inference. Its support popcounts cover the active-only
p25/p50/p75 values 1/2/4 derived from the sealed M1590 ep34 ledger. Residual
and psum values remain synthetic. It does not reproduce the empirical support
frequency and must not be described as representative or production activity.

The future accounting has two non-overlapping terms: standard-cell logic is
the mapped-top PTPX report minus the diagnostic nine-macro Liberty report; the
nine SRAMs are then priced once from public macro read/write counters and the
frozen SRAM datasheet model. The result unit is pJ per directed component
workload. Weight/psum/metadata memories, DRAM, the full C1 schedule, the full
network and energy per frame remain excluded.

Author-side Python 3.6 and 3.10 compilation and five tests passed. Three runtime paths
were exercised without EDA: public-counter conservation (including negative
mutations), exact-window SAIF parsing (duration and TX mutations rejected),
and logic/macro energy separation with a no-double-count check.
No VCS, simv, SAIF, PTPX, license query, attempt or result was created.

The M1743 release and its canonical M1740 timing result are now statically
pinned by release, sidecar, manifest, outer-seal and receipt SHA. The receipt
schema/status, 3-ns coordinate, nine-macro scope, 16,549-point Formality proof,
PrimeTime setup/hold values and exact claim boundary are checked before a
future M1739 energy attempt can consume its own authority.
