# M1261 — additive R12 checker/tests-only source hardening

Status: **SOURCE PASS; one fresh independent hammer is authorized.** Release
authoring and VCS remain forbidden.

M1260 found three P1 families in the M1258 checker while confirming the R12 TB
itself was correctly scoped. M1261 therefore adds only a new checker and test
file. The R12 TB remains exactly
`e13d630f4cf2e2f7e0264dc2325218aee4cc580497be3b37deb1ff7a641ad302`;
the base M1258 checker/tests, DUT RTL, SVA, and docs/359 are unchanged.

The new checker requires an exact allowlist for every force/release target and
exact per-helper statement inventories. It lexes SystemVerilog strings while
ignoring comments, then requires one real display for every phase and exactly
one PASS display with exact claim fields. Finally it counts exactly one
executable `normal_m935_completion()` call after comments and strings are
blanked.

Canonical checking passes with no errors. The new suite passes 18/18. It closes
all four concrete M1260 unexpected accepts: child-prefix shadow, boundary claim
comment decoy, integrated-normal claim comment decoy, and commented-out normal
completion. It also covers parent/child seam substitution, duplicate PASS,
phase decoy, duplicate normal call, headline inflation, semantic normal-task
drift, and a positive parent-force comment that must remain inert.

This is source/checker evidence only. It authorizes one independent hammer, not
a release, VCS, timing, cycle, speedup, PPA, energy, system, headline, or paper
claim.
