# M1759 C1 execution release author self-check

Status: `PASS_M1759_C1_RELEASE_AUTHOR_SELF_CHECK__NO_EDA`.

The double-sealed release binds the exact M1757 runner, checker, test, source contract, author receipt, and the independently sealed M1758 review triple. It authorizes one future campaign only: one fresh VCS compile, one simv run, one checked SAIF, and one PTPX run, with no automatic retry or alternate workload after attempt consumption.

The measurement boundary is intentionally narrow. `UNIT_DELAY` is permitted only to generate functional mapped-gate activity. Timing remains the independently sealed M1740 PrimeTime prelayout result. Primary PTPX accounting is the whole mapped C1 top including exactly nine linked SRAM Liberty macro instances. The mixed TT-standard-cell/SSG-SRAM result is a mixed-corner component estimate, not single-corner signoff. The SRAM datasheet estimate is an alternative sensitivity only and must not be added to whole-top PTPX.

This author check ran no VCS, simv, SAIF generation, PrimeTime, PTPX, license query, attempt, or result. The frozen docs/359 digest remains exact.
