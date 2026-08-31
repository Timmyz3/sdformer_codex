# M446 independent raw-evidence recomputation

This directory independently audits M438 without reading either M438 receipt JSON as a numeric source.

The recomputation used the sealed M425 `memh` workload, the original VCS compile/simulation/assertion artifacts, the raw mapped-gate SAIF, the PrimeTime annotation reports, and direct inspection of the wrapper, testbench, mapped netlist, UCLI, PT Tcl, and exact-SHA runner.

Key independent results:

- Workload: 64 phases, 192,000 rows, 63,067 PWP rows, 921,166 contributions, and 48,435,456 reconstructed lanes.
- Functional replay: all five mismatch classes, accepted-transaction X count, protocol error, and assertion failure count are zero.
- Direct gate SAIF: 22,800 signal entries, 21,827 with nonzero `TC`, 973 with zero `TC`, no nonzero `TX`, and no duration inconsistency.
- PrimeTime: 22,800/22,800 exact-mapped annotated nets (100%); 21,827/22,800 nets have at least one toggle (95.732456%). Inconsistent annotation count is zero.
- PrimeTime never invoked `update_power` or `report_power`; M438 contains no power or energy result.

Decision: GO only for a separate independently controlled PTPX run. M438 itself remains NO-GO for power, energy, system/Conv speedup, paper-PPA, or headline claims.
