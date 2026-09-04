# M1141R6 additive structural reset-chain checker — author review

Verdict: **PASS; authorize only a different-author bounded/static hammer.**

The checker recursively follows one driver and one proven unary input at each level. It accepts zero or more foundry-proven `BUFFD1BWP35P140` stages and requires exactly one foundry-proven `CKND0BWP35P140` inversion before terminating at `rst_core`. The cell-library SHA is frozen, and the primitive bodies were independently checked as `buf(Z,I)` and `not(ZN,I)`.

The frozen quarantine netlist was read only. All 337 shadow clear bits pass: 12 clear nets, 75 registers on direct inverter paths, 262 on buffer-then-inverter paths, and maximum path depth two. Thirteen bounded mutation classes reject unknown gates, constants, cycles, multiple drivers, reconvergence, incorrect polarity/root/pins, zero or two inverters, excessive depth, active async set, and duplicate instances.

No mapped VCS, VCS, DC, launch, or retry ran. The frozen subject and complete failure-directory tree are unchanged. Diagnostic area/setup values were neither emitted nor promoted. This is checker-source evidence only and is not mapped functionality, PPA, performance, or paper-citable evidence.
