# M424 evidence bundle

- `m424_m422_ptsta_independent_hammer_review_r1.json`: machine-readable decision, score, findings and exact claim boundary.
- `m424_m422_ptsta_independent_hammer_review_r1.md`: concise reviewer-facing audit.
- `independent_ptsta/`: separate Synopsys PrimeTime reproduction from the frozen M416 mapped netlist/SDC; it does not invoke the M422 Tcl or parser.
- `SHA256SUMS` and `SHA256SUMS.seal.sha256`: immutable evidence manifest and outer seal.

Accepted scope: M416 logic-only selected-slice pre-layout data-path STA at 3 ns. Not accepted: recovery/removal signoff, SPEF/post-route timing, physical SRAM, power/energy, system speedup, paper-ready PPA or all-internal-state equivalence.
