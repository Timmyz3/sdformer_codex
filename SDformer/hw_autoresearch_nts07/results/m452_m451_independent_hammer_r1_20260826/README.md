# M452 independent M451 hammer

M452 independently exhausts the legal signed arithmetic domain of the frozen
M451 adapter, reruns protocol/fail-closed checks with independent SVA, verifies
the M451/M449 seals, and reviews resource-fairness boundaries.

The VCS run has its own `RUN_MANIFEST.sha256` and outer seal.  The independent
review and VCS seal are additionally bound by `M452_REVIEW_SHA256SUMS` and its
outer seal.

This evidence authorizes standalone DC, mapped-netlist Formality, and
prelayout PT only.  It does not admit memory concurrency, cycle speedup,
resource-normalized performance, power, energy, system speedup, or a DATE
headline.
