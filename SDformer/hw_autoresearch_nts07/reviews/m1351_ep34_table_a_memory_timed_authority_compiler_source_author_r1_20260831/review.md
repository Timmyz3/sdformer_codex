# M1351 Table-A memory-timed authority compiler — source author review

Verdict: **PASS SOURCE ONLY; DIFFERENT-AUTHOR BLIND HAMMER REQUIRED.**

M1351 keeps the production authority allowlist empty and adds five mandatory
system-row boundaries: resolved workspace containment, row-invariant direct
logic rates, a sealed nonempty address trace, independently nonzero DRAM and
all seventeen SRAM planes, and a latency/stall receipt for every row and
population point. Address-timed cycles must equal the already conserved charge
cycles, and memory stalls must equal the SRAM plus DRAM stall partitions.

Frozen M1340 tests pass 10/10, frozen M1342 tests pass 16/16, and new M1351
tests pass 13/13. The source self-check also passes. No production allowlist,
Table-A row, capture, GPU, VCS, Synopsys task, energy result, speedup, or paper
headline was created.

The next step is a different-author blind hammer with fresh traversal, energy
rate, trace substitution, zero-plane, timing-coverage and stall-accounting
mutations. Only a later additive release may pin real production authorities.
