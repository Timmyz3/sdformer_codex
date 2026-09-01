# M1808 source-author attestation

M1808 is an additive recovery source for the uniquely consumed and sealed M1798 failure. I did not modify M1454, M1456, M1798, M1807, docs/359, or any prior result, and I did not launch VCS, PrimeTime PX, DC, Formality, a license query, GPU work, or remote work.

The new TB retains the eight asserted reset posedges and negedge deassertion. From the first post-release posedge it checks all 28 architectural/control outputs for X/Z and forbids accept/issue/retire or other architectural activity. Only the eleven public debug counters use the pre-existing three-cycle quiescent window. At the third edge all eleven must be binary and exactly zero; before any traffic starts, the TB proves the gate closed. Every later cycle restores the complete aggregate public X/Z check.

The one warmup tile, eight measured dense tiles, SAIF window, numeric result/beat checks, public counter conservation, backpressure coverage, release/retire check, and the M1798 ordered nine-tag scoreboard remain present. No force, initializer, ignore-X switch, hierarchy bypass, deleted check, or netlist edit is used.

The future one-shot is inert until a different-author M1815 review and exact double-sealed M1816 release bind the new source, old M1798 failure, M1807 diagnosis, M1456 timing evidence, and frozen docs/359. CPython 3.6 and 3.10 each passed static validation and rejected all 44 mutations. This receipt authorizes no EDA and makes no power, energy, performance, or headline claim.

