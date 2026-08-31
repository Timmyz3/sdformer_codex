# M258 ATLIF trace-executable I/O sensitivity

M258 replays the exact ordered 45-context T10 population shared by all ten H67
trace samples.  One inference contains 7,318,350 factor tiles, 7,318,350
ordered tags and 36,591,750 five-beat results.  Both serial and phase-decoupled
boundaries pay the same one-cycle configuration transfer and one-cycle release
barrier per context; contexts drain before release.

With an always-ready result sink, serial and candidate take 73,183,590 and
36,592,065 cycles, respectively (`1.999985x`).  The candidate speedup is
`1.874988x`, `1.749990x`, and `1.499994x` at periodic ready fractions 93.75%,
87.5%, and 75%.  Thus the near-2x ATLIF result depends on sustaining the
one-beat/cycle result boundary; the full sensitivity table must accompany it.

The executable model preserves tag/beat order and includes finite FIFO
backpressure accounting, but it is not a matched integrated stage1+stage2 RTL
boundary.  It does not admit throughput/area, trained rank3 accuracy, energy,
system speedup, paper PPA or a headline claim.
