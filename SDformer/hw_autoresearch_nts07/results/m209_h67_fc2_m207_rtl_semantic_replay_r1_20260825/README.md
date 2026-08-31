# M209 H67 FC2 M207 RTL-semantic replay r1

The VCS-calibrated M202-to-M207 controller recurrence was replayed over all 120
frozen H67 FC2 payload records: 5,580,000 tokens, 36,480,000 raw 96-bit beats,
18,869,376 nonzero descriptors, 6,523,707 compact windows, and 143,894,510
events.  Every payload SHA, extent, and popcount was rechecked.

M207 takes **92,878,814 isolated-token cycles** with ready group/done sinks.
Per-stage cycles are 23,685,015 / 15,610,475 / 38,740,976 / 14,842,348.  The
maximum M202 queue occupancy is seven and the maximum descriptor hold caused by
two occupied window buffers is 672 cycles.  All 3,716,056 nonzero tokens retire
on the last replay group through terminal collapse.

The earlier M203 analytic schedule is 3.075819% optimistic.  The legacy
S1/F1/W1 analytic baseline divided by this exact M207 result is 1.234355878x,
but it is not yet a matched RTL-control baseline and therefore is deliberately
not admitted as an RTL, physical, complete-FC2, FFN, system, or headline
speedup.  The dominant control gap is stage 0, motivating an adjacent-window
handoff prefetch.  `docs/359` was not modified.
