# M203 frozen-H67 replay of the exact M202 recurrence

M203 replaces the M199 arbitrary-boundary segment recurrence with M202's actual
aligned raw4 packet, eight-entry queue and ready/valid behavior.  It models the
empty-queue fresh bypass and the registered residual path exactly; a nonempty
queue cannot co-emit with a fresh packet.  Per-window close intervals are then
fed into the pinned M199 finite dual-window wall model.

All 120 frozen H67 FC2 payload records reproduce 5,580,000 tokens, 36,480,000
raw96 beats, 18,869,376 nonzero descriptors and 6,523,707 compact windows.
Against M199, 5,387,736 tokens have equal front-end service, 189,567 are faster
and 2,697 are slower.  The observed M202 queue maximum is seven, so the RTL's
eight entries are sufficient on the frozen payload.

The differences almost cancel under the finite-buffer wall model.  M199's
stage-aware S4/F4 point is 90,112,890 cycles; exact M202 recurrence is
90,107,277 cycles, 5,613 cycles lower.  Speed versus the pinned S1/F1 W1
baseline is therefore 1.272322x rather than 1.272243x.  The W1 wall is exactly
unchanged at 94,761,587 cycles; paired-window drain contributes 1.050376x.

This replay resolves the M202 recurrence question but also sharpens the real
RTL gap.  Current M184 accepts one descriptor per cycle and drains one window
at a time.  It implements neither the four-descriptor sink nor paired-window
bank-load fusion assumed by the 90.107M-cycle point.  M203 is exact-payload
cycle evidence, not measured RTL cycles, integrated density, complete FC2,
physical PPA, system speedup, or a headline result.
