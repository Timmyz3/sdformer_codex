# M1169 — C1 depth-one II=2 interval replay source receipt

Status: `PASS_M1169_EXACT_II2_INTERVAL_REPLAY_SOURCE_AND_BOUNDED_ORACLE__FRESH_DIFFERENT_AUTHOR_HAMMER_REQUIRED__NO_PRODUCTION`.

M1169 closes the modeling gap identified by M1166 without replaying the
production schedule.  For each sealed M1141 task, M1137 contributes 87 or 88
contiguous requested beats.  Under M1162's depth-one, zero-request-stall,
one-cycle-response protocol, the first completed issue is
`max(requested_first+1, previous_completed+2)` and subsequent completions in
the task are exactly two cycles apart.  The resulting recurrence retains only
one state record per axis and expands no beats.

The bounded oracle compares the closed form against explicit beat simulation
for 1,008 exhaustive gap/overlap cases plus 2,000 deterministic random trials.
It also injects request-accept and response stalls: those traces are never
silently treated as the fixed zero-stall model and can only delay completion.
Seven unit tests pass, including quota conservation, provenance, bool, drop,
duplicate and ordering attacks.

No production number is reported.  The small 1.1667x and 1.0417x values in the
bounded fixture are synthetic test coordinates only.  A later production
successor must pin (1) the sealed M1161 production result, (2) a fresh
different-author hammer of that result, and (3) a fresh different-author
hammer of this M1169 source.  Even then, ratios are component weight-service
schedule ratios, not RTL cycles, system speedup, traffic, energy, or PPA.

`docs/359` remains at
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
