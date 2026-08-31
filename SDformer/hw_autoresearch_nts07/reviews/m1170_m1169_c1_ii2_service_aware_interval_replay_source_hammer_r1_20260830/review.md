# M1170 independent hammer of M1169 II=2 interval replay source

Status: `PASS_M1170_M1169_II2_INTERVAL_SOURCE_HAMMER__AUTHORIZE_ONLY_FUTURE_GATED_SUCCESSOR_SOURCE`.

The M1169 recurrence is correct for its deliberately narrow service model.  A
depth-one service with zero request stalls and one-cycle responses accepts beat
`k` no earlier than `max(requested[k], previous_completion+1)` and completes it
one cycle later.  Therefore an interval completes first at
`max(requested_first+1, previous_completion+2)` and thereafter at II=2.

The independent hammer did not reuse M1169's explicit oracle.  It compared the
closed form against a separately implemented beat service across 19,074
exhaustive first/last/delay cases, 10,000 random multi-task gap/overlap trials,
and 1,074 request/response-stall attacks.  It also attacked first-completion
off-by-one, wrong II, boolean-as-integer acceptance, drop, reorder, incomplete
terminal, exact-field-set and provenance mutations.  All attacks were either
rejected or detected.

The production floor quota was independently enumerated without opening the
production JSONL: 616,896 tasks carry 87 beats and 195,264 tasks carry 88 beats,
summing to 70,853,184 beats per axis and 212,559,552 over three axes.  Retained
state remains O(axes); no beat stream was materialized.

Authorization is intentionally limited.  This hammer permits a future
successor **source** only after it pins a sealed M1161 production result and a
fresh different-author hammer of that result.  It does not authorize production
execution now, and it does not establish RTL cycles, system speedup, traffic,
energy, or PPA.  The production schedule, canonical rows, and M1161 result were
not opened or consumed.  No VCS/DC/PTPX/GPU/remote action was performed.

`docs/359_DATE终局冻结_20260813.md` remains at
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
