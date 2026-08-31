# M210 stage-0 handoff prefetch VCS calibration r3 exact-input seal

This r3 reseals M210 after the independent hammer review found that the r2
bank48 testbench had gained an elapsed-cycle print after its input receipt was
written.  The current exact testbench was recompiled and rerun with Synopsys
VCS.  The legal worst-case packet reaches 48 events in one bank twice, drains
192 output groups, and completes one token in 195 header-to-done cycles.  The
cycle-exact software recurrence independently returns the same 195 cycles for
eight descriptors carrying twelve bank-0 events apiece.

The broad directed regression, focused stage-0 handoff/stall regression, and
256-case continuous-source sweep remain zero-error.  M210 uses six-bit packet
bank sums, checks the 96-event per-window capacity, and asserts the legal
48-event packet bound.  This closes the M207 five-bit truncation deadlock.

The performance scope is an isolated FC2 sparse frontend.  No complete-FC2,
physical, FFN, system, or headline speedup is claimed here.  `docs/359` was not
modified.
