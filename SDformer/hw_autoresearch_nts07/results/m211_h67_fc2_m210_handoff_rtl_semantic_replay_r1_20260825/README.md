# M211 frozen-H67 FC2 replay on corrected M210 recurrence

M211 rehashes and replays all 120 frozen H67 records through the M210 control
recurrence that is calibrated against Synopsys VCS.  The replay covers
5,580,000 tokens, 36,480,000 raw beats, 18,869,376 nonzero descriptors,
6,523,707 windows, and 143,894,510 events.

The isolated sparse-frontend result is 91,184,539 cycles.  Stage-0 adjacent
window handoff removes 1,694,275 cycles from the non-truncating M209
opportunity architecture, a 1.018580727x incremental factor.  Relative to the
older mixed-fidelity analytic ledger, the ratio is 1.257291107x, but that is
not a matched RTL baseline and must not be presented as an RTL, FC2, FFN, or
system speedup.

M207/M208/M209 admission remains revoked because the original five-bit bank
sum can deadlock on a legal 48-event packet.  M211 uses the corrected M210
six-bit recurrence and the exact-input r3 VCS seal.  No physical or headline
speedup is claimed.  `docs/359` was not modified.
