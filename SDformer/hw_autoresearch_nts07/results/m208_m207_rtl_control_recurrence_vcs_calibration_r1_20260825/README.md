# M208 M207 RTL control recurrence VCS calibration r1

The independent software recurrence matches all 256 continuous-source
Synopsys VCS cases exactly.  The sweep covers all four H67 FC2 geometries,
dense/sparse/prefix/zero payloads, partial and odd-full tails, trailing-zero
scan, dual-buffer backpressure, and terminal-group collapse.  Maximum observed
M202 queue occupancy is two and maximum descriptor hold is 40 cycles.

This admits the recurrence for a frozen-payload replay.  It does not by itself
admit an FC2, FFN, physical, system, or headline speedup.  `docs/359` was not
modified.
