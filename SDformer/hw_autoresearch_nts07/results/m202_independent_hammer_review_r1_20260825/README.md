# M202 independent hammer review

M202 earns **84/100** and a conditional standalone pass.  The fresh-arrival
bypass is real: an independently authored VCS bench (no production TB/SVA)
passes 1,626 tokens and 6,905 descriptors with exact tag, bitmap, beat-index,
compact-window and done-count conservation.  It also covers stable output
stall, raw backpressure, residual descriptors after a mid-lane boundary, reset
flush and four sticky fail-closed attacks, with no drop, duplicate or reorder.

Independent 3 ns Synopsys DC exactly reproduces the sealed numeric result:
5,729.472015 um2, 8,083 cells, 873 sequential cells, 55 logic levels,
+0.2667 ns setup slack and +0.0004 ns hold slack.  This remains ideal-clock,
ZeroWireload, zero-macro logic-only evidence; 0.4 ps hold margin is not a
physical margin.

The central M199 cycle-match claim is rejected.  Exhaustive independent VCS
comparison of every bitmap through one to eight raw beats at all three legal
window depths finds 1,302 equal patterns, 160 where RTL is faster, and 68 where
RTL is slower.  Two minimal depth-two witnesses are:

- raw bitmap occupancy `110` (LSB/earliest first): M199 charges two scan
  segments but M202 accepts the full residual raw packet in one cycle, so the
  measured front-end service is 2 versus 1 cycles;
- occupancy `01111`: M199 restarts an unaligned S4 group after each compact
  boundary and charges 2 cycles, while M202 leaves a registered residual and
  cannot co-emit it with the next fresh arrival, so RTL takes 3 cycles.

Thus M202 closes the empty-reservoir first-arrival bubble but does not implement
M199's full window-segment recurrence.  The 1.272243182967x stage-aware abstract
speed cannot yet be assigned to this RTL.  The recomputed 1.102227394
M186-additive density and 1.002863801 factor over optimistic registered F2 are
arithmetically correct only if the mismatched abstract numerator is assumed;
they remain conditional screens, not admitted RTL-cycle or physical results.

The next milestone should be an exact frozen-H67 cycle replay of the actual
M202 aligned-packet/queue/ready semantics.  If that replay loses material
cycles, add a carried-plus-fresh co-emission merge and/or an explicit
residual-lane rotate/skid contract, then repeat independent VCS and DC.  Because
the current F4 additive advantage over F2 is only 0.286%, a matched bypass F2
comparison is required before freezing width.  Do not upgrade M202 to
integrated-front-end, complete-FC2, physical, system, or headline status.
