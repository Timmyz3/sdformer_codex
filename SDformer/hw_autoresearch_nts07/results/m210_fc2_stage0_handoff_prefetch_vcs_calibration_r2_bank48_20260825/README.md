# M210 stage-0 handoff prefetch VCS calibration r2

This r2 supersedes the pre-adversarial r1 calibration.  Independent review of
M207 found that a legal four-descriptor packet can place 48 events in one bank,
while M207 stored the packet sum in five bits.  The truncation can underflow a
closed window's bank count and deadlock.  M210 widens the packet sum to six
bits, guards the legal 96-event/window capacity, and adds SVA bounds.

The dedicated Synopsys VCS regression reaches the exact 48-event packet bound
twice and drains 192 output groups to one correct token completion.  The broad
directed test, focused stage-0 handoff/stall test, and 256-case continuous-source
calibration all pass; the software recurrence remains 0-mismatch.  M207/M208/
M209 are retained as historical evidence but are not functionally admitted.

M210 also preloads the first group of an already closed next stage-0 window on
the current window's release edge.  The focused dense case improves from seven
M207 cycles to six M210 cycles without crossing window or token ownership.
No complete-FC2, physical, FFN, system, or headline speedup is claimed here.
`docs/359` was not modified.
