# M200 independent hammer review

Score: **86/100**, conditional pass for the standalone compactor only.

The sealed input/output manifests verify, and an independent VCS testbench
passes 307 tokens and 3,424 descriptors with exact order, bitmap, index,
window-boundary, token-tag and done-count conservation.  It explicitly attacks
multi-cycle descriptor stalls, raw data held behind those stalls, release into
same-cycle push/pop, mid-lane compact-window closes, all-zero data, one/two/three
lane raw residuals, reset with a pending stalled output and malformed packets.
There are no drops, duplicates or reorders.

The first exploratory stall bug is adequately repaired for this conservative
standalone interface.  The old run had 22 descriptor-stability assertion
failures; the current gate freezes raw ingress whenever a visible descriptor
is stalled.  Sealed VCS and the independent 493-stall-cycle attack both pass.
This repair does reduce decoupling during output stalls, so it is not an
integrated backpressure-throughput result.

An independent Synopsys DC replay exactly reproduces 4,808.286021 um2, 6,296
cells, 873 sequential cells, 31 logic levels, 1.0599 ns setup slack and 0.0007
ns hold slack at 3 ns.  Hold is MET, but 0.7 ps is too narrow to describe as a
physical margin.  The additive M186+M200 arithmetic also reproduces the
1.099079848 throughput/area proxy; its numerator remains M199 abstract replay,
not measured M200 RTL cycles.

Matched S4/F4 is worth doing and is P0 before choosing the paper point.  M200's
eight-entry F2 reservoir uses 873 sequential cells, while an F4 design may use
shallower storage and simpler shifting.  Halving the descriptor bus therefore
does not establish lower total area or energy.  The comparison must share the
same input, window, stall and queue/bypass assumptions and run the same VCS and
3 ns DC screens.

No physical, complete-FC2, FFN or system-speedup claim is admitted.
