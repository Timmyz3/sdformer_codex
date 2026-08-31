# M200 raw4-to-descriptor2 standalone VCS/DC

M200 is the first FC2 source frontend in this line that consumes raw 96-bit
sn2 bitmaps directly rather than an oracle list of nonzero descriptors.  It
screens four raw beats, stably emits at most two descriptors and uses an
eight-entry queue with fail-closed backpressure and window-boundary packet
gating.

Exact-SHA Synopsys VCS passed 241 tokens, 3,643 raw beats, 2,305 conserved
descriptors, 324 output stalls, 326 raw backpressure cycles, four protocol
attacks and all ten SVA covers with no assertion failure.

TSMC28 3 ns logic-only DC reports 4,808.286021 um2, 6,296 cells, 873 sequential
cells, 31 logic levels, 1.0599 ns setup slack and 0.0007 ns hold slack.  There
are no macros or mapped multipliers.  Adding this standalone area to M186 gives
a conservative 1.099080x throughput/area proxy; it is not integrated density
or physical/complete-FC2/FFN/system speedup.
