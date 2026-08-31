# M201 matched S4/F4 standalone VCS/DC

M201 is a contract-matched ablation of M200.  Both consume four raw bitmap
beats, preserve order and token/window boundaries, use an eight-entry queue
and apply the same fail-closed backpressure contract.  M201 can emit four
descriptors instead of two.

Exact-SHA VCS passes the same 241 tokens, 3,643 raw beats and 2,305 conserved
descriptors, including random stalls and four protocol attacks.  TSMC28 3 ns
logic-only DC reports 5,312.286017 um2, 7,065 cells, 873 sequential cells, 32
logic levels, 0.6761 ns setup slack and 0.0006 ns hold slack.

F4 provides 1.024884x abstract throughput over F2 but costs 1.104819x as much
standalone logic, so F2 wins standalone throughput/area.  When conservatively
added to the much larger M186 island, F4's proxy is 1.113058x versus F2's
1.099080x.  This slight additive advantage excludes downstream 384-bit write
wiring/ports, so the primary width remains conditional rather than frozen.
Both M200 and M201 emit only from registered queue state; neither implements
M199's fresh-arrival same-cycle bypass.  The abstract cycle numerators therefore
remain unbound to these RTL versions pending a bypass-capable VCS/DC screen.
