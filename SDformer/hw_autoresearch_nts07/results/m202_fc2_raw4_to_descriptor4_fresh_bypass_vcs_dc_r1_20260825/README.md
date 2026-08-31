# M202 F4 fresh-arrival bypass standalone VCS/DC

M202 closes the specific M199/M201 semantic gap found by independent review:
when the reservoir is empty, a legal nonzero raw4 packet may compact and emit
the first descriptor packet in that same cycle.  Stable order, token identity,
zero elision and the no-cross-window-packet rule remain unchanged.  Residual
fresh descriptors after the first window boundary enter the eight-entry queue.

The exact-SHA VCS run conserves 2,305 descriptors over 241 tokens and 3,643 raw
beats under random stalls and four protocol attacks.  The M202-specific SVA
observes 604 accepted fresh bypasses, including eight full raw4 nonzero cases.
The reused M201 scoreboard prints a legacy `PASS M201` literal through a
test-only interface adapter; the compiled design and bound fresh-bypass SVA are
M202 and their hashes are sealed by the runner.

TSMC28 3.000 ns logic-only DC reports 5,729.472015 um2, 8,083 cells, 873
sequential cells, 55 logic levels, +0.2667 ns setup slack and +0.0004 ns hold
slack.  This is 1.078532x the area of registered F4 M201, but remains 122.783026
um2 below the independent M201 review's conservative 5,852.255041 um2 ceiling.

Using M199's exact-payload stage-aware S4/F4 front-end point (90,112,890 cycles,
1.272243x) and adding standalone M202 logic to M186 produces a conditional
1.102227x throughput/area proxy.  This is only 1.002864x above the deliberately
optimistic comparison that gives registered F2 its abstract numerator without
charging an F2 bypass.  The margin is real but too small to freeze width before
integrated SRAM write-port and result-backpressure screening.

M202 is standalone logic-only evidence.  It is not complete FC2, physical PPA,
FFN speedup, system speedup, or a headline result.
