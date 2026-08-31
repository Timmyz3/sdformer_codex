# M201 independent hammer review

Score: **88/100**, conditional pass for the matched registered-only standalone
comparison.

All M201 sealed manifests verify.  An independently authored Synopsys VCS
scoreboard passes 322 tokens and 3,578 descriptors with exact tag, bitmap,
index, compact-window boundary and done-count conservation.  It covers 502
descriptor-stall cycles, 258 raw-stall cycles, residual raw packets, all-zero
traffic, mid-lane compact-window closes, one pending-output reset flush and four
sticky fail-closed attacks.  There are no drops, duplicates or reorders.

Independent 3 ns Synopsys DC exactly reproduces 5,312.286017 um2, 7,065 cells,
873 sequential cells, 32 logic levels, 0.6761 ns setup slack and 0.0006 ns hold
slack.  This is ideal-clock, ZeroWireload and pre-macro.  The 0.6 ps hold result
is MET but is not a credible physical margin.

The matched width result is useful but narrow.  F4 has 1.024884x the abstract
throughput and 1.104819x the standalone area of F2, making its standalone
throughput/area only 0.927649x F2.  After amortization against M186, F4's
conditional proxy is 1.113058x versus F2's 1.099080x, a 1.2718% lead.  The
384-bit downstream write path and activity energy are excluded.

The critical P0 is now directly witnessed: M199 adds fresh arrivals and emits
them in the same modeled cycle, whereas both M200 and M201 derive output valid
only from registered queue state.  With an empty queue and four accepted legal
nonzero inputs, independent VCS observes no descriptor acceptance in that same
cycle.  Therefore neither M199 abstract numerator is cycle-matched to these RTL
versions, and the additive ratios remain conditional algebraic screens.

For the performance-first line, advance fresh-arrival bypass F4 first.  M199's
S4/F4 point has zero post-emit backlog, while F2 needs a carried-plus-fresh
merge with backlog four.  A conservative gate is 5,852.255041 um2 at 3 ns: if
the bypass F4 compactor stays below that area, it beats even current F2 with
zero assumed bypass-area cost in the same M186-additive algebra.  If it exceeds
the gate, synthesize matched bypass F2 before freezing width.

No integrated, physical, complete-FC2, FFN or system-speedup claim is admitted.
