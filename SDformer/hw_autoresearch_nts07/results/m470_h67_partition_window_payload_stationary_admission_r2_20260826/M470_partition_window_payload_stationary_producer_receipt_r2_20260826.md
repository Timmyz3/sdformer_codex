# M470 producer admission r2: identity recovery

The r1 admission receipt is revoked.  Independent hammer found that its
hand-copied producer-result SHA was wrong even though the sealed producer
directory itself validated correctly.  The actual producer result SHA is
`7817460e7c13e73c20b80de4224ffac285d3a604edf258c182e3c8c78a9ad165`.
No producer file, seal, cycle equation, capacity equation, or measured number
was changed.

The corrected identity chain reproduces 147 points.  The best full-preflight
point remains P5/four banks/row64/128 B/cycle at 892,869,158 cycles, or
1.286498482x against the strong-zero path forced through the same spill
schedule.  Among the requested P={1,2,4,8} landmarks, P4 remains best at
964,742,918 cycles and 1.263148133x; stored P8 does not pass the macro gate.

The negative frontier result is also unchanged: P5 spills and reloads
18,823,680,000 B in each direction, reaches 38,563,705,600 B total DRAM
traffic, runs at only 0.831195007x of the 742,148,386-cycle strongest-zero
anchor, and at 0.977133951x of the already sealed 872,452,768-cycle M468R3
240-KiB stored-PWP point.  Therefore M470 is still independent-hammer-only,
74/100 producer score, RTL hold, and no performance/system/headline claim.
