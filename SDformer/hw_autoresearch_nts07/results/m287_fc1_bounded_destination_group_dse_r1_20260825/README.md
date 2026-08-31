# M287 FC1 bounded destination-group DSE

M287 combines the frozen ep35 checkpoint with all 100 locally available
binary-FC1 payloads (112,213,979 source events) and the 620,302,905-cycle
compute envelope.  It quantizes each FC1 row symmetrically to signed INT8 and
screens static destination-group task elision.

The deterministic local rule is: omit a source/group task only when every
INT8 weight in the group has magnitude at most beta.  Each omitted task then
changes each destination accumulator by at most beta; a runtime omitted-source
counter gives the exact conservative bound form `beta * count`.  Beta zero
performs no pruning and is the exact engine subset.

Key optimistic task-compaction points for four-lane groups are:

- beta 32: 1.1158x FC1 task opportunity, 1.0202x FC1-only envelope sensitivity;
- beta 48: 1.4787x, 1.0658x;
- beta 64: 2.3905x, 1.1249x;
- beta 80: 4.5691x, 1.1752x, but 78.11% of weighted group tasks are omitted.

Thus FC1 alone crosses the precommitted 1.15 envelope gate only at beta 80
for four-lane groups (or beta 96 for eight-lane groups).  Those are aggressive
points, not yet “micro-lossy”.  FC2/Conv capture is required to test whether a
combined, lower-beta budget can meet the gate.

All speedups here are ideal task-compaction opportunities.  The compactor,
tag router, bank conflicts, accumulator service, accuracy, RTL and physical
cost are not modeled or admitted.

