# M295 independent hammer review

This directory independently reviews M293 without importing the M293 analyzer
or its checkpoint helper.  It rehashes and decodes all 60 little-endian patch
Conv bitpacks, reconstructs exact 3x3 source activity by coordinate mapping,
loads all six checkpoint Conv weights, rebuilds per-row INT8 values, and checks
the full 5x9 destination-group/beta grid.

The frozen 79-row operator ledger is then used to weight each eligible module's
task-retention fraction by its own cycle count.  This replaces M293's aggregate
task-ratio shortcut.  The correction is small (at most 37.284 ppm) and does not
change first-crossing decisions, but it is required before calling the Amdahl
overlay exact or scope-correct.

Main result: `independent_recompute_r1.json`.

Review and decision: `m295_m293_patch_binary_conv_independent_review_r1.json`.

Exact replay:

```bash
results/m295_m293_patch_binary_conv_independent_hammer_r1_20260825/run_m295_exact_replay.sh
```

No RTL, VCS, DC, STA, open-source RTL tool, modified-forward accuracy, or
`docs/359` update is part of this review.
