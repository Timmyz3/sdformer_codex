# NB0 AAE Gap Diagnostic (2026-08-06)

Machine receipt: `neuron_autoresearch/NB0_AAE_GAP_DIAGNOSTIC_20260806.json`.

## Conclusion

- The local metric implementation is not the cause of the gap: released legacy AAE is 2-D direction angle, while `AAE_Benchmark` is the Barron/Middlebury `(u,v,1)` angle and passes the SHA-bound 8-test receipt.
- NB0 AEE is not proven converged, but its two angle metrics are already near a plateau. More epochs may improve AEE; they are not expected by themselves to turn local valid825 into the paper's official hidden-test AE 4.871.
- Paper Table I and local valid825 differ in formula, population, and server aggregation. Direct reproduction requires an official DSEC submission.

## Late Trends

| model | interval | AEE improvement | AAE-2D improvement | AE-3D improvement | status |
|---|---|---:|---:|---:|---|
| NB0 | ep24->ep29 | 5.558% | 0.437% | 1.258% | AEE not_proven_converged; angle near_plateau |
| H67 | ep25->ep30 | 2.468% | -0.075% | 0.179% | AEE not_proven_converged; angle near_plateau |

## Evidence Boundary

The six endpoint profiles are full-resolution 480x640, window 2x15x15, batch-one, BN no-running, and 825-frame local validation. They predate the three-aggregation profile schema, so the queued equal-plus10 re-evaluation must regenerate source points before pixel-global or sequence-balanced claims are made.
