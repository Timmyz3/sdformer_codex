# M394 H67 ep35 q32/O4 burst-streaming rebinding

M394 replaces the rejected PAFT-ep4 runtime population used by M381 with the
frozen H67 ep35/no-running M40 S10 payload.  The q32 catalog remains a disjoint
H67 train-only dictionary; strict fallback preserves every nonzero row exactly.

At the frozen robust point (`cmd32`, descriptor SRAM `L8/II1`) the standalone
four-bottleneck-Conv recurrence is `742148386 / 669012336 =
1.1093194341x`.  Across 17,280 phases, 24,534,432 of 51,840,000 source rows are
exact zeros, and 16,971,357 active rows use exact center-plus-residual replay.

This is a trace-backed cycle-simulator result, not RTL-measured performance.
The M393 17,280-phase controller-cycle miter remains required.  Energy,
physical SRAM, full-network speedup, paper PPA and DATE headline are false.

Two early executions produced no result directory: one failed on a payload
SHA field-name typo; one was deliberately interrupted after 3/10 metric
groups to replace the all-q scan with a q32-only scan.  The final q32-only
implementation cross-checks one phase per sample against the frozen M381/M339
all-q implementation.
