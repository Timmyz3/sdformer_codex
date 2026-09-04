# M528 single-port same-ledger recompute author handoff

## Outcome

The source-only M528 recompute package is ready for an independent static hammer. The author did not run the production CPU analyzer, any EDA/GPU job, or create RTL.

The implementation replays one frozen H67 ep35 row ledger and reconstructs M468 strong-zero, M473 same-coordinate bit, the M473 concurrent-1R1W opportunity ceiling, M504 all-write 1RW, M505 dead-write-only 1RW, and the combined PVRF ablation. The realizable M505 point is gated against M468 strong-zero and M473 bit; distance from the unphysicalized M473 ceiling is diagnostic only.

## Grain discipline

The analyzer emits two separate distributions:

- Sample-major: one continuous four-operator task stream per sample plus exactly 96,000 commit cycles. These ten rows sum to the aggregate.
- Operator-isolated: every sample/operator slice restarts the pipeline and omits commit. These forty rows are diagnostic and must not be summed as a sample-major pipeline.

Both grains report arithmetic mean, geometric mean, min, max, population CV, and ratio-of-sums.

## Capacity, traffic, and conservation

The M505 capacity ledger replaces the nominal 9,216-B rounded scratch with nine generated 128x128-bit 1RW macros (18,432 B), adds a conservatively rounded 1,152-B liveness bitmap, and must reproduce 213,376 B total with 32,384 B remaining under 240 KiB. Response-queue, descriptor-directory, matcher-source-store, resident-psum, and ping-pong obligations are mapped explicitly; standard-cell logic area is never converted into free SRAM bytes.

The traffic ledger separates weight DRAM, source SRAM, descriptor write/search/scan, parent scratch read/write, DMA commands, and commit cycles. Parent traffic is reported for all eight output blocks. Arithmetic issues, parent edges, active-row completions, trace rows, and commit vectors are fail-closed conservation gates.

## One-shot execution boundary

The runner refuses execution without a caller-pinned independent static admission and caller-pinned runner SHA. It also rejects canonical-output reuse, consumes a one-attempt sentinel immediately before Python launch, quarantines incomplete work, rejects this user's Synopsys/VCS/simv collision, and requires conservative memory headroom. A raw result remains non-citable until a separate independent result hammer.

No M528 number is currently admitted. Even after a passing CPU run, the scope remains one sequence and four bottleneck Conv3x3 operators—not RTL, PPA, energy, full-network/system speedup, or a DATE headline.
