# M2056 additive TSBG matched mapped-energy successor source hammer

## Verdict

**PASS at source-review scope, 98/100, P0/P1/P2 = 0/0/2.** M2056 repairs
every M2054 source blocker in a new namespace. No VCS, SAIF, PTPX, compiler,
license query, or retry was run. The sealed M2054 FAIL identity remains intact.

This source set is ready for authoring and separately reviewing an exact-SHA
one-shot runner. An unwritten runner is not authorized to execute.

## Frozen workload and denominator

M2056 does not use M2051's default slot. It requires global `WORKLOAD_SLOT=42`,
which is the pre-registered M2047 semantic anchor slot 0: the first captured
sample, layer 28 FC1, token quartet starting at zero, and 48 real source groups.
Selection is positional, not performance-selected.

The first stop hard-checks all of the following before SAIF can start:

- sample/layer/FC identity: `0 / 28 / FC1 / token0 / G48`;
- 149 rows, 1,278 issues, and 29,472 products;
- ordinary cache miss/hit/eviction `149/0/145`;
- TSBG cache miss/hit/eviction `48/101/44`;
- 1,788 versus 576 weight bundle beats;
- exact descriptor-preload endpoint at cycle 383.

At the selected-axis completion, the wrapper additionally checks all frozen
work and request counters and fixes the measured denominator at 20,292 cycles
for ordinary LRU4 or 7,569 cycles for TSBG. Existing sealed M2051 slot42 output
matches every field, but that prior run is evidence for the denominator—not a
substitute for the future M2056 mapped run.

## M2054 repairs

1. Each filelist now includes exactly its corresponding M2029 mapped Verilog.
   Ordinary includes mode 0 only; TSBG includes mode 1 only.
2. The adapter implements DC's ascending-unpacked flattening convention:
   bank 0 and lane 0 occupy high packed segments. Every `[0:7]` segment uses
   `7-bank`, nested `[0:15]` segments use `15-lane`, and Acc24 commit lanes use
   `(15-lane)*24`. Native packed handshakes remain direct.
3. The wrapper labels and enforces global slot42 instead of falsely calling
   global slot0 layer28/G48.
4. PTPX requires both SSG and TT DBs, loads both before the mapped SDC, and
   explicitly overrides the analysis point to TT/ZeroWireload afterward.
5. Selected public inputs and outputs are rejected on X/Z after every active
   measurement-window rising edge, not merely at the endpoints. Diagnostics
   distinguish load/reset, memory handshakes, bridge/commit/control, counters,
   bank metadata, bank/lane payloads, and accumulator lanes.
6. PTPX accepts only the axis. It derives top, strip path, and 20,292/7,569
   cycle denominator, and rejects a gate SAIF whose duration is not exactly
   `cycles × 3 ns`.
7. Simultaneous ordinary/TSBG compile defines fail at preprocessing.

## Two-stop and UCLI proof

The wrapper contains exactly two `$stop` statements. Each UCLI file contains
exactly three `run` commands and one enable, disable, report, and quit:

1. run to the asserted post-load/pre-execute stop;
2. enable mapped-scope power and run to the selected mapped completion;
3. disable/report power, then run M2051 through the remaining comparison,
   retired-identity replay, stale attack, two-reset recovery, legal service,
   scoreboards, and final machine PASS.

Ordinary and TSBG UCLI scopes exactly match their respective mapped hierarchy.

## Remaining P2 boundaries

- The one-shot runner is not yet present. It must pin these hashes, add no
  source, force only `+WORKLOAD_SLOT=42`, preserve the top/UCLI/PTPX mapping,
  prohibit retries, and receive its own static hammer before execution.
- The ep34 source masks are real, but M2051 uses deterministic directed INT8
  weights and this measured sample has positive nonzero source codes. A future
  result is a single component workload and may not be called real-weight,
  empirically bipolar, full-FC, network, or system energy.

`docs/359` remains unchanged at SHA256
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
