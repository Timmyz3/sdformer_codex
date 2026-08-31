# M523 VCS exit-31 independent failure receipt hammer r1

## Decision

Root cause is confirmed with high confidence: **functional behavior was observed to pass, but the run is not admitted**. The runner exited 31 because one terminal cover was not committed before `$finish`. P0=0, P1=2, P2=0; diagnostic score 98/100.

No tool was rerun. The consumed attempt cannot be reused, the quarantine cannot be promoted, and no functional, C2, performance, energy, PPA, system, or headline claim is authorized.

## What passed

VCS identity, compilation, and simulation each returned 0. Compilation contains no rejected warning/error token. Simulation contains no assertion failure, fatal, watchdog, or timeout token and prints exactly one required line:

`PASS M523 events=6 bundles=8 taps=43 full8=4 tails1=1 stalls=7 replacements=2 boundaries=6 cross_event=2 tag_flush=1 time_flush=1 stream_flush=2 stream_iso=1 fifo_max=18 phases=6/10/10/17 protocol_attack=1`

Nine cover points have positive matches: 4, 1, 2, 2, 4, 2, 2, 2, and 1 respectively. Only `cp_fault_drain_complete` has 25 attempts and 0 matches.

## Root cause

`cp_fault_drain_complete` is `protocol_error && $fell(busy)`. The TB waits until all expected taps are observed and `busy` becomes false, then on that same positive edge immediately checks the ledger, prints PASS, and calls `$finish`. The functional checks see the drained state, but VCS has not yet committed the concurrent cover result for that final sampled transition into `assert.report`. The runner correctly detects the zero-match cover at lines 457–465 and exits 31.

This is a terminal testbench sampling/settlement race, not an RTL or protocol P0. The TB independently confirmed 43 taps, eight bundles, sticky fault, event lockout, empty FIFO, no bundle, and exact counters before PASS.

## Failure-tree and seal audit

The canonical path is absent. The one-shot attempt marker exists, its manifest and outer seal verify, and its exact input identity verifies from the hardware root. The failure tree was atomically moved to `m523_c2d_k8_polyphase_tap_bundler_vcs_r1_20260827.failed_or_incomplete.2195382.quarantine` with exit code 31 and `DO_NOT_CITE` status.

The wrong-runner negative receipt remains double-sealed and proves exit 10 with no nested VCS or attempt. The quarantine has no top-level success manifest, receipt, topology, or RUN_COMPLETE because failure occurred before receipt construction; this is the intended fail-closed state. It contains exactly the historical two VCS symlinks, both resolving to regular in-tree files with recorded hashes. The independent review seals the diagnostic evidence; it does not convert the quarantine into a result.

`docs/359` remains at `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

## Minimum repair

1. TB r3: immediately after the drain `while` loop, add one `@(negedge clk_core);` before final checks and `$finish`. This preserves the strong `$fell(busy)` cover and gives the last positive-edge assertion evaluation time to settle.
2. Do not change RTL, SVA, or filelist.
3. Contract r3: pin TB r3 and repair the inherited stale `full8=3/tails1=3` sentence to `4/1`; keep every performance/C2/PPA claim false.
4. Runner r2: use new canonical and attempt names, pin the new identities, and retain all ten cover gates plus exact-two-symlink and atomic-publication checks.
5. A different independent static hammer must authorize exactly one new VCS attempt. The old attempt and quarantine are permanently ineligible.

## Claim boundary

Even a future clean VCS pass would establish only directed descriptor behavior. M523 still lacks the flattened weight key, bank-conflict deferral, and stored-weight identity needed for direct M218/C2 integration. No decoder speedup, energy, area, timing, PPA, system speedup, or DATE headline follows.
