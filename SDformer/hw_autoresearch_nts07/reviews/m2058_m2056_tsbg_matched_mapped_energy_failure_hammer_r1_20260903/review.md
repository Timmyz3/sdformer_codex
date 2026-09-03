# M2058 / M2056 TSBG matched mapped-energy independent failure hammer

## Verdict

**M2058 is conclusively failed and consumed: no retry.** Failure-hammer quality is 96/100; P0/P1/P2 = 0/3/1.

The only M2058 attempt performed one license preflight, one ordinary-LRU4 VCS compile and one ordinary-LRU4 mapped simulation. It produced no SAIF, ran no PTPX, and never compiled or simulated the TSBG axis. No power or energy number exists.

The attempt token is correctly double-sealed. The hidden failure stage is not: the runner tried to seal the full VCS build tree, encountered two tool-generated symlinks, swallowed the nested failure-publication exception, and left the stage unpublished. This review does not retroactively modify it; it pins all key evidence plus a deterministic fingerprint of the complete 110-file/two-symlink tree.

## Exact failure timeline

1. Reset and all four contexts × 48 groups complete normally.
2. The last descriptor completes at `cycle=383`, simulation time 1,164,000 ps.
3. `full_load_complete cycle=383` and `full_execute_begin cycle=383` are printed.
4. Ten picoseconds later, before the first UCLI `$stop` and before power enable, the wrapper fails at line 75 with `M2056 ordinary mapped bridge/commit/control X/Z`.

The preceding load/reset and memory-handshake `$isunknown` groups passed. The first failing concatenation contains 21 logical fields:

| Class | Signals | Static origin |
|---|---|---|
| Bridge control | valid, ready, accept | frontend state/fault plus TB ready |
| Bridge sideband | context, group, half, slice, bank-valid | resettable `current_*_q` state; bank-valid is packed direct |
| Commit control | valid, ready, accept | frontend state/fault plus TB ready |
| Commit sideband | context, tag, slice, terminal | resettable commit/context state |
| Bundle control | done-valid, done-ready | frontend state/fault plus TB constant |
| Fault/status | protocol-error, stale-response, overflow, busy | frontend plus M803 adapter state |

The fatal is class-level: it does not print individual values and there is no waveform, checkpoint or SAIF. It is therefore impossible to prove which member or bit was X/Z. Counter, bank metadata, payload and accumulator checks were never reached and must not be called passing.

## Root-cause judgment

### Not a demonstrated flatten-order failure

The failure happens before the unpacked bank metadata/payload groups. Scalar controls are named direct connections; `bridge_bank_valid[7:0]` is also a native packed direct port. The high-segment element-zero reversal applies to later bank/lane arrays, so this fatal does not implicate that mapping.

### Most likely: invalid-sideband / mapped initial-state observability

The exact RTL source passes slot 42 under M2057 with 149 rows, 1,278 issues and 29,472 products. M2058 fails only on the mapped ordinary axis, at the boundary before any request/product/commit. Several fields in the failed group are semantically irrelevant while `bridge_valid=0` or `commit_valid=0`, yet M2056 requires them to be known unconditionally. This makes mapped invalid-sideband or initial-state observability the strongest supported class. The M803 adapter cannot be completely excluded because protocol-error, stale-response and busy are in the same coalesced group.

This is not evidence of arithmetic failure. It also does not prove the mapped netlist equivalent—the mapped dynamic gate remained open and has failed.

### `UNIT_DELAY` did not create a one-unit timing transient

Although the command defines `UNIT_DELAY`, the pinned TSMC cell Verilog never consumes that macro, its specify arcs are zero, and no SDF is annotated. Moreover, the endpoint is already sampled at the load-completion negedge plus 10 ps, 1.51 ns after the preceding posedge. This artifact is a zero-delay mapped functional simulation. “Move the monitor to negedge” alone is not an adequate repair.

## Successor decision

One **newly named and independently reviewed source direction** is allowed; execution is not yet authorized. It must be a genuinely different measurement protocol, not an M2058 retry:

- sample continuously at a settled phase, preferably negedge plus a delta margin;
- keep clock/reset, valid/ready/accept, fault and busy controls unconditionally known;
- gate request metadata by request-valid, response metadata/payload by response-valid, bridge sidebands by bridge-valid/bank-valid, and commit sidebands/accumulators by commit-valid;
- print per-signal diagnostics rather than one 21-field fatal;
- preserve slot42, both mapped netlists, the 383-cycle preload, the 20,292/7,569 execute denominators, UCLI scopes and final M2051 PASS;
- retain the `TX=0`, annotation and subtotal gates—valid gating may not excuse X transitions appearing in measured SAIF;
- use a new attempt latch with `automatic_retry=false` and a symlink-aware evidentiary failure seal;
- receive a source hammer before one execution and a result hammer before any power/energy claim.

M2058 itself remains failed forever. Existing `simv` reuse, namespace reuse, “continuation,” or automatic retry is prohibited.

No EDA, GPU job, license query or source/result/predecessor/`docs359` mutation was performed.
