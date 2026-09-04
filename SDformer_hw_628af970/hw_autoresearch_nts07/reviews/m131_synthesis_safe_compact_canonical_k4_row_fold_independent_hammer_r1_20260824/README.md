# M131 synthesis-safe compact canonical K4 row-fold independent hammer

## Verdict

**92/100, conditional pass. P0=0, P1=1, P2=4.** M131 fixes the M130 Synopsys negative predecessor-index elaboration failure and preserves the accepted-descriptor arithmetic/protocol behavior under commercial VCS. The evidence supports a synthesis-safe, descriptor-stream-local module result. It does **not** support complete row losslessness, a free/implemented descriptor producer, a frequency ratio, macro-inclusive PPA, physical/system speedup, or a paper headline.

`docs/359_DATE终局冻结_20260813.md` remained at SHA-256 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

## Independent evidence

| Check | Result |
|---|---:|
| Frozen-source production VCS rebuild | PASS, compile/sim RC 0 |
| Production descriptors / lane checks | 237 / 22,752 |
| Independent descriptors / lane checks | 110 / 10,464 |
| Independent accepted K1/K2/K3/K4 | 1 / 7 / 3 / 99 |
| Within-descriptor duplicate/descending attacks | rejected 1 / 1 |
| Cross-descriptor repeat/backtrack attacks | rejected 1 / 1 |
| Row drift / dirty source padding / dirty negate padding | rejected 1 / 1 / 1 |
| Nonlast source15 / cache miss / resident-block mismatch | rejected 1 / 1 / 1 |
| Maximum output stall | 73 cycles |
| Same-cycle retire/replace after long stall | 1 |
| Cross-row descriptor/update adjacent II=1 intervals | 95 / 95 |
| Tagged-done overlap with next row and correct prior tag | 100 / 100 |
| Idle `group_valid=0` payload perturbation checks | 17 (16 closed row + 1 open row) |
| Reset attack/check | 1, in-flight descriptor aborted and state isolated |
| Independent DC analyze/elaborate/link/check_design | PASS |
| `ELAB-312`, `group_source[-1]`, negative index | 0 |

The production rebuild used the frozen M131 RTL, SVA, production testbench and file list at their contract SHAs. It reproduced the exact production PASS metrics and all seven expected SVA covers. The independent testbench separately recomputed every accepted signed lane result and attacked strict K1-K4 ordering, identity, padding, cache residency, source-15 framing, backpressure, reset, and tagged completion.

Static and dynamic ready/valid review found no internal combinational ready/valid cycle: with `group_valid=0`, `group_ready` is capacity-only, and the descriptor/request audits do not consume `group_ready` or `group_accept`. This does not waive the external producer rule: its `valid` and payload must not combinationally depend on `ready`, or a registered/skid boundary is required.

## Synthesis repair and exploratory DC

M130 has exactly one `group_source[pick-1]` expression. M131 has none and replaces it with fixed guarded comparisons for `[1:0]`, `[2:1]`, and `[3:2]`. Independent Synopsys DC V-2023.12-SP3 analyze/elaborate/link/uniquify/check_design returned true without `ELAB-312` or an out-of-bounds index. This was an elaboration audit, not an independent compile.

The existing M131 and M128 runs used the same DC version, generic compile Tcl, 3.000 ns SDC, max/min libraries, ideal clock, ZeroWireload, and zero macros:

| Fixed 3 ns exploratory metric | M128 | M131 | M131 - M128 |
|---|---:|---:|---:|
| Cell area (um2) | 89,045.585598 | 89,467.055598 | +421.470000 (+0.4733%) |
| Leaf cells | 107,287 | 109,277 | +1,990 |
| Sequential cells | 14,782 | 14,798 | +16 |
| Logic levels | 41 | 32 | -9 |
| Worst setup slack (ns) | +0.3387 | +0.6733 | +0.3346 |
| Worst hold slack (ns) | +0.0005 | +0.0001 | -0.0004 |

Both runs meet the fixed setup/hold constraint. M131 is slightly larger but has substantially more setup margin in this one synthesis point. The runs lack exact-SHA launch manifests, frequency sweeps, SRAM macros, clock-tree/routing, and power evidence; therefore this comparison is not an Fmax ratio or a physical-speedup result.

The independent precompile `check_design` is also not warning-free: 386 `LINT-1` undriven elaboration cells and 780 `LINT-31` direct/shorted sign-extension outputs (1,166 total). The exploratory postcompile report is warning-free after optimization. The safe claim is specifically the negative-index elaboration repair.

## Open findings

### P1 — complete row partition losslessness remains open

M131 checks strict increasing order and nonoverlap within an open row, but it has no expected-row source set or end-of-row coverage ledger. The independent test accepted three legal increasing descriptors that intentionally omitted source IDs. Thus monotonicity prevents duplicate/backtracking work but cannot prove that every required source is delivered exactly once. Implement and exact-SHA verify the descriptor producer plus a row-level exactly-once proof, or retain a trusted-input stream-local claim.

### P2 — claim-boundary issues

- `35` is payload bits only: block3 + row9 + count2 + source IDs16 + negate4 + last1. It excludes ready/valid, framing, queues, producer state, the derived mask, and transport.
- The complete 3 ns DC evidence is exploratory and lacks a sealed launch manifest or frequency sweep; M130 has no valid DC result. The M128 comparison above is same-context but not a frequency claim.
- The 89,467 um2 number is zero-macro logic-only area. The behavioral multi-read cache, ideal clock, and ZeroWireload are not a realizable SRAM/routed PPA point.
- Precompile `check_design` contains 1,166 warnings, so M131 is not a zero-warning RTL result.

## Reproduce and verify

The one-shot runner is `run_m131_independent_hammer.sh`; it intentionally refuses to overwrite the captured `sealed_vcs_replay`, `independent_vcs`, or `dc_elaboration` evidence. It used Synopsys VCS V-2023.12-SP1 Full64 and DC V-2023.12-SP3.

From the hardware root, verify the frozen external evidence and rerun the deterministic machine audit:

```bash
sha256sum -c reviews/m131_synthesis_safe_compact_canonical_k4_row_fold_independent_hammer_r1_20260824/input_manifest.sha256
reviews/m131_synthesis_safe_compact_canonical_k4_row_fold_independent_hammer_r1_20260824/audit_m131_independent.py
(cd reviews/m131_synthesis_safe_compact_canonical_k4_row_fold_independent_hammer_r1_20260824 && sha256sum -c manifest.sha256)
```

Generated VCS build trees (`csrc`, `simv.daidir`, `simv.vdb`) and DC work-library intermediates are reproducible scratch and are excluded from the durable manifest. The executable `simv`, raw logs, assertion reports, RCs, independent DDC, review sources, receipts, and machine audit are included.
