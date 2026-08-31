# M1273 — M1272/R13 final independent checker hammer

Date: 2026-08-30  
Mode: fresh different-author, read-only source hammer; no source mutation, VCS, simv, EDA, GPU or remote work  
Decision: **SOURCE NO-GO; no release and no VCS**  
Score: **69/100**  
P0/P1/P2: **0/5/1**

## Outcome

The frozen R13 TB and M1272 checker/tests have the requested exact identities:

- TB: `b749c7d635dc5b65669320aec7b7edb40cd5e2a5d781a9e474e3d28cbb054263`
- checker: `b0a32b4c77ddcdd526fd04bb66ae19642f1a0f2ea72b03a7c32a2361af107b62`
- tests: `7dcffc4d0238bc487435c46b1b54186fc193f8b108b2a24f4a9195697deab83b`

The clean checker passes, and its bundled regression rejects all 16 M1272
attacks (`17/17 PASS`).  Nevertheless, nine bounded, source-shape mutations
outside that regression are accepted by the checker.  They independently
disable the claimed runtime tokens, disable the real workload, override a
request object, or bypass the operand oracle.  Therefore the checker cannot
authorize exact-byte release authoring.

Per the task boundary this is the final checker-expansion hammer.  The result
is fail-closed: do not create an R13 release, do not run R13 VCS, and do not
patch this checker again in the current chain.

## Findings

### P1-1 — Runtime PHASE/PASS tokens remain guardable

Wrapping each of ENTER, COMPLETE, or PASS in the runtime-false guard
`if (oracle_count < 0)` is accepted.  The checker proves token spelling and
textual order, but not unconditional execution in the authoritative initial
flow.

### P1-2 — The real-M935 workload remains runtime-disableable

The checker accepts a runtime-false guard around `real_m935_completion()` and
also around both `serve_real_m935_beat` calls.  Literal call cardinality is not
a proof that the real task and both real beats execute.

### P1-3 — Request overrides remain expressible through legal LHS shapes

Two bounded override forms are accepted:

- `issue_request_valid[0] = 1'b0;`
- `force {dut.issue_request_first} = 1'b0;`

The first bypasses the assignment suffix regex with a bit select; the second
bypasses the force prefix regex with a concatenated lvalue.  Thus the claimed
`parent_issue_override=0 child_issue_override=0` boundary is not established
by M1272.

### P1-4 — Oracle condition can be clobbered

Adding `condition = 1'b1;` before the operand display is accepted.  This makes
the sole X-safe fatal unreachable for every failing caller while preserving
the checker-required display/flush/fatal text and order.

### P1-5 — Oracle can terminate before printing or failing

Adding `$finish;` before the operand display is accepted.  Simulation can
therefore terminate successfully before either diagnostic operands or the
fatal execute.

### P2-1 — Lexical admission is not a bounded control-flow proof

The checker remains a deliberately narrow lexical recognizer.  The failures
above are different syntax/control forms of already in-scope security
properties, not requests for a general SystemVerilog parser.

## Manual audit of the exact frozen TB

The actual byte-frozen TB itself remains credible independent of checker
weakness:

- row 0 uses mask `16'h0003`; rows 1–63 use zero masks;
- `real_m935_completion()` invokes one first beat and one non-first beat;
- no parent, child, or bare `issue_request_*` assignment, force, or release is
  present in the exact TB;
- only public prep and external weight/psum services drive the workload;
- the sole fatal is in `oracle`, after the operand display and flush;
- frozen M528/M935/M1162 and R3 SVA identities match the contract;
- `docs/359` remains `dedde7ce...7bdfc4`.

This manual observation is source evidence only.  It does not cure the failed
admission checker and does not authorize execution.

## Claim boundary

No functional VCS result, runtime SVA result, timing, cycle count, speedup,
PPA, energy, system speedup, or paper headline is established here.  Since
P1 is nonzero, the only allowed outcome is fail-closed.  Separate exact-byte
release authoring is **not authorized**.
