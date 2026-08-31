# M518 r6 VCS TB-instrumentation compile-failure independent hammer

Date: 2026-08-27  
Verdict: `DIAGNOSTIC_CONFIRMED__R6_TB_HIERARCHICAL_DEPOSIT_VS_ALWAYS_FF_ICPD__R7_STATIC_READMISSION_REQUIRED`  
Diagnostic score: **98/100**  
Findings: **P0=0, P1=1, P2=2**

This review is read-only over the immutable r6 failed result and frozen source.
No author/result file was modified; VCS was not rerun, and no DC, Formality,
PT/PTPX, or open-source EDA tool was executed.

## Verdict

The r6 one-shot was correctly authorized and failed closed at VCS compilation.
The failure is a **testbench-instrumentation P1**, not an RTL P0: V06 performs
ten hierarchical `$deposit` calls on ten variables owned by the DUT's single
`always_ff state` process. VCS rejects each as an illegal combination of
procedural drivers.

The exact r6 result is diagnostic only. It admits no compilation, simulation,
V01--V20 behavior, numeric equivalence, cycles, DC, PPA, energy, speedup,
system result, or headline. r6 must not be rerun.

## Result integrity

- The sealed r6 static authorization verifies. Authorized and observed runner
  SHA are both
  `050db5ce70013ba0b61093ce2abbb544b645542af55e48061a1d9bc3e60c2a4d`,
  and the result is the authorized default canonical directory.
- VCS identity is `V-2023.12-SP1_Full64`. The runner exits 20;
  `compile.rc=255`; the failure marker is present.
- The automatic wrong-SVA control returns exactly 10. Its 35-row preflight has
  the intended single all-zero SVA mismatch, and its member manifest and outer
  seal both pass.
- The positive preflight has 35/35 exact SHA matches. All four prior member
  manifests and outer seals pass. The saved input snapshot still matches all
  35 current frozen members.
- VCS parsed RTL, repaired SVA, and TB, then stopped in driver legality. No
  `simv`, simulation log/RC, assertion report, positive receipt, RUN_COMPLETE,
  or positive publication seal exists. Partial `simv.daidir` compiler scratch
  is not a simulator and is not positive evidence.

The immutable result contains 24 regular files, 40,272 bytes, and no symlinks.
The evidence inventory in this review binds every member by SHA256.

## All reported errors and root cause

VCS reports ten `Error-[ICPD] Illegal combination of procedural drivers`
diagnostics, one for each of:

```text
raw_bank0_q  raw_bank1_q  raw_tag0_q  raw_tag1_q
raw_order0_q raw_order1_q raw_owned_q raw_ready_q
raw_beats_q  tiles_loaded_q
```

The variables are declared at RTL lines 97--100 and 138 and assigned in the
single DUT `always_ff` block beginning at line 436. The only additional source
writes are the ten `$deposit` calls at TB lines 610--619, reached from the
`V01_to_V20_campaign` initial process at line 1012. No hierarchical `force`,
`release`, direct assignment, or other DUT-state write exists in the frozen TB.

The compiler reaches its default maximum-error count at ten, so the log alone
proves “at least ten.” Source enumeration independently finds exactly ten such
writes, all with the same cause. No internal second RTL state owner, arithmetic
defect, protocol defect, or SVA failure is evidenced.

The V06 comment itself explains why the test used injection: the eager
one-fill/one-dense transport cannot naturally retain the two-ready-bank state
without a test harness. This is an unreachable-state instrumentation problem,
not evidence that production transport is incorrect.

## Minimal r7 repair

The preferred repair is a simulation-only **legal-fill harness**, not a wide
state deposit:

1. Under a dedicated `M518_VCS_V06_HARNESS` define, add two test controls: hold
   dense issue and select bank1 for the first empty-bank raw fill.
2. V06 holds dense issue, sends the bank1 payload/tag first and bank0 payload/tag
   second through the normal five-beat raw interface, and requires
   `raw_ready_q==2'b11` plus `raw_order1_q<raw_order0_q`.
3. Release the hold and retain the current pre-edge bank1 selection, post-edge
   ownership/tag transition, conservation, numeric scoreboard, and release
   oracle. Count exactly one harness activation.
4. With the macro absent, production behavior and the ordered 50-port interface
   must remain the r6 design. DC, Formality, and PT must never define this macro.

This approach lets the existing DUT `always_ff` process create every state
transition and tests more real transport than the snapshot deposit.

If that legal-fill approach proves impractical, an acceptable fallback is one
simulation-only packed debug-injection request whose assignments are performed
inside the existing DUT `always_ff` block. It is larger and less realistic but
still preserves single ownership.

The following are not acceptable repairs:

- changing `always_ff` to `always` merely to permit multiple writers;
- replacing `$deposit` with hierarchical assignment, `force/release`, or a
  writing bind process;
- deleting V06, dropping the bank1-oldest sequential check, or weakening any
  V01--V20/numeric/protocol oracle;
- rerunning r6 with a larger error limit.

## r7 gate

r7 needs fresh source, contract, exact-SHA runner, absent canonical result path,
author handoff, and independent static review. The static gate must bind this
sealed failure review and the complete prior chain; reject every hierarchical
DUT-state LHS write; prove single state ownership; prove the test macro is
absent from production flows; preserve 51 assertions, the exact 25 covers, all
V01--V20 closure fields, and all claim-boundary `false` fields.

Only r7 source authoring is authorized now. r7 VCS, DC, Formality, PT/PTPX, and
all performance/system claims remain unauthorized until that fresh static gate.

