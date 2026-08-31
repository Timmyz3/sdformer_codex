# M518 r7 VCS phase16 sampling failure independent hammer

Date: 2026-08-27  
Verdict: `DIAGNOSTIC_CONFIRMED__R7_TB_ACTIVE_REGION_COMBINATIONAL_SAMPLE__R8_TB_ONLY_READMISSION_REQUIRED`  
Score: **98/100**  
Findings: **P0=0, P1=1, P2=2**

This is a receipt-blind, read-only review of the sole r7 VCS result at
`results/m518_matched_fixed_t10_atlif_vcs_r7_exact_20260827`. I did not rerun
VCS or any other EDA tool, and I did not modify the frozen r7 inputs or
`docs/359`.

## Decision

The r7 result is diagnostic only and must not be cited. The exact VCS identity
is V-2023.12-SP1 Full64, the runner identity matches its static admission,
`compile.rc=0`, and `sim.rc=0`; nevertheless the runner correctly exits 23
because `sim.log` contains a TB `$fatal` at line 768 and publishes
`RUN_FAILED_OR_INCOMPLETE.txt`. There is no positive receipt, `RUN_COMPLETE`,
member manifest, or outer seal in the canonical result.

This is **not an RTL P0** and is **not V06-induced phase drift**. It is a
testbench sampling P1: at TB lines 765--768 the task wakes at `negedge`, assigns
`result_ready=0`, and immediately reads combinational `u_dut.fifo_credit`
without yielding a delta cycle. At that point the registered facts are already
deterministic (`dense_active_q=1`, selected cycle 16, FIFO occupancy 16), but
`fifo_credit` can still reflect the pre-assignment `result_ready=1`, hence the
fatal text "phase16 targeted stall did not align" despite the intended phase
having aligned.

## Why V06 is actually exercised and clean up to its closure

The r7 simulation reaches `targeted_phase12_phase16_stalls`, which is called
only after the blocking `oldest_selection_sequential_attack` returns. Returning
from that task requires all of the following to have passed:

- two legal five-beat fills, bank1 then bank0, with exact tags and order;
- the pre-edge and post-edge bank1-oldest selection/ownership checks;
- `finish_context(2,...)`, including result-tag/data/beat scoreboard,
  expected-read/write closure, zero `numeric_mismatches`, tile/issue/push/pop
  conservation, and slot-ledger closure.

The independent SVA report also records exactly one
`cp_dual_ready_oldest_bank1` match. The log contains no V06 fatal, V02 numeric
mismatch, scoreboard-incomplete fatal, or assertion-failure diagnostic. Thus
V06 legal-fill is a real dynamic hit; it did not merely compile. Full V01--V20
campaign admission still remains false because r7 stops in the following V08
task before later fault/reset tests.

V06 cannot shift this V08 oracle. `targeted_phase12_phase16_stalls` begins with
`reset_dut`, clears the observation state, selects manual ready mode, and uses
event/phase conditions rather than absolute simulation time. The r7 reverse
proof also shows that this V08 task is byte-identical to r6. The active-region
bug was latent in r6, whose simulation never started because compilation had
already failed.

## Minimal r8 repair

Change only the TB sampling point:

```systemverilog
@(negedge clk_core); result_ready=1'b0; #0.2;
if (!(u_dut.dense_active_q && u_dut.dense_selected_cycle==16
        && result_fifo_occupancy==16 && !u_dut.fifo_credit))
    $fatal(1,"V08 phase16 targeted stall did not align");
```

The `#0.2` is a post-combinational-settle observation delay, well before the
next posedge; the TB already uses the same convention elsewhere. It must not
alter the phase sequence, registered state, RTL, SVA, V06 legal-fill, numeric
oracle, or any cover requirement. Removing exactly this delay from r8 must
recover the frozen r7 TB SHA256
`a2de78ac5a3c537e03113f06552a09808426170d188d39e462b500b0c865eb12`.

Forbidden repairs remain: `$deposit`, `force/release`, hierarchical DUT-state
writes, writing binds, `always_ff` downgrade, deleting V06, weakening the
phase12/phase16 checks, or lowering any of the 25 nonzero-cover gates.

## r8 admission sequence

1. Freeze a new r8 TB/contract/runner identity; keep r7 RTL and SVA byte exact.
2. Bind this sealed r7 failure review and the immutable r7 failure artifacts;
   the r7 canonical directory remains diagnostic and is never rerun or sealed.
3. Have a different reviewer prove the one-line reverse SHA, preserved 51
   assertions/25 covers/V01--V20 campaign, no forbidden state writes, and a new
   absent canonical r8 path.
4. Permit exactly one VCS run only after static GO; require compile/sim rc 0,
   no fatal/assertion failures, exact PASS signature, all 25 covers nonzero,
   finite receipt, and result seals.
5. Require a separate receipt-blind hammer before any VCS admission. DC,
   Formality, PT/PTPX, cycles, PPA, energy, speedup, system speedup, and headline
   claims remain unauthorized.

## Findings

- **P1 — same-active-region oracle sample:** line 766 reads `fifo_credit`
  before `always_comb` can settle after line 765 changes `result_ready`.
- **P2 — zero simulator rc is not success:** VCS returns zero after the TB
  `$fatal/$finish`; the runner's log gate catches it and exits 23 as designed.
- **P2 — partial coverage is diagnostic only:** covers reached before the stop
  are useful localization evidence, but zero later fault covers reflect an
  incomplete campaign and admit no functional claim.

`docs/359` remains SHA256
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
