# M86 independent hammer review

Verdict: **74/100, scoped GO for registered-bank/FIFO functional closure; NO-GO for any speedup readmission.** P0=0, P1=4, P2=3.

The frozen evidence is reproducible. Exact source/contract/input identities match, the sealed `simv` independently reran with rc=0, and a source-recompiled independent VCS bench passed. The binary oracle reconstructs all 1,728 phases, 221,184 descriptors, 835,383 issue beats, 725,103 cross-row fetches, maximum bank row 459, and the unique escape at phase 1242/pattern 5/block 5.

## What M86 really closes

The RTL implements a real **one-cycle registered bank-array read behavior**: issue rows are registered on one edge and the clocked block dereferences the eight arrays into a FIFO entry on the following edge. The independent bench checked response equals the prior-cycle issue, held a full FIFO for six cycles, observed 28 simultaneous push/pop cycles, drained 12 ordered bit-exact outputs, and rejected missing-row, OOB-row, and duplicate-row cases.

This is not yet evidence for a compiled SRAM macro. `bank_mem` remains behavioral RTL with no macro binding, inference report, DC/STA, or power result. The correct citation is “one-cycle registered eight-bank array interface,” not “paper-ready SRAM.”

## Highest-priority findings

1. **P1: silent simultaneous-valid deadlock.** `descriptor_ready` depends on `!payload_load_valid`, while `payload_load_ready` depends on `!descriptor_valid`. The independent attack held both valids for four cycles and got neither ready, neither accept, no busy, and no protocol error. Add explicit priority/phase state or fail closed and document the producer contract.

2. **P1: 203,200 is a narrow II counter, not total cycles.** It is exactly `1600 * 127` adjacent descriptor intervals inside always-ready phases. It excludes 16,256 stress-phase intervals, first/cross-phase starts, and—most importantly—the mandatory 794,880 row writes plus 1,728 phase commits. The interface-level serialized lower bound is 1,631,991 cycles before fallback or downstream work; loader+commit traffic is 95.36% of the read-issue count and loader rows are 48.71% of that lower bound.

3. **P1: no physical SRAM/PPA.** VCS proves registered functionality, not macro timing, area, power, or frequency.

4. **P1: escape is still a zero control token.** No exact bit-sparse fallback, accumulator, ordered heldout-use consumer, or end-to-end module cycle result exists.

The fixed 460-row completion rule also forces 51,149 zero-row accepts over the actual phase terminals, a 6.88% loader overhead. Terminal-aware loading and ping-pong banks are the direct hardware-performance opportunities.

## Admission boundary

- GO: exact-input functional closure of the registered eight-bank read, four-entry response FIFO, M85 final mask, and M82 signed reconstruction.
- NO-GO: compiled-SRAM wording, M78 1.409x re-admission, RTL/module/system speedup, FPS, energy, PPA, accuracy, DATE, or best-paper claims.

Next gate: fix channel arbitration; implement terminal-aware ping-pong refill or explicitly schedule all refill cycles; connect real escape/downstream execution; then run same-resource SRAM/DC/STA/SAIF/PTPX.
