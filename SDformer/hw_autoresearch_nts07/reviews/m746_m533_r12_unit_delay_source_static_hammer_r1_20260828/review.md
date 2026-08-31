# M746/M533 r12 source static hammer

Verdict: **PASS, 100/100, P0/P1/P2 = 0/0/0**.

This was a fresh read-only audit. The runner, source contract, TB r7, frozen RTL/SVA/macro identities, foundry assets, prior failure chain and claim boundary all match their sealed identities. `bash -n` passed. The sole VCS command uses exactly `+define+UNIT_DELAY`; neither `+notimingcheck` nor `+no_notifier` appears. R7 PASS/COVERAGE gates require both exact RAW recovery paths and reject timing/SVA/error/fatal signatures.

No runner, VCS, simv, HDL compiler, CPU/GPU experiment, remote job or EDA tool was executed. The r12 result path remains absent. This admits only the source identity for candidate review; it does not authorize launch or establish functional, timing, RTL, cycle, PPA, energy, speedup or paper claims.
