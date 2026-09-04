# M872/M803 C2 R16 three-axis DC source handoff

This milestone authors a fresh, source-only Synopsys DC identity for the M803
channel-split successor.  It does not reuse the consumed R15 attempt or its raw
K1 child.  One future attempt must synthesize K1, M803 K8, and equal-bandwidth
K1x8 from zero with the same filelist, Tcl, SDC, library pair, and 3 ns clock.

The frozen axis bindings are `ARCH_MODE=0/1/2`.  Every axis must pass source
analysis, elaboration, `check_design`, and `check_timing` with `TIM-209=0` and
`OPT-150=0` before its single `compile_ultra`.  A partial axis set or a result
assembled across attempts is noncitable.

The runner retains the mature R15 exact environment, resource, collision,
license-status, PID identity, runtime-final, atomic attempt, quarantine, and
artifact publication controls.  The artifact tuple is strengthened to seven
items per axis: mapped Verilog, mapped SDC, DDC, SVF, area, QoR, and setup
timing.  Hold remains diagnostic only.

Author validation passed under Python 3.6: function and exact-file closure,
three duplicate-key JSON attacks, two semantic mutations, 25 artifact attacks,
and the entire candidate-to-contract pre-attempt path.  The full-path replay
exited before resource preflight, license query, attempt publication, or DC.
The wrong-runner-SHA negative returned 3 and emitted a double-sealed pre-attempt
failure receipt.  No canonical result or attempt exists.

This handoff authorizes only a fresh independent source hammer.  It does not
authorize DC, VCS, PT, PTPX, Formality, a license query, or remote execution.
It makes no area, timing, power, energy, PPA, throughput/mm2, or system claim.
