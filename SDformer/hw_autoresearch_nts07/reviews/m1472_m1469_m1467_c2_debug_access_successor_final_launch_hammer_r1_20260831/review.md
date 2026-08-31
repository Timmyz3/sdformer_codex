# M1472 C2 final launch-authority hammer

Verdict: **PASS, 100/100, P0=0, P1=0**.

M1472 authorizes exactly one M1467 production campaign. It does not claim that the campaign has run or that any functional, SAIF, PTPX, power, energy, performance, system-speedup, PPA, or headline result exists.

The hammer replayed all 13 M1467 native source tests and independently rejected 161 mutations: 18 source/order/count attacks, 140 complete M1469 release-leaf attacks, and 3 sidecar corruption attacks. False negatives were zero.

The only admitted source delta from consumed M1432 is one `-debug_access+r` flag in the shared VCS compile prefix. Both `k8` and `k1x8` use it. The campaign remains cases 0--4 for both axes, with exactly 2 VCS compiles, 10 simv runs, 10 production SAIF files, and 10 PTPX runs. All ten mapped correctness/SAIF gates precede the first PTPX run. Partial-axis citation and automatic retry are forbidden.

The M1432 failure remains exactly `SIM_k8_0`, UCLI-117, with one compile, one simulation, zero SAIF, and zero PTPX. Its attempt remains consumed. M1472 did not read or enumerate the private unsealed build.

No license query or EDA tool was invoked by this hammer. `docs/359` remains at `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`; the user-modified `ucli.key` was not touched.
