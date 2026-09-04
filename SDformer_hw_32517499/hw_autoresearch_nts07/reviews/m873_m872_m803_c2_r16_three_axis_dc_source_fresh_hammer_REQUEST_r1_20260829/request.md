# M873 request: fresh M872/M803 three-axis DC source hammer

Review the sealed M872 source identity from scratch.  Do not trust the author
validation and do not run DC, VCS, simv, lmutil, PT, PTPX, Formality, a remote
job, or any production runner path.

The central questions are whether the full no-EDA path truly executes every
pre-attempt JSON/jq/exact-SHA gate; whether K1/M803-K8/K1x8 are the exact
`ARCH_MODE=0/1/2` bindings under one frozen filelist; whether all axes must pass
`TIM-209=0` and `OPT-150=0` before compile; and whether mapped V/SDC/DDC/SVF,
area, QoR, and setup timing are all fail-closed artifacts.

Use independent duplicate-key, missing-key, wrong-SHA, axis-binding, Tcl-order,
filelist, artifact, and publication mutations.  A pass must be 100/100 with
P0/P1/P2 = 0/0/0.  Even then it authorizes release authoring only, not DC.
