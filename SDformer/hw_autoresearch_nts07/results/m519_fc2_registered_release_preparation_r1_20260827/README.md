# M519 registered-release recovery preparation

Status: **PREPARED ONLY — VCS/DC NOT RUN AND NOT AUTHORIZED**

M519 is the single recovery identity permitted by the sealed M496-r3 failure
hammer.  It removes only the M219 response-edge release-to-request reuse:
an accepted response updates the slot and context registers at the active edge,
and that resource is eligible for a new request beginning on the next cycle.

Prepared material:

- six M519 RTL files covering K1, K8, K1x8, standalone and matched wrappers;
- service and K1x8 SVA, including explicit no-same-edge-release/reissue checks;
- two exact-identity VCS tests measuring K1↔K1x8 and K8↔K1x8;
- three filelists and an exact-SHA VCS runner;
- a 3.000 ns TSMC-28 slow/fast, ideal-clock, ZeroWireload, flattened DC Tcl and
  exact-SHA one-attempt runner;
- a precompile hard gate that terminates on any TIM-209 or OPT-150 before
  `ungroup` or any of the three compile commands;
- a blocked launch-admission contract.  It cannot authorize DC until sealed
  VCS and an independent P0=0 static review are bound by canonical SHA256.

Static preparation checks passed: JSON and shell syntax, identity hashes,
filelist closure, removal of both same-edge bypasses, required SVA/test intent,
precompile-gate ordering, exact three-command compile recipe, absence of
canonical M519 VCS/DC output directories, and unchanged M219/M496-r3/docs-359
identities.

This package contains no simulation, synthesis, timing, area, power, cycle,
speedup, energy or paper-admission result.  The next legal action is an
independent static hammer at
`reviews/m519_registered_release_static_hammer_r1_20260827`; no tool may run
from this preparation package.

