# M881 request: fresh M880/M803 R16 terminology-repair source hammer

Review M880 from scratch without trusting the author validation.  The central
question is whether M873's sole P2 is actually closed: current attempt and
hammer instructions must name M872/M803 R16 or M880, while legitimate
historical `r15_*` M800 and artifact-gate provenance must remain intact.

Recompute every sealed identity, run strict duplicate and nonfinite JSON
checks under Python 3.6 and 3.10, inject the stale R15 labels as a negative,
and replay only the explicit no-EDA full-path and artifact self-tests.  The
frozen M803 RTL/SVA/TB/filelist/Tcl/SDC/libraries and production behavior may
not change.

A pass requires 100/100 and P0/P1/P2 = 0/0/0.  Even a pass authorizes only a
fresh launch-release authoring milestone; it does not authorize DC, VCS,
license queries, remote work, or any physical/performance claim.
