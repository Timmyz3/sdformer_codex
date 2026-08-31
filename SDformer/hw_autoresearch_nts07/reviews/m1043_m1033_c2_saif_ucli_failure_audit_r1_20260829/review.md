# M1043 M1033 UCLI-power failure audit

M1033 is consumed and must not be retried.  Its sealed license checkout
preflight passed, K1 mapped-gate compilation and linking succeeded, and a fresh
`simv` was created.  The first UCLI case then stopped before simulation with
`UCLI-117` because the production compile omitted exact `-debug_access+r`.
No case completed and no SAIF file exists.

The additive repair must exercise the actual UCLI power protocol before the
next production attempt: compile a frozen tiny DUT/TB with
`-debug_access+r`, run `power -enable`, `power -disable`, and `power -report`,
then validate that the tiny SAIF is nonempty and names the frozen DUT hierarchy.
Failure must seal an isolated preflight quarantine and leave M1046 unconsumed.

This audit authorizes source and release repair only.  It does not authorize
M1046, PT, PTPX, DC, GPU work, power, energy, or system claims.
