# M1168R2 compile-repair source author receipt

Verdict: **GO for a fresh different-author M1172 source hammer only**.  Direct
VCS execution is forbidden; a separately sealed M1173 release is required after
that hammer passes.

The consumed r1 attempt failed before elaboration with five `DTINPCIL` and five
`IRFPCA-AUTOVAR` diagnostics.  All ten diagnostics trace to five automatic task
formals used on procedural `force` RHS expressions.  The failure quarantine is
recursively sealed with outer-seal file SHA
`72ec416eb80888bb5c30a448c870b0859912097d43564662a3a88953182316c7`;
the r1 attempt is not reusable.

R2 introduces five static module-scope staging variables, assigns every field
before force, and retains ten hierarchical forces against DUT-internal request
state.  A dedicated source check rejects five mutations that restore the
illegal automatic-formal RHS mode.  The package still contains 16 assertions,
six covers, 18 directed protocol cases, 24 deterministic random transactions,
seven DUT attacks, two service-assumption attacks, three reset states, and the
frozen M935 normal row/task case.

The r2 attempt/result/work namespace is new and absent.  No runner, VCS, simv,
license query, or EDA tool was invoked.  Consequently this source receipt does
not claim that compilation, elaboration, simulation, timing, cycles, speedup,
PPA, power, or energy passed.

`docs/359_DATE终局冻结_20260813.md` remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
