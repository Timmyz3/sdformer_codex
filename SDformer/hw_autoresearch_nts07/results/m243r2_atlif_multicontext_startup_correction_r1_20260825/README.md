# M243r2 ATLIF multi-context startup correction

This immutable overlay preserves M243 r1 and revokes only its exact
finite-population cycle values.  The direct raw ordered trace contains 45
distinct T10 ATLIF contexts per inference.  Because a context change requires
drain/release, the five-cycle phase-decoupling fill is charged 45 times:

`5*N + 5*S = 5*7,318,350 + 5*45 = 36,591,975 cycles`.

The corrected conditional ATLIF module ratio is `1.999987702x`; substituting
only this module into the frozen fixed-compute ledger gives `1.062627046x`.
The latter is an Amdahl diagnostic, not system speedup.

The result directly pins the ordered trace, trace manifest, independent M244
review and frozen M37 r10 RTL.  M37 remains a standalone stage2 sidecar;
integrated RTL throughput, matched throughput per area, trained accuracy,
energy, system speedup, paper PPA and headline claims remain unadmitted.
