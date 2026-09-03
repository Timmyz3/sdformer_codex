# M2030 independent M2029 result hammer

Verdict: **PASS 99/100; P0/P1/P2 = 0/0/0.**

The unique consumed attempt and unique published result are complete and
double-sealed.  They record one Design-Compiler license query, two DC runs, one
compile per axis, and no retry.  Both axes have the same 4,551 public ports and
meet the 3 ns setup target.

| Axis | Area (um2) | Setup WNS (ns) | Hold diagnostic WNS (ns) |
|---|---:|---:|---:|
| ordinary-LRU4 | 249,710.451846 | +0.0264 | -0.0164 |
| TSBG-B4 | 249,739.809848 | +0.0688 | -0.0164 |

TSBG/ordinary logic area is `1.000117568175x`: only `29.358002 um2`, or
`0.0117568%`, additional standard-cell logic.  This is citable only as a
matched, pre-macro static-schedule ablation under ideal-clock/ZeroWireload DC.

The 2.533808x CPU premodel remains a CPU premodel, and the 75% request
reduction remains a directed VCS result.  Hold closure, power, energy,
same-area/exact-cycle/component/system speedup, production G48 dynamic
verification, and paper-PPA readiness remain false.

The reviewer launched no EDA, GPU, or license query and did not modify the
result, predecessors, or `docs/359`.
