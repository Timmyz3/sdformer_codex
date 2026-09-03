# M2028 independent source hammer

Verdict: **PASS 98/100; P0/P1/P2 = 0/0/0.**

The exact M2029 runner is authorized once for one license query and two
logic-only DC executions: ordinary-LRU4 (`SCHEDULE_MODE=0`) and TSBG-B4
(`SCHEDULE_MODE=1`).  Both elaborate the same sealed M2018 G48/B4/LRU4 source,
two-row filelist, constraints, and synthesis flow.  Python 3.6 and 3.12 both
passed the independent static hammer; 14/14 in-memory mutations were rejected.

This source review does not admit a DC result.  Even a passing run measures
only matched logic area, setup, and public-port equality.  It cannot upgrade
the 2.533808x CPU premodel or the directed 75% request reduction into an RTL
cycle speedup, same-area result, system speedup, hold closure, power/energy,
paper-PPA, or headline claim.  M2029 still requires independent result review.

The reviewer ran no EDA and made no license query.  `docs/359` is unchanged.
