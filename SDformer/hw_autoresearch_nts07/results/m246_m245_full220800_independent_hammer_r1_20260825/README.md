# M246 independent hammer of M245 full220800 VCS

Score: **94/100**. Severity: **P0=0, P1=2, P2=3**.

Verdict: **GO for the scoped M245 exact-VCS milestone.** This closes the
M240 full-population RTL gap. All 220,800 M233 s10 coefficient pairs pass the
unchanged M235 production RTL with zero integer mismatch, and the six extrema
missing from the old 1,024-vector set are present. A separately coded RTL-order
numeric model also reports zero mismatch, and a fresh exact-source VCS rerun
reproduces all result, stall and fail-closed protocol covers.

The M245 value `max_unstalled_accept_ii=10` contains one test-driver negedge
bubble. It is an observed interval, not intrinsic II. M240's independent
standalone M235 latency/II remains 8/9.

This milestone is only the finalized-moments-to-coefficients engine. Moment
finalization, runtime affine/event equivalence, BN2 residual behavior, valid825,
mapped equivalence, full BN and system speedup remain false. M235 is now the
conditional primary over M236 at the coefficient boundary; downstream semantic
testing still decides whether its larger coefficient error reserve is safe.
