# M240 independent BN coefficient-engine Pareto hammer

Score: **88/100**. Severity: **P0=0, P1=6, P2=4**.

The sealed scoped evidence passes. M235r2 checks 1,024 RTL vectors; M236 checks
all 220,800 M233 s10 coefficient pairs and the six tail extrema missing from
M235's set, with zero integer mismatch. Backpressure and a fail-closed illegal
request against a pending result pass for both. Matched 3 ns TSMC28 logic-only
DC maps one multiplication datapath per candidate and reports 3,148.362002 um2
for M235 versus 3,376.926005 um2 for M236.

An independent no-stall VCS bench corrects the performance interpretation:
M235 latency/II is 8/9 and M236 is 11/12. The production TB's M235 interval 15
contains a five-cycle output stall plus a driver bubble; M236's 13 contains one
driver bubble. At the standalone coefficient boundary M235 therefore has
1.333x M236 throughput and 1.430x area-throughput efficiency.

Select M235 64-entry/one-Newton as the conditional primary. M236's 16-entry LUT
is 75% smaller and its captured coefficient-only bound is 9.009% lower, but the
complete design is 7.260% larger and 25% slower. Keep M236 only as the error
fallback. Do not build a third coefficient engine now: close full220800 M235
VCS, then let ATLIF/BN2/valid825 evidence decide whether M235's 0.001728 bound is
acceptable. Matched SAIF/PTPX is needed only if energy becomes a selection or
paper claim.

Neither point is full dynamic BN. Moment finalization, moment SRAM/barriers,
runtime affine quantization, ATLIF events, BN2 residual behavior, valid825,
system speedup and paper-ready physical PPA remain unadmitted.
