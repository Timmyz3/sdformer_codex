# M236 final delivery

M236 closes the 16-entry LUT plus two-Newton coefficient-engine candidate at
the same finalized-moment interface as M235. The full 220,800-pair
checkpoint-bound vector population passes exact-SHA Synopsys VCS with zero
integer mismatches. First-result latency is 11 cycles and the maximum
unstalled accept interval is 13 cycles.

TSMC28 HPC+ 3 ns logic-only DC is complete: 3,376.926005 um2, 4,345 cells,
319 sequential cells, setup slack +0.1069 ns and hold slack 0.0000 ns. Relative
to matched M235r2, M236 improves the captured coefficient-only maximum bound
by 9.009% and reduces LUT payload by 75%, but **increases area by 7.260%**.
M236 is therefore not the area winner. Choose M235 64+1 if later event/accuracy
admission accepts its error; retain M236 only as the lower-error option.

This milestone excludes moment finalization, runtime affine quantization,
ATLIF/event equivalence, BN2 residual equivalence, valid825, full-BN cycles,
energy and system speedup.
