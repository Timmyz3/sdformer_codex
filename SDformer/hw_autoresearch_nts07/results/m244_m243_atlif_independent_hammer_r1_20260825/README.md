# M244 independent hammer review of M243 ATLIF performance

Score: **78/100**. Severity: **P0=1, P1=6, P2=4**.

Verdict: **NOGO for the current exact frozen-population admission; GO for the
corrected conditional direction.**

All declared hashes pass, a wrong-SHA attack is rejected, and M243's arithmetic
is correct if all 7,318,350 tiles form one uninterrupted context. The raw trace
shows otherwise: a frame contains 45 distinct T10 operator contexts. M31/M37
require drain/release at context changes, so the five-cycle fill is paid 45
times. The corrected equation is `5*N + 5*S` with `S=45`, producing 36,591,975
cycles and 1.999987702x, not 36,591,755 and 1.999999727x. The corrected
fixed-compute-only Amdahl diagnostic is 1.062627046x. It remains not system
speedup.

M37's 63,114.407654 um2 result is a standalone CSD reconstruction sidecar. It
omits the stage1/shared96 pool while including private input banks and FIFO, so
it is neither complete candidate area nor exact normalized incremental area.
M31 is VCS-only, M37 alone has DC/STA/Formality, and M38 remains an abstract
model. Matched throughput-per-area therefore remains NOGO.
