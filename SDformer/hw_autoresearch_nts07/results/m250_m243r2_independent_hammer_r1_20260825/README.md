# M250 independent hammer of M243r2

Score: **95/100**. Severity: **P0=0, P1=2, P2=3**.

Verdict: **GO. M243r2 closes M244's multi-context startup P0.** The directly
bound trace has ten samples; each has 45 unique T10 contexts and exactly
7,318,350 factor tiles. Context names and per-context tile counts are identical
across all samples. Independent arithmetic confirms:

`5*N + 5*S = 5*7,318,350 + 5*45 = 36,591,975 cycles`.

The corrected conditional module ratio is 1.999987702x. The fixed-compute-only
diagnostic is 1.062627046x and is not system speedup. M243r1's old 36,591,755
cycles and 1.999999727x are explicitly revoked. A wrong raw-trace SHA is
rejected before output creation, and a clean producer replay is byte-identical.

Integrated RTL, matched throughput per area, trained accuracy, energy, system
speedup, paper PPA and headline claims remain false. M37's 63,114.407654 um2 is
still standalone stage2-sidecar area, not complete candidate area.
