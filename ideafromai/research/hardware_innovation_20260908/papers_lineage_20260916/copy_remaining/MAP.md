# Remaining-op map (closed skip-family excluded)

Fair bar: extra ≥15% after tax vs nnz(S)×Cout **word-adds** or vs n_tokens×Cout×T×T PSN MACs.

| Paper | Onto leftover | This round |
|---|---|---|
| FireFly-S dual-side | S0.fc1 / r1 word-adds | KILL |
| BitWave vs word-add | 7bit vs 1-cycle word add | **KILL −160%** |
| BitWave vs 7-serial | informational | 60.1% **not the bar** |
| CGNet on FC1 | skipped/total | **KILL live BN** |
| Prosperity unique | remaining rows | 4.6% KILL |
| LUT-GEMM G=4 packed | leftover word-adds | **KEEP +29.2%** lossless |
| Scrooge tot-oracle Σ\|A\|\|Y\| | leftover mix | **KILL** (tot = leftover, tax=1 → −7%) |
| Scrooge ‖A_rest‖₁·max\|Y\| | leftover mix | **KEEP +60.4%** after 1/T tax |
| static Cin k=79 | nnz extra ≥15% | **KEEP 15.7%** AEE 0.711 |
| static Cout drop 15% | hidden prune | **KEEP 15.1%** AEE ~0.71 |
| Phi / K=256 / rank-4 | — | KILL |
