# Remaining-op copies — honest Scrooge tax

Entry: `run_remaining.py` + `remaining_ops.py`. Receipt: `results/remaining_copies.json`.

FC1 bar: nnz(S)×Cout word-adds. PSN bar: n_tokens×Cout×T×T mix MACs.
Scrooge tot=Σ|A||Y| is leftover work → inspect_tax=1 → KILL.
Legal Scrooge bound: rest=‖A_rest‖₁·max|Y| (no MAC on skipped Y); inspect_tax=1/T.

Two launches, KEEP tuples identical:

- BitWave vs word **KILL −1.6051**
- Scrooge tot-oracle **KILL −0.0697**
- Scrooge l1-maxabs **KEEP 0.6042** (extra 0.7042 − tax 0.10)
- LUT-GEMM G4 packed **KEEP 0.2923**
- Cin k=79 **KEEP 0.1568** AEE 0.711 ≪ NB0
- Cout drop15 **KEEP 0.151**

Stack chain extra **0.408** (max per op, weighted FC1+PSN, no product).

Tests: `test_remaining_stats.py` drives shipped `psn_scrooge_extra(..., bound=)` and `psn_scrooge_inspect_tax` (tot_oracle tax is 1, not 1/T).
