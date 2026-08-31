# M692 fresh-result hammer

Verdict: **GO for the M672 static adapter input only**, score **100/100**, P0/P1/P2 = **0/0/0**.

The canonical S10 payload is intact and complete: 40 records cover every `(sample 0..9, decoder D0..D3)` cell, all 40 bitpacks independently match their shape, byte count, SHA-256 and unpacked population, and the exact S00D0 sentinel is 839,586 ones plus 3,768,414 zeros. The canonical tree has 55 files, four directories and no symlink. Output, runtime, weight and attempt seals all pass, while the externally frozen manifest and outer-seal-file roots are exactly `c06de650...` and `e468b03a...`.

D1 is admitted only as `EXACT_SCALED_BINARY_BITPACK`: all ten `{0, runtime scalar theta}` gates pass. The folded-theta weight miter is nonexact in all ten records, so folded-weight deployment, the unmetered sidecar and decoder numerical equivalence remain explicitly unadmitted. This is not a payload defect; it is the correct fail-closed boundary.

Independent attacks changed a sealed member and were rejected. A private copy whose manifest and seals were consistently regenerated passed its internal seal, but changed both frozen external roots and was rejected by identity. No mapper, cycle simulator, performance model, GPU, RTL or EDA flow was run.
