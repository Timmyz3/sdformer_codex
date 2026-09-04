# NSC Line — SC Attention Improvement Ideas

Updated: 2026-06-03

## TX vs SC Architecture Analysis

```
TX (ternary alpha-XNOR):
  S[i,j] = Σ_d α-XNOR(Q_sign[i], K_sign[j])  ← N×N matrix
  
SC (signed consensus):
  S_token[j] = Σ_d (Q_sign[i] × K_sign[j])  ← token-level, O(N)
```

SC is simpler hardware (pure shift, no LUT) but lacks N×N interaction.

## Paper → Experiment Mapping

| Paper | Venue | Key Mechanism | Maps To |
|-------|-------|--------------|---------|
| BSA | NeurIPS 2025 | Ternary Q/K, Shiftmax | SC baseline |
| A²OS²A | CVPR 2025 | Binary Q + ReLU K + ternary V | NSC-10b |
| SITRA | 2025 | Temporal sparsity acceleration | HW co-design |

## Experiment Execution Priority

| Priority | Experiment | Method | Rationale |
|----------|-----------|--------|-----------|
| P0 | Wait for NSC-09d | 30ep full | ATLIF alignment test |
| P1 | NSC-10c λ sweep | 360-step | λ=[0.3,0.5,0.8,1.2] |
| P1 | NSC-10b ReLU K | 30ep full | Biggest potential gain |
| P2 | NSC-10d S02 FFN | 30ep full | Match TX layout |
| P3 | NSC-11 score_scale | short | Fine-tuning |
