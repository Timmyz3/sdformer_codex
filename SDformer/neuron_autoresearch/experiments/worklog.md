# Autoresearch Worklog: Neuron Operators

**Session**: 2026-05-07
**Branch**: `autoresearch/neuron-ops-20260507`
**Goal**: Design hardware-friendly sparse neuron operators. Minimize SOPs while keeping AEE within 10% of PSN baseline.

---

## Data Summary

| Metric | Baseline (PSN) | Best Known (G1) | Target |
|--------|---------------|-----------------|--------|
| SOPs | 3.6219G | 2.7134G (-25.1%) | <2.5G (-30%) |
| AEE | 1.5848 | 1.6056 (+1.3%) | <1.75 (+10%) |
| Firing rate | 0.08496 | 0.06365 | <0.07 |
| Gates | 0 | 6 (layer0 only) | flexible |

---

### Run 1: PSN baseline profile — sops=3.6219 (KEEP)
- Timestamp: 2026-05-08 00:36
- What changed: Initial baseline measurement using epoch59 checkpoint
- Result: SOPs=3.6219G, AEE=1.5848, firing=0.08496
- Insight: This is the starting point. 105 profiled neuron layers across 4 encoder stages.
- Next: Observe H1 training results, then design A1 (FSN on G1 nodes)

## H1 Training Status (2026-05-08 00:36)
- Epoch 5/20 complete
- 36 gates, 14 open (mean_prob=0.388)
- ~9 min/epoch, ~135 min remaining
- gates converging slowly (14→14 over 4 epochs)

## Key Insights

1. Blanket neuron replacement is a dead end — partial/selective gating is the winning strategy
2. G1's 6 layer0 nodes account for disproportionate sparsity gains — they're the highest-impact targets
3. The hardware story needs both: (a) sparse computation AND (b) efficient non-sparse ops
4. FSN (multi-level + ternary) is untested beyond smoke — it's the most promising unexplored direction
5. Hardware co-design means each neuron change must have a clear accelerator mapping

## Next Ideas

- A1 (FSN on G1 nodes): most promising — combines proven G1 targets with richer spike encoding
- A3 (shared gates): safest — simple hardware, proven approach
- A5 (refractory): simplest — guaranteed sparsity improvement, minimal risk
