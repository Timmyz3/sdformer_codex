# A1: FSN Multi-Level on G1 Nodes

**Hypothesis**: Upgrading G1's 6 layer0 nodes from binary HardSparseGate to
FusedSparseNeuron (2-level signed = ternary-style) allows each spike to carry
polarity information, enabling more aggressive gating without accuracy loss.

## Key Config
- `use_fsn: true`, `fsn_num_levels: 2`, `fsn_signed: true`
- `stage_selection: layer0_only` (same 6 nodes as G1)
- `freeze_backbone: true` (gate-only training, BN in eval mode)

## Hardware Mapping
- 2 comparators + sign detection → 2-bit signed spike
- AND-popcount accumulator (multiplier-free)
- 6 neuron units gated → ~1% of total neurons, negligible area overhead

## Baseline Comparison
| | PSN | G1 (expected) | A1 target |
|---|---|---|---|
| SOPs | 3.6219G | 2.7134G | <2.5G |
| AEE | 1.5848 | 1.6056 | <1.75 |
| Firing | 0.0850 | 0.0637 | <0.06 |
