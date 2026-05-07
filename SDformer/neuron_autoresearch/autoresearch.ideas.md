# Autoresearch Ideas Backlog

## Phase 1: Neuron Operator Innovations

### A1 — FSN Multi-Level on G1 Nodes
- **Hypothesis**: G1's 6 layer0 nodes are the highest-impact gating targets. Upgrading from binary HardSparseGate to FusedSparseNeuron (ternary signed spike) allows each spike to carry polarity information (-1, 0, +1), which matches optical flow's positive/negative event encoding. This should allow closing more gates while preserving accuracy.
- **Config**: `use_fsn: true`, `fsn_num_levels: 2`, `fsn_signed: true`, `stage_selection: layer0_only`
- **Hardware mapping**: 2 comparators + sign bit → 2-bit spike bus → signed accumulation
- **Risk**: medium — ternary spikes might confuse BN statistics
- **Expected SOPs**: < 2.5G (target: 30% reduction from baseline)

### A2 — Leakage-as-Gate
- **Hypothesis**: PSN neurons with high decay (τ→0, forgetful) naturally produce fewer informative spikes. Instead of learning a separate gate_logit, derive the gate signal from the neuron's decay parameter. This couples temporal dynamics to sparsity — elegant and zero extra parameters.
- **Mechanism**: gate_prob = sigmoid((τ - τ_min) / (τ_max - τ_min) * scale)
- **Hardware mapping**: decay register → threshold comparator → clock-gate (reuses existing decay parameter, no extra storage)
- **Risk**: medium — decay and sparsity may have conflicting optima
- **Expected SOPs**: 2.8-3.2G

### A3 — Hierarchical Shared Gates
- **Hypothesis**: Neurons in the same stage have correlated spike patterns. Sharing one gate per stage (4 gates total instead of 36) reduces parameter count and hardware control complexity with minimal accuracy loss.
- **Config**: `stage_selection: all_stages_proj`, but with `gate_sharing: stage` (new option)
- **Hardware mapping**: 4 global clock-gate signals → distributed to all neurons in each stage
- **Risk**: low — simpler hardware, but may lose per-node fine-tuning
- **Expected SOPs**: 2.7-3.0G

### A4 — Spike-Timing-Dependent Gate
- **Hypothesis**: In event-based data, early timesteps are noise-dominated (sparse events), late timesteps carry coherent motion. Gating should be time-aware: open gates in early bins should be penalized more heavily than late bins.
- **Mechanism**: gate_penalty[t] = base_penalty * (1 + α * (T - t) / T) — higher penalty for early-timestep spikes
- **Hardware mapping**: bin counter + LUT per neuron group — cheap
- **Risk**: medium — implementation complexity
- **Expected SOPs**: 2.5-3.0G

### A5 — Refractory-Period Pruning
- **Hypothesis**: After emitting a spike, a neuron carries little new information for several timesteps. Enforcing a hardware refractory period (2-3 timesteps) after each spike reduces temporal firing density without spatial degradation.
- **Mechanism**: After spike at time t, block spikes at t+1, t+2 (mask = 0 for refractory window)
- **Hardware mapping**: saturating counter per neuron, 2-3 flip-flops → zero additional ALU
- **Risk**: low — simple, well-understood biologically
- **Expected SOPs**: 2.9-3.3G (modest but guaranteed improvement)

## Phase 2: Voxelization + Neuron Co-Design

### B1 — Sparse Voxel Frontend
- Replace dense voxel grid with sparse voxel representation (only store non-empty voxels)
- Neuron operates on sparse voxel indices instead of dense grid

### B2 — Event-Rate-Adaptive Bin Count
- Dynamically reduce timesteps for low-event-rate regions
- Fewer timesteps → fewer neuron evaluations → lower energy

## Phase 3: Attention Sparsity

### C1 — Window-Aware Gate
- Attention windows with zero events → gate off all neurons in that window
- Pre-compute window sparsity mask before neuron evaluation

### C2 — Sparse Attention Score Threshold
- Don't compute attention for token pairs with score below threshold
- Requires cheap score estimator

## Phase 4: Pruning + Hardware Co-Design

### D1 — Structured Neuron Pruning
- Remove entire neuron columns that never fire (or fire below threshold)
- Structured pruning → regular hardware array → no irregular sparsity overhead

### D2 — Precision-Scaling by Layer Importance
- Critical layers (layer0, bottleneck): 4-bit spike
- Non-critical layers (deep stages): 1-bit spike
- Hardware: mixed-precision datapath with configurable comparator banks

## Hardware Accelerator Design Principles

Every neuron innovation should answer:
1. **Area cost**: How many additional transistors/gates per neuron?
2. **Energy model**: What's the switching activity reduction?
3. **Control complexity**: How many global signals needed?
4. **Regularity**: Does it preserve regular array structure (systolic-friendly)?
5. **Memory**: Additional SRAM/register overhead per neuron?

Target hardware metrics:
- Neuron area: < 500 µm² in 28nm (baseline PSN neuron ~300 µm²)
- Spike energy: < 0.5 pJ/spike (baseline ~1.0 pJ/spike)
- Control overhead: < 5% of total area
