# Autoresearch Ideas Backlog

Updated: 2026-05-08 with ICLR/NeurIPS/ICML 2024-2026 literature integration.

---

## Paper → Experiment Mapping

| Paper | Venue | Key Mechanism | Maps To |
|-------|-------|--------------|---------|
| Bipolar Self-Attention (BSA) | NeurIPS 2025 | Ternary spike neurons, bipolar attention, Shiftmax | A1, A6 |
| SEMM (Experts Mixture) | NeurIPS 2024 | Spike-driven MoE routing, dynamic sparse computation | A6, C3 |
| IMP-SNN | NeurIPS 2024 | Learnable initial membrane potential, richer spike patterns | A7 |
| SpikeSlicer | NeurIPS 2024 | Adaptive event stream slicing, SPA-Loss | A4, A9 |
| Activity Pruning AT-LIF | NeurIPS 2024/2025 | Adaptive threshold, current-based output | A5, H1 (baseline) |
| Sparse SNN (Timescale Het.) | ICLR 2024 | Timescale heterogeneity for pruning recurrent SNN | A2, A8 |
| QP-SNNS | ICLR 2025 | Weight quantization + structured pruning, SVD criterion | A3b, A8 |
| DPRC-SNNs | ICLR 2026 (review) | Channel-level structured pruning, orthogonality regrowth | A3b, D1 |
| Activation Sparsification (Xu et al.) | Neural Networks 2025 | ~5% spike density training on SENECA neuromorphic processor | A8 |
| All-in-One-Timestep | arXiv 2025 | Multi-level spiking, Sparse-ResNet for spike avalanche | A1, B2 |
| Spiking Patches | arXiv 2025 | Asynchronous sparse event tokenization, 10.4× faster inference | B1 |
| TTFSFormer | ICML 2025 | Time-to-first-spike, lossless attention conversion | A4, C4 |
| LoAS (Dual-Sparse Dataflow) | MICRO 2024 | Weight + activation dual-sparsity hardware dataflow | H2 (hardware) |
| MINT (Multiplier-less INT) | ASP-DAC 2024 | Quantization-aware SNN accelerator, multiplier-less integer | D2 |

---

## Phase 1: Neuron Operator Innovations

### A1 — FSN Multi-Level on G1 Nodes [READY]
- **Hypothesis**: G1's 6 layer0 nodes are the highest-impact gating targets. Upgrading from binary HardSparseGate to FusedSparseNeuron (2-level signed = ternary) allows each spike to carry polarity information (-1, 0, +1), matching optical flow's positive/negative event encoding.
- **Paper backing**: BSA (NeurIPS 2025) validates ternary spike for polarity preservation; All-in-One-Timestep validates multi-level spiking with reduced quantization error.
- **Config**: `use_fsn: true`, `fsn_num_levels: 2`, `fsn_signed: true`, `stage_selection: layer0_only`
- **Hardware mapping**: 2 comparators + sign bit → 2-bit spike bus → signed AND-popcount accumulation
- **Risk**: medium — ternary spikes may affect BN statistics in eval mode
- **Target**: SOPs < 2.5G (30% reduction), AEE < 1.75

### A2 — Leakage-as-Gate
- **Hypothesis**: PSN neurons with high decay naturally produce fewer informative spikes. Derive gate signal from decay parameter instead of learning separate gate_logit. Couples temporal dynamics to sparsity with zero extra parameters.
- **Paper backing**: Sparse SNN (ICLR 2024) shows timescale heterogeneity is effective for pruning.
- **Mechanism**: gate_prob = sigmoid((τ - τ_min) / (τ_max - τ_min) × scale)
- **Hardware mapping**: decay register → threshold comparator → clock-gate (reuses existing register)
- **Risk**: medium — decay and sparsity may have conflicting optima
- **Target**: SOPs 2.8-3.2G

### A3 — Hierarchical Shared Gates
- **Hypothesis**: Neurons in the same stage have correlated spike patterns. Share one gate per stage (4 gates instead of 36) for dramatic parameter reduction.
- **Paper backing**: DPRC-SNNs (ICLR 2026) validates channel-level structured sparsity; QP-SNNS (ICLR 2025) validates structured pruning with SVD criterion.
- **Config**: `stage_selection: all_stages_proj`, `gate_sharing: stage`
- **Hardware mapping**: 4 global clock-gate signals → broadcast to all neurons per stage
- **Risk**: low
- **Target**: SOPs 2.7-3.0G

### A3b — Structured Channel + Stage Dual Sparsity (from DPRC-SNNs + QP-SNNS)
- Combine stage-level shared gates (A3) with channel-level structured pruning
- Use singular-value-based pruning criterion on spatiotemporal spike activities (from QP-SNNS)
- Orthogonality-driven regrowth for pruned channels (from DPRC-SNNs)

### A4 — Spike-Timing-Dependent Gate
- **Hypothesis**: Early timesteps are noise-dominated, late timesteps carry coherent motion. Gating should be time-aware: early-bin spikes penalized more heavily.
- **Paper backing**: SpikeSlicer (NeurIPS 2024) adaptive event slicing; TTFSFormer (ICML 2025) time-to-first-spike.
- **Mechanism**: gate_penalty[t] = base × (1 + α × (T-t)/T)
- **Hardware mapping**: bin counter + LUT per neuron group
- **Risk**: medium
- **Target**: SOPs 2.5-3.0G

### A5 — Refractory-Period Pruning [HIGH PRIORITY — low risk]
- **Hypothesis**: After emitting a spike, a neuron carries little new information for 2-3 timesteps. Hardware-enforced refractory period reduces temporal firing density.
- **Paper backing**: Activity Pruning AT-LIF (NeurIPS 2024/2025) — achieves 0.06 avg firing rate on ImageNet. Current-based output (zero when silent) for cleaner sparse forward pass.
- **Mechanism**: After spike at time t, block spikes at t+1, t+2 via refractory mask
- **Hardware mapping**: 2-bit saturating counter per neuron → zero ALU overhead
- **Risk**: LOW — simple, biologically validated
- **Target**: SOPs 2.9-3.3G (modest but guaranteed)

### A6 — Bipolar Spike Attention Gate [NEW — from BSA NeurIPS 2025 + SEMM NeurIPS 2024]
- **Hypothesis**: Apply FSN's signed ternary spike specifically to attention Q/K projection neurons. The signed spike preserves polarity information that binary spikes lose, enabling better attention score discrimination with fewer active neurons.
- **Paper backing**: BSA (NeurIPS 2025) — bipolar self-attention with ternary spike neurons and Shiftmax (bit-shift softmax). SEMM (NeurIPS 2024) — spike-driven MoE routing.
- **Mechanism**: 
  1. Replace attention Q/K projection neuron wrappers with FSN (signed=True, num_levels=2)
  2. Optionally use spike activity as MoE routing signal (only active experts compute)
  3. Optionally replace softmax with Shiftmax for hardware-friendly attention
- **Config**: `use_fsn: true`, `fsn_signed: true`, `stage_selection: attn_qk_only` (new mode)
- **Hardware mapping**: Signed 2-bit spike bus → bipolar accumulation in attention unit
- **Risk**: medium — attention-specific, needs careful Q/K gradient flow
- **Target**: SOPs 2.3-2.7G (attention is ~30% of total ops)

### A7 — Learnable Initial Membrane Potential Gating [NEW — from IMP-SNN NeurIPS 2024]
- **Hypothesis**: Instead of learning a separate gate_logit, learn the initial membrane potential (IMP). Neurons with low IMP naturally fire less → IMP acts as a built-in soft gate.
- **Paper backing**: IMP-SNN (NeurIPS 2024) — learnable IMP generates richer spike patterns, +4.05% on ImageNet.
- **Mechanism**: 
  1. Replace per-neuron gate_logit with per-neuron learnable IMP
  2. gate_prob = sigmoid((IMP - IMP_min) / IMP_scale)
  3. IMP also affects spike timing → joint optimization of sparsity + temporal dynamics
- **Hardware mapping**: IMP register already exists in neuron → zero extra storage for gate
- **Risk**: medium — joint optimization may be unstable
- **Target**: SOPs 2.5-3.0G

### A8 — Dual-Sparsity Training Regularizer [NEW — from Xu et al. 2025 + QP-SNNS ICLR 2025]
- **Hypothesis**: Jointly optimize weight sparsity AND activation sparsity during training. A single loss term pushes both dimensions simultaneously.
- **Paper backing**: Xu et al. (Neural Networks 2025) — ~5% spike density on SENECA processor. QP-SNNS (ICLR 2025) — unified weight quantization + structured pruning.
- **Mechanism**: 
  ```
  Loss = task_loss + λ_firing × mean_firing_rate + λ_weight × L1_weight_norm
  ```
- **Implementation**: Add dual regularizer to existing H1/A1 training loop
- **Hardware mapping**: Benefits both: fewer weights × fewer spikes = quadratic energy reduction
- **Risk**: low — just a loss term addition
- **Target**: Additional 10-15% SOP reduction on top of existing gates

---

## Phase 2: Voxelization + Neuron Co-Design

### A9 — Event-Rate-Adaptive Timestep Reduction [NEW — from SpikeSlicer + Spiking Patches]
- **Hypothesis**: Low-event-rate regions don't need all 10 timesteps. Dynamically reduce timesteps based on local event density.
- **Paper backing**: SpikeSlicer (NeurIPS 2024) — adaptive event slicing. Spiking Patches (arXiv 2025) — asynchronous sparse tokenization, 10.4× faster inference.
- **Mechanism**: Compute per-region event count → skip timesteps where event_count < threshold
- **Risk**: HIGH — changes the fundamental T=10 assumption of the architecture

### B1 — Sparse Voxel Frontend
- Replace dense voxel grid with sparse voxel representation (only store non-empty voxels)
- Paper backing: Spiking Patches (arXiv 2025)

### B2 — Multi-Level Voxel Encoding
- Co-design voxel quantization levels with neuron spike levels for end-to-end sparsity
- Paper backing: All-in-One-Timestep (arXiv 2025)

---

## Phase 3: Attention Sparsity

### C1 — Window-Aware Gate
- Attention windows with zero events → gate off all neurons in that window

### C2 — Sparse Attention Score Threshold
- Don't compute attention for token pairs with score below threshold

### C3 — Spike-Driven MoE Attention [NEW — from SEMM NeurIPS 2024]
- Use spike activity itself as expert routing signal
- Only "active" attention heads compute → dynamic sparse attention
- EMSA (Experts Mixture Spiking Attention) architecture

### C4 — Shiftmax Attention [NEW — from BSA NeurIPS 2025]
- Replace softmax with bit-shift-based approximation
- Hardware: no exponentiation, no division → shift + add only
- Compatible with our 2-bit signed spike datapath

---

## Phase 4: Pruning + Hardware Accelerator Co-Design

### D1 — Structured Neuron Column Pruning
- Remove entire neuron columns that fire below threshold
- Paper backing: DPRC-SNNs (ICLR 2026) — channel-level structured sparsity

### D2 — Mixed-Precision Spike Datapath
- Critical layers: 4-bit spike (FSN num_levels=4)
- Non-critical layers: 1-bit spike (binary)
- Paper backing: MINT (ASP-DAC 2024) — multiplier-less integer quantization

### H2 — Dual-Sparse Hardware Dataflow [NEW — from LoAS MICRO 2024]
- Design hardware dataflow exploiting BOTH weight sparsity + activation sparsity
- Paper backing: LoAS (MICRO 2024) — fully temporal-parallel dataflow for dual-sparse SNNs
- Target: hardware accelerator architecture document

---

## Hardware Accelerator Design Principles

Every neuron innovation must answer:
1. **Area cost**: Additional transistors/gates per neuron?
2. **Energy model**: Switching activity reduction?
3. **Control complexity**: Global signals needed?
4. **Regularity**: Preserves systolic-friendly array structure?
5. **Memory**: Additional SRAM/register overhead?

Target hardware metrics (28nm):
- Neuron area: < 500 µm² (baseline PSN ~300 µm²)
- Spike energy: < 0.5 pJ/spike (baseline ~1.0 pJ/spike)
- Control overhead: < 5% of total area

---

## Execution Priority

| # | Experiment | Risk | Est. SOP Gain | Dependencies |
|---|-----------|------|---------------|--------------|
| 1 | A5 — Refractory pruning | LOW | 15-20% | None (simple mechanism) |
| 2 | A1 — FSN on G1 | MED | 25-30% | H1 overlay (sparse_gate.py) |
| 3 | A8 — Dual-sparsity regularizer | LOW | 10-15% | Works with any gate setup |
| 4 | A6 — Bipolar attention gate | MED | 20-30% | FSN + attention target mode |
| 5 | A3 — Shared gates | LOW | 20-25% | New gate_sharing option |
| 6 | A7 — IMP gating | MED | 20-30% | New IMP neuron wrapper |
| 7 | A9 — Adaptive timestep | HIGH | 20-30% | Architecture change |
