# Draft Section: 3. Co-Design Methodology — Stage-Aware Static Heterogeneous Schedule (Phase 4 start)

**Style**: research-paper-writing (DATE systems co-design lens) + nature-polishing notes. CER discipline enforced. All claims tied to evidence. DATE voice: implementation-focused, static schedule as the hero, not neuron tricks.

## 3.1 Motivation: Why Static Schedule Instead of Uniform or Dynamic Replacement

The SDformerFlow baseline (spike-driven Swin-style encoder on event voxels) exhibits highly non-uniform compute across stages (see Fig. 2 in the generated nature figures: Stage 2 alone accounts for ~23% of GFLOPs proxy). Prior work on SNN sparsity (QKFormer, STAtten, various ternary neurons) has shown that replacing dense operations with binary/ternary or pruned equivalents can reduce SOPs and firing rate significantly. However, naively applying such replacements uniformly across the network either (a) incurs unacceptable accuracy degradation on motion boundaries and large displacements, or (b) creates a heterogeneous datapath whose control logic and buffer management become prohibitively expensive in hardware.

**Claim**: A compile-time, stage-aware static replacement policy, derived from sensitivity and per-stage SOPs profiling, produces a deterministic layer schedule that preserves most efficiency gains while keeping the hardware mapping simple and predictable.

**Evidence** (from EXPERIMENT_REDESIGN_PLAN.md, Phase 2-S02 / S012 experiments):
- SN (signed_shiftnorm) attention with S02 FFN replacement consistently achieves SOPs in the 3.0-3.3G range (vs baseline ~3.6G) with stable AEE across multiple short and longer runs.
- TX (ternary_axnor) with S012 reaches the lowest SOPs (~2.92G in short tests) but shows more variance in accuracy.
- Cross-FFN analysis shows SN is the only attention variant that remains robust whether only high-SOPs stages or more blocks are replaced.
- Firing rates drop from ~0.085 (baseline) to 0.069-0.077 in the selected candidates.

**Reasoning for hardware friendliness**:
- The policy is decided once, offline, using the same profiling that any design-space exploration would perform.
- At deployment the network is emitted with a fixed per-block kernel annotation (dense PSN, binary ATLIF, ternary ATLIF + specific normalizer).
- The accelerator only needs a small schedule table and muxes to select the pre-synthesized kernel for that layer — no runtime mask generation, no complex dynamic sparsity controller, no frequent buffer format conversion.

This directly addresses the DATE requirement for "design methodologies" and "architectural design" rather than pure algorithmic improvement.

## 3.2 Software Primitives and Replacement Policy

### Neuron Level
We retain the upstream PSN for early layers where expressiveness matters most, and introduce ATLIF (adaptive threshold leaky integrate-and-fire) with ternary output for later stages. ATLIF's learnable threshold scaling directly controls sparsity while the symmetric positive/negative firing preserves the polarity information critical for event-based motion.

### Attention Level
We replace the dense QK multiplication + softmax with hardware-light variants explored in the redesign search:
- **TX (ternary_axnor)**: Ternary spikes → sign extraction + XNOR + popcount (or L1 variant).
- **SN (signed_shiftnorm)**: Signed consensus or shift-based normalization that maps to cheap shifts instead of multiplies.
- **SC (signed_consensus)**: Similar consensus mechanism.

All variants operate on the spike-driven Q-K path already present in the SDformerFlow upstream, adding only local temporal bias where beneficial (inspired by STAtten but kept lightweight).

### FFN / MLP Replacement
FFN blocks (the dominant compute consumer after patch embedding) are selectively replaced:
- High-SOPs stages (primarily Stage 0 and Stage 2, per the dataflow breakdown) use binary or ternary ATLIF + sparse linear layers.
- Lower-SOPs or high-sensitivity stages retain denser PSN paths.
- The decision is encoded as the static schedule table (example in Table X, generated from the sensitivity analysis in EXPERIMENT_REDESIGN_PLAN.md).

The resulting network is therefore a **static heterogeneous SNN Transformer**: different blocks use different kernels, but the mapping is known at compile time and does not change at runtime.

## 3.3 Hardware Subsystem Architecture

The software policy is mapped to a **Spiking Event-Transformer Accelerator Subsystem** (not a full SoC; the decoder and post-processing remain in software or a lighter accelerator for future work).

Key engines (directly derived from the primitives above):
- **Spike Neuron Engine**: Supports PSN (dense), binary ATLIF, and ternary ATLIF with on-the-fly threshold adaptation. Small per-neuron state (membrane + threshold) kept in registers.
- **Ternary Attention Engine**: Sign/valid extraction, XNOR or shift-norm datapath, popcount tree or shift-based accumulation, followed by shiftmax/shiftnorm gating. Window-tiled to match the Swin structure and exploit the dual-bank Window SRAM.
- **Sparse FFN Engine**: Binary/ternary sparse matrix-vector units that skip zero activations according to the schedule.
- **Static Schedule Controller**: Simple FSM that walks the layer schedule table and configures the muxes and engine modes for the current block. No dynamic search logic in the critical path.

**Memory system** (targeting <2 MB on-chip SRAM for edge feasibility):
- Weight buffer (256 KB) streamed from DRAM per layer.
- Dual-bank Window SRAM (512 KB total) for the current spatial-temporal window tile (Q/K/V + partial results).
- Explicit ping-pong and reuse analysis per schedule entry (to be quantified in the cost model).

Data types are locked early (see HARDWARE_DATAFLOW_SPEC.md): 1-bit/2-bit spikes, FP16 or INT8 weights, narrow fixed-point for attention scores and gates. This enables the area and energy numbers DATE reviewers expect.

## 3.4 Co-Design Loop and Evidence Chain

The loop is:
1. Software redesign search (H-series in EXPERIMENT_REDESIGN_PLAN.md) → candidate policies + measured SOPs/firing/AEE.
2. Sensitivity + per-stage SOPs profiling → rule-based replacement mask (S02 or S012 template).
3. Emission of static layer schedule table.
4. Hardware cost modeling (operator-level: XNOR vs MAC, SRAM access, etc.) and microarchitecture sizing.
5. Feedback: if accuracy or hardware cost is unacceptable, adjust the replacement scope or primitive parameters and re-search (closed loop, not one-way model-to-hardware).

**Current evidence** (software side, Phase 3 figures):
- Pareto plot (Fig. 1) shows multiple points inside or near the target region (AEE within ~5% of baseline while SOPs reduced >15-20%).
- Stage breakdown (Fig. 2) justifies why S02 is attractive.
- Firing and relative SOPs summary (Fig. 3) quantifies the sparsity win.

**Hardware evidence still required** (to be completed):
- Concrete operator cost table and cumulative energy proxy for the chosen schedule vs baseline.
- Layer schedule table (text + visual).
- Microarchitecture block diagram with data widths and critical paths.
- (Stretch for DATE) Preliminary DC synthesis area/power numbers in 28nm.

This chain is what turns the software experiments into a DATE paper rather than a model improvement paper.

**Nature polishing note (for later pass)**: Keep the language precise and understated ("we demonstrate a mapping", "enables", "reduces the effective cost by"). Avoid "dramatic", "significant" without numbers. Every claim must have a figure or table pointer.

---

**Status**: This is a first-cut Method subsection ( ~ Phase 4 early ). Expand with exact numbers after figure verification and hardware cost model. Can be fed to nature-polishing or academic-paper revision mode later.

**Verification required**: All numbers in this draft cross-checked against EXPERIMENT_REDESIGN_PLAN.md and the generated nature figures.
