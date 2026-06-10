# Phase 2: Chosen Framework (Locked) — SDformerFlow for DATE 2026

**Source**: Synthesized from docs/DATE_PAPER_SKELETON_CN.md + PAPER_CO_DESIGN_PROPOSAL.md + EXPERIMENT_REDESIGN_PLAN.md + HARDWARE_* docs. 
**Method**: Used as strong prior per full-paper-workflow (paper-framework role). One primary arc locked; alternatives noted for flexibility.

## Primary Story Arc (Recommended for DATE)
**Title (working)**: Hardware-Aware Sparse Ternary Spiking Transformer for Energy-Efficient Event-Based Optical Flow

**Core Narrative (DATE-optimized, NOT CVPR-style)**:
"We do not claim a new SNN neuron or attention trick in isolation. Instead, we show a complete **software-hardware co-design loop**:
1. Profile-driven identification of high-SOPs, low-sensitivity stages/blocks in SDformerFlow.
2. Design of hardware-friendly primitives (adaptive ternary PSN+ATLIF, sign/valid ternary attention with XNOR/popcount or shift-norm).
3. Compile-time stage-aware static replacement policy that produces a deterministic layer schedule table.
4. Mapping to a heterogeneous accelerator subsystem (spike neuron engine + ternary attention engine + sparse FFN engine + static controller) with clear operator cost models.
This turns model-level sparsity wins into predictable energy, latency, and area benefits suitable for edge neuromorphic or low-power accelerators."

**Why this works for DATE** (from skeleton):
- Matches: Architectural design, Low-power, Approximate computing (ternary/structured sparsity), Design methodologies (sensitivity + SOPs profiling → static schedule), Neuromorphic/edge.
- Avoids rejection traps: No "black-box replacement", no dynamic pruning controller, explicit hardware evidence chain required.

## Locked Structure (Phase 2 Backbone)
1. **Introduction** (4-5 paras, DATE voice):
   - Event optical flow on edge: accuracy + extreme efficiency needed (real-time, <5W).
   - SDformerFlow baseline (spike-driven Swin/QK-style) + prior SNN improvements.
   - Gap: Software sparsity (SOPs/firing) does not automatically translate to hardware; partial replacement can increase control complexity.
   - Contribution: Stage-aware static co-design that makes sparsity hardware-mappable + implementation evidence.

2. **Background & Motivation**:
   - SDformerFlow architecture recap (focus on stage compute distribution — Stage 2 heaviest).
   - Ternary neuron and attention design space (from redesign experiments).
   - Hardware challenges for SNN Transformers (data types, sparsity mapping, buffer hierarchy).

3. **Co-Design Methodology** (core technical):
   - Profiling: SOPs contribution + sensitivity analysis per stage/block (tie to EXPERIMENT_REDESIGN_PLAN).
   - Primitive design: Adaptive ternary (PSN preserves expression, ATLIF controls sparsity + polarity), hardware-friendly attention (TX/SC/SN variants).
   - Policy: Stage-aware static schedule (S02 or S012 templates for high-SOPs FFN; unified ternary attention). Compile-time mask → schedule table.
   - Hardware subsystem architecture (from HARDWARE_ACCELERATOR_DESIGN + DATAFLOW_SPEC):
     - Spike neuron engine (support PSN/binary/ternary + threshold update).
     - Ternary attention engine (sign extraction, XNOR/popcount, shiftmax/shiftnorm).
     - Sparse FFN engine.
     - Static schedule controller + on-chip buffers (weight 256KB, window SRAM dual-bank).
   - Dataflow & data types (FP16 weights, 1/2-bit spikes, fixed-point scores).

4. **Experiments & Evidence**:
   - Software: Accuracy-efficiency Pareto (nature Fig. 1), firing/SOPs (Fig. 3), cross-FFN stability (SN best).
   - Hardware evidence (to be completed in Phase 3/5):
     - Operator cost model (MAC/add/XNOR/popcount/shift/SRAM access costs).
     - Layer schedule table (example with chosen candidate, e.g. SN S02 or TX S012).
     - Energy/latency/memory traffic proxies vs baseline and vs uniform replacement.
     - Microarchitecture sketches (Fig. from skeleton).
   - Ablations: replacement scope (stage vs block), attention normalizer variants.

5. **Discussion**:
   - Accuracy recovery techniques still needed for aggressive schedules.
   - Static schedule as reusable methodology.
   - Comparison to SOTA accelerators (FireFly-T, SENECA, Bishop etc.).
   - Limitations & future (full end-to-end, FPGA validation, 28nm DC numbers).

6. **Conclusion**.

**Key Figures (Phase 3 delivered)**:
- Fig. 1: Pareto (SOPs vs AEE) with target region.
- Fig. 2: Stage SOPs breakdown (motivates selective replacement).
- Fig. 3: Firing + relative SOPs summary.
- Additional needed (to generate): Layer schedule table visualization, operator cost bar chart, overall co-design flow diagram.

**Chosen Candidate for Main Results** (recommendation):
- SN (signed_shiftnorm) with S02 FFN replacement as primary (stable, good SOPs, hardware friendly).
- TX (ternary_axnor) S012 as strong alternative for lowest SOPs.
- Report both in ablations. Use data from H41 / Phase 2 tables after canonical re-run.

**Alternatives Considered (for flexibility)**:
- More aggressive full-stage replacement (higher risk on accuracy, simpler HW).
- Adding dynamic token pruning on top (more HW control overhead — deprioritized for DATE).

**Version & Verification Note**:
- Locked v0.2 of workflow.
- All claims tied to specific evidence in redesignmd + hardware specs.
- Next: human verify generated figures against source data, then expand to full section drafts.

This framework is now the backbone for Phase 4 writing and hardware implementation push.
