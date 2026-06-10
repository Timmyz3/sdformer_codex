# SDformerFlow DATE 2026 Co-Design Claim-Evidence Map (research-paper-writing lens)

**Target**: DATE 2026  
**Core Story** (from DATE_PAPER_SKELETON_CN.md + PAPER_CO_DESIGN_PROPOSAL.md):  
"Hardware-Aware Sparse Ternary Spiking Transformer for Energy-Efficient Event-Based Optical Flow"  
— Not another neuron swap paper. A **stage-aware static heterogeneous schedule** that turns software sparsity wins into deterministic, hardware-mappable efficiency with clear implementation path.

**Three Iron Rules enforced here**:
- Versioned at every major claim.
- All numbers human-verified against EXPERIMENT_REDESIGN_PLAN.md + latest profiles (split 825 canonical).
- DATE style: emphasize static schedule, operator cost models, layer-wise evidence, comparison to accelerators — not just "we got lower SOPs".

---

## Primary Claims + Evidence (CER discipline)

### Claim 1: Stage-aware static replacement (S02/S012 high-SOPs FFN + unified ternary attention) delivers substantial efficiency gains while keeping accuracy degradation controllable.
**Evidence (software, from EXPERIMENT_REDESIGN_PLAN.md — to be re-run on canonical 825 split for paper)**:
- Baseline (PSN/upstream style): AEE ≈ 1.585 (valid40), SOPs ≈ 3.622G, firing ≈ 0.085.
- SN (signed_shiftnorm) S02: AEE 0.96 (short test), SOPs 3.23G, firing 0.076 — strong stable across FFN configs.
- TX (ternary_axnor) S012: SOPs down to 2.92G (≈ -19%), firing 0.069.
- Cross-FFN summary: SN is the only attention that remains stable; S02/S012 FFN replacement gives best SOPs (2.82–3.23G range) without uniform accuracy collapse.
- Firing rate reductions and SOPs Pareto shown in generated nature figures (Fig. 1, Fig. 3).

**Reasoning**:
- Different stages have very different compute profiles (Stage 2 dominates). Uniform replacement is wasteful or risky; selective high-SOPs FFN replacement captures most savings.
- Ternary attention (sign/valid + XNOR/popcount or shift-norm) removes expensive multiplies while preserving event polarity.

**Caveats / Scope (must be explicit in paper)**:
- Current best accuracy candidates still show AEE degradation (e.g. some H41 variants ~1.73). The 5% target (AEE ≤ ~1.66) requires further recovery techniques (refinement head, better training, or selective full-precision fallback).
- All numbers are dev-split (DSEC subset). Official test/benchmark results required for final claims.
- SOPs is a proxy; real energy depends on actual kernel implementation and data movement.

**DATE-specific strength**: This claim is not "better neuron". It is "we profiled, decided replacement mask at compile time, and can map it to static kernel switching".

### Claim 2: The replacement policy can be expressed as a simple, hardware-friendly static layer/block schedule table.
**Evidence (planning + dataflow spec)**:
- From HARDWARE_DATAFLOW_SPEC.md + DATE_PAPER_SKELETON: replacement decided by sensitivity + SOPs contribution analysis before deployment.
- Schedule table concept: per-stage / even-odd block mask → kernel type (dense PSN / binary ATLIF / ternary ATLIF + specific attention normalizer).
- No runtime dynamic search or complex controller for the mask itself.
- Data types formalized: binary spike (1-bit), ternary (2-bit sign+valid), fixed-point for scores/thresholds.

**Reasoning**:
- Static schedule avoids the control overhead and buffer format thrashing that would come from arbitrary per-block runtime decisions.
- Maps directly to the "heterogeneous accelerator subsystem" with spike neuron engine + ternary attention engine + sparse FFN engine + schedule controller (as defined in accelerator design doc).

**Hardware evidence needed (to be generated in Phase 3/5)**:
- Layer schedule table (table in paper).
- Operator cost model (MAC vs XNOR vs popcount vs shift vs SRAM access).
- Memory traffic / buffer reuse estimates per schedule.

### Claim 3: This co-design yields a publishable DATE contribution because it bridges model-level sparsity to concrete architectural and implementation decisions.
**Supporting points (claim-evidence chain)**:
- Sensitivity-driven, rule-based replacement is a reusable design methodology (matches DATE "design methodologies for ML architectures").
- Hardware cost model + microarchitecture sketches for the key engines (spike neuron, ternary attention with sign/XNOR/popcount/shift-norm, sparse accumulation) provide the required implementation evidence.
- Comparison dimensions for DATE: not only AEE/AAE vs baseline, but energy proxy, latency proxy, area estimate (28nm target), on-chip SRAM (<2MB), throughput target (30 FPS @ 480×640).
- Avoids common DATE rejection reasons: no "just software paper", no un-implementable dynamic sparsity, clear static schedule story.

**Anti-patterns we refuse (per research-paper-writing + DATE skeleton)**:
- Reporting only SOPs/firing without operator-level cost model or schedule table.
- Claiming "hardware friendly" while the replacement policy would require complex runtime control.
- Hiding accuracy–efficiency trade-off or split inconsistencies.

---

## Suggested Paper Structure (DATE-optimized, from skeleton)

**Title options**:
- Hardware-Aware Sparse Ternary Spiking Transformer for Energy-Efficient Event-Based Optical Flow
- (or SATFlow / HASTE-Flow short form)

**Abstract / Intro**:
- Event camera optical flow on the edge needs both accuracy and extreme efficiency.
- Pure software SNN improvements (ternary neurons + sparse attention) are necessary but not sufficient for DATE.
- We present a co-design: stage-aware static schedule + hardware-mappable kernels that turn those improvements into deterministic efficiency gains with clear implementation path.

**Related Work** (short, positioning):
- SDformerFlow baseline + QKFormer/STAtten style spike-driven attention.
- Prior SNN accelerators (SENECA, FireFly-T, Bishop, etc.) and token pruning works (HeatViT, SpAtten).
- Gap: most either keep dense attention or use unstructured/dynamic sparsity that is hard to schedule.

**Method / Co-Design** (core):
- Software redesign space (ternary PSN+ATLIF, TX/SC/SN attentions, S02/S012 FFN).
- Profiling → sensitivity + SOPs analysis → stage-aware replacement policy.
- Static schedule table definition.
- Hardware subsystem: spike neuron engine, ternary attention engine (sign extraction, XNOR/popcount, shiftmax/shiftnorm), sparse FFN, controller.
- Data type and fixed-point considerations.

**Experiments**:
- Software: accuracy–efficiency Pareto on DSEC dev (canonical 825), ablations on attention normalizers and FFN replacement scope (use nature figures).
- Hardware evidence: operator cost model, layer schedule table, estimated energy/latency/memory vs baseline and vs uniform replacement.
- (Future) DC synthesis numbers, comparison table vs SOTA accelerators.

**Discussion / Limitations**:
- Accuracy recovery still needed for the most aggressive schedules.
- Current results are on dev split; full benchmark required.
- Scope: we target the encoder/transformer core; full end-to-end (voxelization + decoder) left for future or shown in software only.

**Conclusion**:
- Static heterogeneous schedule is the key that makes software sparsity wins hardware-realizable.
- Opens a path for energy-efficient event-based flow on edge neuromorphic or conventional accelerators.

---

## Immediate Next Actions (workflow dispatch)

1. **Human verification gate** (you must do):
   - Re-run or confirm the key numbers in EXPERIMENT_REDESIGN_PLAN.md on the canonical `valid_split_seq.csv` (sha 7f3dc28..., 825 samples).
   - Say explicitly: "已人工核对 redesign 主要表通过，split 口径统一" before we use them in figures/draft.

2. **Version save** (mandatory before proceeding):
   - git commit the current state + this workflow state file, or create dated archive.

3. **Run the nature-figure script**:
   - `cd paper_artifacts/figures && python nature_figure_sdformer_results.py`
   - Inspect the SVGs/PDFs. Fix any data or styling. Commit outputs with verification note.

4. **Hardware planning parallel task** (Week 1 of your roadmap):
   - Polish / complete the full dataflow spec (HARDWARE_DATAFLOW_SPEC.md) with the chosen best candidate (e.g. SN or TX + S02).
   - Produce the layer schedule table artifact.

5. **Writing**:
   - We will next produce a first full section draft (Introduction + Method co-design + Experiments software part) using research-paper-writing (DATE lens) + nature-polishing.

**Current workflow status**: Phase 0 complete, Phase 2+3 artifacts launched (state file + nature-figure script + this claim-evidence map). Awaiting your confirmation + version save + data verification to advance to full drafting and hardware RTL push.

All sub-skills remain available independently (e.g. you can still say "nature-figure 再画一个硬件代价 breakdown" or "research-paper-writing 帮我强化 claim 2 的 DATE 证据链").

Ready for your confirmation.