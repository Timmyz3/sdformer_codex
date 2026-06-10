# Hardware Week 1 Action: Full System Dataflow Spec Polish (DATE 2026)

**From user's 6-week roadmap (HARDWARE_RESEARCH_ROADMAP.md)**: Week 1 = "全系统数据流规格 (不写代码，定架构)". Day 1-2: top-level interfaces, Day 3-5: per-module dataflow, Day 6-7: schedule table + buffer sizing.

**Base document**: neuron_autoresearch/HARDWARE_DATAFLOW_SPEC.md (2026-05-29 | H41 SC S012C epoch27 | 28nm target).

**Goal for this week (workflow dispatch)**: Lock a concrete, chosen-candidate version of the spec tied to the Phase 2 framework (recommend SN S02 or TX S012 as primary). Produce:
- Updated schedule table for the chosen replacement policy.
- Explicit operator cost model table (to be used in paper Fig. for energy proxy).
- Refined buffer hierarchy and data type decisions.
- Interface signals ready for RTL (Week 2-3).

## 1. Chosen Candidate for Hardware Mapping (from Phase 2 lock)
- **Primary**: SN (signed_shiftnorm) + S02 FFN replacement (stable across experiments, good SOPs reduction ~10-20%+, hardware friendly normalization).
- **Strong alternative for ablation**: TX (ternary_axnor) + S012 (lowest SOPs in short tests ~2.92G).
- Unified attention: ternary for all blocks (PSN + ATLIF for polarity preservation).
- FFN: selective binary/ternary in high-SOPs stages (Stage 0+2 or 0+1+2).

This produces a **static, compile-time schedule** — no runtime mask search.

## 2. Updated Global Conventions (refined from base spec)
**Data types (locked for 28nm)**:
- Weights: FP16 (1-5-10) or INT8 post-quant.
- Activations: Binary spike (1-bit), Ternary spike (2-bit: 00 silent, 01 +thre, 10 -thre).
- Attention scores: 6-bit signed for popcount or shift-norm results.
- Thresholds/gates: 8-bit fixed point.

**Coordinate system** (event-native):
- T: timesteps (5 for PSN, 2 for window attention).
- Tokens N = T × H_patch × W_patch (after voxel/patch embed).
- Stages increase C and decrease spatial (96→192→384→768).

**Memory hierarchy (target <2MB on-chip for edge)**:
- DRAM (off-chip, HBM2/LPDDR): weights + full activations (8GB modeled).
- Weight Buffer: 256KB SRAM, 256-bit, 2 cyc.
- Window SRAM: 512KB dual-bank, 256-bit (Q/K/V + mid activations for current window tile).
- Spike/ternary state: small per-neuron registers (membrane + threshold).

## 3. Layer/Block Schedule Table (Core Artifact — must go in paper)
Example for SN S02 primary (to be filled with exact numbers from your H41 or best checkpoint after canonical re-run):

Stage | Block | Attention Type | FFN Type | Spike Encoding | Est. Rel. Cost (vs baseline MAC) | Notes
------|-------|----------------|----------|----------------|----------------------------------|-------
0     | 0     | SN ternary     | Dense PSN| Ternary        | 0.6x (XNOR + shift)             | High SOPs, keep expressive
0     | 1     | SN ternary     | Binary ATLIF | Binary     | 0.35x                           | Replace for sparsity
... (full table for all stages/blocks)

**How to generate the real table**:
- Use your latest profile (SOPs per layer from tools/profile_sops.py or similar).
- Apply the S02 rule: replace FFN in stages 0 and 2 (or high-SOPs low-sensitivity).
- Output as CSV + LaTeX table for paper.

## 4. Operator Cost Model (for DATE energy proxy — critical)
Define costs (to be calibrated with Accelergy/Timeloop or your golden sim in 28nm):

Op | Cost (pJ or relative) | Used in
----|-----------------------|--------
MAC (dense) | 1.0 (baseline) | Conv/FC dense
XNOR + popcount (ternary attn) | 0.25-0.4 | TX/SC attention
Shift + norm (SN) | 0.3 | signed_shiftnorm
Add / compare (spike neuron) | 0.1 | PSN/ATLIF threshold
SRAM read (256-bit) | 0.05 per access | All buffers

**Paper figure needed**: Bar chart of cumulative energy proxy per stage for baseline vs chosen schedule.

## 5. Refined Dataflow for One Forward Pass (example for one Swin block)
1. Voxel stream / patch embed → binary/ternary spikes into Window SRAM.
2. For each window tile:
   - Load Q/K/V (spike encoded).
   - Ternary attention: sign extract → XNOR or shift-norm → popcount/score → shiftmax gate → weighted sum (sparse).
   - Membrane update in spike neuron engine (ATLIF threshold adapt).
3. FFN: according to schedule table — dense or sparse binary/ternary path.
4. Write back to buffer / next stage or DRAM for residual.

**Static schedule controller** simply indexes the table and muxes the engine/kernels. No dynamic sparsity logic in critical path.

## 6. Immediate Tasks for You This Week (Week 1)
- Pick exact best checkpoint (e.g. H41 SC S02 or latest SN) and re-profile on canonical 825 split.
- Fill the full layer schedule table (all stages, all blocks).
- Calibrate 2-3 operator costs using your golden_hw_sim.py or add simple energy counters.
- Produce the "Layer Schedule" figure (table + small diagram of kernel switching).
- Update this doc + HARDWARE_DATAFLOW_SPEC.md with the chosen policy.
- Commit as "v0.2-hw-week1".

**Output artifacts to produce**:
- `paper_artifacts/hardware/layer_schedule_table.csv` + .tex
- Updated energy proxy numbers for the nature figures (we can extend the script).

This completes Week 1 "定架构". Week 2 can start RTL for the chosen engines (spike_unit, token_mixer/attention for SN, etc.).

Tie-back to software: The replacement policy comes directly from the sensitivity analysis in EXPERIMENT_REDESIGN_PLAN.md — this is the co-design loop.

**Workflow note**: This is parallel to Phase 3 figures (hardware cost bars can be added to the Pareto figure set). Human verification of costs vs real profiles required.
