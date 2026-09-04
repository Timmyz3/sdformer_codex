# Deep Idea Mining Report (2026-07-11)

Scope: algorithm venues (CVPR/ICLR/ICML/NeurIPS/ECCV) + architecture venues (ISCA/HPCA/DATE/CICC-style neuromorphic) + non-SNN optical-flow transformers adapted to SDformer/DATE.
Method: literature-search skill (OpenAlex/arXiv) + full-text ar5iv/PDF download + GitHub clone (STAtten, MaxFormer, SpiLiFormer, SeqSNN).

Artifacts:
- papers/*.txt full-text extracts
- papers/pdfs/*.pdf (CVPR a-XNOR, STAtten, EDCFlow, DATE2024 focus session)
- repos/{STAtten,MaxFormer,SpiLiFormer,SeqSNN}

## Working constraints for packaging
- Keep all-binary ATLIF + all12 unified formula + no native QKFormer carrier when possible.
- Prefer optional branches; do not delete old modules.
- Final evidence = standard valid825 / MVSEC, not valid10.
- Hardware story must count attention ops (popcount/Shiftmax), not only neuron SOPs.

## Ranked transferable ideas

### P0 — high fit, can package as DATE co-design story

#### I1. Error-Constrained Score/Spike Pruning (from Bishop ISCA'25)
- Paper: Bishop: Sparsified Bundling Spiking Transformers... (ISCA 2025, arXiv:2505.12281)
- Core: Error-Constrained TTB Pruning (ECP) trims spiking Q/K/V **before and after** attention maps with a **defined error bound**; Token-Time Bundle packs tokens over multiple timesteps; density stratifier routes high/low density spikes to dense/sparse cores.
- Code read: N/A (arch paper); method text explicit on ECP + TTB + heterogeneous cores.
- Map to SDformer:
  - Software: prune low |score| TTX candidates or silent tokens in each window with threshold τ chosen so row-sum/gate L1 change ≤ ε.
  - Hardware DATE: TTB natural for T=2 Swin windows; density stratifier ↔ event density per crop.
- Novelty packaging: "Error-bounded sparse dyadic TTX" — not raw token drop; error-constrained relative to full TTX.
- Risk: need valid825; if AEE jumps >5% fail.

#### I2. Temporally Dense Spike-Difference Motion Prior (from EDCFlow CVPR'25)
- Paper: EDCFlow (CVPR 2025, arXiv:2506.03512); PDF on disk.
- Core: high-res **temporal feature difference** complementary to low-res cost volume; attention-based multi-scale temporal difference layer; adaptive fusion.
- Map: inject Δ voxel / Δ encoder features as (a) additive bias on TTX scores, or (b) second cheap scalar gate fused with Shiftmax gate — **no full correlation volume**.
- Packaging: "Spike-difference guided TTX" for event flow; aligns with DATE sequential-task thesis.
- Risk: must keep all12 unified if fusion is same module everywhere; or restrict to decoder only and label as decoder module not attention mix.

#### I3. Density-Guided Token Bypass (from TP-Spikformer ICLR'26 + SparseSpikformer)
- Papers: TP-Spikformer (arXiv:2603.00527); SparseSpikformer (arXiv:2311.08806)
- Core: bypass uninformative tokens **without breaking spatial layout** (early-stop inside blocks, reassemble); co-design token+weight prune.
- Map: use event density / binary activity per Swin token to skip TTX compute; keep zeroed residual path.
- Packaging: "Event-density early-exit for DATE" — software first, RTL skip mux later.
- Risk: flow needs spatial continuity; use soft bypass (scale) not hard delete.

#### I4. Latent Cost Tokens for Window Matching (from FlowFormer, non-SNN)
- Paper: FlowFormer (arXiv:2203.16194)
- Core: compress cost volume into **latent tokens**, then transformer decoder aggregates — avoids all-pairs dense attention at full resolution.
- Map: within each Swin window, pool K into M≪N latent keys (M=4/8), TTX only vs latent keys; still gate*K reconstruction via latent.
- Packaging: "Latent-key dyadic TTX" — ANN flow idea → spike/popcount form.
- Risk: latent aggregation may hurt AAE; test M carefully.

#### I5. Energy-aware Attention Op Accounting (from STEP NeurIPS'25 + Bishop + DATE'24)
- Papers: STEP energy model (NeurIPS 2025 poster); DATE 2024 focus session PDF `1313_pdf_upload.pdf`; Bishop energy claims.
- Core: energy = spike sparsity × bitwidth × memory traffic; SNN wins on sequential/event tasks.
- Map: fix profiler to count TTX popcount + Shiftmax LUT + optional N×N; DATE table uses same model.
- Packaging: not a model innovation alone, but **mandatory** for paper credibility after H66.

### P1 — good secondary ablations / lite variants

#### I6. High-frequency Max Residual (from MaxFormer NeurIPS'25; code MaxFormer/)
- Code: `Max_Mixer` uses MaxPool2d; patch embed uses MaxPool stride-2; SSA_DWC = SSA + depthwise conv.
- Map: add max-pool residual branch in patch embed / stage0 for event edges; keep TTX elsewhere.
- Packaging: "High-frequency residual for all-binary flow encoder".
- Risk: changes early features; needs FT not full retrain ideally.

#### I7. Dual-rail Difference Gate (from SpiLiFormer ICCV'25; code SpiLiFormer/)
- Code: `FF_LiDiff_Attention`: split Q into two groups, `attn = lif(sum(q1)-sum(q2))`, then `mul(attn, k)` — feed-forward inhibition; deeper `FB_LiDiff` uses feedback.
- Map: optional dual-rail Q difference as **extra gate** on TTX (or replace SC). All-binary dual-rail already in DATE story.
- Packaging: "Lateral-inhibition dual-rail TTX gate" without full FB state first (FF only).
- Risk: FB needs state SRAM — hold for RTL later.

#### I8. ST Chunk Linear Attention only where T>1 (from STAtten CVPR'25; code STAtten/)
- Code: `chunk_size=2`, reshape Q/K/V to (num_chunks, B, heads, chunk*N, d), compute `attn = (K^T V)*scale`, `out = Q @ attn` — true space-time block, complexity O(T N D^2) per chunk not O((TN)^2).
- Map: only deepest stages; still all-binary; compare vs TTX factorized.
- Note: H66c was weak TP stencil; STAtten is different (D×D state).
- Risk: H66 already showed weights@K AAE risk; needs independent V or careful K reuse.

#### I9. Hamming Linear Space-Time (SpikeVideoFormer ICML'25)
- Core: SDHA normalized Hamming; O(T) temporal designs analyzed.
- Map: H66b already configured; run once 360+valid40 then decide.
- Packaging: only if beats TTX energy accounting with D×D cost included.

#### I10. Iterative Membrane Refinement (from RAFT + LIF dynamics)
- RAFT: iterative correlation lookup + update operator.
- Map: 2–4 micro-steps of TTX with ATLIF state carry across micro-steps (not more epochs).
- Packaging: "Recurrent spike refine without extra params".
- Risk: latency ×K; DATE must show energy still down.

#### I11. Structured Quant+Prune post-hoc (QP-SNN arXiv:2502.05905)
- Weight rescaling for low-bit; structured prune.
- Map: after TTX freeze, prune MLP channels / unused ATLIF groups; int8 already partially done.
- Packaging: deployment section, not main algorithm claim.

#### I12. Activation Sparsification Training (SENECA event flow arXiv:2407.20421)
- ANN vs SNN on neuromorphic processor with activation sparsify.
- Map: aux loss on intermediate firing rates toward target sparsity while AEE constrained.
- Packaging: training recipe for DATE energy.

### P2 — literature controls / careful only

#### I13. a-XNOR full matrix (CVPR'25 Xiao PDF on disk)
- Already H66a; oracle only; do not full-train.

#### I14. A2OS2A addition-only (arXiv:2503.00226)
- Three activation alphabets; hardware heterogenous — reference only.

#### I15. BESTformer binary CIE (arXiv:2501.05904)
- Coupled Information Enhancement for binary info loss — possible residual module.

#### I16. Global Matching (GMFlow) / convex local upsample (TMA 2412.06439)
- Decoder-side ideas for flow head, not encoder attention mainline.

#### I17. DATE'24 hybrid SNN-ANN positioning
- Hybrid units for temporal SNN + static ANN; use in discussion/future work, not force hybrid now if pure SNN is story.

## Explicit non-recommendations (already falsified or costly)
- Force symmetric ±θ ATLIF (H63–65).
- Replace mainline with full N×N a-XNOR.
- MDR ep20 low-LR calib stack.
- Stage-wise TX/SC mixed deployment if paper claims unified all12.

## Suggested experiment order (minimal)
1. Document H66c valid825 failure (done on disk).
2. I1 ECP-style score prune on TTX ep2 short FT + valid825.
3. I3 density bypass short FT + valid825.
4. I2 spike-difference score bias short FT.
5. I4 latent-key TTX screen.
6. Energy model (I5) before any RTL change.
7. I6/I7 only if AEE still needs help.

## GitHub clones
- STAtten: Intelligent-Computing-Lab-Yale/STAtten (attention_mode STAtten/SDT, chunk_size=2)
- MaxFormer: bic-L/MaxFormer (Max_Mixer, DWC mixers, MaxPool embeds)
- SpiLiFormer: KirinZheng/SpiLiFormer (FF_LiDiff / FB_LiDiff)
- SeqSNN: microsoft/SeqSNN (RPE-related modules)

