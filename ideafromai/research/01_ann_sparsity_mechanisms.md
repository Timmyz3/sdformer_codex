# ANN → SDformer Transferable Mechanisms (Focus A/B)

**Project:** SDformer / Timmyz3 — ISCAS optical-flow SNN-Transformer accelerator  
**Date:** 2026-09-05 (Asia/Shanghai)  
**Scope:** Mechanisms that transfer to optical-flow SNN-Transformer pipelines; novelty scored for ISCAS HW/SW co-design (1–10).  
**Constraint:** Prefer NEW MECHANISMS over generic sparsity slogans. C1/C2+TSBG currently feel derivative (“multiply order change”).

---

## Ranking criteria (novelty for ISCAS HW/SW co-design)

| Score | Meaning |
|------:|---------|
| 9–10 | Distinct microarch hook + OF/SNN semantic story; low prior art in ISCAS OF/SNN space |
| 7–8  | Strong transfer; clear hook; some related prior but rebrandable |
| 5–6  | Useful but crowded; must be tightly coupled to optical-flow / spike-amplitude semantics |
| ≤4   | Saturated / “everybody does sparsity” — avoid as primary claim |

---

## Top 12 ranked mechanisms

### 1. Flow-Coherent Predictive Token Wake (score **9.5**)

**Source papers**
- ERAFT: *An FPGA-based Real-Time Optical Flow Accelerator for Recurrent All-Pairs Field Transforms* — **ISCAS 2025**, DOI [10.1109/iscas56072.2025.11043529](https://doi.org/10.1109/iscas56072.2025.11043529)
- TCAS-I 2025: *An Ultra-High Performance and Scalable Optical Flow Hardware Accelerator…* — adaptive OF via **dynamic direction prediction** + reconfigurable pyramid pipeline — DOI [10.1109/tcsi.2025.3572918](https://doi.org/10.1109/tcsi.2025.3572918)
- Related: FlowAcc — **DATE 2022**, DOI [10.23919/date54114.2022.9774506](https://doi.org/10.23919/date54114.2022.9774506)

**1-line mechanism**  
Use predicted motion direction / temporal coherence to *wake only* the spatial tiles (or spike-token patches) that are expected to change, instead of dense frame or dense token scan.

**Why not “everybody does sparsity”**  
Skip decision is *semantic* (optical-flow prediction residual), not “activation == 0”. ERAFT/TCAS-I already prove direction-prediction reduces OF compute without accuracy loss; nobody has wired that predictor into an SNN-Transformer PE scheduler.

**SDformer remake name (C1)**  
**OF-Predictive Spike Tile Wake (OP-STW)** — a lightweight direction/ΔI predictor feeds a tile wake bitmap into C1’s spike dispatcher (TSBG becomes “wake bitmap + burst”, not just reorder).

**Claim it enables**  
“First optical-flow SNN-Transformer accelerator whose PE wake is driven by flow-prediction residual rather than zero-skip.”

**Experiment that proves it**
1. Middlebury / Sintel / DVS-OF pairs: measure tile wake rate vs. EPE/AEE.
2. Ablate: always-on vs OP-STW vs naive zero-spike skip.
3. Report PE active cycles, SRAM toggles, EPE Δ.

**Derivative risk:** Medium if you only cite ERAFT’s prediction block without SNN-Transformer integration. Mitigate by showing *spike-tile* scheduling + attention token mask co-generated from same predictor.

---

### 2. Amplitude-Preserving Dual-Bit / Dual-Factor Spike MAC (score **9.0**)

**Source papers**
- BBS / BitVert: *BBS: Bi-directional Bit-level Sparsity for Deep Learning Acceleration* — **MICRO 2024**, arXiv [2409.05227](https://arxiv.org/abs/2409.05227)
- BitParticle: *Partializing Sparse Dual-Factors…* — arXiv [2507.09780](https://arxiv.org/pdf/2507.09780) (2025)
- Dyn-Bitpool: two-sided sparse CIM — **TCAS-I 2025**, DOI [10.1109/tcsi.2025.3547001](https://doi.org/10.1109/tcsi.2025.3547001)
- BRIM: dual-sided bit-serial workload balance — arXiv [2607.19431](https://arxiv.org/html/2607.19431)

**1-line mechanism**  
Exploit bit-level sparsity on *both* operands (weights + multi-bit membrane / ATLIF amplitude), with load-balance (slot donation / quasi-sync), not binary spike AND-accumulate only.

**Why not saturated**  
Most SNN accelerators assume binary spikes → AAC. Seed gap: real-valued ATLIF / multi-bit train–binary infer is under-served. Dual-sided bit sparsity + imbalance control is ANN-frontier and maps cleanly to multi-bit Vmem × W.

**SDformer remake name (C2)**  
**ATLIF Dual-Bit Particle MAC (ADP-MAC)** — rebrand C2 away from “multiply order” into *particlized dual-factor bit lanes* with pairwise slot donation when spike amplitude bit-density is skewed across OF patches.

**Claim**  
“C2 is not multiply reordering: it is a dual-sided bit-particle datapath that keeps ATLIF amplitude while skipping zero bits on both W and Vmem.”

**Experiment**
1. Iso-area PE: binary AAC vs ADP-MAC on Quantized Spike-driven Transformer / IE-LIF models.
2. Accuracy vs bitwidth (2/4/8) and PE utilization under OF scenes with sparse vs textured regions.
3. Show utilization >90% with slot donation (BRIM-style) vs without.

**Derivative risk:** High if paper reads as “we do bit-serial.” Must lead with *amplitude-preserving OF-SNN* + imbalance microarch.

---

### 3. Cascade Token/Head Pruning → Spatio-Temporal Bundle Gate (score **8.5**)

**Source papers**
- SpAtten: *Efficient Sparse Attention Architecture with Cascade Token and Head Pruning* — **HPCA 2021**, arXiv [2012.09852](https://arxiv.org/abs/2012.09852), DOI via HPCA
- HeatViT: *Hardware-Efficient Adaptive Token Pruning for Vision Transformers* — **HPCA 2023**, DOI [10.1109/HPCA56546.2023.10071047](https://doi.org/10.1109/HPCA56546.2023.10071047)
- Bishop: *Sparsified Bundling Spiking Transformers… Error-Constrained Pruning* — **ISCA 2025**, DOI [10.1145/3695053.3731063](https://doi.org/10.1145/3695053.3731063), arXiv [2505.12281](https://arxiv.org/html/2505.12281)

**1-line mechanism**  
On-the-fly cascade prune of tokens/heads (or spike *bundles* of token×time), with top-k engine and progressive / error-constrained confidence — not static weight prune.

**Why not saturated**  
Cascade token prune ≠ unstructured zero-skip. Bishop already moves this into spiking transformers with S-stationary AAC/SAC; OF gives a *motion-saliency prior* SpAtten never had.

**SDformer remake name (C1)**  
**Motion-Salient Bundle Cascade (MSBC)** — token importance = attention score ⊗ optical-flow magnitude / event rate; cascade prune heads then spatial tokens then time-bundles.

**Claim**  
“Cascade pruning driven by optical-flow saliency, implemented as hardware top-k over spike-bundles.”

**Experiment**
1. Compare SpAtten-style attention-only prune vs MSBC (attention + flow magnitude).
2. End-to-end EPE vs pruned FLOPs/SOP; top-k engine throughput vs software sort.
3. Show progressive MSB-first attention (SpAtten) on QK for OF scenes.

**Derivative risk:** Medium–high vs Bishop/SpAtten. Differentiate with *flow-saliency score* and OF metrics (EPE), not GLUE/ImageNet only.

---

### 4. Cross-Stage Predictive Sparse Attention (DLZS-style) for Spike Attention (score **8.5**)

**Source papers**
- SOFA: *A Compute-Memory Optimized Sparsity Accelerator via Cross-Stage Coordinated Tiling* — arXiv [2407.10416](https://arxiv.org/abs/2407.10416) (2024; compare-to SpAtten/FACT/Energon)
- Complementary: DOTA (ASPLOS 2022), Sanger (MICRO 2021), FACT (ISCA 2023)

**1-line mechanism**  
Cheap *log/leading-zero* prediction of attention importance → tiled top-k → only then formal QK/V; cross-stage tiling feeds sort order into FlashAttention-like update to cut Exp/compare.

**Why distinctive**  
Not “skip zeros”: it *predicts* which Q–K pairs matter with multiplier-free DLZS, then coordinates precompute / top-k / formal stages. Transfer: spike-driven attention often already approximate; DLZS maps to spike-popcount / LZC on multi-bit envelopes.

**SDformer remake name (C1+C2 joint)**  
**Spike-Envelope Predictive Attention (SEPA)** — LZC/popcount on ATLIF envelope predicts SDSA importance; SU-FA-like sorted update on surviving pairs; TSBG becomes cross-stage tile scheduler.

**Claim**  
“First SNN-Transformer OF accelerator with multiplier-free envelope prediction + sorted-updating spike attention, reducing both SOP and membrane SRAM traffic.”

**Experiment**
1. Prediction hit-rate vs EPE degradation (k = 10–40%).
2. Compare formal-only sparse attention vs SEPA end-to-end latency/energy.
3. Ablate sorted-update vs vanilla FlashAttention-style tiling on spike softmax / IAND attention.

**Derivative risk:** Medium if SOFA is cited as ANN-only clone. Rebrand around *spike envelope* + OF token grids.

---

### 5. Hierarchical Structured Sparsity with Runtime Reshape (score **8.0**)

**Source papers**
- HighLight: *Efficient and Flexible DNN Acceleration with Hierarchical Structured Sparsity* — **MICRO 2023**, arXiv [2305.12718](https://arxiv.org/abs/2305.12718), DOI [10.1145/3613424.3623786](https://doi.org/10.1145/3613424.3623786)
- Related N:M HW: MoE-OPU expert-aware N:M — **ICCAD 2025**, DOI [10.1109/iccad66269.2025.11240807](https://doi.org/10.1109/iccad66269.2025.11240807)

**1-line mechanism**  
Compose sparsity as hierarchy of simple patterns (e.g., C1 2:4 over C0 dense / alternating dense-sparse ranks) so HW only supports cheap intersection SAFs, while runtime can reshape pattern per layer/pyramid level.

**Why not saturated**  
Structured N:M alone is common; *hierarchical composition + modular skip-at-rank* is the HighLight insight. For OF pyramids, each pyramid level can select a different HSS pattern (coarse = denser, fine = sparser).

**SDformer remake name (C2)**  
**Pyramid-Rank HSS Dataflow (PR-HSS)** — C2 PE array supports 2-rank HSS; OF pyramid / transformer block selects pattern via small CSRAM config word (runtime reshape, not retrain-only).

**Claim**  
“Runtime-reshapable hierarchical sparsity matched to optical-flow pyramid levels on one PE fabric.”

**Experiment**
1. Pareto: EPE vs energy for fixed 2:4 vs PR-HSS per pyramid level.
2. Show SAF overhead vs unstructured sparse engine.
3. Iso-accuracy EDP vs dense and vs single-pattern N:M.

**Derivative risk:** Medium. Avoid claiming “we invent N:M”; claim *pyramid-conditioned HSS reshape*.

---

### 6. Pre-Gated Expert / Head Routing for Temporal Blocks (score **8.0**)

**Source papers**
- Pre-gated MoE: *Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference* — **ISCA 2024**, [Microsoft PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/05/isca24_pregated_moe_camera_ready.pdf)
- UbiMoE: MoE-ViT FPGA — **ISCAS 2025**, arXiv [2502.05602](https://arxiv.org/abs/2502.05602), DOI [10.1109/iscas56072.2025.11043956](https://doi.org/10.1109/iscas56072.2025.11043956)
- Edge-MoE: **ICCAD 2023**, DOI [10.1109/iccad57390.2023.10323651](https://doi.org/10.1109/iccad57390.2023.10323651)
- MoE-OPU: hot-expert prediction — **ICCAD 2025**

**1-line mechanism**  
Decide *next* expert / FFN / attention-head set *before* current block finishes (pre-gate), overlapping weight fetch with compute; or predict “hot” experts from history + hidden state.

**Why distinctive**  
Routing HW is about *prefetch & overlap*, not sparsity slogans. For SDformer: treat temporal transformer blocks / direction-specialized subnets / pyramid experts as MoE; pre-gate from previous frame’s flow field.

**SDformer remake name (C1)**  
**Flow-Pregated Temporal Experts (FPTE)** — experts = {coarse-flow, fine-flow, occlusion, static}; gate from t−1 flow + spike rate; prefetch expert weights into C1 weight buffer during current block.

**Claim**  
“Frame-to-frame pre-gated expert routing eliminates expert-fetch bubbles in OF SNN-Transformer.”

**Experiment**
1. Measure HBM/SRAM stall cycles with/without pre-gate.
2. Expert hit-rate of predictor (target >80% like MoE-OPU).
3. EPE vs number of active experts K.

**Derivative risk:** Medium–high for pure MoE copy. Tie experts to *OF semantic roles*.

---

### 7. Progressive / Value-Aware Bit Truncation with Confidence Recompute (score **7.5**)

**Source papers**
- SpAtten progressive quantization (MSB first, LSB recompute if low confidence) — **HPCA 2021**
- GOBO: outlier-aware 3–4b weights — **MICRO 2020**, arXiv [2005.03842](https://arxiv.org/abs/2005.03842)
- OliVe: outlier–victim pair quantization — **ISCA 2023**, arXiv [2304.07493](https://arxiv.org/abs/2304.07493), DOI [10.1145/3579371.3589038](https://doi.org/10.1145/3579371.3589038)
- ANT: adaptive numerical type — **MICRO 2022**
- ISSCC 2023 Tambe: entropy-based early exit + mixed-precision predication — [Columbia PDF](https://sld.cs.columbia.edu/pubs/tambe_isscc23.pdf)

**1-line mechanism**  
Fetch/compute MSBs (or normal values); only bring LSBs / outliers / full precision when confidence or local importance is low — *value-aware* bit skip, not zero skip.

**Why distinctive**  
Bit truncation is common; *confidence-triggered recompute* + *outlier–victim pairing* are distinctive microarch hooks. For OF: high-gradient / occlusion pixels keep full bits; smooth regions truncate.

**SDformer remake name (C2)**  
**Flow-Confidence Progressive Precision (FC-PP)** — per-tile confidence from flow residual / spike entropy; MSB-first MAC; LSB refill + OVP encoding for amplitude outliers in ATLIF.

**Claim**  
“Mixed-precision SNN-Transformer datapath gated by optical-flow confidence, not static layer bitwidth.”

**Experiment**
1. Bit traffic vs EPE for static INT4 vs FC-PP.
2. Fraction of tiles needing LSB refill.
3. Compare GOBO/OliVe-style encoding area vs baseline.

**Derivative risk:** Medium. Must not claim “we invent mixed precision.” Lead with *flow-confidence gate*.

---

### 8. FlashAttention-style Fusion / Einsum Dataflow with Semantic Stationarity Twist (score **7.5**)

**Source papers**
- FLAT: *An Optimized Dataflow for Mitigating Attention Bottlenecks* — **ASPLOS 2023**, DOI [10.1145/3575693.3575747](https://doi.org/10.1145/3575693.3575747), arXiv [2107.06419](https://arxiv.org/abs/2107.06419)
- FuseMax: *Leveraging Extended Einsums to Optimize Attention Accelerator Design* — **MICRO 2024**, DOI [10.1109/micro61859.2024.00107](https://doi.org/10.1109/micro61859.2024.00107), arXiv [2406.10491](https://arxiv.org/abs/2406.10491)
- 3D-Flow / 3D-FlashAttention: register-to-register vertical fusion — **DATE 2026**, arXiv [2602.11016](https://arxiv.org/abs/2602.11016)

**1-line mechanism**  
Fuse QK→softmax→AV (or spike-attention equivalents) so attention footprint grows linearly; map via Einsum cascades for near-100% PE util; optional S-stationary / spike-stationary hybrid.

**Why not “multiply order”**  
FLAT/FuseMax change *operator fusion and tiling semantics*, not PE multiply scheduling trivia. Bishop’s S-stationary for wide attention scores is the SNN-native twist. C2 should be sold as **score-stationary spike attention**, not “we swapped A×B order.”

**SDformer remake name (C2)**  
**Score-Stationary Fused Spike Attention (SS-FSA)** — keep multi-bit S (attention / IAND accum) in PE regs; stream binary Q/K or spike bundles; fuse with membrane update to avoid SRAM round-trips (ASTER-like hybrid WS/SS).

**Claim**  
“C2 implements score-stationary fused spike-attention dataflow that eliminates intermediate A-matrix materialization for OF tokens.”

**Experiment**
1. SRAM energy breakdown: fused vs unfused SDSA.
2. Utilization of AAC/SAC arrays vs FLAT-style baseline.
3. Sequence / spatial-token length scaling (linear vs quadratic buffer).

**Derivative risk:** High if abstract says “novel dataflow” without naming fusion invariant. Use FuseMax-style Einsum diagram in paper.

---

### 9. Parallel Tick-Batching / Membrane-Unrolled Temporal Dataflow (score **7.5**)

**Source papers**
- Parallel Time Batching — **HPCA 2022**, DOI via IEEE HPCA 2022 (Lee, Zhang, Li)
- ISCAS 2025: *Hardware Efficient Accelerator for Spiking Transformer With Reconfigurable Parallel Time Step Computing* — DOI [10.1109/iscas56072.2025.11043330](https://doi.org/10.1109/iscas56072.2025.11043330), arXiv [2503.19643](https://arxiv.org/abs/2503.19643)
- SpinalFlow — **ISCA 2020**
- ASTER: spike-/weight-stationary hybrid PIM for spiking transformers — arXiv [2511.06770](https://arxiv.org/html/2511.06770v1)

**1-line mechanism**  
Unroll LIF across timesteps in parallel (tick-batching) to remove membrane SRAM and amortize weight fetch once per layer across T.

**Why transferable**  
Direct SNN-Transformer fit; OF streams are naturally temporal. Novelty for SDformer = combine tick-batching with multi-bit ATLIF (not only binary) + OF tile wake.

**SDformer remake name (C1)**  
**Amplitude-Aware Parallel Tick Batch (AAPT)** — parallel T lanes share weights; each lane carries ATLIF amplitude; inactive OF tiles clock-gate entire tick columns.

**Claim**  
“Membrane-buffer-free multi-bit tick-batching co-gated by optical-flow tile wake.”

**Experiment**
1. Area: membrane SRAM vs unrolled PE (binary vs multi-bit).
2. Energy/SOP vs serial tick-batching (SpinalFlow-style).
3. Latency for T=4/8 on OF sequences.

**Derivative risk:** High vs ISCAS 2025 spiking transformer paper. Differentiate with *multi-bit amplitude + OF wake*, and optical-flow task (not CIFAR only).

---

### 10. Control-Variate / Self-Healing Approximate MAC with Analytic Error Bounds (score **7.0**)

**Source papers**
- Control Variate Approximation for DNN Accelerators — **DAC 2021**, DOI [10.1109/dac18074.2021.9586092](https://doi.org/10.1109/dac18074.2021.9586092), arXiv [2102.09642](https://arxiv.org/abs/2102.09642)
- Extended control variate — arXiv [2412.16757](https://arxiv.org/pdf/2412.16757)
- MACISH: Internal-Self-Healing approximate MAC — IEEE Access 2019, DOI [10.1109/access.2019.2920335](https://doi.org/10.1109/access.2019.2920335)

**1-line mechanism**  
Use aggressive approximate multipliers but cancel mean error at accumulate (control variate / internal self-healing) with *analytic* error model — no heavy retrain.

**Why distinctive**  
Approximate MAC is old; *runtime error nulling with bounds* is the claimable hook. For SNN-OF: approximate only low-saliency tiles; keep exact MAC on occlusion / high-flow tiles.

**SDformer remake name (C2)**  
**Saliency-Gated Self-Healing MAC (SG-SHMAC)** — PE has exact / approx modes; control-variate column corrects mean error; mode bit from OP-STW / FC-PP.

**Claim**  
“Approximate MAC with proven mean-error cancellation, gated by optical-flow saliency.”

**Experiment**
1. Power vs EPE for exact / approx / SG-SHMAC.
2. Measure mean/variance of convolution or SDSA error with/without control variate.
3. No-retrain vs retrain baselines.

**Derivative risk:** Medium. ISCAS reviewers know approx arithmetic — couple tightly to *saliency gating* and OF accuracy.

---

### 11. Residual / Skip-Path Membrane Reuse (IAND / Potential Shortcut) (score **6.5**)

**Source papers**
- ISCAS 2025 Spiking Transformer accelerator: residual as **IAND** to keep spike-only path — arXiv [2503.19643](https://arxiv.org/abs/2503.19643)
- Spike-driven Transformer / Spikingformer residual-before-activation (membrane shortcuts) — CVPR/related spike-transformer line (e.g. Spike-driven Transformer V3 arXiv [2411.16061](https://arxiv.org/abs/2411.16061))
- ASTER residual/spike reuse patterns — arXiv [2511.06770](https://arxiv.org/html/2511.06770v1)

**1-line mechanism**  
Keep residual on membrane potential (or IAND spike residual) so datapath never leaves spike/membrane domain; HW reuses adders for residual merge without FP convert.

**Why still useful**  
Not sparsity — *domain closure* for energy. For OF: residual across recurrent RAFT-like / temporal transformer iterations can reuse on-chip membrane scratchpad.

**SDformer remake name (C1)**  
**Recurrent Membrane Skip Port (RMSP)** — dedicated residual port on PE merging previous iteration’s membrane with current spike-attention out; zero off-chip residual traffic.

**Claim**  
“On-chip recurrent membrane skip eliminates residual DRAM for multi-iteration OF refinement.”

**Experiment**
1. Traffic: FP residual vs RMSP for N refinement iters.
2. Accuracy of IAND vs add residual under OF.
3. Area of merge port vs save in SRAM ports.

**Derivative risk:** High (many spike-transformer papers). Only viable as *secondary* claim supporting C1, not headline.

---

### 12. Event-Triggered / Differential Activation Tile Controller (score **6.5**)

**Source papers**
- Multiply-and-Fire (MNF) event-driven ANN accel — arXiv [2204.09797](https://arxiv.org/pdf/2204.09797)
- Flexible encoding for event-based DNNs — IEICE ELEX 2023, DOI [10.1587/elex.20.20230379](https://doi.org/10.1587/elex.20.20230379)
- Differential / predictive coding style event wake (emerging; treat carefully) — e.g. predictive surprise gating literature; prefer citing MNF + OF predictors rather than unvetted repos

**1-line mechanism**  
Fire PE work only on *events* (new spikes, Δactivation, prediction surprise), with bitmask/CSR dual encoding selected by event rate.

**Why partially distinctive**  
Zero-skip is saturated; *differential / surprise* wake + encoding mode switch is less so — especially if Δ is optical-flow residual, not ReLU(x)==0.

**SDformer remake name (C1)**  
**Differential Flow-Event Dispatcher (DFED)** — event = spike OR |flow residual|>θ; encoding switches bitmask↔CSR by local density (ELEX insight).

**Claim**  
“Dual-encoding event dispatcher driven by flow residual, not only spike occupancy.”

**Experiment**
1. Encoding energy vs event rate curves.
2. Compare MNF-style vs DFED on DVS + frame OF hybrid input.
3. False-wake rate vs EPE.

**Derivative risk:** High if framed as “event-driven sparsity.” Must stress *differential flow residual* + encoding switch.

---

## Suggested remake map for current C1 / C2

| Current feel | Remake | Primary mechanism # |
|--------------|--------|---------------------|
| C1 “generic sparse dispatch / TSBG reorder” | **OP-STW + MSBC + FPTE** (wake + cascade + pregate) | 1, 3, 6 |
| C2 “just multiply order” | **ADP-MAC + SS-FSA + FC-PP** (dual-bit particle + score-stationary fusion + confidence precision) | 2, 8, 7 |
| Shared glue | **SEPA + PR-HSS + AAPT** | 4, 5, 9 |

**Recommended ISCAS narrative (one sentence)**  
*SDformer replaces dense OF CNN pipelines with an ATLIF SNN-Transformer whose PE wake is flow-prediction-driven, whose attention is envelope-predicted and score-stationary, and whose MAC is dual-bit amplitude-preserving — not another zero-skip systolic array.*

---

## Explicit DO NOT CLAIM list (saturated)

Do **not** make these the primary novelty bullets:

1. **“We skip zero activations / zero spikes.”** — Cnvlutin, SCNN, Eyeriss-v2, Cambricon-S, every SNN paper since 2018.
2. **“We use N:M or 2:4 structured sparsity.”** — Ampere / countless MICRO–ISCA papers; OK as *substrate*, not claim.
3. **“We use mixed precision / quantization.”** — unless gated by a *new* confidence/outlier mechanism (FC-PP / OliVe-style).
4. **“Novel weight-stationary vs activation-stationary.”** — Eyeriss taxonomy era; only claim **score-/spike-stationary fusion** with Einsum/Fusion proof.
5. **“Bit-serial MAC.”** — Laconic et al.; only claim **dual-sided + load-balance** (ADP-MAC).
6. **“Token pruning for ViT.”** — HeatViT/SpAtten crowded; only with **flow-saliency cascade**.
7. **“FlashAttention on hardware.”** — FLAT/FuseMax/GPU FA; only with **spike/OF cross-stage** twist (SEPA/SS-FSA).
8. **“MoE for transformers.”** — Pre-gated MoE / UbiMoE; only with **OF-semantic experts + frame pregate**.
9. **“Approximate computing for DNN.”** — DAC lineage; only with **analytic healing + saliency gate**.
10. **“First spiking transformer accelerator.”** — ISCAS 2025 + Bishop ISCA 2025 already exist; claim *OF task + multi-bit ATLIF + flow wake* instead.
11. **“Pyramid optical flow on FPGA.”** — FlowAcc / ERAFT / TCAS-I; claim *SNN-Transformer OF*, not pyramid alone.
12. **C2 as “we changed multiply order / dataflow order.”** — reviewers will call this incremental; rebrand to SS-FSA or ADP-MAC.

---

## Focus B quick index (distinctive sparsity / skip families)

| Family | Distinctive twist | Best SDformer hook | Score |
|--------|-------------------|--------------------|------:|
| Dual-sided bit sparsity | Both W and A bit-skip + imbalance fix | ADP-MAC (C2) | 9 |
| Hierarchical structured (HSS) | Compose simple patterns; low sparsity tax | PR-HSS pyramid | 8 |
| Predictive / DLZS attention skip | Predict before formal QK | SEPA | 8.5 |
| Cascade token/head prune | Dynamic top-k, not weight prune | MSBC | 8.5 |
| Progressive / value-aware bits | MSB first + confidence LSB | FC-PP | 7.5 |
| Early-exit / entropy exit | Tambe ISSCC23 | Optional block exit on static OF regions | 6 |
| Event / differential wake | Surprise or Δ, not ==0 | DFED + OP-STW | 9.5/6.5 |
| Outlier–victim encoding | Local OVP, memory-aligned | FC-PP outlier path | 7.5 |
| Pre-gated routing | Overlap fetch with compute | FPTE | 8 |
| Tick-batch unroll | Kill membrane buffer | AAPT | 7.5 |

---

## Citation bank (copy-friendly)

- SpAtten — HPCA 2021 — https://arxiv.org/abs/2012.09852  
- HeatViT — HPCA 2023 — https://doi.org/10.1109/HPCA56546.2023.10071047  
- FLAT — ASPLOS 2023 — https://doi.org/10.1145/3575693.3575747  
- FuseMax — MICRO 2024 — https://arxiv.org/abs/2406.10491  
- HighLight — MICRO 2023 — https://arxiv.org/abs/2305.12718  
- BBS/BitVert — MICRO 2024 — https://arxiv.org/abs/2409.05227  
- GOBO — MICRO 2020 — https://arxiv.org/abs/2005.03842  
- OliVe — ISCA 2023 — https://arxiv.org/abs/2304.07493  
- Pre-gated MoE — ISCA 2024 — Microsoft Research PDF (above)  
- Bishop — ISCA 2025 — https://doi.org/10.1145/3695053.3731063  
- SOFA — arXiv 2024 — https://arxiv.org/abs/2407.10416  
- Control Variate approx — DAC 2021 — https://arxiv.org/abs/2102.09642  
- Parallel Time Batching — HPCA 2022 — Lee et al.  
- Spiking Transformer HW — ISCAS 2025 — https://arxiv.org/abs/2503.19643  
- ERAFT — ISCAS 2025 — https://doi.org/10.1109/iscas56072.2025.11043529  
- FlowAcc — DATE 2022 — https://doi.org/10.23919/date54114.2022.9774506  
- TCAS-I OF direction prediction — 2025 — https://doi.org/10.1109/tcsi.2025.3572918  
- UbiMoE — ISCAS 2025 — https://arxiv.org/abs/2502.05602  
- ASTER — arXiv 2025 — https://arxiv.org/abs/2511.06770  
- Dyn-Bitpool — TCAS-I 2025 — https://doi.org/10.1109/tcsi.2025.3547001  
- BitParticle — arXiv 2025 — https://arxiv.org/pdf/2507.09780  

---

## Next actions (for parent / Timmyz3)

1. Pick **headline trio**: OP-STW (C1) + ADP-MAC (C2) + SS-FSA (C2 narrative fix).  
2. Keep SEPA or MSBC as supporting algorithm–HW co-design section.  
3. Rewrite any draft that motivates C2 with “multiply order” → replace with dual-bit particle + score-stationary fusion.  
4. Benchmarks must include **EPE/AEE + SOP/J + PE wake rate**, not only CIFAR accuracy.

