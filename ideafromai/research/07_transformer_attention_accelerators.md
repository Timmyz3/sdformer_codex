# ROUND 2 — Transformer / Attention Hardware Accelerators for SDformer C1*/C2* Remake

**Date:** 2026-09-05 (Asia/Shanghai)  
**Scope:** 2020–2026 · ISCA / MICRO / HPCA / ASPLOS / DAC / DATE / ISCAS / ISSCC / TCAD / FPGA / FPL + arXiv  
**Goal:** Extract **named, transferable mechanisms** for optical-flow Spikeformer (SDformer) hardware — **not** “sparsity good / attention hard” slogans.  
**C1* package:** OP-STW / PRRC / OGEC  
**C2* package:** HBG-RP / ADP-MAC / ARM-Acc / MFBD / SP-Gate  
**Hard rule:** Do **not** claim “first spiking Transformer HW” (Bishop ISCA’25, SMAM arXiv’25, FireFly-T, SpikeTA already exist). Do **not** claim generic zero-skip or “swap multiply order.”

---

## 0. How to read this note

For each **strong mechanism** below:
- **Paper + venue/year + link**
- **1-line mechanism**
- **Transfer hook** → existing C1*/C2* name **or** a **NEW named remake idea**
- **Novelty score 1–10** for an ISCAS optical-flow SNN-Transformer paper (10 = OF-specific + not saturated; ≤5 = saturated / wrong domain)

**Saturation legend:** mechanisms already widely used in ANN Transformer ASICs (cascade prune, weak-omit, score-stationary) score lower unless **re-grounded in OF residuals / ATLIF real payloads / event motion**.

---

## 1. Sparse / cascade attention HW

### M1 — SpAtten cascade token+head prune + progressive quantization
- **Paper:** Wang, Zhang, Han — *SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning* — **HPCA 2021** — https://arxiv.org/abs/2012.09852
- **Mechanism:** On-the-fly **cascade** prune of tokens/heads via cumulative attention-prob / head-magnitude scores + high-parallelism top-k; **progressive quantization** fetches LSB only if softmax max-prob is flat.
- **Transfer hook:** Direct ancestor of **SP-Gate** (attention-mass gate). New OF remake: **MSBC-Cascade** — cascade prune using *motion-saliency × attention-mass* (not NLP redundancy), and progressive bitwidth keyed by residual confidence (ties to **PRRC** budget).
- **Novelty for ISCAS OF-SNN-T:** **6/10** alone (cascade prune is saturated); **8/10** if MSBC-Cascade + OF residual confidence is the claim.

### M2 — A³ approximate candidate selection
- **Paper:** Ham et al. — *A³: Accelerating Attention Mechanisms in Neural Networks with Approximation* — **HPCA 2020** — https://arxiv.org/abs/2002.10941
- **Mechanism:** Content-based search view of attention; **greedy approximate candidate selection** of likely-relevant keys before full QK, then specialized pipeline.
- **Transfer hook:** Map candidates → **ARM-Acc** hypotheses: keep only K motion hypotheses whose coarse flow residual is within PRRC window (**Hyp-Candidate Select**). Avoid claiming “approx attention” generically.
- **Novelty:** **7/10** when fused with OP-STW wake; **4/10** as standalone approx attention.

### M3 — Sanger quantized prediction + score-stationary reconfigurable SA
- **Paper:** Lu et al. — *Sanger: A Co-Design Framework for Enabling Sparse Attention using Reconfigurable Architecture* — **MICRO 2021** — https://dl.acm.org/doi/10.1145/3466752.3480125 · often cited with arXiv companion
- **Mechanism:** Low-bit **quantized attention-mask prediction** → pack into structured blocks → **score-stationary** dataflow on reconfigurable systolic array unifying SDDMM/SpMM.
- **Transfer hook:** Strengthens **SP-Gate** + **MFBD**: predict sparse mask from coarse flow / membrane amp, then score-stationary Acc delivery by motion-bundle ID. New name: **SS-FSA-OF** (score-stationary flow-sparse attention).
- **Novelty:** **7.5/10** (score-stationary is known; OF mask predictor is not).

### M4 — ELSA lightweight filter via sign random projection
- **Paper:** Ham et al. — *ELSA: Hardware-Software Co-design for Efficient, Lightweight Self-Attention* — **ISCA 2021** — https://taejunham.github.io/data/elsa_isca21.pdf · DOI 10.1109/ISCA52012.2021.00060
- **Mechanism:** Cheap **approximate relation filter** (sign / random projection style) drops unlikely Q–K pairs before expensive attention; specialized HW for approx path.
- **Transfer hook:** Cheap prefilter for **OGEC**: unmatched/occlusion tokens skip SDSA exact path. New: **SRP-OGEC** — sign-hash occlusion/unmatch detector driving exact vs propagate.
- **Novelty:** **6.5/10** (approx filter saturated); **8/10** as occlusion prefilter for OF.

### M5 — DOTA detect-and-omit weak attentions (ASPLOS’22 electronic)
- **Paper:** Qu et al. — *DOTA: Detect and Omit Weak Attentions for Scalable Transformer Acceleration* — **ASPLOS 2022** — https://doi.org/10.1145/3503222.3507738 · NSF PDF https://par.nsf.gov/servlets/purl/10357543
- **Mechanism:** Jointly train a **lightweight Detector** with the Transformer to omit weak attention edges at runtime; **Token-Parallel** dataflow + reconfigurable matrix unit for irregular sparse maps.
- **Transfer hook:** Detector features = {event density, residual |F−F0|, occlusion flag} → omit weak edges in SDSA. New: **Flow-DOTA Detector** feeding **SP-Gate**. (Do **not** confuse with photonic DOTA arXiv.)
- **Novelty:** **8/10** if detector is OF-semantic; **5/10** if just another weak-omit.

### M6 — FACT eager correlation prediction (before QKV)
- **Paper:** Qin et al. — *FACT: FFN-Attention Co-optimized Transformer Architecture with Eager Correlation Prediction* — **ISCA 2023** — https://doi.org/10.1145/3579371.3589057
- **Mechanism:** **Eagerly predict** attention correlation **before** full QKV generation (log-domain add-only predictor); skip useless QKV work; mixed-precision FFN from predicted mass; OoO scheduler keeps predictor off critical path; diagonal storage for mixed-precision FFN.
- **Transfer hook:** Highest-leverage ANN idea for SDformer: predict attention/mass from **cheap TDE / coarse flow / ATLIF amp histogram** *before* projection MAC. New: **ECP-QKV** (Eager Correlation Prediction for QKV) — distinct from Bishop’s Error-Constrained Pruning. Ties **OP-STW** wake to projection skip, not only tile wake.
- **Novelty:** **9/10** for OF-SNN-T (eager-before-QKV + motion prior is underclaimed).

### M7 — Energon Mix-Precision Multi-Round Filtering
- **Paper:** Zhou et al. — *Energon: Towards Efficient Acceleration of Transformers Using Dynamic Sparse Attention* — **TCAD 2022** (arXiv 2021) — https://arxiv.org/abs/2110.09310
- **Mechanism:** **MP-MRF**: multi-round low-bit filtering of Q–K pairs, then high-precision only on survivors; Filtering Unit + Attention Unit co-processor.
- **Transfer hook:** Multi-round filter stages = **PRRC pyramid levels** (coarse INT2/4 filter → fine residual INT8/ATLIF payload). New: **Pyramid-MRF**.
- **Novelty:** **7.5/10**.

### M8 — AccelTran / DynaTran runtime activation prune + tiling
- **Paper:** Tuli & Jha — *AccelTran: A Sparsity-Aware Accelerator for Dynamic Inference with Transformers* — arXiv 2023 — https://arxiv.org/abs/2302.14705
- **Mechanism:** **DynaTran** low-overhead runtime activation pruning across layers + diverse tiled dataflows for reuse.
- **Transfer hook:** Runtime prune = **HBG-RP gate** decisions at SDSA/MLP boundaries; tile dataflows inform **ADP-MAC** bank mapping.
- **Novelty:** **5.5/10** (general dynamic prune — saturated unless OF metrics).

---

## 2. Spike / spike-driven attention HW (closest cousins — claim carefully)

### M9 — Bishop TTB + BSA + ECP + AAC attention core
- **Paper:** Xu et al. — *Bishop: Sparsified Bundling Spiking Transformers on Heterogeneous Cores with Error-Constrained Pruning* — **ISCA 2025** — https://arxiv.org/abs/2505.12281 · DOI 10.1145/3695053.3731063
- **Mechanism:** **Token-Time Bundle (TTB)** packing; density **stratifier** → dense vs sparse cores; **BSA** training for structured bundle sparsity; **Error-Constrained TTB Pruning (ECP)** on Q/K with binary-score error bound; multiplier-less **AND-Accumulate (AAC)** attention core; score-stationary-ish S registers.
- **Transfer hook:**
  - TTB ↔ **MFBD** (motion-bundle delivery) — but redefine bundle key as **(tile, Δt, hyp_id)** not generic token×time.
  - ECP ↔ strengthen **SP-Gate** with **error-bounded** prune (not heuristic mass).
  - Stratifier ↔ **HBG-RP** dense/sparse path split on `|amp|` density.
  - **DO NOT CLAIM** first spiking Transformer accelerator.
  - New OF remake: **Motion-TTB** + **OF-ECP** (error bound from AEE / residual energy, not classification CE).
- **Novelty:** **4/10** for “spiking Transformer HW”; **8.5/10** for **Motion-TTB / OF-ECP** remake.

### M10 — SMAM dual-spike Mask-Add for Spike-driven Transformer
- **Paper:** Li et al. — *An Efficient Sparse Hardware Accelerator for Spike-Driven Transformer* — arXiv 2025 — https://arxiv.org/abs/2501.07825
- **Mechanism:** Position-encoded spike skip; **Spike Mask Add Module (SMAM)** for **dual-spike** SDSA (Hadamard Qs⊙Ks → accumulate → mask Vs) — unlike single-spike CNN SNN accelerators.
- **Transfer hook:** SMAM is the right *primitive* family for binary SDSA; your differentiator is **HBG-RP** `{gate, real payload}` on V/linear path and OF gates. New: **SMAM-RP** — Mask-Add on spike gates + real ATLIF payload MAC only when mask=1 (bridges SMAM ↔ ADP-MAC).
- **Novelty:** **6/10** (spike SDSA HW exists); **8.5/10** for SMAM-RP with irreducible ATLIF payload.

---

## 3. Vision Transformer / token-pruning HW

### M11 — HeatViT adaptive token prune on embedded FPGA
- **Paper:** Dong et al. — *HeatViT: Hardware-Efficient Adaptive Token Pruning for Vision Transformers* — **HPCA 2023** — https://arxiv.org/abs/2211.08110 · DOI 10.1109/HPCA56546.2023.10071047
- **Mechanism:** Learnable **head-evaluation token selector** inserted before blocks; image-adaptive prune/consolidate non-informative tokens; reuse backbone HW with tiny control; INT8 + polynomial approx of nonlinearities.
- **Transfer hook:** Token selector features ← event count / residual / occlusion — feeds **OP-STW** wake bitmap and **PRRC** ROI. New: **HeatFlow-Tok** — progressive selectors across OF pyramid levels.
- **Novelty:** **8/10** (ViT prune HW known; event/residual selector for dense OF is not).

### M12 — ViTCoD prune-and-polarize + denser/sparser engines
- **Paper:** You et al. — *ViTCoD: Vision Transformer Acceleration via Dedicated Algorithm and Accelerator Co-Design* — **HPCA 2023** — DOI 10.1109/HPCA56546.2023.10071027 · code https://github.com/GATECH-EIC/ViTCoD
- **Mechanism:** Prune and **polarize** attention maps into enforced **denser vs sparser fixed patterns**; dual engines; lightweight auto-encoder trades DRAM moves for compute.
- **Transfer hook:** Polarize by **motion coherence**: smooth flow regions → sparse fixed pattern; occlusion/boundaries → dense. Dual engines ↔ Bishop-style stratifier / HBG dense-sparse. New: **Polar-Flow-Attn**.
- **Novelty:** **8/10**.

### M13 — Auto-ViT-Acc mixed-scheme quantization FPGA framework
- **Paper:** Li et al. — *Auto-ViT-Acc: An FPGA-Aware Automatic Acceleration Framework for Vision Transformer with Mixed-Scheme Quantization* — **FPL 2022** — DOI 10.1109/FPL57034.2022.00027
- **Mechanism:** Automated FPGA mapping with **mixed-scheme** (fixed-point + power-of-two) quantization of ViT FC layers.
- **Transfer hook:** Quant schemes for **ADP-MAC** bit lanes and progressive MSB/LSB (SpAtten-style) on ATLIF payloads — not a headline claim alone.
- **Novelty:** **4/10** as main contribution; **useful engineering** for C2* bitwidth search.

---

## 4. Efficient attention algorithms → what HW actually did

### M14 — FlashAttention / FA-2 / FA-3 (GPU kernels; HW lessons)
- **Papers:**
  - Dao et al. — *FlashAttention* — **NeurIPS 2022** — https://arxiv.org/abs/2205.14135
  - Dao — *FlashAttention-2* — 2023 — https://arxiv.org/abs/2307.08691
  - Shah / Dao et al. — *FlashAttention-3* — **NeurIPS 2024** — https://arxiv.org/abs/2407.08608
- **Mechanism (algorithm):** IO-aware **tiling**; never materialize full N×N; online softmax; FA-2 better work partition; FA-3 warp-specialize + TMA async + FP8 block quant on Hopper.
- **What HW actually did:** Mostly **GPU software kernels** exploiting HBM↔SRAM hierarchy — **not** a new ASIC microarchitecture family. ASIC follow-ons exist (below).
- **Transfer hook:** For edge SDformer, “Flash” lesson = **fuse SDSA score+mask+V in SRAM tiles**; avoid writing full attention maps. New: **Flash-SDSA Tile** — online accumulate with spike masks, tile size = PRRC residual window.
- **Novelty:** **5/10** claiming FlashAttention HW; **7.5/10** claiming Flash-SDSA fused tile for OF windows.

### M15 — H-FA hybrid float/log FlashAttention ASIC
- **Paper:** *H-FA: A Hybrid Floating-Point and Logarithmic Approach to Hardware Accelerated FlashAttention* — arXiv 2025 — https://arxiv.org/abs/2511.00295
- **Mechanism:** ASIC dataflow for FlashAttention-2-style blocked attention; **hybrid BF16 + fixed-point log** in selected stages → ~22–27% area/power vs pure FP FA-2 datapath at iso latency (28nm).
- **Transfer hook:** Log/hybrid path for attention **scores**; keep **ATLIF real payload** on V path in linear domain (**HBG-RP** dual rail). New: **Hybrid-Score / Linear-Payload**.
- **Novelty:** **7/10**.

### M16 — Linear / butterfly / Performer-style HW (FABNet)
- **Paper:** Fan et al. — *Adaptable Butterfly Accelerator for Attention-based NNs via Hardware and Algorithm Co-design* (FABNet) — **MICRO 2022** — https://arxiv.org/abs/2209.09570
- **Mechanism:** Unified **butterfly** sparsity approximates attention *and* FFN; runtime-reconfigurable butterfly engine shared with FFT-style token mixing (FNet-like).
- **Transfer hook:** Optional **linear-attn backend** for coarse OF pyramid level (global matching / GMFlow-style), while fine residual uses sparse SDSA. New: **Butterfly-Coarse / SDSA-Fine** dual path with **OGEC** choosing path.
- **Novelty:** **7/10** (linear attn HW rare but not OF-specific).
- **Note:** True **Performer / Nyströmformer dedicated ASICs** are scarce; most “linear attention HW” is GPU kernels (FlashLinearAttention, GLA) or FPGA persistent-state decode (Gated DeltaNet FPGA arXiv 2603.05931 — decode/serving oriented). Treat Performer/Nyström as **algorithm options**, not saturated ASIC claims.

### M17 — PagedAttention (serving) — relevance note
- **Paper:** Kwon et al. — *Efficient Memory Management for Large Language Model Serving with PagedAttention* — **SOSP 2023** — https://arxiv.org/abs/2309.06180 · DOI 10.1145/3600006.3613165
- **Mechanism:** OS-paging-style **non-contiguous KV cache blocks** for multi-request LLM serving (vLLM).
- **Transfer hook:** **Mostly irrelevant** to edge optical-flow Spikeformer inference (fixed short T, no multi-tenant KV growth). Weak analogy only: fragmented event-token buffers — do **not** cite as core OF mechanism.
- **Novelty for ISCAS OF:** **2/10**.

---

## 5. Sparseloop-related (methodology, not a PE)

### M18 — Sparseloop taxonomy: format / gating / skipping
- **Paper:** Wu et al. — *Sparseloop: An Analytical Approach To Sparse Tensor Accelerator Modeling* — **MICRO 2022** — https://arxiv.org/abs/2205.05826 · DOI 10.1109/MICRO56248.2022.00096
- **Mechanism:** Analytical model + taxonomy of sparse accel features: **representation format**, **gating**, **skipping**; stochastic density models; Timeloop extension.
- **Transfer hook:** Use as **evaluation language** for C1*/C2* ablations (gate vs skip vs format on ATLIF `{g,p}` and Motion-TTB). Not a mechanism claim.
- **Novelty:** N/A as contribution; **mandatory** for honest design-space paper writing.

---

## 6. Spatio-temporal / video transformer HW

### M19 — FlightVGM spatial-temporal online sparsification (FPGA)
- **Paper:** Liu et al. — *FlightVGM: Efficient Video Generation Model Inference with Online Sparsification and Hybrid Precision on FPGAs* — **FPGA 2025** — DOI 10.1145/3706628.3708864
- **Mechanism:** **Spatial-temporal online activation sparsification** exploiting frame similarity; hybrid FP attention / fixed linear; dynamic-static scheduling for online compression.
- **Transfer hook:** Online ST sparsify ↔ **OP-STW** + temporal skip (**VGTS** in synthesis note). New: **ST-Online-Wake** for event/video OF tokens.
- **Novelty:** **8/10** (video-gen domain, but ST online sparsify transfers cleanly to OF).

### M20 — PARO pattern-aware reorder for 3D full attention (video DiT)
- **Paper:** PARO — pattern-aware **reorder** unifying diverse 3D attention patterns into block-diagonal + mixed-precision PE — Tsinghua / Infinigence (preprint PDF circulating 2025; video generation accelerator)
- **Mechanism:** Reorder 3D attention patterns → unified **block-diagonal**; output-bitwidth-aware mixed-precision PE array.
- **Transfer hook:** Reorder tokens by **epipolar / motion trajectory** before SDSA so Acc contexts = trajectories (**MFBD** / **ARM-Acc**). New: **Trajectory-Reorder Attn**.
- **Novelty:** **8.5/10** if motion-trajectory reorder is demonstrated on event OF.

### M21 — Kaleido latent-space channel reuse for video DiT (algo–HW)
- **Paper:** *Kaleido: Algorithm-Hardware Co-Design for Video Diffusion Transformers by Exploiting Latent Space Correlations* — arXiv (2025/2026 listing) — search title on arXiv
- **Mechanism:** Channel-wise spatiotemporal **partial-result reuse** in latent space; reconfigurable PE for irregular reuse sparsity.
- **Transfer hook:** Reuse partial SDSA/MLP results across Δt when flow warp says tokens align — ties **ReMem-Tok** / membrane reuse.
- **Novelty:** **7.5/10**.

---

## 7. Ranked remake ideas (named) for ISCAS claim writing

| Rank | Named idea | From mechanisms | Hooks C1*/C2* | Suggested novelty |
|---:|---|---|---|---:|
| 1 | **ECP-QKV** (Eager Correlation Prediction before projection) | FACT M6 | OP-STW, SP-Gate | 9 |
| 2 | **Motion-TTB + OF-ECP** | Bishop M9 | MFBD, SP-Gate, HBG-RP | 8.5 |
| 3 | **SMAM-RP** (Mask-Add gate + real ATLIF payload) | SMAM M10 | HBG-RP, ADP-MAC | 8.5 |
| 4 | **Trajectory-Reorder Attn** | PARO M20 | MFBD, ARM-Acc | 8.5 |
| 5 | **HeatFlow-Tok / MSBC-Cascade** | HeatViT M11 + SpAtten M1 | OP-STW, PRRC, SP-Gate | 8 |
| 6 | **Polar-Flow-Attn** | ViTCoD M12 | HBG stratifier, OGEC | 8 |
| 7 | **Pyramid-MRF** | Energon M7 | PRRC | 7.5 |
| 8 | **Flash-SDSA Tile** | FlashAttn M14 | PRRC window, SP-Gate | 7.5 |
| 9 | **SS-FSA-OF** | Sanger M3 | SP-Gate, MFBD | 7.5 |
| 10 | **Flow-DOTA Detector** | DOTA M5 | OGEC, SP-Gate | 8 |
| 11 | **SRP-OGEC** | ELSA M4 | OGEC | 8 |
| 12 | **Hyp-Candidate Select** | A³ M2 | ARM-Acc, OP-STW | 7 |
| 13 | **Butterfly-Coarse / SDSA-Fine** | FABNet M16 | OGEC, PRRC | 7 |
| 14 | **Hybrid-Score / Linear-Payload** | H-FA M15 | HBG-RP | 7 |

**Saturated / avoid as headline:** plain cascade token prune; plain weak-omit; plain score-stationary; plain zero-skip; “first spike Transformer ASIC”; PagedAttention; generic N:M.

---

## 8. How this upgrades C1* / C2* packages

### C1* (OP-STW / PRRC / OGEC) — upgrades

| C1* block | Upgrade from this survey | Concrete change |
|---|---|---|
| **OP-STW** | HeatViT selectors, FlightVGM ST online sparsify, FACT eager predict | Wake bitmap ← learnable/heuristic **token selector** on residual+events; optionally skip **projection** not only tile MAC (**ECP-QKV**) |
| **PRRC** | Energon multi-round filter, Flash tile windows, SpAtten progressive bits | Pyramid levels = **filter rounds** with rising bitwidth; exact capture ROI = **Flash-SDSA tile** |
| **OGEC** | ELSA filter, DOTA detector, ViTCoD polarize | Unmatched → sparse/propagate engine; matched → dense exact; detector trained with OF consistency |

**C1* claim upgrade sentence:**  
“Exact-product capacity is scheduled by **eager motion-correlation prediction** and **occlusion-aware detect-and-omit**, not by zero-skip or flat tile quotas.”

### C2* (HBG-RP / ADP-MAC / ARM / MFBD / SP-Gate) — upgrades

| C2* block | Upgrade | Concrete change |
|---|---|---|
| **HBG-RP** | Bishop stratifier, SMAM dual-spike, H-FA hybrid | `{g,p}`: g drives AAC/Mask-Add; p drives real MAC; density stratify dense/sparse cores |
| **ADP-MAC** | Auto-ViT-Acc mixed scheme, Sanger pack blocks | Dual-side bit skip + donation; **no multiply-order swap narrative** |
| **ARM-Acc** | A³ candidates, PARO reorder | Acc contexts = **motion hypotheses / trajectories** after trajectory-reorder |
| **MFBD** | Bishop TTB, VideoFlow/TMA prior | Bundle = Motion-TTB `(tile,Δt,hyp)`; weight reuse intra/inter-bundle |
| **SP-Gate** | SpAtten mass, Bishop ECP, Sanger score-stationary | Gate = attention-mass **with error bound**; score-stationary issue |

**C2* claim upgrade sentence:**  
“C2* is **dual-rail spike-gate + real ATLIF payload** with **error-constrained motion-bundle attention**, not TSBG multiply reordering.”

### Combined killer story (ISCAS-sized)

1. Frontend **ECP-QKV + OP-STW + OGEC** decides *what* enters SDSA.  
2. Midend **Motion-TTB / MFBD + SP-Gate/OF-ECP** decides *which* Q/K bundles run.  
3. Backend **SMAM-RP + ADP-MAC** executes Mask-Add on gates and real MAC on payloads.  
4. Ablate against: always-on, zero-skip, Bishop-style binary-only AAC, old C1/C2.

---

## 9. Citation checklist (real links only)

| Topic | Primary cite |
|---|---|
| SpAtten | HPCA’21 https://arxiv.org/abs/2012.09852 |
| A³ | HPCA’20 https://arxiv.org/abs/2002.10941 |
| Sanger | MICRO’21 https://doi.org/10.1145/3466752.3480125 |
| ELSA | ISCA’21 https://taejunham.github.io/data/elsa_isca21.pdf |
| DOTA (weak omit) | ASPLOS’22 https://doi.org/10.1145/3503222.3507738 |
| FACT | ISCA’23 https://doi.org/10.1145/3579371.3589057 |
| Energon | TCAD’22 https://arxiv.org/abs/2110.09310 |
| AccelTran | https://arxiv.org/abs/2302.14705 |
| HeatViT | HPCA’23 https://arxiv.org/abs/2211.08110 |
| ViTCoD | HPCA’23 DOI 10.1109/HPCA56546.2023.10071027 |
| Auto-ViT-Acc | FPL’22 DOI 10.1109/FPL57034.2022.00027 |
| Bishop | ISCA’25 https://arxiv.org/abs/2505.12281 |
| SMAM Spike-driven HW | https://arxiv.org/abs/2501.07825 |
| FlashAttention | NeurIPS’22 https://arxiv.org/abs/2205.14135 |
| FlashAttention-2 | https://arxiv.org/abs/2307.08691 |
| FlashAttention-3 | NeurIPS’24 https://arxiv.org/abs/2407.08608 |
| H-FA | https://arxiv.org/abs/2511.00295 |
| FABNet butterfly | MICRO’22 https://arxiv.org/abs/2209.09570 |
| Sparseloop | MICRO’22 https://arxiv.org/abs/2205.05826 |
| PagedAttention | SOSP’23 https://arxiv.org/abs/2309.06180 |
| FlightVGM | FPGA’25 DOI 10.1145/3706628.3708864 |

---

## 10. Explicit DO NOT CLAIM (reconfirmed)

- First spiking Transformer hardware (Bishop / SMAM / others).  
- Cascade token prune or weak-omit as sole novelty.  
- FlashAttention “accelerator ASIC” without hybrid/log or fused-SDSA tile story.  
- PagedAttention relevance to OF-SNN edge.  
- Generic sparsity / N:M / multiply-order swap.  
- Photonic DOTA as if it were ASPLOS DOTA.

---

*End of ROUND 2 survey. Feed into RTL sketches: prioritize ECP-QKV predictor block + Motion-TTB packer + SMAM-RP dual rail.*
