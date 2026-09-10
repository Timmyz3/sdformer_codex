# Optical-Flow Data Properties → HW Mechanisms for SDformer (C1/C2 Remakes)

**Date:** 2026-09-05 (Asia/Shanghai)  
**Scope:** Hardware innovation axes tied to *optical-flow / event-flow data & algorithm structure*, not generic CNN/SNN accel.  
**Target paper:** SDformer / SDformerFlow-class optical-flow SNN-Transformer HW/SW co-design (ISCAS).  
**Problem statement:** Current C1 (limited-capacity exact product capture for Conv residency) and C2 (typed-signed K8 fabric + TSBG weight-row bundle reuse) are competent but feel like “specialized SNN MAC islands.” User wants **remakes with OF-native mechanisms**.

**Current baseline (from project reviews):**
- **C1:** exact product capture under finite on-chip capacity; Conv-resident tile events; ladder baselines (zero / bit / finite-1RW / ceiling).
- **C2+TSBG:** same weight-row identity, multi-context Acc24 delivery; ordinary vs TSBG schedule; FC2 continuation.
- **Algo workload:** Motion C12 ep34 (AEE ~1.20, firing ~5.7%); Motion-XOR/TTX temporal signatures exist but AEE gain is small—do **not** sell as algo novelty; use only as *data interface* for HW.

**SDformerFlow data/alg skeleton (algo paper, no HW yet):**
- Event voxel → Spiking Feature Generator + Shortcut Patch Embedding → Spatiotemporal Swin Spikeformer (3D window / shifted-window SDSA: dot-product or QK-linear) → spike decoder / multi-scale flow heads.
- Tokens live in `T×H×W` with small temporal window (`Tw≈2`) and spatial windows (`9×9` / `15×15`).
- Spike-driven attention is mask/add-friendly; membrane-potential shortcuts preserve spike I/O.
- Citation: Tian & Andrade-Cetto, *SDformerFlow*, ICPR 2024 / arXiv:2409.04082; LNCS 15315.

---

## 0. Optical-flow DATA properties (what HW can actually exploit)

| Property | What it means in OF / event-flow | Classical / DNN OF HW that already uses it | SDformer / SNN-Transformer mapping |
|---|---|---|---|
| **Spatial smoothness / locality** | Neighboring pixels share similar flow; local regularization | Horn–Schunck / LK FPGA (local stencil); FlowAcc regularization; Gong TCAS-I multi-scale LK | 3D Swin windows; skip links at same scale; flow head spatial smoothness prior |
| **Temporal coherence / warm-start** | Flow evolves slowly; prior frame is a strong prior | RAFT warm-start; ERAFT prediction skip; VideoFlow MOP | Spike membrane state across T; Motion-XOR/TTX signatures; recurrent update tokens |
| **Pyramid / coarse-to-fine** | Large motion at coarse, refine residual at fine | FlowAcc multiplexed BNN pyramid; Ultra-Flow hierarchical match; TCAS-I 2025 reconfigurable pyramid pipeline | Hierarchical STSF stages + decoder upsample; patch-merge as “pyramid” |
| **Correspondence / cost volume sparsity** | Only a few matches matter; soft-argmax peaks | RAFT all-pairs + lookup; GMFlow global matching + softmax; FlowFormer cost memory | SDSA attention mass is sparse inside windows; QK token mask |
| **Occlusion / unmatched pixels** | Forward–backward inconsistency; propagate from matched regions | GMFlow flow-propagation self-attn; AccFlow occlusion-aware accumulate | Unmatched tokens should *not* burn full SDSA; gated decode |
| **Aperture / multi-scale direction** | Local aperture ambiguous; need multi-scale or multi-point | hARMS / ARMS aperture-robust multi-scale event OF | Multi-head / multi-window direction hypotheses before Acc |
| **Event sparsity & asynchrony** | Compute only where brightness changes | hARMS event-history (no full frame); ASNA-Flow spatial locality sparse; Kraken SNE | Binary spike tokens; event-addressed PE wake-up |
| **Temporal difference / EMD priors** | Motion = Δt between spatially offset events | TDE / TDE-3; sEMD; Greatorex CVPRF 2026 timing OF | Hardware TDE front-end → tokens; ISI / spike-count velocity code |
| **Iterative residual update** | Few refinements if early motion good | RAFT ConvGRU; ERAFT early-exit prediction; TMA fewer refinements | Early-exit on membrane / flow residual; variable T or update steps |
| **Epipolar / geometric (optional)** | Stereo / ego-motion constrains flow | Automotive OF + ego; MultiCM contrast-max | Not native to SDformer; optional C3-side check, not C1/C2 remake |

---

## 1. Dedicated OF / motion HW landscape (real papers)

### 1.1 Frame / DNN optical-flow accelerators
1. **FlowAcc** — Ling, Yan, Huang, Chen. DATE 2022. Pipelined DNN OF; multiplexed BNN pyramidal features; Hamming matching; hierarchical regularization. Exploits **pyramid + binary descriptor matching**. DOI: 10.23919/DATE54114.2022.9774506.  
2. **Ultra-Flow** — Ling et al. FPL 2022. Single BNN feature map + hierarchical matching (avoid multi-level NN reuse); local regularization; up to ~688 FPS @ 640×480. Exploits **feature reuse across pyramid levels**. DOI: 10.1109/FPL57034.2022.00017.  
3. **JSA follow-on** — Yan, Ling, Huang, Chen. *J. Syst. Arch.* 136:102818, 2023. Same family, real-time high-accuracy FPGA OF.  
4. **ERAFT** — ISCAS 2025. Lightweight RAFT + **prediction mechanism** to skip/reduce iterative updates; 86 FPS @ 640×480 on VCK190. Exploits **iterative residual predictability**. DOI: 10.1109/ISCAS56072.2025.11043529.  
5. **RAFT-Lite FPGA** — Li, Gao, Su, Chen, Liu. CAC 2022. Compressed RAFT + conv scheduling / resource reuse; 10.4 FPS @ 512×396 on ZCU102. DOI: 10.1109/CAC57257.2022.10054761.  
6. **Scalable pyramid FPGA (ISCAS 2024 / TCAS-I 2025)** — Liu et al. Adaptive OF via **dynamic direction prediction**; configurable PE count; reconfigurable pyramid-layer pipeline; 405 FPS, AEE ~0.64–1.02. Exploits **direction predictability + pyramid dependency breaking**. DOI: 10.1109/ISCAS58744.2024.10557997; TCAS-I 10.1109/TCSI.2025.3572918.  
7. **Gong et al. TCAS-I 2023** — Multi-scale LK tracking accelerator for SLAM; 93 FPS @ 752×480 on Zynq. Exploits **sparse keypoints + multi-scale LK**, not dense DNN. DOI: 10.1109/TCSI.2023.3298969.  
8. **VD56G3 on-sensor OF ASIC** — STMicro characterization (arXiv:2305.13087, 2023). Integrated motion-vector unit; up to ~300 FPS reduced res. Exploits **sensor-local block matching**.  

### 1.2 Event / neuromorphic optical-flow HW
9. **hARMS** — Stumpp, Akolkar, George, Benosman. *IEEE Access* 10:58181–58198, 2022 (arXiv:2112.06772). FPGA hybrid SoC for aperture-robust multi-scale event OF; **no full event frame**—small relevant event history; latency independent of sensor resolution; ~1.21 Mevent/s.  
10. **ASNA-Flow** — Wang, Luo, Li, Zhou, Yu, Xiao. *IEEE Trans. VLSI Syst.*, 33(12):3409–3422, Dec 2025. First dedicated neuromorphic ASIC co-design for event OF; **spatial locality for sparse compute**; 104 FPS, 7.9 mW, 0.3 pJ/SOP (TSMC 28 nm). DOI: 10.1109/TVLSI.2025.3600953.  
11. **Kraken SoC** — Di Mauro et al. Hot Chips / arXiv:2209.01065. Event+frame fusion; SNE runs LIF-FireNet-style OF; energy-proportional sparse SNN.  
12. **SENECA optical-flow comparison** — Xu et al. arXiv:2407.20421. ANN vs SNN event OF on neuromorphic processor with activation sparsification.  

### 1.3 Classical / bio OF circuits (still relevant mechanisms)
13. **TDE / TDE-3** — Time Difference Encoder for direction-selective motion; TDE-3 adds inhibitory third input for textured scenes; trains to map velocity → spike count or ISI (arXiv:2402.11662; Neuromorphic Computing & Eng. 2025).  
14. **Digital TDE FPGA** — TechRxiv / related digital TDE implementations for OF obstacle avoidance (<1 mW class).  

**Gap vs SDformer:** Almost all OF HW is (a) frame CNN/RAFT/pyramid **or** (b) classical/event asynchronous OF **without** Swin-spikeformer tokens. **No published accelerator for SDformerFlow / spikeformer OF.** That is the “first accelerator for X” opening—but X must be an *OF-native mechanism*, not “we accelerated SDSA.”

---

## 2. Algorithm-only OF / event-flow / transformer-flow (2022–2026) — “first HW for X” candidates

| Paper | Venue | Mechanism | Why HW-interesting for SDformer |
|---|---|---|---|
| **RAFT** | Teed & Deng, ECCV 2020 | All-pairs corr + recurrent ConvGRU updates | Iterative update early-exit HW (already partially done by ERAFT); less native to spikeformer |
| **E-RAFT** | Gehrig et al., 3DV 2021 | RAFT on event voxel volumes | Event corr volume; still dense |
| **GMFlow** | Xu et al., CVPR 2022 Oral | Global matching + Transformer feature enhance + softmax match + self-attn **flow propagation** for occlusions | Softmax matching & unmatched-pixel propagation → gated token schedules |
| **CRAFT** | Sui et al., CVPR 2022 | Cross-frame attention replaces brittle local corr under blur/large motion | Cross-frame Q/K filtering = OF-specific attention noise rejection |
| **FlowFormer / ++** | Huang et al. ECCV 2022; Shi et al. CVPR 2023 | Cost-volume as tokens + alternate-group Transformer; MCVA pretrain | Cost-memory decoder with positional queries—closest “transformer OF” to accelerate |
| **VideoFlow** | Shi et al., ICCV 2023 | TROF tri-frame bi-dir + MOP motion propagation across sequence | Multi-frame motion feature reuse → C2-style context bundling but for *motion* not weights |
| **TMA** | Liu et al., ICCV 2023 | Temporal Motion Aggregation for events; fewer refinements (−40% time vs E-RAFT) | Temporal split + linear lookup + pattern aggregation → early-exit HW |
| **Spike-FlowNet** | Lee et al., ECCV 2020 | Hybrid SNN–ANN event OF | Historical SNN OF; energy story |
| **SDformerFlow** | Tian & Andrade-Cetto, ICPR 2024 / arXiv:2409.04082 | First spikeformer dense event OF; 3D Swin SDSA; MS shortcut; SPE; QK-linear | **Your workload**—no HW yet |
| **IDNet** | Wu et al., 2023 | Iterative deblurring as alternative to corr volumes | Avoids 4D cost—friendly to SNN UNet |
| **TDE-3** | 2024–2025 | 3-point time-difference prior for OF | Bio front-end before transformer |
| **Event timing OF** | Greatorex et al., CVPRF 2026 | Precise event timing / synaptic gating without heavy train | Timing datapath, not MAC |

**Strongest “first accelerator for X” claims (algo exists, HW does not):**
1. First accelerator for **spikeformer / SDformerFlow-class** event OF.  
2. First HW for **TMA-style temporal motion aggregation** (or VideoFlow MOP) under spikes.  
3. First HW for **GMFlow-style unmatched-pixel flow propagation** in a spike token fabric.  
4. First co-design of **TDE-3 priors + spikeformer residual flow**.  
5. First **early-exit iterative update** fabric for *spiking* OF (ERAFT did it for ANN RAFT).

---

## 3. Remake ideas for C1 / C2 (ranked)

Each idea: **mechanism → why OF-data-specific (not generic CNN) → C1 or C2 remake angle → novelty (1–10) → citations**.

### Idea A — Direction-Predictive Product Skip (DPP-Skip)
- **Mechanism:** Borrow Liu/TCAS-I **dynamic direction prediction**: maintain a coarse flow / membrane-velocity field; only compute exact products for tokens whose predicted Δflow exceeds threshold or whose polarity contradicts prediction; others reuse cached Acc or zero-skip.
- **OF-data fit:** Flow fields are **spatially smooth + temporally coherent**—direction is far more predictable than generic CNN activations. Event OF concentrates energy along motion edges.
- **C1 remake:** Replace “finite-capacity exact product capture” story with **prediction-gated exact capture**: capacity is spent on *prediction residuals*, not uniform tile residency. Ladder becomes: always-exact / bit-skip / **direction-predict skip** / ceiling.
- **Novelty:** **8.5** (OF HW used this on pyramids; never on spikeformer product capture).
- **Cite:** Liu et al. ISCAS 2024 / TCAS-I 2025; ERAFT prediction (ISCAS 2025); RAFT warm-start.

### Idea B — Pyramid-Resident Residual Capture (PRRC)
- **Mechanism:** Map STSF stages to a **hardware pyramid pipeline**: coarse stage writes low-res flow residual into a shared residual SRAM; finer stages only capture products in the **warp-aligned residual window** (Hamming/BNN intuition from FlowAcc, but for spike products).
- **OF-data fit:** Coarse-to-fine is the defining OF inductive bias for large displacement; SDformer already has hierarchical encoders + multi-scale decode.
- **C1 remake:** Capacity bound becomes **per-pyramid-level residual budget**, not flat tile. Reconfigurable pyramid-layer PE folding (Liu / FlowAcc multiplexing).
- **Novelty:** **8.0**.
- **Cite:** FlowAcc DATE 2022; Ultra-Flow FPL 2022; Liu TCAS-I 2025; SDformerFlow multi-scale decoder.

### Idea C — Occlusion-Gated Exact Capture (OGEC)
- **Mechanism:** Lightweight forward–backward / event-density consistency produces an **unmatched mask**; exact product path only for matched tokens; unmatched tokens take a cheap **flow-propagation** path (neighbor fill / MS-shortcut bleed), echoing GMFlow’s self-attn propagation.
- **OF-data fit:** Occlusion/out-of-boundary is OF-specific failure mode; generic CNN sparse skip ignores *semantic unmatchedness*.
- **C1 remake:** Two datapaths under one capacity ledger: ExactMatch vs Propagate. Claim: same AEE with fewer exact products on DSEC occlusions / car-hood artifacts noted in SDformerFlow.
- **Novelty:** **8.5**.
- **Cite:** GMFlow CVPR 2022; AccFlow ICCV 2023; VideoFlow bi-dir consistency.

### Idea D — Temporal Signature Prefetch (TSP) for Motion-XOR/TTX
- **Mechanism:** Treat Motion-XOR/TTX (or TDE-style Δt codes) as **address generators** into C1’s product buffer: signatures that repeat across T hit a small associative CAM → product reuse; novel signatures allocate exact capture slots.
- **OF-data fit:** Temporal coherence ⇒ motion codes are sticky across frames/time-steps; event timing is the signal, not texture.
- **C1 remake:** Exact capture keyed by **temporal signature**, not spatial tile ID. Sell as co-design with existing Motion workload *without* claiming Motion-XOR algo novelty (AEE gain small—HW reuse is the claim).
- **Novelty:** **7.5–8.0** (careful claim discipline).
- **Cite:** TDE-3 arXiv:2402.11662; Greatorex CVPRF 2026; project Motion-XOR/TTX; SDformerFlow event voxel bins.

### Idea E — Aperture-Robust Multi-Hypothesis Acc (ARM-Acc)
- **Mechanism:** For each spatial locus, keep K short **direction hypotheses** (inspired by ARMS/hARMS multi-scale aperture fix); Acc24 lanes are typed by hypothesis ID; commit winner after local evidence (spike count / ISI).
- **OF-data fit:** Aperture problem is classic OF, not CNN classification.
- **C2 remake:** Retype K8 fabric from “sign/destination contexts” to **motion-hypothesis contexts**; TSBG becomes “same weight-row, multi-hypothesis Acc.”
- **Novelty:** **9.0** (strong OF uniqueness).
- **Cite:** hARMS IEEE Access 2022; classical aperture literature; multi-head SDSA in SDformerFlow.

### Idea F — Motion-Feature Bundle Delivery (MFBD) — VideoFlow/TMA → C2
- **Mechanism:** Instead of bundling *weight rows*, bundle **motion features** that VideoFlow-MOP / TMA would propagate: same spatial weight-row fetched once, broadcast to Acc contexts holding adjacent TROF/time-slice motion states.
- **OF-data fit:** Multi-frame temporal cues and intermediate event-motion features are OF-video specific; reduces iterative refinements (TMA −40% time).
- **C2 remake:** TSBG 2.0 = **Motion-Bundle Scheduler** (ordinary vs motion-bundle modes). Metrics: scalar-bank requests vs motion-context hits; early-exit rate.
- **Novelty:** **9.0**.
- **Cite:** VideoFlow ICCV 2023; TMA ICCV 2023; E-RAFT 3DV 2021.

### Idea G — Softmax-Peak / Attention-Mass Token Gate (SP-Gate)
- **Mechanism:** In SDSA (dot or QK-linear), attention mass is sparse; hardware tracks running peak / token importance `A_t` (already in SDformerFlow QK path) and **suppresses V/product traffic** for cold tokens inside the 3D window.
- **OF-data fit:** Correspondence is peaky (matching), unlike dense classification attention; GMFlow softmax matching is the algo cousin.
- **C2 remake:** Typed fabric destinations gated by attention-mass; K8 only schedules hot tokens. Distinct from generic spike zero-skip because gate is **cross-token match score**, not unary spike presence.
- **Novelty:** **8.0**.
- **Cite:** SDformerFlow QK linear SDSA; GMFlow; Spike-Driven Transformer SMAM HW (arXiv:2501.07825) as related but *not* OF-aware.

### Idea H — Event-History Stream Core (EHSC) — hARMS × spikeformer
- **Mechanism:** Abandon frame-batch tiles for an **asynchronous event-history micro-core**: maintain per-pixel short ring of events; wake STSF windows only when history triggers motion; latency ∝ events not resolution.
- **OF-data fit:** Event cameras’ defining property; SDformerFlow paper itself admits chunked voxels under-use asynchrony.
- **C1 or front-end remake:** C1 products only for woken windows; pairs with ASNA-Flow spatial locality.
- **Novelty:** **8.5** (HW exists for classical event OF; not for spikeformer).
- **Cite:** hARMS 2022; ASNA-Flow TVLSI 2025; SDformerFlow §V limitation on async.

### Idea I — Early-Exit Spike Update Controller (EESUC)
- **Mechanism:** Like ERAFT’s prediction / TMA’s early good estimates: monitor flow residual or membrane Δ; halt extra T-steps / decoder refinements when residual < ε.
- **OF-data fit:** Iterative OF converges unevenly—static regions finish early; motion boundaries need more updates.
- **C1+C2 joint remake:** Dynamic T / dynamic context depth; energy ∝ residual map. Strong co-design story with PSN (SDformerFlow-v2) learnable temporal weights.
- **Novelty:** **8.0**.
- **Cite:** ERAFT ISCAS 2025; TMA ICCV 2023; PSN NeurIPS 2023 (Fang et al.).

### Idea J — Hierarchical Hamming / Binary Descriptor Match Assist (HMA)
- **Mechanism:** FlowAcc-style binary descriptors + Hamming distance as a **cheap proposal** for which spike windows deserve exact SDSA/products; transformer path is residual corrector.
- **OF-data fit:** Patch matching is the heart of classical/DNN OF; SDformer currently jumps straight to attention.
- **C1 remake:** Two-level capture: Hamming propose (bit ops) → exact product refine (scarce capacity).
- **Novelty:** **7.5** (FlowAcc already did BNN+Hamming OF; novelty is coupling to spikeformer residual).
- **Cite:** FlowAcc 2022; Ultra-Flow 2022; StereoEngine BNN lineage.

### Idea K — Cross-Frame Noise-Reject Q/K Filter (CRAFT-lite datapath)
- **Mechanism:** CRAFT’s insight: local corr fails under blur/large motion; apply **Query/Key projections as hardware noise filters** before spike attention mask.
- **OF-data fit:** Large displacement + motion blur are OF benchmarks’ hard cases (Sintel Final, DSEC turns).
- **C2 remake:** Typed path includes a fixed **semantic-smoothing** stage on K before TSBG broadcast.
- **Novelty:** **7.0** (algo clear; HW differentiation softer unless tied to DSEC large-motion ablations).
- **Cite:** CRAFT CVPR 2022; SDformerFlow qualitative large-turn wins.

### Idea L — Spatial-Locality Sparse PE Map (ASNA-style)
- **Mechanism:** Place PEs on a spatial mesh; activate neighborhoods around event clusters; Acc drain follows flow vector predicted direction (move activity with the object).
- **OF-data fit:** Optical flow *is* spatially local transport of brightness events.
- **C1 remake:** Product capacity follows a **moving sparse frontier**, not static tiles.
- **Novelty:** **8.0**.
- **Cite:** ASNA-Flow 2025; Kraken SNE; Sparse VideoGen spatial/temporal head split as cross-domain analogy.

### Idea M — Tri-Frame Bi-Directional Context Fabric
- **Mechanism:** Hardware holds forward & backward Acc contexts for frame/event triplet (VideoFlow TROF); consistency check kills inconsistent Acc lanes.
- **OF-data fit:** Bi-directional OF + occlusion detection is standard OF toolkit.
- **C2 remake:** K8 lanes typed as `{fwd, bwd} × {t−1,t,t+1}` motion contexts; TSBG shares weights across bi-dir.
- **Novelty:** **8.5**.
- **Cite:** VideoFlow ICCV 2023; GMFlow bidirectional occlusion.

### Idea N — Cost-Memory Token Compress (FlowFormer-lite)
- **Mechanism:** Compress local matching costs into short cost tokens (FlowFormer AGT idea) before SDSA; C1 stores **cost tokens** not raw products.
- **OF-data fit:** 4D cost structure is OF-unique; transformers on cost memory beat generic ViT.
- **C1 remake:** “Exact product” → “exact cost-token capture” with OF-specific compression.
- **Novelty:** **8.0** (first HW if scoped to spiking cost tokens).
- **Cite:** FlowFormer ECCV 2022; FlowFormer++ CVPR 2023.

### Idea O — TDE-Prior Front-End → Spikeformer Residual (hybrid co-design)
- **Mechanism:** Cheap digital TDE-3 array produces coarse velocity spikes; SDformer path predicts **residual flow only**; HW partitions energy between TDE array and C1/C2.
- **OF-data fit:** Bio EMD/TDE is purpose-built for motion; residual learning is OF-standard (GMFlow refine, pyramid residual).
- **C1 remake:** Capacity spent on residual products; TDE is hardwired prior (not another NPU).
- **Novelty:** **9.0**.
- **Cite:** TDE-3; SDformerFlow; GMFlow refinement; Greatorex 2026.

---

## 4. Recommended C1 / C2 remake packages (for ISCAS narrative)

### Package 1 — “OF-Residual Exact Capture” (C1 remake) — **recommended**
Combine **A (DPP-Skip) + B (PRRC) + C (OGEC)** under one name:

> **C1\*: Prediction- and occlusion-gated residual product capture for hierarchical event flow.**

- Still a *capture* island (keeps your physical ladder / Formality story).  
- Novelty is **what** you choose to capture (OF residuals), not that you capture exactly.  
- Workload still Motion/SDformerFlow ep34; ablations: w/ vs w/o direction predict, w/ vs w/o occlusion gate, pyramid residual budget sweep.  
- Novelty score of package: **~9**.

### Package 2 — “Motion-Hypothesis / Motion-Bundle Fabric” (C2 remake) — **recommended**
Combine **E (ARM-Acc) + F (MFBD) + G (SP-Gate)** (optional M bi-dir):

> **C2\*: Typed motion-hypothesis fabric with motion-feature bundle delivery and attention-mass gating.**

- Retargets TSBG from generic weight-row reuse to **OF multi-hypothesis / multi-time motion reuse**.  
- Distinct from Prosperity/ELSA-style SNN fabrics because types are **flow hypotheses**, not arbitrary Acc contexts.  
- Novelty score of package: **~9**.

### Package 3 — Moonshot co-design (if page allows one sentence)
**O (TDE prior) + H (event-history wake)** as related-work positioning / future work—or a tiny measured TDE stub feeding C1\*. Highest scientific novelty; higher integration risk for 4-page ISCAS.

### Avoid (per prior review discipline)
- New Motion-XOR RTL as a headline contribution.  
- Claiming attention speedups without OF-data gate.  
- Multiplying C1×C2×TSBG system speedups.  
- Generic spike zero-skip / bit-serial MAC without OF residual/occlusion/hypothesis story.

---

## 5. Mapping: OF property → idea → remake slot

| OF property | Best idea IDs | Prefer remake |
|---|---|---|
| Temporal coherence / warm-start | A, D, I, F | C1\* / C2\* |
| Spatial smoothness | A, L, G | C1\* |
| Pyramid / coarse-to-fine | B, J, O | C1\* |
| Occlusion / unmatched | C, M | C1\* / C2\* |
| Aperture ambiguity | E | C2\* |
| Event asynchrony / sparsity | H, L | C1\* front-end |
| Correspondence peakiness | G, N, J | C2\* / C1\* |
| Iterative refinement | I, F | joint |
| Bio temporal difference | D, O | C1\* + prior |
| Multi-frame motion prop. | F, M | C2\* |

---

## 6. Cross-domain transfers (use carefully)

| Domain | Transferable trick | OF-safe use |
|---|---|---|
| Sparse VideoGen / SPADE (video DiT sparse attn) | Spatial vs temporal attention head split; block-sparse kernels | Analogous to 3D Swin window cold/hot tokens (Idea G)—cite as *inspiration*, not OF proof |
| RAFT iterative HW (ERAFT, RAFT-Lite) | Early-exit / prediction skip | Directly portable to spike T / residual (Idea I) |
| Spike-Driven Transformer accelerators (SMAM, Spike-IAND-Former) | Mask-add SDSA datapaths | Necessary substrate; **insufficient novelty alone**—must add OF gate (G/C/E) |
| Stereo BNN Hamming (StereoEngine / FlowAcc) | Binary match propose + refine | Idea J |

---

## 7. Suggested claim language (ISCAS-safe)

**Good:**  
“We retarget product-capture capacity and typed Acc delivery to **optical-flow residuals, occlusion masks, and multi-hypothesis motion contexts**, exploiting spatial smoothness and temporal coherence of event flow—properties absent in generic SNN classification accelerators.”

**Bad:**  
“We propose an SNN-Transformer accelerator for optical flow” (reviewer: so what?).  
“Motion-XOR improves AEE” (gain too small).  
“C1×C2 = system speedup” (unintegrated islands).

---

## 8. Priority next measurements (engineering, not writing)

1. On ep34 spike traces: measure **direction predictability** (% tokens whose flow sign matches previous T) and **occlusion fraction**—feeds Idea A/C ROI.  
2. Histogram of SDSA / QK `A_t` mass → expected skip rate for Idea G.  
3. Count repeated Motion-XOR/TTX signatures across T → Idea D CAM size.  
4. Pyramid-level residual energy (coarse vs fine product counts) → Idea B SRAM budget.  
5. If adopting Package 2: redefine TSBG counters as **motion-context hits** and re-run directed VCS with hypothesis-typed Acc.

---

## 9. Citation cheat-sheet (real papers only)

**HW / systems**  
- Ling et al., FlowAcc, DATE 2022.  
- Ling et al., Ultra-Flow, FPL 2022.  
- Yan et al., JSA 136:102818, 2023.  
- Liu et al., ISCAS 2024; TCAS-I 2025 (405 FPS scalable OF).  
- ERAFT, ISCAS 2025.  
- Li et al., RAFT-Lite FPGA, CAC 2022.  
- Gong et al., TCAS-I 2023 (LK tracking).  
- Stumpp et al., hARMS, IEEE Access 2022.  
- Wang et al., ASNA-Flow, TVLSI 2025.  
- Di Mauro et al., Kraken, arXiv:2209.01065.  
- VD56G3 characterization, arXiv:2305.13087.  
- Sparse Spike-Driven Transformer accel, arXiv:2501.07825.  
- Spike-IAND-Former HW, arXiv:2503.19643.

**Algo OF / event / transformer**  
- Teed & Deng, RAFT, ECCV 2020.  
- Gehrig et al., E-RAFT, 3DV 2021.  
- Xu et al., GMFlow, CVPR 2022.  
- Sui et al., CRAFT, CVPR 2022.  
- Huang et al., FlowFormer, ECCV 2022; Shi et al., FlowFormer++, CVPR 2023.  
- Shi et al., VideoFlow, ICCV 2023.  
- Liu et al., TMA, ICCV 2023.  
- Lee et al., Spike-FlowNet, ECCV 2020.  
- Tian & Andrade-Cetto, SDformerFlow, ICPR 2024 / arXiv:2409.04082.  
- TDE-3, arXiv:2402.11662.  
- Greatorex et al., Event timing OF, CVPRF 2026.  
- Wu et al., AccFlow, ICCV 2023.

**SDformer internals to hook**  
- 3D Swin SDSA (dot / QK-linear), MS shortcuts, SPE, PSN vs LIF, multi-scale spike decoder, event voxel `T×2n×H×W`.

---

## 10. Bottom line

| Slot | Current (weak novelty feel) | Remake (OF-native) | Novelty |
|---|---|---|---|
| **C1** | Finite-capacity exact product capture | **C1\*** residual / direction-predict / occlusion-gated capture (Ideas A+B+C) | High |
| **C2** | Typed K8 + weight-row TSBG | **C2\*** motion-hypothesis + motion-bundle + attention-mass gate (Ideas E+F+G) | High |
| Optional | — | TDE prior + event-history wake (O+H) | Highest risk/reward |

The differentiator vs “another SNN accel” is not ATLIF amplitude or K8 typing—it is **making the hardware schedule, capture, and reuse decisions in the units optical flow actually lives in: residual motion, occlusion, direction hypotheses, and temporal coherence.**

