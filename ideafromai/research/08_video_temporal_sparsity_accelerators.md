# ROUND 2 — Video / Temporal Sparsity Accelerators → SDformer C1*/C2* Remakes

**Date:** 2026-09-05 (Asia/Shanghai)  
**Scope:** Mechanisms that exploit **temporal redundancy, motion, frame difference, sparse video transformers**, transferable to **optical-flow SNN-Transformer** (SDformer / SDformerFlow).  
**Venues / arXiv window:** ~2020–2026; **no fabricated venues**.  
**Relation to ROUND 1 packages:** OP-STW, PRRC, OGEC, HBG-RP, ADP-MAC, ARM-Acc, MFBD, SP-Gate, EESUC, VGTS, ReMem-Tok — each idea below is tagged **enhance** vs **replace**.

**Legend**
- **HW:** published accelerator / chip / FPGA system that implements the mechanism.
- **Algo-only / first-HW candidate:** strong algorithm with GPU/CUDA kernels or training-free sparsity but **no dedicated ASIC/FPGA accelerator yet** — flag for “first HW for X” claims.
- **Transfer:** how it maps onto event-flow / spikeformer OF datapath (Tw×H×W windows, ATLIF payload, SDSA).

---

## 0. Landscape map (what exists vs gap)

| Cluster | Representative work | HW? | Gap vs SDformer OF |
|---|---|---|---|
| Video DiT sparse attention | Sparse VideoGen (ICML 2025); SVG2 (arXiv:2505.18875); RainFusion2.0 (arXiv:2512.24086); SPADE (arXiv:2608.03335); VSA (arXiv:2505.13389) | Mostly **algo + GPU kernels**; RainFusion claims device-agnostic; **FlightVGM** (FPGA’25 Best Paper) and **Kaleido** (arXiv:2607.13770, RTL 16 nm) are true HW | Generation DiT ≠ OF spikeformer, but **spatial vs temporal head / block-sparse layout** transfers to SDSA scheduling |
| Temporal Δ / DiffFrame / delta CNN | VideoTime3 (IEEE LSSC 2023); DeltaCNN (CVPR 2022 / arXiv:2203.03996); MotionDeltaCNN (ICCV 2023); CBinfer (TCSVT 2019); Skip-Conv (arXiv:2104.11487) | VideoTime3 = **28 nm chip**; Delta* = CUDA; CBinfer/Skip = algo | Closest to **event sparsity**; moving-camera Δ needs OF warp (MotionDelta) |
| Early-exit video | TLEE (IEEE IoT-J 2023); ERAFT-style residual early stop (already in R1) | Edge GPU/Jetson; not OF ASIC | Maps to **EESUC / VGTS** |
| Motion / OF-guided skip | MotionDeltaCNN; MaskVD (arXiv:2407.12067 cites OF skip as expensive); FlowAcc/ERAFT family (R1) | Mostly algo or pure OF HW | **OF as skip prior** for transformer tokens — under-served |
| 3D CNN temporal reuse | Morph (MICRO 2018); Systolic Cube / Edge 3D CNN (DAC 2019 + TCAD 2020); TSR@ICECS 2022; HARFLOW3D (arXiv:2303.17218) | Yes (ASIC/FPGA toolflows) | Temporal filter reuse → **ReMem-Tok / MFBD** |
| Event–video hybrid (beyond pure OF chips) | DVS–CIS fusion + NPU trigger (ISCAS 2025 demos); Lele et al. event–frame fused OF (ISCAS 2022); EvGNN (arXiv:2404.19489); ESDA (arXiv:2401.05626); HOMI (arXiv:2508.12637); EDFLOW (TCSVT 2022); AceleradorSNN (arXiv:2603.28429) | FPGA systems | Hybrid **event wake + frame/texture refine** for SDformer front-end |
| Token merge / TSM | ToMe (ICLR 2023); TESTA (EMNLP Findings 2023); TempMe (arXiv:2409.01156); STTM (ICCV 2025); TSM + FPGA DPU demo (MIT Han Lab `tsm_fpga`) | TSM = **FPGA deploy** (shift on CPU, conv on DPU); ToMe-family = **algo-only** | Merge/shift as **token/membrane reuse**, not MAC reorder |

**Explicit “first-HW candidates” (algo strong, dedicated OF/SNN-Transformer HW missing):** Sparse VideoGen / SVG2 / SPADE / VSA / RainFusion2.0 patterns; DeltaCNN/MotionDeltaCNN as **ASIC-class datapath**; STTM/TESTA/TempMe token merge; ToMe for video DiT.

---

## 1. Strong remake ideas (12) — citation → transfer → C1*/C2* name

### Idea 1 — Spatial–Temporal Head Dual Sparse Attention (from Sparse VideoGen)

| Field | Content |
|---|---|
| **Citation** | Xi et al., *Sparse VideoGen: Accelerating Video Diffusion Transformers with Spatial-Temporal Sparsity*, **ICML 2025** / arXiv:2502.01776. Follow-on: SVG2, arXiv:2505.18875 (semantic-aware permutation). |
| **Mechanism** | Online-profile each attention head as **Spatial Head** (within-frame block) vs **Temporal Head** (same spatial locus across frames / slash pattern); layout transform (token-major → frame-major) so temporal sparsity becomes contiguous block-sparse; ~30% sparsity, ≤2.3× e2e GPU speedup, PSNR~29. |
| **HW status** | **Algo + customized Triton/FlashInfer kernels** — **first-HW candidate** for ASIC/FPGA sparse SDSA. |
| **OF / SNN-Transformer transfer** | SDSA windows already have Tw×Hs×Ws; classify heads/tokens as **intra-frame spatial** vs **same-(x,y) temporal** and skip the other. Online 1% row profiling → cheap vs full QK. Frame-major layout matches event voxel `T×H×W`. |
| **Remake name** | **STH-Gate** (Spatial–Temporal Head Gate) — **C2\*** primary; optional C1\* tile wake. |
| **vs packages** | **Enhances SP-Gate** (attention-mass → now *typed* spatial/temporal mass); **enhances MFBD** (temporal-head tokens = motion-bundle peers); does **not** replace HBG-RP. |
| **Novelty** | **8.5 / 10** (first OF-spikeformer HW for SVG-style dual-head sparse would be strong; algo itself published). |
| **Prove-by experiment** | (1) Histogram of SDSA attention mass: spatial-block vs temporal-slash on Motion ep34; (2) PE wake vs always-on / zero-skip / SP-Gate-only; (3) AEE Δ at fixed SOP budget with oracle vs 1% online profile. |

---

### Idea 2 — Spatiotemporal Online Activation Sparsify (from FlightVGM)

| Field | Content |
|---|---|
| **Citation** | Liu et al., *FlightVGM: Efficient Video Generation Model Inference with Online Sparsification and Hybrid Precision on FPGAs*, **FPGA 2025** (Best Paper), DOI:10.1145/3706628.3708864. |
| **Mechanism** | Spatial–temporal **online activation sparsification** (+SU/RU units); hybrid FP/INT DSP58 expansion; dynamic–static scheduling for online compression; 3.17× compute cut; V80 FPGA beats 3090 by 1.30× perf / 4.49× energy on VGMs. |
| **HW status** | **True FPGA HW** for video DiT sparsify. |
| **Transfer** | Replace DiT activation Δ with **spike/ATLIF amplitude Δ** across Tw; SU = sparsity detector on membrane or event histogram; RU = dense restore only at flow head. Prefer **fixed** for linear SDSA path, keep higher precision on residual flow Acc (mirrors HBG-RP split). |
| **Remake name** | **STA-SU** (SpatioTemporal Activation Sparsify Unit) — **C1\*** (tile/spike wake) + C2\* schedule. |
| **vs packages** | **Enhances OP-STW** (direction wake + activation-similarity wake); **enhances VGTS**; complements ADP-MAC (bit sparsity × temporal act sparsity). |
| **Novelty** | **8 / 10** (FlightVGM exists for DiT; OF-SNN remake + ATLIF is new). |
| **Prove-by** | Measure activation/spike cosine or L1 Δ across adjacent Tw; ablation SU threshold vs AEE; energy of SU+skip vs always-on PE on FPGA prototype. |

---

### Idea 3 — Channel-wise Latent/Token Reuse (from Kaleido)

| Field | Content |
|---|---|
| **Citation** | *Kaleido: Algorithm-Hardware Co-Design for Video Diffusion Transformers by Exploiting Latent Space Correlations*, arXiv:2607.13770 (2026); systolic-like accel, **16 nm RTL**. |
| **Mechanism** | RoPE channel groups encode (t,x,y); **channel-wise reuse** skips partial MAC when adjacent tokens’ channel prefixes match; reconfigurable PE + data dispatcher for irregular reuse; up to 5.9× / 16× energy vs prior DiT accelerators. |
| **HW status** | **ASIC-class co-design** (not GPU-only). |
| **Transfer** | Spikeformer tokens: reuse **partial QK or payload MAC** when neighboring (t) or warped-(x,y) tokens share ATLIF channel prefixes; dispatcher = MFBD-like bundle router. |
| **Remake name** | **CW-Reuse** — **C2\*** datapath. |
| **vs packages** | **Enhances ADP-MAC** (bit + channel-prefix reuse); **enhances ReMem-Tok** (membrane reuse when prefix match); does not replace ARM-Acc. |
| **Novelty** | **8 / 10**. |
| **Prove-by** | Channel-prefix hit rate on SDformer features; RTL cycle count with/without CW-Reuse; AEE under forced reuse. |

---

### Idea 4 — DiffFrame / Real-time Temporal Δ Convolution Chip (from VideoTime3)

| Field | Content |
|---|---|
| **Citation** | *VideoTime3: A 40-μJ/frame 38 FPS Video Understanding Accelerator With Real-Time DiffFrame Temporal Redundancy Reduction and Temporal Modeling*, **IEEE Solid-State Circuits Letters (LSSC)**, 2023, DOI:10.1109/LSSC.2023.3286698. |
| **Mechanism** | **DiffFrame** convolution on sparse frame differences + on-chip RefFrame; sorter-free sparse OS dataflow; single-frame latency; temporal modeling without multi-frame batch latency; 40 μJ/frame @ 0.6 V, 28 nm. |
| **HW status** | **Silicon**. |
| **Transfer** | Event voxels are *already* DiffFrame-like; for hybrid RGB/event OF: DiffFrame path for APS frames, spike path for DVS. RefFrame buffer ↔ membrane / residual flow state. Single-frame latency aligns with online OF. |
| **Remake name** | **DF-OS** (DiffFrame Output-Stationary) — **C1\*** residency / capture. |
| **vs packages** | **Enhances PRRC** (residual budget on DiffFrame, not full frame); **enhances EESUC**; orthogonal to OP-STW (direction) — combine. |
| **Novelty** | **7.5 / 10** (chip exists; OF-spikeformer DiffFrame Acc is new claim angle). |
| **Prove-by** | DRAM traffic / SOP of DiffFrame vs dense voxel; AEE with RefFrame refresh every N events; latency single-step vs batch-T. |

---

### Idea 5 — End-to-end Sparse Frame-Difference CNN (DeltaCNN → ASIC candidate)

| Field | Content |
|---|---|
| **Citation** | Parger et al., *DeltaCNN: End-to-End CNN Inference of Sparse Frame Differences in Videos*, **CVPR 2022** / arXiv:2203.03996 (Meta); up to 7× vs cuDNN with CUDA sparse kernels. |
| **Mechanism** | Propagate **sparse update masks** through *all* layers (not only first); truncate insignificant Δ; structured per-pixel sparsity for SIMD. |
| **HW status** | **Algo + GPU kernels — first-HW / ASIC candidate**. |
| **Transfer** | Treat event spike presence as update mask; propagate mask through SPE → STSF → flow head without densifying until decoder. Avoid error accumulation via periodic dense refresh (like Skip-Conv schedule). |
| **Remake name** | **Δ-MaskPipe** — **C1\*** + pipeline. |
| **vs packages** | **Enhances OP-STW** (mask = wake); can **partially replace** naive zero-skip narrative; keep HBG-RP for payload. |
| **Novelty** | **8 / 10** as *first ASIC for OF-spikeformer delta-pipe*. |
| **Prove-by** | Mask density vs time; AEE with truncation ε; PE utilization of mask-gated vs dense. |

---

### Idea 6 — Motion-Warped Delta / Spherical Buffer (MotionDeltaCNN)

| Field | Content |
|---|---|
| **Citation** | Parger et al., *MotionDeltaCNN*, **ICCV 2023**; spherical buffers + padded conv for **moving cameras**; up to +90% FPS vs DeltaCNN on moving cams. |
| **Mechanism** | Warp previous buffer with camera/ego motion; fuse newly unveiled regions; 2D ring / spherical buffer without memory blowup. |
| **HW status** | **Algo + CUDA — first-HW candidate**; *OF-guided skip with HW buffers*. |
| **Transfer** | **Critical for OF:** use coarse flow / TDE prior to warp ReMem / RefFrame, then compute residual spikes only. Spherical buffer = on-chip torus for ego-motion OF. |
| **Remake name** | **MW-ΔBuf** (Motion-Warped Delta Buffer) — **C1\*** memory + **C2\*** schedule. |
| **vs packages** | **Enhances ReMem-Tok** (warp then reuse membrane); **enhances OP-STW / PRRC** (warp residual window); **enhances MFBD** (bundles follow motion). Closest “optical-flow-guided DNN skip” with a concrete buffer story. |
| **Novelty** | **9 / 10** for SDformer (motion-warped residual spike pipe is OF-native). |
| **Prove-by** | Static-cam Δ vs ego-motion Δ sparsity; AEE with/without warp; SRAM footprint spherical vs naive full refresh. |

---

### Idea 7 — Temporal + Layer Early Exit (TLEE) → residual OF exit

| Field | Content |
|---|---|
| **Citation** | Fang et al., *TLEE: Temporal-Wise and Layer-Wise Early Exiting…*, **IEEE Internet of Things Journal**, 2023, DOI:10.1109/JIOT.2023.3293506; Jetson Nano validated. |
| **Mechanism** | Gate which **frame** to stop; branch which **layer** to exit; feature-reuse module aggregates exited features. |
| **HW status** | Algo on edge GPU — **first dedicated OF early-exit HW still open**. |
| **Transfer** | Exit when **flow residual / membrane Δ** below τ (ERAFT-like); temporal-wise = skip quiet Tw; layer-wise = skip deep SDSA blocks on easy tiles. |
| **Remake name** | **TL-Exit** — strengthens **EESUC** / **VGTS**. |
| **vs packages** | **Enhances EESUC** (add temporal-wise + layer-wise dual gate); **enhances VGTS**; does not replace ARM-Acc. |
| **Novelty** | **7.5 / 10**. |
| **Prove-by** | Exit-rate CDF vs AEE; energy vs always-full-T; correlate exit with ERAFT prediction confidence / OF residual. |

---

### Idea 8 — 3D Temporal Reuse + Redundancy Skip (Systolic Cube / Morph / TSR)

| Field | Content |
|---|---|
| **Citation** | Wang et al., *An Edge 3D CNN Accelerator…* (**TCAD** 2020 / Systolic Cube **DAC 2019** DOI:10.1145/3316781.3317919); Hegde et al., *Morph*, **MICRO 2018**; De Alwis & Alioto, Temporal Similarity Removal, **ICECS 2022**. |
| **Mechanism** | 3D PE cube moves data for temporal filter reuse; detect repetitive activations across adjacent time and **skip identical act×weight**; Morph flexible tiling/dataflow for 3D CNNs; TSR tunnels skip similar fmap. |
| **HW status** | **Published ASIC/FPGA-class 3D CNN HW**. |
| **Transfer** | Swin 3D window: reuse weights/activations along Tw; skip when spike maps identical across t (common in static background). Flexible Morph-like config per STSF stage. |
| **Remake name** | **T3D-Reuse** — **C2\*** / fabric. |
| **vs packages** | **Enhances ReMem-Tok** and **MFBD**; **enhances ADP-MAC** when combined with bit sparsity. |
| **Novelty** | **7 / 10** (known in 3D CNN; new on spikeformer OF). |
| **Prove-by** | Temporal identical-activation rate on ep34; energy of skip vs 3D-dense; compare Morph-style dataflow configs for Tw=2 vs Tw=4. |

---

### Idea 9 — Event–Frame Hybrid Wake / Cognitive Trigger (DVS–CIS fusion)

| Field | Content |
|---|---|
| **Citation** | ISCAS 2025 live demo: DVS–CIS fusion + YOLOv3-Tiny NPU, **DVS-triggered NPU** → 31.5% power save (DOI:10.1109/ISCAS56072.2025.11043578); receiver DOI:10.1109/ISCAS56072.2025.11044183; Lele et al., fused event–frame OF, **ISCAS 2022**; EvGNN arXiv:2404.19489; ESDA arXiv:2401.05626; HOMI arXiv:2508.12637; EDFLOW TCSVT 2022. |
| **Mechanism** | Sparse DVS detects ROI / motion → wake dense DNN or ISP; event GNN / sparse dataflow for pure events; fused OF combines leaky event CNN + frame flow. |
| **HW status** | **Multiple FPGA systems** (beyond pure OF ASICs like ASNA-Flow / hARMS already in R1). |
| **Transfer** | Event front-end **wakes** SDformer tiles (OP-STW); optional APS texture path for occlusion fill (**OGEC**). EvGNN-style queues for neighbor search ≈ local SDSA windows. |
| **Remake name** | **EV-Wake** — **C1\*** front-end. |
| **vs packages** | **Enhances OP-STW + OGEC**; does **not** replace pure spike path; optional hybrid mode. |
| **Novelty** | **8 / 10** as *SDformer-specific* hybrid scheduler (fusion demos exist, spikeformer OF HW does not). |
| **Prove-by** | Power with DVS-gated PE vs always-on; AEE hybrid vs event-only on MVSEC/DSEC; latency per event queue. |

---

### Idea 10 — Video Token Merging / Aggregation (ToMe, TESTA, TempMe, STTM)

| Field | Content |
|---|---|
| **Citation** | Bolya et al., *Token Merging (ToMe)*, **ICLR 2023** / arXiv:2210.09461 (incl. video MAE); Ren et al., *TESTA*, **EMNLP Findings 2023**; Shen et al., *TempMe*, arXiv:2409.01156; Hyun et al., *STTM*, **ICCV 2025**. |
| **Mechanism** | Bipartite similarity merge of redundant tokens; TESTA separates temporal vs spatial aggregation (~75% token cut); TempMe progressive temporal merge; STTM quadtree spatial + directed temporal merge, query-agnostic KV reuse. |
| **HW status** | **Algo-only — first-HW candidate** (no OF/SNN merge ASIC found). |
| **Transfer** | Merge quiet background spike tokens inside Swin windows; keep high-|amp| ATLIF tokens; unmerge at flow head. Directed temporal merge ≈ motion correspondence without full attention. |
| **Remake name** | **TokMerge-OF** — **C2\*** token fabric / **C1\*** capacity relief. |
| **vs packages** | **Enhances ReMem-Tok** (merge then share membrane); **enhances PRRC** (coarse tokens = pyramid); **enhances SP-Gate** (merged token = one attention slot). Distinct from ADP-MAC (bits vs tokens). |
| **Novelty** | **8.5 / 10** as first HW token-merge for spikeformer OF. |
| **Prove-by** | Token count vs AEE; merge error on moving edges; PE wake reduction; compare ToMe vs TESTA vs STTM schedules in RTL sim. |

---

### Idea 11 — Temporal Shift as Zero-MAC Temporal Mix (TSM + FPGA)

| Field | Content |
|---|---|
| **Citation** | Lin et al., *TSM: Temporal Shift Module…*, **ICCV 2019**; extended TPAMI/edge; MIT Han Lab **`tsm_fpga`** Vitis-AI DPU demo (shift on CPU, conv on DPU). |
| **Mechanism** | Channel shifts across frames = temporal mixing **with zero FLOPs/params**; online uni-directional shift for streaming. |
| **HW status** | **FPGA deploy exists** but shift not hardened as custom PE — opportunity. |
| **Transfer** | Shift a fraction of ATLIF / membrane channels across Tw **before** SDSA — temporal context without extra MAC; hardware shift register between PE rows. |
| **Remake name** | **TS-Shift** — **C2\*** lightweight temporal path. |
| **vs packages** | **Enhances MFBD / ReMem-Tok** (shift delivers temporal neighbors); does **not** replace HBG-RP; orthogonal to SP-Gate. |
| **Novelty** | **7 / 10** (TSM known; hardened spike-channel shift for OF is fresh). |
| **Prove-by** | AEE with/without channel shift at iso-SOP; area of shift-reg vs extra temporal conv; streaming latency. |

---

### Idea 12 — Block-wise Video Sparse Attn + First-Frame Sink (RainFusion2.0 / SPADE / VSA)

| Field | Content |
|---|---|
| **Citation** | RainFusion2.0, arXiv:2512.24086 (block-mean mask, spatiotemporal permutation, **first-frame sink**, 80% sparsity, 1.5–1.8×, claims ASIC-friendly); SPADE, arXiv:2608.03335 (input-adaptive head-wise sparse engine); VSA, arXiv:2505.13389 (trainable cube sparse attn, HW-aligned tiles). |
| **Mechanism** | Low-overhead block representatives predict sparse mask; permute for denser critical tokens; video-specific **first-frame / sink** tokens always kept; cube↔SM tile mapping. |
| **HW status** | RainFusion/SPADE/VSA = **algo / GPU engine — first-HW candidates**; RainFusion explicitly targets non-GPU. |
| **Transfer** | Keep **reference frame / key event slice** as sink tokens; block-mean of spike counts predicts SDSA mask; cubes = 3D Swin windows. |
| **Remake name** | **SinkSparse** — **C2\*** attention engine. |
| **vs packages** | **Enhances SP-Gate + STH-Gate**; **enhances OGEC** (sink = matched reference); complements EESUC. |
| **Novelty** | **8 / 10**. |
| **Prove-by** | Mask IoU vs oracle attention; quality with/without sink; FPGA block-sparse vs unstructured sparse throughput. |

---

## 2. Package matrix — enhance vs replace

| Existing package | Role today | ROUND 2 interaction |
|---|---|---|
| **OP-STW / DPP-Skip** | Direction/residual → spike-tile wake | **Enhance** with STA-SU, Δ-MaskPipe, EV-Wake, MW-ΔBuf (more wake signals: act-sim, mask, DVS, warped Δ) |
| **PRRC** | Pyramid residual budget | **Enhance** with DF-OS, TokMerge-OF, MW-ΔBuf (residual on DiffFrame / merged coarse tokens) |
| **OGEC** | Occlusion → exact vs propagate | **Enhance** with EV-Wake (APS fill), SinkSparse (reference sink), MW-ΔBuf (unveil regions) |
| **HBG-RP** | Binary gate + real payload | **Keep / enhance** with FlightVGM-style hybrid precision; **do not replace** |
| **ADP-MAC** | Bilateral bit sparsity | **Enhance** with CW-Reuse (channel prefix) + T3D-Reuse |
| **ARM-Acc** | Aperture multi-hypothesis Acc | **Keep**; temporal sparse ideas feed hypotheses, don’t replace |
| **MFBD** | Motion-bundle delivery | **Enhance** with STH-Gate temporal heads, TS-Shift, MW-ΔBuf |
| **SP-Gate** | Attention-mass schedule | **Enhance → specialize** as STH-Gate / SinkSparse (typed sparse patterns) |
| **EESUC** | Residual early stop | **Enhance** with TL-Exit (temporal+layer) |
| **VGTS** | Amplitude-aware time skip | **Enhance** with STA-SU, DiffFrame, Delta masks |
| **ReMem-Tok** | Shifted-window membrane reuse | **Enhance** with MW-ΔBuf warp, TokMerge-OF, T3D-Reuse, TS-Shift |

**Nothing in this round should *replace* HBG-RP or ARM-Acc.** Strongest *narrative upgrades*: **MW-ΔBuf**, **STH-Gate**, **TokMerge-OF**, **EV-Wake**.

---

## 3. Recommended C1* / C2* remake bundles (video-temporal round)

### C1* “Temporal Residual Front-End” (add to R1 C1* pack)
1. **MW-ΔBuf** — motion-warp RefFrame / membrane, compute residual spikes only.  
2. **Δ-MaskPipe / DF-OS** — sparse update propagation + DiffFrame OS capture.  
3. **EV-Wake** — DVS (or event voxel energy) gates tile exact-path.  
4. Keep **OP-STW + PRRC + OGEC** from R1.

**Claim sketch:** *First spikeformer-OF frontend that couples motion-warped temporal residuals with event-triggered exact-product budgets.*

### C2* “SpatioTemporal Sparse Fabric” (add to R1 C2* pack)
1. **STH-Gate** — spatial vs temporal head/token sparse SDSA.  
2. **SinkSparse / CW-Reuse** — block-mask + channel-prefix reuse.  
3. **TokMerge-OF + TS-Shift + T3D-Reuse** — reduce tokens / zero-MAC temporal mix / 3D reuse.  
4. Keep **HBG-RP + ADP-MAC + ARM-Acc/MFBD + SP-Gate**.

**Claim sketch:** *C2 is not multiply reorder — it is typed spatiotemporal sparsity + ATLIF payload under motion-aware token fabric.*

---

## 4. First-HW candidate shortlist (algo → claim carefully)

| Candidate | Source | Safe claim if you build HW |
|---|---|---|
| Dual spatial/temporal sparse SDSA | Sparse VideoGen / SVG2 | First **OF-spikeformer** accelerator with SVG-style dual-head sparse + layout transform |
| Motion-warped delta buffer | MotionDeltaCNN | First **ASIC/FPGA** motion-warped residual pipe for event OF transformer |
| Token merge for spike windows | STTM / TESTA / TempMe / ToMe | First hardware token-merge unit for spikeformer OF |
| Block-sparse video attn engine | RainFusion2.0 / SPADE / VSA | First block-sparse SDSA engine with OF sink tokens |
| Channel-wise reuse PE | Kaleido | Adapt Kaleido PE to ATLIF channels (cite Kaleido; claim OF-SNN workload) |

**Do not claim:** “first video sparse attention” (SVG/FlightVGM/Kaleido exist); “first DiffFrame chip” (VideoTime3); “first TSM on FPGA” (Han Lab demo).

---

## 5. Related notes (requested keywords)

- **EVA:** name collision across systems (CLIP/EVA vision, unrelated edge tools). **No stable “EVA video sparsity accelerator”** found in 2020–2026 top-HW venues for this remake — do not hang a claim on EVA without a specific citation the team owns. Prefer **ToMe / TESTA / STTM** for token reduction.  
- **Sparse VideoGen:** §Idea 1; GPU kernels; ICML 2025; **HW-missing for ASIC**.  
- **Token Merging for video:** ToMe (ICLR’23), TESTA, TempMe, STTM (ICCV’25) — §Idea 10.  
- **Temporal Shift HW:** TSM algo + `tsm_fpga` DPU split deploy — §Idea 11; custom shift PE still open.  
- **Streaming video DiT sparsity HW:** FlightVGM (FPGA’25), Kaleido (16 nm); RainFusion/SPADE/VSA algo — §Ideas 2,3,12.

---

## 6. Suggested prove-by ladder (shared)

1. **Data stats on Motion ep34 / MVSEC:** temporal Δ sparsity, spatial vs temporal attention mass, mergeable token %, warp residual energy.  
2. **Cycle-accurate / FPGA:** PE wake, SRAM traffic, SOP/J vs always-on, zero-skip, R1 C1/C2, new mechanisms.  
3. **Accuracy:** AEE / Fl-all / NPE; ablate each gate.  
4. **Fair baselines:** VideoTime3-style DiffFrame, DeltaCNN mask (software), SVG oracle head type, ERAFT early-exit — all cited, not invented.

---

## 7. Sources (verified; no fabricated venues)

1. Xi et al., Sparse VideoGen, **ICML 2025**, arXiv:2502.01776.  
2. SVG2, arXiv:2505.18875.  
3. FlightVGM, **FPGA 2025**, DOI:10.1145/3706628.3708864.  
4. Kaleido, arXiv:2607.13770.  
5. RainFusion2.0, arXiv:2512.24086.  
6. SPADE, arXiv:2608.03335.  
7. VSA, arXiv:2505.13389.  
8. VideoTime3, **IEEE LSSC 2023**, DOI:10.1109/LSSC.2023.3286698.  
9. DeltaCNN, **CVPR 2022**, arXiv:2203.03996.  
10. MotionDeltaCNN, **ICCV 2023**.  
11. CBinfer, **IEEE TCSVT 2019**, DOI:10.1109/TCSVT.2019.2903421.  
12. Skip-Convolutions, arXiv:2104.11487.  
13. TLEE, **IEEE IoT-J 2023**, DOI:10.1109/JIOT.2023.3293506.  
14. Systolic Cube / Edge 3D CNN, **DAC 2019** + **TCAD 2020**, DOI:10.1109/TCAD.2020.3011042.  
15. Morph, **MICRO 2018**, DOI:10.1109/MICRO.2018.00080.  
16. TSR 3D CNN, **ICECS 2022**, DOI:10.1109/ICECS202256217.2022.9970939.  
17. HARFLOW3D, arXiv:2303.17218.  
18. ToMe, **ICLR 2023**, arXiv:2210.09461.  
19. TESTA, **EMNLP Findings 2023**.  
20. TempMe, arXiv:2409.01156.  
21. STTM, **ICCV 2025**.  
22. TSM, **ICCV 2019**; FPGA demo: mit-han-lab/temporal-shift-module `tsm_fpga`.  
23. DVS–CIS ISCAS 2025 demos, DOI:10.1109/ISCAS56072.2025.11043578 / 11044183.  
24. Lele et al., event–frame fused OF, **ISCAS 2022**.  
25. EvGNN, arXiv:2404.19489; ESDA, arXiv:2401.05626; HOMI, arXiv:2508.12637; EDFLOW, **TCSVT 2022**.  

---

*End of ROUND 2 report. Feed into `04_SYNTHESIS` / C1* C2* microarch sketches next.*
