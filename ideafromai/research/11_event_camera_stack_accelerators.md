# ROUND 3 — Event-Camera / DVS / Event-Vision Full Stack → SDformer C1*/C2* Remakes

**Date:** 2026-09-05 (Asia/Shanghai)  
**Scope:** Sensor → encode → optical-flow / motion → transformer / SNN accelerators; mechanisms transferable to **optical-flow SNN-Transformer** (SDformer / SDformerFlow).  
**Venues / arXiv window:** ~2018–2026; **no fabricated citations**.  
**Relation to ROUND 1–2 packages:** OP-STW, PRRC, OGEC, HBG-RP, ADP-MAC, ARM-Acc, MFBD, SP-Gate, EESUC, VGTS, ReMem-Tok, ECP-QKV, MW-ΔBuf, EV-Wake, STH-Gate, SMAM-RP, Motion-TTB — each idea below is tagged **enhance** vs **replace**.

**Legend**
- **HW:** published accelerator / chip / FPGA system that implements the mechanism.
- **Algo-only / first-HW candidate:** strong algorithm with GPU/CUDA or training-free sparsity but **no dedicated OF-spikeformer ASIC/FPGA** — flag carefully for “first HW for X.”
- **Transfer:** how it maps onto event-flow / spikeformer OF datapath (Tw×H×W voxels, ATLIF payload, SDSA).

**Stack layers covered**
1. Event **sensor** readout / compression / hybrid CIS+EVS  
2. Event **encoding** HW (voxel / histogram / time-surface / TDE)  
3. Event **optical-flow** accelerators (classical + neuromorphic ASIC)  
4. Event **DNN / GNN / sparse CNN** edge systems  
5. **Spiking transformers** accelerators (vision / classification — gap vs OF)  
6. Gaps vs **dense OF-SNN-Transformer ASIC** (SDformerFlow has **no published HW**)

---

## 0. Landscape map (what exists vs gap)

| Cluster | Representative work | HW? | Gap vs SDformer OF |
|---|---|---|---|
| Event sensor + in-sensor compress | Finateu et al. **ISSCC 2020** Prophesee/Sony 1280×720 EVS (1.066 GEPS, rate ctrl, compressive pipeline); Guo et al. **ISSCC/JSSC 2023** 3-wafer CIS+EVS (4.6 GEps, skip/ROI, in-pixel TDC); Krishnan/Hassan **A-SSCC 2023 / LSSC 2024** 3D-ISC 65 nm DVS autoencoder compress (>10×, <6 mW) | **Silicon** | Bandwidth/ROI gate exists; **no bridge into spikeformer OF tokens** |
| Event encode (frame / TS / voxel) | Blachut & Kryjak **SPA 2023** HD event-frame FPGA; HOMI **arXiv:2508.12637** SLTS/SETS shift time-surfaces + MIPI EVT3.0; SSER **arXiv:2505.07556** GRU/MGU per-pixel encoder on ZCU104; 3DS-ISC **arXiv:2512.20073** eDRAM time-surface ISC | **FPGA / sim ASIC** | Encoders stop at CNN classify; **SPE / voxel→spikeformer path unhardened** |
| Classical / block-match event OF | Liu & Delbruck **ISCAS 2017** BMOF; Aung et al. **ISCAS 2018** plane-fit; Liu & Delbruck **EDFLOW TCSVT 2022** ABMOF+SFAST (123 GOp/s, ~100× less power than EV-FlowNet); EventShiftFlow **arXiv:2605.28312** bitvector hyp-shift Artix-7 (<2 kB, 0 DSP) | **FPGA** | Sparse/coarse OF; **not dense transformer OF** |
| Neuromorphic event OF ASIC | ASNA-Flow **TVLSI 2025** TSMC 28 nm, 104 FPS, 7.9 mW, 0.3 pJ/SOP; hARMS **IEEE Access 2022**; Kraken **SNE** HotChips/arXiv:2209.01065 + SNE DATE-class; TrueNorth OF (Haessig TBioCAS 2018) | **ASIC / SoC** | SNN/classical OF; **no Swin-SDSA / ATLIF fabric** |
| Hybrid DVS–CIS + NPU wake | Cha et al. **ISCAS 2025** DVS-CIS fusion + event NPU trigger (31.5% energy); Lele & Raychowdhury **ISCAS 2022** event–frame fused OF (Leaky CNN + Farneback); ColibriUAV / SwiftEagle UAV platforms | **FPGA systems** | Wake for YOLO/CNN — **maps to EV-Wake but not OF residual budgets** |
| Sparse event DNN / GNN | ESDA **FPGA’24** / arXiv:2401.05626 submanifold sparse dataflow; EvGNN **TCAS-AI 2024** / arXiv:2404.19489 event queues 16 µs/evt; EFGCN **JSA 2026** / arXiv:2406.07318; NullHop + DVS (ICONS 2021); Eventor **DAC 2022** EMVS | **FPGA** | Classify/detect/stereo — **not dense OF transformer** |
| Event OF **algorithms** (dense / learnable) | EV-FlowNet RSS’18; E-RAFT **3DV 2021**; TMA **ICCV 2023**; Spike-FlowNet ECCV’20; SDformerFlow **ICPR 2024** / arXiv:2409.04082; TDE-3 Frontiers/arXiv:2402.11662; Greatorex timing OF **CVPRF 2026** | Mostly **algo / GPU**; TDE on Loihi/FPGA | **TMA / SDformerFlow / timing-OF = first-HW candidates** |
| Spiking / event transformers HW | Xpikeformer **IEEE** (arXiv:2408.08794); FireFly-T arXiv:2505.12771; SPARTA **ICCAD 2025**; ASTER arXiv:2511.06770; Bishop AAC; SMAM | **HW for classify / attention** | **No published accelerator for spikeformer *optical flow*** |
| Frame RAFT OF HW (non-event) | FlowAcc **DATE 2022**; ERAFT **ISCAS 2025** (frame RAFT, VCK190 86 FPS); Ultra-Flow FPL’22 | **FPGA** | Prediction early-exit transferable; **input is frames, not events** |

**Explicit gap:** There is still **no published ASIC/FPGA for SDformerFlow-class dense event OF with spatiotemporal Swin spikeformer**. Closest neighbors are ASNA-Flow (SNN OF ASIC), EDFLOW (block-match camera), FireFly-T/Xpikeformer (spikeformer *classification*), ERAFT (frame RAFT FPGA).

**Survey anchor:** Gallego et al. *Event-based Vision: A Survey*, **IEEE TPAMI 2022**; Kryjak *Event-based vision on FPGAs — a survey*, arXiv:2407.08356 / DSD 2024 (~60 FPGA papers through mid-2024; OF still under-served vs filter/track/classify).

---

## 1. Strong remake ideas (≥12) — citation → transfer → C1*/C2* name

### Idea 1 — Adaptive Area-Event Slice Exposure (from EDFLOW ABMOF)

| Field | Content |
|---|---|
| **Citation** | Liu & Delbruck, *EDFLOW: Event Driven Optical Flow Camera With Keypoint Detection and Adaptive Block Matching*, **IEEE TCSVT 2022**, DOI:10.1109/TCSVT.2022.3156653; precursors ISCAS 2017 BMOF, BMVC 2019 ABMOF. |
| **Mechanism** | Accumulate events into **slices** with **area-event-count** exposure; **feedback** from average OF match distance adapts slice density; SFAST corners + multiscale 25×25 ABMOF @ 123 GOp/s; MVSEC accuracy ≈ EV-FlowNet at ~100× less power. |
| **HW status** | **True FPGA HW** (DAVIS+FPGA open Vivado). |
| **Transfer** | Replace fixed Tw voxel dump with **adaptive event-count / residual-driven Tw**; quiet tiles get longer integrate → fewer SPE wakes; busy tiles shorten Tw → OP-STW exact budget. |
| **Remake name** | **AdaptSlice-Tw** — **C1\*** front-end. |
| **vs packages** | **Enhances OP-STW + PRRC** (exposure = residual budget); **enhances EV-Wake**; does not replace HBG-RP. |
| **Novelty** | **8 / 10** (EDFLOW exists for ABMOF; **adaptive Tw for spikeformer OF** is new). |
| **Prove-by** | Tw histogram vs event rate on DSEC/MVSEC; AEE vs fixed Tw; PE wake vs always-fixed Tw. |

---

### Idea 2 — Spatial-Locality Sparse Neuromorphic OF Datapath (from ASNA-Flow)

| Field | Content |
|---|---|
| **Citation** | Wang et al., *ASNA-Flow: An Efficient Asynchronous Neuromorphic Accelerator for Real-Time Event-Based Optical Flow*, **IEEE TVLSI 2025**, DOI:10.1109/TVLSI.2025.3600953. |
| **Mechanism** | Algo–HW co-design for event OF; **spatial locality** enables sparse async compute; 104 FPS, 7.9 mW, **0.3 pJ/SOP**, TSMC 28 nm — claimed **first dedicated neuromorphic ASIC** for event OF. |
| **HW status** | **Silicon ASIC**. |
| **Transfer** | Tile PE wake only in **spatially clustered event neighborhoods** (not full H×W); locality bitmap feeds OP-STW / Motion-TTB packer; energy-proportional SOP matches ATLIF sparsity. |
| **Remake name** | **SpatLoc-Wake** — **C1\*** + **C2\*** fabric schedule. |
| **vs packages** | **Enhances OP-STW, EV-Wake, Motion-TTB**; complements MFBD; keep HBG-RP for payload. |
| **Novelty** | **7.5 / 10** (ASIC exists for SNN OF; remake is **locality bitmap → SDSA tile schedule**). |
| **Prove-by** | Spatial cluster radius vs AEE; pJ/SOP of locality-gated vs full-window SDSA; compare to ASNA-Flow published energy. |

---

### Idea 3 — Bitvector Hypothesis-Shift Coarse Flow Prior (from EventShiftFlow)

| Field | Content |
|---|---|
| **Citation** | Alonso Bizzi, Cladera, Taylor, *EventShiftFlow*, **arXiv:2605.28312** (2026); Artix-7 prototype, <2 kB storage, **0 DSP / 0 BRAM** core, 99.5% directional accuracy on RPG shapes_rotation. |
| **Mechanism** | Time-bin → 1-bit occupancy → shift-register grid → **parallel discrete velocity hypotheses** scored by popcount / cross-multiply (no dividers); sparse quantized velocity, not dense sub-pixel OF. |
| **HW status** | **True FPGA HW** (preliminary; sequential hyp variant published). |
| **Transfer** | Ultra-cheap **direction / |v| prior** for OP-STW, ECP-QKV, ARM-Acc apertures; hyp lanes ≈ ARM-Acc multi-hypothesis Acc; dense SDSA only when hyp residual high. |
| **Remake name** | **BitHyp-Prior** — **C1\*** predictor. |
| **vs packages** | **Enhances OP-STW + ECP-QKV + ARM-Acc**; optional **replace** naive zero-skip narrative for wake; does not replace full flow head. |
| **Novelty** | **8.5 / 10** as *first hyp-shift prior glued to spikeformer residual OF*. |
| **Prove-by** | Direction IoU vs coarse ground-truth; AEE with BitHyp gate vs always-on; LUT/FF of prior vs SPE savings. |

---

### Idea 4 — In-Sensor / Near-Sensor Event Compression Gate (from 3D-ISC)

| Field | Content |
|---|---|
| **Citation** | Krishnan et al., *3D-ISC*, **A-SSCC 2023**, DOI:10.1109/A-SSCC58667.2023.10347978; Hassan et al., *3-D In-Sensor Computing for Real-Time DVS Data Compression*, **IEEE LSSC 2024**, DOI:10.1109/LSSC.2024.3375110 (65 nm, 4-bit AE IMC, >10× compress, <6 mW on 256×256). |
| **Mechanism** | 3D-stacked DVS + IMC autoencoder compresses event volume before IO; footprint-matched tiles; continuous stream latency constraints. |
| **HW status** | **Silicon prototype** (in-sensor concept). |
| **Transfer** | Treat compressed latent / saliency as **token budget** into SPE; drop reconstructed quiet regions before SDSA; bandwidth-aware PRRC. |
| **Remake name** | **ISC-TokBudget** — **C1\*** capacity. |
| **vs packages** | **Enhances PRRC + HeatFlow-Tok + EV-Wake**; orthogonal to HBG-RP. |
| **Novelty** | **8 / 10** (chip exists; **compress→spikeformer OF token quota** is new). |
| **Prove-by** | Off-chip bandwidth vs AEE under AE latent; tile drop rate; energy of AE vs dense voxel DMA. |

---

### Idea 5 — Hybrid CIS+EVS Skip / Border-ROI Readout (from Guo stacked sensor)

| Field | Content |
|---|---|
| **Citation** | Guo et al., *A Three-Wafer-Stacked Hybrid 15-MPixel CIS + 1-MPixel EVS…*, **ISSCC 2023** / **JSSC 2023**, DOI:10.1109/JSSC.2023.3303154 (4.6 GEps, skip of connected-component interiors, ROI, flicker filters, on-chip ESP). |
| **Mechanism** | Shared optics CIS+EVS; **skip** non-border pixels of connected event sets; ROI / subsample; global activity monitor. |
| **HW status** | **Silicon sensor**. |
| **Transfer** | Event **edge/ROI** → exact SDSA; interior skip → propagate (OGEC-like); APS texture path for unmatched/occlusion fill. |
| **Remake name** | **SkipEdge-ROI** — **C1\*** sensor-aware front-end. |
| **vs packages** | **Enhances OGEC + EV-Wake**; **enhances MW-ΔBuf** unveil regions via CIS; keep spike path primary. |
| **Novelty** | **8 / 10** as *SDformer scheduler driven by sensor skip/ROI metadata*. |
| **Prove-by** | AEE with skip-interior vs full events; power with ROI-gated PE; hybrid APS fill vs event-only. |

---

### Idea 6 — Shift-Based Time-Surface / Histogram Encoder HW (from HOMI)

| Field | Content |
|---|---|
| **Citation** | H et al., *HOMI: Ultra-Fast EdgeAI platform for Event Cameras*, **arXiv:2508.12637** (IMX636 + ZU+ MPSoC, SLTS/SETS via bit-shifts, EVT 3.0 MIPI, RAMAN sparse CNN, 1 ms / 1000 fps gesture); Blachut & Kryjak **SPA 2023** HD frames; SSER **arXiv:2505.07556** GRU encoder FPGA. |
| **Mechanism** | Constant-time / constant-event accumulation; **shift-approx** linear/exponential time surfaces (no exp LUT); ping-pong BRAM; feeds sparse CNN. |
| **HW status** | **True end-to-end FPGA**. |
| **Transfer** | Harden SPE front-end as **SETS/SLTS or voxel** unit with shift ALUs; multi-channel polarity → ATLIF channels; constant-event mode = AdaptSlice peer. |
| **Remake name** | **ShiftTS-SPE** — **C1\*** encoder. |
| **vs packages** | **Enhances AdaptSlice-Tw + Δ-MaskPipe**; feeds ReMem-Tok; does not replace SDSA. |
| **Novelty** | **7.5 / 10** (HOMI exists for classify; **SPE for OF spikeformer** is remake). |
| **Prove-by** | AEE SETS vs voxel vs hist on DSEC; LUT of shift-TS vs LUT-ETS; end-to-end latency. |

---

### Idea 7 — TDE / TDE-3 Direction-Selective Front-End (bio prior)

| Field | Content |
|---|---|
| **Citation** | Gutierrez-Galan et al., *An event-based digital time difference encoder…*, **IEEE TNNLS 2021** (FPGA VHDL); TDE-3: *improved prior for OF in SNNs*, **Frontiers Neurosci. 2025** / arXiv:2402.11662 (Loihi2/SpiNNaker/FPGA compatible); Greatorex et al., *Event-Based Optical Flow Leveraging Precise Event Timing*, **CVPRF 2026**. |
| **Mechanism** | Facilitator–trigger **time-difference** → preferred-direction spikes; TDE-3 adds inhibition for textured scenes; timing/synaptic-gate OF without heavy training. |
| **HW status** | **FPGA + neuromorphic HW** for TDE primitive; **algo/system** for TDE-3 / Greatorex OF. |
| **Transfer** | TDE bank as **cheap direction field** before SDSA (parallel to BitHyp-Prior); ISI / spike-count velocity code seeds ARM-Acc hypotheses; skip SDSA when TDE confidence high. |
| **Remake name** | **TDE3-Prior** — **C1\*** bio front-end. |
| **vs packages** | **Enhances OP-STW + ARM-Acc + EESUC**; optional early-exit when TDE agrees with coarse flow. |
| **Novelty** | **9 / 10** for *TDE-3 + residual spikeformer OF co-design* (first-HW-class if hardened together). |
| **Prove-by** | AEE TDE-only vs TDE+SDSA residual; SOP when TDE gates deep blocks; textured-scene ablation (TDE-2 vs TDE-3). |

---

### Idea 8 — Temporal Motion Aggregation HW (from TMA — algo → first HW)

| Field | Content |
|---|---|
| **Citation** | Liu et al., *TMA: Temporal Motion Aggregation for Event-based Optical Flow*, **ICCV 2023**, DOI:10.1109/ICCV51070.2023.00888 / arXiv:2303.11629 (+6% acc, −40% time vs E-RAFT on DSEC-Flow). |
| **Mechanism** | Event **splitting** → dense intermediate correlations; **linear lookup** align motion features across spans; **motion pattern aggregation** emphasizes consistent patterns → fewer RAFT-like refinements. |
| **HW status** | **Algo-only (GPU) — first-HW candidate**. |
| **Transfer** | Split Tw into sub-slices; lookup = MFBD / Motion-TTB delivery; aggregation = early good flow → **EESUC / VGTS** fewer SDSA refinements. |
| **Remake name** | **TMA-Agg** — **C2\*** temporal fabric + **C1\*** early exit. |
| **vs packages** | **Enhances MFBD, Motion-TTB, EESUC, VGTS**; complements STH-Gate temporal heads. |
| **Novelty** | **9 / 10** as *first HW for TMA-style aggregation under spikes*. |
| **Prove-by** | Refinement count CDF vs AEE; energy vs E-RAFT-style full iterate; RTL of split+lookup+agg. |

---

### Idea 9 — Event-Queue Neighbor Search for Local Windows (from EvGNN)

| Field | Content |
|---|---|
| **Citation** | Yang, Kneip, Frenkel, *EvGNN*, **IEEE TCAS-AI 2024**, DOI:10.1109/TCASAI.2024.3520905 / arXiv:2404.19489 (KV260, 16 µs/event, N-CARS 87.8%). |
| **Mechanism** | Directed dynamic graphs; **event queues** find local neighbors in spatiotemporal range; layer-parallel GNN; edge-free single-hop storage. |
| **HW status** | **True FPGA HW** (classify, not OF). |
| **Transfer** | Queue = **hardware neighbor gather** for 3D Swin windows without densifying full voxel; per-event update of SDSA K/V candidates. |
| **Remake name** | **EvQ-Win** — **C2\*** window fabric. |
| **vs packages** | **Enhances MFBD + ReMem-Tok + TokMerge-OF**; complements SpatLoc-Wake. |
| **Novelty** | **8.5 / 10** as *EvGNN queues remade for OF SDSA windows*. |
| **Prove-by** | Neighbor-hit latency; AEE queue-radius ablation; BRAM of queues vs dense window SRAM. |

---

### Idea 10 — Submanifold Sparse Token Dataflow (from ESDA)

| Field | Content |
|---|---|
| **Citation** | Gao et al., *ESDA*, **FPGA 2024**, DOI:10.1145/3626202.3637558 / arXiv:2401.05626 (composable sparse modules, submanifold sparse conv, all-on-chip dataflow). |
| **Mechanism** | Parametrizable sparse layer modules; uniform sparse token–feature interface; keep activation sparsity through depth (submanifold). |
| **HW status** | **True FPGA HW** (event CNN classify). |
| **Transfer** | Propagate **event occupancy mask** through SPE→STSF without densifying until flow head (DeltaCNN-like but event-native); composable STSF stages. |
| **Remake name** | **SubMan-Pipe** — **C1\***+**C2\*** pipeline. |
| **vs packages** | **Enhances Δ-MaskPipe + OP-STW**; can **partially replace** naive dense voxel narrative; keep HBG-RP for ATLIF payload. |
| **Novelty** | **8 / 10** as *submanifold pipe for spikeformer OF*. |
| **Prove-by** | Mask density per STSF stage; AEE with forced densify vs submanifold; throughput vs dense baseline. |

---

### Idea 11 — DVS-Triggered Exact-Path / NPU Wake (deepen EV-Wake)

| Field | Content |
|---|---|
| **Citation** | Cha et al., *Energy-Efficient Daily Surveillance… Event-based NPU Triggering*, **ISCAS 2025**, DOI:10.1109/ISCAS56072.2025.11043213; live demo DOI:10.1109/ISCAS56072.2025.11043578; receiver DOI:10.1109/ISCAS56072.2025.11044183 (31.5% energy, YOLOv3-Tiny 18 ms). |
| **Mechanism** | DVS ROI / scene-change detector **triggers** dense NPU; CIS for texture; always-on DVS, duty-cycled DNN. |
| **HW status** | **FPGA system HW**. |
| **Transfer** | Same trigger semantics for **exact-product / full SDSA** vs membrane reuse; night/low-light = event-primary mode. |
| **Remake name** | **EV-Wake+** (R3 deepen of R2 EV-Wake) — **C1\***. |
| **vs packages** | **Enhances OP-STW + OGEC + ECP-QKV**; hybrid mode with SkipEdge-ROI. |
| **Novelty** | **7.5 / 10** (demos exist; **OF residual trigger policy** is the remake). |
| **Prove-by** | 24h-style energy with duty-cycled SDSA; AEE hybrid vs always-on; false-trigger rate. |

---

### Idea 12 — Multi-Sensor Fusion SoC Path (from Kraken / SNE)

| Field | Content |
|---|---|
| **Citation** | Di Mauro et al., *Kraken* multi-sensor fusion SoC (22 nm FDX), HotChips / ETH reports; SNE *energy-proportional sparse event conv*, arXiv:2204.10687 (0.221 pJ/SOP class); Xu et al. SENECA OF compare arXiv:2407.20421. |
| **Mechanism** | Heterogeneous SoC: **SNE** for sparse event SCNN + RISC-V cluster + TNN for frames; power-gate accelerators; COO event encoding. |
| **HW status** | **Silicon SoC**. |
| **Transfer** | Dual-rail system story: event-sparse STSF on SNE-like engine; APS/frame refine on ANN path for OGEC; power-gate deep SDSA. |
| **Remake name** | **SNE-DualPath** — system / **C2\*** power domains. |
| **vs packages** | **Enhances HBG-RP dual-rail story + OGEC**; does not replace ATLIF MAC semantics. |
| **Novelty** | **7 / 10** (SoC exists; **OF spikeformer dual-path** claim needs careful scoping). |
| **Prove-by** | Power-gate residency vs event rate; AEE event-only vs fusion; pJ/SOP vs Kraken/ASNA published. |

---

### Idea 13 — Plane-Fit / History-Window OF Lite Prior (from Aung + hARMS)

| Field | Content |
|---|---|
| **Citation** | Aung, Teo, Orchard, *Event-based Plane-fitting Optical Flow… FPGA*, **ISCAS 2018**, DOI:10.1109/ISCAS.2018.8351588 (100 M plane-fits/s, sub-µs); Stumpp et al., *hARMS*, **IEEE Access 2022**, DOI:10.1109/ACCESS.2022.3172396 (small event history, resolution-independent latency); Aerospace 2026 plane-fit Savitzky–Golay FPGA ~500 Kevts/s. |
| **Mechanism** | Local (x,y,t) plane slope → OF; store **small recent history**, not full frame; aperture-robust multi-scale (hARMS). |
| **HW status** | **True FPGA HW**. |
| **Transfer** | Lite plane-fit = another **coarse Acc prior** for quiet regions; history buffer ↔ ReMem / MW-ΔBuf; multi-scale ↔ PRRC. |
| **Remake name** | **PlaneHist-Prior** — **C1\*** lite path. |
| **vs packages** | **Enhances ARM-Acc + ReMem-Tok + PRRC**; backup when BitHyp/TDE disagree. |
| **Novelty** | **7 / 10** (priors known; **gated exact SDSA on plane residual** is remake). |
| **Prove-by** | AEE plane-only vs residual refine; history depth vs SRAM; latency independence of resolution. |

---

### Idea 14 — Spikeformer Attention Dual-Engine for OF SDSA (from FireFly-T / Xpikeformer)

| Field | Content |
|---|---|
| **Citation** | FireFly-T, arXiv:2505.12771 (sparse engine + **binary AND-PopCount attention** engine, FPGA overlay); Xpikeformer arXiv:2408.08794 / IEEE journal (AIMC FF + stochastic spiking attention); SPARTA **ICCAD 2025** DOI:10.1109/ICCAD66269.2025.11240724 (token skip + ReRAM CIM); ASTER arXiv:2511.06770 (event-driven spiking transformer PIM). |
| **Mechanism** | Specialized engines for **spike attention** vs linear layers; token skip; sparsity-aware routing. |
| **HW status** | **True HW** for *classification / ImageNet–DVS* — **not OF**. |
| **Transfer** | Binary engine ↔ SDSA mask/AND path; keep **real ATLIF payload** on second rail (**HBG-RP / SMAM-RP**); token skip ↔ TokMerge-OF / HeatFlow. |
| **Remake name** | **DualEng-SDSA** — **C2\*** attention. |
| **vs packages** | **Enhances HBG-RP + SMAM-RP + SP-Gate + STH-Gate**; **do not claim** “first spikeformer HW.” |
| **Novelty** | **8 / 10** as *first dual-engine SDSA for dense event OF*. |
| **Prove-by** | Energy binary-attn vs dense QK on ep34; AEE with ATLIF payload rail; compare FireFly-T-style LUT6 PopCount in OF windows. |

---

### Idea 15 — Prediction Early-Exit from Frame RAFT HW → Event Residual (from ERAFT)

| Field | Content |
|---|---|
| **Citation** | *ERAFT* FPGA RAFT accelerator, **ISCAS 2025**, DOI:10.1109/ISCAS56072.2025.11043529 (prediction skip iterations, 86 FPS @ 640×480 VCK190); algo cousin E-RAFT **3DV 2021** (event voxels, DSEC-Flow). |
| **Mechanism** | Lightweight RAFT + **predict** when further updates unnecessary. |
| **HW status** | **Frame OF FPGA HW**; E-RAFT = **algo** on events. |
| **Transfer** | Port prediction gate to **spike residual / membrane Δ** (strengthen EESUC); E-RAFT corr volume lessons → optional cost-lite path before SDSA. |
| **Remake name** | **PredExit-OF** — **C1\***/**C2\*** exit. |
| **vs packages** | **Enhances EESUC + TL-Exit + VGTS**; cite ERAFT honestly as frame HW inspiration. |
| **Novelty** | **8 / 10** as *first event-spikeformer residual early-exit ASIC/FPGA*. |
| **Prove-by** | Exit-rate vs AEE on DSEC; energy vs full-T; correlate exit with BitHyp/TDE confidence. |

---

## 2. Package matrix — enhance vs replace (R3 vs R1+R2)

| Existing package | Role today | ROUND 3 interaction |
|---|---|---|
| **OP-STW / DPP-Skip** | Direction/residual → spike-tile wake | **Enhance** with BitHyp-Prior, TDE3-Prior, SpatLoc-Wake, AdaptSlice-Tw, PlaneHist-Prior |
| **PRRC** | Pyramid residual budget | **Enhance** with AdaptSlice-Tw, ISC-TokBudget, PlaneHist multi-scale |
| **OGEC** | Occlusion: exact vs propagate | **Enhance** with SkipEdge-ROI, SNE-DualPath APS fill, EV-Wake+ |
| **HBG-RP** | Binary gate + real payload | **Keep**; DualEng-SDSA binary rail must not absorb ATLIF — **do not replace** |
| **ADP-MAC** | Bilateral bit sparsity | **Enhance** with SubMan-Pipe occupancy |
| **ARM-Acc** | Aperture multi-hypothesis Acc | **Enhance** with BitHyp lanes, TDE directions, PlaneHist |
| **MFBD / Motion-TTB** | Motion-bundle delivery | **Enhance** with TMA-Agg lookup, EvQ-Win gather |
| **SP-Gate / STH-Gate** | Attention-mass / typed sparse | **Enhance** with DualEng-SDSA |
| **EESUC / VGTS** | Residual / amplitude time skip | **Enhance** with TMA-Agg early good flow, PredExit-OF |
| **ReMem-Tok / MW-ΔBuf** | Membrane reuse / warp residual | **Enhance** with PlaneHist history, ShiftTS-SPE, AdaptSlice |
| **ECP-QKV** | Eager corr before projection | **Enhance** with BitHyp / TDE cheap priors as ECP features |
| **EV-Wake** | DVS energy gate | **Deepen → EV-Wake+** with ISCAS’25 trigger + SkipEdge-ROI |

**Nothing in this round should *replace* HBG-RP or ARM-Acc.** Strongest *narrative upgrades*: **TDE3-Prior**, **TMA-Agg**, **BitHyp-Prior**, **AdaptSlice-Tw**, **EvQ-Win**, **DualEng-SDSA**.

---

## 3. Recommended C1* / C2* remake bundles (event-stack round)

### C1* “Event-Native Residual Front-End” (add to R2 C1*)

1. **AdaptSlice-Tw** — adaptive event-count Tw exposure (EDFLOW).  
2. **BitHyp-Prior + TDE3-Prior** — dual cheap direction/velocity priors.  
3. **ShiftTS-SPE / ISC-TokBudget** — HW encode + bandwidth token quota.  
4. **SkipEdge-ROI + EV-Wake+** — sensor skip/ROI + duty-cycled exact path.  
5. Keep **OP-STW + MW-ΔBuf + OGEC + ECP-QKV + PRRC**.

**Claim sketch:** *First spikeformer-OF frontend that couples adaptive event-slice exposure with bio/bitvector coarse priors and sensor ROI/skip metadata to schedule exact-product budgets.*

### C2* “Event-Sparse SpatioTemporal Fabric” (add to R2 C2*)

1. **TMA-Agg** — temporal split / lookup / pattern aggregation under spikes.  
2. **EvQ-Win + SpatLoc-Wake + SubMan-Pipe** — queue neighbors, locality, submanifold mask pipe.  
3. **DualEng-SDSA** — FireFly-T-style binary attn engine + ATLIF payload rail.  
4. Keep **HBG-RP + SMAM-RP + ADP-MAC + ARM-Acc/MFBD + STH-Gate**.

**Claim sketch:** *C2* is not multiply reorder — it is **event-submanifold sparse fabric + TMA-style temporal aggregation + dual-rail spike attention**, with ATLIF payloads preserved.*

---

## 4. Ranked shortlist (R3 remakes for ISCAS story)

| Rank | Remake | Why | Novelty |
|---|---|---|---|
| 1 | **TDE3-Prior** | Bio timing prior native to events; HW primitive exists; OF co-design open | 9 |
| 2 | **TMA-Agg** | Strong DSEC gains algo-only; clear first-HW for spike OF | 9 |
| 3 | **BitHyp-Prior** | Ultra-cheap FPGA prior; perfect OP-STW/ECP companion | 8.5 |
| 4 | **EvQ-Win** | Proven microsecond neighbor HW → SDSA windows | 8.5 |
| 5 | **AdaptSlice-Tw** | EDFLOW-proven adaptive exposure → Tw control | 8 |
| 6 | **DualEng-SDSA** | Reuse FireFly-T binary engine under HBG-RP | 8 |
| 7 | **SubMan-Pipe** | ESDA → mask pipe through STSF | 8 |
| 8 | **ISC-TokBudget / SkipEdge-ROI** | Sensor-silicon mechanisms → token/ROI policy | 8 |
| 9 | **SpatLoc-Wake** | ASNA-Flow locality → tile schedule | 7.5 |
| 10 | **ShiftTS-SPE** | HOMI encoder → SPE | 7.5 |
| 11 | **PredExit-OF** | ERAFT prediction → spike residual exit | 8 |
| 12 | **PlaneHist-Prior / SNE-DualPath** | Lite OF / fusion system story | 7 |

---

## 5. First-HW candidate shortlist (claim carefully)

| Candidate | Source | Safe claim if you build HW |
|---|---|---|
| TMA under spikes | TMA ICCV’23 | First **accelerator for TMA-style temporal motion aggregation** in event OF (esp. spikeformer) |
| SDformerFlow / spikeformer dense OF | SDformerFlow ICPR’24 | First **accelerator for spatiotemporal Swin spikeformer optical flow** |
| TDE-3 + residual transformer OF | TDE-3 2025 + Greatorex CVPRF’26 | First **co-designed TDE-3 prior + spikeformer residual OF** datapath |
| Bitvector hyp-shift → transformer wake | EventShiftFlow 2026 | First **hyp-shift prior gating spikeformer OF** (cite EventShiftFlow for prior) |
| EvGNN queues for OF windows | EvGNN 2024 | First **event-queue neighbor unit for OF SDSA windows** |
| Submanifold pipe for OF spikeformer | ESDA FPGA’24 | First **submanifold sparse pipe through spikeformer OF** |

---

## 6. DO NOT CLAIM

- “First event optical-flow hardware” — **EDFLOW, ASNA-Flow, hARMS, Aung plane-fit, TrueNorth OF, EventShiftFlow** exist.  
- “First event camera / DVS silicon” — decades of ISSCC/JSSC (Lichtsteiner, Finateu ISSCC’20, Guo ISSCC’23, …).  
- “First spiking transformer hardware” — **Xpikeformer, FireFly-T, SPARTA, ASTER, Bishop, SMAM** exist.  
- “First hybrid DVS–CIS system” — **Guo stacked sensor; ISCAS 2025 fusion; Kraken; Lele ISCAS’22**.  
- “First time-surface / event-frame FPGA” — **HOTS FPGA, Blachut SPA’23, HOMI, SSER**.  
- “First sparse event CNN FPGA” — **NullHop+DVS, ESDA, RAMAN/HOMI**.  
- Do **not** conflate **E-RAFT (algo, events)** with **ERAFT (FPGA, frames)** — different papers.  
- Do **not** claim FlowAcc/ERAFT as event OF chips.  
- Avoid hanging novelty on **generic AER filtering** alone (many FPGA filters 2015–2023).

---

## 7. Algo-only vs true HW (quick reference)

| True HW (chip/FPGA system) | Algo-only / first-HW for OF-spikeformer |
|---|---|
| Finateu ISSCC’20; Guo JSSC’23; 3D-ISC LSSC’24 | E-RAFT 3DV’21; TMA ICCV’23 |
| EDFLOW TCSVT’22; Aung ISCAS’18; EventShiftFlow’26 | SDformerFlow ICPR’24; Greatorex CVPRF’26 |
| ASNA-Flow TVLSI’25; hARMS Access’22; Kraken/SNE | VideoFlow / FlowFormer (frame transformer OF) |
| HOMI; ESDA FPGA’24; EvGNN; EFGCN; Eventor DAC’22 | EvT CVPRW’22 (event transformer classify) |
| FireFly-T; Xpikeformer; SPARTA; ISCAS’25 DVS-CIS | Sparse VideoGen etc. (R2 — not event-native) |
| ERAFT ISCAS’25; FlowAcc DATE’22 (**frame** OF) | Spike-FlowNet ECCV’20 (hybrid SNN–ANN OF) |

---

## 8. Known gaps vs dense OF-SNN-Transformer ASIC

1. **No end-to-end SDformerFlow accelerator** (SPE + STSF SDSA + multi-scale flow head).  
2. Event OF HW is mostly **block-match / plane-fit / small SNN**, not **Swin spikeformer**.  
3. Spikeformer HW targets **classification**, not **dense regression OF**.  
4. Encoding HW (TS/voxel) rarely **shares occupancy masks** into deep sparse transformers.  
5. Sensor skip/ROI / compress metadata **not consumed** by OF PE schedulers.  
6. TMA / VideoFlow-style **temporal aggregation** not hardened.  
7. Hybrid APS path used for detect, not **occlusion-aware OF refine (OGEC)**.  
8. Evaluation fragmentation (custom sequences) — prefer **DSEC-Flow / MVSEC** for prove-by.

---

## 9. Suggested prove-by ladder (shared)

1. **Data stats:** event-rate vs Tw; spatial cluster size; TDE/BitHyp direction agreement vs GT; mask density through STSF; TMA early-flow quality.  
2. **Cycle-accurate / FPGA:** PE wake, SRAM/queue traffic, pJ/SOP vs always-on, R2 C1*/C2*, new R3 blocks.  
3. **Accuracy:** AEE / Fl-all on DSEC-Flow & MVSEC; ablate AdaptSlice, BitHyp, TDE3, TMA-Agg, EvQ radius.  
4. **Fair baselines (cite, don’t invent):** EDFLOW, ASNA-Flow, E-RAFT (GPU), TMA (GPU), ERAFT (frame), FireFly-T-style attn energy.

---

## 10. C1* / C2* upgrade matrix (R3 delta)

| Layer | R2 package | R3 add / deepen | Engineering note |
|---|---|---|---|
| Sensor/IO | EV-Wake | **EV-Wake+**, **SkipEdge-ROI**, **ISC-TokBudget** | Consume skip/ROI/compress bits in wake CSR |
| Encode | Δ-MaskPipe / DF-OS | **AdaptSlice-Tw**, **ShiftTS-SPE** | Ping-pong voxel/TS; event-count FSM |
| Coarse prior | ECP-QKV, OP-STW | **BitHyp-Prior**, **TDE3-Prior**, **PlaneHist-Prior** | Parallel prior lanes → wake bitmap |
| Temporal | MW-ΔBuf, MFBD, Motion-TTB | **TMA-Agg** | Split/lookup/agg microcode |
| Sparse fabric | STH-Gate, SubMan n/a | **SpatLoc-Wake**, **SubMan-Pipe**, **EvQ-Win** | Queues + occupancy stickiness |
| Attention | HBG-RP, SMAM-RP, DualEng n/a | **DualEng-SDSA** | Binary attn engine + payload MAC |
| Exit | EESUC, VGTS | **PredExit-OF** (+ TMA early) | Residual + prior agreement |

---

## 11. Sources (verified; no fabricated venues)

### Sensors / in-sensor
1. Finateu et al., Prophesee/Sony 1280×720 EVS, **ISSCC 2020**, pp. 112–114.  
2. Guo et al., Hybrid 15 MP CIS + 1 MP EVS, **ISSCC 2023** / **JSSC 2023**, DOI:10.1109/JSSC.2023.3303154.  
3. Krishnan et al., 3D-ISC, **A-SSCC 2023**, DOI:10.1109/A-SSCC58667.2023.10347978.  
4. Hassan et al., 3-D ISC DVS compression, **IEEE LSSC 2024**, DOI:10.1109/LSSC.2024.3375110.  
5. 3DS-ISC time-surface, arXiv:2512.20073.

### Encode / end-to-end event edge
6. Blachut & Kryjak, HD event frames SoC FPGA, **SPA 2023**.  
7. HOMI, arXiv:2508.12637.  
8. SSER (self-supervised event repr. FPGA), arXiv:2505.07556.  
9. ESDA, **FPGA 2024**, DOI:10.1145/3626202.3637558 / arXiv:2401.05626.  
10. EvGNN, **IEEE TCAS-AI 2024**, DOI:10.1109/TCASAI.2024.3520905 / arXiv:2404.19489.  
11. EFGCN, arXiv:2406.07318 / **J. Syst. Arch. 2026**.  
12. Eventor EMVS, **DAC 2022**, DOI:10.1145/3489517.3530452.  
13. NullHop, **IEEE TNNLS 2019**; DVS+NullHop **ICONS 2021**.  
14. Kryjak, Event-based vision on FPGAs survey, arXiv:2407.08356 / **DSD 2024**.

### Event OF HW
15. Liu & Delbruck, BMOF, **ISCAS 2017**, DOI:10.1109/ISCAS.2017.8050295.  
16. Aung, Teo, Orchard, plane-fit OF FPGA, **ISCAS 2018**, DOI:10.1109/ISCAS.2018.8351588.  
17. Liu & Delbruck, EDFLOW, **IEEE TCSVT 2022**, DOI:10.1109/TCSVT.2022.3156653.  
18. Stumpp et al., hARMS, **IEEE Access 2022**, DOI:10.1109/ACCESS.2022.3172396.  
19. Wang et al., ASNA-Flow, **IEEE TVLSI 2025**, DOI:10.1109/TVLSI.2025.3600953.  
20. EventShiftFlow, arXiv:2605.28312.  
21. Haessig et al., spiking OF TrueNorth, **IEEE TBioCAS 2018**.  
22. Aerospace 2026 neuromorphic OF FPGA, DOI:10.1109/AERO66936.2026.11519863.

### Hybrid / fusion / SoC
23. Cha et al., DVS-CIS NPU trigger, **ISCAS 2025**, DOI:10.1109/ISCAS56072.2025.11043213 (+ demos 11043578 / 11044183).  
24. Lele & Raychowdhury, fuse frame+event OF, **ISCAS 2022**.  
25. Di Mauro et al., Kraken / SNE, HotChips + arXiv:2204.10687 / arXiv:2209.01065.  
26. Xu et al., SENECA OF, arXiv:2407.20421.

### Event OF / spikeformer algorithms
27. Zhu et al., EV-FlowNet, **RSS 2018**.  
28. Gehrig et al., E-RAFT, **3DV 2021**, DOI:10.1109/3DV53792.2021.00030.  
29. Liu et al., TMA, **ICCV 2023**, DOI:10.1109/ICCV51070.2023.00888.  
30. Lee et al., Spike-FlowNet, **ECCV 2020**.  
31. Tian & Andrade-Cetto, SDformerFlow, **ICPR 2024** / arXiv:2409.04082.  
32. Sabater et al., Event Transformer (EvT), **CVPRW 2022**, DOI:10.1109/CVPRW56347.2022.00301.  
33. TDE-3, **Frontiers Neurosci. 2025** / arXiv:2402.11662; Gutierrez-Galan TDE, **IEEE TNNLS 2021**.  
34. Greatorex et al., timing OF, **CVPRF 2026**.  
35. Gallego et al., Event-based Vision Survey, **IEEE TPAMI 2022**, DOI:10.1109/TPAMI.2020.3008413.

### Frame OF HW + spikeformer HW (transfer only)
36. Ling et al., FlowAcc, **DATE 2022**, DOI:10.23919/DATE54114.2022.9774506.  
37. ERAFT (frame RAFT FPGA), **ISCAS 2025**, DOI:10.1109/ISCAS56072.2025.11043529.  
38. FireFly-T, arXiv:2505.12771; Xpikeformer, arXiv:2408.08794; SPARTA, **ICCAD 2025**, DOI:10.1109/ICCAD66269.2025.11240724; ASTER, arXiv:2511.06770.

---

*End of ROUND 3 report. Feed into synthesis / C1* C2* microarch sketches; do not modify `hw_autoresearch_nts07`.*
