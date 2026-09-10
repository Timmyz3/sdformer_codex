# ROUND 4 — ISSCC / VLSI / HotChips Edge NPUs → SDformer C1*/C2* Transfer

**Date:** 2026-09-05 (Asia/Shanghai)  
**Scope:** ~2019–2026 · ISSCC digests · VLSI · HotChips slides/papers · JSSC follow-ups · select ISCA/MICRO edge-NPU cousins when they seeded silicon lineages  
**Goal:** Extract **mechanisms transferable** from **mobile/edge AI accelerators, DNN NPUs, vision NPUs, sparse/quantized edge chips** onto optical-flow SNN-Transformer (SDformer) remake packages — **not** “build another mobile NPU,” and **not** marketing TOPS.  
**Prior packages (do not replace cores):**  
- **C1\***: OP-STW / PRRC / OGEC / **ECP-QKV** / **MW-ΔBuf** / EV-Wake / HeatFlow-Tok (+ R3: MW-CIM-TileGate / Het-OF-CIM / TDE3-Prior / …)  
- **C2\***: **HBG-RP** / ADP-MAC / ARM-Acc / MFBD / SP-Gate / **SMAM-RP** / **Motion-TTB** / **OF-ECP** / **STH-Gate** (+ R3: DualRail-CIM / TMA-Agg / EvQ-Win / …)  
**Hard rules:**
- **NO fabricated venues/citations.** Every entry below has a real DOI, HotChips PDF, or arXiv that was checked this round.
- Prefer **enhance** over replace for C1*/C2* names.
- Do **not** modify Codex hw under `hw_autoresearch_nts07` — this note only feeds ideafromai remake naming.
- Do **not** claim first edge NPU / first sparse NPU / generic peak TOPS.

---

## 0. How to read this note

For each **strong transfer**:
- **Citation** | **Mechanism** | **HW status** | **Transfer to OF-SNN-T** | **Remake name** | **vs C1\*/C2\*** | **Novelty 1–10** | **Prove-by**

**Saturation legend:**

| Saturated (do NOT claim as ISCAS headline) | Still-open OF/spike hooks |
|---|---|
| Generic activation zero-skip / N:M structured sparsity | Motion-/residual-/occlusion-keyed skip + wake bitmap |
| Peak TOPS or TOPS/W of flagship mobile NPU | System mJ/frame or AEE-vs-energy under OF sparsity |
| “Systolic / WS / OS dataflow” taxonomy alone | Irregular Motion-TTB / sparse SDSA NoC schedule |
| Mixed-precision INT4/8 support as sole claim | PRRC / ATLIF payload **predicated** bitwidth from residual confidence |
| Always-on vision SoC marketing | Event/CIS cascade that **binds** EV-Wake → ECP-QKV → exact SDSA |
| Edge Transformer µJ/token tables | OF early-exit / head-type DVFS (not NLP entropy alone) |

**Cross-benchmark intent:** steal **control loops** (early-exit, predication, port remap, saliency cascade, hetero digital/AiMC schedule), not CNN classification benchmarks.

---

## 1. Landscape table (clusters → remake hooks)

| Cluster | Representative works (verified) | Transferable mechanism | Remake names (this note) | Novelty band |
|---|---|---|---|---|
| Sparse / zero-skip mobile NPU | Samsung ISSCC’19 butterfly NPU; Samsung ISSCC’21 6K-MAC FM-sparsity NPU (+ ISCA’21 companion); SCNN ISCA’17; Cambricon-X MICRO’16 | Feature-map zero-skip engines; dynamic port assign; compressed sparse delivery | **FMSkip-OPSTW**, **DynPort-MFBD**, **CartProd-TTB** | 6.5–8.5 |
| Mixed-precision / bit-serial / bit-fusion | UNPU ISSCC’18; LNPU ISSCC’19; Stripes MICRO’16; Bit Fusion MICRO’18; Keller VLSI’22 / JSSC’23 per-vector INT4; Tambe ISSCC’23 FP4/8 predication | Precision scales with confidence / layer; per-vector scale; bit-serial lanes | **PRRC-BitFuse**, **PVScale-ATLIF**, **MP-Pred-PRRC** | 7–8.5 |
| Near-/in-memory edge SoC | DIANA ISSCC’22 / JSSC’23; TinyVers VLSI’22 / JSSC’23; Yue ISSCC’21 CIM zero-skip | Het digital + AiMC; state-retentive always-on; block-wise CIM skip | **HetSchedule-OF** (→ Het-OF-CIM), **Retain-Prior-eMRAM**, **BlkSkip-CIM-Wake** | 7–9 |
| Attention / Transformer edge engines | Wang ISSCC’22 asymptotic-sparsity OoO; Tu ISSCC’22 BL-transpose CIM Transformer; Keller VLSI’22; Tambe ISSCC’23 STP; Kim C-Transformer ISSCC’24; Qin Ayaka JSSC’24 | Eager/asymptotic sparse speculate; early-exit; big-little DNN/spike; score pipelines | **Asymp-OoO-ECP**, **Entropy-EE-OF**, **BigLittle-DualRail**, **BLT-CIM-SDSA** | 7.5–9 |
| Vision / ISP+NPU SoCs | Tesla FSD HotChips’19 (NNA+ISP); Huawei DaVinci HotChips’19; AI-ISP TCSVT’24 tightly-coupled strip-tile; Marsellus ISSCC’23 endpoint SoC | ISP↔NPU low-latency strip/tile handoff; SRAM-resident kernels; cube+vector+scalar | **StripTile-OFPipe**, **SRAM-Resident-MFBD**, **CubeVec-HBG Map** | 7–8.5 |
| Always-on / wake / saliency cascade | CogniVision VLSI’24; Alpha-Vision ISSCC’26 AoV; HAR always-on TCAS-II’21; Glimpse MobiSys’17 (system ancestor) | Hierarchical saliency → novelty → DNN; motion-event skip imaging | **SalCascade-EVWake**, **MEDU-OPSTW**, **AoV-Cascade-CardH** | **8–9** |
| Early-exit / prediction skip | EdgeBERT MICRO’21 → Tambe ISSCC’23; QNAP ISSCC’21 / JSSC’21 EWC+ECP | Entropy EE + MP predication + VFS; effective-weight + psum prediction | **Entropy-EE-OF**, **EC-Psum-Gate**, **EWC-Cluster-MAC** | 7.5–9 |
| Classical edge DNN dataflow | Eyeriss ISSCC’16 / JSSC’17 / Eyeriss v2 JETCAS’19; Envision ISSCC’17; DNPU ISSCC’17; Thinker VLSI’17 / JSSC’18 | Row-stationary reuse; DVAFS; hybrid NN reconfig | **RS-MW-Reuse**, **DVAFS-PRRC**, **Thinker-Split-SDSA** | 5.5–7.5 |
| Sparse irregular NoC / fabric | Eyeriss v2 hierarchical mesh; SCNN accumulator array; Samsung ISCA’21 reconfig MAC | Load-balance irregular nonzero traffic | **IrrNoC-MotionTTB**, **ReconfMAC-STH** | 7–8.5 |
| Memory-centric / 3D (caution) | Neurocube ISCA’16 | Vault-driven PEs — low OF priority | *(skip)* | ≤5 |

---

## 2. Strong transfers (mechanisms → remakes)

### M1 — Feature-map sparsity-aware zero-skip mobile NPU (Samsung)

- **Citation:** Park, Jang, Lee, Lee, Lee, Jung, et al. — *9.5 A 6K-MAC Feature-Map-Sparsity-Aware Neural Processing Unit in 5nm Flagship Mobile SoC* — **ISSCC 2021**, pp. 152–154 — DOI [10.1109/ISSCC42613.2021.9365928](https://doi.org/10.1109/isscc42613.2021.9365928). Companion architecture paper: Jang/Park et al. — *Sparsity-Aware and Re-configurable NPU Architecture for Samsung Flagship Mobile SoC* — **ISCA 2021** — DOI [10.1109/ISCA52012.2021.00011](https://doi.org/10.1109/isca52012.2021.00011). Prior: Song et al. — *7.1 An 11.5TOPS/W 1024-MAC Butterfly Structure Dual-Core Sparsity-Aware Neural Processing Unit in 8nm Flagship Mobile SoC* — **ISSCC 2019** — DOI [10.1109/ISSCC.2019.8662476](https://doi.org/10.1109/isscc.2019.8662476).
- **Mechanism:** Energy-efficient **inner-product engine that skips zero input feature-map elements**; **reconfigurable MAC array**; **dynamic internal memory port assignment** to maximize on-chip bandwidth under sparse traffic; mixed-precision arithmetic. Silicon in 5 nm flagship SoC (measured FPS / TOPS/W on quantized Inception-v3 in ISCA companion).
- **HW status:** **Silicon** (commercial mobile SoC + measured silicon reports).
- **Transfer:** OF residual / event tiles are **activation-sparse and non-stationary**. Map FM zero-skip **controller** onto OP-STW wake bitmap (skip = no PE clock for zero residual amp), and port remapping onto MFBD / ADP-MAC banks when Motion-TTB density spikes at occlusion.
- **Remake names:** **FMSkip-OPSTW** (C1\*) · **DynPort-MFBD** (C2\*).
- **vs C1\*/C2\*:** **Enhance** OP-STW (add FM/residual zero-detect microarch, not replace direction wake) · **Enhance** MFBD/ADP-MAC (port remap). Orthogonal to HBG-RP semantics.
- **Novelty:** **7.5/10** (FM zero-skip is saturated as a slogan; **OF residual + Motion-TTB–driven dynamic ports** is not).
- **Prove-by:** Cycle-accurate PE utilization vs always-on MAC under ep34 residual sparsity; ablate DynPort vs static banks at occlusion bursts; report AEE vs energy.

### M2 — Entropy early-exit + mixed-precision predication + sentence V/F (Tambe STP)

- **Citation:** Tambe, Zhang, Hooper, Jia, Whatmough, et al. — *22.9 A 12nm 18.1TFLOPs/W Sparse Transformer Processor with Entropy-Based Early Exit, Mixed-Precision Predication and Fine-Grained Power Management* — **ISSCC 2023**, pp. 342–343 — PDF [sld.cs.columbia.edu/pubs/tambe_isscc23.pdf](https://sld.cs.columbia.edu/pubs/tambe_isscc23.pdf). Algorithm ancestor: Tambe et al. — *EdgeBERT* — **IEEE MICRO 2021** — arXiv [2011.14203](https://arxiv.org/abs/2011.14203).
- **Mechanism:** Per-input **entropy** after early Transformer layers → (1) **early exit** when confident, (2) **predicate FP4 vs FP8** MAC path, (3) **latency-bounded V/F scaling**, plus attention-head null-span skip and sparse bitmask encode/decode.
- **HW status:** **Silicon** 12 nm (4.60 mm² STP).
- **Transfer:** Replace NLP entropy with **OF residual confidence / AEE proxy / occlusion rate / event density**. Early-exit = skip deeper SDSA/MLP stages on calm tiles; MP predication = PRRC bit ladder; VFS = per-tile / per-frame DVFS under latency SLA.
- **Remake names:** **Entropy-EE-OF** · **MP-Pred-PRRC** · **TileVFS-Motion**.
- **vs C1\*/C2\*:** **Enhance** PRRC / HeatFlow-Tok / SP-Gate · **Orthogonal system loop** over ECP-QKV (ECP predicts *before* QKV; Entropy-EE decides *whether later blocks run*). Do **not** rename ECP-QKV.
- **Novelty:** **9/10** for OF-keyed EE+MP+VFS co-loop on Spikeformer; **5/10** if claimed as “early-exit Transformer HW” alone.
- **Prove-by:** Frame-level energy histograms (calm vs high-motion); exit-layer distribution vs AEE budget; ablate EE / MP / VFS independently.

### M3 — Asymptotic sparsity speculation + OoO Transformer (Wang)

- **Citation:** Wang, Qin, Deng, Wei, Zhou, Fan, et al. — *A 28nm 27.5TOPS/W Approximate-Computing-Based Transformer Processor with Asymptotic Sparsity Speculating and Out-of-Order Computing* — **ISSCC 2022** — DOI [10.1109/ISSCC42614.2022.9731686](https://doi.org/10.1109/isscc42614.2022.9731686). Follow-up line: Qin et al. — *Ayaka: A Versatile Transformer Accelerator With Low-Rank Estimation and Heterogeneous Dataflow* — **JSSC 2024** — DOI [10.1109/JSSC.2024.3397189](https://doi.org/10.1109/jssc.2024.3397189).
- **Mechanism:** **Speculate asymptotic attention sparsity** early; **out-of-order** schedule keeps predictor off critical path; approximate compute for sparse survivors.
- **HW status:** **Silicon** 28 nm (ISSCC’22); Ayaka JSSC follow-on.
- **Transfer:** Strengthen **ECP-QKV** scheduler: cheap TDE/coarse-flow predictor issues speculate-mask; OoO SDSA lanes absorb irregular Motion-TTB arrivals without stalling MW-ΔBuf refill.
- **Remake name:** **Asymp-OoO-ECP** (scheduler sibling of ECP-QKV).
- **vs C1\*/C2\*:** **Enhance ECP-QKV** (add OoO + asymptotic speculate); does **not** replace FACT-style eager correlation idea.
- **Novelty:** **8/10** with OF prior; **6/10** as generic asymptotic sparsity.
- **Prove-by:** Predictor hit/miss vs full QK; OoO vs in-order stall cycles under event bursts.

### M4 — Bitline-transpose CIM sparse Transformer (Tu)

- **Citation:** Tu, Wu, Wang, Liang, Liu, Ding, et al. — *A 28nm 15.59µJ/Token Full-Digital Bitline-Transpose CIM-Based Sparse Transformer Accelerator with Pipeline/Parallel Reconfigurable Modes* — **ISSCC 2022**, pp. 466–468 — DOI [10.1109/ISSCC42614.2022.9731645](https://doi.org/10.1109/isscc42614.2022.9731645).
- **Mechanism:** Full-digital CIM for Transformer; **bitline-transpose** path avoids large transpose buffer for QKᵀ pipeline; pipeline/parallel reconfigurable modes; sparsity for attention compute.
- **HW status:** **Silicon** 28 nm.
- **Transfer:** SDSA needs dynamic K streams (not static CNN weights). BL-transpose maps to **Motion-TTB / MFBD** delivery into DualRail-CIM without off-array transpose SRAM.
- **Remake name:** **BLT-CIM-SDSA** (feeds R3 DualRail-CIM / XForm-Split-SDSA).
- **vs C1\*/C2\*:** **Enhance DualRail-CIM / Motion-TTB**; orthogonal to digital HBG-RP if CIM fabric not used.
- **Novelty:** **8/10** for OF dynamic-K transpose issue; **5/10** as “CIM Transformer” alone.
- **Prove-by:** Transpose-buffer energy share vs BLT path on SDSA tiles; accuracy under sparse attention masks.

### M5 — C-Transformer big-little DNN / Spiking-Transformer (Kim)

- **Citation:** Kim, Kim, Jo, Kim, Hong, Yoo — *20.5 C-Transformer: A 2.6–18.1µJ/Token Homogeneous DNN-Transformer/Spiking-Transformer Processor with Big-Little Network and Implicit Weight Generation for Large Language Models* — **ISSCC 2024**, pp. 368–370 — DOI [10.1109/ISSCC49657.2024.10454330](https://doi.org/10.1109/isscc49657.2024.10454330).
- **Mechanism:** **Homogeneous** support for dense DNN-Transformer and **Spiking-Transformer** paths; **big-little network**; implicit weight generation to cut EMA; measured µJ/token.
- **HW status:** **Silicon** (ISSCC’24).
- **Transfer:** Big path = dense occlusion / high-residual tiles (full ATLIF payload MAC); little path = calm tiles (spike-gate only / SMAM Mask-Add). Implicit W gen ↔ low-EMA FFN under PRRC.
- **Remake name:** **BigLittle-DualRail** (system map onto HBG-RP / SMAM-RP).
- **vs C1\*/C2\*:** **Enhance** HBG-RP / SMAM-RP / STH-Gate (path select); **do not claim** first spiking Transformer silicon (Bishop / SMAM / FireFly-T / C-Transformer exist).
- **Novelty:** **8/10** if path select is **OF residual / STH-typed**; **4/10** as “big-little Transformer.”
- **Prove-by:** % tiles on little vs big vs AEE; EMA energy share with/without implicit W.

### M6 — DIANA hybrid digital + analog IMC SoC

- **Citation:** Ueyoshi, Papistas, Houshmand, Sarda, Jain, Shi, et al. — *DIANA: An End-to-End Energy-Efficient Digital and ANAlog Hybrid Neural Network SoC* — **ISSCC 2022** — DOI [10.1109/ISSCC42614.2022.9731716](https://doi.org/10.1109/isscc42614.2022.9731716). JSSC: Houshmand et al. — *DIANA: An End-to-End Hybrid DIgital and ANAlog Neural Network SoC for the Edge* — **JSSC 2023** — DOI [10.1109/JSSC.2022.3214064](https://doi.org/10.1109/jssc.2022.3214064).
- **Mechanism:** RISC-V host + **precision-scalable digital NN core** + **1152×512 AiMC core**; shared memory; **layer-fused** schedule assigning high-precision / low-utilization layers to digital, dense CONV-like layers to AiMC; measured end-to-end TOPS/W on CIFAR/ImageNet.
- **HW status:** **Silicon** 22 nm.
- **Transfer:** Same hetero schedule as R3 **Het-OF-CIM**: frame/ref / high-bit ATLIF payload → digital DigiCIM/MAC; event/spike gates → AiMC or spike-CIM; scheduler keyed by OP-STW / STH-Gate, not ImageNet layer types.
- **Remake name:** **HetSchedule-OF** (operationalizes Het-OF-CIM).
- **vs C1\*/C2\*:** **Enhance Het-OF-CIM / DualRail-CIM**; orthogonal to pure-digital Card A/B.
- **Novelty:** **8.5/10** with OF modality schedule; **6/10** as hybrid IMC SoC.
- **Prove-by:** Layer/tile assignment trace (digital vs AiMC) under event+frame OF; system energy vs single-fabric baseline.

### M7 — TinyVers state-retentive extreme-edge SoC

- **Citation:** Jain, Giraldo, De Roose, Boons, Mei, Verhelst — *TinyVers: A 0.8–17 TOPS/W, 1.7 µW–20 mW, Tiny Versatile System-on-chip with State-Retentive eMRAM for Machine Learning Inference at the Extreme Edge* — **VLSI 2022** — DOI [10.1109/VLSITechnologyandCir46769.2022.9830409](https://doi.org/10.1109/vlsitechnologyandcir46769.2022.9830409). JSSC: Jain et al. — **JSSC 2023** — DOI [10.1109/JSSC.2023.3236566](https://doi.org/10.1109/jssc.2023.3236566).
- **Mechanism:** Extreme-edge SoC with **state-retentive eMRAM**, wide dynamic power range (µW idle → mW active), versatile ML inference.
- **HW status:** **Silicon**.
- **Transfer:** Retain **TDE3-Prior / coarse flow / last Motion-TTB dictionary** across always-on sleep; wake only EV-Wake / MEDU path. System claim for Card H.
- **Remake name:** **Retain-Prior-eMRAM**.
- **vs C1\*/C2\*:** **Enhance EV-Wake / TDE3-Prior** (storage); orthogonal to SDSA datapath.
- **Novelty:** **7.5/10** (retentive ML SoC known; **OF prior retention across AoV sleep** underclaimed).
- **Prove-by:** Cold-start vs retain-prior AEE after N sleep frames; idle power with retained priors.

### M8 — QNAP effective-weight convolution + error-compensation prediction

- **Citation:** Mo, Zhu, Hu, Wang, Li, Li, et al. — *9.2 A 28nm 12.1TOPS/W Dual-Mode CNN Processor Using Effective-Weight-Based Convolution and Error-Compensation-Based Prediction* — **ISSCC 2021**, pp. 146–148 — DOI [10.1109/ISSCC42613.2021.9365943](https://doi.org/10.1109/isscc42613.2021.9365943). JSSC: Mo et al. — *A 12.1 TOPS/W Quantized Network Acceleration Processor…* — **JSSC 2021** — DOI [10.1109/JSSC.2021.3113569](https://doi.org/10.1109/jssc.2021.3113569).
- **Mechanism:** **Effective-weight-based convolution (EWC):** group unique weights, accumulate activations first, multiply once. **Error-compensation-based prediction (ECP)** skips unimportant partial sums (ReLU-aware) with trained compensation — **distinct naming from FACT ECP-QKV**.
- **HW status:** **Silicon** 28 nm.
- **Transfer:** EWC → cluster repeated ATLIF scales / shared hyp weights in ARM-Acc. Prediction skip → gate useless Acc adds before payload MAC (**not** QKV eager correlation — keep names separate).
- **Remake names:** **EWC-Cluster-MAC** · **EC-Psum-Gate**.
- **vs C1\*/C2\*:** **Enhance ADP-MAC / ARM-Acc**; **orthogonal** to ECP-QKV (different “ECP” expansion — document acronym collision explicitly).
- **Novelty:** **7.5/10**.
- **Prove-by:** Multiplier count reduction under hyp-weight reuse; psum-skip vs AEE on ReLU/spike-fire boundaries.

### M9 — Per-vector scaled INT4 Transformer engine (Keller)

- **Citation:** Keller, Venkatesan, Dai, Tell, Zimmer, Dally, Gray, Khailany — *A 17–95.6 TOPS/W Deep Learning Inference Accelerator with Per-Vector Scaled 4-bit Quantization for Transformers in 5nm* — **VLSI 2022** — DOI [10.1109/VLSITechnologyandCir46769.2022.9830232](https://doi.org/10.1109/vlsitechnologyandcir46769.2022.9830232) (also cited in Tambe ISSCC’23 refs). JSSC: Keller et al. — *A 95.6-TOPS/W … Per-Vector Scaled 4-bit Quantization in 5 nm* — **JSSC 2023** — DOI [10.1109/JSSC.2023.3234893](https://doi.org/10.1109/jssc.2023.3234893).
- **Mechanism:** **Per-vector scale factors** for aggressive INT4 while preserving accuracy on Transformer workloads; high measured TOPS/W in 5 nm.
- **HW status:** **Silicon** 5 nm.
- **Transfer:** Quantize **ATLIF real payloads** with **per-Motion-TTB / per-hyp vector scales**, not global INT8. Feeds ADP-MAC / DualRail payload lane.
- **Remake name:** **PVScale-ATLIF**.
- **vs C1\*/C2\*:** **Enhance ADP-MAC / HBG-RP payload path**; engineering, not headline alone.
- **Novelty:** **7/10** (per-vector scale known; **per-hyp / per-bundle ATLIF scale** for OF is fresher).
- **Prove-by:** AEE vs bitwidth with global vs per-vector vs per-hyp scales.

### M10 — CogniVision hierarchical always-on smart vision SoC

- **Citation:** Gupta, Vohra, Alioto — *CogniVision: End-to-End SoC for Always-on Smart Vision with mW Power in 40nm* — **VLSI 2024** — DOI [10.1109/VLSITechnologyandCir46783.2024.10631426](https://doi.org/10.1109/vlsitechnologyandcir46783.2024.10631426).
- **Mechanism:** Full vision SoC: imager with **dual in/near-sensor saliency**, **on-the-fly novelty detection**, DNN with on-chip scheduler, WiFi TX, wake-up RX; **gate each subsequent pipeline stage** from lowest semantic level; ~2.1 mW average @ 30 fps reported.
- **HW status:** **Silicon** 40 nm.
- **Transfer:** Map saliency→novelty→DNN cascade onto **EV-Wake → OP-STW → ECP-QKV → exact SDSA**. Lowest semantic level = event/motion energy; novelty = residual vs MW-ΔBuf ref.
- **Remake name:** **SalCascade-EVWake** (system; Card H).
- **vs C1\*/C2\*:** **Enhance EV-Wake / OP-STW / MW-ΔBuf**; does not replace SDSA cores.
- **Novelty:** **8.5/10** for OF-Spikeformer cascade binding; **6/10** as AoV SoC.
- **Prove-by:** Stage-gating power waterfall; false-wake vs miss-wake vs AEE.

### M11 — Alpha-Vision always-on vision subsystem (ISSCC 2026)

- **Citation:** *31.9 Alpha-Vision: A Real-Time Always-on Vision Processor with 787µs Face Detection Latency in <5mW* — **ISSCC 2026** (IEEE Xplore doc [11409322](https://ieeexplore.ieee.org/document/11409322)); supports CNN and ViT, end-to-end on-chip, fine-grained leakage management, ~4.6 mW average @ 60 fps face detection.
- **Mechanism:** Programmable **always-on vision (AoV)** subsystem for edge SoCs; CNN+ViT; no external memory for target pipeline; leakage-aware power mgmt.
- **HW status:** **Silicon** (ISSCC 2026 presentation).
- **Transfer:** Template for **Card H** “always-on OF front-end” that keeps TDE3/EV-Wake alive while deep SDSA sleeps. Do **not** paste face-detect latency as OF claim.
- **Remake name:** **AoV-Cascade-CardH**.
- **vs C1\*/C2\*:** **System enhance** of EV-Wake stack; orthogonal to Card A/B RTL.
- **Novelty:** **8/10** as OF AoV subsystem claim; **5/10** if “always-on vision processor” alone.
- **Prove-by:** End-to-end mW with deep core gated; wake latency to first Motion-TTB.

### M12 — Motion-event / adaptive-resolution always-on HAR SoC (ancestor)

- **Citation:** (representative always-on CIS+DNN) — *A 0.82 µW CIS-Based Action Recognition SoC With Self-Adjustable Frame Resolution for Always-on IoT Devices* — **IEEE TCAS-II 2021** — DOI [10.1109/TCSII.2021.3067151](https://doi.org/10.1109/tcsii.2021.3067151).
- **Mechanism:** **MEDU** motion-event detection skips imaging+DNN when idle; **adaptive frame resolution** reduces CIS readout; µW idle / mW active.
- **HW status:** Simulated/implemented 65 nm report (paper states simulated in 65 nm).
- **Transfer:** MEDU ↔ OP-STW coarse gate; adaptive resolution ↔ PRRC pyramid / HeatFlow-Tok spatial budget.
- **Remake name:** **MEDU-OPSTW**.
- **vs C1\*/C2\*:** **Enhance** OP-STW / PRRC.
- **Novelty:** **7/10**.
- **Prove-by:** Idle skip rate on static scenes; resolution ladder vs AEE.

### M13 — Tesla FSD HotChips NNA + ISP integration (public slides)

- **Citation:** Bannon, Venkataramanan, et al. — Tesla FSD Chip — **Hot Chips 31 (2019)** slides: [old.hotchips.org/hc31/HC31_2.3_Tesla_Hotchips_ppt_Final_0817.pdf](https://old.hotchips.org/hc31/HC31_2.3_Tesla_Hotchips_ppt_Final_0817.pdf). Secondary analysis: WikiChip Fuse summary (not a primary claim source).
- **Mechanism (public):** Dual custom **NNA** (96×96 MAC, large **on-chip SRAM**, programs resident in SRAM), integrated **ISP + video encode**, CPU/GPU for post-process; batch-1 inference focus; measured/quoted TOPS for Tesla nets (use cautiously).
- **HW status:** **Commercial silicon** (public HotChips disclosure).
- **Transfer:** **SRAM-resident** Motion-TTB / MFBD kernels and wake bitmaps; ISP strip feeding residual OF front-end without DRAM round-trips — system pattern for Card H, not “we match FSD TOPS.”
- **Remake name:** **SRAM-Resident-MFBD**.
- **vs C1\*/C2\*:** **Enhance MFBD / Motion-TTB** placement; orthogonal microarch of SDSA.
- **Novelty:** **7/10** (SRAM-resident NNA known; **OF bundle programs + wake maps resident** is the twist).
- **Prove-by:** DRAM traffic with/without resident MFBD dictionaries under continuous OF.

### M14 — Huawei DaVinci HotChips scalable AI core

- **Citation:** Liao et al. — *DaVinci: A Scalable Architecture for Neural Network Computing* — **Hot Chips 31 (2019)** slides: [old.hotchips.org/hc31/HC31_1.11_Huawei.Davinci.HengLiao_v4.0.pdf](https://old.hotchips.org/hc31/HC31_1.11_Huawei.Davinci.HengLiao_v4.0.pdf).
- **Mechanism (public):** Scalable **AI Core** with cube (matmul), vector, and scalar units; ISP/VPU siblings in SoC story; Ascend-Mini → Ascend-Max scaling narrative.
- **HW status:** **Commercial silicon** family (HotChips disclosure).
- **Transfer:** Map **HBG-RP dual rail** onto cube (payload MAC) vs vector/scalar (gate, Mask-Add, softmax-ish, control). CubeVec scheduling ↔ STH-Gate spatial vs temporal.
- **Remake name:** **CubeVec-HBG Map**.
- **vs C1\*/C2\*:** **Enhance** HBG-RP / SMAM-RP implementation mapping; not a new algorithm.
- **Novelty:** **6.5/10** (mapping aid); useful for Card H floorplan story.
- **Prove-by:** Utilization of cube vs vector lanes under SMAM-RP masks.

### M15 — SCNN compressed-sparse Cartesian-product dataflow

- **Citation:** Parashar, Rhu, Mukkara, Puglielli, Venkatesan, Khailany, Emer, Keckler, Dally — *SCNN: An Accelerator for Compressed-sparse Convolutional Neural Networks* — **ISCA 2017** — DOI [10.1145/3140659.3080254](https://doi.org/10.1145/3140659.3080254) · arXiv [1708.04485](https://arxiv.org/abs/1708.04485). Related sparse ISA: Zhang et al. — *Cambricon-X* — **MICRO 2016** — DOI [10.1109/MICRO.2016.7783723](https://doi.org/10.1109/micro.2016.7783723).
- **Mechanism:** Keep sparse W and A compressed; **Cartesian product** delivery to multiplier array; specialized accumulator array for irregular scatter.
- **HW status:** Architecture (widely cited; influenced later silicon sparse NPUs).
- **Transfer:** Cartesian delivery of **nonzero residual spikes × hyp weights** into ADP-MAC; accumulator scatter keyed by Motion-TTB ID.
- **Remake name:** **CartProd-TTB**.
- **vs C1\*/C2\*:** **Enhance Motion-TTB / ADP-MAC / MFBD**.
- **Novelty:** **7.5/10**.
- **Prove-by:** Mult utilization vs dense under structured OF sparsity; accumulator conflict rate.

### M16 — Bit Fusion / Stripes / UNPU precision-scalable lineages

- **Citation:**  
  - Judd, Albericio, Hetherington, Aamodt, Moshovos — *Stripes: Bit-Serial Deep Neural Network Computing* — **MICRO 2016** — DOI [10.1109/MICRO.2016.7783722](https://doi.org/10.1109/micro.2016.7783722).  
  - Sharma, Park, Mahajan, Amaro, Kim, Shao, et al. — *Bit Fusion: Bit-Level Dynamically Composable Architecture for Accelerating Deep Neural Networks* — **MICRO 2018** (arXiv [1712.01507](https://arxiv.org/abs/1712.01507)).  
  - Lee, Kim, Kang, Shin, Kim, Yoo — *UNPU: A 50.6TOPS/W unified deep neural network accelerator with 1b-to-16b fully-variable weight bit-precision* — **ISSCC 2018** — DOI [10.1109/ISSCC.2018.8310262](https://doi.org/10.1109/isscc.2018.8310262).  
  - Lee, Lee, Han, Lee, Park, Yoo — *7.7 LNPU: A 25.3TFLOPS/W Sparse Deep-Neural-Network Learning Processor with Fine-Grained Mixed Precision of FP8–FP16* — **ISSCC 2019** — DOI [10.1109/ISSCC.2019.8662302](https://doi.org/10.1109/isscc.2019.8662302).
- **Mechanism:** Execution time / energy **scales with bitwidth**; bit-serial or bit-fusion PE; UNPU fully variable 1–16b weights; LNPU fine-grained FP8/16 + sparsity for learning.
- **HW status:** Stripes/Bit Fusion architecture; UNPU/LNPU **silicon**.
- **Transfer:** Fuse with **MP-Pred-PRRC**: bit-fusion lanes widen only when residual confidence / Entropy-EE-OF demands; ADP-MAC bit-sparsity on ATLIF payloads.
- **Remake name:** **PRRC-BitFuse**.
- **vs C1\*/C2\*:** **Enhance PRRC / ADP-MAC**.
- **Novelty:** **7/10** (bit-scalable NPU saturated; **predicated by OF confidence** is the remake).
- **Prove-by:** Average effective bits vs AEE; compare fixed INT8 vs PRRC-BitFuse.

### M17 — Eyeriss / Envision / Thinker classical edge dataflow controls

- **Citation:**  
  - Chen, Krishna, Emer, Sze — *Eyeriss* — **ISSCC 2016** / **JSSC 2017** — DOI [10.1109/JSSC.2016.2616357](https://doi.org/10.1109/jssc.2016.2616357); Eyeriss v2 — **JETCAS 2019**.  
  - Moons, Uytterhoeven, Dehaene, Verhelst — *14.5 Envision: A 0.26-to-10TOPS/W subword-parallel dynamic-voltage-accuracy-frequency-scalable CNN processor* — **ISSCC 2017** — DOI [10.1109/ISSCC.2017.7870353](https://doi.org/10.1109/isscc.2017.7870353).  
  - Yin et al. — *Thinker* hybrid-NN — **VLSI 2017** DOI [10.23919/VLSIC.2017.8008534](https://doi.org/10.23919/vlsic.2017.8008534) / **JSSC 2018** DOI [10.1109/JSSC.2017.2778281](https://doi.org/10.1109/jssc.2017.2778281).  
  - Shin, Lee, Lee, Yoo — *14.2 DNPU* — **ISSCC 2017** — DOI [10.1109/ISSCC.2017.7870350](https://doi.org/10.1109/isscc.2017.7870350).
- **Mechanism:** **Row-stationary** reuse + data gating (Eyeriss); **DVAFS** accuracy–voltage–frequency (Envision); bit-width adaptive + array partition (Thinker); CNN–RNN reconfig (DNPU).
- **HW status:** **Silicon** (all).
- **Transfer:** RS reuse → **MW-ΔBuf** reference tiles stay stationary while residual spikes stream; DVAFS → TileVFS-Motion; Thinker partition → dense vs sparse SDSA engines (ViTCoD/Bishop cousin).
- **Remake names:** **RS-MW-Reuse** · **DVAFS-PRRC** · **Thinker-Split-SDSA**.
- **vs C1\*/C2\*:** **Enhance** MW-ΔBuf / PRRC / STH dual engines.
- **Novelty:** **6–7.5/10** (classic; claim only as OF-grounded remakes).
- **Prove-by:** On-chip reuse distance for warped ref; DVAFS energy under PRRC levels.

### M18 — Tightly coupled AI-ISP strip-tile vision processor

- **Citation:** *A Tightly Coupled AI-ISP Vision Processor* — **IEEE TCSVT 2024** — DOI [10.1109/TCSVT.2024.3510939](https://doi.org/10.1109/TCSVT.2024.3510939) (ACM DL entry).
- **Mechanism:** **Strip-tile conversion dataflow** for fine-grained low-latency ISP↔DLA interaction; pixel pool; layer-fusion DLA; measured EMA and latency cuts for UHD pipelines.
- **HW status:** Architecture / evaluated processor (paper reports quantitative EMA/latency reductions).
- **Transfer:** CIS/ISP strips → residual OF tiles without full-frame DRAM; event ROI windows (**EvQ-Win**) align to same strip schedule as frame ref for **Het-OF**.
- **Remake name:** **StripTile-OFPipe**.
- **vs C1\*/C2\*:** **Enhance** MW-ΔBuf / EvQ-Win / Het-OF-CIM system I/O.
- **Novelty:** **8/10**.
- **Prove-by:** EMA for frame+event OF vs decoupled ISP→NPU baseline.

### M19 — CIM block-wise zero-skip ping-pong (Yue)

- **Citation:** Yue, Feng, He, Huang, Wang, Yuan, et al. — *15.2 A 2.75-to-75.9TOPS/W Computing-in-Memory NN Processor Supporting Set-Associate Block-Wise Zero Skipping and Ping-Pong CIM with Simultaneous Computation and Weight Updating* — **ISSCC 2021**, pp. 238–240 — DOI [10.1109/ISSCC42613.2021.9365958](https://doi.org/10.1109/isscc42613.2021.9365958).
- **Mechanism:** **Set-associative block-wise zero skipping** in CIM; ping-pong CIM compute∥weight update.
- **HW status:** **Silicon**.
- **Transfer:** Block-skip CIM tiles under OP-STW / MW-CIM-TileGate; ping-pong for online hyp/prior refresh (ARM-Acc dictionary).
- **Remake name:** **BlkSkip-CIM-Wake**.
- **vs C1\*/C2\*:** **Enhance** MW-CIM-TileGate / Het-OF-CIM.
- **Novelty:** **7.5/10**.
- **Prove-by:** CIM tile active fraction vs residual energy map.

### M20 — Marsellus endpoint SoC (orchestration reminder)

- **Citation:** Conti et al. — Marsellus — **ISSCC 2023** PDF [pulp-platform.org/docs/isscc2023/isscc2023_marsellus_fconti.pdf](https://pulp-platform.org/docs/isscc2023/isscc2023_marsellus_fconti.pdf); SamurAI VLSI’20 cited therein.
- **Mechanism / transfer:** Endpoint RISC-V + precision-scalable DNN shows **end-to-end mW ≫ AiMC peak**. Card H should claim **orchestration energy**, not TOPS.
- **Remake:** (principle only — no new code).

---

## 3. ≥12 remake ideas — compact cards

| # | Remake | Cluster | Enhance target | Nov. | One-line prove-by |
|---|---|---|---|---|---|
| 1 | **Entropy-EE-OF** | Early-exit Transformer NPU | PRRC / SP-Gate / HeatFlow | 9 | Exit-layer hist vs AEE & mJ/frame |
| 2 | **MP-Pred-PRRC** | MP predication | PRRC bit ladder | 8.5 | FP/INT path select from residual conf. |
| 3 | **TileVFS-Motion** | Fine-grain VFS | System Card H | 8 | V/F vs motion density under latency SLA |
| 4 | **Asymp-OoO-ECP** | Sparse Transformer speculate | ECP-QKV scheduler | 8 | Stall cycles under event bursts |
| 5 | **FMSkip-OPSTW** | Mobile FM zero-skip | OP-STW | 7.5 | PE util vs residual sparsity |
| 6 | **DynPort-MFBD** | Dynamic memory ports | MFBD / ADP-MAC | 8 | Bank conflict @ occlusion |
| 7 | **HetSchedule-OF** | DIANA-style hetero | Het-OF-CIM | 8.5 | Digital vs AiMC assignment trace |
| 8 | **SalCascade-EVWake** | AoV saliency cascade | EV-Wake / OP-STW | 8.5 | Power waterfall per semantic stage |
| 9 | **AoV-Cascade-CardH** | Always-on subsystem | Card H | 8 | Deep-core gated mW + wake latency |
| 10 | **BigLittle-DualRail** | C-Transformer big-little | HBG-RP / SMAM-RP | 8 | % tiles little vs AEE |
| 11 | **BLT-CIM-SDSA** | BL-transpose CIM Attn | DualRail-CIM / MFBD | 8 | Transpose-buffer energy share |
| 12 | **PVScale-ATLIF** | Per-vector INT4 | ADP-MAC payload | 7 | AEE vs scale granularity |
| 13 | **CartProd-TTB** | SCNN sparse delivery | Motion-TTB / ADP-MAC | 7.5 | Mult util + scatter conflicts |
| 14 | **PRRC-BitFuse** | Bit-fusion / UNPU | PRRC / ADP-MAC | 7 | Avg effective bits vs AEE |
| 15 | **StripTile-OFPipe** | AI-ISP strip-tile | MW-ΔBuf / EvQ-Win | 8 | EMA frame+event OF |
| 16 | **SRAM-Resident-MFBD** | FSD-style SRAM programs | MFBD / Motion-TTB | 7 | DRAM traffic continuous OF |
| 17 | **Retain-Prior-eMRAM** | TinyVers retentive | TDE3 / EV-Wake | 7.5 | Cold vs retain AEE after sleep |
| 18 | **BlkSkip-CIM-Wake** | CIM block zero-skip | MW-CIM-TileGate | 7.5 | Active CIM tile fraction |
| 19 | **EC-Psum-Gate** | QNAP prediction skip | ARM-Acc / ADP-MAC | 7.5 | Psum-skip vs AEE (**≠ ECP-QKV**) |
| 20 | **EWC-Cluster-MAC** | Effective-weight | ARM-Acc / ADP-MAC | 7.5 | Multiplier count under hyp reuse |
| 21 | **RS-MW-Reuse** | Eyeriss RS | MW-ΔBuf | 7 | Ref-tile reuse distance |
| 22 | **MEDU-OPSTW** | Motion-event skip imaging | OP-STW | 7 | Idle skip rate static scenes |
| 23 | **IrrNoC-MotionTTB** | Sparse irregular fabric | Motion-TTB NoC | 7.5 | Load imbalance under STH masks |
| 24 | **CubeVec-HBG Map** | DaVinci cube/vector | HBG-RP mapping | 6.5 | Lane util under SMAM-RP |

*(Primary shortlist uses top ~12; extras are engineering siblings.)*

---

## 4. Ranked shortlist (ISCAS claim value)

1. **Entropy-EE-OF + MP-Pred-PRRC + TileVFS-Motion** — strongest **system control loop** stolen from edge Transformer NPUs; OF-grounded confidence replaces NLP entropy.  
2. **SalCascade-EVWake / AoV-Cascade-CardH** — binds always-on vision silicon pattern to EV-Wake→exact SDSA.  
3. **Asymp-OoO-ECP** — upgrades ECP-QKV with silicon-proven speculate+OoO.  
4. **HetSchedule-OF** — operational schedule for Het-OF-CIM / DualRail-CIM from DIANA.  
5. **DynPort-MFBD + FMSkip-OPSTW** — mobile NPU sparsity microarch → OF residual traffic.  
6. **BigLittle-DualRail** — path select for HBG/SMAM without claiming first spike-Transformer chip.  
7. **BLT-CIM-SDSA** — dynamic attention CIM transpose fix for Motion-TTB.  
8. **StripTile-OFPipe + SRAM-Resident-MFBD** — sensor↔NPU and residency for Card H.  
9. **CartProd-TTB + PVScale-ATLIF + PRRC-BitFuse** — datapath precision/sparsity engineering.  
10. **EC-Psum-Gate / EWC-Cluster-MAC** — careful QNAP transfers (acronym hygiene vs ECP-QKV).

---

## 5. How Round-4 upgrades prior packages

| Prior package | R4 upgrade |
|---|---|
| **OP-STW** | + **FMSkip-OPSTW**, **MEDU-OPSTW**, saliency stage of **SalCascade** |
| **ECP-QKV** | + **Asymp-OoO-ECP** scheduler (speculate + OoO); still distinct from QNAP “ECP” |
| **PRRC / HeatFlow-Tok** | + **MP-Pred-PRRC**, **PRRC-BitFuse**, **DVAFS-PRRC**, **TileVFS-Motion** |
| **MW-ΔBuf** | + **RS-MW-Reuse**, **StripTile-OFPipe** I/O |
| **EV-Wake / TDE3-Prior** | + **SalCascade-EVWake**, **Retain-Prior-eMRAM**, **AoV-Cascade-CardH** |
| **HBG-RP / SMAM-RP** | + **BigLittle-DualRail**, **CubeVec-HBG Map** |
| **ADP-MAC / ARM-Acc** | + **PVScale-ATLIF**, **EWC-Cluster-MAC**, **EC-Psum-Gate**, **CartProd-TTB** |
| **MFBD / Motion-TTB** | + **DynPort-MFBD**, **SRAM-Resident-MFBD**, **IrrNoC-MotionTTB**, **BLT-CIM-SDSA** |
| **Het-OF-CIM / DualRail-CIM** | + **HetSchedule-OF**, **BlkSkip-CIM-Wake** |
| **SP-Gate / STH-Gate** | early-exit & big-little path select as outer loops |

**R1→R2→R3→R4 one-liner:** R1 semantic OF+ATLIF · R2 eager/warped/temporal sparse · R3 CIM+event fabric · **R4 edge-NPU control loops & SoC orchestration (EE, VFS, ports, AoV cascade, hetero schedule)**.

---

## 6. Suggested Card H / system-level claims

**Card H (proposed):** *Always-on OF front-end + orchestrator* — not another MAC array.

Suggested contents:
1. **SalCascade-EVWake** FSM (sensor saliency → novelty/residual → wake bitmap).  
2. **Entropy-EE-OF / TileVFS-Motion** policy tables (confidence → exit / bitwidth / V·F).  
3. **HetSchedule-OF** dispatcher (tile → digital MAC vs DualRail-CIM vs sleep).  
4. **Retain-Prior-eMRAM** (or SRAM retain) for TDE3 / last flow.  
5. **StripTile-OFPipe** interface registers to ISP/event DMA.  
6. Host RISC-V-class sequencer (Marsellus/DIANA pattern) — **claim orchestration energy**, not peak TOPS.

**Safe Card H claim sentence (draft):**  
“System energy is gated by an **always-on residual/event cascade** that predicates **precision, voltage, and exact-SDSA entry**, rather than by peak NPU TOPS.”

**Keep Cards A/B** for HBG-RP / OP-STW RTL; **Card F** CIM fabric; **Card G** TMA-Agg/EvQ-Win; **Card H** = R4 SoC loop.

---

## 7. DO NOT CLAIM

- First / only **edge NPU**, **sparse NPU**, **mobile NPU**, or **always-on vision SoC**.  
- Generic **peak TOPS / TOPS/W** from Samsung / Tesla / Huawei / Edge TPU marketing as SDformer results.  
- “We built a flagship-class NPU” — wrong paper; this is **OF-SNN-Transformer remake**.  
- First **spiking Transformer** HW (Bishop ISCA’25, SMAM, FireFly-T, C-Transformer ISSCC’24, …).  
- First **CIM Transformer** or first **hybrid digital/AiMC** (DIANA, Tu ISSCC’22, …).  
- Equating **QNAP ECP** (error-compensation psum prediction) with **FACT/ECP-QKV** (eager correlation before QKV) — **different mechanisms; document acronym collision**.  
- Pasting **face-detect latency** (Alpha-Vision) or **BERT µJ/token** as optical-flow metrics.  
- Google **Edge TPU** as a detailed microarch source without a citable HotChips/ISSCC mechanism paper — mention only as product lineage if needed, **no fabricated internals**.  
- Apple Neural Engine / Qualcomm Hexagon **internals** without public ISSCC/HotChips mechanism cites (Cloud AI 100 decks are product; use only high-level if at all).  
- Neurocube as “near-memory OF solution” headline (ISCA’16 architecture; low transfer priority).  
- Replacing **HBG-RP / ARM-Acc / OP-STW** with generic NPU zero-skip.

---

## 8. Cite checklist (primary DOIs / links used)

| Work | Venue | ID |
|---|---|---|
| Samsung 6K-MAC FM-sparsity NPU | ISSCC’21 | 10.1109/ISSCC42613.2021.9365928 |
| Samsung sparsity NPU architecture | ISCA’21 | 10.1109/ISCA52012.2021.00011 |
| Samsung butterfly sparse NPU | ISSCC’19 | 10.1109/ISSCC.2019.8662476 |
| Tambe STP entropy EE | ISSCC’23 | PDF sld.cs.columbia.edu/pubs/tambe_isscc23.pdf |
| EdgeBERT | MICRO’21 | arXiv:2011.14203 |
| Wang asymptotic sparsity Transformer | ISSCC’22 | 10.1109/ISSCC42614.2022.9731686 |
| Tu BL-transpose CIM Transformer | ISSCC’22 | 10.1109/ISSCC42614.2022.9731645 |
| C-Transformer | ISSCC’24 | 10.1109/ISSCC49657.2024.10454330 |
| DIANA | ISSCC’22 / JSSC’23 | 10.1109/ISSCC42614.2022.9731716 · 10.1109/JSSC.2022.3214064 |
| TinyVers | VLSI’22 / JSSC’23 | 10.1109/VLSITechnologyandCir46769.2022.9830409 · 10.1109/JSSC.2023.3236566 |
| QNAP | ISSCC’21 / JSSC’21 | 10.1109/ISSCC42613.2021.9365943 · 10.1109/JSSC.2021.3113569 |
| Keller per-vector INT4 | VLSI’22 / JSSC’23 | 10.1109/JSSC.2023.3234893 |
| CogniVision | VLSI’24 | 10.1109/VLSITechnologyandCir46783.2024.10631426 |
| Alpha-Vision | ISSCC’26 | IEEE Xplore 11409322 |
| Always-on HAR SoC | TCAS-II’21 | 10.1109/TCSII.2021.3067151 |
| Tesla FSD | HotChips’19 | HC31_2.3 PDF (hotchips.org) |
| Huawei DaVinci | HotChips’19 | HC31_1.11 PDF (hotchips.org) |
| SCNN | ISCA’17 | 10.1145/3140659.3080254 |
| Cambricon-X | MICRO’16 | 10.1109/MICRO.2016.7783723 |
| Stripes | MICRO’16 | 10.1109/MICRO.2016.7783722 |
| Bit Fusion | MICRO’18 | arXiv:1712.01507 |
| UNPU | ISSCC’18 | 10.1109/ISSCC.2018.8310262 |
| LNPU | ISSCC’19 | 10.1109/ISSCC.2019.8662302 |
| Eyeriss | JSSC’17 | 10.1109/JSSC.2016.2616357 |
| Envision | ISSCC’17 | 10.1109/ISSCC.2017.7870353 |
| Thinker | VLSI’17 / JSSC’18 | 10.23919/VLSIC.2017.8008534 · 10.1109/JSSC.2017.2778281 |
| DNPU | ISSCC’17 | 10.1109/ISSCC.2017.7870350 |
| Yue CIM zero-skip | ISSCC’21 | 10.1109/ISSCC42613.2021.9365958 |
| AI-ISP strip-tile | TCSVT’24 | 10.1109/TCSVT.2024.3510939 |
| Ayaka | JSSC’24 | 10.1109/JSSC.2024.3397189 |
| Neurocube (context only) | ISCA’16 | 10.1109/ISCA.2016.41 |

---

## 9. Bottom line for ISCAS OF-SNN-T

Round-4 does **not** add “another sparse MAC.” It adds **edge-NPU-grade control and SoC loops**:

- **Confidence → exit / precision / V·F** (Tambe lineage → Entropy-EE-OF).  
- **Saliency cascade → exact path** (CogniVision / AoV → SalCascade-EVWake).  
- **Hetero digital/AiMC schedule** (DIANA → HetSchedule-OF).  
- **Sparse irregular traffic plumbing** (Samsung ports / SCNN CartProd → DynPort-MFBD / CartProd-TTB).  

Combined with R1–R3 packages, the story is: **motion-semantic front-end + dual-rail ATLIF attention + CIM/event fabric + edge-NPU orchestration** — still **enhance**, never “we are a phone NPU.”

**Files:** this note `13_isscc_vlsi_hotchips_edge_npus.md` (mirrored under `ideafromai/research/` and `hw_innovation_research/`).  
**Next synthesis candidate:** `14_ROUND4_SYNTHESIS_EDGE_NPU.md` (Card H freeze).
