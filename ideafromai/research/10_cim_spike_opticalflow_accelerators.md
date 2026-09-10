# ROUND 3 — CIM / Compute-in-Memory for Spike · SNN · Optical-Flow · Sparse Transformer Workloads

**Date:** 2026-09-05 (Asia/Shanghai)  
**Scope:** ~2018–2026 · ISSCC / VLSI / A-SSCC / HotChips / JSSC / TCAS-I / TCAS-AI / ISCA / MICRO / HPCA / DAC / DATE / ICCAD / ICCD / Nature Electronics–adjacent + arXiv  
**Goal:** Extract **named, transferable CIM mechanisms** for optical-flow Spikeformer (SDformer) remake packages — **not** generic “CIM is energy-efficient / ADC is the bottleneck” slogans.  
**C1\* package (R1+R2):** OP-STW / PRRC / OGEC / **ECP-QKV** / **MW-ΔBuf** / EV-Wake / HeatFlow-Tok  
**C2\* package (R1+R2):** **HBG-RP** / ADP-MAC / ARM-Acc / MFBD / SP-Gate / **SMAM-RP** / **Motion-TTB** / **OF-ECP** / **STH-Gate**  
**Hard rules:**
- **NO fabricated citations.** Every paper below has a real venue / DOI / arXiv.
- Do **not** claim “first CIM for SNN / Transformer / OF.” Silicon and simulation priors exist.
- Do **not** modify Codex hw under `hw_autoresearch_nts07` — this note only feeds **ideafromai** remake naming.
- Round-3 CIM **enhances** C1\*/C2\* (orthogonal fabric / macro options); it does **not** replace HBG-RP / ARM-Acc / OP-STW.

---

## 0. How to read this note

For each **strong idea**:
- **Citation** (venue/year + DOI/arXiv)
- **Mechanism** (1–3 lines, what actually moves bits/charge)
- **HW status** (silicon / measured / NeuroSim·AIHWKit simulation / architecture paper)
- **Transfer to OF-SNN-Transformer**
- **Remake name** (new code, mappable to C1\*/C2\*)
- **vs C1\*/C2\*** — enhance / replace / orthogonal
- **Novelty 1–10** for an **ISCAS OF-SNN** claim (10 = OF+spike-specific + unsaturated; ≤5 = saturated SRAM-CIM MAC slogan)
- **Prove-by experiment sketch**

**Saturation legend (explicit):**
| Saturated (do NOT claim as ISCAS headline) | Still-open OF/spike hooks |
|---|---|
| Generic 6T/8T SRAM-CIM MAC TOPS/W | Fused **W + Vmem** CIM for ATLIF residual |
| “ADC-less because binary spike” alone | Motion-gated WL/BL activity (OP-STW → CIM) |
| Analog crossbar VMM for static FC | Hybrid **static NVM projection + dynamic SRAM attention** for SDSA |
| Bit-serial digital CIM for CNN edge | Dual-rail **gate CIM / payload digital** (HBG-RP on CIM fabric) |
| Twin-column signed weights (known) | Twin-column keyed by **flow hyp / Motion-TTB** |
| Event+frame fusion SoC (tracking) | Same hetero fabric for **dense OF residual + event wake** |

---

## 1. Landscape table (clusters → remake hooks)

| Cluster | Representative works (real) | What is transferable | Remake names (this note) | Novelty band |
|---|---|---|---|---|
| Digital SRAM-CIM + fused Vmem | IMPULSE (arXiv:2105.08217, 65nm); SpiDR (arXiv:2411.02854, 65nm) | Fuse W+Vmem; multi-neuron IF/LIF/RMP; zero-skip inputs | **Fused-WVMEM-CIM**, **SpiDR-Reconf-CIM** | 7–8.5 |
| Neuromorphic SRAM-CIM (ADC-less) | Neuro-CIM VLSI’22 / JSSC’23; Spike-CIM A-SSCC’22 | MSB skip, early stop, charge-domain IF, sparsity-adaptive | **MSB-Skip-CIM**, **CD-IF-SpikeCIM** | 6.5–8 |
| TTFS / twin-column SRAM | TFSRAM TCAS-AI’24 | Twin-col ± weights; timing threshold | **TwinCol-TTFS-OF** | 7 |
| Time-domain SNN CIM | Park et al. TCAS-I’25 (701.7 TOPS/W) | Async time-domain MAC + LIF | **TD-CIM-Gate** | 6.5 |
| RRAM / nvCIM + spike | Yan et al. VLSI’19 (ISNA); Hung/Chang ReRAM CIM ISSCC’21 lineage | In-situ nonlinear / spike conversion; dense static W | **ISNA-nvCIM-FFN** | 6–7.5 |
| Hetero RRAM-CIM + SRAM-CNM for event+frame | Chang/Lele ISSCC’23 → JSSC’24 | Modality-matched: RRAM CNN + SRAM SNN; power gate | **Het-FrameEvent-CIM** | **8.5–9** |
| Transformer-in-memory (ANN) | X-Former TVLSI’23; ReTransformer ICCAD’20; RACE-IT ICCD’25; Spoon Frontiers’21 | Static NVM FFN + dynamic SRAM/ACAM attention | **XForm-Split-SDSA**, **ACAM-NonVMM** | 7–8.5 |
| Spiking Transformer + AIMC | Xpikeformer arXiv:2408.08794 | PCM FFN + stochastic spiking attention engine | **PCM-FFN + SSA-OF** | 7–8.5* |
| Full-precision digital CIM (ANN) | Chih et al. ISSCC’21 22nm 89 TOPS/W | Bit-serial / adder-tree digital CIM for multi-bit payload | **DigiCIM-PayloadLane** | 5–7 |
| Sparse / binary-ish CIM for SNN | Spike-CIM; Neuro-CIM W=1b path; Khwa ISSCC’18 binary | Input-spike sparsity → WL activity | **Spike-WL-Gate-CIM** | 5–7 |

\*Claim carefully: Xpikeformer self-styles as first spiking-Transformer HW accelerator (simulation); **Bishop ISCA’25 / SMAM / FireFly-T** are digital cousins — do **not** claim “first.”

---

## 2. Strong ideas (mechanisms → remakes)

### M1 — Fused weight + membrane-potential digital CIM (IMPULSE)

- **Citation:** Agrawal, Ali, Koo, Rathi, Jaiswal, Roy — *IMPULSE: A 65nm Digital Compute-in-Memory Macro with Fused Weights and Membrane Potential for Spike-based Sequential Learning Tasks* — arXiv:2105.08217 (submitted to IEEE; 65nm silicon report in paper).
- **Mechanism:** 10T-SRAM macro **fuses WMEM and VMEM** on shared BLs; in-memory AccW2V / SpikeCheck / ResetV / AccV2V; staggered mapping for 6b W vs 11b Vmem; supports IF / LIF / **RMP**; input-spike sparsity → instruction count (∼97% EDP cut at 85% sparsity). Measured ∼0.99 TOPS/W @ 0.85 V, 200 MHz for signed 11b ops.
- **HW status:** **Silicon** 65nm.
- **Transfer:** ATLIF / residual membrane is the SDformer pain point (extra SRAM traffic). Fuse **gate + payload amp** storage with projection weights for local tiles woken by OP-STW.
- **Remake name:** **Fused-WVMEM-CIM** (aka **ATLIF-FuseCIM**).
- **vs C1\*/C2\*:** **Enhance C2\*** (HBG-RP / ADP-MAC locality); **orthogonal to C1\*** wake (OP-STW decides *which* fused tile is live). Does **not** replace HBG-RP semantics.
- **Novelty:** **8/10** (fused Vmem CIM is known for SNN FC; **OF residual + ATLIF amp fuse** under OP-STW is not saturated).
- **Prove-by:** RTL/FPGA: Acc energy & Vmem traffic vs separate SRAM baseline on ep34 OF tiles; ablate fuse vs HBG-RP gate-only; report AEE vs energy at matched sparsity.

### M2 — Reconfigurable digital CIM SNN + zero-skip (SpiDR)

- **Citation:** *SpiDR: A Reconfigurable Digital Compute-in-Memory Spiking Neural Network Accelerator for Event-based Perception* — arXiv:2411.02854 (65nm TSMC LP fabrication reported).
- **Mechanism:** Digital CIM with **reconfigurable precision / neuron / size**; Vmem handling; **zero-skip** for sparse spikes without high overhead at low sparsity; async handshake for variable unit latency; up to ∼5 TOPS/W at 95% input sparsity (4b W / 7b Vmem reported in abstract).
- **HW status:** **Silicon** 65nm.
- **Transfer:** Event / residual sparsity is **non-stationary** in OF (occlusion bursts). Reconfig modes map to PRRC pyramid bitwidths and STH-Gate spatial vs temporal heads.
- **Remake name:** **SpiDR-Reconf-CIM** → specialize as **PRRC-CIM-Modes**.
- **vs C1\*/C2\*:** **Enhance C1\*** PRRC / HeatFlow-Tok bit ladders; **enhance C2\*** STH-Gate (mode = spatial-dense vs temporal-sparse). Orthogonal to SMAM-RP algorithm.
- **Novelty:** **7/10** (reconfig SNN CIM exists); **8/10** if modes are **OF-pyramid / head-type** keyed.
- **Prove-by:** Sweep sparsity 50–95% on DVS-warped OF sequences; measure mode-switch overhead vs fixed 8b CIM; keep AEE within budget.

### M3 — MSB word skipping + early stopping + ADC-less neuromorphic CIM (Neuro-CIM)

- **Citation:** Kim, Kim, Um, Kim, Kim, Yoo — *Neuro-CIM: A 310.4 TOPS/W Neuromorphic Computing-in-Memory Processor with Low WL/BL activity and Digital-Analog Mixed-mode Neuron Firing* — **VLSI Technology & Circuits 2022** — DOI:10.1109/vlsitechnologyandcir46769.2022.9830276; extended *Neuro-CIM: ADC-Less …* **JSSC 2023** (operation gating/stopping). HotChips’22 poster.
- **Mechanism:** SNN conversion removes high-precision ADC (1b comparator / spike); **MSB word skipping** (sign-ext / small-mag weights → lower BL activity, 25–38% power); **early stopping** of neuronal ops (∼31–37%); mixed-mode multi-macro aggregation; voltage folding. Peak **310.4 TOPS/W** (I=4b, W=1b); 28nm, CIFAR/ImageNet numbers in JSSC.
- **HW status:** **Silicon** 28nm.
- **Transfer:** Map MSB-skip → **ADP-MAC bit lanes** and **ECP-QKV** “don’t fetch LSBs if coarse correlation flat”; early-stop → **SP-Gate / OF-ECP** when residual energy already below PRRC budget.
- **Remake name:** **MSB-Skip-CIM** + **EarlyStop-Neuron-CIM** (bundle: **NeuroGate-CIM**).
- **vs C1\*/C2\*:** **Enhance** ECP-QKV / SP-Gate / ADP-MAC; **orthogonal** fabric under HBG-RP. Do **not** claim ADC-less as novel alone.
- **Novelty:** **6.5/10** standalone (famous); **8/10** when early-stop criterion = **OF residual / AEE bound** (OF-ECP).
- **Prove-by:** Gate BL toggles vs weight MSB histogram on Spikeformer OF checkpoint; early-stop when |F−F0| energy < ε; plot TOPS/W vs AEE.

### M4 — Spike-encoding sparsity-adaptive charge-domain IF CIM (Spike-CIM)

- **Citation:** Song, Tang, Luo, Xu, Wang, Ji, et al. — *Spike-CIM: A 290TOPS/W Spike-Encoding Sparsity-Adaptive Computing-in-Memory Macro with Differential Charge-Domain Integrate-and-Fire* — **A-SSCC 2022** — DOI:10.1109/a-sscc56115.2022.9980797.
- **Mechanism:** Spike encoding + **sparsity-adaptive** CIM; **differential charge-domain** integrate-and-fire neuron (ADC-light); reported **290 TOPS/W**.
- **HW status:** **Silicon** (A-SSCC macro).
- **Transfer:** Charge-domain IF ≈ hardware LIF accumulator for **binary SDSA scores** (SMAM path) while keeping digital path for ATLIF **payload**.
- **Remake name:** **CD-IF-SpikeCIM** → dual with **DigiCIM-PayloadLane** = **DualRail-CIM**.
- **vs C1\*/C2\*:** **Directly strengthens SMAM-RP / HBG-RP** (gate on CD-IF, payload on digital/ADP-MAC). Orthogonal to OP-STW.
- **Novelty:** **8/10** as dual-rail CIM story for OF-SNN-T; **5/10** if only “290 TOPS/W spike CIM.”
- **Prove-by:** Split SDSA: Mask-Add / score on CD-IF macro model; V/linear ATLIF MAC on digital; ablate vs all-digital SMAM-RP.

### M5 — Twin-column TTFS SRAM CIM (TFSRAM)

- **Citation:** *TFSRAM: A 249.8TOPS/W Timing-to-First-Spike Compute-in-Memory Neuromorphic Processing Engine With Twin-Column SRAM Synapses* — **IEEE TCAS-AI 2024** — DOI:10.1109/tcasai.2024.3452649 (also NSF PDF).
- **Mechanism:** 64×64 **8T-SRAM** twin-column mapping of **+/-** synapses; current-based IF post-neuron; multi-level firing threshold & timing threshold co-design; **249.8 TOPS/W** (8b in / signed 4b W).
- **HW status:** Engine / measured efficiency reported in TCAS-AI.
- **Transfer:** Twin-column ↔ **signed residual / bidirectional flow** (u,v) or excitatory–inhibitory attention; timing threshold ↔ **Motion-TTB Δt** window.
- **Remake name:** **TwinCol-TTFS-OF**.
- **vs C1\*/C2\*:** **Enhance** Motion-TTB / MFBD (time-to-first as bundle priority); orthogonal to ARM-Acc (could supply cheap hyp scores).
- **Novelty:** **7/10** (TTFS CIM known); **8/10** if twin-col stores **flow-signed** weights and timing = OF Δt.
- **Prove-by:** Encode coarse flow hyp as TTFS latency; compare hyp rank quality vs ARM-Acc INT8 scores; energy of twin-col vs digital AAC.

### M6 — Time-domain async SNN CIM (Park TCAS-I)

- **Citation:** Park, Jeong, Kim, Shin, Kim, Lee — *A 701.7 TOPS/W Compute-in-Memory Processor With Time-Domain Computing for Spiking Neural Network* — **IEEE TCAS-I**, 72(1):25–35, 2025 — DOI:10.1109/tcsi.2024.3480350.
- **Mechanism:** Mixed-signal synapse + analog LIF; **no global clock** on neuron/synapse array; MAC in **time domain**; efficiency scales with pre-spike rate; reported peak **701.7 TOPS/W**.
- **HW status:** **Silicon** (TCAS-I).
- **Transfer:** Async event path for **EV-Wake / OP-STW** exact tiles; careful with ATLIF real payloads (time-domain likes spike timing).
- **Remake name:** **TD-CIM-Gate** (gate/event path only).
- **vs C1\*/C2\*:** **Enhance C1\*** EV-Wake; keep **C2\* payload digital** (do not force ATLIF into pure TD). Orthogonal dual path.
- **Novelty:** **6.5/10** as TD-CIM claim; **7.5/10** as **event-gate CIM + digital payload**.
- **Prove-by:** Route DVS/event density through TD-CIM wake; measure false-wake rate vs OP-STW digital; never put irreducible amp on TD path.

### M7 — RRAM nvCIM + in-situ nonlinear / spike activation (Yan VLSI’19)

- **Citation:** Yan, Yang, Chen, Chang, Su, Hsu, Li, Lee, Sheu, Ho, Wu, Chang, Chen, Li — *RRAM-based Spiking Nonvolatile Computing-In-Memory Processing Engine with Precision-Configurable In Situ Nonlinear Activation* — **VLSI Technology 2019** — DOI:10.23919/VLSIT.2019.8776485.
- **Mechanism:** 64Kb 1T1R RRAM macro + **ISNA** (integrate-fire–like) merges A/D and activation; precision **1–8b** configurable; **16.9 TOPS/W**; max spike freq ∼99 MHz; area ≪ ADC scheme.
- **HW status:** **Silicon** hybrid CMOS-RRAM.
- **Transfer:** Park **static** Spikeformer FFN / patch embed / QKV **projection weights** in nvCIM; keep SDSA dynamic on SRAM/digital (endurance!).
- **Remake name:** **ISNA-nvCIM-FFN**.
- **vs C1\*/C2\*:** **Orthogonal fabric** under ECP-QKV (skip programming/fetch of cold projections); enhances energy of C2\* linear layers without replacing HBG-RP.
- **Novelty:** **6/10** nvCIM SNN PE known; **7.5/10** as **static-FFN nvCIM + OF-gated rewrite avoidance**.
- **Prove-by:** Map MLP/FFN W to NVM model (NeuroSim); count writes under MW-ΔBuf (should be rare); AEE vs all-SRAM baseline.

### M8 — Heterogeneous RRAM-CIM (frame CNN) + SRAM near-memory SNN (event) SoC

- **Citation:** Chang, Lele, Spetalnick, Crafton, Konno, Wan, Bhat, Khwa, Chih, Chang, Raychowdhury — *A 73.53TOPS/W 14.74TOPS Heterogeneous RRAM In-Memory and SRAM Near-Memory SoC for Hybrid Frame and Event-Based Target Tracking* — **ISSCC 2023** — DOI:10.1109/isscc42615.2023.10067544; journal: Lele et al. — *A Heterogeneous RRAM In-Memory and SRAM Near-Memory SoC for Fused Frame and Event-Based Target Identification and Tracking* — **JSSC 2024** — DOI:10.1109/jssc.2023.3297411. Related OF algo: Lele & Raychowdhury — *Fusing Frame and Event Vision for High-speed Optical Flow for Edge Application* — **ISCAS 2022** — DOI:10.1109/iscas48785.2022.9937763.
- **Mechanism:** 40nm ULP SoC: **RRAM CIM** for CNN on frames + **SRAM CNM** for SNN on events; dual-level power gating (∼91.8% chip power save with NVM); RRAM **triple error correction**; parallel CNN∥SNN >100 outputs/s; **73.53 TOPS/W** (ISSCC).
- **HW status:** **Silicon** ISSCC/JSSC.
- **Transfer:** Closest **vision-adjacent CIM** prior. Remap: RRAM-CIM → dense **spatial / ref-frame / warped residual CNN-ish front**; SRAM path → **spike SDSA / event wake**. Ties directly to MW-ΔBuf + EV-Wake + OP-STW.
- **Remake name:** **Het-FrameEvent-CIM** (OF remake: **Het-OF-CIM**).
- **vs C1\*/C2\*:** **Enhance C1\*** (frame residual vs event gate) **and** supply a **system template** for C2\* dual fabric. Does **not** replace HBG-RP.
- **Novelty:** **8.5–9/10** for ISCAS OF-SNN-T (tracking≠dense OF, but hetero frame/event CIM is the right *story ancestor*; dense OF residual mapping is still open).
- **Prove-by:** On FlyingChairs/Sintel or event-OF sets: RRAM path = warped-ref residual CNN features; SRAM path = spikeformer SDSA; measure AEE & mJ/frame vs homogeneous SRAM; ablate power-gate with MW-ΔBuf idle tiles.

### M9 — Hybrid NVM projection + SRAM attention for Transformers (X-Former)

- **Citation:** Sridharan, Stevens, Roy, Raghunathan — *X-Former: In-Memory Acceleration of Transformers* — arXiv:2303.07470 → **IEEE TVLSI 2023**, 31(8):1223–1233.
- **Mechanism:** **ReRAM AIMC Projection Engine** for static W (FFN, QKV proj); **8T-SRAM Attention Engine** for dynamic Q·K / attn·V; **sequence blocking** dataflow to pipeline the two; avoids NVM writes for attention.
- **HW status:** Architecture / simulator (PUMA-class), not a single named ISSCC die for the full system.
- **Transfer:** Blueprint for Spikeformer: static proj/FFN → NVM or DigiCIM; **SDSA** → SRAM digital / SMAM-RP (spikes reduce write pressure further).
- **Remake name:** **XForm-Split-SDSA**.
- **vs C1\*/C2\*:** **Orthogonal microarch split** supporting ECP-QKV (fewer live projections) + SMAM-RP. Enhance, don’t replace packages.
- **Novelty:** **7/10** (ANN Transformer IMC known); **8.5/10** if split is **spike-SDSA + ATLIF payload banks** with OF wake.
- **Prove-by:** Cycle-accurate split: % energy in proj vs SDSA on SDformer; show NVM write rate ≈0 under ECP-QKV; compare to all-digital Bishop-style core.

### M10 — ReTransformer / RACE-IT (non-VMM in analog IMC)

- **Citations:**
  - Yang, Yan, Li, Chen — *ReTransformer: ReRAM-based Processing-in-Memory Architecture for Transformer Acceleration* — **ICCAD 2020**.
  - Zhao, Natarajan, Buonanno, et al. — *RACE-IT: A Reconfigurable Analog Computing Engine for In-Memory Transformer Acceleration* — arXiv:2312.06532 → **ICCD 2025**, pp. 103–110.
  - Spoon, Tsai, Chen, et al. — *Toward Software-Equivalent Accuracy on Transformer-Based DNNs With Analog Memory Devices* — **Frontiers in Computational Neuroscience 2021** — DOI:10.3389/fncom.2021.675741 (PCM + quantized attention study).
- **Mechanism:** ReTransformer programs dynamic ops into RRAM (write/endurance pain). RACE-IT: **Compute-ACAM** for Softmax / activations / data-dependent MatMul in analog, ADC reduction. Spoon: noise-aware fine-tune + INT6-ish attention for PCM accuracy.
- **HW status:** Arch / sim (RACE-IT ICCD’25); device-aware sim (Spoon).
- **Transfer:** For **spike** SDSA, Softmax often **absent** — so RACE Softmax is less critical; still useful for **any residual real-valued OF head** or hybrid ANN front. Noise-aware train ↔ ATLIF payload quantization.
- **Remake name:** **ACAM-NonVMM** (use sparingly) + **NoiseAware-ATLIF-Q**.
- **vs C1\*/C2\*:** Mostly **orthogonal / engineering**; do **not** headline Softmax-in-ACAM for spikeformer. NoiseAware helps ADP-MAC bit search.
- **Novelty:** **5/10** Softmax-IMC; **7/10** noise-aware payload for analog/digital mixed OF-T.
- **Prove-by:** If any real-valued attn head remains, compare ACAM vs digital Softmax energy; else skip Softmax claim entirely.

### M11 — Spiking Transformer hybrid AIMC + stochastic spiking attention (Xpikeformer)

- **Citation:** Song, Katti, Simeone, Rajendran — *Xpikeformer: Hybrid Analog-Digital Hardware Acceleration for Spiking Transformers* — arXiv:2408.08794 (v2).
- **Mechanism:** **PCM AIMC** for FFN/FC/embed (row-block mapping, LIF at tile, avoid storing non-binary pre-acts); **SSA engine**: Bernoulli / stochastic computing, AND+count attention, streaming Q/K/V, no intermediate attn-score SRAM write; HWAT + global drift compensation. Reports ∼13× energy vs SOTA digital ANN Transformer ASIC projection; ∼1.9× vs ideal digital SNN Transformer.
- **HW status:** **Simulation** (NeuroSim + 45nm synth for SSA); not a published full-chip measurement.
- **Transfer:** Closest **spiking-Transformer + CIM** cousin. For OF: SSA ↔ **SMAM-RP** family; PCM FFN ↔ ISNA/XForm static path; add **OP-STW / MW-ΔBuf** to cut active tiles (Xpikeformer does not solve OF).
- **Remake name:** **PCM-FFN + SSA-OF** (OF-specialized: **SSA-MotionMask**).
- **vs C1\*/C2\*:** **Enhance** SMAM-RP / Digi path; **orthogonal** C1\* motion front-end. **DO NOT CLAIM** “first spiking Transformer HW” (Bishop ISCA’25, SMAM, FireFly-T, and Xpikeformer’s own claim collide).
- **Novelty:** **7/10** as CIM+spike-T; **8.5/10** with **motion-masked SSA** + ATLIF dual-rail (their SSA is binary/stochastic; your payload is real).
- **Prove-by:** Replace SSA Bernoulli with SMAM-RP mask + ATLIF V path; add ECP-QKV; report energy & AEE on OF benchmarks vs Xpikeformer-like all-spike baseline.

### M12 — Full-precision all-digital SRAM CIM (Chih ISSCC’21) as payload lane

- **Citation:** Chih, Lee, Fujiwara, Shih, Lee, Naous, et al. — *16.4 An 89TOPS/W and 16.3TOPS/mm² All-Digital SRAM-Based Full-Precision Compute-In Memory Macro in 22nm* — **ISSCC 2021** — DOI:10.1109/isscc42613.2021.9365766. Lineage: Si Twin-8T **ISSCC’19**; Khwa binary CIM **ISSCC’18**; Dong 7nm **ISSCC’20**.
- **Mechanism:** Digital 6T-based CIM, bit-serial × adder-tree, programmable act 1–8b / weight 4/8/12/16, signed; **full MAC precision** (no analog SNR loss); 89 TOPS/W, 16.3 TOPS/mm² in 22nm.
- **HW status:** **Silicon** 22nm (TSMC).
- **Transfer:** Perfect **payload / ATLIF amp** CIM lane when binary spike CIM is too lossy for AEE. Pair with Spike-CIM / Neuro-CIM for gates.
- **Remake name:** **DigiCIM-PayloadLane**.
- **vs C1\*/C2\*:** **Enhance ADP-MAC / HBG-RP** implementation; saturated as *standalone* CIM paper.
- **Novelty:** **5/10** alone; **7.5/10** as half of **DualRail-CIM** with CD-IF-SpikeCIM.
- **Prove-by:** Bitwidth sweep on ATLIF amp MAC in DigiCIM vs SRAM+ALU; show AEE cliff if forced to 1b CIM.

### M13 — Motion-warped / sparse update into CIM tiles (OF-specific composition)

- **Citation (composition, not a single CIM paper):** MotionDeltaCNN / MW-ΔBuf from Round-2 `08`; Het SoC M8; IMPULSE sparsity M1.
- **Mechanism:** Only **rewrite / activate** CIM rows/tiles whose **warped residual** or event energy exceeds threshold; cold tiles power-gated (Lele-style). Avoids NVM write & SRAM BL activity.
- **HW status:** Composition of measured SoC gating + algorithmic Δ.
- **Remake name:** **MW-CIM-TileGate** (ties **MW-ΔBuf ⊗ CIM**).
- **vs C1\*/C2\*:** **Enhance C1\*** MW-ΔBuf / OP-STW into the **memory macro**; orthogonal to C2\* math.
- **Novelty:** **9/10** for ISCAS (CIM literature rarely keys activity by **optical-flow warp residual**).
- **Prove-by:** % CIM tile wake on Sintel under MW-ΔBuf vs frame-diff vs always-on; energy & AEE.

### M14 — Dual-rail gate CIM + payload DigiCIM (HBG-RP on CIM fabric)

- **Citation (composition):** HBG-RP / SMAM-RP (R1/R2); Spike-CIM M4; Chih DigiCIM M12; Neuro-CIM M3.
- **Mechanism:** Binary / spike **gate** accumulates in charge-domain or 1b neuromorphic CIM; **real ATLIF payload** only when gate=1, on digital full-precision CIM or ADP-MAC. Same semantic as HBG-RP, new **memory substrate**.
- **Remake name:** **DualRail-CIM** (= CD-IF-SpikeCIM ∥ DigiCIM-PayloadLane).
- **vs C1\*/C2\*:** **Implementation enhance** of HBG-RP / SMAM-RP — **do not replace** the package name in the paper claim.
- **Novelty:** **8.5/10** as fabric; claim as “HBG-RP mapped onto dual-rail CIM,” not as new algorithm.
- **Prove-by:** Gate toggle energy in Spike-CIM model; payload MAC count with/without gate; match digital HBG-RP AEE.

---

## 3. Ranked shortlist for ISCAS OF-SNN-Transformer (CIM remakes)

| Rank | Remake | Why it upgrades SDformer | Primary package link | Suggested novelty |
|---|---|---|---|---|
| 1 | **Het-OF-CIM** | Only strong **vision/event+frame CIM SoC** ancestor; maps to residual vs spike paths | C1\* MW-ΔBuf / EV-Wake + C2\* fabric | 8.5–9 |
| 2 | **MW-CIM-TileGate** | Makes CIM activity **OF-semantic** (unsaturated) | C1\* MW-ΔBuf / OP-STW | 9 |
| 3 | **DualRail-CIM** | Physical substrate for HBG-RP / SMAM-RP | C2\* HBG-RP / SMAM-RP | 8.5 |
| 4 | **Fused-WVMEM-CIM** | Attacks ATLIF Vmem traffic (SNN-specific, silicon) | C2\* HBG-RP / ADP-MAC | 8 |
| 5 | **XForm-Split-SDSA** | Static NVM FFN vs dynamic SDSA — Transformer-correct split | C2\* + ECP-QKV | 8.5 |
| 6 | **PCM-FFN + SSA-OF** | Spiking-T + AIMC prior; specialize with motion mask | SMAM-RP / Motion-TTB | 8–8.5 |
| 7 | **NeuroGate-CIM** (MSB-Skip + EarlyStop) | BL activity × OF-ECP / SP-Gate | SP-Gate / OF-ECP / ADP-MAC | 8 |
| 8 | **PRRC-CIM-Modes** (SpiDR-class) | Pyramid / head-type reconfig | PRRC / STH-Gate | 8 |
| 9 | **TwinCol-TTFS-OF** | Time+sign structure for hyp / Δt bundles | Motion-TTB / ARM-Acc | 7–8 |
| 10 | **ISNA-nvCIM-FFN** | Dense static W, spike-friendly readout | ECP-QKV cold FFN | 7.5 |
| 11 | **TD-CIM-Gate** | Async event wake lane | EV-Wake | 7.5 |
| 12 | **DigiCIM-PayloadLane** | Accurate ATLIF amp CIM (supporting actor) | ADP-MAC | 7.5 as duo |

**Engineering order (does not block Card A/B):**  
Document DualRail / Fused-WVMEM / Het-OF as **Card F (CIM fabric)** after OP-STW + HBG-RP RTL exist; MW-CIM-TileGate shares OP-STW wake bitmap.

---

## 4. How Round-3 CIM upgrades C1\* / C2\*

### C1\* — Temporal Residual + Eager Front-End
| Add | Role |
|---|---|
| **MW-CIM-TileGate** | Wake/power-gate **CIM tiles** from warped residual, not only PE arrays |
| **Het-OF-CIM** | System pattern: frame/ref path on RRAM-CIM, event/spike path on SRAM |
| **PRRC-CIM-Modes** | Bit/neuron mode per pyramid level |
| **TD-CIM-Gate / EV-Wake** | Async event lane into exact OF path |
| **ECP-QKV × ISNA/XForm** | Eager skip also avoids **NVM/CIM activation** of dead projections |

**Upgraded C1\* claim angle (safe):**  
“Front-end residual/event energy schedules **both** PE wake and **in-memory tile activity** (MW-CIM-TileGate / Het-OF), not zero-skip alone.”

### C2\* — Dual-Rail + SpatioTemporal Sparse Fabric
| Add | Role |
|---|---|
| **DualRail-CIM** | Gate = spike/charge CIM; payload = DigiCIM / ADP-MAC |
| **Fused-WVMEM-CIM** | Keep ATLIF amp beside W inside macro |
| **XForm-Split-SDSA** | NVM/static vs SRAM/dynamic attention split |
| **SSA-OF / SMAM-RP** | Stochastic or Mask-Add attention without Softmax IMC cosplay |
| **NeuroGate-CIM** | MSB-skip / early-stop as SP-Gate / OF-ECP physical levers |
| **TwinCol-TTFS-OF** | Optional hyp / Δt ranking macro |

**Upgraded C2\* claim angle (safe):**  
“HBG-RP / SMAM-RP are realized on a **dual-rail CIM fabric** (binary gate CIM ∥ full-precision payload CIM), with Transformer-correct **static/dynamic memory split** — not a generic SRAM-CIM MAC macro.”

---

## 5. DO NOT CLAIM (CIM edition)

1. **“First CIM for SNN”** — IMPULSE, Neuro-CIM, Yan VLSI’19, Spike-CIM, TFSRAM, Park TCAS-I, SpiDR, …  
2. **“First CIM / IMC for Transformers”** — ReTransformer ICCAD’20, X-Former TVLSI’23, RACE-IT ICCD’25, Spoon Frontiers’21, TransPIM HPCA’22, …  
3. **“First spiking Transformer hardware”** — Xpikeformer (sim), Bishop ISCA’25, SMAM arXiv’25, FireFly-T, …  
4. **Generic TOPS/W leadership** from SRAM-CIM MAC without OF workload — saturated ISSCC slogan space (Khwa’18, Si’19, Chih’21, …).  
5. **ADC-less as sole novelty** — Neuro-CIM / Spike-CIM already sold this for SNN.  
6. **Softmax-in-RRAM as spikeformer headline** — spike SDSA typically Softmax-free; ReTransformer write pain is a warning, not a feature.  
7. **Analog SNR miracles for AEE-critical OF** without DigiCIM / dual-rail — AEE is brittle; keep payload digital or full-precision DigiCIM.  
8. Do **not** overwrite `hw_autoresearch_nts07` Codex trees; CIM remakes stay in ideafromai naming until a **new Card F** is opened.

---

## 6. Prove-by experiment ladder (suggested)

1. **Traffic study (ep34 / Sintel):** Vmem + W + QKV projection bytes vs fused-WVMEM; % tiles with non-zero residual under MW-ΔBuf.  
2. **Macro model:** Spike-CIM / Neuro-CIM style energy model for **gate**; Chih DigiCIM model for **payload**; compare DualRail vs all-digital HBG-RP.  
3. **Het ablation:** RRAM-static FFN on/off; SRAM SDSA always; power-gate with OP-STW bitmap.  
4. **Accuracy:** Force 1b CIM on ATLIF amp → show AEE collapse → justify DigiCIM-PayloadLane.  
5. **Paper figure:** one diagram = Het-OF / DualRail / MW-CIM-TileGate overlaid on C1\*→C2\* pipeline.

---

## 7. Citation index (verified anchors only)

| Key | Venue / ID |
|---|---|
| IMPULSE | arXiv:2105.08217 (65nm digital fused W+Vmem CIM) |
| SpiDR | arXiv:2411.02854 (65nm reconfig digital CIM SNN) |
| Neuro-CIM | VLSI’22 DOI:10.1109/vlsitechnologyandcir46769.2022.9830276; JSSC’23 extension; HotChips’22 |
| Spike-CIM | A-SSCC’22 DOI:10.1109/a-sscc56115.2022.9980797 |
| TFSRAM | TCAS-AI’24 DOI:10.1109/tcasai.2024.3452649 |
| Park TD-CIM SNN | TCAS-I 72(1) 2025 DOI:10.1109/tcsi.2024.3480350 |
| Yan RRAM ISNA | VLSI Tech’19 DOI:10.23919/VLSIT.2019.8776485 |
| Chang/Lele Het SoC | ISSCC’23 DOI:10.1109/isscc42615.2023.10067544; JSSC’24 DOI:10.1109/jssc.2023.3297411 |
| Lele OF fuse (algo) | ISCAS’22 DOI:10.1109/iscas48785.2022.9937763 |
| Chih DigiCIM | ISSCC’21 DOI:10.1109/isscc42613.2021.9365766 |
| Khwa binary SRAM-CIM | ISSCC’18 DOI:10.1109/isscc.2018.8310401 |
| Si Twin-8T | ISSCC’19 DOI:10.1109/isscc.2019.8662392 |
| X-Former | arXiv:2303.07470 / TVLSI 31(8) 2023 |
| ReTransformer | ICCAD 2020 |
| RACE-IT | arXiv:2312.06532 / ICCD 2025 pp.103–110 |
| Spoon PCM Transformer | Front. Comput. Neurosci. 2021 DOI:10.3389/fncom.2021.675741 |
| Xpikeformer | arXiv:2408.08794 |
| Bishop (digital cousin) | ISCA 2025 arXiv:2505.12281 |
| SMAM (digital cousin) | arXiv:2501.07825 |

---

## 8. Relation to prior rounds

| Round | File | What it added |
|---|---|---|
| R1 | `01`–`04` | OP-STW, PRRC, OGEC, HBG-RP, ADP-MAC, ARM-Acc, MFBD, SP-Gate |
| R2 | `07`, `08`, `09` | ECP-QKV, MW-ΔBuf, SMAM-RP, Motion-TTB, STH-Gate, … |
| **R3 (this)** | `10_cim_spike_opticalflow_accelerators.md` | **CIM fabric remakes** DualRail / Het-OF / MW-CIM-TileGate / Fused-WVMEM / XForm-Split / … |

**Sync note:** Round-3 CIM feeds **ideafromai**; directory sync to `hw_innovation_research` is handled by the parent agent. **Do not** modify `hw_autoresearch_nts07`.

---

*End of ROUND 3 — CIM / spike / OF / sparse-transformer survey.*
