# V7 — local unread P0 sweep (round 4 wide)

Date: 2026-09-11. Full texts only from `survey_ab_fusion_20260910/p0_txts/`. Not a novelty claim. Not RTL. Not a PPA invention. Not a CIM title.

**Identity freeze** (`IDENTITY_ATLIF.md`): AT-LIF \(o=\theta\cdot H(m-\theta)\in\{0,\theta\}\). Inference: layer-shared \(\theta\) **absorbed** into next \(W\). Transmitted spikes are \(\{0,1\}\). Spike path after absorb is **binary GeMM**. Residual / PED / I24 is a **separate continuous tensor**, not analog AT-LIF amplitude shared by two MACs. Dual-consumer, if it exists: binary gate + continuous residual, never “continuous AT-LIF payload reused by two MACs.”

**Class vs absorb-cut identity**

| Tag | Meaning |
|---|---|
| **A** | Complete prior that must be copied as control on the *post-absorb binary spike path* (or on a named adjacent object). |
| **B** | Strong negative: identity-illegal, deletes the continuous residual consumer, or is a different object sold as the same. |
| **X** | Adjacent (event OF algo, tokenizer, train, compiler, CIM). Keep as background; not the letter’s silicon object. |
| **already A** | Mapped in rounds 1–3; not re-litigated. |

Corpus: priority list + all `2602.*`–`2609.*` except EventShiftFlow (`2605.28312`, already read) and ExSpike (`2606.20414`, already mapped). Depth: title/abstract triage, then full local text on hits for SNN HW, transformer HW, event OF, fusion, last-use, BN, residual.

---

## Relevant deep notes (≥20)

### 1. 2501.07825 — An Efficient Sparse Hardware Accelerator for Spike-Driven Transformer

**Venue (stated):** IEEE (index terms SNN / accelerator); FPGA eval on Xilinx Virtex UltraScale, CIFAR-10 Spike-driven Transformer.

**Mechanism.** Encodes *addresses of fired spikes* into Encoded Spike SRAM instead of dense bitmaps. SEA (Spike Encoding Array) writes `Pos` when membrane ≥ \(V_{th}\). Downstream units consume only those addresses: SMU turns 2×2 maxpool into OR-of-covered positions; SMAM does SDSA on **dual spike** Q/K by address comparison / mask-add, not MAC; SLA is spike-linear (add/compare). Tile Engine does conv on unencoded maps. **ResBuffer + Adder Module** sit after the Tile Engine for residual add, then the residual result is re-encoded into the SDEB core. Dual-spike SDSA is the sold HW object; residual is a separate adder path, not analog spike amplitude.

**Class.** **A** for post-absorb binary SDSA as address-compare / mask-add (same family as FireFly-T AND-PopCount / Spike-IAND). **A** for “residual is a fourth buffer, not the spike FMap.” **B** as dual-consumer last-use of a continuous residual from the *same* producer as the binary gate — residual add is generic skip-add after conv, then re-spike. Classification FPGA, not event OF.

---

### 2. 2501.11554 — Event-based vision for egomotion estimation using precise event timing

**Venue:** neuromorphic / on-chip mixed-signal (180 nm cognigr1 TDE synapse); not a digital SNN-Transformer.

**Mechanism.** Fully event-domain egomotion: no event frames. Time Difference Encoder (TDE) synapse (FAC/TRG decaying traces) converts inter-event delay into analog EPSC, then a silicon LIF bursts with ISI inverse to \(\Delta t\). Local optical-flow velocity is that burst rate; a shallow SNN reads out ego motion. Chip experiment on spatially downsampled car-mounted events; larger TDE net in simulation. Object is **asymmetric correlation / insect EMD**, not dense RAFT cost volume, not transformer SSA.

**Class.** **X** event-OF *sensor-circuit* prior (timing → analog current → LIF). **B** as a digital post-absorb binary GeMM / residual dual-path OF letter. Does not absorb \(\theta\) into \(W\); payload is analog TDE current.

---

### 3. 2501.13610 — Efficient Synaptic Delay Implementation in Digital Event-Driven AI Accelerators

**Venue:** digital neuromorphic (Seneca / IMEC); delay HW.

**Mechanism.** Shared Circular Delay Queue (SCDQ): two FIFOs (PRQ/POQ) in a ring instead of a FIFO *per delay level*. An event enters the shared queue **once**; AER packet is extended with delay metadata. Memory scales with **activation density**, not \(D\times I\times J\) synapses. Evaluated vs Loihi/TrueNorth delay structures on Seneca.

**Class.** **A** for event lifetime / last-use *of AER packets* (circular shared buffer keyed by spike density). **X** for our letter: synaptic delay is not AT-LIF absorb, not residual PED, not transformer. Do not sell SCDQ as typed last-use of a continuous residual.

---

### 4. 2501.14484 — SpikePack: Enhanced Information Flow in SNNs with High Hardware Compatibility

**Venue:** (cs.NE); Spikeformer-8-512 experiments; FPGA XCZU3EG @ 300 MHz.

**Mechanism.** Neuron computes a **global potential** \(v_g^l\) from a time-zipped spike tensor `slzip` so forward can be serial (LIF-like leak/reset) or **parallel O(1)** in \(T\). Claims to keep reset + leak while reducing transmission loss vs step-by-step LIF. Compatible with ANN topologies / conversion. FPGA implements the parallel neuron; sparsity still binary spikes. Spikeformer numbers are GPU/FPGA classification, not OF.

**Class.** **A** for *causal/parallel tick-batching of a reset-LIF* (same question as Spike-IAND mux `111/101/000`). **B** as a T10 noncausal PSN mix substitute, and **B** as analog residual: output is still spikes after threshold. Parallel \(v_g\) is a membrane, not PED/I24.

---

### 5. 2501.14490 — Multiplication-Free Parallelizable Spiking Neurons with Efficient Spatio-Temporal Dynamics

**Venue:** NeurIPS 2025.

**Mechanism.** Mul-free channel-wise PSN: sliding PSN → depthwise conv over time with **sawtooth dilations** to cut order \(k\), then **bit-shifts** replace neuronal multiplies. \(X\in\mathbb{R}^{T\times C}\), integer \(W\), hidden \(H\), Heaviside vs \(V_{th}\) → \(S\in\{0,1\}\). Explicitly contrasts PSN / masked PSN dense float GeMM and sliding PSN large-\(k\) conv. GPU autoselect kernels; INT8 45 nm area/energy *model* of the shift vs MAC (not a chip). Tasks: SHD, seq-CIFAR, DVS-Lip.

**Class.** **A** for *pre-threshold temporal mix that is still a neuron, then binary spike out* — closest local unread cousin of T10 PSN, but **causal dilated conv + shift**, not noncausal dense \(H=WX\). After absorb, the mix is still **before** threshold; it does not make AT-LIF analog. **B** as residual dual-MAC. Hardware-friendly claim is neuron ALU, not a transformer overlay.

---

### 6. 2503.10195 — ST-FlowNet: An Efficient SNN for Event-Based Optical Flow Estimation

**Venue:** journal preprint (cs.CV); event OF.

**Mechanism.** U-Net-like flow net with **ConvGRU** for cross-scale temporal alignment of predicted flow. Train ANN, then derive SNN by (i) standard ANN-to-SNN or (ii) **BISNN** (bio-info fusion, parameter-free conversion). Event frames in → dense flow out. Energy is estimated from spike counts, not silicon. Benchmarks: event OF datasets (not a DSEC silicon loop).

**Class.** **X** SNN event-OF *algorithm*. ConvGRU is a **continuous recurrent state** beside spikes — dual-path in the ANN sense, **not** AT-LIF absorb. **B** as FPGA/ASIC prior. Do not copy BISNN as \(\theta\to W\).

---

### 7. 2503.15986 — SpiLiFormer: Enhancing Spiking Transformers with Lateral Inhibition

**Venue:** (cs.NE); ImageNet-1K / CIFAR / DVS classification.

**Mechanism.** Diagnoses “attention distraction”: SSA/SDSA inherited from softmax(\(QK^\top\))V over-allocates to background. Lateral-inhibition pathway processes Q/K/V **separately**: excitatory center vs inhibitory surround (retina RF), implemented as extra mask / difference paths on spike attention, not a second MAC on a residual tensor. SOTA classification vs E-SpikeFormer with fewer params / \(T\).

**Class.** **X** spiking-transformer *attention algebra*. **B** as hardware last-use or residual PED. Inhibition is a **gate on attention weights**, not absorb-cut identity.

---

### 8. 2505.07556 — Self-Supervised Event Representations: Towards Accurate, Real-Time Perception on SoC FPGAs

**Venue:** SPIE-style (Kryjak / AGH); SoC FPGA GRU.

**Mechanism.** Per-pixel GRU stack updates a dense latent memory from **one event at a time** (no time quantization). Self-supervised to reconstruct timestamps/polarity. After the window, the latent tensor is a dense representation for a frame CNN/ViT detector (Gen1 / 1 Mpx). **First hardware recurrent event representation** on SoC FPGA: sub-µs per event, 1–2 W.

**Class.** **X** event *frontend* (asynchronous dense tensor). **A** only as “event → dense map on FPGA before a dense consumer.” **B** as SNN-Transformer / AT-LIF. GRU state is continuous; not binary GeMM after absorb.

---

### 9. 2506.03512 — EDCFlow: Exploring Temporally Dense Difference Maps for Event-based Optical Flow Estimation

**Venue:** (cs.CV); RAFT-like event OF.

**Mechanism.** Avoids high-res 4D cost volumes. High-res (1/4) **feature-difference** maps between adjacent event frames (\(O(TNC)\)) capture motion boundaries; low-res **correlation** volume is robust to noise. Adaptive fusion of the two motion features; iterative GRU update. Plug-in refinement for RAFT-like event methods (E-RAFT / TMA family).

**Class.** **X** event-OF *fusion of two motion tensors* (difference + correlation). **B** as SNN / absorb / last-use. The dual tensors are both **dense ANN features**, not binary spike + continuous residual from one AT-LIF producer.

---

### 10. 2506.07878 — Spatio-Temporal State Space Model For Efficient Event-Based Optical Flow

**Venue:** CVPR Workshops 2025. DSEC public benchmark.

**Mechanism.** STSSM module (structured SSM / S4-S5 vs Mamba) extracts spatio-temporal event features cheaper than ViT/CNN. Convex upsampling (RAFT-style) replaces U-Net decoder. Claims 4.5× faster / 8× fewer ops than TMA, 2× fewer than EV-FlowNet, competitive DSEC. Explicitly: SNNs/GNNs are cheaper but **lose** accuracy; this paper stays dense SSM, not spike.

**Class.** **X** efficient dense event-OF backbone (DSEC numbers are **theirs**). **B** as SNN HW. Useful only as AEE/compute *control table*, not a mechanism we can absorb.

---

### 11. 2507.09780 — BitParticle: Partializing Sparse Dual-Factors to Build Quasi-Synchronizing MAC Arrays

**Venue:** (cs.AR); DNN bit-sparsity MAC.

**Mechanism.** Sign-magnitude 8-bit; split each operand into particles (1+2+2+2), cross-multiply 16 products, drop zero products, regroup to ≤7 PPs (no PP explosion vs EBS). Approximate variant discards small IRs. **Quasi-synchronous** MAC array: cycle-level elasticity so dual-factor bit-sparsity does not stall the whole array (vs Laconic group-sync).

**Class.** **A** for *dual-factor sparsity scheduling* on a MAC array (product sparsity family, Prosperity-adjacent). After absorb, spikes are 1-bit so “activation bit-sparsity” collapses to value sparsity. **X** for ternary/int8 weights × binary spikes. **B** as SNN-Transformer overlay.

---

### 12. 2508.12637 — HOMI: Ultra-Fast EdgeAI platform for Event Cameras

**Venue:** (cs.AR); Zynq UltraScale+ + Prophesee IMX636.

**Mechanism.** End-to-end: MIPI from IMX636 → PL preproc (constant-time **and** constant-event histogram, linear/exponential **time surfaces**) → RAMAN CNN accelerator that skips structured+unstructured sparsity. DVS Gesture: 94% high-accuracy mode or 1000 fps low-latency; 33% LUT, 10×5 cm board. Frontend is dense maps; backend is **CNN**, not SNN/transformer.

**Class.** **A** for *event-camera FPGA frontend* (histogram / time-surface) that any OF/SNN letter must name as prior if it claims “first event-to-FPGA.” **B** as spikeformer / absorb / residual. RAMAN sparsity is CNN zero-skip.

---

### 13. 2508.19806 — Context-aware Sparse Spatiotemporal Learning for Event-based Vision (CSSL)

**Venue:** IEEE (event OF + detection).

**Mechanism.** Learned **context-aware thresholds** on conv/recurrent modules: threshold depends on input distribution, not a fixed ReLU. No extra sparsity loss. Applied to event object detection and **optical flow**. Activation density drops while accuracy holds. Aimed at neuromorphic MatMul-free processors, but the paper is an **ANN/RNN training method**, not a chip.

**Class.** **X** activity-thresholding (cousin of AT-LIF \(\theta\), but \(\theta\) is **not** absorbed into \(W\); it gates dense activations). **B** as official AT-LIF identity. Do not rename CSSL as absorb-cut.

---

### 14. 2509.18968 — OTTERS: Energy-Efficient Spiking Transformer via Optical Time-to-First-Spike Encoding

**Venue:** (cs.LG) preprint; GLUE (language), 22 nm energy *model*.

**Mechanism.** TTFS (≤1 spike/neuron) normally needs digital decay \(f(t)\) × weight. Authors fabricate InOx optoelectronic synapse whose **physical decay IS \(f(t)\)**; analog output is treated as fused \(w\cdot f(t)\). QNN→SNN conversion to train a transformer under this encoding. Energy model includes compute + movement + memory on 22 nm.

**Class.** **X** device-physics TTFS. **B** vs absorb-cut: TTFS payload is **timing/analog decay**, not \(\{0,1\}\) after \(\theta\to W\). Not event OF. Otters++ (`2606.13016`) is the same family (device-faithful forward, QNN STE backward) — same class, not a second object.

---

### 15. 2510.26614 — Spiking Patches: Asynchronous, Sparse, and Efficient Tokens for Event Cameras

**Venue:** IROS 2026.

**Mechanism.** Tokenizer groups events in **space and time** into tokens that stay asynchronous and spatially sparse (no fixed window frame). Tokens feed GNN, point-cloud net, **or Transformer**. Gesture / detection: up to 3.4× vs voxel tokens, 10.4× vs frames, accuracy matched or +1.4–3.8. “Spiking” names the tokenizer, not a LIF net.

**Class.** **X** event tokenization for transformers. **A** only as “do not convert events to dense frames before attention.” **B** as SNN HW / residual last-use.

---

### 16. 2511.06770 — ASTER: Attention-based Spiking Transformer Engine for Event-driven Reasoning

**Venue:** (cs.AR); **PIM / RRAM** (CIM-class; out of scope *as title*).

**Mechanism.** Hybrid analog-digital PIM for Spike-Driven Transformer (SDT). Software: layer skip when attention firing < TAFT (~5%); confidence-based timestep cut. Hardware: RRAM crossbars, time-multiplexed **membrane buffers**, spike-stationary / weight-stationary dataflow, bitwise mask-add SDSA, IAND residual cited from Spike-IAND. Tasks: ImageNet, CIFAR10-DVS, DVS Gesture. Explicit: **SDT removes non-spiking residuals** for full integer / spike-only inference.

**Class.** **already-illegal residual (B)** if sold as IAND skip (round-2/3). CIM **stop as title**. Useful **A-control** only for: membrane-state persistence across \(T\); spike-stationary reuse; skip of dead attention layers. Not absorb-cut: they keep analog membrane *on-tile*, spikes still 0/1.

---

### 17. 2511.21910 — Platinum: Path-Adaptable LUT-Based Accelerator Tailored for Low-Bit Weight Matrix Multiplication

**Venue:** (cs.AR) ASIC 0.96 mm²; BitNet b1.58-3B.

**Mechanism.** LUT GEMV for integer-weight mpGEMM. **Offline construction paths** (vs Prosperity runtime “shortcuts” + dynamic scheduling). Bit-serial path for general integer W; **switch path** to ternary LUT (no 2-bit serial + partial-sum merge). Compares speed/energy vs SpikingEyeriss, **Prosperity**, T-MAC. Argument: BitNet-uniform ternary → high LUT occupancy → Prosperity’s dynamic scheduler is wasted area (24% / 32.3% power in their telling of Prosperity).

**Class.** **A** for LUT / product-reuse GeMM **after absorb** if W is ultra-low-bit (Prosperity family). **B** as SNN: activations here are **not** binary spikes; they are integer activations × ternary W. Do not retitle Prosperity. Numbers are BitNet LLM, not DSEC.

---

### 18. 2512.17555 — ISSCC 2025 23.2 ConvFormer accelerator (hybrid-attention layer-fusion + cascaded pruning)

**Venue:** **ISSCC 2025** Session 23; 28 nm; SegFormer / PVT Cityscapes. Printed title: memory-compute-intensity-aware CNN-Transformer with HAPU + LFS + CFMP.

**Mechanism.** (1) HAPU: most Q tiles use **linear attention** (\(K^\top V\) first, on-chip \(C\times C\)), few tiles vanilla \(QK^\top\) for global RF; ATM tiles Q if LMB would overflow; RMPI precomputes \(K^\top,V,K^\top V\) and routes \(K^\top\) LMB→RMB. (2) **LFS layer-fusion**: KV-reused VACF then weight-reused LACF; non-overlapped fused conv so VA/LA boundary tiles need no overlapping LF; broken RF recovered by next attention. (3) CFMP: conv \(W_0,W_1\) with expanded sparse Z, same mask restores density; also mapped onto sparse VA. SIMD + 2 HAPU + 2MB LMB + 1MB RMB.

**Class.** **A** for *ANN CNN–Transformer layer-fusion / KV-weight reuse / attention tiling* — the strongest local unread **fusion** silicon. **B** as SNN: dense/int activations, no spikes, no AT-LIF, no event OF. Letter that looks like ISSCC 23.2 + ESTU (hybrid attention overlay + fusion + mW) is desk-reject class. Residual is not discussed as a dual last-use object.

---

### 19. 2512.20073 — 3DS-ISC: Accelerating Time-Surface Construction for Neuromorphic Event Cameras

**Venue:** (cs.AR); 3D-stack in-sensor; DRAM leakage as analog decay.

**Mechanism.** Time-surface \(TS=\exp(-(t-t_{last})/\tau)\) built **in-sensor** using DRAM leakage + MOMCAP + low-leakage switch instead of 16-bit SRAM timestamps. 3D stack vs 2D: power/latency/area reductions (their numbers). Downstream: STCF denoise, GoogleNet on TS (N-MNIST etc.), DAVIS240C reconstruction. Analog timestamp, not a network accelerator.

**Class.** **X** in-sensor TS (CIM-adjacent; **CIM stop as title**). **A** only as time-surface *construction* prior next to HOMI’s digital TS. **B** as SNN-Transformer.

---

### 20. 2601.02613 — Sparsity-Aware Streaming SNN Accelerator with Output-Channel Dataflow for AMC

**Venue:** IEEE J-LAT / FPGA; RadioML 2016 AMC (not vision).

**Mechanism.** FINN-style **streaming** (one HW module per layer, weights local) **plus** sparsity. Gated one-to-all product (**GOAP**): iterate nonzero **weights** (fixed at inference), precompute which IFM pixels hit; extra/empty iterations **baked into the schedule** so no runtime decoder / load balancer. Output-channel dataflow balances work. Claims first streaming SNN that uses both temporal (spike) and spatial (weight) sparsity without extra control. 23.5 MS/s, ~2× baseline throughput.

**Class.** **A** for *compile-time last-use / intersection schedule* on binary IFM × sparse W (GOAP vs LoAS runtime bitmap). After absorb this is legal on the spike path. **X** workload (RF AMC). **B** as transformer / residual / event OF.

---

### 21. 2602.12590 — Unbiased Gradient Estimation for Event Binning via Functional Backpropagation (EventFBP)

**Venue:** ICLR 2026.

**Mechanism.** Event binning is discontinuous → biased grads if you smooth the bin. Lift binning to a functional; integration-by-parts + reconstructed cotangent gives **weak derivatives** matching long-range finite differences, forward unchanged. Improves optimization-based egomotion, **self-supervised OF EPE**, SLAM.

**Class.** **X** training of event binning (OF-relevant). **B** as HW.

---

### 22. 2602.23204 — Motion-aware Event Suppression for Event Cameras

**Venue:** RSS 2026.

**Mechanism.** Joint IMO segmentation + dense OF; warp mask by \(\Delta t\) to **suppress future IMO events**. Downstream: ViT token pruning; VO feature pruning. Disentangles ego-motion edges vs independent movers.

**Class.** **X** event-OF *filter* (IMO vs ego). **B** as SNN. Legal background if a letter gates spikes by predicted flow — but **G2 stop** still applies: do not use current-frame flow as skip oracle.

---

### 23. 2603.15184 — CATFormer: Continual Learning Meets Spiking Transformers With Dynamic Thresholds

**Venue:** AAAI 2026 Neuro workshop (PMLR).

**Mechanism.** **DTLIF**: context-adaptive **per-task thresholds** as the forgetting defense (neuromodulation), not extra synapses. After task 0, freeze W, learn thresholds + Gated Dynamic Head Selection for task-agnostic inference. CIFAR / Tiny-IN / CIFAR10-DVS / SHD.

**Class.** **B** vs absorb-cut: official identity **absorbs layer-shared \(\theta\) into \(W\)**. CATFormer *needs* \(\theta\) **not** absorbed (it *is* the memory). Different object. **X** CL. Do not un-absorb \(\theta\) to copy this.

---

### 24. 2604.03626 — L-SPINE: Low-Precision SIMD Spiking Neural Compute Engine

**Venue:** FPGA (AMD VC707); RISC-V + SNN core.

**Mechanism.** Unified 2/4/8-bit datapath; **multiplier-less shift-add** for neuron + synaptic acc. Neuron 459 LUT / 0.39 ns / 4.2 mW; system 0.54 W, 2.38 ms. INT2/INT4 memory shrink. Classic FC/conv SNN, not transformer.

**Class.** **A** for shift-add LIF ALU (with 2501.14490). **X** scale. **B** as overlay / residual / OF.

---

### 25. 2604.28059 — NeuroRing: Multi-FPGA Bidirectional Ring + Stream-Dataflow SNNs

**Venue:** (cs.AR); HLS FPGA; NEST-compatible.

**Mechanism.** Stream-dataflow SNN tiles on a **bidirectional ring**, 1–N FPGAs. Cortical microcircuit RTF 0.83; Sudoku CSP. Spike communication/sync is the bottleneck they schedule, not GeMM sparsity.

**Class.** **X** multi-FPGA neuroscience sim. **B** as event-OF transformer letter.

---

### 26. 2605.12217 — Heterogeneous SoC integrating ReckOn (recurrent SNN) on FPGA

**Venue:** (cs.AR); X-HEEP RISC-V + Zynq ARM + ReckOn.

**Mechanism.** Wraps the taped-out ReckOn recurrent SNN accelerator in an open SoC; reproduces classification; online learning on Braille digits.

**Class.** **X** recurrent SNN FPGA integration. **B** as transformer/OF.

---

### 27. 2605.13869 — Elastic Spiking Transformers for Efficient Gesture Understanding

**Venue:** (cs.NE); Loihi/SpiNNaker-oriented; EHWGesture + CIFAR/DVS.

**Mechanism.** Matryoshka nested width in Feature Extractor, SSA heads, FFN. One universal net; **slice width/heads at runtime** without retraining. Elasticity cuts **parameter count and average firing rate** (hence synaptic ops) — SNN-specific, not just slim ANN.

**Class.** **X** runtime width elasticity (cousin of ELSA *time* elasticity). **B** as absorb / residual last-use. Firing-rate drop is activity sparsity, already A on the binary path.

---

### 28. 2605.20802 — ELSA: ELastic SNN Inference Architecture

**Venue:** (cs.AR); near-SRAM dataflow; ImageNet / COCO.

**Mechanism.** SNNs can emit early answers (elastic inference). Prior SNN chips are **layer-by-layer** or **T-then-layer**, synchronizing **all spines/tokens** before forward. ELSA: **spine/token-wise pipeline** — forward each completed token immediately (first-response \(O(L)\)). BAER bundles spikes to cut NoC. **Mini-batch spiking Gustavson product** for MM-sc; MM-ss = two MM-sc with spike tracers as continuous. Residual add and image-to-column listed as extra ops beside Gustavson. ST-BIF ≈ quantized ReLU.

**Class.** **A** for *token-wise last-use / streaming pipeline* and **Gustavson on spike×W** (must copy vs GustavSNN / Ventaglio). **A** that residual add is a **separate op** next to sparse MM. **B** as dual-MAC on analog AT-LIF. Elastic *time* is not T10 PSN.

---

### 29. 2605.21333 — SymbolicLight V1: Spike-Gated Dual-Path Language Modeling

**Venue:** (cs.CL) preprint.

**Mechanism.** Binary LIF spikes **gate** selected projections; a **continuous residual stream** carries content. Dual-Path SparseTCAM: exponential-decay state (half-life 8–13 tokens) + windowed softmax attention on the **continuous** residual; fusion gate. Encoder ~90% zero spikes. **Implementation still uses dense kernels** (zeros do not skip MAC). FFN up-proj and local attn consume continuous; decay path and FFN down-proj take binary spikes.

**Class.** **A** as *algorithmic dual-path*: binary gate + continuous residual — **same split as locked identity**. **B** as HW (dense kernels; language; no absorb). Strong **negative** if a letter claims “nobody writes binary spikes beside a continuous residual.” They did, in LM.

---

### 30. 2606.03257 — PSViT: Structurally Pruning Spiking Vision Transformers  
### 30b. 2606.03428 — PrimeSVT: Automated memory-aware structured pruning

**Venue:** IEEE J-LAT family (same authors / SViT pruning).

**Mechanism.** PSViT: uniform then sensitivity-guided **channel-wise** filter prune (structured, single-shot). PrimeSVT: sort layers by size, prune largest-first (prioritized policy), automated vs unstructured STSFP/STATA token prune. ImageNet-1K SViT memory cut ~22% with <3% acc drop (PSViT).

**Class.** **X** structured sparsity on SViT weights — legal **A-control** for “our W is sparse after absorb.” **B** as last-use of residual / OF.

---

### 31. 2606.05362 — MOSAIC: DSE framework for heterogeneous NPUs

**Venue:** (cs.AR) simulator.

**Mechanism.** Searches Big/Little + **Special-Function tiles (FFT / SNN-integrate / polynomial)** vs homogeneous MAC NPU. Analytical 7 nm models; 20-workload suite.

**Class.** **X** HPU DSE. **B** as a mechanism. Mentions SNN as a *tile type*, not AT-LIF.

---

### 32. 2606.09213 — SNN-MLIR: MLIR dialect compiling NIR → bare-metal C

**Venue:** (cs.PL).

**Mechanism.** NIR graph → type-polymorphic CUBA-LIF dialect (f32 and i8/i32) → linalg/arith → dependency-free C11. Feedforward FC only. Compiler IR, not a dataflow invention.

**Class.** **A** as *compiler IR for LIF + quantized synapses* (round-4 compiler bucket). **B** as fusion/last-use HW.

---

### 33. 2606.20675 — VQ4SNN: Vector Quantization for Memory-Efficient FPGA SNNs

**Venue:** (cs.NE); spatial-dataflow FPGA.

**Mechanism.** Shared codebook + pointers replace dense W; interleaved layer execution to avoid multi-port codebook. 52–61% BRAM cut vs uncompressed spatial FPGA SNNs, logic not increased, acc held. LIF spatial pipeline template.

**Class.** **A** for on-chip W compression (codebook), orthogonal to absorb. **X** no transformer/OF.

---

### 34. 2606.26701 — SegFold: Fine-Grained Dynamic Dataflow for SpGEMM

**Venue:** (cs.AR); Segment dataflow + SegFold accelerator.

**Mechanism.** Static inner/outer/Gustavson each lose reuse or balance. **Segment**: dynamic scheduling of a local window of the stationary array + dynamic remap of partial work across PEs (merge network). Codesigned memory controller. Geo-mean 1.95× vs SOTA SpGEMM, 5.3× vs best static dataflow. Mentions sparse attention as a workload.

**Class.** **A** for *dynamic Gustavson-family* SpGEMM (after absorb, spike×sparse-W is SpGEMM). Must sit next to GustavSNN / ELSA mini-batch Gustavson / Ventaglio. **B** as SNN-specific residual.

---

### 35. 2606.30039 — Mega: 22 nm Convolutional SNN Accelerator, 0.375 pJ/SOP

**Venue:** (cs.AR); GF 22 nm FDSOI **fabricated**.

**Mechanism.** Nine clusters × 32 CUs for **parallel 3×3** spike conv. Spike streamer converts **dense spike maps → addresses** (AER only when sparse enough). **Unified memory** for spikes, membranes, weights (early layers state-heavy, late weight-heavy). Threshold unit after all spikes of a tick. 4× SOTA pJ/SOP *their* number.

**Class.** **A** for fabricated conv-SNN: address-from-bitmap + unified state/W memory + 3×3 parallel acc. Same spike-address idea as 2501.07825, without transformer SDSA. **B** as transformer/OF/residual dual-path.

---

### 36. 2607.05445 — BitFair: 12 nm Bit-Serial CNN with Learnable Early Termination (XR / DVS)

**Venue:** IEEE JETCAS (accepted); GF 12 nm; IBM DVS128 Gesture, N-MNIST.

**Mechanism.** Bit-serial CNN (not SNN). Learnable per-layer threshold \(\theta_l\): if bit-serial partial sum predicts **ReLU will be 0**, **early-terminate** remaining bits. Search layer-wise **bit order** to make that prediction earlier. 0.34 mm², 0.8–13.7 mW, sub-ms. Positions vs SNN XR chips: they stay ReLU CNN on event or frame inputs.

**Class.** **A** for *bit-serial early-exit when output is predictably zero* (product-sparsity / skip family). After absorb, spikes already 1-bit — the interesting leftover is **weight** bit-serial vs binary spike, not activation bits. **X** DVS gesture silicon **control** (always-on XR). **B** as LIF.

---

### 37. 2607.22790 — The Sparsity Tax: Weight Sparsity in Event-Driven SIMD vs SIMT Neuromorphic Cores

**Venue:** (cs.AR); GF22FDX+ RTL-to-gates.

**Mechanism.** One core, three datapaths: lockstep SIMD; **bitmap-gated Sparse-SIMD** (disable lanes, weights still dense); **SIMT** with per-PE AGU + run-length sparse W. Event-driven: one input spike updates many postsynaptic P. **Sparsity tax** = control + metadata + memory energy that eats weight-sparsity gains. SIMD throughput ~flat vs prune; Sparse-SIMD saves little (bitmap + dense storage); SIMT scales energy/speed at high sparsity but **sublinear** (metadata, imbalance, dense phases). SRAM dominates area.

**Class.** **A** for the *cost model of skipping zeros* on binary-event × sparse-W (must be copied before claiming skip energy). Directly relevant after absorb. **B** as transformer overlay.

---

### 38. 2607.25504 — Ventaglio: Gustavson sparse tensor contractions on RVV for Transformer inference

**Venue:** (cs.AR); 12 nm FinFET vector cluster; LLaMA-3-8B DuoGPT 40–60% **dual** sparsity.

**Mechanism.** Gustavson: nonzero **activations** trigger indexed accumulate of sparse **weights** into VRF. Adds RVV ops: fused indexed MAC + post-increment so kernels hit roofline without L1 gather/scatter. 6.9–7.4× vs optimized RVV; 3.1% cluster area. Prefill and decode speedups vs dense. Formats: N:M and bitmap. Explicit dual sparsity (act + W).

**Class.** **A** for Gustavson **on transformers** with activation sparsity — after absorb, spikes **are** that activation sparsity. Strongest unread Gustavson×Transformer prior. Residual/LN not the sold object. Not SNN, not OF.

---

### 39. 2607.26648 — The Sparsity Ceiling: Where SNNs Can and Cannot Trade Activity for Energy

**Venue:** (cs.NE) analysis.

**Mechanism.** Matched architecture, swap continuous vs LIF, two-sided target-rate probe. FF perception: firing → 5% free. Recurrent LM: firing pinned ~50% (state must stay live). **Spiking Transformer sparsifies to 2%** with no extra quality cost — ceiling is **recurrent compression**, not “sequence.” Attention pays KV memory instead. Firing-floor bound \(\rho \ge H_b^{-1}(\log_2 M / H)\). Dense (frame-replayed) input caps op reduction → **native event input** is where neuromorphic wins.

**Class.** **A** as *task-structure prior*: event-OF / perception is the regime where binary sparsity is free; transformer attention is also sparsifiable; recurrence is not. Supports keeping residual **continuous** (information) vs spike **gate**. Not HW.

---

### 40. 2608.00595 — Time-Multiplexed SNN Accelerator with Pipelined Readout (Artix-7 MNIST)

**Venue:** FPGA MNIST 784-64-10.

**Mechanism.** 1-bit time-mux spike broadcast + local W + integer LIF + multi-cycle pipelined argmax to kill the combinational readout critical path. Fmax 13.3 → 167 MHz; 82 µs/image; ~0.336 W.

**Class.** **X** toy FPGA. **B** as letter prior (ESTU-class already occupies small FPGA classification).

---

### 41. 2608.12500 — Lonic: Fully Local Online SNN Training, INT4, ICCAD’26

**Venue:** ICCAD 2026.

**Mechanism.** INT4 local online learning (not BPTT). Reconfigurable INT4/8/16 and binary-ternary PEs; dual zero-gating; temporal **prefix** local-learning dataflow; INT4 weight movement. Energy vs M4/V100/TPU-like/H2Learn (their sim).

**Class.** **X** train-HW (round-4 train bucket). **B** as inference absorb/OF.

---

### 42. 2608.19046 — APEX: Dual-Sparsity Accelerator for Precise SNN Inference (PASC-IF on LoAS)

**Venue:** (cs.AR).

**Mechanism.** PASC-IF: three-stage precise ANN–SNN conversion neuron (inhibitory spikes + soft-reset) **mathematically equivalent** to source ANN at few \(T\). Incompatible with vanilla IF HW. **Key HW fact:** LoAS **FTP inner-product** already reduces the full postsynaptic tensor **before** the neuron, so PASC’s three stages unroll as **combinational, 1 cycle each**, no extra latency. Dual sparsity spike+W via CSF; mixed INT4/INT8 TPPEs. +3% acc vs IF; 1.3–5.4% power, 2.1–2.7% area vs LoAS; 40% energy at best-acc configs.

**Class.** **A** for *neuron after dual-sparse GeMM* and for LoAS FTP as the thing that makes a fancy neuron cheap. After absorb, AT-LIF is a **threshold on an already-complete current** — same schedule. **B** as residual dual-consumer. Soft-reset / inhibitory path is conversion algebra, not PED.

---

### 43. 2608.19238 — Spiking Local Interaction + Adaptive Complementary Fusion for Spiking Transformers

**Venue:** (cs.NE); ImageNet / ADE20K / DVS.

**Mechanism.** SSA is sparse/discrete (weak QK co-activation dies). **SLI**: attention-*independent* depthwise–pointwise mix among neighboring spike tokens. **ACF**: layer- and channel-wise learned mix \(A_l(\cdot)\) vs \(S_l(\cdot)\), then **standard residual spiking MLP**. Does **not** change SSA formula. QKFormer+SLI+ACF: 84.37% IN-1K, 37.5% ADE20K mIoU (no IN pretrain).

**Class.** **A** as *two spike pathways fused by learned gates, then residual MLP* — algorithmic dual path, both sides still spikes. **B** vs locked identity if someone calls SLI the continuous PED: SLI is local spike conv, not I24. Residual is vanilla spiking skip (FireFly-T / ESTU class).

---

### 44. 2608.21223 — Event-triggered Implicit Perturbation ZO fine-tune of Spiking Transformers (IPZO)

**Venue:** (cs.AR); **IMC** (CIM stop as title).

**Mechanism.** ZO fine-tune without explicit weight RMW: event-triggered PGU adds perturbation sums to IMC weighted sums; PGU-XOR recombines RNGs; only **spike-active rows** get perturbations. Spikingformer/CIFAR and SpikeGPT.

**Class.** **CIM stop as title.** **X** on-chip ZO train. Spike-gated perturbation is not absorb.

---

### 45. 2609.08446 — FlexSpIM: Event-Based Digital CIM, Flexible Resolution, Layer-Wise Hybrid Stationarity

**Venue:** (cs.AR); 40 nm **digital CIM** (CIM stop as title); IBM DVS Gesture 95.8%.

**Mechanism.** Unified SRAM for **weights and membranes** with arbitrary bit-width/shape (no wasted bits from fixed 1-bit vs 8-bit maps). **Per-layer hybrid WS/OS**: weight-stationary when W dominates, membrane-stationary when state dominates (early vs late SCNN layers — same diagnosis as Mega). Event-driven SCNN backend for DVS. Up to 45% energy / 52% latency vs fixed stationarity in their large-system model.

**Class.** **CIM stop as title.** **A-control** for *layer-wise WS vs OS (membrane) stationarity* and unified W/state memory — same object Mega sells in digital 22 nm. **B** as transformer residual last-use.

---

### Already mapped (do not re-open)

| arXiv | Note |
|---|---|
| **2605.28312** EventShiftFlow | Already read (event OF). |
| **2606.20414** ExSpike | Already **A** (full-event + adjacent AND compression). Residual SRAM still spikes. |

---

## One-line skips (unrelated or out of identity)

| arXiv | Skip |
|---|---|
| 2501.03874 | Scattering-media DVS+SNN imaging; no HW/transformer/OF fusion. |
| 2503.12905 | UCF-Crime-DVS dataset + MSF VAD; no accelerator. |
| 2504.13457 | Learned RGC event *sensor* for interpolation/OF; computational photography, no SNN HW. |
| 2506.12524 | Event gaze post-process / micro-expression (medical). |
| 2510.14172 | Quantum Hamiltonian SpMSpM systolic (DIAMOND). |
| 2510.24231 | Microsaccade *dataset* + Spiking-VGG; eye/medical. |
| 2511.21910 | *Kept* (Platinum). |
| 2604.21952 | DATE focus session on multimodal foundation models / LLM. |
| 2605.00319 | SRAM **CIM** SNN macro (CIM title). |
| 2605.07653 | Aquatic neuromorphic OF (underwater sensing; OCR-broken, not SNN-Transformer HW). |
| 2605.09770 | Spiking bandpass wavelets (ECG/audio DSP). |
| 2606.13016 | Otters++ — same TTFS-optical family as 2509.18968 (see §14). |
| 2606.13354 | SupraSNN mapping/scheduling preprint; synapse-level ILP, not a new GeMM object (optional X if compiler round needs it). |
| 2606.22296 | SCENIC IoT command LLM. |
| 2607.11986 | SpikeDS 3D MRI PNI (medical). |
| 2607.12505 | N:M sparse **ViT GPU** kernels (MD-SpMM); GPU not our FPGA/ASIC letter. Optional X for N:M. |
| 2607.19248 | CNN FPGA column-wise sparsity (no SNN). |
| 2607.19421 | Photonic ViT on-chip fine-tune. |
| 2607.19623 | ECC / bit-position UEP for DNN memory. |
| 2607.24027 | Sol-Attn video-gen sparse attention (NVIDIA). |
| 2607.24396 | SpiNNaker2 many-core *platform paper* (system, not a reusable GeMM/residual mechanism for a 5-page letter). |
| 2608.26595 | LLC SST power electronics. |
| 2609.04949 | Hopf-bifurcation NDR spike *detector* (electrophysiology). |
| 2609.07544 | Flying humanoid MPC (robotics). |

---

## What this sweep changes (evidence, not a proposal)

1. **Fusion silicon to copy as A:** ISSCC 2025 ConvFormer LFS/HAPU (`2512.17555`) is the unread layer-fusion chip. It is ANN, 64K tokens, Cityscapes — **control**, not a same-net prior.
2. **Last-use / pipeline:** ELSA token-wise forward (`2605.20802`) + GOAP baked schedules (`2601.02613`) + SCDQ density-scaled queues (`2501.13610`) are the new last-use objects. None is typed last-use of a **continuous residual beside a binary gate**.
3. **Gustavson after absorb is crowded:** ELSA mini-batch Gustavson, Ventaglio RVV Gustavson on Transformers (`2607.25504`), SegFold dynamic Segment (`2606.26701`), LoAS/APEX FTP (`2608.19046`). Binary spike × W **is** this literature.
4. **Spike-driven transformer FPGA** (`2501.07825`) is ESTU/FireFly-T class: encode spike addresses, SDSA mask-add, **ResBuffer is a separate adder**. Dual-spike attention ≠ dual-consumer residual.
5. **Identity dual-path is not unfound:** SymbolicLight (`2605.21333`) writes binary LIF gates + continuous residual in LM (dense kernels). SLI+ACF (`2608.19238`) fuses two **spike** paths then residual MLP. CATFormer (`2603.15184`) **forbids** absorbing \(\theta\).
6. **Event OF in this unread slice is almost all X:** ST-FlowNet, EDCFlow, STSSM, CSSL, EventFBP, IMO suppression, HOMI/3DS-ISC frontends, TDE egomotion. No unread paper is SNN-Transformer **silicon** on DSEC.
7. **CIM / optical / photonic / in-sensor:** ASTER, FlexSpIM, IPZO, OTTERS, 3DS-ISC — **stop as titles**; steal only named non-CIM objects (membrane stationarity, spike-row gating).
8. **BN hardware:** **not located** in this unread P0 slice (no BN-fold / BN-skip accelerator in the deep-read set).

Paper numbers stay on **their** nets and boards. They are not this-net DSEC AEE or same-port PPA.
