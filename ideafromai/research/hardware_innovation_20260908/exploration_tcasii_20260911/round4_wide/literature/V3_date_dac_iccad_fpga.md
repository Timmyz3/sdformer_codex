# V3 — DATE / DAC / ICCAD / FPGA / FCCM / FPL / ISCAS 2023–2026 (wide venue sheet)

Date: 2026-09-11. Round-4 wide mining. Not a novelty claim. Not a PPA invention. Not a CIM title.

**Identity freeze** (`IDENTITY_ATLIF.md`): AT-LIF \(o=\theta\cdot H(m-\theta)\in\{0,\theta\}\). Inference: layer-shared \(\theta\) **absorbed** into next \(W\). Transmitted spikes are \(\{0,1\}\). Spike path after absorb is **binary GeMM**. Residual / PED / I24 is a **separate continuous tensor**. Dual last-use, if it exists, is **binary gate after absorb** and **continuous residual**, not “continuous AT-LIF amplitude shared by two MACs,” and not two core types assigned to two **layers**.

**G1 (typed last-use).** Retire one producer only when **both** post-absorb binary GeMM **and** continuous PED have completed. That is not “a dense core for layer 1 and sparse cores for the rest,” not FireFly-T dual-**engine**, not ESTU spike-mem vs int-mem mux.

**ESTU same-journal skip.** TCAS-II 72(12) already published a 5-page FPGA spikeformer letter: binary SSA + group-of-4 skip + overlay + classification mW. Any letter still readable as that object is desk-reject class. Copy ESTU as **A** on the post-absorb spike path, then **stop**.

**Class legend.** **A** = complete prior to copy on the named object, then stop as X. **B** = strong negative (identity-illegal, or the paper occupies the slogan under a different algebra). **stop** = illegal as title (CIM; 3D PPA; ESTU reskin; first-OF-HW). **X** is not minted from a bounded miss. Paper numbers belong to **their** nets and boards.

Venue window: DATE / DAC / ICCAD / FPGA / FCCM / FPL / ISCAS **2023–2026**. Adjacent same-family papers that reviewers will stack (IEEE TC, TRETS, TCAS-AI, TCAS-II) are recorded when they are the printed venue of a must-cover object.

---

## 0. Source table (what was actually opened)

| # | Paper | Venue / year | Primary opened this pass | Depth |
|---|---|---|---|---|
| 1 | Aliyev / Lopez / Adegbija hybrid | DATE 2025 | arXiv HTML **full** [2411.15409v1](https://arxiv.org/html/2411.15409); journal-ref DATE 2025 | Methods + results |
| 2 | RISCSparse | ICCAD 2024 | Local author full text `literature/RISCSparse_ICCAD2024_author.txt` (DOI 10.1145/3676536.3676774) | §1–4, Figs. 1–2 |
| 3 | SPARTA ReRAM-CIM token skip | ICCAD 2025 | IEEE abs + HUST faculty/news restatement. DOI 10.1109/ICCAD66269.2025.11240724. **No arXiv. PDF unread.** | Abstract + author restatement |
| 4 | Xu et al. 3D MoE/MHA + D³ head prune | ICCAD 2025 | Author PDF [gtcad Xu-ICCAD25](https://gtcad.gatech.edu/www/papers/Xu-ICCAD25.pdf) **full**; arXiv:2412.05540 is the **2024 preprint without D³**. DOI 10.1109/ICCAD66269.2025.11240876 | Full PDF (9 pp.) |
| 5 | Xu et al. Spiking Transformer 3D Integration | ICCAD 2024 | arXiv HTML **full** [2411.07397](https://arxiv.org/html/2411.07397). DOI 10.1145/3676536.3676826 | Full HTML |
| 6 | Spike-IAND-Former | ISCAS 2025 | Local `p0_txts/2503.19643.txt`; DOI 10.1109/ISCAS56072.2025.11043330 | Full local (already R3L1) |
| 7 | FireFly-T | IEEE TC 75, Jun 2026 | Local `p0_txts/2505.12771.txt`; DOI 10.1109/TC.2026.3672901 | Full local (already R3L1) |
| 8 | ESTU | TCAS-II 72(12) 2025 | AAM in `ADV_ESTU_same_journal.md`; DOI 10.1109/TCSII.2025.3626209 | AAM (already ADV / R3L3) |
| 9 | Sparse Spike-Driven Transformer FPGA | arXiv 2501.07825 (FPGA Virtex US+) | Local `p0_txts/2501.07825.txt` | Full local |
| 10 | LoopTree | IEEE TCAS-AI 2024 | Local `literature/LoopTree_TCASAI2024_author.txt` | Full author (already L4) |
| 11 | da4ml | ACM TRETS 19(1) Mar 2026 | arXiv HTML **full** [2507.04535v2](https://arxiv.org/html/2507.04535); DOI 10.1145/3777387 | Methods + integration |
| 12 | DeepFire2 | IEEE TC 72(10) 2023 | arXiv HTML **full** [2305.05187](https://arxiv.org/html/2305.05187); DOI 10.1109/TC.2023.3272284 | Full HTML |
| 13 | SyncNN | FPL 2021 / TRETS 15(4) 2022 | IEEE/ACM abs + SFU author PDF snippets. DOI 10.1145/3514253 / 10.1109/FPL53798.2021.00058 | Abstract + author PDF (borderline year; must-cover) |
| 14 | FlexSpIM | ISCAS 2025 | Local `literature/FlexSpIM_ISCAS2025_author.txt`; arXiv:2410.23082; DOI 10.1109/ISCAS56072.2025.11043507 | Full local (already R3L3) |
| 15 | ERAFT FPGA | ISCAS 2025 | IEEE abs only. DOI 10.1109/ISCAS56072.2025.11043529 | Abstract; PDF unresolved |
| 16 | COBRA | ICCAD 2025 | arXiv HTML **full** [2504.16269v2](https://arxiv.org/html/2504.16269); DOI 10.1109/ICCAD66269.2025.11240633 | Full HTML |
| 17 | Diff-DiT | ICCAD 2025 | IEEE abs + GitHub `Glinttsd/Diff-DiT`. DOI 10.1109/ICCAD66269.2025.11240791. **No arXiv.** | Abstract + repo README; PDF unread |

`p0_txts` does **not** contain 2411.15409, 2411.07397, 2412.05540, 2504.16269, 2507.04535, 2305.05187. Those were opened from arXiv HTML / author PDF this pass. SPARTA and Diff-DiT and ERAFT remain **no-fulltext**. Do not invent PE geometry, SRAM sizes, or PPA that are not in an opened primary.

---

## 1. Identity used for mapping (do not rewrite)

\[
o_i[t]=\theta_i\cdot H(m_i[t]-\theta_i)=\theta_i s_i[t],\quad s\in\{0,1\},\quad o\in\{0,\theta\}.
\]

At inference, \(W\leftarrow\theta W\). After that fold:

1. **Spike path** = binary \(\{0,1\}\times W\) GeMM (select-add / AC / AND-PopCount on SSA). Prosperity product-sparsity, Gustav NRV/CPTB, FireFly-S Bitmap AND, ESTU group skip, FireFly-T sparse engine are **legal A** on this path.
2. **Residual / PED / I24** = a **different continuous tensor**. Dual last-use is `{binary gate after absorb, continuous residual}`, not two layer types, not two attention engines, not muxed spike-mem vs int-mem.

**Forbidden as titles:** analog/digital CIM; 3D F2F PPA; OpenROAD-as-foundry; first-OF-HW; Prosperity-on-our-net; continuous-\(\theta g\) dual-MAC; deleting 35 RNE; GrokBot three knives.

---

## 2. Must-cover papers (mechanism, then class)

### 2.1 DATE 2025 hybrid — Aliyev / Lopez / Adegbija, arXiv:2411.15409

**Printed title:** *Exploring the Sparsity-Quantization Interplay on a Novel Hybrid SNN Event-Driven Architecture.* DATE 2025. Code: `github.com/githubofaliyev/SNN-DSE/tree/DATE25`.

**What it actually is.** First hybrid **inference** architecture for **direct-coded** SNNs. Direct coding feeds **raw floating-point input** into the first convolution; that layer emits floating-point membranes; a LIF then emits **binary** spikes that drive the rest (eqs. 1–2). Two **core types**, assigned by **layer**, not by consumer of one tensor:

| Core | Assigned to | Datapath (arXiv HTML §IV) |
|---|---|---|
| **Dense core (DC)** | Input layer: “largest feature map dimensions, **non-binary, and non-sparse** activations” | Weight-stationary systolic MAC, 27 PEs for 3 ch × 3×3. Activ: bias + leak \(\beta\) + threshold \(\theta\); spike=1 and subtract \(\theta\), else 0. Spike trains to BRAM, timestep-major. |
| **Sparse cores (SC)** | Remaining CONV/FC: event-driven spiking convolutions | ECU compresses spike trains (`SpikeEvents` via priority encoder + bit-reset). Address gen maps each spike to the 3×3 neighborhood. Neural cores **accumulate weights into membrane BRAM**, then Activ for LIF. Output-channel unroll \(N\). Spike max-pool = **OR** over \(N\times N\). |

QAT: int4 W/bias; **neuronal parameters stay float**; membranes dequantized for LIF. Quantization **increases** spike sparsity vs fp32 by 6.1 / 10.1 / 15.2% on SVHN / CIFAR-10 / CIFAR-100 (Fig. 1). Rate-coded ablation **turns the dense core off** (V-D).

**Their numbers (theirs).** Xilinx Virtex UltraScale+ XCVU13P, 100 MHz. Direct vs rate on CIFAR-10 LW: 2 vs 25 timesteps, 87.01% vs 77.37%, 7.6 vs 201 mJ/image → **26.4×** energy (Table II). vs Gerlinghoff DATE 2022 on CIFAR-100: **51×** throughput, ~½ power (Table III). int4 vs fp32 energy “3.4×” is **their** fp32/int4 pair (Fig. 4).

**G1?** **No.** Dense core consumes **pixels**. Sparse cores consume **binary spike trains of later layers**. BRAM between cores is a **layer FIFO**, not a fork of one compiled T10 / post-absorb word into (i) binary GeMM and (ii) continuous PED/I24. LIF (eq. 2) **throws the membrane away as a binary spike**. No residual/PED consumer. No optical flow. No absorb-then-Prosperity.

**ESTU?** Different venue. Overlay of dense+sparse cores is **not** ESTU’s microcode spikeformer, but a letter that only says “we also have a dense core and a sparse core” is a DATE 2025 reskin **and** still reads as ESTU-class skip+overlay if the headline is FPGA spikeformer + mW + class-%.

| Mechanism | Class |
|---|---|
| Hybrid overlay control: dense MAC **input** / sparse event conv **rest**; layer-wise spike-count workload model (eq. 3) | **A** then **stop as X** |
| QAT-induced extra spike sparsity | **A** (empirical, their nets) |
| Same-producer dual last-use (G1) | **B** (does not exist here) |
| “We have dense+sparse cores” as the letter | **stop** |

### 2.2 ICCAD 2024 RISCSparse — Lin / Zhu / Xie / Chen / Zhuo / Sun / Yu

**Printed title:** *RISCSparse: Point Cloud Inference Engine on RISC-V Processor.* ICCAD 2024. Local author text.

**What it actually is.** Submanifold sparse convolution (SSC) on a Chipyard RISC-V SoC: BOOM + RVV + Gemmini systolic array. Three bottlenecks they name: Rule Map Construction (Mapping), Gather-MatMul-Scatter (GMS), uncombined operators. Vectorized hash rulebook; Gemmini GEMM; SA + vector units to cut intermediate footprint; **BN fused to \(Y=aX+B\)** with ReLU/Add.

**Not SNN. Not transformer. Not event OF.** Point-cloud MinkUNet / SparseResNet segmentation and detection. Submanifold constraint: output sites do not expand. FireSim/Verilator SoC eval vs TorchSparse: average 11.73× / 13.1× vs Edge-CPU (seg/det); 1.63× / 1.07× vs Edge-GPU. Not chip silicon.

**G1?** **No.** One sparse-conv pipeline. Affine BN fusion is **frozen running-stat** \(Y=aX+B\), the **opposite** of this net’s live full-domain projection BN (`10×96×120×160` actual batch stats). Copy the **mapping-tax / gather-scatter-tax / fused affine** as A, then stop.

**ESTU?** No collision. Different object (point-cloud SSC vs spikeformer overlay).

| Mechanism | Class |
|---|---|
| Rule Map → Gather → GEMM → Scatter; vectorized hash; GMS offload | **A** as sparse-frontend tax (not our X) |
| BN folded to frozen \(Y=aX+B\) | **A** as fused affine; **B** if retitled live proj BN |
| Dual last-use binary GeMM + PED | **B** (not present) |

### 2.3 ICCAD 2025 SPARTA — Jiang / Wang / Fang / Wang / Zhu / Miao / Wang (HUST)

**Printed title:** *SPARTA: Spike-Aware Token Skipping Co-Optimization with Heterogeneous ReRAM-CIM Architecture for Spiking Transformer Acceleration.* ICCAD 2025, Munich, pp. 1–9. DOI 10.1109/ICCAD66269.2025.11240724. **No arXiv. PDF unread this pass.**

**Opened objects (faculty page + HUST news, matching IEEE landing):**

- Algorithm: RL-based dynamic token skipping (RL-TS) + spike-aware token prediction (STP). Structured token-level sparsity in space **and** time. PPO adapts per-layer skip ratio. STP sets a prediction threshold from weights to drop inactive tokens early.
- Hardware: **heterogeneous ReRAM-CIM**: analog ReRAM CIM engine for **linear** layers; **token-spike fusion digital engine** for token routing + spike attention. Two engines because operators differ (matmul vs non-matmul attention), not because one producer has two last-uses.
- Abstract metrics (faculty page): up to **543.1× / 10.2×** speedup and **308.0× / 5.2×** energy vs GPU and COMPASS; accuracy loss ≤1.1%. **Do not reuse as our PPA.** Process node, array geometry, and skip-mispredict cost are **not in the opened abstract**.

**Illegal as title: analog ReRAM CIM.** Copy as A only the named **token-skip** object (lossy, structured, RL), then **stop as X**.

**G1?** **No** (from abstract). Token skip drops **tokens**, not a dual-ready fork of binary GeMM + PED. Analog CIM linear engine is out of identity.

**ESTU?** Token skip is a **different skip unit** (tokens, not groups-of-4 spikes), but a TCAS-II letter whose skip story is “we also skip inactive tokens in a spikeformer” still maps to ESTU’s skip + class-%. CIM makes it worse, not better.

| Mechanism | Class |
|---|---|
| Structured token skip (RL-TS + STP) on a spiking transformer | **A** (lossy skip family) then **stop** |
| Heterogeneous ReRAM analog CIM + digital token/attention engine | **stop** (CIM title) |
| Dual last-use G1 | **B** / unknown-not-claimed |
| Extra 543× / 308× | **stop** (their abstract, PDF unread) |

Do not confuse with arXiv:2508.01646 “SPARTA” (Jang/Kim, spike-timing sparse **attention algorithm**, not ICCAD CIM).

### 2.4 ICCAD 2024 — Xu / Hwang / Vanna-iampikul / Lim / Li, *Spiking Transformer Hardware Accelerators in 3D Integration*

arXiv:2411.07397. DOI 10.1145/3676536.3676826. ICCAD ’24 session “Innovations in Neuromorphic Hardware and 3D Integration.”

**What it actually is.** First dedicated **3D F2F** accelerator for **binary** spiking transformers (Spikformer-class). Two kernels:

- **Spiking MLP:** bottom-tier spatiotemporal systolic array (1-bit spikes on columns, multi-bit W on rows, synaptic-integration-stationary PE); top-tier spiking generators (membrane + LIF). Kernel fusion of synaptic integration / Vmem / spike gen (Alg. 1). Weight reuse across tokens **and** timesteps.
- **Spiking self-attention:** reconfigurable array, Mode 1 \(A=QK^\top\) (AND + accumulate, attention-stationary), Mode 2 \(X=AV\). Kernel fusion so the multi-bit attention map **never leaves the array**. Then top-tier LIF.

Physical: 28 nm PDK, memory-on-logic and logic-on-logic, pin-3D flow. vs 2D: ~50% area, +6–7% Fmax, memory-access latency −68% (MLP) / −74% (SSA). CIFAR10-DVS / DVS-Gesture **classification**. 1-bit spikes, 4/8-bit W, 12/16-bit synaptic integration.

**G1?** **No.** One binary spike tensor, one accumulate consumer, LIF binarizes. Attention-stationary register is **A**, not PED. 3D stacking is a **packaging** prior, not a last-use contract.

**ESTU?** Different venue. The **object** (binary SSA + LIF + classification) is the same family ESTU already put in TCAS-II. 3D PPA does not escape the ESTU script if the letter is still “spikeformer hardware with skip.”

| Mechanism | Class |
|---|---|
| Binary SSA AND-accumulate + kernel-fused \(QK^\top\) then \(AV\) without writing A off-array | **A** on post-absorb SSA (FireFly-T / ESTU already occupy FPGA form) |
| 3D F2F memory-on-logic / logic-on-logic PPA vs 2D | **stop** as title (packaging, not this net) |
| Dual last-use G1 | **B** |

### 2.5 ICCAD 2025 — Xu / Hwang / Vanna-iampikul / Yin / Lim / Li, *3D Acceleration for Mixture-of-Experts and Multi-Head Attention Spiking Transformers with Dynamic Head Pruning*

DOI 10.1109/ICCAD66269.2025.11240876. McCalla Best Paper. Author PDF opened in full. arXiv:2412.05540 is the **preprint without D³ pruning** — do not cite the arXiv as the ICCAD 2025 mechanism.

**Adds to the 2024 3D paper:**

1. **Spiking MoE kernel.** Five steps: token conditional routing (SCR) → synaptic integration (SSI) → membrane accumulate (SMA) → conditional spike gen (SCG) → aligned merge (SAM). Top-K experts. Four modularized SE cores, two-tier router, shared weight GLBs. 1-bit spikes × expert-specific multi-bit \(W^{(e)}\).
2. **Distributed MHA.** Heads dispatched to separate 3D attention expert cores; concat. Reconfigurable PE: Mode \(A=QK\), Mode \(X=AV\).
3. **D³ (dynamic detect-and-drop) head pruning (Alg. 3).** Importance \(s_i=\sum_{t,n,j} Q_{t,n,i,j}\) on **binary Q** only (grouped bit-count). Keep top \(\lceil p H\rceil\) heads; pruned heads skip K/V/attention except the Q-score. Must be **trained in**; pruning a pretrained model costs ~2.7% CIFAR-10. With in-train D³, they report **no accuracy drop** (average +0.24%); pruning 75% of heads → 25% attention work, ~0.19% acc loss. At \(p=0.5\), 8-head → 4-head parallel on 4 cores, half the tiling.

End-to-end vs 2D (their Figs. 8–9): 3D kernels −16.82% energy; 3D+D³ **−54.89% energy, −48.64% latency, EDP 0.23×**. 28 nm F2F, CIFAR-10/100 classification. 8-bit W, 16-bit synaptic integration.

**G1?** **No.** Head prune is **lossy structured skip of binary attention heads**, using Q popcount as a proxy. Dropped heads are **not** a continuous PED consumer. MoE routing is **token → expert**, mutually exclusive per token (top-K), not simultaneous gate+PED on one producer. Membrane stays inside each expert’s LIF.

**ESTU?** Head prune is another skip encoding. Reviewer who just handled ESTU hears “spikeformer + skip.” 3D + MoE does not change the journal object if the letter still measures class-% / EDP.

| Mechanism | Class |
|---|---|
| 3D modularized spiking MoE + MHA kernels | **A** as binary-expert parallelism; **stop** as 3D-PPA title |
| D³ head prune: Q-popcount importance, keep-\(p\), in-train | **A** as lossy structured skip; **B** as G1; **stop** as ESTU-class skip title |
| Dual last-use G1 | **B** |

### 2.6 ISCAS 2025 Spike-IAND — Chen & Chang, arXiv:2503.19643

Already mapped in `round3_gap/literature/R3L1`. Short remap under locked absorb identity.

- Residual **ADD replaced by IAND** \(x\odot(1-\mathrm{ConvBN}(x))\) so the **whole model is spike I/O**.
- Fully parallel tick-batching; unrolled LIF mux `111/101/000` for \(T=4/2/1\); membrane SRAM deleted.
- TSMC 28 nm, 3456 GSOPS, 38.334 TSOPS/W (their Table II). ImageNet / CIFAR-10.

**IAND is B / identity-illegal on PED/I24** (deletes the continuous consumer). Mux unroll is **A** for **causal** T-parallel LIF, **B** as T10 PSN.

**G1?** Anti-prior: they kill the residual to stay binary.

**ESTU?** Same object family (binary spikeformer, skip-by-construction via IAND, class-%, TSOPS/W). Different venue, but a TCAS-II letter that “also maps a spikeformer and removes residual add” collides with ESTU **and** IAND.

### 2.7 FireFly-T — IEEE Transactions on Computers, vol. 75, Jun 2026, pp. 2185–2199

arXiv:2505.12771. DOI 10.1109/TC.2026.3672901. Full local text. Already R3L1; printed venue is now **IEEE TC 2026** (not DATE/DAC/ICCAD/FPGA). Recorded because the owner listed it and because dual-**engine** is the naming trap vs G1.

- Sparse engine: bitmap spike decode → multi-bit \(W\) → LIF membrane (activation sparsity).
- Binary engine: 2D systolic **AND-PopCount** SSA.
- Orchestrator + attention-enable overlay; latency-hide \(QK^\top\) behind V.
- Residual I/O: **fourth AXI port**, **pre-neuron membrane** (SDT MS), not PED MAC.
- Overlay **serializes** engines per layer.

**G1?** **B.** Dual-engine = two **operators** (sparse conv/linear vs binary attention) on **binarized** QKV, plus MS add into the **same** LIF. Not `{binary gate, continuous PED}` from one producer.

**ESTU?** Strongest **name-collision** after ESTU itself. “Dual-engine overlay FPGA spikeformer + skip + GOP/s/W + CIFAR” is ESTU’s SOTA row plus FireFly-T. Desk-reject if that is the letter.

### 2.8 ESTU — TCAS-II 72(12) 2025 (same journal)

Already ADV. Object in one line:

**binary spike tensor → one accumulate consumer → skip inactive groups-of-4 → classification % + mW on a 5k-LUT FPGA.**

Table I overlay: `Dense(spike)`, `Dense(int)`, `Sum(spike,*)`, `Mul(spike,spike)` (16 AND + 16-input popcount), `Mul(spike,int)`. Typed **destinations**, not a fork. 3.76 mW / 0.23 mW / 4.28 μJ; NinaPro 87.21%.

**G1?** **No.** Muxed destination. Slot retirement = operator-done × group sparsity, not `gate ∧ PED`.

**Same-journal rule.** Copy as **A** (binary SSA overlay + group skip) on the post-absorb spike path, then the letter **must** measure a different object (dual last-use of binary GeMM **and** continuous PED on DSEC valid825, same-port net service). DATE dense/sparse, FireFly-T dual-engine, IAND, 3D head prune, and “we also skip” **do not escape**.

### 2.9 FPGA Spike-Driven Transformer — Li / Mao / Zhang / Dong / Wang, arXiv:2501.07825

**Printed title:** *An Efficient Sparse Hardware Accelerator for Spike-Driven Transformer.* Virtex UltraScale FPGA. CIFAR-10 Spike-driven Transformer.

**Mechanism (local full text).** Encode **positions of valid spikes**; convert linear / maxpool / **SDSA** to **address comparison** so zeros are bypassed. Specialized SDSA module takes **dual spike inputs** (Q and K both binary). All spike arithmetic = add + compare, no multiply. Two cores: SPS (tile + adder) and SDEB (spike linear array). Claimed 13.24× throughput / 1.33× energy vs prior SNN accelerators (**their** numbers).

**Naming trap.** “Dual spike inputs” = two **binary** tensors into attention, **not** gate + residual. Continuous PED is not a second spike tensor.

**G1?** **No.**

**ESTU?** FPGA spikeformer + zero-skip + CIFAR = ESTU-class object on a bigger Xilinx part.

| Mechanism | Class |
|---|---|
| Position-encode + address-compare zero skip on linear / maxpool / SDSA | **A** on post-absorb spike path |
| “Dual spike inputs” as dual last-use | **B** (two binary tensors, one SSA) |
| 13.24× / CIFAR FPGA spikeformer as the letter | **stop** (ESTU + FireFly-T family) |

### 2.10 LoopTree — Wu et al., IEEE TCAS-AI 2024

Not DATE/DAC/ICCAD/FPGA; recorded because the owner listed it and because retain/recompute is the fusion-DSE A.

**What it is.** Fused-layer dataflow **analyzer** (Timeloop + ISL + Accelergy): per-tensor retain vs recompute vs reread; sequential or pipelined inter-layer schedule; explicit Buffet assumption (reordering does not increase latency; pipeline stalls ignorable). Fusion **set is exogenous** — LoopTree does not choose which layers to fuse.

**G1?** **No hardware last-use.** Can *describe* a fusion set `{source, PED}` as retain choices; that vocabulary is A for Stage B occupancy accounting, not X.

**ESTU?** No collision.

| Mechanism | Class |
|---|---|
| Retain / recompute / reread per intermediate fmap; tile-shape → action counts | **A** as DSE vocabulary |
| Analyzer as accelerator X | **B** / **stop** |
| Assumption “explicit data orchestration hides stall” | **B** vs this net’s finite same-port backpressure (8088) |

### 2.11 da4ml — Sun / Que / Loncar / Luk / Spiropulu, ACM TRETS 19(1) Art. 13, Mar 2026

arXiv:2507.04535v2. DOI 10.1145/3777387. Code: `github.com/calad0i/da4ml`. Integrated into hls4ml as `Strategy: distributed_arithmetic`.

**What it is.** Exact (not approximate) **constant matrix–vector multiply** compiler for fully unrolled FPGA NNs. Graph decomposition \(M=M_1 M_2\) (MST on columns) then cost-aware two-term CSE on CSD, emitting adder graphs \(a\pm(b\ll s)\). \(O(N^2)\) vs Hcmvm \(O(N^3)\); ~2% extra adders, \(10^5\times\) faster compile. Up to ~1/3 LUT reduction on highly quantized nets. Production use: CMS AXOL1TL trigger.

**Map after absorb.** T10 PSN mix \(H=WX\) is **pre-threshold continuous** CMVM — da4ml is the complete **A** for that compile (ordinary 260 add/sub; lifting 159 + 35 RNE). It does **not** know dual consumers, AT-LIF absorb, or same-port backpressure. Sharing the 159-op cone across **gate and PED** is not a da4ml result.

**G1?** **No.** One CMVM, one output vector.

**ESTU?** Different object (compiled dense mix vs binary SSA skip). A letter that only “compiled T10 with da4ml” is da4ml + ESTU if the headline is still FPGA spikeformer.

| Mechanism | Class |
|---|---|
| Exact CMVM adder-graph CSE + delay constraint | **A** for **pre-threshold** T10 mix |
| Share one adder cone across binary gate **and** PED | **not in the paper** (do not mint X from silence; G1 still unmeasured) |
| LUT/latency numbers from LHC triggers | **stop** (their PPA) |

### 2.12 DeepFire2 — Aung / Gerlinghoff / Qu / Yang / Huang / Goh / Luo / Wong, IEEE TC 72(10) 2023

arXiv:2305.05187. DOI 10.1109/TC.2023.3272284. Predecessor DeepFire: FPL 2021.

**What it is.** Spatial convolutional SNN FPGA IP for **multi-SLR** UltraScale+ (VU9P / VCU118). IF neurons, **T=1** spike train. Two mapping knobs: layer-wise SLR placement (binary spikes only cross SLR) and **split-kernel** (partition \(\omega\) parallel weight/neuron-core units across SLRs). Neuron core: 8 spikes × 8-bit W; **AND implemented as register reset** (inverted spike → FF.R) to save LUTs vs DF1 LUT-AND; DSP SIMD adder tree; accumulate & fire. **Transduction layer** is the first layer: 8-bit pixels × 8-bit W **MAC** (DSP multiply), then threshold → spikes. Two-stage feature buffers; backpressure when FBF(n+1) is full.

**Their numbers.** CIFAR-10: 87.1%, 23 kFPS, 550 MHz, 518 GOPS/W on 1 SLR. ImageNet: 40.1%, 1.56 kFPS, 21 TOPS, 3 SLRs — first claimed full ImageNet SNN on an FPGA. MNIST 79 kFPS @ 600 MHz.

**G1?** **No.** Transduction MAC is the **input image**, same DATE-hybrid shape (dense first layer, sparse-binary rest) but **spatial** not overlay. IF membrane is not a second GeMM consumer. T=1 kills T10.

**ESTU?** Conv-SNN spatial, not transformer. Still: FPGA SNN + skip-by-binary-AND + class-% / GOP/s/W is the FireFly/DeepFire row ESTU cites. Do not retitle.

| Mechanism | Class |
|---|---|
| Register-AND IF core; split-kernel SLR mapping; spatial pipeline | **A** on post-absorb spike conv path |
| Transduction dense 8b×8b first layer | **A** as DATE-class input-layer dense core (spatial sibling) |
| T=1 / ImageNet 40.1% / 21 TOPS | **stop** as our PPA or as T10 |
| Dual last-use G1 | **B** |

### 2.13 SyncNN — Panchapakesan / Fang / Li, FPL 2021 / ACM TRETS 15(4) 2022

DOI 10.1145/3514253. Borderline year; owner-listed because FireFly-T / DeepFire2 SOTA tables still use it.

**What it is.** **Synchronous** rate-encoded SNN: aggregate spikes over the encoding window, one forward pass per sample (eliminates per-tick sequential fire). SNN-friendly 16/8/4-bit quant. Parameterized compute engines for irregular encoded-neuron access. ZCU102 LeNet-S MNIST: 13,086 FPS, 99.3%, 200 MHz, 4-bit. NiN / VGG-13 on CIFAR-10. Power: board active 24.5 W minus static (FireFly-S footnote).

**G1?** **No.** Synchronous rate encoding is the **opposite** of event-driven dual last-use.

**ESTU?** Historical FPGA SNN baseline. Cite-and-drop.

| Mechanism | Class |
|---|---|
| Synchronous rate-encode; parameterized sparse engines | **A** as 2021–22 FPGA SNN control |
| Dual last-use / PED / spikeformer OF | **B** |

### 2.14 ISCAS 2025 FlexSpIM — Chauvaux / Kneip / Posch / Makinwa / Frenkel

Local author full text. DOI 10.1109/ISCAS56072.2025.11043507. arXiv:2410.23082.

**Illegal as title: digital CIM-SRAM.** Unified 16 kB 6T array for **weights and membrane**. Event-driven layer-first IF SCNN, DVS-Gesture 95.8%. Hybrid stationarity (HS): each **layer** chooses WS or OS (membrane-stationary). Operand shaping \(N_R\times N_C\); unused columns PC standby.

**G1?** **No.** Unified W **and** Vmem of the **same IF update**, not binary GeMM **and** a separate PED tensor. HS is per-layer stationarity, not DATE dense/sparse cores, not dual last-use.

**ESTU?** CIM silicon, different journal. Transferable skip is ordinary event-driven binary skip + (digitally reimplemented) stationarity. Then **stop**.

### 2.15 ISCAS 2025 ERAFT FPGA — DOI 10.1109/ISCAS56072.2025.11043529

**Printed title:** *An FPGA-based Real-Time Optical Flow Accelerator for Recurrent All-Pairs Field Transforms.* **IEEE/scholar abstract only; PDF unresolved.**

**Opened objects.** Lightweight **frame** RAFT-style DNN (“ERAFT” here is **this** hardware RAFT, **not** Gehrig E-RAFT 3DV 2021). VCK190. Middlebury, **86 fps at 640×480**.

**Not event, not SNN, not transformer.** CONTROL for “FPGA optical flow.” Different algorithm family from DSEC event-SNN-Transformer.

**G1?** No. **ESTU?** No. **Do not claim “first event-flow FPGA”** — ROFD ISCAS 2025 (5k-fps event flare flow, DOI 10.1109/ISCAS56072.2025.11044162) and plane-fit ISCAS 2018 already sit on that slogan. Full PE/SRAM: **unresolved**.

### 2.16 ICCAD 2025 COBRA — Qiao / Chen / Wang / Zhang / Deng / Huang, arXiv:2504.16269

DOI 10.1109/ICCAD66269.2025.11240633. Best Paper candidate. Edge FPGA **ANN binary BERT**, not SNN.

**Mechanisms (full HTML).**

1. **SPS (Shifted Polarized Softmax):** replace softmax+elastic binarize with head-wise threshold \(\lambda_{i,k}\); output \(\{0,1\}\). GLUE average 98.2% of BiT.
2. **RBMM:** real 1-bit engine for \(\{-1,1\}\times\{-1,1\}\) (XNOR+popcount) **and** \(\{0,1\}\times\{-1,1\}\) (AND+popcount+DC count). Pack to 768-bit datapacks. Quantization fused into threshold compare. One engine, six modes (M1–M4 MHA, F1–F2 FFN). II=1. 6:3 compressor popcount.
3. ZCU102 \(N_{pe}=32\): **3894.7 GOPS, 448.7 GOPS/W**; 3.5× throughput vs BAT. KV260 \(N_{pe}=16\): 846 GOPS, 208 GOPS/W. Ablation: without SPS, throughput 564× worse (softmax dominates).

**Map after absorb.** Post-absorb spike path is \(\{0,1\}\times\) **multi-bit \(W\)**, or 1-bit×1-bit SSA. COBRA’s alphabet is **ANN binary BERT** \(\{-1,1\}/\{0,1\}\) with scaling \(\alpha,\gamma\), LayerNorm integer path, **no LIF, no residual PED, no spikes**. RBMM XNOR/AND-popcount is a **complete A for binary matmul ALU**, already occupied on the spike path by ESTU `Mul(spike,spike)` and FireFly-T AND-PopCount. SPS is a **binary attention surrogate**, not SSA.

**G1?** **No.** One binary consumer. Residual in BERT is integer add then LayerNorm, then re-binarize — ESTU-class typed mem, not PED last-use.

**ESTU?** Different model family (BERT GLUE vs spikeformer). A letter that “also does 1-bit transformer on KV260 with popcount” is COBRA+ESTU.

| Mechanism | Class |
|---|---|
| XNOR/AND + popcount binary matmul; 6:3 compressor; fused threshold | **A** for **binary GeMM ALU** (then stop; ESTU/FireFly-T already) |
| SPS as SSA / AT-LIF | **B** (ANN softmax replacement) |
| Dual last-use G1 | **B** |
| 3894 GOPS / KV260 spikeformer | **stop** |

### 2.17 ICCAD 2025 Diff-DiT — Tang / Zheng / Chen / Lv / Da Silva / Ling

DOI 10.1109/ICCAD66269.2025.11240791. GitHub `Glinttsd/Diff-DiT`. **No arXiv. PDF unread.**

**IEEE abstract (opened).** FPGA accelerator for **low-bit Diffusion Transformers**. Temporal differential computation (activation similarity across **diffusion timesteps**). Challenges: applying diff to DiT **attention** costs memory. Three named objects:

1. **ADA (approximated differential attention):** significance score; selectively approximate attention across time steps; low-bit on-chip; cut memory.
2. **Cross-cast data access** + flexible reuse for matmul intensity.
3. **HCS (half-condition splitting)** dataflow + fine-grained pipeline.

vs V100: 1.39× throughput, 5.60× energy. vs SOTA diffusion accelerators: 2.81× / 2.77×. Modes 0–12 in the repo (HC/LN/Proj, QK+SoftMax, SV, and Diff-* variants) confirm **ANN DiT**, residual add (mode 3/10 `Proj3+Res`), softmax, GELU — not SNN.

**G1?** **No.** Temporal difference is **across diffusion steps of ANN activations**, cousin of Ditto/Cambricon-D / Comperity-DSE **bit residual**, not AT-LIF ticks and not PED. Residual in DiT is the standard ANN skip.

**ESTU?** No. Cite as CONTROL-drop unless someone retitles T10 as “temporal differential.”

| Mechanism | Class |
|---|---|
| Temporal differential / ADA for DiT attention | **A** only as **ANN diffusion-step residual** (not SNN T10) |
| Dual last-use G1 / spike path | **B** |
| FPGA DiT PPA | **stop** |

---

## 3. Venue enumeration 2023–2026 (located, not invented)

Search date 2026-09-11. Depth marked. “Not located in this boundary” ≠ “does not exist.”

### 3.1 DATE

| Year | Paper | Depth | Class vs this net |
|---|---|---|---|
| 2025 | Aliyev hybrid dense/sparse SNN (2411.15409) | **Full arXiv HTML** | **A** overlay control; **B** G1; **stop** as title |
| 2025 | SparseInfer (LLM activation sparsity, 2411.12692) | abs | CONTROL-drop (LLM, not SNN) |
| 2026 LBR | Dense-sparse DNN scheme | DATE PDF abs | CONTROL-drop (DNN) |
| 2024 | CRISP class-aware prune | DATE PDF | CONTROL-drop |
| 2022 (out of window, stacked) | Gerlinghoff emerging-encoding SNN (2206.02495) | known A | DATE 2022 FPGA SNN; Aliyev’s 51× baseline |
| 2022 (out) | SNE event conv (2204.10687); FlowAcc frame OF | known | event-conv / frame-OF; not dual last-use |

**DATE 2023–2026 SNN-Transformer OF accelerator: not located.** Strongest DATE object is Aliyev layer-split hybrid.

### 3.2 DAC

| Year | Paper | Depth | Class |
|---|---|---|---|
| 2025 | DJP: Dynamic Joint Pruning for SNN energy (IEEE 11132570) | abs | **A** as lossy prune; identity-changing train; **stop** |
| 2025 | GRASP LLM activation sparsity | program | CONTROL-drop (LLM) |
| 2024 | Dyn-Bitpool two-sided sparse CIM | program | **stop** CIM |
| 2024 | 3 nm SRAM-CIM SNN (2410.09130) | abs | **stop** CIM; 2024 |
| 2023 | (no SNN-Transformer OF accelerator located) | — | — |
| 2022 (out) | SATO temporal-oriented SNN; Eventor | known | CONTROL |

**DAC 2023–2026 SNN-Transformer OF accelerator: not located.** No DAC hit is G1.

### 3.3 ICCAD

| Year | Paper | Depth | Class |
|---|---|---|---|
| 2025 | SPARTA ReRAM-CIM token skip (11240724) | **abs + faculty** | **A** token skip then **stop CIM** |
| 2025 | Xu 3D MoE/MHA + D³ head prune (11240876) | **full author PDF** | **A** 3D binary MoE/MHA + lossy head skip; **B** G1; **stop** 3D-PPA title |
| 2025 | COBRA binary BERT FPGA (11240633) | **full arXiv HTML** | **A** binary matmul ALU; **B** as SNN/G1; **stop** as spikeformer |
| 2025 | Diff-DiT temporal diff DiT FPGA (11240791) | **abs + GitHub** | **A** ANN diffusion-step residual; **B** SNN T10; PDF unread |
| 2025 | DANCE dual-side N:M digital CIM LLM (11240835) | abs | CONTROL-drop CIM LLM |
| 2025 | ScanNow sparse GEMM (11240912) | abs | CONTROL-drop (not SNN) |
| 2025 | MoE-OPU FPGA overlay LLM (11240807) | abs | CONTROL-drop |
| 2024 | RISCSparse point-cloud SSC | **full author** | **A** mapping/GMS/fused affine; **B** live BN / G1 |
| 2024 | Xu 3D spiking transformer (2411.07397) | **full arXiv HTML** | **A** binary SSA kernel-fusion; **stop** 3D PPA |
| 2023 | Edge-MoE ViT (10323651) | abs | CONTROL-drop (ANN MoE) |

**ICCAD 2023–2026 SNN-Transformer OF / dual last-use: not located.** Crowded **binary spikeformer** (3D, CIM token skip, head prune) and **ANN binary transformer** (COBRA).

### 3.4 FPGA (ACM/SIGDA)

| Year | Paper | Depth | Class |
|---|---|---|---|
| 2025 | 2501.07825 Spike-driven Transformer sparse FPGA | **full local** | **A** address-compare skip; **B** “dual spike” as G1; **stop** ESTU-class |
| 2024 | ESDA event-vision sparse dataflow (2401.05626) | local txt exists | **A** event sparse dataflow (not transformer OF SNN) |
| 2023–2026 | dedicated SNN-Transformer OF FPGA at the FPGA symposium: **not located** | — | 2501.07825 is arXiv FPGA eval, not confirmed FPGA ’25 ToC this pass |

### 3.5 FCCM

**No 2023–2026 FCCM SNN-Transformer, event-OF SNN, or dual last-use paper located in this boundary.** Treat as **search-incomplete**, not as a hole that mints X. Historical FCCM SNN (Bluehive 2012, etc.) is out of window.

### 3.6 FPL

| Year | Paper | Depth | Class |
|---|---|---|---|
| 2026 | ExSpike + APEC (2606.20414) | local full | **A** full-event spikeformer + spatial event compression; **B** if residual/PED must stay continuous (they force spike I/O) |
| 2024 | NeuraLUT (2403.00849) | abs | **A** LUT NN; not SNN dual-path |
| 2021–22 (out / stacked) | SyncNN; DeepFire (FPL 2021) | abs / DF2 HTML | **A** historical FPGA SNN |
| 2022 (out) | Ultra-Flow FPGA OF | abs | frame OF, not SNN |

### 3.7 ISCAS

| Year | Paper | Depth | Class |
|---|---|---|---|
| 2025 | Spike-IAND (2503.19643) | **full local** | **B** IAND residual; **A** causal T-unroll mux |
| 2025 | FlexSpIM CIM | **full local** | **A** then **stop CIM** |
| 2025 | ERAFT FPGA OF | **abs only** | **A** frame OF FPGA; **not** event SNN |
| 2025 | CSA array for spiking transformers (11043823) | abs | **A** 1b-attn vs 8b-linear **mode mux**; **B** G1 |
| 2025 | UbiMoE MoE-ViT FPGA (2502.05602) | abs | CONTROL-drop (ANN MoE) |
| 2025 | Cha DVS/CIS NPU trigger (11043213) | abs | event trigger, not transformer OF |
| 2025 | FlexAcc BN GPU-FPGA (ISCAS PDF) | abs | **A** as BN offload; **B** as live full-domain proj BN X |
| 2025 | ROFD 5k-fps event OF (11044162) | abs | event OF **non-neural**; CONTROL-drop for neural letter |
| 2023–24 | (no additional spikeformer dual last-use located) | — | — |

**ISCAS 2025 is the densest CASS-adjacent spikeformer year:** IAND, CSA, FlexSpIM, ERAFT, UbiMoE. None is G1. IAND is the **anti-prior** for keeping PED.

---

## 4. Master table (17 must-cover + stacked venue hits)

Legend: **G1** = same-producer last-use of post-absorb binary GeMM **and** continuous PED. **ESTU-skip** = would a TCAS-II reviewer who just handled ESTU map this letter onto “FPGA spikeformer + skip + mW + class-%”?

| # | Paper | Venue | Depth | Binary after first spike? | Hybrid of what? | G1 dual last-use? | ESTU-skip collision? | Copy as A | Illegal / stop |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Aliyev hybrid | DATE 2025 | full HTML | After input LIF: yes. Input: **no** (direct-coded dense) | **Layer-split** dense MAC input / sparse event rest | **No.** Different layers, different tensors | If letter is “FPGA SNN overlay + skip + class-%”: **yes** | Overlay control; QAT extra sparsity | “We have dense+sparse cores.” Not G1 |
| 2 | RISCSparse | ICCAD 2024 | full author | n/a (ANN SSC) | Mapping + GMS + fused affine | **No** | No | Rulebook / gather-scatter tax; frozen \(Y=aX+B\) | Live proj BN; point-cloud PPA |
| 3 | SPARTA | ICCAD 2025 | abs | yes (spike tokens) | ReRAM analog CIM linear + digital token/attn | **No** (token skip) | Token skip ≈ ESTU skip family | Token skip (lossy) at abs grain | **CIM.** 543× unread-PDF |
| 4 | Xu 3D transformer | ICCAD 2024 | full HTML | yes | 3D mem-on-logic MLP + fused SSA | **No** | Object family **yes** (binary SSA + class) | Kernel-fused binary SSA; 3D as packaging A | 3D PPA title |
| 5 | Xu 3D MoE/MHA + D³ | ICCAD 2025 | full PDF | yes | 3D MoE experts + head-parallel MHA + Q-popcount prune | **No.** Head drop is lossy; MoE is top-K mux | Head prune ≈ ESTU skip | D³ Q-score head keep-\(p\); 3D expert parallel | 3D PPA; prune as G1 |
| 6 | Spike-IAND | ISCAS 2025 | full local | **whole model** spike I/O | none (one PE + unrolled LIF) | **Anti.** IAND deletes PED | **yes** (binary spikeformer + class + TSOPS/W) | Causal T-unroll mux `111/101/000` | IAND residual; unroll as T10 |
| 7 | FireFly-T | IEEE TC 2026 | full local | yes | Dual-**engine**: sparse conv + binary attn overlay | **No.** Serial overlay + MS membrane skip | **yes** (name collision dual-engine) | Overlay sparse+binary; MS port | Dual-engine as G1 |
| 8 | ESTU | TCAS-II 2025 | AAM | yes | Microcode overlay, typed spike/int mem | **No.** Muxed dest, one accumulate | **is** the collision | Binary SSA + group-4 skip + tiny-FPGA overlay | Same-journal reskin |
| 9 | 2501.07825 SDT FPGA | FPGA eval 2025 | full local | yes | SPS + SDEB; dual **spike** SDSA | **No.** Two binary tensors | **yes** | Position-encode + compare skip | “Dual spike” as G1; 13.24× |
| 10 | LoopTree | TCAS-AI 2024 | full author | n/a | Retain/recompute DSE | **No** (analyzer) | No | Fusion-set occupancy vocabulary | Analyzer as X; zero-stall assumption |
| 11 | da4ml | TRETS 2026 | full HTML | n/a (ANN CMVM) | none | **No.** One output vector | No unless letter is “FPGA compile + spikeformer” | Exact T10 adder-graph CSE | Share-across-PED as if already done |
| 12 | DeepFire2 | IEEE TC 2023 | full HTML | after transduction | Spatial SLR split-kernel; dense **input** MAC | **No.** Input pixels vs later spikes | Conv-SNN SOTA row, not transformer | Register-AND IF; split-kernel; transduction dense | T=1; ImageNet 40%; GOP/s/W |
| 13 | SyncNN | FPL’21 / TRETS’22 | abs+PDF | rate-encoded | sync window, parameterized engines | **No** | Historical FPGA SNN | Sync rate-encode baseline | 13k FPS MNIST as ours |
| 14 | FlexSpIM | ISCAS 2025 | full author | yes (IF events) | WS/OS per layer in unified W/Vmem CIM | **No.** Same IF update | No (CIM, other journal) | Event skip; stationarity of **membrane vs W** | **CIM**; 90% many-macro extrap. |
| 15 | ERAFT FPGA | ISCAS 2025 | abs | no (ANN RAFT) | none | **No** | No | Frame OF FPGA control | Event-SNN OF; “first OF FPGA” |
| 16 | COBRA | ICCAD 2025 | full HTML | ANN binary \(\{\pm1,0\}\) | one RBMM, six modes | **No** | Binary transformer FPGA, different task | XNOR/AND popcount ALU; SPS as **ANN** attn | SPS as SSA; KV260 spikeformer |
| 17 | Diff-DiT | ICCAD 2025 | abs+repo | no (DiT) | full vs differential **modes** | **No.** Diffusion-step residual | No | Temporal-diff as ANN cousin of DSE | T10 rename; PDF unread |

**15+ rows satisfied.** Additional stacked hits (DANCE, ScanNow, UbiMoE, CSA, DJP, ExSpike, ESDA, ROFD) are in §3; none flips G1 or ESTU.

---

## 5. vs G1 (dual last-use of binary GeMM **and** continuous PED)

G1 is a **graph** property: one compiled producer, two live consumers, retire on `gate ∧ PED`.

None of the opened DATE/DAC/ICCAD/FPGA/FCCM/FPL/ISCAS 2023–2026 papers implement that graph.

| Slogan a letter might use | What the venue actually contains | G1? |
|---|---|---|
| “Dense core + sparse core” | DATE 2025: **layers** (pixels vs later spikes). DeepFire2 transduction vs IF. | **No** |
| “Dual engine” | FireFly-T: sparse conv/linear **engine** + binary SSA **engine**, serialized per layer; residual is MS membrane AXI | **No** |
| “Dual spike inputs” | 2501.07825: Q and K both binary into SDSA | **No** |
| “Hybrid CIM” | SPARTA analog linear + digital token/attn; FlexSpIM W+Vmem of **one** IF | **No** (and CIM stop) |
| “Typed spike vs int mem” | ESTU Table I muxed destination | **No** |
| “Head prune / token skip / group skip” | Xu D³, SPARTA RL-TS, ESTU group-4 | Lossy **binary** skip, one consumer |
| “MoE / top-K experts” | Xu 2025: token routed to experts, mutually exclusive | Mux, not dual last-use |
| “Temporal differential” | Diff-DiT: ANN diffusion steps | Wrong time axis |
| “IAND residual” | Spike-IAND: **deletes** continuous residual | **Anti-G1** |
| “BN fusion” | RISCSparse frozen \(Y=aX+B\) | Opposite of live proj BN |
| “Compiled adder graph” | da4ml: one CMVM | Does not fork to PED |

**Absence of a G1 sentence is not a novelty license.** FireFly-T already has binary GeMM **and** a continuous residual **port**. ESTU already has integer mem. SDT/FireFly-T already write pre-neuron membrane residual. DATE already has two datapaths because **layers** differ. G1, if it is anything, is still: same-producer last-use of **binary GeMM + continuous PED** under same-port 8088, with DSEC AEE gates — **unmeasured**, not “unfound in ICCAD.”

---

## 6. vs ESTU (same-journal skip)

ESTU’s object: **binary spike tensor → one accumulate consumer → skip inactive groups → classification % + mW.**

Triggers that still fire after this venue sweep (any two ⇒ desk-reject):

| If the letter says… | Reviewer maps it to |
|---|---|
| FPGA spiking transformer | ESTU + FireFly-T + 2501.07825 + SpikeTA |
| overlay / reused datapath / dual-engine | ESTU §III + FireFly-T orchestrator + DATE cores |
| skip zeros / tokens / heads / groups / NRV | ESTU group-4 + SPARTA tokens + Xu D³ heads |
| AND + popcount SSA | ESTU + FireFly-T + COBRA RBMM + 2501.07825 |
| LIF then binary as the **output** of the block | ESTU eq. 2 + IAND + DeepFire2 IF |
| LUT/DSP/mW/μJ or GOP/s/W as the story | ESTU Tables + FireFly-T + IAND 38 TSOPS/W + COBRA 448 GOPS/W |
| CIFAR / ImageNet / DVS-Gesture / sEMG **accuracy** | ESTU Table V + every classification accelerator above |
| “first edge spikeformer FPGA” | ESTU already claimed this, same journal |

**What does not save a 5-page TCAS-II letter:** bigger Xilinx part; 3D F2F; ReRAM CIM; MoE experts; D³ head prune; DATE dense core; COBRA SPS; da4ml LUT cut on T10 **alone**; ERAFT fps.

**What ESTU still does not measure** (unchanged by this sweep): (i) residual/PED as a **separate** continuous tensor still live after binary last-use, (ii) DSEC valid825 AEE, (iii) same-port dual-completion service that **moves 8088**. Those are measurement requirements, not “ICCAD forgot G1.”

---

## 7. Copy / do not sell / still unmatched

**Copy (controls) if the letter touches that object.**

- DATE 2025 layer-split hybrid (dense **input** / sparse **rest**).
- FireFly-T dual-engine overlay + MS residual AXI; FireFly-S Bitmap AND on the **spike** path.
- ESTU binary SSA + group skip + 5-page FPGA letter (mandatory relative prior).
- Spike-IAND causal T-unroll mux; IAND as **negative** residual.
- 2501.07825 position-encode skip; Xu kernel-fused binary SSA; Xu D³ Q-popcount head keep.
- da4ml exact CMVM CSE for **pre-threshold** T10.
- DeepFire2 register-AND IF + split-kernel SLR; transduction dense first layer.
- LoopTree retain/recompute vocabulary.
- RISCSparse mapping/GMS tax; frozen affine BN as **contrast** to live proj BN.
- FlexSpIM event skip + stationarity **if reimplemented digitally**, then stop CIM.
- COBRA RBMM popcount as binary-ALU sibling of ESTU `Mul(spike,spike)`.
- ERAFT / ROFD as **task-side** FPGA OF (frame / non-neural event).

**Do not sell.** Dual-engine as G1; DATE cores as G1; IAND as PED hardware; D³/token skip as dual last-use; 3D PPA; CIM; COBRA/SPS as SSA; Diff-DiT as T10; LoopTree as an accelerator; RISCSparse fused BN as live proj BN; any invented FPS vs ESTU 3.76 mW / IAND 90 mW / FireFly-T GOP/s/W / COBRA 3894 GOPS.

**Still unmatched by this venue set (not a novelty claim).**

- Same-producer last-use `{binary GeMM after absorb, continuous PED/I24}` with `gate ∧ PED` retirement.
- Live full-domain projection BN \(10\times96\times120\times160\).
- Noncausal T10 mix **shared** by two consumers without duplicating the da4ml cone.
- DSEC valid825 AEE + same-port net service ≥15% that **moves 8088**.

Those holes, if any, live in measurement (Codex 8088 wait-class) and in other round-4 venue sheets — not in “DATE/ICCAD/FPGA forgot to write dual last-use.”

---

## 8. Open / unread (do not close by slogan)

- SPARTA **IEEE PDF**: unread. RL-TS/STP and ReRAM/digital split are from faculty/news + IEEE abstract.
- Diff-DiT **IEEE PDF**: unread. ADA/HCS from abstract + GitHub modes.
- ERAFT **IEEE PDF**: unread. 86 fps / VCK190 / Middlebury from abstract.
- DATE **IEEE PDF** of Aliyev: unread; arXiv HTML used as methods primary.
- ESTU GitHub tree: still search-incomplete (ADV).
- FCCM 2023–2026 SNN-Transformer: **not located**; search-incomplete.
- DAC 2025 DJP SNN prune: abstract grain only.

No silicon number in this note was synthesized. If a figure is not in a row of §0, it is not used.

---

## 9. Ten-line verdict

| Line | Object | Verdict |
|---|---|---|
| 1 | DATE 2025 hybrid | **A** layer-split overlay; **B** G1; **stop** as title |
| 2 | FireFly-T dual-engine (IEEE TC 2026) | **A** sparse+binary overlay; **B** as dual last-use; ESTU name-trap |
| 3 | ESTU TCAS-II | **A** then **same-journal stop**; not G1 |
| 4 | Spike-IAND ISCAS 2025 | **B** IAND residual; **A** causal T-unroll |
| 5 | Xu ICCAD 2024/2025 3D + D³ | **A** binary SSA/MoE + lossy head skip; **stop** 3D PPA; **B** G1 |
| 6 | SPARTA ICCAD 2025 | **A** token skip at abs grain; **stop CIM** |
| 7 | COBRA ICCAD 2025 | **A** binary popcount ALU (ANN BERT); **B** as SNN G1 |
| 8 | 2501.07825 / DeepFire2 / SyncNN | **A** FPGA spike skip / spatial IF / sync rate; ESTU-class if retitled spikeformer letter |
| 9 | da4ml / LoopTree / RISCSparse | **A** T10 compile / retain vocab / mapping tax; none forks PED |
| 10 | FlexSpIM / ERAFT / Diff-DiT | CIM stop / frame OF / ANN diffusion-diff; **not** G1 |

**No 2023–2026 DATE/DAC/ICCAD/FPGA/FCCM/FPL/ISCAS paper located that already implements absorbable AT-LIF binary GeMM **and** a separate live PED last-use on event-camera dense OF.** The year is crowded with **binary spikeformer skip/overlay/3D/CIM** — exactly ESTU’s object plus packaging. A 5-page TCAS-II letter that still looks like that object is desk-reject. G1 remains a **measurement**, not a missing sentence.
