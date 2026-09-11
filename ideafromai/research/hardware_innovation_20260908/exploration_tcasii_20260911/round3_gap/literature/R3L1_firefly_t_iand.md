# R3L1 — FireFly-T overlay vs Spike-IAND-Former T-unroll (and FireFly-S Bitmap AND)

Date: 2026-09-11. Round-3 gap sheet. Full local texts only. Not a novelty claim. Not a PPA invention. Not a CIM title.

**Question.** After AT-LIF \(\{0,\theta\}\) with layer-shared \(\theta\) absorbed into next \(W\), the spike path is binary GeMM. Residual / PED / I24 is a **separate** continuous tensor. Is FireFly-T’s dual-engine overlay a complete prior for “hybrid overlay of sparse conv + binary attention”? Does either engine consume a continuous residual beside that binary GeMM from the **same** producer? Is Spike-IAND’s T-unroll mux a causal LIF unroll or a noncausal T10 PSN mix?

**One-line answer.** FireFly-T is **A** for overlay control of *sparse conv/linear + binary AND-PopCount attention*. Residual I/O is **pre-neuron membrane**, not a second MAC on PED/I24 from the same producer as the binary gate — that dual last-use is **B** (strong negative), not an unfound-sentence novelty. Spike-IAND IAND residual is **B** (deletes the continuous consumer). Spike-IAND mux `111/101/000` is **A** for *causal* T-parallel LIF unroll / membrane-free tick-batching, **B** as a T10 PSN substitute.

---

## 0. Identity freeze (do not rewrite)

From `IDENTITY_ATLIF.md` / round-3 `SCOPE.md`:

- Official AT-LIF: \(o=\theta\cdot H(m-\theta)\in\{0,\theta\}\). Inference: layer-shared \(\theta\) **absorbed** into next \(W\). Transmitted spikes are \(\{0,1\}\). Spike path after absorb is **binary GeMM**.
- Residual / PED / I24 is a **separate continuous tensor**, not analog AT-LIF amplitude shared by two MACs.
- Pre-threshold T=10 PSN mix \(H=WX\) is **continuous**, then threshold, then absorb. Causal LIF leak-and-reset over \(T\) is a different object.
- Venue: TCAS-II, 5 pages. Same-journal ESTU (`ADV_ESTU_same_journal.md`) is already **binary SSA + skip + classification mW** on a 5k-LUT FPGA. A letter that still looks like ESTU/FireFly-T/IAND (binary spikeformer overlay, AND-PopCount, activity skip, CIFAR/ImageNet %, TSOPS/W) is desk-reject class.
- Forbidden: silicon PPA invention; CIM as title; selling IAND as a residual win; claiming novelty from “we did not find a sentence.”

**Read in full (local texts):**

| File | Role |
|---|---|
| `survey_ab_fusion_20260910/p0_txts/2505.12771.txt` | FireFly-T, arXiv:2505.12771v1 [cs.AR] 19 May 2025. Li, Li, Shen, Zhao, Zhang, Zeng. Dual-engine overlay: sparse engine + binary engine. |
| `survey_ab_fusion_20260910/p0_txts/2503.19643.txt` | Spike-IAND-Former accelerator, arXiv:2503.19643v1 [cs.AR] 25 Mar 2025. Chen & Chang, NYCU. IAND residual + unrolled LIF. Printed title: *Hardware Efficient Accelerator for Spiking Transformer With Reconfigurable Parallel Time Step Computing*. |
| `survey_ab_fusion_20260910/p0_txts/2408.15578.txt` | FireFly-S (optional contrast). Bitmap AND is **sparsity detector**, not residual algebra. |

Round-2 already mapped FireFly-S Bitmap AND as **A** on the post-absorb spike path (`R2L3`) and IAND residual as identity-illegal (`R2L5`, `L5`). This sheet does not re-litigate those as titles. It answers the **overlay dual-engine** and **T-unroll mux** questions that round 2 left unread.

---

## 1. Title confirmation

| arXiv | Printed title | What it actually is |
|---|---|---|
| 2505.12771 | **FireFly-T: High-Throughput Sparsity Exploitation for Spiking Transformer Acceleration with Dual-Engine Overlay Architecture** | FPGA overlay (KV260 / `xczu5ev`). Sparse engine = bitmap multi-lane decoder + 3D load-balanced array for **conv/linear**. Binary engine = 2D systolic **AND-PopCount** for \(QK^\top\) / \(QK^\top V\). Orchestrator = padding + layout + attention-enable. Task: CIFAR-Net / Spikingformer classification. 4-bit W following FireFly-S. |
| 2503.19643 | **Hardware Efficient Accelerator for Spiking Transformer With Reconfigurable Parallel Time Step Computing** | TSMC 28 nm ASIC for Spike-IAND-Former. Residual **ADD replaced by IAND** so the whole model is spike I/O. Fully parallel tick-batching + unrolled LIF, mux `111/101/000` for \(T=4/2/1\). Vector PE for 3×3, 1×1, matmul. Classification (ImageNet / CIFAR-10). |
| 2408.15578 | **FireFly-S: Exploiting Dual-Side Sparsity for Spiking Neural Networks Acceleration with Reconfigurable Spatial Architecture** | Spatial FPGA, **not** overlay. Bitmap AND of spike vector × weight mask, then IF/LIF. SCNN5/7/9 conv stacks, \(T=4\). No transformer, no residual skip algebra. |

Paper numbers belong to **their** nets and boards. They are not this-net DSEC AEE or same-port service.

---

## 2. Quote index (section pins)

Quotes are from the local `p0_txts` extracts. Line numbers are extract lines, not IEEE PDF pages.

### Q1 — FireFly-T dual-engine contract (Abstract)

> “we propose FireFly-T, a dual-engine overlay architecture that integrates a sparse engine for activation sparsity and a binary engine for spiking attention.”  
> Source: 2505.12771 Abstract.

### Q2 — Overlay pipeline order (III.A)

> “Within the FireFly-T IP, the architecture is implemented as an overlay consisting of three main modules: the orchestrator, the sparse engine, and the binary engine. During inference, the feature maps of each network layer are processed in a pipelined manner—first by the orchestrator, then by the sparse engine, and finally by the binary engine if attention is enabled. This processing sequence is applied uniformly across all layers, reusing the same hardware throughout inference.”  
> Source: 2505.12771 §III.A.

### Q3 — Sparse engine: spike bitmap in, membrane accumulate, optional handoff to binary engine (III.A)

> “The sparse engine accelerates convolutional and linear operations by exploiting fine-grained activation sparsity. It employs high-throughput multi-lane decoders alongside a scalable, workload-balanced 3D array architecture. The decoder concurrently extracts multiple non-zero indices from vectorized input spikes … pipelined accumulation across output channels enables sequential, group-wise membrane updates, thereby allowing a more compact and resource-efficient neuronal dynamics module. The resulting outputs are passed through a max-pooling module and, if attention is enabled, forwarded to the binary engine for further processing.”  
> Source: 2505.12771 §III.A Sparse Engine.

### Q4 — Binary engine: 1-bit AND-PopCount on QKV from the sparse engine (III.A / III.B)

> “The binary engine employs a resource-efficient 2D systolic array to accelerate binary attention operations, notably the \(QK^T\) and \(QK^T V\) computations. … At its computational core is the AND-PopCount operation … the binary engine interfaces directly with the sparse engine, consuming intermediate results and operating in a pipelined manner.”  
> “Upon receiving intermediate results from the sparse engine, each PE in the binary engine performs an AND-PopCount operation on two input vectors, each of length \(P_{Bk}\).”  
> Source: 2505.12771 §III.A Binary Engine; §III.B.

### Q5 — Latency-hiding: two engines, two **different** operators on QKV, not residual (III.C)

> “Fig. 5 illustrates the proposed latency-hiding pipeline, which overlaps spiking attention by having the sparse engine compute \(Q\), \(K\), and \(V\), while the binary engine simultaneously calculates \(QK^T\) and \(QK^T V\).”  
> “the execution time complexity is reduced from \(O(3T_s L d^2 + 2T_s L^2 d)\) to \(O(3T_s L d^2)\)”  
> Source: 2505.12771 §III.C.

### Q6 — Residual I/O is a **fourth AXI port**, not the spike FMap port (III.A system)

> “four 128-bit S-AXI-HP ports are enabled. These ports handle specific tasks: loading weights, biases, and membrane thresholds; fetching feature map inputs; writing feature map outputs; and providing a read-and-write interface for residual input and output data.”  
> Source: 2505.12771 §III.A. Fig. 3 labels `S_HP2` as `In/Out Res.` next to the sparse engine; `S_HP0` In FMap, `S_HP3` Out FMap.

### Q7 — Residual algebra is **pre-neuron membrane**, chosen so conv still sees spikes (II.B + Table I)

> “Residual connections in spiking transformers can be implemented either post-neuron, by adding spikes, or pre-neuron, by adding membrane potentials as illustrated in Fig. 2. We adopt the pre-neuron strategy since it maintains spike-based inputs for all convolutions, preserving computational efficiency, while also allowing residuals to carry richer membrane potential information, which benefits performance [12], [13].”  
> Table I: Spikformer = No binary attn, **Post-Neuron**; Spikingformer = No binary attn, **Pre-Neuron**; SDT / SDTv2 / BESTformer = Yes binary attn, **Pre-Neuron**. Preferred combo: binary attention + pre-neuron residual.  
> Source: 2505.12771 §II.B, Table I. Refs [12],[13] are Spike-driven Transformer / v2 (membrane shortcut).

### Q8 — Sparse decoder consumes **bitmap spikes**, not residual membrane (IV.A1)

> “bitmap-based decoders produce only one output (a one-hot vector or index) per cycle … a 4-lane decoder (\(M=4\)) can process the input bitmap `0x9042` in a single cycle by simultaneously extracting four active spikes”  
> “Each worker is responsible for extracting valid weight data corresponding to \(M\) non-zero indices—provided by the \(M\)-lane sparse decoder—for membrane potential accumulation.”  
> Source: 2505.12771 §IV.A.

### Q9 — Orchestrator metadata includes attention enable, not a PED tensor (IV.C)

> “these registers define control parameters (e.g., stride, rotation, layer-specific settings) … while also propagating metadata—such as accumulation loop counts and attention enable flags—to the downstream sparse and binary engines.”  
> Source: 2505.12771 §IV.C.

### Q10 — FireFly-T conclusion: two engines + orchestrator (VII)

> “a dual-engine overlay architecture: the sparse engine employs multi-lane sparse decoder and a bank-conflict-free workload balancer to effectively exploiting sparse inputs … the binary engine utilizes implicit dataflow manipulation and LUT6-Optimized AND-PopCount logic, enabling resource-efficient spiking attention support. Complementing the dual-engine, an orchestrator flexibly manages dataflow, ensuring adaptability to diverse network topologies for the overlay architecture.”  
> Source: 2505.12771 §VII.

### Q11 — Spike-IAND: residual ADD is the non-spike problem; IAND is the fix (I, II)

> “Spikformer only reduces the computational complexity of self-attention. It still needs non-spike calculation due to the residual summation.”  
> “the residual summation in this model causes non-spike computations (values are no longer 0/1) in the convolution layers due to the addition operations. This not only limits the energy efficiency advantages of SNN Transformers but also complicates their deployment and optimization on SNN hardware instead of simple AND gates for multiplication.”  
> “we adopt the element-wise-IAND [11] as the operator in the residual block … The IAND operation is \(x * (1 - \mathrm{ConvBN}(x))\), where \(x\) is the input, and ConvBN is convolution and BN operations as in Fig. 1. Since both \(x\) and ConvBN(\(x\)) are spike, their multiplication can be simplified as an AND operation.”  
> Source: 2503.19643 §I–II. [11] = Fang et al. SEW/IAND, NeurIPS 2021.

### Q12 — Spike-IAND: whole model spike I/O (V Conclusion)

> “The proposed design solves the non-spike computation problem in spiking transformers by replacing residual addition with element-wise-IAND to make the whole model spike I/O only.”  
> Source: 2503.19643 §V.

### Q13 — Spike-IAND: fully parallel tick-batching; causal LIF dependency; unroll (III.A, III.C)

> “In the SNN operations, the multiplication between input and weight has no data dependency between time steps. This allows parallel processing of input for all time steps. However, the neuron output of each time step depends on the membrane potential of previous time steps. To resolve this data dependency, we unroll the LIF neuron loop for all time steps.”  
> “By unrolling the LIF neuron to simultaneously compute four time steps, we achieve spatial-temporal parallel acceleration. This approach of parallel processing outputs of all time steps reduces hardware computation delay while also avoiding the need to access membrane memory.”  
> Source: 2503.19643 §III.A / §III.C.

### Q14 — Spike-IAND mux `111/101/000` (Fig. 5 caption)

> “Fig. 5. The proposed reconfigurable unrolled LIF neuron. The MUX selector input (from left to right) will be set to 111/101/000 for the time step=4/2/1, respectively.”  
> “This neuron can be configured to support different time steps with three multiplexers to control how to propagate data to neighbor neurons for different time steps.”  
> Source: 2503.19643 Fig. 5 caption + §III.C. LIF \(\theta=0.5\), leak \(=0.25\); design supports up to four time steps (§II).

### Q15 — Spike-IAND PE: spike SRAM in, unrolled LIF out (III.A)

> “The system accesses weights and inputs from off-chip memory through the memory controller and stores them into weight SRAMs and spike SRAMs. … The unrolled LIF neurons convert the output of the output channel into output spikes and can generate output for four time steps simultaneously. The results are saved in spike temp SRAMs”  
> Source: 2503.19643 §III.A. No residual-membrane AXI. No second continuous tensor.

### Q16 — FireFly-S Bitmap AND is sparsity decode, not residual IAND (V.B)

> “At CLK0, a bitwise AND operation is executed between the vector pair, yielding a result denoted as \(x\), which identifies active events.”  
> “a one-hot encoded output, \(y\), is generated through the logical expression \(y = x \land \neg(x-1)\) … The register holding \(x\) is then updated using \(x = x \land \neg y\)”  
> “at CLK6, the weight data is fetched from the RAM and used to perform IF/LIF neuron model computation.”  
> Source: 2408.15578 §V.B. Operand pair = **spike bitmap × weight mask**. Not \(x \odot (1-\mathrm{ConvBN}(x))\). FireFly-S grep of “residual”: **no matches**.

---

## 3. A. FireFly-T overlay: engine contracts, residual I/O, dual consume?

### 3.1 Sparse engine vs binary engine (contracts)

| | Sparse engine | Binary engine |
|---|---|---|
| **Consumes** | Vectorized **spike bitmaps** length \(P_{C_i}\) (Q8). Weights multi-bit (4-bit in experiments, following FireFly-S). Bias & threshold into neuronal dynamics (Fig. 4). | **1-bit** vectors from the sparse engine: \(Q,K,V\) after SN (Q4, Q5). |
| **Does** | Fine-grained sparse conv / linear: multi-lane bitmap decode → load-balanced 3D array \((P_{T_s}, P_{F_x}, P_{C_o})\) → group-wise **membrane** update → SN → optional max-pool (Q3). | 2D systolic **AND-PopCount** of two 1-bit vectors of length \(P_{B_k}\) for \(QK^\top\) and \(QK^\top V\) (Q4). Implicit SRAM byte-write transpose \((L,d,T_s)\to(T_s,d,L)\) (Fig. 8). LUT6 6:2 / 6:3 compressors. |
| **Produces** | Spike feature maps in overlay layout \((P_{T_s}, P_{F_x}, P_{C_i})\). If attention enabled, QKV handed to binary engine. Membrane updated **inside** neuronal dynamics. | Attention (or “Output Conv. Or Attn.” in Fig. 4C) reformatted back to the same overlay layout (Q4). |
| **When live** | Every layer (SPS 3×3, SSA’s extra conv, MLP 1×1). | **If attention is enabled** (Q2, Q9). Not on pure conv layers. |
| **Parallelism** | \((P_{T_s}, P_{F_x}, P_{C_i}, P_{C_o})\); decoder lanes \(M\), workers \(P_{W_o}\). Eval config \((P_{T_s}\times P_{F_x}, P_{C_i}, P_{C_o})=(8,16,64)\). | \(P_{B_m}\times P_{B_n}\) systolic, inner \(P_{B_k}\). Sized so \(3(W_s/P_s)\approx 2(W_b/P_b)\) (Eq. 3–4). |

Orchestrator is **not** a third compute engine. It is config + padding + push/pop layout so the sparse engine can tile arbitrary \(K_h\times K_w\) (they contrast Fang et al. fixed 3×3 grid). Attention-enable is a **flag**, not a residual tensor (Q9).

### 3.2 What residual I/O actually carries

- **Port:** dedicated R/W AXI (`S_HP2` `In/Out Res.`), **separate** from In FMap / Out FMap / Weight+bias+threshold (Q6).
- **Algebra:** **pre-neuron**, “adding **membrane potentials**” (Q7). Explicitly **not** post-neuron spike ADD (Spikformer column in Table I).
- **Intent:** keep **spike-based inputs for all convolutions**; residual carries “richer membrane potential information” (Q7). This is the SDT / Spikingformer **membrane-shortcut** restated in hardware, with refs [12],[13].
- Fig. 4 places `Residual In/Out` on the **sparse engine** next to Neuronal Dynamics, **not** on the binary engine.

So residual I/O carries **membrane / pre-activation skip**, not the transmitted spike map. After SN, the sparse engine’s conv still sees binary spikes (Q7 + Q8). That is the same split as SDT MS (`R2L5` §2.3): linear ops see \(\{0,1\}\); add is on \(U\).

### 3.3 Pre-neuron vs post-neuron

**Pre-neuron (adopted).** Add on membrane, then SN. Spike tensors stay binary. Continuous object that leaves the skip is the **same neuron family’s membrane stream**.

**Post-neuron (rejected here, used by Spikformer in Table I).** Add spikes. That is SEW-ADD class: integer spikes, which IAND-Former also refuses (Q11).

FireFly-T does **not** implement IAND. It implements MS-style pre-neuron residual **and** binary attention (Table I preferred row).

### 3.4 Same-producer continuous PED/I24 beside binary GeMM?

**As written: no.**

1. Sparse engine’s MAC operand is the **spike bitmap**, not residual/PED/I24 (Q8). Membrane is the **accumulator / skip**, not a second dense GeMM operand from the same producer as a binary gate.
2. Binary engine’s MAC operand is **1-bit QKV** produced **after** SN by the sparse engine (Q4–Q5). It does not read `In/Out Res.`.
3. Dual-engine overlap is \(Q,K,V\) (sparse) ∥ \(QK^\top, QK^\top V\) (binary) — two **attention operators**, one data family (binarized QKV), not `{binary gate, continuous PED}` from one source (Q5).
4. Overlay order is **layer-serial reuse**: orch → sparse → (binary if attn) (Q2). That is ESTU-class **operator overlay**, not two live last-uses of one producer vector.

Do **not** retitle FireFly-T residual as PED \(1\times1\) on continuous `r1out`, or as I24 skip that is also a dense conv operand. The paper never names PED, I24, or a second MAC on the residual tensor.

---

## 4. B. Is FireFly-T dual-engine a complete prior?

### 4.1 “Hybrid overlay control of sparse conv + binary attention” → **A (complete prior, copy)**

The paper’s own sentence is the control:

> “a dual-engine overlay architecture that integrates a sparse engine for activation sparsity and a binary engine for spiking attention” (Q1, Q10).

Must copy if a TCAS-II letter says “we overlay sparse conv and binary attention”:

- Overlay reuse of one sparse array + one binary systolic across layers (Q2).
- Sparse path: bitmap decode of **binary activations** × multi-bit \(W\), then LIF (Q3, Q8). After AT-LIF absorb, this **is** the spike-path ALU (same class as FireFly-S Bitmap AND on the spike path; FireFly-T is activation-sparse, FireFly-S is dual-side).
- Binary path: AND-PopCount SSA, 1-bit × 1-bit (Q4). Same journal already has ESTU `Mul(spike,spike)` = 16 AND + popcount.
- Orchestrator + attention-enable flag for topology (Q9).
- Latency hiding of \(QK^\top\) behind V (Q5) — copy as **pipeline scheduling**, not as dual-consumer last-use.

**Not** a title. ESTU (same journal, 5 pages, binary SSA skip, classification mW) plus FireFly-T (FPGA overlay, GOP/s/W, CIFAR/ImageNet) already occupy that desk. A 5-page letter that only restates dual-engine overlay is ESTU/FireFly-T class.

### 4.2 “Dual last-use of binary gate + continuous residual” → **B (strong negative)**

FireFly-T has:

- binary spike GeMM (sparse engine) **and**
- a continuous residual **port** (membrane skip) **and**
- a binary attention engine.

It does **not** have: one producer vector whose **last use** is both (i) binary gate after absorb and (ii) continuous residual/PED MAC.

The residual is MS-style add **into the same LIF membrane**, then SN. The binary engine is a **downstream attention operator** on already-binarized QKV. Overlay **serializes** engines per layer (Q2).

This is the same negative already frozen in `R2L5`: MS is **A** for “add on \(U\), then SN, spike tensors stay \(\{0,1\}\)”; dual-path X is last-use of `{binary gate, continuous residual/PED, BN-stat}`, **not** “we put the shortcut before SN.” FireFly-T hardwareizes that MS row. It does not create the dual last-use.

**Absence of a dual last-use sentence is not a novelty license.** ESTU already stores spike mem vs integer mem without forking one source into gate **and** PED. SDT already writes pre-neuron membrane residual.

---

## 5. C. Spike-IAND-Former

### 5.1 T-unroll mux `111/101/000`

Confirmed (Q14). Three multiplexers chain neighbor unrolled LIF cells:

| \(T\) | MUX select (L→R) | Meaning in Fig. 5 |
|---|---|---|
| 4 | `111` | all four unrolled neurons coupled (full chain) |
| 2 | `101` | every-other coupling |
| 1 | `000` | no neighbor propagate |

This is **reconfigurable spatial unroll of a causal LIF recurrence**, not a noncausal mix matrix over \(T=10\).

### 5.2 Residual algebra

**IAND**, not ADD, not MS (Q11–Q12):

\[
g(x,\mathrm{ConvBN}(x)) = x \odot (1-\mathrm{ConvBN}(x))
\]

Both operands are spikes → AND-gate. Whole model **spike I/O only**. Fang et al. SEW Table: IAND keeps \(\{0,1\}\); identity by \(A\equiv 0\).

This is **B / identity-illegal** on this net (`R2L5`, `L5`): IAND exists to **destroy** the continuous residual that PED / I24 / live proj BN **are**. Copy IAND as a **negative control** (what happens if you force spike-only skip). Do not sell it as hardware for PED.

### 5.3 T-parallel causal LIF, not noncausal T10 PSN

The paper states the dependency they unroll (Q13):

- \(W x\) **across \(t\)**: independent → parallel PEs (4 PE arrays per PE block, 12 blocks, four time steps × 12 channels).
- **Neuron output at \(t\)** depends on **membrane at \(t-1\)** → unroll the LIF **loop** so \(t=0..3\) sit in space, mux-coupled.

That is **causal** leaky-integrate-and-fire (leak 0.25, \(\theta=0.5\)), spatially unrolled, membrane SRAM deleted because \(V[t]\) is a wire to the next unrolled cell.

This net’s pre-threshold object is **noncausal T=10 mix** of continuous \(H=WX\), **then** threshold, **then** absorb. IAND unroll:

- does not mix future ticks into a PSN \(H\);
- never holds a continuous \(H\) after threshold;
- \(T\le 4\), not \(T=10\);
- encoding layer bitplanes an 8-bit **image**, not event-OF voxels.

**A** for: T-parallel causal LIF, single weight fetch, membrane-free residency, mux-reconfigurable \(T\in\{1,2,4\}\).  
**B** as a stand-in for T10 PSN / lifting CSE.

### 5.4 Hardware shape (paper numbers only; not a PPA claim)

TSMC 28 nm, 500 MHz, 198.46 kGE logic, 139.25 KB SRAM, 90.153 mW, 3456 GSOPS, 38.334 TSOPS/W (their Table II). CIFAR-10 \(T=4\): 46.72 FPS, 73.88% zeros. Vector PE: 8×9, 3×3 / 1×1 / matmul. **No dual engine.** One spike PE + unrolled LIF.

IAND’s “first Spiking Transformer accelerator” sentence (Abstract / §IV.B) is **their** 2025-03 claim; FireFly-T is 2025-05 FPGA overlay. Neither is event-OF silicon. Do not fight first-ness in a TCAS-II letter.

---

## 6. Contrast: FireFly-S Bitmap AND vs residual vs IAND

Three different ANDs. Do not collapse.

| AND | Where | Operands | Role vs this net |
|---|---|---|---|
| FireFly-S CLK0 Bitmap AND (Q16) | Spatial sparse detector | spike bitmap \(\land\) **weight mask** | **A** on **post-absorb spike GeMM** (`R2L3`). Not residual. No residual word in the paper. |
| FireFly-T multi-lane bitmap decode (Q8) | Overlay sparse engine | spike bitmap → indices; then \(W\) gather into membrane | **A** on spike path (activation-sparse sibling of FireFly-S). Residual is a **different** AXI membrane port (Q6–Q7). |
| Spike-IAND residual IAND (Q11) | Model skip | spike \(x\) \(\land\) \(\neg\mathrm{ConvBN}(x)\) | **B**: deletes continuous residual. Makes later conv AND-gates by **removing** PED/I24. |
| FireFly-T / ESTU AND-PopCount | Binary SSA | 1-bit \(\land\) 1-bit, then popcount | **A** for binary attention ALU. Same-journal ESTU already has this as `Mul(spike,spike)`. |

FireFly-T vs FireFly-S architecture: T is **overlay dual-engine** (reuse across layers, transformer + conv); S is **spatial** one-core-per-layer, conv-only, dual-side sparsity. T’s sparse decoder is multi-lane (M indices/cycle); S’s decoder is single-nonzero-per-cycle Bitmap walk (CLK0–CLK6). T does **not** claim dual-side weight sparsity as the engine contract (Table III: Ours = Fine-Grained **activation** sparsity + all four parallelisms).

---

## 7. D. Mechanism classes (A copy / B strong negative / X possible / stop)

Rule: never claim novelty from not finding a sentence. “Possible X” only if a **named object in this net** is still unmatched **and** the paper does not already occupy it under another name.

| # | Mechanism | Class | Why |
|---|---|---|---|
| 1 | Overlay dual-engine: sparse conv/linear + binary AND-PopCount attention + orchestrator | **A** | Q1–Q2, Q10. Complete prior. Copy in related work. Not a title (ESTU + FireFly-T). |
| 2 | Sparse engine: bitmap spike decode → multi-bit \(W\) → LIF membrane | **A** | Q3, Q8. After absorb this **is** the spike path. Copy FireFly-S/T together. |
| 3 | Binary engine: 1-bit QKV AND-PopCount, latency-hide \(QK^\top\) behind V | **A** | Q4–Q5. Copy as SSA scheduling. ESTU already binary SSA skip. |
| 4 | Pre-neuron residual AXI carrying **membrane**, conv still binary | **A** as MS hardware | Q6–Q7. Copy as SDT membrane-shortcut on a port. **B** if retitled PED/I24 dual MAC. |
| 5 | Dual last-use `{binary gate after absorb, continuous residual/PED}` from one producer | **B** | FireFly-T does not do it (overlay serial + MS add + binary engine on QKV). IAND deletes B. ESTU typed mem is not a fork. Absence ≠ X. |
| 6 | IAND residual / spike-I/O-only transformer | **B** | Q11–Q12. Identity-illegal on PED/I24. Negative control only. |
| 7 | Unrolled LIF mux `111/101/000`, \(T=4/2/1\), membrane-free, W fetched once | **A** for **causal** T-parallel LIF residency | Q13–Q15. Copy vs SpinalFlow serial tick-batching. |
| 8 | Same unroll as noncausal T10 PSN mix \(H=WX\) then threshold | **B / stop** | Q13 is explicit causal \(V[t]\!\leftarrow\!V[t-1]\). T10 is pre-threshold continuous mix. Do not rename. |
| 9 | FireFly-S Bitmap AND as residual skip | **B / stop** | Q16. AND is sparsity detect, not IAND/MS/PED. |
| 10 | Silicon PPA / TSOPS/W / 3.76 mW ESTU-style as this-net contribution | **stop** | Venue 5-page TCAS-II; ESTU already classification mW; FireFly-T GOP/s/W on KV260; IAND 38.334 TSOPS/W on 28 nm. No new PPA. No CIM title. |

No **X** row. Dual last-use is **B** (occupied by a different algebra: MS skip + separate attention engine), not an open hole created by silence.

---

## 8. What a 5-page TCAS-II letter must still copy vs must not sell

**Copy (controls).** FireFly-T dual-engine overlay; FireFly-S Bitmap AND on the **spike** path after absorb; SDT/FireFly-T pre-neuron membrane residual; Spike-IAND causal T-unroll + mux `111/101/000` + membrane-free tick-batching; ESTU binary SSA + group skip + mW classification.

**Do not sell.** Dual-engine as dual last-use; IAND as residual for PED; unroll as T10 PSN; Bitmap AND as residual; overlay as event-OF same-port service; any invented PPA vs ESTU 3.76 mW / IAND 90.153 mW / FireFly-T GOP/s/W.

**Still unmatched by these two papers (not a novelty claim):** live full-domain projection BN; I24 / PED as a **second consumer** of a producer that is also a binary gate; noncausal T10 mix before threshold; DSEC valid825 AEE; same-port/same-state/same-backpressure net service. Those holes, if any, live in other round-3 sheets — not in “FireFly-T forgot to write dual last-use.”

---

## 9. Ten-line verdict table

| Line | Object | Verdict |
|---|---|---|
| 1 | FireFly-T sparse engine (bitmap spikes × W → LIF) | **A** spike-path overlay after absorb |
| 2 | FireFly-T binary engine (AND-PopCount SSA) | **A** binary attention ALU / hide \(QK^\top\) |
| 3 | FireFly-T orchestrator + attention-enable overlay | **A** topology reuse; **stop** as title (ESTU overlay) |
| 4 | FireFly-T residual I/O | **Membrane skip (pre-neuron MS)**; **not** spike map; **not** PED/I24 MAC |
| 5 | Dual-engine = sparse conv + binary attn overlay | **A complete prior** for that sentence |
| 6 | Dual-engine = dual last-use binary gate + continuous residual | **B** (serial overlay + MS add + QKV handoff) |
| 7 | Spike-IAND residual | **IAND** \(x(1-\mathrm{ConvBN}(x))\); **B** deletes continuous consumer |
| 8 | Spike-IAND mux `111/101/000` | **A** causal LIF unroll \(T=4/2/1\); **B** as T10 PSN |
| 9 | FireFly-S Bitmap AND vs residual | **Sparsity detector**, not skip algebra; **A** on spike GeMM only |
| 10 | TCAS-II 5p / ESTU mW / no CIM / no PPA invention | Dual-engine restatement **desk-reject**; no X from silence |
