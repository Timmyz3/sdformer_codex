# V10 — Compiler / mapping / fusion / last-use priors (DNN + SNN)

Date: 2026-09-11. Round-4 **wide** literature. Not a title. Not RTL. Not a novelty claim from a bounded miss.

**Focal question.** After official AT-LIF \(\{0,\theta\}\) with layer-shared \(\theta\) absorbed into next \(W\) (spike path = binary GeMM; residual / PED / I24 = a **separate continuous tensor**), does any compiler / mapping IR already have **typed last-use of a binary tensor AND a continuous tensor from the same producer**? Or only single-tensor lifetime / layer fusion / fmap prune?

**One-line answer.** In the named set, **not located.** The stack is: (i) **single-tensor** keep / bypass / retain–recompute / last-rank tile, (ii) **layer fusion** of one intermediate fmap, (iii) **fmap prune** (CFMP ≠ LFS), (iv) **operator fusion of frozen BN**, (v) **affine CSE** of a continuous mix, (vi) **overlay / layer-split** by operator class or layer type, (vii) **Skip / Fetch / Exec** and **FTP** on binary spikes. None of those IRs fork **one** post-absorb producer into `{s ∈ {0,1} → GeMM, u continuous → PED}` with independent last-use predicates.

A bounded miss is **not** X. Copy the complete objects as **A**. The leftover, if any, is still the dual-consumer wait-class on **this** net — not a new compiler dialect.

---

## 0. Identity freeze (do not rewrite)

From `IDENTITY_ATLIF.md` / round-4 `SCOPE.md`:

- AT-LIF: \(o=\theta\cdot H(m-\theta)\in\{0,\theta\}\). Inference: \(W\leftarrow\theta W\). Inter-layer token on the spike path is **0/1**, not analog amplitude, not unabsorbable int8.
- Pre-threshold T10 PSN mix is **continuous**, then threshold, then absorb. Prosperity / Gustav / FireFly / LoAS apply **after** absorb as complete A.
- Residual / PED / I24 is a **different continuous tensor**. Dual last-use, if it exists, is **binary gate after absorb** vs **continuous residual**, not “continuous θg shared by two MACs.”
- Native projection BN: actual batch stats over `10×96×120×160`, not frozen running stats unless a new student passes AEE. Local-window replay with free μ/σ undercharges wait/storage.
- Venue: TCAS-II Express Briefs. No CIM title. No OpenROAD-as-PPA. No first-OF-HW. No delete-35-RNE title. No RTL this agent.

**Object this sheet tests (G1 / H1 grain, not a title):**

```
producer  (compiled T10 / r1out / post-SN word)
    ├─ last_use_bin  : s ∈ {0,1} consumed by next-layer binary GeMM
    └─ last_use_ped  : continuous residual / PED / I24 consumed by add / q24
free the physical line  iff  last_use_bin ∧ last_use_ped
               (and BN-stat has sampled, if stats are still live)
```

A paper covers this object only if it **names two typed live-ranges of two datatypes from one producer** and retires storage on the **conjunction**. Layer-split cores, dual engines, fmap prune, and single-tensor retain are **different graphs**.

---

## 1. Source table (depth)

| Prior | Primary opened | Depth | Role in this sheet |
|---|---|---|---|
| **LoopTree** TCASAI 2024 | Local author full text `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/literature/LoopTree_TCASAI2024_author.txt` (also `p0_local_txt/`). Gilbert, Wu, Emer, Sze. DOI `10.1109/TCASAI.2024.3461716`. arXiv:2409.13625. | Full method: fused-layer taxonomy, retain/recompute, per-tensor retain, Buffet assumption. | Inter-layer **single-fmap** last-use language. |
| **RISCSparse** ICCAD 2024 | Local author full text `literature/RISCSparse_ICCAD2024_author.txt`. Lin, Zhu, Xie, Chen, Zhuo, Sun, Yu. | Full method: Mapping + GMS + frozen-BN operator fusion. | Coordinate tax + **frozen** BN fold. |
| **ISSCC 23.2 ConvFormer / CFMP** | Local author digest `literature/ISSCC2025_23_2_ConvFormer_author.txt` **=** `p0_txts/2512.17555.txt` (same paper; not a second CFMP). Dong, Tan, et al. DOI `10.1109/ISSCC49661.2025.10904499`. | Full digest: HAPU, **LFS** layer-fusion, **CFMP** cascaded fmap prune. | LFS ≠ CFMP. CFMP is **not** layer fusion. |
| **MAESTRO** | MICRO 2019 paper identity + local open-source card (`idea_cards/os_MAESTRO.md`). Kwon, Chatarasi, Pellauer, Parashar, Sarkar, Krishna. *Understanding Reuse…* | Paper-level reuse/occupancy model; **repo inventory, not a second full-text dump**. | Data-centric **single-tensor** reuse. |
| **Timeloop** | ISPASS 2019 (Parashar et al.) + Sparseloop MICRO 2022 extension named in the Timeloop repo. Local card is open-source inventory. | Mapping = annotated loop nest + `keep`/`bypass` per **named tensor**. | Intra-layer tile lifetime. LoopTree **extends** this to fusion sets. |
| **ZigZag** | IEEE TC 2021 / arXiv:2007.11360. Mei, Houshmand, Jain, Giraldo, Verhelst. Local card: open-source inventory. | Uneven mapping: operands **W / I / O** decoupled from memory levels. | Still three **standard** operands, not binary+PED fork. |
| **SparseTIR** | ASPLOS 2023, arXiv:2207.04606v4. Ye, Lai, Shao, Chen, Ceze. **Not in `p0_txts`.** Opened via arXiv abs + PDF text (composability, three IR stages, buffer lowering). | Sparse **format** IR on TVM TensorIR. | Format composability ≠ typed last-use. |
| **da4ml** | TRETS / arXiv:2507.04535. Official docs in-tree: `psn/cmvm_20260909/da4ml_official/docs/cmvm.md`. Local compile already run on T10. | Stage-1 MST decompose + Stage-2 bitwidth-weighted **CSE of affine CMVM**. | Continuous mix A. No binary GeMM last-use. |
| **FireFly-T** | Local full text `p0_txts/2505.12771.txt`. Li et al. Dual-engine overlay + orchestrator. | Overlay pipeline: orchestrator → sparse engine → (optional) binary engine. Residual = **pre-neuron membrane AXI**. | Operator-class overlay, not same-producer last-use. |
| **DATE 2025 hybrid** | arXiv HTML full text 2411.15409 (Aliyev, Lopez, Adegbija). IEEE PDF unread. Already mapped in `round3_gap/literature/R3L3_hybrid_cores.md`. | Dense MAC **input layer** / sparse event cores **rest**. | **Layer-split**, not G1. |
| **GustavSNN** | Local full text `literature/GustavSNN_HPCA2026_public_mirror.txt`. Hwang, Lee, Koo, Kung. HPCA 2026. | CPTB + NRV; PE = **Fetch** then **Exec**; Skip¢ weight / Skip£ row. | Binary-spike Skip/Fetch/Exec. |
| **LoAS** | Local full text `p0_txts/2407.14073.txt`. Yin, Kim, Wu, Panda. MICRO 2024. | **FTP** Algorithm 1: `t` innermost, spatially unrolled; dual-sparse spMspM. | Binary dual-sparse time-parallel. |
| **PSN** | Local full text `p0_txts/2304.12760.txt`. Fang et al. NeurIPS 2023. | \(H=WX\) continuous, then \(S=\Theta(H-B)\) binary. | Algorithmic cut, not a last-use IR. |

Related already-read notes this sheet does **not** re-derive: `literature/L4_isscc_fusion_looptree.md`, `round3_gap/literature/R3L1_firefly_t_iand.md`, `R3L3_hybrid_cores.md`, `round2_absorb/literature/R2L4_psn_before_threshold.md`, `literature/L2_gustav_loas.md`.

---

## 2. What “last-use” means in each IR (and what it does **not**)

### 2.1 LoopTree — last-use of **one intermediate fmap tile**

LoopTree’s contract (intro, local author text):

> The amount of intermediate fmap that needs to be retained to exploit inter-layer reuse can be reduced by tiling the layers and scheduling the processing of the tiles such that an intermediate fmap tile … can be computed and immediately used between layers. **When the intermediate fmap tile is not needed anymore, it is released from the buffer.**

That **is** last-use — of **one** tiled fmap, of **one** precision, with consumers that LoopTree treats as the **next layer** in a fusion set.

Four axes they add on top of TANGRAM / DeFiNES / ConvFusion / Optimus / SET / FLAT / TileFlow: partitioned ranks; recomputation; **per-intermediate-fmap** recompute; **per-tensor retain**. Recompute is not a separate knob: if the schedule + retention choice leaves an activation nowhere, it is recomputed (same as refetch for non-intermediate tensors).

Always-kept schedule constraint:

> Our scheduling follows the constraint that the output from one layer is immediately consumed by the next layer.

Buffet assumption (do not import as free on this net):

> LoopTree assumes explicit data orchestration using Buffets such that pipeline stalls can be assumed to be negligible.

**Typed dual last-use?** **No.** Per-tensor retain can keep **W vs fmap vs a second fmap** at different tile ranks. It cannot encode “this word is a **bitmap** for GeMM and a **q24 residual** for PED, and the physical bank dies only when both consumers retire.” Dual consumers of r1out (`conv_res` + `proj.sn` + BN moments) are **two (three) live tensors**, which LoopTree can name **if they are listed in the fusion set**. The type split `{binary, continuous}` after absorb is **not** in the IR. Projection BN **violates** “immediately consumed”: μ,σ are a global reduction over `10×96×120×160`.

**Copy as A:** fusion-set + retain vs recompute vs refetch **per tensor**; takeaway 5 (small buffers → layer-by-layer can beat tiled fusion). **Stop as X:** “we fused r1.”

### 2.2 Timeloop / Sparseloop — `keep` / `bypass` of named dataspaces

Timeloop mapping is an annotated loop nest. At each storage level a tensor is `keep` or `bypass`. Tile analysis charges the first / second / last iteration of each loop. Sparseloop (Timeloop 2.0) adds compressed-sparse **occupancy** of those same dataspaces.

**Typed dual last-use?** **No.** Dataspaces are the Einsum operands (typically Inputs / Weights / Outputs). Bypass is “this tensor is not stored at this level,” not “this producer has two typed unfinished uses.” LoopTree’s paper states it **extends** Timeloop to fused-layer design spaces; intra-layer Timeloop is the control, not a dual-type last-use IR.

### 2.3 MAESTRO — reuse and occupancy of **W / I / O**

MAESTRO (MICRO 2019) is a data-centric cost model: directives → tensor analysis (which dimensions couple which tensor) → cluster / reuse / performance / cost engines. Occupancy is how long a **data tile** of a standard operand stays in a buffer.

**Typed dual last-use?** **No.** Reuse classes are temporal / spatial multicast of the usual three tensors. A residual skip is, at best, a **fourth named tensor** if the user writes it as another operand — still one dtype per name, last-use = last consumer of **that** name. No absorb cut. No binary-vs-PED pair from one producer.

### 2.4 ZigZag — uneven mapping of **W / I / O**

ZigZag’s increment vs even mapping: operands at a shared memory level need not use the same loop index. Memory hierarchy can be **unbalanced** (different depths for W vs I vs O). Search is still over conv / GEMM Einsums.

**Typed dual last-use?** **No.** Uneven mapping is the closest **A** for “binary bitmap RF vs continuous RF at different hierarchy depths” **if** those two tensors are already distinct operands. ZigZag does not **create** the fork at absorb, and does not emit a last-use bit. Copy uneven mapping as a **search language** after the two tensors exist; do not retitle absorb as ZigZag.

### 2.5 SparseTIR — composable **formats**, not typed last-use

SparseTIR (ASPLOS 2023): three IR stages — (I) coordinate-space computation, (II) position-space after sparse iteration lowering, (III) loop-level TensorIR / affine. Optional `DecomposeFormat` splits one sparse tensor into hybrid formats (e.g. dense tiles + CSR remainder). Buffer lowering flattens `A[i,j]` to indptr/indices. Horizontal fusion of extra CUDA kernels is a **launch** pass, not dual-consumer last-use.

**Typed dual last-use?** **No.** Composable formats can store the **same** sparse matrix two ways. They do not type one producer as `{bitmap spike, continuous PED}`. A legal **A** after absorb: compile the **spike** tensor in a SparseTIR format (bitmap / CSR / hybrid) for GeMM. Illegal A: putting PED into that sparse format, or claiming format composability is dual last-use.

**Not in `p0_txts`.** Mechanism from arXiv v4. Do not invent GPU speedups as this-net service.

### 2.6 da4ml — CSE of an **affine** CMVM; one number type

Official two-stage algorithm (`da4ml_official/docs/cmvm.md`):

1. Column-graph MST → \(W=W_1 W_2\).
2. CSD + greedy bitwidth-weighted CSE of two-term \(a \pm (b\ll s)\).

Local compile on this net already used that object as **A**: ordinary T10 260 add/sub; lifting 159 add/sub + 35 intermediate RNE/sat. Always-ready moved; **long backpressure stayed 8088**. CSE is **pre-threshold continuous mix**. da4ml never sees a binary GeMM and never annotates last-use of a residual.

**Typed dual last-use?** **No.** Shared subexpressions are **values of the mix**, not `{s, u}` from one producer. Two-sorted CSE (boolean Sort-S vs affine Sort-M) is a **candidate rewrite on this net**, not something da4ml already emits. Copy da4ml whole as mix A; stacking da4ml + Prosperity is **A+A** (`R2L4`), not X.

### 2.7 ISSCC 23.2 — LFS is layer fusion; CFMP is **not**

Same digest as arXiv:2512.17555. Three features:

| Unit | Object | Last-use? |
|---|---|---|
| **HAPU** | Hybrid LA/VA; KTV-first O(C²) | Attention **reordering**, trained hybridization. Not last-use. |
| **LFS / DR-LFS** | Finish all VA that need on-chip KV → **replace KV slot with conv weights** → fuse conv on VA tiles, reuse weights on LA tiles | **Slot last-use of tensor family A**, then physical replace by family B. Still **one precision class** (ANN fmap / W). |
| **CFMP** | Factor one linear conv \(W_0,W_1\); expand Z; **pre-trained tiled mask**; FMS indices; DRU density restore. I/O stay dense; only Z sparse | **Prune + restore**, not fusion. Not last-use of two types. |

ISSCC challenge 2 already says conventional LF cannot co-resident K, V, and conv W. LFS’s answer is **time-multiplex the SRAM**, which is complete **A** for “finish consumers of A, then replace the bank.” Illegal copy onto r1: replace the bank **before** both gate and PED have consumed, or before full-domain BN moments exist.

Non-overlapped LF + **zero-pad missing halo** + “later attention recovers RF” is an **explicitly lossy model change** (`<0.5%` Cityscapes). Residual 3×3 + projection BN cannot use that repair silently.

**CFMP is not layer fusion.** Do not write “ConvFormer fusion” when the unit in play is FMS/DRU. CFMP also does not factor `conv–PSN–conv2` (AT-LIF sits between the two convs; a linear \(W_0 W_1\) does not commute with threshold).

### 2.8 RISCSparse — operator fusion of **frozen** BN; GMS last-use of coordinates

Three bottlenecks: Rule Map, Gather–MatMul–Scatter, uncombined ops. BN fusion (local §3.4):

> During inference, Batch Normalization (BatchNorm) operates with **fixed parameters: mean (μ) and variance (σ²)** … simplifies to \(Y = aX + B\).

ONNX Runtime fuses BN+ReLU and Add+ReLU. That is **compile-time dead-code of the reduction**, not last-use of a live stats tensor.

**Typed dual last-use?** **No.** Order-preserving mapping exists because GMS reorders; dual consumers on this net would need a **stable order** for (compressed gates, dense `conv_res`, full-domain BN). That is a **tax**, not a type system. Until projection BN is frozen and passes AEE, RISCSparse’s BN fold is **illegal** on the captured student (`PROBLEM.md` / L4).

### 2.9 FireFly-T orchestrator — layer/operator pipeline, residual is **membrane**

Overlay (`p0_txts/2505.12771.txt` §III.A):

> three main modules: the orchestrator, the sparse engine, and the binary engine. During inference, the feature maps of each network layer are processed in a pipelined manner—first by the orchestrator, then by the sparse engine, and finally by the binary engine if attention is enabled.

Orchestrator: padding, layout, register-file configs (stride, attention-enable). Sparse engine: bitmap spike decoder → membrane accumulate. Binary engine: AND-PopCount on QKV **from the sparse engine**. Latency hiding overlaps **different operators** (sparse QKV vs binary \(QK^\top\)), not two last-uses of one residual tensor.

Residual I/O is a **fourth AXI port** for **pre-neuron membrane** (SDT MS). Thresholds are **loaded as run-time parameters**, not absorbed into \(W\).

**Typed dual last-use of binary GeMM + PED from the same producer?** **No.** Dual-engine = two **operator classes**. Residual last-use, if any, is membrane writeback, which this identity **forbids** as the PED object. Copy overlay + orchestrator as **A** (`R3L1`); stop as X.

### 2.10 DATE 2025 hybrid — layer-split cores

Aliyev et al.: dense systolic MAC for the **direct-coded input layer** (raw float, non-binary, non-sparse); sparse event cores for **later** binary-spike convs. BRAM between cores is a **layer FIFO**. Rate-coded ablation **turns the dense core off**.

**Typed dual last-use?** **No.** Different layers, different tensors. Strongest **overlay-control A** for “two datapaths because the network is not uniformly binary.” G1 is two consumers of **one** compiled source. Copy DATE whole, then stop as X (`R3L3`).

### 2.11 Gustav Skip / Fetch / Exec — binary NRV pipeline

Gustav PE (`literature/GustavSNN_HPCA2026_public_mirror.txt`): two pipeline stages.

- **Fetch:** up to NR NRV spike rows; shared-weight buffer; **Skip¢ weight skipping** (index not in NRV); **Skip£ row skipping** (all-zero P-column row).
- **Exec:** merger tree + in-situ psum / LIF.

Last-use inside a PE is **this tick’s P-column potential** after LIF, then the next tick. CPTB holds potentials **in-situ** so they are not a second global tensor.

**Typed dual last-use?** **No.** Both Skip classes are on **binary spikes × multi-bit W**. Psum last-use is GeMM completion of that column. PED must not hijack the psum slot — that interference graph is **this net’s candidate**, not Gustav’s IR. Copy Skip/Fetch/Exec + NRV as **A** on the post-absorb spike path.

### 2.12 LoAS FTP — innermost `t`, dual-sparse binary

Algorithm 1 (local `2407.14073.txt`): `parallel-for t ∈ T` inside the k-reduction; then LIF, also parallel in `t`. Goals: no T-fold refetch, few temporal psums, hide sparsity-unit latency. FTP-friendly compression + inner-join. Silent neurons skipped **across all T**.

**Typed dual last-use?** **No.** FTP is a **loop permutation** of dual-sparse \(\{0,1\}\times\) sparse-W. Ready = “this output spike tensor of the layer is complete.” Residual/PED is not in the nest. Gustav **rejects** FTP as the right answer for long-T / temporal coding; they are **parallel priors**, not a stack (`L2`). Copy FTP as time-parallel **A**; do not use it as a dual-ready predicate for PED.

### 2.13 PSN 2304.12760 — algorithmic cut only

\(H=WX\) (continuous T×T mix), \(S=\Theta(H-B)\) binary. Inter-layer object is \(S\), never \(H\). That is **why** da4ml (on \(H\)) and Prosperity (on \(S\)) do not compose into X. PSN has **no** compiler last-use of residual/PED.

---

## 3. Joint map: same-producer typed last-use is **not located**

Legend: **Y** = the paper’s IR actually has that object. **A** = complete prior to copy, then stop as X. **—** = not that object.

| IR | Single-tensor lifetime | Layer fusion | Fmap prune | Frozen-BN op fusion | Affine CSE | Overlay / layer-split | Binary skip/FTP | **Same-producer typed last-use {bin GeMM, cont PED}** |
|---|---|---|---|---|---|---|---|---|
| LoopTree | **Y** (release tile) | **Y** (fusion set) | — | — | — | — | — | **—** |
| Timeloop | **Y** (`keep`/`bypass`) | intra-layer | — | — | — | — | sparse occupancy (v2) | **—** |
| MAESTRO | **Y** (occupancy) | layer-by-layer DSE | — | — | — | — | — | **—** |
| ZigZag | **Y** (uneven W/I/O) | — | — | — | — | — | — | **—** |
| SparseTIR | buffer lowering | horizontal kernel fuse | format hybrid | — | — | — | sparse formats | **—** |
| da4ml | live CSE nodes (untyped) | — | — | — | **Y** | — | — | **—** |
| ISSCC LFS | **Y** (KV-then-W slot replace) | **Y** | — | — | — | — | — | **—** |
| ISSCC CFMP | — | **— (not LF)** | **Y** | — | — | — | index FMS/DRU | **—** |
| RISCSparse | GMS buffers | — | — | **Y** (μ,σ fixed) | — | — | submanifold skip | **—** |
| FireFly-T | layer fmap FIFO | — | — | — | — | **Y** (orchestrator + dual engine) | bitmap decoder | **—** (membrane residual ≠ PED) |
| DATE hybrid | layer FIFO | — | — | — | — | **Y** (dense input / sparse rest) | event SC | **—** |
| Gustav | in-situ psum | — | — | — | — | — | **Y** Skip/Fetch/Exec | **—** |
| LoAS | FTP psums | — | — | — | — | — | **Y** FTP + silent neuron | **—** |
| PSN | — | — | — | — | mix is affine | — | binary \(S\) after \(\Theta\) | **—** (algorithm, not IR) |

**Search-incomplete (this boundary, not closed by slogan):**

- Full IEEE PDF of DATE 2025 (arXiv HTML opened; mechanism not in doubt).
- SparseTIR / Timeloop / MAESTRO / ZigZag: paper+docs opened; no local `p0_txts` dump of SparseTIR. Unlikely to hide G1 (wrong object class); still mark **docs-not-mirrored**.
- MLIR sparse dialect / TVM `qnn` last-use bits / Buffet occupancy tokens: **adjacent**, not in the assigned named set. Classical RA last-use (Chaitin, SASS `.reuse`, IBM last-use operand) is **single-value** last-use A, already named in `independent/P05_dual_consumer.md` — still not `{binary, continuous}` from one absorb cut.

**Not located in this named set:** an IR type `Producer → (Tensor[bin], Tensor[q24])` with `live_bin` and `live_ped` and a retire rule `free iff both`.

---

## 4. A / B / X for a five-page letter (compiler island)

### A — copy whole, then stop as title

1. **LoopTree** fusion-set + per-tensor retain/recompute/refetch, including takeaway 5.
2. **Timeloop / MAESTRO / ZigZag** as the **intra-layer** mapping search. ZigZag uneven W/I/O if (and only if) bitmap vs continuous are already two operands.
3. **ISSCC LFS** slot replace: finish family A, replace SRAM with family B. HAPU only if the student **is** hybrid attention.
4. **CFMP** only as **trained** \(W_0/W_1\) + mask + FMS/DRU on a **linear** conv (not `conv–PSN–conv2`). Name it prune, not fusion.
5. **RISCSparse** Mapping+GMS as coordinate tax; BN fold **only** on a frozen-stats student that passes AEE.
6. **da4ml** two-stage CMVM CSE on **pre-threshold** T10. Already in-tree.
7. **SparseTIR** formats for the **spike** tensor after absorb.
8. **FireFly-T** orchestrator + dual-engine overlay; residual port as **membrane** control, not PED.
9. **DATE** dense-input / sparse-rest layer-split overlay.
10. **Gustav** Skip/Fetch/Exec + NRV + in-situ psum; **LoAS** FTP as the competing time mapping.
11. **PSN** \(H=WX\) then binary \(S\): the algebraic reason mix CSE and spike GeMM do not fuse.

### B — hole on **this** net (not in those papers)

- After absorb, **one** producer, **two** datatypes, **two** last-uses.
- Noncausal T10: no causal prefix tile (LoopTree T-partition of PSN is recompute, not fusion).
- Full-domain projection BN: global reduction; gifted μ/σ is RISCSparse’s deletion without paying it.
- Same-port long-backpressure **8088** on ordinary and lifting; wait-class dump still missing. Source-only 8088 is **not** evidence of PED last-use until a dual-consumer kernel is classified (`ADV_H1`).
- Integer two-consumer −5.78%; serialized full chain −6.4% unclosed. Neither is 15%.
- Lifting AEE +0.013 vs +0.005 relative gate (fails as default student).

### X — why a reskin is not the letter

| Slogan | Why it is A or illegal, not X |
|---|---|
| “compiler last-use / liveness” | LoopTree release + Timeloop keep/bypass + classical RA. Need the **type split forced by absorb**. |
| “layer fusion of r1” | Alwani + LFS + LoopTree. Must charge halo **and** full-domain BN. |
| “CFMP / cascaded prune” | Not fusion; linear factor; PSN in the middle. |
| “hybrid overlay” | FireFly-T (operator split) or DATE (layer split). |
| “Skip/Fetch/Exec” | Gustav, binary only. |
| “FTP schedule” | LoAS, binary dual-sparse; Gustav disagrees for long T. |
| “CSE / da4ml” | Mix A; 8088 invariant; A+A with Prosperity. |
| “sparse IR / SparseTIR” | Format of **one** sparse tensor. |
| “fuse BN” | Legal iff μ,σ frozen. Captured proj.BN is not. |
| “uneven mapping / two RFs” | ZigZag W/I/O. Dual RF without absorb-typed last-use is a dataflow zoo. |

A true compiler increment, **if** later measured, would have to be a **wait-class / dual-ready / typed-spill** artifact whose dump shows PED last-use (or BN last-use) as the mass on a **dual-consumer** kernel, moving same-port net service ≥15% with AEE gates held. That measurement does **not** exist in this sheet. Round-3 already parked H1 as **revise-not-title** until that dump.

---

## 5. Direct answers to the assigned question

1. **Does any compiler IR here already have typed last-use of a binary tensor AND a continuous tensor from the same producer?**  
   **Not located** in LoopTree, RISCSparse, MAESTRO, ZigZag, Timeloop, SparseTIR, da4ml, ISSCC LFS, ISSCC CFMP, FireFly-T orchestrator, DATE hybrid, Gustav Skip/Fetch/Exec, LoAS FTP, or PSN.

2. **What do they have instead?**  
   Single-tensor lifetime (keep/bypass/retain/release), layer fusion of one fmap, cascaded **fmap prune** (CFMP ≠ fusion), frozen-BN operator fusion, affine CSE, overlay/layer-split by **operator or layer**, binary skip and FTP.

3. **Is “we did not find the sentence” a contribution?**  
   **No.** It is a search result inside a named boundary. Classical last-use bits and Buffets remain A. Dual-consumer last-use stays a **hypothesis on this net**, gated by wait-class, union density, and 15% — not by this literature miss.

---

## 6. What this note does not do

- Does not choose a title X or add modules.
- Does not take over Codex RTL or edit `main.tex`.
- Does not treat ISSCC 0.22 μJ/token, DATE 51×, Gustav/LoAS PPA, or SparseTIR GPU speedups as local service.
- Does not treat `2512.17555` as a second paper.
- Does not reopen lifting CSE or delete-35-RNE as titles.
- Does not claim LoopTree’s 4% model error under finite backpressure.
- Does not equate FireFly-T membrane residual with PED.

**Sources (absolute):**

- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/literature/LoopTree_TCASAI2024_author.txt`
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/literature/RISCSparse_ICCAD2024_author.txt`
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/literature/ISSCC2025_23_2_ConvFormer_author.txt`
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/survey_ab_fusion_20260910/p0_txts/2512.17555.txt`
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/survey_ab_fusion_20260910/p0_txts/2505.12771.txt`
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/survey_ab_fusion_20260910/p0_txts/2407.14073.txt`
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/survey_ab_fusion_20260910/p0_txts/2304.12760.txt`
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/literature/GustavSNN_HPCA2026_public_mirror.txt`
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/psn/cmvm_20260909/da4ml_official/docs/cmvm.md`
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/exploration_tcasii_20260911/IDENTITY_ATLIF.md`
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/exploration_tcasii_20260911/literature/L4_isscc_fusion_looptree.md`
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/exploration_tcasii_20260911/round3_gap/literature/R3L1_firefly_t_iand.md`
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/exploration_tcasii_20260911/round3_gap/literature/R3L3_hybrid_cores.md`
- SparseTIR arXiv:2207.04606v4; Timeloop ISPASS 2019; MAESTRO MICRO 2019; ZigZag IEEE TC 2021 / arXiv:2007.11360; DATE arXiv:2411.15409.
