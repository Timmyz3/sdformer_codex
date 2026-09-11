# V1 — ISSCC / JSSC / CICC / Symp. VLSI / ESSERC 2023–2026

Date: 2026-09-11. Round-4 **wide** silicon-venue mine. Not a title pitch. Not PPA. Not first-event-OF-HW.

Identity freeze (`IDENTITY_ATLIF.md`): AT-LIF output is \(\{0,\theta\}\). Inference **absorbs** layer-shared \(\theta\) into the next \(W\leftarrow\theta W\). After that fold:

- **Spike path** = binary \(\{0,1\}\times W\) GeMM (select-add / AC). Prosperity / Gustav / FireFly Bitmap-AND are **legal A** on this path.
- **Residual / PED / I24** = a **separate continuous tensor**. Dual last-use, if it exists, is **binary gate after absorb** **and** **continuous PED**, not “continuous AT-LIF amplitude shared by two MACs.”
- **CIM as title = A then STOP.** Copy a complete CIM object, then do not sell CIM as the letter.
- Do not invent silicon PPA. Do not recycle “first optical-flow hardware.”

**Question.** Which located 2023–2026 ISSCC / JSSC / CICC / Symp. VLSI / ESSERC papers are complete priors (A), strong negatives (B), **stop-as-title**, or **not-applicable** for **absorb-cut dual last-use** (binary GeMM last-use ∧ continuous PED last-use of the **same** producer)?

**Class column (vs absorb-cut dual last-use):**

| Tag | Meaning here |
|---|---|
| **A** | Complete prior to copy on a **named** object (binary AC GeMM, fusion slot-replace, unstructured NZ fetch, binary/ternary GeMM). Copy, then do not retitle. |
| **A→stop** | A on a CIM / analog / always-on-vision object; **illegal as letter title**. |
| **B** | Strong negative: the paper’s sold object is **not** dual last-use, or it **rejects** the hybrid we would need, or last-use is one accumulate consumer. |
| **n.a.** | Wrong workload (LLM face-detect ISP, spike-sorting, frame OF) with no transferable absorb-cut object at the grain we opened. |
| **unresolved** | Title/abstract located; PDF unread. Mechanism grain = abstract only. |

Search-incomplete is allowed. **Do not invent papers or numbers.**

---

## 0. What was opened this pass

| Source | Depth |
|---|---|
| `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/literature/ISSCC2025_23_2_ConvFormer_author.txt` | **Full** ISSCC digest (same as `p0_txts/2512.17555.txt`) |
| `survey_ab_fusion_20260910/p0_txts/2408.15578.txt` FireFly-S | **Full** local (venue is **TCAS-I**, not this list; opened because the task named it) |
| Idea cards `unresolved_C_Transformer*`, `unresolved_ISSCC22_UED_SNN`, `unresolved_Neuro_CIM_JSSC_extension`, `unresolved_Tu_Bitline*` | **unresolved_no_fulltext** locally; IEEE/KAIST/HKUST **abstracts** opened on the web |
| ISSCC 2025 session 23 ToC (press kit / advance program / `fengbintu/Neural-Networks-on-Silicon`) | Title-level for 23.1–23.10; **full** only for 23.2 |
| ISSCC 2026 press kit session 30 CIM + NVIDIA Alpha-Vision page | Abstract / press-kit grain |
| HKUST SPD CICC 2024 Fang abstract; IEEE JSSC Fang abstract; IEEE CICC 2025 11-5 abstract; KAIST C-Transformer ISSCC+JSSC abstracts | Abstract |
| Tambe ISSCC 2023 author PDF snippet (`sld.cs.columbia.edu/pubs/tambe_isscc23.pdf`) | Digest-level quotes (not a local `p0_txts` file) |

**Venue-out local that was still opened (do not put in the 15-row ISSCC table as if it were ISSCC):** FireFly-S, TCAS-I, arXiv:2408.15578. Bitmap-AND is **A on the post-absorb spike path**. Dual-**side** ≠ dual-**consumer**.

---

## 1. Table (≥15 located entries)

Depth: **full** = local digest/author text quoted below; **abstract** = IEEE/lab/press-kit abstract; **unresolved** = idea-card no-PDF, abstract only or title only.

| # | Paper | Venue / year | DOI / arXiv | Depth | 1-line mechanism | vs absorb-cut dual last-use |
|---|---|---|---|---|---|---|
| 1 | ConvFormer / CFMP 23.2 (Dong, Tan, Tu, Cheng) | **ISSCC 2025** 23.2 | [10.1109/ISSCC49661.2025.10904499](https://doi.org/10.1109/ISSCC49661.2025.10904499); arXiv:[2512.17555](https://arxiv.org/abs/2512.17555) | **full** (local author txt) | HAPU hybrid LA/VA + LFS KV-then-weight **layer fusion** + CFMP cascaded dense-I/O prune; ANN SegFormer Cityscapes | **A** (fusion slot-replace + cascaded prune). **B** for dual last-use: one ANN MAC consumer; no AT-LIF, no PED fork. **Stop** as “ISSCC SNN-Transformer OF.” |
| 2 | T-REX 23.1 (Moon, Seok, Intel) | **ISSCC 2025** 23.1 | IEEE ISSCC 2025 23.1 (DOI not captured this pass; press kit / Intel blog) | **abstract** | Factorizing training: shared dense \(W_S\) + per-layer sparse \(W_D\); EMA 31–65.9× cut; 16 nm ANN transformer | **A** (trained shared-dense + sparse complement). **B** dual last-use. **n.a.** as SNN/event-OF. |
| 3 | Bit-level-weight-compressed LLM 23.8 (Qin et al.) | **ISSCC 2025** 23.8 | [10.1109/ISSCC49661.2025.10904774](https://doi.org/10.1109/ISSCC49661.2025.10904774) | **abstract** | ABAF ≤3 ones/word + cluster-aligned INT-FP GEMM + Taylor SFU; LLM KV/weight compress | **A** (bit-sparse GeMM codec). **B** dual last-use. LLM **n.a.** as OF/SNN. |
| 4 | Slim-Llama 23.9 (Kim, Lee, Yoo) | **ISSCC 2025** 23.9 | [10.1109/ISSCC49661.2025.10904761](https://doi.org/10.1109/ISSCC49661.2025.10904761) | **abstract** | Binary/ternary **weights** + bit-linear for billion-param Llama | **A** (binary/ternary **W** GeMM). **B**: weight bits ≠ absorbed spike bits; no PED consumer. |
| 5 | C-Transformer 20.5 (Kim, Yoo) | **ISSCC 2024** 20.5 | [10.1109/ISSCC49657.2024.10454330](https://doi.org/10.1109/ISSCC49657.2024.10454330) | **abstract** (idea card unresolved PDF) | Homogeneous **DNN-Transformer / Spiking-Transformer** core; big-little + implicit weight gen for LLM | **A** (homogeneous DNN/SNN transformer PE). **B** dual last-use (one token path, LLM). **Stop** as “we also do SNN-Transformer silicon.” |
| 6 | C-Transformer JSSC extension | **JSSC 2025** 60(10):3802–3815 | [10.1109/JSSC.2025.3554699](https://doi.org/10.1109/JSSC.2025.3554699) | **abstract** | HDSC+HMAU, OSSU output-spike speculation, IWGU+ESC implicit weights; 28 nm | Same as #5 at journal grain. **A→stop** as title. PDF unread: do not invent HMAU schematic. |
| 7 | Sparse Transformer Processor / STP (Tambe et al.) | **ISSCC 2023** 22.9 | [10.1109/ISSCC42615.2023.10067817](https://doi.org/10.1109/ISSCC42615.2023.10067817) | **digest quotes** (author PDF) | Entropy early-exit + mixed-precision predication + sentence V/F; attention-head prune | **A** (lossy early-exit / head prune). **B** dual last-use. Classification entropy ≠ AEE. **Stop** as OF early-exit. |
| 8 | Butterfly zero-skipper sparse Transformer (Liu et al.) | **ISSCC 2023** 16.2 | [10.1109/ISSCC42615.2023.10067360](https://doi.org/10.1109/ISSCC42615.2023.10067360) | **abstract** | In-memory **butterfly zero skipper** for unstructured prune + CIM local-attention-reusable engine; 8 b ANN | **A** (unstructured zero-skip). **A→stop** CIM title. **B** dual last-use. |
| 9 | Fang 3-D AC array | **CICC 2024** | [10.1109/CICC60959.2024.10529019](https://doi.org/10.1109/CICC60959.2024.10529019) | **abstract** (PDF unread; HKUST SPD) | Spiking-**only** AC; 3-D timestep-parallel array; parallel NZ spike/weight fetch; vs hetero SNN+CNN MAC cores | **A** (post-absorb binary AC GeMM + unstructured NZ fetch). **B**: **rejects** hetero MAC cores; no PED. **Stop** pJ/SOP as local PPA. |
| 10 | Fang 3-D array JSSC | **JSSC 2025** 60(3):977–989 | [10.1109/JSSC.2024.3507095](https://doi.org/10.1109/JSSC.2024.3507095) | **abstract** (PDF unread) | Same chip family: 3-D array, parallel NZ fetcher, multimode scheduler for SCONV / QKV / SSA; ImageNet spiking transformer | **A** as #9 at journal grain. **Do not** fill array geometry from unread body. |
| 11 | Sample-wise-adaptive SNN (Yang, Zou, Fudan) | **CICC 2025** 11-5 | [10.1109/CICC63670.2025.10983259](https://doi.org/10.1109/CICC63670.2025.10983259) | **abstract** | Tiny PN tags IR neurons; prune IN per sample; unstructured-model-aware pipeline-event PE; hierarchical clock-gate | **A** (sample-wise neuron prune + event pipeline). **B**: lossy per-sample; frozen OF student cannot drop neurons without AEE. Not dual last-use. |
| 12 | 4D-CIM CNN/Transformer (Wang, Lin, Yoo) | **CICC 2025** | [10.1109/CICC63670.2025.10983287](https://doi.org/10.1109/CICC63670.2025.10983287) | **abstract** | Reconfigurable 4-D CIM map + adaptive feature reuse for diverse CNN/Transformer ops | **A→stop** CIM. ANN, not SNN dual last-use. |
| 13 | SparseTrim (Li, He et al.) | **CICC 2025** | [10.1109/CICC63670.2025.10982861](https://doi.org/10.1109/CICC63670.2025.10982861) | **abstract** (title+DOI) | On-chip decompression of fine-grained sparse models | **A** (sparse codec). **n.a.** as SNN/PED. PDF unread. |
| 14 | ROM-LTE (Jeong, Park, Jeon) | **CICC 2026** | [10.1109/CICC65509.2026.11509629](https://doi.org/10.1109/CICC65509.2026.11509629) | **abstract** | ROM LUT tensor, output-stationary, complementary ROM, latch PE skips irregular zeros | **A** (constant-tensor LUT / unstructured zero skip). Not SNN; not two consumers of one producer. |
| 15 | Neuro-CIM (Kim, Yoo) | **JSSC 2023** 58(10):2931–2945 (VLSI 2022 conference paper is **2022**) | [10.1109/JSSC.2023.3273238](https://doi.org/10.1109/JSSC.2023.3273238) | **abstract** (idea card unresolved PDF) | ADC-less SNN CIM; sign-extend BL gating; 1-b comparator; **early stopping** of neuronal ops | **A→stop** CIM. Early-stop is **neuron-fire**, not dual last-use of GeMM+PED. |
| 16 | TranCIM (Tu et al.) | **JSSC 2023** 58(6):1798–1809 (ISSCC **2022** 11.5) | JSSC [10.1109/JSSC.2022.3213542](https://doi.org/10.1109/JSSC.2022.3213542); ISSCC22 [10.1109/ISSCC42614.2022.9731645](https://doi.org/10.1109/ISSCC42614.2022.9731645) | **abstract** (idea card unresolved PDF) | Full-digital **bitline-transpose CIM**; pipeline vs parallel modes; sparse attention scheduler | **A→stop** CIM. Sparse attention scheduler is **A** for attention skip, **B** for PED last-use. Conference year 2022 is **out of 2023–2026 ISSCC**; JSSC 2023 is in-window. |
| 17 | Alpha-Vision / ALPhA-Vision 31.9 (Keller, NVIDIA/Stanford) | **ISSCC 2026** 31.9 | [10.1109/ISSCC49663.2026.11409322](https://doi.org/10.1109/ISSCC49663.2026.11409322) | **abstract** (NVIDIA page) | Always-on CNN **and ViT**, end-to-end **no EMA**, leakage power-gate; face detect | **n.a.** / **stop**. Frame ViT, not event SNN, not OF. No dual last-use. |
| 18 | Hybrid SNN–CNN digital SRAM CIM 30.8 (Yeh, Chang, NTHU/ITRI) | **ISSCC 2026** 30.8 | [10.1109/ISSCC49663.2026.11409250](https://doi.org/10.1109/ISSCC49663.2026.11409250) | **abstract** / press kit | Shared SNN–CNN MAC in digital CIM; 1-to-8 b config; CFD-SC-RC + PS-WR-ODM | **A→stop** CIM. Hybrid is **precision/model mux in one macro**, not same-producer GeMM∧PED. |
| 19 | Lele fused frame–event SoC | **JSSC 2024** 59(1):52–64 | [10.1109/JSSC.2023.3297411](https://doi.org/10.1109/JSSC.2023.3297411) | **abstract** (idea card unresolved PDF) | RRAM-CIM **CNN** (frames) + SRAM-CNM **SNN** (events) for **target tracking**; parallel CNN/SNN modules | **A** (modality-split CNN vs SNN **modules**). **B** dual last-use of **one** producer. **Stop** CIM title. **Stop** “first event-OF HW” (this is tracking, and OF HW already exists elsewhere). |
| 20 | UED clock-free SNN wake-up | **ISSCC 2022** 22.7 (**out of 2023–2026**); kept because idea card named it | [10.1109/ISSCC42614.2022.9731795](https://doi.org/10.1109/ISSCC42614.2022.9731795) | **abstract** (unresolved PDF) | Clock-free UED bionic SNN + CIM for AIoT wake-up | **A→stop** CIM / always-on wake-up. **Out of year window.** Not dual last-use. |

**Not a 21st ISSCC row:** FireFly-S (`2408.15578`, TCAS-I). Opened in full. Bitmap AND of spike mask × weight mask is **A on the absorbed spike path**. Dual-**side** sparsity is **not** dual last-use.

---

## 2. Quote pins (full local / digest only)

### 2.1 ISSCC 2025 23.2 ConvFormer — full local

Pins from `literature/ISSCC2025_23_2_ConvFormer_author.txt` (same body as `p0_txts/2512.17555.txt`).

**Identity of the chip (ANN ConvFormer, not SNN):**

> 23.2 A 28nm 0.22μJ/Token Memory-Compute-Intensity-Aware CNN-Transformer Accelerator with Hybrid-Attention-Based Layer-Fusion and Cascaded Pruning for Semantic-Segmentation

> DOI: 10.1109/ISSCC49661.2025.10904499

**Challenge 2 = cannot co-resident K, V, and conv weights (the fusion prior):**

> 2) While Layer-Fusion (LF) [13-18] is a common technique to reduce Fmap EMA, it is infeasible to buffer key (K), value (V), and convolution weights on-chip simultaneously.

**LFS slot replace (legal A for “finish consumers of A, replace the physical buffer with B”):**

> The LFS first reuses on-chip buffered KV to compute all VA tiles, then replaces the KV with off-chip convolution weights. Afterward, convolution layers are sequentially fused with each VA output tile and LA input tile, reusing both KTV and convolution weights.

**Non-overlapped LF is explicitly lossy (illegal silent copy onto residual r1 / PED):**

> we propose a non-overlapped LF processing scheme wherein the FC is broken into several non-overlapped FCs. The subsequent attention layer recovers the broken receptive field by its inherent long-range dependency. The boundary fusion issue is then resolved by zero-padding the unavailable boundary tiles, resulting in 50% GB usage and a 20% reduction in operations, with <0.5% accuracy drop.

**CFMP: I/O dense, only intermediate Z sparse:**

> The CFMP decomposes the convolution weights into two cascaded components W0 and W1, injecting redundancy by enlarging the intermediate Fmap Z, which is then pruned using pre-trained tiled masks. Since Z is the only sparse Fmap, with the input and output Fmaps remaining dense

**Their PPA (cite as prior, never as local):** 28 nm, 200–625 MHz, 0.65–1.0 V, peak 52.90 TOPS/W, 0.22 μJ/token SegFormer-B0. **Do not multiply HAPU × LFS × CFMP into FPS.**

**Map after absorb.** Slot multiplexing is **A** for SRAM occupancy of `{spike bitmap, W, residual PED}`. Dual last-use forbids replacing the source slot after **only** the binary GeMM finishes if PED/I24 still needs the continuous tensor. CFMP density-restore before a dense consumer is **A** for “I/O stay dense”; it does **not** retire PED. Non-overlapped pad is a **new student**, not a schedule of the frozen net.

### 2.2 FireFly-S — full local, venue **TCAS-I** (opened on instruction)

Pins from `p0_txts/2408.15578.txt`.

> FireFly-S employs a Bitmap-based sparse decoding logic in the dual-side sparsity detector, enabling efficient exploitation of sparsity found in both activation spikes and synaptic weights. This approach directly skips all computations involving zero values

> At CLK0, a bitwise AND operation is executed between the vector pair, yielding a result denoted as x, which identifies active events.

> This vector is then subjected to a bitwise AND operation with the weight mask to pinpoint active indices that align with the mask.

**Map after absorb.** Bitmap `spike ∧ Wmask` **is** the post-absorb spike-path object. Copy as **A**. Dual-**side** (W and spike) is **not** dual-**consumer** (gate and PED). Continuous PED cannot be AND-popcounted. Gradient-rewire prune is a **new student**.

### 2.3 Tambe ISSCC 2023 22.9 — digest quotes (author PDF, not `p0_txts`)

> Key contributions of this work are as follows: (1) A specialized datapath for entropy-based early exit assessment reduces BERT latency by up to 6.13× … (2) a mixed-precision (MP) FP4/FP8 MAC supports per-vector exponent biases … (3) a fine-grained sentence-level power management scheme

> Attention head pruning (AP) further extends these savings by 1.5×.

**Map.** Early-exit / head-prune is **A** as a **lossy** control. Optical-flow AEE is not SST-2 entropy. Not dual last-use.

---

## 3. Abstract-grain notes (PDF unread; do not upgrade)

### 3.1 C-Transformer ISSCC 2024 20.5 + JSSC 2025 — idea cards still `unresolved_no_fulltext`

ISSCC IEEE/KAIST abstract (opened): LLM EMA 68% of power; pruning cannot reach high sparsity on translation/QA; title object is **homogeneous DNN-Transformer / Spiking-Transformer** + big-little + implicit weight generation.

JSSC 2025 IEEE abstract (opened): three blocks **HDSC+HMAU**, **OSSU** (output spike speculation), **IWGU+ESC**; Samsung 28 nm 1P8M; 0.7–1.1 V, 200 MHz; GPT-2 / T5 / mT5 / FSMT.

**Map.** Closest ISSCC **SNN-Transformer silicon** in-window. After absorb, a homogeneous PE that muxes MAC vs AC is **A for PE arithmetic**, **B for dual last-use**: LLM token path is one consumer. OSSU speculates **output spikes**, not PED completion. **Stop** as letter title (“homogeneous SNN-Transformer processor”). PDF still missing locally — do not invent HMAU gate-level.

### 3.2 Fang CICC 2024 + JSSC 2025 3-D array — PDF unread

HKUST SPD abstract (opened, DOI 10.1109/CICC60959.2024.10529019):

> Previous work [2] mitigates this problem by using heterogeneous SNN and CNN cores but incurs increased area and power consumption due to costly MAC array implementation. Therefore, a spiking-only accelerator with enhanced energy efficiency across all sparsity levels is highly desired.

Three named challenges: redundant W/psum across timesteps; unstructured NZ fetch one-by-one; one-size scheduler.

JSSC abstract (IEEE 10.1109/JSSC.2024.3507095): 3-D array (parallel timesteps, weight reuse); parallel non-zero fetcher; multimode scheduler for **SCONV, spiking Q/K/V, SSA**; 40 nm; **0.078 pJ/SOP** and ImageNet **77.6%** are **their** headline — not local PPA.

**Map.** After absorb, spike path **is** AC. Fang is the strongest **in-venue A** for unstructured binary GeMM + timestep-parallel reuse + NZ fetch + SSA. It is also a **B** against DATE-style dense MAC + sparse SNN cores: they **sell spiking-only**. No continuous PED. **Copy AC+NZ as A, stop as X / stop pJ/SOP.**

### 3.3 CICC 2025 11-5 sample-wise adaptive SNN

IEEE abstract (DOI 10.1109/CICC63670.2025.10983259; program paper 11-5, Fudan):

> (1) A dynamic pruning scheme utilizing a tiny Perception Network (PN) that identifies inference-relevant (IR) neurons and prunes inference-irrelevant ones In the Inference Network (IN) for each sample … (2) An efficient unstructured-model-aware architecture parallelly scheduling active weights for IR neurons into a pipeline-event coupled processing strategy … (3) A hierarchical workload monitor … workload-aware clock gating

**Map.** Sample-wise neuron drop is **lossy** and **not** last-use of a frozen producer’s two tensors. **A** for unstructured event pipeline + clock-gate. **B** / **stop** as adaptive-OF-SNN title (G2-class spatial skip already stopped elsewhere as ASNA-Flow).

### 3.4 ISSCC 2026 30.8 hybrid SNN–CNN digital CIM

Press kit + Google-Scholar restatement (DOI 10.1109/ISSCC49663.2026.11409250): first fully-digital 16 nm SRAM CIM for **hybrid SNN–CNN**; shared MAC (CFD-SC-RC); 1–8 b; **444.21 TOPS/W** (1bIN–8bW–14bOUT, **SNN**) and **62.84 TOPS/W** (8bIN–8bW–22bOUT, **CNN**) are **their** macro numbers.

**Map.** Hybrid = **one macro, two precisions/models**, not dual last-use of one T10/PED producer. **A→stop CIM.** Do not quote TOPS/W as local.

### 3.5 ISSCC 2026 31.9 Alpha-Vision

NVIDIA page (DOI 10.1109/ISSCC49663.2026.11409322):

> ALPhA-Vision is an always-on low-power subsystem for DNN-inference-based vision tasks in edge SoCs. … supports CNN and ViT inference and employs hardware/software co-design to enable fully end-to-end execution with no external memory accesses. … face detection with 787µs latency and 99.3% detection accuracy with 4.6 mW average power at 60fps.

**Map.** **n.a.** Always-on **frame** ViT. Not event camera, not SNN, not OF. Fine-grained power-gate is a distant A for leakage, not dual last-use.

### 3.6 Neuro-CIM JSSC 2023 — idea card unresolved PDF

IEEE abstract (10.1109/JSSC.2023.3273238): SNN to drop CIM ADC; sign-extended-bit BL gating; 1-b comparator; **early stopping** of unnecessary neuronal operations (−31% power in **their** text).

**Map.** **A→stop CIM.** Early-stop = membrane/neuron halt, **not** `last_use(binary GeMM) ∧ last_use(PED)`.

### 3.7 TranCIM / Tu bitline-transpose — idea cards unresolved PDF

JSSC abstract (10.1109/JSSC.2022.3213542): bitline-transpose digital CIM; **pipeline mode** (attention, cut off-chip) vs **parallel mode** (FC); INT16 attention / INT8 FC; **sparse attention scheduler (SAS)**. ISSCC 2022 conference paper is **out of year**; JSSC June 2023 is in-window.

**Map.** **A→stop CIM.** SAS is A for sparse attention. Pipeline vs parallel is **layer-type mux**, not GeMM∧PED of one source.

### 3.8 Lele JSSC 2024 fused frame–event — idea card unresolved PDF

IEEE/ADS abstract (10.1109/JSSC.2023.3297411): RRAM CIM **CNN** on frames + SRAM CNM **SNN** on events; tracking; two-level power-gate; parallel CNN and SNN modules; **>100 outputs/s** is **their** number.

**Map.** **A** for **modality-split** CNN/SNN engines (different sensors, different nets). **B** for same-producer dual last-use. **Stop** CIM. **Stop** first-OF-HW (tracking ≠ DSEC dense OF; event-OF silicon already exists: SENECA, ASNA-Flow title-level, TrueNorth OF, plane-fit FPGA).

### 3.9 Slim-Llama / ISSCC 23.8 / T-REX — binary/sparse **ANN** GeMM

- Slim-Llama: binary/ternary **weights**, bit-linear Llama. **A** for `{−1,0,+1}×act` GeMM. Absorbed AT-LIF is `{0,1}×W`, different alphabet, and PED is still a second tensor.
- 23.8 ABAF: bit-level weight/KV compress, cluster INT GEMM. **A** for bit-sparse codec.
- T-REX: shared dense \(W_S\) + sparse per-layer \(W_D\). **A** for factorized W; Phi-adjacent, not dual last-use.

None is SNN. None forks PED.

### 3.10 ISSCC 2022 UED SNN — out of window, idea card named

IEEE abstract 10.1109/ISSCC42614.2022.9731795: clock-free UED SNN + CIM, AIoT wake-up, 82 nW / 0.53 pJ/SOP / 40 μs are **their** numbers. **A→stop CIM.** Keep unresolved PDF. Do not use as 2023–2026 prior except as older relative.

---

## 4. ISSCC 2025 session 23 — rest of the session (title-level)

Opened ToC (press kit / advance program / GitHub `fengbintu/Neural-Networks-on-Silicon`). **No SNN + Transformer + event-OF paper in this session.** 23.2 is the only ConvFormer/fusion/prune hit with full local text.

| # | Title (as printed) | This pass | vs dual last-use |
|---|---|---|---|
| 23.1 | T-REX 16 nm transformer, reduced EMA | abstract (§3.9) | A factorized W; n.a. SNN |
| 23.2 | ConvFormer HAPU/LFS/CFMP | **full** | A fusion/prune; B dual last-use |
| 23.3 | EdgeDiff few-step diffusion | title only | n.a. |
| 23.4 | Nebula 3-D PNN, multi-skipping | title only | n.a. (point-cloud PNN, not SNN-Transformer OF) |
| 23.5 | MAE 3 nm mini autoencoder | title only | n.a. |
| 23.6 | MEGA.mini big/little NPU | title only | n.a. (name collision with “big-little” C-Transformer) |
| 23.7 | BROCA social-agent SoC | title only | n.a. |
| 23.8 | Bit-level-weight-compressed LLM | abstract | A bit-sparse GeMM |
| 23.9 | Slim-Llama binary/ternary Llama | abstract | A binary **W** |
| 23.10 | HuMoniX text-to-motion, output sparsity | title only | n.a. |

---

## 5. Symp. VLSI / ESSERC 2023–2026 — search-incomplete

**Recorded queries this pass:** `Symposium VLSI 2023 2024 2025 2026 SNN spiking transformer event camera`; `ESSERC 2023 2024 2025 2026 SNN accelerator`.

| Hit | Status |
|---|---|
| Neuro-CIM **VLSI 2022** | **Out of year.** JSSC 2023 extension is row 15. |
| Keller per-vector INT4 **VLSI 2022** / JSSC 2023 | Out of VLSI year; JSSC 2023 exists (`10.1109/JSSC.2023.3234893`). Idea card unresolved PDF. ANN INT4, **n.a.** as SNN dual last-use. |
| CogniVision **VLSI 2024** | Always-on smart vision SoC (catalog). Not SNN/event-OF. **n.a.** Title-level only this pass. |
| VLSI 2024 Canon SPAD “event vision sensing” (SSC magazine report) | Imager, not SNN accelerator. **n.a.** |
| **ESSERC / ESSCIRC 2023–2026 dedicated SNN / spiking-transformer / event-OF accelerator** | **Not located** in this pass. Search-incomplete, not a claim of absence in nature. |
| Catalog `MAIN-R270` “Zhang optical-flow processor, CICC 2026” | Title **incomplete** in local CSV; **not re-identified** on the web this pass. Keep **unresolved**. Do not invent a CICC 2026 OF chip. |

No VLSI 2023–2026 paper was opened at PDF depth that is jointly *SNN + transformer + event OF*.

---

## 6. Cross-cut vs absorb-cut dual last-use

**No opened ISSCC/JSSC/CICC/VLSI/ESSERC 2023–2026 paper implements:** post-absorb binary GeMM **and** a second live continuous PED/I24 consumer of the **same** producer, with typed last-use on both, on event-camera dense optical flow.

What the window **does** contain:

1. **Binary / AC SNN silicon is A on the spike path, then stop as title.** Fang CICC/JSSC (spiking-only AC + NZ fetch + SSA), Neuro-CIM (ADC-less CIM + neuron early-stop), ISSCC 2026 30.8 (SNN/CNN mux in digital CIM), C-Transformer (homogeneous DNN/SNN transformer, LLM). After \(\theta\) absorb, copying any of these onto DSEC OF is **prior-copy**, not X.
2. **CIM is A then STOP as title.** Neuro-CIM, TranCIM/Tu, ISSCC 2023 16.2, ISSCC 2026 30.8, CICC 2025 4D-CIM. SCOPE forbids CIM-as-contribution.
3. **Transformer fusion/pruning silicon is ANN, complete A for schedules.** ConvFormer LFS slot-replace + CFMP dense-I/O prune; T-REX factorized W; 23.8 bit-compress; Slim-Llama binary W. Slot-replace is the only fusion object that **could** be copied onto `{bitmap, W, PED}` occupancy — and it **fails** if PED outlives the binary GeMM.
4. **Heterogeneous CNN+SNN is layer/modality split, not G1.** Lele: frame CNN CIM vs event SNN CNM. Fang **explicitly rejects** hetero SNN+CNN MAC cores. DATE 2025 hybrid (not this venue list) is the same split. Dual last-use is **not** “a dense core and a sparse core.”
5. **Event cameras in this venue list are sensors or tracking, not DSEC spikeformer OF accelerators.** Alpha-Vision is frame ViT. Lele is tracking. ISSCC 2023 stacked CIS+EVS (catalog `MUSHA-IS013`) is an **imager**, not opened this pass. **Do not write first-OF-HW.**
6. **Sample-wise / entropy / non-overlapped-pad pruning is lossy A.** CICC 2025 PN/IN, Tambe entropy exit, ConvFormer zero-pad RF repair. Frozen student + AEE gates forbid silent copy.

**Mandatory relative priors if a 5-page letter talks silicon:** ConvFormer 23.2 (fusion+prune, **full** local), C-Transformer (only ISSCC SNN-Transformer chip in-window at abstract grain), Fang 3-D AC (strongest binary GeMM A), TranCIM/Neuro-CIM (CIM stop), Lele (event SNN module, not OF). FireFly-S Bitmap-AND is A but **TCAS-I**, not ISSCC.

---

## 7. Idea cards that stayed unresolved (PDF still missing)

| Card | Located this pass? | Residual |
|---|---|---|
| `unresolved_C_Transformer` / `_ISSCC24` | Yes: ISSCC DOI + JSSC 2025 extension DOI + abstracts | Body unread. Do not invent HMAU/OSSU circuits. |
| `unresolved_ISSCC22_UED_SNN` | Yes: DOI 10.1109/ISSCC42614.2022.9731795 | **2022.** Abstract only. CIM wake-up. |
| `unresolved_Neuro_CIM_JSSC_extension` | Yes: JSSC DOI 10.1109/JSSC.2023.3273238 | Abstract only. CIM A→stop. |
| `unresolved_Tu_Bitline*` | Yes: ISSCC 2022 + **TranCIM JSSC 2023** | Conference out of year; JSSC in-window, abstract only. CIM A→stop. |
| `unresolved_ISSCC25_ConvFormer` | **Superseded.** Full author txt + `2512.17555` are the same digest. | Card’s `unresolved_no_fulltext` is stale relative to `isscc2025_convformer.md`. |
| `unresolved_Lele_fused_frame_event_optical_flow` | JSSC 2024 tracking SoC located | Title on the card said “optical flow”; the JSSC object is **tracking**, plus a **2022 arXiv OF algorithm** (`2207.10720`) that is **not** this silicon. Do not merge them. PDF unread. |
| `unresolved_Keller_Per_Vector_INT4` | JSSC 2023 DOI known | PDF unread. ANN. |
| `unresolved_Spike_CIM` | **Not re-identified** this pass | Stay unresolved. |

---

## 8. Search-incomplete (explicit)

- **ESSERC 2023–2026:** no SNN / spiking-transformer / event-OF accelerator located. Not a proof of empty set.
- **Symp. VLSI 2023–2026:** no dedicated SNN-transformer OF chip opened; CogniVision / SPAD imager are title-level n.a.
- **ISSCC 2026 session 31 AI Accelerators** beyond 31.9 Alpha-Vision: ToC not fully opened.
- **ISSCC 2025 23.3–23.7, 23.10:** titles only.
- **Fang CICC 2-page PDF and JSSC body:** unread. Array size, clocks, scheduler microarchitecture **unknown**.
- **C-Transformer ISSCC digest and JSSC body:** unread.
- **CICC 2026 Zhang OF processor** (local catalog stub): not found on the web this pass.
- **T-REX DOI:** not captured (press kit / Intel blog only).
- **CICC 2025 SparseTrim / 4D-CIM:** abstracts/titles; PE/SRAM unread.
- **ISSCC 2023 stacked CIS+EVS, ISSCC 2025 15.2 spike-sorting:** catalogued, not mechanism-read (wrong object).

**Do not convert a miss in this boundary into novelty.**

---

## 9. Sources (absolute / URL)

Local:

- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/literature/ISSCC2025_23_2_ConvFormer_author.txt`
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/survey_ab_fusion_20260910/p0_txts/2512.17555.txt`
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/survey_ab_fusion_20260910/p0_txts/2408.15578.txt`
- idea cards: `unresolved_C_Transformer.md`, `unresolved_C_Transformer_ISSCC24.md`, `unresolved_ISSCC22_UED_SNN.md`, `unresolved_Neuro_CIM_JSSC_extension.md`, `unresolved_Tu_Bitline_Transpose_CIM_Transformer.md`, `unresolved_Tu_ISSCC22_BitlineTranspose_CIM.md`, `isscc2025_convformer.md`
- `exploration_tcasii_20260911/IDENTITY_ATLIF.md`
- prior notes used as maps, not as invented papers: `literature/L4_isscc_fusion_looptree.md`, `round3_gap/literature/R3L3_hybrid_cores.md`, `web/W1_2025_2026_venues.md`

Web (recorded 2026-09-11): KAIST Pure C-Transformer; IEEE Xplore abstracts 10454330, 10964134, 10777513, 10983259, 10130013, 9931922, 10067817, 10067360, 11409322, 11409250; HKUST SPD `1783.1/138276`; NVIDIA Alpha-Vision page; ISSCC 2025/2026 press kits; CICC 2025 program PDF; Columbia `tambe_isscc23.pdf`.
