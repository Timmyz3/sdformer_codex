# R3L3 — Hybrid cores: DATE dense/sparse overlay, FlexSpIM, SpikePool, Fang CICC, COMPASS, ESTU

Date: 2026-09-11. Literature remap only. Freeze: `round3_gap/SCOPE.md` + `IDENTITY_ATLIF.md`.  
**CIM is not a title.** Copy a CIM paper as **A** if it is a complete prior on a named object, then **stop using it as X**.

Locked identity used for mapping (not a novelty claim):

\[
o_i[t]=\theta_i\cdot H(m_i[t]-\theta_i)=\theta_i s_i[t],\quad s\in\{0,1\},\quad o\in\{0,\theta\}.
\]

At inference, layer-shared \(\theta\) is absorbed \(W\leftarrow\theta W\). After that fold:

1. **Spike path** = binary \(\{0,1\}\times W\) GeMM (select-add / AC).
2. **Residual / PED / I24** = a **different continuous tensor**. Dual last-use, if it exists, is **binary gate after absorb** and **continuous residual**, not “continuous AT-LIF amplitude shared by two MACs,” and not two core types assigned to two **layers**.

G1 (typed last-use) is: retire one producer only when **both** post-absorb binary GeMM **and** continuous PED have completed. That is not “a dense core for layer 1 and sparse cores for the rest.”

Numbers below that are 0.078 pJ/SOP, 95.8%, 51×, 3.76 mW, etc. belong to those papers’ own chips/nets. They are not this-net cycle share. **Do not invent silicon numbers** that are not in a cited primary.

---

## 0. Source table (what was actually opened)

| Paper | Primary opened this pass | Depth | Not opened |
|---|---|---|---|
| **DATE 2025 hybrid** (Aliyev, Lopez, Adegbija) | arXiv HTML **full text** [2411.15409v1](https://arxiv.org/html/2411.15409); abs [arxiv.org/abs/2411.15409](https://arxiv.org/abs/2411.15409) (journal-ref: DATE 2025). IEEE Xplore 10992893 landing **empty fetch**. | Methods + results from arXiv HTML. | IEEE PDF / IEEE HTML. |
| **FlexSpIM** ISCAS 2025 | Local author full text `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/literature/FlexSpIM_ISCAS2025_author.txt` (IEEE ISCAS 2025, DOI 10.1109/ISCAS56072.2025.11043507). Idea cards `survey_ab_fusion_20260910/idea_cards/flexspim_iscas2025.md` and `arxiv_2410.23082.md`. Cross-check arXiv HTML [2410.23082](https://arxiv.org/html/2410.23082). | §I–IV, Figs. 1–7, Table I. | — |
| **SpikePool** | Local full text `survey_ab_fusion_20260910/p0_txts/2510.12102.txt` (arXiv:2510.12102v1, 14 Oct 2025). | Full paper including appendix. | — |
| **Fang CICC 2024** | **Not in** `p0_txts`. IEEE abstract DOI [10.1109/CICC60959.2024.10529019](https://doi.org/10.1109/CICC60959.2024.10529019) via IEEE authors page + **HKUST SPD** [hdl.handle.net/1783.1/138276](https://hdl.handle.net/1783.1/138276) (same abstract; CICC 2024 pp. 1–2, art. 10529019). Author-lab restatement: Westlake CenBRAIN news 2024-01-25. | Abstract + author restatement of the 2-page CICC. | **CICC PDF unread. JSSC 2025 extension PDF unread** (`IEEE JSSC` 60(3):977–989, DOI 10.1109/JSSC.2024.3507095). |
| **COMPASS** MICRO 2024 | **Not in** `p0_txts` (`rg COMPASS` over that tree: 0 hits). Card `idea_cards/unresolved_COMPASS.md`: `unresolved_no_fulltext`. IEEE/MICRO abstract DOI 10.1109/MICRO61859.2024.00083. | Title + publisher abstract only. | **Full PDF unread.** AE repo `ZongwuWang/COMPASS_AE` not cloned. |
| **ESTU** TCAS-II 2025 | Already opened in `exploration_tcasii_20260911/adversarial/ADV_ESTU_same_journal.md` (author AAM, DOI 10.1109/TCSII.2025.3626209). This note does **not** re-open the PDF. | AAM-level, as recorded there. | IEEE HTML empty; GitHub tree search-incomplete (same as ADV). |

`p0_txts` also has `2410.23082.txt` (FlexSpIM preprint). Mapping below uses the **ISCAS author full text**, which matches the arXiv HTML on mechanism.

---

## 1. DATE 2025 hybrid — layer-split overlay, not G1

**Bibliographic identity (from arXiv abs + HTML).**  
Ilkin Aliyev, Jesus Lopez, Tosiron Adegbija, *Exploring the Sparsity-Quantization Interplay on a Novel Hybrid SNN Event-Driven Architecture*, arXiv:2411.15409v1, 23 Nov 2024; journal-ref **DATE 2025**. IEEE record 10992893 (added 21 May 2025; HTML not retrieved here). Code: `github.com/githubofaliyev/SNN-DSE/tree/DATE25`.

**What the paper actually is.** First hybrid **inference** architecture for **direct-coded** SNNs. Direct coding feeds **raw floating-point input** into the first convolution; that layer emits floating-point membranes; a LIF then emits **binary** spikes that drive the rest of the net (Introduction; eqs. 1–2).

Two **core types**, assigned by **layer**, not by consumer of one tensor:

| Core | Assigned to | Datapath (arXiv HTML §IV) |
|---|---|---|
| **Dense core (DC)** | Input layer: “largest feature map dimensions, **non-binary, and non-sparse** activations” | Weight-stationary systolic MAC array, 27 PEs for 3 input ch × 3×3. Activ unit: bias + leak \(\beta\) + threshold \(\theta\); spike=1 and subtract \(\theta\), else 0. Spike trains written to BRAM (timestep-major). |
| **Sparse cores (SC)** | Remaining CONV/FC: event-driven spiking convolutions | ECU compresses spike trains (`SpikeEvents` via priority encoder + bit-reset). Address gen maps each spike to the 3×3 neighborhood. Neural cores **accumulate weights into membrane BRAM**, then Activ for LIF. Output-channel unroll factor \(N\). Spike max-pool = **OR** over \(N\times N\). |

QAT: int4 weights/biases; **neuronal parameters stay float**; membranes dequantized for the LIF (II-B, IV-D). Quantization **increases** spike sparsity vs fp32 by 6.1 / 10.1 / 15.2% on SVHN / CIFAR-10 / CIFAR-100 (Fig. 1). Rate-coded comparison **turns the dense core off** (V-D): rate coding is sparse-only.

**Their numbers (theirs, not ours).** Xilinx Virtex UltraScale+ XCVU13P, 100 MHz. Direct vs rate on CIFAR-10 LW: 2 vs 25 timesteps, 87.01% vs 77.37%, 7.6 vs 201 mJ/image → **26.4×** energy (Table II). vs Gerlinghoff DATE 2022 on CIFAR-100: **51×** throughput, ~½ power, 56.9% vs 60.1% (Table III). int4 vs fp32 energy “3.4×” on CIFAR-10 is **their** fp32/int4 pair (Fig. 4), not a foundry PPA for this net.

**Answers to the assigned questions.**

- **Is this class A for hybrid overlay control?** **Yes.** Copy in full: *direct-coded input is dense/non-binary → dedicated dense MAC; subsequent spike convs are event-driven sparse cores; resources partitioned from a layer-wise spike-count workload model (eq. 3).* That is the complete prior on “two datapaths because the **network is not uniformly binary**.”
- **Is this G1 typed last-use of binary GeMM + continuous PED from the SAME producer?** **No.** The dense core consumes **pixels** of the input image. The sparse cores consume **binary spike trains of later layers**. BRAM between cores is a **layer FIFO**, not a fork of one compiled T10 / θg word into (i) post-absorb binary GeMM and (ii) continuous PED/I24. There is no residual/PED consumer, no optical flow, no absorb-then-Prosperity. LIF (eq. 2) **throws the membrane away as a binary spike**. Rate-coded ablation even **disables** the dense core.

**Copy as A, then stop as X.** A letter that says “we also have a dense core and a sparse core” is a DATE 2025 reskin. The leftover, if any, is still G1: same-producer last-use of **binary GeMM + continuous PED**, which this paper does not have.

---

## 2. FlexSpIM — CIM; WS/OS hybrid dataflow; skip is event-driven IF, not PED last-use

**Bibliographic identity (author IEEE text).**  
Chauvaux, Kneip, Posch, Makinwa, Frenkel, *An Event-Based Digital Compute-In-Memory Accelerator with Flexible Operand Resolution and Layer-Wise Weight/Output Stationarity*, ISCAS 2025, DOI 10.1109/ISCAS56072.2025.11043507. TU Delft author package locally transcribed. arXiv:2410.23082 is the preprint of the same paper.

**Illegal as title.** Digital **CIM-SRAM**. Unified 16 kB 6T array for **weights and membrane potentials**. Dual wordline / dual SA / five-phase in-array AND/NOR → 1-bit full adder. SCOPE: analog/digital CIM is not the letter. After copying any complete prior below, **stop using FlexSpIM as X**.

**What the paper actually is.** Event-driven **layer-first** IF SCNN classifier on IBM DVS-Gesture. Execution (Fig. 1c): for each timestep, collect events; for each layer, execute that layer; **retain membrane**. Binary spikes; IF; XNOR-and-accumulate of weight bit A and membrane bit B inside the CIM (five phases, Fig. 2c); compare vs threshold → output spike.

Three circuit/system objects (not dual-consumer):

1. **Arbitrary operand resolution** (1-to-512×256, bitwise) for W and Vmem independently.
2. **Operand shaping** \(N_R\times N_C\) via 2-bit control cells + carry-select chaining; unused columns **PC standby** (−87% energy on inactive columns; ≤4.3× energy/op vs row-wise stacking; shape-to-shape energy variation <24% at 16 b / 32 och — **their** Fig. 7a).
3. **Hybrid stationarity (HS):** unified W/Vmem storage lets each **layer** choose WS or OS (membrane-stationary). HS-min (stationary = smaller operand) vs WS-only: **+46%** stationary operands on a two-macro map of their 6-conv SCNN (Fig. 4). Not a fork of one producer.

**Their silicon (Table I / §III; do not reuse as our PPA).** 40 nm bulk CMOS, core 1.37 mm², 16 kB CIM macro, 0.9–1.1 V, 75.5–157 MHz system / 942 MHz internal CIM phases, 6.8–17.9 mW **(macro only)**, 5.7–7.2 pJ/SOP at 8 b W / 16 b Vmem, DVS-Gesture **95.8%**. “Up to 90% energy in large-scale systems” is a **many-macro + DRAM extrapolation** (Fig. 7b–d; 16 or 18 macros; 85–99% input sparsity), not a measured full-chip energy for a transformer-OF net.

**Skip semantics — what transfers as A, what does not.**

| FlexSpIM skip / storage | Transferable as A after absorb? | Why |
|---|---|---|
| Event-driven: only process input events; IF compare emits binary spike | **A on the post-absorb spike path** (same object as ordinary event-driven SNN skip) | After \(\theta\) fold, this net’s spike GeMM is also \(\{0,1\}\times W\). This is not new vs FireFly/Gustav/Prosperity. |
| PC standby of unused CIM columns | **A as idle-lane clock/energy gating**, only if implemented on **digital SRAM/RF**, not as a CIM claim | Mechanism is “idle columns don’t pay”; the 4.3× is CIM-phase energy. |
| Layer-wise WS vs OS (who stays in the unified array) | **A as stationarity choice** (weights vs **membrane**), for F5-style “who occupies the limited RF” | Membrane ≠ PED. OS here is Vmem of IF, not I24 residual/PED last-use. |
| Unified W **and** membrane in one CIM | **Not transferable as dual-consumer** | One array, two **operand classes** of the **same IF update**. Not binary GeMM **and** a separate continuous PED tensor. |
| Dual WL / dual SA / five CIM phases / 90% system-energy extrapolation | **Do not copy as X or as PPA** | CIM identity. Extrapolation is not silicon of this net. |

**Verdict.** Complete CIM prior: copy stationarity + event-driven binary skip as **A**, then **stop**. It does not cover G1. HS “hybrid” is **WS/OS per layer**, not DATE’s dense/sparse cores, and not dual last-use.

---

## 3. SpikePool — algorithm (max-pool attention). No hardware. No dual path.

**Bibliographic identity.** Lee, Sima, Li, Stinis, Panda, *SpikePool: Event-driven Spiking Transformer with Pooling Attention*, arXiv:2510.12102v1, 14 Oct 2025. Local: `p0_txts/2510.12102.txt`.

**What it is.** Frequency-domain analysis: spiking transformers behave as **high-pass** filters vs ViT **low-pass**. SpikePool **replaces SSA** with **2D max-pooling attention** (low-pass) so the stack is a selective band-pass for event data.

Pooling attention (eqs. 7–9):

\[
Z_{\mathrm{lif}}=\mathrm{LIF}(Y_{\mathrm{out}}),\quad
Z_{\mathrm{pool}}=\mathrm{MP}(Z_{\mathrm{lif}}),\quad
Z_{\mathrm{out}}=\mathrm{Conv\text{-}BN}(Z_{\mathrm{pool}})+Y_{\mathrm{out}}.
\]

S-MLP then residual-adds again (\(H_{\mathrm{out}}=H_2+Z_{\mathrm{out}}\)). This is the Spike-driven Transformer **membrane/feature residual** (continuous \(Y_{\mathrm{out}}\) lives across SN), **not** a hardware fork of one producer into binary GeMM and PED.

**Hardware?** None. Timing is **GPU wall-clock** on A5000/A100 (Fig. 5): training −19.7%/−42.5% vs SDT on N-Caltech101; encoder train/infer −31.9%/−32.8% vs SpikingViT on Gen1. No FPGA, no ASIC, no skip unit, no dual core.

**Dual path?** Algorithmic residual ADD around pooling and MLP. Classification (CIFAR10-DVS, N-Caltech101) and event **object detection** (PAF / Gen1 / 1Mpx) with a YOLOX head. **Not** event-camera 2D optical flow hardware; **not** dual-consumer PED.

**Copy as A, then stop as X.** A = “SSA can be replaced by max-pool + Conv-BN + residual” and the high-pass/low-pass story. That is a **model** prior. It does not supply a hybrid core, a last-use contract, or a spike-path GeMM engine.

---

## 4. Fang CICC 2024 — 3-D AC-only, unstructured spike sparsity, **spiking-only** (against SNN+CNN cores)

**Status: full CICC PDF unread.** Mapping uses (i) IEEE / HKUST SPD abstract (DOI 10.1109/CICC60959.2024.10529019; CICC 2024, Denver, 21–24 Apr 2024; **pp. 1–2**), (ii) Westlake CenBRAIN news restatement of the same CICC paper, (iii) existence of a JSSC 2025 journal extension (Fang et al., *An Energy-Efficient Unstructured Sparsity-Aware Deep SNN Accelerator With 3-D Computation Array*, IEEE JSSC 60(3):977–989, DOI 10.1109/JSSC.2024.3507095) whose **abstract** restates the same three features. **Do not treat JSSC body numbers as read.**

**Authors (IEEE).** Chaoming Fang, Ziyang Shen, Shiqi Zhao, Chuanqing Wang (Westlake); Fengshi Tian (HKUST); Jie Yang (Westlake); Mohamad Sawan. Title: *A 0.078 pJ/SOP Unstructured Sparsity-Aware Spiking Attention/Convolution Processor with 3D Compute Array*.

**IEEE / HKUST abstract (quoted objects, not extra silicon).**

- SNNs use **accumulation (AC)** features; efficiency collapses at low sparsity.
- **Previous work [2] uses heterogeneous SNN and CNN cores** but “incurs increased area and power consumption due to costly **MAC** array implementation.”
- Therefore they want a **spiking-only** accelerator across all sparsity levels.
- Three challenges: (1) redundant W / psum access **across timesteps**; (2) unstructured spike sparsity — fetching irregular NZ spikes and weights **one-by-one** kills throughput; (3) one-size-fits-all scheduler cannot serve distinct operators.

**Author-lab restatement of the CICC chip (Westlake news; matches JSSC abstract).** 3-D array to reuse weights **across timesteps** and cut EMA; **parallel non-zero fetcher** with a **priority-encoder array** to search/fetch multiple NZ spike–weight pairs; **unified / multimode scheduler** for operators in spiking transformers. Fabricated **40 nm**. Headline numbers **in the title and abstracts**: **0.078 pJ/SOP**, ImageNet **77.6%** with a spiking transformer. Westlake news also prints a chip micrograph caption “TSMC 40nm CMOS.” **Area, frequency, power, array geometry, and scheduler modes beyond the names SCONV / QKV / SSA are not in the opened CICC abstract — do not invent them.** JSSC abstract (ADS/IEEE, unread PDF) names the scheduler as configurable for **SCONV, spiking Q/K/V generation, and SSA**.

**Class A for binary AC GeMM.** After AT-LIF absorb, this net’s spike path **is** AC (select-add). Fang is a complete **spiking-only AC + unstructured NZ fetch + timestep-parallel 3-D array** prior on **that** path. Copy that object as A.

**Explicitly not DATE hybrid, not G1.** Fang **rejects** heterogeneous SNN+CNN (MAC) cores — the opposite of Aliyev’s dense MAC input core. Consumers are binary spikes / AC. No continuous PED last-use of the same producer. Classification (ImageNet spiking transformer), not DSEC OF.

**Copy as A, then stop as X.** Do not retitle the letter as “3-D array” or “0.078 pJ/SOP.” Do not fill CICC 2-page figures from the JSSC body until that PDF is opened.

---

## 5. COMPASS MICRO 2024 — CIM, full text missing locally

**Not in `p0_txts`.** Card remains `unresolved_no_fulltext`.

**Publisher abstract only** (Wang, Liu, Yang, Huang, Li, Jiang; MICRO 2024 pp. 1090–1106; DOI 10.1109/MICRO61859.2024.00083):

- SRAM-based **CIM** for SNNs; dominant op named **spike-wise Accumulate-Compare**.
- Speculation on irregular sparsity of **input spikes (explicit)** and **output spikes (implicit)**; CIM modified for dynamic spike-pattern generation; adaptive dataflow with temporal spike representation.
- Abstract metrics: **26.7×** end-to-end speedup vs “recent SNN accelerators hardware implementation,” **up to 386.7×** less energy per inference. **Process node, area, frequency, and speculation-mispredict cost are not in the opened abstract — do not invent them.**

**Rule.** CIM → copy as **A only at abstract grain** (binary AC + speculation on spike presence), then **STOP as X**. Without the PDF, speculation window, mispredict datapath, and whether any continuous residual exists are **unknown**. Cannot claim it covers or fails G1 beyond: the named object is CIM Accumulate-Compare on spikes, not dual last-use of binary GeMM + PED.

---

## 6. ESTU TCAS-II 2025 — same-journal collision; **not** dual-consumer of the same producer

Source of record: `adversarial/ADV_ESTU_same_journal.md` (AAM opened; DOI 10.1109/TCSII.2025.3626209; vol. 72 no. 12 pp. 2027–2031). ADV still frames the letter identity as continuous θg; **this freeze does not.** Under the locked absorb identity the collision object is unchanged.

**ESTU’s object, one line (from the opened letter, as ADV recorded):**

**binary spike tensor → one accumulate consumer → skip inactive groups-of-4 → classification % + mW on a 5k-LUT FPGA.**

| ESTU mechanism (ADV) | Dual-consumer of same producer? |
|---|---|
| Binary SSA \(QK^\top V/\mathrm{scale}\) on spikes | No. One SSA consumer. |
| LIF \(s(t)=v(t)>\theta\) then reset; integer ops **then binarize** into spike mem | No. Continuous membrane is **not** a second live consumer. |
| Table I overlay: `Dense(spike)`, `Dense(int)`, `Sum(spike,*)`, `Mul(spike,spike)` (16 AND + 16-input popcount), `Mul(spike,int)` | **Typed destinations** (spike mem **or** int mem **or** through LIF). Muxed overlay, not a fork. |
| Activity stack, groups of 4; software sparsity 0.95 → exploited 0.82 | Binary-group skip. After absorb this **is** legal A on the spike path — and it is **already in this journal**. |
| NinaPro / EEG accuracy; iCE40UP5K 3.76 mW / 0.23 mW / 4.28 μJ | Classification + mW, not DSEC AEE, not PED service. |

**Confirm: ESTU is NOT dual-consumer of the same producer.** Integer mem is a **mode** of the microcode datapath, not PED/I24 last-use of a compiled T10 source that must stay live for a continuous residual after the binary gate would have retired. Slot retirement in ESTU is operator-done × group sparsity, not `gate ∧ PED`.

**Same-journal implication (unchanged).** A five-page letter readable as “FPGA spiking transformer with skip + overlay + mW + class-%” **collides** with TCAS-II 72(12). Copy ESTU as **A** (binary SSA overlay + group skip) on the post-absorb spike path, then the letter must measure a **different object**: dual last-use of binary GeMM **and** continuous PED on DSEC valid825, same-port net service. DATE dense/sparse cores, FireFly-T dual-**engine**, and “we also skip” do **not** escape that script (ADV table).

ADV’s checklist item “continuous θg is the source value” is **retired** by `IDENTITY_ATLIF.md`. The surviving non-collision objects are: (i) residual/PED as a **separate** continuous tensor still live after binary last-use, (ii) DSEC AEE gates, (iii) same-port dual-completion service. ESTU still has none of those.

---

## 7. Verdict table

Legend: **A** = complete prior to copy on the named object, then stop as X. **Control** = reviewer will map a slogan onto this paper. **G1** = typed last-use of **same-producer** binary GeMM + continuous PED. **CIM** papers are A-then-stop, never title.

| Paper | Venue | Primary depth | Binary after first spike layer? | Hybrid of what? | Same-producer dual last-use (G1)? | Copy as A | Illegal as title / stop as X |
|---|---|---|---|---|---|---|---|
| Aliyev et al. | DATE 2025 / arXiv 2411.15409 | **Full arXiv HTML** | After LIF of input layer: yes. Input layer: **no** (direct-coded dense). | **Layer-split:** dense MAC **input** / sparse event conv **rest** | **No.** Different layers, different tensors. | Hybrid **overlay control** (dense-input vs sparse-rest); QAT-induced extra spike sparsity | “We have dense+sparse cores.” Not Prosperity-after-absorb. Not G1. |
| FlexSpIM | ISCAS 2025 | **Local author full text** | Yes (IF events) | **WS/OS per layer** in unified W/Vmem CIM | **No.** W and Vmem of the same IF update | Event-driven binary skip; layer-wise stationarity **of membrane vs weights**; idle-column gating **if reimplemented digitally** | **CIM.** Dual-WL macro, 90% many-macro energy, DVS-Gesture 95.8% as our story |
| SpikePool | arXiv 2510.12102 | **Local full text** | SSA replaced by max-pool on LIF spikes | None in hardware | **No hardware.** Residual ADD is SDT-style feature skip | Max-pool attention **algorithm**; GPU-time reduction is their number | Any FPGA/ASIC dual-path claim |
| Fang et al. | CICC 2024 (2 pp.) | **IEEE+HKUST abstract; PDF unread** | Yes; **AC-only**, unstructured NZ spike fetch | **Anti-hybrid:** spiking-only vs hetero SNN+CNN MAC cores | **No** (from abstract) | **Binary AC GeMM** + timestep-parallel 3-D reuse + parallel NZ fetch + multimode SCONV/QKV/SSA scheduler | Extra pJ/SOP, array size, clocks **not in opened abstract**. Do not use as dual-core X |
| COMPASS | MICRO 2024 | **Abstract only; not in p0_txts** | Accumulate-Compare on spikes (abs) | CIM + spike speculation | **Unknown / not claimed** | Abstract-grain: CIM AC + in/out spike speculation | **CIM. Full text missing.** 26.7× / 386.7× are **their** abstract, not ours |
| ESTU | TCAS-II 2025 72(12) | AAM (ADV) | Yes. Integer path **LIF-binarized** | Microcode **overlay**, typed spike/int mem | **No.** Muxed destination, one accumulate consumer | Binary SSA + group-of-4 skip + tiny-FPGA overlay | Same-journal **binary SSA skip + class-% + mW**. Dual-engine ≠ dual-consumer |

---

## 8. Mapping onto this net (after absorb)

Post-absorb spike GeMM **already has** complete A: Prosperity / Gustav / FireFly-S **and**, from this round, Fang’s AC+NZ-fetch (abstract), ESTU’s binary SSA skip (same journal), DATE’s sparse-core event conv, FlexSpIM’s event-driven IF. Copying any of those onto DSEC OF is **prior-copy**, not X.

Continuous residual / PED remains a **separate tensor**. None of the six papers:

- forks one producer into binary GeMM **and** PED,
- retires a slot on `gate ∧ PED`,
- measures DSEC valid825 AEE,
- or moves a same-port long-backpressure class analogous to 8088.

DATE is the strongest **overlay control**: two core types because **layers** differ (dense pixels vs sparse spikes). G1 is two **consumers** because **one** compiled source has two last-uses. Those are different graphs.

Fang is the strongest **anti-hybrid control**: a reviewer who knows CICC 2024 will say heterogeneous MAC+SNN cores were already considered and **rejected** in favor of spiking-only AC. A letter that reintroduces a dense MAC core must say it is **not** Fang’s [2] and **not** DATE’s input layer — it is the PED/I24 path of a **different tensor**.

FlexSpIM’s transferable skip is ordinary event-driven binary skip plus (digitally reimplemented) stationarity. Unified W/membrane is **not** dual-consumer.

SpikePool does not enter the hardware A stack.

ESTU remains the **desk-reject script** for any title that still reads as FPGA spikeformer + skip + mW + accuracy.

---

## 9. Open / unread (do not close by slogan)

- Fang **CICC 2-page PDF** and **JSSC 2025 body**: unread. AC-only and anti-hetero-core claims are from the abstract; {0,1} GeMM algebra and any θ-folding were **not** line-checked in a PDF (same uncertainty as `round2_absorb/web/DEEP_RESEARCH2.md`).
- COMPASS **PDF**: unread; not in `p0_txts`.
- DATE **IEEE PDF**: unread; arXiv HTML used as the methods primary.
- ESTU GitHub tree: still search-incomplete (ADV).

No silicon number in this note was synthesized. If a figure is not in a row of §0, it is not used.
