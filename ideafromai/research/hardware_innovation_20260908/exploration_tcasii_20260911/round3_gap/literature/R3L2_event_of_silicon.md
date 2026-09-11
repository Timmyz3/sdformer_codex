# R3L2 — Event-OF silicon vs dual last-use after absorb

Freeze: 2026-09-11. Round-3 gap note. Identity: `IDENTITY_ATLIF.md` / round-3 `SCOPE.md`. This is a prior map, not a paper claim, not Stage-B schedule evidence, and **not** a first-event-OF-HW pitch.

**Locked identity.** AT-LIF \(o=\theta\cdot H(m-\theta)\in\{0,\theta\}\). Inference: layer-shared \(\theta\) absorbed \(W\leftarrow\theta W\). Spike path after absorb is binary GeMM. Residual / PED / I24 is a **different continuous tensor**. Task: event-camera 2D optical flow, SDformerFlow family, DSEC valid825 AEE.

**Question.** Does existing **silicon / FPGA** already implement (i) dual last-use of binary GeMM **and** continuous PED after absorb, (ii) live full-domain BN \(10\times C\times H\times W\) as a completion barrier, or (iii) optical-flow spatial-locality skip as a letter? Also: is SDformerFlow’s “disable tracking of running states” live batch stats or FrozenBN?

**Two hard bans.**
1. Do not write “first event optical-flow hardware.” SENECA FireNet OF, TrueNorth OF, FPGA plane-fit OF, and (title/abstract-level) ASNA-Flow already exist in this tree.
2. Do not treat an IEEE abstract as a full-text mechanism. PPA on an abstract is **their** number, never local same-port %.

---

## 0. Depth of each source (do not flatten)

| ID | Printed title | Local text | Depth this sheet may use |
|---|---|---|---|
| **SENECA** | Event-based Optical Flow on Neuromorphic Processor: ANN vs. SNN Comparison based on Activation Sparsification (Xu et al., Neural Networks 2025; arXiv:2407.20421) | `p0_txts/2407.20421.txt` | **Full local.** FireNet on SENECA. Already event-OF on a neuromorphic processor. |
| **SDformerFlow** | SDformerFlow: Spatiotemporal swin spikeformer for event-based optical flow estimation (Tian & Andrade-Cetto, ICPR 2024; arXiv:2409.04082) | `p0_txts/2409.04082.txt` | **Full local.** GPU algorithm + Horowitz energy **model**. Not silicon. |
| **EventShiftFlow** | EventShiftFlow: Towards Hardware-efficient FPGA-based Flow Estimation (Alonso Bizzi, Cladera, Taylor; arXiv:2605.28312) | `p0_txts/2605.28312.txt` | **Full local.** Artix-7 1-bit occupancy-grid velocity. Not SNN, not DSEC dense AEE. |
| **SNE** | SNE: an Energy-Proportional Digital Accelerator for Sparse Event-Based Convolutions (Di Mauro et al.; arXiv:2204.10687, DATE 2022) | `p0_txts/2204.10687.txt` | **Full local.** 22 nm event-driven eCNN. **IBM-DVS-Gesture classification, not optical flow.** |
| **ASNA-Flow** | ASNA-Flow: An Efficient Asynchronous Neuromorphic Accelerator for Real-Time Event-Based Optical Flow | **no PDF** (`idea_cards/unresolved_ASNA_Flow.md`) | **`unresolved_no_fulltext`.** IEEE abstract only: DOI `10.1109/TVLSI.2025.3600953`, TVLSI 33(12):3409–3422, 26 Aug 2025. |
| **ERAFT FPGA** | An FPGA-based Real-Time Optical Flow Accelerator for Recurrent All-Pairs Field Transforms | **no PDF** (`idea_cards/unresolved_ERAFT_FPGA.md`) | **`unresolved_no_fulltext`.** IEEE abstract only: DOI `10.1109/ISCAS56072.2025.11043529`, ISCAS 2025. |
| **Aung plane-fit FPGA** | Event-based Plane-fitting Optical Flow for Dynamic Vision Sensors in FPGA (ISCAS 2018) | **no PDF** (`idea_cards/unresolved_FPGA_plane_fitting_event_optical_flow.md`) | **`unresolved_no_fulltext`.** Mechanism quotes below are **EventShiftFlow’s related-work paraphrase**, not Aung’s body. |
| **Haessig TrueNorth OF** | Spiking optical flow for event-based sensors using IBM’s TrueNorth neurosynaptic system (TBioCAS 2018) | **no PDF** | **`unresolved_no_fulltext`.** Cited by EventShiftFlow [13] and Spike-FlowNet `2003.06696` [13]. |

Round-1 `literature/L6_event_of.md` already scored SENECA as the hardware paper in a six-paper GPU/HW mix. This sheet adds **silicon that L6 did not own** (SNE, EventShiftFlow, ASNA/ERAFT abstracts) and answers the **post-absorb dual-consumer** question that L6 asked as producer-support.

---

## 1. Direct answers (four questions)

### Q1. Dual last-use of binary GeMM + continuous PED after absorb?

**No inspected silicon implements that object.**

After absorb this net has **two last-use maps**: (A) binary spike GeMM (Prosperity/Gustav/FireFly legal A), (B) continuous residual / PED / I24. Gustav CPTB is one death time. Dual last-use is G1’s remaining hole (`round2_absorb/DECISION_LOG.md`: G1 revise-not-title until 8088 wait-class is split).

What the chips actually last-use:

| Chip | What dies | Second consumer? |
|---|---|---|
| SENECA FireNet | One membrane per pixel group; ANN 3×3 can release after last RF event; LIF **cannot** | No PED/I24. Recurrent hidden state is a **same-family** tensor, not a deformed \(1\times1\) continuous skip. |
| SNE | FIRE dumps membrane to output FIFO; UPDATE keeps state; C-XBAR wait-all slaves | FIRE/UPDATE are two **uses of one LIF membrane**, not binary GeMM vs PED. |
| EventShiftFlow | Occupancy-grid bits; scorer last-use = bin-complete | No neural residual. |
| ASNA-Flow | **Unknown.** Abstract names event-driven SNN + spatial-locality sparse compute. Dual last-use is **not** in the abstract. Do not invent it. |

SENECA grouping=4 is **one** load/store of neuron state for up to four ac./sp. at the **same pixel**. That is same-consumer reuse, not two typed free events on a shared SRAM port.

### Q2. Live full-domain BN \(10\times C\times H\times W\) as a completion barrier?

**No inspected silicon uses that barrier.** Native projection BN on this freeze (`PROBLEM.md`) uses **actual batch statistics** over `10×96×120×160`. A site is live until \(\mu/\sigma\) over the full domain is done. Active-site BN is a different student.

- SENECA: **no BN** in `2407.20421` (grep empty). FireNet 32-ch 3×3; resolution capped by on-chip **neuron-state** SRAM (56² / 120²), not a BN population.
- SDformerFlow (algorithm, not HW): eval **disables tracking of running states** → **live batch stats**, not FrozenBN. Energy model then **ignores** BN (~0.01%). Opposite of a completion barrier.
- SNE / EventShiftFlow: no projection BN.
- ASNA-Flow / ERAFT FPGA: BN handling **unknown** (paywalled). Round-2 `DEEP_RESEARCH2.md` already logged this as an uncertainty. Stay unknown.

### Q3. Optical-flow spatial-locality skip — already ASNA-Flow title-level?

**Yes at title/abstract level. G2 already stopped as a title.** Do not reopen.

IEEE abstract (not body) of ASNA-Flow:

> “novel exploitation of optical flow’s spatial locality characteristics to enable efficient sparse computing.”

> “This work establishes the first dedicated neuromorphic computing solution that simultaneously addresses the temporal sparsity, event-driven processing, and energy constraints inherent in optical flow estimation tasks.”

Round-2 `ADV_G2.md` / `DECISION_LOG.md`: **G2 stop as title**; keep only as Prosperity/ExSpike A-control. Spatial locality of OF support is **their** sold object. Local post-absorb binary GeMM still copies Prosperity/FireFly/ExSpike APEC; translating NRV by previous flow is MCP/oracle-adjacent, not a new reused object.

**Do not upgrade the abstract into a datapath.** How they skip (tile, event, PE, SRAM) is unresolved without PDF.

### Q4. SDformerFlow BN: live batch stats or FrozenBN?

**Live batch stats. Aligns with native proj BN, not FrozenBN.**

Pinned (`2409.04082`, Experiments §IV-A):

> “During the evaluation test, we disable the tracking of running states for batch normalization layers.”

PyTorch name of that switch is `track_running_stats`. `False` ⇒ BN **never** keeps a running \(\mu/\sigma\) and **always** uses the current minibatch. FrozenBN is the other contract: freeze and **use** running stats at eval. They are not the same.

They also write that BN is “negligible” for a Horowitz energy model (~0.01%). That sentence is an **algorithm-side** license to ignore BN traffic. Local `PROBLEM.md` forbids it: live full-domain stats are a wait/storage barrier; delayed-V until BN completes is integer 0-diff with `arithmetic_saving=0` (round-3 `SCOPE.md`).

SPE deformed shortcut is **not** dual last-use. Eq. 13–14:

\[
z_{\mathrm{res}}=\mathrm{Conv}_{\mathrm{deformed}}(I),\qquad
z=\mathrm{BN}(\mathrm{Conv}(\mathrm{SN}(I)))+z_{\mathrm{res}}.
\]

`Conv_deformed` is \(1\times1\), stride 2, on the residual to match embedding shape. MS shortcut (Fig. 4) adds **before** SN so spike tensors stay binary. That is SDT/MS **A** (`R2L5`). PED here is a **shape-matching skip**, still one residual stream, not a second last-use of a binary GeMM product.

---

## 2. Quote-pinned notes (full local first)

### 2.1 SENECA FireNet OF (`2407.20421`) — already event-OF HW

**What it is.** Lightweight FireNet (Hagenaars 2021), ANN FATReLU vs LIF, mapped to SENECA, MVSEC-style event OF. Hardware-in-loop time/energy on GF-22 nm FDX (Xcelium+JOULES). Resolution 56² and 120² because **neuron states** must fit SRAM.

> “SENECA has an event-driven processing mechanism that can exploit the sparsity in ANN activations and SNN spikes to accelerate the inference of both types of neural networks.” (Abstract)

> “The ANN and the SNN for comparison have similar low activation/spike density (∼5%) thanks to our novel sparsification-aware training.” … “SNN’s higher efficiency attributes to its lower pixel-wise spike density (43.5% vs. 66.5%) that requires fewer memory access operations for neuron states.” (Abstract)

> “SENECA reuses neuron states by a mechanism called ac./sp. grouping. … We set the group size to four. Firstly, the neuron states are loaded … four ac./sp. are integrated … stored back. … If there are fewer than four … dummy zero ac./sp. fill(s) the group.” (§3.2)

> “the resolution is constrained by the size of data memory due to the need to store the neuron states.” (256 KB → 56²; 2 MB → 120²) (§3.2)

> “This is the first work showing an SNN’s advantageous energy and time efficiency over an ANN in a regression task of event-based vision by a fair comparison supported by experimental measurements on a neuromorphic processor.” (§1) — **their** first (ANN vs SNN on SENECA). Do not recycle as first event-OF HW.

Measured numbers in **their** loop (do not convert to same-port %): SNN 44.9 ms / 927 μJ vs ANN 71.8 ms / 1233 μJ on selected 56² frames (Abstract / L6 already tabulated). 120² in Table 3 of the paper. I/O excluded.

**Map after absorb.**

- A: event-driven skip of zero ac./sp.; pixel-clustered membrane traffic; trainable thresholds + \(L_s\). Copy as sparsity-training **A** if a student trains support (G4 already **stop as title**).
- Not dual last-use: grouping amortizes **one** membrane. Depth-first fire-when-RF-complete is SENECA ANN’s 3-row release. Local T10 is **noncausal**; LIF “keep all membranes” is the closer, expensive contract (L6 §5).
- Not PED: FireNet has no deformed \(1\times1\) continuous skip and no I24 integer residual consumer.
- Not full-domain BN barrier.
- Binary SNN spikes vs local absorbed \(\{0,1\}\) GeMM: the spike path **is** binary after absorb, so Prosperity/Gustav apply **on that path**. SENECA never runs a second continuous PED MAC on the same source.

**Verdict:** **A** (event-OF neuromorphic mapping + sparsity training). **Stop** “we also ran OF on a neuromorphic chip” and **stop** grouping=4 as X.

### 2.2 SDformerFlow (`2409.04082`) — task family, not silicon

**What it is.** STTFlowNet (ANN swin) + SDformerFlow (fully spiking swin spikeformer) for **dense** event OF on **DSEC** (left camera, 640×480 GT 10 Hz) and MVSEC. First **spikeformer for dense OF** (their claim, algorithm). GPU train. Energy = FLOPS × rate × \(T\) × Horowitz \(E_{AC}\).

MS vs SEW (Fig. 4, §III):

> “in SDformerFlow, we opt for using membrane-potential shortcuts (MS).” … “with MS shortcuts, residuals are applied before the spikes to preserve the spike-driven property.”

SPE (Eq. 13–14): deformed \(1\times1\) stride-2 shortcut on patch embedding. Ablation: SPE “notably improved the performance.”

BN eval: disable tracking of running states (§IV-A, quote in Q4).

BN in energy:

> “ignoring the negligible contribution of batch normalization layers (around 0.01%).” (§IV-D)

**Map after absorb.**

- A: MS residual algebra (copy SDT; `R2L5`). Deformed PED \(1\times1\) is the **student’s existing skip**, not a hardware last-use.
- B (this net, not their paper): live proj BN over `10×96×120×160` is a **barrier they explicitly ignore**; continuous PED/I24 outlives the binary GeMM.
- X is not “we also use MS + SPE.” That is the student.

**Verdict:** **A** for MS/SPE/DSEC task. **Stop** Horowitz mJ as PPA. BN sentence is a **negative** prior: they treat BN as free; we cannot.

### 2.3 EventShiftFlow (`2605.28312`) — FPGA occupancy grid, not SNN

**What it is.** Stream events → time bins → **1-bit** x/y occupancy → shift-register grid \(N_x\times L\) → discrete velocity hypotheses scored by popcount / cross-multiply. No frame recon, no FP, no DSP, no dividers. Sparse **quantized** velocity, not dense sub-pixel DSEC flow. Artix-7 xc7a100t prototype: **x-axis only**, hypotheses \(\{-4,-2,0,2,4\}\), sequential scorer.

Pinned hardware claims (theirs, not ours):

> “It requires no frame reconstruction, no floating-point arithmetic, and no iterative optimization.” (Abstract)

> “the proposed datapath requires less than 2 kB of storage, and implement a single-axis prototype on a low-cost Xilinx Artix-7.” (Abstract)

> “EDFLOW [11] uses 390 Block RAMs (855 kB) and 669 DSP48E units … Aung et al. [12] use 138 Block RAMs and 16 DSPs. Our design requires zero Block RAMs … and zero DSP units.” (§V-F) — **their** comparison.

Post-impl prototype (Table V): 100 MHz, 0.142 W on-chip estimate, 13,326 LUT / 5,517 FF, 0 BRAM, 0 DSP, UART-to-motion **including** I/O. Scoring 2400 cycles / 24 μs. **Do not quote as local PPA.**

**Related-work pins this sheet is allowed to use (paraphrase of Aung / Haessig, not those PDFs):**

> “Aung et al. [12] implemented Benosman’s plane-fitting algorithm, achieving sub-microsecond latency with a fully pipelined design capable of 100 million plane fits per second, but requiring division and square-root pipelines for the least-squares computation.” (§II-B)

> “Haessig et al. [13] implemented a spiking optical flow estimator on IBM’s TrueNorth neuromorphic chip, achieving low power consumption but limited to sparse flow estimates.” (§II-B)

**Map after absorb.** Occupancy 1-bit is **not** AT-LIF after absorb. Hypothesis popcount is **not** Prosperity prefix reuse. No BN, no PED, no Transformer. Opposite of dense DSEC AEE student.

**Verdict:** **A** as “event-flow FPGA exists and is integer/occupancy.” **Stop** as a dual-consumer or SDformerFlow mapping prior.

### 2.4 SNE (`2204.10687`) — event-conv silicon, **not OF**

**What it is.** Digital sparse neural engine, 4-bit eCNN, linear-leak LIF, explicit event tuple, energy proportional to input events. Proof task: **IBM DVS-Gesture 92.8%**. Peak 51.2 GSOP/s, 4.5 TSOP/s/W, 0.221 pJ/SOP, 80–261 μJ/inf at 1.2–4.9% activity. **Do not quote as local PPA. Do not retitle as optical-flow silicon.**

> “our accelerator performs a number of operations proportional to the number of events contained into the input data stream.” (Abstract)

> “As a proof of concept, we show that SNE consumes 0.221 pJ/SOP and achieves 92.8% accuracy on a classification task performed on the IBM DVS-Gesture data set.” (§I)

Datapath that is **A** for event addressing (already in `L7_cmvm_lut_temporal.md` §2.2): C-XBAR ready-valid **wait-all slaves**; dual-buffer state + TLU skip idle ticks; filter buffer ≤256; FIRE vs UPDATE FIFOs; on-slice all-layers vs tile-through-external-memory.

**Map after absorb.** Wait-all slaves is the right **picture** of dual consumers of one source word, but the slaves are **neuron slices of one LIF**, not `{binary GeMM, continuous PED}`. Gesture classification has no DSEC BN domain.

**Verdict:** **A** (energy-proportional event conv + broadcast wait-all). **Stop** “SNE already did event OF” (it did not). **Stop** pJ/SOP as same-port %.

### 2.5 ASNA-Flow — IEEE abstract only (`unresolved_no_fulltext`)

**Title (IEEE):** *ASNA-Flow: An Efficient Asynchronous Neuromorphic Accelerator for Real-Time Event-Based Optical Flow.* TVLSI vol. 33, no. 12, Dec 2025, pp. 3409–3422. DOI 10.1109/TVLSI.2025.3600953. Authors on the IEEE record include Jinghai Wang, Jilong Luo.

**Abstract-level facts (not mechanisms):**

- Event-driven cameras + SNNs for optical flow at the edge.
- Three named “innovations” in the abstract: (1) hardware-aware algorithm optimization, (2) data-pattern analysis, (3) **optical flow’s spatial locality** → sparse computing.
- TSMC 28 nm; **104 FPS**; **7.9 mW**; **0.3 pJ/SOP**. These are **their** abstract numbers. **Never** convert to same-port % or local AEE.
- Abstract also claims “first dedicated neuromorphic computing solution that simultaneously addresses the temporal sparsity, event-driven processing, and energy constraints inherent in optical flow estimation tasks.” That is **their** “first.” Relative to SENECA (2025, FireNet on a **programmable** neuromorphic processor) and TrueNorth OF, the sentence is a **dedicated-accelerator** claim, not a license for this letter to say first. **Do not recycle.**

**What the abstract does *not* say (so this sheet does not say):** BN live vs frozen; dual last-use; absorb of \(\theta\); PED/I24; DSEC valid825; Prosperity product sparsity; T10; which SNN (FireNet vs transformer).

**Verdict:** **A at title/abstract** for “dedicated event-OF SNN ASIC + spatial-locality sparse compute.” **Stop G2 as title** (already stopped). **Keep unresolved_no_fulltext** for any circuit/BN/last-use claim. If a PDF appears later, re-read; until then **do not write datapath X from this paper.**

### 2.6 ERAFT FPGA — IEEE abstract only (`unresolved_no_fulltext`)

**Title (IEEE):** *An FPGA-based Real-Time Optical Flow Accelerator for Recurrent All-Pairs Field Transforms.* ISCAS 2025, London, 25–28 May 2025. DOI 10.1109/ISCAS56072.2025.11043529.

**Abstract-level facts:**

> “we present ERAFT: a novel lightweight deep neural network architecture based on the Recurrent All-Pairs Field Transforms (RAFT) algorithm, which is more suitable for hardware deployment.”

> “The hardware accelerator was evaluated on the Xilinx VCK190 evaluation board.” … “high accuracy on the **Middlebury** dataset, with speeds of up to **86 frames/s** for **640 × 480** pixel images.”

**Confirm:** this is **frame RAFT on VCK190 / Middlebury**, **not** Gehrig E-RAFT 3DV 2021 (event voxels, DSEC), **not** an event camera, **not** an SNN. The acronym collision with E-RAFT is the trap. Round-1 `W1_2025_2026_venues.md` P31 already marked CONTROL.

**Verdict:** **A** as “FPGA optical-flow accelerator exists (ANN RAFT).” **Stop** as event-SNN-Transformer prior. **Stop** using it as G2 geometry. Full datapath remains unresolved without PDF; the abstract is already enough to **exclude** it from the dual-consumer island.

### 2.7 Plane-fit FPGA / TrueNorth OF — secondary only

Keep as **existence proofs** that event-OF hardware predates this letter. Do not invent PPA.

| Work | How this sheet knows it | What it is |
|---|---|---|
| Aung et al., ISCAS 2018 | EventShiftFlow §II-B paraphrase; Spike-FlowNet refs | FPGA Benosman plane-fit; division + sqrt; 100 M plane-fits/s in the paraphrase |
| Haessig et al., TBioCAS 2018 | EventShiftFlow §II-B; Spike-FlowNet §1 | TrueNorth spiking OF; sparse estimates; rotating spirals/pipes (Spike-FlowNet’s words) |

`unresolved_no_fulltext` on both PDFs. Enough to kill “first event-OF HW.” Not enough to copy a last-use protocol.

---

## 3. Verdict table (A / B / X / stop)

Legend. **A** = copy whole onto the matching object, or name as complete relative prior. **B** = hole that exists only under **this** freeze (absorb + dual tensor + live BN + DSEC SDformerFlow family). **X** = increment that is not a reskin of A. **stop** = do not title, do not reopen.

| Prior | Object they actually ran | A (copy / name) | B they do **not** cover | X from this prior? | Disposition |
|---|---|---|---|---|---|
| SENECA FireNet | Event OF, LIF/FATReLU FireNet, grouping=4, 56/120, GF-22 | Event-driven skip; pixel vs neuron density; threshold+\(L_s\) training | Dual last-use GeMM+PED; T10 noncausal; live BN \(10\times96\times120\times160\); C96 r1 | No | **A.** **stop** first-HW / grouping-as-X |
| SDformerFlow | GPU spikeformer OF, MS + SPE \(1\times1\), DSEC | MS residual; SPE deformed skip; DSEC task | Silicon; BN as barrier (they ignore it); dual last-use | No | **A** for task/MS. **stop** Horowitz mJ |
| EventShiftFlow | FPGA 1-bit occupancy, Artix-7 x-axis | Integer event-flow FPGA exists | SNN, absorb, PED, BN, dense DSEC | No | **A** existence. **stop** as mapping |
| SNE | 22 nm eCNN, DVS-Gesture | Event-proportional SOP; C-XBAR wait-all | Optical flow; PED; live BN | No | **A** wait-all picture. **stop** OF rename |
| ASNA-Flow abs. | Dedicated event-OF SNN ASIC (claimed) | Spatial-locality sparse compute **as a sold title**; 28 nm OF chip exists | Dual last-use, BN, absorb — **unknown** | **No until PDF** | **stop G2 title.** unresolved body |
| ERAFT FPGA abs. | Frame RAFT, VCK190, Middlebury | FPGA OF accelerator (ANN) | Event, SNN, DSEC | No | **stop** as event-SNN prior |
| Aung / TrueNorth | Plane-fit FPGA / TrueNorth OF | Event-OF HW exists | Dual last-use / BN / transformer | No | **stop** first-HW. unresolved body |
| Prosperity/Gustav/FireFly | Binary GeMM last-use **one** consumer | **Must copy** on post-absorb spike path | Second death time of PED/I24 | Dual last-use is **G1**, not these chips | A on spike path; not this sheet’s X |

**Island-level X (not granted by these papers).** If anything remains after A is copied, it is **typed last-use of `{binary GeMM after absorb, continuous PED/I24}` under a live full-domain BN completion barrier**, measured as same-port / same-state / same-backpressure net service vs conservative `max(GeMM, residual, BN)` last-use. That is G1’s revise-not-title, **not** a new event-OF silicon claim. These papers do not close it and do not make it a title by themselves.

---

## 4. Spatial locality (G2) — closed as title, open only as A-control

Round-2 already:

> G2 OF-geometry on binary support — **stop as title**. Keep as Prosperity / ExSpike A-control. (`DECISION_LOG.md`)

This round **confirms** with the ASNA-Flow abstract quote in Q3. Additional spatial-structure priors that are **A**, not X:

- SENECA: hardware time scales with **pixels that have ≥1 ac/sp**, not nnz alone; group=4 dummy fills. Spatial **distribution**, still one consumer.
- EventShiftFlow: skip inactive occupancy bits; that **is** the algorithm.
- EV-FlowNet 20–30% event-pixel **eval** mask (L6): not production skip.

Legal leftover on the binary path after absorb: finish Prosperity Dispatcher / ExSpike APEC / FireFly 3×3 AND on DSEC edges. Illegal: current-frame final flow as skip oracle (L6 hard ban). Paid previous-flow warp already lost once as untrained exact-Δ (**1.529×** more work, L6/ADV_G2).

---

## 5. BN completion barrier — still hygiene, not a silicon prior

| Source | BN contract | Barrier? |
|---|---|---|
| This freeze `PROBLEM.md` | Actual batch stats, full `10×96×120×160` | **Yes.** Local-window replay with free \(\mu/\sigma\) undercharges. |
| SDformerFlow eval | `disable the tracking of running states` = **live batch stats** | They **have** live stats in software and then **ignore** BN in the energy model. |
| FrozenBN (not this paper) | Use running \(\mu/\sigma\), no batch wait | Would **remove** the barrier. Native student is not this. |
| SENECA / SNE / EventShiftFlow | No proj BN | N/A |
| ASNA-Flow body | unknown | Stay unknown |

Round-2 G3 = **hygiene** (must bookkeep). Round-3 observation: delayed-V until native BN completes is 0-diff, `arithmetic_saving=0`. Silicon in this set does not implement the barrier, so “BN-aware last-use” is not copying a chip; it is finishing the freeze’s accounting.

---

## 6. What a TCAS-II sentence may **not** say

Forbidden (relative priors already take them):

- “First event optical-flow hardware / first neuromorphic OF accelerator.”
- ASNA-Flow 104 FPS / 7.9 mW / 0.3 pJ/SOP, SENECA 44.9 ms / 927 μJ, EventShiftFlow 0.142 W, SNE 0.221 pJ/SOP as **this student’s** service.
- ERAFT FPGA as E-RAFT or as event-SNN.
- SDformerFlow Horowitz mJ; “BN is 0.01% so drop it.”
- Dual last-use as a rename of SENECA grouping, SNE FIRE/UPDATE, or MS/SPE.
- Spatial-locality skip as a letter (ASNA-Flow abstract + G2 stop).

Allowed relative sentence:

*Event-OF silicon already exists (TrueNorth OF, FPGA plane-fit, SENECA FireNet, ASNA-Flow ASIC at abstract level). After AT-LIF absorb the spike path is binary GeMM, so Prosperity/Gustav/FireFly copy there. None of the inspected full texts last-use a second continuous PED/I24 tensor, and none stall on live full-domain BN \(10\times C\times H\times W\). SDformerFlow’s eval BN is live batch stats, matching native proj BN, but their energy model treats BN as free. Spatial locality of OF is already a sold ASIC title. The remaining object, if any, is typed dual last-use under the BN barrier — G1, not a new OF chip.*

---

## 7. Quote index

| Need | Pin |
|---|---|
| SENECA ~5% density; pixel 43.5 vs 66.5; 44.9 ms / 927 μJ | 2407.20421 Abstract |
| grouping=4 dummy fill; SRAM caps 56/120 | 2407.20421 §3.2 |
| “first … regression … on a neuromorphic processor” | 2407.20421 §1 — do not reuse as first-OF-HW |
| SDformerFlow MS before spikes | 2409.04082 Fig. 4 / §III |
| SPE deformed \(1\times1\) stride-2 | 2409.04082 Eq. 13–14 |
| disable tracking of running states | 2409.04082 §IV-A |
| ignore BN ~0.01% | 2409.04082 §IV-D |
| EventShiftFlow 1-bit occupancy, <2 kB, Artix-7 | 2605.28312 Abstract, Table V |
| Aung plane-fit / Haessig TrueNorth (paraphrase) | 2605.28312 §II-B |
| SNE DVS-Gesture, event-proportional SOP | 2204.10687 Abstract / §I |
| ASNA-Flow spatial locality + 104 FPS / 7.9 mW / 28 nm | IEEE abs. 10.1109/TVLSI.2025.3600953 — **not body** |
| ERAFT FPGA = RAFT, VCK190, Middlebury 86 fps | IEEE abs. 10.1109/ISCAS56072.2025.11043529 — **not body** |
| TrueNorth OF, spirals/pipes | 2003.06696 §1 citing Haessig |

Primary local texts: `p0_txts/2407.20421.txt`, `2409.04082.txt`, `2605.28312.txt`, `2204.10687.txt`. Session cross-links: `literature/L6_event_of.md`, `round2_absorb/literature/R2L5_residual_ped_bn.md`, `round2_absorb/adversarial/ADV_G2.md`, `idea_cards/unresolved_ASNA_Flow.md`, `idea_cards/unresolved_ERAFT_FPGA.md`.
