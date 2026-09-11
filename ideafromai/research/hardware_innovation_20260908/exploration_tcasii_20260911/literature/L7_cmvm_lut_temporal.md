# L7 — After a compiled 10×10 time mix: remaining cost, LUT/CMVM priors, temporal papers

Freeze: `PROBLEM.md` 2026-09-11. This note is literature + mapping, not an experiment and not a title claim.

**Focal question.** Once a dense T10 mix is handed to official da4ml CMVM (CSE + cutoff fold), what work is still expensive (RNE boundaries, dual-consumer live-until-last, BN domain), and which priors are already **A** so that a TCAS-II **X** must beat ordinary CSE + terminal cutoff fold + generic round→sat fusion.

**Read status.** Full local txts of the five assigned papers; official da4ml `cmvm.md` in-tree; grep of `survey_ab_fusion_20260910/p0_txts/` for the keyword list. LUT-DLA / T-MAC / da4ml papers are **not** in that txt dump; they are cited only from local method notes already in this tree.

---

## 0. What the freeze already spent (A, not X)

Compiled **source** graphs after official CSE (`PROBLEM.md`, `TECH_ARCH_SOFT_HARD_FULL.md`):

| Student | CSE add/sub per T10 vector | Intermediate RNE/sat | Terminal cutoff |
|---|---:|---:|---|
| ordinary dense-source/raw | **260** (66 shared nodes; result bits 8606, max 41) | **0** | RNE/sat **fold into** `postprocess.rows` exact cutoff |
| learnable 40-coeff lifting T10 | **159** (already includes identity; do not add 40) | **35** (must keep: guard/sticky/parity, conditional +1, sat) | last **5** may merge into cutoff; times `[6,7,8,9,5]`, source gates `[0,7,8,1,9]` |

Same two-stage SIMD source resource (always-ready slots **6938 → 5354**, **−22.83%**). Long backpressure **both 8088** — source-node gain is absorbed at the exit. Part of −22.83% is **generic round→sat fusion** (lifting unfused 6194 → fused 5354, **−13.56%**, 24 SIMD batches each drop 35 issues). That fusion is ordinary instruction combining, **not** X.

Separate integer **consumer** model (different resource point): **758777 → 714889 (−5.78%)**. Do not add the two tables.

AEE: ordinary raw **1.219801338**; lifting raw **1.232979368** (Δ **+0.013178**; absolute 1.259 passes, relative +0.005 **fails**). Integer gates / I24 / PED q24 on two captured windows: **0** vs model.

**Hard A list for this island (borrow ≠ title):**

1. Official da4ml CMVM: MST column graph → \(W=W_1W_2\), then bitwidth-weighted greedy CSE of \(a\pm(b\ll s)\) on CSD (`psn/cmvm_20260909/da4ml_official/docs/cmvm.md`; paper arXiv:2507.04535, **not** in `p0_txts`).
2. Terminal RNE/sat folded into exact cutoff on ordinary; last-5 cutoff merge on lifting.
3. Generic round→sat / norm24 fusion on the SIMD issue stream.
4. LUT-NN / LUT-DLA / T-MAC / Platinum table-MAC (see §3).
5. Multiplierless shift-add neuron dynamics (MFPSN, L-SPINE, DeepShift-Q).
6. Wavelet / lifting **as a time basis** (already the lifting student; wavelet papers in the dump are encoders, not T10 compilers).

**X kill-gate for this island.** Any candidate that, under same-port / same-state / same-backpressure, does not beat **ordinary 260-node CSE + cutoff fold + generic fusion** on net service of the **union** of (source DAG + dual consumers + native proj/BN), or that only restates −22.83% always-ready / −13.56% fusion, is a rename. Full-chain net service ≥15% is the SCOPE number; it is **unknown**. Relative AEE +0.005 already failed for lifting raw.

---

## 1. Keyword grep of `p0_txts/` (what is actually on disk)

Directory: `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/survey_ab_fusion_20260910/p0_txts/` (~150 txts).

| Query | Hits in `p0_txts` | Meaning for this island |
|---|---|---|
| `da4ml` | **none** | Compiler A lives in `psn/cmvm_20260909/`, not in the survey dump. |
| `CMVM` / `constant matrix` | **none** | Same. |
| `adder net` / `AdderNet` | **none** | No Chen/Wang AdderNet (add-only CNN) txt. Do not invent it as a local prior from this dump. |
| `LUT-DLA` | **none** | Local method notes exist (`idea_cards/LUT_DLA.md`, `bn_state/next_training_proposal.md`); fulltext not in dump. |
| `T-MAC` | **only** `2511.21910.txt` (Platinum) | CPU table-MAC baseline: 2-bit encoding of ternary, Apple M2 Pro 16-thread. |
| `lifting scheme` | **none** as wavelet lifting | `2602.12590` uses “lifting” as integration-by-parts; `2503.12905` is the word “Shoplifting”. |
| `wavelet` | `2605.09770`, `2608.19238`, `2405.16790`, `2510.12102` | Time-causal **encoders**, not T10 CMVM. |
| `multiplierless` / `multiplier-less` | `2608.00595`, `2604.03626` | FPGA LIF shift-add / L-SPINE SIMD neuron. A for dynamics, not for compiled mix. |

**Implication.** The dump does **not** contain the CMVM/LUT stack that this island actually uses. Those priors must be taken from the in-tree da4ml mirror and the already-read LUT-NN/LUT-DLA/T-MAC notes, and labelled **A**. The five assigned papers are about **what remains after a matrix/stage has been reduced**: stage boundaries, dual consumers of one source, residual terms, and time-dense optical-flow heads.

---

## 2. Assigned five papers (titles confirmed from first page of each txt)

### 2.1 arXiv:2407.10416 — SOFA

**Title (confirmed):** *SOFA: A Compute-Memory Optimized Sparsity Accelerator via Cross-Stage Coordinated Tiling* (Wang et al., Tsinghua / SJTU).

**Architecture (methods, not PPA).** Dynamic-sparsity attention is three stages: 4-bit pre-compute of \(\hat A\), top-k, then 16-bit formal QK/softmax/SV. Three remaining costs after “the mix is predicted”:

1. **Prediction can exceed formal compute.** With top-k = 20%, 4-bit predict vs 16-bit formal, prediction power is already **1.4×** formal; it grows with token parallelism. A later stage that is “already sparse” is not free.
2. **Whole-row ready.** Top-k and softmax are row-serial. Pre-Atten and A go DRAM then come back by row. Fine-grained tiling is blocked unless the **boundary certificate** (global Max / top-k indices) is forwarded. FA-2 tiling **increases** exp/compare (lines 5–8 of their FA-2 listing); at \(S=2048\) they quote \(9\times10^6\) extra exp and \(3\times10^5\) extra compares vs vanilla.
3. **Dual-stage consumers of the same score.** Formal compute still needs Max-ensure because DLZS is log-domain **approximate**. They add an Auxiliary Process (AP) module; SU-FA default **descending** order uses the top-k Max to skip FA-2’s per-tile max refresh. RASS reorders KV (~33% less memory vs left-to-right) — generic schedule, not a new arithmetic.

Mechanism: **DLZS** = leading-zero encoder into log domain, then shift-add (explicitly multiplier-free). **SADS** = split a long row into sub-segments, sphere-search so dominant Type-I tokens are not missed. **SU-FA** = fuse QK+softmax+SV into one tiled pipeline **using the previous stage’s sorted Max**. Hardware: 128×4 PE array, reusable DLZS engine, flexible-input SADS, two systolic arrays + AP + O-update, 192/96/28 KB SRAM banks. Reported 9.5× / 71.5× vs A100 and 15.8× energy vs 8 SOTA **must not** be copied; they are LLM/LTPP, not valid825.

**Map to freeze.** Ordinary CSE + cutoff fold + round→sat fusion is exactly SOFA’s “fuse QK+softmax+SV and fold Max into the next tile.” After that, **SOFA still pays** (a) a live source until the **last** consumer of the score (SV and O-update), (b) an **inexact** boundary (DLZS error → AP), (c) DRAM/SRAM at stage cuts. Local analog: lifting’s **35 intermediate RNE** are the inexact/identity boundaries that **cannot** be fused the way terminal cutoff can; dual PED+gate are two consumers of the same compiled vector; 8088 long backpressure is “whole-row / whole-vector ready.” **X is not another SU-FA rename.** If anything is left, it is a **dual-consumer-aware identity** at RNE cuts that ordinary fusion does not have, with the AP-style extra unit **charged**.

**Kill.** If the proposal is “forward a Max/certificate and fuse the next stage,” it is SOFA §III-C + local generic fusion. Stop that layout; keep the family as A.

---

### 2.2 arXiv:2204.10687 — SNE

**Title (confirmed):** *SNE: an Energy-Proportional Digital Accelerator for Sparse Event-Based Convolutions* (Di Mauro, Prasad, Huang, Spallanzani, Conti, Benini; ETH / Bologna).

**Architecture.** Digital 22 nm FDX, 400 MHz, 4-bit weights / 8-bit state LIF with **linear** leak (not exponential). Event tuple \(E_i=(\mathrm{OP},t,x,y)\), OP ∈ {RST, UPDATE, FIRE}. Outer loop is **time**, then explicit events; inner loops are output neurons, not a dense H×W scan. One input event is consumed in **48 cycles** while all sensitive neurons update. Energy **0.221 pJ/SOP**, 4.54 TSOP/s/W, 51.2 GSOP/s peak; IBM-DVS-Gesture 92.8%, 80–261 µJ/inf at 1.2–4.9% activity. **Do not quote these as local PPA.**

Datapath that still costs after “the event is encoded”:

- **C-XBAR ready-valid.** Point-to-point **or** broadcast that **pauses until every slave has the event**. Dual (or N) consumers of one source word = wait-all, not skip-one.
- **Dual-buffer neuron state** + time-of-last-update (TLU) to skip idle timesteps — ordinary zero-Y / skip, A.
- **Filter buffer ≤256** weight sets, address-selected per event — multi-consumer weight residence.
- **FIRE vs UPDATE** as two uses of the same membrane: FIRE dumps to per-cluster output FIFOs so TDM does not stall; collector packs sparse outputs to one DMA. Analog of gate-FIFO vs continuous residual FIFO.
- Mapping: all layers on slices (events never leave) **or** time-multiplex tiles through external memory. The second mode is the local “source DAG + later proj/BN still in another resource point” split.

Neuron is **not** ATLIF continuous \(\theta_g\). Linear leak and 4-bit synapses are a different identity.

**Map to freeze.** Energy-proportional SOP and event encoding are **A** (Gustav NRV / source∩W language). Remaining local cost that SNE actually implements and that CSE does not erase: **broadcast wait-all until the last slave** (dual consumers), **48-cycle full-receptive-field update per source event** (compiled mix still touches many live neurons), and **FIRE/UPDATE split buffers**. Native proj+BN is closer to SNE’s “tile through external memory” than to the on-slice all-layer mode.

**Kill.** “Charge energy only for events / spikes” without dual-consumer wait-all and without the continuous PED path is a weaker denominator than PROBLEM.md. Stop that layout.

---

### 2.3 arXiv:2307.05033v3 — EVA-Flow

**Title (confirmed):** *Towards Anytime Optical Flow Estimation with Event Cameras* (Ye, Shi, Yang, Wang, Yin, Sun, Wang, Wang; ZJU / HNU). GitHub `Yaozhuwa/EVA-Flow`.

**Architecture (not a compiler).** **Unified Voxel Grid (UVG):** each bin has fixed support \(2\tau\) centered at \(t_b\), bilinear in \(x,y,t\); a bin is ready after \(\tau\) (5 ms on DSEC with 21 channels) instead of waiting the full 100 ms Voxel Grid. Encoder → 4-level pyramid → stacked **SMR**: vertical coarse-to-fine warp+ConvGRU+residual FlowHead; horizontal time recurrence with **shared weights across \(j\)**, **distinct weights across pyramid \(i\)**.

\[
\hat f_j^i=\mathrm{Warp}(f_j^i,F_{0,j}^{i-1}),\quad
H_{t,j}^i=\mathrm{ConvGRU}(H_{t,j-1}^i,\mathrm{concat}(\hat f_j^i,\mathrm{Up}(F),\mathrm{Up}(H))),\quad
F_{0,j}^i=F_{0,j}^{i-1}+\mathrm{FlowHead}(H).
\]

**Only the last flow \(F_{0,B-1}\) is L1-supervised** (E-RAFT-style). Intermediate timesteps are implicitly regularized by dense warp. **RFWL** rectifies FWL by normalizing IWE variance by total event count so events warped off-frame do not fake a worse score.

DSEC-Flow test (their Table 1): EVA-Flow 5.0M params, **16.8 GMACs per prediction**, 5 ms latency, 200 Hz, 10 Hz GT only, **EPE 0.88**. E-RAFT 0.79 EPE / 100 ms / 10 Hz. **DSEC EPE ≠ valid825 AEE**; do not import 0.88.

**Map to freeze.** This paper is the **task-side** temporal prior, not CMVM:

- Sequential bins + last-step-only loss **do not compile a 10×10 mix**. Local T10 is **noncausal**, dense, already compiled.
- Shared-weight SMR across time is **shared-Q language**. Local shared-Q already measured: weighted terms −1.13%, H8 −4.44%, NRV +0.166%, BN residual constants 288→**2880 B**, extra BZ+inverse. Ablation only.
- RFWL is an **unsupervised intermediate identity** (contrast after warp). Local analog is “does an intermediate RNE/sat still match the integer consumer,” not a new arithmetic.
- Anytime output (a flow after each bin) is **not** dual-consumer of one compiled vector. Local dual consumers are gate **and** continuous PED of the **same** T10 output.

**Kill.** Replacing T10 PSN by UVG+SMR, or advertising 200 Hz / 5 ms / DSEC EPE, is a different paper. Stop that layout; keep EVA-Flow as a **task** control (anytime vs batch T10), not as compile X.

---

### 2.4 arXiv:2403.07953v3 — TASDER / TASD (MLSys 2025)

**Title (confirmed):** *Enabling Unstructured Sparse Acceleration on Structured Sparse Accelerators* (Jeong, Tsai, Bambhaniya, Keckler, Krishna; Georgia Tech / NVIDIA).

**Architecture.** **TASD:** any tensor \(A \simeq A^{s_1}_1 + A^{s_2}_2 + \cdots\) by repeatedly extracting an N:M view of largest-magnitude entries and leaving a residual. Distributive GEMM: \((A_1+A_2)B = A_1B+A_2B\). **TASDER** searches per-layer series so accuracy stays ≥99% of original (MLPerf-style). **TASD-W** offline on unstructured weights; **TASD-A** **runtime** after ReLU (or pseudo-density on GELU/Swish). Hardware: TASD Tensor Core on VEGETA/STC; extra **TASD units** extract terms from activation tiles (up to M cycles/block; 16 units hide latency on TTC-VEGETA-M8). Decomposition-aware dataflow: **B and C stationary**, stream \(A_1\) then \(A_2\) (Figure 11) — two terms **share the other operand and the psum**. Reported EDP up to 83%/74% and 39% on RTX 3080 **must not** be copied.

**Map to freeze.** Closest compile analog in the assigned set:

- Stage-1 da4ml MST + Stage-2 CSE is a **lossless** structured decomposition of a **constant** 10×10. TASD is **lossy** N:M on large GEMM. Local integer identity is **0-diff** on gates/I24/PED; TASD’s 99% accuracy bar is the wrong contract.
- **Residual terms after the first structured view are still paid.** TASD keeps a second 1:8 (or 2:8) term because dropping it loses magnitude. Local analog: after CSE, **35 RNE nodes** and **dual consumers** are the residual. You do not get to delete them because the first 159 adders look cheap.
- Dual TASD terms sharing B/C is the right **dataflow picture** for dual consumers sharing a compiled source: keep the source/B live, accumulate two uses into C. That is **A** (stationary psum + multicast A), not X.
- TASD-A after ReLU is a **dynamic** residual extractor. Local BN uses **actual batch stats on 10×96×120×160**, not a ReLU sparsity view. Do not treat BN as a TASD-A layer.

**Kill.** “Approximate the 10×10 by N:M + residual” violates 0-diff integer identity. “Two-term stationary C” without beating CSE+fusion is VEGETA dataflow. Stop those layouts.

---

### 2.5 arXiv:2412.11284 — Learning Normal Flow / VecKM

**Title (confirmed):** *Learning Normal Flow Directly From Event Neighborhoods* (Yuan, Burner, Wu, Liu, Chen, Aloimonos, Fermüller; UMD). Code `dhyuan99/VecKM_flow`.

**Architecture (algorithm, not RTL).** Per-event neighborhood \(N(e_k)\) in normalized camera coords; **VecKM** local encoder (adjacency × random Fourier features, no explicit sample-and-group). MLP predicts \(\hat n_k\). Loss = **radial** (Thales circle of GT optical flow as diameter, Eq. 2/5) + **angular** (kill the zero-flow trivial solution, Eq. 6); gradients orthogonal. Augmentations: random rotation / scale / sampling. **UQ:** rotation-equivariant ensemble, circular std as \(\sigma\); drop high-\(\sigma\) events. Downstream: IMU-derotated normal flow → linear SVM on depth-positivity (Alg. 1).

Qualitative dual use of one encoding: rich texture → full optical flow; edge → normal flow (their Fig. 9). Metrics PEE / %Pos on MVSEC/EVIMO2; DSEC is qualitative (no per-event GT). **Not** valid825 AEE.

**Map to freeze.** Dual **semantic** consumers of one local code (normal vs full flow; keep vs drop by \(\sigma\)) rhyme with gate vs continuous PED, but:

- The split is **lossy, trained, and UQ-filtered**. Local dual consumers are **both required**; deleting PED historically wrecks ten-frame AEE. UQ-drop is F1-like source-word suppression, second queue, not compile X.
- Ensemble inference (K rotations) **multiplies** the encoder. Opposite of reducing post-CSE work.
- No CMVM, no RNE, no BN domain.

**Kill.** “Predict only normal flow / drop uncertain events” as a replacement for T10+PED. Keep as a **gating** control for F1, not as LUT/CMVM X.

---

## 3. LUT / table-MAC / multiplierless / wavelet — A stack (mostly off-dump)

These are the priors a reviewer will name if the title says “lookup / adder tree / lifting.” All are **A** unless a measured hole remains after they are given the same ports.

### 3.1 da4ml CMVM (in-tree, not in `p0_txts`)

`psn/cmvm_20260909/da4ml_official/docs/cmvm.md` + README:

- Stage 1: columns of \(W\) as vertices, distance = nnz of sum/difference, Prim MST, \(W=W_1W_2\).
- Stage 2: CSD, greedy \(a\pm(b\ll s)\) CSE, **bitwidth-weighted**.
- Optional min-delay constraint **increases** nodes (S0: 236 @ depth 8 vs 263 @ depth 6). Pipelining is not free.
- S0 10×10 Aq: CSD 464 nodes vs CMVM **236**; 737.28M U and gates **0-diff**. Node count −49% **≠** cycle −49%. Same known-zero: 59.7M 96-lane MAC vs 153.9M vector add/sub (ratio 2.579 is **not** a period).
- Patch T10 graphs: row34 108/5, common3 96/6; **common prefix can have more nodes** than row34. Prefix U is not a full gate.
- RF: default temp 1124 bit/scalar; pressure-first 12.9 KB/96-lane. Fully unrolled wires still pay fanout, retiming, shifts.

**After this compiler, remaining local cost is not “too many constant multiplies.”** It is: intermediate **quantize/round identity**, **dual live consumers**, **BN full-domain stats**, and **finite-port schedule of a wide DAG** (max 41-bit ordinary, 38/39-bit lifting).

### 3.2 LUT-NN / LUT-DLA / T-MAC / Platinum

Not in `p0_txts` except Platinum’s T-MAC citations.

| Prior | What was actually read in-tree | Why it is A here |
|---|---|---|
| LUT-NN (MobiCom 2023, 2302.03213) | Centroid learning, hard-fwd/soft-bwd, **PSum table INT8 QAT**, SIMD lookup, hierarchical accumulate | Table MAC + trained codes. Local forced-support already −37.94% FC1→PSN steps **and** 13.68→34.85 MB/frame coefficient traffic. |
| LUT-DLA (HPCA 2025, 2501.10658) | CCM distance/argmin, index FIFO, IMM PSum LUT, LUT-stationary, index reuse, double buffer, LUTBoost staged train. §IV-B: hiding table-swap latency ≠ reducing traffic. DSE “pruning” drops **configs**, not PSum entries. | Encoder + cold fill + wide table **must** be charged. Incomplete local port ≠ “LUT-DLA lost.” |
| T-MAC (EuroSys 2025, 2407.00088) | Bit-serial table, register-resident LUT, layout, **complement-mirror** \(L(1-D)=\sum W-L(D)\), table quant | Mirror needs ±1 pairing. Local 0/1 support codes are **not** complementary; extra constant term required. 2-bit ternary encoding (Platinum’s quote) still >1.58 bit. |
| Platinum (2511.21910, **in dump**) | Offline construction **paths**, online LUT[dst]=LUT[src]±a[j], bit-serial **or** ternary LUT, PPE query+aggregate, extra adders because construct vs query port/adder ratios differ. LUT+weight SRAM **83.3%** area. T-MAC-T16 is the CPU baseline. | Offline path vs Prosperity **runtime** shortcuts is F3-adjacent A. Cold-fill, banks, overflow still eat FC1 savings. BitNet ×speed ≠ AEE. |

**Local LUT hole that is still not X.** Forced LUT reduced **service steps** and **increased** coefficient reads. Zero-response directory on \(L=DW^\top\) was a constrained-block-sparsity rename (~4.5/10) and stopped. A new LUT title would need **both** consumers to read **fewer** physical table words **after** encoder/fill, vs complete LUT-DLA + ordinary INT10/BDI + da4ml CSE. No such measurement.

### 3.3 Multiplierless neurons (in dump)

- **2501.14490 MFPSN:** channel-wise parallel PSN, sawtooth dilation, PoT+shift, STE. Direct **algorithm** control for T10/lifting. Replacing lifting by MFPSN/PoT without a consumer-interface delta is not X.
- **2604.03626 L-SPINE:** 2/4/8-bit SIMD, multiplier-less shift-add LIF, picoRV32 host, FPGA 459 LUT/neuron. Dynamics A; no compiled mix, no dual PED, no BN domain.
- **2608.00595:** time-mux 1-bit spike feed, integer LIF, pipelined argmax, MNIST FPGA. Multiplierless leak is a right-shift. Toy net, not T10.

AdderNet (add-only CNN replacing MAC by \(\ell_1\)) is **absent** from the dump. If used as a reviewer prior, treat as A for “multipliers already gone after CMVM,” not as X.

### 3.4 Wavelets / lifting (in dump, wrong kind)

- **2605.09770** *Encoding and Decoding Temporal Signals with Spiking Bandpass Wavelets* (Pedersen, Lindeberg, Gerstoft): time-causal DoE/DoT frames, spike-quantized bandpass, reconstruction bounds, ECG/audio. Maps to neuromorphic **encoders**. Local T10 is a **learned dense mix on already-binned events**, noncausal, compiled. Do not retitle PSN as a wavelet ADC.
- **2608.19238 / 2510.12102 / 2405.16790:** SWformer / WGSE citations. Same encoder family.
- Tunable lifting CNN units (ARX-133/149 in the merged CSV, **no txt**): invertibility of filter banks. Local 40-coeff lifting is already that **model** family. Hardware X cannot be “we used lifting.”

---

## 4. Remaining expensive work (the only legal B for this island)

After ordinary 260-node CSE **and** lifting 159-node CSE **and** terminal cutoff fold **and** generic round→sat fusion, PROBLEM.md still leaves three billed islands.

### 4.1 RNE boundaries

- Ordinary: **0** intermediate RNE; terminal round/sat **already folded**.
- Lifting: **35** intermediate RNE/sat **kept** (guard, sticky, parity, conditional +1, sat). Last 5 may merge. Full half-step gate-graph merge was **241 ≥ 159+35** and **stopped**.
- Two-stage SIMD: fused lifting still issues **35 norm24** (table: add/round/sat/norm24/gate = 159/0/0/35/10). Fusion removed **round and sat issue slots**, not the **identity**.
- Half-step RNE microprobe (box MP, ~76 cells) only shows the checkpoint is expressible. Not net service.
- SOFA’s lesson: folding Max into SU-FA still left an **AP** unit because the predictor was approximate. Local integer path is **exact**; the 35 RNE are **exact identity cuts**, not approximations. Early-stop / MSB bounds (CompRRAE/SPARK/USEFUSE in `psn_decision_one_page.md`) are **A** and were already judged insufficient as X without a concrete bitwidth/fanout/RF save.

**What would still be X (narrow).** A **consumer-specific** rounding identity: the **gate** consumer can legally use a cheaper cut (threshold only) while the **PED** consumer keeps full sticky RNE, **without** cloning the 159-node DAG and without violating 0-diff on both captured windows. Ordinary CSE+fusion applies the **same** cut to the whole vector. That split is **not** in SOFA, da4ml, LUT-DLA, or TASD. It is also easy to fake: if it reduces to “two output formats in `postprocess.rows`,” it is cutoff fold.

### 4.2 Dual-consumer live-until-last

Consumers after source (PROBLEM.md):

1. Spike/gate: `proj.sn` → \(\theta_g\) → native `proj.conv` (96×96×3×3, stride 2).
2. Continuous residual/PED: `U_ped→V_ped` + merge.
3. Native projection **conv + BN + residual add** still exist.

Source DAG parents must stay live until the **true last** consumer. Always-ready −22.83% vs long-backpressure **8088=8088** is the measurement that **exit stalls eat source-node savings**. Integer consumer-only −5.78% is the measurement that **the expensive work moved downstream**.

SNE’s C-XBAR wait-all and TASD’s B/C-stationary two-term replay are the **A** pictures. SOFA’s whole-row ready is the **failure mode** of “wait for the compiled vector.” EVA-Flow’s last-step-only supervision is **not** a license to drop the continuous consumer.

**What would still be X (narrow).** A schedule/identity that lets the **gate** retire a source parent while **PED** still holds a **smaller residual state**, with both integer windows 0-diff, vs giving ordinary CSE the same two-output postprocess and the same FIFO. If ordinary two-output cutoff already matches, X is dead. Do not multiply source-slot % by consumer %.

### 4.3 BN domain

Native projection BN on captured students uses **actual batch statistics over the full 10×96×120×160 domain**, not frozen running stats. Local-window replay with **free** mean/var **undercharges wait/storage**. SCOPE lists this cost as **unknown**.

None of the five papers, and none of the LUT/CMVM stack, compute a **dense 4-D batch mean/var** as a billed service. TASD-A’s ReLU residual, SNE’s TLU, SOFA’s tile Max, and VecKM’s local neighborhood are all **local** reductions. Full-domain BN is closer to a **reduction tree + broadcast** that sits **after** the compiled mix and **before** residual add.

**What would still be X (narrow).** Fuse BN’s full-domain reduction with the **already-live** dual-consumer reduction (gate count / PED moments) so the extra pass over 10×96×120×160 is not a third consumer of the source — **after** giving ordinary CSE+fusion a fused BN-scale into V (fixed-BN student already folds BN2 gain into F). If frozen running stats match AEE, the hole disappears and X dies. If local-window stats are given **free**, the experiment is invalid under PROBLEM.md.

---

## 5. Relative-prior table (reviewer view)

| Claimed move | Completeness of A | Hole in **this** net | X or rename? |
|---|---|---|---|
| More CSE / CSD / MST | da4ml two-stage, already 260 vs 159 | Intermediate RNE + ports | Rename |
| Fold terminal round/sat | ordinary cutoff; lifting last-5; SIMD fusion −13.56% | 35 **intermediate** RNE; dual identity | Rename unless split per consumer |
| LUT / codebook MAC | LUT-NN, LUT-DLA, T-MAC, Platinum, Prosperity shortcuts | Dual consumers still want **dense** residual; table traffic went **up** | Rename; encoder/fill must be in the union |
| Multiplierless neuron | MFPSN, L-SPINE, DeepShift, SNE linear leak | Mix is already add/shift after CMVM | Rename |
| Wavelet / lifting title | 40-coeff lifting student; 2605.09770 encoder | Relative AEE already **fails** +0.013 | Not hardware X |
| Cross-stage attention tiling | SOFA DLZS/SADS/SU-FA/RASS | T10 is 10×10, not S×S attention | Wrong island |
| Event-proportional SOP | SNE | Continuous PED + BN domain | Incomplete A if used as title |
| Anytime flow / UVG+SMR | EVA-Flow | Noncausal compiled T10; dual PED | Task control only |
| N:M residual series | TASDER | Breaks 0-diff integer | Forbidden lossy |
| Normal-flow UQ gate | VecKM | PED cannot be dropped | F1 control only |
| Dual-consumer-aware RNE / retire | **not** in the five papers; SPARK/USEFUSE/CompRRAE are A for **single** consumer early-stop | 35 RNE + 8088 backpressure + dual PED | **Only remaining legal X direction on this island**, must beat two-output cutoff |
| Full-domain BN fused into live reductions | **not** in dump; local-window free μ/σ is **illegal** | 10×96×120×160 stats unknown | Legal B; X only if ordinary fused-BN / running-stats **fail** AEE or still pay a third pass |

---

## 6. TCAS-II constraints for ideation on this island

- One mechanism. Circuits+systems. Relative prior **must** name da4ml CSE, cutoff fold, generic fusion, and (if lookup is mentioned) LUT-DLA/T-MAC/Platinum.
- Do not change identity to binary ATLIF. Do not sell analog CIM. Do not quote OpenROAD/Yosys as foundry PPA. Do not multiply −22.83% × −5.78% into FPS.
- Strongest controls: ordinary 260-node CSE + cutoff; lifting 159+35 with **same** fusion permission; two-output postprocess; frozen vs actual-batch BN; complete LUT-DLA encode+table if lookup is in the claim; same-port / same-state / same-backpressure; 0-diff integer windows.
- Kill numbers: valid825 AEE ≤1.259 and Δ vs ordinary ≤+0.005; union net service ≥15% vs CSE+cutoff+fusion; long-backpressure must **move**, not stay 8088=8088; BN experiment must **not** gift mean/var.

**Bottom line.** The five assigned papers confirm titles and supply **boundary / dual-consumer / residual-term** pictures. They do **not** supply a compile X. The dump has **no** da4ml, AdderNet, LUT-DLA, or lifting-scheme paper; T-MAC appears only as Platinum’s CPU baseline. After a dense 10×10 is compiled, expensive work is **not** the constant multiplies. It is **RNE identity cuts**, **two consumers that keep the DAG live**, and **BN over the full 10×96×120×160 domain**. Candidate X must beat ordinary CSE + cutoff fold + generic fusion on that union; anything else is a reskin.
)
