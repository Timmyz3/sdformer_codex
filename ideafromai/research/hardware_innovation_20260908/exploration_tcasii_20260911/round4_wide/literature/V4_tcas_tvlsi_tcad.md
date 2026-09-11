# V4 — TCAS-I / TCAS-II / TVLSI / TCAD / TBioCAS / IEEE Micro / TRETS (2023–2026)

**Freeze:** 2026-09-11. Round-4 **wide** CASS/EDA journal scan.  
**Target venue:** IEEE TCAS-II Express Briefs, **5 pages**. Same-journal collisions are first-class.  
**Identity:** `IDENTITY_ATLIF.md` — official AT-LIF \(\{0,\theta\}\) with layer-shared \(\theta\) **absorbed** into next \(W\). Spike path after absorb = **binary GeMM**. Residual / PED / I24 = a **separate continuous tensor**. Task = event-camera 2D optical flow, DSEC valid825 AEE. Not CIM-as-title. Not first-event-OF-HW. Not Prosperity-on-our-net. Not continuous-\(\theta g\) dual-MAC.

This sheet answers one reviewer question for every located paper:

> Would a TCAS-II reviewer who **just handled ESTU** (72(12), 5 pages) **desk-reject a letter that looks like this paper?**

Collision is the **object**, not the board. ESTU’s object, one line (`ADV_ESTU_same_journal.md`, AAM opened; absorb freeze does **not** change it):

**binary spike tensor → one accumulate consumer → skip inactive groups-of-4 → classification % + mW on a 5k-LUT FPGA overlay.**

A five-page letter that can be rewritten as “FPGA spiking transformer with sparsity skip” without (i) dual last-use of **post-absorb binary GeMM and continuous PED**, (ii) DSEC AEE gates, (iii) same-port dual-completion service, **collides**. Copy ESTU as **A** on the spike path, then stop as **X**.

**Depth legend.** `full` = local PDF/txt or opened AAM/arXiv HTML. `abs` = IEEE/arXiv/publisher abstract only. `unresolved` = no PDF this pass; do not invent PE/SRAM. Abstracts are legal evidence for collision **objects**; they are not datapath copies.

**Card merge (do not double-count).** `unresolved_Liu_optical_flow_FPGA_TCAS_I_2025.md` and `unresolved_TCAS_I_2025_adaptive_OF.md` are the **same IEEE paper**: DOI `10.1109/TCSI.2025.3572918`. The “adaptive OF / reconfigurable pyramid” slogan **is** Liu et al. The 405 FPS / AEE 1.02 numbers are **frame** OF on ZCU104, **not** event SNN.

---

## 0. ESTU control (same journal) — copy, then never look like it

| Field | Record |
|---|---|
| Title | ESTU: Enabling Spiking Transformers on Ultra-Low-Power FPGAs |
| Venue | IEEE TCAS-II Exp. Briefs, vol. 72, no. 12, pp. 2027–2031, Dec 2025 |
| DOI | [10.1109/TCSII.2025.3626209](https://doi.org/10.1109/TCSII.2025.3626209) · IEEE 11218914 |
| Authors | Leone, Busia, Orrù, Raffo, Meloni (Univ. of Cagliari) |
| Depth | **full AAM** (`adversarial/ADV_ESTU_same_journal.md`). IEEE HTML empty. GitHub `EOLAB-2025/ESTU` tree **search-incomplete**. |
| Object | Binary SSA \(QK^\top V/\mathrm{scale}\) on spikes. LIF \(s(t)=v(t)>\theta\) then reset. Microcode overlay: `Dense(spike/int)`, `Mul(spike,spike)` = 16 AND + 16-input popcount. Activity stack, groups of 4 (software sparsity 0.95 → exploited 0.82). Typed spike-mem **or** int-mem **or** through LIF — muxed destination, **not** a fork. |
| Their numbers (theirs) | iCE40UP5K, 4301 LC / 30 BRAM / 6 DSP / 21 MHz; 3.76 mW inf, 0.23 mW standby, 4.28 μJ/inf @ 12 MHz; NinaPro DB-5 87.21% (86.97% after 8-bit W); 29% time from sparsity. |
| After absorb | Group skip + AND-popcount SSA are **complete A** on the post-absorb binary GeMM. Integer path is **then binarized**. Continuous PED/I24 is never a second live consumer. Task is class-%, not DSEC AEE. |
| ESTU desk-reject if letter looks like this? | **This *is* the script.** Any 5-page “spikeformer FPGA + skip + overlay + mW + accuracy” is ESTU 72(12). |

ADV’s retired item (“continuous \(\theta g\) is the source value”) is **killed** by `IDENTITY_ATLIF.md`. Surviving non-collision objects: dual last-use of binary GeMM **and** continuous PED; DSEC AEE; same-port net service. ESTU has none.

---

## 1. Master table (20 located objects)

**ESTU-DR** = desk-reject a *TCAS-II letter that looks like that paper*?  
**A/B/X** under absorb: **A** = copy then stop as title; **B** = strong negative / anti-prior; **X** leftover only if a *different* object is measured; **CIM-stop** = copy at most as A then illegal as title.

| # | Paper | Venue / year | DOI | Depth | ESTU-DR? | Collision object (one sentence) | A/B/X after absorb |
|---|---|---|---|---|---|---|---|
| 1 | **ESTU** | **TCAS-II** 72(12) 2025 | 10.1109/TCSII.2025.3626209 | full AAM | **YES — it is the control** | Binary SSA overlay that skips inactive spike groups and reports class-% + mW on a 5k-LUT FPGA. | **A** (binary SSA + group skip) then **stop**. Same journal. |
| 2 | **FireFly-S** | **TCAS-I** 72(8) 2025 pp. 4007–4020 | 10.1109/TCSI.2024.3496554 · arXiv:2408.15578 | full local txt | **YES** | Spatial FPGA that ANDs a spike bitmap with a pruned-weight mask, then IF/LIF, and sells FPS/W on MNIST/DVS-Gesture/CIFAR. | **A** (dual-side Bitmap skip on post-absorb GeMM). Dual-**side** ≠ dual-**consumer**. Prune+LSQ train is identity-changing. |
| 3 | **FireFly-T** | **IEEE TC** 75(6) 2026 pp. 2185–2199 *(not CASS; mandatory relative)* | 10.1109/TC.2026.3672901 · arXiv:2505.12771 | full local txt | **YES** | Dual-**engine** overlay: sparse decoder for conv/linear + AND-PopCount binary attention; residual is a **fourth AXI port of pre-neuron membrane**, not PED last-use. | **A** (overlay + AND-PopCount attn). Name trap: dual-engine ≠ dual-consumer. CIFAR-Net / Spikingformer class-%. |
| 4 | **ASNA-Flow** | **TVLSI** 33(12) 2025 pp. 3409–3422 | 10.1109/TVLSI.2025.3600953 | **abs only — unresolved PDF** | **NO as ESTU-reskin; YES as “first event-OF neuromorphic”** | 28 nm event-driven SNN OF ASIC that skips by **optical-flow spatial locality**, 104 FPS / 7.9 mW / 0.3 pJ/SOP, and claims “first dedicated neuromorphic OF.” | **A** at title/abstract for spatial-locality skip (G2 already **stop**). PE/SRAM **unknown**. Recycle “first” = desk-reject vs this abstract, not vs ESTU. |
| 5 | **Liu adaptive OF FPGA** | **TCAS-I** 72(12) 2025 pp. 7835–7847 | **10.1109/TCSI.2025.3572918** | **abs only — unresolved PDF** | **NO** | **Frame-camera** pyramid OF on ZCU104: dynamic-direction prediction + configurable PEs + reconfigurable pyramid pipeline; **405 FPS, AEE 1.02**. Not events, not SNN, not transformer. | **A** only as “FPGA OF exists and already reports AEE.” **B** if the letter’s headline is FPS/AEE on a Xilinx board. Cards `unresolved_Liu_*` and `unresolved_TCAS_I_2025_adaptive_OF` **merge here**. |
| 6 | **Xpikeformer** | **TVLSI** 33(6) 2025 pp. 1596–1609 | 10.1109/TVLSI.2025.3552534 · arXiv:2408.08794 | arXiv full + method card | **NO as ESTU-reskin; YES if analog-hybrid is the title** | AIMC crossbar for FFN/FC + **stochastic** Bernoulli-AND attention; ImageNet/symbol-detect; 13× vs ANN digital, 1.9× vs digital SNN-transformer **projection**. | **CIM-stop**. SSA stochastic bits ≠ absorbed binary SSA and ≠ PED. Card `unresolved_Xpikeformer.md` is a stub; use `arxiv_2408.08794.md` + this row. |
| 7 | **LoopTree** | **TCAS-AI** 1(1) 2024 pp. 97–111 *(CASS sibling, not TCAS-II)* | 10.1109/TCASAI.2024.3461716 · arXiv:2409.13625 | full local author txt | **NO** | Analyzer (not an accelerator) of fused-layer **tile × retain/recompute × pipeline hidden-latency**. | **A** as same-port occupancy/transfer **vocabulary**. Not a circuit object. Fusion set is exogenous. No spike/NRV. |
| 8 | **SNE** | **DATE 2022** (family: Kraken 22 nm) | 10.23919/DATE54114.2022.9774552 · arXiv:2204.10687 | full local txt | **NO as ESTU-reskin** | Energy-proportional digital eCNN: RST/UPDATE/FIRE on COO events; IBM-DVS-Gesture **classification**; 0.221 pJ/SOP. Kraken maps a 4-layer LIF-FireNet **per-pixel OF** on the same engine (ISSCC-family SoC, not a CASS letter). | **A** (event-proportional conv + FIRE/UPDATE split of **one** LIF membrane). FIRE/UPDATE ≠ GeMM vs PED. Gesture % ≠ DSEC AEE. Pre-window DATE; keep as family prior. |
| 9 | **TrueNorth OF** | **TBioCAS** 12(4) 2018 pp. 860–870 | 10.1109/TBCAS.2018.2834558 · arXiv:1710.09820 | **abs** (PDF unread this pass) | **NO** | Barlow–Levick spike-time OF on TrueNorth + ATIS; ~11% AEE, <80 mW **sensor+compute**. | **A** that event-OF silicon **already exists**. **B** for “first event-OF HW.” Not a transformer, not dual last-use. Outside 2023–2026; required prior. |
| 10 | **plane-fit FPGA** | **ISCAS 2018** | 10.1109/ISCAS.2018.8351588 | **abs** (PDF unread; EventShiftFlow paraphrase of body) | **NO** | Fully pipelined 3×3 plane-fit on DVS events; 100 M fits/s, sub-μs latency; AAEE vs density trade-off. | **A** that event-OF **FPGA** already exists. **B** for first-event-OF-FPGA. Not SNN-Transformer. Outside window; required prior. Card still `unresolved_FPGA_plane_fitting_*`. |
| 11 | **FireFly** | **TVLSI** 31(8) 2023 pp. 1178–1191 | 10.1109/TVLSI.2023.3279349 · arXiv:2301.01905 | abs + family full later | **YES** | DSP48E2 multiplex-accumulate of **binary spike × W**, on-the-fly fire, 5.53 TSOP/s @ 300 MHz, classification. | **A** (post-absorb spike×W is exactly this MAC). Overlay/spatial FPGA + class throughput = ESTU/FireFly stack. |
| 12 | **FireFly v2** | **TCAD** 43(9) 2024 pp. 2647–2660 | 10.1109/TCAD.2024.3380550 · arXiv:2309.16158 | abs | **YES** | Spatiotemporal FPGA, 4-D parallel, **membrane-storage-free** tick, systolic 500–600 MHz, plus a **non-spike** op path so SOTA SNNs can deploy end-to-end. | **A** (tick-batch / on-the-fly). Non-spike path is **layer-typed extra op**, not PED last-use of the same producer. ESTU already has `Dense(int)` then LIF. |
| 13 | **SpikeTA** | **TCAD** 44(9) 2025 pp. 3465–3478 | 10.1109/TCAD.2025.3547275 | abs | **YES** | “First” FPGA **TSNN**: parameterized HEs, DSP-efficient adder tree for binary-spike×W, split-engine streaming, 28.99 TOPS, class speedups vs CPU/GPU. | **A** (FPGA TSNN + adder-tree AC). ESTU already claimed low-power spikeformer FPGA in **this** society’s letters. Headline TOPS is ESTU-class. |
| 14 | **SpikeX** | **TCAD** (accepted; arXiv 2505.12292) | arXiv:2505.12292 (IEEE DOI not printed on abs) | abs | **YES if “sparse SNN systolic + skip”** | Systolic array for unstructured spatial+temporal spike sparsity; HW-aware train + arch search; 15.1–150.87× EDP. | **A** (unstructured NZ spike skip). Co-design train **changes the frozen student**. One accumulate consumer. |
| 15 | **SYNtzulu** | **TCAS-I** 72(2) 2025 | 10.1109/TCSI.2024.3456552 (IEEE 10666827) | abs | **YES — ESTU’s parent** | Same Cagliari group, **same iCE40UP5K**, RISC-V + SNN PE + encode/decode; 12.05 mW / 1.45 mW avg; biosignal SNN **without** attention. | **A** (tiny-FPGA SNN SoC). ESTU = this overlay **plus SSA**. A letter that is SYNtzulu+attention **is** ESTU. |
| 16 | **sEMG 5k-LUT SNN** | **TBioCAS** 19(1) 2025 pp. 68–81 | 10.1109/TBCAS.2024.3456552 | abs | **YES on the mW/sEMG axis** | Same group, same 5k-LUT FPGA, vanilla SNN (not transformer) on NinaPro/Hyser; 11.31 mW / 83.17% / 0.875 corr. | **A** that ESTU’s **task+board** already had a TBioCAS sibling. Adding SSA on that board is exactly ESTU. |
| 17 | **SYNtzulA** | **TCAD** 45(8) 2026 pp. 3614–3625 | 10.1109/TCAD.2025.3645186 | abs | **YES if “open-source tiny SNN + skip + nJ”** | OpenROAD ASIC of SYNtzulu on IHP 130 nm; RISC-V + SNN; 36.5 pJ/SOP; biosignal. | **A** (open SNN PE). OpenROAD-as-PPA is out of SCOPE. Not transformer, not OF. |
| 18 | **da4ml** | **TRETS** 19(1) Art. 13, 2026 | 10.1145/3777387 · arXiv:2507.04535 | abs (verified) | **NO** | Exact CMVM → adder-graph / distributed-arithmetic compiler for fully unrolled FPGA NNs (hls4ml plugin); ~1/3 LUT. | **A** for compiling a **constant** T10 / lifting matrix. Does **not** serve two consumers or measure same-port 8088. Closest TRETS object to source-compile. |
| 19 | **Park TS-CIM** | **TCAS-I** 72(1) 2025 | 10.1109/TCSI.2024.3480350 | abs | **NO as ESTU; CIM-stop as title** | 65 nm time-domain CIM SNN, 701.7 TOPS/W, phase coding, no ADC. | **CIM-stop**. Analog/time-domain MAC is not digital dual last-use. Card `unresolved_Park_TCAS_I_TD_SNN_CIM.md`. |
| 20 | **RT-FLOW** | **TCAS-I** 72(11) 2025 pp. 6423–6435 | 10.1109/TCSI.2025.3566090 | abs | **NO** | FPGA OpF-SLAM (frame): correlation engine + mixed-precision flow update + Householder pose; 65 fps on XCZU7EV. | **A** that TCAS-I already hosts **frame** OF/SLAM FPGA. Not events, not SNN. |
| 21 | **LK tracking FPGA** | **TCAS-I** 70(12) 2023 pp. 4914–4927 | 10.1109/TCSI.2023.3298969 | abs | **NO** | Multi-scale Lucas–Kanade tracker on Zynq; 93 fps @ 752×480; SLAM RMSE 0.189 m. | **A** (frame LK FPGA in this society). Not the DSEC student. |
| 22 | **A-NPU** | **TVLSI** 33(7) 2025 pp. 1886–1898 | 10.1109/TVLSI.2025.3558xxx (IEEE 11010182) | abs | **YES if “edge SNN FPGA, LUT/BRAM, CIFAR %”** | First claimed spiking-SSL accelerator; 16-b ResNet on ZC702, 29k LUT / 8 BRAM, 94.08%. | **A** (tiny FPGA SNN). Classification. No transformer, no OF, no PED. |
| 23 | **speech RSNN** | **TCAS-I** (journal-ref on arXiv 2503.21337) | abs 71.2 μW / 28 nm | abs | **NO as ESTU-reskin** | Recurrent SNN speech, T=1–2, zero-skip + merged spike, 71.2 μW real-time. | **A** (zero-skip on binary spikes). Different task. T=1–2 **kills T10**. |
| 24 | **OPN** | **TCAS-II** 2024 | 10.1109/TCSII.2024.3354738 | metadata (prior local read of III-B/C/D) | **NO** | One-pass BN for **on-device training** memory. | **A** only as BN-traffic vocabulary. Training BN ≠ live full-domain **inference** BN barrier on `10×C×H×W`. Same journal, different object. |

IEEE Micro **2023–2026:** no located SNN-transformer, event-OF, or dual-consumer accelerator in this identity. Loihi (`IEEE Micro` 38, 2018) is out of window. Record as **search-null**, not as “does not exist in nature.”

---

## 2. Must-include close-ups (absorb mapping)

### 2.1 ESTU — TCAS-II 2025 (same 5-page slot)

Opened AAM. See §0 and `ADV_ESTU_same_journal.md`. Under absorb, ESTU’s skip is **legal A** on the spike path **and already published in this journal**. Dual-engine (FireFly-T), dual-side (FireFly-S), “we also skip,” a bigger Xilinx part, or tick-batch do **not** escape.

**Desk-reject script (unchanged):** *“We published ESTU in 72(12). This is another spikeformer FPGA with skip. The increment is a larger FPGA, a different classifier, or a different skip encoding. Reject.”*

### 2.2 FireFly-S — TCAS-I 2025

Local full: `p0_txts/2408.15578.txt` + card `arxiv_2408.15578.md`.  
IEEE: vol. 72 no. 8, pp. 4007–4020, pub. 15 Nov 2024, DOI 10.1109/TCSI.2024.3496554.

Joint gradient-rewire prune + SNN LSQ during training → >85% W sparsity, 4-bit W/bias/Vth. Hardware: **spatial** (not overlay) FPGA; Bitmap detector ANDs 1-bit spike vector with 1-bit W mask; silent-channel prune via bias-only V over T (Algo 1). SCNN5/7/9, T=4, MNIST / DVS-Gesture / CIFAR-10. Their FPS/W: 10047 / 3683 / 2327.

After absorb the spike path **is** binary×sparse-W, so Bitmap AND is complete **A**. It is **not** dual last-use: one IF/LIF accumulate consumer. Residual/PED never appears. Training prune is forbidden as identity for a frozen student.

**ESTU-DR = YES.** A letter that is “FPGA SNN, skip zeros, FPS/W, CIFAR/DVS” is ESTU without even the transformer. Dual-**side** in the title is a naming trap against dual-**consumer**.

### 2.3 FireFly-T — IEEE TC 2026 (10.1109/TC.2026.3672901)

Mandatory relative even though TC is not a CASS journal. Local full: `p0_txts/2505.12771.txt` + `round3_gap/literature/R3L1_firefly_t_iand.md`.  
IEEE TC vol. 75, Jun 2026, pp. 2185–2199; pub. 12 Mar 2026.

Dual-engine **overlay**: orchestrator → sparse engine (multi-lane bitmap decoder, 3-D load-balanced array, membrane accumulate) → binary engine (2-D systolic **AND-PopCount** for \(QK^\top\), \(QK^\top V\)). Residual I/O is **AXI port 2**, pre-neuron **membrane** add so convs still see spikes. Latency-hiding overlaps QKV (sparse) with attention (binary). KV260. vs FireFly v2 / SpikeTA: 1.39× / 2.40× energy, 4.21× / 7.10× DSP (**their** nets).

**ESTU-DR = YES.** ESTU already overlays binary SSA and skips. FireFly-T is ESTU’s cited high-end cousin: two **engines for two operators**, not two consumers of one producer. Pre-neuron residual is SDT/MS **A**, then LIF-binarize — ESTU already has integer then LIF.

### 2.4 ASNA-Flow — TVLSI 2025 (unresolved PDF)

IEEE abstract (11142472): Wang, Luo, Li, Zhou; TVLSI 33(12):3409–3422, pub. 26 Aug 2025, DOI 10.1109/TVLSI.2025.3600953. ResearchGate: no full text. Card `unresolved_ASNA_Flow.md` stays unresolved.

Abstract objects only:

1. hardware-aware algorithm;
2. data-pattern analysis;
3. **“novel exploitation of optical flow’s spatial locality … sparse computing”**;
4. TSMC 28 nm, **104 FPS, 7.9 mW, 0.3 pJ/SOP**;
5. “**first** dedicated neuromorphic computing solution” for temporal sparsity + event-driven + energy in OF.

**Do not copy PE/SRAM from an abstract.** Spatial-locality skip is title-level **A**; G2 already **stop as title**. “First event-OF neuromorphic” is **false against TrueNorth OF + SENECA FireNet + Kraken-SNE FireNet** and must not be recycled.

**ESTU-DR = NO** (task is event OF, not spikeformer class-%). **Collision with a different script:** a letter that is “event-driven SNN OF ASIC/FPGA, spatial skip, FPS/mW, first” is an ASNA-Flow reskin in a sister journal.

### 2.5 Liu / adaptive OF — TCAS-I 2025, DOI 10.1109/TCSI.2025.3572918

**Cards merged.** `unresolved_Liu_optical_flow_FPGA_TCAS_I_2025.md` title = *An Ultra-High Performance and Scalable Optical Flow Hardware Accelerator Based on FPGA for Autonomous Driving*. `unresolved_TCAS_I_2025_adaptive_OF.md` slogan = *Adaptive optical flow via dynamic direction prediction; reconfigurable pyramid*. IEEE 11030859, vol. 72 no. 12, pp. 7835–7847, pub. 11 Jun 2025. Authors on RG: Ye Liu, Shuang Hao, Kun Huang. **PDF unresolved.**

IEEE abstract (quoted object):

- Adaptive OF **based on dynamic direction prediction** (cut compute, keep accuracy);
- scalable PE-configurable architecture;
- **reconfigurable pyramid-layer pipeline** (perf + memory);
- Xilinx **ZCU104, 405 FPS, AEE 1.02** vs SOTA hardware accelerators.

**This is FRAME optical flow.** Image pyramid, autonomous driving, AEE on (implicitly Middlebury/KITTI-class) **frames**. Not DVS, not SNN, not transformer. The AEE 1.02 number is **not** DSEC valid825 and must not be stacked against 1.219 / 1.259.

**ESTU-DR = NO.** A letter that looks like Liu is not an ESTU reskin; it is a **TCAS-I frame-OF FPGA** reskin. Selling 405 FPS / AEE 1.02, or “FPGA optical flow,” as the 5-page story **collides with this DOI**, not with ESTU.

### 2.6 Xpikeformer — TVLSI 2025

IEEE TVLSI 33(6):1596–1609, Jun 2025, DOI 10.1109/TVLSI.2025.3552534. arXiv:2408.08794v2 (3 Apr 2025). Local `p0_txts/2408.08794.txt` + method card `arxiv_2408.08794.md`. Stub card `unresolved_Xpikeformer.md` must **not** be treated as unread once the arXiv full text is in `p0_txts`.

AIMC for FFN/FC; **stochastic spiking attention**: Bernoulli streams, AND in place of QK multiply; BNL replaces LIF on the attention path. Tasks: image classification + wireless symbol detection. 13× energy vs SOTA digital ANN transformer at similar throughput; up to 1.9× vs **projected** digital SNN-transformer.

**ESTU-DR = NO** (analog hybrid ≠ 5k-LUT digital overlay). **CIM-stop as title.** Stochastic attention is **B** against bit-exact DSEC AEE. Hybrid analog-digital is **not** dual last-use of binary GeMM + PED.

### 2.7 LoopTree — TCAS-AI 2024

Local author full text. Gilbert, Wu, Emer, Sze. DOI 10.1109/TCASAI.2024.3461716. TCAS-AI vol. 1 no. 1, pp. 97–111. Card `looptree_tcasai2024.md`.

Not an accelerator. Design-space **model**: tiling ranks × retain vs recompute of intermediate fmaps × per-tensor retain × sequential vs pipeline-hidden latency; Accelergy energy; Buffet-style orchestration. Worst-case 4% model error vs prior fused designs; up to ~10× buffer for same off-chip traffic (**their** case studies).

**ESTU-DR = NO.** Useful **A** as the language for “who stays in the two-port RF until **both** binary GeMM and PED complete.” LoopTree does not choose the fusion set, has no spike/NRV, and assumes stalls can be ignored — **B** against finite 8088 backpressure if used as a PPA oracle.

### 2.8 SNE family — DATE 2022 + Kraken

SNE: Di Mauro et al., DATE 2022, DOI 10.23919/DATE54114.2022.9774552, arXiv:2204.10687. Local full `p0_txts/2204.10687.txt`. Card `arxiv_2204.10687.md`. 4-bit eCNN, energy ∝ events, IBM-DVS-Gesture, 80–261 μJ/inf at 1.2–4.9% activity, 0.221 pJ/SOP. OPs: RST / UPDATE / FIRE. Slice + C-XBAR + filter buffer.

Kraken (22 nm FDX SoC; ISSCC-adjacent, not this venue list): SNE as a block; abstract states SNE **can** run a 4-layer 4-bit LIF-FireNet **per-pixel optical flow** (98 mW @ 222 MHz; activity-dependent inf/s). That is **already event-OF on SNE**, classification-engine reused — not a CASS letter, not dual-consumer.

**ESTU-DR = NO** (DATE/SoC, not spikeformer FPGA letter). **A** for event-proportional conv. FIRE vs UPDATE is two uses of **one membrane**, not GeMM vs PED.

### 2.9 TrueNorth OF — TBioCAS 2018 (required, out of year)

Haessig, Cassidy, Alvarez, Benosman, Orchard. TBioCAS 12(4):860–870. DOI 10.1109/TBCAS.2018.2834558. arXiv:1710.09820. **PDF unread this pass; abstract used.**

Fully spike-based Barlow–Levick OF from DVS/ATIS on TrueNorth; 1 ms tick; ~11% average endpoint error; <80 mW sensor+compute.

**ESTU-DR = NO.** **B** for any “first event-based optical-flow hardware” sentence. Not a transformer. Not dual last-use.

### 2.10 Plane-fit — ISCAS 2018 (required, out of year)

Aung, Teo, Orchard. DOI 10.1109/ISCAS.2018.8351588. VHDL+Matlab at `gorchard/FPGA_event_based_optical_flow`. **PDF unread;** EventShiftFlow related-work paraphrase: least-squares plane fit, division/sqrt pipelines, 100 M 3×3 fits/s, sub-μs.

**ESTU-DR = NO.** **B** for first-event-OF-FPGA. Classical plane-fit, not SNN-Transformer, not DSEC dense AEE.

---

## 3. Venue enumeration 2023–2026 (what the scan actually found)

Queries (2026-09-11): IEEE/arXiv/Scholar for `{TCAS-I, TCAS-II, TVLSI, TCAD, TBioCAS, IEEE Micro, TRETS} × {2023,2024,2025,2026} × {spiking, SNN, spikeformer, transformer accelerator, optical flow FPGA, event-based OF, neuromorphic}`. Local catalog (`literature_merged_300plus.csv`, W1, R3L*) used as a **checklist**, not as invented papers. Mechanism lines from opened full texts or publisher abstracts.

### 3.1 TCAS-II (target; 5-page collisions)

| Paper | Year | Depth | Notes |
|---|---|---|---|
| **ESTU** | 2025 72(12) | full AAM | **The** same-slot control. |
| OPN one-pass BN | 2024 | prior local body | Training-memory BN. Not inference dual-consumer. |
| (null) SNN-Transformer OF | 2023–2026 | — | **No other** TCAS-II letter located that is event-OF SNN-Transformer. |

A 2026 TCAS-II letter in this identity will be stacked against ESTU first, then SYNtzulu (TCAS-I, same group/board).

### 3.2 TCAS-I

| Paper | Year | Depth | ESTU-DR if letter looks like it |
|---|---|---|---|
| SYNtzulu tiny RISC-V SNN FPGA | 2025 72(2) | abs | **YES** (ESTU parent) |
| FireFly-S dual-side spatial FPGA | 2025 72(8) | full | **YES** |
| Liu adaptive pyramid OF, 405 FPS / AEE 1.02 | 2025 72(12) | abs **unresolved PDF** | **NO** (frame OF) |
| RT-FLOW OpF-SLAM FPGA | 2025 72(11) | abs | **NO** |
| Park TS-CIM 701.7 TOPS/W | 2025 72(1) | abs | CIM-stop |
| Analog subthreshold SNN reservoir | 2025 10.1109/TCSI.2025.3550876 | abs | analog-stop |
| Speech RSNN 71.2 μW | 2025 (arXiv 2503.21337 journal-ref) | abs | NO as ESTU; T=1–2 kills T10 |
| LK OF tracking FPGA | 2023 70(12) | abs | **NO** |
| Clock-free multilevel-event SNN CIM | 2023 | metadata | CIM-stop |
| Dyn-Bitpool two-sided sparse CIM | 2025 10.1109/TCSI.2025.3547001 | metadata | CIM-stop; dual-**side** name trap |

### 3.3 TVLSI

| Paper | Year | Depth | ESTU-DR |
|---|---|---|---|
| **ASNA-Flow** event-OF neuromorphic | 2025 33(12) | abs **unresolved** | NO as ESTU; YES as first-OF-HW / spatial-skip OF |
| **Xpikeformer** AIMC+SSA | 2025 33(6) | arXiv full | NO as ESTU; CIM-stop as title |
| FireFly DSP48 spike×W | 2023 31(8) | abs | **YES** |
| A-NPU spiking SSL FPGA | 2025 33(7) | abs | YES if LUT/class-% story |
| Spatio-temporal redundancy SNN | 2024 32(4) 10.1109/TVLSI.2023.3335232 | abs | YES if skip-redundancy + TOPS/W |
| 3-D stacked SENECA memory | 2024 32(11) | abs | packaging; not a letter object |
| ESSA sparse SNN FPGA | 2022 30(11) *(border)* | abs | YES if sparse-SNN FPGA |

### 3.4 TCAD

| Paper | Year | Depth | ESTU-DR |
|---|---|---|---|
| FireFly v2 spatiotemporal + non-spike ops | 2024 43(9) | abs | **YES** |
| SpikeTA FPGA TSNN | 2025 44(9) | abs | **YES** |
| SpikeX systolic unstructured sparsity | accepted / 2025 arXiv | abs | **YES** if sparse-SNN skip |
| SYNtzulA open ASIC of SYNtzulu | 2026 45(8) | abs | YES if tiny-SNN + nJ |
| Minifloat MACC Versal | 2025 44(6) | metadata | NO (ANN arithmetic) |

### 3.5 TBioCAS

| Paper | Year | Depth | ESTU-DR |
|---|---|---|---|
| **TrueNorth OF** (required) | **2018** | abs | NO; **B** for first-OF-HW |
| sEMG SNN on 5k-LUT FPGA (Scrugli/Leone/Meloni) | 2025 19(1) | abs | **YES** on board+task+mW (ESTU sibling) |
| (null) 2023–2026 event-OF **transformer** | — | — | None located. |

### 3.6 IEEE Micro

**Search-null 2023–2026** for SNN-transformer accelerators and event-OF hardware. Historical Loihi (IEEE Micro 2018) is the name reviewers still know; it is not a 2023–2026 hit and not a 5-page CASS letter. Do not pad the table with Micro survey/product pieces.

### 3.7 TRETS

| Paper | Year | Depth | ESTU-DR |
|---|---|---|---|
| **da4ml** CMVM adder-graph | 2026 19(1) Art. 13 | abs | **NO**. Strongest TRETS **A** for T10 compile. |
| REATA ViT on Versal ACAP | 2026 19(1) Art. 12 | abs | NO (ANN ViT) |
| TernaryGNNs CPU-FPGA | 2026 19(1) Art. 2 | abs | NO |

No TRETS 2023–2026 SNN-transformer or event-OF accelerator located.

### 3.8 Adjacent (not in the seven venues; already mandatory or already stopped)

FireFly-T **IEEE TC 2026** (§2.3). SNE **DATE 2022** / Kraken SoC. Plane-fit **ISCAS 2018**. LoopTree is TCAS-**AI**, listed because required. ERAFT FPGA / ROFD are **ISCAS 2025**, not this sheet’s venues (see R3W1).

---

## 4. Local idea cards — status after this pass

| Card | After V4 |
|---|---|
| `ADV_ESTU_same_journal.md` | **Opened AAM remains the control.** ADV still narrates continuous \(\theta g\); **collision object unchanged** under absorb. Use this sheet’s absorb rewrite when citing ADV. |
| `unresolved_Liu_optical_flow_FPGA_TCAS_I_2025.md` | **Same DOI as next row.** Abstract now filled (405 FPS, AEE 1.02, **frame** pyramid). Body **still unresolved**. Do not copy PE counts. |
| `unresolved_TCAS_I_2025_adaptive_OF.md` | **Merge into Liu.** Not a second paper. |
| `unresolved_Xpikeformer.md` | **Superseded for mechanism** by `arxiv_2408.08794.md` + local `p0_txts/2408.08794.txt` + IEEE 10.1109/TVLSI.2025.3552534. Keep the stub only as a pointer. |
| `unresolved_ASNA_Flow.md` | Still **no PDF**. Abstract objects recorded. Spatial-locality skip = title A / G2 stay stopped. |
| `unresolved_FPGA_plane_fitting_event_optical_flow.md` | Still **no PDF**. IEEE abs + EventShiftFlow paraphrase only. |
| `looptree_tcasai2024.md` | Full local. A as analyzer, not X. |
| `arxiv_2204.10687.md` (SNE) | Full local. A as event-proportional conv. |
| `arxiv_2408.15578.md` (FireFly-S) | Full local. A then stop. |

---

## 5. Would ESTU’s reviewer desk-reject *our* letter if it looked like X?

Read the left column as “if the 5-page PDF is indistinguishable from …”

| If the letter looks like … | ESTU reviewer |
|---|---|
| ESTU / SYNtzulu / TBioCAS sEMG 5k-LUT / SYNtzulA | **Desk-reject.** Same group’s object, now with SSA, already in TCAS-II 72(12). |
| FireFly / FireFly-S / FireFly v2 / FireFly-T / SpikeTA / SpikeX / A-NPU | **Desk-reject.** Binary spike FPGA/ASIC, skip, overlay or spatial, class-% or TSOPS/W. FireFly-T dual-engine and FireFly-S dual-side are **name collisions**, not increments. |
| Xpikeformer / Park CIM / Dyn-Bitpool | **Not ESTU**; **out of identity** if analog/CIM is the title. Cite and drop. |
| ASNA-Flow | **Not ESTU.** Different desk-reject: “TVLSI already ran event-driven neuromorphic OF with spatial-locality skip and claimed first.” Need dual last-use + DSEC AEE + no “first.” PDF still missing — do not copy their datapath. |
| Liu 405 FPS / RT-FLOW / LK tracking | **Not ESTU.** Different desk-reject: “TCAS-I already has frame OF FPGA with AEE/FPS.” Event SNN-Transformer OF is a **different object**; do not borrow 405 FPS or AEE 1.02. |
| TrueNorth OF / plane-fit / SNE-FireNet | **Not ESTU.** They **forbid** “first event-OF hardware.” They do **not** close spikeformer + PED last-use. |
| LoopTree / da4ml / OPN | **Not ESTU.** Tools/compilers/BN. Using them as the **title** is a different (weaker) reject: “this is an analysis/compiler letter.” Using them as **A** for occupancy or T10 compile is allowed. |

**Allowed 5-page object (not found in this venue window):** one compiled source whose **post-absorb binary GeMM** and **continuous PED/I24** are both live, retired on `gate ∧ PED`, measured on DSEC valid825 inside the AEE gates, with same-port net service — **and** the abstract cannot be rewritten as ESTU, FireFly-T, SpikeTA, or ASNA-Flow.

No 2023–2026 paper in TCAS-I/II, TVLSI, TCAD, TBioCAS, IEEE Micro, or TRETS was located that already implements that object. ASNA-Flow’s **body** could still hide a surprise; it is **unresolved**, not a license to claim the hole.

---

## 6. Unresolved / search-incomplete (do not close by slogan)

| Item | Missing | Risk if faked |
|---|---|---|
| ASNA-Flow PDF | PE, SRAM, how “spatial locality” is skipped, whether any residual/BN exists | Inventing dual last-use **or** inventing a skip unit |
| Liu / adaptive OF PDF | Exact dataset for AEE 1.02, PE microarch, pyramid depth | Treating 405 FPS / 1.02 as event-SNN numbers |
| TrueNorth OF / plane-fit PDFs | Body equations, mapping | Over-claiming “we read the silicon” |
| SpikeX IEEE issue/pages | Volume not on arXiv comments | Fine at arXiv grain |
| IEEE Micro 2023–2026 SNN-transformer | **Null this pass** | Padding with Loihi 2018 as if it were new |
| ESTU GitHub tree / IEEE HTML | Code layout | Fine: AAM is enough for the collision object |

---

## 7. One-page takeaway for the 5-page letter

CASS journals 2023–2026 already contain:

1. **The same 5-page object ESTU** (binary spikeformer FPGA overlay + skip + mW + class-%), with a TCAS-I parent (SYNtzulu) and a TBioCAS sibling (sEMG 5k-LUT).
2. **The FireFly stack** across TVLSI / TCAD / TCAS-I / IEEE TC: DSP spike×W, tick-batch, dual-side Bitmap, dual-engine AND-PopCount attention.
3. **FPGA TSNN** as SpikeTA (TCAD) — ESTU’s high-end relative.
4. **Event-OF neuromorphic as a TVLSI paper** (ASNA-Flow, abstract-level spatial skip, “first” — **unresolved body**).
5. **Frame-OF FPGA as TCAS-I papers** (Liu 405 FPS AEE 1.02; RT-FLOW; LK 2023) — **not** our task.
6. **Hybrid analog spikeformer** as TVLSI (Xpikeformer) — illegal as title.
7. **Fused-layer retain/recompute** as TCAS-AI LoopTree and **CMVM compile** as TRETS da4ml — tools, not the letter.
8. **TrueNorth OF (TBioCAS 2018) and plane-fit (ISCAS 2018)** as the historical event-OF hardware floor.

After AT-LIF absorb, **Prosperity / Gustav / FireFly-S Bitmap / ESTU group-skip are all legal A on the spike path** and therefore **cannot be the contribution**. The only object that does not already sit in this reviewer pool as ESTU-or-FireFly-or-ASNA is dual last-use of that binary path **together with** a separate continuous PED, on DSEC, under same-port backpressure.

**Cite ESTU in the relative-prior paragraph. If the abstract still reads as ESTU, do not send.**
