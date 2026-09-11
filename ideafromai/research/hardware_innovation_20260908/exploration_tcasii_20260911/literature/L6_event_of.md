# L6 — Event optical-flow mechanisms vs patch-r1 PRODUCER support

Freeze: 2026-09-11. Full local texts under
`/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/survey_ab_fusion_20260910/p0_txts/`.
This note answers one question. It is not a paper claim, not Stage-B schedule evidence, and not a first-event-OF-HW pitch.

**Question.** Which algorithm mechanisms can change **PRODUCER support of patch r1** enough to **pay hardware**, without using **current-frame final flow as scheduler oracle**, and without claiming **first event-OF HW**.

**Local B (from `PROBLEM.md` / `SCOPE.md`).** Patch-embed residual r1 = two convs + noncausal T10 PSN; dual consumers after source (spike/gate path and continuous residual/PED path); native projection conv + BN + residual add. Neuron is ATLIF with continuous θg. Expensive region historically ~patch 35% of an old activity-weighted-dot proxy; proxy ≠ new student’s cycle share. Valid825 ordinary AEE 1.219801338; lifting raw 1.232979368. Same-port net service of the full r1 chain is still UNKNOWN.

**PRODUCER support** here means the trained spatial–temporal–channel set of non-zeros that r1 must actually produce: sn1/sn2 θg, Conv1/Conv2, identity, and whatever the continuous PED/`conv_res` still demands. Hardware pays only if that support is sparse or clustered enough that skip + grouping + state retirement beat dense always-on production under the same-port / same-state / same-backpressure contract, after paying mask, halo, dual-consumer union, full-domain BN, and T10 completion.

**Two hard bans.**
1. Do not schedule current-frame r1 from that same frame’s **final** flow (r1 is upstream of the flow head; using the answer to skip the producer is an oracle). Intermediate same-frame flow used as a quality map is the same class of oracle unless it is strictly causal (past events / previous frame only) and is paid.
2. Do not write “first event optical-flow hardware.” TrueNorth OF, FPGA plane-fit OF, and SENECA FireNet OF already exist in these texts.

---

## 0. GPU-accuracy vs hardware (do not mix denominators)

| arXiv | Short | Venue identity | What was actually measured | Class |
|---|---|---|---|---|
| 1802.06898 | EV-FlowNet | RSS 2018 | MVSEC AEE; 40–48 ms forward on GTX 1050 | **GPU accuracy.** No accelerator, no skip schedule. |
| 2003.06696 | Spike-FlowNet | ECCV 2020 | MVSEC AEE; encoder spike rate; Horowitz 45 nm AC/MAC *estimate* | **GPU accuracy + paper energy model.** Not measured HW. Claims “first SNN SOTA” on event OF — do not recycle. |
| 2404.08135 | SciFlow | CVPRW 2024 | Sintel/KITTI EPE/Fl-all; Snapdragon 8 Gen 3 HTP INT8 latency/power of a **frame** OF net | **GPU accuracy + on-device whole-net latency.** Not event, not r1 support, not SNN. |
| 2503.03256 | BAT | AAAI 2026 | DSEC-Flow / MVSEC EPE; RTX 3090 training | **GPU accuracy.** Correlation/iteration head. No HW. |
| 2407.20421 | SENECA ANN vs SNN | Neural Networks 2025 | FireNet on SENECA; GF-22 nm FDX gate-level Xcelium/JOULES; 56² and 120² | **The hardware paper.** Already event-OF on a neuromorphic processor. |
| 2107.07305 | Delta Activation Layer | arXiv 2021 | UCF101 ResNet-50/MobileNet sparsity vs accuracy; memory-footprint arithmetic | **Video-DNN algorithm + HW-cost discussion.** Not event OF, not a chip. |

Spike-FlowNet’s 214× encoder “energy benefit” is `AC=5.1×MAC` from Horowitz ISSCC 2014, not a chip. SciFlow Table 5 is phone-HTP latency of MobileFlow, not producer-support service. SENECA’s ms/μJ are the only numbers in this set that may be cited as hardware-in-loop, and they are FireNet 32-ch 3×3 on 56/120, not Motion C12 / H67 / DSEC 240×320 T10 dual-consumer r1.

---

## 1. EV-FlowNet (1802.06898) — GPU accuracy

**Mechanisms (method).** 4-channel event image = pos/neg counts + latest timestamps; timestamps normalized by the window so fast-small and slow-large windows look similar; U-Net + residual; photometric warp of grayscale + smoothness; AEE only on pixels that saw ≥1 event.

> “The first two channels encode the number of positive and negative events that have occurred at each pixel, respectively.” … “we encode the pixels in the last two channels as the timestamp of the most recent positive and negative event at that pixel.” (§III-A)

> “we normalize the timestamp images by the size of the time window … ensuring that fast motions with a small time window and slow motions with a large time window that generate similar displacements have similar inputs.” (§III-A)

> “as the input event image is relatively sparse, the network only returns accurate flow on points with events. As a result, we limit the computation of AEE to pixels in which at least one event was observed.” … “this results in the error being computed over 20-30% of the pixels.” (§V-E)

> “In practice, these problems can be avoided by choosing time windows large enough so that sufficient information is available while avoiding saturating the event image. One possible solution … would be to have a fixed number of events in the window each time.” (§V-F)

> “A single forward pass takes, on average, 40ms for the smaller network, and 48ms for the larger network, when run on a NVIDIA GeForce GTX 1050.” (§V-B)

**Can it change r1 PRODUCER support?** Weakly, and only at the **input representation**, not inside r1.

- Event-count / timestamp / window-N changes which pixels carry signal into `patch_embed`. That can change stem/r1 activity **if the student is retrained on that representation**. It is not a scheduler.
- The 20–30% event mask is an **evaluation mask**, not a producer skip. The CNN still runs dense on the 4-channel image. “Predict zero where no events” is a photometric consequence, not a hardware retirement of Conv2/PSN.
- Counts vs timestamps ablation: counts-only is worst; timestamps carry order/speed; combined is best. Counts “carry information about the importance of each pixel. Pixels with few events are likely to be just noise.” That is a **cheap input prior** (event count, not flow) that could gate first-layer production. It is A, not X. Kill-gate: a static “skip r1 where input event count = 0” control must be given for free to the ordinary student; if that already eats the skip, training a fancier mask does not pay.

**Oracle?** Photometric loss uses grayscale frames at **train** time only. Inference is events-only. No current-frame flow oracle.

**HW?** None. Do not convert 40 ms GPU into same-port %.

---

## 2. Spike-FlowNet (2003.06696) — GPU accuracy + Horowitz energy

**Mechanisms.** Sequential 4-channel event frames through an IF encoder; last SNN layer **accumulates and does not fire**; residual+decoder stay ANN because of vanishing spikes and because regression needs analog precision.

> “deep SNNs suffer in terms of performance due to the spike vanishing phenomenon.” (Abstract / §1)

> “To the best of our knowledge, this is the first SNN demonstration to report the state-of-art performance on event-based optical flow estimation.” (§1) — **ban recycling this “first” sentence for HW.**

> “only the encoder block is built as an SNN, while the residual and decoder blocks maintain an ANN architecture.” (§3.4)

> “The outputs from encoder layers are collected in their corresponding output accumulators until all consecutive event images have passed.” (§3.4)

> “the final SNN layer neurons just integrate … while not producing any spikes at the output.” (Alg. 1 / §3.5)

> Encoder mean spike activity “0.48% and 1.01% for dt = 1 and dt = 4.” Overall compute-energy reduction vs full ANN “~17.5%” after applying Horowitz 5.1× AC/MAC; encoder-only 214× / 25.5×. “the proportion of required computations in encoder-block compared to the overall architecture is 17.6%.” (§4.4 / Table 2)

> Ablation: converting residual blocks to SNN **worsens AEE** (Spike-FlowNet 1R/2R vs encoder-only). “the spike vanishing phenomenon aggravates with the network depth.” (Supp. §6.1)

**Can it change r1 PRODUCER support?** Encoder IF threshold/N changes **early** spike support. r1 is a **residual block of two convs**, i.e. the layer class they **refused** to spike because AEE died.

- Allowed borrow: IF / ATLIF threshold as a trained producer of sparse θg; sequential T bins rather than one concatenated voxel. Local net already has T10 and continuous θg — binary IF is **not** the identity (`PROBLEM.md`: do not change to binary ATLIF).
- Accumulator-until-all-N-pass is the **opposite** of early retirement: they wait for the full window before the ANN residual sees a vector. That is a dense handoff, not a support shrink of the residual itself.
- ~17% whole-net energy cut is an estimate on EV-FlowNet-shaped U-Net, and it comes from the encoder, not the residual. Local r1 **is** the residual. Their residual-SNN ablation is a **negative** prior: spiking r1’s two convs in the Spike-FlowNet style is likely to fail AEE.

**Oracle?** Self-supervised grayscale photometric, train only.

**HW?** None. “TrueNorth / Loihi” appear as motivation, not this paper’s measurement. Haessig et al. TrueNorth OF is cited as related shallow SNN, not this architecture.

---

## 3. SciFlow (2404.08135) — GPU accuracy + phone HTP (frame OF)

**Mechanisms.** Self-Cleaning Iteration: warp F2 by the **current iteration’s flow**, Gaussian similarity SCI map, concat into ConvGRU. Regression Focal Loss: GT-vs-pred confidence, heavier weight on hard pixels; **final-iteration** map applied to **all** iterations at train. SCI+RFL add “negligible to zero” params/latency.

> “in each iteration, we compare the feature maps of the two frames using the current estimated optical flow and warping. The pixel-wise differences provide an indication for consistency of the optical flow.” (§1)

> “SCI only requires the network to process an additional channel of the quality map and RFL only affects loss computation during training.” (§1)

> “Having the confidence map ready, we apply the map to li … This RFL-based confidence weighting is derived by the final iteration of prediction and applied li of all iterations.” (§3.2)

> On-device, Snapdragon 8 Gen 3 HTP, INT8, 6 iterations: MobileFlow 29.02 ms / 392 mW; MobileFlow+SCI+RFL 28.88 ms / 393 mW. RAFT-S OOM because of all-pairs cost volume. (§4.2.6 / Table 5)

**Can it change r1 PRODUCER support?** No.

- SCI **adds** a quality channel to the iterative **head**. It does not skip, mask, or re-support an early residual conv.
- SCI’s quality map is computed from **same-frame current flow**. Using that map to skip current-frame r1 is exactly the forbidden oracle (even if the flow is intermediate rather than “final”; r1 has already been produced before the first iteration).
- RFL is train-only GT weighting. Optional “RFL as confidence at inference” (Fig. 6) still needs GT or a flow estimate — not a producer prior.
- Lightweight backbone (MobileNetV2 + coarse cost volume) is ordinary A. Phone latency is whole-net, 6 GRU iterations, Sintel/KITTI, not DSEC r1.

**Oracle?** Yes, if used as a scheduler. SCI is a same-frame flow-consistency oracle by construction. **Stop SCI-as-r1-gating.** Keep RFL only as a training reweight control (uses GT, not a runtime oracle).

**HW?** On-device HTP of a frame OF net. Not event-OF HW. Compiler luck even *reduced* latency 0.48% — cannot be a same-port story.

---

## 4. BAT (2503.03256) — GPU accuracy (DSEC head)

**Mechanisms.** Split reference/target voxels into N groups; bidirectional temporal correlation (forward to future groups, backward to past groups); linear motion `df = f/N` to warp; learnable sampling radius `lr = α·r`; SATMA = deformable sparse attention + spatial sigmoid gate to fuse consistent motion, suppress inconsistent; ConvGRU updates; **future flow from past events only** via backward correlation.

> “Given two consecutive event streams E(ti−1, ti) and E(ti, ti+1) … our goal is to estimate the optical flow f_ti→ti+1.” Then voxels are split into N groups. (§ Feature Extraction)

> “we assume that the optical flow is linear with respect to time. Thus, we can derive the optical flow df between adjacent temporal groups as: df = f / N.” Then forward/backward correlation warps with `j·df`. (§ BTC)

> “Our BAT (bwd corr) can predict future optical flow f_ti→ti+1 using only the past event stream E(ti−1, ti).” Table 5: only-Ref BAT 1.163 EPE vs E-RAFT warm-start 4.518. (§ Future Optical Flow Prediction)

> “Limitations. Our BAT does not bring significant gains under rapid motion changes. … there is a significant difference between backward and forward temporal motion.” (Conclusion)

> Context net “takes event frames from timestamp ti − Δt/N to ti+1.” (Appendix, iterative update)

**Can it change r1 PRODUCER support?** Not the correlation head. One **causal** prior is usable.

- BTC/SATMA/ATS live **after** features exist. They change how a RAFT-style updater consumes motion, not which r1 conv locations are produced. Local identity is T10 PSN + dual PED, not ConvGRU correlation. Borrowing SATMA as an r1 skip is a reskin.
- `df = f/N` warping **uses the current iterative flow of the frame being solved**. That is an in-head oracle, acceptable inside a correlation updater, **forbidden** as a gate on the same frame’s r1 producer.
- **Allowed:** Table 5 backward-only future flow. Scheduler input is **past** events E(ti−1, ti), not current-frame final flow. That prior can, in a later frame, cheaply hint *where motion will be*. It still does not skip r1 unless a **trained student** actually changes r1 support when conditioned on that prior, and the prior’s feature/corr cost is paid. Warm-start of E-RAFT (previous flow as init) is the strong control; BAT’s own limitation is violent shake (bwd ≠ fwd).
- Adaptive radius α is a learned correlation hyperparameter, not producer support.

**Oracle?** Forward BTC uses current f. Backward-only future prediction does not. Do not smuggle current f into r1.

**HW?** None. DSEC 1PE −39% vs E-RAFT is GPU accuracy. Not same-port %.

---

## 5. SENECA ANN vs SNN (2407.20421) — hardware paper (already event-OF HW)

**Mechanisms that actually change support.** Channel-wise trainable FATReLU / LIF thresholds; L1 on membranes + L2 on 1/T²; ANN threshold init = per-channel median of a dense teacher (50% target), then fine-tune; replace dense TanH hidden state in ConvRNN by FATReLU so the recurrent tensor is sparse. **Mechanisms that consume support, not change it:** event-driven depth-first conv; spatial-order fire-when-RF-complete; ac/sp grouping = 4 (one membrane load/store per up-to-4 events at the same pixel); weights **not** reused.

> “The ANN and the SNN for comparison have similar low activation/spike density (∼5%) thanks to our novel sparsification-aware training.” … “SNN’s higher efficiency attributes to its lower pixel-wise spike density (43.5% vs. 66.5%) that requires fewer memory access operations for neuron states.” (Abstract)

> FATReLU: `x · 1[x > T]`, T channel-wise, surrogate as in LIF. Sparsification loss `Ls = Σ_i λ_i · ( Σ ReLU(x_i) + Σ_j 1/T_{i,j}² )`. Densest layer gets `λ_i = 2` so one dense layer cannot bottleneck the pipeline. (§4.1–4.2, Eq. 1–2)

> “SENECA reuses neuron states by … ac./sp. grouping. … We set the group size to four. Firstly, the neuron states are loaded … four ac./sp. are integrated … stored back. … If there are fewer than four … dummy zero ac./sp. fill(s) the group. … Weights are not reused by our processing mechanism.” (§3.2)

> Controlled 16×16 layer: time linear in **number of pixels that have ≥1 ac/sp**, not in nnz alone. Going from 1→4 events/pixel is cheap (one group); 4→6 jumps because a second group appears. “higher neuron density does not necessarily mean more time cost, the spatial distribution of ac./sp. is an important factor.” (§6.1 / Fig. 3)

> FireNet 56², selected average-density frames: SNN 44.9 ms / 927 μJ vs ANN 71.8 ms / 1233 μJ → 62.5% time, 75.2% energy. Pixel density 43.5% (SNN) vs 66.5% (ANN) at similar ~5% neuron density. 120²: SNN 225.77 ms / 4174 μJ vs ANN 326.39 ms / 5753 μJ. GF-22 nm FDX, 0.8 V, 25 C, 500 MHz, Xcelium+JOULES, ±15% of signoff, I/O excluded. (§6.3 / Table 3)

> ANN 3×3 depth-first can **release** membrane after last RF event and keep ~3 rows; LIF **cannot**: “the memory for the neuron states cannot be released like ANN.” Recurrent FireNet blocks force ANN to keep hidden state too, so max resolution matched. Inputs to layer-0 must be spatially sorted; sort overhead “not counted … because it is the same for ANN and SNN.” Depth-first needs in-order arrival. (§ App. D)

> “it is still only a lightweight network without state-of-the-art accuracy. The obstacle is our current limited capacity of mapping bigger-size networks.” (§7)

**Can it change r1 PRODUCER support?** **Yes — this is the only paper in the set that both (i) trains support and (ii) measures that support on hardware.**

What transfers as **A** (complete prior, not X):

1. **Train thresholds + membrane/threshold regularizers until neuron density is ~5% and no single layer is the dense bottleneck.** This changes which r1 channels fire without reading flow. Local ATLIF already has continuous θg and a threshold; the increment is a **sparsity-aware loss on the actual dual consumers**, not FATReLU as a name.
2. **Pixel-clustered vs uniform nnz.** Same 5% neuron density, 43.5% vs 66.5% pixel density, different membrane traffic. A student that only reports “spike rate down” has not paid hardware. Kill-gate must log **pixel density, events-per-pixel histogram, and group=4 dummy fills**, not just nnz.
3. **Do not treat grouping=4 or depth-first as X.** They are SENECA mapping. Local T10 is **noncausal**: you cannot fire-and-release after the last spatial RF event the way SENECA ANN does, because all ten times are still live for PSN. LIF-style “keep all membranes” is the closer of their two contracts, and it is the **expensive** one.

What does **not** transfer:

- FireNet 32-ch, no downsample, MVSEC dt=4 AEE ~4, 56/120 because of on-chip state. Not C96 3×3 r1 on 240×320, not dual PED, not full-domain BN over 10×96×120×160.
- Binary spike vs local continuous θg. SENECA ANN activations are analog above T; SNN is binary. Local residual PED consumes **amplitude**. Sparse binary gates do not automatically skip `conv_res(r1out)`.
- “SNN more efficient than ANN on the same core” is **their** comparison at matched ~5% density. It is not a license to claim first event-OF HW, nor that swapping ATLIF for LIF wins r1.
- Weight non-reuse is their mechanism **boundary**. Gustav-style W broadcast is a different contract; do not add grouping on top and call both wins.

**Oracle?** None. Thresholds and regularizers never see flow.

**HW?** Yes, and therefore **the “first event-OF HW” claim is already taken.** Cite as relative prior. Incremental X, if any, is dual-consumer T10 + continuous PED under a SENECA-style sparsity loss, measured on **our** same-port r1 chain — not “we also ran OF on a neuromorphic chip.”

---

## 6. Delta Activation Layer (2107.07305) — video algorithm, HW-cost discussion

**Mechanisms.** Replace activation by quantized `fq(Z,q)=round(Z/q)·q`; train q (channel-wise) with L1 on `|ΔO|`; at inference, propagate `ΔO = O(t)−O(t−1)` and reconstruct `Z(t)=ΔZ(t)+Z(t−1)`. No level-crossing threshold, so quantized dense and quantized-delta match **exactly** (no drift, no periodic reset). Casts temporal sparsity into spatial zeros for sparse MAC. **Selective** layers: skip sigma or delta at dense/delta boundaries; residual ADD junctions are the **least** temporally sparse.

> “A Delta Activation Layer casts temporal sparsity into spatial activation sparsity to be exploited when performing sparse tensor multiplications in hardware.” (Abstract)

> “there is however no threshold or PID parameters to be learned. Instead a generic activation quantization method is used.” … “When using quantization … quantized inference and quantized delta-inference provide exact same results and therefore no error will be accumulated over time.” (§1, §2.2)

> Related AMS (Buckler ISCA 2018): “using optical flow information from P-frames to do motion compensation directly on the activation maps of previous frames.” (§1.1) — **this is a flow-oracle scheduler. Ban if the flow is current-frame.**

> ResNet-50 UCF101: baseline 73% / 49.1% op sparsity; spatial L1 65.4% / 65.3%; temporal DAL 67.6% / 93.1% (~3× vs spatial). Full DAL memory 41 MB vs 25 MB dense (~+40% neurons stored). “using full Delta Activation Layers in ResNet-50 results in 40% increase in the usage of memory. We suggest partial use of stateful layers.” (§3 Table 1, §4)

> Residual ADD after skip: “layers where skip connections are merging … exhibit the lowest temporal sparsity in their neighbourhood.” (Fig. 6)

> If on-chip SRAM is only 1 MB, delta can force **weights and states** into DRAM: “three times more external memory accesses … almost equal to three to five times more energy.” Sparse MAC “on average, by a factor of five” vs spatial L1 can still lose energy. (§4)

> Input-delta-only (no state) “is equivalent to providing inputs from a DVS sensor.” Deeper temporal sparsity remains high even with a moving camera, because high-level features change slowly. (§3, Fig. 8)

**Can it change r1 PRODUCER support?** Yes **if trained**, and only on **activation deltas**, not on flow-warped maps.

- Allowed: train q / Δ-penalty so r1’s θg or conv outputs are temporally redundant across T or across frames, then skip MACs on exact zeros. No flow in the loop. This is the mechanism MotionDeltaCNN/DeltaCNN already name; 2107.07305 adds **trainable q + exact quantized-delta equivalence + selective layers**.
- Local **negative layout** (README / motion probe): untrained causal motion-compensated exact residual of real adjacent frames was **1.529× more work**, plus 294,912,000 B of old continuous bases. That kills the **untrained exact-Δ** layout, not the trained quantized-Δ family. Kill-gate for a new student: after paying previous-state storage, halo, dynamic BN, and dual-consumer union, same-port service must still drop ≥15% and AEE stay inside 1.259 / +0.005.
- Residual ADD is their **worst** temporal-sparsity site. Local r1 **is** a residual (`identity + conv2`). Expect Δ(r1out) denser than Δ(sn2). Continuous PED/`conv_res` still wants amplitudes; skipping zeros in g does not skip non-zero θ.
- Partial layers: they already say do not put DAL on every layer. Candidate sites are layers whose downstream MAC fan-out is large (they set λ ∝ fan-out). r1 Conv2 96×96×3×3 is such a site; source PSN T10 is another — but PSN is noncausal, so “t vs t−1” inside T10 is not a frame-Δ.
- Event cameras **already** difference log-intensity. Input-delta-only is EV-FlowNet’s counts, not a second mechanism. The live question is Δ of **r1 activations**, i.e. a second difference.

**Oracle?** DAL: no. AMS/P-frame OF: yes if used. **Do not import AMS warping of activation maps with current-frame flow.** Previous-frame DAL state is causal and allowed.

**HW?** Discussion and DRAM/SRAM arithmetic, not a chip. SCNN/EIE/NullHop/A100 sparsity are cited as context. Do not quote +3× sparsity as same-port %.

---

## 7. Direct answer: which mechanisms can change r1 PRODUCER support enough to pay hardware

Ranked by whether they **change the producer** (not the OF head), **avoid the current-frame flow oracle**, and have a **hardware-shaped kill-gate**. All are **A** until a local student beats ordinary dense r1 + the named controls under `PROBLEM.md` gates. None is “first event-OF HW.”

### Keep (can change support; no current-frame flow oracle)

| ID | Mechanism | Source | What must change in r1 | Strong controls (must win) | Why it might pay | Why it might not |
|---|---|---|---|---|---|---|
| **M1** | Trainable channel thresholds + membrane/threshold regularizers; densest-layer extra λ | SENECA §4 | sn1/sn2 θg support of r1, hence Conv1/Conv2 requests | Ordinary ReLU/ATLIF without Ls; static T; channel prune / narrow C to same nnz; SENECA FATReLU copied without dual-consumer loss | Only paper that **measured** ~5% density → fewer membrane accesses | Binary/FATReLU ≠ continuous θg; PED still dense; T10 cannot release like 3-row ANN; FireNet ≠ C96 |
| **M2** | Train **pixel-clustered** support (same nnz, fewer live pixels, fatter per-pixel groups) | SENECA §6, Fig. 3–5 | Spatial support shape of r1, not just rate | Matched-nnz uniform scatter; group=4 dummy-fill accounting; H8/P4 word skip without clustering | Grouping and state load/store scale with **pixels**, not nnz | Local ports are word/H8/NRV, not SENECA pixel groups; dummy zeros can eat the gain |
| **M3** | Trained quantized activation-Δ (no OF warp); **partial** layers; skip DAL on residual-add if Δ is dense | 2107.07305 §2–4 | Which r1 activations change across T or across frames | Untrained exact Δ (already 1.529×); spatial L1 only; MEET/DeltaCNN full cache cost; freeze-BN Δ | If skip MAC > (state store + 2 extra add + possible DRAM) | r1 **is** residual-add (their worst site); +40% state; local BN is full-domain; previous Y already ~295 MB in an old probe |
| **M4** | Input event-count / fixed-N window / polarity-count as a **cheap production mask** (not eval mask) | EV-FlowNet §III-A, §V-E/F; Spike-FlowNet §3.2 | Stem/r1 only where the window actually has events | Always-on dense r1; eval-only event mask; voxel-grid T10 as now | 20–30% event pixels on MVSEC; local patch already sees sparse events | Mask at input ≠ skip inside r1 after 3×3 halo; BN population is full domain; continuous PED on anchors still wants values |
| **M5** | Causal **past-only** motion prior (BAT bwd corr / previous-frame flow) as a **paid** hint to next-frame r1 mask, never as same-frame skip | BAT Table 5, Fig. 6 | Next frame’s r1 spatial support, trained | E-RAFT warm-start; copy previous flow; static event-count mask; zero prior | Does not read current-frame final flow | Feature/corr cost can exceed r1 skip; violent shake is their stated failure; still must retrain r1, not just lookup |

### Stop / do not use as r1 producer scheduler

| Mechanism | Why banned or dead as r1 support |
|---|---|
| SciFlow SCI quality map | Same-frame current flow → warp → quality. Oracle. Also **adds** a channel, does not skip r1. |
| BAT `df = f/N` warp as r1 gate | Uses current iterative f of the frame being solved. Legal inside a correlation head, illegal on r1. |
| AMS / P-frame OF compensation of activation maps | 2107.07305 cites it; it **is** a flow oracle. Ban for current frame. |
| Spike-FlowNet residual-as-SNN | Their own ablation: AEE gets worse as residual becomes SNN. r1 is that residual. |
| Spike-FlowNet encoder accumulators as “skip residual” | They **wait for all N** then dump a dense vector into ANN residual. Opposite of producer skip. |
| EV-FlowNet “AEE only where events exist” | Evaluation, not production. |
| Horowitz 5.1× AC/MAC, 214× encoder, 17% whole-net | Not hardware. Encoder fraction 17.6% ≠ r1 share. |
| SciFlow HTP 29 ms | Frame OF, 6 GRU iters, not r1. |
| “First SNN/event-OF hardware” | False relative to SENECA 2025, TrueNorth OF, FPGA plane-fit OF. |

### Dual-consumer / T10 filters (any keep-list item that fails these does not pay)

1. **Union of consumers.** Gate/sn2 zero does not retire a source if `conv_res`/PED still needs the amplitude. Report gate-only skip and union skip separately.
2. **Noncausal T10.** SENECA ANN 3-row release is illegal. A position is live until the PSN has all ten times or a **trained** complete-T10 predictor is paid (separate prior, not these six papers).
3. **Full-domain BN.** Native projection BN uses actual batch stats over 10×96×120×160 (`PROBLEM.md`). Active-site BN (SSCN/SBNet class) is another student. DAL/FATReLU zeros that still enter μ/σ cannot be dropped from the population.
4. **Halo.** 3×3 then 3×3: a zero output pixel can still force a (B+4)² source window. Event-count masks must be OR-dilated.
5. **Same-port ≥15% and AEE.** Absolute ≤1.259, relative ≤+0.005 vs ordinary dense/raw 1.219801338. nnz↓ or MVSEC AEE↓ is not the gate.
6. **No component-FPS multiply.** SENECA 62.5% of FireNet time is not r1 FPS.

---

## 8. What “pay hardware” means on this island (one paragraph)

On patch r1 the producer is not FireNet’s 32-ch conv and not a RAFT updater. It is two C96 3×3 residual convs plus T10 PSN, then **two** consumers (θg gates and continuous PED). The six papers show only three **algorithm** knobs that actually move that producer’s support without reading the current frame’s flow: **(M1) trained thresholds/regularizers**, **(M3) trained quantized temporal Δ**, **(M4) input-event production mask**; plus one **shape** knob **(M2) pixel clustering at matched nnz**, and one **causal** prior **(M5) past-only motion**. Hardware in this set exists only for M1+M2 on a different net (SENECA FireNet). Local untrained Δ already lost. SciFlow/BAT heads are GPU accuracy (and SciFlow’s on-device number is the wrong task). A TCAS-II sentence that survives is: *sparsity-aware training that changes r1’s dual-consumer support shape, measured as same-port service against dense r1, event-count mask, matched-nnz scatter, and SENECA-style threshold loss, without a same-frame flow oracle and without a first-OF-HW claim.* Anything else in these six texts is either an OF-accuracy head or a mapping prior to copy in full (grouping, depth-first, selective DAL) and then subtract.

---

## 9. Quote index (section pins)

| Need | Pin |
|---|---|
| Event image = count + last timestamp; window normalize | 1802.06898 §III-A |
| AEE on 20–30% event pixels; fixed-N window suggestion | 1802.06898 §V-E, §V-F |
| GPU 40–48 ms | 1802.06898 §V-B |
| Hybrid encoder SNN, residual ANN; vanish if residual spiked | 2003.06696 §3.4, supp §6.1 |
| Accumulator waits for all N | 2003.06696 §3.4, Alg. 1 |
| Horowitz energy, 17.6% encoder share | 2003.06696 §4.4 Table 2 |
| “First SNN SOTA event OF” | 2003.06696 §1 — do not reuse |
| SCI = warp by current flow | 2404.08135 §1, §3.1 Eq. 5–6 |
| RFL uses final-iter GT confidence on all iters | 2404.08135 §3.2 Eq. 8–9 |
| HTP Table 5 | 2404.08135 §4.2.6 |
| BTC `df=f/N`; SATMA; future flow from past only | 2503.03256 BTC, SATMA, Table 5 |
| BAT limitation: shake, bwd≠fwd | 2503.03256 Conclusion |
| FATReLU + Ls; grouping=4; pixel vs neuron density; 44.9 ms / 927 μJ | 2407.20421 §4, §3.2, §6 |
| ANN 3-row release vs LIF keep-all | 2407.20421 App. D |
| Already event-OF on neuromorphic HW; still lightweight | 2407.20421 Abstract, §7 |
| DAL quantized Δ, no threshold drift; +40% mem; residual ADD worst; AMS uses OF | 2107.07305 Abstract, §1.1, §2.2, §4, Fig. 6 |

Primary texts: `p0_txts/1802.06898.txt`, `2003.06696.txt`, `2404.08135.txt`, `2503.03256.txt`, `2407.20421.txt`, `2107.07305.txt`.
