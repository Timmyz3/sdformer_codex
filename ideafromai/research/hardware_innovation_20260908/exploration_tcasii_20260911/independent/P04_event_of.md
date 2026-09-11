# P04 independent ideas — event-camera optical flow, algorithm+hardware

Perspective: DSEC 2D flow on frozen student Motion C12 / H67 / ep34 (ATLIF continuous θg). Axes: voxel vs count+timestamp, causal vs noncausal T10, motion-warp costs, occlusion, confidence, residual encoding.

Hard constraints from freeze: do not retitle as binary ATLIF or analog CIM; do not quote Yosys/OpenROAD as PPA; do not multiply component speedups into FPS; do not add the two resource tables (source SIMD vs integer consumer); do not use this network’s **final flow of the same frame** as a free oracle for scheduling, warping, or skipping that frame. Previous-window flow is allowed only if it is treated as a stored state with its warp/storage cost in the same-port ledger.

Kill numbers reused unless an idea tightens them: valid825 AEE absolute ≤ 1.259; vs ordinary dense-source 1.219801338, relative ≤ +0.005; complete-chain same-resource **net service** ≥ 15% after backpressure (lifting’s −22.83% always-ready with unchanged 8088 long-backpressure is the cautionary control). Integer gates / I24 / PED q24 must stay 0-diff vs model on the two captured windows.

---

## P04-I1 Dual-moment T10 source: count → gate, timestamp moment → PED

**One sentence.** Compile EV-FlowNet/EST dual moments (event count and time) into the existing T10 PSN adder tree so the two consumers finally read the statistics they actually use, instead of both reading a polarity-mixed voxel.

**A (complete prior to copy).** Copy the full EV-FlowNet input (Zhu et al., RSS 2018): per-pixel, per-polarity **event count** plus **timestamp** channels, including the normalization of timestamps into the window. Copy the full EST construction (Gehrig et al., ICCV 2019): events as a measure, bilinear/trilinear kernel onto a grid, optional polarity split, and the published projections (voxel = collapse time kernel; SAE = collapse all but last timestamp; count image = collapse time). Do not copy only the slogan “use timestamps.” Copy the accumulator math, polarity split, and the two published collapses.

**B (hole in THIS net).** T10 PSN is noncausal and voxel-like; ordinary source is 260 add/sub per T10 vector. After CSE, two consumers share that source: a spike/gate path and a continuous residual/PED path (I24 / q24). A mixed voxel is the wrong sufficient statistic for both: gates need occupancy, PED needs a continuous time residual. Patch-embed r1 (two convs) then re-mixes the mismatch at the historically expensive region. Lifting cut the add graph (159 add/sub + 35 RNE/sat) but moved AEE to 1.232979368 (relative fail) and left long backpressure at 8088.

**X (not a reskin).** The claim is not “retrain with 4-channel input.” The mechanism is one event-loop dual-moment accumulator that **is** the T10 source: saturating count feeds the gate consumer; first-moment time (Σt, or Σt/N with the divider only on the PED path) feeds I24/PED. Same two-stage SIMD ports, same ready/valid, no extra representation SRAM. Latest-only SAE (EV-FlowNet’s last timestamp) is a **control**, not the title; first-moment vs latest-only is the circuit differential.

**Strongest controls.** (1) Ordinary dense T10 voxel, AEE 1.219801338. (2) EV-FlowNet latest-timestamp collapse, matched ports. (3) EST bilinear voxel, same bin count. (4) 40-coeff lifting T10 (failed relative). (5) Polarity-split vs polarity-sum, all else equal. (6) Same-port source SIMD; do not add integer-consumer −5.78%.

**Kill-gate.** valid825 AEE ≤ 1.259 and Δ vs ordinary ≤ +0.005; complete-chain net service ≥ 15% on the source SIMD **including** the extra moment add; I24/PED q24 0-diff vs the dual-moment student on captured windows. Kill if either moment needs a second pass over events, or if latest-only and first-moment both miss the relative AEE gate.

**TCAS-II pitch.** Event cameras already produce the two moments that this SNN’s two consumers want; the circuit is a shared accumulator, not two front-ends. One causal graph: event → {count, Σt} → {gate, PED} → r1, with add/sub and AEE on the same page.

**Biggest objection.** First-moment time is a linear collapse of T10 and may throw away the bin structure the transformer still expects, repeating lifting’s AEE regression.

**Assumptions.** T10 bins are a discretized event measure; gate consumer is occupancy-like; PED is continuous and can ingest I24 first-moment; polarity split is optional.

**Predictions.** (P1) First-moment PED + count gate matches ordinary AEE within +0.005, latest-only does not on DSEC night. (P2) Source add/sub per event falls vs two independent representations, but not vs a single voxel unless r1 zeros increase. (P3) Long-backpressure 8088 moves only if PED can start from a running Σt rather than a finished voxel.

**Disconfirmers.** Dual-moment AEE > 1.2248; or service gain vanishes once the divider/normalization sits on the critical port; or integer PED 0-diff breaks because Σt dynamic range exceeds q24.

---

## P04-I2 Causal T10 prefix with bin-complete partial-ready (not a future-bin drop)

**One sentence.** Replace noncausal T10 “wait for 10” with a copied streaming voxel handshake (EVA-Flow UVG + IDNet TID) so r1/gate can fire on a causal prefix, and kill the idea if any prefix that actually reduces backpressure fails AEE.

**A (complete prior to copy).** Copy EVA-Flow Unified Voxel Grid (Ye et al.): a bin is emitted when its interval is full, representation is consistent across first/middle/last bins, network consumes bins sequentially, Rectified Flow Warp Loss is **not** required for the circuit. Copy IDNet TID (Wu, Paredes-Vallés, de Croon, ICRA 2024): one iteration per streaming batch, residual flow from sequential event bins, no correlation volume. Copy the streaming contract: ready when the bin is complete, not when the whole window is complete.

**B (hole in THIS net).** PSN T10 is noncausal. Native projection BN uses actual batch statistics over the **full** 10×96×120×160 domain, so local-window replay with free μ/σ undercharges wait/storage. Lifting’s −22.83% always-ready was absorbed: long backpressure stayed 8088. Arithmetic on a volume that cannot be released early cannot deliver net service.

**X (not a reskin).** Not “delete future bins and retrain.” The increment is a **bin-complete token** compiled into the source ready/valid graph: causal bins 1…K release gate + r1 stage-0; PED may wait for K′≥K. K is a frozen compile-time constant, not a runtime function of this frame’s final flow. Same two consumers, same ports. If K must be 10 to pass AEE, the idea is dead rather than renamed “causal.”

**Strongest controls.** (1) Noncausal T10 ordinary. (2) Causal K=10 (same data, handshake only — isolates protocol). (3) Causal K∈{6,7,8,9} with identical weights vs fine-tuned-on-prefix. (4) UVG-style consistent bin vs raw truncated voxel (EVA-Flow’s first/last bin mismatch). (5) Backpressure measured, not add/sub.

**Kill-gate.** A causal K<10 that is the one used in the service ledger must have valid825 AEE ≤ 1.259 and Δ vs noncausal ordinary ≤ +0.005, **and** complete-chain net service ≥ 15% with long-backpressure < 8088. Kill if only K=10 passes AEE, or if prefix service “wins” by stalling PED until 10 anyway.

**TCAS-II pitch.** Noncausal T10 makes every arithmetic win wait on the last bin; a bin-complete handshake is a circuits object, not a dataset trick. One graph: event-bin → complete → {gate, r1} → optional PED, with AEE vs K on one plot.

**Biggest objection.** DSEC flow GT is aligned to a timestamp that may require events on both sides of t; a causal prefix can be information-theoretically insufficient, not just a hardware issue.

**Assumptions.** Event windows can be re-cut as (t−Δ, t] without changing the student family; BN either streams (see I7) or is replaced for the prefix experiment; K is global, not per-patch.

**Predictions.** (P1) Handshake-only K=10 does not change AEE and does not change 8088. (P2) Some K∈{7,8,9} passes relative AEE if bins are UVG-consistent, and fails if they are naive truncations. (P3) Net service ≥15% appears only when r1 stage-0 is actually released before bin 10.

**Disconfirmers.** All K<10 exceed +0.005 AEE even after prefix fine-tune; or BN full-domain wait reintroduces the stall; or GT alignment makes “causal” still need future events (then label the window noncausal and kill).

---

## P04-I3 Previous-window motion-compensated residual T10, warp charged on the same ports

**One sentence.** Copy CMax/IDNet warping as a residual encoder for T10, using **previous-window** flow only, and accept the idea only if residual-sparse r1 savings exceed bilinear warp cost on the same SIMD.

**A (complete prior to copy).** Copy Gallego et al. contrast maximization (CVPR 2018): warp events along a motion field to a reference time, form IWE, residual = unaligned events. Copy IDNet iterative deblurring: x′ = x + (t_ref−t) F(x), then re-encode residual bins. Copy HEVC/H.264 motion-compensated prediction: encode residual against a delayed predictor, never against the current reconstruction of the same picture. Copy bilinear sampling’s 4-tap gather as the warp circuit (the cost model, not a GPU op).

**B (hole in THIS net).** r1 is two convs on **unwarped** patch residual; PED is continuous on that volume. Motion-blurred voxels stay dense, so skip logic never fires. Warp cost is absent from both published resource tables. Using this frame’s final flow to warp the same frame is forbidden (oracle). Backpressure 8088 says idle arithmetic is not the bottleneck — **work shape** is.

**X (not a reskin).** Not “add a warp module in front of the CNN.” The mechanism is: stored previous-window flow (state) warps current T10 into a residual volume; r1/PED run on residual; bilinear 4-tap **competes for the same two-stage SIMD source ports** as T10 PSN. Net service is after warp, not before. Occlusion holes are a 1-bit mask into the PED valid (interface to I4), not a second network.

**Strongest controls.** (1) Unwarped ordinary T10. (2) Warp with **zero** flow (identity, measures residual-encoder overhead). (3) Warp with previous-window student flow, cost in ledger. (4) Forbidden oracle: current-frame final flow (report as an illegal upper bound, never as the design). (5) Nearest-neighbor warp vs bilinear. (6) Lifting without warp (arithmetic-only fail).

**Kill-gate.** valid825 AEE ≤ 1.259, Δ ≤ +0.005 vs ordinary; (T10 + warp + r1 + both consumers) same-port cycles ≤ 0.85 × ordinary complete chain. Kill if warp+state-read ≥ residual-skip savings, or if the only passing warp field is the same-frame final flow.

**TCAS-II pitch.** Residual coding is only cheaper if the predictor is delayed state and the warp is on the same ports as the source. One graph: F_{n−1} → warp → residual T10 → r1/PED, with warp cycles and AEE on one axis.

**Biggest objection.** Previous DSEC window flow is a stale predictor under acceleration/turns; bilinear gather at 120×160×10 can dominate two r1 convs.

**Assumptions.** A previous-window flow exists at the same spatial grid; linear motion inside one window is a usable predictor; residual volume uses the same I24/q24 path; storage of F_{n−1} is counted.

**Predictions.** (P1) Residual energy (L1 of T10 after warp) drops on roads, not at object boundaries. (P2) Nearest-neighbor warp almost matches bilinear AEE here (events are already discrete) and is the only warp that can pass the 15% gate. (P3) Oracle current-frame warp beats AEE by more than +0.005 the other way — that gap is **not** a legal claim.

**Disconfirmers.** Residual L1 does not drop enough to skip r1 tiles; bilinear is required for AEE and blows the cycle gate; first window / scene cuts have no F_{n−1} and the fallback is ordinary (kills average service).

---

## P04-I4 Occlusion-gated PED skip from delayed warp-holes, never from same-frame FB flow

**One sentence.** Copy UnFlow/CMax occlusion masking, but build the mask from previous-window warp occupancy vs current event-rate, and use it only to invalidate the expensive continuous PED consumer.

**A (complete prior to copy).** Copy UnFlow (Meister, Hur, Roth, AAAI 2018): bidirectional flow, forward–backward consistency, **mask the data term** on occluded pixels, keep smoothness/prior on those pixels. Copy Shiba–Aoki–Gallego time-aware flow (ECCV 2022 / T-PAMI 2024): occlusion is a transport/time-of-arrival problem, not a 2D hole in a frame pair; warp along characteristics. Copy IWE hole statistics from CMax as the occupancy test. Do **not** copy a second backward student on the current window (that is another full net).

**B (hole in THIS net).** Dual consumers **always** both run after the source. Integer-consumer table only moved 758777 → 714889 (−5.78%) at a different resource point. Continuous PED q24 is the consumer that should be skippable on occlusions and disocclusions (DSEC vehicles, poles, night HDR). Gate/spike path must still run: occupancy is exactly what spikes encode. No occlusion bit exists in the ready graph.

**X (not a reskin).** Not “train with an occlusion loss.” The circuit is a 1-bit occupancy from {previous-window warp of predicted event mass, current T10 count}. If predicted mass ≫ 0 and current count = 0 (or ratio ∉ [1/ρ, ρ]), PED valid is deasserted; gate consumer stays asserted. No same-frame final flow, no backward pass. ρ is a compile-time constant.

**Strongest controls.** (1) Always-both-consumers ordinary. (2) Skip PED on **zero-event** patches only (classical DVS skip). (3) Illegal FB using current-frame flow (upper bound). (4) Random skip at the same skip rate (must lose AEE). (5) Gate-also-skip vs PED-only-skip.

**Kill-gate.** On valid825, PED skip rate ≥ 20% of **nonzero-event** patches, AEE ≤ 1.259 and Δ ≤ +0.005 vs always-both. Report AEE on skipped vs kept patches separately. Kill if skip rate < 10% after excluding zeros, if random skip matches AEE, or if gate-path skip is required to hit 15% service (that would be a different, likely dead, idea).

**TCAS-II pitch.** Occlusion in event flow is a delayed occupancy mismatch, and this net already has two consumers with different physics. One graph: F_{n−1} → predicted mass ≷ count → PED valid, with skip rate and AEE.

**Biggest objection.** Delayed occupancy is a motion-error detector as much as an occlusion detector; it may skip the exact patches that need PED (object boundaries).

**Assumptions.** Previous-window flow is stored; ρ and the mass predictor are frozen; PED skip does not change ATLIF θg dynamics on the gate path; service model can deassert one of two consumers without collapsing the SIMD issue width in a way that loses the gain.

**Predictions.** (P1) PED-only skip at ≥20% costs ≤0.005 AEE; gate-also-skip at the same map fails AEE. (P2) Random skip at the same rate fails relative AEE. (P3) Night DSEC skip rate > day (HDR flicker ≠ occlusion — must be filtered by requiring predicted mass > τ, else kill).

**Disconfirmers.** Occupancy map correlates with |flow| rather than occlusion (then it is illegal confidence-from-flow); or integer 0-diff fails when PED is replaced by hold-last; or backpressure stays 8088 because the gate consumer still serializes the source.

---

## P04-I5 Continuous θg × event-count as source-port confidence arbiter

**One sentence.** Use the student’s already-computed continuous threshold amplitude θg, times local T10 count, as a 2-bit priority for the shared source SIMD, copying confidence-weighted early-exit without reading this frame’s flow.

**A (complete prior to copy).** Copy confidence-aware optical flow and early-exit as a **full policy**: RAFT-style iteration stop on a confidence proxy; event-count as uncertainty (low count ⇒ ill-posed flow, used in EV-FlowNet visualizations and in CMax IWE weighting by n(x)); mixture-density / learned uncertainty heads (report as the heavy prior, then refuse to add a head). Copy hardware priority arbitration for a shared SIMD/issue port (oldest-ready vs QoS), including starvation rules.

**B (hole in THIS net).** The measured hole is **scheduling**, not the add graph: lifting 6938 → 5354 always-ready (−22.83%) with long backpressure still 8088. Dual consumers fight for one source. ATLIF is defined by **continuous** θg, already on the datapath, unused as a scheduler. Final flow of this frame is not a legal confidence oracle.

**X (not a reskin).** Not “early-exit the transformer” and not a new confidence CNN. Tap θg and T10 count (both already live), quantize to a 2-bit class {hold-last, gate-only, both-consumers, both+priority}. Arbiter sits on the existing ready/valid, same ports, same backpressure counters. Mapping is frozen before valid825 (no per-frame search). Continuous θg is the paper identity; using it as QoS is the increment.

**Strongest controls.** (1) Always-both ordinary. (2) Count-only skip (ignore θg). (3) θg-only skip (ignore count). (4) Random class with same histogram. (5) Illegal: schedule with same-frame |flow| or AEE. (6) Lifting without arbiter (arithmetic-only).

**Kill-gate.** Complete-chain net service ≥ 15% **with the arbiter inside the ready/valid model** (not a MAC proxy); AEE ≤ 1.259, Δ ≤ +0.005 vs always-both. Kill if θg is uncorrelated with per-patch error (Spearman |ρ| < 0.15 on a held-out train split before touching valid825), or if the only working map is a disguised function of this frame’s flow.

**TCAS-II pitch.** This student already spends energy computing a continuous threshold; that scalar is a free confidence for a shared-port arbiter. One graph: {θg, count} → 2-bit QoS → source issue, against the 8088 backpressure number.

**Biggest objection.** θg tracks neuron excitability, not geometric flow confidence; the arbiter may starve textured, high-threshold patches that dominate AEE.

**Assumptions.** θg is observable at patch granularity without extra full-width dumps; hold-last residual is defined; starvation bound ≤ N cycles is part of the compile; valid825 is not used to fit the 2-bit thresholds (use train or a captured window).

**Predictions.** (P1) Count-only already recovers most skip; θg adds a night/HDR split (high θg + mid count = noise). (P2) Long-backpressure falls below 8088 only if low-class patches **release the source**, not if they still occupy an always-ready slot. (P3) Illegal |flow| scheduler beats AEE — that is a disallowed ceiling.

**Disconfirmers.** Spearman(θg, per-patch AEE) ≈ 0; arbiter histogram collapses to “both” on >95% of patches; integer 0-diff fails on hold-last; service gain is only on the activity-weighted-dot proxy, not the SIMD ready graph.

---

## P04-I6 Temporal DPCM T10 (slice residuals) as residual-sparse r1 source

**One sentence.** Encode T10 as slice-0 plus nine saturated temporal differences (full DPCM/HEVC residual coding), run BN and r1 in the residual domain, and skip near-zero residual tiles — without any flow predictor.

**A (complete prior to copy).** Copy DPCM and HEVC residual coding end-to-end: prediction (here: previous temporal bin, not spatial intra), residual, saturating quantizer, skip flag on |res|<ε, reconstruct at the consumer if needed. Copy Spike-FlowNet’s split (Lee et al., ECCV 2020): SNN/encoder on events, **residual blocks** on accumulated features — the residual is after a temporal accumulate, not a new backbone. Copy signed polarity voxels (EST signed measure) so residuals are allowed to be negative.

**B (hole in THIS net).** Source is absolute T10 (260 add/sub). Lifting already tried a different linear factorisation and failed relative AEE (+0.013178) while adding 35 RNE/sat. Native BN statistics are over the full absolute 10×96×120×160 domain. r1 two convs do not see temporal sparsity; DSEC static background still occupies the expensive residual chain.

**X (not a reskin).** Not lifting, not a wavelet, not a new stem. The source graph becomes `s0, sat(s_t−s_{t−1})` with the **same** RNE/sat already in the integer path. BN moments are compiled in residual domain (smaller range → I24). Dual-consumer map: gate from residual zero/sign, PED from residual magnitude. Reconstruct-to-absolute is a control (must be off the critical port if service is to count).

**Strongest controls.** (1) Absolute ordinary T10. (2) Lifting 40-coeff (failed). (3) DPCM without r1 zero-skip (isolates encoding vs skip). (4) Reconstruct-to-absolute before r1 (should erase the gain). (5) ε=0 vs ε kill-gated. (6) Same SIMD, same RNE.

**Kill-gate.** Residual-domain student AEE ≤ 1.259, Δ vs ordinary ≤ +0.005; complete-chain net service ≥ 15% **after** the extra subtract; skip fraction measured on valid825, not a toy window. Kill if AEE regression looks like lifting (≥ +0.013) or if skip only appears after reconstruct-to-absolute is “forgotten” in the ledger.

**TCAS-II pitch.** The expensive r1 chain is already named residual; the source is still absolute, so sparsity is fake. One graph: Δ_t T10 → sat → {gate, PED, r1-skip}, with the lifting fail as the linear-algebra control.

**Biggest objection.** Adjacent T10 bins in a moving scene are **shifted**, not subtracted-sparse; DPCM without warp (I3) may densify rather than sparsify.

**Assumptions.** T10 slices are temporally ordered with comparable scale; sat-I24 residual matches the integer 0-diff contract; BN in residual domain is retrained or folded, not naively reused.

**Predictions.** (P1) Residual L0 sparsity is high on night static pixels, low on near-field motion — skip service ≥15% only if the static mass dominates valid825 cycles. (P2) Reconstruct-before-r1 returns AEE to ordinary and kills service. (P3) DPCM+lifting is worse than DPCM alone (two linear transforms).

**Disconfirmers.** Residual histograms are not concentrated at 0 on DSEC driving; BN residual stats break AEE by >0.005; extra subtract + RNE exceeds skip savings (repeat of lifting’s sat tax).

---

## P04-I7 Overlapping Chan/Welford BN that reproduces full-domain batch stats

**One sentence.** Copy PreciseBN/GroupNorm as complete answers to the train-time batch-stat problem, then fuse Chan pairwise moments over the T10 ring so hardware matches the student’s **actual** 10×96×120×160 batch μ,σ² without a stall.

**A (complete prior to copy).** Copy Ioffe BN, including the train (batch μ,σ²) vs eval (running) split. Copy Wu & He GroupNorm (ECCV 2018) as the batch-axis-free vision default. Copy Wu & Johnson “Rethinking ‘Batch’ in BatchNorm” (PreciseBN: recompute true population/batch stats rather than hope running averages). Copy Welford and Chan pairwise algorithms as the numerically stable online moment circuits, including the merge of disjoint tiles.

**B (hole in THIS net).** “Native projection BN on captured students uses **actual batch statistics over the full 10×96×120×160 domain**, not frozen running stats. Local-window replay given free mean/var undercharges wait/storage.” This is a frozen measurement. Frozen running stats would stream but may break AEE. Free μ/σ in a local replay is a dishonest service number. Projection + BN + residual add sit next to the dual consumers.

**X (not a reskin).** Not “flip BN to eval mode.” Two compiled mechanisms, one title: (i) Chan merge of per-bin/per-tile moments that **equals** full-domain μ,σ² to the integer tolerance of I24, overlapped with T10 fill; (ii) GroupNorm (spatial/channel groups, no batch axis) as the complete copied alternative with the same residual add. Scale/shift fused into native projection. Dual-consumer residual add sees the same numeric BN as the student.

**Strongest controls.** (1) Native full-domain batch BN (AEE reference). (2) Frozen running stats (service upper bound, AEE likely fail). (3) Local-window BN with **charged** μ/σ (honest replay). (4) PreciseBN two-pass (correct, no overlap). (5) GroupNorm G∈{1,8,32}. (6) InstanceNorm (G=C).

**Kill-gate.** Compiled BN AEE ≤ 1.259 and Δ vs native full-domain BN ≤ +0.005 (and vs ordinary 1.219801338 if BN is the only change). Moment storage + merge cycles are in the same-port ledger; complete-chain net service ≥ 15% vs wait-for-full-volume-then-BN. Kill if only running stats hit 15% and they miss the AEE gate; kill if Chan merge does not match full-domain μ,σ² to the I24 0-diff budget.

**TCAS-II pitch.** The student is not a running-stat network; quoting service with free μ/σ is a measurement error. One graph: tiles → Chan merge → fused scale/shift → residual add, with AEE against running-stat and GroupNorm controls.

**Biggest objection.** Full-domain moments may be load-bearing for AEE; overlapping Chan still needs a barrier before projection, so backpressure 8088 does not move.

**Assumptions.** Native projection BN is on the critical path of r1/consumers; I24 can hold pairwise moments; GroupNorm retrain is allowed as a control student in the same family, not a new architecture.

**Predictions.** (P1) Running-stat AEE fails relative +0.005 (the student was captured with batch stats). (P2) Chan-overlapped BN matches native AEE to integer 0-diff and reduces wait vs two-pass PreciseBN. (P3) GroupNorm passes AEE only at G that preserve T10 channel structure; G=1 (LN-like) fails.

**Disconfirmers.** Running stats already pass AEE (then B is false); Chan merge needs a second full read (no overlap); GroupNorm wins AEE but loses the “same student” identity; moment SRAM exceeds any r1 saving.

---

## P04-I8 Shared decay-integrator time surface as PED, count as gate (T10 storage optional)

**One sentence.** Copy HOTS/HATS/EV-FlowNet time surfaces as a single IIR per pixel/polarity that **is** the PED source, with a saturating count as the gate source, and drop T10 volume storage if AEE survives a causal decay.

**A (complete prior to copy).** Copy Lagorce et al. HOTS time surfaces (T-PAMI 2017): S_a(t)=exp(−(t−T_a(t))/τ), local context, polarity. Copy Sironi et al. HATS (CVPR 2018): averaged time surfaces, cell histograms — histogram part is a control, the surface is the circuit. Copy EV-FlowNet’s two timestamp channels plus two count channels. Copy the IIR identity: one multiply-add (or shift-add if τ is dyadic) per event per pixel.

**B (hole in THIS net).** Noncausal T10 stores a 10-slice volume; BN walks 10×96×120×160. Dual consumers want a spike occupancy **and** a continuous time residual; a decay surface is exactly the latter, a count is exactly the former. Voxel T10 is a batching artifact that creates the wait I2/I7 are trying to schedule around.

**X (not a reskin).** Not “retrain a CNN on HATS.” The mechanism is one write-port pair on event arrival: count++ (gate); PED ← PED + f(Δt) with q24 sat, τ compiled (dyadic shift-add preferred). T10 SRAM becomes optional. Causal by construction. If τ must be a learned per-patch map, the circuit dies. Lifting’s adder tree is unrelated; do not hide a wavelet in τ.

**Strongest controls.** (1) Ordinary T10 voxel. (2) EV-FlowNet last-timestamp (no decay). (3) HOTS exponential with global τ. (4) Dyadic τ=2^{k} Δt vs learned τ. (5) HATS histogram extra (should be unnecessary). (6) Causal T10 prefix (I2) at matched latency.

**Kill-gate.** AEE ≤ 1.259, Δ vs ordinary T10 ≤ +0.005; T10 volume storage removed or shown unused in the ledger; complete-chain net service ≥ 15% including IIR updates on event rate (DSEC peaks, not mean). Kill if a learned per-patch τ is required, if only noncausal (future-aware) decay passes AEE, or if IIR updates at peak event rate exceed voxel T10 cycles.

**TCAS-II pitch.** Count and time-surface are the two consumers’ sufficient statistics, and they share one event write port. One graph: event → {count IIR, decay IIR} → {gate, PED}, with τ dyadic and DSEC AEE.

**Biggest objection.** The frozen student was trained on T10 voxels; a decay surface is a different function class, and matching +0.005 AEE without breaking the family may be impossible in the “weeks, CPU student” horizon.

**Assumptions.** A student with IIR front-end can be trained or last-layer-adapted without leaving Motion C12 / H67 / ep34; peak event rate is the service case; polarity-split IIR is the default copy of HOTS.

**Predictions.** (P1) Dyadic τ in a small set {2^{k}} passes AEE if one τ is chosen on train, not if τ is global-arbitrary. (P2) Last-timestamp (no decay) fails night DSEC relative to exponential. (P3) Service win is storage + BN, not MAC; at peak rate IIR can **lose** unless events are spatially sparse enough that most pixels are idle.

**Disconfirmers.** Best dyadic-τ AEE > 1.2248 vs ordinary; peak-rate IIR cycles > voxel T10; student identity requires keeping T10 anyway (then this is I1, not I8).

---

## Cross-idea notes (not extra modules)

- **One mechanism per TCAS-II paper.** I1/I8 are representation; I2 is handshake; I3 is delayed warp residual; I4 is occupancy valid; I5 is θg QoS; I6 is DPCM; I7 is BN moments. Do not stack into a zoo.
- **Illegal ceiling.** Same-frame final flow as warp/oracle/scheduler may improve AEE; report only as a forbidden bound.
- **Failed arithmetic control.** Learnable 40-coeff lifting T10 (AEE 1.232979368, backpressure 8088) is the default “linear source rewrite that did not buy net service.”
- **Composition later, not now.** I3’s warp residual and I6’s DPCM are cousins; I2’s handshake and I7’s overlapping BN both attack the full-volume barrier. Pick one title.
