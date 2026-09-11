# P06 — Training-induced structure that hardware can legally charge

Participant: P06. Independent round. Inputs used: `SCOPE.md`, `PROBLEM.md` freeze of 2026-09-11 only.

Perspective: quantization, N:M, product sparsity of **real** ATLIF θg (not binary spikes), temporal lifting coefficients, shared time coordinates, and pruning that changes **producer** support for **both** consumers. Relative AEE +0.013 is a reported observation, not a license to ship unconstrained lifting.

Identity (hard): ATLIF stays continuous-θg. No binary-ATLIF paper. No analog CIM. No Yosys/OpenROAD as foundry PPA. No component-rate × FPS. No module zoo. Lifting family may be kept; the layout may die.

## Frozen numbers this file is allowed to use

- Ordinary dense-source/raw valid825 AEE = 1.219801338
- Learnable 40-coeff lifting T10 raw AEE = 1.232979368 (Δ = +0.013178 vs ordinary)
- Absolute AEE budget 1.259 (lifting passes). Relative strong-control budget +0.005 (lifting fails)
- Ordinary source graph after official CSE: 260 add/sub per T10 vector
- Lifting source graph: 159 add/sub + 35 intermediate RNE/sat
- Same two-stage SIMD source resource: always-ready 6938 → 5354 (−22.83%); long backpressure **both** 8088 (gain absorbed). Part of −22.83% is generic round→sat fusion, not lifting
- Separate integer consumer model (different resource point): 758777 → 714889 (−5.78%). Do not add the two tables
- Native projection BN: actual batch statistics over the full 10×96×120×160 domain, not frozen running stats. Local-window replay with free mean/var undercharges wait/storage
- Integer gates / I24 / PED q24 on two captured windows: 0 difference vs model

## Shared recovery protocol (every training idea)

Relative AEE +0.013 is already on the table. Structure induction without a paired recovery is not a TCAS-II result.

- **Pair:** structure constraint + one recovery fine-tune, started from the captured Motion C12 / H67 / ep34 student named in the idea.
- **Budget:** one run, compute ≤ original ep34 (prefer a short FT with a single λ / step-size / mask-refresh schedule). Architecture search, extra heads, and λ grids do **not** count as the pair.
- **AEE kill (number):** after that one pair, valid825 AEE must satisfy absolute ≤ 1.259 **and** Δ vs ordinary 1.219801338 ≤ +0.005, i.e. AEE ≤ 1.224801338. If relative AEE stays **> 0.005**, the idea is dead. No second attempt.
- **Legal-charge test:** a trained zero, alphabet, N:M pattern, knot, or prune is chargeable only if it changes **issued work** or **ready/completion** under the same ports, same state, same backpressure model. Proxy FLOPs, old activity-weighted-dot “patch ~35%”, and 260→159 add/sub are not service.
- **Service kill (number, once full_chain scheduling closes):** complete-chain same-resource net service ≥ 15%. Until then, component kill is idea-specific (always-ready **and** long-backpressure, or the consumer table, never the sum).
- **Do not** treat 5354 / 8088 as the paper’s gain: long-backpressure did not move.

---

## P06-I1 — CSD/dyadic alphabet on the 40 lifting coefficients

One sentence: Restrict the already-learned 40 T10 lifting coefficients to a CSD/dyadic alphabet so each multiply becomes a compile-time shift-add, then spend one paired recovery to pull relative AEE back under +0.005 while deleting the 35 data-dependent RNE/sat nodes.

### A (complete prior to copy)
Copy the whole chain, not the slogan: (1) Sweldens lifting and Daubechies–Sweldens FIR factorization into predict/update steps; (2) Calderbank–Daubechies–Sweldens / JPEG2000 integer lifting as the quantized-coefficient ancestor; (3) CSD and multiplier-block FIR (Hartley; Dempster–Macleod) plus TCAS-I/II multiplierless lifting DWT datapaths; (4) QAT of coefficients with a straight-through estimator (Jacob integer-arithmetic CNN; Esser LSQ). The copied object is “learned FIR → finite signed-digit alphabet → CSE’d shift-add graph,” applied to **this** 40-coeff noncausal T10 source.

### B (hole in this net)
The 40 coeffs are unconstrained reals. That is why the lifting graph still carries 35 intermediate RNE/sat, why AEE already regresses +0.013178, and why dual consumers (gate and continuous PED) both inherit a rounded, non-exact source. Round→sat fusion already explains part of the always-ready drop; unconstrained reals do not give the compiler a legal multiplierless graph that **both** consumers can share.

### X (why not a reskin)
Fixed-basis wavelet VLSI quantizes CDF-5/3 or 9/7 once. Here the 40 coefficients are task-learned for event-camera T10 PSN, the neuron is continuous-θg ATLIF, and two heterogeneous consumers must accept the same alphabet. The increment is not “quantize lifting.” It is: train the alphabet so the **compiled source graph** loses the 35 RNE/sat **and** the dual-consumer integer path stays 0-mismatch, with AEE recovered in one pair. If the 35 nodes are not on the long-backpressure path, X is empty — that is a kill, not a pivot.

### Strongest controls
- Ordinary dense-source AEE 1.219801338 / 260 add/sub / always-ready 6938 / long-BP 8088
- Unconstrained 40-coeff lifting AEE 1.232979368 / 159 add/sub+35 RNE/sat / 5354 / 8088
- Post-hoc round-to-CSD **without** the paired FT (expect AEE death)
- Same two-stage SIMD, same ports, same state; no extra predict/update step
- Gate-consumer integer vs PED-consumer integer, separately, after the alphabet freeze

### Kill-gate (number)
After **one** paired CSD/LSQ recovery: AEE > 1.224801338 **or** relative Δ > +0.005 kills. Hardware kill if 35 RNE/sat do not go to 0 (or to a compile-time sat policy with 0 data-dependent RNE) **or** if long-backpressure remains 8088 at the same ports (alphabet did not touch the stall that absorbed −22.83%). Consumer table may move; do not add it to the source table.

### Two-sentence TCAS-II pitch
Unconstrained learnable lifting already buys 260→159 add/sub and then spends it on 35 rounded intermediates and +0.013 AEE; the always-ready win is absorbed at 8088 long-backpressure. Training those 40 coefficients onto a CSD alphabet, with one paired recovery, is a circuits-and-training co-design: the source becomes a multiplierless, dual-consumer-legal graph whose AEE is measured against the dense student, not against a weaker lifting baseline.

### Biggest objection
Reviewers will say this is JPEG2000 lifting plus QAT. The only rebuttal is the dual-consumer 0-mismatch plus a moved long-backpressure number; add/sub count alone is already published as an observation.

### Assumptions
Lifting family is retainable. CSD with ≤2 signed digits per coefficient is rich enough for T10 motion. Dual consumers can share one coefficient ROM. Integer gates/I24/PED q24 remaining 0-difference after alphabet freeze is feasible because those paths already match at 0 on two windows.

### Predictions
1. Post-hoc CSD without recovery: AEE Δ stays near +0.013 or worse (> +0.005).
2. One paired recovery: AEE ≤ 1.224801338, source RNE/sat intermediates = 0, add/sub ≤ 159 and CSE-legal.
3. If RNE/sat were on the ready path, long-BP < 8088; if not, prediction 3 fails and I1 dies in favor of I5.

### Disconfirmers
CSD recovery that meets AEE but leaves 35 RNE/sat in the graph. Long-BP still 8088 after RNE deletion (legal-charge failure). Either consumer’s integer model leaving 0-difference. A second training run required to hit +0.005.

---

## P06-I2 — Product sparsity of real θg, intersection support for both consumers

One sentence: Train exact zeros of the real product |θg|·|source| so the **producer** may skip a lane only when **both** the gate consumer and the continuous PED consumer ratify that skip; never treat θg as a binary spike.

### A (complete prior to copy)
Copy: (1) product-of-gates compute (GLU/SwiGLU; CATS magnitude-skip on the gate factor); (2) activation-sparsity training with a thresholded identity (Hoyer / transformed-ℓ1 dual-sparse training; Kurtz et al. sparse ReLU); (3) product zero-skip microarchitecture (SCNN, SparTen) that skips a MAC iff a product is zero; (4) dual-path residual gating (Highway / GRU reset) where one gate does not license skipping the other path. The copied object is “train a product toward zero, then skip the product in hardware,” **not** “SNN event-driven spike skip.”

### B (hole in this net)
ATLIF θg is a continuous amplitude. Binary spike-skip is identity-illegal and factually false. Dual consumers after the source (spike/gate path **and** continuous residual/PED) mean a zero that only the gate can ignore does **not** change producer support: PED still demands the lane. Integer gates already match the model at 0 difference on two windows, so the hardware **can** skip an exact integer-zero product without a new comparator story; what is missing is trained, **shared** support.

### X (why not a reskin)
This is not “SNNs are sparse.” The skip predicate is `prod = θg ⊙ x == 0` at the existing integer alphabet, and the producer mask is `S = S_gate ∩ S_PED`, not `S_gate ∪ S_PED` and not `S_gate`. Union sparsity is a consumer-side curiosity; intersection sparsity is the only pattern the **source SIMD** can legally drop. That intersection, trained on real θg, is the increment.

### Strongest controls
- Dense ordinary student (AEE 1.219801338) vs product-thresholded student after one pair
- Mask = gate-only vs PED-only vs union vs intersection; only intersection may reduce source issued slots
- Unstructured |θg| threshold vs SIMD-grouped zeros (see also I3)
- Same ports / same state / same backpressure model; skip only exact integer zeros (preserve 0-difference)

### Kill-gate (number)
One paired attempt (threshold + STE, one λ). Kill if AEE Δ vs 1.219801338 > +0.005. Kill if intersection sparsity on the source tensor is < 25% of SIMD lanes **or** if intersection sparsity ≥ 25% but always-ready **and** long-BP are both unchanged (skip was not issued). Kill if PED integer mismatch leaves 0 on the captured windows (producer skip corrupted the continuous consumer).

### Two-sentence TCAS-II pitch
The expensive source has two consumers; sparsity that only the gate can see cannot retire a producer slot, and θg is real, so spike-skip is a wrong paper. Training the real product |θg|·|x| onto exact integer zeros with an **intersection** mask gives one skip network that both consumers accept, which is the only sparsity TCAS-II can charge on this SIMD.

### Biggest objection
“You just thresholded a continuous neuron into a spike.” Rebuttal must keep multi-bit θg amplitude and show non-zero θg values still flow; only the product hits exact zero. If the trained state collapses toward binary ATLIF, the idea has changed paper identity and is dead.

### Assumptions
Exact-zero products at the existing I24/q24 alphabet are frequent enough to train without collapsing θg to {0,1}. Dual-consumer saliency is positively correlated on T10 (otherwise intersection ≈ 0 and there is nothing to charge). Skip logic fits in the existing two-stage SIMD issue window.

### Predictions
1. Gate-only mask: consumer-gate work drops, source always-ready and long-BP do not.
2. Intersection mask after one pair: AEE ≤ 1.224801338; source issued lanes drop ≥ 25%; 0 integer mismatch holds on both consumers.
3. Product histogram of |θg|·|x| grows a Dirac at 0 without a Dirac at all other θg bins.

### Disconfirmers
Intersection density stays ≈ union density (one consumer already dominated). AEE recovers only when θg is binarized. Skip bitmap traffic cancels the lane-drop under same-port accounting. A second λ sweep is required.

---

## P06-I3 — SIMD-native N:M on the source layout, one mask for both consumers

One sentence: Impose N:M structured sparsity on the **existing two-stage SIMD grouping** of the source tensor, with the kept-N chosen as the intersection of both consumers’ saliency, then recover AEE in one SR-STE-style pair.

### A (complete prior to copy)
Copy: Zhou et al. ICLR 2021 SR-STE (N:M from scratch / dense-to-sparse with sparse-refined STE); NVIDIA ASP 2:4 as the commodity pattern; column-wise N:M for SIMD/vector CPUs (Chu et al., column-wise tile N:M on RISC-V / XNNPACK); N:M transformer co-design (inherited dynamic pruning + sparse engine). Copy the **training rule and the layout contract**, not Ampere Sparse Tensor Core as this chip.

### B (hole in this net)
Unstructured zeros in θg or in lifting coeffs do not change the two-stage SIMD schedule. The measured −22.83% always-ready already failed to move long-backpressure. Dual consumers can disagree on which of M lanes matter; a weight-side 2:4 borrowed from Ampere will not match this source’s time-channel packing. Hardware can charge N:M only if the mask is the SIMD’s native group and both consumers accept it at compile time or at a cheap ready-valid granularity.

### X (why not a reskin)
ASP/SR-STE are weight N:M for GEMM. Here N:M is on the **source activation/coefficient layout the SIMD already issues**, the two consumers are heterogeneous (gate vs continuous PED), and the mask is their intersection. That is a different legal-charge object than “put 2:4 on conv kernels.”

### Strongest controls
- Dense ordinary vs SR-STE N:M with M = SIMD group width (not necessarily 4)
- Intersection N vs independently pruned gate-N and PED-N
- Weight-only N:M on r1 convs vs source-layout N:M (the latter must win on producer slots)
- Same two-stage SIMD resource; compile-time mask vs per-window mask (per-window must pay bitmap ports)

### Kill-gate (number)
One SR-STE pair, one λ_w. Kill if AEE Δ > +0.005 vs 1.219801338. Kill if M does not equal the SIMD grouping actually compiled (pattern not issuable). Kill if always-ready does not drop ≥ 15% **and** long-BP stays 8088 (structured sparsity that the scheduler cannot issue). Prefer N/M ≤ 1/2; if N/M > 1/2 is required to meet AEE, chargeable work is too small — kill.

### Two-sentence TCAS-II pitch
N:M is only a circuit when the N and the M are the lanes the SIMD already owns. Training one intersection  N:M mask that both the gate and the continuous PED accept turns a dense T10 source into a compile-time sparse issue schedule without renaming Ampere 2:4.

### Biggest objection
Reviewers will ask why this is not ASP on the patch-embed convs. Answer with the dual-consumer intersection and a same-port SIMD slot number; a GEMM 2:4 ablation that does not move source long-BP is the control that must lose.

### Assumptions
SIMD group width M is known and small (4–8). Both consumers’ top-N in each group overlap enough to train. Compile-time masks are legal because T10 PSN is noncausal and the pattern can be frozen after the pair.

### Predictions
1. Weight-only 2:4 on r1: consumer MACs drop, source 6938/8088 unchanged.
2. Source-layout intersection N:M: AEE ≤ 1.224801338; always-ready ≤ 0.85 × baseline at that resource; integer 0-difference preserved.
3. Gate-only N:M meets AEE more easily but leaves PED dense — producer slots do not drop.

### Disconfirmers
Best AEE-feasible N is N=M (no sparsity). Per-window masks need extra ports that erase the slot gain. Intersection N is systematically smaller than min(N_gate, N_PED) by enough that AEE dies.

---

## P06-I4 — Shared time-coordinate knots for T10, one table, two consumers

One sentence: Replace 40 independent lifting coefficients with K≪40 shared time knots on the noncausal T10 axis; both the gate consumer and the PED consumer read kernels interpolated from that **one** coordinate table, so unused knots delete producer taps for both.

### A (complete prior to copy)
Copy: polyphase/filter-bank shared delay lines (Vaidyanathan); Temporal Shift Module shared integer offsets (Lin et al. ICCV 2019); learnable lifting / second-generation wavelets with trained predict-update (LSNet; AMSW-NN); spline/RBF kernels with shared knots. The copied object is “one time grid, many readers,” not a new temporal mixer block.

### B (hole in this net)
Forty independent coeffs over T10 mean two consumers can demand different temporal supports, so the producer cannot drop a tap. Hardware today stores 40 unconstrained numbers and still emits a dense T10 vector. Shared coordinates are the structure that makes **time-support pruning** legal for both consumers at once. This is the dual of I6 (channel support) on the time axis.

### X (why not a reskin)
Reducing 40→K coefficients by PCA or by “smaller lifting” is a reskin. The increment is that the only learned time parameters are coordinates **shared by both consumers**, so a knot that interpolates to ~0 for both is a deleted producer tap (support change), and the ROM is one table, not two kernels. If K shrinks storage but not issued taps, X is empty.

### Strongest controls
- Unconstrained 40-coeff lifting (AEE 1.232979368, 159+35)
- Independent low-rank kernels per consumer with total parameter count = K (storage matched, producer not shared)
- Tied taps without learnable coordinates (hard TSM shifts)
- Ordinary dense T10 (AEE 1.219801338)
- Same SIMD, same ports; knot table is a compile-time constant after the pair

### Kill-gate (number)
Declare K before the pair (suggested K ∈ {4,6,8}, K≤10). One pair. Kill if AEE Δ > +0.005 vs 1.219801338. Kill if issued T10 taps remain 10 (coordinates did not change producer support). Kill if the two consumers’ interpolated kernels diverge enough that the compiler must emit two source graphs (shared table failed). Hardware: long-BP still 8088 **and** source add/sub not below the lifting 159 **and** consumer table not below 714889 — no legal charge.

### Two-sentence TCAS-II pitch
Dual consumers of a noncausal T10 source cannot skip a time tap unless they share the time axis. Training one knot table that both the gate and the continuous PED interpolate from is a coordinate constraint, not a new filter: unused knots delete producer work for both readers.

### Biggest objection
“This is just fewer lifting coefficients.” The control with two independent K-parameter kernels (same storage, no shared support) must lose on producer slots; if it does not, the objection stands and I4 is dead.

### Assumptions
T10’s 40 coeffs are overcomplete relative to optical-flow temporal bandwidth. Gate and PED need similar group delay. Lifting family remains; only the parameterization of time changes.

### Predictions
1. Independent per-consumer K-kernels: AEE easier, source taps still 10.
2. Shared knots K≤8 after one pair: AEE ≤ 1.224801338; at least 2/10 taps compile to zero for both consumers; coefficient ROM is one table.
3. Knot locations concentrate on a subset of the 10 noncausal indices rather than uniformly.

### Disconfirmers
Optimal knots remain 10 distinct occupied taps. Consumers require two tables to meet +0.005. AEE recovery needs extra residual adapters that re-densify the source.

---

## P06-I5 — Prefix-sufficient T10: train the remainder to zero so ready/completion actually moves

One sentence: Keep noncausal T10 as a family but train a prefix of length k<10 to be sufficient for **both** consumers, with the suffix forced to exact integer zero, so the producer’s completion token can fire before slot 10 and the 8088 long-backpressure can finally move.

### A (complete prior to copy)
Copy: anytime / early-exit networks (BranchyNet; Figurnov SACT; MSDNet) **without** adding a second head — the original output is reused; online/prefix FIR and polyphase “output ready before last tap”; causal lifting as a constrained subset of Sweldens; trained remainder bounds from approximate computing (chip-level: skip trailing terms when the trained residual is below ULP). The copied object is “train a prefix to replace the full sum,” not an extra classifier.

### B (hole in this net)
The most important frozen observation is not 260→159 add/sub. It is: always-ready 6938→5354 while **long backpressure stays 8088 for both**. Completion still waits on full noncausal T10, for both consumers. Add/sub reduction is absorbed. No training signal currently exists for ready/valid. Dual consumers sharing one completion token is the hole: a prefix that only the gate can use does not retire the token.

### X (why not a reskin)
Early-exit papers add heads. Here there is no extra head and no extra loss tower: the suffix coefficients / suffix source lanes are trained to the **same exact integer zero** both consumers already implement at 0-difference. That changes the **completion/ready dependence** of the existing two-stage SIMD, which is the quantity the observation says did not move. If k=10 still wins, X is empty.

### Strongest controls
- Ordinary T10 and unconstrained lifting T10, both long-BP 8088
- Hard-causal k-prefix with **no** recovery (expect AEE death)
- Prefix for gate only vs prefix for both (only both may move the shared completion)
- Same ports, same state; suffix lanes clock-gated or not issued, not computed-and-discarded

### Kill-gate (number)
Declare k∈{4,5,6,8} before the pair. One pair. Kill if AEE Δ > +0.005 vs 1.219801338. **Hardware kill is primary:** long-backpressure remains 8088 at the same two-stage SIMD resource. Also kill if suffix integer values are not 0 on the captured windows (completion cannot legally fire). If k≥8 is required, chargeable occupancy is too small unless long-BP still drops proportionally — otherwise kill.

### Two-sentence TCAS-II pitch
Lifting already reduced arithmetic and did not reduce long backpressure: both graphs sit at 8088 because T10 is noncausal and both consumers wait on the last tap. Training a dual-consumer prefix with a suffix of exact zeros is the rare training idea that targets the stall the circuit actually has, not the add/sub ledger.

### Biggest objection
Noncausal T10 exists because optical flow on DSEC needs future context; a prefix will not recover +0.005. That is a fair kill, and it must be allowed to kill in one pair rather than by adding a future-buffer module.

### Assumptions
A large fraction of T10 energy for this student is causal or near-causal. Both consumers can accept the same k. Suffix zeros at the integer alphabet do not require extra sat logic (reuse existing q24).

### Predictions
1. Hard k=6 with no FT: AEE Δ ≫ +0.005.
2. One paired remainder-zero training at k=6: AEE ≤ 1.224801338; long-BP < 8088, roughly scaling with k/10 if issue is tap-serial.
3. Gate-only prefix: AEE maybe recoverable, long-BP still 8088 because PED holds the token.

### Disconfirmers
Long-BP is not tap-serial (k/10 does not scale 8088) — then prefix training cannot charge, even if AEE passes. Suffix refuses to hit exact 0 without AEE death. Dual consumers need two different k.

---

## P06-I6 — Intersection prune of patch-r1 producer support for both consumers

One sentence: Prune r1 residual channels (the historically expensive patch-embed residual chain) using `min(saliency_gate, saliency_PED)` so removed channels leave the **producer** graph, not just one consumer’s MAC list, then recover AEE in one pair.

### A (complete prior to copy)
Copy: channel pruning (He et al. ICCV 2017; Network Slimming Liu 2017; Molchanov importance); residual-branch skipping (SkipNet; Conv-AIG) as the “maybe the residual is unused” ancestor; official common-subexpression elimination as the **already applied** compiler pass (260 vs 159) that pruning must beat, not repeat. The copied object is structured channel removal with a saliency rule, compiled into a smaller source.

### B (hole in this net)
Old activity-weighted-dot put whole patch ~35% of a **proxy**. Proxy ≠ this student’s cycle share, so “prune the expensive layer” is not an observation. What is observed: dual consumers after the source; official CSE already spent the easy sharing; unconstrained lifting still leaves 35 RNE/sat and +0.013 AEE. A channel that only PED uses cannot be deleted from the producer. Intersection support is the only prune that changes **producer** add/sub and **both** consumer integer counts.

### X (why not a reskin)
Standard Slimming uses one BN-γ per channel. Here saliency is the **min** of two consumer-specific scores on the same r1 tensor, and the compile target is the source graph that feeds both. Gate-only prune and PED-only prune are mandatory losers in the control table. Beating official CSE (not ignoring it) is part of X.

### Strongest controls
- Ordinary 260 add/sub vs lifting 159+35 vs intersection-pruned graph
- Gate-only vs PED-only vs union vs intersection channel masks at the same keep-ratio
- Magnitude prune without recovery vs one paired recovery
- Do not quote the old 35% proxy as a result

### Kill-gate (number)
Declare keep-ratio before the pair (suggested keep ≤ 0.75 of r1 output channels). One pair. Kill if AEE Δ > +0.005 vs 1.219801338. Kill if source add/sub does not fall **below** the relevant baseline (ordinary 260 if starting dense; lifting 159 if starting lifting) **or** falls only by CSE-equivalent identities the compiler already had. Kill if consumer integer cycles do not fall on **both** consumers (one-sided prune). Never add source and consumer tables.

### Two-sentence TCAS-II pitch
A residual channel is not expensive because an old proxy said 35%; it is expensive when both consumers still demand it from the producer. Training an intersection support on r1, then compiling the smaller source, is pruning that the dual-consumer graph can actually retire.

### Biggest objection
CSE already removed sharing; further channel prune will hit the +0.005 wall in one shot, as lifting already did at +0.013. If that happens, I6 is dead — the freeze already warns that structure regresses AEE.

### Assumptions
r1 still contributes measurable producer work on the new student’s cycle share (must be re-measured; if it does not, I6 is the wrong island). Gate and PED saliency correlate on r1 channels. Prune is compile-time, not dynamic.

### Predictions
1. Gate-only keep-0.75: gate consumer cycles drop, source add/sub ≈ baseline.
2. Intersection keep-0.75 after one pair: AEE ≤ 1.224801338; source add/sub strictly below baseline; both consumer tables drop.
3. Union mask matches dense producer (almost no legal charge).

### Disconfirmers
New-student cycle share of r1 is negligible (B was a proxy ghost). Intersection keep-ratio needed for AEE is ≥ 0.95. Pruned graph is CSE-isomorphic to the unpruned graph (no new legal charge).

---

## P06-I7 — Train frozen projection-BN statistics so full-domain mean/var become compile-time constants

One sentence: Distill the native projection BN’s full 10×96×120×160 batch statistics into frozen running (or folded) constants with one paired recovery, then legally charge the disappearance of the mean/var reduction wait that local-window replay currently gets for free.

### A (complete prior to copy)
Copy: BatchNorm (Ioffe–Szegedy) and the inference contract of frozen running stats; conv–BN folding into integer affine (Jacob QAT; TensorRT-style fold); “precise BN” / eval-BN re-estimation over the real activation domain (Wu & Johnson; cross-iteration BN) as the method to **match** full-domain stats rather than hope EMA is close; integer BN (Banner et al.) only as the arithmetic after freeze. The copied object is “replace batch stats with compile-time affine,” including the re-estimation step people skip.

### B (hole in this net)
The freeze says native projection BN uses **actual batch statistics over the full 10×96×120×160 domain**, not frozen running stats, and that local-window replay given free mean/var **undercharges wait/storage**. Integer gates/I24/PED q24 are already 0-difference — BN is the remaining non-compile-time statistic on the projection residual path. This is a real producer/consumer stall that no lifting add/sub ledger currently accounts for.

### X (why not a reskin)
“Switch BN to eval()” is not a paper. The increment is a measured AEE gap between full-domain batch stats and frozen stats, closed by **one** paired training/re-estimation so the folded integer affine is 0-mismatch on the captured windows **and** AEE stays inside +0.005. Hardware then drops the full-volume reduction and the BN-ready dependence. If the gap is already < +0.005 without training, there is no X — report it as a negative observation and kill I7.

### Strongest controls
- Full-domain batch BN (native student) vs naive `eval()` running stats **without** the pair
- Re-estimated frozen stats over valid825 (precise-BN) without weight FT vs with one paired FT
- Local-window BN with **charged** mean/var (not free) vs folded constants
- Same ports; BN reduction tree is either issued or compiled out, no hybrid “free stats”

### Kill-gate (number)
First measure the AEE gap: AEE(frozen running) − AEE(full-domain batch). If that gap is already ≤ +0.005 **and** integer mismatch is 0, I7 has no training story — kill as “just fold BN.” If the gap is > +0.005, one paired freeze+FT: kill if AEE vs ordinary 1.219801338 stays > +0.005 **or** vs the native batch-BN student stays > +0.005 (declare the stricter of the two as the gate; both must pass). Kill if mean/var still require a full 10×96×120×160 reduction at runtime. Kill if folded integer affine leaves 0-difference on I24/PED.

### Two-sentence TCAS-II pitch
This student’s projection BN is not the textbook running-stat inference path; it is a full-volume statistic that window replay currently forgets to charge. Training those statistics into a folded integer affine, with a one-shot AEE gate against both the dense student and the batch-BN student, removes a real wait without inventing a new normalizer.

### Biggest objection
Reviewers will call this BN folding. Survive only with (i) the measured full-domain vs frozen gap on **this** 10×96×120×160 student and (ii) a service number that charges the reduction, not a claim that BN “is expensive” in general.

### Assumptions
The AEE gap is real and > +0.005 (to be measured first). Projection BN is on the dual-consumer path (if BN is only on a side branch neither consumer waits on, I7 cannot charge). Folding into the existing integer projection does not need extra bits beyond I24/q24.

### Predictions
1. Naive `eval()` without pair: AEE Δ > +0.005 vs batch-BN (otherwise I7 dies immediately).
2. One pair: AEE ≤ 1.224801338 vs ordinary **and** Δ vs batch-BN ≤ +0.005; runtime mean/var issue count = 0; integer 0-difference holds.
3. Local-window replay with charged reductions will show a wait that folded BN deletes; free mean/var will not.

### Disconfirmers
Gap already inside +0.005 (no paper). Folded affine needs wider than I24 to stay 0-difference. BN wait is hidden behind the 8088 T10 backpressure so folding does not move service (legal-charge failure even if AEE passes).

---

## P06-I8 — Integer-to-integer T10 lifting with trained rounding, dual-consumer bit-exact

One sentence: Replace float lifting + 35 data-dependent RNE/sat with Calderbank-style integer-to-integer lifting whose rounding policy is trained so **both** consumers stay bit-exact at the existing I24/q24 alphabet, then recover AEE in one pair.

### A (complete prior to copy)
Copy the whole integer-lifting chain: Calderbank–Daubechies–Sweldens 1998 lossless integer wavelet; JPEG2000 5/3 predict/update with floor; TCAS lifting-DWT architectures that implement rounding as a wired floor/sat (Andra et al. and subsequent multiplierless 5/3–9/7 datapaths). Copy QAT STE on the rounding dead-zone so the **learned** predict/update (the 40 coeffs) remain trainable through floor. Do not copy analog or approximate-compute rounding as the identity.

### B (hole in this net)
Lifting’s 159 add/sub still sit next to **35 intermediate RNE/sat**. Those nodes are why the graph is not integer-to-integer, why part of always-ready movement is generic round→sat fusion, and why unconstrained lifting already lost +0.013 AEE. Integer gates/I24/PED q24 are 0-difference **after** this source — the hole is the source’s own rounding, which dual consumers both see. A rounding policy that is not trained will either break 0-difference or fail +0.005.

### X (why not a reskin)
I1 quantizes **coefficients** to CSD (multiplierless). I8 quantizes the **datapath** to integer lifting (rounding-as-floor, no 35 RNE). A paper can run both, but they are different legal-charge objects: I1 deletes multiplies; I8 deletes intermediate float rounding and makes the source as bit-exact as the already-exact integer consumers. JPEG2000 5/3 uses fixed coeffs; here rounding is trained through 40 learned steps for two consumers. If 35 RNE/sat compile away by the existing round→sat fusion without training, I8 has no X — the freeze already said part of −22.83% is that fusion.

### Strongest controls
- Unconstrained float lifting (1.232979368, 159+35, 5354, 8088)
- Wired floor on the same 40 coeffs **without** the pair
- I1-style CSD with float rounding still present (separates coeff alphabet from datapath integer)
- Ordinary dense integer source (0-difference gates/PED as the bit-exact north star)
- Same SIMD; count remaining data-dependent RNE, not “we sat at the end”

### Kill-gate (number)
One pair. Kill if AEE Δ > +0.005 vs 1.219801338. Kill if data-dependent intermediate RNE/sat count does not go from 35 to 0. Kill if either consumer leaves 0-difference on the two captured windows. Kill if long-BP remains 8088 **and** the only “gain” is the generic round→sat fusion already included in −22.83% (cannot charge it twice).

### Two-sentence TCAS-II pitch
The lifting source is not yet an integer circuit: 35 RNE/sat stand between 159 adds and two consumers that are already bit-exact at I24/q24. Training integer-to-integer rounding on the learned T10 steps, with a one-pair AEE gate against the dense student, is the datapath co-design that makes lifting chargeable rather than merely smaller on an add/sub ledger.

### Biggest objection
Integer lifting is 1998 and JPEG2000. Survive only by showing (i) learned 40-coeff T10, (ii) dual-consumer bit-exactness, (iii) AEE recovered under +0.005 after the freeze’s +0.013, (iv) a stall that is not the already-counted round→sat fusion.

### Assumptions
Floor/sat STE is stable on this student for one FT. Dual consumers can share one rounding policy. Integer lifting does not expand dynamic range beyond q24/I24. Lifting family stays.

### Predictions
1. Wired floor, no FT: AEE Δ > +0.005, likely worse than +0.013; bit-exactness may still hold.
2. One paired trained-rounding run: AEE ≤ 1.224801338; intermediate RNE/sat = 0; gates/I24/PED remain 0-difference.
3. Combined with I1 (CSD+integer lifting) is a **fusion candidate**, not this idea’s identity; this idea must pass alone.

### Disconfirmers
Round→sat fusion already removed the 35 nodes in the current compiler (nothing left to train). Trained floor meets AEE only by reintroducing float residuals. Dynamic range overflows q24, breaking the 0-difference that is the paper’s integer credibility.

---

## Cross-idea notes for the owner (not extra modules)

- **AEE is the shared wall.** Unconstrained lifting already failed +0.005 at +0.013178. Every idea above spends its **one** pair on that wall. A second pair is a kill, not a revision.
- **8088 is the shared hardware wall.** Ideas that only shrink 260→159 without moving long-backpressure are legally unchargeable. I5 is the direct attack; I1/I8 attack it only if RNE/sat or taps sit on that path; I2/I3/I6 attack issued lanes; I4 deletes shared taps; I7 attacks a different wait (BN reduction) that must be shown not hidden behind 8088.
- **Both consumers or no producer skip.** Gate-only structure is the standard false positive in this net.
- **θg stays real.** Product zeros (I2) and N:M (I3) must not collapse the paper to binary ATLIF.
- **Do not add the source SIMD table to the consumer integer table.** Report one island per idea.
- **Suggested measurement order (still one pair per idea, not a zoo):** I7 gap probe (maybe dies in a day) → I5 long-BP probe → I2/I3 intersection sparsity → I6 r1 share re-measure → I1/I8 lifting datapath if lifting remains the title.

End of P06 independent set (I1–I8).
