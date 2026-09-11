# P01 independent ideas — hostile-but-constructive TCAS-II circuits reviewer

Perspective: IEEE TCAS-II Express Briefs (5-page letter). One mechanism, one causal occupancy/critical-path figure, component PPA only after a numeric kill-gate, relative prior mandatory. Reviewers punish CSE-count theater, component-FPS multiplication, OpenROAD-as-foundry, analog CIM, and binary-ATLIF identity changes.

Freeze used: `PROBLEM.md` dated 2026-09-11 only. Every item below is an **idea**, **assumption**, or **prediction**, never a finding. Origin of all eight: AI-assisted. Stage of all eight: independent.

Tensions forced into the set: performance vs ports; sparsity vs dual-consumer residual; compile-time vs run-time; source-gain vs backpressure.

Numeric rails copied from the freeze (not claims of novelty): ordinary valid825 AEE = 1.219801338; learnable 40-coeff lifting T10 AEE = 1.232979368 (delta +0.013178; abs 1.259 passes, relative +0.005 fails); CSE ordinary 260 add/sub per T10 vector vs lifting 159 add/sub + 35 intermediate RNE/sat; same two-stage SIMD always-ready slots 6938 → 5354 (−22.83%) with long backpressure both 8088; part of −22.83% is generic round→sat fusion; separate integer consumer model 758777 → 714889 (−5.78%) at a different resource point — do not add the two tables; native projection BN uses actual batch statistics over 10×96×120×160, not frozen running stats; integer gates / I24 / PED q24 = 0 difference vs model on two captured windows.

Same-resource rule for every kill-gate: same port count, same architectural state class, same backpressure protocol family, fusion-matched arithmetic, full r1→T10→{gate, PED} chain. Target net service ≥15%. Do not quote consumer −5.78% as chain gain.

---

## P01-I1 Dual-credit elastic tap at the T10 dual-consumer fork

- **id:** P01-I1
- **one-sentence statement:** Idea: convert the absorbed T10 source-gain into dual-consumer occupancy with a depth-1/2 elastic tap that gives gate and continuous PED independent credits on one source sample, without extra SRAM ports.
- **origin:** AI-assisted
- **stage:** independent
- **A complete prior to copy:** Full latency-insensitive / synchronous-elastic protocol, not a nameless FIFO. Copy Carloni–McMillan–Sangiovanni-Vincentelli latency-insensitive design (IEEE TCAD 2001) plus Cortadella–Carmona–Kishinevsky synchronous elastic circuits (IEEE TCAD 2006), including the relay-station / elastic-buffer firing rules and a 1-bit credit (Dally-style) per consumer. The copied artifact is the handshake + occupancy invariant, not a tool flow.
- **B hole in THIS network (PROBLEM.md only):** Same two-stage SIMD resource already shows always-ready 6938 → 5354 (−22.83%) while long backpressure stays 8088 on both ordinary and lifting — source arithmetic gain is absorbed. Dual consumers after the source (spike/gate and continuous residual/PED) imply a joint ready that can hold the source even when only one consumer is stalled. Part of −22.83% is generic round→sat fusion, so the remaining lifting-specific always-ready movement is even smaller and still does not move the 8088 class.
- **X why not a reskin:** Not “deeper FIFO” and not “lifting uses fewer adds.” The mechanism is a two-tap elastic register with *independent* credits so source stall becomes max(T_gate, T_PED) minus documented overlap, rather than a single AND-of-readys. Ordinary and lifting both sit behind the same tap; lifting is not the title.
- **strongest controls:** (C1) ordinary dense source, joint ready, no tap; (C2) ordinary + same elastic tap; (C3) lifting + joint ready; (C4) lifting + same tap. All fusion-matched (round→sat on or off together). Same two-stage SIMD ports. Depth cap = 2 registers (same-state class). Do not add the consumer-table −5.78%.
- **kill-gate number:** Full-chain same-resource net service ≥15% versus C1, **and** long-backpressure count must drop ≥15% from 8088 (always-ready-only improvement is insufficient). valid825 AEE vs ordinary ≤ +0.005 and absolute ≤ 1.259. If long-backpressure stays 8088, the idea is dead.
- **two-sentence TCAS-II pitch:** Copy the elastic-buffer protocol onto the T10 fork so gate and PED accept one source vector on staggered credits under the same SRAM ports. The letter lives or dies on one occupancy figure showing the 8088 class shrinking at equal state, not on a 260→159 add/sub table.
- **assumptions:** Assumption: a non-trivial fraction of the 8088 long-backpressure cycles is joint-ready coupling of the two consumers rather than a pure RAW from T10 result to PED. Assumption: depth-1/2 is enough because the freeze already has always-ready slack of 6938→5354 on the source side.
- **predicted observations:** Prediction: C2 (ordinary+tap) already moves long-backpressure and service; C4 vs C2 is within measurement noise if fusion is matched. Prediction: causal figure is occupancy of {source, gate, PED} vs cycle, with a visible overlap band that C1 lacks.
- **disconfirming evidence:** If an instrumented stall breakdown shows 8088 ≈ 100% PED-RAW on the current T10 vector, credits cannot create overlap and the idea dies. If depth-2 already violates a same-state comparison the venue will demand, the idea dies.
- **biggest objection:** “You added skid buffers and called it a protocol paper.” Hostile control: report C2 vs C1 first; if ordinary+tap already captures the 15%, lifting must leave the title.

---

## P01-I2 Compile-time union-CSE channel mux (ordinary ∪ lifting) under the +0.005 AEE rail

- **id:** P01-I2
- **one-sentence statement:** Idea: meet the failed lifting relative AEE by a compile-time per-channel enable mask on a single union CSE DAG, keeping AEE-sensitive T10 channels on the ordinary 260-add graph and the rest on the 159-add lifting graph.
- **origin:** AI-assisted
- **stage:** independent
- **A complete prior to copy:** Complete lifting-based DWT datapath family (Sweldens lifting construction; Andra–Chakrabarti–Acharya VLSI lifting wavelet, IEEE TSP 2002; multiplierless / CSE FIR and DWT letters in TCAS) **plus** quality-constrained hybrid exact/approximate datapath mapping (Chippa/Venkataramani-style quality knobs, used here only as a static mask, not as run-time approximate MAC). Copy the adder-graph construction and the static mapping procedure, not a second accelerator.
- **B hole in THIS network (PROBLEM.md only):** Drop-in learnable 40-coeff lifting T10 AEE = 1.232979368 vs ordinary 1.219801338 (delta +0.013178) fails the relative +0.005 rail while still passing absolute 1.259. CSE already shrank 260 → 159 add/sub + 35 RNE/sat, and the same-SIMD always-ready gain is absorbed by 8088 backpressure. A pure lifting island therefore has neither relative accuracy nor demonstrated chain service.
- **X why not a reskin:** Not “train lifting until AEE moves” and not two side-by-side sources (that would explode ports). X is one union DAG with compile-time edge enables so the SIMD resource stays two-stage and both dual consumers still see a single T10 vector per channel. The mask is frozen after a one-shot valid825 sensitivity sweep; it is not a run-time mux opcode farm.
- **strongest controls:** All-ordinary (AEE 1.219801338); all-lifting (AEE 1.232979368); random channel masks at the same ordinary-fraction as the sensitivity mask; union-DAG sensitivity mask. Fusion-matched. Same ports. Report add/sub as the enabled-edge count of the union, not 159×(1−f)+260×f invented in software.
- **kill-gate number:** valid825 AEE vs ordinary ≤ +0.005 **and** enabled add/sub ≤ 0.85 × 260 (arithmetic room for a 15% service attempt) **and** full-chain same-resource net service ≥15% vs all-ordinary. If the mask that meets +0.005 keeps >70% of channels ordinary, kill (assumption: that leaves no 15% chain room after 8088 absorption).
- **two-sentence TCAS-II pitch:** Copy a lifting CSE adder-graph, then publish a static channel mask that is the only increment, with ordinary as the relative prior. One Pareto figure (ordinary-fraction vs AEE vs always-ready) must cross +0.005 before the arithmetic budget collapses, or the letter is a negative result.
- **assumptions:** Assumption: AEE sensitivity of T10 channels is concentrated, not linear in ordinary-fraction. Assumption: dual consumers can share the muxed vector without a second source fire (performance vs ports).
- **predicted observations:** Prediction: a minority of channels account for most of the +0.013178. Prediction: random masks at the same fraction miss the +0.005 rail (this is the anti-reskin control).
- **disconfirming evidence:** Linear AEE vs ordinary-fraction with intercept near +0.013178. Union-DAG area/ports exceeding the two-stage SIMD envelope. Mask that works on two captured windows but fails full valid825.
- **biggest objection:** “This is a design-space sweep, not a circuit mechanism.” Counter required in the letter: the union DAG with gated edges must be the *same* PE as ordinary CSE, shown in one schematic, or it reads as two priors glued.

---

## P01-I3 Same-port two-phase Welford for native full-domain projection BN

- **id:** P01-I3
- **one-sentence statement:** Idea: stop undercharging projection-BN wait by streaming exact batch moments over the true 10×96×120×160 domain with Welford on the existing SIMD ports, riding the noncausal T10 buffer instead of a second statistics memory.
- **origin:** AI-assisted
- **stage:** independent
- **A complete prior to copy:** Welford’s online algorithm (1962) and Chan–Golub–LeVeque pairwise moments for the numeric method; two-pass streaming BN / online-moments circuits for the microarchitecture. Do **not** copy inference BN-folding (TensorRT-style running-stat fold-into-conv): that prior assumes frozen running stats, which this student does not use. Ioffe–Szegedy defines the math identity to preserve.
- **B hole in THIS network (PROBLEM.md only):** Native projection BN uses **actual batch statistics over the full 10×96×120×160 domain**, not frozen running stats. Local-window replay given free mean/var undercharges wait/storage. Integer gates / I24 / PED q24 already 0-diff on two windows, so the unaccounted sequential barrier on the native projection+BN+residual-add path is a plausible absorber of any T10 source-gain (8088 long backpressure on both graphs). Unknown (SCOPE): true cost of those full-domain stats.
- **X why not a reskin:** Not “fold BN” (would change the frozen student). Not analog CIM. X is exact batch moments computed on the same ports as the projection conv, with a compile-time tile order such that pass-1 Welford of tile k+1 overlaps pass-2 affine of tile k only inside the T=10 noncausal buffer that already exists for PSN. Storage of 96 means/vars is not the claim; the claim is barrier hiding.
- **strongest controls:** (C1) captured actual-batch BN (identity); (C2) frozen running-stats BN, same integer pipeline; (C3) local-window replay with *free* mean/var (the undercharge); (C4) two-pass Welford, no overlap; (C5) two-pass Welford overlapped on T10 buffer, same ports. Fusion-matched T10 source held fixed (ordinary, AEE 1.219801338).
- **kill-gate number:** First gate: if C2 AEE vs C1 ≤ +0.005 and abs ≤ 1.259, **kill the island and freeze running stats** (no circuit letter). Else: C5 bit-true vs C1 on projection output **and** full-chain same-resource net service ≥15% vs C4. Do not claim service vs C3 (C3 is an undercharged fantasy baseline).
- **two-sentence TCAS-II pitch:** The relative prior is two-pass batch BN, not running-stat folding; the increment is Welford that rides the noncausal T10 line buffer on the same SIMD ports. One causal figure (reduction barrier vs T10 window complete) plus a frozen-stat AEE control decides whether the letter exists.
- **assumptions:** Assumption: the “10” in 10×96×120×160 is the same noncausal T10 axis as the PSN, so the line buffer already holds the reduction domain. Assumption: BN wait is on the r1 residual completion path that can hold T10’s dual consumers.
- **predicted observations:** Prediction: C2 misses +0.005 (otherwise this island is unnecessary). Prediction: C3 under-reports wait by a large factor versus C4. Prediction: C5 recovers most of C4’s wait without a second 10×96×120×160 store.
- **disconfirming evidence:** C2 already inside +0.005. Pairwise/Welford integer drift vs captured BN (must be 0 at the same q24 policy that already matches gates/I24/PED). Overlap illegal because projection output is not in the T10 buffer (assumption broken).
- **biggest objection:** “O(C) mean/var is cheap; you invented a paper around a barrier you have not measured.” Fair. The first figure must be C3 vs C4 wait on the captured student, before any architecture cartoon.

---

## P01-I4 Fork-fused RNE/sat presentment, with fusion-matched ordinary as the real prior

- **id:** P01-I4
- **one-sentence statement:** Idea: make the generic round→sat fusion a single I24 presentment at the source–consumer fork so both gate and PED see one quantized sample, and only then measure whether lifting still has any same-port service left.
- **origin:** AI-assisted
- **stage:** independent
- **A complete prior to copy:** Parhi operator fusion / strength reduction for DSP graphs; saturating SIMD arithmetic as a fused round-saturate PE (the QADD-class circuit, copied as a boundary stage, not as an ISA slogan). Elastic boundary register from the P01-I1 prior family if a skid is required, but the copied core is the fused RNE+sat cell and its placement.
- **B hole in THIS network (PROBLEM.md only):** Explicit freeze sentence: part of the −22.83% always-ready (6938 → 5354) is generic round→sat fusion. Long backpressure is 8088 on **both** graphs, so lifting CSE (260 → 159 + 35 RNE/sat) has not demonstrated a fusion-isolated chain gain. Dual consumers need quantized I24 and continuous PED; a source-interior sat can make the 35 lifting RNE/sat look like work while the fork still pays twice. Consumer-table −5.78% is a different resource point and must not be added to −22.83%.
- **X why not a reskin:** The title is the **fork-fused quantizer**, not lifting. Ordinary+fusion is the missing relative prior the current tables do not provide. X is collapsing two consumers’ quant waits into one presentment without extra ports; the 35 interior RNE/sat of lifting become a liability unless they disappear into that single boundary sat.
- **strongest controls:** Ordinary, unfused; ordinary, source-interior fusion; ordinary, fork fusion; lifting, interior 35 RNE/sat; lifting, fork fusion only (interior sats compiled off). Same two-stage SIMD. Never add source and consumer tables.
- **kill-gate number:** Versus ordinary+**same** fork fusion, full-chain same-resource net service ≥15%. If lifting+fusion ≈ ordinary+fusion, lifting leaves the title. Fork-fusion vs unfused ordinary must itself clear ≥15% or the idea is ordinary engineering. AEE vs ordinary ≤ +0.005, abs ≤ 1.259 (integer 0-diff on captured gates/I24/PED q24 must remain).
- **two-sentence TCAS-II pitch:** TCAS-II will subtract generic fusion from your −22.83% unless you publish the fusion-matched ordinary control; this letter *is* that control plus a fork placement that serves both consumers once. The causal figure is sat-stage occupancy at the dual-consumer fork, not an add/sub count.
- **assumptions:** Assumption: a measurable slice of dual-consumer wait is duplicated quantization/presentment rather than PED arithmetic. Assumption: PED q24 can legally consume the same I24 presentment as the gate path without an extra RNE (supported by the freeze’s 0-diff integer windows, but that is still an assumption for the full valid825 chain).
- **predicted observations:** Prediction: ordinary+fork-fusion captures most of the reported always-ready movement. Prediction: lifting’s residual always-ready after fusion matching is <<22.83% and still does not move 8088. Prediction: fork fusion can move consumer occupancy even when source CSE is held ordinary.
- **disconfirming evidence:** Fusion-matched ordinary already equals lifting on always-ready **and** fork placement does not move 8088. PED requires a different rounding mode than gate, forcing two presentments (ports blow up).
- **biggest objection:** “Operator fusion is expected; five pages need a mechanism.” If the occupancy figure does not show a fork-specific stall class, this is a checklist item in the methods, not a letter. That is an acceptable kill.

---

## P01-I5 Asymmetric issue: skip-gate, never-skip PED, steal the freed port

- **id:** P01-I5
- **one-sentence statement:** Idea: spend gate-path inactivity on extra residual/PED work under a fixed 2-port budget, never skipping the continuous consumer that makes naive sparsity illegal on this net.
- **origin:** AI-assisted
- **stage:** independent
- **A complete prior to copy:** Zero-skipping MAC handshake from SCNN / Cambricon-X / SparTen / Eyeriss-class skip, **copied together with** the residual-mandatory path of ResNet-style accelerators that still execute the add. Do not copy binary-SNN accelerators as identity. Do not title a threshold comparator (SCOPE forbids GrokBot-comparator-as-title).
- **B hole in THIS network (PROBLEM.md only):** Dual consumers after the source: spike/gate **and** continuous residual/PED. Neuron is ATLIF with **continuous** θg, not binary spikes. Historical activity-weighted-dot ledger put whole patch ~35% of a proxy that is explicitly *not* this student’s cycle share. Any skip that drops the source sample starves PED; any skip that idles the port fails the same-port service test. Long backpressure 8088 on both dense graphs shows the consumer side can absorb source-gain; reclaiming skipped-gate cycles is one way to spend that side.
- **X why not a reskin:** Not “sparse SNN” and not MX3P/binary ATLIF. Mechanism: source sample always fires once (dense residual identity); gate PE may skip on a *compile-time integer test of already-quantized* gate operands (not a new learned threshold); the skipped issue slot is legally stolen by PED/residual on the same two ports. Continuous θg is unchanged.
- **strongest controls:** Dense both; skip-gate but idle the port (skip without steal); skip-gate with steal; skip-both (illegal, expect AEE fail). Same ports, same state, fusion-matched ordinary T10 (do not mix in lifting until AEE of the skip itself is clean).
- **kill-gate number:** Dual-consumer pair same-resource net service ≥15% vs dense-both. Residual/PED path remains 0-diff on the two captured integer windows; valid825 AEE vs dense ≤ +0.005, abs ≤ 1.259. Assumption used as a numeric tripwire: if measured gate-skip *rate* on valid825 < 20%, kill (overhead cannot credibly yield 15% net).
- **two-sentence TCAS-II pitch:** The relative prior is zero-skip MAC plus a mandatory residual; the increment is an issuer that converts a skipped gate cycle into PED work at equal ports. One figure (skip rate vs stolen-slot occupancy vs AEE) must show service moving while the continuous consumer stays bit-true.
- **assumptions:** Assumption: the gate path has ≥20% skippable issue slots on valid825 even with continuous θg. Assumption: PED has legal extra work in those same cycles (backlog), otherwise steal has nothing to eat. Assumption: the skip test uses existing quantized gate values, not a new analog or learned comparator.
- **predicted observations:** Prediction: skip-without-steal shows occupancy holes and ~0 service gain (proves steal is the X). Prediction: skip-both breaks AEE far beyond +0.005. Prediction: steal reduces 8088 only if those stalls are PED-side backlog, not source-side emptiness.
- **disconfirming evidence:** Gate path too dense under continuous θg (skip rate <20%). Steal requires a third port or extra scoreboard state (fails same-port/same-state). AEE moves when gate skips even if PED is dense (hidden coupling).
- **biggest objection:** Continuous θg was chosen exactly to avoid binary sparsity stories; a skip-gate letter will be read as identity drift unless the residual 0-diff control is the first figure, not an appendix.

---

## P01-I6 Static MCM binding of the already-CSEd T10 DAG versus SIMD interpretation

- **id:** P01-I6
- **one-sentence statement:** Idea: bind the frozen T10 CSE graph (ordinary 260 or lifting 159+35 RNE/sat) to a hardwired MCM/CSD adder-graph with a compile-time as-soon-as-ready schedule, instead of interpreting that graph at run time on the two-stage SIMD that still posts 8088-class stalls.
- **origin:** AI-assisted
- **stage:** independent
- **A complete prior to copy:** Multiplierless MCM family in full: RAG-n (Dempster–Macleod), Hcub (Voronenko–Püschel, ACM TECS 2007), CSD/SOPOT FIR (Lim discrete-coefficient FIR, Hartley CSE). Folded FIR / static schedule from Parhi VLSI DSP. If lifting coeffs are kept, also copy the lifting DWT adder-graph (Andra–Chakrabarti–Acharya) as the *source graph*, then re-bind; do not claim CSE itself as X (CSE is already in the freeze).
- **B hole in THIS network (PROBLEM.md only):** CSE is done: 260 vs 159+35. Constants are compile-time complete. The same two-stage SIMD resource still shows long backpressure 8088 on both graphs; always-ready movement is contaminated by generic fusion. The hole is run-time interpretation of a compile-time DAG, including 35 intermediate RNE/sat as run-time ops rather than prescribed pipeline stages.
- **X why not a reskin:** Not “we counted 159 adds.” X is static binding of that DAG onto the **same** two-stage resource with a proven source-internal ready schedule, so the source itself cannot emit 8088-class self-stalls. Optional sub-X: requantize the 40 lifting coeffs onto SOPOT so interior RNE/sat → ≤1 output sat, but only if AEE vs ordinary ≤ +0.005 (currently +0.013178).
- **strongest controls:** SIMD-interpreted ordinary CSE; SIMD-interpreted lifting CSE; hardwired MCM ordinary, fusion-matched; hardwired MCM lifting with 35 sats as stages; SOPOT-requant lifting with output-sat only. Same port count. Report area-class state (adder-graph registers) against the same-state rule.
- **kill-gate number:** If an instrumented breakdown shows source-internal backpressure ≈ 0 and 8088 is 100% consumer-side, **kill** (MCM cannot move completion). Else: full-chain same-resource net service ≥15% vs SIMD-interpreted ordinary, AEE vs ordinary ≤ +0.005 (bit-true MCM of the same coeffs must be 0-diff on captured integer windows). SOPOT sub-variant additionally requires intermediate RNE/sat ≤ 1 and still ≤ +0.005 AEE; if SOPOT AEE stays ~1.233, kill that sub-variant.
- **two-sentence TCAS-II pitch:** Copy Hcub/RAG-n onto the freeze’s CSE DAG and show, against SIMD interpretation of the *same* DAG, that static binding is the increment. The letter needs a reservation-table figure at equal ports; a 260 vs 159 table without the SIMD-interpreted control is a reskin.
- **assumptions:** Assumption: a non-zero slice of 8088 is source-internal issue bubbles from interpreting CSE/RNE/sat. Assumption: folding MCM back onto two SIMD stages does not reconstruct the interpreter (otherwise X vanishes). Assumption: same-state can be met by using the existing SIMD registers as the MCM pipeline, not a fully unrolled 40-coeff array.
- **predicted observations:** Prediction: source-only occupancy improves; chain occupancy does not, unless P01-I1/I7 co-exist — this idea must still clear 15% **alone** or it is not a letter. Prediction: SOPOT will fail +0.005 just as the float 40-coeff lifting did, unless the discrete search is AEE-constrained rather than coefficient-MSE.
- **disconfirming evidence:** Source-internal ready already saturated. Folded MCM II equals SIMD-interpreted II. SOPOT add count ≥ 260.
- **biggest objection:** Same-resource folding reconstructs the two-stage SIMD; the “hardwired” claim is then a synthesis option, not a mechanism. Hostile reviewer asks for the reservation table of the folded MCM vs the existing SIMD, cycle by cycle.

---

## P01-I7 Compile-time modulo fill of the 8088-class stalls with independent r1 work

- **id:** P01-I7
- **one-sentence statement:** Idea: replace dynamic ready on r1→T10→{gate,PED} with a compile-time modulo schedule whose empty slots are filled by independent r1-conv work, turning absorbed source-gain into residual-chain occupancy at equal ports.
- **origin:** AI-assisted
- **stage:** independent
- **A complete prior to copy:** Synchronous dataflow static scheduling (Lee–Messerschmitt, Proc. IEEE 1987), cyclo-static SDF, and iterative modulo scheduling (Rau; Lam software pipelining). Parhi unfolding/folding for the circuit binding. Copy the topology → reservation-table → II procedure in full, including how dual-rate actors are encoded. Not “we ran HLS.”
- **B hole in THIS network (PROBLEM.md only):** Always-ready 6938 → 5354 with long backpressure **both** 8088: arithmetic CSE does not fill the stall class. Finite ports, complete/ready dependencies, complete constant compilation are in-scope. Expensive region is patch-embed residual r1 (two convs) plus T10 PSN with dual consumers. Dynamic ready leaves holes that r1’s two convs could legally occupy if they do not depend on *this* T10 vector.
- **X why not a reskin:** Not retiming slogans and not C-slow (C-slow multiplies state and fails the same-state test). X is a reservation table whose II is derived from the measured 8088-class stall and the dual-consumer fork, with legal fillers restricted to r1 work that is data-independent of the in-flight T10 vector. Backpressure cycles become scheduled issue, not stalls.
- **strongest controls:** Dynamic ready (freeze baseline); static II with *nops* in the 8088 slots (schedule without fill); static II with r1 fill; fusion-matched ordinary source held fixed. Same ports. Register count must not grow beyond the dynamic-ready scoreboard/skid class (no C-slow).
- **kill-gate number:** Captured-window integer 0-diff on gates/I24/PED q24 preserved; valid825 AEE vs ordinary ≤ +0.005, abs ≤ 1.259; full-chain same-resource net service ≥15% vs dynamic ready. If the 8088 class is a true RAW from this T10 result into PED/gate with **no** independent r1 actor legally issuable on those ports, kill.
- **two-sentence TCAS-II pitch:** Copy SDF modulo scheduling onto the freeze’s r1+T10 dual-consumer graph and use the 8088 stall class as the II constraint, not as a regret. The letter is one Gantt figure of those slots before/after fill, at equal ports and equal state.
- **assumptions:** Assumption: r1’s two convs have a legal pool of work independent of the in-flight T10 vector (channel/spatial tiling). Assumption: constants and shapes are compile-time complete enough for a static II (SCOPE: complete constant compilation). Assumption: filling does not require extra ports to r1 memories.
- **predicted observations:** Prediction: nop-static matches dynamic-ready service (proves fill, not “static” as such, is X). Prediction: fill moves net service even when T10 CSE is held ordinary, so this island does not need lifting. Prediction: AEE is bit-true if the schedule is only an issue permutation.
- **disconfirming evidence:** Port conflict on r1 weights/activations in the stolen slots. Hidden RAW from T10 into r1 residual add on the same vector. Static table that only works by adding scoreboard state (same-state fail).
- **biggest objection:** “Static scheduling of a frozen DAG is a compiler exercise.” The five-page defense is the circuit reservation table tied to *this* dual-consumer fork and the 8088 number; if the figure could be any conv+FIR pipeline, reject.

---

## P01-I8 Consumer-split exits on one T10 CSE DAG (lifting-class gate, ordinary-class PED)

- **id:** P01-I8
- **one-sentence statement:** Idea: give the AEE-sensitive continuous PED the ordinary T10 exit and the gate path a lifting-class exit on a *shared-prefix* CSE DAG, so dual consumers are no longer forced through the same accuracy/cost point.
- **origin:** AI-assisted
- **stage:** independent
- **A complete prior to copy:** Multi-filter shared-subexpression / MCM-for-several-constants family (Hartley CSE across multiple FIR transfer functions; Dempster/Macleod RAG-n for multiple constants; polyphase complementary filter banks with shared delay line, Vaidyanathan). Copy the joint adder-graph construction for two transfer functions, not two accelerators. Lifting DWT adder-graph is the cheap-exit prior; ordinary dense CSE is the exact-exit prior.
- **B hole in THIS network (PROBLEM.md only):** Dual consumers after source: gate and continuous residual/PED. Lifting drop-in fails relative AEE (+0.013178 vs +0.005) so PED, which already matches integer q24 at 0-diff, cannot take the cheap graph. Gate is a different consumer; forcing both through ordinary 260 add/sub or both through lifting 159+35 is the hole. Same-port SIMD and 8088 absorption forbid a second independent source (performance vs ports). Source-gain vs backpressure: extra arithmetic on PED is worthless if it serializes behind a joint ready (couple with I1 only as a control, not as this idea’s identity).
- **X why not a reskin:** Not two sources and not P01-I2’s per-channel all-or-nothing mask (that still feeds both consumers the same vector). X is **two exits on one DAG**: shared prefix CSE, cheap lifting-class exit into gate, ordinary-class exit into PED. One fire, two presentments from intermediate nodes, same two-stage ports via a compile-time interleaved drain.
- **strongest controls:** Both consumers ordinary (AEE 1.219801338); both lifting (AEE 1.232979368); this split; swapped split (cheap PED, exact gate — expect AEE fail). Shared-prefix disabled (two graphs, illegal-port control). Fusion-matched fork presentment (I4 as control, not mixed into the claim).
- **kill-gate number:** valid825 AEE vs ordinary ≤ +0.005 (PED must carry the identity) **and** union enabled add/sub ≤ 0.85 × 260 **and** full-chain same-resource net service ≥15% vs both-ordinary. If measured shared-prefix fraction < 20% of the ordinary 260, kill (two graphs in a trench coat; ports/state will fail). Swapped split must *fail* AEE or the accuracy story is not consumer-causal.
- **two-sentence TCAS-II pitch:** Copy joint MCM/CSE for two transfer functions and hang them on this net’s actual dual consumers: exact PED, cheap gate. One DAG figure with two exits, plus the swapped-split AEE control, is the letter; two SIMD sources is a reject.
- **assumptions:** Assumption: ordinary and lifting CSE graphs share a usable prefix (or can be rebuilt to share one without AEE drift on the ordinary exit). Assumption: gate-path T10 error of lifting magnitude is tolerated by valid825 when PED stays ordinary. Assumption: interleaved drain of two exits fits the existing two-stage ports (performance vs ports).
- **predicted observations:** Prediction: split AEE ≈ ordinary (≤ +0.005) while both-lifting stays ≈ 1.233. Prediction: swapped split lands near both-lifting, proving PED is the AEE owner. Prediction: service gain tracks the removed suffix on the gate exit, not 260→159 of the whole vector.
- **disconfirming evidence:** Gate-path lifting error leaks into PED through ATLIF continuous θg or residual coupling (AEE follows both-lifting). Prefix sharing <20%. Two exits require two concurrent source ports.
- **biggest objection:** “You implemented two filters and duty-cycled one SIMD; that is a reskin of both priors.” The shared-prefix fraction and the swapped-split AEE control are mandatory; without them this is a module zoo, which SCOPE forbids.

---

## Reviewer cross-cuts (still ideas, not findings)

These are constraints on how any of the eight should be written as a 5-page letter, not extra modules.

1. **Relative prior is the fusion-matched ordinary graph at the same two-stage SIMD ports.** Lifting 159 vs 260 without that control is not a submission. Consumer −5.78% must never be added to source −22.83%.
2. **Always-ready 6938→5354 is not throughput.** Long-backpressure 8088 is the number a causal figure must move, or the letter is arithmetic theater.
3. **Dual consumers make sparsity and source-skip illegal unless PED stays dense.** P01-I5 and P01-I8 exist because of that hole; they are not binary-ATLIF papers.
4. **BN actual-batch stats are a first-class barrier candidate.** Measure frozen-running-stat AEE before building P01-I3; that measurement is allowed to kill the island.
5. **Main title may drop lifting.** SCOPE allows changing the island if A+X is stronger. P01-I1, I3, I4, I5, I7 do not need lifting in the title; P01-I2, I6, I8 copy lifting as *a* prior, not as identity.
6. **Kill-gates are conjunctive:** AEE abs ≤ 1.259 **and** AEE vs ordinary ≤ +0.005 **and** full-chain same-resource net service ≥15%. Any two of three is a reject at this venue.

End of P01 independent set (8 ideas).
