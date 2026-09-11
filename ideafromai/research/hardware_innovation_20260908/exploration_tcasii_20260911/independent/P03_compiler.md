# P03 independent ideas — compiler / constant-matrix / DA / LUT / CSE / lifting

**Participant:** P03  
**Perspective:** compiler, constant-matrix (MCM/CMVM), da4ml, LUT/DA, CSE, lifting/wavelet, fast-transform hardware.  
**Inputs used:** `SCOPE.md`, `PROBLEM.md` only. Freeze date 2026-09-11.  
**Status:** ideas, not findings. No RTL, no `main.tex`, no catalog.

Frozen numbers this round treats as measurements, not novelty:

- Ordinary dense-source valid825 AEE = 1.219801338
- Learnable 40-coeff lifting T10 AEE = 1.232979368 (Δ +0.013178 vs ordinary; abs 1.259 passes, relative +0.005 fails)
- After official CSE: ordinary 260 add/sub per T10 vector; lifting 159 add/sub + **35 intermediate RNE/sat**
- Same two-stage SIMD, two writeback slots: always-ready 6938 → 5354 (−22.83%); **long backpressure both 8088** (algebraic gain absorbed). Part of −22.83% is generic round→sat fusion
- Separate integer consumer model (different resource point): 758777 → 714889 (−5.78%). Do not add the two tables
- Native projection BN: **actual batch statistics over full 10×96×120×160**, not frozen running stats. Local-window replay with free mean/var undercharges wait/storage
- Integer gates / I24 / PED q24 on two captured windows: 0 difference vs model

Author kill-gates this round inherits: abs AEE ≤ 1.259; relative vs strong ordinary control ≤ +0.005 (so AEE ≤ 1.224801338); complete-chain same-port / same-state / same-backpressure net service ≥ 15%. Lifting family may stay; **layout is what is allowed to die**. Title may leave lifting if A+X is stronger.

Relative AEE already kills unconstrained 40-coeff lifting as a publishable student unless a later idea recovers ≤ +0.005. Long-backpressure 8088=8088 already kills add-count as the circuit figure of merit.

---

## P03-I1 — Write-port-optimal MCM, not adder-count CSE

**One sentence.** Recompile the T10 constant matrix with the cost function equal to committed writes on the measured two-slot file, copying time-multiplexed MCM rather than add-optimal CSE.

**A (complete prior to copy).** Copy the whole MCM stack, not a slogan: Voronenko & Püschel, *Multiplierless Multiple Constant Multiplication* (ACM TALG 2007, Hcub) plus SPIRAL’s multiplier-block generator; Tummeltshammer, Hoe & Püschel, *Multiple constant multiplication by time-multiplexed mapping of addition chains* (DAC 2004) for a **shared adder bound to a small number of issue slots**; HLS resource-constrained / modulo scheduling (Paulin force-directed; Rau iterative modulo). Official CSE already in the freeze is the add-optimal sibling of this stack.

**B (hole in this net).** After official CSE the lifting graph is 159 add/sub + 35 RNE/sat writes. Always-ready drops 6938 → 5354, but long backpressure stays **8088 = 8088**. The machine is two-stage SIMD with **two writeback slots**. Add-count is therefore the wrong IR cost: extra live intermediates from CSE compete for the same two slots that must also retire outputs. Dual consumers (gate and continuous PED) further duplicate sinks. Part of the always-ready −22.83% is generic round→sat fusion, so the remaining CSE story is already thinner than 22%.

**X (not a reskin).** Do not publish “we ran CSE.” Re-objective the rewrite to **minimize writes bound to two slots under a live-range cap**, even if add/sub rises back toward 260. That can look like a *worse* DAG and a *better* long-backpressure schedule. The circuit claim is a 2-port time-multiplexed CMVM for a dual-consumer temporal mix, not a smaller adder tree.

**Strongest controls.** (i) Ordinary dense + official CSE + **the same** round→sat fusion. (ii) Lifting + add-optimal CSE (159+35) on the same two slots. (iii) Always-ready and long-backpressure reported as separate columns; never substitute 22% always-ready for chain service. (iv) Consumer 758777 table not added to 8088. (v) ATLIF stays continuous θg.

**Kill-gate.** On the same two-slot SIMD, lifting-or-rewrite long-backpressure slots ≤ 6874 (8088×0.85). valid825 AEE ≤ 1.224801338 and ≤ 1.259. If only always-ready moves and long-BP stays 8088, kill I1. If ordinary+fusion already eats most of −22.83%, residual must still meet 15% or the title cannot be “CSE.”

**Two-sentence TCAS-II pitch.** Reviewers will treat adder-count CSE as FIR-MCM déjà vu; the frozen schedule already shows that win was absorbed by two write slots. Copy Hcub + time-multiplexed MCM completely, then make the figure of merit write-slot occupancy on a mixed spike/continuous T10 source.

**Biggest objection.** 8088 may already be the output-port lower bound (retire N outputs through 2 ports plus pipeline fill). Then no CSE variant can move long-BP, and I1 is dead by construction.

**Assumptions.** Intermediate writes, including the 35 RNE/sat nodes, sit on the long-backpressure critical path rather than being fully overlapped with output retirement.

**Predictions.** Add-optimal CSE raises peak live values near the 35 RNE nodes; port-optimal MCM cuts concurrent lives toward the slot count and is the first rewrite that can move 8088. Always-ready may worsen while long-BP improves.

**Disconfirmers.** Occupancy trace shows 8088 = ceil(outputs/2)×fill independent of intermediate count. Port-optimal add/sub climbs with no slot movement. AEE of the rewritten integer DAG exceeds +0.005 vs ordinary.

---

## P03-I2 — Integer-to-integer lifting: delete the 35 intermediate RNE/sat

**One sentence.** Copy integer-to-integer lifting so rounding exists only at the two consumer type-exits, erasing the 35 intermediate RNE/sat that official CSE left behind.

**A (complete prior to copy).** Calderbank, Daubechies, Sweldens & Yeo, *Wavelet Transforms That Map Integers to Integers* (ACHA 1998) and the ICIP 1997 integer-lifting recipe; JPEG2000 reversible 5/3 (ISO/IEC 15444) as the deployed I2I instance; Sweldens lifting factorization (*Factoring wavelet transforms into lifting steps*). Faithful / truncated MCM (Aksoy et al.) as the sibling for error-bounded intermediate drop. hls4ml/da4ml explicit quant-node graphs as the compiler form: delayed quantization, not per-add Q.

**B (hole in this net).** Lifting after official CSE is not “159 adds.” It is 159 add/sub **+ 35 intermediate RNE/sat**. Those 35 are Q-format barriers on learned real 40-coeff steps, not algebraic commons. They (1) are extra writes on the two-slot file, (2) break further CSE across nearly-equal subexpressions, (3) sit next to an AEE that already fails relative (+0.013178). Ordinary 260-add source has a different rounding texture; comparing add/sub without rounding nodes is an apples-to-oranges IR.

**X (not a reskin).** Not post-hoc output quant, not “fewer bits.” Retrain or reparameterize T10 into an **I2I predict/update graph with zero intermediate RNE**; RNE/sat only at gate-integer and PED-q24 sinks. Compiler deletes 35 write nodes. Numerical contract is either exact integer invertibility (Calderbank) or a residual to the FP teacher that still passes +0.005.

**Strongest controls.** Current 40-coeff lifting with 35 RNE. Delayed rounding **without** retrain. Frozen JPEG2000 5/3 and 9/7 as unmodified complete priors. Ordinary dense AEE 1.219801338. Same two-slot schedule. Window 0-diff is a measurement, not a requirement once AEE is legal.

**Kill-gate.** Intermediate RNE/sat in the source DAG ≤ 2 (one per consumer type), ideally 0 inside the mix. AEE ≤ 1.224801338. Long-BP ≤ 6874 vs 8088 on the same ports. If delay-rounding without retrain misses +0.005, QAT is mandatory; if QAT cannot land AEE, kill the I2I island (and lifting-as-title with it).

**Two-sentence TCAS-II pitch.** The frozen CSE output still carries 35 rounding barriers; that is the unadvertised cost and a plausible contributor to the +0.013 AEE miss. Integer-to-integer lifting is a 1998 complete prior; the increment is compiling it onto dual-type ATLIF exits so RNE is a sink, not a per-step tax.

**Biggest objection.** A 40-coeff learned temporal mix for optical flow may not be representable as dyadic I2I without collapsing AEE, reducing the idea to “use 5/3 DWT,” which will fail valid825.

**Assumptions.** A non-trivial fraction of +0.013178 is Q-format noise from the 35 barriers, so I2I+retrain can enter [1.2198, 1.2248]. The 35 nodes are real writes, not bookkeeping flags.

**Predictions.** Post-hoc delay-rounding fails AEE; I2I QAT either recovers relative AEE and drops long-BP, or fails AEE and kills lifting. Add/sub may stay ~159 or rise; slots still drop if writes fall.

**Disconfirmers.** I2I codebook sweep all AEE > 1.2248. Occupancy shows the 35 RNE nodes are not on the write-back critical path. 5/3 or 9/7 as drop-in T10 miss abs 1.259.

---

## P03-I3 — Typed dual-exit MCM (gate integer vs PED q24)

**One sentence.** Compile one T10 CMVM with two typed sinks so the 35 RNE/sat collapse to the spike/gate versus continuous-PED frontier instead of repeating after every lifting step.

**A (complete prior to copy).** Multi-output MCM / MIMO constant multiplication (Aksoy–Costa–Flores–Monteiro line; SPIRAL multi-output transforms); polyphase/lifting filter banks with predict and update as two exits; mixed-precision HLS with per-sink Q-format; compiler qnn graphs with distinct output quantizers (TVM/XLA canonical forms). Copy a **complete multi-output MCM + per-sink quant** flow, not “mixed precision.”

**B (hole in this net).** Dual consumers after the source are explicit: spike/gate path and continuous residual/PED path. Official CSE is counted **per T10 vector**, as if one type. Captured windows show integer gates / I24 / PED q24 at **0 difference vs model**, i.e. the current compile pays exactness for both types. The 35 intermediate RNE/sat are the likely place where one vector is recast over and over so both sinks stay 0-diff. Consumer −5.78% is a different resource point and cannot be added, but it shows the consumer side still barely moves — consistent with a source that already fully materializes a high-precision vector.

**X (not a reskin).** Not two T10 copies, not “quant-aware CSE.” One DAG, **two typed exits**: gate sink may legally trunc/sat (ATLIF θg is continuous, the gate is integer); PED sink keeps RNE q24. Shared prefix is integer-exact; rounding only at the type-change frontier. The ATLIF dual-consumer graph is the reason typed sinks exist; FIR MCM did not have a spike/continuous pair.

**Strongest controls.** Single-type CSE then two casts (current). Duplicated DAG per consumer (cost upper bound). Precision ablation: drop gate-path extra bits, keep PED q24. Same two slots. AEE, not window 0-diff, is the accuracy gate.

**Kill-gate.** Source long-BP ≤ 6874 vs 8088. Full-chain re-measured at **one** resource point (do not add 8088 and 758777) net service ≥ 15%. AEE ≤ 1.224801338 even if window 0-diff breaks. If typed exits only relocate the same 35 writes to the sinks with unchanged port occupancy, kill I3.

**Two-sentence TCAS-II pitch.** A mixed spike/continuous transformer is two number types, not two neuron circuits. The mechanism is a typed-sink CMVM whose rounding frontier is the gate versus PED split that this net actually has.

**Biggest objection.** Both consumers may need the same T10 vector at full q24; then typed sinks buy zero slots and only a prettier diagram.

**Assumptions.** Gate-path Lipschitz in continuous θg allows coarser rounding than PED q24 without +0.005 AEE. A material subset of the 35 RNE exists to keep the unused extra precision.

**Predictions.** Intermediate RNE count falls from 35 to 2. Window 0-diff on gates may break while AEE holds. Long-BP moves only if the deleted RNE were writes; consumer table must be rebuilt at the source’s resource point.

**Disconfirmers.** Gate-path precision ablation changes neither AEE nor slot count. PED and gate bit-exact requirements coincide on the full T10 vector. Full-chain service < 15% after re-bind.

---

## P03-I4 — Distributed-arithmetic IR bound to the two writeback accumulators

**One sentence.** Copy Peled–Liu / da4ml CMVM compilation, but bind DA accumulators to the already-measured two write slots so the IR matches port geometry instead of a parallel CSE DAG.

**A (complete prior to copy).** Peled & Liu, *A new hardware realization of digital filters* (IEEE TASSP 1974); White’s DA survey (IEEE ASSP Mag 1989); **da4ml** (Sun, Que, Loncar, Luk, Spiropulu, ACM TRETS 2025 / arXiv:2507.04535) and its hls4ml DA strategy: CMVM → adder graph with CSE, bit-exact, fully unrolled or bit-plane, LUT/adder on FPGA, no DSP. Copy the **entire** da4ml/hls4ml DA path (Q-format in, generate DA graph, emit HLS/RTL), including their CSE, then change binding. LUT-ROM of bit-sliced partial products is optional, not the identity (ASIC two-accumulator DA is enough).

**B (hole in this net).** Two-slot writeback is geometrically a **pair of accumulators**. The parallel CSE DAG (159 adds, 35 live RNE) fights those ports; long-BP tie at 8088 is what a wide DAG on two drain pipes looks like. da4ml’s default fully-unrolled CMVM is the wrong occupancy for this machine (it assumes reuse_factor 1 and combinational/pipelined unroll). Serial or 2-bit-at-a-time DA writes one or two accumulators per cycle — isomorphic to the SIMD that already exists.

**X (not a reskin).** Not “FPGA LUTs because LUTRAM exists” and not “we used da4ml.” The increment is **port-isomorphic DA**: bit-planes scheduled so each cycle uses both write slots, with RNE only at LSB flush into the two typed consumers. ATLIF continuous θg (not binary) is the input alphabet; dual-consumer flush is the extra quant node da4ml’s jet-tagging CMVM does not have. Title is a 2-accumulator constant-matrix engine, not a LUTNet.

**Strongest controls.** Parallel official CSE on the same two slots (current). da4ml default unroll as complete A (area/latency, not our FOM). Serial DA vs 2-bit vs 4-bit Pareto. Ordinary-dense DA vs lifting DA (AEE). Same round→sat fusion applied to ordinary. No Yosys/OpenROAD as foundry PPA; no LUT-count as novelty.

**Kill-gate.** Long-BP cycles ≤ 6874 vs ordinary **on the same ports and state**. AEE ≤ 1.224801338. If DA latency ≈ bitwidth × vector and that exceeds 8088, DA is slower — kill I4. Always-ready may worsen under serial issue; the chain metric is realistic ready/backpressure, not the 22% always-ready column.

**Two-sentence TCAS-II pitch.** Copy da4ml/Peled–Liu completely, then stop unrolling a CSE DAG onto a 2-port SIMD that already absorbed 22%. The circuit is two DA accumulators whose flush types are gate versus PED, which is this net’s bottleneck rather than FPGA LUT count.

**Biggest objection.** DA serializes by input bits; for this T10 width/bitwidth the lower bound may exceed 8088, making DA a regression.

**Assumptions.** Two-bit-at-a-time (or two-output) DA can keep both slots busy and avoid the 35 intermediate writes. DA flush rounding can be trained or shown equivalent to per-tap RNE within +0.005 AEE.

**Predictions.** Intermediate RNE → one flush per output. Long-BP drops only for 2-bit/two-output schedules; 1-bit serial misses 15%. Ordinary-dense DA vs lifting DA: if dense DA already meets 15%, lifting is unnecessary in the title.

**Disconfirmers.** Best DA schedule ≥ 8088 long-BP. DA flush AEE > 1.2248 vs per-tap RNE model. LUT/adder area quoted as the result without a same-port cycle win.

---

## P03-I5 — Dyadic/CSD lifting student (QAT), or lifting cannot be the title

**One sentence.** Retrain the 40-coeff T10 as a dyadic/CSD codebook lifting so the compiler emits shift-add with sink-only rounding, simultaneously attacking the +0.013 AEE miss and the 35 RNE writes.

**A (complete prior to copy).** CSD recoding (Reitwiesner) + MCM; JPEG2000 dyadic predict/update (`x += 2^{-k} y`); AdderNet (Chen et al., CVPR 2020), DeepShift, ShiftAddNet, power-of-two / INQ-style codebook quantization; integer DWT coefficient design. Copy a **complete QAT + multiplierless compile** loop (discrete coeffs in the graph, STE or codebook, then Hcub/da4ml), not PTQ of the frozen 40 reals.

**B (hole in this net).** Unconstrained 40-coeff lifting **already fails the author’s relative AEE gate** (1.232979368 vs 1.219801338, Δ +0.013178 > 0.005) while still inserting 35 intermediate RNE. The accuracy failure and the rounding-node tax are the same root: real coeffs expanded to shift-add with per-node Q. A TCAS-II titled “lifting CSE” with a student that loses to ordinary dense on AEE is a reject even if slots later move.

**X (not a reskin).** Not PTQ of the current 40 coeffs (that should worsen 1.233). A constrained student at the same Motion C12 / H67 / ep34 identity: discrete dyadic/CSD lifting, compiler output with **no intermediate RNE**, dual-consumer sinks only. If no codebook recovers ≤ +0.005, **SCOPE’s island-change clause fires** — lifting is not the title; I6/I7 become the main island.

**Strongest controls.** Ordinary dense (1.219801338). Unconstrained 40-coeff lifting (1.232979368). PTQ-CSD of those 40 coeffs (must be shown dead or alive). Frozen 5/3 and 9/7. Same CSE/ports after the new coeffs exist. No binary ATLIF.

**Kill-gate.** AEE ≤ 1.224801338 and ≤ 1.259. Intermediate RNE ≤ 2. Full-chain same-resource net service ≥ 15%. PTQ-only: if PTQ AEE > 1.2248, PTQ is dead. If all QAT codebooks > 1.2248, **kill lifting-as-title**, do not weaken the AEE gate.

**Two-sentence TCAS-II pitch.** You cannot headline lifting CSE while the student is +0.013 AEE versus dense. The mechanism is a dyadic lifting student whose shift-add graph has sink-only rounding, restoring both the relative AEE contract and the two-port schedule.

**Biggest objection.** Optical-flow T10 may need unconstrained real taps; the codebook’s AEE floor may sit above 1.233, not below 1.225.

**Assumptions.** +0.013178 is not an information-theoretic floor of sparse temporal mixing. QAT can trade a little ordinary-dense margin (1.2198 has 0.005 room) for multiplierless structure.

**Predictions.** PTQ-CSD fails AEE. QAT dyadic either lands in [1.220, 1.225) and drops RNE, or fails and forces an island change. 5/3/9/7 miss AEE and exist only as A.

**Disconfirmers.** Sweep of `{2^{-k}}`, CSD Hamming ≤ 2, 5/3-like predict/update, all AEE > 1.2248. QAT passes AEE but RNE count stays ~35 (compiler did not exploit dyadic exactness). Chain service < 15% despite legal AEE.

---

## P03-I6 — Full-domain BN as the same-class 2-port reduction (fold or Welford)

**One sentence.** Cost the native projection BN’s real full-domain 10×96×120×160 batch statistics on the same two write slots, then either fold running stats into the projection CMVM or compile Welford onto that engine.

**A (complete prior to copy).** Ioffe & Szegedy BN and the **standard inference fold** of γ(x−μ)/σ+β into conv constants (TensorRT/cuDNN/TF-TRT); Welford online variance (1962) and Chan/Pébay pairwise parallel moments; training-accelerator fused reduction+broadcast; SPIRAL-style reduction-tree compilation. Copy **one** complete path: either the production BN-fold+CMVM compiler, or a bit-specified parallel Welford. GhostBN/SyncBN/instance-norm are related priors, not drop-ins (different domains).

**B (hole in this net).** Native projection BN uses **actual batch statistics over 10×96×120×160** (= 18,432,000 elements), not frozen running stats. Local-window replay with free mean/var **undercharges wait/storage**. T10 is already noncausal over 10. Two-slot writeback applies verbatim to moment accumulators (sum, sumsq). A source-only 8088 story that omits this barrier cannot claim complete-chain ≥ 15%. SCOPE already flags this cost as unknown.

**X (not a reskin).** Not “we also accelerate BN.” Two mutually exclusive forks, one paper identity:

- **FOLD** (if AEE allows): measure running-stat vs full-batch-stat AEE. If relative ≤ +0.005 and abs ≤ 1.259, copy BN-fold completely and compile projection as one constant matrix (da4ml/Hcub). X is that this student was captured with batch stats that inference hardware must not pay — a legal-inference compile, not a new neuron.
- **REDUCE** (if fold fails AEE): compile full-domain Welford onto the **same** two-slot SIMD as T10, overlapping noncausal temporal mix with spatial/channel reduction. X is **one full-domain linear engine** (moments + mix), not two islands. Free-stats replay is banned as a reported control.

**Strongest controls.** Frozen running stats, folded (AEE + slots). Full-batch exact stats (current capture). Local-window replay with free μ/σ (illegal FOM; must show undercharge in cycles/storage). Two-pass vs Welford, bit-exact vs AEE. Ordinary dense T10 + honest BN as the chain baseline. Do not multiply component speedups into FPS.

**Kill-gate.** BN reduction slots **must be measured** on the same 2-slot model; without that number, chain ≥ 15% is unpublished. FOLD: AEE vs ordinary ≤ +0.005 after fold. REDUCE: full chain (T10 + proj + BN + residual add + dual consumers) same-resource net service ≥ 15% vs ordinary+honest BN. If honest BN dominates and neither fork beats 15%, kill source-only papers (including I1–I5 as titles).

**Two-sentence TCAS-II pitch.** The unmeasured first-order cost is full-domain BN, not T10 add count. Either fold it into a constant matrix (complete inference prior) or compile Welford onto the same two write slots that already absorbed CSE; the causal graph must include the stats barrier.

**Biggest objection.** Batch stats at this capture may be an artifact; then FOLD is a bugfix, too thin for TCAS-II unless the folded projection CMVM still delivers ≥ 15% chain service.

**Assumptions.** Free μ/σ local replay is first-order undercharge, as frozen. T10 noncausal wait can overlap REDUCE if compiled as one engine; as two islands it cannot.

**Predictions.** Honest BN slots ≫ 8088 unless folded. FOLD AEE either passes (title becomes folded-proj CMVM; lifting optional) or fails (BN engine is the title). REDUCE without overlap misses 15%.

**Disconfirmers.** Running-stat AEE equals batch-stat AEE **and** reduction hides inside existing T10 wait — BN is a footnote, not a paper. FOLD passes AEE but folded CMVM does not beat ordinary+fold by 15%. REDUCE bit-exact Welford on 2 ports exceeds any 15% budget.

---

## P03-I7 — Fusion-complete mixed-Q IR; residual lifting is the only publishable X

**One sentence.** Copy quant-node canonicalization (round→sat fusion) onto **both** ordinary and lifting graphs, then publish the residual table; lifting stays in the title only if residual long-BP is still ≥ 15%.

**A (complete prior to copy).** IEEE-754 fused round-and-sat / integer sat packs; HLS Q-format canonicalization (Vitis/Catapult); TVM/Glow/XLA `qnn` canonicalizations (adjacent quant ops merge); hls4ml/da4ml quant-node merging and bit-exact DA (accumulator precision ignored under DA because the graph is already canonical). Copy a **complete quant-IR pass**, including running it on the dense control, not only on lifting.

**B (hole in this net).** Freeze text: **part of −22.83% is generic round→sat fusion**, and long-BP is 8088=8088. Headline “lifting CSE −22.83%” is not a measured advantage of lifting. Ordinary 260-add source also has round/sat. The 35 intermediate RNE/sat are extra fusion opportunities that **ordinary does not have**, so fusion and lifting are confounded. Consumer −5.78% is another resource point and must not be stacked.

**X (not a reskin).** The mechanism is **quantization-node fusion as a first-class pass on mixed spike/continuous graphs**. Lifting is a case, not the identity. Increment vs TVM qnn: typed fusion (gate sat vs PED RNE) on ATLIF dual consumers, and a mandatory residual table:

`(ordinary)`  
`(ordinary + fusion)`  
`(lifting + fusion, no CSE)`  
`(lifting + CSE + fusion)`  

all on the **same** two-slot, same always-ready and long-BP columns. If residual < 15%, retitle to fusion-complete mixed-Q IR or kill hardware.

**Strongest controls.** The four-row table above. Fusion applied to ordinary **first**. Always-ready vs long-BP split. Full-chain one resource point. AEE unchanged by fusion if the integer model is 0-diff on windows — if fusion changes values, AEE re-gate.

**Kill-gate.** Residual lifting-only long-BP vs **ordinary+fusion** ≥ 15% to keep lifting in the title (≤ 6874 vs the fused-ordinary slot count, not vs unfused 8088). If residual < 15% but fusion itself ≥ 15% on ordinary, **retitle to I7**. If neither long-BP column meets 15%, kill the hardware claim. AEE ≤ 1.224801338 still applies to any surviving lifting student.

**Two-sentence TCAS-II pitch.** Reviewers will subtract generic sat fusion from a lifting headline; do the subtraction in the paper. Copy qnn canonicalization, then show whether any lifting-specific service remains on the two-slot machine that tied at 8088.

**Biggest objection.** Fusion-complete IR is “an HLS pass,” too thin unless typed spike/continuous fusion is the circuit, and unless the 15% lands on long-BP not always-ready.

**Assumptions.** “Part of −22.83%” is large enough that residual may miss 15%. Long-BP, not always-ready, is what TCAS-II must survive.

**Predictions.** Ordinary+fusion captures a large share of always-ready −22.83%; long-BP stays tied; lifting residual fails 15%; title must leave lifting or die. If fusion does **not** move ordinary always-ready, fusion was lifting-specific (the 35 nodes) and belongs inside I2, not as a generic pass.

**Disconfirmers.** Ordinary+fusion does not move always-ready or long-BP, while lifting+fusion does — then fusion is not generic and I7’s “subtract from both” is a null control, not a title. Fusion changes integer values and breaks window 0-diff **and** AEE.

---

## P03-I8 — Flipping-structure / cascade cell: keep lifting family, stop CSE-netlist layout

**One sentence.** Copy Huang’s flipping-structure DWT cell so the two write slots are the predict/update memories by construction, replacing the wide CSE DAG that absorbed its own add-count win.

**A (complete prior to copy).** Huang, Tseng & Chen, *Flipping structure: an efficient VLSI architecture for lifting-based discrete wavelet transform* (IEEE TSP 2004); follow-on PLS / line-based 2-D DWT (e.g. TSP 2006 mapping of 1-D registers to temporal buffers); Wu/Lin pipeline lifting; JPEG2000 5/3 and 9/7 lifting datapaths (the TCAS-I/II DWT decade). Copy the **complete cell**: flipped predict/update, forwarding, precision analysis as in Huang §precision, not a cartoon ladder.

**B (hole in this net).** Learned 40-coeff T10 was compiled as a **wide CSE netlist** (159 adds, 35 RNE lives) onto a 2-slot SIMD. Classic DWT hardware does the opposite: a **narrow cascade** of predict/update cells whose write arity is two. SCOPE: lifting family may stay, **layout is what stops**. Long-BP 8088=8088 is the layout failure mode (wide DAG, two drain pipes). Flipping also exists to cut multiplier-on-path accumulation — the same accumulation that creates intermediate sat.

**X (not a reskin).** Not “we implemented DWT.” Map **learned T10 PSN + dual consumers (gate and PED)** onto a flipping cell whose two memories **are** the measured two writeback slots. Same 40 coeffs first (layout-only ablation), then QAT inside the cell if cell rounding breaks AEE. Increment vs 2004 DWT: noncausal T10 optical-flow mix, continuous-θg ATLIF dual exits, and a same-port service number rather than critical-path in isolation.

**Strongest controls.** Wide CSE DAG on two-slot SIMD (current). Same 40 coeffs, only datapath/schedule change (isolates layout vs family). Unmodified 5/3 and 9/7 flipping cells as complete A (AEE kill). Dual-consumer exits from the cell. Ordinary dense. Fusion applied equally (I7 control nested).

**Kill-gate.** Layout-only (same coeffs): long-BP ≤ 6874 vs 8088; integer AEE vs current lifting within the +0.005 envelope of ordinary (current lifting is already +0.013 vs ordinary — layout-only **does not fix AEE**). Therefore I8 **cannot be the title alone** unless paired with I5-style retrain in the cell loop that reaches AEE ≤ 1.224801338. If flipping with same 40 coeffs does not move long-BP, bottleneck is output-port bound → kill layout-X.

**Two-sentence TCAS-II pitch.** Flipping-structure DWT is the venue’s own complete prior; copy the cell and stop the CSE-netlist layout that left 35 live RNE on two write slots. Keep lifting as family, bind the cell’s two memories to the measured two-slot writeback, and only then retrain coeffs if AEE still misses +0.005.

**Biggest objection.** Reviewers will call it a 2004 DWT reskin unless dual-consumer ATLIF + recovered relative AEE + 15% same-port service are all real. Layout-only leaves AEE at 1.233 and should not be submitted.

**Assumptions.** SCOPE “停的是布局” is correct: family is viable, wide CSE is the failure. Cell rounding can be matched to the current integer model or corrected by in-cell QAT.

**Predictions.** Same-coeff flipping: service may move, AEE stays ~1.233 and still fails relative — I8 without I5 is unsubmittable. In-cell QAT either recovers AEE or kills the lifting family. If 8088 is output-bound, flipping and CSE tie, and the main island must leave T10 (I6).

**Disconfirmers.** Flipping long-BP stays 8088. Cell rounding diverges from the CSE DAG by > 0.005 AEE and in-cell QAT cannot recover. 5/3 flipping as drop-in already fails abs 1.259, and learned-cell QAT does too.

---

## Cross-idea notes (still ideas, not a ranking)

- **AEE vs ports are different kill axes.** I1/I4/I8 can win slots and still be unsubmittable at AEE 1.233. I5/I2 exist to repair AEE. I7 exists to stop a false 22% headline. I6 exists because chain service may not live in T10 at all.
- **Do not add 8088 and 758777.** Any survivor must re-measure one complete target chain at one resource point, with honest BN stats.
- **Generic fusion is a covariate.** I7 is the control that every other lifting title must nest.
- **If 8088 is output-port lower bound,** I1/I2/I3/I4/I8 all die together; remaining islands are I6 (BN/proj CMVM) and possibly r1 residual convs as frozen CMVM — still compiler/constant-matrix, but not T10 lifting.
- **Binary ATLIF, analog CIM, Yosys-as-PPA, component×FPS** stay out.
