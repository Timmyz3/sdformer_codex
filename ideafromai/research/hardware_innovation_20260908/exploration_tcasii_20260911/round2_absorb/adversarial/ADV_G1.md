# Adversarial review — G1 Typed last-use at the absorb cut

**Reviewer did not originate G1.** Fusion of Q01-I2/I5/I7, Q02-I2/I6/I7, Q03-I3/I7, Q04-I1/I3/I4, Q05-I4 after R2L1–L5. Steelman first, then attack. Frozen facts from `round2_absorb/PROBLEM.md` + `IDENTITY_ATLIF.md` only. Priors from R2L1, R2L2, R2L5 (and R2L3/R2L4 where they name the same hole). No papers invented. Ideas and scores are proposals, not findings. No RTL, no new AEE, no closed service number in this sheet.

Round-1 sibling: `adversarial/ADV_F1.md` (continuous-θg dual-completion). Absorb **does not** salvage F1. It makes the default reviewer reading *worse*: path A is now legally Gustav/Prosperity/FireFly, so a last-use letter that does not copy those engines whole and then change the **retire contract** is a reskin of A.

---

Idea ID: G1 (main island in `fusion/FUSION_R2.md`)

| Slot | Card as clustered (do not steelman yet) |
|---|---|
| **A** | Prosperity one-consumer Y; Gustav single-consumer NRV retire; FireFly Algo-1 silent channels; SDT MS. |
| **B** | 8088 long-BP unchanged after mix CSE; residual/PED/I24 is a second tensor; integer 0-diff windows exist. |
| **X** | occupancy = last-use of **binary gate after absorb** AND **continuous residual/PED**, not analog AT-LIF amplitude shared by two MACs. |
| **Kill** | 8088 unmoved; chain <15%; AEE gates. |

Hypothesis (`hypotheses/H_G1_typed_last_use.json`): 8088 is dominated by the continuous residual remaining live after the binary spike consumer would have retired; binary-only NRV/product-sparsity cannot move 8088.

---

## Strongest version

One mechanism, not the thirteen-ID stack. After \(W\leftarrow\theta W\), copy **one** complete binary-GeMM prior onto the matching layers (`r1 conv2`, `proj.conv`, later SN→conv/linear): Prosperity EM/PM forest **or** Gustav CPTB+NRV **or** FireFly-S Bitmap AND, checklist-complete (R2L1 §5 / R2L2 §3.3 / R2L3 §1.4). That copy is **A**, reported as a control, never the title.

The letter object is a **typed dual-completion** of a producer the freeze actually has:

- Consumer A = post-absorb **binary gate** (Prosperity/Gustav/FireFly may skip zeros here).
- Consumer B = **continuous residual / PED / I24** (dense 0-diff q24; not AT-LIF \(o\) carrying amplitude; not a second binary GeMM).
- BN-stat over `10×96×120×160` is a **third conjunct if live**, charged as G3 hygiene, not a second island.

A SRAM / source word stays occupied until \(\mathrm{last\text{-}use}=\max(t_A,t_B[,t_{\mathrm{BN}}])\). Fetch/skip occupancy, if any, is \(\mathrm{support}(s_{\mathrm{after\,absorb}})\cup\mathrm{support}(\mathrm{PED})\), **not** Gustav’s P-bit `{0,1}` NRV at one tick, **not** Prosperity’s one-Y prefix table, **not** FireFly Algo-1 “channel never spikes,” **not** SDT’s single membrane stream \(U\). Relative prior is ordinary dense-source T10 (AEE 1.219801338) at the **same** two-stage SIMD ports, state class, and backpressure model, **with the same dual last-use protocol applied to ordinary**. Lifting is a sensitivity (already +0.013178 relative fail). Always-ready 6938→5354 is not service. Integer gates / I24 / PED q24 stay 0-diff on the two captured windows.

That is the only steelman that still looks like one TCAS-II island: R2L2 X1 (dual-retire across the absorb cut) plus R2L2 X2 (union occupancy, two algebras) plus R2L5’s leftover sentence, with Q04-I1’s \(\max(t_A,t_B)\) as the 8088 readout. It lives or dies on moving long-backpressure off 8088 and on complete-chain net service ≥15%.

Do **not** steelman G1 as “we also NRV,” as “two-color scoreboard ISA,” as “membrane shortcut so spikes stay binary,” or as “analog AT-LIF shared by two MACs.” Those are the attacks below. The steelman forbids them.

---

## Observation that would count against it

A cycle-accurate occupancy trace on the **same** two-stage SIMD resource that produced 8088, split into

\[
\{\mathrm{wait\_source\_ALU},\ \mathrm{wait\_FIFO\_full},\ \mathrm{wait\_gate},\ \mathrm{wait\_PED},\ \mathrm{wait\_BNstat},\ \mathrm{wait\_1RW\_port},\ \mathrm{wait\_T10\_fill},\ \mathrm{wait\_I24\_reread}\}
\]

showing that typed dual-completion is **not** the dominant term. Sufficient kills from that trace (any one deletes the steelman without RTL):

1. Gate, PED, and BN-stat become ready on the same cycle for ≥95% of live windows (AND-retire has nothing to fire; lazy fork already optimal). Q05-I4 already named this: histogram of \((\mathrm{residual\_last}-\mathrm{gemm\_last})\) is a delta at 0.
2. Union occupancy density \(\approx 1\) (continuous PED almost always live → NRV-style skip ≈ 0). Then G1 collapses to “wait for both readys,” which is a scoreboard, and skip language must be dropped.
3. 8088 is delay-line fill, 2-slot output-port bound, or FIFO-full, **not** last-use of gate vs PED. Then dual last-use cannot legally drop a committed source write.
4. After a bit-true last-use tag (no skip), 8088 is unchanged. Phantom-hold was not the occupant.
5. Binary-only Prosperity/NRV/FireFly **on this student** already moves 8088 ≥15% at 0-diff. Then B was never holding the port (Q04-I3 kill); G1 is empty; the letter is a complete A, which is a dead letter (R2L1 §8).

Until that histogram exists, 15% is a copied kill-gate, not a prediction.

---

## Attack 1 — Reskin of scoreboard / Gustav NRV / SDT?

**Yes. That is the default TCAS-II reading, and it is fair.** Absorb made it the *first* paragraph, not a footnote. Round-1 F1 could still say “NRV is the wrong skip unit because θg is continuous.” That sentence is **retired**. Path A is binary. Reviewers subtract A, then ask what remains besides a 2-bit valid/ready.

### 1.1 Scoreboard / credit / last-use bookkeeping (the honest relative prior)

Q02-I2’s own title is “two-color last-use **scoreboard**.” Q01-I2 is a two-color retirement FSM. Q01-I5 is residual-first credits. Q02-I6 is drain-first under backpressure. Q02-I7 is S-last-use as R-release. Q03-I3/I7 are typed live-ranges and dual issue queues. Q04-I4 is “stop co-allocating.” Those are Tomasulo/scoreboard last-use bits, Carloni latency-insensitive credits, Cortadella SELF eager vs lazy fork, Dally–Towles reverse credits, Buffet occupancy = fills not yet last-used, AXI-Stream valid/ready, architected last-use, LoopTree retain/recompute of a live tensor. P05-I1’s round-1 objection still applies: “you added valid/ready.”

After absorb, the *type* split (binary token vs q24 residual) is forced by `IDENTITY_ATLIF.md`. Forced identity ≠ a circuit. A compiler that keeps one last-use tag on two dtypes is a miscompile; fixing a miscompile is not a five-page datapath. Q01-I2’s own objection is the review: “Dual last-use is reference counting / bookkeeping, too thin for TCAS-II even at 5 pages.” Q02-I7’s objection: “Free-on-last-read is basic SRAM.”

If the measured 15% appears only vs ordinary+joint-ready **without** the same 2-credit tap on ordinary, or only vs a serialized 1RW strawman, it is accounting. Relative prior for a handshake letter is ordinary + the **same** protocol. Then lifting, NRV, and FireFly leave the title.

### 1.2 Gustav NRV / Prosperity one-Y / FireFly silent-channel (incomplete A + different predicate)

G1’s A line is four adjectives, not one copied engine.

| Card’s “A” | What the paper actually requires | What G1 keeps |
|---|---|---|
| Prosperity one-consumer Y | Binary rows, one shared \(W\), EM/PM forest, one prefix, TCAM detector, NO-order, remainder XOR, lossless Y, **one** consumer (R2L1 §1, checklist §5) | The *fact* that Y has one consumer. That is the hole statement, not a copy. |
| Gustav single-consumer NRV retire | CPTB \(P\)-psum in-situ, NRV = row index + \(P\)-bit occupancy **of binary \(s\) at one tick**, NR4 merger, same-ID barrier, column-major time-second, commit = `LIF` after this-tick sweep (R2L2 §2–§3) | Occupancy as a packed support vector **is** NRV. Changing `{0,1}` to \(\mathrm{support}(s)\cup\mathrm{support}(\mathrm{PED})\) is R2L2 **X2**, which R2L2 already said dies if the union is dense. |
| FireFly Algo-1 silent channels | Eval-time drop of channels whose **bias-only causal V** never crosses \(V_{\mathrm{th}}\) (R2L3 §1.3–1.4) | **Illegal transfer** on this net: a never-spike channel can still be required as continuous PED/I24. Algo-1 as A is a kill, not a control. Bitmap AND of spike×W-mask **is** A on path A; Algo-1 is not. |
| SDT MS | Add on continuous \(U\) **before** SN so spike tensors stay `{0,1}` (R2L5 §2.3). Already the student’s `MS_ResBlock`. After absorb, binary communication is the **neuron identity**, not a residual invention. | Placement is A. Last-use of two dtypes on one SRAM is not MS. |

R2L2’s honest leftover **is** G1: X1 dual-retire, X2 union occupancy. Incomplete Gustav/Prosperity + the transfer note’s X is a reskin of NRV with a different predicate, not a measured letter. Prosperity intersection-of-supports is **explicitly unused** (R2L1 §1.1); union occupancy is not EM/PM and must not inherit product-sparsity language. FireFly dual-side is Bitmap AND of two **binary** masks, not “binary gate + continuous PED of the same source” (R2L1 §3.2).

A reviewer can write one paragraph:

> Credits from 2001–2007, occupancy vector from HPCA 2026 NRV, one-Y retire from Prosperity, silent-channel skip from FireFly-S Algo 1, membrane add from SDT 2023. None of them is dual-lifetime of a q24 residual. You did not copy any of them whole. After absorb the spike path **is** their object, so the missing relative prior is fatal.

That is a binary reject on novelty + relative prior, before the 15% is checked.

### 1.3 SDT / SDformerFlow SPE (the graph is already the student)

R2L5 §5.1 / §6 objection, quoted because it is the review:

- “We use membrane shortcut so spikes stay binary” = SDT §3.2 + SDformerFlow Fig. 4 + local `MS_ResBlock`. After absorb, binary communication is the locked identity.
- Deformed \(1\times1\) skip / PED = SDformerFlow Eq. 13–14 = `SpikingPEDLayer`. Algebra is A.
- “Dual residual like ResNet projection downsample” names the **graph**, not last-use of two dtypes on one SRAM.

PED Eq. 13–14 **is** the dual-path graph. A reviewer can say: *you implemented SDformerFlow SPE and called last-use a circuit.* The only defense is a **measured** same-port occupancy split \(\{\mathrm{wait\_gate},\mathrm{wait\_PED},\mathrm{wait\_BNstat},\mathrm{wait\_I24\_reread}\}\) that moves 8088, with MS/SEW/FrozenBN as published controls. If the split is simultaneous ready, dual-path is empty and MS was always enough.

G1’s X sentence correctly forbids “analog AT-LIF amplitude shared by two MACs.” That forbids the identity break; it does **not** create a circuit. The producer of the fork is `r1out` / I24, a post-add continuous tensor. AT-LIF only binarizes path A. Last-use of that fork is register allocation of a named SRAM object (round-1 P05-I2 class; still a hole after absorb; still not MS; still not automatically 15%).

**Mitigation that would survive 1.1–1.3:** copy **one** A complete (not four names); one retire-contract figure that cannot be redrawn as valid/ready **and** cannot be redrawn as NRV with a denser predicate **and** cannot be redrawn as SDT Fig. 2 with “optical flow” in the caption; BN-stat stays G3; no Algo-1.

---

## Attack 2 — Union occupancy dense?

**Predicted yes on the tensors the freeze actually names. If true, skip-G1 is empty and last-use-G1 is a throttle.**

### 2.1 Algebra

G1 writes occupancy as last-use of gate **AND** residual. Two different objects got glued:

- **Retirement (AND of completions)** = \(\max(t_A,t_B)\). Occupancy **lengthens** vs gate-only retire. This is the conservative single last-use Q05-I4 already listed as the control. Dual last-use *helps* only if the current schedule **over-holds** (phantom I24 reread / pin after both consumers are done) or **over-issues** (source still producing after last-use).
- **Skip occupancy (union of supports)** = \(\mathrm{support}(s)\cup\mathrm{support}(\mathrm{PED})\). Density **rises** vs gate-only NRV. A zero on the absorbed spike map is a legal skip for consumer A and can be a **nonzero live residual** for consumer B (R2L1 §3.2, R2L2 §3.2). Union skip ≤ PED skip. If PED is dense, union skip ≈ 0.

R2L5 already wrote the collapse: if union density \(\approx 1\) (PED almost always live), skip-on-gate is empty and the sentence collapses to last-use / reread elimination / BN accounting. Those are still not MS. They may still die as register-allocation / Welford-on-fill (L4 A).

Producer-side issue cannot charge union (round-1 P06-I2, still true): a producer slot drops only on **intersection** of skip masks. Union is a consumer-side curiosity. Intersection of `{gate silent} ∩ {PED silent}` is smaller than either, and PED-silent is the scarce one.

### 2.2 This net’s PED is dense by construction

`SpikingPEDLayer`: \(z_{\mathrm{res}}=\mathrm{Conv}_{1\times1,s=2}(I)\) on **continuous** `r1out`; integer PED q24 is **0-diff** vs model. That is a dense multi-bit MAC on the skip tensor, not a sparse event. Even/even (stride-2) sites need the **value** on path B. I24 is the residual stream that is written and later reread for PED **and** for post-BN ADD (R2L5 §3.2–3.3). A gate-silent site is not a license to drop `r1out`.

T10-before-threshold is worse: the mix is a dense \(T\times T\) of continuous pre-threshold values; every spatial site that feeds either consumer keeps mix parents live until \(\max(t_A,t_B[,t_{\mathrm{BN}}])\) (R2L4 §4.2, Q04-I8). Ordinary vs lifting already proved mix **op-count** is not occupancy (260→159, 8088=8088). Union occupancy of a dense parent DAG is dense.

### 2.3 Absorb makes union *more* lopsided, not less

Round-1 F1 union was \(\mathrm{support}(\theta_g)\cup\mathrm{support}(\mathrm{PED})\) with both sides continuous. After absorb, path A is **more** skippable (binary NRV/Bitmap/EM-PM become legal A) while path B **did not become a spike**. The union is then dominated by B. Binary-only A is predicted to print a large skip count and **leave 8088 stuck** — which is exactly H-G1’s first prediction, and also exactly “successful A, dead letter” (R2L1 §8).

H-G1’s second prediction is the honest kill: charging residual last-use moves 8088 **iff residual occupancy is not denser than the binary gate.** The freeze gives no density number. The graph gives a dense 1×1 on continuous `r1out`. Prior: do not import FireFly 70–90% spike sparsity or Gustav Fig. 6 \(P=8\) skip ~80% (wrong net, wrong tensor).

If a later dump shows union density low enough to skip, that is a **new measurement**, not a reason to keep NRV language in the title today.

---

## Attack 3 — 8088 FIFO artifact?

**Open, and currently the strongest alternative. Dual last-use does not get the benefit of the doubt.**

### 3.1 What is actually measured

Same two-stage SIMD source, two writeback slots:

- Always-ready 6938→5354 (−22.83%, **part generic** round→sat fusion).
- Long backpressure **both 8088** (ordinary 260 add/sub and lifting 159+35 RNE/sat).
- Separate integer-consumer model 758777→714889 (−5.78%); do not add tables.
- ALU shortening **raised** FIFO-full wait 1135→2719 (round-1 F3 / REPORT). Fusion already removed ~70 issue ops without moving 8088.

Venue gate is same-port / same-state / same-backpressure **net service ≥15%**, not always-ready, not add count, not −5.78%.

H-G1 disconfirmer 1 is exactly this attack: “8088 is a FIFO-depth artifact independent of residual liveness.”

### 3.2 Two mutually exclusive 8088 stories

**Story FIFO (kills G1).** 8088 ≈ output-port lower bound \(\lceil N_{\mathrm{out}}/2\ \mathrm{slots}\rceil\times\mathrm{fill}\) (ADV_F3 alt-3; Q03-I4: mix CSE is 8088-invariant). Source commits a dense stream of mix outputs. Consumers being “typed” does not reduce committed writes unless a write is **legally droppable**. Dual last-use AND-retire cannot drop a write still required by PED. Drain-first / residual-first credits can **reorder** under backpressure but cannot beat a hard 2-slot drain of a dense output count. Then G1 is the same failure mode as lifting CSE: occupancy mix ≠ service.

**Story JOIN (the steelman).** 8088 is wait-for-consumer-ready: PED/I24/BN not accepting, or I24 reread colliding on 1RW, while the binary gate would already have retired. Then \(\max(t_A,t_B)\) is the occupant, and a last-use tag / independent maps / drain-first **might** move the number — or they might only stop over-issue (throttle): long-BP *count* falls because the producer does less, while completed r1→gate+PED+BN+add vectors per cycle do not rise 15%. PROBLEM.md already showed that cartoon: always-ready −22.83% with long-BP stuck at 8088.

These two stories are **not** both G1. They are F1-vs-FIFO-vs-G3. ADV_F2 already: 8088 is on the **source arms**; projection BN is a later consumer of `proj.conv` at `10×96×120×160`; magically-instant BN can leave 8088 untouched. G1’s “at the absorb cut” names the **threshold/absorb boundary**. The measured 8088 is **pre-threshold T10 writeback**. Q04-I8 is the only parent that places last-use on the mix parent; Q04-I1/I3/I4 place it on gate vs residual tiles; Q02-I2 places it on a scoreboard of GeMM issue vs residual live-range. Those are three different tensors. Fusing them as “typed last-use at the absorb cut” is a category error until the histogram says which tensor owns 8088.

### 3.3 AND-retire is anti-performance unless a phantom exists

If 8088 is real consumer wait, \(\max(t_A,t_B)\ge t_A\) and \(\max\ge t_B\): dual-completion **extends** the conservative hold vs either single consumer. Speedup then requires a **phantom** in the current schedule:

- I24 written and reread after last real use (register allocation / in-place last-use; P05-I2; R2L5 §3.3).
- Co-allocation of bitmap and q24 on one live window (Q04-I4).
- Gate-only NRV that would have been illegal (correctness fix, not 15%).

Phantom-hold elimination is LoopTree retain/recompute and compiler liveness. It may be real. It is not union-NRV, and it is not “absorb cut.” If the 15% is only vs a baseline that kept I24 pinned for the whole frame, the baseline was sloppy, not Gustav-incomplete.

Q04-I3’s gradient (−22.83% / −5.78% / 0%) is **diagnosis**, not a mechanism. “B holds the port A already drained” is consistent with FIFO-full **and** with join. It does not pick G1.

---

## At least two alternative explanations for any predicted 15% win

1. **Throttle / occupancy rename, not service.** Typed credits `min(credit_gate, credit_PED, credit_BNstat)` or AND-retire stop the source from over-issuing into the measured 8088 wall. Long-BP count falls; completed dual-consumer vectors per cycle do not rise 15%. Already observed at always-ready.

2. **Generic fusion, extra skid, or self-inflicted 1RW serialize.** Part of 6938→5354 is already generic round→sat. Depth-1/2 elastic taps hide burstiness under a same-state violation. Serialized dual-read on 1RW is an Eyeriss-multicast / buffet-credit miss: a 1R latch plus refcount 2→0 can print “15%” against a baseline that illegally issued the second consumer on a second grant. 2RW is an illegal upper bound.

Further alternatives that must be named if a number appears: ordinary T10 plus the **same** last-use protocol matching lifting (H-G1 alternative: X is the protocol, not dual-completion occupancy); BN barrier overlap that is actually G3 smuggled in via “BN-stat if live”; I24 reread elimination that is register allocation rather than union-NRV; issue-locked dual accumulator (Q01-I7) stealing a −5.78%-class op-count and quoting it as port service; FIFO-depth change that violates same-state.

---

## Measurement or analysis failure modes

- Reporting always-ready slots, add/sub (260→159), Prosperity \(\Delta S\), Gustav Fig. 6 skip ratios, FireFly FPS/W, or the separate integer-consumer −5.78% as chain service; adding the two tables.
- Single stall bit instead of the wait-class histogram; synthetic ready patterns.
- Free \(\mu/\sigma\) on a local window as the baseline BN wait (undercharge; G3).
- FIFO depth, extra anti-token SRAM port, 2RW, or operand-collector ports not held to the frozen two-stage point.
- Enabling a skip from union occupancy that changes the integer law, then quoting two-window 0-diff as if valid825 AEE were unchanged.
- Measuring source-only or gate-only completion, not residual add after actual full-domain batch stats.
- Comparing dual-completion against a lazy-fork or 2RW oracle the authors introduced, not against frozen joint-ready ordinary.
- **Category error on 8088’s tensor:** attributing source-arm 8088 to `proj.norm_layer` (wrong rank, later in the graph) or to post-absorb GeMM skip (Prosperity sits *after* \(S\) exists; R2L4 §4.1).
- **Algo-1 / silent-channel as a control that drops PED channels.** Function change, not A.
- **Incomplete A counted as X.** 1RW serial subset scan of an \(m=256\) tile is not Prosperity’s Detector (R2L1 §3.4). Building a 1RW Detector for the same forest is how you **finish A**, not X (R2L1 §6).
- Title language: EM/PM/XOR/CAM, “we also Gustavson,” “dual-side sparsity,” “membrane shortcut,” “continuous AT-LIF two MACs.”

---

## Prior evidence that challenges it

**R2L1.** After absorb, `conv2` / `proj.conv` **are** Prosperity’s binary GeMM; copy the forest there as A. Dual-completion last-use is named as a *candidate* X (token `{gate_done, ped_done, bn_stats_ready}`) and immediately fenced: if the only hardware story is “binary-mask prefix of the gate GeMM,” it is still ProSparsity applied to a mask. A letter whose mechanism is product sparsity on the absorbed spike path has no increment, even if the copy is complete and lossless.

**R2L2.** CPTB/NRV copy 1-1 onto the binary GeMM; they do not copy onto T10-before-threshold or onto the residual continuous path. Union occupancy is X2, kill if dense. Dual-retire is X1, kill if same-state budget grows. Gustav+LoAS as stacked A contradicts both papers.

**R2L3.** FireFly-S Bitmap AND is complete A on path A. Algo-1 is illegal if PED needs the channel. Dual-consumer last-use is not in the paper.

**R2L4.** Remaining hole after da4ml+Prosperity is dual last-use **across** \(\Theta\) and 2-port occupancy of the mix. Control: ordinary CSE with the **same** two-output postprocess. If that already matches 8088, this X is dead. Mix CSE already matched 8088.

**R2L5.** MS / SEW / SPE graph are A. Dual-path X must copy binary GeMM on path A, keep path B dense and bit-true, pay BN, retire I24 at union last-use, and move 8088. If the figure is SDT Fig. 2 with “optical flow” in the caption, it is a reskin.

**Freeze.** Source-side arithmetic already failed this exact test: −22.83% absorbed at 8088 on both ordinary and lifting. Any mechanism that only changes producer occupancy or skip of gate-only zeros has a published negative.

**ADV_F1.** Same island, continuous-θg wording. Novelty 1, performance 1, revise. Absorb removed F1’s only identity wedge against NRV. G1 is F1 with that wedge deleted.

---

## Zoo defect (not a fifth attack, a fusion defect)

G1 clusters thirteen independent IDs that are **not one circuit**:

| Mechanism | Parents | Object |
|---|---|---|
| Dual-color / typed last-use FSM | Q01-I2, Q03-I3, Q04-I1, Q04-I4, Q05-I4 | refcount 2→0 on one line |
| Residual-first credits / drain-first | Q01-I5, Q02-I6 | port policy under 8088 |
| Issue-locked dual accumulator | Q01-I7 | one pointer, two ALUs |
| Two-color scoreboard delaying S-issue | Q02-I2 | GeMM issue vs R live-range |
| Two-window release S frees R | Q02-I7 | coherence between bitmap and q24 |
| Dual-queue issue | Q03-I7 | two typed queues, one port |
| Gradient diagnosis | Q04-I3 | not a circuit |

P05 forbade that stack in round 1. Q04’s own cross-note says keep I1/I3/I4 separable under kill-gates. Five-page TCAS-II punishes a zoo. **One** mechanism, or it is not G1.

---

## Mitigation (only if the island is not stopped today)

1. **Do not implement protocol RTL first.** Cheap occupancy dump on captured windows: split 8088 into the wait classes above; print `live(s)` vs `live(I24/PED)` vs `live(T10 parents)` vs `live(BN-stat)`; print union vs intersection vs gate-only density; print stall correlation of gate vs PED; print FIFO-full vs consumer-not-ready. If join is not the bottleneck, **stop G1 the same day**.
2. **One mechanism.** Dual-completion last-use **or** I24 recolor **or** typed credits **or** drain-first — not Q01-I2+I5+I7 plus Q02-I2+I6+I7 plus Q03-I3+I7 plus Q04-I1+I3+I4 plus Q05-I4. BN-stat “if live” is a fact of the net or it is G3.
3. **One A, copied complete.** Prosperity checklist **or** Gustav §3.3 **or** FireFly Bitmap AND, as a **control**, with dual-retire as the only X. Not four names. Algo-1 out. SDT MS is already the student; cite it as A, do not retitle it.
4. **Controls that steal the win if omitted:** ordinary + joint ready; ordinary + same tap/credits (depth cap published); eager duplicate vs lazy barrier vs hybrid; binary NRV vs union occupancy vs dense (no skip); fusion-matched round→sat on both arms; paid full-domain BN vs free μ/σ (illegal as a service claim); serialized 1RW vs 1R multicast as an honest 1RW baseline, 2RW as an illegal upper bound; MS-only projection (no `conv_res`) as a graph control; FrozenBN as a numeric control.
5. **Kill numbers stay conjunctive:** AEE ≤ 1.224801338 relative and ≤ 1.259 abs; 8088 moves; complete-chain same-port/state/backpressure service ≥15%; 0-diff integer windows. If union density is too high to skip, drop NRV language and keep only last-use lifetime — then survive only if phantom hold is the measured 8088 term.

---

## Residual uncertainty

Complete-chain same-resource net service is **not closed**. True paid cost of full-domain `10×96×120×160` BN stats is unknown (G3). Union occupancy density is unmeasured; the graph predicts dense. Live-range overlap of gate vs I24/PED vs T10 parents vs BN-stat is unmeasured. Whether 8088 is join, BN barrier, delay-line 1RW, or FIFO depth is unmeasured — G1, G3, and FIFO-bound are mutually exclusive titles until that split exists. Two-window integer 0-diff is not valid825. Post-absorb \(\rho\) and \(W\) density on this student are unmeasured; importing \(P=8\) or 85% W-sparsity is a different network.

Absorb closed the round-1 identity fight. It did **not** close any of the measurements ADV_F1 already demanded.

---

## Disposition: revise

Keep the **question** (why 8088 did not move when the source ALU did; dual-consumer last-use vs binary NRV after absorb). Do not keep the fused wording, the four-name A, FireFly Algo-1 as A, or a build plan.

- **Not retain.** As written, G1 is F1 with absorb lipstick plus a scoreboard/NRV/SDT reskin surface. Main-island status does not make a slogan a letter.
- **Not pause-as-unworkable.** The question is still the only one aimed at the measured 8088 hole. Pause *circuit* work until the 8088 mix and union density are printed (same day-one dump as ADV_F1). That is sequencing, not a pause of the island.
- **Not stop today.** Stopping now would be claiming FIFO-bound or dense-union without the histogram. Those are the **stop triggers**, not the current evidence.
- **Stop triggers (predeclared):** (i) 8088 histogram is FIFO-full / 2-slot drain / T10 fill, join <5% of stalls; (ii) union density too high to skip **and** phantom I24 hold is not first-order; (iii) binary-only complete A already moves 8088 ≥15% at 0-diff; (iv) simultaneous ready ≥95%. Any one converts this sheet to **stop**.

If those prints show simultaneous ready, dense union without phantom hold, or a non-join bottleneck, **stop** rather than pivot into G3 under this ID (G3 is hygiene). If they show phantom I24 hold on a dense union, the surviving sentence is last-use/reread elimination with Gustav/Prosperity as copied controls — still revise, still not “typed NRV.”

---

## Novelty 0–4 and performance-plausibility 0–4

Rubric (same session): novelty 0 none; 1 prior only; 2 hole but X not located; 3 measurable X not closed; 4 title-level. Performance 0 cannot help; 1 15% as standalone structurally implausible; 2 unmeasured, could matter if the histogram says join; 3 plausible 15% after named controls; 4 expected to clear 15% and AEE as the letter. 4 almost never.

- **Novelty: 1.** Dual-lifetime of a **binary gate after absorb** vs a **continuous residual/PED** is a real *identity* hole vs Gustav (single binary \(C\), commit at LIF), Prosperity (one Y), FireFly-S (one accumulate/LIF; Algo-1 spike-only), and SDT MS (one stream \(U\), add-before-SN). That hole is already written as R2L2 X1/X2 and R2L5’s leftover sentence. Absorb **shrunk** the hole: path A is now those papers’ object, so “typed” is “two dtypes,” which a TCAS-II reviewer is trained to subtract as scoreboard/liveness. The circuit objects G1’s parents actually name — credits, last-use tags, elastic fork/join, packed occupancy vectors, dual queues — are those relative priors. Thirteen IDs fused into one letter is a zoo. A 2 would require a single copied A plus a measured retire-contract figure that cannot be redrawn as valid/ready **or** as NRV-with-OR. As fused: 1. (Round-1 F1 was already 1; absorb did not add a degree.)

- **Performance-plausibility: 1.** The 15% is a copied kill-gate, not a bound from occupancy. Historical evidence cuts **against** a source-side occupancy trick: −22.83% always-ready, 8088 unchanged; FIFO-full wait scaled the wrong way when the ALU shortened. AND-retire lengthens hold. Union occupancy of a dense 1×1 PED / dense T10 parent is expected dense (R2L2 mechanism kill; R2L5 collapse; producer can only charge intersection). A 15% chain win then requires an unmeasured phantom (I24 reread or over-pin) to dominate 8088 *and* to be removable at equal ports — which, if true, may be register allocation or G3, not typed NRV. Throttle can fake the 8088 number. Until the wait histogram exists, 15% is not a prediction; it is a hope. A 2 would need that histogram showing last-use slack ≥15% of 8088 with union skip unused. As fused: 1.

| Axis | Score | One line |
|---|---:|---|
| **Novelty** | **1** | Hole is real and already written in R2L2/R2L5; the named circuits are scoreboard/NRV/MS. Absorb made A legal, so the reskin reading is the default. |
| **Performance** | **1** | Union predicted dense; AND-retire lengthens; 8088 already invariant to mix CSE; FIFO artifact unrefuted. |

---

## Decision card

| Item | Call |
|---|---|
| **Disposition** | **revise** (stop if histogram says FIFO or dense-union-without-phantom) |
| **Novelty** | **1 / 4** |
| **Performance** | **1 / 4** |
| Reskin of scoreboard? | **Yes** (default reading; Q02-I2 names it) |
| Reskin of Gustav NRV? | **Yes**, unless one complete A is copied and the *retire contract* is the only X; union-predicate NRV is X2 and dies if dense |
| Reskin of SDT MS? | **Yes**, if the figure is add-before-SN / SPE graph; **no** only if last-use of `{gate, PED, BN, I24 reread}` is measured and moves 8088 |
| Union occupancy dense? | **Predicted yes**; unmeasured; skip-G1 empty if true |
| 8088 FIFO artifact? | **Open, strongest alternative**; unmeasured; G1 does not get the benefit of the doubt |
| Do not do | protocol RTL, four-name A, Algo-1, analog-amplitude wording, title before 8088 split |
| Day-one dump | wait-class histogram + union/intersection density + FIFO-full vs consumer-not-ready |
