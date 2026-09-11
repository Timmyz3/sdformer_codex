# ADV_G4 — Train binary support at matched rate

Reviewer did **not** originate G4. Sources under review: `round2_absorb/fusion/FUSION_R2.md` G4 (from Q06-I2/I5/I6/I8); freeze `round2_absorb/PROBLEM.md`; identity `IDENTITY_ATLIF.md`; parents `round2_absorb/independent/Q06_train.md`; FireFly-S / Bishop / Phi training contract `round2_absorb/literature/R2L3_firefly_bishop_phi.md`. Pins only: R2L1 (Prosperity does not retrain), R2L2 (Gustav “no architecture constraints”), R2L4 §4.5 (Prosperity cannot repair lifting AEE), H-G1 (binary-only skip vs 8088).

This file is an attack, not a salvage pitch. Scores are not accept probabilities.

---

## Verdict

| Field | Call |
|---|---|
| **Disposition** | **stop** as title |
| **Novelty** | **1** / 4 |
| **Performance** | **1** / 4 |
| Fusion “second queue” | **meters only** (ordinary vs lifting vs shuffle/Bernoulli on frozen checkpoints). Not a train–eval letter. |
| The one paired recovery | **Do not spend it here.** |

Not retain. Not revise-in-place (rewriting G4 as last-use is G1; as OF-geometry is G2; as lifting-mix recovery is Q06-I1, a different parent G4 did not take). Not pause-for-a-kinder-G4: R2L3 already named the object as **A’s software half**. Cheap untrained dumps of Q06’s shared meters are allowed; they are not G4.

---

## 0. Candidate as written (not endorsed)

From `FUSION_R2.md`:

> **G4 Train binary support at matched rate (second queue)**
>
> From Q06-I2/I5/I6/I8. One paired recovery. Must beat shuffle/Bernoulli at same \(\rho_S\) and still move 8088. Lifting AEE +0.013 is a wall.

Parents (compressed; Q06’s own header says they are **competing objectives for one pair**, pick at most one):

| Parent | Trained object | Copied engine | Alleged X |
|---|---|---|---|
| **I2** | All-zero spatial/channel rows at matched mean \(\rho_S\) | Gustav NRV/CPTB | Beat rate-matched shuffle/Bernoulli |
| **I5** | Structure/prune absorbed \(W'=\theta W\); spike side **AEE-only**, no \(\|S\|_0\) | FireFly-S Bitmap AND | W-side structure + unregularized \(S\) on a flow student |
| **I6** | Consecutive all-zero timesteps on C12 | Gustav CPTB + FireFly temporal bitmaps | Flow loss prefers block zeros; \(\lambda\|S\|_0\) does not |
| **I8** | Threshold-margin + \(S[t]\oplus S[t-1]\) flicker on pre-threshold \(m\) | NRV/CPTB packing of chatter-poor \(S\) | Same \(\rho_S\), fewer isolated ones |

G4’s title sentence (“matched rate”) is I2/I6/I8. I5 **refuses** a spike-rate term. The cluster is already four letters.

Shared Q06 law that G4 inherits and then breaks: one train–eval pair (frozen ordinary vs one trained student, same ep34 / C12 / H67); AEE must land \(\le 1.224801338\) and \(\le 1.259\); skip accounting copies Prosperity / Gustav / FireFly / LoAS **whole**; “Training is not a new skip circuit.” If meters 2–5 are explained by unstructured \(\Delta\rho_S\), every Q06 idea is a reskin and the training letter is not written.

---

## 1. Attack

### 1.1 I5 is FireFly-S contribution 1, copied onto DSEC

FireFly-S (`R2L3` §1.1–1.3) is a **joint** HW/SW paper. The software half is not optional garnish.

**What MUST be trained (FireFly-S):**

| Item | In the paper | If skipped |
|---|---|---|
| Gradient-rewiring + Laplace/ℓ2 on `θ`, conv **and** FC **and** bias | **Yes, during training** | Detector still skips *existing* W zeros; 85–95% W sparsity and the Bitmap vs COO/CSR region are **not free**. Dual-side collapses toward spike-side. |
| Joint 4-bit LSQ of **W + bias + Vth**, shared per-channel scale | **Yes, during training** (STE) | Integer-only IF/LIF and “scale absorbed into threshold” break. |
| Silent-channel Algo 1 | Eval on trained IF/LIF, then drop | Legal **only** if the only consumer is “does this channel ever spike”. |
| Spike sparsity | **Not** a training loss. “Natural sparsity of spikes inherent to SNNs.” | Inference-time, from thresholding. |

They **reject** prune-then-quant or quant-then-prune as a separate post-process.

Q06-I5, quoted against that table:

> Structure or prune weights **after** the absorb rewrite \(W'=\theta W\), copy FireFly-S bitmap AND as the dual-side engine, and spend the one training pair’s spike-side budget only on AEE (**no extra \(\|S\|_0\)**).

That is FireFly-S’s software contract with the names swapped: rewiring/LSQ → “structure \(W'\)”; natural spike sparsity → “AEE-only, no \(\|S\|_0\)”; Bitmap detector → copied A. The claimed X is “they did not train DSEC AT-LIF / absorb \(\theta\) / leave a residual consumer.” Task transfer plus a second tensor the paper does not skip is **not** a new training method. It is “run FireFly-S’s train loop on our net.”

R2L3 already closed this as X:

> Copying HW skip **without** the paper’s training/calibration is not “the method”: W-side dual-side needs prune+recover …
>
> **Training hooks that move the frozen student** (BSA, ECP-aware, PAFT, rewiring+LSQ) are **not** X by themselves; they are **A’s software half**. Using them on this net is a new training run under the AEE kill-gate, **not a circuit title**.

G4-as-I5 is that paragraph, retitled. Two-way bind:

- Copy Bitmap AND **without** rewiring+LSQ → incomplete A (R2L3 §4.2). Dual-side is almost only spike-side.
- Copy rewiring+LSQ **and** call it X → the venue subtracts FireFly-S §I.1 and rejects for missing relative prior.

I5’s own kill already says the interesting outcomes are (a) AEE restored without giving back skip, (b) AEE fail, or (c) AEE-only training **hurts** dual-side skip by increasing \(S\cap\mathrm{supp}(W')\). Those are **control measurements for a copied FireFly**, not a five-page increment. Outcome (c) is a predicted **negative** of A’s software, not X.

**Attack result:** I5 is FireFly-S’s software half. It cannot be G4’s title and it cannot share the one pair with I2/I6/I8 (those *do* regularize \(S\)).

### 1.2 I8 is Phi PAFT with a flicker alias

Phi (`R2L3` §3): binary activation rows → 128 Hamming patterns (k-means calibration) + optional **Pattern-aware Fine-tuning**. PAFT is a few-epoch Hamming regularizer \(R=\sum H(\mathrm{Act},\mathrm{Pattern})\), \(\mathrm{Loss}=\mathrm{Loss}_{orig}+\lambda R\), on a **pre-trained** SNN. Without PAFT they claim lossless vs bit-sparsity; with PAFT ~1.26× extra runtime and a student change. Frozen ep34 does not include PAFT. AEE gates apply.

Q06-I8: penalize \(|m-\theta|<\varepsilon\) and \(S[t]\oplus S[t-1]\) so post-absorb \(S\) is “crisper 0/1 with longer zero runs,” \(\theta\) still absorbed, \(\rho_S\) held.

| Phi PAFT | I8 |
|---|---|
| Hamming(Act, Pattern) | Hamming-like flicker \(S[t]\oplus S[t-1]\) plus margin mass |
| Goal: fewer L2 nonzeros; packable rows | Goal: fewer isolated ones; packable zeros |
| Optional FT of a frozen student | The one paired recovery |
| Alphabet: 16-bit binary patterns | Alphabet: blocky 0/1 for NRV/CPTB |
| One LIF consumer after L1+L2 | Residual/PED still a second tensor |

“Packing engines hate isolated ones” is **why Phi exists** (`R2L3` §3.1: binary 0/1 “enforces a more structured distribution”; L2 corrects leftover Hamming). I8’s objection column already admits the reskin: “Surrogate-gradient SNNs already use margins.” The rebuttal offered is “show the hardware meter moving at matched rate on this DSEC student.” That is a **meter protocol**, not a new regularizer. Matched-rate Hamming/flicker that feeds copied NRV/Phi is PAFT retargeted from classification to flow.

Phi codebook calibration (k-means, not a gradient) is also A, and also a student-adjacent change: a codebook fit on **this** net’s spike rows, PWPs rebuilt after \(W\leftarrow\theta W\). G4 never even names the codebook, so if someone later claims Phi L1 speedup they have neither PAFT nor calibration — incomplete A again.

**Attack result:** I8 is Phi’s software hook (plus a stock margin term). Hardware remains copied packing. X is empty once PAFT is in the denominator.

### 1.3 I2 / I6 are Bishop BSA, not a Gustav/FireFly hole

FireFly-S **does not train spike structure**. GustavSNN claims compatibility “without imposing any constraints on the choice of SNN architectures or neuron models” (`R2L2` §9) and **consumes** NRV-friendly rows at inference. Prosperity **does not retrain** (`R2L1` §1.4: lossless / algorithm-agnostic on GeMM). So I2/I6 cannot hide behind “Gustav forgot to train.” The paper that **does** train structured firing is in the same R2L3 sheet:

Bishop BSA (`R2L3` §2.3, §4.1): \(L_{tot}=L_{CE}+\lambda L_{bsp}\), \(L_{bsp}=\sum Z\), \(Z=\|X_{\text{tokens in bundle, times in bundle, feature }d}\|_0\), on MLP / projection / Q / K. Without BSA, Model 1: **29% of bundles active** — “restricted opportunities for computation skipping.” BSA is **not** Han W-prune; it is a **firing-structure** loss. FireFly rewiring and Phi PAFT are different losses; BSA is the one that matches I2/I6.

| G4 parent | BSA work unit | G4 work unit |
|---|---|---|
| I2 | TTB L0 (tokens × times × feature) | Spatial / channel **row** L0 (NRV empty-row) |
| I6 | Time-group L0 inside the bundle | Consecutive-timestep all-zeros (CPTB / FireFly temporal bitmap) |

Retargeting the bundle axis from (token, time) to (spatial row) or (C12 timestep run) is how you **copy** BSA onto r1, not how you invent it. I2’s sentence “Gustav assumes NRV-friendly rows exist; it does not train a flow student to emit them” is true of Gustav and **false** of the R2L3 prior that actually trains occupancy. I6’s sentence “classification accelerators treat extra spikes as evidence of class; they do not train temporal block-sparsity so a 2D flow field stays accurate” is a **task** claim. TCAS-II 5 pages does not give a circuits title for “BSA on DSEC.”

I2/I6’s “X test” is beat shuffle/Bernoulli at the same \(\rho_S\). That test is how one **demonstrates BSA worked** (structured L0 vs i.i.d. rate). Q05 already wrote the same reskin test for *inference* geometry (`Q05` cross-idea: equal-density shuffle must hurt or X was Prosperity). A control protocol is not X.

**Attack result:** matched-rate row/time silence is Bishop BSA retargeted to Gustav/FireFly packing units. FireFly’s spike side remains untrained, as in the paper. The shuffle foil belongs in the methods appendix of a copied-A letter, not in a title.

### 1.4 The cluster is four letters; Q06 already forbade it

Q06 header: **one** pair; ideas are competing **objectives**; pick **at most one**; the rest are kill-controls or disconfirmers on the **same** checkpoints. I8 additionally: this term **xor** I1’s mix-AEE term, not both.

G4 concatenates I2+I5+I6+I8 and still says “one paired recovery.” Incompatibilities:

| | Regularize \(S\)? | Regularize \(W'\)? | Time vs space | Uses lifting mix as DoF? |
|---|---|---|---|---|
| I2 | Yes, row L0 at matched \(\rho_S\) | No (Prosperity is a foil) | Space | No |
| I5 | **No** \(\|S\|_0\) | **Yes** (the point) | — | No |
| I6 | Yes, temporal run-length | No | Time | Mix must stay noncausal |
| I8 | Yes, flicker/margin at matched \(\rho_S\) | No | Time (flicker) | xor I1 |

A single λ cannot be “NRV rows and not \(\|S\|_0\) and CPTB runs and Hamming flicker and FireFly rewiring.” Architecture search over those four is a **grid**, which Q06’s shared protocol forbids as “the pair.” Fusion of four software halves is the F1 zoo failure mode on the training axis.

**Attack result:** G4-as-written is not an island. It is a queue of A’s training hooks. A 5-page letter cannot carry them.

### 1.5 Lifting +0.013 already spent the AEE slack; the pair cannot recover two graphs

Frozen numbers (`PROBLEM.md`):

- Ordinary AEE **1.219801338**
- Lifting raw **1.232979368** (relative **+0.013178**; abs 1.259 **pass**, +0.005 **fail**)
- Recovery must land AEE \(\le 1.224801338\) and \(\le 1.259\)
- Ordinary CSE 260 add/sub; lifting 159 + 35 intermediate RNE/sat
- Same-port always-ready 6938→5354 (−22.83%, part generic fusion); long backpressure **both 8088**

R2L4 §4.5: Prosperity is lossless on binary GeMM and **cannot** repair lifting +0.013178. FireFly Bitmap AND and Phi L1/L2 are likewise lossless **given** \(S,W'\). Skip engines do not buy AEE. Only a **student change** can.

Two starts, both lethal for G4 as title:

1. **Pair from lifting.** Start already **+0.013** dead on the relative gate. Need \(\ge 0.008\) absolute recovery **and** structured support **and** 8088 movement, in one run. The parent that is allowed to touch the mix is **Q06-I1**, which G4 **did not include**. I8 explicitly xor’s I1. So G4 on lifting is I1’s job with a support garnish — the garnish Q06-I1 already listed as a kill if meters do not move vs “AEE recovery, no support term.”
2. **Pair from ordinary.** Relative slack is **0.005**. No lifting CSE to keep. Then G4 is “train FireFly/Phi/BSA on the frozen flow student.” 8088 is still 8088 on ordinary. Binary-support training does not own a mix rewrite, so it cannot cite 260→159. Always-ready −22.83% is already absorbed and partly generic fusion; Q06 shared meters forbid spending the 15% gate on it unless trained \(S\) actually changes that net.

ADV_F3 already stopped spending the **same** one pair on dyadic/CSD QAT of the 40 coeffs. F4 wanted the pair for prefix-zero. G4 wants it for A’s software. The freeze’s recovery budget is **one**. Using it to retarget FireFly-S/PAFT/BSA is how a circuits letter becomes a second-rate training paper **and** still misses 8088.

**Attack result:** lifting AEE is not G4’s “wall to climb.” It is a published fail that skip-training cannot patch without becoming I1, and I1 is not G4. Ordinary start has no CSE prize and no 8088 theory.

### 1.6 Binary support cannot legally move 8088

G4’s hardware kill, copied from the fusion card: “still move 8088.”

After absorb, two objects stay distinct (`IDENTITY_ATLIF.md`, `PROBLEM.md`):

1. Spike GeMM — binary select-add. FireFly Bitmap AND, Gustav NRV/CPTB, Phi patterns, Prosperity EM/PM apply **here as complete A**.
2. Residual / PED / I24 — a **different continuous tensor**. Dual-side sparsity and pattern hierarchy **do not** apply (`R2L3` §0, §4.3).

H-G1 (already on this session): 8088 is dominated by the continuous residual remaining live after the binary spike consumer would have retired; **binary-only NRV/product-sparsity cannot move 8088**.

Q06-I3’s own objection, which G4 did not take as a parent but which still binds the stall: long backpressure both 8088 is a warning that sparsity in \(S\) may not touch that number. Q06-I7 (last-use licenses silence) is the training cousin of **G1**, and G4 **dropped** it.

FireFly-S remaining kills after absorb (`R2L3` §1.5): Algo 1 is a **function change** if a never-spike channel still feeds PED/BN/add; bias-as-bubble is not a PED retirement rule; one accumulate/LIF consumer. I2’s empty rows and I5/Algo-1 silent channels are **illegal** as net skip if consumer B is live. If training also zeros the residual to make rows “empty,” identity/AEE/0-diff die. If it does not, 8088 does not move.

Venue gate is complete-chain same-port/state/backpressure **net service ≥15%**, not Bitmap-AND density, not NRV empty-row rate, not Phi’s 3.45× vs Stellar. G1 is the island that names dual last-use. G4 training of **gate** support is the wrong consumer.

**Attack result:** even a perfect matched-rate NRV/PAFT/rewiring student is a **spike-path** result. The freeze’s hostile number is a **wait** on a path those engines do not see. G4 cannot charge 8088 without stealing G1’s object, at which point G4 is a student sidecar and should not exist as an ID.

### 1.7 Shuffle/Bernoulli is not X; it is the reskin detector

Fusion card: “Must beat shuffle/Bernoulli at same \(\rho_S\).” Q06: if meters 2–5 are all explained by \(\Delta\rho_S\) at unstructured support, **do not write the training letter**.

That sentence is a **suicide pact**, not a contribution. Beating shuffle means the trained \(S\) has structured L0. Structured L0 is BSA/PAFT/NRV-friendly occupancy — **A’s training target**. Failing shuffle means unstructured rate, which Q06 already called a Prosperity/Gustav/FireFly reskin.

Neither branch yields a circuit:

| Outcome | What it is | Letter? |
|---|---|---|
| Trained NRV/CPTB/AND \(\le\) shuffle | Unstructured \(\rho_S\) | Reskin; Q06 says do not write |
| Trained meters beat shuffle, 8088 unmoved | BSA/PAFT succeeded as software | Control table for copied A; service fail |
| Beat shuffle **and** 8088 moves | Then 8088 moved because residual/BN/last-use moved, or because the source issued less (throttle) | That mechanism is G1/G3, not G4; prove it without the train first |

Prosperity product formula \(\rho_S\rho_{W'}\) is the unstructured foil I3 named. G4 did not even take I3 (alignment of \(S\) with \(W'\)). Matched-rate **product** structure would still be FireFly dual-side / Prosperity accounting, not X.

**Attack result:** the shuffle test is necessary hygiene on any copied skip engine. It is not a title.

### 1.8 Additional kills G4 left on the table

**Identity laundering.** Margin on \(m\) (`I8`) is one rewrite away from “analog spike amplitude.” Kill-gate already forbids calling the margin that or refusing \(\theta\)-absorb. A TCAS-II reviewer who just read round-1 continuous-\(\theta_g\) drafts will assume the laundering unless absorb is shown on the trained checkpoint (\(S\in\{0,1\}\), \(W'=\theta W\)).

**Algo 1 / silent rows vs full-domain BN.** Projection BN uses **actual batch stats over \(10\times 96\times 120\times 160\)**. Training channels/rows toward never-fire changes the BN domain if those sites still enter \(\mu,\sigma\), or changes the function if they are dropped (FireFly Algo 1). G3 already owns BN as **hygiene**. G4 must not get a BN-stat side effect for free, and must not freeze BN as a cheat (`PROBLEM.md`).

**Do not add tables.** Integer-consumer 758777→714889 (−5.78%) is a different resource point. Always-ready −22.83% is partly generic fusion. G4 is forbidden from stacking either with a Bitmap-AND cycle model.

**Quotes not to launder** (`R2L3` §5): FireFly-S 85–95% W sparsity, 70–90% spike sparsity, 4-bit, FPS/W; Bishop 5.91× / 29% bundles without BSA; Phi 3.45× / 1.26× PAFT. None of these are DSEC valid825 or 8088.

**G2 already owns task-structured support at inference.** G2 (conditional): OF-geometry on absorbed binary support; kill if gain is only unstructured \(\Delta\rho_S\). G4 is “train G2’s occupancy.” If ordinary ep34 **already** has edge-like rows / C12 zero-runs, I2/I6 die on ordinary-vs-trained (`I6` objection: events are already sparse in time). That is a **dump**, not a pair. If ordinary does **not**, training it is still BSA.

---

## 2. A / B / X after the attack

| Piece | G4 as written | After attack |
|---|---|---|
| **A** | Implicit: Gustav NRV/CPTB, FireFly Bitmap AND, shuffle as foil | **Incomplete.** Complete A on the spike path is FireFly-S **Bitmap + rewiring+LSQ** (if W sparsity claimed), Bishop **TTB/BSA** (if structured firing claimed), Phi **codebook + PWP + optional PAFT** (if pattern skip claimed), plus Prosperity EM/PM (no train) and Gustav NRV/CPTB (no train) as **parallel** engines (`R2L2` §8: not a stack). Shuffle/Bernoulli/time-shuffle are **controls inside A**, not X. |
| **B** | Binary support of a flow student can be trained at matched \(\rho_S\); lifting +0.013 is a wall; 8088 must move | **Mis-typed.** Real freeze holes are pre-threshold T10, residual/PED dual-consumer, full-domain BN, same-port last-use, DSEC AEE (`SCOPE.md` / `R2L3` §4.3). “Train \(S\)” is not among them. Lifting +0.013 is an AEE **fail**, not a training license. 8088 is wait-not-ALU on a path binary skip does not retire. |
| **X** | Beat shuffle at matched \(\rho_S\) with copied skip | **Empty as title.** That beat is BSA/PAFT/rewiring doing what those papers train them to do, on a new task. Task transfer is not TCAS-II X. Dual residual consumer is G1’s object; OF geometry is G2’s; mix recovery is I1’s. G4 names none of those as the mechanism. |
| **Kill** | AEE; shuffle; 8088; lifting wall | **Necessary, not sufficient.** Add: pair spent on I2 **and** I5 **and** I6 **and** I8; W-prune defined on unabsorbed \(W\); Algo 1 / empty rows while PED live; margin written as analog \(\theta_g\); citing −22.83% / −5.78% / FireFly 85% / Phi 3.45×; 8088 movement that is throttle or G1 last-use smuggled in; second recovery after miss. |

---

## 3. Scores

### Novelty **1** / 4

Rubric: 0 none; 1 prior only; 2 hole exists but X not located; 3 measurable X, not closed; 4 title-level.

- **Not 0:** DSEC flow + absorb + a second continuous residual is a setting FireFly-S/Phi/Bishop did not evaluate. The matched-rate shuffle dump is a real (and cheap) measurement.
- **Not 2:** X is located and already classified. `R2L3` §4.3 item 6: training hooks that move the frozen student are **A’s software half**, not a circuit title. G4’s four parents map onto FireFly rewiring+LSQ (I5), Phi PAFT (I8), Bishop BSA (I2/I6) with one-to-one loss types. “They did not run it on optical flow” is the sentence a TCAS-II reviewer subtracts.
- **Not 3/4:** no new skip circuit; Q06 itself says training is not a new skip circuit. Four-ID fusion is a zoo. Relative prior is complete once the software halves are copied.

A 2 would require a training object those papers **cannot** name even after retargeting work units — e.g. a dual-consumer loss that is legal for residual last-use (Q06-I7) or a mix-only recovery that keeps lifting CSE (Q06-I1). G4 **excluded** both.

### Performance **1** / 4

Rubric: 0 already falsified / cannot pass conjunctive gates; 1 local negatives dominate; 2 one gate plausible, the other likely fail; 3 both gates look reachable; 4 measured.

Conjunctive gates: valid825 AEE \(\le 1.259\) **and** \(\Delta\le +0.005\) vs 1.219801338 **and** beat shuffle/Bernoulli at matched \(\rho_S\) **and** 8088 moves **and** full-chain same-port service \(\ge 15\%\) **and** I24/PED q24 0-diff **and** \(S\in\{0,1\}\) after absorb **and** one pair, no λ grid.

Why 1, not 0: the *family* (log support meters; optionally copy FireFly’s train loop as A) is not pre-falsified on ordinary. Shuffle-vs-true occupancy on ep34 has not been printed. Why not 2: every local constraint G4 must pass is already hostile, and the cluster cannot even be executed as one pair.

| Local negative | Number / fact | What it does to G4 |
|---|---|---|
| Ordinary dense T10 | AEE **1.219801338** | Strong control; relative slack **0.005** |
| Lifting T10 | AEE **1.232979368** (\(+0.013178\)); 8088 unchanged | Skip engines cannot repair AEE (`R2L4` §4.5); G4∉I1 |
| Source SIMD | always-ready −22.83%, long-BP **both 8088** | Arithmetic win absorbed; binary \(S\) is the wrong stall |
| Dual consumer | residual/PED continuous; 0-diff q24 | Empty-row / Algo 1 skip is illegal if B live |
| FireFly spike loss | **none** in the paper | I2/I6/I8 are extra vs FireFly; I5 is FireFly and then cannot beat shuffle by training \(S\) |
| One pair | Q06 law | I2 ⊥ I5 ⊥ I6 ⊥ I8; I8 xor I1; F3 already claimed the pair and was stopped |
| Full-chain service | **UNKNOWN** (`SCOPE.md`) | Cannot claim 15% from NRV rate or Bitmap density |

Expected order if someone spends the pair anyway: (i) I5-style W-prune, frozen \(S\): AND-skip up, AEE dies or residual disagreement; recovery FT restores AEE and **gives the skip back** (I5 outcome a/c). (ii) I2/I6/I8-style \(S\) structure: either shuffle already matches ordinary (no train job) or AEE exceeds +0.005 when rows/times are forced quiet on a flow field. (iii) 8088 stays 8088 unless residual last-use was the stall, which is G1’s measurement, not G4’s reward. That is performance 1.

---

## 4. Disposition: **stop** as title

**Not retain.** A is FireFly-S/Phi/Bishop **software**. X is a shuffle protocol. Four parents violate Q06’s one-objective law. Spending the only recovery budget here burns I1/F3/F4/G1’s option value after lifting already failed +0.005.

**Not revise-now.** The honest rewrites are already other IDs:

| Tempting rewrite | Actual ID | Why not “G4 revise” |
|---|---|---|
| Train mix until lifting AEE \(\le+0.005\), log support | Q06-I1 | G4 did not take I1; I8 xor I1 |
| Dual last-use; maybe silence after residual consume | G1 / Q06-I7 | Hardware retire contract; training optional sidecar |
| OF geometry / paid translation of NRV | G2 | Inference occupancy, not a student change |
| Paid full-domain BN | G3 | Hygiene |

Morphing G4 into those is how F4 tried to live. Stop the ID.

**Not pause.** Pause was correct for F4 because 8088’s *wait class* was untyped and might have been prefixable. G4’s object is typed: it is A’s train loop. A dump of Q06 shared meters on **frozen** ordinary and lifting (no pair) is cheap and should be done **as G1/G2 controls**. If shuffle already equals true NRV/CPTB, I2/I6 are dead without training. If not, training is still BSA/PAFT. Either way G4 is not a letter.

**Second queue, demoted.** `FUSION_R2` queued G4 behind G1. That queue is **logging**, not a second paper:

1. On ordinary freeze **and** lifting-raw, after \(\theta\)-absorb, spike path only: \(\rho_S\), \(\rho_{W'}\), \(\rho(S\odot W')\) vs \(\rho_S\rho_{W'}\); NRV empty-row vs spatial shuffle; temporal run-length vs time-shuffle; FireFly-style AND-skip vs frozen-\(S\) weight-structure foil.
2. Identity check: \(S\in\{0,1\}\), \(W'=\theta W\), residual/PED still continuous, q24 0-diff.
3. Do **not** train. Do **not** spend the pair. Do **not** cite −22.83% / −5.78%.

If a later letter **claims** FireFly dual-side W sparsity or Phi PAFT extra, it must copy that software as **A** and re-gate AEE. That run is completeness of the denominator, still not G4.

---

## 5. What a TCAS-II reviewer would punish (if G4 were submitted as written)

- Renaming FireFly-S gradient rewiring + LSQ, Bishop BSA \(L_{bsp}\), or Phi PAFT as the contribution.
- Quoting 85–95% W sparsity, 70–90% spike sparsity, 5.91×, 3.45×, 1.26× PAFT, 4-bit FPS/W as this student’s service.
- Copying Bitmap AND / NRV / Phi L1 without the paper’s train/calibration, then claiming the method.
- One pair that is secretly four λ’s, or a second recovery after AEE miss.
- Keeping unconstrained lifting (relative +0.013 fail) and asserting skip will fix AEE (lossless engines cannot).
- Empty-row / Algo 1 skip that drops a channel PED still reads; or secretly binarizing residual to fake empty spike rows.
- Calling a margin on \(m\) analog \(\theta_g\), or refusing absorb on the trained checkpoint.
- Reporting always-ready −22.83% or integer-consumer −5.78% as G4’s gain.
- Shuffle/Bernoulli as the **title** rather than the reskin detector that kills the title.
- “First structured-sparsity training for event optical flow hardware” (task transfer; SENECA already is event-OF hardware; FireFly/Bishop/Phi already train structure).

Venue: 5 pages, one mechanism, circuits+systems, binary accept/reject. Missing **relative prior** (the software half) and missing **measured 8088/15%** are the standard reject.

---

## 6. Identity and stacking

- Spike path after absorb stays \(\{0,1\}\times W'\). Residual/PED/I24 stays a different continuous tensor. G4 may not collapse them to fake NRV rows.
- Do not revive unabsorbable \(\theta_g\), HBG-RP int8, CIM, OpenROAD-as-PPA, FPS products.
- Do not stack FireFly + Bishop + Phi as one “sparsity training” layer (`R2L2` §8 analogue: parallel priors, not a fusion title).
- G4 is not a stack with G1 last-use or G3 BN. If meters ever show 8088 moving, the mechanism is last-use or BN until proven otherwise; G4 **stops** rather than claiming the stall.
- G2 may use the **untrained** shuffle dump. That does not resurrect G4.

---

## 7. One-paragraph owner summary

G4 is the FireFly-S / Phi software half of A, with Bishop BSA filling the structured-firing slot FireFly explicitly does not train, clustered as four incompatible objectives for a budget Q06 already set at **one** pair. I5 is rewiring+LSQ and AEE-only spikes — FireFly’s actual train contract. I8 is PAFT (Hamming to packable 0/1). I2/I6 are BSA retargeted to NRV rows and CPTB runs; beating shuffle at matched \(\rho_S\) is how those papers show structure, not X. Lifting already failed the relative +0.005 gate by **+0.013178**; lossless skip cannot repair it; recovering it is I1, which G4 did not take. Binary support training cannot legally move **8088** while residual/PED stays live (H-G1; R2L3 dual-consumer kill). Novelty 1 (prior only). Performance 1 (AEE slack gone or 0.005, stall on the wrong tensor, cluster un-runnable). **Stop** as title. Keep Q06’s meters as untrained controls on G1/G2. Do not spend the paired recovery here.

End of ADV_G4.
