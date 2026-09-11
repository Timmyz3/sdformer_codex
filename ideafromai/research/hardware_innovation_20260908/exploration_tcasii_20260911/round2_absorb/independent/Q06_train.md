# Q06 — training that changes binary support after absorb

Date: 2026-09-11. Ideas, not claims. Identity lock: AT-LIF \(o=\theta\cdot H(m-\theta)\in\{0,\theta\}\); inference absorbs layer-shared \(\theta\) by \(W\leftarrow\theta W\); the spike path is binary \(\{0,1\}\times W\). Residual / PED / I24 stay a different continuous tensor.

**Training object.** Not analog \(\theta_g\), not unabsorbable int8. Training may change only the **support** of the post-absorb binary tensor \(S\in\{0,1\}\) (and, if used, the support of absorbed \(W'=\theta W\)): product-sparsity density \(1-\rho(S\odot W')\), Gustav NRV empty-row rate, FireFly dual-side bitmap-AND skip. AEE gates stay in force.

**Observation, not novelty.** Ordinary AEE \(1.219801338\); lifting raw \(1.232979368\) (relative \(+0.013178\); abs \(1.259\) pass, \(+0.005\) fail). Recovery must land AEE \(\le 1.224801338\) and \(\le 1.259\).

**One paired recovery only.** Do not run a campaign per idea. The letter gets **one** train–eval pair: frozen ordinary student vs one trained student (from the same ep34 / C12 / H67 motion setup). Ideas below are competing **objectives for that pair**, plus the meters that pair must report. Pick at most one objective; the rest are kill-controls or disconfirmers on the same checkpoints.

**Copied A (complete, not a garnish).** After absorb, Prosperity product-sparsity, GustavSNN NRV/CPTB, FireFly-S bitmap AND, LoAS sparse format are legal inference engines. Copy the named prior whole as the skip accounting. Training is not a new skip circuit.

**Non-goals.** Continuous unabsorbable \(\theta_g\); HBG-RP int8 payload; CIM; 18 modules; OpenROAD-as-PPA; FPS products; extra integer-consumer tables (freeze already has \(758777\to 714889\), \(-5.78\%\)); treating \(+0.013178\) as a contribution.

---

## Q06-I1 — Lifting-mix recovery that rewrites who crosses \(\theta\)

- **id:** Q06-I1
- **one sentence:** One paired train from the lifting student pulls AEE back under \(+0.005\) while the pre-threshold T10 mix is the only extra degree of freedom that is allowed to change post-absorb binary support.
- **A:** Prosperity product-sparsity density, Gustav NRV empty-row rate, FireFly-S bitmap-AND skip — copied whole as meters and as the inference skip model on \(S\times W'\) after absorb.
- **B:** Lifting CSE lives in the noncausal T10 mix **before** threshold (source: ordinary 260 add/sub vs lifting 159 + 35 intermediate RNE/sat). It edits continuous \(m\), hence \(S=H(m-\theta)\), hence binary GeMM support. The AEE hole is DSEC valid825 optical flow, not a missing MAC-skip mux.
- **X:** Prosperity / Gustav / FireFly consume an already-binary spike map. They do not own a lifting predict/update graph, PSN mix, or a flow-field loss. A reskin would be “add \(\lambda\|S\|_0\) on any SNN and run Prosperity.” Here the trained object is the **mix that emits \(S\)**, under a relative-AEE gate that raw lifting already fails.
- **controls:** (i) ordinary freeze, AEE \(1.219801338\); (ii) lifting raw, AEE \(1.232979368\), no extra loss; (iii) same lifting graph with AEE recovery but **no** support term (rate / NRV / product meters still logged). Exactly one trained checkpoint besides (i).
- **kill-gate:** Trained AEE \(>1.224801338\) or \(>1.259\); or all three binary-support meters match lifting-raw within sampling noise; or \(\theta\) is left unabsorbed / spikes are stored as analog amplitudes.
- **two-sentence pitch:** Raw lifting already buys CSE and already breaks the relative AEE gate by \(+0.013178\); one recovery train is the only legal way to keep that graph. After absorb the datapath is the copied binary skip engine, so the letter’s increment is a mix that emits hardware-shaped \(S\) without changing neuron identity.
- **objection:** “You just fine-tuned lifting until AEE came back; sparsity engines do the rest.” If support meters do not move relative to control (iii), the objection wins and I1 dies.
- **assumptions:** T10 mix coefficients (or equivalent lifting predict/update) remain trainable; \(\theta\) stays layer-shared and absorbed; projection BN and residual add stay in the graph; student is the DSEC flow net (motion C12 / H67 / ep34), not a classifier.
- **predictions:** The recovered student beats lifting-raw on AEE by \(\ge 0.008\) absolute (enough to cross \(+0.005\)) and moves at least one of {product-sparsity density, NRV empty-row rate} by a margin larger than the ordinary-vs-lifting-raw gap on the same meter. Mean spike rate change alone does not explain the NRV/product move (see I2 rate-matched control on the same pair).
- **disconfirmers:** Recovery needs analog spike amplitudes; or AEE recovers only when lifting CSE is disabled (mix identity lost); or Prosperity’s independent-rate product formula \(\rho_S\rho_{W'}\) already accounts for every skip the trained \(S\) produces.

---

## Q06-I2 — Rate-matched NRV row-silence, not salt-and-pepper

- **id:** Q06-I2
- **one sentence:** At matched mean spike rate, train post-absorb \(S\) into all-zero spatial or channel rows that Gustav NRV can drop, instead of lowering unstructured fire probability.
- **A:** GustavSNN NRV/CPTB copied whole (empty-row skip + temporal pack). Prosperity product-sparsity is logged as a foil, not the objective.
- **B:** After absorb the spike path is binary select-add. Empty rows are a packing of that binary tensor. Optical-flow support on DSEC \(96\times 120\times 160\) (captured projection BN batch \(10\times 96\times 120\times 160\)) can concentrate on edge-like rows; that is a task-shaped hole in row occupancy, not a new neuron.
- **X:** Gustav assumes NRV-friendly rows exist at inference. It does not train a flow student so that a pre-threshold mix and a residual chain **emit** those rows. Equal-rate unstructured sparsity is exactly the reskin; beating it is the X test.
- **controls:** Same checkpoint pair as I1. On both ordinary and trained \(S\): (i) true NRV empty-row rate; (ii) rate-matched Bernoulli rows with the same mean \(\rho_S\); (iii) rate-matched spatial shuffle of \(S\) that preserves \(\rho_S\) but destroys row all-zeros.
- **kill-gate:** AEE gates fail; or trained NRV rate \(\le\) shuffled control; or the NRV gain vanishes after matching \(\rho_S\) to ordinary (pure activity pruning).
- **two-sentence pitch:** Same number of ones, more skippable rows — that is the only training claim Gustav does not already make. The letter still copies NRV/CPTB as the engine and spends the one recovery budget on AEE plus row occupancy.
- **objection:** “Row-sparsity is just spike-rate on a coarser axis.” The shuffle control answers: if occupancy is unstructured, NRV skip equals the Bernoulli baseline and the idea is a reskin.
- **assumptions:** Gustav’s row definition (neuron-row / spatial-row / packed channel group) can be bound to this student’s feature layout without a new datapath; patch r1 residual remains a separate continuous tensor and is not secretly binarized to fake empty spike rows.
- **predictions:** At \(\Delta\rho_S\approx 0\), empty-row rate rises enough that a copied NRV cycle model moves more than Prosperity’s product-sparsity model on the same \(S\). Shuffled \(S\) loses that gap. AEE stays inside gates.
- **disconfirmers:** Empty rows appear only on dead channels that a frozen ep34 already has; or forcing row-silence collapses flow on those scanlines and AEE exceeds \(+0.005\); or CPTB (temporal) not NRV (row) is what actually moves — then the idea is I6, not I2.

---

## Q06-I3 — Dual-consumer product sparsity: starve binary GeMM, keep residual/PED

- **id:** Q06-I3
- **one sentence:** Train consumer A (post-absorb binary gate) toward high product sparsity while consumer B (continuous residual / PED / I24) keeps the flow amplitude, using the historically heavy patch-r1 residual chain as the budget that is allowed to stay dense.
- **A:** Prosperity product sparsity copied whole on the spike GeMM only (skip iff \(s=0\) or \(w'=0\)). Do not pretend Prosperity skips residual adds.
- **B:** Freeze splits two objects: binary select-add after absorb vs residual/PED/I24 continuous tensor. Patch r1 residual chain is an old proxy \(\sim 35\%\) of activity-weighted dots (proxy \(\ne\) new cycle share). Training that moves information across that split changes **binary support** without giving AT-LIF an analog payload.
- **X:** Prosperity and FireFly have one sparse binary operand stream. They have no second consumer that is a full-width residual add with captured BN. “Lower \(\rho_S\), run Prosperity” is the reskin; I3 requires AEE to survive **because** B stayed dense, not because \(S\) stayed informative.
- **controls:** (i) spike-rate penalty without residual path (illegal identity if it forces analog \(o\); legal only if it only edits \(S\)); (ii) residual-off / residual-frozen; (iii) product-sparsity of \(S\odot W'\) vs independent \(\rho_S\rho_{W'}\). Same single paired recovery.
- **kill-gate:** AEE gates fail when \(\rho(S\odot W')\) drops; or product density equals \(\rho_S\rho_{W'}\) (no extra alignment, pure rate); or residual/PED is rewritten as “analog AT-LIF shared by two MACs.”
- **two-sentence pitch:** After absorb the spike MAC is a binary skip problem those engines already solve; the letter’s hole is a second tensor that they do not see. Train the split so Prosperity’s product-sparsity meter moves on A while optical-flow AEE is carried on B.
- **objection:** “You hid the energy in the residual; net service will not move.” Valid if residual-add cycles dominate. I3 only lives if the copied spike-GeMM skip is a large enough share that same-port/backpressure can still clear \(15\%\) **without** double-counting the freeze’s always-ready \(6938\to 5354\) (\(-22.83\%\), part generic fusion). Long backpressure both \(8088\) is a warning: sparsity in \(S\) may not touch that number.
- **assumptions:** Dual consumers already exist in the student (projection conv + BN + residual add + AT-LIF). Last-use of the two tensors can be named (see I7). Two-window integer gates/I24/PED q24 stay 0-diff vs model; training must not break that bit-match on the continuous path.
- **predictions:** \(\rho(S\odot W') < \rho_S\rho_{W'}\) (spikes systematically miss nonzero weights, or vice versa). Residual/PED energy does not fall in proportion. AEE inside gates. Independent-rate Prosperity accounting under-predicts the trained skip.
- **disconfirmers:** Product-sparsity gain is fully explained by \(\rho_S\downarrow\) at fixed \(W'\); or residual-off matches I3 AEE (B was not carrying the field); or I24/PED q24 diffs appear (continuous path left the freeze).

---

## Q06-I4 — Projection-BN stats as the support knob

- **id:** Q06-I4
- **one sentence:** Recalibrate or constrain the captured full-frame projection BN (actual batch stats over \(10\times 96\times 120\times 160\)) so \(m-\theta\) becomes row-structured or more often negative, changing \(S=H(m-\theta)\) after absorb without touching AT-LIF identity.
- **A:** After BN+threshold+absorb, log Prosperity product density, Gustav NRV, FireFly bitmap AND on the resulting \(S\) — copy those engines as accounting, not as BN hardware.
- **B:** Full-frame BN is continuous arithmetic **before** threshold. It is in the freeze allow-list and is not a GeMM skip. Captured stats are a measured object, not a new module.
- **X:** Prosperity / Gustav / FireFly / LoAS do not own a projection BN with captured NCHW batch stats on a flow pyramid. If a BN-policy change (train vs freeze-to-eval-stats vs scale-clip) is enough to move binary support, that policy is X relative to those papers even when the skip engine is copied.
- **controls:** (i) freeze BN to the captured eval stats; (ii) train BN scale/bias only (all other weights frozen from ep34) — this is still the one paired recovery if it is the chosen objective; (iii) shuffle BN channels. Do not also train a spike-rate loss in the same pair.
- **kill-gate:** AEE gates fail; or \(S\) meters match frozen-BN; or BN is folded into \(W\) in a way that breaks \(\theta\)-absorb identity (double-absorb / analog fold of \(\theta_g\)).
- **two-sentence pitch:** The cheapest legal way to reshape who crosses \(\theta\) is the BN the net already runs at full frame. The skip papers start after \(S\) exists; this pair starts one operator earlier.
- **objection:** “BN fusion is generic and already in the \(-22.83\%\) same-port number.” Generic fusion is a compile/graph fact. I4 is a **trained** change of BN stats that changes \(S\), then uses copied skip on that new \(S\). If always-ready cycles do not move after BN-stat change, do not cite \(-22.83\%\) as I4’s gain.
- **assumptions:** Projection BN remains an explicit operator at train and at the captured-stat eval; it is not replaced by a per-event analog gain on \(o\). Batch \(10\times 96\times 120\times 160\) remains the stat window.
- **predictions:** Frozen-BN vs trained-BN (or scale-clipped BN) yields a measurable NRV and/or product-sparsity delta at small \(\Delta\rho_S\). AEE stays inside gates. Channel-shuffled BN loses the NRV delta (structure, not just shift).
- **disconfirmers:** Any BN change that recovers AEE also **destroys** lifting CSE (graph rewritten); or BN change only rescales \(\theta\) in disguise (identity break); or support meters move solely because mean activations clip to all-zero maps that fail flow.

---

## Q06-I5 — Dual-side on absorbed \(W'\); spike side AEE-only

- **id:** Q06-I5
- **one sentence:** Structure or prune weights **after** the absorb rewrite \(W'=\theta W\), copy FireFly-S bitmap AND as the dual-side engine, and spend the one training pair’s spike-side budget only on AEE (no extra \(\|S\|_0\)).
- **A:** FireFly-S dual-side bitmap AND copied whole. LoAS compressed sparse format may be the alternate complete A if the letter copies LoAS instead of FireFly — not both half-copied.
- **B:** Layer-shared \(\theta\neq 0\) does not change \(\mathrm{supp}(W)\), but pruning thresholds on \(|W'|\) are absorb-defined. The hole is: a flow student with a pre-threshold T10 mix and a residual path, under AEE gates, with dual-side skip applied only to the binary GeMM.
- **X:** FireFly demonstrates skip **given** bitmaps. It does not train DSEC AT-LIF, does not absorb \(\theta\) into \(W\), and does not leave a continuous residual consumer. The non-reskin test is weight-side structure on \(W'\) plus an **unregularized** spike generator; if the skip equals weight-prune-only \(\times\) frozen \(\rho_S\), X is empty.
- **controls:** (i) ep34 \(W\) absorbed, no prune; (ii) prune/structure \(W'\) with frozen \(S\) (no train); (iii) the one paired recovery = AEE train with frozen structured \(W'\) (spike generator may move \(S\) under AEE only).
- **kill-gate:** Dual-side AND skip on (iii) equals (ii) within noise (training did not change spike-side support); or AEE gates fail; or prune is defined on unabsorbed \(W\) while reporting \(S\times W'\) (identity bookkeeping error); or any int8 spike payload is reintroduced.
- **two-sentence pitch:** Dual-side sparsity is already FireFly’s complete prior; the letter only gets to use it because absorb made the spike path binary. The training increment is AEE-feasible \(S\) on top of absorb-defined \(W'\), not a new AND gate.
- **objection:** “This is magnitude pruning plus FireFly.” Yes unless (iii) moves spike bitmaps relative to (ii) **and** that move is forced by the flow loss interacting with T10/BN/residual, not by an \(\|S\|_0\) term.
- **assumptions:** \(\theta\) absorbed once, then prune/structure \(W'\); FireFly tile/bitmap geometry can be bound to the student’s channels without a new PE. Residual/PED not pruned as if they were spike bitmaps.
- **predictions:** Frozen-\(S\) prune (ii) already raises AND-skip vs (i). The paired recovery (iii) either (a) restores AEE under gates without giving back that skip, or (b) fails AEE — both are publishable measurements. A third outcome, (c) AEE-only training **increases** overlap of \(S\) with \(\mathrm{supp}(W')\) and **hurts** dual-side skip, is the interesting negative.
- **disconfirmers:** Need spike-rate loss to keep AEE (collapses to activity pruning reskin); or \(W'\) structure that FireFly needs destroys residual-add accuracy via shared projection weights; or LoAS/FireFly formats disagree on the same \(S,W'\) so hard that a 5-page letter cannot copy either prior whole.

---

## Q06-I6 — Motion-C12 temporal block silence for CPTB / FireFly-S

- **id:** Q06-I6
- **one sentence:** Use optical-flow temporal smoothness on motion C12 / H67 / ep34 to train consecutive all-zero timesteps in post-absorb \(S\), matching Gustav CPTB packs and FireFly-S temporal bitmaps rather than spatial NRV rows.
- **A:** Gustav CPTB + FireFly-S temporal bitmap AND, copied whole. Spatial NRV is a foil meter (I2).
- **B:** Binary support after absorb is a space–time \(0/1\) volume. Flow wants slowly changing fields; event cameras are already bursty. The hole is **task-structured temporal occupancy** of \(S\), produced after a noncausal T10 mix that itself spans time **before** threshold.
- **X:** Classification SNN accelerators treat extra spikes as evidence of class presence. They do not train temporal block-sparsity so that a 2D flow field stays accurate. A \(\lambda\|S\|_0\) reskin has no preference for consecutive zeros; CPTB does.
- **controls:** Same pair. (i) run-length of zeros per channel vs (ii) i.i.d. rate-matched spikes; (iii) time-shuffle of \(S\) that preserves per-frame \(\rho_S\). Ordinary vs trained on C12 specifically, not a global average that hides H67.
- **kill-gate:** AEE gates fail; or trained zero run-length \(\le\) time-shuffle; or T10 mix is made causal to fake temporal blocks (breaks the freeze’s noncausal mix).
- **two-sentence pitch:** CPTB and FireFly already skip temporal empty bitmaps; they never trained those bitmaps from a flow loss sitting behind a noncausal mix. One recovery pair either produces longer zero-runs at matched rate or the idea is dead.
- **objection:** “Events are already sparse in time.” Then ordinary ep34 already saturates CPTB, the trained pair cannot move the meter, and I6 dies on the ordinary-vs-trained comparison.
- **assumptions:** C12 motion channels are a real named tensor in the student; H67 / ep34 identify the block/checkpoint, not extra modules. Noncausal T10 mix stays pre-threshold. Temporal pack geometry matches the copied prior (do not invent a third pack).
- **predictions:** Trained C12 zero run-length exceeds time-shuffled and ordinary at matched \(\rho_S\). CPTB skip moves more than spatial NRV. AEE inside gates. Noncausal mix still present (CSE counts remain lifting or ordinary, not a new graph).
- **disconfirmers:** Zero-runs appear only because the mix averaged adjacent times into a constant \(m<\theta\) (then I1/I4 already own it); or temporal silence on C12 is compensated by chatter on H67 so net CPTB is flat; or long-backpressure \(8088\) is temporal and unchanged, so service never sees the block silence.

---

## Q06-I7 — Dual-path last-use licenses binary row-empty

- **id:** Q06-I7
- **one sentence:** Train a consume-then-silence policy: after the continuous residual’s last-use (the add), the post-absorb binary support on that patch is allowed to go NRV-empty because consumer B already carried the value.
- **A:** Gustav NRV (and Prosperity product sparsity as foil) on the spike GeMM after absorb, copied whole. Last-use itself is not claimed as their invention.
- **B:** Freeze allow-list: dual-path last-use. Consumer A is the binary gate after absorb; consumer B is the residual continuous path. Last-use is a lifetime fact of two tensors, not “analog AT-LIF amplitude reused by two MACs.”
- **X:** Prosperity / Gustav / FireFly model one spike stream’s zeros. They do not model a residual tensor’s last-use **licensing** subsequent spike silence. That coupling is the training object and the hardware object (port-free after last-use), still inside same-port/state/backpressure accounting.
- **controls:** (i) last-use annotated but no silence loss; (ii) silence loss without last-use (illegal if it zeros \(S\) while residual still needs a later read — would break 0-diff I24/PED); (iii) the one pair: silence only on rows whose residual last-use has already fired in the schedule.
- **kill-gate:** AEE gates fail; or silence appears before residual last-use and I24/PED q24 leaves 0-diff; or same-port/backpressure net service cannot be argued \(\ge 15\%\) even after counting freed last-use (and without eating the generic-fusion \(-22.83\%\)); or long backpressure stays \(8088\) on both and last-use never was the stall.
- **two-sentence pitch:** Empty binary rows are legal once the residual add has consumed the continuous tensor; training that alignment is not a reskin of product sparsity. The copied NRV engine then skips a silence that last-use already made semantically cheap.
- **objection:** “Last-use is compiler 101; training does not belong.” If ep34 already silences \(S\) after residual last-use, the compiler observation is enough and training I7 is unnecessary — kill on control (i).
- **assumptions:** A schedule can name residual last-use vs spike last-use on patch r1 without adding modules. Two-window integer gates/I24/PED q24 remain the bit-true continuous path. Spike path remains \(0/1\) after absorb.
- **predictions:** On patch r1, NRV empty-row rate **after** residual last-use rises; **before** last-use it does not. AEE inside gates. A copied NRV model attributes skip to post-last-use rows. Backpressure moves only if those rows were on the stalled port — otherwise I7 is a correctness idea, not a service idea, and should not claim the \(15\%\) gate.
- **disconfirmers:** Residual last-use and spike last-use are the same cycle (no license window); or training silence before the add is the only AEE-feasible point (identity/quality break); or patch r1’s \(35\%\) proxy is unrelated to the new cycle share so the letter cannot spend pages on it.

---

## Q06-I8 — Pre-threshold margin / flicker penalty; \(\theta\) still absorbed

- **id:** Q06-I8
- **one sentence:** Penalize threshold-margin violations and \(S[t]\oplus S[t-1]\) chatter on the pre-threshold field \(m\) so post-absorb \(S\) is crisper 0/1 with longer zero runs, without ever training a continuous unabsorbable \(\theta_g\).
- **A:** Gustav NRV/CPTB packing of the resulting chatter-poor \(S\); FireFly temporal bitmaps as the dual-side foil. Copy complete.
- **B:** Chatter is produced by T10 mix + BN + residual add **before** \(H(m-\theta)\). The freeze forbids selling analog \(\theta_g\). The legal object is \(m\)’s distance to \(\theta\) and the stability of the already-binary \(S\).
- **X:** Activity pruning minimizes spike count. Prosperity/FireFly do not train margins. I8 holds \(\rho_S\) (rate-matched) and minimizes flicker / \(|m-\theta|<\varepsilon\) mass. That is a different support shape: fewer isolated ones, more packable zeros. Optical-flow AEE cares about *when* a fire happens, not only how many.
- **controls:** Rate-matched I2/I6 shuffles; margin loss vs flicker loss ablated **inside the same pair’s logging**, but only one of them is the trained objective (no second train). Ordinary ep34 chatter as baseline.
- **kill-gate:** Any writeup that calls the margin “analog spike amplitude” or refuses \(\theta\)-absorb; AEE gates fail; or flicker drops only because \(\rho_S\to 0\).
- **two-sentence pitch:** Packing engines hate isolated ones; flow training can push \(m\) off \(\theta\) so \(S\) is blocky 0/1 while \(\theta\) still disappears into \(W\). That is still binary GeMM, with a support prior those engines never trained.
- **objection:** “Surrogate-gradient SNNs already use margins.” Then the letter must show the **hardware meter** (NRV/CPTB/FireFly AND) moving at matched rate on this DSEC student, not a classification accuracy table.
- **assumptions:** Surrogate around \(H(m-\theta)\) already exists; adding a margin/flicker term does not change neuron identity. One paired recovery includes this term **xor** I1’s mix-AEE term, not both (budget).
- **predictions:** At matched \(\rho_S\), Hamming flicker \(\sum_t S[t]\oplus S[t-1]\) falls, zero run-length rises, NRV/CPTB skip rises, product-sparsity vs independent-rate barely moves (I8 is not I3). AEE inside gates.
- **disconfirmers:** Margin term is equivalent to raising \(\theta\) and then not absorbing it; or flicker falls but packing skip does not (isolated ones were never the packing bottleneck); or AEE recovery from \(+0.013178\) requires the margin term **and** a second loss, violating the one-pair budget.

---

## Shared meters for the one pair

Report on ordinary freeze vs the single trained student, after \(\theta\)-absorb, on the spike path only:

1. AEE on DSEC valid825 (gates: abs \(\le 1.259\), relative \(\le +0.005\) vs \(1.219801338\)).
2. Mean \(\rho_S\), \(\rho_{W'}\), product density \(\rho(S\odot W')\), and independent foil \(\rho_S\rho_{W'}\).
3. NRV empty-row rate; rate-matched spatial shuffle foil.
4. Temporal zero run-length / flicker; time-shuffle foil.
5. FireFly-style bitmap-AND skip; frozen-\(S\) weight-structure foil if I5 is **not** the objective.
6. Identity check: \(S\in\{0,1\}\), \(W'=\theta W\), residual/PED/I24 still continuous, two-window q24 still 0-diff.
7. Do **not** add integer-consumer tables. Do **not** spend the \(15\%\) same-port gate on the freeze’s generic-fusion \(-22.83\%\) unless the trained \(S\) actually changes that net.

If meters 2–5 are all explained by \(\Delta\rho_S\) at unstructured support, every Q06 idea is a Prosperity/Gustav/FireFly reskin and the training letter should not be written.
