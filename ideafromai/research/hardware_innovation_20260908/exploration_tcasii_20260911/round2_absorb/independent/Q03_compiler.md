# Q03 — compiler (round 2 independent)

Perspective: **compiler**. Locked identity: AT-LIF \(o=\theta\cdot H(m-\theta)\in\{0,\theta\}\); inference absorbs layer-shared \(\theta\) into the next \(W\); after that the spike path is binary \(\{0,1\}\times W\).

Pass geometry this file is not allowed to blur:

1. Noncausal T10 mix (PSN / lifting CSE) is **continuous arithmetic before** the AT-LIF threshold.
2. Threshold, then absorb, then the next layer is **binary GeMM** — Prosperity / Gustav / FireFly / LoAS’s object. Copy that backend whole. Do not reskin it.
3. Residual / PED / I24 is a **different continuous tensor**. Dual consumers, if any: consumer A = post-absorb binary gate; consumer B = continuous residual path.
4. **Do not title-delete the 35 lifting RNE/sat.** Ordinary and lifting already share long-backpressure **8088**. Mix op-count is empirically not the 8088 knob. Always-ready 6938→5354 (−22.83%) is generic fusion and is not the gated metric.

Gated objects: same-port / state / backpressure net service ≥15% on the 8088 series; AEE abs ≤1.259 and relative ≤+0.005 vs ordinary 1.219801338; two-window integer gates / I24 / PED q24 stays 0-diff vs model. Lifting raw AEE 1.232979368 already fails the relative gate (+0.013178). No CIM, no non-absorbable int8, no “continuous θg” contribution.

Ideas, not a module farm.

---

## Q03-I1 — Absorb is a constant-fold that switches ISA, not a MAC

**One sentence.** A compiler pass folds layer-shared \(\theta\) into next-layer \(W\) (\(W\leftarrow\theta W\)), dead-code-eliminates the \(\theta\)-scale on the spike path, and **switches instruction selection** from scale-and-add to binary select-add; the mix IR on the other side of the threshold is left as continuous add/sub.

**A.** Post-absorb binary GeMM copies Prosperity product sparsity, Gustav NRV/CPTB, FireFly-S Bitmap AND, and LoAS as published. Absorb itself is the locked identity, not a new operator.

**B.** The freeze splits two objects: pre-threshold continuous mix, post-absorb binary GeMM. A single untyped MAC IR issues both through the same port. That is the hole: legalization never built Prosperity’s object, so the port still sees a scaled payload that the identity says cannot exist after absorb.

**X.** Prosperity/Gustav/FireFly assume \(\{0,1\}\) tokens already exist. They do not fold AT-LIF \(\theta\), do not DCE a scale that only lives on the spike path, and do not refuse to select mix add/sub as GeMM. The X is the **typed ISA cut at threshold+absorb**, not a new bitmap.

**Controls.** (i) identity-faithful absorb fold + copied binary backend; (ii) no-absorb scale-and-add on \(\{0,\theta\}\); (iii) absorb fold but mix and GeMM still share one untyped opcode. Report 8088-series cycles, always-ready (non-gated), AEE, q24 0-diff. Same-port/state/backpressure net.

**Kill-gate.** If absorb+ISA-switch does not cut 8088 by ≥15% vs the no-fold same-port baseline, kill as a letter title (keep it as a correctness pass only). If AEE relative >+0.005 or q24 breaks, kill.

**Pitch.** After absorb the next layer is binary GeMM; the compiler’s job is to actually emit that ISA instead of leaving a \(\theta\)-multiply on the issue port. Prosperity then runs on the object it was designed for, which this net did not previously present.

**Objection.** “That is just implementing the identity, not a compiler contribution.” If 8088 is already binary select-add occupancy, this pass is mandatory plumbing and cannot be the title — the letter must then spend pages on I3/I5/I7, not on absorb.

**Assumptions.** Some fraction of 8088 is still scale-and-add or untyped mix+GeMM issue. \(\theta\) is layer-shared, so the fold is a compile-time constant, not a per-spike load.

**Predictions.** After the pass, spike-path codegen contains no \(\theta\)-multiply; residual/PED codegen is unchanged; binary backend counters (skipped zeros, bitmap ANDs) become non-zero. If 8088 was scale residue, it drops; if not, 8088 is flat and I1 is demoted.

**Disconfirmers.** Post-fold 8088 = 8088. Mix opcodes still issued as GeMM. Residual path accidentally DCE’d. AEE moves.

---

## Q03-I2 — Fuse mix→threshold; never fuse mix→GeMM

**One sentence.** Fuse the noncausal T10 mix with Heaviside into one producer kernel that **retires the continuous membrane** and emits binary tokens; treat mix→GeMM fusion as a type-illegal rewrite.

**A.** Token consumer is a full copy of Prosperity/Gustav/FireFly/LoAS. Generic fusion that already moved always-ready 6938→5354 is the control, not the claim.

**B.** Always-ready improved −22.83% while **long backpressure stayed 8088 on both mix flavors**. The freeze says the mix is before threshold and GeMM is after absorb. The hole is fusion of the **wrong edge**: an untyped producer-consumer fuse that keeps the membrane live on the port under backpressure.

**X.** Prosperity fuses inside \(\{0,1\}\times W\). FireFly ANDs bitmaps of spikes. Neither paper has a continuous noncausal T10 mix that must complete before the spike even exists. The X is **fusion legality across the threshold cut**.

**Controls.** Four graphs: mix‖threshold‖GeMM (unfused); mix+threshold | GeMM (legal); mix+GeMM (illegal type fuse); generic fusion that produced 5354/8088. Kill any fuse that changes lifting/ordinary AEE or q24.

**Kill-gate.** Legal mix+threshold fusion must move 8088 ≥15%. If only always-ready moves again, it is the same generic-fusion story already measured — kill the title.

**Pitch.** The mix is not a GeMM tile; it is the last continuous reduction before a binary token is born. Fusing it through the threshold shortens membrane occupancy; fusing it into GeMM is how a Prosperity reskin accidentally gets written.

**Objection.** “Fusion already failed: 8088 is identical for ordinary and lifting.” That kills **mix-op fusion**, not **mix-threshold fusion**. If membrane last-use is already identical to token last-use, I2 dies and last-use split (I3) is the remaining fuse-shaped object.

**Assumptions.** 8088 holds a live pre-threshold membrane on the same port as the next-layer issue. Mix add/sub count is *not* assumed to be the knob (ordinary=lifting=8088).

**Predictions.** Mix+threshold fusion reduces live continuous bytes at the port and cuts 8088; mix+GeMM fusion either breaks AEE/q24 or leaves 8088 unchanged. Always-ready may move without 8088 moving — that is a negative control, not success.

**Disconfirmers.** 8088 unchanged under legal fusion. AEE relative >+0.005. Token stream after fusion differs from threshold(mix(·)).

---

## Q03-I3 — Typed dual last-use: binary gate vs residual/PED/I24

**One sentence.** Run last-use / liveness as two typed live-ranges: consumer A’s post-absorb **binary gate** may die when the next GeMM tile completes; consumer B’s **continuous residual/PED/I24** dies at residual add / two-window q24; do not let B’s last-use extend A’s token occupancy on the same port.

**A.** Consumer A’s GeMM copies Prosperity/Gustav/FireFly/LoAS, including their spike-buffer lifetime tricks, **only on A**. Residual arithmetic is not rewritten into a bitmap.

**B.** Freeze: two objects stay distinct; dual consumers are gate vs residual, not two MACs sharing analog AT-LIF amplitude. Patch r1 residual chain is historically expensive (old proxy ~35% of activity-weighted dots; proxy ≠ cycle share). Same-port 8088 is the hole: one last-use tag on two types.

**X.** Prosperity’s consumers are binary. Dual last-use of a **binary token and a continuous residual** is not product sparsity, not NRV, not Bitmap AND. Those papers do not have PED/I24/q24 as a second live tensor.

**Controls.** (i) unified last-use (today); (ii) typed split, same port; (iii) typed split plus dual queue (I7). Measure residual last-cycle vs token last-cycle vs 8088. Keep q24 0-diff. Do not use the 35% dot-proxy as a cycle claim.

**Kill-gate.** If splitting last-use does not move 8088 ≥15%, kill. If the split changes residual numerics (AEE or q24), kill. If the only win is on always-ready, kill.

**Pitch.** After absorb the spike is a gate, not a value; the value that still needs a register is the residual path. A compiler that keeps one last-use on both is paying residual lifetime on the binary port.

**Objection.** “This is ordinary liveness; TCAS-II will call it software.” The letter has to show the *type* split is forced by AT-LIF absorb (gate vs residual), that unified last-use is exactly the 8088 occupant, and that Prosperity applied to both tensors is a miscompile.

**Assumptions.** Residual/PED/I24 is actually live across the same issue port as post-absorb GeMM. The 35% proxy is not assumed to equal 15% of 8088.

**Predictions.** Token buffer occupancy ends earlier than residual occupancy after the split; 8088 tracks the **max**, not the sum, unless the port is strictly serialized (then I7). q24 remains 0-diff because B’s values are untouched.

**Disconfirmers.** Residual already dies before GeMM issue (split is a no-op). 8088 is entirely inside binary GeMM (then copy Prosperity is the title, this is not). AEE moves when last-use tags change — that means a real data dependence was broken, not a lifetime bug.

---

## Q03-I4 — Mix CSE is 8088-invariant; 35 RNE is not a title

**One sentence.** Treat T10 mix CSE (ordinary 260 add/sub vs lifting 159 add/sub + 35 intermediate RNE/sat) as a **pre-threshold rewrite that the freeze already shows does not move 8088**, and forbid any letter title whose delta is “delete 35 RNE.”

**A.** Ordinary vs lifting source CSE as arithmetic controls. After threshold+absorb, still copy the binary GeMM stack. Lifting AEE 1.232979368 is a **failed** accuracy control (relative +0.013178 > +0.005), not a candidate default.

**B.** Same-port: always-ready 6938→5354 from generic fusion; **long backpressure both 8088**. CSE reduced add/sub 260→159 and added 35 RNE, and the gated cycle number did not move. The hole is a compiler that still uses mix op-count or RNE count as the service metric.

**X.** Prosperity/Gustav/FireFly never see T10 lifting CSE; it happens before threshold. Not a reskin: this idea **refuses** to sell a mix-arithmetic trick as if it were sparse GeMM. The X is the empirical 8088-invariance of the only mix CSE we have.

**Controls.** Ordinary mix, lifting mix, and a “RNE packed into unused mix slots without changing results” schedule. All three must quote 8088, AEE, q24. No table that leads with 260→159.

**Kill-gate.** This idea is already a kill-map for mix-op titles. It becomes a letter title only if a **different** mix rewrite (live-range of the membrane, not op-count) moves 8088 ≥15% **and** stays within AEE relative +0.005 — which lifting currently does not. If the only mix rewrite that shortens live-range is lifting, the mix pass is dead for the letter.

**Pitch.** The compiler already ran the obvious CSE; backpressure did not notice, and lifting already misses the relative AEE gate. Pages spent on 35 RNE cannot be the TCAS-II story when 8088 is the gated object.

**Objection.** “Then why mention mix at all?” Because mix is still the **producer barrier** for tokens (I2/I6). Mentioning it as an arithmetic saving is the error, not mentioning it as a schedule constraint.

**Assumptions.** The 8088 equality of ordinary and lifting is not a measurement bug. RNE/sat are legalization of lifting, not extra memory traffic that a better encoding would remove from the port.

**Predictions.** Any pass whose sole IR delta is −35 RNE or 260→159 leaves 8088 = 8088 and, if it is lifting, fails relative AEE. A mix pass that only changes membrane last-use (without those op deltas) is a different idea and must be scored as I2, not as CSE.

**Disconfirmers.** A re-measure where lifting 8088 ≠ ordinary 8088 (then mix op-count *is* a knob and this kill-map is wrong). An ordinary-mix schedule that deletes RNE-like saturations and *does* move 8088 without AEE loss — still do not call it “delete 35 RNE”; call it whatever actually occupied the port.

---

## Q03-I5 — Pass order: BN-fold, θ-absorb, residual-add are three legalizations

**One sentence.** Fix a legalization order: fold projection BN (actual batch stats over 10×96×120×160) into projection \(W\) where residual-safe; absorb \(\theta\) into **next-layer** \(W\) on the spike path only; emit residual add as a continuous kernel that neither fold may cross.

**A.** Textbook BN folding + copied post-absorb binary GeMM (Prosperity/Gustav/FireFly/LoAS). BN stats are captured, not learned-fake.

**B.** Native graph has projection conv + BN + residual add beside AT-LIF. Full-width BN is listed as an object those SNN GeMM papers do not cover. The hole is one “absorb scales into W” pass that either skips BN traffic (8088 stays) or illegally folds through residual (AEE/q24 die).

**X.** Prosperity’s \(W\) is the binary GeMM weight, not a projection-BN-residual bundle. FireFly does not legalize full-width BN over 10×96×120×160. The X is **which scale is foldable on which tensor**, forced by absorb identity plus a residual that is not AT-LIF output.

**Controls.** (i) no folds; (ii) BN-fold only; (iii) \(\theta\)-absorb only; (iv) both folds with residual barrier; (v) illegal BN-fold through residual. Cycle 8088-series, BN full-width traffic, AEE, q24 0-diff.

**Kill-gate.** Legal order must move 8088 ≥15% **or** this is not a title (BN-fold may still be required correctness). Illegal order must be shown to break AEE or q24. If BN-fold does not change port traffic, BN is not the occupant.

**Pitch.** Three scales sit in one student: BN on projection, \(\theta\) on AT-LIF, nothing absorbable on residual add. A compiler that has one fold pass will either leave full-width BN on the port or smash the residual path that the identity says is a different tensor.

**Objection.** “BN folding is 1990s compiler 101.” Yes; the letter’s novelty is not the fold, it is that **θ-absorb must not be written as the same fold**, and that full-width BN plus residual are why copying Prosperity onto “the weights” is a miscompile.

**Assumptions.** Projection BN is still executed full-width at inference with those batch stats (not already folded). Residual add is after BN and cannot be commuted into next-layer binary GeMM.

**Predictions.** BN-fold reduces full-width scale/shift traffic; \(\theta\)-absorb does not touch projection BN; residual checksum (q24) is invariant only when the residual barrier is present. 8088 moves iff BN or the unfolded scale stream was on the critical port.

**Disconfirmers.** BN already folded in the captured graph. 8088 independent of BN-fold and of \(\theta\)-absorb (points back to last-use / dual-queue). AEE moves under the legal order.

---

## Q03-I6 — Schedule the noncausal T10 window; do not emit it as a causal GeMM timestep

**One sentence.** Model T10 mix as a **noncausal window reduction** with loop-carried reads of past and future taps, software-pipeline it against optical-flow structure (motion C12 / H67 / ep34, patch r1), then threshold **once per window**; never lower the mix to Prosperity’s per-timestep spike GeMM.

**A.** After the window fires, binary GeMM is a full Prosperity/Gustav/FireFly/LoAS copy. Causal SNN timestep schedules are negative controls.

**B.** Freeze: noncausal T10 mix happens **before** threshold. Event-camera 2D flow on DSEC valid825 is the student. The hole is a compiler that schedules the mix as another GeMM timestep, holding window state on the same port until a causal firing rule that this mix does not have.

**X.** Prosperity/FireFly/Gustav consume spikes in time-major tiles after neurons have fired. They do not own a pre-threshold noncausal PSN/lifting mix, and they do not own epipolar/motion reuse of this OF student. The X is **window scheduling of the producer**, not sparse GeMM of the consumer.

**Controls.** Causal-as-if-SNN schedule vs window-complete-then-threshold vs window pipelined with C12/H67/ep34 neighbor reuse. Do not change mix CSE flavor (I4). AEE gates as usual; lifting still illegal as default.

**Kill-gate.** Window schedule must move 8088 ≥15% vs the causal-as-if-SNN lowering. If 8088 is downstream GeMM-only, I6 is a producer constraint for I1/I2, not a title. AEE relative >+0.005 kills.

**Pitch.** Tokens cannot legally exist until the noncausal mix window is closed; issuing GeMM as if they did is a schedule bug that looks like backpressure. Optical-flow neighbors are the reuse that can close the window without extra port fills.

**Objection.** “Scheduling a stencil is not TCAS-II.” It is only a letter if 8088 is window-state occupancy and the OF reuse is the thing that releases it — not if this is a software-pipeline essay with a static 8088.

**Assumptions.** T10 mix really is noncausal (future taps in the window). Some 8088 cycles are mix-window state or repeated neighbor fills, not binary GeMM math. C12/H67/ep34 actually share mix taps spatially.

**Predictions.** Token-ready time aligns to window completion, not to a fake timestep. Neighbor reuse cuts mix-state fills. 8088 drops only if those fills were on the port; binary GeMM counters stay in the Prosperity copy.

**Disconfirmers.** Mix is already scheduled as a closed window and 8088 is unchanged. OF motion structure does not alias mix taps. Any window reorder breaks AEE (illegal CSE of a noncausal read).

---

## Q03-I7 — Dual-queue issue: residual/PED continuous vs binary GeMM

**One sentence.** Give the same-port machine two typed issue queues: queue A feeds post-absorb binary GeMM (Prosperity’s object); queue B feeds residual add / PED / I24 continuous ops; backpressure is per-queue, not a single 8088 covering both.

**A.** Queue A’s backend is a complete Prosperity/Gustav/FireFly/LoAS copy (product sparsity, NRV/CPTB, Bitmap AND). Queue B is ordinary integer add/shift for residual/PED/q24, not a second sparse GeMM.

**B.** Long backpressure **both 8088** under a same-port model, while a separate integer consumer model only moved 758777→714889 (−5.78%) and must not grow tables. The hole: one port serializes a binary selector and a continuous residual chain that the identity forbids merging.

**X.** Not “two sparse GEMMs.” Not a FireFly reskin with an extra bitmap. The second queue exists because residual/PED/I24 is **not** \(\{0,1\}\times W\). Optical-flow r1 residual is task structure those accelerators do not schedule.

**Controls.** Same-port unified queue (8088 baseline); dual queue, shared datapath; dual queue, split datapath (report area honestly, no OpenROAD-as-PPA, no FPS product). Last-use split (I3) on vs off. q24 0-diff required. Do not quote −5.78% as the gated win.

**Kill-gate.** Dual-queue must move the **same-port/state/backpressure** 8088 series by ≥15% (target ≲6875). If the win appears only in the separate-integer-consumer model, kill (already −5.78%, below gate, do not add tables). If queue B is secretly a binary GeMM, kill as Prosperity reskin.

**Pitch.** Absorb made the spike path a binary selector; it did not make the residual path binary. One issue queue is how a letter accidentally claims sparse GeMM while the port is waiting on PED.

**Objection.** “Dual issue is architecture, not compiler.” The compiler artifact is **typed enqueue**: which IR node is legal in A vs B. The hardware can stay one datapath; illegally enqueueing residual into A is the miscompile.

**Assumptions.** 8088 is serialization of A and B, not A alone. Residual/PED ops are numerous enough under backpressure to matter; the old 35% dot-proxy is not the proof.

**Predictions.** Per-queue stall counters sum to ~8088 today and diverge after the split; A’s counter should then respond to Prosperity (copied), B’s should not. q24 stays 0-diff. AEE unchanged.

**Disconfirmers.** Queue B occupancy ≈ 0 under backpressure (residual is free; copy Prosperity on A and stop). Dual-queue 8088 ≥ 0.85×8088. Enqueue rules change outputs.

---

## Q03-I8 — Split allocation: membrane buffer vs token bitmap

**One sentence.** Allocate two buffers at the threshold cut — a wide continuous **membrane/mix** buffer that dies at Heaviside, and a compact **token** buffer that is the only legal input to next-layer GeMM — and copy FireFly/Gustav compression **only** onto the token buffer.

**A.** Token buffer: FireFly-S Bitmap AND, Gustav NRV/CPTB, Prosperity’s sparse fetch, LoAS as published. Membrane buffer: dense (or modestly quantized) continuous storage; no bitmap of analog mix.

**B.** Same-port 8088 with unified storage is the hole: the allocator keeps a membrane-shaped object live as if it were spikes. Freeze forbids calling that object analog AT-LIF amplitude; after threshold it must be 0/1 tokens.

**X.** Gustav/FireFly compression is a complete prior **after** spikes exist. They do not allocate a pre-threshold T10 membrane, and they do not have a second residual tensor competing for the same port (I3). The X is **type-changing allocation at absorb**, not a new sparse format.

**Controls.** Unified buffer; split buffers, same port; split buffers + last-use split (I3). Token-buffer bits vs membrane-buffer bits vs 8088. q24 on PED/I24 must not be stored in the token bitmap.

**Kill-gate.** Split alloc must move 8088 ≥15%. If the only delta is capacity while backpressure cycles stay 8088, kill as a memory-diagram paper. Putting residual/PED into the token bitmap kills the idea as an identity violation.

**Pitch.** Prosperity’s object is a token buffer; this net’s producer is a membrane. One allocation makes a sparse-GeMM title that still ships analog occupancy through the port.

**Objection.** “Double buffering is obvious.” It is obvious *after* the identity lock; the letter has to show unified occupancy **is** 8088, and that copying bitmap compression onto the membrane is the reskin to avoid.

**Assumptions.** Unified allocation currently sizes the live object to the membrane (or to a {0,θ} payload) through GeMM. Port backpressure tracks that occupancy, not mix add/sub (I4).

**Predictions.** Membrane last-use ends at threshold; token occupancy is 1 bit/site (or bitmap words) and is what GeMM fetches; 8088 falls with occupancy. Residual/PED remains a third storage class, not a bitmap.

**Disconfirmers.** 8088 independent of buffer type (compute-bound GeMM: then A on I1’s backend is the title). Token buffer bitmaps change AEE (threshold not actually binary after absorb). Membrane compressed with FireFly AND “for density” — that is the reskin this idea exists to forbid.

---

## Compiler stance (not extra ideas)

- **Copy** Prosperity / Gustav / FireFly / LoAS on the post-absorb binary GeMM. That is A everywhere. X is never “our bitmap is slightly different.”
- **Do not** try to win 8088 by mix add/sub or by deleting 35 RNE. Ordinary and lifting already tied at 8088; lifting already fails relative AEE.
- **Do** aim the compiler at objects the freeze says those papers do not cover: T10 mix **before** threshold (as a producer barrier / window, not as CSE savings), residual/PED/I24 continuous path, full-width BN, typed dual last-use, optical-flow task structure.
- **Legalize** \(\theta\) into next \(W\), then stop talking about analog spike amplitude.
- **Gate** on 8088-series same-port/state/backpressure ≥15% and AEE abs/relative; always-ready 5354 and integer-consumer −5.78% are not substitutes; do not add those tables.
- If I1’s ISA switch and I8’s token buffer make 8088 move **only** because the binary backend finally matches Prosperity, the letter title is that legalization plus a copied prior — still not a mix-RNE title, and still not enough unless X (I3/I5/I6/I7) is why Prosperity alone does not already cover the port.
