# P02 independent ideas — HPCA/ISCA/MICRO architecture

Status: **ideas, not findings**. Generated 2026-09-11 from `SCOPE.md` + `PROBLEM.md` only.

Perspective: dataflow (inner / Gustavson / outer-product), tick-batch occupancy, next-reuse vectors (NRV), 1RW SRAM tax, dual-consumer last-use, compiler calendar vs hardware scoreboard.

Frozen facts used as B, not as novelty: Motion C12 / H67 / ep34; ATLIF **continuous** θg; expensive patch r1 residual + T10 PSN; dual consumers (gate and continuous PED); native projection conv+BN+add; noncausal T10; ordinary AEE 1.219801338 vs lifting AEE 1.232979368; source add/sub 260 vs 159+35 RNE/sat; always-ready 6938→5354 with **identical** long backpressure 8088; integer consumer −5.78% at a **different** resource point; projection BN uses full-domain 10×96×120×160 batch stats; integer gates/I24/PED q24 match on two windows.

Shared numeric gates (every idea that claims a chain win must pass these; arithmetic-preserving ideas must additionally keep AEE):

- valid825 AEE absolute ≤ 1.259; relative vs ordinary dense-source ≤ +0.005.
- Complete-chain **same-port / same-state / same-backpressure model** net completed service ≥ 15%.
- Long-backpressure must **move below 8088**; always-ready slot cuts that leave 8088 unchanged are not a win.
- Do not add the SIMD-source table to the integer-consumer table.
- Subtract generic round→sat fusion from any lifting credit.

---

## P02-I1 — One-grant spatial fork for dual-consumer last-use

**one-sentence:** Force the post-source gate path and the continuous PED path to share a single 1RW SRAM grant in the same tick-batch window, converting a temporal double-read into a spatial multicast.

**A (complete prior to copy):** Copy the whole data-orchestration stack, not the slogan: (i) Buffets (MICRO 2019) credit/fill/read on **1RW** scratchpads; (ii) Eyeriss spatial multicast + row-stationary delivery; (iii) latency-insensitive ready/valid with fork-join credits (Carloni); (iv) compiler last-use from SSA/VLIW live-range (Chaitin / Multiflow), which already knows a value’s consumer count. The copied artifact is a buffet with **one read pointer**, a multicast bus, and a compiler-emitted last-use epoch — not a 2RW SRAM and not an extra copy buffer.

**B (hole in THIS net):** After T10 PSN the source vector has **two** consumers (spike/gate and continuous residual/PED). A 1RW bank legally yields one read per cycle. Naive RTL either clones the tensor (state tax) or serializes the two consumers (cycle tax). That fork sits downstream of the source ALU, which is the only place the 8088-vs-8088 identity can live: lifting cut always-ready slots 6938→5354 (−22.83%) and **did not** cut long backpressure. Dual-consumer last-use is the first place a source-ALU win can be absorbed.

**X (why not a reskin):** Not “add a second read port” and not “Eyeriss multicast” as a rename. The increment is a **legality proof that consumer co-issue is allowed under continuous θg and exact I24/PED q24**: gate and PED may have different bitwidths and different FU latencies, but they must be scheduled to the **same SRAM grant**. The circuit is a 1R multicast latch plus a 2-credit last-use (refcount 2→1→0). The system claim is completed source vectors per 1RW grant, not add/sub counts.

**controls:** (C1) identical 1RW, identical bank count, identical dual-consumer graph; (C2) a 2RW oracle and a double-buffer (2×1RW) as **illegal** upper bounds, reported separately; (C3) serialized dual-read as the honest 1RW baseline; (C4) round→sat fusion enabled on both arms; (C5) ordinary dense T10 as the AEE-safe source; lifting T10 only as a sensitivity; (C6) integer gates/I24/PED q24 two-window zero-diff must hold.

**kill-gate:** On the same two-stage SIMD resource point used for 6938/5354/8088: dual-consumer **source-bank reads per T10 vector** drop from 2 to 1, **and** long backpressure falls ≥15% below 8088, **and** full-chain completed service ≥15% vs serialized 1RW. If 8088 is unchanged, kill even if always-ready looks better. AEE: two-window integer delta = 0; valid825 ≤ 1.259 and ≤ ordinary+0.005.

**two-sentence pitch:** Dual-consumer last-use, not the T10 adder tree, is what taxes a 1RW source bank in this residual chain; we copy buffet credits and compiler last-use, then legally co-issue gate and PED on one grant. TCAS-II gets one mechanism (spatial fork on 1RW), one causal graph (ALU ready ≠ consumer complete), and a component metric (grants per vector, long backpressure).

**assumptions:** Gate and PED can be co-issued in one tick-batch window without changing ATLIF continuous θg or the frozen student; the 8088 stall is at least partly the serialized second read rather than an unrelated downstream lock.

**predictions:** (P1) A cycle-accurate 1RW service model will show the second consumer’s ready bit waiting on a port, not on arithmetic. (P2) Spatial co-issue moves 8088; cloning the tensor moves area/state but not 8088. (P3) Lifting without co-issue still prints 8088.

**disconfirmers:** If a port-true 1RW model already serves both consumers in one grant (hidden 2R), B is false. If 8088 is entirely inside projection-BN reduction, co-issue cannot move it. If co-issue changes q24/PED vs the two-window oracle, X is illegal.

**objection:** “Just use 2RW SRAM / two banks.” That violates the venue’s same-port rule and hides the tax; the brief is the 1RW legality of the fork, not a memory-compiler upgrade.

---

## P02-I2 — In-place lifting is illegal under dual last-use; holding-bank multicast restores 1RW

**one-sentence:** Classical in-place lifting (the reason 40-coeff T10 looks cheap) is illegally overwritten before the second consumer’s last-use; a single holding bank after the last lifting write restores 1RW without cloning the live range.

**A (complete prior to copy):** Copy Sweldens lifting as an **in-place 1RW algorithm** (predict/update on even/odd, 40 learned coeffs here), plus modulo-scheduled software pipelining with **recurrence- and port-constrained II** (Rau, Lam), plus the Buffet “occupancy = live fills not yet last-used” accounting. The copied claim is: lifting was invented to cut auxiliary storage, not merely to cut add/sub.

**B (hole in THIS net):** Ordinary source is 260 add/sub per T10 vector; lifting is 159 add/sub **plus 35 intermediate RNE/sat**. Those 35 intermediates are extra 1RW writebacks. Dual consumers then need the **final** source after the last in-place write. In-place reuse of the delay-line bank aliases with last-use of the published source: the second consumer (PED or gate) may legally read a location the next lifting step is already overwriting. Always-ready improved −22.83% (part of that is generic round→sat fusion, not lifting); long backpressure stayed 8088. Integer consumer at another resource point only saw −5.78%. ALU counts never measured the alias.

**X (why not a reskin):** Not “we implemented lifting.” The increment is the **alias theorem**: in-place lifting and dual-consumer last-use cannot share the same 1RW rows unless a **post-commit holding row** is inserted and both consumers multicast from it. Ordinary dense T10 can keep the source in an output-stationary register and never alias. So lifting’s arithmetic win is legal only after paying a measured holding-bank tax; if that tax returns II to the dense schedule, lifting cannot be the title.

**controls:** (C1) same 40-coeff lifting graph, same RNE/sat; (C2) report II limited by 1RW ports vs II limited by recurrences, separately; (C3) subtract round→sat fusion credit; (C4) ordinary dense T10 on the **same** 1RW holding/multicast hardware; (C5) AEE of lifting vs ordinary on valid825 (currently +0.013178, relative fail); (C6) do not quote −22.83% and −5.78% as one number.

**kill-gate:** Port-true compiled II per T10 vector, 1RW, dual consumers. Kill lifting-as-title if (i) holding-bank + 35 intermediates make completed-service gain <15% vs ordinary on the same ports, or (ii) valid825 AEE stays > ordinary+0.005 (today 1.232979368 vs 1.219801338). Kill the alias story if a 1RW in-place schedule already has unique last-use before the next overwrite (no dual-consumer live overlap).

**two-sentence pitch:** Lifting is an in-place 1RW algorithm whose intermediates and dual last-use destroy in-place; the circuit is one holding row plus multicast, not a smaller adder. Either that tax still leaves ≥15% completed service and AEE within +0.005 of ordinary, or lifting is the wrong island for a five-page brief.

**assumptions:** The 35 RNE/sat values are architecturally visible writebacks, not fused into a combinational lifting pipeline of illegal depth; dual consumers remain after CSE.

**predictions:** (P1) A modulo schedule of the lifting DFG will show II bound by 1RW writes of intermediates, not by 159 adds. (P2) Adding the holding row does not change AEE; removing in-place (private output buffer) matches ordinary’s port occupancy. (P3) Always-ready can fall while completed service does not.

**disconfirmers:** If intermediates are fully forwarded in RF with no SRAM write, the 35-count is not a port tax. If lifting AEE can be retrained under +0.005 without changing the graph, AEE stops being this idea’s kill (service remains). If ordinary also materializes ≥35 intermediates, X collapses to I1.

**objection:** Reviewers will say “lifting already cut 260→159, that is the contribution.” The frozen 8088 identity is the rebuttal: add/sub is not service.

---

## P02-I3 — Continuous-θg kills Gustavson at r1; bake-off inner-product vs outer-product vs Gustavson on 1RW

**one-sentence:** Patch r1’s two convs were historically costed as if spike sparsity made Gustavson (row-wise) the default; continuous θg makes A dense-valued, so the legal 1RW dataflow is a measured inner-product / output-stationary residual, not a sparse-gather story.

**A (complete prior to copy):** Copy the **complete** sparse-dense taxonomy and one exemplar per point: Gustavson 1978 row-wise (InnerSP / MatRaptor / GAMMA); outer-product rank-1 (OuterSPACE, SpArch, DSTC); inner-product / output-stationary (TPU, SIGMA dense mode). Copy Timeloop/MAESTRO-style mapping enumeration under a **hard 1RW** partial-acc and 1RW activation constraint. Copy the residual-fusion prior (fused-CNN / fused ResNet blocks) so C-stationary SRAM is allowed to be the residual accumulator.

**B (hole in THIS net):** Frozen neuron is ATLIF with **continuous threshold amplitude θg, not binary spikes**. Event-camera sparsity exists at the sensor, not necessarily at r1 after two convs + residual. Old activity-weighted-dot ledger put whole patch ~35% of a proxy that is **not** this student’s cycle share. Dual consumers after source still need the r1 residual path. A Gustavson mapper that gathers “spike rows” is the wrong prior for this identity; an outer-product mapper that explodes partial C fights the same 1RW bank the dual consumers need.

**X (why not a reskin):** Not “we do sparse conv” and not a CIM tile. The increment is a **negative result that is still a mechanism**: under continuous θg, Gustavson’s skipped-zero advantage is ≤ε, so the only remaining dataflow question is **where the 1RW partials live**. We pick the dataflow that minimizes 1RW grants to dual last-use (predicted: output-stationary C that **is** the residual, inner-product along the kernel), and we **kill** Gustavson and outer-product on the same port budget rather than leaving them as related-work adjectives.

**controls:** (C1) same 1RW activation bank, same 1RW acc bank, same MAC count; (C2) inject true event-sparsity only at ingest as an ablation, never as the r1 default; (C3) binary-ATLIF Gustavson as a **forbidden identity** control (must not become the paper); (C4) ordinary T10 source; (C5) report grants and writebacks, not proxy dots; (C6) two-window integer match.

**kill-gate:** Same-resource completed MAC-pipeline service for the r1 two-conv+residual island. Kill if Gustavson or outer-product matches OS/inner-product within 5% completed service under continuous θg (then dataflow is not the story). Kill if the winner does not deliver ≥15% chain-level completed service vs the current unfused mapping, or if AEE moves beyond +0.005. Kill if the win requires 2RW acc SRAM.

**two-sentence pitch:** This student is not a binary SNN, so Gustavson is the wrong copy; we copy the full three-way dataflow taxonomy and keep only the 1RW mapping that makes the residual the stationary C for both later consumers. The brief’s figure is three dataflows, one port model, one winner, not a sparse-accelerator zoo.

**assumptions:** r1 activations under continuous θg are dense enough that zero-skipping <10% of MAC cycles; the two convs plus residual add share a legal fused live range.

**predictions:** (P1) Density of r1 tensors is ≫ event-frame density. (P2) Outer-product partial-C writebacks exceed OS under 1RW. (P3) Gustavson gather indices cost more than they skip. (P4) OS residual cuts dual-consumer copies because PED reads C in place.

**disconfirmers:** If measured r1 zero fraction is high despite continuous θg, Gustavson may win and this idea inverts (still publishable, but as the opposite mechanism). If fusion of two convs + add changes AEE, numerics — not dataflow — dominate.

**objection:** “Sparse event cameras obviously want Gustavson.” Sensor sparsity ≠ r1 sparsity under this frozen neuron; the control is a density histogram, not an intuition.

---

## P02-I4 — Tick-batch occupancy identity: always-ready is not completed service

**one-sentence:** Noncausal T10 forces a 10-deep tick-batch delay line whose 1RW fill/read occupancy, not the source ALU, sets completed service; always-ready slot cuts that ignore delay-line grants are a false win.

**A (complete prior to copy):** Copy (i) SNN tick-batch vs event-driven queues (TrueNorth dense tick, Loihi event queues) as two complete occupancy models; (ii) noncausal temporal convolution delay lines (TCN / WaveNet) that **must** hold the full temporal kernel; (iii) double-buffering vs true 1RW occupancy (DAE, Buffets); (iv) the classical identity occupancy = fills − last-uses, throughput = min(ALU, port grants). Copy the whole accounting, including that a delay line of T=10 is already a tick-batch even if the RTL is called “streaming.”

**B (hole in THIS net):** PSN is **noncausal T10**: all 10 ticks must land before the source vector exists. That is a 10-row NRV delay line (produce tick t, last-use at the T10 publish). Dual consumers then read the published vector. Frozen measurement: same two-stage SIMD source resource, always-ready 6938→5354, **long backpressure both 8088**. The cheaper lifting ALU increased producer-ready, then waited on the same occupancy-saturated port. Local-window tools that ignore the 10-deep fill undercharge the same way free BN mean/var undercharge wait/storage.

**X (why not a reskin):** Not “we batch 10 frames.” The increment is an **occupancy identity** with a circuit: a rotating 10-row 1RW delay line that **shares the fill port with PSN reads** under a compiled grant calendar, vs an illegal 2-port line or a 10-copy register file. The paper’s causal graph is: noncausal T10 ⇒ delay-line occupancy ⇒ 1RW grants ⇒ completed vectors, with always-ready as a non-service metric. That identity is what makes the 8088 result unsurprising rather than mysterious.

**controls:** (C1) T=10 noncausal frozen; do not switch to causal T to make streaming legal; (C2) 1RW delay line vs 2-bank ping-pong vs RF of 10 vectors, all reported; (C3) lifting vs ordinary on the **same** delay-line hardware; (C4) count delay-line grants separately from ALU-ready; (C5) full 10×96×120×160 domain, not a window with free history.

**kill-gate:** Completed T10 vectors per delay-line 1RW cycle. Kill if delay-line grant utilization is already <50% (then occupancy is not the bottleneck; look at I1/I6). Kill if a legal 1RW calendar cannot raise full-chain completed service ≥15% without adding a read port. Kill if making T causal is required (identity change).

**two-sentence pitch:** Noncausal T10 already bought a 10-deep 1RW delay line; we copy tick-batch occupancy accounting and score completed vectors per grant, not always-ready slots. The five-page claim is that the 8088 stall is delay-line occupancy, and a compiled fill/read calendar is the only legal fix under 1RW.

**assumptions:** T10 stays noncausal; the delay line is in SRAM not an infinite RF; producer ALU and delay-line port compete.

**predictions:** (P1) Grant-level traces will show PSN reads colliding with tick fills. (P2) Lifting reduces idle ALU cycles, not collisions. (P3) Widening the ALU without extra grants leaves 8088 unchanged.

**disconfirmers:** If T10 operands are supplied from a already-resident RF of all 10 ticks with spare ports, B is false. If 8088 is after publish (pure dual-consumer or BN), this idea loses to I1/I6. If causal T10 is later allowed, the delay line shrinks and X evaporates.

**objection:** “T=10 is tiny, just keep it in flops.” 10 × 96 × 120 × 160 at I24 is not a flop file under the same-state rule; the control is the bit×row 1RW cost, not the digit “10.”

---

## P02-I5 — NRV last-use cell vs compiled calendar (compiler vs hardware)

**one-sentence:** Classify every produced vector as an NRV (next-reuse vector) with a 2-bit last-use count, then decide whether a 2-bit SRAM side-cell or a fully compiled 1RW calendar is the cheaper legal way to free the bank at dual-consumer last-use.

**A (complete prior to copy):** Copy **both** complete stacks, not a hybrid slogan: (i) hardware scoreboard / refcount (CDC 6600 scoreboard, store-refcount in memory allocators, hardware GC last-use); (ii) static VLIW/CGRA calendar with explicit last-use opcodes (Cydrome, Softbrain, Stripe, TVM static schedules). Copy Buffet credits as the storage idiom both stacks share. The copied question is the old compiler-vs-hardware one: is consumer-complete **data-dependent** on this net, or is it a compile-time constant?

**B (hole in THIS net):** In-scope constraints already name 同资源服务、有限端口、完成/ready 依赖、完整常量编译. Dual consumers imply refcount=2 for source vectors, refcount=1 for most lifting intermediates, refcount=T-k for delay-line ticks. Integer gates/I24/PED q24 are **deterministic** on captured windows (0 diff vs model), so last-use epochs are not value-dependent the way binary-spike skip would be. Continuous θg **removes** the usual excuse for a runtime sparse scoreboard. The hole is that a hardware ready-bit fabric may be taxing the same 1RW path that already absorbs lifting’s −22.83%.

**X (why not a reskin):** Not “we added a scoreboard” and not “the compiler schedules it.” The increment is a **head-to-head under identical 1RW**: a 2-bit last-use cell beside each source row (hardware, tiny, TCAS-II-circuit-native) versus a fully unrolled grant calendar with zero runtime ready state (compiler-complete). Because ATLIF is continuous and integer consumers match the model, the calendar is **legal** here even if it is illegal for binary SNNs. One winner, one mechanism.

**controls:** (C1) same grants, same dual-consumer graph, same T10; (C2) scoreboard with data-dependent stall vs calendar with only structural 1RW stall; (C3) binary-spike scoreboard as a forbidden-identity ablation; (C4) measure extra SRAM bits (2 per row) vs extra compiler IR; (C5) two-window zero-diff as evidence that last-use is not data-dependent.

**kill-gate:** Full-chain completed service, same ports. Kill the hardware cell if the calendar matches its service within 2% (then 2 bits are a reskin of compile-time last-use). Kill the calendar if any captured window needs a data-dependent stall beyond 1RW structural hazards (then “complete constant compilation” is false on this net). Kill either if chain gain <15% or 8088 does not move. AEE must not move.

**two-sentence pitch:** Dual-consumer last-use is a 2-bit refcount; continuous θg and exact integer consumers make that count a compile-time constant, so a 1RW calendar can retire the scoreboard. TCAS-II either ships a 2-bit last-use cell as the circuit or ships the proof that the cell’s entropy is zero — one of those, not both as a zoo.

**assumptions:** Last-use of gate vs PED is structural (epoch known) on valid825, not gated on spike presence; NRV rows are the T10 source and delay-line, not the entire net.

**predictions:** (P1) Two-window traces show last-use cycle independent of θg amplitude. (P2) A calendar with II set by 1RW matches a scoreboard’s completed count. (P3) A spike-skip scoreboard (binary identity) would diverge here and must not be used.

**disconfirmers:** If PED last-use depends on continuous values (data-dependent residual skip), the calendar is illegal and the 2-bit cell must stay. If compiler IR cannot name last-use without runtime complete tokens from BN (I6), this idea defers to I6. If 2-bit cells change SRAM cycle time, PPA-after-gate may invert the winner (not a pre-RTL kill).

**objection:** “Compiler vs hardware is a MAG paper, not TCAS-II.” The circuit is the 2-bit cell **or its absence**; the five-page figure is occupancy-with-cell vs occupancy-with-calendar on the same 1RW row.

---

## P02-I6 — Projection-BN Welford-on-fill: the 1RW tax of full-domain batch stats

**one-sentence:** Native projection BN legally reduces actual batch statistics over 10×96×120×160 on the same 1RW ports as the residual add; overlapping Welford with tick-batch fill is the mechanism, free local mean/var is the undercharge.

**A (complete prior to copy):** Copy Welford’s online mean/var, BN-fusion-into-conv as used in inference compilers (TVM/Glow running-stats vs batch-stats), and tree/ring reduction microarchitectures (allreduce as a dataflow, not a collective library). Copy the blocking-reduction occupancy model: a reduce-then-broadcast is a **complete/ready barrier** on the residual add. Copy “running stats for inference” as a **numeric-changing** prior that must be measured, not assumed.

**B (hole in THIS net):** Frozen observation: native projection BN on captured students uses **actual batch statistics over the full 10×96×120×160 domain**, not frozen running stats. Local-window replay given free mean/var **undercharges wait/storage**. That reduction is a third 1RW client next to T10 fill and dual-consumer reads, and it is a barrier in front of residual add — a ready/complete edge that can hold 8088 even when source ALU is always-ready. This is independent of lifting’s 260→159.

**X (why not a reskin):** Not “approximate BN” and not “skip BN.” The increment is a **same-state schedule**: one Welford pair (mean, M2) updated on every 1RW fill of the projection tensor, so the barrier length is **hidden in fills already paid**, vs a post-hoc second pass that re-reads 18,432,000 elements. Running-stats inference is a **numeric control**, not the title, unless it passes the AEE relative gate. The circuit is a two-word streaming accumulator + a broadcast of (μ, σ) on the same multicast bus as I1.

**controls:** (C1) exact full-domain batch stats vs running stats vs local-window free μ/σ (the last is illegal as a service claim); (C2) same 1RW ports as projection writes; extra stats SRAM counted as state; (C3) AEE of any stats change vs ordinary 1.219801338; (C4) do not hide the reduction in a second port; (C5) report barrier cycles between last projection write and first residual add.

**kill-gate:** (Service) extra 1RW reads for stats must go to **0** beyond the fills already required to write projection; barrier cycles must drop enough that full-chain completed service ≥15%. (Numeric) if Welford-on-fill is exact, two-window delta = 0 and valid825 stays on the frozen student. If the idea switches to running stats, valid825 AEE must still be ≤ 1.259 and ≤ ordinary+0.005 — currently we do **not** know that. Kill if local-window free μ/σ is the only way to “win.”

**two-sentence pitch:** Full-domain batch BN is a blocking 1RW reduction, not a BN IP; we copy Welford and hide the reduce in the fills the projection already pays. Either the barrier vanishes at zero numeric delta, or the undercharged local-window model is why previous service numbers cannot be believed.

**assumptions:** Projection writes visit each domain element at least once (so Welford can ride along); BN is on the critical complete/ready path of residual add; 18.4M is the true reduction set.

**predictions:** (P1) A service model with free μ/σ will under-report stalls vs a model that pays the second pass. (P2) Welford-on-fill matches batch μ/σ at q24 if accumulation width is sufficient. (P3) 8088 moves only when the BN barrier is overlapped, even if T10 ALU is unchanged.

**disconfirmers:** If projection BN is off the long-backpressure path (8088 unchanged when BN is magically instant), B is false for the 8088 identity (still true as a storage tax). If Welford in integer q24 diverges from the student’s actual batch stats, exactness fails and we must widen acc or drop the idea. If the student can legally switch to frozen running stats **and** pass AEE, the reduction dataflow dies and the paper becomes a numeric BN-for-inference brief (different X).

**objection:** “BN fusion is 2015.” Fusion into conv is not the copied prior we need; the copied prior is **batch stats over a 10-tick spatial domain on 1RW**, which those papers do not pay.

---

## P02-I7 — Port-limited II of the lifting graph (intermediates as writeback tax)

**one-sentence:** Treat the lifting DFG’s 35 RNE/sat materializations as the recurrence/port bound on initiation interval, and only claim a source win if compiled II, not add/sub, beats ordinary dense T10 on 1RW.

**A (complete prior to copy):** Copy iterative modulo scheduling (Rau) with **RecMII** vs **ResMII**, register-pressure spilling (Chaitin), and the writeback-port model of SIMD pipelines (one 1RW write per cycle unless fused). Copy round/sat fusion as a **writeback elision** prior (two architectural writes → one), which the freeze already notes ate part of −22.83%. The copied artifact is a reservation table for a 2-stage SIMD source, not a flop-count table.

**B (hole in THIS net):** Ordinary 260 add/sub vs lifting 159 add/sub + 35 intermediate RNE/sat, **same two-stage SIMD source resource**. Always-ready slots fell 22.83%; long backpressure both 8088. Add/sub is not II. The 35 intermediates are candidates for ResMII = ceil(writes/1RW). Ordinary dense may have a larger adder but fewer architecturally visible writes if the 10×10 is output-stationary in the SIMD RF. Comparing 260 vs 159 without a reservation table is the hole that let a false source win be believed.

**X (why not a reskin):** Not “modulo-schedule the FIR.” The increment is **making II the paper’s source metric** and proving (or disproving) lifting under that metric with dual-consumer last-use writes included. Round→sat fusion is a control that must be applied to **both** ordinary and lifting, because it is generic writeback elision. If after fusion ResMII_lifting ≥ ResMII_ordinary, lifting has no 1RW story; the title must move to I1/I4/I6.

**controls:** (C1) identical 2-stage SIMD, identical 1RW writeback; (C2) RecMII vs ResMII reported separately; (C3) round→sat on/off as a 2×2; (C4) dual-consumer holding write included or excluded as two rows; (C5) AEE of the two sources (lifting currently fails +0.005); (C6) never add integer-consumer −5.78% to this table.

**kill-gate:** Compiled II per T10 vector. Kill lifting-as-source-win if II_lifting ≥ II_ordinary under 1RW after round→sat fusion. Kill the paper-as-lifting if II_lifting wins but (a) 8088 does not move (gain absorbed downstream) or (b) AEE 1.232979368 is not brought within +0.005 of 1.219801338. A **downstream** idea may still use ordinary T10; this idea is specifically the source-II claim.

**two-sentence pitch:** Source service is initiation interval on a 1RW writeback, not 260 vs 159 adds; we copy modulo scheduling and count the 35 RNE/sat writes. If II does not win, lifting is not the TCAS-II mechanism — which is itself a result the brief must not bury.

**assumptions:** Two-stage SIMD is the real resource; intermediates are not infinitely forwarded; ordinary dense can legally keep the T10 output in RF.

**predictions:** (P1) ResMII_lifting is set by writes ≈ 1 (publish) + 35/k fused groups, not by 159. (P2) Round→sat fusion shrinks both arms and shrinks the gap. (P3) Dual-consumer publish write is on the II critical path for both arms.

**disconfirmers:** If a combinational lifting pipeline of 159 adds fits the same two-stage SIMD without 35 writes, B is false. If ordinary also spills ≥35 temps, the II gap tracks add/sub after all. If software-pipelined II wins but AEE fails, lifting still cannot title the brief.

**objection:** “II is a compiler metric, TCAS-II wants circuits.” The circuit is the 2-stage SIMD + 1RW write port whose reservation table **is** II; the figure is that table.

---

## P02-I8 — Output-stationary fused residual island (r1 two-conv + T10 publish + dual last-use)

**one-sentence:** Collapse patch r1’s two convs, residual add, T10 publish, and both source consumers into one output-stationary island whose C SRAM is the only 1RW live range, replacing a module pipeline that pays NRV tax at every seam.

**A (complete prior to copy):** Copy fused-CNN (Alwani MICRO), residual-block fusion as implemented in production inference engines, TPU-style output-stationary acc, and Buffet-composed **multi-stage** data orchestration (fill of A/B, accumulate in C, drain to two consumers). Copy the “no inter-layer DRAM, but **also** no inter-layer 1RW copy” rule — fusion papers often stop at DRAM. Copy a complete mapping search (Timeloop-style) **restricted to OS** so this does not become a dataflow zoo (the zoo is I3).

**B (hole in THIS net):** Historically expensive region: patch embed residual r1 (two convs) plus T10 PSN; dual consumers after source; native projection conv+BN+add also exist. Today’s service models treat these as staged producers with always-ready/complete bits at each seam. Each seam is an NRV: not last-used, therefore a 1RW resident. The −22.83% source-only gain with 8088 unchanged is the symptom of **seam occupancy**. Integer-consumer −5.78% at another resource point cannot be multiplied with source slots, and cannot be read as chain FPS.

**X (why not a reskin):** Not “operator fusion” and not “one accelerator for the whole net.” The increment is **C-stationary as the dual-consumer holding bank** (I1’s multicast source **is** r1’s residual acc), with T10 applied as a **temporal mix on C rows in place** rather than as a producer of a new tensor. That is a different legal dataflow from “run conv, write, run T10, write, fork.” Projection-BN (I6) stays **outside** the island unless it passes the exact-stats overlap; stuffing BN in without Welford-on-fill would be the module zoo the scope forbids.

**controls:** (C1) unfused staged pipeline on the same 1RW budget; (C2) fusion without in-place T10 (still extra publish tensor) as a half-measure; (C3) ordinary vs lifting T10 **inside** the island, II from I7; (C4) chain-level completed service only, no component-FPS product; (C5) two-window integer match for I24/PED/gates; (C6) same backpressure model.

**kill-gate:** Complete-chain same-resource net completed service ≥15% vs unfused, **and** long backpressure <8088 by ≥15%, **and** AEE gates. Kill if fusion requires extra banks to hide intra-island hazards (not same-port). Kill if T10-in-place on C changes valid825 beyond +0.005 vs ordinary. Kill if the island cannot include **both** consumers (then it is just fused conv, a reskin of 2016).

**two-sentence pitch:** The residual C is the only tensor that should occupy 1RW: two convs accumulate into it, T10 mixes it in place, gate and PED drain it once. That single live-range is the brief’s mechanism; component speedups that stop at a seam are how 8088 absorbed the last ALU win.

**assumptions:** T10 can legally mix along the time axis of C without a private source buffer; both consumers can drain OS C without a layout transpose that needs another bank; projection-BN can sit outside without re-imposing a barrier that restores 8088 (or I6 is stacked after the gate).

**predictions:** (P1) Unfused NRV occupancy ≥ 3 live tensors (r1 out, T10 out, residual); fused occupancy = 1. (P2) 8088 drops only when the T10 publish tensor disappears. (P3) Lifting vs ordinary inside the island is a second-order II effect (I7), not the title.

**disconfirmers:** If in-place T10 on C is numerically illegal (layout / noncausal mix needs a private buffer), fusion stops at “fused conv” and X dies. If PED needs a different layout than gate, drain becomes two transposes and 1RW occupancy returns. If chain service is already bound in projection-BN (I6) or delay-line fill (I4), fusing r1 cannot hit 15%.

**objection:** “Fusion is not novel.” DRAM-fusion is not novel; **1RW live-range fusion through a noncausal T10 and a dual last-use drain** is the copied-then-differenced object, and the 8088 identity is the measurement that older fusion papers never had on this net.

---

## Cross-idea notes (still ideas, not a ranking)

- I1, I4, I5, I8 are arithmetic-preserving candidates (ordinary T10, AEE 1.219801338). They should be tried before any lifting-title claim because lifting already fails the relative AEE gate.
- I2 and I7 are the honest lifting-as-source checks: alias + II. If they miss 15% or AEE, stop using lifting as the paper identity (layout may still exist).
- I3 is the identity-protecting dataflow bake-off against binary-SNN Gustavson temptation.
- I6 is the observation most likely to falsify any source-only win: full-domain BN is a barrier the local-window model gives away for free.
- Stacking I8 with I6 is allowed only as **one** island plus one overlapped reduction, not as eight modules.
- RTL/PPA is out of horizon until a CPU service model with 1RW grants + integer AEE passes the kill-gates above.
