# Adversarial review — F1 Dual-completion last-use

Reviewer did **not** originate F1 (fusion of P01-I1/I5, P02-I1/I2/I5, P05-I1/I2/I6). Steelman first, then attack. Frozen facts from `PROBLEM.md` only. Priors from `literature/L2_gustav_loas.md` and Prosperity identity in L2 §3 / L1. No papers invented.

---

Idea ID: F1

Strongest version:

One mechanism, not the six-ID stack. On the frozen two-stage SIMD source, a compiled T10 / continuous-θg word (and the I24 residual row that currently gets written and reread) stays occupied until `gate_sampled ∧ PED_sampled ∧ (BNstat_sampled if that location is still a legal full-domain-BN use)`. Fetch/skip occupancy is the packed predicate `support(θg-gate) ∪ support(PED)`, **not** GustavSNN’s P-bit `{0,1}` NRV and **not** LoAS’s packed-T unary silent fiber. Relative prior is ordinary dense-source T10 (AEE 1.219801338) at the **same** ports, state class, and backpressure model; lifting is a sensitivity, not the title. The letter lives or dies on moving long-backpressure off 8088 and on complete-chain net service ≥15%. Always-ready 6938→5354 is not service. Integer gates / I24 / PED q24 stay 0-diff on the two captured windows. ATLIF is not binarized.

That is the only steelman that still looks like one TCAS-II island: L2’s own local X (dual-retire + union occupancy) plus P05’s last-use join, with BN-stat as a **measured** third use rather than a stacked F2 sidecar.

Observation that would count against it:

A cycle-accurate occupancy trace on the same two-stage SIMD resource that produced 8088, split into `{wait_source_ALU, wait_gate, wait_PED, wait_BNstat, wait_1RW_port, wait_delay_line_fill, wait_I24_reread}`, showing that the dual-completion join is **not** the dominant term. Sufficient kills from that trace: (i) gate, PED, and BN-stat become ready on the same cycle for ≥95% of T10 vectors (hybrid/AND-retire has nothing to fire; lazy fork already optimal); (ii) union occupancy density ≈ 1 (continuous PED almost always live → NRV-style skip ≈ 0); (iii) 8088 is delay-line fill/read collision or BN reduce-then-broadcast, not last-use of the published source; (iv) after a bit-true last-use tag, 8088 is unchanged. Any one of these deletes the steelman without needing RTL.

At least two alternative explanations for any predicted 15% win:

1. **Throttle / occupancy rename, not service.** Typed credits `min(credit_gate, credit_PED, credit_BNstat)` or AND-retire stop the source from over-issuing into the measured 8088 wall (P05-I6). Long-backpressure *count* can fall because the producer does less, while completed r1→gate+PED+BN+add vectors per cycle do not rise 15%. PROBLEM.md already showed this failure mode: always-ready −22.83% with long-BP stuck at 8088. Occupancy mix ≠ net service.

2. **Generic fusion, extra skid, or a self-inflicted 1RW serialize.** Part of 6938→5354 is already generic round→sat fusion (PROBLEM.md). Depth-1/2 elastic taps (Carloni LI; Cortadella SELF; P01-I1) hide burstiness under a same-state violation. Serialized dual-read on 1RW (P02-I1) is an Eyeriss-multicast / buffet-credit miss, not a new retire contract: a 1R latch plus refcount 2→0 can print “15%” against a baseline that illegally issued the second consumer on a second grant. If the 15% appears only vs that serialized strawman, or only vs a local-window replay with **free** μ/σ, it is accounting.

Further alternatives that must be named in the letter if a number appears: ordinary T10 plus the **same** credit/last-use protocol matching lifting (H-F1 alternative: X is the protocol, not dual-completion occupancy); BN barrier overlap that is actually F2 (Welford-on-fill) smuggled in via “BN-stat if live”; I24 reread elimination that is register allocation / in-place last-use (P05-I2) rather than union-NRV.

Measurement failure modes:

- Reporting always-ready slots, add/sub (260→159), or the separate integer-consumer −5.78% (758777→714889) as chain service; adding the two tables.
- Single stall bit instead of the wait-class histogram above; synthetic 896/1024-style ready patterns (H-F1).
- Free mean/var on a local window (PROBLEM.md undercharge) as the baseline BN wait.
- FIFO depth, extra anti-token SRAM port, 2RW, or operand-collector ports not held to the frozen two-stage point (same-port / same-state fail).
- Enabling a skip from union occupancy that changes the integer law, then quoting two-window 0-diff as if valid825 AEE were unchanged.
- Measuring source-only completion, not residual add after actual full-domain batch stats.
- Comparing dual-completion against a lazy-fork or 2RW oracle the authors introduced, not against the frozen joint-ready ordinary arm.

Prior evidence that challenges it (cite L2 / Prosperity identity if relevant):

**GustavSNN NRV is the wrong skip unit here, and L2 already wrote F1’s X.** Gustav NRV is a P-bit `{0,1}` row at **one tick**; skip iff all P columns are silent; commit is `LIF(V_θ)` after the current-tick `d`-sweep; one consumer of `O`; activations are binary (L2 §1.3, §1.8–1.10). Local source is compiled T10 with continuous θg and two (three) consumers. L2 §1.12 already names the only honest local X as dual-retire of P-column context (X-G1) and occupancy `support(gate) ∪ support(PED)` (X-G2), and L2 §1.14 kills that X if union density makes skip ≈ 0 or dual-retire grows the same-state budget. F1 as fused is those two sentences plus handshake vocabulary. It does **not** copy Gustav’s complete A (CPTB, in-situ P REG, NR4 merger, same-ID barrier, column-major time-second — L2 §1.11). Incomplete A + the transfer note’s X is a reskin of NRV with a different predicate, not a measured letter.

**LoAS last-use/fiber is not dual-consumer last-use.** LoAS skip is packed-T unary silent-across-all-T; consumer is unrolled P-LIF; Gust is an *rejected* dataflow (L2 §2.1, §2.8). Fiber last-use of `0000` does not retire a continuous PED/I24, and T=10 is already the regime LoAS §VI-B flags as fewer silent neurons. Citing Gustav **and** LoAS as stacked A contradicts both papers (L2 §0, §3).

**Prosperity identity forbids treating union support as product sparsity.** Prosperity EM/PM reuses inner products of **binary** GeMM rows; same support ⇒ same product; one consumer of Y; intersection-of-supports is explicitly unused (L2 §3; Prosperity §III-B as recorded in L1). Continuous θg breaks the reuse theorem. Dual consumers mean skipping a gate-silent source still leaves PED live. Fusion already parked Prosperity-on-real-θg until the F1 object exists — correctly: F1 must not inherit EM/PM. P06-I2 is the harsher sibling: **producer** slots drop only on **intersection** of skip masks; union occupancy is a consumer-side curiosity and cannot legally cut SIMD issue. If PED is a dense residual, union occupancy is dense, X-G2 dies, and F1 collapses to “wait for both readys.”

**Source-side arithmetic already failed this exact test.** Always-ready −22.83% absorbed at 8088 on **both** ordinary and lifting (PROBLEM.md). Any mechanism that only changes producer occupancy or skip of gate-only zeros has a published negative.

Could a TCAS-II reviewer call this a reskin of credit flow-control or Gustav NRV?

**Yes.** That is the default reading, and it is fair.

- **Credit / elastic flow-control reskin:** independent credits per consumer, AND of completions, last-use refcount 2→0, anti-tokens, typed credit min() are Carloni latency-insensitive design, Cortadella SELF eager vs lazy fork and early-eval (DAC 2006/2007 as named in P05-I1), Dally–Towles reverse credits, Buffet occupancy = fills not yet last-used, AXI-Stream valid/ready, architected last-use bits. P05-I1’s own objection is the review: “you added valid/ready.” P05-I6: credits without extra consumer parallelism decorate a stall. P01-I1: “skid buffers called a protocol paper.” Five-page TCAS-II punishes missing relative prior; the relative prior here is ordinary+joint-ready vs ordinary+the same 2-credit tap. If that delta is the 15%, lifting, NRV, and LoAS must leave the title.

- **Gustav NRV reskin:** F1’s A line is “GustavSNN CPTB in-situ + NRV.” Occupancy as a packed P-wide support vector **is** NRV. Changing `{0,1}` to union-of-continuous-supports is L2 X-G2, which L2 already said is empty if dense. Without the rest of Gustav’s checklist, the sentence is “we also NRV.” W1 already classified GustavSNN as **control**, not X, unless NRV is redefined over quantized θg **and** PED is still served.

A reviewer can write both sentences in one paragraph: *credits from 2001–2007, occupancy vector from HPCA 2026 NRV, fiber last-use from MICRO 2024, none of them dual-lifetime continuous T10, and you did not copy any of them whole.* That is a binary reject on novelty + relative prior, before the 15% is even checked.

Mitigation:

1. **Do not implement protocol RTL first.** Cheap CPU experiment on captured windows: split 8088 into the wait classes above; dump `live(Z)` vs `live(U)` including BN-stat and residual-add; print union vs intersection vs gate-only occupancy density; print stall correlation of gate vs PED. If join is not the bottleneck, **stop F1** the same day (P05’s own I1/I2 first experiments).
2. **One mechanism.** Dual-completion last-use **or** I24 recolor **or** typed credits **or** 1RW multicast — not P01-I1+I5 plus P02-I1+I2+I5 plus P05-I1+I2+I6. P05 forbids that stack. BN-stat “if live” is a **fact of the net today** or it is F2; it is not a license to add a Welford sidecar.
3. **One A, copied complete.** Either Cortadella/Carloni elastic fork+join (handshake letter) **or** Gustav checklist §1.11 as a **control** with dual-retire as the only X. Not Gustav+LoAS+credits as adjectives. Prosperity stays a parallel control, never a layer (L2 §3).
4. **Controls that steal the win if omitted:** ordinary + joint ready; ordinary + same tap/credits (depth cap published); eager duplicate vs lazy barrier vs hybrid; binary NRV vs union occupancy vs dense (no skip); fusion-matched round→sat on both arms; paid full-domain BN vs free μ/σ (illegal as a service claim); serialized 1RW vs 1R multicast as an **honest** 1RW baseline, 2RW as an illegal upper bound.
5. **Kill numbers stay conjunctive:** AEE ≤ 1.224801338; 8088 moves; complete-chain same-port/state/backpressure service ≥15%; 0-diff integer windows. If union density is too high to skip, drop NRV language and keep only last-use lifetime — then survive only if phantom hold (I24 reread / over-pin) is the measured 8088 term.

Residual uncertainty:

Complete-chain same-resource net service is **not closed** (`SCOPE.md`). True paid cost of full-domain `10×96×120×160` BN stats is unknown. Whether the BN `10` is the T10 axis is unknown (P05-I7 immediate kill if not). Union occupancy density under continuous θg is unmeasured; P06-I2 predicts intersection, not union, is what a producer can charge. Live-range overlap of Z vs U including BN-stat is unmeasured. Whether 8088 is join, BN barrier, delay-line 1RW, or FIFO depth is unmeasured — F1, F2, and P02-I4 are mutually exclusive titles until that split exists. Two-window integer 0-diff is not valid825.

Disposition: revise

Keep the **question** (why 8088 did not move when the source ALU did; dual-consumer last-use vs binary NRV). Do not keep the fused wording, the dual Gustav/LoAS A, or a build plan. Pause any circuit work until the 8088 mix and union density are printed. If those prints show simultaneous ready, dense union, or a non-join bottleneck, **stop** rather than pivot into F2 under this ID.

Novelty 0-4 and performance-plausibility 0-4 with reasons. 4 almost never.

- **Novelty: 1.** Dual-lifetime of continuous θg vs residual/PED is a real *identity* hole vs Gustav (single binary `C`, commit at LIF), LoAS (one P-LIF), and Prosperity (binary GeMM EM/PM, one Y consumer). That hole is already written as L2 X-G1/X-G2. The circuit objects F1 actually names — credits, last-use tags, elastic fork/join, packed occupancy vectors — are the relative priors a TCAS-II reviewer is trained to subtract. Six independent IDs fused into one letter is a zoo, which the venue punishes. A 2 would require a single copied A plus a measured retire-contract figure that cannot be redrawn as valid/ready. As fused: 1.

- **Performance-plausibility: 1.** The 15% is a copied kill-gate, not a bound from occupancy. Historical evidence cuts **against** a source-side occupancy trick: −22.83% always-ready, 8088 unchanged. AND-retire does not add consumer parallelism. Union occupancy of a continuous PED path is expected dense (L2 mechanism kill; P06-I2: union cannot drop producer issue). A 15% chain win then requires an unmeasured phantom (I24 reread or BN pin) to dominate 8088 *and* to be removable at equal ports — which, if true, may be F2 or P05-I2, not dual-completion NRV. Throttle can fake the 8088 number. Until the wait histogram exists, 15% is not a prediction; it is a hope. A 2 would need that histogram showing last-use slack ≥15% of 8088 with union skip unused. As fused: 1.
