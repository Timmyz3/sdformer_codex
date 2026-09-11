# Q02 — HPCA architect (independent, round 2 absorb)

Identity lock: official AT-LIF \(o=\theta\cdot H(m-\theta)\in\{0,\theta\}\); inference absorbs layer-shared \(\theta\) into next \(W\). After absorb the spike path is binary \(\{0,1\}\times W\). Residual / PED / I24 is a different continuous tensor. Noncausal T10 mix (PSN / lifting CSE) is continuous arithmetic **before** threshold. Dual consumers, if any: A = binary gate after absorb; B = continuous residual path.

A to beat, not dodge: Prosperity product-sparsity forest, Gustav NRV/CPTB, FireFly-S bitmap AND. Those theorems start after \(s\in\{0,1\}\) exists. X is only mix-before-threshold, dual last-use with residual, full-domain BN, or optical-flow spatial structure.

Venue: TCAS-II 5 pages; same-port/state/backpressure net service \(\ge 15\%\); AEE abs \(\le 1.259\) and relative \(\le +0.005\). No CIM, no OpenROAD-as-PPA, no FPS products, no unabsorbable int8, no continuous-\(\theta_g\) contribution. One mechanism per idea.

---

## Q02-I1 — Mix-retire into threshold (T10 last-use dies at the comparator)

**One sentence.** Issue the noncausal T10 mix so a CSE value whose only consumer is \(H(m-\theta)\) is retired into the comparator and never written back; the port then sees a FireFly-legal bitmap plus a separate residual last-use.

**A.** FireFly-S bitmap AND and Prosperity forest on the post-absorb \(0/1\times W\) GeMM; Gustav CPTB packs the emitted spike bitmap. Copy those units whole.

**B.** Freeze: mix is continuous and **before** threshold; spike GeMM is **after** absorb. Same-port always-ready 6938→5354 (−22.83%, part generic fusion) while long-backpressure both 8088 — fusion that still writes mix intermediates back does not free the port under stall. Binary-GeMM last-use is of \(s\) and \(W\), not of pre-threshold \(m\).

**X.** Prosperity/Gustav/FireFly have no operand called “pre-threshold mix intermediate.” The new rule is a live-range kill at the AT-LIF comparator: mix temps are not GeMM tiles and are not residual tiles. Residual/PED/I24 is not retired this way. If this is implemented as “run FireFly on spikes” without the mix-retire rule, it is a reskin and is out.

**Controls.** Absorb on; same SRAM port/state machine; ordinary AEE path as golden (1.219801338); residual q24 0-diff vs two-window integer model; no extra memory port; A’s forest/AND/CPTB instantiated, not replaced.

**Kill-gate.** Under long backpressure, mix-retire + copied A must cut net service cycles \(\ge 15\%\) vs copied A alone on the same port. If the delta is only the always-ready generic-fusion −22.83% and backpressure stays 8088, die. If any mix-retire rounding moves AEE beyond abs 1.259 or relative +0.005, die.

**Two-sentence pitch.** After \(\theta\) is absorbed, the letter copies a bitmap GeMM; the missing SRAM last-use is the T10 mix that exists only before \(H(m-\theta)\). Retiring that mix into the comparator, while leaving residual as a second live tensor, is the occupancy win those binary-GeMM theorems do not state.

**Objection.** “You just fused a neuron with a MAC; FireFly already pipelines spike generation.” Counter: FireFly pipelines **spikes**, not a noncausal 10-step continuous CSE graph whose last-use is the threshold.

**Assumptions.** A measurable fraction of mix CSE values have threshold as sole consumer (lifting source CSE already drops 260→159 add/sub, so the live set is real). Residual last-use is not secretly the same buffer as \(m\). Same-port occupancy under backpressure is mix-write dominated, not GeMM-zero dominated.

**Predictions.** (1) Mix SRAM returns fall roughly with the sole-consumer fraction; residual returns do not. (2) Always-ready improves, but the gated figure is backpressure, not 6938→5354. (3) Post-absorb bitmap entropy stays close to ordinary, so A’s skip rate does not collapse.

**Disconfirmers.** Mix temps are already register-only in the baseline (no SRAM last-use to kill). Residual and mix share one live range, so “retire mix” corrupts residual. Copied A already owns the port under backpressure; mix-retire adds \(\lt 15\%\).

---

## Q02-I2 — Two-color last-use scoreboard (bitmap vs residual)

**One sentence.** Scoreboard color-S for post-absorb spike bitmaps (Prosperity-legal) and color-R for the continuous residual/PED/I24 tensor; delay a color-S fire that would extend color-R’s live range.

**A.** Prosperity forest last-use / product-sparsity issue order on binary GeMM tiles, plus FireFly AND as the inner product.

**B.** Freeze splits two objects: spike GeMM after absorb, and residual/PED/I24 as a **different** continuous tensor. Dual consumers: binary gate vs residual path — not “analog AT-LIF amplitude shared by two MACs.” Patch r1 residual chain is historically expensive (old proxy ~35% of activity-weighted dots; proxy ≠ new cycle share). Separate integer consumer model only 758777→714889 (−5.78%) — two datapaths without a coupled last-use do not hit 15%. Prosperity last-use is of products \(s\cdot W\), not of a second type-R tensor.

**X.** The forest orders **nonzero binary products**. Color-R is not a product, not a spike, not a weight. The scoreboard’s only new predicate is “does this S-issue extend R’s live range?” That predicate is undefined in Prosperity/Gustav/FireFly.

**Controls.** Window-S = copied forest+AND; window-R = existing q24 residual/PED/I24 (0-diff vs model); same port; absorb on; no int8 spike payload; no extra consumer table.

**Kill-gate.** Same-port/state/backpressure net service \(\ge 15\%\) vs Prosperity-only issue (color-S greedy). If color-R stalls are already hidden, or if the coupling is two FIFOs with −5.78%-class gain, die. AEE must stay on the ordinary side of 1.259 / +0.005.

**Two-sentence pitch.** Product sparsity schedules zeros on the absorbed spike path; it does not schedule the residual tensor that still occupies the same port. A two-color last-use rule is the smallest occupancy ISA those GeMM papers do not have.

**Objection.** “Last-use is last-use; Prosperity already has a scoreboard.” Counter: Prosperity’s scoreboard keys are spike-index × weight-index. Residual tiles have neither.

**Assumptions.** Color-S and color-R contend for the same port/state (the freeze’s same-port number). Residual live ranges are long enough that greedy S-issue extends them. Dual-consumer identity stays gate vs residual, not dual MAC on AT-LIF \(o\).

**Predictions.** (1) Mean color-R live range shortens when S-issue is delayed at R-conflicts. (2) Backpressure cycles drop more than always-ready cycles. (3) Product-sparsity skip count stays within a few percent of greedy Prosperity (we delay, not skip differently).

**Disconfirmers.** Color-R already dies before any S-issue (no conflict). Delaying S lengthens total occupancy (priority inversion) and misses 15%. Gate bits are stored as residual (identity break).

---

## Q02-I3 — Full-domain BN prefix, then disappear from the GeMM

**One sentence.** Run projection BN as a dense one-pass reduction over the captured \(10\times 96\times 120\times 160\) domain, fold affine into the T10 mix, threshold-and-absorb, and only then run copied bitmap GeMM — BN never becomes a sparse binary kernel.

**A.** Gustav CPTB + FireFly AND + Prosperity forest on the **post-BN, post-mix, post-absorb** spike GeMM. After the fold, A applies without modification.

**B.** Freeze: captured projection BN uses **actual batch stats** over \(10\times 96\times 120\times 160\) — full-domain, dense, not product-sparse. Native projection conv + BN + residual add exist. Binary-GeMM theorems have no BN, no batch reduction, no affine-into-mix fold. Same-port stats/affine collide with spike refill; they are not zeros the forest can skip.

**X.** Prosperity/FireFly skip \(s=0\) or \(W=0\). BN mean/var are reductions over a dense tensor that is not a spike bitmap. Folding \(\gamma/\sigma\) into mix constants is a pre-threshold rewrite. If the idea is “quantize BN into the weight bitmap,” that is a reskin and is out.

**Controls.** Same batch stats as capture (not running estimates unless 0-diff proven); absorb after BN+mix+threshold; residual add remains a continuous path; A instantiated on the spike GeMM; AEE golden = ordinary 1.219801338.

**Kill-gate.** BN-prefix must (i) be 0-diff or inside AEE 1.259 / +0.005 vs captured stats, and (ii) contribute to \(\ge 15\%\) same-port/backpressure net service when combined with copied A, vs A plus unfused BN on the same port. If BN is off-port already, the occupancy claim dies even if arithmetic is correct.

**Two-sentence pitch.** Full-domain BN is a dense reduction the binary-GeMM priors never named; once affine is folded into the mix, the spike path is allowed to be a textbook FireFly/Prosperity GeMM. The letter’s X is the reduction’s port occupancy and the fold, not a new AND-gate.

**Objection.** “BN is just another layer; absorb \(\theta\) and run the forest.” Counter: BN’s working set is batch statistics over the full ST domain, not a \(0/1\times W\) product.

**Assumptions.** Projection BN is on the same-port critical path (not already a host prepass). Affine is layer-shared enough to fold into mix without per-event payload. Residual add after BN is still the type-R tensor of I2, not absorbed into \(W\).

**Predictions.** (1) Post-fold spike GeMM matches A’s interface (plain bitmaps). (2) BN SRAM traffic becomes one reduction + broadcast, not per-tile affine. (3) AEE stays on ordinary unless stats are faked.

**Disconfirmers.** Captured “actual batch stats” cannot be reduced in one pass without changing AEE past the gate. BN already off the same port. Fold into mix changes T10 CSE last-use enough to break I1/I5 without a 15% win.

---

## Q02-I4 — Patch/epipolar residual nest around a copied bitmap inner kernel

**One sentence.** Outer-schedule a patch-r1 / motion-C12 / H67 / ep34 residual reuse window; inner-kernel is unmodified FireFly AND + Prosperity forest on absorbed spikes.

**A.** FireFly-S bitmap AND as the inner product; Prosperity forest as the inner skip; Gustav CPTB as the inner temporal pack of **spikes**.

**B.** Freeze task is event-camera 2D optical flow (DSEC valid825 AEE), with motion C12 / H67 / ep34 and a historically expensive patch r1 residual chain. Binary-GeMM theorems are neuron-index agnostic: they do not know patches, epipolar lines, or warp-shaped residual reuse. Product sparsity of neighboring pixels is not the same object as residual spatial reuse.

**X.** Nesting is the contribution: the outer window’s operand is the **continuous residual tile** (and its last-use), not a denser bitmap AND. A is copied inside. If the idea “uses flow to prune spikes” by rewriting the forest key, it becomes a reskin of product sparsity and is out.

**Controls.** Inner kernel bit-matches a standalone FireFly/Prosperity GeMM on the same bitmaps; outer window only reorders residual loads/stores and dual last-use with those bitmaps; AEE vs ordinary; same port; no FPS.

**Kill-gate.** Residual SRAM returns in the r1 chain must drop enough that **end-to-end** same-port/backpressure net service \(\ge 15\%\) vs inner A with a naive raster outer schedule. If only inner skip counts improve (Prosperity already counted those), die. AEE abs/relative gates hold.

**Two-sentence pitch.** Flow structure lives in the residual/warp, not in the absorbed 0/1 GeMM the forest already knows how to skip. An outer patch/epipolar window around a copied bitmap kernel is an HPCA dataflow claim those papers’ theorems do not cover.

**Objection.** “Spatial blocking is standard GeMM tiling.” Counter: standard tiling reuses \(s\) and \(W\). This window reuses a **third** tensor (residual) whose geometry is optical-flow structure, and it is dual last-use with the gate, not a GeMM operand.

**Assumptions.** Patch r1 / epipolar neighborhood has residual reuse \(>1\) on this student. Same-port is residual-traffic heavy under backpressure (consistent with 8088 not moving after generic fusion). Inner A’s skip rate is roughly invariant to outer order.

**Predictions.** (1) Residual reuse in r1 windows \(>\) raster. (2) Spike-bitmap reuse may be unchanged (A not the hero). (3) Backpressure net service moves; always-ready may already be partly spent (−22.83%).

**Disconfirmers.** Residual r1 is not spatially reused (random access, reuse \(\approx 1\)). Outer order wrecks Prosperity’s forest locality and **loses** GeMM cycles enough to miss 15% net. AEE fails because windowing changes residual add order/rounding.

---

## Q02-I5 — Mix-contraction unit that emits CPTB (temporal structure is the mix, not the spike train)

**One sentence.** Build a 10-wide pre-threshold contraction (ordinary mix arithmetic) that writes **only** a Gustav-legal spike bitmap plus a residual tensor; mix intermediates die in the unit.

**A.** Gustav NRV/CPTB on the **emitted** bitmap; Prosperity forest + FireFly AND on the subsequent \(0/1\times W\). CPTB is copied, not redesigned.

**B.** Freeze: T10 mix is **noncausal** and happens before threshold. Gustav CPTB packs temporal **spikes** under the assumption that time’s structure **is** the spike train. Here time’s structure is a dense continuous mix (ordinary 260 add/sub; lifting 159 + 35 intermediate RNE/sat) that **then** thresholds. Lifting-raw AEE 1.232979368 vs ordinary 1.219801338 (relative +0.013178: abs 1.259 pass, +0.005 fail) — so the contraction’s arithmetic is accuracy-gated, not a free CSE.

**X.** CPTB’s theorem does not include a dense 10-step mix as a compressor. The new unit’s outputs are `{bitmap, residual}`, never a stored mix cube. Using CPTB to pack ten already-fired binary frames without the contraction is a reskin.

**Controls.** Default arithmetic = ordinary mix (AEE 1.219801338), not lifting-raw. Lifting CSE only if spike bitmap 0-diff vs ordinary (see I8). Absorb on; residual q24 0-diff; A copied on the bitmap; same port.

**Kill-gate.** Contraction must remove mix-cube SRAM from the same-port live set and yield \(\ge 15\%\) backpressure net service vs A that loads a stored mix/spike cube. If lifting is required for the cycle win, AEE relative +0.005 kills it (lifting-raw already fails that gate).

**Two-sentence pitch.** Gustav packs spikes in time; this net mixes time **before** spikes exist. A contraction that emits CPTB-legal bitmaps and a residual tensor copies Gustav after the one place Gustav does not apply.

**Objection.** “NRV already reuses a neuron across time.” Counter: NRV reuses a neuron on **binary** temporal codes. Noncausal mix is continuous all-to-all among T=10 **before** \(s\) exists.

**Assumptions.** Mix cube, if materialized, is a same-port occupant. Ordinary 260 add/sub fit a 10-wide unit without becoming an 18-module design. Residual is produced alongside, not extracted from AT-LIF \(o\).

**Predictions.** (1) Mix-cube traffic → 0; bitmap + residual traffic remain. (2) CPTB density matches ordinary-path spikes. (3) Cycle win is last-use, not add/sub 260→159 (that CSE is not the 15% story).

**Disconfirmers.** Mix cube is not in SRAM (nothing to emit-instead-of-store). Contraction width blows the 5-page / one-mechanism budget. Ordinary mix in the unit misses 15%; lifting in the unit misses AEE.

---

## Q02-I6 — Backpressure drain-first (mix and residual last-use before forest insert)

**One sentence.** Under long backpressure, the issue policy drains mix last-use and residual add **before** inserting another Prosperity-skipped GeMM tile, because the freeze’s stall is live-set occupancy, not unskipped MACs.

**A.** Prosperity forest (greedy product-sparsity insert: more zeros skipped ⇒ fewer GeMM issues). FireFly AND executes whatever the forest inserts.

**B.** Same-port always-ready 6938→5354 (−22.83%, part generic fusion); **long backpressure both 8088**. Separate integer consumer −5.78%. The venue gate is net service under backpressure, not always-ready. Binary-GeMM theorems optimize issued products, not a stalled port holding mix+residual.

**X.** Drain-first is a **port policy** over two non-GeMM live sets (mix-before-threshold, residual). The forest remains the GeMM scheduler when the port is ready. If we “improve sparsity” to beat 8088, we are reskinning Prosperity and the freeze already says both sides sit at 8088.

**Controls.** Identical A datapath; only the issue arbiter changes under the same long-backpressure stimulus that produced 8088. Absorb on; no extra port; AEE untouched (policy must be 0-diff on values).

**Kill-gate.** Long-backpressure cycles must fall \(\ge 15\%\) vs greedy forest on the same stall trace. Value 0-diff vs ordinary (policy is occupancy, not arithmetic). If drain-first equals greedy because GeMM insert was never the occupant, die.

**Two-sentence pitch.** Product sparsity does not retire a mix or residual line that is already sitting on a backpressured port. A drain-first arbiter copies Prosperity when ready and beats it when the freeze’s 8088 stall is live-set, not MAC, bound.

**Objection.** “Ready/valid already prioritizes consumers.” Counter: the measured long-backpressure number is **identical** with and without generic fusion; a consumer-ready policy that does not name mix vs residual last-use is what produced 8088.

**Assumptions.** Under that stall, the port’s head-of-line is mix writeback or residual live, not a nonzero product waiting for AND. Drain is legal (consumers exist now). Policy cannot change spike bits or residual q24.

**Predictions.** (1) 8088 moves; 6938→5354 is not the claimed delta. (2) Forest skip **count** stays ~constant; stall **cycles** drop. (3) Mixing this arbiter with I1/I2 is additive only if they kill different live ranges.

**Disconfirmers.** Stall trace is GeMM-bound (forest insert **is** the occupant). Drain-first deadlocks (drain needs a GeMM result). Any 0-diff break.

---

## Q02-I7 — Two-window release protocol (S-gate last-use frees R residual)

**One sentence.** Architecturally couple the freeze’s two integer windows: window-S is copied FireFly/Prosperity on absorbed gates; window-R is q24 residual/PED/I24; last-use of a gate bit is the **release** that allows overwrite of the matching residual line.

**A.** FireFly bitmap GeMM + Prosperity forest entirely inside window-S. Gustav CPTB may pack S-bits. Window-R is not claimed as a GeMM.

**B.** Two-window integer gates/I24/PED q24 is **0-diff vs model**. Dual last-use: consumer A = binary gate after absorb; consumer B = continuous residual. Papers have one consumer (the MAC). Separate integer consumer without release is the −5.78% model — below 15%.

**X.** The protocol is a **coherence/release** between a binary bitmap and a continuous tensor of different type, not an AND of two bitmaps and not a second forest. Reskin test: if window-R is rewritten as int8 spike payload or unabsorbed \(\theta_g\), reject.

**Controls.** Arithmetic 0-diff vs the two-window integer model; absorb on; window-S = copied A; same port/state; do not add consumer tables; no unabsorbable int8.

**Kill-gate.** Release must shorten residual occupancy enough for \(\ge 15\%\) same-port/backpressure net service vs two windows that only sit side-by-side (−5.78% class). Break 0-diff ⇒ die. Identity break (R treated as AT-LIF amplitude) ⇒ die.

**Two-sentence pitch.** The freeze already has a bit-exact two-window integer split; the missing architecture is the release that ties gate last-use to residual overwrite. That coupling is dual last-use, which binary-GeMM theorems never state.

**Objection.** “Free-on-last-read is basic SRAM.” Counter: last-read of **which** object? Prosperity frees a product. Here S-last-use frees **R**, a tensor A does not name.

**Assumptions.** A residual line is live until its matching gate has been consumed as a binary select (dual consumer). Release is same-port visible. q24 residual is not secretly equal to \(\theta s\).

**Predictions.** (1) Residual line lifetime tracks S-consumption, not S-generation. (2) Occupancy win \(\gt\) −5.78% separate-consumer. (3) Window-S skip rate equals standalone Prosperity on the same gates.

**Disconfirmers.** Residual last-use is independent of gate consumption (release never fires). Coupling adds scoreboard stalls that cancel the occupancy win. Implementers store residual **as** the spike (identity violation).

---

## Q02-I8 — Bitmap-invariant mix (CSE only if ordinary spikes 0-diff)

**One sentence.** Constrain T10 mix CSE/rounding so the post-threshold, post-absorb spike bitmap equals the ordinary-path bitmap; residual stays q24 0-diff; then A’s forest skip set is an invariant, not a side effect.

**A.** Prosperity product-sparsity **skip set** is a function of the 0/1 bitmap. FireFly AND and Gustav CPTB see that same bitmap. Copy A on the invariant bitmap.

**B.** Lifting CSE: 260→159 add/sub plus 35 intermediate RNE/sat, but AEE 1.232979368 vs 1.219801338 (relative +0.013178, +0.005 fail). Mix rounding is **before** threshold, so it can flip \(H(m-\theta)\) and therefore flip every binary-GeMM skip. Those papers do not have a pre-threshold rounding contract. Ordinary mix is the AEE-legal path.

**X.** This is a **bitmap-invariance contract** on continuous mix arithmetic, not a new GeMM. Prosperity cannot state “CSE the 10-mix but keep \(s\).” If we ship lifting-raw and let the forest adapt to new spikes, we reskin A and fail the AEE relative gate.

**Controls.** Golden bitmap = ordinary path; golden residual = two-window q24 (0-diff); same port; A copied; lifting CSE admitted only as a candidate rewrite, not as the default.

**Kill-gate.** Any mix rewrite that is not spike-bitmap 0-diff vs ordinary is killed **even if** AEE abs \(\le 1.259\). Cycle claim: CSE/occupancy from the invariant mix must contribute to \(\ge 15\%\) same-port/backpressure net service vs ordinary mix + copied A. If invariance holds but CSE last-use was never on the port, die (then I1/I5/I6 must carry occupancy, not I8).

**Two-sentence pitch.** Lifting already shows that cheaper mix arithmetic changes AEE by more than the letter allows, because it can change the binary spikes A is designed to skip. A mix unit that is CSE-reduced only under a 0-diff bitmap contract lets the letter copy Prosperity honestly.

**Objection.** “Small AEE deltas are fine; 1.232 is still \(\lt 1.259\).” Counter: the freeze’s relative gate is +0.005; lifting-raw is +0.013. Abs-only is a known fail mode of this exact number.

**Assumptions.** Ordinary vs lifting differ in intermediate RNE/sat (35 sites), i.e. rounding, not a different algorithm family. A measurable subset of CSE is bitmap-invariant. Residual q24 can stay 0-diff independently of mix CSE (different tensor).

**Predictions.** (1) Invariant subset of lifting CSE \(< 101\) add/sub saved (260−159), possibly much less. (2) Where invariance holds, Prosperity skip set matches ordinary exactly. (3) Occupancy win, if any, is from fewer mix writes, not from a different forest.

**Disconfirmers.** No CSE remains once bitmap 0-diff is enforced (empty rewrite). Invariance on spikes but not on residual (q24 breaks). Invariance holds and CSE is real, but same-port backpressure is GeMM-bound (I8 is then a numerical lemma, not the 15% architecture).

---

## Cross-idea notes (not extra modules)

- **A is always instantiated.** Ideas that “beat Prosperity by skipping more zeros” are out of this file.
- **Occupancy vs arithmetic.** I6/I2/I7/I1 target the 8088 backpressure hole. I8/I5 target T10 mix legality (lifting-raw already fails +0.005). I3 is BN. I4 is flow geometry on residual.
- **Do not stack all eight.** A 5-page letter can carry one occupancy policy (I1 or I2 or I6 or I7) plus one legality constraint (I8) if needed; I3/I4/I5 are alternative X objects, not a checklist.
- **Identity tests.** Spike path after absorb is \(0/1\times W\). Residual is not AT-LIF amplitude. No HBG-RP int8 payload. No CIM.
