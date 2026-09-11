# Q01 — TCAS-II circuits (round-2 independent ideas)

Status: **ideas, not findings**. Identity locked: AT-LIF \(o=\theta\cdot H(m-\theta)\in\{0,\theta\}\); inference absorbs layer-shared \(\theta\) by \(W\leftarrow\theta W\); after absorb the spike path is binary \(\{0,1\}\times W\). Residual / PED / I24 is a different continuous tensor. Noncausal T10 mix (PSN / lifting CSE) is continuous arithmetic **before** threshold, then threshold, then absorb. No continuous-θg contribution, no non-absorbable int8, no CIM, no 18-module map.

Numeric rails used as kill-gates (from freeze, unchanged measurements):

| rail | number |
|---|---|
| ordinary AEE | 1.219801338 |
| AEE abs cap | ≤ 1.259 |
| AEE relative cap | ≤ +0.005 → AEE ≤ **1.224801338** |
| lifting raw AEE | 1.232979368 (rel +0.013178: abs pass, rel fail) |
| CSE ordinary | 260 add/sub |
| CSE lifting | 159 add/sub + 35 intermediate RNE/sat |
| same-port always-ready | 6938 → 5354 (−22.83%, part generic fusion) |
| same-port long backpressure | **both 8088** (generic fusion did not move this) |
| 15% of 8088 | ≤ **6874** cycles |
| integer consumer ops | 758777 → 714889 (−5.78%); **not** a table, **not** the 15% gate |
| two-window gates / I24 / PED q24 | 0-diff vs model |

Hard service gate is long-backpressure same-port/state, not always-ready (already −22.83% by generic fusion) and not integer-op count (−5.78%).

---

## Q01-I1 — Bound-kill CSE: decide \(H(m-\theta)\) before the T10 mix finishes

- **id:** Q01-I1
- **one sentence:** A remaining-sum bound circuit on the pre-threshold T10 mix kills leftover add/sub as soon as \(m\gtrless\theta\) is decided, then the post-absorb path is an unmodified binary GeMM engine.
- **A:** Copy whole onto \(\{0,1\}\times W\) after absorb: Prosperity product-sparsity skip, FireFly-S bitmap AND, Gustav NRV/CPTB pack, LoAS dual-sparse temporal-parallel dataflow. No analog payload on this bus.
- **B:** Freeze: T10 mix is continuous and **before** threshold (ordinary 260 add/sub; lifting 159 + 35 RNE/sat). Priors start only after a spike bitmap exists. Lifting-raw already fails the relative AEE rail, so a cheaper mix is not free.
- **X:** Product sparsity is a property of spike×weight after absorb. Early-out is a comparator-plus-kill on real mix terms versus layer-shared \(\theta\), which never enters Prosperity/Gustav/FireFly/LoAS. Residual/PED is not this mix.
- **controls:** (i) full ordinary CSE + threshold + absorb + A; (ii) bound-kill CSE + same threshold/absorb + A; (iii) lifting-raw + A (known AEE-fail reference); (iv) A off (dense binary GeMM) to isolate mix vs GeMM.
- **kill-gate:** AEE ≤ 1.224801338 and ≤ 1.259 on DSEC valid825. If the idea claims exact Heaviside, spike 0-diff vs full ordinary mix+threshold. Mix add/sub < 260 on the measured student (else no circuit vs ordinary CSE). Same-port long-backpressure cycles ≤ 6874; always-ready ≤ 5354. Kill if bound-kill does not reduce mix ops **and** long-BP stays 8088 (mix-local 260 is ~3% of 8088; 15% cannot be claimed from CSE count alone unless mix spills share that port).
- **two-sentence pitch:** After \(\theta\)-absorb the letter is allowed to be a binary SNN accelerator; the only arithmetic those accelerators do not already own is the T10 mix that **creates** the bits. A bound-kill adder/comparator is a TCAS-II block, not a new GeMM format.
- **objection:** 260 mix adds are too small to move 8088; reviewers will say this is a software CSE flag, not a circuit.
- **assumptions:** Mix terms have usable magnitude structure (layer-shared \(\theta\), motion/PSN-like sums). Remaining-sum bounds are cheaper than the killed adds (max-norm or bit-width, not another 260-deep exact sum). Mix intermediates currently compete for the same port/state as GeMM/residual under long-BP, or the letter explicitly does **not** claim the 15% from this block and pairs it with A for service.
- **predictions:** Ordinary-path spike map matches (exact bound) or AEE stays inside +0.005 (inexact bound). Mix add/sub drops. Binary GeMM cycle count equals A-alone. Long-BP moves only if mix spills were on that port.
- **disconfirmers:** Bounds too loose → add/sub stays ~260. Any AEE > 1.224801338. Long-BP remains 8088 and the pitch still claims 15% from CSE. Spike path secretly carries \(\theta\) (identity break).

---

## Q01-I2 — Dual-color last-use FSM for binary GeMM and residual/PED

- **id:** Q01-I2
- **one sentence:** A two-color retirement FSM holds a same-port SRAM line until **both** the post-absorb binary GeMM consumer and the residual/PED consumer have used it, then frees the port.
- **A:** FireFly-S bitmap AND + Prosperity product skip + Gustav NRV/CPTB on consumer A only (the 0/1 gate after absorb). LoAS temporal-parallel issue on that same binary stream.
- **B:** Freeze splits two objects: spike GeMM after absorb vs residual/PED/I24. Dual consumers, if they exist, are **binary gate after absorb** and **continuous residual**, not two MACs sharing analog AT-LIF amplitude. Same-port long-BP is 8088 on both measured nets; generic fusion already spent the always-ready −22.83%.
- **X:** FireFly/Gustav last-use is spike-buffer lifetime. Residual/PED is another tensor. Spike-only retire either drops residual (AEE/0-diff death) or over-holds the line (port death). That second color is not a reskin of bitmap AND.
- **controls:** spike-only last-use vs residual-only vs dual-color; always-ready vs long-BP; two-window I24/PED q24 0-diff on/off as oracle; A on/off.
- **kill-gate:** Two-window I24/PED q24 remains 0-diff vs model. AEE ≤ 1.224801338 and ≤ 1.259. Long-BP same-port cycles ≤ 6874. Always-ready ≤ 5354 (do not give back generic fusion). Kill if dual-color is only extra state bits and long-BP stays 8088 — that is the already-measured generic-fusion failure.
- **two-sentence pitch:** The shared SRAM port is the letter resource; the circuit is the retirement FSM that knows two different tensors. Priors never had a continuous residual consumer glued to a binary bitmap engine.
- **objection:** Dual last-use is reference counting / bookkeeping, too thin for TCAS-II even at 5 pages.
- **assumptions:** The student actually dual-consumes one live line (projection/residual chain, historically r1 ~35% of activity-weighted dots; proxy ≠ new cycle share). Under long-BP, hold time of that line is a first-order term in 8088, not hidden behind unrelated waits.
- **predictions:** Spike-only last-use either mismatches q24 residual or reproduces 8088. Dual-color cuts hold/re-fetch enough to take long-BP ≤ 6874 without touching AEE. Always-ready stays at or below 5354.
- **disconfirmers:** Consumers already strictly sequential → 0% port win. Long-BP still 8088. 0-diff breaks. Design stores analog \(o=\theta s\) for the GeMM consumer (identity break).

---

## Q01-I3 — End-point RNE lifting CSE (make 159-add mix AEE-legal)

- **id:** Q01-I3
- **one sentence:** Keep lifting’s 159 add/sub factoring on the pre-threshold T10 mix but move the 35 intermediate RNE/sat to a **single end-point** round (optional 1–2 guard bits) so the value presented to \(H(\cdot-\theta)\) matches ordinary CSE.
- **A:** After threshold and \(\theta\)-absorb, copy Prosperity/Gustav/FireFly/LoAS whole on binary GeMM. A is not modified to “fix” lifting.
- **B:** Lifting raw AEE 1.232979368 is +0.013178 vs 1.219801338: abs 1.259 **pass**, relative +0.005 **fail**. Source CSE is the hole: ordinary 260 vs lifting 159 + 35 intermediate RNE/sat. Those 35 rounds sit **before** threshold, so they change which bits A will see.
- **X:** FireFly/Prosperity never round a lifting graph. This is a rounding-placement circuit on continuous pre-threshold arithmetic, not a bitmap format.
- **controls:** ordinary 260; lifting-raw 159+35 mid-RNE; lifting-endRNE 159+1 end-round; lifting-endRNE+g guard bits \(g\in\{1,2\}\); each followed by identical threshold, absorb, A.
- **kill-gate:** AEE ≤ 1.224801338 (must recover at least 0.008178 from lifting-raw 1.232979368) and ≤ 1.259. Mix add/sub < 260 (else ordinary already wins). If claiming exactness: 0-diff vs ordinary mix at the threshold input, or spike 0-diff. Long-BP ≤ 6874 and always-ready ≤ 5354 **must** be delivered by A (binary GeMM) plus any mix-spill reduction; kill if the write-up uses 159 vs 260 as the 15% service number (101/8088 ≈ 1.25%). Kill if end-point RNE still sits at 1.232979368.
- **two-sentence pitch:** Lifting already lost the letter on the relative AEE rail because it rounded 35 times too early. One end-point RNE block makes the cheap mix legal, then the absorb-binary stack is a full prior rather than a reskin that inherits a failing AEE.
- **objection:** Rounding-mode on 35 nodes is a numerical-analysis note, not a circuit; 15% service is still entirely A.
- **assumptions:** Intermediate RNE/sat is the dominant cause of the +0.013178, not the algebraic lifting identity. Guard bits fit the existing integer width next to q24 PED (two-window 0-diff path stays untouched). Service 15% is claimed from A on the post-absorb port, with this block only as the AEE gate that makes a 159-add mix admissible.
- **predictions:** End-point RNE brings AEE to ≤ 1.224801338 while add/sub stays 159+O(1). Spike bitmap matches ordinary (exact) or AEE-legal (guarded). A’s GeMM sparsity numbers are unchanged vs ordinary-mix+A.
- **disconfirmers:** Error is in the lifting algebra, not the 35 rounds → AEE stays ~1.233. Guard bits blow width until mix is slower than 260 ordinary. Letter claims 15% from 159/260.

---

## Q01-I4 — Captured-BN fold + in-place residual add on the projection line

- **id:** Q01-I4
- **one sentence:** Freeze the captured projection BN (actual batch stats over 10×96×120×160) into affine \(y=ax+b\), fold \(a\) into the projection weights (then \(\theta\)-absorb into the **next** \(W\)) and \(b\) into the residual add, and perform residual add **in-place** on one SRAM line so one port visit feeds both later binary gating and the continuous residual tensor.
- **A:** FireFly-S / Prosperity / Gustav / LoAS run on the 0/1 bits **after** this in-place add and AT-LIF threshold/absorb. Fold does not put \(\theta\) into the residual tensor.
- **B:** Native projection conv + BN + residual add exist. BN is full-width. Patch r1 residual chain is historically expensive (old proxy ~35% of activity-weighted dots; proxy ≠ new cycle share). Residual/PED is not the spike GeMM. Long-BP 8088 means 15% must come from **fewer port visits**, not from overlapping compute (always-ready already −22.83%).
- **X:** Prosperity/FireFly never fold a dense BN or in-place residual. Those ops are on the continuous projection/residual tensors **before** threshold. Textbook CNN BN-fold becomes a letter circuit only because the same line has a second consumer that is a **binary gate after absorb**, so fold+in-place changes same-port last-use, not just compiler graphs.
- **controls:** BN live vs folded; residual out-of-place vs in-place; absorb on (mandatory) vs illegally off; A on/off; q24 two-window 0-diff as oracle.
- **kill-gate:** Folded BN + in-place add remains 0-diff vs captured BN+add at two-window q24, **or** AEE ≤ 1.224801338 if 0-diff is abandoned (must say which). AEE ≤ 1.224801338 and ≤ 1.259 either way. Long-BP ≤ 6874. Always-ready ≤ 5354. Projection+BN+residual **port transactions** under the long-BP test drop ≥ 15% vs unfused (must show on 8088, not on integer-op 758777). Kill if fold is compile-time only and 8088 is unchanged.
- **two-sentence pitch:** The port-bound student is projection–BN–residual, not the binary GeMM. Fold the captured BN and add residual in-place so one visit retires both the continuous tensor and the bit that FireFly will AND; that is a same-port circuit, not an SNN format paper.
- **objection:** Inference BN fold is textbook; in-place add is a one-line SRAM FSM; reviewers will send it to a compiler workshop.
- **assumptions:** Captured batch stats over 10×96×120×160 are **frozen constants** at inference (not online per-batch \(\mu,\sigma\)). Residual is pointwise on the projection layout. \(\theta\)-absorb still applies only to the next-layer spike GeMM, never to residual/PED. Long-BP 8088 has a large share of visits from this chain (the 35% proxy is a prior, not a cycle claim).
- **predictions:** \(y=ax+b\) fold is 0-diff at q24. In-place add cuts long-BP visits ≥ 15%. Post-absorb spike bits match baseline if threshold sees the same folded sum. A’s product sparsity is unchanged.
- **disconfirmers:** BN is truly online over each 10×96×120×160 batch → fold illegal. Residual layout ≠ projection. Long-BP waits are elsewhere → 8088 unchanged. Fold accidentally scales residual by \(\theta\) (identity break).

---

## Q01-I5 — Residual-first credit arbiter; FireFly eats leftover same-port credits

- **id:** Q01-I5
- **one sentence:** Under long backpressure, a credit FSM gives the dense residual/PED consumer first claim on the shared port and lets the post-absorb FireFly/Prosperity engine issue only on leftover credits, with dual last-use still binding retirement.
- **A:** Full FireFly-S bitmap AND, Prosperity product skip, Gustav NRV/CPTB, LoAS dual-sparse schedule — but they may issue only with leftover credits. No dropped products: leftover credits plus product sparsity must still complete the binary GeMM.
- **B:** Always-ready fusion already 6938→5354 (−22.83%, part generic). Long-BP **both 8088**. Residual/PED is a different dense tensor; r1 residual chain was historically a large activity-weighted share. Generic fusion treated two consumers as one stall domain.
- **X:** Prosperity’s skip is **inside** GeMM. It does not arbitrage a scarce SRAM port against a dense continuous residual tensor that is not a spike bitmap. The arbiter + dual last-use is the circuit; the bitmap engine is the copied prior.
- **controls:** equal-priority (generic fusion) vs residual-first vs GeMM-first; credit depth \(c\in\{1,2,4\}\); dual last-use on/off; always-ready vs long-BP; A on/off (dense GeMM with residual-first should starve or blow cycles).
- **kill-gate:** Long-BP ≤ 6874. Always-ready ≤ 5354. AEE ≤ 1.224801338 and ≤ 1.259. Two-window I24/PED q24 0-diff. Binary GeMM **completeness**: 0 missing products vs A-alone (starvation = kill, even if cycles look good). Kill if GeMM-first matches residual-first (arbiter is a no-op) or if equal-priority already would have moved 8088 (contradicts freeze).
- **two-sentence pitch:** The freeze’s failure mode is not “need more sparsity”; it is a dense AEE-critical residual glued to a sparse binary GeMM on one port when the port is rarely ready. Residual-first credits plus leftover product-sparsity is a 5-page arbiter, not a new neuron.
- **objection:** Priority arbiters are textbook; leftover-credit plus sparsity is just QoS naming.
- **assumptions:** Under the long-BP test the two consumers’ ready/need patterns differ (residual dense/must-issue; GeMM product-sparse/skippable in time). Dual last-use from Q01-I2 is available or reimplemented. A is complete: every nonzero spike×weight still executes.
- **predictions:** Residual-first + A takes long-BP ≤ 6874; GeMM-first either hurts AEE (residual late) or stays ~8088; equal-priority stays 8088 (freeze). 0-diff and AEE hold. Always-ready does not regress past 5354.
- **disconfirmers:** Both consumers assert need on every ready cycle of the 8088 test → arbiter cannot save a visit. GeMM starves (nonzero missing products). AEE > 1.224801338. Cycle win appears only in always-ready.

---

## Q01-I6 — Temporal XOR-delta issue of post-absorb bitmaps; residual stays dense

- **id:** Q01-I6
- **one sentence:** Cache timestep \(t{-}1\)’s post-absorb spike bitmap and issue FireFly AND / Prosperity skip only where the XOR with timestep \(t\) is 1; the residual/PED path remains a full dense tensor each \(t\) because that is where optical-flow motion lives.
- **A:** FireFly-S bitmap AND and Prosperity product sparsity on the XOR-1 locations; Gustav NRV/CPTB may pack the delta bitmap; LoAS time-parallel engine is the **contrast**, not the delta (LoAS parallelizes independent timesteps, it does not incrementalize identical bits).
- **B:** Task is event-camera 2D flow, DSEC valid825 AEE, motion slices C12 / H67 / ep34. After absorb the spike path is binary, so successive bitmaps are legally XOR-able. Residual/PED is another tensor and must not be silently delta’d.
- **X:** FireFly/Prosperity/LoAS treat each \(t\) as an independent dual-sparse GeMM. The split “incremental binary spikes / dense residual” is a **task tensor split** (flow), not a neuron-format trick and not product sparsity itself.
- **controls:** per-\(t\) full bitmap vs XOR-delta vs illegal residual-delta; A on/off; report AEE overall **and** on C12 / H67 / ep34; XOR density = Hamming\((s_t,s_{t-1})/N\).
- **kill-gate:** AEE overall ≤ 1.224801338 and ≤ 1.259; same rails on C12 / H67 / ep34 (no hiding motion error in the mean). XOR-delta is algebraically exact on the spike path if the cache holds the true previous bitmap: spike 0-diff vs full issue. Long-BP ≤ 6874; always-ready ≤ 5354. Kill if mean XOR density ≥ 0.90 of full bitmap density on valid825 (no issue reduction). Kill if residual is also delta’d unless a separate 0-diff/AEE gate is passed (default: residual-delta is out of scope).
- **two-sentence pitch:** Optical flow is the reason successive 0/1 maps are sticky while the residual is not. Copy FireFly on the bits that actually change; leave PED/I24 dense; that split is the letter, not another bitmap AND.
- **objection:** Temporal spike reuse is an SNN classic; LoAS already “does time.”
- **assumptions:** After absorb, Hamming distance across \(t\) is low on this student (flow-smooth, not a reset-every-tick encoder). Residual/PED is **not** similarly sparse. Spike-side visits are a large enough fraction of 8088 that delta-issue can move 15% **without** thinning residual. Cache of one bitmap fits the same-port state budget.
- **predictions:** XOR density ≪ spike density. GeMM port traffic drops; residual port traffic unchanged. Net long-BP ≤ 6874 iff spike-side was a large share. AEE and motion slices stay inside +0.005 because the spike path is exact.
- **disconfirmers:** Event-driven student re-spikes a new pattern each \(t\) (XOR density high). Residual dominates 8088 → delta-issue cannot reach 15%. Any implementation that deltas PED/I24 blows AEE. LoAS-style time-parallel already issues the same products (no incremental win).

---

## Q01-I7 — Issue-locked dual accumulator: binary select-add + q24 PED/I24

- **id:** Q01-I7
- **one sentence:** One PE, two accumulators, one issue pointer: Prosperity/FireFly binary select-add for absorbed spikes, and a q24 integer add for I24/PED, so dual last-use is the same beat rather than two state machines.
- **A:** Binary accumulator is exactly Prosperity product-sparsity select-add / FireFly AND-gate MAC / Gustav packed NRV operand. No analog \(\theta\) on that adder.
- **B:** Two-window integer gates/I24/PED q24 is already 0-diff vs model — the residual path has a **proven** integer circuit contract. Separate integer consumer model is only 758777→714889 (−5.78%); freeze says do not add tables. That 5.78% is **below** the 15% service gate, so the idea is illegal if it reports ops.
- **X:** Prosperity’s PE is spike×W only. A second q24 residual accumulator welded to the **same issue pointer** is a dual-tensor circuit. The novelty is shared state/backpressure, not integer-op reduction those priors never counted.
- **controls:** separate issue (the −5.78% world) vs locked issue; q24 vs wider residual; A on/off; two-window 0-diff oracle; always-ready vs long-BP.
- **kill-gate:** Keep two-window I24/PED q24 0-diff. AEE ≤ 1.224801338 and ≤ 1.259. Same-port/state long-BP ≤ 6874 **and** always-ready ≤ 5354. **Forbidden win:** quoting 758777→714889. Kill if locked-issue cycle counts equal already-measured generic fusion (5354 / 8088): that is a reskin of fusion, not a PE. Kill if the only saved resource is integer ops < 15%.
- **two-sentence pitch:** The letter PE is not an analog AT-LIF MAC; it is a binary select-add welded to a 0-diff q24 residual adder on one issue pointer. That weld is the circuit; the −5.78% op count is a non-result the freeze already has.
- **objection:** Dual-accumulator DSP PEs are 1990s; 5.78% already says compute is not the bottleneck, so a PE diagram cannot hit 15% same-port.
- **assumptions:** Same-port/state cycles are **issue/index dominated** when the two consumers walk separately (two pointers, two last-use FSMs). Locking issue halves that state traffic under long-BP, not the MAC energy. Residual add is pointwise-aligned with the GeMM reduction’s spatial index often enough to share a pointer; misaligned channels fall back to split issue (must be measured, not assumed 100%).
- **predictions:** State/index/port transactions drop ≥ 15% under **both** always-ready and long-BP. 0-diff holds. AEE holds. Integer-op count may still show ~5.78% and is not reported as the win.
- **disconfirmers:** Issue pointers were already shared in generic fusion → 5354/8088 reproduced. Residual index ≠ GeMM index (alignment < the 15% budget). Reviewers read it as the 5.78% table. Analog \(o\) sneaks into the binary adder.

---

## Q01-I8 — Split-layout packer: q24 residual words vs FireFly bitmap view of the same lines

- **id:** Q01-I8
- **one sentence:** A packer/unpacker circuit presents one physical SRAM row as a q24 residual/PED (or pre-threshold mix scratch) **and** as a packed 0/1 bitmap for FireFly, so the two freeze tensors share visits without sharing a MAC.
- **A:** FireFly-S expects a packed spike bitmap for AND with weight bitmaps; Prosperity/Gustav/LoAS consume that packed view only. They do not see q24 residual lanes.
- **B:** After absorb, consumer A wants N packed bits; consumer B wants N q24 words. T10 mix before threshold may also need a small scratch (ordinary 260 / lifting 35 intermediates). Those layouts fight on the same port. Long-BP 8088 is the measured fight; always-ready generic fusion did not fix it.
- **X:** FireFly’s contribution **is** the bitmap. The hole is that this net also has a dense residual/PED tensor and a pre-threshold mix that are **not** bitmaps. A packer that preserves dual last-use is not a reskin of AND; AND is A, packing two freeze objects onto one row is X.
- **controls:** two arrays (q24 + bitmap) vs packed dual-view; packer before vs after threshold (after is identity-legal: bit = \(H(m-\theta)\), residual stays q24); mix intermediates in a 35-slot RF vs spilled to the same SRAM; A on/off.
- **kill-gate:** Packed dual-view is 0-diff on two-window I24/PED q24 **and** spike 0-diff vs unpacked bitmap. AEE ≤ 1.224801338 and ≤ 1.259. Long-BP ≤ 6874; always-ready ≤ 5354. Same-port **reads+writes** of residual+bitmap+mix-scratch drop ≥ 15% vs two-array baseline under the 8088 test. Kill if the packer adds a visit (pack then consume) so 8088 grows. Kill if mix scratch is claimed as 15% from 35 registers alone.
- **two-sentence pitch:** FireFly wants bits packed; residual wants q24; the letter is the packer that lets one row serve both without a second SRAM port. That is a circuit on the freeze’s two objects, not a third GeMM format.
- **objection:** Layout/packer is a compiler/memory-system note; TCAS-II reviewers may call it wiring.
- **assumptions:** Dual consumers of the same spatial line exist (gate after absorb + residual, or mix output + later residual). Packer width matches the existing SRAM word (q24 + 1 bit, or bit-serial pack of 24 gates per residual word — must pick one and freeze it). Mix 35 intermediates, if included, live in a tiny RF and **stop competing** for the FireFly/residual port.
- **predictions:** Dual-view cuts residual+bitmap visits ≥ 15% under long-BP. 0-diff on both tensors. FireFly AND rate equals A-alone (same bits, better layout). 35-slot mix RF removes CSE spills from 8088 but is not the 15% story by itself.
- **disconfirmers:** Residual and spike-gate do not share spatial lines → packer is a transpose tax. Packer extra pass ≥ the visits it saves. Long-BP still 8088. Pre-threshold continuous values packed as if they were post-absorb bits (identity break).

---

## Cross-idea notes (still ideas)

- **A is copied, not re-derived.** Every idea above treats Prosperity product sparsity, Gustav NRV/CPTB, FireFly-S bitmap AND, and LoAS dual-sparse time-parallel dataflow as the **complete** post-absorb binary GeMM stack. X is only: T10 mix before \(\theta\) (I1, I3, I8), residual/PED continuous path (I2, I4, I5, I7, I8), full-width captured BN (I4), dual-path last-use (I2, I5, I7, I8), optical-flow task structure (I6, and residual-first I5).
- **15% is long-BP 8088→≤6874**, not always-ready 6938→5354 and not integer 758777→714889.
- **AEE binding rail is relative +0.005** (1.224801338), tighter than abs 1.259. Lifting-raw is a known fail on that rail.
- **Identity kills:** analog \(\theta g\) on the GeMM bus; non-absorbable int8 payload; skipping \(W\leftarrow\theta W\); calling residual “AT-LIF amplitude.”
- **Do not build all eight.** I2+I5+I8 are the same-port family; I1+I3 are the pre-threshold family; I4 is BN/residual; I6 is task-temporal; I7 is the PE/state weld. A 5-page letter can carry **one** family plus copied A.
