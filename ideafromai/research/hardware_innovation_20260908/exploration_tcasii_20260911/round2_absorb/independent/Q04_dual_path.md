# Q04 — dual-path last-use (independent, round 2 absorb)

Perspective: consumer A = binary spikes after absorb; consumer B = continuous residual / PED / I24. They are different tensors. Focal objects: dual last-use, full-width BN barrier, same-port 8088.

Identity used, not argued: AT-LIF \(o\in\{0,\theta\}\); inference absorbs layer-shared \(\theta\) into next \(W\); after absorb the spike path is binary select-add. Residual / PED / I24 is not analog AT-LIF amplitude. Pre-threshold T10 mix is continuous, then threshold, then absorb.

A on every idea = copy the complete post-absorb binary GeMM prior (Prosperity product sparsity / Gustav NRV·CPTB / FireFly-S bitmap AND / LoAS loop occupancy). X is only an object those priors do not contain.

6–8 ideas. Not 18 modules. Not non-absorbable int8. Not “continuous \(\theta g\)”.

---

## Q04-I1 Dual last-use is \(\max(t_A,t_B)\), which is the 8088

**id:** Q04-I1

**one sentence:** After absorb, last-use of a shared live window is the later of binary GeMM completion (A) and residual/PED/I24 completion (B); GeMM-only last-use cannot move the long-backpressure 8088.

**A:** Prosperity / Gustav / FireFly / LoAS last-use: free the spike tile when the next-layer binary select-add finishes.

**B:** Freeze keeps two objects distinct (binary gate after absorb vs continuous residual/PED/I24). Same-port always-ready drops 6938→5354 (−22.83%, part generic fusion); long backpressure is **both** 8088. Ordinary vs lifting CSE (260 vs 159+35) does not change that 8088, so the stall is not mix-op count. If A and B share port/state, last-use \(=\max(t_A,t_B)\) and long backpressure saturates the max.

**X:** Those priors have one consumer of spikes. They have no second last-use bit on a residual tensor. Copying bitmap lifetime on A is mandatory and does not touch \(\max(t_A,t_B)\).

**controls:** Same-port, same state, same backpressure law. Compare A-only last-use vs \(\max(t_A,t_B)\). Always-ready vs long-backpressure. Do not change the numeric path (AEE).

**kill-gate:** If an explicit dual last-use token does not cut long-backpressure net service off 8088 by ≥15% at iso-state, or AEE exceeds abs 1.259 / rel +0.005 vs 1.219801338, kill.

**two-sentence pitch:** The letter’s hardware claim is a two-bit last-use, not a sparser binary GeMM. 8088 equality under backpressure is the measurement that A-only occupancy priors are the wrong last-use.

**objection:** “8088 is just a full pipeline; add Prosperity and the GeMM drains.” Then A-sparsity must move 8088; the freeze says both designs stay at 8088.

**assumptions:** Shared port/state is the letter config. B remains live after A’s GeMM would have retired the tile. The old r1 ~35% activity-weighted-dot proxy is not treated as the new cycle share.

**predictions:** Always-ready keeps a large A-side gap; long-backpressure gap stays ~0 until last-use becomes \(\max(t_A,t_B)\) with split retire. Splitting last-use without changing W or \(\theta\) leaves AEE at the ordinary number.

**disconfirmers:** Long-backpressure 8088 splits once A-only FireFly/Prosperity is turned on. Independent B last-use does not change occupancy. AEE moves when only last-use metadata changes.

---

## Q04-I2 Full-width projection BN is the dual-path barrier

**id:** Q04-I2

**one sentence:** Captured projection BN over \(10\times96\times120\times160\) is a dense continuous barrier on B; absorbing \(\theta\) into \(W\) and skipping zeros on A does not retire BN or the residual add that waits on it.

**A:** Post-absorb channel scale in \(W\leftarrow\theta W\) plus Prosperity/FireFly skip-zero on the binary GeMM.

**B:** Native projection conv + BN + residual add exist. Captured BN uses **actual batch stats** over \(10\times96\times120\times160\), not a per-spike scale. B (residual add) cannot complete until that reduction produces the continuous tensor. A’s binary skip does not skip B’s BN. Last-use of the projection buffer is BN-complete, not GeMM-complete. Under same-port backpressure that barrier is a candidate for the 8088 hold.

**X:** Prosperity/Gustav/FireFly/LoAS do not host a full-width batch-stat BN between spike GeMM and residual add. Folding \(\theta\) into \(W\) is the locked identity; batch-stat BN is a different tensor and a different lifetime.

**controls:** Batch-stat BN vs identity-BN vs frozen running-stat folded into absorbed \(W\). Measure last-use of projection vs spike bitmap. Same-port 8088. AEE vs 1.219801338.

**kill-gate:** If BN folds into absorbed \(W\) at 0-diff AEE and last-use collapses to A-only, the barrier claim dies. If BN-aware last-use does not buy ≥15% net service under long backpressure, kill for the letter.

**two-sentence pitch:** Absorb removes \(\theta\) from the spike path; it does not remove the batch BN that residual add waits on. The 5-page X is BN as a dual-path last-use barrier, not BN-as-PPA.

**objection:** “Inference BN is running stats, fold it like \(\theta\).” Freeze captured actual batch stats on that \(10\times96\times120\times160\) map; folding is a control, not the default identity.

**assumptions:** The captured BN is on the projection/residual path (B), not on the post-absorb 0/1 GeMM (A). Spatial reduction for batch stats cannot be tile-killed the way a FireFly bitmap is killed.

**predictions:** A-only sparsity leaves BN-buffer last-use unchanged. Replacing batch BN by folded running stats, if 0-diff, should also collapse 8088 toward the always-ready regime; if AEE breaks, BN cannot be folded and remains X.

**disconfirmers:** BN already folded in the student with 0-diff. BN last-use ends before A’s GeMM. Long-backpressure 8088 is independent of BN on/off.

---

## Q04-I3 The 22.83% / 5.78% / 0% gradient is B holding the port A already made sparse

**id:** Q04-I3

**one sentence:** Always-ready −22.83% is A-side generic fusion; separate integer consumers only −5.78%; long backpressure both 8088 — B’s continuous traffic occupies the same port after A has nothing left to skip.

**A:** Complete Prosperity/Gustav/FireFly/LoAS on post-absorb binary GeMM.

**B:** Freeze numbers, same-port: always-ready 6938→5354 (−22.83%, part generic fusion); long backpressure **both** 8088; separate integer consumer model 758777→714889 (−5.78%, do not add tables). That gradient is the hole: A-side fusion appears when the port is free; under backpressure service is identical; modeling B as a separate integer consumer without changing last-use/port only moves −5.78%. B is residual/PED/I24, not a second binary GeMM.

**X:** Not a reskin of product sparsity. The uncovered object is port-share between binary select-add and a continuous residual consumer, read off the freeze gradient. Priors never had two tensors on one port.

**controls:** Dual-port vs same-port (letter stays same-port). B-only stall injection vs A-sparsity on/off under long backpressure. Always-ready vs long-backpressure. Do not put 758777→714889 in a letter table.

**kill-gate:** If full A-prior under long backpressure moves 8088 by ≥15%, then 8088 was A-starvation and dual-path port-hold dies. If merely dual-porting with no last-use X already yields ≥15%, it is extra ports, not a letter.

**two-sentence pitch:** Same-resource service ≥15% has to come from not letting B pin the port A already drained. The three freeze points are the control story, not a table of integer-consumer speedups.

**objection:** “−22.83% already meets 15%, ship generic fusion.” That number is always-ready; the letter gate is same-port/**state**/**backpressure**. Backpressure is 8088.

**assumptions:** Always-ready and long-backpressure are the same net besides ready/credit. −22.83% is partly generic fusion, so it is not claimed as dual-path X. −5.78% is an observation, not a result table.

**predictions:** A-sparsity on/off is visible in always-ready and invisible at 8088. Removing B traffic (identity residual, drop PED/I24) would move 8088 even without a better A prior. Keeping B and splitting port-credit would move 8088 without changing sparsity.

**disconfirmers:** A-prior alone moves 8088 ≥15%. Dropping B does not move 8088. Dual-port with identical last-use already clears the 15% gate.

---

## Q04-I4 Independent last-use maps for bitmap vs residual (stop co-allocating A and B)

**id:** Q04-I4

**one sentence:** The post-absorb 0/1 bitmap and the residual/PED/I24 tensor must have independent last-use kills; co-allocating them is what turns two consumers into one 8088 window.

**A:** FireFly bitmap / Prosperity occupancy / LoAS loop live-set on the 0/1 tile — kill when A GeMM done.

**B:** Freeze: two objects stay distinct; consumer A is the binary gate after absorb, consumer B is the continuous residual path. Patch r1 residual chain is historically expensive (old proxy ~35% of activity-weighted dots; proxy ≠ new cycle share). One live window for both ⇒ last-use is the union. The hole is a second last-use map for B, not another A scheduler.

**X:** Bitmap kill on A is the prior. A last-use map for residual/PED/I24 is not product sparsity, NRV packing, or bitmap AND.

**controls:** Co-allocated window vs two last-use maps. Iso-state (do not grow buffers to fake 15%). Same-port 8088. AEE 0-diff expected if only allocation/lifetime metadata changes.

**kill-gate:** Independent last-use maps that do not reduce long-backpressure service vs 8088 by ≥15% at iso-state: kill. Any AEE movement: the split corrupted a live tensor — kill.

**two-sentence pitch:** Dual-path is two lifetimes, not two MACs sharing analog AT-LIF. The letter mechanism is “kill A when GeMM done, kill B when residual/PED/I24 done,” under the same port.

**objection:** “That is just more SRAM / ping-pong.” Extra storage without a last-use rule is out of gate; iso-state is the control.

**assumptions:** Co-allocation is the current same-port state. B’s tensor is not a view of A’s bitmap. Cycle share of r1 residual is unknown; the idea does not use 35% as a cycle number.

**predictions:** Bitmap occupancy falls with A-priors even while 8088 stays until B’s map exists. Iso-state split last-use moves 8088; extra buffering without split does not (or fails same-state). AEE stays 1.219801338.

**disconfirmers:** They are already separately allocated and 8088 remains. Split maps change AEE. Iso-state split <15% and only over-provisioned buffers pass 15%.

---

## Q04-I5 Two-window I24/PED last-use is not spike-tile last-use

**id:** Q04-I5

**one sentence:** Two-window integer I24/PED is 0-diff vs model at q24 and is consumer B; its window last-use is not the spike-tile last-use Prosperity assigns to consumer A.

**A:** Binary gates after absorb → FireFly/Prosperity/LoAS tile last-use.

**B:** Freeze: two-window integer gates/I24/PED q24 is 0-diff vs model. Separate integer consumer 758777→714889 (−5.78%). Gates (A) and I24/PED (B) already exist as two windows. The hole is last-use of the I24/PED window against the binary-gate window under same-port 8088, not an analog AT-LIF payload and not a new table of −5.78%.

**X:** I24/PED is the continuous residual-family tensor SNN GeMM papers do not contain. 0-diff at q24 says B cannot be dropped as approximation; last-use of that window is the uncovered object.

**controls:** Kill I24/PED at B-done vs kill with gates. One-window fused vs two-window. Long-backpressure 8088. AEE must remain 0-diff vs the two-window integer model.

**kill-gate:** Two-window last-use decoupling that does not beat 8088 by ≥15%, or any break of q24 0-diff: kill. Do not add 758777→714889 as a letter table.

**two-sentence pitch:** The integer dual-path is already numerically exact; the letter is the lifetime of the I24/PED window next to the binary-gate window. Absorb makes A binary; it does not make I24 a bitmap.

**objection:** “I24 is just a wider spike.” Freeze forbids writing dual consumers as shared analog AT-LIF amplitude; I24/PED is a different tensor, 0-diff at q24.

**assumptions:** Two-window gates vs I24/PED is the student integer path. q24 0-diff remains the numeric contract. −5.78% is not the ≥15% gate.

**predictions:** Coupling I24 last-use to gate-tile last-use reproduces 8088. Decoupling at iso-state moves service without changing q24 outputs. A-only bitmap scheduling leaves I24 occupancy flat.

**disconfirmers:** I24/PED already dies with the gate tile and 8088 still unexplained. Decoupling changes q24 outputs. Decoupling <15% under long backpressure.

---

## Q04-I6 Residual add is dense B; Gustav NRV packs only sparse A

**id:** Q04-I6

**one sentence:** Post-absorb residual is binary select-add of \(\theta\)-absorbed \(W\) plus a dense continuous residual add; NRV/CPTB packs A and leaves B’s dense add holding the same pipeline.

**A:** Gustav NRV/CPTB and Prosperity product sparsity on \(s\cdot W_{\mathrm{absorbed}}\).

**B:** Native residual add exists. After absorb, layer transfer is 0/1, not \(\theta s\) payload; the skip/projection path is still a continuous add (BN output + skip). Freeze warns the r1 ~35% activity-weighted-dot proxy is not the new cycle share — so B’s occupancy must be re-measured as hold time, not dots. Under same-port, B’s dense add is occupancy Gustav does not model; that is consistent with 8088 equality when backpressured.

**X:** Gustav/FireFly/Prosperity/LoAS have no residual skip tensor. Residual add is not another sparse GeMM. That tensor is the X.

**controls:** Residual-add on the GeMM port vs a decoupled last-use/credit for the add. NRV on A only. Measure B-add hold vs A-select-add under long backpressure (not the old dot proxy). AEE.

**kill-gate:** If residual-add hold under backpressure is negligible (35% proxy false for cycles) and NRV-on-A already ≥15% at 8088, X is empty — kill the residual-occupancy claim. If decoupling the add violates AEE 1.259 / +0.005, kill.

**two-sentence pitch:** Copy Gustav on the binary term; the letter X is that the residual add does not NRV-pack and still owns last-use. Same adder port makes B the backpressure source.

**objection:** “Residual add is one extra add, not 15%.” Then the measured B-add hold under 8088 must be small — that is exactly the kill-gate, and the freeze already distrusts the 35% proxy.

**assumptions:** Residual add is on B, after projection/BN, not absorbed into binary \(W\). Same-port means A select-add and B dense add compete. Cycle share is unmeasured.

**predictions:** NRV occupancy of A falls; 8088 does not, until B-add has its own last-use/credit. Identity-residual (B=0) moves 8088 without a better NRV. AEE unchanged if only scheduling of the add changes.

**disconfirmers:** B-add hold ≪ 15% of 8088. NRV-on-A alone meets the backpressure gate. Identity-residual leaves 8088 unchanged.

---

## Q04-I7 OF motion (C12/H67/ep34) couples A and B last-use

**id:** Q04-I7

**one sentence:** Event-camera optical flow makes A’s binary spikes and B’s residual/PED live intervals co-fire on motion C12/H67/ep34, so last-use is a task-structured dual handshake, not i.i.d. SNN product sparsity.

**A:** Prosperity/FireFly skip zeros wherever spikes are absent (unstructured product sparsity).

**B:** Task is DSEC valid825 2D flow; motion C12 / H67 / ep34 exist in the student. Residual chain is an OF feature skip, not a classifier residual. Events (A) and residual/PED (B) peak on the same moving structure, so last-use conflict is worst when both consumers are hot — the backpressure regime that reads out as 8088. Always-ready −22.83% is the regime where that coupling is not binding.

**X:** Optical-flow task structure is an allowed uncovered object. Bitmap AND does not schedule a motion-aligned residual tensor.

**controls:** Motion-conditioned dual last-use vs static \(\max(t_A,t_B)\) (Q04-I1). C12/H67/ep34 features on vs off as schedule inputs, not as extra modules. AEE vs 1.219801338. Same-port 8088.

**kill-gate:** If motion-conditioned last-use equals static dual last-use (no extra ≥15% under 8088), OF structure is not X — fall back to I1/I4. AEE >1.259 or rel >+0.005: kill. No 18-module motion stack.

**two-sentence pitch:** Dual-path stall is not random sparsity; it is motion-aligned A+B heat. The letter may use C12/H67/ep34 only as last-use hints, not as a second network.

**objection:** “SNN accelerators already like bursty spikes.” Burstiness of A is the prior; co-liveness of B’s residual on the same burst is the OF object.

**assumptions:** C12/H67/ep34 correlate with residual/PED energy, not only with spike rate. Using them as last-use hints does not change the numeric graph. Static dual last-use is the baseline X; motion is an increment.

**predictions:** A-sparsity and B-occupancy correlate on moving pixels and anti-correlate on static background. Static \(\max(t_A,t_B)\) already moves 8088; motion hints only help if residual last-use is shorter than worst-case B on background tiles. AEE unchanged if hints are schedule-only.

**disconfirmers:** A and B last-use uncorrelated given motion. Motion hints change AEE. Motion hints = static dual last-use within measurement noise.

---

## Q04-I8 Pre-threshold T10 mix last-use forks into both A and B

**id:** Q04-I8

**one sentence:** Noncausal T10 mix (PSN / lifting CSE) is continuous arithmetic **before** threshold; its intermediates must stay for both post-absorb binary A and residual B, so absorbing \(\theta\) into \(W\) does not let FireFly kill T10 buffers at spike-tile end.

**A:** After threshold + absorb, copy FireFly/Prosperity/Gustav/LoAS on binary GeMM.

**B:** Freeze: noncausal T10 mix happens before AT-LIF threshold, then threshold, then absorb. Source CSE ordinary 260 add/sub vs lifting 159+35 intermediate RNE/sat. Lifting raw AEE 1.232979368 (rel +0.013178) **fails** +0.005 though abs 1.259 passes — so lifting CSE is not the letter’s accuracy story. T10 outputs fork: threshold → binary A, and residual/PED → B. Last-use of T10 state is mix-depth plus \(\max(t_A,t_B)\). 8088 can be T10 state held for B, not missing binary skip. Noncausal mix also blocks time-causal tile-kill that SNN priors assume.

**X:** T10-before-threshold is an allowed uncovered object. Prosperity-class priors start at spikes. Absorb does not shorten T10 lifetime.

**controls:** T10 last-use to threshold-only vs to \(\max(t_A,t_B)\). Ordinary mix vs lifting CSE, but letter AEE must stay on the ordinary 1.219801338 side of +0.005. Same-port 8088.

**kill-gate:** If T10 buffers already die at threshold and 8088 is unchanged, T10 is not the dual last-use source — kill. Any T10-X that ships lifting raw AEE (rel +0.013178) fails the relative gate — kill. ≥15% not shown at iso-state under 8088 — kill.

**two-sentence pitch:** Dual-path last-use starts before the neuron: T10 is shared parent of binary A and continuous B. Copy the binary prior after absorb; do not claim lifting CSE as the accuracy win.

**objection:** “Lifting already cuts 260→159, that is the X.” CSE count did not move 8088, and lifting AEE fails +0.005. X is T10 **lifetime** across the fork, not op count.

**assumptions:** T10 intermediates are live until both consumers of the fork are done, unless proven otherwise. Ordinary (not raw lifting) is the AEE baseline. Noncausal mix forbids killing by timestep the way a causal SNN tile is killed.

**predictions:** Threshold-only T10 kill breaks B (AEE or 0-diff I24/PED) or does not move 8088. Fork-aware T10 last-use can move 8088 without adopting lifting’s AEE. Ordinary vs lifting CSE still ties at 8088 until last-use changes.

**disconfirmers:** T10 already last-used at threshold with B intact. Fork-aware last-use <15% at 8088. Any accuracy path that needs lifting raw 1.232979368.

---

## Cross-idea notes (not extra modules)

- A is always the full binary-GeMM prior after absorb. None of these ideas replace Prosperity/Gustav/FireFly/LoAS on A; they refuse to let that prior define last-use of B.
- 8088 is the dual-path readout. Always-ready −22.83% is not the letter gate. Integer −5.78% is not a table.
- BN barrier (I2), residual add (I6), I24/PED windows (I5), and T10 fork (I8) are different B tensors/lifetimes; do not merge them into “continuous AT-LIF shared by two MACs.”
- I1 is the last-use rule; I3 is the port-credit diagnosis; I4 is the allocation map. Keep them separable under kill-gates.
- I7 is optional increment on I1; kill it first if motion hints = static \(\max(t_A,t_B)\).
- Venue: TCAS-II 5 pages; same-port/state/backpressure net service ≥15%; AEE abs ≤1.259 and rel ≤+0.005 vs ordinary 1.219801338; no OpenROAD-as-PPA, no FPS products, no CIM.
