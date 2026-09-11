# ADV_F4 — Prefix-sufficient T10

Reviewer did **not** originate F4. Sources under review: `fusion/FUSION_CANDIDATES.md` F4 (from P04-I2, P06-I5); freeze `PROBLEM.md`; literature `L5_venom_bitfair_spiketransformer.md`, `L6_event_of.md`. Supporting pins only: P04-I2, P06-I5, L7 EVA-Flow map, local SpikePack-style pack AEE.

This file is an attack, not a salvage pitch. Scores are not accept probabilities.

---

## Verdict

| Field | Call |
|---|---|
| **Disposition** | **pause** |
| **Novelty** | **2** / 4 |
| **Performance** | **1** / 4 |
| If 8088 is not prefixable | **stop** the island (do not morph into F1/F2) |
| If 8088 is tap- or window-serial **and** both consumers share one k | **revise** A/X, then one pair |
| As written | **not retain** |

Pause-gate (must fire before any paired recovery): decompose the frozen **8088** long-backpressure. F4 is chargeable only if a non-trivial fraction of 8088 is *prefixable wait* (tap-serial issue, or data-window wait that a compile-time K<10 actually cuts). If 8088 is BN-stat barrier, dual-consumer RAW on the current vector, write-back/bank conflict, or “events of the full noncausal window have not arrived,” suffix-zero training cannot legally move it.

---

## 0. Candidate as written (not endorsed)

From `FUSION_CANDIDATES.md`:

> **F4 Prefix-sufficient T10** (from P04-I2, P06-I5; BitFair/EVA-Flow as A)
>
> **A:** BitFair learnable prefix; EVA-Flow anytime OF.
> **B:** noncausal T10 waits for all 10; 8088 is wait not ALU.
> **X:** train suffix ticks to exact 0 for **both** consumers so commit can fire on a prefix; not ReLU-zero early terminate.
> **Kill:** AEE; or only the gate consumer can prefix.

Parents (compressed):

- **P04-I2:** streaming *bin-complete* handshake (EVA-Flow UVG + IDNet TID as A). Causal bins 1…K release gate+r1; PED may wait for K′≥K. K frozen. Kill if K=10 is the only AEE-legal point, or if prefix “wins” by stalling PED until 10.
- **P06-I5:** keep noncausal T10 family; train prefix length k<10 sufficient for **both** consumers; suffix forced to **exact integer zero** so the shared completion token can fire before slot 10. A is anytime/early-exit without a second head, online prefix FIR, causal lifting, remainder bounds. Hardware kill is primary: 8088 unchanged.

F4 collapses those two objects into one letter and names the wrong papers as A.

---

## 1. Attack

### 1.1 BitFair ReLU-zero ≠ dual-consumer continuous θg

BitFair (L5, arXiv 2607.05445) is a **weight-bit-serial CNN** on SpikingJelly event *bins*, not an SNN and not this PSN. The stop predicate is explicit:

- reverse nest: all spatial/channel inputs of bit-plane j, then next plane;
- hard stop `P_k ≤ θ^l` ⇒ predict **ReLU(y)=0**, write 0, skip remaining bits;
- learnable θ + survival loss + ABO as offline prefix *order*;
- per-PE comparator + FSM gather that stops **further fetches**.

Their own limitation: speed-up is **bounded by the ReLU-negative fraction**; GELU/smooth activations “rarely produce exact zeros and would benefit less.”

This identity has **no ReLU-zero consumer** (`PROBLEM.md`: ATLIF emits continuous threshold amplitude θg; dual consumers = gate/θg path **and** continuous residual/PED; do not change to binary ATLIF).

| BitFair object | This net |
|---|---|
| Output contract `{0, ReLU(y)}` | Continuous θg, including **non-zero** amplitudes |
| One activation consumer | Two consumers of one source word |
| Prefix that proves “will be zero” | Prefix that proves “gate silent” is **not** a PED value |
| Weight bit-planes | T10 taps / lifting intermediates (35 RNE/sat are a different island, F3) |
| Dynamic, per-output terminate | F4’s parents declare **compile-time** k/K before the pair |

L5 already classified BitFair as **CONTROL, not title-level X**, and already wrote the only candidate increment: prefix accept/continue on the **union of unfinished dual consumers**, emit predicted **non-zero** θg, charge the last pending physical request. That sentence lives in `conditional_execution_followup_20260908.md` / `sparsity_hardware_next_20260908.md`. F4’s “not ReLU-zero early terminate” is that disclaimer, not a new circuit.

Worse: F4’s actual mechanism (P06-I5 / P04-I2) is **static k**, not BitFair’s dynamic bit-plane stop. Static T=k is a different prior family (causal FIR, T-reduction, Spike-IAND-Former T=4/2/1 mux). Citing BitFair as A invites a reviewer to grade a T-reduction student as a reskin of a ReLU-zero bit-serial CNN — and still find the predicate illegal.

**Attack result:** BitFair is a *wrong-A* plus a *predicate kill* unless F4 (i) copies the full BitFair skeleton (bit-plane nest, survival loss, ABO, PE+FSM group-close) **or** drops BitFair from A and admits static T=k, and (ii) still beats **independent BitFair + the same closer** and **gate-only prefix**. Continuous θg on the accepted prefix is mandatory; zeroing θg to make the stop fire is identity-illegal.

### 1.2 EVA-Flow anytime is GPU OF

EVA-Flow (L7 pin of arXiv 2307.05033; L6 family is event-OF *algorithm* vs r1 producer) is:

- **Unified Voxel Grid** bins of support 2τ, a bin ready after τ (5 ms on DSEC, 21 channels) instead of a 100 ms voxel;
- encoder + 4-level pyramid + stacked **SMR** (warp + ConvGRU + residual FlowHead);
- **only the last flow** L1-supervised; intermediate times implicit via dense warp;
- DSEC-Flow test: 5.0M params, **16.8 GMACs/prediction**, 5 ms, 200 Hz, **EPE 0.88**.

None of that is a T10 PSN handshake.

L7 map, already on freeze:

- Sequential bins + last-step-only loss **do not compile a 10×10 mix**. Local T10 is **noncausal**, dense, already compiled.
- Anytime output is a **flow after each bin**, not dual-consumer of one compiled T10 vector (gate **and** PED of the **same** output).
- Replacing T10 PSN by UVG+SMR, or advertising 200 Hz / 5 ms / DSEC EPE, is a **different paper**. Keep EVA-Flow as a **task** control (anytime vs batch T10), not compile X.
- DSEC EPE ≠ valid825 AEE. Do not import 0.88.

L6 additional negatives on “stream then skip residual”:

- Spike-FlowNet **waits for all N** encoder bins, then dumps a **dense** vector into an ANN residual. That is the opposite of prefix retirement. Their residual-as-SNN ablation **worsens AEE**. Local r1 **is** that residual class.
- SciFlow SCI quality maps and BAT `df=f/N` are same-frame flow oracles; illegal as r1 schedulers.
- SENECA ANN 3-row release is illegal on noncausal T10: a position is live until PSN has all ten times **or** a trained complete-T10 predictor is paid.

P04-I2 copies UVG+TID as if they were a ready/valid circuit. They are a GPU streaming *representation* plus a recurrent OF head. IDNet TID is “one iteration per streaming batch,” still GPU accuracy, still a flow updater, not a dual-consumer T10 completion token.

**Attack result:** EVA-Flow cannot be A for a TCAS-II circuits letter. It is a **task control** (does a causal/anytime student even exist on DSEC?). Using it as A is a reskin risk *and* a denominator cheat (GMACs, 200 Hz, EPE). F4 must not sell “anytime optical flow hardware.”

### 1.3 Noncausal T10 may need all ticks for PED

Frozen time identity: **noncausal T10 at this PSN**. Future source samples can change the first output. Dual consumers after source; integer PED q24 is 0-diff vs model on captured windows — PED is a **value** consumer, not a {fire, silent} consumer.

Reasons a prefix is information-theoretically hostile:

1. **DSEC GT alignment.** P04-I2’s own biggest objection: GT may require events on **both sides of t**. A causal (t−Δ, t] cut can be insufficient, not just a hardware issue. If that is true, every K<10 fails relative AEE after one pair, and the island is dead rather than “almost causal.”
2. **PED is the AEE owner.** P01-I8’s swapped-split prediction (control, not this island): cheap PED / exact gate should fail AEE; exact PED / cheap gate might not. F4’s kill already says “only the gate consumer can prefix.” That is the **expected** outcome, not a rare failure. A shared completion token is the **max**(k_gate, k_PED). If k_PED=10, F4 does not move 8088.
3. **Union occupancy, not intersection skip.** F1’s occupancy is `support(θg)∪support(PED)`. F4 needs the **intersection of “prefix-sufficient”** across consumers. Intersection of “can stop” is the complement of the union of “still needs suffix.” One live PED lane holds the source word (already measured on GP suffix bounds / H8: seven heads done at C192, one live at C384, still read the C384 word).
4. **Full-domain BN.** Native projection BN uses **actual batch statistics over 10×96×120×160**, not frozen running stats (`PROBLEM.md`). Local-window replay with free μ/σ **undercharges wait/storage**. Even a perfect T10 prefix does not retire the projection residual if BN still barriers on the size-10 axis. Axis identity (`10` ↔ T10) is **unproven** (P05-I7). If the 10 is T10 and suffix bins still enter μ/σ, they are not droppable. F4 stacked with F2 is a zoo; F4 alone must show 8088 is not that barrier.
5. **Lifting already rewrote T10 and lost relative AEE.** Ordinary 1.219801338 vs lifting 1.232979368 (Δ **+0.013178**; abs 1.259 passes, rel +0.005 fails). Prefix-sufficient is another restriction of the same 10-tap mix. There is no remaining AEE slack for “drop future context.”
6. **T-reduction prior.** Spike-IAND-Former (L5, 2503.19643) already ships T=4/2/1 mux; CIFAR accuracy falls 95.69 → 92.93 → 91.34. Causal LIF unroll ≠ noncausal T10, but “compile a shorter T” is not X.

**Attack result:** the letter’s hardware story (commit before slot 10) is likely false for the consumer that owns AEE. Gate-only prefix is a predicted **false positive**, already written as a kill, and should be treated as the default experimental outcome.

### 1.4 Local SpikePack AEE 1.31 already failed a packing layout

SpikePack (L5, 2501.14484): rank-1 `v_g = W(S q)`, integer zip of T spikes, dynamic-θ decode, binary spike I/O. **CONTROL** for any “compress T10 into fewer words.” Replacing ATLIF with SpikePack is **identity-hostile**.

Local layout already run (`README.md` / L5 kill-gate): **single-projection + uniform 8-level quant** (SpikePack Eq. 8–9 control) on six S2 sources, valid825 **AEE 1.310533**, original time-linear terms/vector = 10. Absolute budget 1.259 **fails**. That does not kill the SpikePack *family*; it kills “pack T10 to one scalar then decode” as a free lunch on this student.

F4 is not SpikePack. It is still a **temporal degrees-of-freedom cut**: k<10 taps instead of 10, suffix forced to 0. On this student, every measured attempt to make T cheaper has either:

- failed **absolute** AEE (pack layout 1.310533), or
- failed **relative** AEE (lifting 40-coeff T10 +0.013178), or
- not moved **8088** (lifting always-ready −22.83% absorbed).

O(1) words ≠ O(1) bits. Dual consumers still need enough bits for amplitude θg **and** the residual. Rank-1 `q` is not full-rank T10 PSN / lifting. Zip-and-threshold does not produce a residual addend.

**Attack result:** a packing/T-fold layout is a **mandatory negative control**, not a method. If F4’s k-prefix is numerically a low-rank time fold, it inherits the 1.31 death. Do not retitle prefix-zero as SpikePack; do not claim O(1) service.

### 1.5 Additional kills F4 left on the table

**8088 is not shown to be tap-serial.** Ordinary CSE 260 add/sub and lifting 159 add/sub **both** post long-backpressure **8088**. That is the freeze’s strongest hardware sentence, and it already says the stall is *not* ALU depth. P06-I5’s own disconfirmer: if 8088 does not scale with k/10, prefix training cannot charge even if AEE passes. SCOPE: complete-chain same-resource net service is **UNKNOWN**. Always-ready 6938→5354 is not throughput.

**Two mechanisms fused, neither title-ready.**

| | P04-I2 handshake | P06-I5 remainder-zero |
|---|---|---|
| What becomes ready | a **bin of events** | a **compiled tap / suffix lane** |
| Wait it can cut | data arrival of future bins | serial issue of suffix arithmetic |
| If 8088 is the other wait | no | no |
| AEE object | causal window vs noncausal GT | same window, truncated mix |
| P04 prediction | handshake-only K=10: AEE and 8088 **unchanged** | hard k=6, no FT: AEE Δ ≫ +0.005 |

F4’s X sentence (“train suffix ticks to exact 0”) is P06-I5. F4’s A list (BitFair, EVA-Flow) is a mash of P06’s wrong CNN prior and P04’s GPU OF prior. A 5-page letter cannot be both a streaming voxel protocol and a trained FIR remainder.

**Computed-and-discarded suffix is a non-result.** P06-I5 already requires suffix lanes not issued / clock-gated. If the compiler still emits 10 taps and the scheduler still waits on tap 10, X is empty.

**One pair, already on a failed T10 rewrite.** Relative +0.013 is on the table. A second recovery is a kill, not a revision (`P06` shared protocol). Architecture search over k∈{4,5,6,8} is a **grid**, which that protocol forbids as “the pair.”

**Do not add resource tables.** Source SIMD −22.83% and integer-consumer −5.78% are different resource points. Prefix that moves only the consumer table does not pay the 8088 story.

---

## 2. A / B / X after the attack

| Piece | F4 as written | After attack |
|---|---|---|
| **A** | “BitFair learnable prefix; EVA-Flow anytime OF” | **Incomplete and mis-typed.** Copy as CONTROL, not as title: BitFair+BitSET+SparseInfer+whole-word (dynamic terminate skeleton); BranchyNet/SACT/MSDNet **without** extra head; causal/prefix FIR + causal lifting; Spike-IAND-Former T=4/2/1 as T-reduction; SpikePack rank-1/zip as time-fold; Spike-FlowNet wait-all residual as **negative** handshake; EVA-Flow UVG+SMR as **task** control only. Ordinary T10 and lifting T10 at the **same** two-stage SIMD (8088 both) are the hardware A. |
| **B** | “noncausal T10 waits for all 10; 8088 is wait not ALU” | **Half-true.** 8088 is wait-not-ALU is a frozen observation. That it is wait-*for-the-tenth-tap* is an **untested interpretation**. BN full-domain, dual-consumer RAW, and data-arrival of the whole window are competing B’s (F2, F1, P04-I2). F4 does not own 8088 until a stall breakdown says so. |
| **X** | “train suffix ticks to exact 0 for both consumers so commit can fire on a prefix; not ReLU-zero” | **Not title-level as stated.** “Not ReLU-zero” is L5’s already-named increment disclaimer. Static dual-consumer T=k with integer suffix 0 is a T-reduction student plus a completion-token claim. Dynamic union-prefix with **non-zero** θg is the L5 increment, and it still must beat independent BitFair+same closer. Either way, X is empty if k_PED=10 or if suffix is computed-and-discarded. |
| **Kill** | “AEE; or only the gate consumer can prefix” | **Necessary, not sufficient.** Add: 8088 unchanged; BN barrier reintroduces the stall; suffix not integer-0 on captured windows; k≥8 without proportional 8088 drop; SpikePack-style pack or lifting-class AEE regression; identity collapse to binary ATLIF / IAND residual; EVA-Flow head replacement; adding −22.83% to −5.78%. |

---

## 3. Scores

### Novelty **2** / 4

Rubric: 0 none; 1 prior only; 2 hole exists but X not located; 3 measurable X, not closed; 4 title-level.

- **Not 1:** B is real. Always-ready 6938→5354 with long-BP **both 8088** is the freeze’s hardware hole; F1–F3 do not uniquely own “completion token.”
- **Not 3:** F4 did not locate X. It named two incompatible parents, cited GPU OF and ReLU-zero CNN as A, and restated L5’s dual-consumer prefix disclaimer as X. Static k vs dynamic terminate is not even chosen.
- **Not 4:** TCAS-II will read “anytime OF” + “learnable early terminate” and ask for the relative prior. The relative prior is complete. The increment is not.

### Performance **1** / 4

Rubric used here: 0 already falsified / cannot pass conjunctive gates; 1 local negatives dominate; 2 one gate plausible, the other likely fail; 3 both gates look reachable; 4 measured.

Conjunctive gates from freeze: valid825 AEE ≤ 1.259 **and** Δ vs ordinary 1.219801338 ≤ +0.005 (i.e. ≤ 1.224801338) **and** full-chain same-port/state/backpressure net service ≥15% **and** integer windows 0-diff **and** 8088 moves **and** no binary ATLIF.

Why 1, not 0: the *family* (train a shorter sufficient time support) is not the dead SpikePack *layout*. Why not 2: every local T10 restriction already failed a gate that F4 must pass, and PED/BN/union make 8088 movement the unlikely one.

| Local negative | Number | What it does to F4 |
|---|---|---|
| Ordinary dense T10 | AEE **1.219801338** | Strong control; relative slack is **0.005** |
| Lifting T10 | AEE **1.232979368** (Δ +0.013178); 8088 unchanged | T10 rewrite already spent the slack |
| SpikePack-style 8-level pack | AEE **1.310533** | T-fold / fewer time DoF already dead as a layout |
| Lifting source SIMD | always-ready −22.83%, long-BP **8088** | Arithmetic win absorbed; F4 must move 8088, not add/sub |
| Integer consumers (other resource) | −5.78% | Forbidden to add; not a prefix result |
| Spike-FlowNet residual-SNN / wait-all | AEE worse / opposite of early retire | r1 is that residual |
| BitFair on smooth/non-ReLU | their speed-up bound | θg and PED are the smooth case |
| Full-chain service | **UNKNOWN** (SCOPE) | Cannot claim ≥15% from 8088 folklore |

Expected experimental order if someone ignores the pause: hard k=6, no FT, AEE death (P06-I5 P1); gate-only prefix recovers AEE, 8088 stays 8088 (P06-I5 P3); both-consumer k=10 is the only AEE-legal point (P04-I2 kill). That is performance 1.

---

## 4. Disposition: **pause**

**Not retain.** A is wrong. Two mechanisms are fused. X is a disclaimer. Spending the one pair on F4-as-written burns the only recovery budget on a T10 restriction after lifting already failed +0.005.

**Not stop yet.** Stop would throw away the only fusion island that *names* 8088 as wait-not-ALU. That observation is still the right B. The island dies **if** the pause-gate says 8088 is not prefixable, or if a later one-pair run hits the kills. Do not pre-kill the family; do not keep the layout.

**Not revise-now.** Revising A/X without a stall breakdown writes a fictional letter (handshake vs remainder-zero vs BN vs last-use). That is how F4 got here: P04-I2 and P06-I5 clustered because both say “don’t wait for 10,” not because they share a circuit.

**Pause** until all of the following are printed on captured windows / the same two-stage SIMD model:

1. Stall histogram of the 8088 class: fraction that is (a) T10 tap-serial, (b) last-bin data arrival, (c) PED-RAW on this vector, (d) gate∧PED joint-ready, (e) BN-stat barrier, (f) write-back/bank. **If (a)+(b) is negligible → stop F4** (do not retitle as F1 or F2).
2. Hard causal / hard k∈{6,8} **without** recovery: valid825 AEE vs 1.219801338. **If already ≤ +0.005**, F4 has no training story (the mix was already prefix-sufficient) — then only a handshake experiment remains, and that is P04-I2 alone, still needing 8088 to move. **If Δ ≫ +0.005**, that is the expected prior, not a license to grid-search k.
3. Gate-only vs PED-only vs both: which consumer’s suffix energy is non-zero. **If PED suffix is required → stop** (F4’s own kill, treated as default).
4. Integer suffix of the current student: is it already ~0 on q24? If not, exact-0 is a real training ask; if yes, completion should already be able to fire and the hardware bug is elsewhere.

Cheap probes. Not the pair. Not RTL.

### Branch after pause

- **stop** if 8088 is not prefixable, or PED needs k=10, or hard-prefix AEE is already inside +0.005 and 8088 still does not move (nothing to train, nothing to schedule).
- **revise** (only if (a) or (b) dominates 8088 **and** both consumers can share one k):
  - Split letters: handshake (data prefix) **xor** remainder-zero (arithmetic prefix). One mechanism.
  - Recast A as in §2. EVA-Flow is task control. BitFair is terminate CONTROL only if the design is *dynamic*; if k is compile-time, BitFair is the wrong A — use T-reduction + prefix FIR.
  - X must be a **shared completion token** on `k* = max(k_gate, k_PED)` with suffix **not issued**, integer 0-diff on both sinks, continuous θg preserved on the prefix (non-zero amplitudes legal). Not ReLU-zero. Not UVG+SMR. Not SpikePack neuron. Not IAND residual.
  - Strong controls: ordinary T10; lifting T10; hard k no FT; UVG-consistent bins vs naive truncate; gate-only prefix; PED-only prefix; SpikePack-style pack; T=k student with no completion-token change (isolates protocol); frozen-BN vs charged full-domain BN (must not get 8088 movement for free).
  - One pair, one declared k, no λ grid. Kill on AEE abs/rel, 8088, suffix not 0, k≥8 without proportional stall drop, identity collapse.
- **retain** is not on the menu until those controls exist as measurements.

---

## 5. What a TCAS-II reviewer would punish (if F4 were submitted as written)

- Renaming BitFair ABO / ReLU-zero, EVA-Flow anytime, SpikePack zip, or T=4/2/1 mux as the contribution.
- Quoting 2.12× / 22.1× / 200 Hz / EPE 0.88 / 38 TSOPS/W as this student’s service.
- Changing ATLIF into binary SpikePack / IAND-Former / ReLU-zero so the stop predicate matches.
- Early-stopping the gate and silently dropping PED.
- Claiming O(1) in T when the packed integer widened, or claiming k/10 service when 8088 did not move.
- Reporting always-ready −22.83% as the result of prefix training (already absorbed; part is generic round→sat fusion).
- Adding the integer-consumer −5.78% table.
- Free μ/σ on local windows while BN is full-domain.
- Same-frame flow as a prefix oracle (SciFlow/BAT class; L6 ban).
- “First event optical-flow hardware” (SENECA 2025 already is; L6).

Venue: 5 pages, one mechanism, circuits+systems co-design, binary accept/reject. Missing **relative prior** and missing **measured advantage** are the standard reject.

---

## 6. Identity and stacking

- Do not change the paper identity to binary ATLIF. Suffix-zero of **source taps** is legal; collapsing θg to {0,1} is not.
- Do not sell analog CIM. Do not quote OpenROAD/Yosys as foundry PPA. Do not multiply component speedups into FPS.
- F4 is not a stack with F1 last-use or F2 paid BN. If the pause-gate says 8088 is last-use or BN, that work belongs on those islands; F4 **stops**.
- F3 (delete 35 RNE) is orthogonal and must not be smuggled in as “the prefix is integer-exact.”

---

## 7. One-paragraph owner summary

F4 points at the right frozen hole (8088 is wait, not ALU) and then attaches the wrong priors and two different circuits. BitFair’s stop is ReLU-zero on a bit-serial CNN; this net’s consumers are continuous θg and PED. EVA-Flow anytime is GPU optical flow with a recurrent head; it is not a T10 completion token. Noncausal T10 plus DSEC alignment plus PED-as-AEE-owner make “both consumers accept k<10” the unlikely branch — and F4 already listed that as a kill. Local SpikePack-style packing already died at AEE 1.310533; lifting already died on relative +0.005 without moving 8088. Novelty 2 (hole yes, X not located). Performance 1 (local T10 restrictions lose). **Pause** for a stall breakdown; **stop** if 8088 is not prefixable; **revise** only if it is, and then as one mechanism with BitFair/EVA-Flow demoted to control, not A.

End of ADV_F4.
