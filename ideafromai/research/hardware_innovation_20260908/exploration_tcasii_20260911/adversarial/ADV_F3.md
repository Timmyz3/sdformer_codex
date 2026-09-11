# ADV_F3 — Integer-exact lifting that deletes 35 RNE

**Reviewer (did not originate F3):** adversarial stand-in, 2026-09-11.  
**Object:** `fusion/FUSION_CANDIDATES.md` F3, clustered from P03-I2/I5/I7 and P06-I1/I8. Hypothesis `hypotheses/H_F3_integer_lifting.json`.  
**Role constraint:** attack the strongest written version; do not invent a kinder F3; do not kill the lifting *family* (SCOPE: stop layouts).  
**Inputs used:** `PROBLEM.md` freeze, L7, W1 (da4ml / ROM-LTE / ESTU), two-stage writeback README, `constant_compilation.md` / result JSON, `gate_collapse_probe.md`, `CURRENT_LINES_AND_PLAN_20260910.md` §4–5, TECH_ARCH novelty table, P03/P06 source cards.

---

## 0. Scores and disposition

| Axis | Score (0–4) | One line |
|---|---:|---|
| **Novelty** | **1** | Priors only. Claimed X is QAT + da4ml CSE + Calderbank I2I + generic quant fusion, all already A. |
| **Performance** | **1** | Always-ready ALU win already taken by fusion; long-BP 8088 unmoved; relative AEE already fails +0.013 and a smaller alphabet should make it worse. |

**Disposition: stop** this fusion candidate as a title-level island.

Not retain (no unlocated circuit X). Not revise-in-place (typed dual-exit rounding is a *different* idea, closer to F1 / P03-I3, and is not “delete 35 RNE by dyadic QAT”). Not pause-as-F3 (CURRENT_LINES already queues PoT as a **control after** full-chain ≥15%; rebranding that control as F3 would launder A into X).

Lifting-as-family may stay. F3-as-written dies.

---

## 1. Strongest version under review

As written:

- **A:** da4ml / MCM / CSD; generic round→sat fusion as control (already −13.56% of source-ready).
- **B:** after CSE, lifting still has 35 intermediate RNE/sat; relative AEE +0.013.
- **X:** one paired QAT so the 40 lifting coeffs live in a dyadic/CSD alphabet and the source graph is integer-to-integer with **zero** intermediate RNE; ordinary gets the same alphabet permission.
- **Kill:** after exactly one paired recovery, relative AEE still >+0.005, or long-BP still 8088.

That is already the strongest charitable reading: one pair, ordinary matched on alphabet, fusion nested as A, 8088 as hardware kill. The attack is that this “strongest X” is still a rename, and the freeze already contains the numbers that kill it.

---

## 2. Attack 1 — Is deleting 35 RNE just QAT + da4ml, already A?

**Yes.** F3 concatenates two complete priors and mislabels the join as X.

### 2.1 Multiplierless compile is already spent

The 40 coeffs are **not** unconstrained reals in the IR F3 would hand to hardware. They are already `signed16/f12` RNE, compiled by official da4ml 0.6.0 (`constant_compilation_result.json`):

```
N = (old << 12) + signed_q12 * other
state = clip_signed24(RNE(N / 4096))
```

Per T10 vector, raw forward is **159 add/sub** (identity included; do not add 40 multiplies). Stage-1 MST + Stage-2 **CSD + bitwidth-weighted CSE** is exactly da4ml (`psn/cmvm_20260909/da4ml_official/docs/cmvm.md`; TRETS 2026 / arXiv:2507.04535). W1 P24 already called this the strongest **A** for T10 source compile. Ordinary is the same compiler on a dense 10×10: **260 add/sub, 0 intermediate RNE**, terminal cutoff fold.

So “put coeffs in a CSD alphabet then CSE” is a description of the **current** compile, not a new one. da4ml already recodes q12 into CSD internally. Hcub / RAG-n / Hartley CSE / SPIRAL multiplier-blocks are the same A stack (P01-I6, P03-I1). ROM-LTE (W1 P10, CICC 2026) is the LUT twin of the same constant tensor. Claiming MCM/CSD after this freeze is a reskin.

### 2.2 The 35 RNE are a Q-format identity, not missing CSD

`constant_compilation.md`: 40 half-step RNE/sat events; last 5 raw outputs fold into gate cutoff; **35 intermediate RNE must stay** because layer-3a still feeds layer-3b. They are guard / sticky / parity / conditional +1 / sat on a signed24 writeback. They are **not** algebraic commons da4ml failed to eliminate.

The compile JSON already records that CSD Hamming of the *coefficients* does not commute with this rounding:

```
RNE_noncommutation_example:
  old=1, q12=2048, other=1
  correct_RNE_combined = 2
  incorrect_old_plus_RNE_product = 1
```

`q12=2048` is **exactly** coefficient `0.5 = 2^{-1}`, the JPEG2000 dyadic predict. Even a pure dyadic tap still needs the combined RNE if the identity is RNE-of-sum rather than floor. F3’s sentence “dyadic/CSD alphabet ⇒ zero intermediate RNE” is therefore **false as an arithmetic identity**. Alphabet QAT can at best make some half-steps shift-add; it does not delete sticky rounding unless the **rounding rule** changes (Calderbank floor) or the **10×10 is composed** (no half-step checkpoints).

Those two moves are also A, and both were already tried or named:

| Move | Prior | Local status |
|---|---|---|
| Compose half-steps, drop intermediate Q | delayed quant / da4ml quant-node graphs | `gate_collapse_probe.md`: full merge **241 add/sub vs 159+35**; **stopped** that layout |
| Integer-to-integer floor | Calderbank–Daubechies–Sweldens 1998; JPEG2000 reversible 5/3 | P06-I8 copies it as A; changes the integer function (breaks current 0-diff) |
| QAT discrete codebook then MCM | DeepShift / INQ / ShiftAddNet / AdderNet; **ARITH 2025** hardware-aware training into MCM adder graphs | CURRENT_LINES: “只将系数改为二次幂**不算 X**”; TECH_ARCH table: **A: PoT / DeepShift** |
| Fold round→sat | HLS/TVM `qnn` canonicalization; IEEE fused round-and-sat | already measured, see §4 |

P03-I7 is not X; it is the **covariate** F3 must subtract. F3 lists it as a parent and then still sells residual RNE deletion as the letter.

### 2.3 Project text already classified this as A

`CURRENT_LINES_AND_PLAN_20260910.md` §5 Step 2: ordinary dense, current lifting Q12, and **zero-including PoT**, same train domain; “迁入 DeepShift 有用部分以建立**强对照**；只将系数改为二次幂不算 X.”  
`TECH_ARCH_SOFT_HARD_FULL.md` §4.1: CSE, PoT/DeepShift, and generic round→sat/norm24 fusion are **not** title sentences.  
L7 relative-prior table: “More CSE/CSD/MST → rename”; “Fold terminal round/sat → rename unless **split per consumer**.”

F3 does **not** split per consumer. It wants **zero** intermediate RNE for the whole vector. L7’s only remaining legal X on this island was *gate may trunc, PED keeps sticky, without cloning the 159-node DAG*. That is P03-I3 (typed dual-exit), which F3 dropped. Zero-RNE-everywhere is the ordinary student’s **already achieved** rounding texture, not an increment.

**Verdict on Attack 1:** deleting 35 RNE by dyadic/CSD QAT + da4ml is A. The join is still A.

---

## 3. Attack 2 — Does +0.013 AEE get worse under a dyadic alphabet?

**Expected yes**, for PTQ certainly, for one-pair QAT very likely. The freeze already fails the relative gate **before** shrinking the alphabet.

### 3.1 Numbers that already exist

| Student | valid825 AEE | vs ordinary 1.219801338 |
|---|---:|---:|
| ordinary dense-source/raw | 1.219801338 | 0 |
| lifting40 raw (unconstrained 40 reals → q12) | 1.232979368 | **+0.013178** (abs 1.259 pass, relative +0.005 **fail**) |
| lifting40 shared (more structure) | 1.247808610 | **+0.028** |
| relative kill line | 1.224801338 | +0.005 |

Unconstrained 40-coeff lifting, trained 320 steps from the same parent, already sits **2.6×** the relative budget above ordinary. Shared-Q, a tighter structure on the same family, got **worse**. Hard-sign lifting (CURRENT_LINES fusion table) died at ten-frame AEE ~1.59/1.62. The pattern is: **more constraint on this T10 mix → worse AEE**. F3 adds a strictly smaller codebook.

### 3.2 The current q12 alphabet is far from dyadic/CSD Hamming ≤2

Deployed raw q12 (40 values; `constant_compilation_result.json`): range **[-19046, 18426]** corresponding to coeffs **≈ [-4.65, +4.50]**.

| Property | Count / 40 |
|---|---:|
| Pure PoT (`|q12|` power of two) | **0** |
| Binary popcount ≤ 2 | **0** |
| CSD Hamming ≤ 1 | **0** |
| CSD Hamming ≤ 2 | **2** |
| Mean CSD Hamming | **4.45** |
| Mean binary popcount | **6.55** |
| `|coeff| > 1` (DeepShift-default clip zone) | **13** |
| `|coeff| > 2` | **7** |

JPEG2000 5/3 lives in `{±1/2, ±1/4}`. These taps do not. PTQ-to-CSD/PoT is a large perturbation, not a rounding of already-sparse digits. CURRENT_LINES already warned: source PoT must **allow shifts of coefficients >1**; author-default DeepShift ranges would clip 13/40 taps.

### 3.3 One paired recovery is the wrong budget for this gap

P06’s protocol: one FT ≤ original ep34, no λ grid. The +0.013 hole was left by a **320-step** recovery that was allowed to use unconstrained reals. A one-shot STE/LSQ onto `{2^{-k}}` or CSD Hamming ≤2 is asking a smaller set to beat a larger set’s failed optimum. P03-I5’s own prediction was “PTQ-CSD fails AEE”; F3 offers no new reason that QAT would land in [1.220, 1.225) when unconstrained 1.233 did not.

Even if QAT **hits** ≤+0.005, that is an **algorithm** result (discrete FIR codebook on DSEC). TCAS-II still needs a circuit increment after ordinary is given the **same** alphabet (F3’s own control). Ordinary 10×10 under a dyadic codebook is still one da4ml CMVM with terminal cutoff — the thing that already has 0 intermediate RNE.

### 3.4 Deleting RNE without QAT is already known-lossy

Gate-collapse composed q12 half-steps with `Fraction` and re-quantized to signed16. The new 10×10 is a **different function**; the probe forbade inheriting 1.233 AEE. So “just delete the 35 barriers” is not a free accuracy win. F3’s assumption that a non-trivial fraction of +0.013 **is** Q-format noise from the 35 barriers is untested and, given unconstrained training already saw those barriers, probably false: the network was trained **with** the half-step RNE in the numeric helper.

**Verdict on Attack 2:** dyadic/CSD is a subset of the alphabet that already lost +0.013. Shared and hard-sign (more structure) lost more. PTQ will be worse; one-pair QAT is not a plausible recovery of 0.008 AEE against a 320-step unconstrained run. If QAT somehow passes, ordinary-with-same-alphabet remains the hardware control.

---

## 4. Attack 3 — Would generic fusion already capture the hardware win?

**Yes, on the only axis that moved, and the axis that matters did not move.**

Same two-stage SIMD, 8 lanes, 2R1W RF, two writeback slots, 50 B extra pipe state charged to **all** arms (`two_stage_writeback/README.md`):

| Arm | always-ready | long-BP | FIFO-full wait | add / round / sat / norm24 / gate |
|---|---:|---:|---:|---|
| ordinary (fusion on or off; nothing to fuse) | **6938** | **8088** | 1135 | 260 / 0 / 0 / 0 / 10 |
| lifting unfused | 6194 (−10.72%) | **8088** | 1879 | 159 / 35 / 35 / 0 / 10 |
| lifting fused `round→sat` → `norm24` | **5354 (−22.83%)** | **8088** | **2719** | 159 / 0 / 0 / **35** / 10 |

Facts F3 is not allowed to re-spend:

1. **Fusion already deleted the 35 round and 35 sat issue slots.** 24 SIMD batches × 35 = 840 slots = 6194→5354 = **−13.56%**. Independent review (`pipeline_review.md`): fusion is a **generic** instruction combine; ordinary has no such interior pairs, so it cannot be charged a matching 13%. That 13.56% **is** the hardware story of “don’t issue round and sat separately.” It is A, and it is already in the −22.83% headline F3 must not retitle.
2. **Long backpressure is 8088 on all four arms.** Source-node gain is absorbed at the exit. PROBLEM.md: do not sell −22.83%. F3’s own kill says “or long-BP still 8088.” The freeze already trips that kill **after** fusion.
3. **Faster source makes the stall worse, not better.** FIFO-full wait **rises** 1135 → 1879 → 2719 as lifting ALU shrinks. Deleting the remaining 35 `norm24` identities would be another ~840 always-ready slots (5354→~4514) and **more** FIFO-full wait, with 8088 unchanged. That is the opposite of a chain-service win.
4. **Composition (true zero interior RNE) increased arithmetic.** 241 ≥ 159+35. You cannot claim both “zero RNE” and “keep the 159-node graph” unless you keep 35 **writebacks of lifting state** (predict/update memories). Those writes are the lifting **layout**, not rounding. Huang flipping-structure DWT (TSP 2004) already bound that layout to two memories — the same two write slots this SIMD has. P03-I8 is that prior; F3 does not add it.
5. **Integer consumers are already 0-diff** on gates / I24 / PED q24. The 35 RNE exist to **preserve** that identity under half-step Q. Deleting them either (a) keeps the same values, in which case they were never a numeric problem, only issue slots — and fusion already ate the slots — or (b) changes values, in which case 0-diff and likely AEE break.

SCOPE / CURRENT_LINES: do **not** start paired recovery from source-ready grey; full chain is not PASS. F3 wants to spend the one pair **now**, on an island whose measured bottleneck is not the 35 RNE.

**Verdict on Attack 3:** generic fusion captured the issuable hardware win of the 35 RNE. Residual 35 `norm24` are identity cuts, not extra ALUs. Deleting them cannot move 8088 on this port geometry; it would likely deepen the already-larger FIFO stall.

---

## 5. Attack 4 — What a TCAS-II reviewer says after HPCA lifting/MCM literature

Five-page Express Brief, binary accept/reject, punishes missing **relative prior** and missing **measured advantage**. After ESTU + da4ml + the MCM/lifting stack, the letter is a reject.

### 5.1 Same-journal and 2025–2026 A the reviewer will name (W1)

- **ESTU, TCAS-II Dec 2025:** “spiking transformer on a small FPGA.” Any letter whose mechanism is “we mapped a spikeformer datapath” is a reskin. F3 is not even that; it is a **constant FIR compile**. ESTU still occupies the journal’s SNN-transformer slot; F3 must differ in mechanism. “Integer lifting PSN” is a DWT datapath, which this journal has published for two decades.
- **da4ml, TRETS 2026:** complete CMVM adder-graph compiler, CSD+CSE, exact arithmetic. Local compile **is** da4ml. Reviewer: “you ran their compiler on a 10×10.”
- **ROM-LTE, CICC 2026:** ROM/LUT tensor for constants, fine-tune-free. Alternate A if anyone says “lookup the mix.”
- **Phi, ISCA 2025:** Level-1 offline pattern×W precompute. Cousin of “compile a constant mix.”
- **NeuroFlex, 2025:** “integer-exact” ANN/SNN column hybrid. Name collision with F3’s title; mutually exclusive per column, not dual-consumer T10. Still: integer-exact is not a 2026 invention.

### 5.2 HPCA-class constant-arithmetic / QAT stack (the “HPCA lifting/MCM” hit)

Architecture reviewers (HPCA/MICRO/ISCA) will not treat multiplierless FIR as new:

| Prior | Why it covers F3 |
|---|---|
| **LUT-DLA, HPCA 2025** | Constant/table MAC, staged train, LUT-stationary. Compile-once constants. |
| **Prosperity, HPCA 2025 / GustavSNN, HPCA 2026** | Wrong sparsity model for continuous θg, but they set the bar: a hardware letter needs a **dataflow** increment (NRV, CPTB, product reuse), not a smaller adder count. F3 has no dataflow X. |
| **Hcub (Voronenko–Püschel), RAG-n, DAC 2004 time-multiplexed MCM** | MCM onto a **bounded adder set** — this machine already *is* two write slots. Add-optimal CSE vs port-optimal MCM is P03-I1, not F3. |
| **ARITH 2025 *Hardware-Aware Training for Multiplierless CNNs*** | QAT that **costs the MCM adder graph during training**, progressive freeze, emit the adder graph. F3’s “one paired QAT so coeffs live in a dyadic/CSD alphabet then da4ml” is this paper’s loop on 40 taps. Local projection note already marked it **A**. |
| **DeepShift / INQ / ShiftAddNet / AdderNet / MFPSN (2501.14490)** | PoT+STE. L7: replacing lifting by MFPSN/PoT without a consumer-interface delta is not X. CURRENT_LINES: not X. |
| **Calderbank 1998; JPEG2000 5/3; Andra–Chakrabarti–Acharya TSP 2002; Huang flipping DWT TSP 2004; TCAS multiplierless 5/3–9/7 letters** | Integer lifting **is** the venue’s own archive. Learned 40-coeff T10 does not make I2I new unless dual-consumer 0-diff **and** moved long-BP **and** recovered +0.005 all hold. None hold. |

### 5.3 Likely first-paragraph reject

> This is JPEG2000 integer lifting plus QAT (DeepShift/ARITH 2025) compiled with da4ml/Hcub, on a student that already **loses** to the dense 10×10 by +0.013 AEE. Generic round→sat fusion already removed the 35 extra issues (−13.56%); long backpressure is tied at 8088. Ordinary already has zero intermediate RNE. Where is the circuit?

If the authors answer “event-camera SNN-Transformer,” the reviewer points at ESTU / FireFly-T / SpikeTA / Bishop and asks for a **dual-consumer continuous-θg** mechanism. F3’s X does not mention the two consumers except as 0-diff witnesses. If they answer “we share one adder graph between gate and PED,” that share is already true of the 159-node CSE (both consumers read the same T10 vector). Sharing is A; F1 is the island that actually attacks last-use / 8088.

Measured-advantage failure is independent of novelty: relative AEE fail, chain service unknown, 8088 tied, fusion already counted.

---

## 6. Alternative explanations of the 35 RNE (all cheaper than F3)

1. **They are half-step Q checkpoints of a lifting delay line.** Predict/update must write a 24-bit state. Rounding policy is RNE because the numeric helper is RNE. Fusion packs round+sat. Nothing left for a letter.
2. **They are why 0-diff holds.** Delete them without retraining → function change (noncommutation example; gate-collapse). Retrain → algorithm paper, ordinary must get the same codebook.
3. **8088 is output-port bound** (`ceil(outputs / 2 slots) × fill`). Then **no** CSE/QAT/I2I variant moves long-BP (P03-I1’s own biggest objection). FIFO-full wait already scales the wrong way. Occupancy traces in the two-stage compare are consistent with this.
4. **Always-ready −22.83% is CSE + fusion, not lifting magic.** Unfused lifting is only −10.72%. Residual after fusion is still absorbed at 8088.

Any one of (1)+(3) kills F3 without training.

---

## 7. Measurement / sequencing failures F3 would commit

- Spending the **one** paired recovery **before** full-chain same-port service is known, against CURRENT_LINES / two-stage README: “完整链仍未 PASS，也不据源收益启动精度恢复.”
- Reporting 260→159 or 35→0 as service. Node count ≠ cycles (L7; da4ml S0 10×10 node −49% ≠ cycle −49%).
- Adding source-table −22.83% to consumer-table −5.78%.
- Quoting fused 5354 vs 6938 as F3’s gain (fusion is A; long-BP tied).
- PTQ-CSD without the ordinary-same-alphabet arm.
- Inheriting 1.233 AEE onto a floor/I2I or composed 10×10.
- Calling PoT QAT “circuits+systems co-design” if 8088 does not move (P06-I1 already called that a legal-charge failure).
- OpenROAD/Yosys on the ~76-cell half-step RNE iverilog toy (TECH_ARCH MP) as PPA.

---

## 8. What is *not* F3 (do not salvage by rename)

L7’s narrow leftover — **consumer-specific** rounding: gate trunc/sat, PED q24 sticky, one DAG, no clone — is P03-I3. F3 explicitly demands **zero** interior RNE for both sinks. If I3 is ever run, it belongs under the dual-consumer island (F1), with kill: if typed exits only relocate the same 35 writes, or if both consumers need full q24 (likely: integer 0-diff today is on **both** I24 and PED q24).

PoT QAT remains CURRENT_LINES **Step 2 control**, after a full-chain ≥15% opportunity exists, with ordinary given the same codebook. That experiment must not be retitled F3 if it runs.

---

## 9. Residual uncertainty (does not save the candidate)

- Full-chain same-resource net service is still **unknown**. That uncertainty argues **against** spending the pair on F3, not for it.
- Whether 8088 is a hard output-port lower bound is not theorem-proved; FIFO-full wait scaling is strong circumstantial evidence. An occupancy trace that showed the 35 `norm24` on the long-BP critical path would be surprising given fusion already removed 70 issue ops without moving 8088.
- A dyadic QAT that **recovers** ≤+0.005 is not logically impossible; it is inconsistent with unconstrained 1.233, shared 1.248, hard-sign ~1.6, and 0/40 PoT digits. If it happened, F3 would still lack a circuit X vs ordinary+same-alphabet+fusion.

---

## 10. Decision

| Item | Call |
|---|---|
| **Disposition** | **stop** F3 as a fusion / title island |
| **Novelty** | **1 / 4** |
| **Performance** | **1 / 4** |
| Family | lifting T10 may remain; this **layout** (I2I-by-alphabet deleting 35 RNE) stops |
| Do not do | one-pair dyadic/CSD QAT as the next hardware action |
| Allowed later, not as F3 | PoT/DeepShift **control** after full-chain 15% exists; typed dual-exit only if parked under F1 and pre-registered against “both sinks need q24” |

**One-sentence kill:** F3 is QAT + da4ml + JPEG2000 rounding, already A; generic fusion already cashed the 35 RNE as −13.56% always-ready; 8088 did not move; +0.013 AEE is the unconstrained student, so a dyadic subset should miss the relative gate even harder.

End of ADV_F3.
