# ADV_F2 — Paid full-domain projection BN

**Freeze:** 2026-09-11  
**Reviewer (not an originator):** adversarial stand-in. Did not write P01–P06 or cluster F2. Same model family as the originators — this is **not** independent-lab review.  
**Object reviewed:** `fusion/FUSION_CANDIDATES.md` **F2**, clustered from P01-I3, P02-I6, P03-I6, P04-I7, P05-I3, P06-I7.  
**Contract:** `PROBLEM.md` / `SCOPE.md` / scientific-brainstorming step 7 (`idea_evaluation.md` adversarial template). Review the strongest version first.  
**Ask (controlling):** is F2 a **TITLE**, or only a **mandatory cost that any honest Stage B must pay** (hygiene, not X)?

Ideas and scores below are **proposals**, not findings. No RTL, no new AEE, no service number is closed in this sheet.

---

## Card as clustered (do not steelman yet)

| Slot | F2 text |
|---|---|
| **A** | LoopTree retain/recompute; Welford; RISCSparse frozen `Y=aX+B` as **negative** control. |
| **B** | Captured native proj BN uses actual batch stats on `10×96×120×160`; free μ/σ on local windows undercharges. |
| **X** | Welford-on-fill on the same 2 ports; a tile is not a closed endpoint until the stats barrier; no frozen-BN cheat. |
| **Kill** | If charging the barrier makes lifting worse than ordinary, **or** if switching to running-stats changes AEE >+0.005. |

Shared session gates still bind: AEE ≤1.259 and ≤ ordinary+0.005; full-chain same-port/state/backpressure net service ≥15%; integer windows 0-diff; 8088 long-BP must move; no binary ATLIF identity; do not add −22.83% and −5.78%.

---

## Strongest version of the idea

Keep the **captured** student’s projection BN: actual batch moments over the full `10×96×120×160` domain. Do not freeze, do not gift μ/σ to a local window, do not fold `a,b` at compile time unless a measured AEE control licenses it.

On the **same two write ports** that already emit `proj.conv` (or the projection tensor the BN legally reads), update a Welford pair `(μ, M2)` with **zero extra 1RW** beyond fills the projection already pays. Domain-ready is a **broadcast barrier**. Affine `γ(x−μ)/√(σ²+ε)+β` of any tile is illegal until that barrier. After the barrier, LoopTree’s trichotomy is explicit for the activation tensor: **retain**, **refetch**, or **recompute** — never “tensor already in a size-1 buffer.” Gate and PED may last-use a tile after BN-stat has **sampled** that tile; residual add waits on **domain-ready**. RISCSparse frozen `Y=aX+B` is a **numeric control**, allowed to kill the island. Two-window integer 0-diff vs the captured BN at the same q24 policy that already matches gates/I24/PED. Service is reported only against ordinary + **honest** two-pass BN on the same ports, never against free-μ/σ replay.

That is the strongest REDUCE reading (P01-I3 / P02-I6 / P05-I3). It still has to beat BNFF and LoopTree, not just beat a cheated local window.

The cluster also contains the opposite fork (P03-I6 FOLD / P06-I7): one paired freeze+re-estimation so RISCSparse §3.4 becomes legal. The strongest FOLD reading is: **measure** the AEE gap, close it with one pair, delete the 18.4M reduction. It is **not** the same X as REDUCE. Treating them as one island is already a defect of F2 as clustered.

---

## TITLE or hygiene? (the ask)

**Hygiene, not a title.** F2 as written is the accounting identity that `PROBLEM.md`, LoopTree takeaway 5, and RISCSparse §3.4 already require of every honest Stage B on this net. It is a **shared kill-gate** on F1/F3/F4, not a five-page mechanism.

Why the written X is not X:

1. **“A tile is not a closed endpoint until the stats barrier”** is LoopTree’s schedule constraint applied to a global reduction. L4 already maps `proj.norm_layer` as an **untiled (or two-pass) fusion constraint**. That sentence is a legality rule for F1 last-use, not a new circuit.
2. **“No frozen-BN cheat”** is a measurement rule. RISCSparse’s inference fusion **is** `Y=aX+B` with compile-time `a,b`. Forbidding that cheat is how you stop undercharging. It is not a contribution.
3. **“Welford-on-fill on the same 2 ports”** is Welford (1962) plus BNFF Mean-Variance Fusion (SysML 2019): statistics ride the producer write so you do not pay a dedicated stats reread. Affine still cannot start until the last of 18.4M sites has contributed. Fill-hidden stats do **not** hide the affine barrier. Storage during the barrier is still retain / refetch / recompute of the **activations**, which LoopTree already names.
4. **F1 already owns the occupancy union.** F1’s X is: a source word retires only when `gate ∧ PED ∧ (BN-stat if live)` complete. F2 is that third conjunct expanded into a fake island.
5. **The clustered kill-gate is incoherent**, which is what a cost-cluster looks like when forced into title shape. Independent REDUCE kills if **frozen** AEE is already inside +0.005 (then fold; no circuit letter). Independent FOLD kills if the gap is already inside +0.005 (no training story) **or** if freeze+FT still misses +0.005. F2’s card says kill **if running-stats changes AEE >+0.005** — that inequality is the case that *keeps* REDUCE, not the case that kills it. The other clause, “charging the barrier makes lifting worse than ordinary,” is an **F3 / lifting-title** kill, not an F2 existence proof. Hygiene can invalidate lifting. Hygiene does not become the letter.

What F2 **is**: Stage B must (i) put full-domain moments in the live set, (ii) report retain/recompute/refetch for that tensor, (iii) run frozen-BN as a numeric control, (iv) never quote service against gifted μ/σ. Any island that skips those four is an invalid experiment, not a competitor.

---

## Observation that would count against it

Any one of these, predeclared:

1. **Frozen running-stats (RISCSparse `Y=aX+B`) on projection BN only**, same integer pipeline, valid825 AEE ≤ 1.259 and ≤ ordinary+0.005 **and** two-window 0-diff. Then B is a capture-mode artifact; FOLD is a compiler footnote; REDUCE has no island. This is P01-I3’s first gate, and it is the honest reading of F2’s AEE clause once the inequality is un-inverted.
2. **Magically-instant BN** (gifted μ/σ, illegal as a *claim*, legal as a *counterfactual*) leaves long-backpressure at **8088**. Then BN is not the 8088 absorber. F2 cannot be the service title. It remains a storage tax that Stage B still pays.
3. **Shape dump:** `proj.norm_layer`’s reduction tensor is **not** the T10 PSN delay line. L4’s r1 graph is `sn2/PSN(T10) → conv2 → r1out → proj.sn → proj.conv → proj.norm_layer`. Projection BN sits **after** stride-2 `proj.conv` at `10×96×120×160`. T10 PSN occupancy is a different tensor at r1 (spatially `240×320` before downsample). Then P01-I3’s “ride the noncausal T10 buffer” and P05-I7’s alias are category errors. Remaining REDUCE is BNFF-on-`proj.conv`-write, which is A.
4. Integer/pairwise Welford vs the captured BN exceeds the 0-diff / +0.005 contract, and the only repair is FrozenBN. Then REDUCE dies by its own rule (P05-I3: do not “fix” by FrozenBN).
5. After honest charging, **no** fused schedule beats ordinary + honest BN by 15% same-port (LoopTree VI-F / takeaway 5: small buffers, intra-layer 3×3 reuse ≫ one extra inter-layer read+write). Then F2 as title is dead, and F2 as hygiene has done its job: it killed a dishonest fusion win.

---

## At least two alternative explanations

**Alt-1 — ledger bug, not missing hardware.** Local-window replay was handed μ/σ. LoopTree: missing data is retained, refetched, or recomputed. RISCSparse: inference BN **deletes** μ,σ from the live set by freezing. The “hole” is that the service model took both deletions without a line in the schedule. The correction is to **charge** two-pass BN or BNFF-fused stats, then keep F1/F3/F4 as the actual islands. Welford-on-fill is how you stop lying, not how you get 15%.

**Alt-2 — 8088 is F1, not BN.** PROBLEM.md: same two-stage SIMD source, always-ready 6938→5354, **long backpressure both 8088**. That number is on the **source arms**. Projection BN is a later consumer of `proj.conv` output, after `proj.sn`. Dual-consumer last-use of compiled T10 / I24 reread can hold 8088 with BN magically instant (P02-I6’s own disconfirmer). Paying BN then **adds** wait on a different tensor and cannot move 8088. F2 “absorbs the source gain” is a conjecture SCOPE itself marks **unknown**.

**Alt-3 (kept, not counted as a third required):** capture left projection BN in train/batch mode by accident; production inference on this family was always running-stats. Then F2 is a bugfix of the student, not a circuit. P06-I7’s first measurement decides this. Until that number exists, REDUCE is an untested identity claim.

---

## Measurement or analysis failure

- **Gifted μ/σ as the winner’s baseline.** PROBLEM.md forbids it. Any ≥15% that exists only vs C3 is unpublished.
- **Adding −22.83% and −5.78%.** Different resource points. BN cycles land in whichever table actually issues the reduction; they must not be stacked with source CSE.
- **Welford ALU occupancy reported as chain service.** Sidecar moments are O(C=96) state. The billed object is the **barrier** (last fill → first legal affine → residual add) plus retain/recompute of activations, under the same 2 ports.
- **Laundering residual `norm1`/`norm2` freeze as projection-BN AEE.** `algorithm/patch_probe/fixed_bn_valid825_summary.json` freezes **r0/r1 residual** BNs on a **different parent** (parent frame AEE 1.165 → 1.203, Δ+0.038). L4: captured residual BNs may already be fixed; the barrier that kills closed-window HW is **`proj.norm_layer`**. That file is not F2’s C2.
- **`fixed_bn2_probe` (valid10, MLP BN2)** is another layer family. Not projection BN, not valid825.
- **Assuming the leading `10` is T10 PSN occupancy.** It is the time extent of the **projection** tensor after downsample (`10×96×120×160 = 18,432,000`). Axis identity with the PSN delay line is a kill, not a belief (P05-I7).
- **Inverted clustered kill.** REDUCE dies when freeze **passes** AEE; FOLD dies when freeze **passes** without a pair, or **fails** after a pair. F2’s card collapses both into one inequality that matches neither.
- **Overlap cartoon that starts affine of tile k before domain-ready.** Within one frame, Welford-on-fill makes pass-1 free in the write. Pass-2 affine is still a full-domain-ready barrier. Overlap with “tile k+1 fill” cannot legally emit normalized outputs until pass-1 of **all** tiles is done. Across-frame overlap is running-stats / next-batch PreciseBN — a different student.

---

## Sampling or generalizability failure

- Integer 0-diff is two captured windows. Full-domain moments are a **reduction over the whole cube**; two windows do not sample night/day DSEC shift of μ,σ.
- Ordinary AEE 1.219801338 / lifting 1.232979368 are source-family numbers, not a projection-BN freeze table.
- SCOPE: true cost of full-domain stats is **unknown**; Codex full-chain service is **not closed**. F2 currently has B as a sentence and X as a cartoon.
- If later stages also use unfrozen BN, paying only `proj.norm_layer` understates the tax; if only projection BN is unfrozen, F2 cannot claim a net-wide BN engine.

---

## Prior evidence that challenges it (challenging priors)

Complete A, already named in L4 and in the independent cards. None of these is a local PPA import.

| Prior | What it already is | What F2 does not add |
|---|---|---|
| **Welford 1962** / Knuth / **Chan–Golub–LeVeque 1983** | Online and pairwise moments | A numeric method, not a TCAS-II circuit |
| **Ioffe–Szegedy 2015** | Two-pass train BN: reduce, then affine | The barrier F2 “discovers” |
| **BNFF, SysML 2019** (Jung et al., arXiv:1807.01702; already in `projection_chain_prior_review.md`) | Fission BN into stats + normalize; fuse stats into preceding CONV (MVF: one scan of mean and second moment on the producer write); fuse affine into successor. **Reduces DRAM; does not delete the full-domain completion dependence** | This **is** Welford-on-fill. Local review already wrote the sentence F2 needs as A: source must wait, retain, or reread; you may not write the barrier as lifted by a local tile |
| **LoopTree, TCASAI 2024** (local author text) | Retain vs recompute vs refetch; per-tensor retain; fusion set is an **input**; takeaway 5: small buffers → layer-by-layer can beat tiled fusion | Charging `proj.BN` moments is LoopTree-complete mapping of this fusion set, not an increment |
| **RISCSparse, ICCAD 2024** §3.4 (local author text) | Inference BN is frozen `Y=aX+B`, folded into SA | Negative control **and** the production default. If C2 passes AEE, F2’s REDUCE is a refusal to copy A |
| **TensorRT / TVM / Glow / RepVGG** | Fold running stats into conv | FOLD’s entire hardware story |
| **PreciseBN / Wu–Johnson “Rethinking Batch in BN”** | Recompute true batch/population stats rather than hope EMA | P04-I7 / P06-I7 method prior, not X |
| **ISSCC 2025 23.2 LFS** | KV-then-weight slot replace; **lossy** pad of missing halo | Does **not** contain a global BN. Cannot repair moments by attention RF |
| **cuDNN / TPU-style two-pass BN** | Sidecar reduction `(count, mean, M2)` beside the datapath | P05-I3’s sidecar is this unit |

L4’s own closer (not an F2 pitch): a brief on r1 fusion that does not keep projection BN’s full-domain moments in the live set will read as a **rename**. That is a **reviewer hygiene demand**, which is the opposite of a title grant.

---

## Reskin risk (LoopTree / Welford / RISCSparse frozen BN)

High. Predicted first-paragraph mappings:

| If the letter says… | Reviewer maps it to… |
|---|---|
| “online / streaming BN,” “Welford accumulator,” “moments on the fill” | Welford 1962 + BNFF MVF (stats fused into CONV write) |
| “tile cannot retire until stats ready,” “retain vs recompute the fmap” | LoopTree, including takeaway 5 against dishonest local fusion |
| “we do not freeze BN” without a measured AEE gap | Capture artifact; why is this not eval-mode? |
| “we freeze BN and fold `Y=aX+B`” | RISCSparse §3.4, TensorRT, RepVGG — **complete A** |
| “GroupNorm / PreciseBN instead” | Copied alternative student (P04-I7 controls), new AEE, not this net’s identity |
| “third consumer of the source” | F1 last-use with a 2-word sidecar; occupancy union, not a BN paper |
| “same 2 ports” as the novelty | Port constraint from PROBLEM.md, not a mechanism |

A TCAS-II Express Brief has **one** mechanism. “We finally billed BN” is not one. “Welford rides the write port” is BNFF. “We folded BN” is RISCSparse. Surviving as title would require a **measured** same-port win of overlapped two-pass vs BNFF/LoopTree-complete two-pass **after** frozen-BN fails AEE — and even then the increment is a compiled grant calendar, which F1 already claims.

P06-I7 inside F2 is a reskin attractor: if freeze+FT works, the hardware disappears and the remaining sentence is QAT of running stats. That is not this journal’s object unless the measured gap on **this** `10×96×120×160` student is the whole brief, which it should not be.

---

## Potential harm, inequity, or misuse

Not applicable at clinical / dual-use level. Practical harms: (i) spending the only five-page slot on accounting; (ii) shipping a dishonest ≥15% if C3 remains the baseline; (iii) changing paper identity by freezing BN without a valid825 row, or by renaming ATLIF into a binary skip so BN’s domain becomes sparse (forbidden). Unpublished traces stay local.

---

## Mitigation

1. **Demote F2 from letter-island to shared Stage B gate.** Rewrite the card from X to: *every F1/F3/F4 service table must include full-domain projection-BN retain/recompute/refetch; gifted μ/σ invalidates the run; RISCSparse freeze is a numeric control.*
2. **Split REDUCE and FOLD.** They are mutually exclusive. FOLD is RISCSparse A plus one AEE number (P06-I7’s first day). REDUCE is not a title unless M1 fails **and** M2 shows the barrier is first-order on the 8088 path **and** overlap beats BNFF two-pass by 15% on the same 2 ports — three conditions that F2 has not earned.
3. **Two measurements before any BN RTL** (CPU / service model; SCOPE horizon):
   - **M1.** valid825 AEE: actual-batch `proj.norm_layer` vs frozen running stats vs precise re-estimation **without** weight FT vs one paired FT. Ordinary 1.219801338 is the relative rail. Do not proxy with residual `norm1/norm2` or MLP BN2.
   - **M2.** Same 2 ports: C3 free μ/σ (illegal FOM, show undercharge only) vs C4 two-pass no overlap vs BNFF/Welford-on-fill vs magically-instant BN counterfactual. Report barrier cycles **last projection write → first residual add**, plus extra 1RW beyond fills, plus extra SRAM vs O(C).
4. **Print shapes the same day.** `proj.norm_layer` input vs T10 PSN buffer vs `conv_res`. If the `10` is not the PSN delay line, delete P01-I3 overlap-in-T10 and P05-I7 alias. Do not “fix” with a second cube (that is the storage F2 promised to avoid).
5. **If M1 passes:** fold, stop REDUCE, cite RISCSparse/TensorRT, put the AEE row in the supplement of whichever island survives. **If M1 fails and M2 tax is off 8088:** charge as footnote in F1. **If M1 fails and M2 tax is on 8088:** F2 still does not automatically become the title; F1’s dual-completion already includes `BN-stat if live`. Add a 2-word sidecar and a domain-ready bit, do not retitle the letter “Welford.”
6. **Fix the clustered kill-gate** if F2 remains in any register: kill REDUCE iff C2 AEE ≤ ordinary+0.005; kill FOLD iff gap ≤+0.005 with no pair **or** pair still >+0.005; kill any service claim that uses C3 as the control to beat.

---

## Residual uncertainty

- M1 (frozen vs batch AEE on **projection** BN) does not exist on the freeze date. Residual r0/r1 freeze Δ+0.038 on another parent is **not** a substitute, but it is a warning that freeze is not free on this family.
- M2 (cycle/storage tax) is the SCOPE unknown. Without it, “BN absorbs 8088” is a story.
- Whether integer Welford matches the student’s CUDA BN inside 0-diff (PROBLEM.md already records FP32 CUDA rounding diffs on other paths).
- Whether later unfrozen BNs exist on the captured student; F2 names only native projection BN.
- Whether Codex full-chain closure will show BN wait inside or behind 8088.
- Whether a compiled grant calendar that includes domain-ready is distinguishable from F1 in five pages. If not, F2 has no separate figure.

---

## Disposition

**Revise — stop as title; retain as mandatory Stage B hygiene / shared gate.**

- **Stop** writing F2 as a TCAS-II mechanism or as a peer island to F1/F3/F4.  
- **Retain** M1/M2 and the four hygiene rules (live moments; LoopTree trichotomy; freeze as control; no gifted μ/σ).  
- **Pause** any Welford-on-fill / sidecar RTL until M1/M2 exist.  
- **Do not** keep P06-I7 FOLD in the same X sentence as “no frozen-BN cheat.”

Revisit trigger: M1+M2 closed. Only then may someone argue for a **footnote circuit** (domain-ready bit + 2-word accumulator) inside F1. That still is not a new letter identity.

---

## Scores (anchors; not selection)

Session grok rubric for novelty: 0 none; 1 prior only; 2 hole but X not located; 3 measurable X not closed; 4 title-level. Performance anchors for this sheet: 0 cannot help / only makes reported service worse; 1 accounting-only, 15% as standalone structurally implausible; 2 unmeasured, could matter if BN is the 8088 absorber; 3 plausible 15% after overlap vs two-pass; 4 expected to clear 15% and AEE as the letter.

| Criterion | Score | Reason |
|---|---|---|
| **Novelty** | **1** | Welford + BNFF + LoopTree + RISCSparse already name stats-on-fill, the affine barrier, retain/recompute, and frozen `Y=aX+B`. The undercharge is a **measurement bug**. A sympathetic reading is 2 (SCOPE lists the cost as unknown); this review does not grant 2: an unmeasured tax of a 1962 algorithm is not an unlocated X. |
| **Performance** | **1** | Honest charging can only **reduce** reported service vs C3. Welford-on-fill saves at most a dedicated stats reread; affine remains a full-domain barrier; 3×3 intra-layer reuse dominates one extra pass (LoopTree takeaway 5). Source 8088 is on a different tensor. FOLD, if M1 passes, **deletes** work but that win is RISCSparse A, not F2 X. |

Do not multiply these scores into an accept probability. Qualitative gate: **not a title**.

---

## What this does to the other islands

- **F1** must list `BN-stat if live` in last-use **and** pay M2. If M1 passes, BN-stat is not live at runtime (folded); F1 occupancy is `support(θg)∪support(PED)` only. If M1 fails, F1’s third conjunct is a 2-word sidecar + domain-ready, not a second paper.
- **F3** (integer lifting) must nest honest BN in the same-port ledger. F2’s “charging makes lifting worse than ordinary” is exactly the hygiene that can kill lifting without becoming the replacement title.
- **F4** (prefix-sufficient T10) cannot prefix past a full-domain BN barrier unless the prefix student also changes BN (new AEE) or BN is frozen (M1).

---

## Sources (absolute)

- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/exploration_tcasii_20260911/fusion/FUSION_CANDIDATES.md`
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/exploration_tcasii_20260911/PROBLEM.md`
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/exploration_tcasii_20260911/SCOPE.md`
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/exploration_tcasii_20260911/literature/L4_isscc_fusion_looptree.md`
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/exploration_tcasii_20260911/independent/P01_circuits.md` (I3)
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/exploration_tcasii_20260911/independent/P02_architecture.md` (I6)
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/exploration_tcasii_20260911/independent/P03_compiler.md` (I6)
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/exploration_tcasii_20260911/independent/P04_event_of.md` (I7)
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/exploration_tcasii_20260911/independent/P05_dual_consumer.md` (I3, I7)
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/exploration_tcasii_20260911/independent/P06_train_hw.md` (I7)
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/algorithm/patch_probe/residual_consumer_probe/projection_chain_prior_review.md` (BNFF as A)
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/algorithm/patch_probe/fixed_bn_valid825_summary.json` (wrong-layer freeze; do not launder)
- LoopTree / RISCSparse / ISSCC 23.2 local author texts cited from L4
- Scientific-brainstorming adversarial template: `idea_evaluation.md` step 7

**Evidence status:** priors above are **located** in local texts. F2’s own M1/M2: **not-checked** (unmeasured). This sheet does not close them.
