# Decision log (K-Dense step 10)

Do not rewrite after outcomes are known. Append only.

---

## D001 — 2026-09-07

| Field | Value |
|---|---|
| Owner | P01 (human). **Not signed.** P02 (facilitator) drafted this entry. |
| Scope | Next information-gathering action for the TCAS-II co-design object. Not ethics approval, not a study, not a title lock. |
| Candidates considered | I001, I002, I003, I004, I005, I007 (Orchestra shortlist). I006, I008–I013 visible as rejected/deferred/parked. |
| Criteria / weights | information_gain 3, originality_vs_search 3, feasibility 2, aee_risk 2 (lower-better). Anchors in `WORKFLOW.md` §6. Weight-delta 0.10 one-at-a-time. |
| Raw ratings | `scores.csv`. P02 only; no second rater; no abstentions. |
| Matrix | `matrix.json` (13 ideas, rebuilt after I012/I013). Tool `decision` = **null**. Presentation ranks: I001 87.5, **I012 77.5**, I002 67.5, I007=I004 60.0 (rank range 4–5), I003 52.5, I005 50.0. I012 is a term-skip **measurement** hanging on I001, not a sixth title. I001 rank range 1–1 under ±10% weight perturbation, **but** its score interval [62.5, 100] overlaps I012/I002 — do not treat 87.5 as a gap. |
| Literature dates | 2026-09-07 bounded search; see `LITERATURE_REOPEN.md` |
| Adversarial | `ADVERSARIAL_REVIEWS.md`; I002 merged into I001 |
| Gates | ethics not-applicable; biosafety/dual-use not-applicable; regulatory not-assessed (PDK/export institutional); unpublished traces keep-local |
| **Decision** | Advance **I001 including I002** (and I012 as a census add-on) to measurement protocol **M0–M2**. If T0 shows attention **score leaf** under about 2% of ep34 cycles, switch the *circuit face* to **I004**. **I003** is the cheapest algorithm-face probe (optional parallel). **I005** remains a comparator, not a title default. **I007** stays fifth. P01 may override. |
| Rationale | Orchestra Phase 3 requires one sharpened plan. I001 is algorithm-native, has SV, and T1 already forbids dishonest skip claims. K-Dense forbids automatic winners; the matrix only ranked. The next_action is simulation, not protocol development or preregistration. |
| Rejected as title defaults | I006 (Prosperity *is* the prior); I008 (no Table-A); I009 (macro unknown); I010 (training-only); I011 (PAFT 1.47 identity); I013 (different object, CICC analog). |
| Dissent | P03: I001’s numeric lead is conditional on T0 and on XOR being accuracy-necessary; if either fails, ranking first is misleading. P02: agrees the lead is a decision aid. P01: **not yet recorded**. |
| Uncertainties | T0 unmeasured; no ep34 QK pack; 2% threshold is an assumed gate; single rater; same-model adversary. |
| Next action (allowed vocabulary) | **simulation** (M0, M1, M2). Not further search as the blocker; not preregistration. |
| Revisit when | ep34 T0 envelope exists, **or** P01 files dissent, **or** I003 10-frame AEE exists, **or** 14 days after this log with no M0. |
| Unresolved risks | Selling a leaf as system PPA; ep35/ep34 identity confusion; authority bias from P02 drafting D001. |

D001 does **not** authorize RTL, training, or Codex implementation. Those start only if P01 says so. Codex’s first work, if accepted, is `../MEASUREMENT_CONTRACTS.md` M0+M1.
