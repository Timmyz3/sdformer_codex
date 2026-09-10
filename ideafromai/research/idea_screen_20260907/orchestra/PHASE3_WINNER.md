# Orchestra Phase 3 — Refine the winner

Skill goal: turn the top idea into a concrete research plan.

**Status of this “winner”:** process output of Orchestra Phase 3. It is **not** a locked paper title and **not** a scientific conclusion. K-Dense forbids automatic selection; grill-me forbade picking one title before screening. P01 (human owner) has not signed. If M0 shows the attention leaf < ~2% of ep34 cycles, the *circuit face* switches to I004; I001 remains the algorithm-native leaf/ablation.

---

## 1. Two-sentence pitch (F10)

Event-camera spike Transformers for optical flow currently score Q-K with Motion-XOR (overlap + co-silence + temporal-peer K XOR), so published AND-PopCount and linear-Q-K accelerators do not implement the datapath the net actually uses — the brief would otherwise be a Prosperity/LoAS copy.

We implement one 32-lane triple-popcount leaf with a t0 `K_peer` shadow, and we clock-gate that leaf per token when K is zero both timesteps or Q/K are clean versus t0 (not a 15×15 Shiftmax memo, because window-clean is ~10% on the ep35 preview).

---

## 2. Core tension (F3)

**Mechanism novelty (temporal XOR) ↔ system acceleration that may be ~0.**

If the tension is an artifact of an unmeasured envelope, T0 removes it. If it is real (leaf < ~2%), characterizing the Pareto frontier *is* the contribution: publish the leaf honestly, and put system PPA on I004.

---

## 3. Abstraction level (F2)

**Down:** a combinational leaf and its enable. Not a new backbone, not a whole-net accelerator, not a skip theory paper.

---

## 4. Three concrete experiments

All three are already named in `../MEASUREMENT_CONTRACTS.md`. They validate the *idea*, they do not train a new SOTA.

1. **M0 / T0.** ep34 cycle or MAC-proxy share of score leaf vs Q/K proj vs conv vs FC vs decoder vs ATLIF. Gate: leaf < ~2% ⇒ I001 is not the circuit face of the abstract.
2. **M1 + M2.** ep34 packed Q/K census (same fields as `../T1_T4_ep35.json`) and Shiftmax-row alignment. Gate: if the true row is ~100% dirty, skip is energy-gate only.
3. **Same-resource leaf cycles.** MX3P vs FireFly-T-style AND-PopCount vs software Q7, equal lanes/ports. Cheap add-on: **I012** term-wise skip of overlap (ep35 mean 0.013) vs same_zero vs motion — high matrix rank (77.5) because it is a cheap discriminator, **not** because it is a second title.

Not in the two-week window: A800 valid825, production RTL, DC/PT.

---

## 5. Strongest objection and response

**Objection:** a circuits editor will not take a leaf with no system PPA, and FireFly-T already did AND-PopCount.

**Response:** (a) T0 is the first experiment, not a footnote; (b) the third popcount is temporal-peer XOR, which AND-PopCount does not do — that is the native claim, and it dies if I003’s one-block swap keeps AEE; (c) we will not label ep35 as ep34; (d) we will not call token skip a Shiftmax skip; (e) if T0 fails, I004 is the pre-declared circuit-face switch, not a late pivot.

---

## 6. Two-week feasibility pilot

Still measurement, not production RTL.

| Week | Work | Signal |
|---|---|---|
| 1 | Obtain or unpack **ep34** Q/K; rerun `analyze_t1_t4.py` identity; write M1 JSON with the same keys as the ep35 preview | If window-clean stays ~10% and S3=0, skip-claim stays token-only |
| 2 | M0 share table on the same ep34 book as C1/C2; if leaf share ≥ ~2%, isolated sim of token-enable on `h67_motionxor_score_q7` (not a new island) | If leaf share < ~2%, stop I001-as-circuit-face and open I004 M4 |

A800 full valid825 is **not** in these two weeks. I003’s 10-frame overlay is the cheapest *algorithm* probe and may run in parallel if GPU idle, but it is not required to finish this pilot.

---

## Completion checklist (skill)

- [x] Two-sentence pitch is specific (Motion-XOR + token enable, not “improve efficiency”)
- [x] Problem-first check passed (`F1_MODE.md`)
- [x] Simplicity: existing SV leaf; skip is an enable; window memo dropped
- [x] Stakeholder: circuits author / TCAS-II reader (`F8_STAKEHOLDERS.md`)
- [x] Core experiments specified (M0, M1+M2, leaf vs AND-PopCount)
- [x] Feasibility pilot defined (two weeks, measurement)
- [x] Strongest objection has a response (T0 switch to I004; no dishonest skip)

**Human gate:** P01 confirms or overrides D001. This agent does not implement.
