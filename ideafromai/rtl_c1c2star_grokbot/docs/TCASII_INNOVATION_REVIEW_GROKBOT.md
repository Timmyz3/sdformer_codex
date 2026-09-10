# TCAS-II Innovation Review (Grok Bot)

**Tree:** `/workspace/sdformer_c1c2star_grokbot/`  
**Branch:** `tcasii/c1c2star-oss`  
**Date:** 2026-09-06 (Asia/Shanghai)  
**Priority:** sharpen what is NEW; do not inflate cousin claims. Letter = **3 knives** only.

---

## 1. What is NEW vs old C1/C2 “multiply reorder / zero-skip” narrative

| Old narrative (must retire) | C1*/C2* replacement (claimable) |
|---|---|
| C1: skip zeros / reorder multiplies for sparsity | **OP-STW**: optical-flow Δ + event density schedules *which tiles wake* for exact work |
| C1: flat exact-product quota | **PRRC + OGEC + exact_capture**: pyramid residual *budget* × occlusion match gate → finite exact enqueue |
| C1: always project QKV | **ECP-QKV** / **MW-ΔBuf**: supporting layers (predict before proj; residual-only) — not letter knives |
| C2: Bishop-style binary AAC / absorb spike into W | **HBG-RP**: binary *gate* + **non-absorbable** ATLIF **int8 proposal** (ep35 capture is binary) |
| C2: “multiply A then B reorder” for bit-sparsity | **ADP-MAC**: bilateral bit-skip — hygiene / ablation, not letter knife |
| — | **TDE3-Prior** (Card G): digital TDE-3 time-difference prior seeds OF wake → enhancer of OP-STW |
| — | **TMA-Agg** (Card G): temporal motion aggregation (split/lookup/agg) under spike/OF schedule — supporting fabric |

**One-sentence letter spine:**  
PE wake is scheduled by *optical-flow + optional TDE3 prior* (C1*), MAC traffic is *binary-gate + non-absorbable ATLIF proposal payload* (C2*), and exact work is *match-gated × budgeted* (OGEC×PRRC) — **not** zero-skip or multiply-reorder.

---

## 2. Contribution candidates ranked for TCAS-II letter

| Rank | Candidate | Why for a letter | Risk |
|---:|---|---|---|
| **1 (Knife 1)** | **OP-STW** (+ **TDE3-Prior** prior lane) | Clearest algo↔HW binding; TDE3 deepens wake novelty without fake analog CIM | Phrase TDE3 as *digital prior*, not Loihi silicon |
| **2 (Knife 2)** | **HBG-RP** (int8 **proposal**) | Dual-rail gate×payload; must keep proposal vs ep35 binary capture split | Mis-sell as frozen int8 → hostile reject |
| **3 (Knife 3)** | **OGEC×PRRC→exact_capture** | Exact path is *budgeted + match-gated* | Needs EXACT ablation rung (now landed) |
| **Support** | **TMA-Agg** | First-HW sketch of TMA-style split+lookup+agg for MFBD/Motion-TTB | Supporting fabric — not a 4th knife unless space |
| **Ablation** | ECP-QKV + MW-ΔBuf | Separates predict-before-proj / residual-only | Easy to sound like FACT / FlightVGM |
| **Ablation** | SMAM-RP + ADP-MAC | Dual-rail Mask-Add × bilateral sparse MAC | Thin vs FireFly / SMAM literature |
| **Demoted** | STH / SP-Gate / ARM-Acc / MFBD / stats | Glue / roadmap / meters | Laundry-list hazard — **demote** from contribution rank |
| **Demoted** | 18-module OpenROAD zoo | Completeness only | Never list as innovations |

**Recommendation for letter Table I:**  
C1*: always-on → **OP-STW** → +ECP → +MW → **+OGEC/PRRC (EXACT)**; optional +TDE3 wake merge.  
C2*: always-on / binary-only → **HBG-RP** → +SMAM → +ADP; TMA-Agg as temporal support.

---

## 3. Compare vs cited cousins — what we must NOT claim

| Cousin | Their knife-edge | We may cite as related | We must **NOT** claim |
|---|---|---|---|
| **Bishop / AAC-style binary** | Binary spike / AAC sparsity | Contrast baseline for HBG-RP | That HBG is “just binary AAC” — payload is real ATLIF *proposal* |
| **SMAM** | Mask-Add attention patterns | Dual-rail inspiration for SMAM-RP | Inventing Mask-Add |
| **FACT** | Eager / forecast attention compute skip | Motivation for ECP-QKV | Being FACT |
| **FlightVGM** | Video / motion-aware sparsity | Motivation for MW-ΔBuf / TMA-Agg | Their end-to-end accuracy |
| **ERAFT** | Event + RAFT-like flow | Event/flow features for OP-STW | Claiming ERAFT algorithm novelty |
| **Loihi / analog TDE** | Neuromorphic TDE silicon | Bio motivation for TDE3-Prior | Loihi / analog CIM silicon claims |
| **ICCV TMA (GPU)** | Temporal motion aggregation accuracy | Motivation for TMA-Agg HW sketch | GPU accuracy / DualRail-CIM arrays |

**Hard negatives (letter hygiene):**
- Do **not** claim silicon µm² / mW; OpenROAD = completeness only.
- Do **not** claim AEE until harness exists.
- Do **not** revive “multiply reorder” as C2* novelty.
- Do **not** conflate HBG int8 proposal with ep35 binary capture.
- Do **not** laundry-list ~18 modules.

---

## 4. Gaps vs silicon TCAS-II

| Gap | Status now | Needed for silicon letter claims |
|---|---|---|
| Liberty (.lib) / tech LEF | Best-effort sky130hd on box | Foundry liberty + LEF |
| OpenROAD / ORFS P&R | Best-effort DRC=0; **NO PDN / NO SPEF** | Not signoff |
| AEE co-sim | Placeholder | ep34 / algo harness ↔ RTL counters |
| TDE3 / TMA depth | Card G directed RTL + TB | System co-sim with OP-STW / MFBD |

---

## 5. Card G remediation (2026-09-06)

1. **TDE3-Prior** + **wake_merge** — digital TDE bank seeds OF wake; OR into OP-STW|Δ.
2. **TMA-Agg** — split Tw → lookup-align → aggregate → early_exit / hyp_hint.
3. **C1 EXACT ablation rung** — PRRC budget caps exact_hit vs MW.
4. Claim draft compressed to **3 knives**; laundry-list modules demoted.

---

## Module → one-line claim (cheat sheet)

| Module | Claim sentence | Letter role |
|---|---|---|
| OP-STW | Optical-flow predictive spike-tile wake schedules exact work. | **Knife 1** |
| TDE3-Prior | Digital TDE-3 time-diff prior seeds OF wake (not zero-skip). | Knife 1 enhancer |
| wake_merge | OR merge: op_stw \| tde_wake \| delta_nz. | Glue |
| HBG-RP | Hybrid binary-gate + real non-absorbable ATLIF **proposal**. | **Knife 2** |
| OGEC×PRRC→exact | Exact enqueue only if match∧budget∧capacity. | **Knife 3** |
| TMA-Agg | Temporal motion agg (split/lookup/agg) under spike/OF schedule. | Support |
| ECP/MW/SMAM/ADP/… | Ablation / roadmap. | Demoted |
