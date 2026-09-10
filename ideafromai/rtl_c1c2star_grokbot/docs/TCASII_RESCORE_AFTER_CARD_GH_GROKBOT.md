# TCAS-II Rescore After Card G/H + OpenROAD (Grok Bot)

**Document type:** Hostile-but-fair **independent rescore** (not inflated)  
**Subject:** Timothee Z — Fork A / C1\*+C2\* HW · tip ≈ `334e8fd`  
**Tree:** `/workspace/sdformer_c1c2star_grokbot/` · **Branch:** `tcasii/c1c2star-oss`  
**Prior:** `docs/TCASII_INDEPENDENT_REVIEW_SCORE_GROKBOT.md` · mean ≈ **3.24** · **Borderline**  
**Date:** 2026-09-06 ~12:50+08 (Asia/Shanghai) · **Author tag:** GROKBOT NEW FILE  
**Hard negatives (unchanged):** OpenROAD = completeness **not** PPA; **no PDN / no SPEF / NOT signoff**; SCI setup WNS **−0.019 ns** = placement-parasitics only — **do not claim timing closed**; int8 HBG = **proposal** vs ep35 **binary** capture; soft knives in idea-18 pack = **research-only (no RTL)**.

---

## 0. What landed since prior review / 相对前评落地了什么

| Item | On-disk fact | Letter-safe reading |
|---|---|---|
| **Card G** | TDE3-Prior + wake_merge + TMA-Agg RTL+TB; regress includes `sim_tde3_prior` / `sim_tma_agg` | Knife-1 **enhancer** (TDE3) + temporal **support fabric** (TMA) — **not** new letter knives |
| **Card H** | CFP-ConfGate, SCI-CleanExit, BiSAT-Agg, BUI-GuardSDSA | Knife-3 **sharpeners** (CFP/SCI) + TMA/HBG **guards** (BiSAT/BUI) — still **3-knife** letter |
| **Regress** | **25/25** sim PASS (`out/REGRESSION_REPORT.md`) | Was 19; +6 Card G/H scripts |
| **OpenROAD** | Old **18** leaf+pipe DRC=0 **+** Card G/H **7/7** DRC=0 = **25/25** | Completeness only; SCI WNS **−0.019** @10 ns **not** closed |
| **C1 EXACT rung** | always→OPSTW→ECP→MW→**EXACT** (`exact_hit` 49→**19** under PRRC budget) | Fills prior gap; strengthens knife-3 evidence |
| **3-knife hygiene** | Claim draft compressed; laundry demoted | Clarity help — **if** authors keep it |
| **Still missing** | AEE co-sim; PDN/SPEF; soft-knife RTL from idea 18 | **Caps** score trajectory |

**Hostile one-liner:** More named RTL ≠ more novelty. Card G/H mostly **deepen / guard** the same three knives; OpenROAD row-count is **engineering completeness**, not a fourth contribution.

---

## 1. Score table / 评分表（1–5）

**Scale:** 1–5（一位小数）。5=该信件轨顶尖；3=可发表边缘；2=大修；1=拒。  
**Calibration rule this pass:** do **not** award novelty for module count; do **not** treat DRC=0 as PPA; do **not** credit SCI timing as closed.

| # | Category | Prior | **Now** | Δ | Rationale (hostile-but-fair) |
|---|---|---:|---:|---:|---|
| 1 | Novelty / originality | 3.2 | **3.35** | +0.15 | TDE3 digital time-diff **prior→wake** is the only clear novelty bump; CFP/SCI/BiSAT/BUI/TMA are policy/consistency/guard enhancers — useful, incremental, near cousins (SciFlow/TMA/BAT-ish). Soft idea-18 knives **not** scored (no RTL). |
| 2 | Technical depth / HW story | 3.0 | **3.15** | +0.15 | TDE3/TMA/SCI have more state than pure threshold glue; many Card H blocks still comparator/ledger depth. ADP/STH/SP/BUI remain thin. Pipe still thin wrappers. |
| 3 | Experimental / verification | 3.5 | **3.75** | +0.25 | **25/25** sim + **25/25** DRC=0 + EXACT ablation rung — strong completeness. Cap: no SPEF/PDN/AEE; SCI WNS negative under placement parasitics. |
| 4 | Clarity vs prior art / ISCAS C1–C3 | 2.8 | **3.15** | +0.35 | **Largest clarity move:** 3-knife claim draft + proposal↔capture discipline + EXACT rung. Residual risk: enhancer naming still invites laundry-list relapse. |
| 5 | Significance / impact | 3.0 | **3.05** | +0.05 | Story spine clearer; **impact still unproven** without workload AEE / system share. |
| 6 | Presentation readiness | 3.2 | **3.40** | +0.20 | Claim draft + Card G/H docs + scoreboard addendum tighter AS-IS; still overclaim-prone if authors quote µ²/WNS as silicon. |
| 7 | Reproducibility (OSS flow) | 4.0 | **4.15** | +0.15 | Scripts/DEFs/CSV for 25 rows; rare OSS letter hygiene. Does **not** buy novelty. |
| | **Mean (unweighted)** | **3.24** | **≈3.43** | **+0.19** | |
| | **Letter-weighted\*** | **≈3.1** | **≈3.30** | **+0.20** | *↑ N/D/E/C；repro 降权* |

\*Letter-weighted ≈ (1.2·N + 1.2·D + 1.2·E + 1.2·C + 1.0·S + 0.8·P + 0.6·R) / 7.2.

---

## 2. Overall recommendation / 总评

**Borderline → Weak-Accept-leaning Borderline**（仍 **非** Weak Accept）。

**一句话理由：** Card G/H + 25× OpenROAD + EXACT 消融抬高了**实验完整度与声称卫生**，但**新颖度仅小幅上移**（TDE3），且 **AEE 缺失 + SCI 负松弛不可宣称闭合 + int8/二值分叉** 仍卡在 TCAS-II Express Briefs 的“可测电路结果”门槛之下。

| Lens | Reading |
|---|---|
| If venue wants **silicon-ish PPA / closed STA** | Still closer to **Weak Reject–Borderline** |
| If venue accepts **reproducible RTL + ablation co-design letter** with honest proposal language | **Borderline–Weak Accept** edge — needs AEE to cross |
| vs prior ISCAS ~WA old C1/C2 | Engineering + knife hygiene **upgraded**; **do not** auto-translate to TCAS-II WA |

---

## 3. Delta vs prior ~3.24 Borderline — what moved the needle / 针动了什么

| Moved ↑ | Did **not** move (or moved little) |
|---|---|
| **Clarity (+0.35):** 3-knife spine written down; EXACT rung closes prior ablation hole | **Significance (~flat):** no AEE / no system % |
| **Experimental (+0.25):** 19→25 sim; 18→25 DRC=0; Card G/H prove-by counters | **Novelty (only +0.15):** more modules ≠ new pillars; soft knives unpaid |
| **Presentation (+0.20):** claim/Card docs reduce laundry risk *if obeyed* | **Depth (+0.15 only):** still many thin gates |
| **Repro (+0.15):** OSS scoreboard completeness | OpenROAD util/µ² **must not** be sold as silicon |

**What must *not* be claimed as needle-movers:** “7 new innovations”; “timing closed” (SCI −0.019); “PPA ready”; “int8 already deployed”.

---

## 4. What still caps the score (top 3) / 仍封顶的三项

1. **No AEE / algo–RTL co-sim Pareto** — especially int8 non-absorbable HBG **proposal** vs ep35 **binary** capture. Without this, Knife-2 is a contract story, not a circuits result.  
2. **Shallow circuit depth + zoo relapse risk** — TDE3 helps Knife-1; CFP/SCI/BiSAT/BUI remain policy/guard fabric. Soft idea-18 knives stay research-only. Laundry-list in Abstract would erase clarity gains.  
3. **No signoff physical story** — no PDN, no SPEF/RCX, single tt corner; SCI setup WNS **−0.019 ns** (placement parasitics) **cannot** be claimed closed; OpenROAD = completeness only.

---

## 5. Chance estimate / 投稿概率（主观）

| Package | Est. accept-class chance | Note |
|---|---|---|
| **AS-IS**（25 RTL + 3-knife hygiene + OpenROAD completeness; **no AEE**) | **~22–32%** Weak Accept；其余 Borderline / WR | Was ~15–25%. Overclaim / depth / missing AEE still dominate rejects. |
| **After AEE + glue**（int8 vs binary Pareto + 1 closed front↔back pipe metric + keep strict 3-knife; soft knives stay out or appendix） | **~55–70%** Weak Accept | Was ~55–65%. Upper bound still limited by **no silicon signoff**. |
| Strong Accept trajectory | **Unlikely AS-IS / even post-AEE** | Needs closed STA+RCX or taped-out numbers beyond this OSS sky130hd path. |

---

## 6. One-page CN executive（可直接贴聊天） / 一页中文执行摘要

**对象：** Fork A C1\*+C2\*，tip≈`334e8fd`；相对前评 mean≈3.24 Borderline 的 **Card G/H + OpenROAD 复评**。

**落地事实（别夸大）：**
- Card G：TDE3-Prior / wake_merge / TMA-Agg；Card H：CFP / SCI / BiSAT / BUI —— **增强三刀**，不是新开四刀动物园。  
- 回归 **25/25 PASS**；OpenROAD 旧 18 + G/H 7 = **25/25 DRC=0**；SCI setup WNS **−0.019 ns**（仅 placement parasitics）—— **不可声称时序闭合**；无 PDN/SPEF。  
- C1 消融已有 **EXACT** 档（exact_hit 49→19）。三刀卫生与 proposal↔capture 分句已写进 claim draft。  
- idea-18 soft knives **仍无 RTL**；仍无 AEE；int8 HBG = **提案**，ep35 = **二值捕获**。

**新分（1–5）：** 新颖 **3.35** · 深度 **3.15** · 实验 **3.75** · 清晰度 **3.15** · 影响力 **3.05** · 呈现 **3.40** · 可复现 **4.15** → **均分 ≈3.43**（+0.19）· **信件加权 ≈3.30**（+0.20）。

**推荐：** **Borderline（偏 Weak Accept 边缘，仍非 WA）** —— 完整度与声称卫生抬升了针，但 AEE 缺失 + 电路深度有限 + 非签核物理，挡在 TCAS-II 稳 WA 之前。

**概率：** AS-IS ~**22–32%** WA；补 AEE+glue 后 ~**55–70%** WA；Strong Accept 不现实。

**针动了什么：** 清晰度与实验完整度（三刀 + EXACT + 25×P&R）；新颖仅因 TDE3 **小幅**上移。  
**别把针动算成：** 模块变多、DRC=0、SCI 负松弛“差不多过了”。

**仍封顶 Top3：** (1) 无 AEE/Pareto；(2) 深度浅 + 动物园复辟风险；(3) 无 PDN/SPEF，SCI 时序未闭。

**给作者一句话：**  
工程与 OSS 证据链已经够“像一封可复现的硬件信”；**下一刀必须是 AEE，不是再开一张 Card。** 忍住模块清单，守住三刀与 int8/二值分句——否则复评分会原路退回。

---

## Appendix — Evidence anchors

| Item | Path |
|---|---|
| Prior independent review | `docs/TCASII_INDEPENDENT_REVIEW_SCORE_GROKBOT.md` |
| Claim draft (3-knife) | `docs/TCASII_LETTER_CLAIM_DRAFT_GROKBOT.md` |
| Card G / H docs | `docs/CARD_G_TDE3_TMA_GROKBOT.md`, `CARD_H_CFP_SCI_GROKBOT.md`, `CARD_H_BISAT_BUI_GROKBOT.md` |
| OpenROAD scoreboard | `docs/OPENROAD_PNR_SCOREBOARD_GROKBOT.md` (25/25; SCI WNS −0.019) |
| Regress | `out/REGRESSION_REPORT.md` (25/25 PASS) |
| C1/C2 ablations | `out/ABLATION_C1_LADDER.md` (incl. EXACT), `out/ABLATION_C2_LADDER.md` |
| Tip | `334e8fd` (`tcasii: Card G/H liberty-map + OpenROAD P&R 7/7 DRC=0 + STA IO delays (NOT signoff)`) |

---

*GROKBOT NEW FILE — independent rescore for parent / Timothee Z. Local tree only; no nts07; no push.*
