# Cross-AI read — Grok46 / Codex independent vs Grok Bot (iscas_ssh)

**Date:** 2026-09-05 · **Source of truth on ismd:** `/home/zhumd/work/ideafromai/`  
**Mirror note:** Grok46 + Codex independent packs pulled to box for this memo.

---

## 1. Three packs on ismd `ideafromai`

| Pack | Path | Core story |
|---|---|---|
| **Grok Bot (us)** | `research/00–14`, Cards A/B, `sdformer_c1c2star_grokbot` | C1*/C2* remake: OP-STW, **HBG-RP int8 ATLIF payload**, ECP/MW/SMAM/STH… |
| **Grok 4.6** | `research/grok46_20260905/` + Card C | **Motion-XOR MX3P** + dirty-lane score memo + **mixed-T binary ATLIF**; C1/C2 demoted |
| **Codex independent** | `codex_independent_20260905/` | C1/C2 novelty weak; hypothesis = selective shared partial-sum by destination mask (Mailman-adjacent); **not validated** |

Canonical README on ismd says: read **both** Grok packs; follow **frozen-ep34 identity**, not the louder claim.

---

## 2. Hard identity conflict (must resolve before more RTL)

| Question | Grok 4.6 (frozen ep34) | Grok Bot (us) |
|---|---|---|
| ATLIF capture | **93/93 binary** fire | **int8 non-absorbable payload** (new co-design default) |
| Card B HBG-RP | **Do not implement as frozen** | Already implemented + Yosys + regress |
| Card C MX3P | Draft; needs T0–T5 stats first | Not started |
| C1/C2 as headline | **Kill** as novelty (Prosperity / TSBG priors) | Remake as OP-STW / HBG narrative |

**Grok46 explicit:** “Do not implement HBG-RP as if it were the deployed contract.”  
**Card C:** “Conflicts with Card B… Do not run A+B+C as one island.”

---

## 3. What Grok46 says is actually new (ranked)

1. **MX3P** — triple popcount: `pop(Q∧K)`, `pop(K⊕K_peer)`, co-silence → Q7 score (α=0.125 frozen)  
2. **Dirty-lane / temporal-peer gate** (same island as MX3P) — not empty-tile  
3. **Mixed-horizon binary ATLIF** — T=10 membrane private, 1-bit public; T=2 attention  
4–8. LoAS-style fiber join, Bishop bound-without-S, K-zero gating, dirty-run scheduler, inter-frame dirty tile  

**Kill list (agree with our earlier “don’t claim”):** Prosperity-as-ours, TSBG-as-ours, empty-tile, Shiftmax 2^k, ASNA spatial-locality-as-ours, analog CIM, component-ratio product speedups.

**System caveat (Workflow B):** attention may be ~0.59% of cycle envelope → MX3P may be **leaf paper** only unless T0 shows otherwise on ep34.

---

## 4. What Codex independent says

- C1 ≠ Prosperity novelty without answering “what non-obvious problem under finite ports”  
- C2 ≠ new math for broadcast / loop interchange  
- Preferred **hypothesis**: selective shared reduction by destination mask (changes arithmetic graph) — related to Mailman; **cannot claim new algebra**; needs workload opportunity stats + PPA  
- Status: `RESEARCH_HYPOTHESIS_ONLY_NOT_VALIDATED`

---

## 5. Implication for our box HW (`sdformer_c1c2star_grokbot`)

| Keep as | Modules |
|---|---|
| **Valid under Grok Bot co-design branch** (int8 ATLIF *new* contract + AEE Pareto required later) | OP-STW, HBG-RP, ECP, MW, OGEC, PRRC, SMAM, ADP, STH, ARM, MFBD, SP, pipes, stats, liberty-mapped area, C1 ablation |
| **Not aligned with frozen-ep34 binary path** | Treating HBG int8 as “already true of ep34” |
| **Missing vs Grok46 innovation knife** | MX3P / K_peer / dirty-lane score island (Card C) |
| **Missing vs both** | ep34 T0–T5 census before claiming system speedup |

**Honest TCAS-II fork (user must pick):**

**Fork A — Co-design letter (our current RTL):** claim *new* ATLIF payload contract + OF-scheduled wake (OP-STW/ECP/MW) + dual-rail HBG/SMAM. Must open AEE Pareto vs binary baseline; cite Grok46 kill list so we don’t claim Prosperity/TSBG.

**Fork B — Frozen-ep34 letter (Grok46):** demote C1*/C2* RTL to context; run T0–T5; then Card C MX3P(+dirty). Our int8 HBG becomes contrast baseline, not headline.

**Fork C — Hybrid (hard):** binary MX3P leaf **and** optional int8 payload path as ablation — two islands, two claims, never conflate.

---

## 6. Recommended next actions (innovation-first)

1. **User lock fork A/B/C.**  
2. If A: strengthen letter draft with “new co-design vs frozen binary” paragraph; keep OSS/OpenROAD; sd5 A800 for AEE.  
3. If B: pause more C1*C2* modules; implement MX3P leaf on box under new dir `rtl_mx3p/`; pull ep34 dirty stats on ismd/sd5.  
4. If C: MX3P first (stats-gated), keep C1*C2* as separate appendix.  
5. Do **not** merge Card A+B+C into one island without Pareto.

---

*Written by Grok Bot (iscas_ssh) after reading ismd ideafromai packs. Local only until user syncs.*
