# TCAS-II design plan (Grok Bot) — C1*/C2* letter track

**Venue:** IEEE Transactions on Circuits and Systems II: Express Briefs (letter, ~4–5 pages).  
**Tree:** `sdformer_c1c2star_grokbot` only. Do **not** touch `hw_autoresearch_nts07`.  
**Date:** 2026-09-05 (Asia/Shanghai).  
**Idea packs:** `/workspace/ideafromai/research/` esp. **09**, **12**, **14** synthesis (+ R1 `04`).

## Contribution story (letter spine)

**Core (claim now, RTL exists):**
1. **OP-STW** — optical-flow predictive spike-tile wake (Card A). Exact-product / PE wake from flow-Δ + event count, not flat tile quotas.
2. **HBG-RP** — hybrid binary-gate + real ATLIF payload (Card B). Gate clocks/mem; payload MAC only when `|amp| > EPS` (payloads not absorbable into W — ATLIF r1).

**Roadmap (cite as future / ablation layers — do NOT claim first):**
| Block | Role | Pack |
|---|---|---|
| **ECP-QKV** | Eager correlation prediction **before** QKV projection; skip useless proj MAC (FACT-style) | 09 / 07 |
| **MW-ΔBuf** | Motion-warped residual buffer; spike only Δ | 09 / 08 |
| **SMAM-RP** | Mask-Add on gate × real ATLIF payload when mask=1 | 09 / SMAM |
| **DualRail / DualRail-CIM** | Gate on spike-CIM, payload DigiCIM fabric | 12 / 14 |
| MW-CIM-TileGate, Motion-TTB, STH-Gate, EV-Wake | later cards | 12 / 14 |

Safe claim sentence (from 09): PE wake scheduled by **eager motion-correlation (ECP-QKV) + motion-warped residual (MW-ΔBuf) + event/occlusion gate**, not zero-skip alone; C2* is **dual-rail spike-gate + non-absorbable ATLIF payload (HBG-RP/SMAM-RP)**.

## Ablation ladder (letter Table I)

**C1*:** always-on → zero-skip → **OP-STW** → OP-STW+ECP-QKV → +MW-ΔBuf → +OGEC/PRRC (full).  
**C2*:** always-on / binary-only (Bishop-style AAC narrative) → **HBG-RP** → HBG-RP+**SMAM-RP** → +ADP-MAC → +SP-Gate/STH.  
Vs baselines: always-on, zero-skip, binary-only — report Δ wake %, Δ SOP proxy, Δ Yosys cells.

## Metrics

| Metric | Now | Later |
|---|---|---|
| Yosys generic cell / wire-bit counts | OSS flow `out/synth/` | — |
| Activity×gate **power proxy** | `power_proxy.py` (NOT silicon) | liberty + OpenSTA |
| Directed TB PASS | Card A/B (+C) | — |
| PE wake % | TB counters / co-sim | ep34 traces |
| SOP proxy | gated MAC enables | full SDSA |
| **AEE** | placeholder | algo co-sim (do not claim Motion-XOR as AEE novelty) |

## Open-source toolchain vs Synopsys

See `docs/OSS_EDA_TOOLCHAIN_GROKBOT.md` and `flows/oss/README_GROKBOT.md`.

## In-scope vs placeholder (first silicon-accurate claim)

| In-scope for letter draft v0 | Placeholder / not claimable yet |
|---|---|
| Functional correctness of OP-STW, HBG-RP, ECP-QKV directed tests | Foundry area (µm²), STA slack |
| Relative Yosys cell counts (generic) | Silicon mW / PrimePower |
| Ablation story + roadmap naming | AEE numbers without co-sim |
| Toolchain reproducibility (OSS) | OpenROAD P&R, DualRail-CIM tapeout |

**First silicon-accurate claim** requires liberty + OpenSTA (or DC/PT) and preferably OpenROAD; until then all power/area are **proxies**.
