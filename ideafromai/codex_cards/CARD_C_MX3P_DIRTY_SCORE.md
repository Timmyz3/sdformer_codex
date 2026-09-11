# Card C — MX3P + dirty-lane Motion-XOR score island (Grok 4.6)

**Path:** `/home/zhumd/work/sdformer_codex/ideafromai/codex_cards/CARD_C_MX3P_DIRTY_SCORE.md`  
**Status:** DRAFT. Do not execute until T0–T5 in `research/grok46_20260905/07_next_stats_rtl_gates.md` exist **and** the user picks Rank 1+4.

Conflicts with Card B: Card B assumes int8 ATLIF payload. This card assumes **binary** frozen ep34 capture. Do not run A+B+C as one island.

## Goal

A new **attention score island** for frozen Motion C12 ep34:

- Inputs: binary Q, K, K_peer (T=2), 1RW foundry SRAM class.
- Compute: `overlap = pop(Q∧K)`, `motion = pop(K⊕K_peer)`, `silence = co-silence`, Q7 `round_even` as frozen formula, α=0.125.
- Gate: skip leaf when dirty=0 **if T1 proves row-level legality**; else compute always, clock-gate only.
- Output: Q7 scores + `attn = gate ⊙ K` with K-zero gather skip.
- Baseline: ordinary AND-PopCount QK (FireFly-T-class) on the **same** Q/K tiles, same ports.
- Not in scope: C1 forest, TSBG rename, empty-tile skip, analog CIM, int8 ATLIF payload.

## Identity contracts

- Bit-exact vs `h67_motionxor_score_q7` / software Q7 on sealed ep34 descriptors.
- Signed protocol (if present) is polarity, not analog ATLIF.
- Shiftmax row denominator: if any lane dirty, define whether the **row** recomputes. Document in the TB.
- Lossless: dirty=1 ⇒ recompute. No ε-threshold unless a separate AEE Pareto is opened.

## Related-work sentences the RTL comments should not violate

- “AND-PopCount is the QK term prior (FireFly-T); we add temporal-peer XOR and co-silence mapped from α-XNOR with frozen α=0.125.”
- “XOR here is a score term, not Comperity GeMM differential encoding, not C1 conv source-mask residual.”
- “Skip is temporal score memo / K-zero gating, not empty-tile.”

## Deliverables if admitted

1. Cycle-accurate TB vs ep34 score descriptors (start small: one head, one window).
2. DC + PT + Formality on the leaf (pre-macro OK, label it).
3. Ablation: full MX3P vs AND-only vs dirty-gated vs K-zero-gated. **No product of ratios.**
4. Do not put 1.6945× or TSBG 1.83× in the new abstract.

## Out of card (Rank 3, later)

Membrane-private T=10 ATLIF firewall. Keep C3 exact coverage. Do not mux-unroll away Vmem unless T5 + a T=10 unroll study shows bit-exact ATLIF.

## Isolation

New RTL in a new directory. Do **not** patch `rtl_m935` C1 or `rtl_m2018` C2 production as if they became MX3P. Do not touch `docs/359`.
