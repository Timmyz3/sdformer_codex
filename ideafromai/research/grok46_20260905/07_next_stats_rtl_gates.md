# 07 — Next gates (stats before RTL)

Canonical folder: `/home/zhumd/work/ideafromai/`.  
Do not open a new production island until T0–T5 exist on **frozen ep34**, same ledger as C1/C2 (not ep35 P0).

## T0 — Attention / conv / FC cycle share (Workflow B gate)

**Question:** If MX3P were free, what is the upper bound on end-to-end cycles?

**Output:** table of cycles or MAC-proxy by {bottleneck conv, encoder linear, attention score leaf, decoder, ATLIF T=10}. Label model vs VCS.

**Fail closed:** if attention leaf < ~2% of sealed envelope, Rank 1 is a **leaf paper** only. Do not write “system 1.5×”.

## T1 — Lane vs row dirty rate (Rank 4 life/death)

For each attention row (token×head or Shiftmax row, match the RTL row that owns the denominator):

- `lane_dirty = OR(Q_t ⊕ Q_{t-1}, K_t ⊕ K_{t-1})` per lane
- `row_dirty = OR(lane_dirty)`
- fraction of rows with row_dirty=0 (full score memo legal)
- fraction of lanes with lane_dirty=0 inside dirty rows (partial)

**Fail closed:** if row_dirty ≈ 100%, do not sell 97% paired-score equality as cycle skip. Keep MX3P as the ALU (Rank 1) and use dirty only as energy gate / ablation.

## T2 — Term-wise invariance

When lane_dirty=0, verify overlap, motion_xor, same_zero, and Q7 score are bit-identical to t0. When `motion_xor==0` but Q toggled, which terms still skip.

## T3 — K-zero ∩ dirty (Rank 6)

Joint histogram: (K_t==0, K_{t-1}==0, dirty). Count V-gather skips that are legal under `attn=gate⊙K`.

## T4 — Dirty run lengths (Rank 7)

Histogram of contiguous dirty token runs at 1/2/4/8 packing. Needed for scheduler, not for the paper’s first figure.

## T5 — Island-boundary bitwidth (Rank 3 premise)

For every live ATLIF output consumed by conv/FC/attention: is the captured tensor 1-bit? If 100% binary, post-neuron MAC → add/sub is a **fact**, not a slogan. If any analog leak remains, list the wrappers (do not assume 93/93 without a script on ep34).

## After T0–T5: RTL policy

| Outcome | Allowed RTL | Forbidden |
|---|---|---|
| T0 attention share high **and** T1 row_dirty low | MX3P + dirty memo island (Card in `08`) | Rebuilding C1 forest as novelty |
| T0 attention share tiny, T1 still interesting | MX3P as **leaf** with energy/area vs AND-PopCount baseline; no system × | Abstract system FPS |
| T1 row_dirty high | MX3P ALU + K_peer port only; dirty as energy | Claiming 97% skip |
| T5 not 100% binary | Stop Rank 3 add-only claim | Grok Bot int8 HBG-RP on “frozen” ep34 |

## Same-resource rule (existing project gate)

Any new island vs ordinary/AND-PopCount/C2-ordinary must be equal lanes/banks/SRAM class or labeled as a model. Do not compare a 2R1W score SRAM against 1RW C1.

## What Codex must not start from this pack alone

- New C1* OP-STW / PRRC / OGEC RTL from `/home/zhumd/work/ideafromai/`
- New C2* HBG-RP int8 payload RTL
- Correlation volume, RAFT residual, decoder ConvTranspose engine
- Weight pruning / FireFly-S
- Analog CIM
- `codex exec resume` of session `01a01043-...`
- Edits to `docs/359`, paper TeX, or TCAS-II/ISCAS drafts unless the user later asks
