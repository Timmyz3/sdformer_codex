# H82 Class-Major TTX operator contract

Status: `FROZEN_OPERATOR_SHA_PENDING_TRAIN`. Date: 2026-08-17.

This is C8.3 plus a C8.1 regularizer. It is not C1–C7.

## Forbidden

- Motion-XOR (H67). Mixing Motion + class-major is C6 identity leak.
- Local5 five-neighbor stencil.
- Prosperity TCAM/Forest, Bishop TTB, FuseMax einsum, FLAT score-kill,
  FusionArch LFT-as-main-innovation, EXION approximate gate reuse.
- Multiplicity-weighted Shiftmax that is algebraically equal to token Shiftmax.
  That rewrite is C7 and scores 3.6, not 4.0.

## New algorithm contract

Token scores are still H81 TX(+SC). The nonlinear is not token Shiftmax.

1. Quantize scores onto the frozen Q7 grid (`step=1/128`, `[-2,2]`).
2. Emit a **Class File** record per occupied code:
   `(class_id, multiplicity, temporal_pair_mask, class_score)`.
3. Shiftmax over **occupied classes only**. Each class casts one vote.
   Multiplicity is not a vote in the partition function.
4. Expand `gate_c` onto tokens, then `attn = K ⊙ gate`.
5. C8.1: spatial score TV so adjacent T450 rows can keep the same members.
   This is what later lets a delta-directory touch the 41% active store.

## New storage / execution object

| Old (H60/H67/H81) | New (H82) |
|---|---|
| 450 token scores → token Shiftmax → 450 gates | Class File → class Shiftmax → expand |
| Directory = 450-entry active list | Directory = occupied classes + multiplicity + mask |
| K addressed per token | K expanded after the class nonlinear |

K-store can stay. The object that changed is the **directory / ISA**, not a
new skip/merge slogan.

## Identity

- Parent: H81 no-motion TTX (`binary_motion_xor_alpha=0`).
- Mode: `h82` / `class_major_ttx`.
- Init: H81 DSEC ep29.
- Seed 0. Do not mix MDR/MVSEC/transfer tables into the H82 identity.
