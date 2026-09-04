# H86 algorithm review (docs/433)

Date: 2026-08-18. Score: **4.0 / 5.** Frozen DATE model stays **H67**. H82 keeps the GPU.

docs/433 §4: 4.0 needs **both** a new algorithm-operator contract **and** a changed storage/execution object. H86 is H82 window class-major (not token Shiftmax, not C7) plus a member-delta file that expand cannot drop.

## Checklist

| Gate | Verdict |
|---|---|
| Executed object = member-delta CSR? | **Yes, as the membership operand.** File has `row0_member_idx` only. No `member_idx`. `member_insert` / `member_delete` are column ids `0..14`. Expand walks `members = prev + insert − delete`. `test_insert_is_executed` mutates `member_insert` and attn moves. After the walk the MAC is still dest-owned `K_i * gate_c(class(i))`. That is the H82 *apply*, not a hidden T450 `token_gate`. |
| 513-bin one-hot as expand operand? | **No.** `class_id` is packed length `P`. File last dims are `P` or `15`. Expand raises on last-dim `513` and on `{codes, token_gate, member_mask, member_idx, occupied, shared_ids, insert_ids, delete_ids, reuse_set, n_bins}`. |
| C8.1 surviving-class **member** Jaccard + STE? | **Yes.** `member_jaccard_surviving` is column inter/union of STE membership, masked by hard `class_shared`. Not class-name `reuse_set`. Loss is `1 − Jaccard`. |
| H82 window class-major kept? | **Yes.** One Shiftmax over occupied Q7 classes. Not H85’s per-row ≤15 Shiftmax. Motion-XOR raises. |
| `apply = shared\|insert` ≡ `live`? | **Class mux is paint.** `rebuilt = prev + insert − delete` is already `curr`. Delete is used in `rebuilt`. This is not H85’s `live * (shared\|insert)` on a stored full table. |
| Expand == H82 token expand? | **Yes, unit test** on a constant-across-rows pattern. Do not fail Gate 1 for the numbers. This is not token Shiftmax / C7. |
| Completeness | Config exists. Regularizer wired. No H86 result dir. H82 is the live job. |

## Why 4.0

**Gate 1 — operator contract: pass.** Occupied-Q7 class Shiftmax is the C8.3 partition. H86 does not fork it and does not go back to token Shiftmax.

**Gate 2 — storage/execution object: pass.** H82 threw the file away (3.3). H83 gathered `codes` (3.4). H84 `member_mask` was the codes transpose (3.5). H85 banned T450 keys and still expanded from a full `member_idx` table (3.6). H86 drops that table. If you remove `member_insert` / `member_delete`, row>0 has no members.

Ugly, not a fail: last-dim is padded `15`, so the PyTorch tensor is a batched CSR, not a 41% proof. No delete-mutation test. No measured CSR saving.

## Must not claim

No DATE model swap. No H82 AEE/SHA transfer. No 41% directory ungated. No C8.2. No Motion / Local5 mix. No H81 G0 percentages. No production RTL. Do not start this graph while H82 owns the A800. Do not call padded-15 a measured CSR saving.

`CLAIM_4_0=YES`. Still missing: frozen H86 **checkpoint** SHA (none), any H86 train, DSEC/MVSEC AEE, rank-1 surviving-class member Jaccard. DATE identity stays **H67**.
