# H85 algorithm review (docs/433)

Date: 2026-08-18. Score: **3.6 / 5. Not 4.0.** Frozen DATE model stays **H67**. H82 keeps the GPU.

docs/433 §4: 4.0 needs **both** a new algorithm-operator contract **and** a new hardware storage/execution object. H85 clears the H84 T450-key ban and changes the Shiftmax *domain*. It does not change the executed object.

## Checklist

| Gate | Verdict |
|---|---|
| Per-row packed file? | **Yes, batched.** `_build_h85_row_files` cuts the 2×15×15 window and packs each spatial row. Axes are `[B,H,T=2,row=15,max_class=15,…]`. Not 15 Python files; one dict with a row axis. Acceptable as packing. |
| Expand raises on T450 tensors? | **Yes, on keys.** `_expand_k_from_h85_row_files` raises if `codes`, `token_gate`, or `member_mask` are present. Returns only `attn`. Tests cover the raise and the non-return of a 450-gate. |
| `class_id` used for shared/insert? | **Gathered, not a selector.** For row>0, `class_id` indexes `shared_ids` / `insert_ids`. For any *live* class on the current row, `(shared\|insert) ≡ curr_set ≡ live`. `apply = live * (shared\|insert)` is `apply = live`. `delete_ids` is never read. |
| `member_idx` last dim 15 not 450? | **Yes.** `member_idx` is `[B,H,2,15,15,15]`. Last dim is the row’s 15 columns, ids in `0..14`. |
| `reuse_set` / delta executed or only stored? | **Stored + loss. Not executed.** `shared_ids` / `insert_ids` / `delete_ids` / `reuse_set` sit in the file. `regularize_h85_delta` maximizes `reuse_set.mean()`. Expand does not read `reuse_set` or `delete_ids`. The shared/insert gather is a tautology. |

Config path `configs/generated/dsec_fullres_w15_H85_row_delta_class_file_ft15.yml` was missing at review time. Completeness ding, not a 4.0 issue.

## Why not 4.0

Same two gates as H82.

**Gate 1 — new operator contract: weak / mixed.** Per-row Shiftmax over ≤15 classes is not H82/H84 window class-major. Then expand throws the fork away as an ISA: exclusive hard membership plus `attn[row, col] += K[row, col] * gate_c[class]` is dest-owned `K_i * gate_c(class(i))`. Banishing the `member_mask` **key** does not ban the T450 **object**.

**Gate 2 — new storage/execution object: fail.** Adjacent-row payload is a **class-name** set built as Q7 **513-bin one-hot**. Class-set Jaccard can be 1.0 while surviving-class member Jaccard is `< 0.5`. The 41% store is the **member CSR**. H85 regularizes the cheap column. `reuse_set` is H84 Jaccard with a new name.

## What H85 did fix (why 3.6, not 3.5)

- H84 `member_mask` was `[packed, 450]`. H85 last-dim is 15.
- Expand no longer returns or allocates a 450 token gate.
- The raise on `{codes, token_gate, member_mask}` is real.
- Delta names exist as directory fields, not only a sidecar `row_jaccard`.
- Row Shiftmax is a different partition than H82.

That is a 0.1 step (3.3 → 3.4 → 3.5 → **3.6**). It is not a new DATE object.

## Must not claim

No 4.0. No DATE model swap. No H82 AEE/SHA transfer. No 41% directory ungated. No C8.2. No Motion / Local5 mix. No H81 G0 percentages. Do not start this graph while H82 owns the GPU.

## Next algorithm step

H86: keep H82 window class-major. Per shared class, pack `member_insert` / `member_delete` as column ids in `0..14`. Expand reconstructs `prev + insert − delete`. Ban 513-bin one-hot as an expand operand. C8.1 is surviving-class member Jaccard, not class-name `reuse_set`.
