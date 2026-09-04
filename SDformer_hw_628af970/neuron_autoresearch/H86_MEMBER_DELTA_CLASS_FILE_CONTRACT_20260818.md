# H86 member-delta Class File (algorithm only)

Date: 2026-08-18. Parent H81 / H82 vote rule. No Motion. No Local5. No hardware.
No GPU launch while H82 trains.

H85 review: **3.6**. Class-name `shared|insert` was tautological. 513-bin one-hot
was the directory. `reuse_set` was a loss, not an expand operand.

H86:

1. Shiftmax stays **window class-major** (H82). One vote per occupied Q7 class.
2. Row 0 stores packed column ids `0..14`. Rows after 0 store only
   `member_insert` / `member_delete` (column ids) plus packed
   `class_shared` / `class_insert` / `class_delete`.
3. Expand reconstructs `members = prev + insert − delete`, then muxes
   shared / insert / delete. It **raises** on `codes`, `token_gate`,
   `member_mask`, `member_idx`, 513-bin occupancy, or H85 class-name sets.
4. C8.1 is `1 − member_jaccard(surviving classes)`, not class-name reuse.

Review 2026-08-18: **4.0**. `CLAIM_4_0=YES` on the operator/object pair only.

DATE model stays H67. No H86 train while H82 owns the A800. No checkpoint SHA yet.
Do not call padded-15 a measured 41% CSR saving. No C8.2. No RTL from this agent.
