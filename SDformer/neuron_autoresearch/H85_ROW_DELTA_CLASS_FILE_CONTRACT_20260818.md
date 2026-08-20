# H85 row-delta Class File (algorithm only)

Date: 2026-08-18. Parent H81. No Motion. No Local5. No hardware. No GPU launch
while H82 trains.

H84 review: **3.5**. Packed `member_mask` was still T450. Expand still did `K ⊙ g`.

H85:

1. One packed file **per spatial row** (15 columns, T=2). Member ids are in `0..14`.
2. Adjacent rows store `shared_ids / insert / delete / reuse_set`.
3. Expand scatters `K[row, member_idx] * gate_c` and **raises** if `codes`,
   `token_gate`, or `member_mask` are present. `class_id` selects shared vs insert.
4. C8.1 is the `reuse_set` field (same class set as previous row), also a train loss.

DATE model stays H67. 4.0 is not claimed until review says both gates pass.
