# H84 packed Class File (algorithm only)

Date: 2026-08-18. Parent H81. No Motion. No Local5. No hardware. No GPU launch
while H82 is training.

H83 review: **3.4**. Expand was still `codes.gather`. T0/T1 Jaccard was not C8.1.

H84:

1. Occupied classes are **packed**. The file is
   `(class_id, valid, member_mask, gate_c, multiplicity)`.
2. `_expand_k_from_packed_class_file` **raises** if `codes` is present.
   Token gates come only from `sum_c member_mask[c] * gate_c[c]`.
3. Membership is hard one-hot + STE, so C8.1 can train.
4. C8.1 is adjacent **spatial-row** class-set Jaccard inside the 15×15 window.

4.0 is still not claimed. DATE model stays H67.
