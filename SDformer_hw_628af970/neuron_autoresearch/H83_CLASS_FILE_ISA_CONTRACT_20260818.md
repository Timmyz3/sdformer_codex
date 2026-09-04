# H83 Class File ISA (algorithm only)

Date: 2026-08-18. Parent: H81. No Motion. No Local5. No hardware in this contract.

H82 review scored **3.3**: class-major partition is real, but the Class File was discarded
and `attn = K ⊙ token_gate` stayed the execution object.

H83 is the algorithm fix:

1. Occupied Class File is **emitted** (`class_id`, `occupied`, `multiplicity`,
   `class_score`, `gate_c`, `temporal_pair_mask`, T0/T1 member Jaccard).
2. Shiftmax runs on occupied classes only. Multiplicity is not a vote.
3. K is applied by `_expand_k_from_class_file`. Destinations keep their own K
   rows. This is not class-wise K folding.
4. No `preserve_mean × 450`. Class-domain gates stay in Shiftmax's natural scale.
5. C8.1 on H83 is **1 − T0/T1 member Jaccard**, not spatial score TV.

H82 training is not interrupted. H83 is the next operator SHA. It does not
launch until the H82 GPU job finishes. 4.0 is still not claimed.
