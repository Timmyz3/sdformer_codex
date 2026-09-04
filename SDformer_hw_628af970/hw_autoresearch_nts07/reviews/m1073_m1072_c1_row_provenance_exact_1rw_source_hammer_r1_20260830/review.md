# M1073 independent receipt-blind hammer of M1072

Verdict: **PASS / GO to author a separately sealed M1074 full-replay release source; do not launch.** Score **100/100**, P0/P1/P2 = **0/0/0**.

This review did not consume the M1072 author receipt, did not modify M1072, did not advance the production generator, and did not run the 51.84M-row replay, EDA, GPU or remote work.

## What closed

- The sole production boundary is the zero-argument generator `iter_canonical_full_replay_results()`. Record, sample, work, preprocess, capacity and coverage arguments reject.
- The canonical reader is fixed to the regular nonsymlink 466,560,000-byte M410 file at SHA256 `6e03352b...`; it uses no-follow plus internally derived `pread`, and checks initial/final fd identity and SHA.
- Independent task-0 reconstruction through M1016 reproduced shared preprocess 210; candidate work 1,664; zero/bit work 4,392; and exact parent summaries.
- The manual M1065 forgery (candidate 0, baselines 999,999, preprocess 0), all-zero masks, row reorder, wrong offset, digest mutation, short read and file path/size/SHA/stat drift all reject.
- Coverage order is internal. Its execution digest consumes record provenance, which binds task/order/coordinate/offset/exact row and mask digests/preprocess/work/parent/common receipt. Dynamic work/preprocess and row-digest mutations changed that digest.
- Empty, partial, duplicate and out-of-order populations plus boolean/extra/duplicate-key schema attacks reject.
- An independent service-only M1016 traversal reproduced 812,160 tasks, ten commits and service digest `a38589ba...`.
- The 214,912-byte arithmetic and the four-group, one-1RW, RAW-aware 20-to-22-cycle cascade remain intact. Capacity remains **capacity-only**, not admitted performance.

## Narrow authorization

M1074 may now author a new one-shot CPU-only full-replay release source that pins the M1072 and M1073 double seals. It may not launch until a different-author release hammer passes. Automatic retry remains forbidden.

This receipt does not admit full-replay completion, matched cycles, speedup, RTL cycles, PPA, energy, or paper readiness. `docs/359` remains `dedde7ce...`.
