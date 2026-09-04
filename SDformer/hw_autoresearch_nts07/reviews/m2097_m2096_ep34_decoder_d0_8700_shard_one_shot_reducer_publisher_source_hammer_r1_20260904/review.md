# M2097 independent source hammer: FAIL CLOSED

M2096 does **not** authorize M2098. Score: **84/100**, with **P0/P1/P2 = 0/2/1**.

The reducer publisher has a sound transaction core: the outer `O_EXCL` attempt precedes all M2090/M2093 result and shard reads, the exact M1704→M1688 reducer enumerates 8,700 ordinals, ratios remain integer ratio-of-sums, and both success and in-transaction failure use sealed work trees plus `RENAME_NOREPLACE`. CPython 3.6 and 3.12 compile/describe/preflight agree, the source contract is double sealed, and docs/359 remains unchanged.

Two authority gaps nevertheless block release:

1. M2096 accepts raw M2090 and M2093 results that explicitly still say `independent_result_hammer_pending=true`. Its runtime gate and future release identity do not bind either predecessor result-hammer. A synthetic otherwise-valid M2097/M2098 pair with neither hammer was accepted.
2. M1688 lets each shard receipt self-select its `release_sha256`, and M2096 does not pin all 8,700 rows to the frozen M1706 release. Thus exact source/checkpoint/resource hashes alone do not establish exact campaign authority.

The minimum successor must bind the sealed M2093 result hammer and the future sealed M2090 result hammer before execution, and require every reduced shard row to carry the exact M1706 release identity. It should also close or explicitly contract the attempt-only failure window around `WORK.mkdir()`.

No production result, shard receipt, payload, reducer, shard, GPU, or EDA path was opened or executed. M2096 source/contract and all prior evidence were left untouched. Only successor-source authoring is authorized; M2098 release authoring and reducer execution are denied.
