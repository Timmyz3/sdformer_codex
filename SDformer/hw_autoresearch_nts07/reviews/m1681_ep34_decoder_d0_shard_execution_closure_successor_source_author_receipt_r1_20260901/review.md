# M1681 author receipt: decoder D0 shard execution closure

Status: `PASS_M1681_M1672_EXECUTION_AND_REDUCER_P1_REPAIRED_SOURCE__DIFFERENT_AUTHOR_REVIEW_REQUIRED__NO_PAYLOAD_NO_EXECUTION`.

M1681 leaves the exact M1671 8,700-shard grid and scheduler untouched. It repairs only the three M1672 P1 findings: a private payload-to-shard execution target with fixed result/attempt/work/failure namespaces; attempt consumption before any population hashing or payload open; immutable opened-FD/hash timestep snapshots; recursively sealed atomic shard publication; no-retry failure/resume accounting; and a reducer which reads only the complete set of exact sealed receipts.

Every metric row now requires positive cycles and request count, request/kind conservation, nonnegative byte totals, exact commit count/bytes, address digest, common commit digest, destination-state chain and a recomputed final-state digest. The reducer performs integer ratio-of-sums only after all 8,700 receipts verify.

The two existing M1666 `__pycache__/*.pyc` files are explicitly treated as ignored runtime cache, never evidence. Only regular non-symlink pyc files immediately inside `__pycache__` are ignored for that predecessor; every other unsealed member is rejected. M1681 shard results forbid pycache entirely.

CPython 3.6 and the current Python each pass 12 tests and compilation with resource warnings promoted to errors. No canonical payload, replay, reducer, GPU or EDA action occurred. M1682 review and M1683 release remain mandatory.
