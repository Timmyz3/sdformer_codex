# M1517 independent hammer of M1516

Verdict: **FAIL CLOSED; M1518 release is blocked.**

The mechanism itself is mostly sound. Exact M1458/M1510/M1512/M1513
identities close, all 12 author tests pass, and independent attacks confirm
the positive/negative split, extent, tail padding, all-zero negative plane,
SHA checks, path traversal and symlink rejection, O_EXCL/no-retry attempt,
exclusive plane creation, renameat2 no-replace, 122-member population, inert
CLI, and the requirement that `execute_once` call the M1517 gate before any
materialization.

One P0 remains. `verify_materialized_seal` proves that a directory is
self-consistent, but not that its `manifest.json` still expresses the frozen
M1510/M1516 semantics. Two independent executable forgeries were accepted:

1. A pre-seal manifest changed D0's scale to ONE, changed its encoding to
   `exact_binary`, enabled weight folding/normalization/coercion, duplicated a
   capture global order, and set `cycles=true`. `seal_staging` accepted it.
2. The canonical first output path was changed to
   `payloads/renamed_attack.bin`, with the record, file and seal made mutually
   consistent. `seal_staging` accepted it.

This is not a weakness in SHA-256 or the 122-member count. The missing trust
edge is an independently regenerated expected manifest. A successor must
regenerate the exact 120 rows from sealed M1458/M1510 authority and compare
all semantic fields, canonical paths and payload SHAs during both staging seal
and post-publication verification. M1516 must remain immutable.

Independent result: 22/24 checks passed; the two failed checks are the two P0
reproductions above. No production action, GPU, EDA, SSH, or remote access was
used.
