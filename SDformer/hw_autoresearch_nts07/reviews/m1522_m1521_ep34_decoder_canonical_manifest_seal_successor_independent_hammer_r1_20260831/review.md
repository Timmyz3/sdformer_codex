# M1522 independent hammer of the M1521 canonical-manifest seal successor

Verdict: `PASS_M1522_M1521_CANONICAL_MANIFEST_SEAL__M1523_RELEASE_ONLY`.

The independently reproduced author suite passed 11/11. The blind hammer then
passed 65/65 checks over 33 deliberate mutations. The original M1517 semantic
forgeries (scale, encoding, fold, normalization, coercion, duplicated global
order, performance claim, and renamed output) were independently rejected at
both pre-seal and post-publication verification. No failed pre-seal attack left
a seal behind.

Additional attacks covered bool/int JSON aliasing, 119/121-call populations,
record reordering, record payload-SHA drift, payload byte drift, and 119/121
payload populations. Public `seal_staging(root)` and
`verify_materialized_seal(root)` accept no caller-supplied expected manifest;
their tested path regenerated expectations via canonical M1458 capture,
M1510 audit, and M1516 enrichment.

This is source-only authority. M1523 may now bind this sealed review to author
the exact one-shot release. Production materialization, address-timed replay,
cycles, traffic, energy, PPA, and Table-A claims remain unauthorized.
