# M1576 — M1574 permit provenance independent rehammer

Status: **NO_GO_M1576_M1574_EXACT_TYPE_PROVENANCE_FORGEABLE_VIA_OBJECT_NEW__SUCCESSOR_FIX_ONLY__NO_REMOTE_NO_CAPTURE**.

Both CPython 3.10.18 and 3.6.8 ran the same 30 independent attacks. Each passed
28/30. The two failures are identical: `object.__new__` can allocate either
exact permit class without invoking its secret-checking constructor; ordinary
name-mangled slots can then be populated. `consume()` checks exact type and the
slot values but does not check that the instance was minted by the closure.
The forged production object therefore emitted a `PRODUCTION_REAL_DISK`
receipt without executing the production issuer or `shutil.disk_usage`.

The narrower M1574 repairs are real. The public and closure production
signatures accept only `output`; the normal production path called the actual
`shutil.disk_usage` twice per runtime, and the first returned free-byte value
exactly equals its receipt. Normal production/synthetic cross-type attacks
were rejected. The tiny synthetic roundtrip validated and occupied 40960
allocated bytes (9947 logical bytes). These positives do not rescue the
authority claim because an exact-type forged object bypasses the issuer.

The next authorized work is source-only: keep a closure-owned registry of
minted production and synthetic instances, require and atomically remove
membership in `consume()`, and retain the two `object.__new__` attacks in both
runtimes. Remote wrapping, SSH, checkpoint load, GPU, capture, production
payload, release, RTL and EDA remain forbidden. No accuracy or performance
claim is created.
