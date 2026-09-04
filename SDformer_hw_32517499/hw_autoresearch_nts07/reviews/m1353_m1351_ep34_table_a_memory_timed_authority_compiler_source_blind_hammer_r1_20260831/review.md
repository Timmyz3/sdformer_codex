# M1353 — M1351 memory-timed Table-A authority compiler blind hammer

## Verdict

`FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED`

The author seal and frozen identities verify, and all inherited/author tests
reproduce: M1340 10/10, M1342 16/16, M1351 13/13, plus the source self-check.
The new compiler correctly rejects 21 fresh attacks covering absolute and
`..` escapes, arbitrary B0/Ours direct-energy rates, malformed or unbound
trace payloads, missing memory planes, missing latency/population, broken
cycle/stall conservation, and invented production allowlists.

One fresh attack is accepted.  A read-only symlink leaf inside the workspace
can be supplied as the config path when it points to the genuine config.  The
implementation calls `candidate.resolve()` before walking the ancestry, then
walks components of the resolved real path.  The symlink leaf has disappeared
by that point, so the build returns a valid source-fixture result instead of
rejecting the symlink.

This does not create a production Table-A row because the production allowlist
is still empty.  It does prevent source admission under the explicit
zero-false-negative rule.  The minimum additive successor must lstat every
lexical path component, including the leaf, before resolution; separately
check resolved containment; add this exact leaf-symlink regression; and then
receive another independent hammer.

No production row, capture, GPU, VCS, DC, PT, PTPX, EDA or remote task ran.
`docs/359` remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
