# M1031 license-complete C2 SAIF source receipt

M1029 closes the consumed M1022 failure: Full64 VCS launched but the caller's
clean environment removed both license routes, so compilation stopped before
simv, gate simulation, or SAIF.  M1022 is not retryable.

M1030 adds a frozen tiny-SV checkout gate before the new M1033 attempt.  The
runner requires at least one nonempty caller license route, preserves the
caller value, and never prints, records, hashes, sizes, or embeds that value in
receipts.  It fixes `VCS_HOME` and `PATH`, verifies VCS, `vcsMsgReport`, and the
tiny source, then performs a fresh `vcs -full64` compile with compiler output
suppressed.  A generic sealed preflight receipt is published; failure exits
without consuming the M1033 attempt.

An author-side noncanonical smoke compiled and linked the frozen tiny source,
created `simv`, ran it, and observed
`PASS_M1030_VCS_LICENSE_CHECKOUT_PREFLIGHT_SOURCE`.  This proved checkout but
did not run M1033 or create production SAIF.

The production body remains the frozen three axes by five cases, fresh compile
per axis, DUT-only SAIF, with a fresh M1033 result/attempt namespace.  Eight
source tests cover missing license, secret non-disclosure, wrong tiny hash,
missing preflight, occupied namespace, collision, axis loss, and case loss.

Only an independent M1032 hammer is authorized.  M1033, PT, PTPX, DC, GPU,
power, energy, and system claims remain unauthorized.
