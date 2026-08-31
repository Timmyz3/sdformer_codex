# M987 independent copy-only promotion release hammer

Verdict: **STOP**, score 80/100, P0/P1/P2 = 1/0/0.

All frozen pins pass: promotion script, M975 source contract, M975 review,
M976 release and both sidecars, source-quarantine recursive exact-set seals,
and `docs/359`. The source quarantine has 31 manifest entries, no symlink,
and remains unchanged. The recovery target and all copy-work prefixes are
fresh. The script contains no EDA invocation and M987 did not execute the
promotion.

The release is not fail-closed under concurrency. It creates a PID-specific
work directory without a shared lock or consumed-attempt marker, checks the
target only before copying, and ends with `mv WORK TARGET` without `-T`,
no-replace, or a final target check. A temporary-directory attack reproduced
the exact GNU `mv` behavior: two concurrent publishes both returned zero, and
the second work directory was nested inside the first target. That mutates an
already sealed canonical directory and invalidates its recursive exact set.

M988 promotion is therefore not authorized. The source quarantine stays
immutable and the target must remain absent. A repaired successor must add an
atomic lock/attempt plus no-replace `mv -T`, then receive a fresh independent
hammer before execution.
