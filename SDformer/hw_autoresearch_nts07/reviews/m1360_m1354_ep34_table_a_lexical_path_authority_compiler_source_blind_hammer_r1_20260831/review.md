# M1360 blind review of M1354 Table-A lexical-path authority

Verdict: **FAIL; additive successor required.**

All 45 inherited/new tests and the source self-check pass.  Of 55 fresh attacks,
54 are rejected.  Static leaf, ancestor, broken, escape, and symlink-to-genuine
cases close correctly.  Energy, trace, latency, all 17 SRAM planes, DRAM,
population, stall, and allowlist smuggling are also rejected.

One P0 false negative remains.  A regular leaf can be replaced after its
`lstat` check but before `resolve` with a symlink to a genuine file inside the
workspace.  Because the resolved target remains contained, M1354 accepts it.
Thus the `lstat`-then-`resolve` sequence is not an atomic no-symlink authority.

The successor should traverse from an opened workspace directory descriptor,
using descriptor-relative `O_NOFOLLOW` checks for every component and retaining
the verified file/parent descriptor for subsequent reads or exclusive output
creation.  A second pathname lookup must not become the security authority.

No production candidate, Table-A row, capture, GPU, VCS, or EDA action was
performed.  The production allowlist remains empty and docs/359 is unchanged.
