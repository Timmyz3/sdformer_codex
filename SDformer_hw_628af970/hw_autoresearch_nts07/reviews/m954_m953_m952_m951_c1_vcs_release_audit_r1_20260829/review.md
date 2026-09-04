# M954 | M953/M952/M951 C1 VCS release audit

Verdict: `GO`, review status
`PASS_M954_M953_M951_VCS_LAUNCH_RELEASE_AUDIT`, score 98/100,
P0=0/P1=0/P2=1. This audit did not launch VCS or any other EDA tool.

M953 release SHA is
`4cf1ece7aad593c8faa753e88ee51f346dcbc39531586229cd85d0e388b9c873`;
both release sidecars validate. The M951 runner and source-contract SHAs match
the release, and the sealed M952 review/manifest/outer-seal chain validates at
the exact recorded identities. The release grants exactly one VCS compile and
one simv run, with zero authorization for all other EDA.

The M951 inline release assertions were replayed read-only and pass. Its unique
attempt, result and work-prefix paths are absent. The consumed M943 identity
hash and recursively sealed failure quarantine remain exact and immutable. The
abandoned M950 runner still exits unconditionally with status 98 before any
environment, path, attempt or tool action, and M951 contains no M950 reference.

At audit time the same-UID EDA scan found zero processes and MemAvailable was
421,537,536 KiB, above the 67,108,864 KiB gate. These are live facts, so the
runner correctly rechecks both immediately before consuming its attempt.

The authorized launch environment must inject the exact M953 SHA above, M952
review SHA `2d4d8acc...`, M952 outer-seal-file SHA `851d23e9...`, and no runner
arguments. The literal descriptive placeholder in M953 is not a launch value.

P2: process and memory state is necessarily transient between audit and launch;
the runner's immediate fail-closed recheck is the controlling gate. No P0/P1
was found. This GO authorizes only one functional foundry-UNIT_DELAY M951
attempt; it admits no timing, workload cycles, speedup, PPA, power, energy,
system, headline or paper claim.
