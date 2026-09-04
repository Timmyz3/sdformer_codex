# M1188 fresh hammer of the M1187/R4 C1 VCS release

Verdict: **GO for exactly one foundry-UNIT_DELAY functional VCS compile and
exactly one simv run through the exact M1187/R4 launcher.**  This review did
not invoke the launcher, VCS, simv, any other EDA executable, or a license
client.

The repaired pre-attempt gate was executed on valid sealed fixtures twice and
returned `PASS_M1187_R4_PRE_ATTEMPT_GATE` without the prior identity-key
`KeyError`.  The author suite rejected 12 mutations.  An independent suite
rejected 22 additional mutations covering hammer schema/status/verdict/score,
P0/P1, release/runner/contract identities, execution and authorization counts,
all five runtime digest bindings, recursive review integrity, and the outer
seal.

The launcher checks the fresh M1188 review and recursive outer seal before the
R4 attempt directory is created.  R1 reuse is forbidden, the failed R2
quarantine remains recursively sealed, R3 remains absent, and the new R4
attempt/result/work/quarantine namespace is fresh.  The runner contains one
UNIT_DELAY VCS compile, one timeout-bounded simv run, same-UID EDA collision and
64-GiB memory gates, plus recursive failure and success sealing.

This authorization is functional only.  It does not establish timing, cycles,
speedup, PPA, power, energy, system performance, paper readiness, or a
headline result.
