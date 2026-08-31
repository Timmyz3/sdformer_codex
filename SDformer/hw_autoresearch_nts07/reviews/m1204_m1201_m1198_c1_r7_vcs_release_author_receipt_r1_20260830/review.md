# M1204 — C1 R7 VCS launch/release source author receipt

Status: **source-only PASS; fresh different-author M1206 release hammer is
mandatory; no VCS, simv, license checkout or EDA has run.**

M1204 consumes M1201's exact authorization and creates a new R7 launch chain.
The launcher pins the M1198 checker/contract/author receipt, exact clean R6 TB,
R7 filelist, frozen R3 SVA, M1162, M935, the parent macro wrapper, foundry
UNIT_DELAY model, VCS/Python binaries and `docs/359`.  M1201 is recursively
verified and additionally pinned by the requested `review.md` digest
`a4310900...`, its review JSON, manifest and outer-seal digests.

The release is inert until a fresh M1206 review directory is recursively
sealed, has zero P0/P1, score at least 95, exact GO schema/status, and binds the
M1204 runner, source contract and release by SHA.  This check occurs before the
fresh attempt token is created.  The future authorization is exactly one VCS
compile plus one simv run; all other EDA is zero.

The future simulation gate exact-matches all four R6 coverage lines and the
terminal PASS line.  It therefore requires the inherited 16 assertions, six
covers, seven DUT attacks, two service-assumption attacks, 24 deterministic
legal transactions, 29 legal mask-clear observations, three reset states,
attained II=2, isolated service skew with no reachable core-ready force, and
one normal frozen-M935 row/task.  A bare PASS token is insufficient.

The author ran only read-only source validation, Python compilation, JSON
parsing, sidecar/recursive-seal checks and shell syntax parsing.  No M1204
attempt/result/work/quarantine namespace exists.  No functional, timing,
cycle, speedup, PPA, power, energy, system or paper-ready claim is created.

