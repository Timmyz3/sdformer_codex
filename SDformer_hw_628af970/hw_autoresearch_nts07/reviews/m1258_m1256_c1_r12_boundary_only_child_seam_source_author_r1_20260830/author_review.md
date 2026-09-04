# M1258 — additive R12 boundary-only child-seam TB source

Status: **SOURCE GO for one fresh independent source hammer only.**  This
milestone does not authorize release authoring, VCS, simv, EDA, GPU, or remote
work.

M1256 proved that R11's random transaction forced the M1162 parent connection
while the reset M935 child had no real request.  R12 changes only the TB object
boundary.  Every synthetic directed/random/service request now forces and
releases the child M935 request-output seam.  Synthetic core-ready control is
likewise applied only to the child's `issue_data_ready` output.  There is no
executable force or release of `dut.issue_request_*` or
`dut.core_issue_data_ready`.

This is explicitly an M1162 boundary protocol test, not integrated M935
traffic.  Boundary-only phase, coverage, and PASS tokens say so.  Random was
renamed `random_boundary_transaction`; the PASS contract states
`integrated_random=false` and `integrated_m935_claim=false`.

The independent real M935 normal path remains intact.  `load_normal_task`,
`serve_normal_beat`, and `normal_m935_completion` are byte-identical to R11,
with exact task hashes respectively
`fa86553341c84a31e0715dd751ce6f41161eed7557932bb35fbbaafc20b9a669`,
`9c27568bd89e590de7c40fad88c1eeedb818875ab7ac8f01c492965abff838c5`,
and `bf30589e69f52f856edb269fcdde20f837eb1a646264d22e60d0ca70ef6a51f4`.

The fail-closed checker passes, and 12 tests pass: canonical source plus a
comment-decoy positive, and ten negative mutations covering parent request
force, parent ready force, missing boundary label, integrated-random claim
inflation, integrated-M935 claim inflation, real-normal drift, missing random
reset, missing child seam force, missing integrated-normal phase token, and
renaming random back to `legal`.

Frozen identities remain unchanged: R11, M528, M935, M1162, R3 SVA, M1256,
and `docs/359`.  A fresh independent hammer must challenge the seam scope,
reset isolation, claim partition, and exact normal-task preservation before
any release can be authored.

No hardware performance, functional VCS, timing, cycle, PPA, energy, system
speedup, headline, or paper claim is admitted by this source-only result.
