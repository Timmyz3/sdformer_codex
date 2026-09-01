# M1813 independent source hammer of M1812 TSBG campaign

Status: `FAIL_CLOSED_M1813_M1812_TSBG_PRODUCTION_CAMPAIGN_SOURCE_HAMMER__P1_1__NO_VCS_NO_EDA`

Score: 91/100. Findings: P0=0, P1=1, P2=0. M1814 and the M1812 VCS campaign are not authorized.

## What closed

The concrete M1812 runner is presently well governed. It has eight exact
external authority pins; verifies the M1812 contract, future M1813 review, and
future M1814 release seals; requires the all-false prelaunch boundary; excludes
same-UID/shared-queue collisions; and budgets exactly one license query, one VCS
compile, and one simv run. The compile command includes `-assert svaext`.
After attempt consumption, failure is sealed into a do-not-retry quarantine;
success uses no-replace atomic publication.

M1795's replay/reset mutation request is also substantively closed. Both Python
3.6 and 3.10 independently pass the source checker and reject all 48 declared
mutations. Those attacks cover accepted bank-3 epoch, slot, generation, tag,
and the loop-wide 16-lane signed payload on capture and replay; zero replay
acceptance; both three-clock resets; full post-reset issue, commit, terminal,
and recovery ledgers; and the SVA 1..8-clock recovery plus later clean terminal.

## Blocking finding

The campaign overclaims the runner/release portion of its mutation hammer. A
separate nine-attack in-memory probe was accepted 9/9 by `validate_semantics`.
It can bypass the calls to `verify_authority`, source validation, namespace,
collision, or resource gates; keep `state["attempt"]` false after creating the
attempt directory; or remove M1794, M1795, and docs359 from the M1814 identity.
The current frozen runner does not contain those regressions, but its semantic
checker cannot reject them. Exact contract SHA protection is real, yet M1795
correctly established that a SHA mismatch is not semantic-regression proof.

This is one P1 because it weakens the exact governance closure used to authorize
the sole VCS attempt. Do not overwrite M1812. An additive successor should add
explicit call-reachability and complete identity-key predicates, mutate every
one of them—including the self-runner pin—and obtain another different-author
zero-severity review before any release or VCS launch.

No EDA tool or license was invoked, no attempt/result/release was created, and
M1794, M1795, M1812, and docs/359 were not modified.
