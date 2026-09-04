# M1344 C2 mapped production-activity additive release source

M1344 is the additive successor required by the M1337 rejection. The old M1336
runner, checker, tests, contract, and failure review remain unchanged.

The checker now has explicit, disjoint modes. `source_absent` is used only for
authoring and requires M1345/M1346/M1347 not to exist. `runtime_present` is the
only mode called by the runner; it requires all three future authorities,
verifies their recursive seals and exact external SHA values, and checks their
semantic runner/contract/cardinality/claim bindings. A disposable complete
authority chain passes runtime mode, while the old future-present/absence
contradiction is retained as a regression.

Every attempt, successful candidate, and failure receipt now records nine exact
identities: runner, source contract, source-hammer review/manifest/outer,
launch release, and final-hammer review/manifest/outer. The existing 2×5
workload anchors, same-UID/resource/license gates, attempt-before-VCS rule,
DUT-only SAIF, endpoint-zero case, success/failure seals, renameat2 NOREPLACE,
no retry, no workspace UCLI key, and false performance/headline boundaries are
preserved.

New tests are 12/12 PASS; inherited M1336 directed tests are 10/10 PASS and
M1334 source tests are 12/12 PASS. No license, VCS, simv, SAIF, or EDA action
ran. A fresh different-author M1345 hammer is required before any launch release.
