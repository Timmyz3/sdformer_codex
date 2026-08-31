# M1250 — C1/R11 one-shot VCS release source authoring

Status: **source GO; a fresh disjoint M1251 hammer is mandatory before the
runner may execute.** This step ran no VCS, simv, EDA, GPU, or remote work.

The release binds the byte-frozen R11 TB, M1246 checker and 24 tests, M1247's
100/100 release-authoring authorization and both seal layers, plus the frozen
M528, M935, M1162, R3 SVA, macro model, tool binaries, and `docs/359`.

The future runner consumes its attempt namespace before opening work, admits
one compile and one simulation only, and applies independent 1200 s compile
and 1800 s simulation timeouts. Any nonzero tool exit, internal watchdog,
missing phase/random/normal/coverage/PASS evidence, or Error/Fatal/Assertion
line fails closed. A failed/incomplete work directory receives a recursive
double seal and moves to a unique quarantine; the consumed attempt prevents
retry. Canonical result promotion occurs only after every log gate and the
recursive success seal.

Final static validation passed 75 checks. Twelve release mutation tests reject an
older TB, a second compile, a second simulation, loss of either timeout,
removal of Error/Fatal/Assertion rejection, missing phase/random/normal gates,
loss of quarantine, missing digest pins, destructive cleanup, and retry.
The inherited M1246 checker exits zero and its 24 tests pass.

No functional claim is admitted yet. M1251 must independently bind and seal
the runner, filelist, contracts, author package, and all fail-closed gates,
then explicitly authorize no more than one compile plus one simulation.
