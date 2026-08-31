# M1278 decoder completion gate / diagnostic annex author receipt

Status: **SOURCE PASS; current 74/120 live state exits 75 with no output and no replay.**

The additive zero-argument gate pins the exact M1111DR2 PID, work/result/lock/
attempt namespaces and frozen runner SHA.  While the producer is alive it
validates its canonical prefix and returns `INCOMPLETE`; it cannot publish an
annex.  After natural producer exit it reuses the frozen runner's
`validate_publish_candidate`, rechecks the exact 120 ordinals, D1 theta,
identity, completion token and atomic seals, and may atomically publish only an
ep35 decoder-only diagnostic projection.

Eight temporary-fixture tests pass: incomplete prefix, duplicate ordinal,
valid projection, wrong D1 theta, damaged seal, wrong checkpoint identity,
wrong live workdir, and no-Table-A annex publication.  No test reads or writes
the live work directory.

The real preflight returned exit code 75 with
`M1278_INCOMPLETE_ROWS_74__NO_OUTPUT_NO_REPLAY`.  The M1278 annex namespace
remained absent, PID 4122290 remained healthy, and the long producer was not
stopped or modified.

This receipt does not authorize replay, GPU, remote, RTL, EDA, Table-A,
system-speedup, energy, PPA, or headline claims.  A successful future annex
still requires a different-author result hammer and final-checkpoint rebind.

