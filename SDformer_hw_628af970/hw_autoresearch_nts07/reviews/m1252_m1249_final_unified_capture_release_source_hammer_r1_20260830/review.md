# M1252 independent hammer: M1249 final unified-capture release source

Verdict: `GO_FUTURE_PRODUCTION_LAUNCH_AUTHORING_AFTER_EXACT_M1237__NO_GO_NOW`.

The exact M1249 source, test, source contract, recursively sealed author receipt,
M1243 source/test/contract pins, and recursively sealed M1244 source hammer all
match.  M1244 carries only `production_capture=true`.  No production M1237 result
hammer exists, so no production launch contract may be authored or executed now.

The controlled suite passes 18/18.  The independent hammer rejects 30/30 fresh
mutations across M1243 identities, all three M1244 seal fields, all three M1237
entry seal fields, all M1234 selection seal/member fields, every M1237 authority
cross-field, release identity, each result/attempt/log namespace, retry policy,
and top-level shape.  Separate occupancy attacks reject each canonical namespace.

Dynamic ordering is `lease_enter -> all_preflight -> attempt_O_EXCL -> capture ->
lease_exit`.  Rejected preflight exits the lease without consuming the attempt.
The marker is mode 0400, the second create is rejected by `O_EXCL`, and automatic
retry is false.  Checkpoint, configuration, and profile identities are hashed by
the exact M1233 selection validator; cohort and frozen runtime-source evidence are
also revalidated inside the lease before the marker is created.

Capture behavior is unchanged by identity alias: 259 static modules, 247 runtime
live modules per sample, 12 dead `sn_v`, 9,880 ordered records, 480 attention
records, 640 payload files, and per-sample atomic snapshots.

Two nonblocking test-quality findings remain.  Nested M1233 selection failures are
fail-closed but are not normalized to `M1249Error`.  In addition, the author test's
M1237 shape attacks mutate a shared fixture, making later seal-field assertions
less specific.  Fresh independent fixtures reject every seal and cross-field
attack, so neither finding is an admission bypass.  The eventual production
launch author should use fresh deep copies and treat any validation exception as
failure.

This hammer authorizes only a separately authored production launch after the
exact, recursively sealed M1237 entry exists.  It performed no remote access, GPU
work, checkpoint load, capture, release, EDA run, or paper measurement.
