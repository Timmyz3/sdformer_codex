# M1248 one-shot cross-run binder release author receipt

M1248 is source-only and production-inert in this milestone.  It binds the
exact M1241 source/test/contract and recursively verifies the exact M1245
manifest, outer seal, PASS status, release-authoring permission, and unchanged
no-rebind/result-hammer boundary.

All four checkpoint paths, all four strict-valid825 profile paths, both config
paths, and the resume manifest must be regular non-symlink files before the
attempt marker can be created.  Output, attempt, and log namespaces must be
fresh and pairwise distinct.  After all preflight gates, the attempt is created
atomically with `O_EXCL`; exactly one pinned M1241 child may run.  Child failure,
seal failure, or claim-boundary failure leaves the attempt consumed and permits
no retry.

The child's only writable production products are a small double-sealed
selection receipt and a small hashed launcher log.  The wrapper rechecks exact
receipt members, manifest, outer seal, payload hashes, M1234 schema/status,
terminal token, selected candidate/epoch domain, and false hardware/system
claims.  It does not run valid825, train, load a model, use a GPU, launch
capture, contact the remote host by itself, copy a checkpoint, or invoke EDA.

Ten controlled temporary-fixture tests pass.  They cover every missing
profile/checkpoint/config/manifest, M1241 and M1245 drift, all three namespace
collisions, child failure/no retry, unsealed output, over-authorized output,
runtime identity drift, and docs/359 drift.

Production execution is not authorized now: ep30/32/34 strict-valid825 inputs
are not yet all present, this exact package still needs a fresh different-author
hammer, and no M1248 attempt/output/log namespace has been touched.
