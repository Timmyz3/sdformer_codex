# M1565 — M1564 permit-gate successor independent rehammer

## Verdict

**NO-GO for remote integration wrapper authoring.** M1564 removes the original
module-global raw mint and correctly rejects zero free space, equality at the
16-GiB reserve boundary, direct construction with an arbitrary token, permit
reuse, and path/inventory drift. However, real free-space verification can
still be bypassed through caller-controlled inputs and synthetic provenance.

## Reproduced bypass

Three paths accept a caller-supplied free-space value:

1. public `issue_preload_permit(output, free_bytes=...)`;
2. module-global `_checked_issue_permit(..., free_bytes=...)`;
3. `issue_synthetic_permit(...)`, which also accepts the exact production
   inventory and 40-sample population.

The rehammer replaced `shutil.disk_usage` with a sentinel that raises if
consulted, then supplied 24,778,606,553 bytes to each path. All three returned
the same exact `_PreloadPermit` type and consumed successfully. Each receipt
reported 17,179,869,185 bytes after the 7,598,737,368-byte result estimate,
without consulting real disk state.

Thus `production_inventory=True` cannot distinguish a production permit from
a synthetic permit, and the checked production disk gate is not enforced.

## Required narrow successor

- Production permit issuance must expose no caller-controlled free-space
  parameter and must query real disk state internally.
- Synthetic and production permits need distinct, immutable provenance.
- `ReducedBinaryProducer(..., production_inventory=True)` must require
  production provenance and reject synthetic permits, even when inventory and
  sample count match.

After that fix, another independent rehammer is required before remote wrapper
authoring. Actual capture still needs a separately sealed one-shot release.

## Boundary

Both CPython 3.10.18 and 3.6.8 local source/synthetic regressions passed. No
checkpoint, GPU, SSH, capture, release, retry, RTL or EDA operation ran, and no
AEE/performance/paper claim is created.
