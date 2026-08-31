# M1249 final unified capture one-shot release source authoring

## Outcome

The final release source/template is authored and 18/18 tests pass. It binds the exact
M1243 source, test, and contract plus the recursively sealed M1244 hammer and its explicit
`production_capture=true` authorization.

No executable production launch contract was created. The selected final checkpoint's
M1237 result hammer does not yet exist, so production remains fail-closed.

## Final M1237 admission

A future production launch must supply the exact M1237 entry with only `path`,
`review_sha256`, `manifest_sha256`, and `outer_file_sha256`. M1249 delegates validation to
the frozen M1243/M1233 selection path, which recursively verifies the seal, fixed M1237
schema/status, selection/checkpoint/config/profile cross-SHAs, different-author status,
and the exact hardware-rebind authorization.

## One-shot ordering

The result, attempt, and log namespaces are pairwise disjoint, fresh, and disjoint from
M1227/M1233/M1243. Under the exclusive GPU lease, all source, checkpoint, configuration,
profile, cohort, selection, and hammer admission checks run before the attempt marker is
created with `O_EXCL` and mode 0400. Once created, the marker is not removed on failure;
automatic retry is fixed false.

## Boundary

This authoring performed no remote access, GPU execution, checkpoint selection, capture,
release, EDA, cycle measurement, speedup, energy, PPA, or paper-result admission.

