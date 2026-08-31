# M1347 different-author blind hammer of M1343

## Verdict

**FAIL_SOURCE / DO NOT AUTHORIZE RELEASE.** M1343 correctly closes the consumed
M1329 failure and correctly changes the ep34 graph arithmetic to 105 live ATLIF
modules, 259 hooks per sample and 10,360 ordered records.  Its release gate is
nevertheless unusable for two independent P0 reasons.

First, three read-only CPU reconstructions on the remote source host loaded the
exact checkpoint and config with `missing=0` and `unexpected=0`.  The profile
and ATLIF overlay source hashes also equal the local sources.  Both runs found
105 ATLIF modules, zero `.sn_v` modules and the canonical sorted-name digest
`6a616f16...94cb7`, not M1343's `ca7dab07...40265`.  The complete 105-name
sorted array is sealed in `remote_cpu_inventory.json`; recomputing SHA-256 over
the names joined by LF plus a terminal LF reproduces the observed digest.  The
real writer therefore fails during `expected_live105_inventory` before any canonical capture can be
published.  No GPU, forward, capture, attempt token or remote write was used.

Second, `--source-self-check` rejects M1343's own sealed contract.  The source
requires exact equality against a two-key `test` object, while the contract's
object also contains `passed` and `failed`.  The observed terminal error is
`M1343Error: M1343 test identity mismatch`.

The author's 12/12 tests were independently reproduced.  They do not cover the
first P0: their fixture replaces `EXPECTED_ATLIF_NAMES_SHA256` with the digest
of synthetic `unit.atlif.*` names.  The author double seal, M1329 failure token
and log, docs/359 hash, exception restoration, namespace collision rejection,
checkpoint/config binding, no-production-CLI boundary, and narrow patch surface
all passed independent checks.  Those strengths do not repair either P0.

## Minimum additive successor

1. Produce a sealed, read-only CPU inventory authority from the exact remote
   checkpoint/config and pinned profile/overlay sources.  Include the complete
   ordered ATLIF name list, count, explicit sort policy and digest.
2. Add a new successor source (do not overwrite M1343) bound to that authority.
   Use the measured `6a616f16...94cb7` only after the list itself is sealed.
3. Make source-policy validation accept exactly the intended contract schema;
   either compare the four `test` fields or project `path`/`sha256` explicitly.
4. Add a test that rebuilds or consumes the sealed real inventory without
   monkey-patching the expected digest, then rerun the different-author hammer.
5. Only a later exact-SHA one-shot release author may consume the fresh M1343
   successor namespace.  This review authorizes no GPU/capture/attempt.

## Claim boundary

This is source review only.  It makes no cycle, speedup, energy, PPA, model
accuracy or paper-readiness claim.  `docs/359` remains unchanged.
