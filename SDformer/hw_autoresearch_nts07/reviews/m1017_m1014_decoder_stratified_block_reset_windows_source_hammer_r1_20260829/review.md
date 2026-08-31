# M1017 independent hammer of M1014 decoder stratified block-reset source

## Verdict

**FAIL — block execution release.** M1014 r1 passes its seven author tests and
correctly rejects D1, zero-commit tails, windows above 10K, selections above
32, a pilot other than eight, and the literal `cycles` field.  It is not yet a
fail-closed execution source because three independent attacks cross the
measurement boundary.

## P0 findings

1. The selector uses a four-name blacklist.  `total_cycles` is accepted, so
   the selection API is not cycle-blind.  Replace the blacklist with a strict,
   recursively validated metadata schema.
2. Candidate/baseline reset equality is count-only.  An injected baseline
   boundary changed from `compute` to `external_read`, kept reset count three,
   and was accepted; cycles became 649 candidate versus 652 baseline.  Compare
   canonical reset semantics and separately account boundary/fill/drain charge.
3. The frozen CI hard stop is not implemented.  A sample with relative CI
   halfwidth 472.79% still returns point speedup 1.0.  Above 10%, point cycles
   and speedup must be suppressed; 5–10% must remain diagnostic/adaptive.

## What remains sound

- M1009 authority and M1014 receipt seals are exact and intact.
- M768/M861/M890/M896 route identities and M785/M890/M896/M946 frozen sources
  match their declared hashes.
- The small M890 synthetic four-way exact replay is 649 versus 649 cycles with
  64 commit requests.
- Transaction compression is explicitly barred from being reported as
  speedup.
- No real payload, real window, EDA, GPU, remote run, or docs/359 edit occurred.

## Next gate

Author one additive M1014-r2 repair, extend its tests with the three attacks,
seal it, and request a fresh independent hammer.  This review authorizes that
source repair only; it does **not** authorize an execution release or runner.
