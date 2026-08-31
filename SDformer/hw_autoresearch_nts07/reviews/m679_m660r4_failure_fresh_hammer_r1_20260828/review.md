# M679 — M660-r4 one-shot failure fresh independent hammer

## Verdict

**GO_CLEAN_NEGATIVE_RESULT_ONLY — 97/100, P0=0 / P1=0 / P2=1.**

The M660-r4 one-shot is consumed and must not be rerun. It produced no
canonical payload, but it did produce a clean, fail-closed negative result:
the current S10 execution does not reproduce the frozen M649/M511 `s00_d0`
activation ledger.

This GO admits only that negative result. It does not admit an M660 payload,
decoder equivalence, cycles, speedup, RTL, EDA, energy, PPA, system speedup, or
a DATE headline.

## Independently reproduced failure

The retained `s00_d0.activation.le.bitpack` is exactly 576,000 bytes, little-bit
order, and covers 4,608,000 elements. Independent decoding gives:

| identity | active | zero | packed SHA256 |
|---|---:|---:|---|
| Current M660-r4 | 838,404 | 3,769,596 | `10981fb3970dd6918c0f89645ce7cf4b3cfb73816cd2fa9ab0e9ea3dc4895d5d` |
| Frozen M649/M511 | 839,586 | 3,768,414 | `ad2251f1fb8a470651044456e0b7182bd6db0e0a89fb63018efa3a9e6fcd6447` |

The current run has 1,182 fewer active values and 1,182 more zeros. The retained
payload therefore independently reproduces the exact `zero_count` gate that
raised the failure.

The divergence is larger than the net population delta suggests: direct
comparison with the M511 payload finds 264,066 changed bits (5.7306%), composed
of 132,624 old-active→current-zero and 131,442 old-zero→current-active changes.
This is a true frozen-trace reproducibility failure, not a corrupt or truncated
bitpack.

## Failure-package integrity

- Canonical M660-r4 output is absent.
- The consumed-attempt receipt, runtime receipt, original-weight package, and
  top-level failure package all pass their inner manifest and outer-seal checks.
- Manifest population checks find no unsealed extra files and no symlinks.
- The failure tree retains only the four original weights and the causal D0
  bitpack. It contains no D1 candidate directory, D1 activation, folded weight,
  sidecar, partial file, or stale success manifest.
- `FAILED.json` is consistent: zero completed admitted records, zero D1 records,
  fail-closed status, and no canonical result. Its empty scrub list is valid
  because failure occurred before any D1 candidate was created.
- Runtime argv, package versions, host/GPU identity, deterministic controls,
  TF32 controls, CUBLAS setting, and allowlisted environment are double-sealed.
- All attempt identity hashes still resolve to the reviewed runner, producer,
  contract, tools, preflight root, and docs/359.

The failure package is cryptographically self-contained and retains enough
causal data to reproduce the mismatch against the double-sealed M649 root.

## P2-1 — diagnostic receipt completeness

`FAILED.json` records only the generic `zero_count` mismatch; it does not embed
the retained record's shape, bit order, payload SHA, observed/expected counts,
or deltas. That is a diagnostic-completeness issue, not an integrity issue,
because the bitpack is retained and independently decodable.

For future one-shots, add those fields to the failure receipt. Do not mutate
this sealed package or rerun the consumed one-shot merely to repair P2.

## Disposition

- Preserve the attempt and failure staging exactly as sealed.
- Treat M649 count equality as not reproducible for this capture identity.
- Do not use this run as a decoder payload or performance point.
- Do not restore or repeat the one-shot under M678 authorization.
