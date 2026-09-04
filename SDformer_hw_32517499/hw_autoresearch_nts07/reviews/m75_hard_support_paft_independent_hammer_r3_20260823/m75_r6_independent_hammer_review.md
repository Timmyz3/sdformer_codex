# M75 r6 independent hammer R3

## Verdict

**GO for the r6 directed hard-support mechanism and admission hardening.**

**NO-GO remains for formal PAFT training and any accuracy/speedup/RTL/PPA paper
claim.**  r6 fixes the r5 numerical P0, and no new P0 was found.  The remaining
gaps are provenance and end-to-end evidence gaps that the receipt itself
correctly keeps outside its claim boundary.

Reviewed immutable identities:

- `pattern_paft.py`: `d3eac645e5b4b2e1d9d2d5dcf9e535f936adb3be15abd86d86a4d6836120a066`;
- validator: `449aa672cb65bc645d343bfafe6fa276191e4a88cdb3cfb8699ba770b5ad1133`;
- r6 receipt: `1a84e07b296c652f1701cc25b4b27ce69ffa71ee9d75e2a59c17c4d1e40d53e2`.

R2 remains the permanent negative review of r5 and was not overwritten.

## P0

None found in the r6 directed scope.

The former r5 P0 is closed: `hard + (vectors - vectors.detach())` produced
bit-exact binary forward values and bit-exact identity gradients under fp16,
bf16, fp32 and fp64 for zero, positive, negative, small and non-dyadic
amplitudes.  Maximum forward error was exactly `0.0`.  Three nonfinite cases
(NaN, +Inf, -Inf) were rejected.

The corrected production proxy also matched the independently expressed oracle
exactly in candidate cost, baseline cost and every candidate-gradient element.
The r5 gradient discrepancy disappeared (`candidate gradient L1 = 13793`).

## P1

### P1-1: real catalog derivation still lacks an independent trust root

Strictly interpreted, `_load_catalog` can still accept a self-consistent
synthetic catalog and contract whose two SHA values are supplied by the same
config.  The production validator intentionally uses such a synthetic loader
positive control.  This does **not** create a config-only full-install bypass:
the installer independently requires the hard-coded train-list, valid-list and
checkpoint identities plus a bound runtime M73 trace, and the missing-artifact
attack was rejected before model mutation.

Nevertheless, the loader does not recompute or cryptographically bind the
catalog's 432 x 16 pattern contents to the M73 trace.  For formal scientific
admission, pin a reviewed contract/config SHA outside the launch config and/or
independently reproduce the catalog using a pinned builder SHA, seed,
tie-breaker rules and input manifest.  Until then, “no config self-attestation”
is satisfied operationally for full installation, but not derivationally for
catalog contents.

### P1-2: current-schema full positive path is not exercised

r6 retains hook plumbing evidence only from M71 r1.  It has not completed a
positive install with the real M77 catalog, real train/valid lists, real M73
trace and real checkpoint, followed by a four-operator forward/backward and one
optimizer step.  The two generated PAFT configs remain disabled and correctly
point at the revoked M71 catalog.

### P1-3: no measured benefit is admitted

The `8 -> 4` directed result is an arithmetic fixture, not a distributional or
system speedup.  No valid825/heldout accuracy, catalog hit rate, full-network
cycle/FPS, energy, area, frequency, DRAM traffic, RTL equivalence or Synopsys
PPA result exists for M75.  The r6 receipt correctly marks all those claims
false.

## P2

- Pattern chunks are copied from CPU to the activation device inside the proxy
  loop.  Cache immutable per-device/per-dtype pattern tensors to avoid repeated
  training-time transfers.
- The all-binary H67 source-domain assumption is not asserted at each live
  hook.  Add a domain/profile receipt or document how negative support maps to
  the hardware encoding.
- `regularization_weight` is checked for negativity but not NaN/Inf.  Require a
  finite nonnegative scalar before forming the loss.

## Independent evidence

- Four-dtype exact support and gradient: PASS; maximum error `0.0`.
- Production-versus-independent cost/gradient oracle: PASS exact equality.
- Directed arithmetic: PASS, baseline `8`, candidate `4`, fixture ratio `2.0x`.
- Admission/state attacks: 19/19 rejected; preexisting state preserved; failed
  install left no state or hooks.
- Old config override: rejected by permanent revoked-SHA denylist.
- `unit_test_only=true` and missing field: both rejected.
- Production r6 validator replay: PASS and byte-identical to the submitted r6
  receipt (same SHA `1a84e07b...53e2`).
- Receipt identity/claim boundary, disabled configs and revocation: PASS.
- Review scripts: Python compilation PASS.

## Scores

| dimension | score / 100 |
|---|---:|
| admission fail-closed correctness | 97 |
| numerical/software correctness | 96 |
| scientific evidence integrity | 64 |
| hardware-mechanism relevance | 62 |
| measured performance evidence | 18 |
| innovation evidence | 58 |
| DATE paper completeness contribution | 31 |
| overall M75 r6 milestone | **67** |

The mechanism score improves materially because r5's central numerical P0 is
closed.  The overall score remains capped by the absence of the real M77
catalog/full-install path and any algorithm-to-silicon outcome.
