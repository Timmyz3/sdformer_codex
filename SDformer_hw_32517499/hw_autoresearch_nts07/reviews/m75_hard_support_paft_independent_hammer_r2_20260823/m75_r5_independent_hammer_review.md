# M75 r5 independent hammer R2

## Verdict

**NO-GO for M75 r5 hard-support PAFT or formal training.**

The admission hardening itself is a narrow **GO**: all 19 independent negative
cases were rejected before hook/state mutation, including a preexisting PAFT
state and both `unit_test_only=true` and missing `unit_test_only`.  However, the
central r5 claim that the STE forward value is *exactly* binary is false for all
four tested floating types.  This makes the r5 mechanism and its receipt
insufficient for a training launch or paper claim.

This review is immutable historical evidence for:

- `pattern_paft.py` SHA `bf15a2ea328a16d1d8c676de11a041fc3768c6717bcf73a21c1c6e3c2378087f`;
- validator SHA `d882af175785cdcfb3a6ec5478039969a465bf156abae2e201d040b2208d59cd`;
- r5 receipt SHA `9832aa7c96a8a8699cde2bd29e249c124d29684ed42d7e2669e8b8c164fd7aae`.

Later source fixes do not change this r5 verdict.

## P0

### P0-1: the advertised exact-binary forward is not exact

r5 used:

```python
hard + vectors - vectors.detach()
```

Left-to-right floating arithmetic first rounds `hard + vectors` and then
subtracts `vectors`.  The independent counterexample found:

| dtype | example non-binary result | maximum observed error |
|---|---:|---:|
| fp16 | `0.99951171875` | `4.8828125e-4` |
| bf16 | `0.99609375` | `3.90625e-3` |
| fp32 | `0.9999999403953552` | `5.9604645e-8` |
| fp64 | `0.9999999999999999` | `1.1102230e-16` |

The direct STE derivative remains exactly one, but the error is not merely a
printing issue.  In a two-vector proxy case the r5 candidate was
`3.0000000000000004` instead of `3.0`, and the candidate gradient differed from
the exact-support oracle (`L1 13796` versus `13793`, maximum element delta
`1.0`).  Equality points of the Hamming term therefore take different
subgradients.  Mixed precision makes the forward deviation materially larger.

Required fix: evaluate the zero-valued differentiable term before addition,
for example `hard + (vectors - vectors.detach())`, and enforce `torch.equal`
plus exact-one gradients over fp16, bf16, fp32 and fp64, including zero,
positive, negative, small and non-dyadic amplitudes.

## P1

### P1-1: catalog content is not derivationally bound to the trace

The loader now binds a catalog to hard-coded train/valid/checkpoint identities,
an externally supplied contract, and a runtime M73 trace.  That correctly
prevents a config-only full-install bypass.  It does not independently prove
that the 432 x 16 catalog contents were deterministically derived from that
trace.  The production validator's synthetic positive control copies the
revoked M71 pattern contents, relabels them M77, writes a self-consistent
contract with `unit_test_only=false`, and `_load_catalog` accepts it.  Runtime
artifact gates stop that fixture from completing installation, so this is a
provenance gap rather than the old r5 security bypass.

Before formal launch, pin the builder source SHA, seed/tie-break rules and input
manifest, and independently reproduce the final M77 catalog SHA from the M73
trace.

### P1-2: no current-schema full positive installation evidence

r5 explicitly admits only the loader and arithmetic unit path.  Hook plumbing
is inherited from M71 r1; there is no positive installation using the real M77
catalog, real train/valid lists, real M73 trace and real checkpoint, followed by
a four-operator forward/backward optimizer step.  This must be completed after
the real train-only artifacts exist.

### P1-3: no algorithm or hardware outcome is admitted

The receipt correctly sets formal launch, accuracy, heldout speedup, cycle
speedup, and RTL/PPA claims to false.  The directed `8 -> 4` result is a toy
arithmetic point, not expected acceleration.  M75 r5 therefore contributes no
DATE-ready system-performance evidence yet.

## P2

- Pattern tensors remain on CPU and are moved to the activation device inside
  every partition chunk.  This can produce repeated host-to-device traffic and
  synchronization overhead during training; cache one immutable device/dtype
  copy per operator.
- The support function treats every nonzero value, including negative values,
  as the same active bit.  H67 is intended to be all-binary, but the live hook
  does not assert/bind that input-domain invariant.  Add a runtime/profile
  domain receipt or explicitly document dual-polarity cost semantics.
- `regularization_weight` rejects negative values but not NaN or infinity.
  Require a finite nonnegative value before forming the loss.

## Independent evidence

- Independent oracle (no production import): PASS for intended `8 -> 4`,
  amplitude-independent exact-support arithmetic, finite nonzero gradients,
  receipt identities, disabled configs, and revocation state.
- Production validator replay: PASS and byte-identical to r5 receipt; both SHA
  values are `9832aa7c...fd7aae`.
- Admission attacks: 19/19 rejected; failed installation left no PAFT state or
  model hook.
- Exact-support counterexample: FAIL across fp16/bf16/fp32/fp64; gradient
  mismatch demonstrated.
- Review scripts: Python compilation PASS.

## Scores

| dimension | score / 100 |
|---|---:|
| admission fail-closed correctness | 96 |
| numerical/software correctness | 60 |
| scientific evidence integrity | 53 |
| hardware-mechanism relevance | 60 |
| measured performance evidence | 18 |
| innovation evidence | 58 |
| DATE paper completeness contribution | 28 |
| overall M75 r5 milestone | **50** |

The overall score is capped by one central P0 and by the explicitly absent
real-catalog, accuracy, cycle, energy and PPA evidence.
