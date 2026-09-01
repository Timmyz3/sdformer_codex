# M1559 — M1556 immutable-input ordinary consistency review

## Verdict

PASS. Commit `bf502765` satisfies the input-fixedness and dual-runtime
prerequisites for authoring a separately sealed, exactly-one-shot D0/call0
diagnostic release over the three non-product configurations. This review does
not authorize or execute that run; production, product capture and automatic
retry remain forbidden.

## Independently checked

- Clean import freezes call0 metadata in immutable closure values: sample 10,
  module D0, call ordinal 0, shape `10x1x1536x15x20`, member
  `payloads/c000_s10_d0.positive.le.bitpack`, and SHA-256
  `37208563da5f5b218f3aff5b292f05e10a5db16b078672762b2cb9ed60678a1c`.
- `stream_actual_call` exposes only the `config` parameter and does not retain
  the selector function or mutable row metadata in its executable closure.
- A synthetic file with the exact 576000-byte size was read into an exact
  `bytes` object. The file descriptor was closed before consumer access.
  Changing the source file afterward changed its SHA but did not change the
  copied bytes or observed bit value.
- The author test, synthetic self-test and preflight passed under CPython
  3.10.18 and CPython 3.6.8.
- Pinned source, test, contract, receipt, and docs/359 SHA values all matched.

## Boundary

This was an ordinary data-consistency regression, not a security or attack
test. It did not call `stream_actual_call`, did not reach request zero, and did
not run a real pilot, production, product capture, GPU, SSH, RTL or EDA task.
No transaction, cycle, traffic, energy, speedup, PPA, Table-A or paper claim is
created.

The next permissible step is to author and independently seal a one-shot
release that is fixed to D0/call0 and exactly the three non-product
configurations. That separate release must preserve zero automatic retries and
must not expand to production or product capture.
