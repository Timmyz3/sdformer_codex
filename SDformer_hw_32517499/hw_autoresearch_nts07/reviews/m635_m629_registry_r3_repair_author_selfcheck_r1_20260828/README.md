# M635 author self-check: M629 registry repair

## Outcome

This package supersedes the **methodology** of M628 r2 without changing any
sealed M628 artifact.  It is an author self-check, not an independent review.

- CPU tests: **14/14 PASS**.
- Canonical builder: `sources=12 bundles=0 table_a_eligible=0 headline_admitted=false analytical_admitted=false`.
- M629 blocking findings addressed in code: **4/4**.
- Table A result admitted: **false**.
- Paper headline admitted: **false**.
- GPU/EDA/remote/paper-body run: **none**.
- `docs/359` SHA-256 remains
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

## Repair mapping

| M629 finding | M635/r3 repair |
|---|---|
| P1-01 self-attested Table A | Table A source IDs now name a dedicated five-artifact `direct_unified_` bundle. Row numbers, identities and closures must exactly project SHA-bound result/completion/resource/coverage/hammer evidence. Aggregate/view values are recomputed from raw samples and Table-A cycle operands sum the raw `iso_service` samples. Existing B/C source IDs are a disjoint namespace and are explicitly rejected. |
| P1-02 config-controlled ladder | Six IDs, roles and fidelities plus numerator/K1x8/candidate anchors are code constants. Coordinated deletion, rename, role mutation and anchor mutation are tested. |
| P1-03 hash-only JSON evidence | Every registered source and every nested bundle/coverage receipt is strict-parsed after confinement and SHA verification. Duplicate keys and NaN/Infinity/overflow-to-infinity are rejected. |
| P1-04 M518 provenance | `m518` now binds the r11 post-run receipt-hammer verdict SHA `513c5d...6665`; its schema, P0/P1 state, authorization and 17-cycle anchor are checked. |

M629 P2-01 is also narrowed: ratios and percentages in Tables B/C are stored as
exact Decimal strings; rounding is deferred to paper rendering.

## Coordinated attack disposition

1. Delete `exact_bit_k1x8` from both rows and the config's required list:
   **rejected by the code-level ladder**.
2. Relabel M618 as all six direct-unified rows, insert positive numbers, fake
   SHA strings and all closure booleans: **zero eligible rows; headline false**.
3. Add a SHA-correct JSON source with duplicate keys or a non-finite value:
   **strict parser rejects it**.

## Boundary

M635 only repairs the registry gate.  It cannot admit Table A without a real
decoder-complete common-resource run, completion/resource/coverage receipts,
closed energy/PPA/accuracy evidence, and a fresh independent result hammer.
The next authority is a fresh independent M635 review with P0=0 and P1=0.
