# M1302 author receipt: M1288 C3 Fixed-T10 PTSTA exact-closed launch

## Author verdict

**SOURCE PASS; EXECUTION STOP pending a different-author receipt-blind hammer.**

M1302 supplies the exact admission pathname frozen into M1288, but uses a new
M1302 schema and wrapper to close the open-world admission weakness identified
by M1299.  The admission and source contract are independently double-sealed.
They bind the exact M1288 source DAG, M1299 hammer, M917/M928/M1285 seals,
mapped netlist and SDC, PT/lmutil/base tools, slow-max and fast-min libraries,
and `docs/359`.

The future wrapper order is exact admission/seals, same-UID collision gate,
resource gate, real PrimeTime license availability gate, repeated collision and
freshness gates, fixed M1302 attempt consumption, and then exactly one call to
the unchanged M1288 runner.  M1288 retains its own fixed attempt, private HOME,
isolated process group, descendant drain, failure quarantine and output seal.

M1302 then independently adjudicates the sealed M1288 reports.  PASS requires
setup and hold state `MET`, both slacks at least zero, zero constraint violators,
zero unconstrained-path diagnostics, and complete setup/hold/out_setup/out_hold
coverage (`total>0`, `met==total`, `violated=0`, `untested=0`).  Any negative
hold, missing coverage, unconstrained path, parse failure or runner failure is a
sealed STOP or failure quarantine.  A future PASS still requires a separate
receipt-blind result hammer and remains pre-layout component timing only.

Author static tests passed eight groups.  Synthetic result tests produced PASS
only for the fully closed fixture and STOP for negative hold, nonzero
unconstrained paths, and untested coverage.  No PT, EDA, license query, GPU,
remote action or result namespace mutation occurred.  `docs/359` remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

## Frozen identities

- Wrapper: `3f24a7d38df4e5c9df6b5316cc747b272fc4161d09b9a1580ea07f9998f18446`
- Static test: `db6d7de633e45629ffcc3308612b45669c0be9cada6f903578ffb60b06650e08`
- Source contract: `21294ec80d8447a128c14201247d768b48cb3c8833d8752bd8e3da91479e6b92`
- Launch admission: `1ea53ea55a8cc2bbc992aa932f73e7865561f7dde16e53f5d74efe3a7b146e3e`

The different-author hammer must verify these exact bytes, both payload seals,
all upstream seals, the exact keysets and booleans, attack mutations, the
preflight/attempt order, result mocks, and fresh namespaces before root may run
the wrapper once.
