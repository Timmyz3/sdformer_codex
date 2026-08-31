# M1303 independent blind hammer — M1301 claim-authority successor

Verdict: **93/100; P0=0, P1=1, production STOP.**

M1301 correctly repairs the M1298 finding. It restores the exact seven M1292
claim keys as exact `false` booleans, removes `paper_ppa_ready`, pins the exact
M1297 source/test/contract, and verifies the exact M1298 manifest, outer seal,
blocking status, and denial authority before delegation. Independent attacks
against missing/extra/true/non-boolean claims, predecessor SHA drift, M1298
member/manifest/outer drift, interpreter path replacement, and attempt reuse all
fail closed. The inherited M1297 data path remains the same policy object with
11 snapshots, three sealed execution sources, `/proc/self/fd` execution, four
exact passed FDs, retained interpreter identity, entity-bound attempt, and
`O_EXCL` no-retry semantics.

## Blocking P1

The wrapper does not preserve one production preflight from frozen M1297:

```text
M1297.main:
  M.verify_frozen_authorities()   # exact M1257 source/test/contract
  M1297.execute_once(...)

M1301.main -> M1301.execute_once:
  verify M1297 triplet + M1298 seals
  validate repaired claims
  M1297.execute_once(...)         # inherited preflight is skipped
```

The hammer reproduced this without touching production: an injected failure in
`M1297.M.verify_frozen_authorities()` was never called and the delegated stub
returned normally. The M1257 executable source is still transitively SHA-pinned
at import, so this is P1 rather than arbitrary-code P0; however, M1257 test and
contract drift are no longer rejected at the production gate that M1297 had.

Required additive repair: pin M1301, then call frozen
`M1297.M.verify_frozen_authorities()` after the M1301 seal/claim checks and before
`M1297.execute_once()`. No entity, FD, snapshot, candidate, F1–F4, E0–E8,
attempt, or no-retry behavior should change. A fresh different-author hammer is
required afterward.

No M1301 author receipt was read or trusted. No remote, production, checkpoint,
GPU, VCS, DC, PT, or other EDA action was performed. `docs/359` remains at
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
