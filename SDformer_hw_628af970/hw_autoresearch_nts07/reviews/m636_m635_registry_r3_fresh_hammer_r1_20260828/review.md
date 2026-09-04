# M636 fresh independent hammer of M635 registry r3

## Verdict

**NO_GO — registry methodology remains unadmitted; no Table-A result or paper headline is admitted.**

- Score: **72/100**
- Severity: **P0=0, P1=2, P2=0**
- Canonical state: 12 sources, 0 direct bundles, 0 eligible Table-A rows,
  headline false, analytical range false.
- Review authority: methodology only.  This review does not admit any
  speedup, energy, PPA, accuracy, full-system or DATE headline.

M635 fixes the four simple M629 attacks: the six-row ladder and three anchors
are code constants; all JSON sources are strict-parsed; M618 cannot be promoted
directly; and M518 now cites the real post-run hammer.  It nevertheless fails
the fresh-review pass boundary for two independent reasons.

## Blocking findings

### M636-P1-01 — frozen test target drifted after the request and contract

The M636 request and M635 contract both bind the unit-test SHA to
`ad13fca130662c1726054b3769109996edf789b473564bae66a7f883c8973f8e` and
declare 14 tests.  The reviewed file is instead SHA
`495ba40363d0cc2edd076438301e7da78b9f5ab15ad7cfa4b9fdd36899869636`
and contains 16 tests.  File timestamps place the modification after the
request.  Builder, config, contract, M518 and docs359 identities match.

The two appended tests are useful, but an unbound post-request improvement is
not the frozen target that the fresh review was asked to authorize.

**Minimal repair:** issue a superseding contract/request that binds the exact
current test SHA and exact test count, then request a fresh review.  Do not
rewrite or reinterpret this sealed review.

### M636-P1-02 — complete five-file bundles are still rootless self-attestation

M635 correctly requires five distinct repo-local files and cross-checks their
hashes, identities, projections, raw aggregate arithmetic, sequence set and
hammer fields.  However, none of those files is rooted in an executable run.
The independent attack created, from scratch:

1. a direct result with arbitrary raw cycles and arbitrary positive
   energy/area/accuracy;
2. a resource manifest whose entire `resource_tuple` was only
   `{"author_declared":"not_measurement_rooted"}`;
3. a completion receipt whose six closure booleans were self-declared true;
4. three invented sequence receipts and a coverage receipt; and
5. a JSON in `reviews/` self-labelling itself an independent PASS hammer.

All hashes and internal projections were consistent.  The builder admitted
6/6 mandatory rows and a **2.0x paper headline**.  This is not merely the
unavoidable fact that measurements can be wrong: the schema does not require
the already-frozen M527 executable configuration manifests, simulator/producer
identity, checkpoint/trace/population manifests, per-sample raw-run receipts,
charge/fallback rules or an independently sealed review identity.  Thus a row
name is still accepted as proof that the corresponding baseline was executed.

This also shows why current test 15 is not sufficient evidence of a healthy
reachable path: it proves that a syntactically complete synthetic bundle is
admitted, including arbitrary measurement values.

**Minimal repair:**

- bind every mandatory row to a distinct SHA-verified M527 executable
  configuration manifest, all referencing one frozen common-resource
  manifest and resolving every resource, charge and fallback field;
- bind direct result/completion/coverage to the exact simulator/producer,
  invocation contract, checkpoint, complete trace, population and aggregation
  manifests, plus SHA-bound per-sample raw-run receipts or logs;
- replace scalar-positive/closure-boolean self-attestation for energy, area,
  STA and accuracy with typed evidence/report manifests and exact projections;
- make the independent hammer bind its request, reviewed target identities,
  its double outer seal and the complete raw-run bundle; a file merely located
  under `reviews/` is not an authority boundary; and
- add a negative test requiring the rootless full-fake five-file construction
  used here to fail.  A reachable positive fixture must carry real measurement
  roots, not only synchronized JSON.

## Checks that passed

- Current CPU suite: **16/16 PASS**.  This is informative but cannot cure
  P1-01 because its SHA is outside the frozen request.
- Canonical builder: `sources=12 bundles=0 table_a_eligible=0
  headline_admitted=false analytical_admitted=false`.
- All 12 registered sources independently rehashed and strict-parsed.
- Exact six mandatory row IDs/order/roles/fidelity and the three headline
  anchors are code-level constants.
- Coordinated deletion, rename, role, fidelity and anchor mutations reject.
- M618 positive values, fake receipt/resource strings and all closure booleans
  remain at zero eligible rows and headline false because external source IDs
  are structurally forbidden.
- Duplicate-key, NaN, Infinity and `1e999` SHA-correct evidence rejects.
- A one-file fake direct bundle rejects; five artifact paths must be distinct
  and lie in their dedicated namespaces.
- Changing aggregate values consistently in result, coverage and hammer still
  rejects when they do not recompute from raw samples.
- Table-B headline and Table-C `ours` mutations reject.
- M518 binds SHA `513c5d...6665`; its real verdict has P0=0/P1=0,
  `rtl_cycle_anchors_admitted=true`, and `issue_cycles_per_tile=17`; the
  registry Table-B value is 17 cycles.

## Mandatory-check disposition

| Check | Result |
|---|---|
| Exact target and docs359 identities | **Fail:** test target drift; others pass |
| Strict 12-source parsing | Pass |
| Code-level ladder/roles/fidelity/anchors | Pass |
| Deletion/rename/role/fidelity/anchor attacks | Pass |
| External M618 promotion | Pass fail-closed |
| Duplicate/nonfinite attacks | Pass |
| Five distinct artifacts and namespaces | Pass syntactically |
| Numeric/identity/closure projection | Pass internally; **fail measurement-root boundary** |
| Raw aggregate/view recomputation | Pass |
| Sequence/density evidence | Pass internally; receipts remain rootless |
| Independent hammer binding | **Fail:** self-declared hammer JSON is sufficient |
| M518 real receipt and 17-cycle anchor | Pass |
| Canonical zero eligible/headline false | Pass |

## Claim boundary

M635 remains a conservative canonical registry and may be used as a repair
prototype.  This review does **not** admit its methodology, any Table-A row,
the analytical 1.79–1.82x range, or a paper headline.  Existing Table-B/C
values retain their already-stated local/external scopes only.

## Execution and protected file

Python 3.6.8, CPU only.  No GPU, EDA, remote job, production simulator, M511
capture or paper-body task was run.  `docs/359` before/after SHA-256 remained
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

Re-review requires both P1 repairs and a newly frozen request.  Only a later
fresh independent review with P0=0/P1=0 may admit the methodology; a
methodology review can never admit measured speedup by itself.
