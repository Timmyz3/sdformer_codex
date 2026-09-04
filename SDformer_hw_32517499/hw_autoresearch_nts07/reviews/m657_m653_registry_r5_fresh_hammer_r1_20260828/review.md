# M657 fresh independent hammer of M653 Table-A registry r5

## Verdict

**NO_GO — canonical fail-closed behavior is sound, but the future
measurement-admission methodology is not yet admitted.**

- Score: **86/100**
- Severity: **P0=0, P1=2, P2=0**
- Canonical state: 12 sources, 0 trusted authorities, 0 bundles, 0 eligible
  Table-A rows, headline false, analytical range false.
- Authority: methodology review only. This review admits no measurement,
  speedup, effective GOPS, energy, PPA, accuracy, Table-A row or paper claim.

M653 closes most of M651's concrete attack surface. It runtime-rehashes the
actual M527 contract and 591,167,876-byte checkpoint, requires a typed fixed
numerator receipt, semantically binds request/review targets, projects raw PPA
numbers into typed values, and normalizes density to `low/medium/high`.
Resealed wrong-target and evidence-map attacks now fail. Two positive probes,
however, still let a semantically incomplete graph reach six rows and
`headline=true`.

## Blocking findings

### M657-P1-01 — the frozen M527 operator population is not executable

The frozen M527 identity requires ten scope entries: `patch_embed`, `Conv2d`,
`ConvTranspose2d`, `fc1`, `fc2`, `dynamic_BN`, `ATLIF`, `attention`,
`prediction_head`, and `all_required_preprocess_and_completion`.

M653 runtime-rehashes M527, but `_runtime_m527_contract()` does not project
that required scope. `_validate_numerator_receipt()` only requires the
included/excluded partition to equal the author-controlled
`measurement.operator_ids`. The accepted future fixture sets that population
to only `patch`, `conv`, and `decoder`; its numerator includes those same three
items and excludes nothing. It still reaches six eligible rows and
`headline=true`.

This is not a naming-only issue: the executable gate does not prove that FC1,
FC2, dynamic BN, ATLIF, attention, prediction head, and all mandatory
pre/post-processing are present in the numerator population. A later hammer
could truthfully recompute the wrong three-item population and satisfy every
current scalar equality.

**Minimal repair:** parse the exact frozen required scope at runtime and make
the trace and numerator partition cover it. If concrete operator IDs differ
from M527 class names, add a typed, complete class-to-instance mapping and
prove every required class has at least one mapped instance; bind that mapping
into the numerator receipt, request evidence map and independent recomputation.
Add a negative test that removes `fc1` while consistently resealing the graph.

### M657-P1-02 — PPA values are projected, but their tool provenance is absent

The numerical repair works: changing raw logic area from 0.6 to 9999.0 or SRAM
power from 0.1 to 77.0 is rejected even after the PPA receipt is rehashed.
However, the admitted raw-report schema is just strict two-line text. The
positive fixture uses manually written files:

```text
logic_area_mm2 0.6
logic_power_mw 0.2
```

with analogous SRAM and STA files. No raw Synopsys report SHA, extractor source
SHA, extraction command, tool/version, library/corner or unit identity exists
in the PPA receipt. This graph also reaches six rows and `headline=true`.
Therefore M651's requested extractor/tool identity binding remains incomplete:
M653 proves typed-to-text equality, not that the text came from DC/PTPX/STA.

**Minimal repair:** introduce a machine-readable extraction receipt per PPA
row that binds the raw DC/PTPX/STA report SHA values, extractor source SHA,
exact extraction argv, Synopsys tool/version, technology library/corner and
units. Project typed values from those receipts, include all raw and extractor
roots in the request/review evidence map, and add a coordinated handwritten-
text substitution attack. A later human review is still required; the schema
must make the evidence it reviewed explicit.

## Repairs independently verified

- Final M654 request and author handoff manifests and outer seals pass.
- Exact builder/config/tests/contract identities match the sealed request.
- Target tests pass **14/14** under Python 3.6.8 and 3.10.18.
- Canonical CLI under both runtimes reports exactly
  `sources=12 authorities=0 bundles=0 eligible=0 headline=false analytical=false`.
- Runtime M527, checkpoint and docs/359 SHA values match their frozen roots.
- A fully resealed arbitrary `{"target":"fixture"}` request rejects on typed
  schema, rather than only on a stale SHA.
- A fully resealed valid-JSON but wrong registry-contract target rejects on
  exact target path.
- A fully resealed review with P1=1 rejects.
- A fully resealed request with a recomputed but incomplete evidence map rejects.
- Raw logic-area and SRAM-power contradictions reject on numeric projection.
- `mid` rejects; `medium` is the admitted frozen vocabulary.
- Fixed numerator population drift and missing checkpoint file spec reject.
- Disposable positive fixtures are removed; none is retained as production
  evidence.

## Claim and future authority boundary

M653 may remain the canonical zero-output registry and a strong repair
prototype. P1 is nonzero, so this review does **not** authorize adding M657 to
`TRUSTED_HAMMER_AUTHORITIES`, does not authorize a future result by implication,
and does not admit any Table-A or headline number.

A successor must repair both P1 findings and receive a new exact sealed request
and fresh independent review with P0=0/P1=0. Even that would admit methodology
only. The real decoder-complete, multi-sequence production bundle requires its
own independently sealed result hammer before any paper metric becomes valid.

## Execution boundary

CPU/static only. No GPU, EDA, M511, production simulator, remote job or paper
task was run. No reviewed target, predecessor or docs/359 was modified.
`docs/359` remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
