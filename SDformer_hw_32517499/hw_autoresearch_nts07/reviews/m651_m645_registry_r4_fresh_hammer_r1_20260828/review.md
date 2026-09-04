# M651 fresh independent hammer of M645 rooted Table-A registry r4

## Verdict

**NO_GO — the canonical registry remains safely empty, but the future
measurement-admission methodology is not yet admitted.**

- Score: **79/100**
- Severity: **P0=0, P1=3, P2=1**
- Canonical state: 12 sources, 0 trusted hammer authorities, 0 bundles,
  0 eligible Table-A rows, headline false, analytical range false.
- Authority: methodology only.  This review admits no speedup, energy, PPA,
  accuracy, full-system result, Table-A row or paper headline.

M645 materially repairs M636's immediate fabricated-island problem.  The old
five-file topology, a shape-complete self-signed bundle and config-injected
authority all fail before admission.  Its future graph also reaches the six-row
path under a temporary code-injected trust root, and the raw cycle, energy and
accuracy arithmetic is substantially stronger than M635.  Three contract-level
gaps nevertheless allow a later `headline=true` without all evidence that the
frozen M527 and M636 repairs require.

## Blocking findings

### M651-P1-01 — the frozen M527 headline gate is referenced but not implemented

The bundle carries the literal M527 SHA, but M645 neither runtime-rehashes the
actual M527 contract nor validates its independent headline gates.  In
particular, frozen M527 says the fixed-throughput-numerator receipt gate is
required before any headline and effective GOPS.  M645 has no numerator receipt,
schema or gate, yet its temporary future fixture reaches six eligible rows and
`headline=true`.

The checkpoint has the same weakness: producer, simulator and trace members are
file specs that are rehashed, while the checkpoint is only a repeated SHA
string.  No checkpoint path/file spec is in the bundle, so the claimed frozen
checkpoint need not be present for admission.

**Minimal repair:** add the actual M527 contract and checkpoint as exact rehashed
file specs; add the M527 fixed dense-equivalent/original-useful numerator receipt
bundle, operator convention and population/scope projections; and make all
frozen M527 independent gates executable prerequisites of the headline.  If
Table A intentionally excludes effective GOPS, issue a superseding M527 contract
that explicitly narrows the gate instead of silently bypassing it.

### M651-P1-02 — the double seal authenticates bytes, not the reviewed target

The code-level empty authority map successfully blocks current author files.
However, the future-positive fixture is admitted even though its sealed request
document is only `{"target":"fixture"}`.  The hammer receipt contains the
request outer-seal SHA and bundle hashes, but no exact reviewed builder, config,
test, contract, invocation or bundle identity fields.  The review manifest need
only seal that hammer JSON; no strict request/review schema or reviewed-target
projection is checked.

Thus M645 has not completed M636's requested repair that the independent hammer
bind its request and exact reviewed target identities.  A later source release
could hard-code a byte-valid but semantically empty request/review package.  A
human reviewer might catch that, but the claimed executable authority boundary
does not.

**Minimal repair:** define and strict-parse request and review manifests.  The
request must bind the exact builder/config/tests/contract, M527/checkpoint,
bundle ID and complete evidence-root manifest before measurement review.  The
review must bind that request outer seal, the exact reviewed targets and its
receipt; both outer seals must be code-pinned.  Add a negative test using the
currently accepted `{"target":"fixture"}` request.

### M651-P1-03 — typed PPA scalars are not projected from the raw reports

Energy components and accuracy are recomputed from every raw-run value.  PPA is
different: logic/SRAM/STA reports are only rehashed.  Their contents are never
parsed or compared with `logic_area_mm2`, `sram_macro_area_mm2`, setup/hold WNS
or the total.  The independent probe rewrote a raw logic report to
`area 9999.0`, updated its SHA, left the typed scalar at 0.6 mm2, and the internal
typed-PPA validator accepted it.

The later hammer boolean `typed_receipts_recomputed=true` is useful reviewer
attestation, but it is not an exact numeric projection and the hammer receipt
does not carry recomputed PPA rows separate from the direct-result rows.

**Minimal repair:** introduce machine-readable raw-report extraction receipts
for logic area, SRAM macro area and setup/hold WNS, bind their extractor/tool
identities and compare extracted values exactly with the typed PPA receipt.
Alternatively, make the independent hammer carry and bind explicit recomputed
PPA components and test a coordinated contradictory-report reseal.

## Nonblocking finding

### M651-P2-01 — density vocabulary drifts from frozen M527

M527 requires `low/medium/high`; M645 and its positive fixture require
`low/mid/high` and reject `medium`.  This is small but breaks literal contract
compatibility and can create incompatible aggregation manifests.  Normalize on
M527's vocabulary or supersede it explicitly.

## Checks that passed

- Every frozen M646 target identity, the handoff outer seal and docs359 match.
- Exact target test count is 18; **18/18 pass** under Python 3.6.8 and 3.10.18.
- Canonical runs under both interpreters are exactly
  `sources=12 authorities=0 bundles=0 eligible=0 headline=false analytical=false`.
- The M636 five-file bundle and expanded self-signed authority cannot reach one
  row; config content cannot populate the code trust map.
- Six future rows bind distinct manifests and sources, including
  `c2_exact_typed_k8`, one common resource/charge policy and fully charged
  per-configuration fallback partitions.
- Producer, simulator, invocation SHA, checkpoint string, decoder flag, trace
  members, population and aggregation roots are checked; directed mutations
  reject.
- All 36 temporary raw logs cover exactly 3 samples x 2 views x 6 rows; omitted
  and duplicate runs reject.  Cycles and both view aggregates are recomputed
  without multiplying isolated ratios.
- Logic/SRAM/DRAM energy component swaps and accuracy mutations reject; PPA
  total must equal logic plus SRAM and setup/hold WNS must be nonnegative.
- M618 remains Table B only: a coordinated promotion builds with zero eligible
  rows and headline false.  The 1.79--1.82x analytical range remains false.
- Row role/fidelity, sealed-base anchor, SHA, duplicate-key, NaN/Infinity/1e999,
  symlink and outer-seal mutations all reject.
- The future graph is not dead: a temporary fixture reaches six rows and is
  removed after the test.  This proves structural reachability, not measurement
  legitimacy; P1-01 through P1-03 still block admission.

## Claim boundary

M645 may remain as the canonical fail-closed zero-output registry and as a
repair prototype.  Because P1 is nonzero, this review does **not** admit the
registry methodology, the test fixture, an actual direct bundle, any Table-A
row, system speedup, energy, PPA, accuracy, analytical range or DATE claim.

A superseding source/contract/request must repair all three P1 findings and
receive a new fresh independent hammer with P0=0/P1=0.  Even such a methodology
review cannot admit measurement results by implication; the real decoder-
complete bundle needs its own independently sealed result hammer.

## Execution boundary

CPU/static only.  No GPU, EDA, M511, production simulator, remote job or paper
task was run.  No target was edited.  `docs/359` remained
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
