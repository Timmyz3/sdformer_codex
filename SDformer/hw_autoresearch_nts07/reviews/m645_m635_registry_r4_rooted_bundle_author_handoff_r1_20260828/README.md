# M645 / M635 registry-r4 rooted-bundle author handoff

## Outcome

M645 is an author-side methodology repair for the two M636 P1 findings.  It
does not mutate M635 and admits no Table-A row, speedup, energy, PPA, accuracy,
analytical range or paper headline.

Canonical execution is:

```text
M645_REGISTRY_PASS sources=12 trusted_authorities=0 bundles=0 table_a_eligible=0 headline_admitted=false analytical_admitted=false
```

Both Python 3.6.8 and Python 3.10.18 pass the exact 18-test CPU suite.  A
temporary test-only rooted fixture exercises the complete positive path,
including a transitively sealed hammer receipt; it is deleted after each test
and does not create a production bundle or trust root.

## What changed from M635

M635's five-file bundle could be fabricated as a synchronized island.  M645
requires a rooted evidence graph:

1. the exact frozen M527-r3 contract and common resource tuple;
2. six distinct executable configuration manifests/sources, including the
   real `c2_exact_typed_k8` row rather than a renamed generic candidate;
3. exact unified producer, simulator and frozen invocation identities;
4. the checkpoint plus decoder-complete trace, population and preregistered
   aggregation manifests;
5. every population sample x both views x six rows as SHA-bound raw logs with
   direct cycle counts, separated logic/SRAM/DRAM energy and accuracy values;
6. independently recomputed typed PPA, energy and accuracy receipts;
7. completion/coverage projections; and
8. a code-trusted independently double-sealed hammer that binds the entire raw
   and typed evidence graph.

The code-level hammer-authority map is deliberately empty.  A config cannot
add entries, and a JSON under `reviews/` cannot declare itself independent.
A later real measurement hammer must first exist, then a separately reviewed
source release must pin its exact request/review outer seals and receipt SHA.
This prevents the M636 fabricated 2.0x path while keeping the future admission
procedure explicit.

## Bound targets

- Builder: `system_simulator/scripts/build_m645_h67_paper_metric_registry_r4.py`
- Canonical overlay: `system_simulator/config/m645_h67_paper_metric_registry_r4_20260828.json`
- CPU suite: `system_simulator/tests/test_m645_h67_paper_metric_registry_r4.py`
- Contract: `contracts/m645_h67_paper_metric_registry_r4_contract_r1_20260828.json`
- Sealed base: M635 builder/config at their previously frozen SHA values
- Resource/ladder root: M527-r3 SHA `83ea25e4...55b`
- Protected docs359 SHA: `dedde7ce...fc4`

Exact full hashes are in `SHA256SUMS` and the contract.  The test file is
frozen at 18 test methods; append-after-request edits are not authorized.

## Mandatory fresh-hammer questions

1. Do target SHA values and the exact test count match before and after review?
2. Can the M636 old five-file fabricated bundle still reach Table A?
3. Can a shape-complete synchronized bundle create its own hammer authority?
4. Can config content populate the code-level trust map?
5. Are all six rows tied to distinct M527 configuration identities and the
   same exact resource/charge/fallback policy?
6. Do raw direct cycles and energy/accuracy values cover the complete frozen
   Cartesian population and recompute Table-A rows and aggregates?
7. Are PPA values typed as logic + SRAM macro, setup/hold closed, and bound to
   raw reports rather than positive scalars?
8. Is the authority model too restrictive, dead, circular, or bypassable?
9. Does any current output imply a paper result despite zero trusted authority?

## Claim and execution boundary

This handoff is CPU/static methodology work only.  It did not run GPU, EDA,
M511, a production simulator or remote jobs; it did not modify the paper body
or docs/359.  A fresh independent review with P0=0/P1=0 may admit only the
methodology.  It cannot admit future measurement results by implication.
