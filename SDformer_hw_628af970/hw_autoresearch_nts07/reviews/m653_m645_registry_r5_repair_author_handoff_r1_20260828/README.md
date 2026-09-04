# M653 author handoff: rooted Table-A registry r5 repair

## Outcome

M653 is a methodology-only, fail-closed successor to M645 r4.  It does not
admit a production measurement.  The canonical run remains:

```text
M653_REGISTRY_PASS sources=12 trusted_authorities=0 bundles=0 table_a_eligible=0 headline_admitted=false analytical_admitted=false
```

The test-only future graph reaches six Table-A rows and a 2.0x direct-cycle
fixture solely to prove schema reachability.  It is deleted after each test and
is not paper evidence.

## M651 repairs implemented

1. The exact repo-local M527 r3 contract is runtime rehashed and its three
   independent headline gates are parsed.  Every future bundle must include
   exact rehashed M527 and 591,167,876-byte ep35 checkpoint file specs.
2. A strict `m527_h67_fixed_throughput_numerator_receipt_v1` is required.  It
   projects the frozen population, frame definition, trace/population/weight
   roots, complete operator-scope partition, machine-readable op convention,
   two positive fixed numerators and all six configuration IDs.  The
   independent hammer must recompute the two scalars.
3. Request and review are typed schemas, not arbitrary JSON.  Both bind exact
   builder/config/tests/registry-contract/M527/checkpoint/direct-result/numerator
   SHA256 identities and the complete raw/typed evidence-root map.  The review
   must be `GO` with exactly `P0=0/P1=0`; its outer seal transitively seals the
   typed review and hammer receipt.
4. PPA raw reports are strict numeric inputs.  Logic area/power, SRAM macro
   area/power and setup/hold WNS are parsed and exactly projected into the typed
   receipt; totals recompute.  Directed attacks with raw logic area `9999.0`
   versus typed `0.6`, and SRAM power `77.0` versus typed `0.1`, reject.
5. Density vocabulary is exactly M527 `low/medium/high`; legacy `mid` rejects.

## Files

- `system_simulator/scripts/build_m653_h67_paper_metric_registry_r5.py`
- `system_simulator/config/m653_h67_paper_metric_registry_r5_20260828.json`
- `system_simulator/tests/test_m653_h67_paper_metric_registry_r5.py`
- `contracts/m653_h67_paper_metric_registry_r5_contract_r1_20260828.json`

## Validation

- Python 3.6.8: 14/14 unit/attack tests PASS.
- Python 3.10.18: 14/14 unit/attack tests PASS.
- Canonical CLI: PASS with zero authority/bundle/eligible row and both headline
  and analytical admission false.
- `jq empty`, `py_compile`, and `git diff --check`: PASS.
- M527 SHA: `83ea25e43b53d12800ac64e971069a682e3077411ff10851a7861636ef77355b`.
- checkpoint SHA: `4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158`.
- docs/359 SHA remains
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

## Claim boundary

No GPU, EDA, M511, production simulator, capture, remote job or paper task was
run.  No predecessor, production result or docs/359 was modified.  This author
handoff requests a fresh methodology hammer only; even a GO cannot admit a
future measurement bundle by implication.
