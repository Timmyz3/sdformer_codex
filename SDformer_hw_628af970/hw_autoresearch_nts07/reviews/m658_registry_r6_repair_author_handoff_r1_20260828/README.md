# M658 author handoff: Table-A registry r6 repair

## Canonical outcome

M658 is methodology-only.  Its canonical output is still:

```text
M658_REGISTRY_PASS sources=12 trusted_authorities=0 bundles=0 table_a_eligible=0 headline_admitted=false analytical_admitted=false
```

No production measurement or paper metric is admitted.  The disposable
positive fixture reaches six rows only to prove that the tightened graph is
structurally reachable.

## M657-P1-01 repair: exact frozen M527 scope

- The runtime-rehashed M527 r3 contract supplies the required ordered ten-item
  scope: patch embed, Conv2d, ConvTranspose2d, FC1, FC2, dynamic BN, ATLIF,
  attention, prediction head, and all required preprocess/completion.
- `measurement.operator_ids` must equal that exact ordered list.
- `complete_trace_manifest.operator_scope` must independently equal it.
- The fixed numerator included/excluded partition must exactly cover it.
- The scope digest and trace-manifest SHA are explicit request/review evidence
  roots in addition to the transitive measurement/numerator roots.
- Directed fixtures removing `fc1` independently from trace, measurement, or
  numerator all reject.

## M657-P1-02 repair: rooted Synopsys PPA provenance

Every one of the six PPA rows now has a strict extraction receipt binding:

- raw DC area, PTPX power, PrimeTime STA and SRAM compiler report file specs;
- exact reviewed extractor source SHA and deterministic extraction argv;
- tool name/version parsed from every report;
- library and operating-corner identities parsed from every report, plus a
  library-identity digest;
- explicit mm2, mW and ns units;
- extracted logic/SRAM area and power plus setup/hold WNS.

The registry reruns the exact extractor on the SHA-bound reports and requires
the extraction receipt and typed PPA row to match it exactly.  All 24 raw
reports, six extraction receipts and the extractor source are explicit hammer
evidence roots.  A three-line handwritten numeric summary, tool version drift,
argv drift, library/corner drift and extracted-value drift all reject.

The positive fixture contains synthetic Synopsys-format text solely for parser
reachability.  It is not retained, not a real EDA result and cannot populate
the canonical registry because the code-trust authority map is empty.

## Reviewed files

- `system_simulator/scripts/build_m658_h67_paper_metric_registry_r6.py`
- `system_simulator/scripts/extract_m658_synopsys_ppa_reports.py`
- `system_simulator/tests/test_m658_h67_paper_metric_registry_r6.py`
- `system_simulator/config/m658_h67_paper_metric_registry_r6_20260828.json`
- `contracts/m658_h67_paper_metric_registry_r6_contract_r1_20260828.json`

## Validation and boundary

- Python 3.6.8: 15/15 PASS.
- Python 3.10.18: 15/15 PASS.
- Canonical CLI, `py_compile`, `jq empty`, and `git diff --check`: PASS.
- No GPU, EDA, M511, capture, production simulator or remote job was run.
- No predecessor was modified.
- docs/359 remains
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

A fresh independent methodology hammer with P0=0/P1=0 is required.  A GO
would not admit any later production result by implication.
