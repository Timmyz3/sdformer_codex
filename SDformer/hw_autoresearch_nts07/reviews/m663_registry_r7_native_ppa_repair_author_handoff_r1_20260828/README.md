# M663 author handoff: native Synopsys PPA registry r7 repair

## Canonical outcome

M663 remains methodology-only.  Its canonical output is:

```text
M663_REGISTRY_PASS sources=12 trusted_authorities=0 bundles=0 table_a_eligible=0 headline_admitted=false analytical_admitted=false
```

No production result, Table-A row, analytical range or paper headline is
admitted.  The disposable six-row fixture proves only that the repaired graph
is reachable when every required root is supplied.

## M661-P1-01 repair: consume native reports, not author wrappers

The r7 extractor directly parses the native report grammars used by the
repository:

- Synopsys DC `report_area`, including Report, Design, Version, libraries and
  total cell area;
- PrimeTime `report_timing` for both max/setup and min/hold paths, including
  Design, Version, delay type and worst slack;
- PrimeTime PX `Averaged Power`, including the memory row and the internal,
  switching, leakage and total summaries;
- a TSMC memory-compiler `.ds`, including compiler version, macro/library
  identity, PVT, geometry and diagnostic read/write/leakage currents.

An extraction is driven by a typed run manifest and five native report
classes.  It is not legal to substitute a compact numeric summary.  The
directed three-line-wrapper attack now rejects.  Real repository DC,
PrimeTime and PTPX reports are parsed in the dual-runtime test suite; the
synthetic native excerpts used for six-row reachability are disposable tests,
not retained evidence.

## M661-P1-02 repair: bind report identity to row, operator and run

Each row is bound to its M527 configuration ID, configuration-manifest SHA,
exact ordered ten-operator scope SHA, deterministic design name, macro name
and deterministic run ID.  The run ID commits to the row/configuration and all
five native report SHA256 values.  All four logic reports must independently
parse the expected Design, and the SRAM report must parse the expected macro.

The extraction receipt, run manifest and all five reports are explicit
evidence roots for every row.  A complete six-row fixture therefore exposes
43 rooted objects: one extractor, six receipts, six run manifests and 30
native reports.  Consistently rehashed wrong-design, cross-row report reuse,
macro-identity drift and report omission attacks reject.

## M661-P2-01 repair: exact power semantics

Every logic, SRAM and chip-total power row names all five quantities in mW:
internal, switching, dynamic, leakage and total.  The extractor checks
`dynamic = internal + switching` and `total = dynamic + leakage`; the builder
checks the typed logic/SRAM/chip composition again.  Missing leakage, total
arithmetic drift and typed-value drift fail closed.  Memory-compiler currents
remain diagnostics; SRAM operating power comes from the PTPX memory group.

## Reviewed files and SHA256

- `system_simulator/scripts/extract_m663_native_synopsys_ppa_reports.py` —
  `2a7456d8fe0c6336f094c857cb37c9d54a48425f77f5c0fd914c34436f0733a4`
- `system_simulator/scripts/build_m663_h67_paper_metric_registry_r7.py` —
  `19f436f05937845805ddd08ce4989e33cef7f59b7be772a7214a9f4b9b357279`
- `system_simulator/tests/test_m663_h67_paper_metric_registry_r7.py` —
  `d9dc8130b79ccfe60fd99e74d6d6bf721b439b858c2254f36d76654529bb5358`
- `system_simulator/config/m663_h67_paper_metric_registry_r7_20260828.json` —
  `e404326ee531a62a8fb27e26159d5a588652da9e2bbd760aadc809bc0d5fc662`
- `contracts/m663_h67_paper_metric_registry_r7_contract_r1_20260828.json` —
  `6ca6abc1d4d61ecb7b3f8e36ea782e05d7a7d3bc3890ec61c6aa469a8d14066d`

## Author validation and boundary

- Python 3.6.8: 18/18 PASS.
- Python 3.10.18: 18/18 PASS.
- Canonical CLI on both interpreters, `py_compile`, `jq empty`, native-report
  direct parsing and static checks: PASS.
- No GPU, EDA, M511, capture, production simulator or remote job was run.
- No predecessor was modified.
- docs/359 remains
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

M664 requests a fresh independent methodology hammer.  A methodology GO
requires P0=0 and P1=0, and would not admit a production metric by implication.
