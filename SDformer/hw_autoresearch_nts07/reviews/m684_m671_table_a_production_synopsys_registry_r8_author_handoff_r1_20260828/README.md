# M684｜M671 Table-A production Synopsys registry r8 author handoff

## Author verdict

`AUTHOR_GO_FOR_FRESH_INDEPENDENT_METHODOLOGY_HAMMER`, self-score `96/100`.
This handoff closes the author implementation of the future production gate; it
does **not** admit a production row.  Canonical state remains exactly
`authority=0 / bundle=0 / Table-A eligible=0 / headline=false / analytical=false`.

## What changed

- Requires one exact native VCS production PASS and one native Formality
  `Verification SUCCEEDED`, in addition to native DC area/environment, PT
  setup/hold, PTPX power/environment and three memory-compiler reports.
- Roots design RTL, testbench, assertions, mapped netlist, SDC, SAIF, all six
  logic/SRAM DB files, five exact tool executable snapshots and versions,
  per-step argv/scripts/logs/exit status and every output report.
- Treats the mapped netlist as a DC output, SAIF as a VCS output, and checks the
  corresponding per-step input/output roots in both command scripts and logs.
- Projects three exact compiled macro organizations to the M527 resource:
  128 KiB weight + 96 KiB state + 16 KiB parent scratch = exactly 245760 B.
  The parent scratch is `1R1W`, matching M527; the partial `1RW` identity was
  internally inconsistent and was corrected before release.
- Rejects nonpositive logic/macro area, zero integrated SRAM PTPX power,
  negative component power, and negative setup/hold WNS.  The exact M527
  configuration SHA, full ten-operator scope and full design name continue to
  prevent selected-slice or pre-macro substitution in the registry path.
- Standalone extraction rejects absolute, dot/dot-dot, repeated-separator,
  backslash, repository-escaping and symlinked paths before resolution.

## Validation

- Python 3.6.8 compile: PASS.
- Python 3.6.8 author suite: `8/8 PASS`.
- Canonical CLI: `M671_REGISTRY_PASS sources=12 trusted_authorities=0 bundles=0 table_a_eligible=0 headline_admitted=false analytical_admitted=false`.
- No EDA/GPU/remote run was launched.  Synthetic files are grammar fixtures,
  are removed by the tests, and are explicitly non-authoritative.
- Protected docs/359 SHA remains
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

## Review boundary

The fresh hammer must independently attack consistently rehashed macro,
corner, DB, netlist, SAIF, VCS/Formality, argv/log and selected-slice
substitutions.  No production admission is permitted from this author review.

