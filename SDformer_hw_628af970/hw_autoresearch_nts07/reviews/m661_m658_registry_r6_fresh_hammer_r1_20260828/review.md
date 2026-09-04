# M661 fresh independent hammer of M658 Table-A registry r6

## Verdict

**NO_GO — canonical fail-closed behavior and the ten-operator scope repair pass,
but future PPA admission is not yet safe.**

- Score: **84/100**
- Severity: **P0=0, P1=2, P2=1**
- Canonical state: 12 sources, 0 trusted authorities, 0 bundles, 0 eligible
  Table-A rows, headline false, analytical range false.
- This is a methodology review only. It admits no production result, PPA,
  energy, speedup, Table-A row or paper claim.

M658 correctly closes the two concrete r5 scope/evidence-map defects: the
runtime-rehashed M527 scope is now the exact ordered ten-item population, and
the PPA graph explicitly carries 24 report roots, six extraction receipts and
the reviewed extractor source. All requested stale-field and missing-root
attacks reject. The remaining blockers are deeper semantic provenance gaps.

## Verified repairs

- The required ordered scope is exactly `patch_embed`, `Conv2d`,
  `ConvTranspose2d`, `fc1`, `fc2`, `dynamic_BN`, `ATLIF`, `attention`,
  `prediction_head`, and `all_required_preprocess_and_completion`.
- Measurement `operator_ids`, complete-trace `operator_scope`, and the fixed
  numerator partition project that exact scope.
- Independently deleting `fc1` from trace, measurement, or numerator rejects.
- A disposable positive graph contains six PPA rows, 24 raw-report roots, six
  extraction receipts and 31 PPA provenance evidence roots.
- A three-line numeric report, receipt-only tool-version drift, extractor or
  argv drift, library/corner/unit drift, raw or extracted value drift, a wrong
  reviewed target, and requests omitting either a raw report or extraction
  receipt all reject.
- Both Python runtimes pass 15/15 target tests and the independent harness.
- Canonical CLI under both runtimes remains exactly zero-authority,
  zero-bundle, zero-row and headline false.

## Blocking findings

### M661-P1-01 — the extractor does not consume native tool reports

The positive fixture's four report classes are 9–10-line hand-authored text
documents with a Synopsys-looking delimiter and custom fields such as
`Tool : dc_shell` and `Total cell area (um2): ...`. That is sufficient for
parser reachability, but it is not the native output grammar.

As a direct compatibility probe, the sealed repo-native DC `report_area`
artifact at
`dc_handoff/runs/m62_p48_dc_3p000ns_r1b_20260823/reports/area.rpt`
(SHA256 `0eae6b6d23ff9816b01cffed3a1a70e33713d58243673c8ea597e514decb72c3`)
is rejected with `Tool header must occur exactly once`. The native report has
the normal `Report`, `Design`, `Version`, `Date`, `Library(s) Used`, and
`Total cell area` sections; it does not contain M658's custom wrapper fields.

Therefore a real DC/PT/PTPX/memory run cannot enter r6 without a manual
normalization/wrapper step whose producer source, argv and input/output roots
are not modeled. Calling the accepted 9–10-line documents "raw Synopsys
reports" would overstate the evidence.

**Minimal repair:** either parse the native report formats directly, or define
a typed normalizer receipt that binds native input SHA values, exact normalizer
source/argv, normalized output SHA values, tool transcript/run manifest, and
return code. Add a positive test using immutable native reports from the
repo-local Synopsys runs.

### M661-P1-02 — report identity is not bound to its configuration row

The extractor parses tool, version, library, corner and six numbers, but it
does not parse `Design` or `Macro` identity. The extraction receipt contains a
row ID and configuration-manifest SHA, yet that association is author-declared
rather than derived from the report.

Two independent probes reach the row provenance validator:

1. Replace the DC `Design` label with `wrong_configuration`, rehash the report
   and extraction receipt, and validate it under the original row: accepted.
2. Build the row-1 extraction receipt from row-0's four raw reports while
   claiming row-1's configuration SHA: accepted.

These probes do not install a production authority or admit the full graph,
but they prove the machine-checked row gate does not prevent cross-row report
reuse. An independent reviewer might notice it manually; the claimed exact
configuration binding does not enforce it.

**Minimal repair:** parse design/macro/top identity from every native report;
bind each identity to a typed configuration-to-design manifest; require the
four reports in a row to agree on the relevant design/run ID; reject report
reuse across distinct configurations unless an explicit same-netlist alias
contract proves equivalence.

## Caveat

### M661-P2-01 — `total_power_mw` is dynamic-only

The extractor exposes `logic_power_mw` and `sram_macro_power_mw` from fields
named dynamic power. It has no leakage projection, yet the typed row sums them
as `total_power_mw`. Until leakage is extracted and charged, this field must be
labeled dynamic power rather than total chip power.

## Claim and authority boundary

M658 remains a useful zero-output registry prototype and its exact scope repair
is sound. P1 is nonzero, so this review does not authorize adding M661 to any
trusted authority map. It admits neither the disposable six-row fixture nor
any later result by implication. A successor needs a new sealed source/contract
request and a fresh P0=0/P1=0 review; a real decoder-complete production bundle
still needs a separate result hammer.

## Execution boundary

CPU/static only. No GPU, EDA, M511, capture, production simulator, remote job
or paper task was run. No reviewed target, predecessor, canonical registry or
docs/359 was modified. `docs/359` remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
