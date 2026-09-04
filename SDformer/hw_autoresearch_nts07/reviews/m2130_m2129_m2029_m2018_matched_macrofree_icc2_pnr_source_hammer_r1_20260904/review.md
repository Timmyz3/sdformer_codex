# M2130 independent source hammer: M2129 matched macro-free ICC2 P&R

## Verdict

**FAIL — M2131 is not authorized.** Score: **91/100**; P0/P1/P2 = **1/0/0**.

M2129 correctly closes both findings from M2122. The Tcl now performs direct post-link `get_cells -hierarchical` queries for `is_unbound`, `is_unmapped`, and `is_black_box`, independently counts black-box reference names, and requires all four counts to be zero. The machine facts, dedicated census report, parser, and fact/report mutations are mutually bound. The parser also requires `design_library.rpt`, `final_design.rpt`, and `vectorless_power_diagnostic.rpt`; it cross-checks setup and hold slacks against real timing reports and the area/cell census against a live-query metric report. Controlled `VIOLATED -999`, contradictory slack, area, and leaf-count fixtures are rejected.

The inherited route, library, physical-equality, tool-identity, Milkyway, one-shot, and SDC gates also pass static and mutation review. However, a new production-blocking mismatch exists at the SPEF producer/consumer boundary.

## P0 — ICC2 cannot produce the literal SPEF name required by the runner

The Tcl invokes:

```tcl
write_parasitics -output "$axis_dir/output/routed.spef" -format spef -corner tt_power
```

The installed ICC2 V-2023.12-SP3 `write_parasitics` command reference states that `-output` supplies a filename prefix; ICC2 appends the parasitic-technology and temperature suffix and then `.spef`. It also emits a separate `.spef_scenario` mapping file. Thus the command above generates a name of the form `routed.spef.<technology>_<temperature>.spef`, not the literal `routed.spef`.

Immediately after each ICC2 process returns, the one-shot runner requires:

```bash
[[ -s "${axis_dir}/output/routed.spef" && ! -L "${axis_dir}/output/routed.spef" ]]
```

The parser likewise correctly permits only `routed.spef` or `routed.spef.gz`. Consequently, a real run would deterministically fail at the runner check, consume the unique attempt, and quarantine the otherwise completed axis. This is not a reason to relax the parser: accepting `routed*.spef*` would recreate the M2111 scenario-file vulnerability.

Minimum repair under a fresh source identity:

1. Write ICC2 parasitics to a dedicated raw prefix.
2. After ICC2 returns, require exactly one regular, nonsymlink corner SPEF for `tt_power`; explicitly exclude the scenario map.
3. Atomically canonicalize that file to `output/routed.spef`.
4. Preserve the current strict parser and add a static/fixture test proving the producer-to-canonical-name path.

## Verified repairs and regressions

- Direct unbound, unmapped, black-box, and black-box-reference census: PASS.
- Separate accepted/nonaccepted mismatch gates: PASS.
- Missing design-library/final-design/vectorless-power reports: rejected.
- Setup/hold report-to-fact cross-check and `VIOLATED -999` attack: rejected.
- Live-query area and cell-census cross-check: PASS; contradictory fixtures rejected.
- Route open=999 and DRC=777 attacks: rejected.
- Scenario-only and deleted-real-SPEF parser attacks: rejected.
- TT/SS/FF/physical coverage: 94/94 each before NXTGRD.
- Actual DEF die/pin and routing/CTS/hold/scenario equality: bound across axes.
- ICC2 and `lmutil lmstat` path, kind, executable bit, and SHA: exact.
- Milkyway inventory: 1051/1051 files, including 1044 FRAM and 2 CEL, exact hashes.
- One-shot ordering, no retry, same-UID guard, and SDC removal cardinalities: PASS.
- M2121/M2122 sealed evidence and source identities: unchanged.
- Protected docs/359 SHA: `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

The independent checker performed 97 gates: 96 passed and the intentional producer/consumer compatibility gate failed. Parser tests passed 10/10. No ICC2, `lmutil/lmstat`, VCS, DC, PT, Formality, GPU, or license query was executed.

## Decision

The release contract requires at least 95/100 and P0/P1/P2 = 0/0/0. M2130 has one deterministic P0, so authorization remains zero and M2131 is prohibited. Keep M2129/M2130 immutable, repair canonical raw-SPEF handling under a new identity, and obtain a new independent source hammer before consuming any physical-design attempt.
