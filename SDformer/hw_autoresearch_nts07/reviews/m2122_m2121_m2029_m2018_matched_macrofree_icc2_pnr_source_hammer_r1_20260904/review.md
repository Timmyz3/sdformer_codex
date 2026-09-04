# M2122 independent source hammer: repaired matched macro-free ICC2 P&R

## Verdict

**FAIL — M2123 is not authorized.** Score: **88/100**; P0/P1/P2 = **1/0/1**.

M2121 repairs most of the M2111 findings correctly. In particular, it now parses real `check_routes` open-net and DRC counts, rejects 999/777 mutations, admits only the exact `routed.spef(.gz)` names, queries accepted mismatches explicitly, pins the executing ICC2/lmutil binaries, exhaustively seals all 1051 Milkyway files, checks 94/94 masters in TT/SS/FF/physical views before NXTGRD, and compares actual DEF/policy evidence across axes. The one-shot ordering, same-UID guard, common normalized SDC, and narrowed macro-free claim are also sound.

Two M2111 requirements remain open, so the contract's all-zero severity release gate is not met.

## P0 — post-link reference completeness is still not directly measured

`unresolved_reference_count` is populated from:

```tcl
get_mismatch_objects -repair_status not_repaired
```

That collection describes logical/physical mismatch objects; it is not the direct post-link reference-object query required by M2111. The source contains no `get_cells -hierarchical -filter "is_unbound == true"` or `is_unmapped == true` census. Consequently, an imported cell may remain unbound or unmapped while all three mismatch-state collections are zero, and the emitted `unresolved_reference_count=0` can still be false as a reference-completeness statement.

The installed ICC2 V-2023.12-SP3 documentation independently defines `is_unbound` as a cell having no reference block and `is_unmapped` as an unmapped cell. The accepted-mismatch repair is valid—the installed `get_mismatch_objects` documentation confirms that accepted objects require an explicit `-repair_status accepted` query—but it does not replace the cell-reference census.

Minimum repair in a fresh source identity: immediately after `link_block`, query and require zero for direct unbound and unmapped cell collections (and record them as separate machine facts); retain the explicit accepted and all-other mismatch-state gates. The parser and negative tests must reject either direct count being nonzero. M2121 must remain immutable and M2123 must not run.

## P2 — generated report inventory and semantic cross-check remain incomplete

The Tcl generates `design_library.rpt`, `final_design.rpt`, and `vectorless_power_diagnostic.rpt`, but the parser does not require any of them. A controlled fixture with all three absent is accepted. A second controlled fixture changed `timing_setup.rpt` to report `slack (VIOLATED) -999.000` while leaving `machine_facts.txt` positive; the parser still accepted the pair.

Minimum repair: require every declared/generated report and cross-check the admission WNS and routed-area/cell census against anchored report text or a separately emitted machine-readable post-route metric report. Add negative tests for a missing report and contradictory setup/hold/area values.

## Permitted checks performed

- Exhaustive double-seal validation: contract, author receipt, M2111 failure review, M2029 input, and physical-technology addendum.
- Exhaustive Milkyway validation: 1051/1051 regular files, including 1044 FRAM and 2 CEL files, all hashes and sorted-path identity exact.
- Installed-tool identity: ICC2 and lmutil are exact regular, non-symlink executables with the contracted SHA-256 values.
- `bash -n`: PASS; parser unit tests: 6/6 PASS.
- Synthetic parser attacks: open=999 rejected, DRC=777 rejected, scenario-only SPEF rejected, no SPEF rejected, accepted/unresolved fact=1 rejected, actual DEF mutation rejected by the existing suite.
- Synthetic parser weakness confirmation: three generated reports may be absent; contradictory timing report text is ignored.
- Source immutability and protected docs/359 SHA verified.

No ICC2, lmutil/lmstat, VCS, DC, PT, GPU, or license query was executed.

## Decision

The mechanical result is fail-closed, and the requested release condition is P0/P1/P2 = 0/0/0. Therefore authorization remains zero and M2123 is prohibited. Repair both findings under a fresh source identity, then obtain a new independent source hammer before consuming any physical-design attempt.
