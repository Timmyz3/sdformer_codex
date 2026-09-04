# M518 matched Fixed-T10 versus rank3 DC static hammer — author handoff

Date: 2026-08-27  
Execution boundary: source-only. Do not run DC, VCS, Formality, PT, PTPX, or an open-source EDA tool.

## Purpose

This package closes the comparison defect identified by M302: Fixed-T10 and
rank3 now have the same synthesis-visible 50-port protocol boundary and are
prepared for one same-corner, same-SDC, same-Tcl, same-source-corpus logic-only
DC campaign. The package does not authorize that campaign.

## Frozen subjects

- Contract: `contracts/m518_matched_fixed_rank3_logic_only_dc_contract_r1_20260827.json`
  (`cb35b98d42fcca5801e41f0e3c7dfa8233eeae2a51f439d74556ed7d4639fb48`)
- Exact-SHA runner: `dc_handoff/scripts/run_dc_m518_matched_fixed_rank3_logic_only_exact_sha.sh`
  (`0ef552f1d557f6dfbcdf320327a403b7e277bf8666644924ba9df2fad954e157`)
- Common Tcl: `dc_handoff/scripts/run_dc_m518_matched_fixed_rank3_logic_only.tcl`
  (`ca4ece27943a2f18773ad5c6df6c31938f2757a89b846425c216122f6d751bbd`)
- Common two-source filelist:
  `dc_handoff/filelists/date_m518_matched_fixed_rank3_logic_only_dc.f`
  (`bd4454fdb4c86c5ead9e56bf61447dc637916b5258ab5ad8382499a3dfba6b00`)
- Common 3.000 ns/fanout-24 SDC:
  `dc_handoff/constraints/date_m289_m273r2_logic_only_3ns_fanout24.sdc`
  (`73030f70b27909c1f8100bbc02af75c77fed246908027980912afd6499beb6e3`)

## Fair denominator

The only permitted area denominator is the new Fixed area from this exact
two-point campaign. The old M289 rank3 area is historical diagnostic evidence,
not a denominator. Cycle anchors are independently sealed VCS observations:
Fixed N1/N4 = 29/80 cycles and rank3 N1/N4 = 24/39 cycles. Primary
throughput/mm² uses N4, and both N1 and N4 must be disclosed. No steady-state
extrapolation is admitted.

## Physical and claim boundary

Both points are flattened TSMC28 standard-cell logic, 3.000 ns, ideal clock,
ZeroWireload and zero macros. The runner requires clean setup, hold,
max-capacitance, max-transition, max-fanout, unconstrained-endpoint, latch,
multi-driver, unresolved-reference and black-box audits. Even a clean run is
not macro-inclusive PPA, power, energy, trained-rank3 accuracy, system speedup,
paper-ready PPA, or a headline until a separate independent receipt review.

The author ran only Bash/JSON/Python source checks. No EDA command was invoked.
