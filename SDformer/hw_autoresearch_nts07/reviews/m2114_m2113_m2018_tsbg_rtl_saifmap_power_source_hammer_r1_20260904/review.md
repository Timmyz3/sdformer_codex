# M2114 independent source hammer: M2113 matched TSBG power campaign

## Verdict

**FAIL, 84/100; P0/P1/P2 = 1/0/0.  M2115 is not authorized.**  This
review invoked no license query, VCS, `simv`, Design Compiler, PrimeTime, or
GPU tool.  The source inventory, one-shot controls, activity gates, mapping
gates, and power parser are otherwise well formed, but the positive Design
Compiler launcher is wrong for this installed Synopsys wrapper.

## P0: direct `snps_shell` cannot select Design Compiler

M2113 changed the positive DC executable from the `dc_shell` launcher symlink
to the resolved regular file:

```text
DC = /opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell
run([DC, "-f", DC_TCL])
```

That is not an equivalent invocation on this installation.  The frozen
`dc_shell` path is a symbolic link with raw link text `snps_shell`; it resolves
to the regular target above, whose SHA256 is
`23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2`.
Static inspection of that target shows `script_name` starts empty and is set to
the original launcher basename only inside the symlink-dereference loop.  The
`dc_shell|dc_shell-t|dc_shell-xg-t` case then selects
`common_shell_exec -shell dc_shell`.  Direct invocation of the regular
`snps_shell` file never captures that basename and reaches the unsupported
default.

This is not hypothetical.  The exhaustive, double-sealed
`m522_m514_dc_tool_invocation_failure_hammer_r1_20260827` records the identical
failure mode and the exact `Error: The  script is not supported.` outcome.  The
subsequent double-sealed `m522_m514_dc_static_hammer_r6_20260827` establishes
the working source pattern: positive `argv[0]` is the literal `dc_shell`
symlink path; the resolved `snps_shell` target is used only for identity/SHA
checking and collision detection.

Consequently, M2113 would consume the one-shot attempt and license budget but
fail before reading its DC Tcl.  No M2115 attempt, lock, result, or power number
exists, and this review deliberately leaves all of them absent.

## What did pass

Ninety-four static and synthetic checks completed.  The 19/19 source inventory
and contract double seal match; `docs/359` remains frozen; the old M2107 and new
M2115 result/attempt/lock identities are all fresh.  After normalizing milestone
names, every parser, testbench, filelist, UCLI, DC Tcl, and PTPX Tcl file is
byte-identical to M2105.  The runner differs only in its explanatory block, DC
and PT paths, and addition of `snps_shell` to the process guard.

The declared execution budget is exactly one license query, one VCS compile,
two serial `simv`/SAIF runs, two serial DC runs, and two serial PTPX runs, with
no retry or old-artifact reuse.  Independent review, freshness, and same-UID
collision checks precede lock/attempt creation and the license query.  A
synthetic missing-review mutation created no result, attempt, or lock.

Parser mutations independently reject nonzero SAIF `TX`, duration drift,
`T0+T1+TX` nonconservation, fewer than 20 toggling records, a missing critical
cone, an intra-class mapping conflict, annotation below 95%, nonzero-toggle
coverage below 20%, a critical-report header-number spoof, and inconsistent
power-component arithmetic.  The common 294,912-byte weight SRAM remains
explicitly outside standard-cell PTPX and exposes only the two read counts for
a separately labelled model.

These passing checks do not mitigate the launcher P0 because no DC netlist or
transformation map can be produced by the current positive command.

## Required additive repair

M2113 is sealed and must remain immutable; M2115 must not be run.  A new source
and result identity must make only this semantic correction:

1. Execute `/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell -f <Tcl>`.
2. Special-case that one launcher symlink in preflight: require `lstat` to show
   a symlink, raw link text exactly `snps_shell`, resolution exactly to the
   frozen regular target, and the resolved-target SHA above.
3. Keep both `dc_shell` and `snps_shell` in the same-UID collision guard.
4. Retain the corrected regular PrimeTime path
   `/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell` and preserve all other
   workload, budget, SAIF, map, PTPX, and SRAM boundaries byte-for-byte.
5. Require a new independent source hammer before any license or EDA call.

No direct `snps_shell` form, including adding a `-shell` argument, is an
acceptable repair because wrapper dispatch happens from the launcher basename.

`docs/359_DATE终局冻结_20260813.md` was not modified and remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
