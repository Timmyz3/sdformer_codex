# M2118 independent source hammer: M2117 matched TSBG power campaign

## Verdict

**PASS, 100/100; P0/P1/P2 = 0/0/0.**  This review authorizes exactly one
M2119 campaign: one license query, one VCS compile, two serial `simv`/RTL-SAIF
runs, two fresh DC transformation-map runs, and two serial PrimeTime PX runs.
Any failure consumes the authorization; automatic retry and reuse of old
artifacts are forbidden.  The review itself invoked no license query, VCS,
`simv`, Design Compiler, PrimeTime, or GPU process.

## Launcher P0 is repaired

The M2113 predecessor directly executed the regular `snps_shell` target and
was correctly rejected by M2114.  M2117 instead has exactly one positive DC
launch site whose `argv[0]` is the literal
`/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell` followed by `-f`.  Static AST
inspection found zero launch sites for the resolved `snps_shell` target.

The installed launcher was independently checked at four levels: `dc_shell`
is a symlink; its raw link text is exactly `snps_shell`; it resolves exactly
to the regular, non-symlink target
`/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell`; and that target's SHA256 is
`23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2`.
Static inspection of the frozen wrapper proves that the original `dc_shell`
basename selects `common_shell_exec -shell dc_shell`.  Both the exhaustive
M522 failure receipt and the subsequent positive r6 source review remain
double-sealed and match this repair.

The PrimeTime path is the installed regular, non-symlink
`/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell`.  The same-UID collision guard
covers `vcs`, `simv`, `dc_shell`, `snps_shell`, `common_shell_exec`,
`common_shell_exe`, `pt_shell`, and `lmstat`.

## Exhaustive source and negative audit

All 21 contract inventory members are regular files with exact frozen hashes.
The contract's inner and outer seals and the M2117 selfcheck's exhaustive
directory/outer seals pass.  `docs/359` remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
The prohibited M2115 and fresh M2119 result, attempt, and lock identities are
all absent.  A synthetic missing-review invocation failed before creating any
persistent state or reaching a subprocess.

The fixed workload is global slot 42, sample 0, FC1 layer 28, token 0, G48.
Ordinary and TSBG use the same implementation and differ only in
`SCHEDULE_MODE=0/1`.  Each UCLI window records only the selected DUT
implementation, excluding the testbench and directed SRAM model.  DC starts
fresh native transformation tracking and exports both default and essential
PTPX maps; PT sources them in that order before reading RTL SAIF.

Eighty-five mechanical checks passed.  Synthetic mutations independently
reject nonzero SAIF `TX`, duration drift, `T0+T1+TX` nonconservation, fewer than
20 toggling records, a missing critical cone, an intra-class map conflict,
annotation below 95%, nonzero-toggle coverage below 20%, a critical-report
header-number spoof, and inconsistent power arithmetic.  PT gates annotation,
toggle coverage, inconsistent mappings, eight public critical cones, and
`check_power` before `report_power`.

## Claim boundary

This is source admission, not a power result.  If M2119 completes, its activity
will still be mapped-netlist power driven by transformation-mapped RTL DUT-only
SAIF, not mapped-gate VCS activity.  Standard-cell PTPX excludes the common
294,912-byte weight SRAM; its two read counts may be combined with a foundry
QRT/model only in a separately labelled column.  This campaign cannot establish
hold closure, Fmax, energy per frame, system speedup, or paper-ready PPA.

An exhaustive independent M2120 result hammer remains mandatory before any
M2119 power or energy number is cited.
