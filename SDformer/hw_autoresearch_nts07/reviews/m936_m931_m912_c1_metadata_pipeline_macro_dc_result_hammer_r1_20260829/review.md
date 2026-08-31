# M936 — M931/M912 C1 metadata-pipeline macro-aware DC result hammer

## Verdict

**Result integrity PASS 100/100, P0/P1/P2 = 0/0/0; physical admission
FAIL.**  The canonical M931 run is a complete, internally consistent and
recursively sealed Synopsys DC result.  It does not meet the frozen 3 ns setup
gate and is therefore not an admitted timing point, paper-ready PPA result,
speedup result, or system result.

This review was read-only.  It did not run DC, VCS, PT, Formality, another EDA
tool, or a license query, and it did not modify the run, source, predecessors,
or `docs/359`.

## Sealed identity and execution

The canonical run has 32 manifest entries and 32 non-symlink regular files,
with no missing, extra, or linked entry.  Its inner manifest and outer seal
verify.  The frozen M931 contract, M933 one-shot release, M932 source hammer,
M929 functional VCS authority, and consumed attempt marker all verify through
their exact hashes and seals.  `dc.rc` is zero and the Tcl terminal token is
present; this is a successfully completed negative physical measurement, not
a failed tool invocation.

The exact DC V-2023.12-SP3 binary, license identity, standard-cell slow/fast
views, SRAM slow/fast views, macro manifest, RTL, adapter, filelist, SDC and Tcl
all match the source contract.  The mapped Verilog contains exactly nine
`TS1N28HPCPHVTB128X128M4S` instances, matching the pre- and post-compile macro
counts.  Mapped Verilog, mapped SDC, DDC and SVF artifacts are all nonempty.

## Area and timing

At TSMC 28 nm, 3.000 ns, ideal clock and `ZeroWireload`, the design has
85,396 hierarchical cells: 73,030 combinational cells, 12,356 non-macro
sequential cells, and nine SRAM macros.  Logic area excluding macros is
80,149.859039 µm², macro area is 78,825.243164 µm², and total cell area is
158,975.102204 µm².  Net area is undefined under the zero-wireload model.

Setup fails decisively:

- WNS: **−4.9058 ns**
- TNS: **−15,026.33 ns**
- violating paths: **3,128**
- critical data arrival/required: 7.6852/2.7794 ns
- critical logic levels: **511**

All 100 reported setup paths start at `match_bank_q_reg` and end at
`directory_q_reg`; ten endpoint bits each appear ten times under the requested
`-nworst 10 -max_paths 100`.  Thus the metadata pipeline removed the old
accept/commit endpoint class from the worst-100 set, but the 64-row
matcher-to-directory update cone is still the dominant setup bottleneck.

Hold is diagnostic only and also fails: WNS −0.0894 ns, TNS −115.02 ns, and
10,113 violating paths.  Every top-100 hold path runs from `slot0_data_q` to
one of the nine SRAM `D` ports.  No hold-fix command or signoff claim is
present.  There are no max-capacitance, max-transition, or max-fanout
constraint violations.

## Precompile gate and warnings

The precompile `check_timing` report contains zero TIM-209 and zero OPT-150
diagnostics, and both the loop-gate and terminal files record zero.  The four
literal occurrences of each code in `dc.log` are only echoed Tcl expressions
and error-message source text, not tool diagnostics.

Precompile lint reports 268 undriven cells and 13 shorted outputs; postcompile
`check_design` emits no diagnostic.  Eleven VER-318 conversion warnings,
TIM-216 on the reset input, and PWR-428 for unannotated macro outputs remain
nonblocking diagnostics.  PWR-428 reinforces that this result cannot support
a power or energy claim.

## Capacity and claim boundary

The nine bound macros provide 18,432 B of physical capacity for a 9,216 B
logical parent payload.  The same-ledger total storage obligation is 213,376 B,
leaving 194,944 B outside this DC top.  The complete storage system is not
macro-integrated.

M931 may be retained as negative timing/DSE evidence only.  It must not be
reported as setup-admitted, hold-clean, power/energy measured, PPA-ready,
speedup, system speedup, or a paper headline.  It also cannot promote the CPU
same-ledger cycle projection into RTL speedup because no fair zero/bit RTL
baseline or trace bridge is part of this result.
