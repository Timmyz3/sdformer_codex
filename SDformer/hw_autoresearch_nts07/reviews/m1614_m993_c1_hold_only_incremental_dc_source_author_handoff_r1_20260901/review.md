# M1614 — C1 hold-only incremental DC source author handoff

Status: **author static checks PASS; ready for a fresh M1615 different-author
source hammer. No EDA is authorized.**

The additive package contains one Tcl, one fail-closed one-shot runner, a sealed
source contract, and a 12-test static author suite. CPython 3.6 and 3.10 each
pass 12/12; `bash -n`, JSON parsing and `git diff --check` pass. No DC, PT,
Formality, PTPX, VCS, GPU or remote process ran. The M1614 result and attempt
identities remain absent.

The Tcl reads the exact M993 DDC and mapped SDC. It binds the same standard-cell
and SRAM slow/max plus fast/min views, checks the 3 ns clock and nine SRAM
macros, captures exact setup/hold evidence before repair, applies exactly one
`set_fix_hold` and exactly one
`compile -incremental_mapping -only_hold_time`, then captures the complete
post-repair timing, DRC, area, hierarchy, reference, clock and macro evidence.
There is no `compile_ultra`, generic incremental pass, second hold-only pass,
constraint rewrite, timing exception, disabled arc or case analysis.

The runner consumes the attempt before DC and never removes it. Structural or
tool failures become a sealed quarantine. A completed run is positive only if
setup and hold both have WNS >= 0, TNS zero and zero violations, design-rule
violating nets are zero, macros remain exactly nine, the output SDC preserves
the clock/uncertainties and zero exceptions, and area is no more than
`154,608.7116945 um²` (+5% over M993). A timing or area miss is published as a
sealed negative with `retry=false`; it cannot trigger a second optimization.

The runner intentionally stops before attempt/resource/tool work until a sealed
M1615 hammer and separately sealed M1616 release exist and the caller pins both
runner and release SHA. M1615 should independently mutate command count,
constraints, macro/library binding, area and timing predicates, artifact
completeness, attempt ordering and forbidden authority. It must run no EDA and
must not self-create the M1616 release.

Even a future positive DC result remains pending a different-author result
hammer, gate-to-gate and direct-RTL Formality, and inert PrimeTime slow/max plus
fast/min. This source package creates no hold, timing, area, power, energy,
speedup, system or paper claim. docs/359 remains `dedde7ce...`.
