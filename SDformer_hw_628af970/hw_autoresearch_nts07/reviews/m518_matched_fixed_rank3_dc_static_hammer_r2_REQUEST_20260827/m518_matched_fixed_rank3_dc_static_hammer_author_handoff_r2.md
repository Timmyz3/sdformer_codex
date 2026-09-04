# M518 matched Fixed-T10 versus rank3 DC static hammer r2 — author handoff

Date: 2026-08-27  
Execution boundary: source-only. Do not run DC, VCS, Formality, PT, PTPX, or open-source EDA.

## r1 disposition

The sealed r1 review scored 94/100 with P0/P1/P2 = 0/1/2 and status
`NEEDS_REVISION__R1_LAUNCH_NOT_AUTHORIZED`. r1 is permanently ineligible for a
launch admission. Its sole P1 was that the precompile hard gate counted
TIM-209/OPT-150 only in `check_timing_precompile.rpt`, after earlier build/link
operations had already occurred outside the scanned transcript.

## Narrow r2 repair

r2 redirects the complete precompile construction into
`precompile_build.rpt`: analyze with `SYNTHESIS`, elaborate, current-design
selection, link, uniquify, slow/fast min-library setup, operating condition,
common SDC, ZeroWireload and fix-hold. It then appends precompile
`check_design` and `check_timing` text to the same in-memory audit string.

Both TIM-209 and OPT-150 are counted across all three sources. Any nonzero
count writes `TCL_EXPLICIT_FAILURE.txt` and executes process `exit 36` before
the only branch containing `ungroup` or any compile command. The shell runner
also requires all three files, the exact provenance line, and zero diagnostic
tokens in each individual source.

## Frozen r2 identities

- Contract SHA256:
  `18ae1c4fc48e421720ea41ffeb76528c2efe56264d3d3eaf5affda4ba364860d`
- Runner SHA256:
  `05ada3ea4e2b653262f2693602eab83c3cc75ea7af35fc4e501f9da2a481147e`
- Tcl SHA256:
  `2bb5cfda31fd04b8ec796c253b22db732413a9796549220b4d0a0f0f86735fe5`
- Unchanged two-source filelist SHA256:
  `bd4454fdb4c86c5ead9e56bf61447dc637916b5258ab5ad8382499a3dfba6b00`
- Unchanged SDC SHA256:
  `73030f70b27909c1f8100bbc02af75c77fed246908027980912afd6499beb6e3`

All r1-approved 50-port, same-scope, cycle-denominator, new-area-only,
resource, structural and zero-macro logic-only boundaries remain unchanged.
The author performed source syntax, JSON, ordering and seal checks only; no EDA
command was invoked. A different independent reviewer must close P1 before a
double-sealed r2 launch admission may exist.
