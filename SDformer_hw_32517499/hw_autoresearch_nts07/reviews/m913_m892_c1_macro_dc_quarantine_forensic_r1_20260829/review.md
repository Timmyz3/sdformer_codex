# M913 — M892 C1 macro-DC quarantine forensic review

## Verdict

`PASS_FORENSIC_REVIEW`, while the M892 artifact remains
`FAILED_OR_INCOMPLETE_DO_NOT_CITE`.  The quarantine is intact and complete enough
for diagnosis, but it is neither a publishable result nor a candidate that may be
promoted to the canonical M892 path.

The wrapper stopped first because its strict log scan saw one DC startup
`Error:`: `env -i` removed `HOME`, and the Design Vision initialization script
tried to read `::env(HOME)`.  That error was survivable: `dc_shell` returned zero,
the Tcl reached its terminal marker, and all requested reports/netlists were
written.  This is a runner false positive, but it is not the scientific root cause
of failure.

The real physical result fails the 3.0-ns point decisively.  Setup WNS/TNS are
-7.05 ns/-73958.98 ns with 12,553 violating paths.  The worst reported path is
`exec_bank_q_reg -> psum_write_valid`, with 9.5968-ns arrival and 2.5500-ns
required time; QoR reports 623 logic levels.  Hold is diagnostic-only and also
fails (-0.08-ns WNS, -121.60-ns TNS, 12,481 paths), led by
`slot0_data_q_reg[0] -> u_parent_scratch/g_slice[0].u_parent_sram` at -0.0799 ns.

## Diagnostic-only physical numbers

- DC total cell area: 156,394.874050 um^2.
- Standard-cell area: 77,569.630886 um^2 (combinational plus
  non-combinational).
- Nine SRAM macro area: 78,825.243164 um^2, or 8,758.360352 um^2 per macro.
- Macro share of DC cell area: about 50.4%.
- Nine `TS1N28HPCPHVTB128X128M4S` instances are present before compile, after
  compile, and in mapped Verilog.  Slow/fast macro views are bound.
- Max capacitance, transition, and fanout reports have no violations.  Power is
  not usable: the log explicitly reports unannotated black-box outputs.

All of these values are quarantine-only diagnostics.  They must not enter a
paper table, an area-efficiency ratio, a speedup claim, or any PPA/energy/system
claim.

## Runner defects, separately from the RTL result

1. The exact M892 `env -i` launch omits the real `HOME`, causing the single
   startup `Error:` that trips the fatal-log regexp.
2. A second deterministic mismatch would be reached after that is repaired:
   the runner requires `status=PASS_M892_RESOLVED_LIBRARY_MACRO_STRUCTURE`, while
   the frozen Tcl writes `status=PASS_M884_RESOLVED_LIBRARY_MACRO_STRUCTURE`.

These are additive-successor defects; the consumed M892 identity must not be
edited or retried.

## M912 feedback

M912 must not be a wrapper-only rerun of unchanged R21.  A publishable 3-ns
successor first needs an RTL/timing repair for the 64-row serial selection and
completion cone feeding `psum_write_valid` (for example a balanced selector or
registered selection/commit boundary), with the added latency reflected in the
same-ledger cycle model and re-proved by Synopsys VCS/SVA.  Only then should a
fresh, one-shot macro DC identity run.

The successor runner must preserve the actual user home in its isolated tool
environment (or otherwise prevent the GUI initializer from observing a missing
`HOME`), retain strict fatal-log checking, and make the terminal/macro-audit
status predicate match its own additive Tcl identity.  It must keep nine-macro
pre/post/netlist gates, TIM-209/OPT-150 gates, setup violation rejection, hold as
diagnostic-only, immutable attempt semantics, and all paper/headline claims
false pending a fresh independent result hammer.

## Integrity

Both the quarantine and consumed-attempt directories pass their inner and outer
SHA-256 seals.  The M892 canonical result is absent.  `docs/359` remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
