# M1484 — M1467 C2 second production failure forensic

## Verdict

M1467 remains `FAILED_OR_INCOMPLETE` and `DO_NOT_CITE`.  This was a read-only
forensic of its sealed one-shot attempt and failure quarantine plus the minimum
three small M1467 private log files needed to identify the first SAIF exit.  No
EDA tool, license query, or retry was run, and the old M1432 private build was
not read.

The exact executed counts are **1 VCS compile, 1 simv run, 0 production SAIF,
and 0 PTPX**.  The canonical result is absent and no partial axis is citable.

## First failure

The K8 mapped compile completed and produced `simv`.  The first simulation
entered the frozen UCLI command:

`power -gate_level all mda sv`

This demonstrates that M1467 crossed M1432's previous observability barrier:
the old `UCLI command without '-debug_access+r'` error is absent.  At simulation
time 0 ps, however, VCS stopped on:

`Error-[LCA_FEATURES_NEED_OPTION]`

The accompanying diagnostic says the `SV-SAIF` flow requires the command-line
option `-lca`.  M1467 added `-debug_access+r` but not `-lca`.  All five assertion
covers therefore show zero attempts, no SAIF file was produced, and PTPX was
never entered.  The sealed failure preserves only that `simv` returned nonzero;
it does not preserve the numeric return code, so M1484 does not invent one.

This is a compile-option/tool-flow failure.  It is not evidence of an RTL,
mapped numeric, protocol, timing, energy, or performance result.

## Disposition

M1467 was a consumed one-shot and must not be edited or retried.  M1484 permits
only one of two next decisions:

1. Author a fresh additive successor whose sole execution change is the
   required `-lca` compile option, while preserving the frozen two-axis,
   five-case mapped campaign and requiring a fresh independent source hammer
   and launch authority before any EDA execution.
2. Declare the production SAIF/PTPX path `NO_GO`.

M1484 itself authorizes no EDA launch.  No M1467 functional, timing, cycle,
speedup, area, power, energy, system, headline, or paper claim is admitted.

`docs/359` remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
