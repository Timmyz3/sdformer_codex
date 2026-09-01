# M1740 revised author source handoff

M1740 repairs all four P1 findings in the sealed M1737 fail-closed review without rerunning any EDA tool. The future canonical result copies and exact-hash verifies the complete eight-file M1722 Formality proof beside the M1733 `ptsta` payload. It discloses the aggregate setup/hold/min-pulse-width coverage and the `out_setup`/`out_hold` checks, each with one untested `no_paths` endpoint.

`runtime_scope.rpt` must now equal the exact ordered 14-row scope. The exact-SHA M1733 PrimeTime Tcl collapses to 89 logical commands; every command must appear with exact cardinality and in order in the frozen raw log, from `set design_name` through `quit`. Only the two exact pre-main startup `Error:` lines remain admitted, and no unaccounted error is allowed.

The underlying evidence is unchanged: Formality has 16,549 passing compare points and zero failure classes; PrimeTime has setup/hold WNS `+0.027871 ns` / `+0.001827 ns`, zero TNS and zero violating paths at 3 ns. The physical boundary remains prelayout, ideal-clock, `ZeroWireload`, no parasitics, and not paper-PPA-ready.

Python 3.6 and 3.12 each pass 12/12. No M1740 runner, EDA, license query, network action, attempt, result or M1743 release was executed or created. A different-author M1742 review is mandatory.
