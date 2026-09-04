# M477 M476r2 pre-macro DC diagnostic

The exact Synopsys DC run completed, but the runner correctly refused admission.  The mapped design measured 42,370.649130 um2 (41,849 cells; 5,508 sequential; 36,340 combinational) at the 3.000 ns setup target.  The timing summaries are positive, but `constraint_violators.rpt` contains one max-transition group at -0.2161 total slack, two max-capacitance violations (-0.0363 total slack), and three max-fanout violations (-102 total fanout slack).

This is a diagnostic cost result, not a paper PPA result.  It shows that the dual 1152-bit response slots plus hold repair cost about 13.5% more area than the M475 baseline and motivates a macro-output-register or one-slot/skid-buffer DSE.  No performance, power, energy, system-speedup, Formality, or DATE-headline claim is admitted.
