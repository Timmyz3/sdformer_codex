# M214 independent hammer review

Verdict: **92/100, P0=0**.  M214's only new authorization is the accepted
`upstream_done_accept` fence.  Independent current-source VCS proves that two
unaccepted malformed fences do not create a causal load, while one legal fence
does; the loaded group remains stable for 72 stalled cycles.  Independent
bank-96 and same-edge next-header tests also pass without identity drift.

Fresh recompilation of both M212 and M214 reproduces the 256-case result:
47 improve by one cycle, 209 are unchanged, and none regress.  The independent
M214 recurrence is byte-identical to production with zero mismatch, and both
the model's causal count and the VCS SVA cover are exactly 47.

The frozen H67 r2 opportunity ledger is identity- and arithmetic-consistent:
90,196,785 cycles, saving 191,982 over M212 (1.002128479x), with no stage-0
change.  Its own `RTL exists` and `VCS calibrated` fields remain false by
design, so it must be cited together with the separate exact VCS receipt.

Exact-SHA logic-only DC passes at 3 ns: 20,587.392080 um2, 30,667 cells,
2,773 sequential cells, 82 logic levels, +0.0004 ns setup slack, and no
violated constraints.  This is ideal-clock ZeroWireload pre-macro evidence,
not physical PPA.  The honest admission is therefore an isolated FC2-frontend
control-bubble ablation, not complete FC2, FFN, physical, system, or headline
speedup.

The directories `_superseded_tb_expectation_r1` and
`_superseded_wrong_top_compile_r1` are reviewer setup attempts excluded from
the evidence manifest: the first expected five headers instead of the correct
four, and the second used a wrong top-module name.  Neither is a DUT failure.
