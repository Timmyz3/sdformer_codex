# M1594 — M1593 C2 first-fault independent cone review

Decision: **M1593 is a consumed failed diagnostic and is not citable. The
first proven defect is an active-region checker sampling defect, not a proven
mapped reset defect. Do not resynthesize yet.** M1594 ran no VCS, `simv`, DC,
or PTPX and authorizes none.

## Execution and identity

The frozen evidence supports exactly one VCS compile and exactly one simulation:
`compile.log` contains one executed `Command: vcs`, `sim.log` contains one
executed `simv` command, and the one-shot attempt marker is present. The compile
uses the frozen 16-file list and explicit top
`tb_m1578_c2_rtl_vs_mapped_k8_case0_first_fault`; all 16 parse records occur in
that order. There are no compile errors. The three TFIPC warnings are unused
`HA1D0` carry outputs (`CO`), not floating inputs.

The executed commands contain no UCLI, initreg, SAIF, PTPX, or force/release
control. Generated VCS Makefiles can link the normal UCLI runtime library, and
`simv.daidir` contains VCS's `saifNetInfo.db`; neither is an executed UCLI
session or a SAIF/PTPX activity/power run.

## Exact first fault

Cycles 1--5 are clean. At cycle 6 the event interfaces remain idle/equal and
the sole top-level difference is `top_pns=000/X00`: RTL protocol/numeric/stale
is `000`, while mapped protocol error is `X` and mapped numeric/stale are `00`.
Both endpoint-fault vectors are `00000000`; all six named registered
fault/stale taps in both DUTs are `000000`. The run stops fail-closed at 28.5 ns
before any completion, so it is neither an RTL/mapped PASS nor equivalence
evidence.

## Cone finding and root classification

RTL `protocol_error` is intentionally combinational: it ORs registered fault
state with current-cycle consistency and nested illegal-condition terms. In the
mapped netlist, `protocol_error` is driven directly by
`ND3D1BWP35P140/U160335`, not by a fault register. The conservative static cone
walk finds 10756 combinational cells, 2066 state instances, 1033 primary
input bits, and zero undriven leaves. All 2066/2066 observed state D-cones
have a structural path to `rst_core`; that is useful evidence but is not a
proof of reset dominance.

The checker executes `trace_edge()` and all X/fault decisions immediately in
`always @(posedge clk_core)`, with no `#1step`, timeprecision delay, or clocking
input skew. The mapped standard-cell flops and the checker therefore react in
the same time slot, while a large zero-delay combinational cone is still
settling across deltas. Reading an RTL and a gate netlist at that instant is not
a valid stable-edge comparison.

Accordingly the present classification is
`CHECKER_ACTIVE_REGION_SAMPLING_DEFECT_WITH_COMBINATIONAL_X_OBSERVATION`.
Reset/invalid-period isolation remains an **unresolved secondary risk**; M1593
does not prove it is either correct or defective after settling.

## Unique minimum repair

Create a new testbench identity whose only semantic change is one 1 ps
(one-timeprecision) post-posedge settle between the cycle increment and
`trace_edge()`. The trace and every difference/fault/done decision must occur
after that same delay. Keep DUTs, filelist, memories, stimulus, X-aware
comparisons, and stop priority unchanged. Do not use initreg, force/release,
fault masking, or a longer reset as a substitute.

This checker-only repair does **not** require DC or resynthesis. It must receive
a separate source review before any new one-shot compile/simulation. If mapped
`protocol_error` is still X at the stable delayed sample, then repair valid/reset
isolation in RTL and resynthesize before another mapped run; only that second
outcome would justify a DC rerun.

CPython 3.6.8 and 3.10.6 produced byte-identical static-forensic reports. The
review preserves docs/359 at SHA-256 `dedde7ce...` and changes no author file.
