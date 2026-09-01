# M1612 — C1/M993 hold closure first-principles read-only review

Verdict: **one fresh DC hold-only incremental attempt from the frozen M993 DDC
is technically justified, but closure is not yet proven. Use PrimeTime as the
independent validator, not as the primary ECO engine.** M1612 wrote no runner,
ran no EDA, and authorizes no DC, PT, Formality, VCS, GPU, or remote work.

## What is actually open

M993/M1006 is a valid macro-aware setup/area component point: 3.000 ns setup
WNS is `+0.001795 ns`, TNS is zero, there are no setup or design-rule
violations, total cell area is `147,246.392090 um²`, and all nine
`TS1N28HPCPHVTB128X128M4S` macros are present. Its QoR also reports 9,992 hold
violations, rounded worst slack `-0.09 ns`, and rounded total violation
`-120.72 ns`. The implied mean negative slack is only `12.08 ps/path`.

That distribution makes local delay insertion plausible. It does not make the
result automatic: the violation population is large and setup has only 1.795 ps
of margin. Aggregate QoR does not reveal shared fanout or whether the few worst
min paths overlap setup-critical cones. Therefore the correct decision is one
bounded experiment, not a promise that hold will close.

The complete M993, nested original-quarantine and M1006 trees and their outer
seals pass. The frozen mapped SDC retains 3.000 ns, 0.200 ns setup uncertainty,
0.050 ns hold uncertainty, ideal clock and ZeroWireload. It contains zero false
paths, multicycle paths, min/max-delay exceptions, disabled arcs, or case
analysis. M962 ran one `compile_ultra` and deliberately no hold repair; the DDC
is consequently the right immutable starting point rather than a new RTL
resynthesis.

## DC incremental versus PT ECO

DC is the only recommended optimizer here. It owns the mapped DDC, can bind the
same slow/max and fast/min standard-cell and SRAM views, preserve the nine
macros, apply `set_fix_hold` to `core_clk`, execute one
`compile -incremental_mapping -only_hold_time`, and emit a complete new mapped
Verilog/SDC/DDC/SVF identity for equivalence and STA.

Do not precede or follow that command with `compile_ultra -incremental`, a
generic `compile -incremental_mapping`, or another hold-only pass. Project
evidence already shows a generic pass can undo a preceding hold repair, while a
multi-pass flow can spend substantial area and runtime repairing the same min
paths twice. A second pass would also destroy the clean one-variable experiment.

PrimeTime is the stronger independent timing observer, but not the right first
implementation boundary in this repository. There is no sealed PT-ECO flow,
DEF/SPEF, or physical insertion context. `fix_eco_timing` would produce a change
list that still needs implementation and a new netlist identity, while adding
license and Formality ambiguity without improving the current ideal-clock,
ZeroWireload model. Use inert PT slow/max plus fast/min only after DC emits the
new netlist.

## Unique future flow

1. Bind the exact M993 DDC (`d301d6b5...`), mapped SDC (`cf7a0c4a...`), old
   mapped Verilog (`9f96c10a...`), slow/fast cell and macro databases, M935 RTL,
   macro wrapper, M993/M1006 seals, and docs/359. Consume a fresh additive
   result identity; never mutate M993.
2. `read_ddc`, link, restore the same min libraries and exact mapped SDC,
   preserve the 3 ns clock and both uncertainty values, ZeroWireload, ideal
   clock, and nine dont-touch macros. Before optimization, report exact min
   top paths and all min violators because the present `-0.09 ns` is rounded.
3. Apply `set_fix_hold [get_clocks core_clk]` and execute exactly one
   `compile -incremental_mapping -only_hold_time`.
4. Report setup, hold, all violators, QoR, area, clocks, constraints, design
   checks and macro binding; write new mapped Verilog/SDC/DDC/SVF. No exception,
   frequency reduction, uncertainty relaxation, disabled arc, or case analysis
   is permitted.
5. Under separate authorizations, prove M993-mapped to repaired-mapped
   gate-to-gate Formality, then close direct frozen RTL-to-new-netlist Formality
   for complete C1 admission. Independently run inert PT slow/max and fast/min.

## Pass gate and stop gate

Both DC and independent PT must report setup and hold WNS `>= 0`, TNS zero and
zero violating paths under the unchanged constraints. There must be zero
unconstrained diagnostics and timing exceptions, exactly nine SRAM macros, and
zero design-rule/loop/link errors. Formality must pass with zero failing,
unmatched, unverified or aborted points.

The predeclared positive-PPA area cap is `154,608.7116945 um²`, exactly 5% over
M993. A larger hold-closed result may be preserved as engineering evidence but
does not preserve the C1 PPA point. This 5% total-area cap already permits about
10.8% growth in the non-macro portion because the nine macros are fixed.

Any failed timing/formal/macro/constraint gate, area above the cap, tool crash,
or incomplete artifact set stops the attempt. Seal it as negative; do not append
a generic incremental pass, second hold-only pass, manual path surgery, or
exception under the same identity. A new method would require a new independent
source review.

Even a full pass remains a 28-nm pre-layout, ideal-clock, ZeroWireload,
macro-aware component result. It is not routed timing, power, energy, cycle
speedup, system speedup, or paper-ready full-system PPA. docs/359 remains
`dedde7ce...`.
