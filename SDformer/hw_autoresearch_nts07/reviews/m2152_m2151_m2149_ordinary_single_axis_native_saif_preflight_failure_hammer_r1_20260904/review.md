# M2152 independent M2151 failure hammer

## Verdict

**The read-only failure hammer passes 100/100; review P0/P1/P2 = 0/0/0.**
M2151 itself is a consumed, sealed, noncitable failure with campaign findings
P0/P1/P2 = 2/2/0.  Its VCS compile and ordinary RTL arithmetic complete
successfully, but the native-SAIF acquisition does not establish a measurement
window: VCS explicitly ignores `power -reset`, and the 356-byte raw SAIF has a
whole-simulation duration but no instance or activity record.

This review ran no license query, VCS, `simv`, SAIF acquisition, DC, PT/PTPX,
ICC2, or GPU job.  It does not authorize a retry of M2151 or any direct EDA run.

## Seals, budget, and disposition

The one-member attempt sentinel and eight-member failure quarantine are
exhaustive, symlink-free, and double-sealed.  Their manifest/outer hashes are
`cae90b37...` / `49b1ddbb...` and `c550ea84...` / `90f9413d...`, respectively.
The M2150 source review is also exhaustively double-sealed and bound to the
exact M2149 source identity.

M2151 consumed exactly one license query, one VCS compile, one `simv`, and one
raw SAIF write.  It admitted zero SAIF files and invoked no DC, PT/PTPX, ICC2,
or GPU work.  There is no canonical M2151 result and no live launch lock;
automatic retry is false.  The existing M2151 identity must never be reused.

## What passed

The VCS log compiles the exact single top with six modules and no compile error.
The sole elaborated compute frontend is the previously hammered schedule-mode-0
ordinary axis; there is no TSBG execution axis.  Simulation reaches its normal
terminal report without an assertion failure.  The testbench checks and passes:

- all 228 boundary-state elements known after reset and the 383-cycle preload;
- 20,292 measurement cycles, 149 rows, 1,278 issues, 29,472 signed products,
  24 commits, 1,788 bundles, and 14,304 scalar weight reads/responses; and
- all 24 context/slice accumulators against its independent INT8 scoreboard.

Therefore M2151 is **not** evidence of an RTL topology, scheduling, completion,
or arithmetic failure.  This conclusion remains functional only; it is not an
admitted activity or power result.

## Exact acquisition failure

At the first `$stop`, after the boundary census, UCLI issues `power -reset`.
VCS responds:

> `Warning-[SAIF_REPORT_BEFORE_RESET] Toggle reporting not done`
>
> `This request to reset power information will be ignored.`

The following pass marker says `power_reset_at_first_stop=1`, but that is an
unverified testbench/UCLI intention, not a simulator acknowledgement.  The raw
SAIF confirms rejection: its duration is 62,043.01 ns, exactly 1,167.01 ns of
reset/preload history plus the intended 60,876.00-ns execution window.  It also
has an empty `DESIGN`, zero `INSTANCE` blocks, zero `(T0 ...)` activity records,
and size 356 bytes.  It cannot be annotated or used for power.

There is a second, independent source defect.  The runtime parser requires UCLI
phase 2 (`run_reset_and_preload`) to precede the census and window-begin markers.
In UCLI, however, phase 2 is printed only **after** the first `run` returns from
the testbench `$stop`; the census and begin markers necessarily occur inside
that run.  The parser correctly fails closed on its encoded rule, but the rule
is statically impossible for this command placement.  Even if it were corrected,
the header-only, wrong-duration SAIF would still fail the activity gates.

## Why M2150 missed this

M2150 proved textual command order and parser mutation rejection, but did not
model the VCS invariant that accumulated switching must be reported before a
`power -reset`.  It also mutated synthetic logs consistent with the parser's
impossible phase-2 ordering instead of deriving marker order from UCLI `run` /
SystemVerilog `$stop` semantics.  Finally, the pass token elevates a reset
request to a reset fact without a negative warning gate.  These are source-
review gaps, not reasons to weaken the SAIF parser.

## Only legal additive successor

Authority now is **source authoring only**.  A legal fresh chain is M2160 source
authoring, M2161 independent source hammer, M2162 one ordinary preflight, and
M2163 independent result hammer.  Those names may be changed if already claimed,
but every artifact must use a new identity; M2151 is permanently consumed.

The minimum successor protocol is:

1. Keep the exact single schedule-mode-0 frontend, slot-42 fixture, INT8
   scoreboard, 3-ns clock, census, and frozen completion ledgers.
2. Select the DUT and enable observation before reset/preload.  At the first
   stop, **disable and report prehistory to a separate diagnostic SAIF before
   requesting reset**; reject any `SAIF_REPORT_BEFORE_RESET`, `ignored`, or
   equivalent warning.  Then reset the counters, re-enable if required by VCS,
   run only the 20,292-cycle window, disable, and report a distinct measurement
   SAIF.  The exact enable/disable form must be source-hammered rather than
   assumed; VCS acceptance is proven by warning absence and the final duration.
3. Seal both raw files.  The prehistory file is diagnostic and never annotated.
   The measurement file alone must have duration 60,876 ns, the expected
   93,971 records, TX=0 for every record, exact conservation, and nonzero
   activity in all critical request/response/bridge/commit cones.
4. Place runtime markers according to actual control flow: census and begin
   occur during the first `run`, before the UCLI marker printed after that run;
   end and functional pass occur during the second run, before its return marker.
5. Rename the pass field to `power_reset_requested=1`, or otherwise make reset
   acceptance a parser-derived fact.  A testbench must not attest to UCLI tool
   acceptance it cannot observe.

An alternative native mechanism is permissible only if it observes reset and
preload yet starts the exported interval exactly at the first stop, and its
window semantics are independently demonstrated.  Late enable is not legal:
M2139 already showed the resulting 223 nonzero-TX internal-state records.

Forbidden repairs remain: editing/retrying M2151, relaxing TX/duration/record
gates, rewriting TX, deleting internal/MDA records, masking X, changing RTL
reset semantics, or proceeding to two-axis/DC/PTPX before a fresh ordinary
preflight passes independent review.

## Evidence boundary

M2151 supplies no admitted SAIF, RTL activity comparison, mapped activity,
logic/SRAM power or energy, component/system speedup, paper-ready PPA, or
evidence against TSBG energy.  It establishes only that the ordinary single-axis
RTL workload and scoreboard completed while the acquisition protocol failed.
Protected docs/359 remains exactly `dedde7ce...`.
