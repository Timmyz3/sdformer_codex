# M97 M85 logic-only Synopsys post-run independent hammer

Date: 2026-08-24

Status: `CONDITIONAL_GO_SAME_FLOW_CELL_AREA_DENOMINATOR_NO_GO_TIMING_OR_PPA`

This review is read-only with respect to the production RTL, DC scripts, and
the M97 run directory.  It independently checks the frozen evidence and writes
only this review directory.

## Verdict first

The receipt field
`usable_as_same_flow_diagnostic_area_denominator=true` is defensible **only as
a qualified cell-area denominator** for an exact M85-versus-M99 logic-island
comparison.  It is not an unconditional baseline.

The exact permitted denominator is:

```text
34,758.486144 um^2 mapped standard-cell area
TSMC28 HPC+ ssg0p9v125c, DC V-2023.12-SP3
3.000 ns constraint, ideal clock, ZeroWireload, zero macros
M82 + M85 logic island, with the documented bank-data port cut
```

It may not be called total area, a 3 ns timing baseline, a complete PWP
frontend, macro-inclusive PPA, energy, Fmax, or paper PPA.  The setup constraint
fails by 1.1583 ns, net area is undefined, the SRAM path is cut, and the wrapper
did not seal.

## Evidence integrity and completion state

Running the run's `evidence_manifest.sha256` from the repository root verifies
all 18 listed files.  The manifest pins the receipt, admission, DC log, port-cut
boundary, negative completion marker, key reports, mapped Verilog/SDC/DDC, and
resource/reference reports.

The backend and wrapper outcomes must remain separate:

- `dc.log` identifies Design Compiler `V-2023.12-SP3` and contains no line
  beginning `Error:`.
- Analyze, elaborate, link, all three compile passes, timing update, reports,
  mapped Verilog, mapped SDC, DDC, and SVF close all execute.
- The log ends with normal memory/CPU statistics and Synopsys `Thank you...`.
- The mapped outputs are nonempty and frozen by the evidence manifest.
- The launch wrapper returned code 2 after `dc_shell` returned because the
  pre-lock launcher reached an unmatched-quote shell parse error.  It did not
  create `RUN_COMPLETE.txt`; the negative marker correctly says backend
  complete but run not admitted.

Therefore `backend_completed=true` and `wrapper_completed=false` are mutually
consistent.  Neither should be rewritten to make the other disappear.

## Launch provenance

The observed run began with pre-lock launcher SHA:

```text
3720b19e55dbe0c81bc45ff941ff6b563a3878f5f258c751c6a1ef76280a8540
```

The later hardened launcher currently on disk has SHA:

```text
670dbe8b40f35f2d5ceaa536141c440179acc7ae5a05e2b679910fe97f8e6fc0
```

The hardened launcher adds the stronger backend guard, atomic lock, identity
checks, and repaired post-run shell.  It cannot be attributed retroactively to
M97.  The receipt and preflight review correctly preserve this distinction.

The old launcher did not self-attest every filelist/SDC/Tcl/DB hash in its own
admission file.  The receipt's `semantic_inputs_unchanged_during_run=true`
therefore depends on the contemporaneous independent-review observation, not
on a complete launch-time manifest emitted by the old wrapper.  This is
adequate for a diagnostic data point after the present post-run cross-check,
but not equivalent to a hardened wrapper-sealed pass.  A rerun is still needed
if a sealed-run claim is desired.

## Independent result reconstruction

### Area and cells

The area and QoR reports reconcile exactly:

| Quantity | Independently read value |
|---|---:|
| combinational cells | 31,711 |
| sequential cells | 3,108 |
| leaf cells | 34,819 |
| hierarchy cells | 1 |
| area-report cells | 34,820 |
| combinational area | 28,492.254043 um^2 |
| noncombinational area | 6,266.232101 um^2 |
| total cell area | 34,758.486144 um^2 |
| macro/black-box count and area | 0 / 0.000000 um^2 |
| net area | undefined, ZeroWireload |
| physical total area | undefined |

`28,492.254043 + 6,266.232101 = 34,758.486144`, and
`31,711 + 3,108 = 34,819`.  The receipt's 34,820 number is the area report's
cell count including the one hierarchical cell; it must not be confused with
the QoR leaf-cell count.

### Timing and constraints

The frozen 3 ns logic-only point does not meet setup:

- worst setup path: input `phase_metadata[3]` to
  `phase_poison_q_reg/D`;
- data arrival: 3.9579 ns;
- required time: 2.7996 ns after 0.200 ns setup uncertainty;
- setup WNS/TNS: -1.1583/-1.1583 ns;
- violating setup paths: 1;
- QoR critical path length: 3.71 ns, 198 logic levels;
- hold: `MET`, no hold violation, worst displayed slack 0.0000 ns after
  0.050 ns uncertainty;
- max transition, capacitance, and reported fanout violations: zero.

The hold result is technically MET but rounded to zero at four decimal places;
it is not a comfortable hold margin.  This further argues against treating the
run as paper timing.

`check_timing_postcompile.rpt` reports no generated-clock, loop, missing input
delay, unconstrained endpoint, pulse-clock, driving-cell, or partial-delay
problem.  `check_design_postcompile.rpt` reports only three unconnected M82
observability outputs (`beat_width[3]`, `beat_accept`, `collecting`).  Link and
postcompile reference reports show no unresolved production reference, and
both area and QoR report zero macros/black boxes.

There are expected diagnostic warnings: nine signed/unsigned conversion
warnings, DesignWare auto-loading, two high-fanout nets modeled with fanout
1000, three unconnected ports, and a name-rule warning.  None is an unresolved
reference, but the high-fanout/ZeroWireload combination is another reason the
result is pre-layout diagnostic evidence only.

## Port timing boundary

`PORT_TIMING_BOUNDARY.md` is accurate and is necessary to every use of this
number:

- `bank_row_addresses[79:0]` stop at output ports;
- `bank_words[255:0]` start at input ports with a generic 0.250 ns delay;
- there is no address-to-SRAM-to-data arc, SRAM clock-to-Q, decoder, physical
  routing, or macro timing check;
- the external-bank model is only a combinational timing cut, not a realizable
  combinational SRAM interface;
- `synchronous_sram_interface=false` and
  `complete_pwp_lookup_timing=false`.

The failing path is the unrolled metadata poison audit, not a PWP SRAM lookup.
M99 may use the result to test whether serializing that audit removes the exact
logic-island bottleneck.  It may not infer complete lookup timing.

## Is the diagnostic area denominator rigorous?

**Conditional GO.**  The boolean is correct in the context of the receipt's
scope and negative gates.  A safer prose expansion is:

```text
usable_as_exact_same_backend_semantics_and_port_cut_standard_cell_area_denominator_only
```

The denominator is useful because the mapped area is internally consistent,
the backend completed, no macros/unresolved references contaminate it, its
source scope is explicit, and all used result files are frozen.  Wrapper exit 2
does not change the cells DC already mapped.

It is conditional because the run was not wrapper-sealed, the pre-lock launcher
did not self-attest all semantic inputs, M97 violates setup, and the quantity is
cell area without interconnect or SRAM.  It must never appear as a bare
`M99 area / baseline area` number without the words `same-flow logic-only
diagnostic`.

The M100 contract's proposed 35% gate is arithmetically correct:

```text
34,758.486144 * 0.35 = 12,165.4701504 um^2
```

Passing that gate can admit a same-flow standard-cell reduction.  It cannot
turn M97 into a timing-met or PPA baseline.

## M99 A/B usage conditions

M97 can be the M85 arm/denominator for M99 only if all of the following hold:

1. The M99 run pins this exact receipt SHA and evidence-manifest SHA; no value
   is copied manually from an unfrozen report.
2. M99 uses the same DC version, setup/hold DB hashes, `ssg0p9v125c`, 3.000 ns
   period, uncertainties, I/O delay/transition/load, ideal clock,
   ZeroWireload, `SYNTHESIS`, and compile/hold-repair Tcl.
3. The common M82 RTL and parameters are identical.  Only M85 versus M99 is the
   intentional logic change.
4. Both arms retain the identical external bank-address/bank-word timing cut
   and zero-macro scope.  If an M99 variant adds an SRAM/ROM macro, M97 is no
   longer an equal-scope denominator; rerun a macro-inclusive A/B.
5. Every M99 state bit, parser counter, table, and mux is included.  A
   base-precompute candidate may not exclude its 128x13-bit table or silently
   price it as a free ROM.
6. The M99 backend and hardened wrapper both complete; its own input/output
   manifest passes, unresolved references and macro count are zero, and mapped
   Verilog is nonempty.
7. M99 independently meets setup and hold.  M97's setup failure does not block
   an area ratio, but it forbids an M97-versus-M99 Fmax ratio.  Timing wording is
   limited to `M99 meets the fixed 3 ns target and the M85 phase-poison path is
   removed/migrated`.
8. Functional replacement is already admitted by the exact 1,728-phase VCS
   differential and negative attacks, with parser latency aligned.  DC area
   alone cannot authorize M99.
9. Compare the report's `cell_area`, leaf/cell/resource counts, and exact
   critical endpoints.  Do not compare `total area`, which is undefined here,
   or power/energy, for which this run has no activity/macro evidence.
10. Preserve launcher provenance per arm: M97 remains the pre-lock run; M99 may
    use the hardened launcher but must not claim M97 did.

If these gates hold, M99 A/B may report the exact diagnostic area fraction and
the fixed-target bottleneck migration.  Any complete-PWP, synchronous-SRAM,
paper-PPA, energy, throughput, or system statement still requires a separate
macro-inclusive implementation and timing/power flow.

## Scores and findings

| Dimension | Score / 100 |
|---|---:|
| frozen output integrity | 97 |
| backend-result reconstruction | 98 |
| launch provenance | 82 |
| diagnostic-denominator rigor | 86 |
| timing/PPA completeness | 32 |
| scoped post-run milestone | 89 |

Severity counts: `P0=0`, `P1=3`, `P2=3`.

P1 findings are the conditional-denominator boundary, the failed 3 ns setup
plus incomplete SRAM timing, and non-retroactive/unsealed launch provenance.
P2 findings are the rounded-zero hold margin, three unconnected observability
ports, and high-fanout/ZeroWireload plus signed-conversion warning debt.

Final status remains conditional GO for the exact same-flow standard-cell area
denominator and NO-GO for a 3 ns timing baseline, complete PWP, physical PPA, or
paper claim.
