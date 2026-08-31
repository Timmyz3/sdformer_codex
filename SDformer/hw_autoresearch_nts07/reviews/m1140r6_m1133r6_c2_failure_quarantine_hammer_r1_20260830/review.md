# M1140R6 independent M1133R6 C2 failure-quarantine hammer

Verdict: **PASS; this is a structural-checker false negative, not a reset
provenance break. Authorize only an additive structural-checker repair source.**
No automatic retry, launch, DC, mapped VCS, or subject modification is
authorized by this receipt.

The namespace contains exactly one consumed attempt and exactly one quarantined
failure; the canonical result and work directories are absent. The failure is
sealed as `FAILED_DIAGNOSTIC_DO_NOT_CITE` at
`MAPPED_RESET_PROVENANCE_337`, and mapped VCS did not run. Every retry flag in
the applicable contract lineage is false.

Independent cell-level tracing covered all 337 shadow-register asynchronous
clear pins. All 12 clear nets have one driver chain, terminate at `rst_core`,
and have exactly one inversion. Seventy-five register bits use a direct
inverter; 262 use a legal non-inverting buffer followed by an inverter. No
constant, cycle, reconvergent/multiple-driver path, or non-buffer combinational
logic was accepted.

The reported register `shadow_service_result_count_q_reg_22_` clears from
`n186651`. The netlist path is `CKND0BWP35P140 U114871` driven by
`BUFFD1BWP35P140 U104338`, so its equation is
`n186651 = NOT(BUF(rst_core)) = NOT(rst_core)`. The foundry model independently
binds `BUFFD1` to polarity preservation and `CKND0` to inversion. The frozen
checker instead requires the inverter input itself to equal `rst_core`, thereby
rejecting a valid synthesized reset-tree buffer.

The authorized repair must remain fail-closed: traverse only a bounded,
single-driver chain of library-proven polarity-preserving buffers plus exactly
one inversion to `rst_core`, while rejecting unknown gates, constants, cycles,
multiple drivers, reconvergence, and wrong polarity. Existing DC area/setup
reports remain diagnostic and non-paper-citable; hold and mapped functionality
are not closed.
