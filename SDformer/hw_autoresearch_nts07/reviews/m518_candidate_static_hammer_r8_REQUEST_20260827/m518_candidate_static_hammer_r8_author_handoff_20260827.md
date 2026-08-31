# M518 r8 TB-only phase16 settle author handoff

Date: 2026-08-27  
Status: `AUTHOR_HANDOFF_ONLY__INDEPENDENT_STATIC_REVIEW_REQUIRED__NO_TOOL_AUTHORIZATION`

## Exact source delta

The only functional source edit relative to frozen r7 is in V08. Immediately
after the phase16 `negedge` drives `result_ready=0`, the TB waits `#0.2` before
sampling combinational `fifo_credit`:

```systemverilog
@(negedge clk_core);result_ready=1'b0;#0.2;
if (!(u_dut.dense_active_q && u_dut.dense_selected_cycle==16
        && result_fifo_occupancy==16 && !u_dut.fifo_credit))
    $fatal(1,"V08 phase16 targeted stall did not align");
```

Deleting only `#0.2` from its unique occurrence mechanically recovers frozen
r7 TB SHA256
`a2de78ac5a3c537e03113f06552a09808426170d188d39e462b500b0c865eb12`.
The r8 TB SHA256 is
`d03fd23a19046d7b96819f2f8b7753a03cb2cf3454564579b03647026a480de2`.

RTL and SVA remain byte exact to r7:

- RTL: `8a7ec11843b1b9c13c22ab679f69d70f73a8f5874f9ccee51c8873f4f7f142d6`
- SVA: `89d4d711e2913e49ed14d3368c786f069cf11b2ec3f89371dd8582358917c1f5`

V06, all V01--V20 phase steps, the numeric/conservation/protocol oracles, the
exact PASS signature, 51 assertions, and all 25 nonzero-cover requirements are
unchanged. No `$deposit`, DUT `force/release`, hierarchical DUT-state LHS,
writing bind, or `always_ff` downgrade was introduced.

## New runnable identity

```text
contract: contracts/m518_matched_fixed_t10_atlif_vcs_contract_draft_r8_20260827.json
contract SHA256: 68055a1385918909eaee5f881b1e226ca8d8f03a1609917c51e0faf55d42fe9b
runner: dc_handoff/scripts/run_vcs_m518_matched_fixed_t10_atlif_r8_exact.sh
runner SHA256: fe457d7bbf93e72e913c55427696fb782dcc00dee80c74b1f4dba9c3edd01a52
runner mode: 0755
result: results/m518_matched_fixed_t10_atlif_vcs_r8_exact_20260827
```

The canonical r8 result path is absent. `M518_RUN_DIR` is rejected. Exact
runner SHA and a reviewer-created, double-sealed static admission SHA are both
checked before canonical `mkdir`, so missing/wrong authorization creates no
result and launches no tool. After that, the wrong-TB control must exit 10
before VCS and double-seal only its negative subdirectory.

`bash -n` passes. All three embedded Python blocks parse under the Python 3.6
grammar and avoid `Path.is_relative_to`, assignment expressions, f-string
debug syntax, `subprocess.run(text=...)`, and other post-3.6-only APIs.

## Launch requirement after independent GO

The independent reviewer must generate and double-seal
`contracts/m518_matched_fixed_t10_atlif_vcs_launch_admission_r8_20260827.json`
with the five exact fields required by the review request. Only then may the
root invoke:

```text
M518_EXPECTED_RUNNER_SHA256=fe457d7bbf93e72e913c55427696fb782dcc00dee80c74b1f4dba9c3edd01a52
M518_EXPECTED_STATIC_ADMISSION_SHA256=<reviewer-generated exact admission SHA256>
M518_RUN_DIR must be unset
```

The author did not execute the runner, its negative campaign, VCS, DC,
Formality, PT/PTPX, or any open-source EDA tool. No compile, runtime, numeric,
cycle, physical, power, energy, speedup, PPA, system, or headline claim is made.

`docs/359` remains SHA256
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
