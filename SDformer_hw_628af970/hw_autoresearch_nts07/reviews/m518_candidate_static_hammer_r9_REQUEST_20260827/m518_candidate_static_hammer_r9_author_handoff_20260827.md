# M518 r9 TB-only V16 release-cadence author handoff

Date: 2026-08-27  
Status: `AUTHOR_HANDOFF_ONLY__INDEPENDENT_STATIC_REVIEW_REQUIRED__NO_TOOL_AUTHORIZATION`

## Exact source delta

The only functional source edit relative to frozen r8 is in
`release_partial_raw_attack`. `send_config` already returns at a `negedge`, so
the redundant additional `@(negedge clk_core)` is replaced by a `#0.2`
post-edge stimulus skew:

```systemverilog
#0.2;release_valid=1'b1;raw_valid=1'b1;
```

Replacing only that unique r9 fragment with the frozen r8 fragment
`@(negedge clk_core);release_valid=1'b1;raw_valid=1'b1;` mechanically recovers
frozen r8 TB SHA256
`d03fd23a19046d7b96819f2f8b7753a03cb2cf3454564579b03647026a480de2`.
The r9 TB SHA256 is
`8877512040c0677de58bc88c1cacd8056bb6f20026c24e3794f633682d962e56`.

The V08 line-765 fragment remains byte exact and unique:

```systemverilog
@(negedge clk_core);result_ready=1'b0;#0.2;
```

RTL and SVA remain byte exact to r8:

- RTL: `8a7ec11843b1b9c13c22ab679f69d70f73a8f5874f9ccee51c8873f4f7f142d6`
- SVA: `89d4d711e2913e49ed14d3368c786f069cf11b2ec3f89371dd8582358917c1f5`

V06, all V01--V20 phase steps, every expected cycle, the
numeric/conservation/protocol oracles, the exact PASS signature, 51 assertions,
and all 25 nonzero-cover requirements are unchanged. No `$deposit`, DUT
`force/release`, hierarchical DUT-state LHS, writing bind, or `always_ff`
downgrade was introduced.

The runner's 28-entry exact-SHA map has zero mismatches and binds the frozen r8
contract/runner/static admission, the double-sealed r8 independent static
review, all eight immutable r8 failure artifacts, and the double-sealed r6
failure hammer that recommends this sole r9 edit.

## New runnable identity

```text
contract: contracts/m518_matched_fixed_t10_atlif_vcs_contract_draft_r9_20260827.json
contract SHA256: f99767c17e33000012de31873169544e68f1e9b8eaf3724257595d666004b11b
runner: dc_handoff/scripts/run_vcs_m518_matched_fixed_t10_atlif_r9_exact.sh
runner SHA256: f43a5d48bdf38d0d98663243a522f7bd26e44edeb51df0b03a25629d4d2d5933
runner mode: 0755
result: results/m518_matched_fixed_t10_atlif_vcs_r9_exact_20260827
```

The canonical r9 result path is absent. `M518_RUN_DIR` is rejected. Exact
runner SHA and a reviewer-created, double-sealed static admission SHA are both
checked before canonical `mkdir`, so missing/wrong authorization creates no
result and launches no tool. After that, the wrong-TB control must exit 10
before VCS and double-seal only its negative subdirectory.

`bash -n` passes. All three embedded Python blocks parse under the Python 3.6
grammar and avoid `Path.is_relative_to`, assignment expressions, f-string
debug syntax, `subprocess.run(text=...)`, and other post-3.6-only APIs.

## Launch requirement after independent GO

The independent reviewer must generate and double-seal
`contracts/m518_matched_fixed_t10_atlif_vcs_launch_admission_r9_20260827.json`
with the five exact fields required by the review request. Only then may the
root invoke:

```text
M518_EXPECTED_RUNNER_SHA256=f43a5d48bdf38d0d98663243a522f7bd26e44edeb51df0b03a25629d4d2d5933
M518_EXPECTED_STATIC_ADMISSION_SHA256=<reviewer-generated exact admission SHA256>
M518_RUN_DIR must be unset
```

The author did not execute the runner, its negative campaign, VCS, DC,
Formality, PT/PTPX, or any open-source EDA tool. No compile, runtime, numeric,
cycle, physical, power, energy, speedup, PPA, system, or headline claim is made.

`docs/359` remains SHA256
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
