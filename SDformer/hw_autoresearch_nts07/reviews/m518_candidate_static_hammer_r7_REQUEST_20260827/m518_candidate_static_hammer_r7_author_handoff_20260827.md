# M518 r7 legal-fill harness author handoff

Date: 2026-08-27  
Status: `AUTHOR_HANDOFF_ONLY__INDEPENDENT_STATIC_REVIEW_REQUIRED__NO_TOOL_AUTHORIZATION`

## What changed

V06 no longer writes DUT state. All ten hierarchical `$deposit` calls were
removed. Under `M518_VCS_V06_HARNESS`, two simulation-only input controls hold
dense issue and steer only the first empty raw fill to bank1. V06 sends the
bank1 payload/tag and then the bank0 payload/tag through the existing legal
five-beat raw interface. It requires dual-ready/dual-owned state, bank1 older
than bank0, exact tags, ten raw beats, two loaded tiles, and zero issue while
held. It then releases the hold and preserves the original bank1 selection,
post-edge dense state, ownership/tag checks, two-tile conservation, and numeric
scoreboard.

The campaign adds exactly one closure field:
`v06_legal_fill_harness=1`. All prior V01--V20 fields remain in the PASS line.

## Production boundary

With the macro absent, static preprocessing finds exactly 50 ports and the
ordered signature is exact to M273. With the macro present, there are 52 ports;
the only suffix additions are `v06_hold_dense_issue` and
`v06_first_empty_fill_bank1`.

The runner's mechanical inverse reconstructs frozen r6 RTL SHA
`90e0304bd8fa5bae5f4cf523d8ab7c62b42878a0ce17b75bd62f8b9288600a6a`
and frozen r6 TB SHA
`e7973a91d04b9f20542b04c58a213e2c8929259768701664dda76721720d2888`.
Therefore every source delta is explicitly bounded to the simulation harness,
V06 stimulus, and its one-count closure. DC, Formality, PT, and PTPX must never
define the macro.

## Static author checks

- zero `$deposit`, hierarchical `force/release`, direct DUT-state LHS, or
  writing bind;
- exactly one DUT `always_ff`, with no `always @(posedge)` downgrade;
- 46/46 exact runner SHA-map bindings;
- runner mode 0755 and `bash -n` pass;
- two embedded Python blocks parse; the source-static block executes cleanly;
- exactly one Full64 VCS identity query, one Full64 compile, one simulation,
  and one harness compile define;
- `M518_RUN_DIR` is mechanically rejected and canonical creation uses atomic
  `mkdir` after exact runner-SHA gating;
- r7 canonical result is absent;
- no runner, negative control, VCS, DC, Formality, PT/PTPX, or open-source EDA
  was executed by the author.

## Frozen runnable identity

```text
runner: dc_handoff/scripts/run_vcs_m518_matched_fixed_t10_atlif_r7_exact.sh
SHA256: 971c61367d84eea7f6dec73fa4a6efa79cc82fe308e00ffa800409c68b9edce0
result: results/m518_matched_fixed_t10_atlif_vcs_r7_exact_20260827
```

This handoff does not authorize VCS. A different independent reviewer must
finish the requested static hammer and explicitly authorize that exact runner
SHA. Any source, contract, runner, dependency, or result-path drift voids the
request. Post-run independent receipt review remains mandatory.

No compile, runtime behavior, production equivalence, cycles, accuracy, DC,
Formality, STA, power, energy, speedup, PPA, system, or headline claim is made.

