# M518 r10 runner-schema author handoff

Date: 2026-08-27  
Status: `AUTHOR_HANDOFF_ONLY__INDEPENDENT_STATIC_REVIEW_REQUIRED__NO_TOOL_AUTHORIZATION`

## Sole functional repair

r9 failed before VCS because its Python preflight queried the nonexistent
historical field `decision.r9_vcs_authorized`. The double-sealed r9 pretool
review identifies the real field in the older r8 release-failure review as:

```text
decision.r9_vcs_authorized_by_this_review
```

r10 first requires that the sealed r9 pretool review points to that exact path.
It then loads the sealed r8 review, requires the `decision` object to be a JSON
object, requires the exact key to exist, and requires its value to be the JSON
boolean `false`. Missing, `null`, `0`, string `"false"`, and `true` all fail.
The boolean must not be inverted: the historical diagnostic review is not
launch authority. A new independent, double-sealed r10 admission is still the
only object allowed to carry `vcs_authorized: true`.

The r9 pretool review itself is also required to keep
`decision.r10_vcs_authorized_by_this_review` exactly `false`.

## Preserved source and campaign identity

- RTL: `8a7ec11843b1b9c13c22ab679f69d70f73a8f5874f9ccee51c8873f4f7f142d6`
- SVA: `89d4d711e2913e49ed14d3368c786f069cf11b2ec3f89371dd8582358917c1f5`
- TB: `8877512040c0677de58bc88c1cacd8056bb6f20026c24e3794f633682d962e56`
- filelist: `09e435600ded03f79ff4eb1462135ce67d4987725e07111b230fbbd1a2f22fea`

These are byte exact to r9. V01--V20, the r9 release/raw `#0.2` cadence repair,
the r8 line-765 settle repair, expected N1/N4 cycles, numeric and conservation
oracles, exact PASS signature, 51 assertion labels, and 25 required nonzero
covers are unchanged.

## New identities

```text
contract: contracts/m518_matched_fixed_t10_atlif_vcs_contract_draft_r10_20260827.json
contract SHA256: ba545cd5a351b31652e6e60415382dd7fb00ae3a3d8b665ad24524537b4c4d15
result identity: contracts/m518_matched_fixed_t10_atlif_vcs_result_identity_r10_20260827.json
result identity SHA256: 117546bd618378997530333c5435ee488457b1fe047a18bb21a882329647a464
runner: dc_handoff/scripts/run_vcs_m518_matched_fixed_t10_atlif_r10_exact.sh
runner SHA256: ca1873f864891b6da02260b10feedbdf7d18b9e1ecca52dbcc77e0bed8c91ef6
runner mode: 0755
canonical result: results/m518_matched_fixed_t10_atlif_vcs_r10_exact_20260827
```

The 31-entry runner SHA map has zero mismatch. `PREFLIGHT_COMPLETE.txt` is
written after all semantic checks and before the first possible VCS command.
On a positive run, `RUN_ARTIFACTS_COMPLETE.txt` is included in the manifest;
both manifest copies and both outer seals are checked before
`RUN_COMPLETE.txt` is created. Thus r9's RUN_COMPLETE-before-seal P1 is closed.

The canonical r10 result and r10 launch admission are absent at this author
handoff. `M518_RUN_DIR` remains forbidden. Exact runner and independent
admission SHAs are checked before canonical `mkdir`.

## Required independent review

The independent reviewer must perform every mechanical check in the JSON
request, especially exact external JSON-path existence and strict boolean
typing. Only a P0-zero review may create and double-seal the r10 one-shot launch
admission. The author did not create an admission and did not execute the
runner, negative campaign, VCS, DC, Formality, PT/PTPX, or open-source EDA.

No compile, simulation, numeric, cycle, physical, energy, speedup, PPA,
system, or headline claim is made. `docs/359` remains SHA256
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
