# M518 r5 static-hammer author handoff

This is an author handoff, not a review or authorization.

The r5 production RTL changes exactly one block-local identifier. Six word
tokens—one declaration and five uses—were renamed from the reserved token
`within` to `tap_within`. Replacing exactly those six tokens back in memory
reconstructs the frozen r4 RTL SHA
`09b1d976595f...1379a93412a`; therefore there is no other RTL byte change.
SVA, TB, and filelist identities are unchanged.

Frozen r5 identities:

- RTL: `90e0304bd8fa...b9288600a6a`
- contract: `51b81bbad3ea...c68fb2ff996`
- runner: `854f152ad23b...3c3560a7312a`
- canonical result: `results/m518_matched_fixed_t10_atlif_vcs_r5_exact_20260827`
  (absent at handoff)

The runner statically has 24 exact input bindings. `bash -n` passed. An
all-zero *runner identity* check returned 4 before result creation; the positive
runner and its automatic wrong-RTL campaign were not invoked. No VCS, DC,
Formality, PT/PTPX, or open-source RTL tool was run by the r5 repair author.

A different reviewer must perform the requested receipt-blind static hammer.
Only that reviewer may issue an exact-SHA, one-shot r5 VCS authorization. This
handoff authorizes neither VCS nor DC.
