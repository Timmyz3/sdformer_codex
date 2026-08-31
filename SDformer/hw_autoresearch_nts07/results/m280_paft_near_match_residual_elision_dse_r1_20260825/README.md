# M280 PAFT near-match residual-elision DSE

This milestone screens a bounded lossy extension of the exact M251 PAFT Conv
path on the frozen M248 ten-sample running-BN trace.  A partition with
population at least two and nearest-pattern Hamming distance at most `tau` is
served by one PWP and no signed correction vectors.  Zero and singleton
fallbacks remain exact.

The exact threshold-zero miter reproduces M251.  The wide isolated-Conv model
reaches 2.000245457x versus the matched bit-sparse path at `tau=2`, and reaches
2.411370253x at `tau=3`.  These are trace-cycle opportunities only: no snapped
network forward, accuracy, RTL, energy, system speedup, PPA, or headline claim
is admitted.

Promotion requires a modified-forward S10 screen followed by paired
running-BN valid825.  The absolute AEE increase must be at most 0.02 versus the
same PAFT checkpoint without snapping.

Primary evidence:

- `m280_paft_near_match_residual_elision_dse_r1.json`
- `m280_paft_near_match_residual_elision_dse_receipt_r1.json`
- `RUN_MANIFEST.sha256`

