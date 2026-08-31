# M518 r6 static-hammer author handoff

This is an author handoff, not a review or authorization.

The r6 SVA inserts exactly one right parenthesis immediately before the
semicolon in `ap_dense_start_ownership`. The SVA is exactly one byte longer than
r5. Removing that unique character in memory restores r5 SVA SHA
`977f95652bb7...582c910f4` byte for byte. The target assertion is balanced at
eight opening and eight closing parentheses. RTL, TB, filelist, public behavior,
and the V01-V20 campaign were not edited.

Frozen r6 identities:

- SVA: `89d4d711e291...582358917c1f5`
- contract: `153f733bb231...4f329ef4341`
- runner: `050db5ce7001...9bc3e60c2a4d`
- canonical result:
  `results/m518_matched_fixed_t10_atlif_vcs_r6_exact_20260827` (absent)

The runner has 35 exact input bindings, including the sealed r4 and r5 failure
chains. `bash -n` passed. A wrong *runner identity* check returned 4 before
result creation. Neither the positive runner nor its automatic wrong-SVA
negative campaign was invoked. No VCS, DC, Formality, PT/PTPX, or open-source
EDA tool was run by the r6 repair author.

A different reviewer must perform the requested receipt-blind static hammer.
This handoff authorizes neither r6 VCS nor downstream physical tools.
