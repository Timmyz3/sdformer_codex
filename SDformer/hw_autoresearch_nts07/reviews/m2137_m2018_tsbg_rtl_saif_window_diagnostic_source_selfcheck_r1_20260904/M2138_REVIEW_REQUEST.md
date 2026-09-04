# M2138 independent source-hammer request

Review the exact additive M2137 source identity frozen by contract
`m2137_m2018_tsbg_rtl_saif_window_diagnostic_source_contract_r1_20260904.json`.
Do not invoke a license query, VCS, `simv`, DC, PT, PTPX, ICC2, a GPU, or any
other EDA executable.

The review must independently and exhaustively establish all of the following:

1. M2125--M2128 are immutable and exact-pinned.  M2127 is a consumed failed
   attempt with no retry and no citable result; its M2128 failure hammer is
   exhaustive and double-sealed.
2. The only M2137 semantic delta is the timing-contamination guard.  RTL, TB,
   parser, filelist, UCLI, workload slot42, initialization, settled-negedge
   windows, ledgers, 93,971-record SAIF gates, TX=0, conservation, and critical
   activity gates are inherited byte-for-byte from M2125.
3. A harmless `/SDformer/` pathname must pass when it appears in `-Mdir`, the
   filelist operand, and the output path.  Do not treat a pathname substring as
   an SDF option.
4. At least these three negative mutation classes must fail closed:
   explicit `-sdf*`/`+sdf*` command tokens; `+define+UNIT_DELAY` including
   combined/value/case variants; and active filelist/source content containing
   `$sdf_annotate` or `UNIT_DELAY`.
5. Every source inventory SHA, contract sidecar/outer seal, predecessor review
   seal, tool identity inherited through M2125, and docs/359 SHA must validate.
6. The only future execution is one fresh M2139 license query, one shared VCS
   compile, two strictly serial `simv` axes, and two SAIF files.  DC/PT/PTPX,
   reuse, retry, caller-selected workload/axis, and a second launch path are
   absent.  Review and freshness precede attempt creation and the license call.
7. M2139 result/attempt/lock are fresh now.  Any execution failure consumes
   the one authorization and must be quarantined without automatic retry.
8. Even a passing M2139 remains a diagnostic pending exhaustive M2140 result
   review; it is not a mapped-activity, power, energy, speedup, or paper claim.

Required M2138 outputs:

- `review.json` schema
  `m2138_m2137_m2018_tsbg_rtl_saif_window_diagnostic_source_hammer_r1_v1`;
- status beginning `PASS_M2138` only if P0/P1/P2 are all zero;
- `severity_counts={"p0":0,"p1":0,"p2":0}` and score at least 95;
- `authorization` exactly equal to `future_m2139_budget` in `selfcheck.json`;
- `identity.runner_sha256=a1a72dcdfbbf0f1f0cbae52424b1dac08b023edd612223236f9c2fb77e7445d4`
  and
  `identity.contract_sha256=42d2394942f25e80a28b6b448ad966715366dc3d71ea60e5cf1899b07b89b2cd`;
- independent positive and negative mutation evidence, `review.md`, mechanical
  checks, `RUN_COMPLETE.txt`, exhaustive `SHA256SUMS`, and
  `SHA256SUMS.seal.sha256`.

Any uncertainty is a fail-closed review, not an authorization.
