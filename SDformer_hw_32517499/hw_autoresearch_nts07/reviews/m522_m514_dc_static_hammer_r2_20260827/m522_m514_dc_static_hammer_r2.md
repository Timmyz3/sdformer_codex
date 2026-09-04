# M522/M514 logic-only DC static hammer r2

Verdict: **STATIC GO — exactly one positive DC run is authorized for runner `2f2124e03cfe937aadd6432ce42bb198badb39c14823973cf373f4d58db35faa`.** Score 96/100, P0=0, P1=2, P2=2.

## r1 P0 closure

Both blocking defects are closed.

1. Tool identity now treats `/opt/.../dc_shell` as an explicitly pinned launcher symlink and `/opt/.../snps_shell` as the resolved regular executable. The observed link text, resolved path, and executable SHA all match the contract. The non-symlink rule remains intact for ordinary evidence files.
2. Publication is fail-closed. The exact topology, finite strict receipt, member manifest, and outer seal are verified in staging before `mv`; the canonical path is verified again after `mv`; only then is `m522_complete=1` assigned. Any earlier or post-move failure selects canonical first, otherwise staging, and moves the incomplete directory to a collision-guarded quarantine.

## Other hard gates

All 12 contract inputs match their SHA pins. The M514 VCS package and independent hammer retain passing member and outer seals. The runner performs a wrong-runner-SHA negative preflight expecting exit code 10, persists runner identity and the negative result, and requires this sealed r2 review to authorize the exact same runner SHA before resource checks or DC.

The synthesis path explicitly locks `SYNTHESIS`, the slow target and fast min library, 3 ns constraints, `ZeroWireload`, and an ideal `clk_core` network. Its precompile TIM-209/OPT-150 gate combines build/link, `check_design`, and `check_timing` reports before flatten or compile. The post-run gate requires nonnegative setup/hold slack, five clean constraint classes, the mapped netlist/SDC/DDC/SVF population, a finite read-back-equal JSON receipt, exact file/directory topology, no symlinks, and two seal verifications.

Two nonblocking P1 issues remain: gate counts are copied into the receipt after prior exact shell checks instead of reparsed inside the receipt builder, and seal exclusions use basenames rather than root-relative seal paths. Neither applies a false claim or opens a known path in the planned output population.

## Authorization boundary

This review authorizes exactly one positive execution of the reviewed runner, solely to measure standalone M514 additive decoder-support logic area and 3 ns timing cost. It does **not** authorize any cycle/system speedup, energy, full-decoder, physical-SRAM, Formality, paper-ready PPA, or headline claim. The resulting canonical package still requires an independent receipt-blind DC hammer.

No runner, DC, or VCS command was executed during this review. Production files and `docs/359` were not modified; the latter remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
