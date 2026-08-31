# M973 independent M962 DC result-hammer request

This is an inert, source-only request. It was authored without reading the
running M962 work directory and without running DC, VCS, PT, PTPX, Formality,
another EDA tool, or a license query.

The future reviewer must wait for an explicit completion notice. The script's
default invocation reads no result. Even with
`--review-complete-canonical`, it returns `WAIT` before payload access unless
the launch lock is absent and the exact canonical directory already contains
regular `RUN_COMPLETE.txt`, `SHA256SUMS`, and
`SHA256SUMS.seal.sha256` files. The transient
`.m962_m935_three_stage_match_macro_aware_dc_work.*` namespace is never read.

The result-integrity decision is separate from physical admission. A complete
3 ns setup-negative run with finite WNS/TNS, a violation count, and the
preserved top paths is valid negative evidence. It is not timing-admitted.
Missing reports, a tool/link/macro failure, or an inconsistent seal is an
integrity failure or sealed quarantine, not a negative timing result.

The macro gate requires nine mapped
`TS1N28HPCPHVTB128X128M4S` single-port 1RW instances, each 128 rows by
128 bits. Their physical capacity is 18,432 B; only 64 rows are logically
addressed, yielding 9,216 B of parent payload. This does not integrate the
full 213,376 B same-ledger storage obligation.

No DC setup/area result can promote the upstream CPU same-ledger
`1.746753x` opportunity into RTL cycles or speedup. There is no fair RTL
zero/bit baseline or trace bridge in this DC point. Hold, power, energy, full
PPA, system speedup, paper readiness, and headline claims remain false.
