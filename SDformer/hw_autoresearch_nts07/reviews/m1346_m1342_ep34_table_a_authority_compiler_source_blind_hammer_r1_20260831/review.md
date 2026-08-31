# M1346 — M1342 Table-A authority compiler source blind hammer

## Verdict

`FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED`

M1342 is fail closed against ordinary self-created production JSON today: its
code-pinned production allowlist has zero entries, caller fixture authorities
are ignored in production mode, and no production candidate or Table-A row can
be emitted.  The frozen M1340 10/10 tests, M1342 16/16 tests, author double seal
and M1341 failed-predecessor double seal all reproduce.

It is not yet safe to populate that allowlist.  Six fresh attacks pass:

1. `..` traverses outside the supplied workspace because containment is
   lexical rather than resolved.
2. Direct B0/Ours logic-energy rates may differ arbitrarily; an unchanged
   charge obtains greater than 99% apparent reduction.
3. `address_trace_sha256` accepts non-hex text and has no sealed trace payload.
4. The entire DRAM plane may be zero while SRAM is nonzero.
5. All seventeen SRAM planes may be zero while DRAM is nonzero.
6. No SRAM/DRAM latency or memory-stall authority is represented.

Thus no fake JSON crosses the current empty production gate, but the future
release interface would not yet prove equal-rate, trace-bound,
memory-inclusive production evidence.  The minimum successor is limited to
resolved containment, equal-rate/direct evidence discipline, sealed trace
bytes, separate nonzero SRAM/DRAM gates, and latency-aware transaction
authority.  It must be hammered before any allowlist entry is populated.

No Table-A row, capture, GPU, VCS, DC, PT, PTPX, EDA or remote task ran.
`docs/359` remains `dedde7ce...`.
