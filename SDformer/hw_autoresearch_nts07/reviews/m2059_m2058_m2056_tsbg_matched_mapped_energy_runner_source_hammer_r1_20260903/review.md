# M2059 independent source hammer of the M2058 mapped-energy runner

## Verdict

**PASS, 99/100, P0/P1/P2 = 0/0/1. One and only one M2058 execution is
authorized when invoked with the final exact source-review pins.** No EDA,
`lmstat`, license query, attempt, SAIF, or result was launched by this review.

## Why the runner is fail-closed

- It accepts no command-line arguments and verifies its own, the parser's,
  contract's, M2059 review's, all M2056 sources', M2029 netlist/SDC, tool,
  standard-cell model, SSG DB, and TT DB identities.
- It holds the shared same-UID queue for the whole campaign. The fixed order is
  ordinary compile/sim/SAIF/PTPX, followed by TSBG compile/sim/SAIF/PTPX.
- A no-retry attempt latch is sealed before the sole `lmstat` call. Failure at
  license, compile, runtime, SAIF, PTPX, parsing, or publication consumes the
  attempt and goes to a failure quarantine.
- Each simulator command contains exactly one runtime plusarg:
  `+WORKLOAD_SLOT=42`. Any other runtime plusarg fails before launch.
- Both builds use the exact M2056 top and axis filelist. PTPX receives only the
  axis plus exact TT/SSG/netlist/SDC/SAIF/output paths; design, strip scope, and
  measurement cycles remain derived inside the sealed M2056 Tcl.

## Required result evidence

For each axis the parser requires exactly one M2056 begin marker, one matching
end marker and one complete M2051 PASS with every identity/ledger field frozen.
It rejects runtime fatal/assertion/X/Z diagnostics, any nonzero SAIF `TX`, a
duration other than 60,876 ns or 22,707 ns, empty switching activity, incomplete
PT annotation, failed power checks, duplicate/nonfinite power fields, or a
power subtotal mismatch.

The candidate receipt reports switching, internal, leakage and total power in
mW and the corresponding execute-window energies in pJ. External weight SRAM
is excluded from numeric energy: it reports only 14,304 versus 4,608 scalar
128-bit reads and the formulas `Nread × Eread_128b` and
`Nread × 128 × Eread_bit`.

## Claim boundary and remaining P2

The only admitted scope after a future result hammer is one pre-registered
ep34 G48 component workload, selected positionally rather than by performance,
using real activity masks but deterministic directed INT8 weights. Power is
mapped standard-cell logic only, TT0.9V25C averaged prelayout, ideal clock,
ZeroWireload, zero macros, and excludes the external weight SRAM. It is not
full FC1, checkpoint-weight, system, or paper-ready PPA evidence.

P2 is solely the mandatory independent result hammer. The runner publishes a
candidate, never an admitted result.

`docs/359` remains unchanged at SHA256
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
