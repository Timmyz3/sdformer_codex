# M1332 — C2 headline mapped production-activity source receipt

## Result

`PASS_M1332_SOURCE_ONLY__INDEPENDENT_HAMMER_REQUIRED__NO_EDA_AUTHORIZED`

M1332 implements the source-only successor required by M1331 for the two
headline comparison axes.  It does not touch M803 or any other headline RTL.
Instead, it preserves the exact frozen M979 workload/reference driver and
replaces only its external test-memory model with a reset-safe, four-state
fail-closed drop-in model.

The K8 and equal-bandwidth K1×8 coordinates have separate exact filelists.
Each binds its admitted M872 mapped netlist, enables runtime SVA, and excludes
both diagnostic K1 and the old unqualified M349 memory model.  A common wrapper
observes all eight endpoint fault bits without placing wrapper/assertion/test
memory activity inside the DUT-only SAIF scope.

## Gates authored

- All stored request payload and held-response state is explicitly reset.
- Invalid or unknown request payload cannot index pending state.
- Unknown valid/ready/accept or valid payload is sticky-faulted.
- Accept equations and payload knownness are asserted for header, source,
  request, response, result commit, and token completion.
- Result and token payloads must remain stable under backpressure.
- Each future case requires nonzero source, commit, stall, and done coverage.
  Nonzero cases 0–3 also require endpoint traffic.  Frozen zero-event case 4
  must have exactly zero endpoint traffic, avoiding manufactured activity.
- Both the original 100,000-cycle workload watchdog and a 1 ms assertion
  watchdog fail closed.
- SAIF covers only
  `tb_m1332_c2_headline_mapped_production_activity.core.dut`, excludes reset
  and idle tails, requires zero TX/reset TC, exact 3 ns × cycle duration, and
  the original M903 cycle tuple.

Six author tests pass, including mutation rejection for K1 injection, old
memory fallback, cycle drift, nonzero TX/reset activity, and missing endpoint
activity.  No VCS, DC, PT, PTPX, GPU, or remote job was launched.

## Boundary

This is source evidence only.  It records zero completed mapped cases and zero
SAIF files.  A different-author receipt-blind source hammer is required before
any one-shot VCS release may be authored.  Power, energy, performance, system
speedup, and paper-ready PPA remain false.  `docs/359` remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
