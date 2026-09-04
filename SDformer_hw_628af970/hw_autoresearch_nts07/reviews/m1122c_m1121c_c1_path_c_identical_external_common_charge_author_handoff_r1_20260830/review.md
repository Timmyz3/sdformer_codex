# M1122C Path-C identical external common-charge author handoff

Verdict: **GO only for a different-author M1123C static hammer. No RTL, external-memory implementation or EDA is authorized.**

This additive source contract implements M1121C's Path-C accounting boundary without pretending that the entire capacity ledger is live hardware. Candidate, strongest-zero and same-coordinate-bit each receive the same `214,912 B` external capacity charge under a `245,760 B` ceiling.

The geometry is intentionally split:

- `93 × 2,048 B = 190,464 B` is the known parent/psum/weight capacity geometry (`9 + 60 + 24` macro-equivalents);
- the remaining `24,448 B` is an identical conservative external capacity common charge for all three axes;
- that residual is not live state, not instantiated memory, has no physical macro count, and has no admitted numeric area/timing/energy model.

Rounding the residual to 2-KiB macros would require 12 instances and 128 B of padding, but this is diagnostic arithmetic only. The contract explicitly forbids calling those macros implemented or calling 93 macros a complete `214,912 B` realization.

Logic-only DC, external memory and total results remain separate. The logic-only boundary includes compute/control/interface and axis-specific adapters but has zero storage macro area. In particular, a candidate top may not keep nine parent macros internally and then add the same nine-equivalent parent charge externally. Parent macro area, leakage and dynamic energy each appear exactly once.

Future common-memory parameters—technology, geometry, ports, latency, area/leakage coefficients and per-access energy coefficients—must have one identity across all axes. Actual read/write counts are still taken from each axis's address-timed trace and may differ; Path C must not erase traffic differences by forcing dynamic energy totals equal.

The frozen raw CPU same-ledger result remains `434,242,823` versus `763,908,050` cycles, or `1.7591725402×`. It is not RTL/mapped speedup, total/system speedup, PPA or energy. A change in external ports, schedule or latency requires replay.

Future matched aggregation is frozen symbolically:

- `A_total_axis = A_logic_axis + A_ext_common`;
- execution time comes from joint address-timed logic+memory replay;
- external dynamic energy uses actual per-axis reads/writes with common coefficients;
- external leakage energy is common leakage power times each axis's execution time;
- total energy adds logic, external dynamic, external leakage and a symmetric residual model term;
- throughput/mm² and speedup use the resulting total area and joint replay time.

Author validation passed 174 checks and rejected 22 mutations, including unequal axis charge, invented 105/12 macro claims, live residual, parent double counting, forced equal traffic, logic-only-to-total relabeling, raw-ratio promotion and EDA authorization.

`docs/359_DATE终局冻结_20260813.md` was not touched and remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
