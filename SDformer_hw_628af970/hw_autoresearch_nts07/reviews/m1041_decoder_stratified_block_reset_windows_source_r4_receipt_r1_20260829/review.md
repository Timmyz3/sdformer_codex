# M1041 decoder stratified block-reset source r4 receipt

Status: `PASS_M1041_R4_SOURCE_ONLY__M1042_INDEPENDENT_HAMMER_REQUIRED`.

M1041 closes the M1035 recursive value-shape hole without modifying M1034 r3 or the frozen M1023 selector/reset path. It reuses the r3 publication constructor, changes only the public schema identity, and validates the completed envelope with an additive strong type.

Every bound is now an exact JSON list containing two finite non-boolean scalars in nondecreasing order. Every uncertainty field is an explicitly allowed finite scalar with a checked range. Coverage rows require a known unique stratum, exact positive integer counts, `sample <= population`, and a finite fraction equal to `sample/population`. A recursive public-JSON walk rejects custom mappings, tuples, NaN, infinity, and non-JSON leaves.

All eleven M1035 escape attacks are author regressions and are rejected. Semantic cycle, mean, sum, estimate, speedup, FPS, throughput, latency, time, and runtime keys are rejected at every unapproved depth. Additional tests reject nested containers without semantic names, booleans or floats used as counts, invalid intervals, malformed uncertainty values, invalid coverage ranges, and inconsistent coverage fractions.

Fourteen tests, the synthetic self-test, static checker, source validator, and Python compilation pass. The additional regressions cover case/plural/camelCase semantic aliases and bind each state to its exact status/action pair. No real payload/window, runner, full row, EDA, GPU, or remote task ran. M1042 independent receipt-blind hammer remains mandatory; this receipt does not authorize execution release or paper citation.
