# M2226 independent source-repair review request

Review only the additive M2225 repair. Do not execute VCS, DC, PTPX, license tools, GPU work, or Git mutations.

Required checks:

- Verify the M2172 helper SHA is `42fd87d6991c46366e80db1d08c20ec5e0d463f3bca8c6050673093d04f3bfe2` and the M2117 helper SHA is `2787e8858799577db8f87297d2d1c1c16ccf0a3933b00f9a039071e968ea3547`.
- Verify both helpers are members of the contract source inventory and the runner rejects either helper mutation before contract validation or EDA.
- Verify the future M2226 review identity must bind runner, contract, and both helpers; verify the future M2227 result emits both helper hashes.
- Verify DC mapping is SSG0P9V125C slow/max with FFG1P05VM40C fast/min, while PTPX is TT0P9V25C on the SSG-mapped netlist and is explicitly labeled mixed-corner.
- Verify the 22.213 pJ per actual accepted bank activation model and the 3.826774326764422 mW leakage proxy and their labels are unchanged.
- Verify the only production namespace is M2227/M2225 and it does not read, write, consume, or reuse M2219.
- Verify M2227 retains one attempt, no automatic retry, and the original budget of 2 VCS compiles, 6 simulations, 6 measurement plus 6 diagnostic SAIF files, 2 DC runs, and 6 PTPX runs.
- Verify no M2217 source file or `docs/359_DATE终局冻结_20260813.md` was modified; docs/359 must remain SHA-256 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

If and only if the score is at least 95/100 with P0/P1/P2 all zero, seal a review using status `PASS_M2226_M2225_MATCHED_POWER_SOURCE_REPAIR_RELEASE`. Its identity object must contain exact `runner_sha256`, `contract_sha256`, `m2172_helper_sha256`, and `m2117_helper_sha256` values. Production remains a sole M2227 attempt, with M2228 independent result review required.
