# M2218 independent source review request

Please independently review the frozen M2217 source chain. No EDA execution is authorized by this request.

Required checks:

- Rebuild the 2,880-row pre-power population and verify the low/median/high selections, identities, descriptor hashes, and fixed one-third weights.
- Verify two separately elaborated single-DUT axes and DUT-only native SAIF with reset-separated diagnostic and measurement windows.
- Verify all six measurement ledgers, exact T0/T1/TX conservation, TX=0, nonzero critical activity, and no diagnostic SAIF annotation.
- Verify two fresh matched DC maps and six matched PTPX points, with no netlist or raw activity reuse.
- Verify identical 288 KiB / 16-macro SRAM capacity, area, and leakage across axes; SRAM dynamic must use actual accepted bank activation.
- Verify logic, SRAM dynamic, SRAM leakage, and component-model totals remain separate and the mixed-corner caveat is preserved.
- Verify the execution budget is exactly 2 VCS compiles, 6 simulations, 6 admitted measurement SAIF files plus 6 diagnostic-only SAIF files, 2 DC runs, and 6 PTPX runs, with no automatic retry.
- Verify M2204 is methodology-only and no M2203 raw result is reused.
- Verify post-read/selective-bank-fill, full-network, FPS, silicon, and energy/frame claims are excluded.
- Verify `docs/359_DATE终局冻结_20260813.md` remains SHA-256 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

Production authorization requires an independently sealed M2218 result with score at least 95/100 and P0/P1/P2 all zero. There is one M2219 attempt and no automatic retry.
