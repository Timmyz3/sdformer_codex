# M2218 independent source hammer

## Verdict

**FAIL — M2219 is not authorized.** Score: **88/100**; severity: **P0/P1/P2 = 1/0/0**.

The core experiment design is substantially sound. I independently rebuilt all 2,880 pre-power rows and reproduced the three frozen representatives, their descriptor hashes, and their one-third weights. The representatives come from three different DSEC sequences, and their identities are selected from reuse density rather than measured power, energy, PPA, or cycle values. The two production axes use one DUT each and differ only in `SCHEDULE_MODE`; the six measurement and six diagnostic SAIF boundaries, two fresh DC maps, six PTPX points, PT annotation/critical-cone gates, and equal 288-KiB/16-macro SRAM model are present.

Production remains blocked by one decisive source-identity defect.

## P0 — transitive admission code is not frozen

`parse_m2217_ep34_tsbg_matched_power.py` imports and executes:

- `parse_m2172_m2018_ordinary_native_saif_balanced_scope_preflight.py` (`42fd87d...`) for file seals, balanced SAIF parsing, hierarchy, and conservation;
- `parse_m2117_m2018_tsbg_rtl_saifmap_power.py` (`2787e885...`) for transformation maps, annotation, switching coverage, critical cones, and power arithmetic.

Neither file appears in the contract's 26-entry `source_inventory`. The production runner validates only those 26 entries. Therefore either helper can change after M2218 without tripping the M2219 source gate, changing exactly the semantics that decide whether power evidence is admitted.

This is not a numerical-result defect and does not invalidate the chosen workloads. It is nevertheless a P0 authorization defect: the independent review cannot freeze an incomplete executable dependency closure.

## Verified surface

- Selection: 2,880 rows, 960 per stratum, three sequences, exact descriptor and ledger hashes, fixed 1/3 weights.
- Matched axes: same inputs, ports, cache, clock, PVT, single DUT, and same testbench; only `SCHEDULE_MODE=0/1` differs.
- Activity: DUT-only scope, 93,971 records, measurement `TX=0`, exact duration and T0/T1/TX conservation; diagnostic SAIF is never annotated.
- Physical flow: two fresh DC maps and six PTPX points; no M2203 raw reuse.
- PT gates: ≥95% direct-net and fully annotated-leaf coverage, ≥20% nonzero-toggle coverage, and eight live protocol cones.
- SRAM: identical 294,912 B, 16 macros, 558,507.032 µm², and leakage on both axes; dynamic energy uses actual accepted bank activations at the conservative 22.213-pJ deep-segment value. The 3.826774-mW leakage value remains labeled as a mixed-corner proxy.
- Tests: M2217 unit tests 9/9 PASS; independent mutations 12/12 PASS; static runner/parser PASS.
- Execution: no VCS, DC, PTPX, license, GPU, or Git action; M2219 result/attempt/lock remain absent; `docs/359` is unchanged.

## Required successor

Do not run M2219. Create a fresh source identity that pins both imported helper files in the contract and production review binding, rejects either helper's drift, and undergoes a new independent source hammer. M2217 files and prior raw evidence must remain unchanged.
