# M2179: M2178 produced timed but empty SAIF containers

## Technical summary

M2178 is a correctly isolated permanent failure, not a power result. Its one authorized license query, VCS compile, and ordinary simulation ran exactly once; the RTL workload passed with 20,292 cycles and 29,472 products. Both emitted SAIF files nevertheless contain only headers and `DURATION`: **0 INSTANCE, 0 NET block, and 0 activity record**. The parser stopped at the first diagnostic file, admitted neither file, created no canonical result, quarantined the evidence, and disabled retry.

The high-confidence cause is the missing UCLI monitoring policy. M2178 executes `power <dut_ordinary>` and `power -enable`, but never executes `power -gate_level ...`. The installed VCS documentation prescribes an explicit UCLI gate-level policy for MDA and SystemVerilog objects. A sealed local preflight and a same-M2018 workload both generate nonempty SAIF under the same `-debug_access+r` compile surface when `power -gate_level all mda sv` precedes scope selection.

**Verdict: FAIL, 93/100; P0/P1/P2 = 1/0/0. M2178 is permanently consumed and must not be retried. Only fresh M2185 source authoring is allowed.**

## What ran and what failed

| Gate | Evidence | Interpretation |
|---|---:|---|
| Attempt | one exhaustive sealed member | M2178 identity was consumed once |
| Quarantine | 14 exhaustive sealed members | failure evidence is complete and immutable |
| VCS compile | completed; `simv up to date` | compilation is not the failure |
| RTL runtime | PASS; 149 rows, 1,278 issues, 29,472 products | DUT execution and workload activity exist |
| Prehistory SAIF | 355 bytes, 1,167.01 ns, 0 records | timed empty container |
| Measurement SAIF | 356 bytes, 60,876 ns, 0 records | timed empty container |
| Parser | `target INSTANCE dut_ordinary count 0 != 1` | correct first fail-closed gate |
| Admission | 0 diagnostic candidates admitted; 0 measurement files admitted | no activity or energy claim is legal |

The correct `DURATION` values show that enable/disable/report timing progressed. Duration alone does not mean that any object was monitored.

## Why this is not “the DUT never toggled”

The functional runtime completed the exact ledger, including 29,472 products, 14,304 scalar reads, 1,788 bundles, and 24 commits. More decisively, an earlier same-M2018, same-60,876 ns, `-debug_access+r` run emitted **93,971 records** after using `power -gate_level all mda sv`. That older campaign was later rejected for unknown activity in 223 records, so it is not power evidence; it is valid diagnostic evidence that this RTL hierarchy has monitorable activity.

Thus “DUT activity absent” is ruled out. The failure is the collection surface, not computation.

## Why monitoring policy is the leading cause

The M2178 UCLI script starts with:

```tcl
power tb_m2160_m2018_ordinary_native_saif_report_reset_preflight.dut_ordinary
power -enable
```

It omits the command present in all inspected successful local SAIF flows:

```tcl
power -gate_level all mda sv
```

The installed VCS V-2023.12-SP1 SAIF help describes the UCLI order as gate-level policy, region selection, enable, run, disable, and report. Its SystemVerilog section specifically requires the `sv` keyword; the general section identifies `mda` for multidimensional arrays. Consistently:

- M2178 files say only that no explicit gate-level-monitoring command was issued and contain zero hierarchy.
- The exhaustive sealed M1046 control uses the command, the same `-debug_access+r`, and emits 18 records.
- The same-M2018 diagnostic uses the command, the same `-debug_access+r`, and emits 93,971 records.

The official SystemVerilog help also recommends `-debug_access+pp`. That is not the minimal repair here: two local controls, including the same M2018 RTL, prove `-debug_access+r` can emit the required hierarchy in this installed release. Changing both UCLI policy and compile access would confound the successor experiment.

## Scope-command alternative

No scope-resolution error appears in the UCLI log. The elaborated top is exact, and the same fully qualified DUT path is used for region selection and both reports. This makes a scope typo unlikely. However, an empty SAIF cannot independently prove that the region-selection command took effect. Therefore M2185 must preserve the parser's exact requirement of one `dut_ordinary` INSTANCE and 93,971 owned records; it must not infer scope success merely from absence of a warning.

## Minimal successor, not a retry

Use a fresh chain: **M2185 source → M2186 independent source hammer → M2187 one-shot → M2188 result hammer**. M2187 is not authorized by this review.

M2185 should create a new UCLI file and new runner/contract identities. It must not edit or reuse M2178. Relative to the frozen M2160 UCLI, the only protocol change should be the first effective command:

```tcl
power -gate_level all mda sv
power tb_m2160_m2018_ordinary_native_saif_report_reset_preflight.dut_ordinary
power -enable
```

Keep the exact M2160 testbench, workload slot 42, fixture, one ordinary `SCHEDULE_MODE=0` frontend, report-before-reset ordering, M2176 parser, and `-debug_access+r`. M2186 must verify the one-line semantic delta and seal the new source before any license query.

## Limits and next gate

- This review ran no VCS, license query, simulation, SAIF acquisition, DC, PT/PTPX, ICC2, or GPU job.
- M2178's VCS/runtime PASS does not rescue either empty SAIF.
- M2139 is used only as a same-RTL monitoring diagnostic, not as admitted power evidence.
- Even a future nonempty M2187 SAIF remains an acquisition candidate until M2188 independently validates hierarchy ownership, TX=0, conservation, critical toggles, duration, reset separation, and all file seals.

No additional question is needed before M2185 source authoring; execution remains blocked pending M2186.
