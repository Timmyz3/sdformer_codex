# M1318 — independent hammer of failed M1311/M1288 PT result

## Verdict

**FAIL_DO_NOT_CITE — diagnostic only — M1311 no retry.** Both quarantine directories are internally and externally sealed, and the independent hammer passes 8/8 checks. The reports are trustworthy as failure diagnostics, but they are not a MET result and cannot support timing closure or paper PPA claims.

## Why this is physical nonclosure, not a PT process failure

PrimeTime returned `0`, the runtime monitor returned `0`, the internal completion marker is present, and all required reports exist. The raw log contains a nonterminal PT-063 Library Compiler-path diagnostic, which the outer fail-closed runner conservatively converted to exit 26 and quarantine. Independently of that diagnostic, the generated timing reports fail the admission gate:

| Gate | Measured result | Verdict |
|---|---:|---|
| Setup WNS | −0.001154 ns | FAIL |
| Global setup | 16 violations, TNS ≈ −0.01 ns | FAIL |
| Hold WNS | −0.022628 ns | FAIL |
| Global hold | 10,047 violations, TNS −101.91 ns | FAIL |
| Recognized unconstrained paths | 0 | PASS only for this sub-gate |
| Output coverage | 132 `out_setup` and 132 `out_hold` checks untested | FAIL |

Coverage rows are exact: setup `10573/10557/16/0`, hold `10573/526/10047/0`, out_setup `607/475/0/132`, and out_hold `607/475/0/132` in total/met/violated/untested order. `check_timing` succeeded and the M1311 parser recognizes zero unconstrained paths, but that does not repair negative slack or incomplete output coverage.

## Authority and claims

The single M1288 and M1311 attempts are consumed. No retry or alternate namespace is authorized. This review ran no EDA tool and made no license query. The quarantines must not be renamed or presented as canonical MET results.

Permitted use is limited to a diagnostic statement: the exact M917 Fixed-T10 netlist, under the frozen 3 ns pre-layout ideal-clock/ZeroWireload/zero-macro setup, does not close the strict setup, hold, and coverage gates. Power, energy, speedup, system, paper-ready PPA, and headline claims remain false.
