# M1816 independent M1811 runner hammer

Status: `PASS_M1816_M1811_C2_REGISTERED_FAULT_MATCHED_TWO_AXIS_DC_RUNNER_HAMMER__P0_0_P1_0_P2_0__ONE_EXECUTION_AUTHORIZED__NO_EDA`

Score: 99/100. Findings: P0=0, P1=0, P2=0.

## Verdict

The final M1811 runner SHA is
`6cac176e737a6393dfda2c81952e099a88689d08a5772a3d9b872022f305fff7`.
It pins the exact M1809 wrapper, 13-row filelist and all 13 source files, the
M1801 K8 identity, frozen M519 K1x8 identity, Tcl, SDC, and docs359. It verifies
both the sealed M1810 source review and this sealed M1816 runner review. M1816's
status, severity, exact runner SHA, exact M1810 review SHA, and exact
authorization dictionary are hard gates inside the runner.

Before consuming the attempt marker, the runner requires a fresh namespace,
rejects same-UID DC-family processes, checks memory and commit headroom, and
queries a Design-Compiler license. Its single DC command site is inside exactly
two iterations: K8 `ARCH_MODE=0` and K1x8 `ARCH_MODE=1`. No retry path exists.
Post-attempt failure is sealed and quarantined as
`FAILED_OR_INCOMPLETE_DO_NOT_CITE`; success is sealed and atomically published.

The output receipt remains explicitly raw and pending independent result review.
It cannot admit timing, area, power, energy, performance, system speedup, or
paper PPA by itself. The result hammer must inspect unexpected DC errors,
unresolved references, design/timing reports, hierarchy, provenance, and all
claim boundaries.

## Authorization

One execution of the exact runner is authorized, totaling exactly two
`dc_shell` runs and zero other EDA runs. Launch must pin:

```
M1811_EXPECTED_RUNNER_SHA256=6cac176e737a6393dfda2c81952e099a88689d08a5772a3d9b872022f305fff7
M1811_EXPECTED_REVIEW_SHA256=33cea13ec28ee9e6d05c18100b1a1322ddccbc3a24a4ad8eb1a0838f4b8c690b
```

Automatic retry is forbidden.
