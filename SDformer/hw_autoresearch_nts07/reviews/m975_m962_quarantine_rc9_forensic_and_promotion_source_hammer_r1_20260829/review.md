# M975 — M962 quarantine rc9 forensic audit and copy-only recovery source

## Verdict

`PASS_M975_M962_QUARANTINE_FORENSIC_RECOVERY__GO_COPY_ONLY_PROMOTION_SOURCE`

The M962 synthesis itself completed. `dc_shell` returned zero, optimization
completed, the Tcl wrote its PASS terminal, nine SRAM macros survived before
and after compile, all mapped artifacts were written, and the tool reached
`quit`/`Thank you`. The runner's exit 9 is not a fatal, link, unresolved
reference, TIM-209, or OPT-150 failure.

The sole runner-regex hit is `dc.log:32`:

```
Error: Error during sourcing of /opt/synopsys/syn/V-2023.12-SP3/auxx/gui/dv/.synopsys_dv.tcl
```

Lines 34–37 identify the cause: `env -i` in runner line 250 omitted `HOME`,
while the optional Design Vision startup script read `::env(HOME)`. DC
continued for another ten minutes and returned zero. Thus the error is real,
but the runner's classification of it as fatal/link/loop evidence is false.

## Recovered physical point

| Field | Evidence |
|---|---:|
| Clock | 3.000 ns, ideal, setup uncertainty 0.200 ns |
| Setup | MET; WNS +0.001795 ns; TNS 0; 0 violating paths |
| Top-100 | 100 startpoints, 100 endpoints, 100 MET slacks, 0 violated |
| SRAM macro | `TS1N28HPCPHVTB128X128M4S`, 9 pre / 9 post / 9 mapped |
| Logic area | 41,912.009793 + 26,509.139132 um2 |
| Macro area | 78,825.243164 um2 |
| Total cell area | 147,246.392090 um2 |
| Net area | undefined under ZeroWireload |

This is setup/area evidence only. QoR also shows a diagnostic worst hold
violation of -0.09 ns and 9,992 hold violations. Hold is not signed off;
power and energy were not measured.

The mapped hard-macro capacity is 18,432 B, with 9,216 B logically addressed
by this wrapper. It is not the complete 213,376 B same-ledger storage model.

## Citation and recovery boundary

The underlying reports are synthesis-complete and physically interpretable,
but the current directory remains directly non-citable because its immutable
marker says `FAILED_OR_INCOMPLETE_DO_NOT_CITE` and the runner never created
its normal receipt. M975 therefore supplies an additive, copy-only recovery
source—not a completed promotion.

The future script copies the original quarantine byte-for-byte beneath
`original_quarantine/`, preserving its seal, log, rc9 marker, and runner-bug
provenance. It only adds a recovery receipt and provenance at a new identity.
It cannot execute until a separately double-sealed M976 one-shot release is
created. No promotion was executed in M975.

After promotion and an independent result hammer, the legal statement is:

> The macro-aware C1 component meets 3 ns setup in a 28-nm pre-route,
> ideal-clock, ZeroWireload DC point and occupies 147,246 um2 including nine
> 128x128-bit 1RW SRAM macros.

It remains illegal to call the CPU same-ledger `1.746753x` an RTL cycle
speedup. Hold signoff, power, energy, full-storage PPA, system speedup,
paper-readiness, and headline admission all remain false.

## Residual finding

`reports/check_design_postcompile.rpt` contains only `1`, so it is not an
inspectable postcompile diagnostic report. This does not negate the completed
mapping and setup/area reports, but it limits this result to a raw component
candidate. A later signoff flow must produce an inspectable post-link design
check; M962 must not be rerun merely to repair this artifact.
