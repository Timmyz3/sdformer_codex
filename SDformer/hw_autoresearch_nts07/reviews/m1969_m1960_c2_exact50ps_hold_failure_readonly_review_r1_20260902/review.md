# M1969 — M1960 C2 exact-50-ps hold failure review

Verdict: **PASS read-only audit; M1960 is permanently `FAILED_OR_INCOMPLETE_DO_NOT_CITE`.**

## What the run actually proved

- The attempt and failure quarantine both verify under their inner and outer SHA-256 seals.
- The execution census is exact: one authorized/observed license query, one authorized/observed `dc_shell`, and no retry.
- The frozen K8 input started at 130,822.775176 µm², setup WNS +0.00176597 ns, and hold WNS −0.0189998 ns with 29,351 violating paths under the 50 ps hold uncertainty.
- One `set_fix_hold` plus one hold-only incremental compile moved DC's printed estimated min-delay cost from −188.42 to `−0.00`.

That final optimizer cost is **not** a timing-closure receipt. The hard area check ran before post-hold timing reports and mapped netlist publication, so no post-repair machine timing summary exists.

## Why the hard failure is correct

| Quantity | Value |
|---|---:|
| Frozen baseline area | 130,822.775176 µm² |
| Frozen +5% ceiling | 137,363.913935 µm² |
| Post-optimization mapped leaf area | 141,886.71 µm² |
| Growth over baseline | 11,063.934824 µm² / **8.457193%** |
| Excess over ceiling | 4,522.796065 µm² / **3.457193 percentage points of baseline** |

The Tcl raised the terminal area error at this point. Although `dc_shell` itself returned zero after stopping the sourced Tcl, the outer runner fail-closed on the missing mandatory post-hold reports/netlists, exited 6, and sealed the failure quarantine. There is no canonical M1960 result.

## Paper boundary and next action

The surviving honest statement remains the M1877 diagnostic boundary: K8 setup is met (+0.001767 ns), while fast-min hold is open (−0.023259 ns, 30,442 violating paths); K1×8 was not run in that campaign. M1960 adds no citable timing, area, power, or PPA point.

Stop further buffer-only uncertainty sweeps. The earlier 70 ps attempt and this exact-50-ps attempt show the same physical tax; scalar retuning is no longer a credible closure path. Any successor should use a fresh identity and one of:

1. structural RTL repair of the high-fanout/short-path state cones, with VCS and Formality;
2. placement/clock-tree-aware implementation and useful-skew analysis;
3. post-CTS/post-route hold repair with extracted parasitics and complete setup/hold/DRC/area/power accounting for both K8 and equal-bandwidth K1×8.

Do not cite `−0.00` as hold closure, do not admit 141,886.71 µm², do not retry or overwrite M1960, and do not relax the +5% gate to manufacture a pass.
