# M1018 — M1013 C2 mapped-gate SAIF compile-failure audit

**Verdict: M1013 is consumed and must not be retried. Only an additive environment repair is allowed.**

Both the attempt directory and failure quarantine verify against their inner manifests and outer seals. M1013 reached `COMPILE_k1`, returned 1, and was quarantined. Its sole compile log contains `Cannot find vcsMsgReport script in /bin; make sure VCS_HOME is set`. No `simv`, simulation PASS, SAIF, or canonical result exists. Therefore the measured boundary is zero completed gate simulations and zero SAIF files.

The root cause is launch-environment completeness, not RTL, mapped-netlist, numerical, or SAIF-window behavior. The clean caller environment supplied `/usr/bin:/bin`, while the runner invoked the absolute VCS binary without exporting its installation root. VCS consequently searched `/bin` for `vcsMsgReport`.

M1015’s prior GO verdict is retracted for the consumed M1013 attempt. M1015 correctly exercised pre-attempt SHA, seal, status, namespace, and collision guards, but—because it deliberately ran no EDA—did not emulate VCS startup under `env -i` and missed the absent `VCS_HOME`. Its sealed evidence remains untouched as an audit trail.

The only permitted successor is a new runner and namespace that export and exact-pin `VCS_HOME` plus `vcsMsgReport`, followed by a new independent hammer. This audit does not authorize VCS, PT, PTPX, DC, GPU work, power, energy, or system claims.
