# M1020 repaired C2 mapped-gate SAIF release hammer

**Verdict: GO for exactly one M1022 mapped-gate VCS plus SAIF attempt.**
Score 100/100; P0/P1/P2 = 0/0/0.

The M1013 attempt and failure quarantine remain sealed.  M1018 proves that
M1013 stopped in `COMPILE_k1` before creating a simulator, completing a gate
simulation, or writing SAIF, and explicitly forbids retrying M1013.  M1022 is
an additive namespace bound to runner SHA
`dbaa5b0b9619cb60b556a42f27e9e926a56bcb22d4627c13048b70a3fc74af1b`.

The repaired environment was exercised, not merely inspected.  Exact clean
environment `vcs -full64 -ID` returned zero and identified
`VCS V-2023.12-SP1_Full64` on `linux64`; it did not search for
`/bin/vcsMsgReport`.  `-full64` is essential because the installation contains
the linux64 compiler and is also the mode used by every M1022 compile.

The runner performs three fresh mapped-netlist compiles, K1/K8/equal-bandwidth
K1x8, followed by five frozen cases per axis.  All 15 case scripts dump DUT-only
SAIF.  VCS and `vcsMsgReport` identities, M1002/M1018/M1020 authorities,
collision checks, and namespace freshness precede the atomic attempt `mkdir`.

Independent faults removed the VCS_HOME export, selected a wrong VCS_HOME,
changed the support-script SHA, occupied the attempt namespace, and introduced
an active process named `vcs1`.  Every fault stopped before attempt consumption
or design compilation; the occupied sentinel remained unchanged.

Authorization is limited to one M1022 VCS mapped-gate plus SAIF attempt.  This
does not authorize PT, PTPX, DC, GPU work, power, energy, system speedup, or
paper-ready PPA claims.
