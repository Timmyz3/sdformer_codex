# M1021 — C2 mapped-gate SAIF environment-repair source receipt

**Verdict: the additive source chain is ready for an independent M1020 hammer; M1022 is not yet authorized to run.**

M1018 verifies the consumed M1013 attempt and quarantine. Failure occurred during K1 compilation because clean-env startup omitted `VCS_HOME`; no simulation or SAIF was produced. M1013 must not be retried, and M1015’s missed clean-environment coverage is recorded without altering its prior sealed evidence.

The new M1022 runner uses a fresh result and atomic attempt namespace. It exports `/opt/synopsys/vcs/V-2023.12-SP1` as `VCS_HOME`, fixes `PATH` to `${VCS_HOME}/bin:/usr/bin:/bin`, and exact-pins both `vcs` and `vcsMsgReport` before attempt creation. All other execution semantics remain three axes by five cases, fresh compile per axis, DUT-only SAIF, one shot, and collision-gated.

The M1019 release binds the M1001 contract, M1002 source hammer, M1018 failure audit, and exact M1022 runner. It reserves the independent M1020 directory and status. The runner requires the caller-pinned M1020 outer seal and cross-checks the hammer’s release, runner, M1002, and M1018 identities before consuming an attempt.

Static checker, `bash -n`, canonical JSON validation, and six unit tests pass. Tests reject a missing `VCS_HOME` export, incorrect support-script SHA, absent M1020 pin, stale M1013 namespace, and reduced axis/case geometry. No M1022 runner or EDA tool was executed.

This receipt authorizes only independent M1020 review. It does not authorize VCS, PT, PTPX, DC, power, energy, system-speedup, or paper-ready PPA claims.
