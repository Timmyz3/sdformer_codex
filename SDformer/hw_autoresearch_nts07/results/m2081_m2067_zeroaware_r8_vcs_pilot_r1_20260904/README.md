# M2081 zero-aware VCS pilot

Synopsys VCS compiled the R8 candidate once and ran two directed frozen workloads.

- Slot 0 is nonzero and retains positive ordinary/TSBG address coverage with zero numerical mismatch.
- Slot 5 is the R7 failure case. Its frozen nonzero-code count is zero; both engines correctly issue zero weight-memory requests while still retiring 96 commits and 1,536 exact checks per axis.

This pilot validates the checker correction only. It is not the 960-workload production result, not an admitted speedup, and not a system or energy result.
