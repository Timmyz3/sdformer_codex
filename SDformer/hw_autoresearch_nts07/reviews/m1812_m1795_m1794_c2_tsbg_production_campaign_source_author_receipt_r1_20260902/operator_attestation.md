# M1812 source-author attestation

M1812 adds only campaign governance and semantic mutation checking around immutable M1794/M1795. I did not modify either predecessor or docs/359, and I did not query a license, run VCS/simv/DC/PTPX, create an attempt, create a result, or create M1814.

The future runner requires exact external pins for itself, the M1812 contract, all three M1813 review authorities, and all three M1814 release authorities. M1814 must semantically bind the runner, source contract and both seals, review JSON/manifest/outer seal, M1794, M1795 and docs/359. It also fixes the all-false prelaunch boundary, directed-only workload boundary, exact one-query/one-compile/one-sim budget, unique namespaces, no retry, same-UID/shared-queue exclusion, and no-replace result or post-attempt failure-quarantine publication.

CPython 3.6 and 3.10 each rejected 48/48 in-memory semantic mutations. Contract SHA mismatch is not counted. The attacks cover real bank3 accepted identity and 16-lane payload capture/replay, zero acceptance and sticky faults, both three-clock resets and full legal recovery ledgers, the SVA 1–8 reset/terminal sequence, and every release/governance boundary.

This package is still source-only. A fresh different-author M1813 review with P0=0 and P1=0 is required before M1814 may be created or the single directed VCS attempt may run.
