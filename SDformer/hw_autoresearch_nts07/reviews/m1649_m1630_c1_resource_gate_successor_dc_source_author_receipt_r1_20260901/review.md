# M1649 C1 resource-gate successor — source author receipt

Status: **PASS source-only authoring; M1650 different-author review required.**

M1649 is not a new physical optimization. It is an execution-authority successor to exact M1630. It reuses the M1630 Tcl byte-for-byte and binds the exact M1630 runner, source contract, M1631 review and M1632 release as immutable predecessors. The original admitted M993/M1006 DDC, tools, libraries, 3 ns clock, 0.200 ns setup uncertainty, one 0.051 ns optimization-only hold pass, restored 0.050 ns reported hold uncertainty, nine SRAM macros, five-percent area ceiling, setup/hold predicates, zero-DRC predicate and no-retry policy are unchanged.

The only scheduling change is `CommitLimit-Committed_AS >= 50,331,648 KiB` (48 GiB), down from 67,108,864 KiB. The five-minute M1630 observation sampled ten times at no more than 30-second intervals: commit headroom remained 64,543,320–66,055,868 KiB, no same-UID DC existed, and MemAvailable was approximately 389,803,916 KiB (371.75 GiB). Thus the former 64 GiB commit-only floor rejected a host with ample resident-memory margin because unrelated shared-host reservations counted in `Committed_AS`. The successor still requires at least 100,663,296 KiB MemAvailable, 16,777,216 KiB SwapFree, zero same-UID DC, a valid license, and every exact tool/input/seal identity.

M1649 uses fresh result, attempt, work, lock, M1650 review and M1651 release namespaces. The runner remains fail-closed until both a different-author M1650 review and a separately sealed M1651 release exist and the caller pins both exact hashes.

CPython 3.6 and 3.10 each passed 18/18 static tests and rejected 16/16 resource-boundary mutations. Bash syntax and the contract inner/outer seal pass. No DC/EDA, attempt, result, release, retry, remote action, or protected-file modification occurred. Only M1650 source review is authorized next.
