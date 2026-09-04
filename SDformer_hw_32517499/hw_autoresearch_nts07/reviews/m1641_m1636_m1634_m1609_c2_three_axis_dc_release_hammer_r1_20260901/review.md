# M1641 independent review of M1636 three-axis DC release

Status: **PASS; one M1634 launch is admitted with exact caller pins.**

The release, its two sidecars, the M1634 runner/source contract, and the complete M1635 review tree all match their exact hashes and pass inner/outer seals. The release grants one attempt containing exactly three `dc_shell` executions: fresh K1, K8, and equal-bandwidth K1×8. All axes use the same common top, 12-source filelist, Tcl, SDC, slow/fast libraries, 3 ns clock, uncertainties, ideal-clock/ZeroWireload assumptions, and logic-only zero-macro boundary; only `ARCH_MODE=0/1/2` varies.

No M872 DDC or mapped netlist is read or copied. M872 is sealed provenance only. The filelist selects M1609 as the unique compactor definition and excludes frozen M214, so every freshly mapped axis contains the registered-fault seam. Each axis executes one non-incremental `compile_ultra`; VCS, Formality, PT, PTPX, SAIF, GPU, and remote execution are not released.

The runner verifies M1635 and M1636 before caller pins, verifies both exact pins before namespace or lock creation, consumes one attempt before the first DC process, has no attempt deletion or retry path, quarantines incomplete work, and publishes with no replacement. Hold remains diagnostic only. This run does not refresh the frozen 1,913/1,945 directed component cycles and creates no system-speedup or headline claim.

The unpinned static preflight passed all identity/seal gates and failed closed at the missing runner pin with exit code 3. It created no attempt, work, result, lock, or quarantine and launched no DC. Independent checks passed 12/12 under CPython 3.6 and 3.10; 25 release plus 25 runner mutations were rejected under both runtimes.

Authorized invocation parameters are:

`M1634_EXPECTED_DC_RUNNER_SHA256=da9cd0d118021eb85c8b548d93f6779ec6d25b6fec7ca5894bdae988a95840b7`

`M1634_EXPECTED_DC_RELEASE_SHA256=0b1945b7060e5b2af9557ceb4b72f5c0a1fb862af48534c3abc59669cbfa5088`

The runner should be invoked through `bash` with both variables set. Any resulting directory remains non-citable until a fresh different-author result hammer passes.
