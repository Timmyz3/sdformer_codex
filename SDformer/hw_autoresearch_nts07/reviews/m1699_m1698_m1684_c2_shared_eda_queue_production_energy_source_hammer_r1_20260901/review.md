# M1699 independent hammer of M1698 C2 shared-queue energy source

Status: `FAIL_M1699_M1698_C2_SHARED_EDA_QUEUE_PRODUCTION_ENERGY_SOURCE_HAMMER__NO_M1700_RELEASE__RUNTIME_SOURCE_AND_M1686_GATES_REQUIRED`

Score: **88/100**; P0/P1/P2 = **0/2/1**. This was a source-only review. No license query, VCS, simulation, SAIF generation, PrimeTime PX, attempt, result, or release was run or created.

## What closes cleanly

The M1685 queue finding is structurally repaired. M1698 uses the common `/tmp/date_dual_synopsys_same_uid_eda_queue.lock`, holds its exclusive flock through the production campaign and publication, rescans ancestry-aware same-UID collisions after locking, and invokes the same collision gate immediately before each VCS and PrimeTime-PX subprocess. Four independent queue mutations were rejected, and the extracted ancestry helpers accepted the current runner while rejecting PID 1 as external.

The fresh M1661/M1677 mapped chain, two axes, five cases, 3 ns clock, 261 accepted sources per axis, and future count budget of 2 VCS compiles + 10 simv + 10 SAIF + 10 PTPX remain intact. Attempt consumption precedes the first VCS launch and no automatic retry exists. Current execution sources contain neither active `force` nor `initreg`.

## Release-blocking P1: execution sources are not bound at launch

The clean-source scan happens only in the author checker. The launch-capable runner exact-checks the M1684 contract file but never traverses its `source_files` inventory. Consequently the assertion module, wrapper, UCLI Tcl, PTPX Tcl, and both VCS filelists can drift after review and before production while the runner still passes. This is a transitive-identity error: hashing a manifest does not hash its members unless the runner verifies them.

A successor must exact-check every M1684 execution member inside the launch-capable runner before admission and attempt consumption. This preserves both the active-force ban and exact workload geometry at the instant of execution.

## Release-blocking P1: M1686 denial is not permanent

M1686 is absent now, but M1698 has no runtime path or `os.path.lexists` gate for it. The checker uses `exists()` only during author review, which also misses dangling symlinks. The successor and future release author must reject the M1686 JSON, digest sidecar, and outer-seal sidecar before consuming the attempt.

## P2: inline Tcl force bypass

The Tcl scanner recognizes only line-leading `force`. It accepts the active command `if {1} { force dut/q 0 }`. The repair should conservatively recognize Tcl command separators and active brace bodies, with inline, semicolon, comment, and string tests.

M1700 is not authorized. M1686 remains forbidden. Only a newly numbered additive runtime-binding repair source is authorized.
