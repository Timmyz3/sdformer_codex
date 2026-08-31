# M1010 independent M1007 source hammer

Verdict: **GO to author the full-replay launch/runner only (96/100, P0/P1/P2 = 0/1/0).** No 51.84M-row replay, VCS, or EDA was run.

The M1007 source/checker/tests/contract and all frozen M504/M505/M528/M410 identities match their pinned SHA-256 values. The M1000 authority and M1007 source receipt both verify as symlink-free flat exact sets with valid outer seals. `docs/359` remains `dedde7ce...`.

Independent execution closes the source-level positives: checker PASS, 9/9 tests PASS, four parent cases match frozen M505 exactly in cycles, reads, writes, forwarding, issue, and stalls, and every parent cycle has at most one READ or WRITE. Three-design common-charge accepts logical equality with timestamp shifts while retaining `cycle_merge_pending=true`; deleting one transaction from each of candidate, strongest-zero, and same-coordinate-bit is rejected. Missing design and incomplete service specifications are also rejected. Paired-psum 1RW conflicts, weight conflicts, half-slot overlap, and incomplete coverage all block capacity admission.

The streaming path is lazy and memory bounded: production reads at most 576 ledger bytes per tile and materializes at most one tile trace, with a static event bound of 1,160. A tiny independent replay exercised the `pread` path without touching the frozen full ledger.

One P1 must be repaired in the future runner: `packing_summary` currently trusts a caller-supplied `coverage_complete` boolean. Passing `true` with empty synthetic traces admits the 214,912-B capacity hypothesis. The runner must derive coverage internally from exact frozen-row conservation, all 10x4x432x3000 rows, all eight blocks, and completed three-design service merges; it must not accept a naked coverage flag.

The frozen `435,293,339` cycles and `1.7467534301x` remain M505 CPU same-ledger references only. This milestone admits neither 214,912 B, matched total cycles, RTL/system speedup, nor PPA.
