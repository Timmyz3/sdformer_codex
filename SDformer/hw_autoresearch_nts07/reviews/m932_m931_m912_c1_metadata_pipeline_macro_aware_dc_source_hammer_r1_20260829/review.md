# M932 independent source hammer: M931 C1/M912 macro-aware DC

## Verdict

**PASS 100/100, P0/P1/P2 = 0/0/0.** The final M931 identity may consume exactly one Synopsys DC attempt after a separate launch release binds this review. The reviewer did not run DC, VCS, any other EDA tool, a license query, GPU work, or a remote command.

The frozen source point is narrow: M912 metadata-pipelined C1 functional RTL, the M528 nine-macro 1RW adapter, 3.000 ns ideal-clock/ZeroWireload constraints, TSMC 28 nm standard-cell and SRAM slow/fast pairs, one `compile_ultra`, and zero incremental compile. M929 remains the functional UNIT_DELAY VCS authority only; it is not timing, speedup, PPA, energy, or system evidence.

## What was independently closed

- Contract SHA `9e617a...eab7`, runner SHA `9c4a43...5421`, both contract seals, all exact-file hashes, `docs/359`, and the recursively sealed M929 authority recompute.
- `dc_shell`, `lmutil`, license, standard-cell slow/fast DB, macro slow/fast DB, and the macro asset manifest match the contract exactly.
- The DC filelist contains only the adapter and M912 RTL. The foundry behavioral macro `.v` is absent. One adapter generates exactly nine `TS1N28HPCPHVTB128X128M4S` cells, and Tcl requires nine before and after compile.
- The SDC is the frozen 3 ns point. Reset is the only false path; no debug path is exempted. `SYNTHESIS`, `ssg0p9v125c`, `ZeroWireload`, paired min libraries, and the precompile loop gate all precede the sole `compile_ultra`.
- The runner checks exact identities before resource admission, scans same-UID DC/common-shell processes, owns its lock explicitly, checks both licenses before consuming the permanent attempt, quarantines every post-attempt failure, captures the DC pipeline status, seals staging, and atomically promotes only a raw result.
- An isolated bad-argument branch proved that a non-owner cleanup leaves a foreign lock intact. It exits before tool, license, or EDA access.

## Capacity wording

The final contract now separates the layers correctly. The logical parent payload is 9,216 B; the nine bound 128x128b macros contribute 18,432 B of physical capacity. The same-ledger total capacity obligation is 213,376 B, leaving 194,944 B not physically macro-bound in this DC top. Therefore this attempt cannot claim that the complete 240 KiB store is integrated or that its result is paper-ready PPA.

## Fail-closed boundary

The runner may preserve a completed negative physical measurement as a sealed **raw** directory; its receipt fixes `independent_result_hammered=false`, `setup_admitted=false`, and every PPA/speedup/system/headline flag to false. It cannot self-admit a result. A separate independent result hammer must inspect the complete DC log and reports, verify all nine macros and artifacts, require setup WNS >= 0 for admission, and recheck the seals before any citation.

No fair zero/bit RTL baseline is present in this package. Even a setup-clean result cannot turn the CPU-ledger 1.74x estimate into RTL or system speedup without a separate same-resource trace bridge.
