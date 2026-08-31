# M530/M528 DW1RW r2 source-static hammer

Verdict: **FAIL, 83/100, P0/P1/P2 = 0/3/2.** This review authorizes no VCS launch.

The two original RTL correctness repairs close under static inspection. A
nonzero synthetic parent-only payload is rejected by the combinational
preaccept predicate before ready/accept/commit/scratch/elision/current-beat
counters, and the dedicated atomic assertions no longer inherit the imminent
fault disable. Stale or nonmatching parent-slot data is masked, and final
overflow debug/fault requires a matching authoritative response. The frozen
matcher, ordering, one-lookahead, no-consume-credit queue, dead-write-only
policy, arithmetic widths and nine-macro organization did not drift.

Three launch-blocking verification/contract gaps remain:

1. `oracle_macro_reads`, `oracle_forwards` and `oracle_deadline_holds` are
   accumulated directly from DUT debug pulses. The stall oracle also consumes
   DUT ready. These are counter mirrors, not an independent deterministic
   microevent oracle.
2. `stalled_raw_recovery` remains high forever after any stalled RAW and can be
   satisfied by an unrelated forward in a later cycle or task. The runner does
   not consume the bounded SVA cover.
3. The future runner does not reject all forbidden non-VCS admission counts;
   notably `pt_runs` is absent, as are iverilog/Verilator/remote and the generic
   CPU count.

Minimum repair is a cleanroom cycle oracle for read/forward/hold/stall, a
consumer/parent/epoch-matched bounded stalled-RAW recovery cover, and a strict
one-VCS/all-other-zero admission schema check. M530 r2 must remain a sealed
failed identity; a new repair identity needs a new admission and fresh static
hammer.
