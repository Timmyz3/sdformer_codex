# M1221 C1/R9 one-shot UNIT_DELAY VCS release author review

## Verdict

The release source is ready for a fresh different-author M1222 hammer. It does not authorize VCS or EDA now.

M1219 bounds the four waits implicated by the consumed M1213/R8 timeout and adds flushed phase observability without changing DUT RTL or SVA. M1221 freezes a new R9 filelist and a disjoint one-shot execution namespace around that source.

## Execution contract

After a fresh M1222 recursive double seal is runtime-bound, the runner may consume one M1221 attempt, invoke exactly one foundry `UNIT_DELAY` VCS compile, and invoke exactly one simulation bounded to 1800 seconds. Automatic retry is forbidden.

Success requires all seven phase enter/complete pairs, all 24 indexed random-transaction enter/complete pairs, zero `TIMEOUT_M1219R9` lines, four exact coverage lines, and one exact PASS line. Failure after work creation emits `phase_watchdog_timeout_dump.txt`, records compile/simulation exit codes, recursively seals the incomplete work, and moves it to a disjoint quarantine. The attempt remains consumed.

## Evidence

- Static source gate: 74/74 PASS before author sealing.
- Local mutation tests: 9/9 PASS.
- M1219 canonical checker: PASS.
- M1218 failure, M1219 author, and M1220 hammer recursive seals: exact.
- M1220 triple: review `7004b6f3...`, manifest `d3a06420...`, outer file `fc05610e...`.
- New attempt/result/work/quarantine namespaces are fresh and disjoint from consumed M1213/R8.
- TB, checker, filelist, M528/M935/M1162 RTL, SVA, foundry model, Python, and VCS binary are exact-pinned.

No VCS, simv, EDA, GPU, or network work was executed during this authoring step. Functional VCS, timing, cycles, performance, power, PPA, and paper admission remain false. `docs/359` remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
