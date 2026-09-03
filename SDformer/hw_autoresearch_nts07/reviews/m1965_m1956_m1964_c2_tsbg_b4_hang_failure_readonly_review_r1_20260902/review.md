# M1965 — M1956/M1964 TSBG hang-failure read-only review

## Verdict

**PASS diagnosis, 97/100; P0/P1/P2 = 0/3/0.** M1956 is permanently `FAILED_OR_INCOMPLETE_DO_NOT_CITE`. The roughly 15-minute stop was conservative and correct. This review authorizes only a new additive TB source; it ran no EDA, queried no license, created no attempt, and authorizes no VCS run.

## Why the 15-minute stop was justified

M1964 observed `simv` PID 3970837 at 99.3--99.4% CPU with 120,676 KiB RSS. From 11:33:01 to 11:36:01 CST, CPU time advanced one-for-one with wall time while RSS, minor/major faults, and the 480-byte log did not change. The live log contained only the ASLR notice and VCS banner.

The sealed final log records:

```text
Interrupt at time 592502218500
```

At a 3 ns clock this is roughly 197.5 million clocks. The intended watchdog is 300,000 clocks, so this run advanced more than 658 watchdog windows. The reason no timeout fired is structural: the first watchdog is created only after `load_workload()` returns. Therefore it never covered the loading handshake that hung.

The owner-side interrupt produced exit code 130 and `retry=false`. The failure and consumed-attempt directories pass both SHA layers. The success directory, owned work directory, and launch lock are absent, and PIDs 3970837/3968146 are gone. This is a clean failure quarantine, not a result.

## Proven pre-watchdog deadlock

The initial thread calls `load_workload()` at TB line 616. The first watchdog fork starts at line 617. Inside the task, one shared `load_valid` drives both instances, and lines 483--484 wait for:

```systemverilog
do @(posedge clk_core);
while (!(base.load_accept && tsbg.load_accept));
```

This assumes two independently stateful DUTs accept the descriptor in the same clock. It does not latch an acceptance that happens on only one side. It also deasserts, prepares the next descriptor, and reasserts the shared valid in the active region of a DUT sampling edge. A split accept can therefore advance one side while the TB continues presenting the descriptor to both. The advanced side can then see `load_valid && load_ready && !load_accept`, which the RTL deliberately converts to sticky `ST_FAULT`. The simultaneous conjunction is thereafter impossible.

Because reset is finite, the full-load task contains no other unbounded operation, and simulation time advanced far beyond the absent watchdog, this is the primary and sufficient hang mechanism. The sealed run has no phase markers or waveform, so it cannot identify which side or descriptor split first. `SCHEDULE_MODE` does not alter the ST_LOAD acceptance equation; this is a TB dual-instance handshake defect, not proof of a schedule-mode RTL defect.

## Separate fail-closed option defect

The compile log says exactly:

```text
Warning-[SVAA-RNF] Invalid compile time argument to -assert
Switch "-assert global_finish_maxfail=1" is a runtime option.
This option cannot be given at compile time. Ignoring ...
```

Thus M1956 did not activate runtime SVA termination. A future runner must compile with `-assert svaext` and invoke:

```text
simv -assert global_finish_maxfail=1
```

It must also reject `SVAA-RNF`, ignored-option diagnostics, assertion/fatal text, nonzero exit, and a non-unique PASS token. This option alone would not cure the present hang because no SVA failure was reported; bounded liveness guards are still required.

## Minimal additive TB repair

Copy the exact M1942 TB to a new path and change only TB orchestration:

1. Give the base and TSBG instances separate valid signals. Keep one shared payload, stable until both sides independently accept it.
2. Latch `base.load_accept` and `tsbg.load_accept` separately. Drop only the accepted side's valid before the next sampling edge; never wait for a same-cycle conjunction.
3. Drive payload/valid away from active posedge, preferably on negedge or through a clocking block with explicit skew.
4. Add a bounded watchdog inside every descriptor load and a whole-test watchdog before the first load. Timeout diagnostics must include phase, context/group, per-side valid/ready/accept/pending, protocol state, and busy.
5. Replace completion `join_any` blocks with named forks plus `disable fork`, or an equivalent bounded wait for both done signals, so orphan watchdogs do not survive a phase.
6. Emit unique BEGIN/END tokens for reset, full load/execute, both attacks, recovery load/execute, and final checks, plus a timeout token.

RTL, adapter, SVA source, reference arithmetic, work ledgers, cache expectations, attacks, reset recovery, local directed gate, PASS token, and `docs/359` must remain frozen.

## Next gate

Only the additive TB source is authorized now. A different-author source hammer must approve it before any filelist or runner is written. The later runner must place the max-failure option on `simv`, obtain a new release and separate audit, and consume at most one fresh VCS attempt.
