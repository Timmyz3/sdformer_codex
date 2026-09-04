# M1051 independent M1040/M1016 full-replay result hammer

## Verdict

`PASS_M1051_M1040_M1016_C1_FULL_REPLAY_RAW_RESULT_HAMMER`

`ADMIT_RAW_CPU_CYCLE_OPPORTUNITY_ONLY__BLOCK_214912B_AND_SPEEDUP`

The sealed M1040 payload is internally reproducible, but it closes a negative
packing gate rather than a speedup admission. M1051 independently read all
51,840,000 frozen rows without importing or executing M1016, rebuilt all three
schedules through the earlier frozen M505 recurrence, and reproduced every
sample boundary, common-service digest, parent counter, raw cycle total, and
reported packing count.

## What was reproduced

Coverage was rederived as 10 samples x 4 operators x 432 partitions x 3,000
rows = 51,840,000 rows, or 17,280 phases, 812,160 row tiles, and 6,497,280
output blocks per design. Source addresses cover exactly `[0, 51,840,000)`.

The independently rebuilt logical common charges are identical for candidate,
strongest-zero, and same-coordinate-bit:

| Service | Count/design |
|---|---:|
| psum | 12,994,560 |
| weight | 70,853,184 |
| source | 51,840,000 |
| DMA | 1,476,108 |
| commit | 960,000 |

All three canonical receipt streams hash to
`a38589ba99715b0962fb88744c03dd6019a68c72bae35d3787ca9f48eb3680ea`.
Candidate parent conservation is also exact: 131,926,088 reads, 79,581,608
writes, 13,717,024 forwards, and 409,734,336 work cycles.

The raw conflict-unrepaired schedule recomputes to:

| Design | Raw CPU cycles | Candidate opportunity |
|---|---:|---:|
| candidate | 434,242,823 | 1.000000x |
| strongest-zero | 753,067,320 | 1.734208x |
| same-coordinate-bit | 753,067,320 | 1.734208x |

This is not the old M528 1.746753x point. M1040 adds the M1016 matched
common-service merge and changes both candidate and denominator. The legal
description is a new raw CPU-cycle opportunity on the frozen four-Conv scope.

## First-principles packing decision

The M1016 source counts only whether each newly appended event has the same
cycle as the last appended event for that packed macro group. Tasks can overlap
in time, so this is not generally a globally ordered collision sweep. The
`row` argument is unused as well, which means the task-local lifetime number
does not prove address-level absence of overwrite or alias hazards.

M1051 therefore independently materialized all 12,994,560 psum accesses per
design and sorted them by packed macro group and absolute cycle. For each of
the three designs the exact global result is:

- 329,816 distinct conflicting macro-group/cycle slots;
- 403,922 excess accesses beyond one 1RW operation per slot;
- 733,738 total accesses participating in those slots;
- maximum multiplicity 3.

Thus the published 403,922 count happens to equal the globally sorted excess
access count on this frozen trace. It is a real negative witness, not an
append-order artifact. However, it is not fed back into the pipeline cycles.
The 434,242,823 candidate cycles therefore do not execute on the claimed
paired-psum 1RW organization.

This invalidates pairing the raw cycles with the 214,912-byte capacity-only
packing. It does not prove that 214,912 bytes is mathematically impossible:
a conflict-aware reschedule might hide accesses in slack, while bank
duplication or an extra port might remove conflicts at a capacity/area cost.
Any such repair changes the coordinate and must be replayed.

## Minimum repair: M1056

The smallest useful successor is not another matcher. Add exact psum
group/address port calendars to the matched-service merger, globally arbitrate
overlapping tasks with a fixed deterministic 1RW queue/port policy, and
recompute all three designs under the same scheduling policy. Every delayed
read/write must feed task readiness, downstream work start, completion, and
sample commit. Simply adding 403,922 to the old cycle total is prohibited:
some conflicts may occupy existing slack while others may cause cascaded
stalls. The successor must conserve all 12,994,560 psum accesses per design,
prove at most one operation per packed 1RW group/cycle, include address-level
lifetime/overwrite checks, and recalculate 240-KiB capacity after any banking
or port change.

M1056 must report two separate gates. `capacity_bytes_pass` answers whether the
stored bits fit 240 KiB; `port_feasibility_pass` answers whether the chosen 1RW
organization can execute the schedule. A passing byte sum never overrides a
failing port calendar.

If this is not closed in the DATE window, use the lower-risk boundary: keep
psum/weight/source/DMA as identical external common charges, cite the promoted
C1 island implementation in the component table, and place 1.734208x only in
a separate CPU-model opportunity table with the conflict caveat.

## Claim boundary

Legal: the frozen 51.84M-row, four-bottleneck-Conv CPU model has equal logical
common charges and a conflict-unrepaired 1.734208x raw opportunity; its paired
psum audit finds 403,922 excess 1RW accesses and fails capacity admission.

Illegal: 214,912-byte capacity feasibility, executable/RTL speedup,
M528-1.746753x promotion, throughput/mm2 paired with component DC area, system
speedup, or paper-ready PPA.

No EDA, GPU, or remote job was run. M1040, M1016, its runner, and `docs/359`
were not modified.
