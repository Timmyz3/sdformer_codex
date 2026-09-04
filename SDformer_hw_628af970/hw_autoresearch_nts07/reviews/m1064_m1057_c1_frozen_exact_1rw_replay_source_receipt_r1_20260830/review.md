# M1064 frozen exact-1RW replay source receipt

## Verdict

`PASS_M1064_SOURCE_ONLY__M1065_REQUIRED_NO_FULL_REPLAY`

M1064 is an additive production-boundary repair after the M1057 STOP. It keeps
the M1056 one-port arbitration and cascade behavior, but removes caller control
over every coordinate that could make a future full result fail open.

## Frozen workload boundary

The future iterator has no arguments. It reads only the frozen M410 row file
and derives exactly 812,160 task IDs in sample/operator/chunk/partition order.
Rows are `task_id mod 64`; chunks 0--45 contain 64 rows and chunk 46 contains
56. Each task creates three explicit design receipts. Their task ID, row, row
count, shared preprocess and canonical M1016 common receipt must be identical;
only work cycles may differ.

The shared preprocess is the maximum of the three frozen M1016 preprocess
values. This removes the prior ability to compare different frontend schedules
under a nominally common task identity.

`FrozenCoverage` has a zero-argument constructor and admits only IDs 0 through
812,159 in order. It inserts ten 96,000-cycle commits internally. Completion
requires the exact service totals and the frozen M1016 digest
`a38589ba99715b0962fb88744c03dd6019a68c72bae35d3787ca9f48eb3680ea`.
Empty, duplicate, missing, reordered or partial streams cannot pass.

## Internally derived capacity

The production API accepts no capacity argument. Its frozen ledger is:

| Component | Bytes |
|---|---:|
| Packed psum: 4 groups x 15 macros | 122,880 |
| Weight: 24 macros | 49,152 |
| Parent scratch | 18,432 |
| Active bitmap | 1,152 |
| Descriptor ping-pong | 2,304 |
| FIFO/control reserve | 16,384 |
| Parent liveness class | 1,152 |
| Psum-valid sidecar | 1,152 |
| Source-mask ping-pong | 2,304 |
| Total | 214,912 |

The derived margin under 240 KiB is 30,848 B. This establishes only the byte
gate. Full-trace port feasibility, matched cycles, area and speedup remain
false until execution and independent result review.

## M1057 attack closure

All five M1057 failures are directed regressions:

- empty/common caller-relative services cannot pass;
- bool counts, duplicate JSON keys, extra keys and coverage booleans reject;
- three-design ID, row, preprocess, receipt or population mismatches reject;
- caller capacity values are impossible at the sanctioned API;
- only the exact canonical double-sealed contract path and hashes validate.

The checker also rehashes the 466,560,000-byte frozen row file. Fifteen tests
pass, and neither the full iterator nor any result namespace was used.

## Next gate

A different author must perform the receipt-blind M1065 hammer. Only then may
a separate M1066 runner/release authorize one CPU full replay. M1066 must use
`iter_frozen_task_records()` and `replay_frozen_sample()`; it may not call the
caller-configurable M1056 comparison API directly.

No full replay, EDA, GPU or remote work ran. M1016, M1056, M1057 and
`docs/359` were not modified.
