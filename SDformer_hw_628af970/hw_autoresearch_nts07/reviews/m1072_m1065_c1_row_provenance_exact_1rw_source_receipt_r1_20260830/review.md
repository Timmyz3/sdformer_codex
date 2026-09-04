# M1072 row-provenance exact-1RW source receipt

## Verdict

`PASS_M1072_SOURCE_ONLY__M1073_REQUIRED_NO_FULL_REPLAY`

M1072 is the additive repair for the M1065 STOP. The exact-1RW arbitration and
214,912-byte capacity model are preserved, but caller-created task records can
no longer provide any cycle-driving field to the production scheduler.

## Unique production boundary

`iter_canonical_full_replay_results()` is the sole production cycle entry and
has zero arguments. It internally opens the exact 466,560,000-byte M410 row
file using `O_NOFOLLOW`, validates its initial fd SHA, and uses `pread` only at
offsets derived from task ID. Each task's exact raw bytes are parsed into masks;
the frozen M1016 functions then rederive the common receipt, three preprocess
values and shared maximum, three work values, and normalized parent summaries.

The file stat signature is checked throughout. No result is yielded until all
812,160 tasks have been read in exact order and the context manager has closed
with a final stat and full fd SHA check. A short read, file drift, wrong offset,
or row reorder rejects.

## Per-task execution provenance

Every internal record binds task/order/coordinate, file offset, row count,
exact raw-row SHA, canonical little-endian mask SHA, shared preprocess, all
three work values, all three parent summaries and the common receipt. Coverage
hashes this execution provenance in addition to the frozen service digest.

External records remain usable only for read-only validation. The validator
reopens the frozen row file and rederives the entire record before equality;
external records never reach the scheduler. Directed tests reject the M1065
candidate-work-zero/baseline-work-999999/preprocess-zero forgery, an all-zero
mask replacement, wrong digests, reordered rows, short reads and stat drift.

Task 0 is anchored at 210 preprocess cycles, 1,664 candidate work cycles and
4,392 zero/bit work cycles. Its candidate parent summary is 408 reads, 248
writes and 32 forwards. These are small-oracle checks, not a full-result claim.

## Preserved physical boundary

The frozen ledger remains 122,880 B psum + 49,152 B weight + 42,880 B
parent/other = 214,912 B, leaving 30,848 B under 240 KiB. It establishes only
capacity arithmetic. Full-trace port feasibility, matched cycles, speedup,
RTL cycles and paper PPA remain false.

## Evidence and next gate

Fifteen directed tests and the fail-closed checker pass. Neither the generator
body nor any full-result namespace was used. A different author must now run
the receipt-blind M1073 source hammer. Only an M1073 PASS may authorize a
separate one-shot M1074 CPU runner. No automatic retry, EDA, GPU or remote work
is authorized.

M1064, M1065 and `docs/359` were not modified.
