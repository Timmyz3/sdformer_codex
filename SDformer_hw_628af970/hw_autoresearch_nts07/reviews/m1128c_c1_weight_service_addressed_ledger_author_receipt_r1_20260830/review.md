# M1128C — weight-service addressed-ledger source author receipt

Status: `PASS_M1128C_WEIGHT_SERVICE_ADDRESSED_LEDGER_SOURCE_AUTHOR_RECEIPT__DIFFERENT_AUTHOR_HAMMER_REQUIRED__CANONICAL_STOP`

## Verdict

The addressed-event schema, exact-once conservation rule, 24-slice 1RW scheduler and bounded three-axis synthetic oracle are source-complete. The real ledger remains **STOP** because the frozen M1102/M1016 iterator does not contain native addressed weight events.

Only a different-author static hammer is authorized next. No successor real-ledger source, canonical row open, 51.84M replay, runner, RTL, EDA, GPU, remote work, traffic, cycle, energy or performance claim is authorized by this receipt.

## Actual frozen interface

Independent source inspection in the author check binds the actual frozen call:

`weight_task(global_offset + start - preprocess, receipt.counts.weight, index & 1)`

The common receipt provides only a count and global `weight_beat_first`; the packing interface adds an interval and task-parity half slot. It still lacks native READ/WRITE, logical bank, native slice set, local row, bytes/byte enable, native activation multiplicity and the exact-once service-beat-to-store relation. None may be inferred from count, ordinal or 24-macro capacity geometry.

Consequently, M1128C opens no canonical row, reads none of the 51.84M population, emits zero canonical weight transactions and keeps `canonical_ready=false`.

## Source mechanism

The source defines the required transaction schema for 24 independent `128x128-bit 1RW` native slices. Each event carries logical bank/half slot/local row, explicit native slice set, op, bytes, per-slice byte enable, native activation count, source provenance and transaction identity. Refill service beats must map exact-once to one 128-byte store transaction covering eight native slices; full-record reads cover all 24 slices. Arbitration charges an explicit stall until no activated native slice conflicts in the same final cycle. The mapping function is identical for candidate, strongest-zero and same-coordinate-bit axes.

This is a proposed/synthetic mapping contract, not evidence that the frozen H67 iterator follows it.

## Bounded oracle

The bounded oracle produces nine transactions across three axes: six refill stores and three full-record reads. Six expected service beats map exact-once, three requests stall explicitly, and the final schedule has zero native 1RW conflicts and zero half-slot overlaps. The author check passed 192 checks and rejected 12 attacks covering count expansion, duplicate/missing beat mapping, bad slices/bytes/enables/activation, final conflicts and claim escalation.

Source SHA256: `d25f9e4fdfda62f56e7efb120fe0c8f6108a4b23ba4eee712e3ec471b5fa493e`.

`docs/359` remains SHA256 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
