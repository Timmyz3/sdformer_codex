# M1146CA independent expected-digest compiler author review

Verdict: **PASS for source and bounded synthetic evidence only**.

The source independently reconstructs the complete 17-field refill event, exact-once ID, M1137C source-row provenance, M1135C fixed-endian serialization, and the per-axis 24-slot 1RW scheduler. It imports or calls none of the M1137C/M1135C/M1130C/M1132C subject runtime and reads no producer output.

The bounded oracle covers three tasks, all three axes, and variable task intervals of 2/3/3 beats. Its 24 events match a separately written golden implementation. Sixteen attack classes were rejected, including schedule ordering/provenance/cycle drift, serializer field/order/endianness drift, ID/provenance/scheduled-cycle drift, caller-supplied expected digests, partial input, and failure atomicity.

Production remains stopped before opening the 2,436,480-record schedule JSONL. No production digest, real driver, full replay, EDA, performance, energy, or PPA claim is authorized. The next allowed action is a different-author bounded hammer.
