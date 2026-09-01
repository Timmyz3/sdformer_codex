# M1787 source-author attestation

M1787 is an additive source-only successor. I did not run VCS, `simv`, Design Compiler, PrimeTime PX, a license query, an attempt, or a result. M1780 is unchanged and remains `FAILED_DO_NOT_RELEASE` under the sealed M1781 review.

The repair composes the frozen M803 channel-split adapter instead of presenting an atomic eight-bank proxy. The external SRAM boundary now has eight independent request/ready channels and eight response channels with the exact M803 epoch, slot, generation, tag, output-block, slice, and source-channel request identity. The source and directed TB retain independent-bank backpressure, bank response reorder, stale and duplicate response attacks, sticky fail-closed behavior, and reset recovery.

The signed bridge is also explicit. The existing binary active source with sign zero is `+1`; an additive sign bit represents `-1`; inactive sources do not issue. A negative source performs exact nine-bit two's-complement negation before the shared per-token Acc24 update. The directed `INT8_MIN` corner therefore evaluates `-(-128)` as `+128`, not as an eight-bit wraparound.

Both modes own the same LRU8 row data, active/sign state, one frozen M803 adapter, eight independent 96-value Acc24 contexts, tags, and commit work. Only traversal order changes. The reference ledger preserves 96 row accesses, 1,152 issues, 18,432 signed products, and 48 commits per mode. The 1,152 versus 144 values are aggregate eight-bank bundle-beat expectations; the corresponding scalar-bank expectations are 9,216 versus 1,152. They are not VCS measurements, cycle speedups, layer speedups, or paper results.

A different author must complete M1788 before any exact-SHA release. VCS, same-coordinate DC, mapped activity/energy, and an independent result hammer remain mandatory. TSBG remains a C2 memory specialization rather than a fourth contribution.
