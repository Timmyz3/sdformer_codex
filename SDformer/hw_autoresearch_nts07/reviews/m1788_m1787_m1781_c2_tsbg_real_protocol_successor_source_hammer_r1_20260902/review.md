# M1788 independent source hammer: M1787 true-protocol TSBG

## Verdict

**82/100, FAIL-CLOSED.** M1787 does close the two architectural P0 findings from M1781: it composes the frozen M803 eight-bank adapter and implements an executable typed-signed bridge with eight independent token accumulator contexts. However, this exact source cannot be released for the single directed VCS run because its sealed reduced TB configuration triggers the RTL's own time-zero parameter fatal.

No VCS, `simv`, DC, PTPX, license query, attempt or result was run or created by this review.

## What independently passed

- `docs/359`, M1780 and the frozen M803 adapter retain their expected identities. Author contract and receipt seals pass.
- The source exposes eight independent request/response valid-ready channels and preserves request identity as epoch, slot, generation, tag, output block, slice and source channel.
- The signed bridge is executable: active/sign encodes `+1/-1`, inactive sources contribute zero, and a nine-bit widened negation maps `-(-128)` to `+128` without eight-bit wraparound.
- Both modes retain the same LRU8 data, B8 active/sign state, M803 adapter, eight 96-value Acc24 contexts and commit work. The only intended axis is traversal order.
- Independent models reproduce 96 row accesses, 1,152 issues, 18,432 signed products and 48 commits in each mode. Token-major gives 0 hit / 96 miss / 88 eviction; row-major gives 84 hit / 12 miss / 4 eviction. The corresponding counts are 1,152 versus 144 aggregate eight-bank bundle beats and 9,216 versus 1,152 scalar bank beats.
- The production worst-case bound remains `48*16*128 = 98,304 < 2^23`.
- Author tests pass 12/12 under CPython 3.6 and 3.10. The independent hammer is identical under both interpreters and detects ten central mutations.

These are source facts, not VCS measurements, not a same-resource DC/energy result, and not a 5.12x or 8x hardware result.

## P0: directed parameter tuple is illegal

The TB uses `GROUPS=12` and overrides both DUTs with `.SOURCE_GROUPS(GROUPS)`. RTL computes `STATIC_ACC24_ABS_BOUND = SOURCE_GROUPS * 16 * 128`, while `PARAMETERS_LEGAL` requires that value to equal 98,304. At twelve groups the value is 24,576. Both instances therefore execute the generated `initial $fatal` at time zero.

The repair is narrow: either run the directed test at 48 groups and update its expected ledgers/timeout, or separate the fixed production worst-case bound from the reduced directed geometry and explicitly admit reduced source groups without weakening the 48-group proof. A new different-author source hammer is required before VCS.

## P1 coverage repairs

1. The two response attacks both inject the same fabricated dead identity. The second is a duplicate of a rejected malformed response, not a replay of a previously accepted and retired legal bank response. Preserve the stale/mismatched attack and add a separate legal-identity replay.
2. Reset recovery only checks that sticky flags clear. It never completes a legal request-response-bridge-commit transaction after reset. In addition, the SVA recovery cover expects one reset clock while the TB holds reset for three, so that named cover cannot close. Run a minimal legal workload after reset and align the cover with the actual reset duration.

## Disposition

M1787 is a viable additive successor, not a killed idea. Its exact SHA is not authorized for VCS, DC, energy, attempt/result creation or release. M1780 remains `FAILED_DO_NOT_RELEASE`. A repaired successor should preserve the current real M803 and typed-signed architecture, close the one P0 and two P1 findings, and return to an independent source hammer.
