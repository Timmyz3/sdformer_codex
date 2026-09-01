# M1781 independent source hammer: M1780 B8 TSBG

## Verdict

**74/100, FAIL-CLOSED.** M1780 gets the standalone scheduling experiment right, but it is not yet a real M803/M519 C2 specialization. I do **not** authorize the exact M1780 SHA for M1784 VCS, DC, PTPX, attempt/result creation, or launch release.

This does not kill TSBG. It requires an additive protocol/typed-signed repair before physical evaluation.

## What independently passed

The review did not import the author reference model for its central calculations.

- Both modes retain the same source FIFO, ordinary persistent LRU8 data/tag/age structures, eight-token Acc24 state and commit structure. The only schedule branch in this RTL is token-major versus group-major.
- For eight contexts, twelve live groups, two halves and six slices: both modes perform 96 row accesses, 1,152 issue accepts, 18,432 signed products and 48 commits.
- Ordinary LRU8 independently gives baseline 0 hit / 96 miss and candidate 84 hit / 12 miss. The authored 1,152 versus 144 numbers are aggregate eight-bank row-response beats.
- Every token retains an independent `{-1,0,+1}` source value. RTL add/subtract is per context; there is no product reuse or lossy drop.
- The explicit default storage arithmetic is correct: row cache 12,288 B + Acc24 2,304 B + signed source FIFO 6,144 B + tags 24 B + active bitmap 48 B = **20,808 B excluding control**. This is a total source-island account, while M1763's 2,128 B was only an incremental B8-over-B1 lower bound; they must not be described as the same quantity.
- Stalls, terminal commits, mixed signs and malformed `+2` source attack exist. `docs/359` remains `dedde7ce...bdfc4`.
- The independent hammer produces identical output under CPython 3.6 and 3.10 and detects schedule, capacity, signed-branch and state-account mutations.

## Blocking findings

### P0-1: atomic proxy is not the M803 physical port

M803 exposes eight independently handshaken bank request/response channels, with epoch, slot, generation and tag ownership. M1780 exposes one request and one response carrying all eight banks atomically, identified only by group/half/slice.

Therefore M1780 removes unequal bank readiness, out-of-order bank return, live-slot ownership and stale-response behavior. A VCS/DC result from this exact source would be a standalone proxy result, not a same-resource M803/C2 result.

### P0-2: the frozen C2 service is not typed-signed

Frozen M218/M519 accepts a source count, binary bank-valid mask and source channels. It has no signed source-value field. M1780 invents that field and performs the signed add/subtract in a local eight-token accumulator array.

The frozen C2 accumulator contexts also represent output-block contexts within a token; they are not directly the eight token contexts needed for TSBG. Thus the source comment saying the local ledger can later be replaced by the admitted M803 service is not executable without a new additive typed-signed service/bridge and context mapping.

### P1 coverage gaps

- The memory model never injects a mismatched, duplicate or stale response. The only protocol attack is malformed source value `+2`.
- Twelve groups over LRU8 imply eviction, but there is no explicit eviction cover/counter.
- Overflow is not dynamically covered. For the default legal point it is statically unreachable: `48 groups * 16 sources * 128 = 98,304 < 2^23`. Seal that bound and assert no overflow rather than inventing an unreachable attack.

### P2 terminology

The 1,152/144 counters are aggregate eight-bank responses. At M803's scalar bank pins, fully active rows imply 9,216/1,152 scalar bank responses. Future receipts must carry both names.

## Required successor before launch

1. Add a typed-signed C2 service/bridge with explicit eight-token identity and per-token Acc24 ownership.
2. Use the real eight-bank independent M803 request/response protocol, including epoch/slot/generation/tag and per-bank skew.
3. Keep baseline and candidate at the same eight banks, LRU8 capacity, FIFO/tag/control, eight token contexts and commit work; only schedule/fetch broadcast may differ.
4. Add illegal/stale/duplicate response attacks, eviction coverage, terminal coverage, backpressure coverage, and the sealed Acc24 bound.
5. Submit the repaired exact SHA to a new different-author source hammer. Only that review may authorize one VCS/DC campaign.

Until then, M1763 remains screening-only, M1780 remains source-only, and no 5.12x, 8x, 1.15x, layer speedup, system speedup, area or energy claim is admitted.
