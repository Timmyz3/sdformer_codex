# M192 independent hammer review

Status: `CONDITIONAL_PASS_EXACT_GLOBAL_PAIR_PHASE_TOKEN_FLUSH_RESEAL_REQUIRED`  
Score: **86/100**

I independently decoded all 120 frozen H67 FC2 payloads without importing
the M192 or M172 analyzer.  Two runs used 4,099 and 65,521 tokens per chunk;
624 versus 16 pairs crossed a chunk boundary, yet the full-pair identity SHA,
tail identity SHA and every semantic total were identical.  All 437,760,000
bytes passed file SHA/size checks and every manifest popcount matched.

## Sealed result reproduced

The exact flattened-pair result is sound:

| policy | replay cycles | speed vs W1 | scope |
|---|---:|---:|---|
| W1 | 79,397,844 | 1.000000x | one window |
| same-token fuse, cross-token pair serial | 75,099,527 | 1.057235x | sealed global pair phase |
| ideal dual-token W2 | 71,233,088 | 1.114620x | unimplemented upper point |

There are 3,261,820 full global pairs and 67 record-tail partials.  The full
pairs split exactly into 1,417,458 same-token and 1,844,362 cross-token pairs.
`1.057234941x` means **5.723494% speedup** and **5.413644% fewer replay
cycles**; it must not be described as a 57.2% improvement.

Token IDs do not alias in the production calculation.  IDs are absolute
within each record across chunks, and pending state is flushed at every
record boundary.  My recomputation additionally used
`(record_ordinal << 32) + local_token_index`, making cross-record identity
explicit.  Both chunk sizes produced the same full-pair identity SHA
`3d3187b7...f380b` and record-tail SHA `c8d319e4...b4ba0`.

## P0: reset pair phase at token boundaries

M192 keeps one flattened W2 pairing phase across token boundaries.  When a
pair crosses tokens it falls back to W1, but it also consumes the first
window of the next token, preventing that window from pairing with its own
following window.  Thus the reported 43.456046% same-token fraction is a
property of that pairing phase, not a one-context hardware limit.

A natural M184-style engine owns one token at a time.  Restarting the pair
phase at each token boundary preserves arrival order, performs no window
reordering and still needs only one token-owned Acc24.  The independently
exact result is:

- **71,596,122 cycles = 1.108968500x versus W1**;
- 2,770,902 full same-token pairs plus 981,903 token-local tails;
- only 0.509642% slower than the ideal dual-token replay point.

This is the correct first RTL policy to reseal.  Dual-token accumulation is
not justified for the remaining half-percent replay gap.

## One Acc24 is sufficient, but base state is not free

Two windows of the same token partition the FC2 source events.  Signed weight
vector addition is associative, so sources selected from either window can
update the same 96-lane Acc24 vector once per cycle.  A 20,000-case independent
bank-service and signed-sum test had zero mismatches.  The worst binary-input
INT8 bound `3072*128=393216` fits signed 20 bits.

This proves **zero extra token context relative to W1**, not zero storage.
M186 receives its 2,304-bit Acc24 context from outside the synthesized
boundary.  BN2, residual and ATLIF must see only the final completed FC2
token, never an intermediate window result.

## RTL and physical gate

M184 already contains two window payload buffers and admits only one active
token, so it is a useful base.  It does not fuse them today: its selector,
bank counts and release logic reference only one `candidate_window`.  The
successor needs a per-bank two-buffer choice and correct independent
decrement, then must release the pair only after the combined populations
drain.

Two buffers are functionally sufficient, but finite performance is open.
Fusion waits for the second window to close, and both buffers remain occupied
during pair drain.  Stage 3 can contain two pairs, so a two-buffer design
cannot fill the next pair concurrently; a third buffer may help but must earn
its area.  The replay DSE charges none of this fill/wait/backpressure.

Against M186's 37,144.673821 um2 flat logic reference, the corrected
token-flush replay-only density screen is 41,192.273200 um2, allowing
4,047.599379 um2.  This is not an admission threshold.  Matched finite-cycle
W1/W2 runs under the same descriptor and SRAM-response timing must replace
it.  M186 has only +0.0002 ns setup slack at 3 ns under ideal clock and
ZeroWireload, so the extra selector is a real timing risk.

VCS must cover all buffer0/buffer1/empty bank choices, 981,903-style
single-window tails, stage3 two-pair tokens, every interface stall, exact
96-lane conservation, zero tokens, malformed traffic and final-only token
completion.  The successor must also close M186's known stale response
`A/reset/B/stale-A` alias with an epoch or proved flush-ack contract.

No RTL, finite-wall, complete-FC2, FFN, physical, energy, FPS, system or
headline speedup is admitted by this review.  `docs/359` remains unchanged at
SHA-256 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
