# M191 independent hammer review

Status: `CONDITIONAL_PASS_EXACT_WINDOW_BATCH_REPLAY_CONTEXT_LABEL_REJECTED`  
Score: **78/100**

I independently decoded all 120 frozen H67 ep35 FC2 payloads without
importing the M191 or M172 analyzer.  All 437,760,000 payload bytes passed
manifest size and SHA checks; every record popcount matched.  Rebuilding the
arrival-order 96-bit descriptor stream with a different chunk size reproduced
all five replay totals exactly:

| window batch W | replay cycles | speed vs W1 | bank utilization | serial-K1 / W |
|---:|---:|---:|---:|---:|
| 1 | 79,397,844 | 1.000000x | 65.004976% | 5.200398x |
| 2 | 71,233,088 | 1.114620x | 72.455864% | 5.796469x |
| 4 | 67,218,210 | 1.181195x | 76.783582% | 6.142687x |
| 8 | 64,622,733 | 1.228636x | 79.867481% | 6.389399x |
| 16 | 62,671,956 | 1.266880x | 82.353500% | 6.588280x |

The W1 total is exactly M187's K8 replay total.  A separate 200,000-case
queue simulation also confirmed that, once a batch is resident and each bank
may independently select a window, service time is
`max_b(sum_w(population[w,b]))` with at most one read per bank per cycle.

## P0: W is not the number of independent contexts

M191 batches **windows**, but labels W as accumulation contexts.  These are
not equivalent because one token often creates several adjacent windows.
For W2, 1,417,525 of 3,261,887 batches (43.457207%) contain windows from only
one token.  For W4, only 143,722 of 1,630,971 batches (8.812051%) actually
contain four tokens; the mean is 2.704529 distinct tokens.

The replay totals remain useful if renamed as an ideal arrival-order
W-window bank-disjoint opportunity.  A correct implementation must choose one
of two contracts:

- windows from the same token share one token-owned Acc24 and all bank
  contributions for that token are reduced once per cycle; or
- every window owns a partial accumulator, in which case merge arithmetic,
  storage and cycles must be charged.

The current result models neither implementation.  Cross-token FC2 arithmetic
is mathematically independent inside one FC2 call, but BN2, residual, ATLIF
ordering and complete-FC2 completion remain outside the scope.

## Physical boundary

The ideal schedule can return different token tags from the eight banks in
one cycle.  Retaining its gain therefore requires simultaneous context-owned
Acc24 updates, not one shared read-modify-write port.  Acc24 is wide enough:
the worst pure FC2 bound is `3072*128=393216`, which fits signed 20 bits.
Porting and ownership are the missing proof.  Raw context state alone is
2,304 bits per token, 4,608 bits for two tokens and 9,216 bits for four.

The replay model also assumes resident batches.  It excludes 18,869,376
nonzero descriptor accepts, 36,480,000 raw 96-bit scan beats, finite buffers,
weight-response latency, output backpressure and context completion.  Reusing
M187's 18,209,963-cycle overhead unchanged is an optimistic proxy, not a wall
model.  Context tags additionally need a generation/epoch or proven flush-ack
quarantine; `ceil(log2(C))` slot bits alone can alias delayed responses after
slot reuse or reset.

## Admission gates

C2 may proceed only as a fail-closed RTL screen.  VCS must cover same-token
and cross-token windows, finite arrivals, request/response and result stalls,
partial tails, resets, stale responses, one-read-per-bank conservation and
exact 96-lane signed accumulation.  DC must include the routing and context
state in the declared boundary at 3.000 ns.

Against M186's 37,144.673821 um2 flat C1 reference, the replay-only C2
throughput/area break-even is 41,402.206478 um2, an allowance of only
4,257.532657 um2 or 11.462027%.  This is a screening bound, not physical
admission; matched finite-cycle throughput must replace it.

C4 stays an upper DSE point.  It gives only 1.059729x replay improvement over
C2 and can spend at most another 2,472.913336 um2 at the replay-only
break-even.  It should advance only if a token-aware implementation beats C2
in finite-cycle throughput/area and energy after buffering, ports and tagged
routing are included.

No complete-FC2, FFN, wall, physical, energy, FPS, PPA, system or headline
speedup is admitted by this review.
