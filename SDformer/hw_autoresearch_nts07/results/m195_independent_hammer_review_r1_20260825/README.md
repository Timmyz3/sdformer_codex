# M195 independent hammer review

Verdict: **84/100 — conditional pass for exact token-flush replay only; RTL
and finite wall time remain open.**

I did not import the M192 or M195 production analyzer.  The independent
recompute reads all 120 frozen FC2 payloads (437,760,000 bytes) directly.  One
pass uses 4,093-token chunks with explicit little-endian shifts; a second uses
32,749-token chunks with `unpackbits(bitorder="little")`.  Aggregate and all
four stage ledgers are identical between the two passes.  All payload SHA and
size checks pass, and all 120 path-encoded sample/module/call IDs agree with
their manifest records.

The independent result exactly reproduces:

- W1 replay: 79,397,844 cycles.
- Token-flush pair replay: 71,596,122 cycles, or 1.1089684997184623x.
- 2,770,902 full same-token pairs, 981,903 odd tails, and zero cross-token
  pairs.
- 1,863,944 zero-event tokens.  They do not inherit or propagate a pair phase.
- Window conservation: 6,523,707 = 2 * 2,770,902 + 981,903.
- Stage factors: 1.166643168x, 1.121004678x, 1.094884806x and 1.089641355x.

## What is valid

Resetting the pair phase at each token is consistent with M184's single-token
ownership contract: adjacent windows share a token and odd tails retire before
the next token starts.  For a fixed token and output block, merging two
windows' per-bank queues is also mathematically sound.  Every signed weight
term still reaches the same Acc24 exactly once, and integer addition is
order-independent.  The payload audit proves the schedule counts, however,
not channel-tagged RTL numerical behavior.

## What is not yet valid

The 1.108968500x value is replay arithmetic.  It excludes header/done cycles,
the wall time of 1.864 million zero-event tokens, finite two-buffer fill/drain,
odd-tail release latency, SRAM response latency, stalls, BN2 and residual
commit.  For intuition only, adding an equal one, two, four or eight cycles per
token to both points reduces the ratio to 1.10109x, 1.09427x, 1.08307x or
1.06712x.  Thus the current number cannot be presented as RTL, physical,
complete-FC2, FFN or system speedup.

The most important correctness attack is stale SRAM response.  Prior M186
review evidence exposed reset aliasing; resetting the pair phase does not stop
an old response from updating the new token.  The integrated design needs an
epoch/token/request tag or a proved quarantine, plus VCS assertions that pair
release waits for all queues and outstanding responses.  Token done must also
wait for final Acc24, BN2 and residual/ATLIF commit under backpressure.

Two buffers may be unable to fill the next pair while the current pair is
being serviced.  The finite replay must therefore compare a bounded dual
buffer with an explicitly charged third-buffer option.  A third buffer is a
hypothesis, not a free overlap assumption.

## Matched 3 ns density gate

Against sealed flat M186 at 37,144.673821 um2, the replay-only upper-bound
break-even area is 41,192.273200 um2, leaving at most 4,047.599379 um2 of
incremental logic.  Once finite cycles are known, the actual gate is:

`M195 area <= 37144.673821 * (79397844 / finite M195 cycles)`

Both sides must meet 3.0 ns and use identical storage, SRAM latency/ports,
stall and commit contracts.  The provisional 551 um2 M194 selector would leave
3,496.599379 um2, but it was not sealed during this review and excludes queue
storage, head advancement, tagged responses and integration, so it is not
admitted as evidence.

The next admissible milestone is finite exact-payload RTL with token/epoch
response safety and channel-tagged numerical conservation, followed by this
matched DC gate.
