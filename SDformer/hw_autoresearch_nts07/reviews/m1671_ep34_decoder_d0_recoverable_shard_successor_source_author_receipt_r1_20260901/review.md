# M1671 author receipt: recoverable full-D0 shard source

Status: `PASS_M1671_EP34_DECODER_D0_RECOVERABLE_SHARD_SUCCESSOR_SOURCE__DIFFERENT_AUTHOR_REVIEW_REQUIRED__NO_PAYLOAD_NO_EXECUTION`.

M1671 turns the independently hammered M1656 D0/call0/t0 prefix into a fixed, recoverable source plan for all 30 frozen D0 calls, ten timesteps per call, 1,200 destinations per timestep and four output blocks. The plan contains 8,700 unique contiguous shards: 28 shards of 42 destinations and one final shard of 24 for each call/timestep pair.

The future engine retains the exact M1539 reference scheduler, M1610 compact scheduler, three non-product configurations, same 240 KiB resource manifest, per-request comparison, per-destination cumulative comparison and dual RSS gates. Performance may be reduced only after all 8,700 sealed shards are present and only with integer ratio-of-sums.

The reset at every shard boundary is explicit. Therefore a future complete result is full-D0 population coverage under a shard-isolated cycle model, not a monolithic full-call execution. D2/D3 require separately reviewed geometry rebinds. D1 remains excluded until an exact numeric bridge exists. Nothing here is a full-decoder, system-speedup or paper result.

Both CPython 3.6 and the current Python passed 11 tests plus compilation. No canonical payload was opened, no attempt/result/release was created, and no GPU or EDA tool ran.
