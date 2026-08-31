# M192 independent-review admission overlay

The sealed M192 global-pair-phase arithmetic is reproducible, but it is not the
selected hardware policy.  Keeping W2 pairing phase across token boundaries
creates cross-token fallback pairs and shifts away otherwise fuseable pairs in
the next token.  M184-style hardware owns one token at a time and can reset the
pair phase at each token boundary.

The independent token-flush recomputation gives 71,596,122 replay cycles, or
1.108968500x over W1, with one Acc24 and 981,903 token-local odd tails.  This is
only 0.509642% slower than the global ideal dual-token W2 point.  The successor
must reseal that policy, model finite pair fill, implement dual-buffer per-bank
selection/release, close stale response aliasing, and run matched VCS/DC.

Neither this overlay nor its successor changes `docs/359`.
