# M195 independent-review admission

Independent dual-decoder replay confirms the exact token-flush result:
79,397,844 W1 cycles become 71,596,122 pair-fusion replay cycles
(1.108968500x), with 2,770,902 full same-token pairs, 981,903 odd tails and
zero cross-token pairs.  One Acc24 is mathematically sufficient because both
windows own the same token and output block.

Admission stops at replay arithmetic.  The next result must charge finite
buffer fill/drain, tagged SRAM responses, exact channel conservation and the
integrated frontend.  At the replay-only upper bound, total matched area must
remain at or below 41,192.273200 um2; this threshold must be recomputed with
finite cycles.
