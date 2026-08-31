# M195 H67 FC2 token-flush pair fusion

M195 resets the adjacent-window pairing phase at every token boundary.  This
matches a single-token-owned FC2 frontend: all 2,770,902 full pairs share one
Acc24, while 981,903 odd tails drain alone.  Across all 120 frozen H67 ep35 FC2
payloads, replay falls from 79,397,844 to 71,596,122 cycles (1.108968500x).
The result is only 0.509642% slower than the global ideal dual-token W2 point,
without any cross-token pair or second accumulator context.

Stage replay factors are 1.166643x, 1.121005x, 1.094885x and 1.089641x.
The result is exact replay arithmetic, not finite wall time: two-buffer pair
fill/waiting, weight responses, reset quarantine, integrated RTL area, BN2,
residual and complete-FC2 are still excluded.
