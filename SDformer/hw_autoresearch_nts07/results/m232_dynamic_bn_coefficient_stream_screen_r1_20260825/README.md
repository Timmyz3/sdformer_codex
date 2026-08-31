# M232 dynamic-BN coefficient stream screen

The frozen no-running FFN path retains 24 current-batch BN phases and 22,080 per-channel coefficients per frame. A single coefficient stream at output interval 16 cycles can feed double-buffered 16-channel BN1 tiles and 96-channel BN2 tiles without a post-first-tile rate stall in any stage. The slowest boundary is stage-3 BN2: 1,536 coefficient-fill cycles versus 3,000 replay cycles.

Without overlap, coefficient service is 353,280 cycles/frame. The exact ping-pong recurrence exposes only the first tile of each of 24 phases, 21,504 cycles/frame (16.428571x lower), equal to 0.010470% of the 205.384M accounted FFN cycles and 0.003467% of the existing global envelope. II31 still rate-matches every phase; II32 first misses stage-3 BN2. Therefore the performance-critical work is the moment barrier and state replay, not a large rsqrt farm.

Using an illustrative Q24 alpha plus Q24 offset payload, the largest block would materialize 184,320 coefficient bits. Two shared 96-channel tile banks hold 9,216 bits, a 20x local payload reduction. BN1 prefold can use M167's mutually exclusive PREFOLD phase instead of adding a second wide multiplier pool.

This milestone does not prove reciprocal-sqrt error, fixed-point equivalence, moment SRAM ports, complete BN cycles, PPA, energy or system speedup. The next numeric gate is an A800 capture of real mean/variance/gamma/beta and activation ranges.
