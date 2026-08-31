# M196 token-flush finite-buffer DSE

M196 charges one nonzero96 descriptor fill per cycle, finite pair readiness,
serial drain and B={2,3,4} resident window buffers.  It exactly cross-checks
the M187 W1/B2 wall count of 97,607,807 cycles and the M195 pair replay count
of 71,596,122 cycles.

The finite result rejects buffer-only promotion.  Two buffers take 97,909,442
cycles (0.996919x, slower than W1); three take 97,628,132 (0.999792x); four
take 97,389,935 (1.002237x).  Stage 0 becomes 5.88% slower because pair
readiness waits for descriptor fill, while only stage 3 benefits materially.
Paying for more buffers cannot justify a 0.224% aggregate cycle gain.

The next hardware lever is descriptor packing/fill width, not a second Acc24
or more buffers: test two or more 96-bit nonzero descriptors per fill cycle.
SRAM response latency, backpressure and integrated RTL remain excluded.
