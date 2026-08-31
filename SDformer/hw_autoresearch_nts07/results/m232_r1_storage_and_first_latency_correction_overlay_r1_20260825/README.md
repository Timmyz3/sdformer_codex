# M232 r1 storage and first-latency correction

The independent review reproduced M232's 12-block/24-phase geometry,
17,664/4,416 BN1/BN2 coefficient counts, II16 no-overlap count of 353,280,
conditional ping-pong count of 21,504, and the II31/II32 rate boundary.

It also found that the reported 20x coefficient payload comparison was not
matched: BN1 and BN2 are mutually exclusive phases, so a phase-reused baseline
holds only the larger phase.  The corrected comparison is 3,072 channels x 48
bits = 147,456 bits versus two 96-channel x 48-bit tile banks = 9,216 bits, or
16x.  The 20x statement is revoked.

The 21,504-cycle result is still conditional on first coefficient latency
equaling II=16.  The coefficient engine, M167 PREFOLD's 7,728 issues/frame,
channel-major state SRAM, BN2 affine and residual ports are not yet included.
M233 supplies the real checkpoint-bound ranges needed to implement and test
that engine.  No physical, system, accuracy or headline claim is admitted.
